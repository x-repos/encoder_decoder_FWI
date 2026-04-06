"""DENISE-PyTorch bridge for encoder-decoder FWI.

Provides DeniseInterface (wraps pyapi_denise for forward/FWI calls) and
DeniseFWIFunction (torch.autograd.Function that injects DENISE adjoint
gradients into PyTorch's backward pass).
"""

import contextlib
import io
import os
import shutil
import warnings

import numpy as np
import torch

import pyapi_denise as api


# Default MPI launch command
RUN_CMD = "mpirun --mca coll ^hcoll -np 14"


class DeniseInterface:
    """High-level wrapper around pyapi_denise for the encoder-decoder FWI loop.

    Handles:
      - Generating observed (target) data by running DENISE forward on the true model
      - Running single-iteration FWI to compute adjoint gradients
      - Reading misfit values and gradient arrays
    """

    def __init__(
        self,
        root="./",
        obs_folder="./outputs_sth_obs/",
        fwi_folder="./outputs_sth_fwi/",
        run_command=RUN_CMD,
        verbose=0,
    ):
        self.root = root
        self.obs_folder = os.path.abspath(obs_folder)
        self.fwi_folder = os.path.abspath(fwi_folder)
        self.run_command = run_command
        self.verbose = verbose

        # Create a single Denise instance; we reconfigure its save_folder as needed
        self._denise = api.Denise(root, verbose=verbose)
        self._configure_common()

        # Will be set after generate_observed_data()
        self._obs_data_dir = None

    def _configure_common(self):
        """Set shared DENISE parameters."""
        d = self._denise
        d.NPROCX = 7
        d.NPROCY = 2
        d.PHYSICS = 1       # 2D-PSV elastic
        d.FREE_SURF = 1     # free surface on top
        d.FW = 10           # PML width in grid points
        d.QUELLART = 1      # Ricker wavelet
        d.SNAP = 0          # no snapshots
        d.TIME = 5.0        # 5 s recording
        d.DT = None         # auto CFL

    def _set_save_folder(self, folder, makedirs=False):
        """Point DENISE at a given output folder and rebuild paths."""
        d = self._denise
        d.save_folder = folder
        d.set_paths(makedirs=makedirs)

    # ------------------------------------------------------------------
    # Observed data generation
    # ------------------------------------------------------------------

    def generate_observed_data(self, model, src, rec):
        """Run forward modelling on the *true* model to produce observed data.

        The seismograms are written to ``obs_folder/su/`` and their path prefix
        is stored so that subsequent FWI calls know where the target data lives.

        Returns:
            list[np.ndarray]: Vx-component shot gathers (28 arrays).
        """
        self._set_save_folder(self.obs_folder)
        d = self._denise
        log_file = os.path.join(self.obs_folder, "denise_forward.log")
        cmd_suffix = f" > {log_file} 2>&1"
        with contextlib.redirect_stdout(io.StringIO()):
            d.forward(model, src, rec,
                      run_command=self.run_command + cmd_suffix)
        # After forward(), DATA_DIR = {obs_folder}/su/seis  (prefix for shot files)
        self._obs_data_dir = d.DATA_DIR
        shots = d.get_shots(keys=["_x"])
        return shots

    def load_observed_shots(self, keys=None):
        """Load previously-generated observed shot gathers."""
        if keys is None:
            keys = ["_x"]
        self._set_save_folder(self.obs_folder)
        d = self._denise
        # Make sure obs_data_dir is set even if generate was not called this run
        self._obs_data_dir = d.DATA_DIR
        return d.get_shots(keys=keys)

    # ------------------------------------------------------------------
    # Single-iteration FWI (gradient computation)
    # ------------------------------------------------------------------

    def compute_gradient(self, vp, vs, rho, dx, src, rec):
        """Run 1-iteration FWI on the given model and return (misfit, grads).

        Parameters
        ----------
        vp, vs, rho : np.ndarray, shape (nz, nx), float32
        dx : float
        src : api.Sources
        rec : api.Receivers

        Returns
        -------
        misfit : float
        grad_vp, grad_vs, grad_rho : np.ndarray, each (nz, nx)
        """
        # --- Build DENISE model ------------------------------------------------
        model = api.Model(
            np.ascontiguousarray(vp, dtype=np.float32),
            np.ascontiguousarray(vs, dtype=np.float32),
            np.ascontiguousarray(rho, dtype=np.float32),
            float(dx),
        )

        # --- Point DENISE at the FWI output folder -----------------------------
        # makedirs=True because fwi() writes the workflow file before _engine()
        # creates the subdirectories
        self._set_save_folder(self.fwi_folder, makedirs=True)
        d = self._denise

        # --- Ensure taper/ directory exists in CWD ----------------------------
        # DENISE writes gradient taper files to a relative "taper/" path
        os.makedirs(os.path.join(d.cwd, "taper"), exist_ok=True)

        # --- Copy observed data into FWI su/ folder ---------------------------
        # DENISE reads observed (target) data from DATA_DIR = {save_folder}/su/{seis_name}.
        # _engine() calls set_paths() which resets DATA_DIR to the fwi_folder,
        # so we cannot simply override the attribute.  Instead, copy the observed
        # SU files from the obs folder into the FWI folder before each run.
        obs_su = os.path.join(self.obs_folder, "su")
        fwi_su = os.path.join(self.fwi_folder, "su")
        if os.path.isdir(obs_su):
            for fname in os.listdir(obs_su):
                src_file = os.path.join(obs_su, fname)
                dst_file = os.path.join(fwi_su, fname)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

        # --- Ensure DT/NT are set before FWI ------------------------------------
        # pyapi's _calc_nt_dt only computes NT in forward mode (MODE<1).  For
        # FWI mode we must pre-compute DT/NT so the binary knows the time grid.
        if d.DT is None:
            d.set_model(model)
            maxvp = np.max(model.vp[np.nonzero(model.vp)])
            maxvs = np.max(model.vs[np.nonzero(model.vs)])
            d.DT = d._check_stability(maxvp, maxvs)
        d.NT = int(d.TIME / d.DT)

        # --- Write lambda/mu model files for INVMAT1=3 -------------------------
        # INVMAT1=3 (lambda/mu/rho parameterization) requires .lam and .mu files
        from pyapi_denise import _write_binary
        mu_arr = rho * vs**2
        lam_arr = rho * (vp**2 - 2.0 * vs**2)
        _write_binary(lam_arr, d.MFILE + '.lam')
        _write_binary(mu_arr, d.MFILE + '.mu')

        # --- Configure single-iteration FWI ------------------------------------
        d.ITERMAX = 1
        d.INVMAT1 = 3       # gradient in lambda/mu/rho parameterization
        d.GRAD_METHOD = 1   # PCG (L-BFGS=2 segfaults on 1 iteration)
        d.fwi_stages = []
        d.add_fwi_stage(
            pro=0.0,       # don't terminate early
            time_filt=0,   # no frequency filtering
            fc_low=0.0,
            fc_high=0.0,
            lnorm=2,       # L2 norm
            gamma=1,       # full step length
            stf=0,         # no source-time-function inversion
        )

        # --- Run FWI -----------------------------------------------------------
        log_file = os.path.join(self.fwi_folder, "denise_run.log")
        cmd_suffix = f" > {log_file} 2>&1"
        with contextlib.redirect_stdout(io.StringIO()), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d.fwi(model, src, rec,
                  run_command=self.run_command + cmd_suffix)

        # --- Read misfit -------------------------------------------------------
        misfit = self._read_misfit(d.MISFIT_LOG_FILE)
        if misfit is None:
            misfit = self._compute_misfit_from_seismograms(d)

        # --- Read gradients ----------------------------------------------------
        # With INVMAT1=3, DENISE writes three gradient files:
        #   _c.old     -> dE/dlambda
        #   _c_u.old   -> dE/dmu
        #   _c_rho.old -> dE/drho
        # Convert to Vp/Vs/rho using the chain rule:
        #   lambda = rho*(Vp^2 - 2*Vs^2),  mu = rho*Vs^2
        #   dE/dVp  = dE/dlambda * 2*rho*Vp
        #   dE/dVs  = dE/dlambda * (-4*rho*Vs) + dE/dmu * 2*rho*Vs
        #   dE/drho = dE/dlambda * (Vp^2 - 2*Vs^2) + dE/dmu * Vs^2 + dE/drho_direct
        grad_lam, grad_mu, grad_rho_direct = self._read_lame_gradients(d)

        grad_vp = grad_lam * (2.0 * rho * vp)
        grad_vs = grad_lam * (-4.0 * rho * vs) + grad_mu * (2.0 * rho * vs)
        grad_rho = (grad_lam * (vp**2 - 2.0 * vs**2)
                    + grad_mu * (vs**2)
                    + grad_rho_direct)

        return misfit, grad_vp, grad_vs, grad_rho

    @staticmethod
    def _read_misfit(log_path):
        """Parse the DENISE FWI misfit log file.

        Returns the misfit value, or None if the file is missing/empty.
        """
        if not os.path.isfile(log_path):
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = np.loadtxt(log_path)
        except ValueError:
            return None
        if data.size == 0:
            return None
        if data.ndim == 1:
            # single iteration: shape (ncols,)
            return float(data[1]) if data.size > 1 else None
        else:
            # multiple iterations: take the last row, second column
            return float(data[-1, 1])

    def _read_lame_gradients(self, d):
        """Read lambda, mu, and rho gradients from DENISE jacobian output.

        With INVMAT1=3, DENISE writes merged gradient files:
            gradient_<prefix>_c.old       ->  dE/dlambda
            gradient_<prefix>_c_u.old     ->  dE/dmu
            gradient_<prefix>_c_rho.old   ->  dE/drho

        Returns (grad_lambda, grad_mu, grad_rho) as (nz, nx) numpy arrays.
        """
        jacobian_dir = os.path.join(self.fwi_folder, "jacobian")
        nz, nx = d.NY, d.NX
        expected_size = nx * nz

        grad_lambda = np.zeros((nz, nx), dtype=np.float32)
        grad_mu = np.zeros((nz, nx), dtype=np.float32)
        grad_rho = np.zeros((nz, nx), dtype=np.float32)

        if not os.path.isdir(jacobian_dir):
            return grad_lambda, grad_mu, grad_rho

        # Find merged gradient files (skip PE fragment files like .0.0)
        for fname in os.listdir(jacobian_dir):
            fpath = os.path.join(jacobian_dir, fname)
            if os.path.isdir(fpath):
                continue
            parts = fname.rsplit(".", 2)
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                continue

            data = np.fromfile(fpath, dtype="<f4")
            if data.size != expected_size:
                continue

            # Reshape: DENISE stores (NX, NY), need transpose+flip
            arr = np.flipud(data.reshape(nx, nz).T).copy()

            # Identify gradient type from filename (INVMAT1=3 convention)
            # Strip control characters for matching
            clean = fname.replace(chr(14), "")
            if "_c_rho.old" in clean:
                grad_rho = arr
            elif "_c_u.old" in clean:
                grad_mu = arr
            elif "_c.old" in clean:
                grad_lambda = arr

        return grad_lambda, grad_mu, grad_rho

    def _compute_misfit_from_seismograms(self, d):
        """Compute L2 misfit from synthetic vs observed seismograms.

        After FWI runs, DENISE writes synthetic seismograms with '.it1' suffix
        (e.g. seis_x.su.shot1.it1).  Observed data lives in {obs_folder}/su/.
        The L2 misfit is 0.5 * sum((syn - obs)^2).
        """
        # Load synthetics: DENISE writes .it1 files for the current iteration
        self._set_save_folder(self.fwi_folder)
        synth_x = d.get_shots(keys=["_x", ".it1"])

        # Load observed from the obs folder
        self._set_save_folder(self.obs_folder)
        obs_x = d.get_shots(keys=["_x"])

        # Restore FWI folder setting
        self._set_save_folder(self.fwi_folder)

        if len(synth_x) == 0 or len(obs_x) == 0:
            return 0.0

        misfit = 0.0
        for syn, obs in zip(synth_x, obs_x):
            residual = syn - obs
            misfit += 0.5 * np.sum(residual ** 2)

        return float(misfit)


# ---------------------------------------------------------------------------
# Custom autograd function
# ---------------------------------------------------------------------------

class DeniseFWIFunction(torch.autograd.Function):
    """Inject DENISE adjoint gradients into PyTorch's autograd graph.

    forward():  Vp/Vs/Rho tensors -> numpy -> DENISE 1-iter FWI -> misfit scalar
    backward(): Returns stored DENISE gradients (dE/dVp, dE/dVs, dE/dRho)

    The chain rule then gives  dE/dw = (dm/dw) * (dE/dm)  where w are CNN
    weights, m = (Vp, Vs, Rho), and dE/dm is the adjoint gradient.
    """

    @staticmethod
    def forward(ctx, vp, vs, rho, denise_interface, src, rec, dx):
        """
        Parameters
        ----------
        ctx : context object for saving tensors for backward
        vp, vs, rho : torch.Tensor, shape (nz, nx), requires_grad=True
        denise_interface : DeniseInterface instance
        src : api.Sources
        rec : api.Receivers
        dx : float

        Returns
        -------
        misfit : torch.Tensor, scalar (shape [])
        """
        # Detach and convert to numpy
        vp_np = vp.detach().cpu().numpy().astype(np.float32)
        vs_np = vs.detach().cpu().numpy().astype(np.float32)
        rho_np = rho.detach().cpu().numpy().astype(np.float32)

        # Run DENISE
        misfit_val, grad_vp, grad_vs, grad_rho = denise_interface.compute_gradient(
            vp_np, vs_np, rho_np, dx, src, rec,
        )

        # Store gradients for backward; convert to tensors on the same device
        device = vp.device
        ctx.save_for_backward(
            torch.tensor(grad_vp, dtype=torch.float32, device=device),
            torch.tensor(grad_vs, dtype=torch.float32, device=device),
            torch.tensor(grad_rho, dtype=torch.float32, device=device),
        )

        return torch.tensor(misfit_val, dtype=torch.float32, device=device)

    @staticmethod
    def backward(ctx, grad_output):
        """Return DENISE adjoint gradients scaled by upstream grad_output.

        Returns a tuple matching forward() args:
            (grad_vp, grad_vs, grad_rho, None, None, None, None)
        Non-tensor arguments get None.
        """
        grad_vp, grad_vs, grad_rho = ctx.saved_tensors
        # Scale by upstream gradient (scalar)
        return (
            grad_output * grad_vp,
            grad_output * grad_vs,
            grad_output * grad_rho,
            None,  # denise_interface
            None,  # src
            None,  # rec
            None,  # dx
        )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def denise_fwi_misfit(vp, vs, rho, denise_interface, src, rec, dx):
    """Compute the FWI misfit as a differentiable scalar.

    This is the function you call in the training loop.  It returns a scalar
    ``torch.Tensor`` whose ``.backward()`` propagates DENISE adjoint gradients.
    """
    return DeniseFWIFunction.apply(vp, vs, rho, denise_interface, src, rec, dx)
