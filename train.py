"""Training loop for encoder-decoder FWI.

Assembles the STH model, encoder-decoder CNN, DENISE bridge, and Adam
optimizer into a complete training pipeline.

Usage:
    uv run python train.py --n_iters 350 --lr 0.0025 --save_dir checkpoints
"""

import argparse
import os
import time

import numpy as np
import torch

from sth_model import (
    NX,
    NZ,
    DX,
    create_true_model,
    create_initial_model,
    create_acquisition,
)
from network import EncoderDecoder
from denise_fwi import DeniseInterface, denise_fwi_misfit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_shot_gathers(di, time_downsample=4):
    """Load observed shot gathers and prepare CNN input tensors.

    Returns
    -------
    vx_tensor, vy_tensor : torch.Tensor
        Shape ``(1, n_shots, nrec, nt_down)``, dtype float32.
    """
    shots_x = di.load_observed_shots(keys=["_x"])  # list of (nrec, nt)
    shots_y = di.load_observed_shots(keys=["_y"])

    # Downsample time axis by `time_downsample`
    vx_list = [s[:, ::time_downsample] for s in shots_x]
    vy_list = [s[:, ::time_downsample] for s in shots_y]

    # Stack into tensors: (1, n_shots, nrec, nt_down)
    vx_tensor = torch.from_numpy(np.stack(vx_list)).float().unsqueeze(0)
    vy_tensor = torch.from_numpy(np.stack(vy_list)).float().unsqueeze(0)

    return vx_tensor, vy_tensor


def compute_model_mse(net, vx, vy, true_model):
    """Compute MSE between CNN output and the true model (monitoring only).

    Returns
    -------
    mse_vp, mse_vs, mse_rho : float
    """
    with torch.no_grad():
        vp, vs, rho = net(vx, vy)
        vp_true = torch.from_numpy(true_model.vp).float()
        vs_true = torch.from_numpy(true_model.vs).float()
        rho_true = torch.from_numpy(true_model.rho).float()
        mse_vp = ((vp.squeeze() - vp_true) ** 2).mean().item()
        mse_vs = ((vs.squeeze() - vs_true) ** 2).mean().item()
        mse_rho = ((rho.squeeze() - rho_true) ** 2).mean().item()
    return mse_vp, mse_vs, mse_rho


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 60)
    print("Encoder-Decoder FWI Training")
    print("=" * 60)

    # ── 1. Create models and acquisition ──────────────────────────────
    print("\n[1/5] Creating true model, initial model, and acquisition...")
    true_model = create_true_model()
    init_model = create_initial_model()
    src, rec = create_acquisition()
    n_shots = len(src.x)
    print(f"  Grid: {NX} x {NZ}, dx = {DX} m")
    print(f"  Sources: {n_shots}, Receivers: {len(rec.x)}")

    # ── 2. DENISE interface and observed data ─────────────────────────
    print("\n[2/5] Setting up DENISE interface...")
    di = DeniseInterface(
        root="./",
        obs_folder="./outputs_sth_obs/",
        fwi_folder="./outputs_sth_fwi/",
    )

    # Generate observed data if the SU directory is empty
    obs_su_dir = os.path.join(di.obs_folder, "su")
    if not os.path.isdir(obs_su_dir) or len(os.listdir(obs_su_dir)) == 0:
        print("  Generating observed data (forward on true model)...")
        di.generate_observed_data(true_model, src, rec)
    else:
        print("  Observed data already exists, skipping generation.")

    # ── 3. Prepare CNN inputs ─────────────────────────────────────────
    print("\n[3/5] Loading and preparing shot gathers...")
    vx_tensor, vy_tensor = prepare_shot_gathers(
        di, time_downsample=args.time_downsample
    )
    print(f"  Vx tensor shape: {vx_tensor.shape}")
    print(f"  Vy tensor shape: {vy_tensor.shape}")

    # ── 4. Create CNN ─────────────────────────────────────────────────
    print("\n[4/5] Creating encoder-decoder CNN...")
    net = EncoderDecoder(
        n_shots=n_shots,
        nx=NX,
        nz=NZ,
        initial_vp=init_model.vp,
        initial_vs=init_model.vs,
        initial_rho=init_model.rho,
    )

    # Warm-up forward pass to initialise lazy linear layers
    with torch.no_grad():
        _ = net(vx_tensor, vy_tensor)
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")

    # ── 5. Optimizer and history ──────────────────────────────────────
    print("\n[5/5] Setting up Adam optimizer (lr={})...".format(args.lr))
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    history = {
        "iter": [],
        "data_loss": [],
        "mse_vp": [],
        "mse_vs": [],
        "mse_rho": [],
    }

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Starting training for {args.n_iters} iterations")
    print("=" * 60)

    for it in range(1, args.n_iters + 1):
        t0 = time.time()

        # (a) Zero gradients
        optimizer.zero_grad()

        # (b) CNN forward
        vp, vs, rho = net(vx_tensor, vy_tensor)

        # (c) Squeeze to (nz, nx) for DENISE
        vp_sq = vp.squeeze(0).squeeze(0)
        vs_sq = vs.squeeze(0).squeeze(0)
        rho_sq = rho.squeeze(0).squeeze(0)

        # (d) DENISE misfit (differentiable)
        misfit = denise_fwi_misfit(vp_sq, vs_sq, rho_sq, di, src, rec, DX)

        # (e) Backward: injects DENISE gradients, backprops through CNN
        misfit.backward()

        # (f) Update weights
        optimizer.step()

        elapsed = time.time() - t0

        # (g) Log
        if it % args.log_interval == 0 or it == 1:
            mse_vp, mse_vs, mse_rho = compute_model_mse(
                net, vx_tensor, vy_tensor, true_model
            )
            history["iter"].append(it)
            history["data_loss"].append(misfit.item())
            history["mse_vp"].append(mse_vp)
            history["mse_vs"].append(mse_vs)
            history["mse_rho"].append(mse_rho)

            print(
                f"  iter {it:4d}/{args.n_iters} | "
                f"misfit {misfit.item():.6e} | "
                f"MSE(Vp) {mse_vp:.2f}  MSE(Vs) {mse_vs:.2f}  "
                f"MSE(rho) {mse_rho:.2f} | "
                f"{elapsed:.1f}s"
            )

        # (h) Save checkpoint
        if it % args.save_interval == 0 or it == args.n_iters:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_{it:04d}.pt")
            torch.save(
                {
                    "iter": it,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )
            print(f"  [checkpoint saved: {ckpt_path}]")

    # ── Save final artefacts ──────────────────────────────────────────
    print("\nSaving final artefacts...")
    final_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(
        {
            "iter": args.n_iters,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        final_path,
    )
    print(f"  Saved {final_path}")

    history_path = os.path.join(args.save_dir, "history.npz")
    np.savez(
        history_path,
        iter=np.array(history["iter"]),
        data_loss=np.array(history["data_loss"]),
        mse_vp=np.array(history["mse_vp"]),
        mse_vs=np.array(history["mse_vs"]),
        mse_rho=np.array(history["mse_rho"]),
    )
    print(f"  Saved {history_path}")

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Encoder-Decoder FWI training loop"
    )
    parser.add_argument(
        "--n_iters", type=int, default=350,
        help="Number of training iterations (default: 350)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0025,
        help="Learning rate for Adam (default: 0.0025)",
    )
    parser.add_argument(
        "--time_downsample", type=int, default=4,
        help="Downsample factor along the time axis (default: 4)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10,
        help="Print loss every N iterations (default: 10)",
    )
    parser.add_argument(
        "--save_interval", type=int, default=50,
        help="Save checkpoint every N iterations (default: 50)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints",
        help="Directory for checkpoints (default: checkpoints/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
