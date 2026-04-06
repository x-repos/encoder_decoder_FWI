# Encoder-Decoder FWI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement physics-guided encoder-decoder FWI for the STH toy model, faithfully reproducing Dhara & Sen (IEEE TGRS 2023).

**Architecture:** A PyTorch CNN maps shot gathers to velocity models. DENISE (MPI elastic FD solver) runs forward+adjoint to compute waveform misfit gradients. These gradients are injected into PyTorch's autograd via a custom Function to update CNN weights with Adam.

**Tech Stack:** PyTorch 2.11 (CUDA 12.8), DENISE (C + MPI), pyapi_denise, numpy, matplotlib, segyio

**Spec:** `docs/superpowers/specs/2026-04-05-encoder-decoder-fwi-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `sth_model.py` | Exists | STH true/initial model + acquisition (already validated) |
| `network.py` | Create | Encoder-decoder CNN from Table I |
| `denise_fwi.py` | Create | PyTorch-DENISE bridge (custom autograd, gradient I/O) |
| `train.py` | Create | Training loop (Adam, logging, checkpoints) |
| `notebook.ipynb` | Create | Experimentation and visualization |
| `tests/test_network.py` | Create | CNN unit tests |
| `tests/test_denise_fwi.py` | Create | DENISE bridge integration tests |

---

### Task 1: Verify DENISE forward modeling on STH model

Confirm DENISE can simulate the STH model and produce shot gathers before building the CNN.

**Files:**
- Modify: `sth_model.py` (no changes needed, already validated)
- Use: `pyapi_denise.py`, `bin/denise`

- [ ] **Step 1: Run DENISE forward on STH true model**

Create a quick test script that runs forward modeling. Run from project root:

```python
# test_denise_sth.py
import os
os.chdir('/home/x/Workspace/7-encoder_decoderFWI')

import pyapi_denise as api
from sth_model import create_true_model, create_acquisition, NX, NZ, DX

model = create_true_model()
src, rec = create_acquisition()

d = api.Denise('./', verbose=1)
d.save_folder = './outputs_sth/'
d.set_paths()

# STH grid parameters
d.NPROCX = 7
d.NPROCY = 2
d.PHYSICS = 1      # 2D-PSV elastic
d.FREE_SURF = 1    # Free surface on top
d.FW = 10           # PML width
d.QUELLART = 1     # Ricker wavelet
d.SNAP = 0          # No snapshots (faster)
d.TIME = 5.0        # 5s recording
d.DT = None         # Auto-compute from CFL

d.forward(model, src, rec, run_command='mpirun --mca coll ^hcoll -np 14')
```

Run: `MPLBACKEND=Agg uv run python test_denise_sth.py`
Expected: DENISE runs without errors, creates .su files in `outputs_sth/su/`

- [ ] **Step 2: Verify shot gathers are readable**

```python
# Append to test_denise_sth.py or run interactively
d.verbose = 0
shots_x = d.get_shots(keys=['_x'])
shots_y = d.get_shots(keys=['_y'])
print(f'Vx shots: {len(shots_x)}, shape: {shots_x[0].shape}')
print(f'Vy shots: {len(shots_y)}, shape: {shots_y[0].shape}')
# Expected: 28 shots each, shape (NT_out, 223)
```

Expected: 28 shots per component, shape approximately (5000, 223) or similar depending on DT/NDT.

- [ ] **Step 3: Plot a shot gather to visually verify**

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, shot, title in zip(axes, [shots_x[14], shots_y[14]], ['Vx', 'Vy']):
    vmax = 0.05 * np.max(np.abs(shot))
    ax.imshow(shot.T, cmap='Greys', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_title(f'{title} shot 15')
    ax.set_xlabel('Time samples')
    ax.set_ylabel('Receiver')
plt.tight_layout()
plt.savefig('sth_shot_test.png', dpi=150)
print('Saved sth_shot_test.png')
```

Expected: Seismograms showing direct wave, reflections from interface and anomalies.

- [ ] **Step 4: Record actual dimensions for CNN design**

```python
nt = shots_x[0].shape[0]
nrec = shots_x[0].shape[1]
nsrc = len(shots_x)
nt_down = nt // 4  # 4x time downsample
print(f'nsrc={nsrc}, nrec={nrec}, nt={nt}, nt_down={nt_down}')
print(f'CNN input per component: ({nsrc}, {nrec}, {nt_down})')
```

Write down these values — they determine the CNN linear layer sizes.

- [ ] **Step 5: Clean up and commit**

```bash
rm test_denise_sth.py
git add sth_model.py
git commit -m "feat: add STH model creation with validated acquisition geometry"
```

---

### Task 2: Encoder-Decoder CNN (`network.py`)

Build the CNN architecture from Table I of the paper. Key design: compute linear layer sizes dynamically from a dummy forward pass through the encoder, and crop decoder output to match (NZ, NX).

**Files:**
- Create: `network.py`
- Create: `tests/test_network.py`

- [ ] **Step 1: Write failing test for CNN output shapes**

```python
# tests/test_network.py
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

def test_encoder_decoder_output_shapes():
    from network import EncoderDecoder

    n_shots = 28
    nrec = 223
    nt_down = 1250  # approximate, adjust after Task 1
    nz, nx = 150, 294

    # Initial model arrays (constant values)
    init_vp = np.full((nz, nx), 1500.0, dtype=np.float32)
    init_vs = np.full((nz, nx), 900.0, dtype=np.float32)
    init_rho = np.full((nz, nx), 1800.0, dtype=np.float32)

    net = EncoderDecoder(n_shots, nx, nz, init_vp, init_vs, init_rho)

    vx = torch.randn(1, n_shots, nrec, nt_down)
    vz = torch.randn(1, n_shots, nrec, nt_down)

    vp, vs, rho = net(vx, vz)

    assert vp.shape == (1, 1, nz, nx), f"Vp shape {vp.shape} != (1, 1, {nz}, {nx})"
    assert vs.shape == (1, 1, nz, nx), f"Vs shape {vs.shape} != (1, 1, {nz}, {nx})"
    assert rho.shape == (1, 1, nz, nx), f"Rho shape {rho.shape} != (1, 1, {nz}, {nx})"


def test_output_includes_initial_model():
    """CNN output = residual + initial model. With zero weights, output should be near initial."""
    from network import EncoderDecoder

    n_shots = 28
    nrec = 223
    nt_down = 1250
    nz, nx = 150, 294

    init_vp = np.full((nz, nx), 1500.0, dtype=np.float32)
    init_vs = np.full((nz, nx), 900.0, dtype=np.float32)
    init_rho = np.full((nz, nx), 1800.0, dtype=np.float32)

    net = EncoderDecoder(n_shots, nx, nz, init_vp, init_vs, init_rho)

    # Zero out all weights so residual is ~0
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()

    vx = torch.randn(1, n_shots, nrec, nt_down)
    vz = torch.randn(1, n_shots, nrec, nt_down)
    vp, vs, rho = net(vx, vz)

    # Output should equal clamped initial model
    assert vp.min().item() >= 1500.0
    assert vs.min().item() >= 900.0
    assert rho.min().item() >= 1800.0


def test_kaiming_init_nonzero():
    from network import EncoderDecoder

    nz, nx = 150, 294
    init_vp = np.full((nz, nx), 1500.0, dtype=np.float32)
    init_vs = np.full((nz, nx), 900.0, dtype=np.float32)
    init_rho = np.full((nz, nx), 1800.0, dtype=np.float32)

    net = EncoderDecoder(28, nx, nz, init_vp, init_vs, init_rho)

    # Check that conv weights are non-zero (Kaiming init)
    has_nonzero = False
    for name, p in net.named_parameters():
        if 'weight' in name and p.dim() >= 2:
            if p.abs().sum().item() > 0:
                has_nonzero = True
                break
    assert has_nonzero, "All weights are zero after Kaiming init"
```

Run: `uv run python -m pytest tests/test_network.py -v`
Expected: FAIL — `network` module not found.

- [ ] **Step 2: Implement `network.py`**

```python
# network.py
"""
Encoder-decoder CNN for physics-guided elastic FWI.
Architecture from Table I of Dhara & Sen (IEEE TGRS 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> LeakyReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderHead(nn.Module):
    """One decoder branch (Vp, Vs, or rho). 4 upsampling blocks + final 1x1 convs."""

    def __init__(self, kernel_size, padding):
        super().__init__()
        channels = [64, 32, 16, 8]
        self.blocks = nn.ModuleList()
        for i in range(4):
            in_ch = channels[i]
            out_ch = channels[i + 1] if i < 3 else 8
            self.blocks.append(nn.Sequential(
                ConvBlock(in_ch, out_ch, kernel_size, padding),
                nn.UpsamplingBilinear2d(scale_factor=2.0),
            ))
        self.final = nn.Sequential(
            nn.Conv2d(8, 8, 1),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.final(x)


class EncoderDecoder(nn.Module):
    """
    Physics-guided encoder-decoder for elastic FWI.

    Input: multicomponent shot gathers (Vx, Vz)
    Output: Vp, Vs, rho model arrays (added to initial model, clamped)

    Args:
        n_shots: number of shots (input channels per component)
        nx, nz: model grid dimensions
        initial_vp, initial_vs, initial_rho: starting model arrays (nz, nx), numpy float32
    """

    # Clamp bounds for STH model
    VP_MIN, VP_MAX = 1500.0, 3000.0
    VS_MIN, VS_MAX = 900.0, 1700.0
    RHO_MIN, RHO_MAX = 1800.0, 2300.0

    def __init__(self, n_shots, nx, nz, initial_vp, initial_vs, initial_rho):
        super().__init__()
        self.nx = nx
        self.nz = nz

        # Store initial model as non-trainable buffers
        self.register_buffer('init_vp',
            torch.from_numpy(initial_vp).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('init_vs',
            torch.from_numpy(initial_vs).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer('init_rho',
            torch.from_numpy(initial_rho).float().unsqueeze(0).unsqueeze(0))

        # Input: transform shot gathers to single channel each
        self.vx_conv = nn.Conv2d(n_shots, 1, 3, stride=1, padding=1)
        self.vz_conv = nn.Conv2d(n_shots, 1, 3, stride=1, padding=1)

        # Encoder: 4 blocks
        self.encoder = nn.ModuleList([
            ConvBlock(2, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        ])
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(4)])

        # Compute flattened size dynamically
        self._enc_shape = None  # set during first forward
        self._flat_size = None
        self.fc1 = None  # lazy init
        self.fc2 = None

        # Decoder heads: Vp/Vs use 5x5 kernels, rho uses 3x3
        self.decoder_vp = DecoderHead(kernel_size=5, padding=2)
        self.decoder_vs = DecoderHead(kernel_size=5, padding=2)
        self.decoder_rho = DecoderHead(kernel_size=3, padding=1)

        # Compute decoder reshape target
        # After 4 upsamples of 2x, spatial dims multiply by 16
        self._dec_h = -(-nz // 16)  # ceil division
        self._dec_w = -(-nx // 16)
        self._dec_size = 64 * self._dec_h * self._dec_w

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for ReLU-family activations (ref [59])."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in',
                                        nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    def _build_fc_layers(self, flat_size):
        """Build linear layers once we know encoder output size."""
        device = next(self.parameters()).device
        self.fc1 = nn.Linear(flat_size, 8).to(device)
        self.fc2 = nn.Linear(8, self._dec_size).to(device)
        # Init the new layers
        for fc in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(fc.weight, a=0.1, mode='fan_in',
                                    nonlinearity='leaky_relu')
            nn.init.zeros_(fc.bias)

    def forward(self, vx, vz):
        """
        Args:
            vx: (batch, n_shots, nrec, nt) Vx shot gathers
            vz: (batch, n_shots, nrec, nt) Vz shot gathers

        Returns:
            vp, vs, rho: each (batch, 1, nz, nx) clamped model tensors
        """
        # Transform shot gathers to single channel
        x_vx = self.vx_conv(vx)  # (B, 1, nrec, nt)
        x_vz = self.vz_conv(vz)  # (B, 1, nrec, nt)

        # Concatenate
        x = torch.cat([x_vx, x_vz], dim=1)  # (B, 2, nrec, nt)

        # Encoder
        for conv, pool in zip(self.encoder, self.pools):
            x = pool(conv(x))

        # Flatten
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # Lazy init linear layers on first forward
        if self.fc1 is None:
            self._flat_size = x_flat.shape[1]
            self._enc_shape = x.shape[1:]
            self._build_fc_layers(self._flat_size)

        # Latent space
        z = self.fc1(x_flat)   # (B, 8)
        z = self.fc2(z)        # (B, dec_size)

        # Reshape for decoder
        z = z.view(B, 64, self._dec_h, self._dec_w)

        # Decode each parameter with independent head
        vp_res = self.decoder_vp(z)
        vs_res = self.decoder_vs(z)
        rho_res = self.decoder_rho(z)

        # Crop to (nz, nx) — decoder output may be slightly larger
        vp_res = vp_res[:, :, :self.nz, :self.nx]
        vs_res = vs_res[:, :, :self.nz, :self.nx]
        rho_res = rho_res[:, :, :self.nz, :self.nx]

        # Add initial model (residual learning) and clamp
        vp = torch.clamp(vp_res + self.init_vp, self.VP_MIN, self.VP_MAX)
        vs = torch.clamp(vs_res + self.init_vs, self.VS_MIN, self.VS_MAX)
        rho = torch.clamp(rho_res + self.init_rho, self.RHO_MIN, self.RHO_MAX)

        return vp, vs, rho
```

- [ ] **Step 3: Run tests**

Run: `uv run python -m pytest tests/test_network.py -v`
Expected: All 3 tests PASS. Note the actual `nt_down` value from Task 1 — if it differs from 1250, update the test.

- [ ] **Step 4: Verify on GPU**

```python
# Quick GPU check
import torch
import numpy as np
from network import EncoderDecoder

nz, nx = 150, 294
net = EncoderDecoder(28, nx, nz,
    np.full((nz, nx), 1500.0, dtype=np.float32),
    np.full((nz, nx), 900.0, dtype=np.float32),
    np.full((nz, nx), 1800.0, dtype=np.float32)).cuda()

vx = torch.randn(1, 28, 223, 1250, device='cuda')
vz = torch.randn(1, 28, 223, 1250, device='cuda')
vp, vs, rho = net(vx, vz)
print(f'GPU output: vp={vp.shape}, device={vp.device}')
print(f'Params: {sum(p.numel() for p in net.parameters()):,}')
```

Expected: Shapes correct, device=cuda:0, prints parameter count.

- [ ] **Step 5: Commit**

```bash
git add network.py tests/test_network.py
git commit -m "feat: add encoder-decoder CNN architecture from Table I"
```

---

### Task 3: DENISE-PyTorch Bridge (`denise_fwi.py`)

Build the interface between PyTorch and DENISE. The key component is `DeniseFWIFunction` — a custom `torch.autograd.Function` that injects DENISE's adjoint gradients into PyTorch's backward pass.

**Files:**
- Create: `denise_fwi.py`
- Create: `tests/test_denise_fwi.py`

- [ ] **Step 1: Write integration test for observed data generation**

```python
# tests/test_denise_fwi.py
import sys
import os
sys.path.insert(0, '.')
os.chdir('/home/x/Workspace/7-encoder_decoderFWI')

def test_generate_observed_data():
    from denise_fwi import DeniseInterface
    from sth_model import create_true_model, create_acquisition

    true_model = create_true_model()
    src, rec = create_acquisition()

    di = DeniseInterface(
        denise_root='./',
        save_folder='./outputs_sth_test/',
        nprocx=7, nprocy=2,
    )
    di.generate_observed_data(true_model, src, rec)

    # Should have created shot gather files
    shots_x, shots_y = di.load_observed_shots()
    assert len(shots_x) == 28
    assert len(shots_y) == 28
    assert shots_x[0].shape[1] == 223  # nrec
    print(f'PASS: {len(shots_x)} shots, shape {shots_x[0].shape}')
```

Run: `uv run python -m pytest tests/test_denise_fwi.py::test_generate_observed_data -v -s`
Expected: FAIL — module not found. (Will take ~minutes to run once implemented due to DENISE.)

- [ ] **Step 2: Implement `DeniseInterface` class**

```python
# denise_fwi.py
"""
Bridge between PyTorch and DENISE for physics-guided FWI.

DeniseInterface: wraps pyapi_denise for forward modeling and gradient computation.
DeniseFWIFunction: custom torch.autograd.Function that injects DENISE gradients.
"""

import os
import numpy as np
import torch
import pyapi_denise as api


class DeniseInterface:
    """Manages DENISE forward modeling and single-iteration FWI for gradient computation."""

    def __init__(self, denise_root='./', save_folder='./outputs_sth/',
                 nprocx=7, nprocy=2, time=5.0):
        self.denise_root = denise_root
        self.save_folder = save_folder
        self.nprocx = nprocx
        self.nprocy = nprocy
        self.nproc = nprocx * nprocy
        self.time = time

        self._init_denise()

    def _init_denise(self):
        """Initialize DENISE API with STH parameters."""
        self.d = api.Denise(self.denise_root, verbose=0)
        self.d.save_folder = self.save_folder
        self.d.set_paths()

        # STH model parameters
        self.d.NPROCX = self.nprocx
        self.d.NPROCY = self.nprocy
        self.d.PHYSICS = 1       # 2D-PSV
        self.d.FREE_SURF = 1     # Free surface on top
        self.d.FW = 10           # PML width
        self.d.QUELLART = 1      # Ricker wavelet
        self.d.SNAP = 0          # No snapshots
        self.d.TIME = self.time
        self.d.DT = None         # Auto CFL

        self.run_cmd = f'mpirun --mca coll ^hcoll -np {self.nproc}'

    def generate_observed_data(self, true_model, src, rec):
        """Run forward modeling on true model to create observed data.

        The seismograms are saved to save_folder/su/ and serve as
        the target DATA_DIR for FWI gradient computation.
        """
        self.d.forward(true_model, src, rec, run_command=self.run_cmd)

    def load_observed_shots(self):
        """Load shot gathers (Vx and Vy components).

        Returns:
            (shots_x, shots_y): lists of numpy arrays, each (nt, nrec)
        """
        shots_x = self.d.get_shots(keys=['_x'])
        shots_y = self.d.get_shots(keys=['_y'])
        return shots_x, shots_y

    def compute_gradient(self, model, src, rec):
        """Run 1-iteration FWI to get adjoint gradients.

        Args:
            model: api.Model with current Vp, Vs, rho
            src: api.Sources
            rec: api.Receivers

        Returns:
            (misfit, grad_vp, grad_vs, grad_rho): misfit scalar and gradient arrays (nz, nx)
        """
        # Configure for single-iteration FWI
        self.d.ITERMAX = 1
        self.d.fwi_stages = []
        self.d.add_fwi_stage(
            pro=0.0,        # Don't terminate early
            time_filt=0,    # No frequency filtering for STH
            fc_low=0.0,
            fc_high=0.0,
            lnorm=2,        # L2 norm
            stf=0,          # No source time function inversion
        )

        self.d.fwi(model, src, rec, run_command=self.run_cmd)

        # Read gradients from jacobian/
        grads = self.d.get_fwi_gradients()

        # Parse misfit from log file
        misfit = self._read_misfit()

        # Gradients come as list, typically [grad_vp, grad_vs, grad_rho]
        # The exact order depends on DENISE output naming
        grad_vp = grads[0] if len(grads) > 0 else np.zeros((model.nz, model.nx))
        grad_vs = grads[1] if len(grads) > 1 else np.zeros((model.nz, model.nx))
        grad_rho = grads[2] if len(grads) > 2 else np.zeros((model.nz, model.nx))

        return misfit, grad_vp, grad_vs, grad_rho

    def _read_misfit(self):
        """Read the latest misfit value from DENISE FWI log."""
        log_path = self.d.MISFIT_LOG_FILE
        try:
            data = np.loadtxt(log_path)
            if data.ndim == 1:
                return float(data[0])
            return float(data[-1, 0])
        except Exception:
            return 0.0
```

- [ ] **Step 3: Run observed data generation test**

Run: `uv run python -m pytest tests/test_denise_fwi.py::test_generate_observed_data -v -s`
Expected: PASS (takes 1-5 minutes for DENISE to run 28 shots).

- [ ] **Step 4: Write test for gradient computation**

Add to `tests/test_denise_fwi.py`:

```python
def test_compute_gradient():
    from denise_fwi import DeniseInterface
    from sth_model import create_true_model, create_initial_model, create_acquisition

    true_model = create_true_model()
    init_model = create_initial_model()
    src, rec = create_acquisition()

    di = DeniseInterface(
        denise_root='./',
        save_folder='./outputs_sth_test/',
        nprocx=7, nprocy=2,
    )

    # Observed data must already exist from previous test
    # Compute gradient using initial model (which differs from true)
    misfit, gvp, gvs, grho = di.compute_gradient(init_model, src, rec)

    print(f'Misfit: {misfit}')
    print(f'Grad Vp: shape={gvp.shape}, max={np.abs(gvp).max():.6f}')
    print(f'Grad Vs: shape={gvs.shape}, max={np.abs(gvs).max():.6f}')
    print(f'Grad Rho: shape={grho.shape}, max={np.abs(grho).max():.6f}')

    assert misfit > 0, "Misfit should be positive for wrong model"
    assert np.abs(gvp).max() > 0, "Vp gradient should be non-zero"
```

Run: `uv run python -m pytest tests/test_denise_fwi.py::test_compute_gradient -v -s`
Expected: PASS — non-zero misfit and gradients.

- [ ] **Step 5: Implement `DeniseFWIFunction` (custom autograd)**

Add to `denise_fwi.py`:

```python
class DeniseFWIFunction(torch.autograd.Function):
    """Custom autograd function that bridges PyTorch and DENISE.

    Forward: takes Vp/Vs/rho tensors, runs DENISE 1-iter FWI, returns misfit.
    Backward: returns DENISE adjoint gradients as upstream grad for the model tensors.
    """

    @staticmethod
    def forward(ctx, vp, vs, rho, denise_interface, src, rec, dx):
        # Detach and convert to numpy
        vp_np = vp.detach().cpu().squeeze().numpy()
        vs_np = vs.detach().cpu().squeeze().numpy()
        rho_np = rho.detach().cpu().squeeze().numpy()

        # Create DENISE model
        model = api.Model(vp_np, vs_np, rho_np, dx)

        # Run 1-iter FWI to get gradients
        misfit, grad_vp, grad_vs, grad_rho = denise_interface.compute_gradient(
            model, src, rec
        )

        # Store gradients for backward pass
        ctx.save_for_backward(
            torch.from_numpy(grad_vp.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(grad_vs.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(grad_rho.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        )
        ctx.device = vp.device

        return torch.tensor(misfit, dtype=torch.float32, device=vp.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad_vp, grad_vs, grad_rho = ctx.saved_tensors

        # Move gradients to the same device as the model tensors
        grad_vp = grad_vp.to(ctx.device) * grad_output
        grad_vs = grad_vs.to(ctx.device) * grad_output
        grad_rho = grad_rho.to(ctx.device) * grad_output

        # Return gradients for (vp, vs, rho, denise_interface, src, rec, dx)
        return grad_vp, grad_vs, grad_rho, None, None, None, None
```

- [ ] **Step 6: Write test for autograd gradient flow**

Add to `tests/test_denise_fwi.py`:

```python
def test_gradient_flows_through_cnn():
    """Verify that DENISE gradients backprop through the CNN."""
    import torch
    from network import EncoderDecoder
    from denise_fwi import DeniseInterface, DeniseFWIFunction
    from sth_model import (create_true_model, create_initial_model,
                           create_acquisition, NX, NZ, DX)

    init_model = create_initial_model()
    src, rec = create_acquisition()

    di = DeniseInterface(
        denise_root='./',
        save_folder='./outputs_sth_test/',
        nprocx=7, nprocy=2,
    )

    # Load shot gathers as CNN input
    shots_x, shots_y = di.load_observed_shots()
    nt_down = shots_x[0].shape[0] // 4

    # Downsample and stack into tensors
    vx_list = [s[::4, :] for s in shots_x]
    vy_list = [s[::4, :] for s in shots_y]
    vx_tensor = torch.from_numpy(np.stack(vx_list)).float().unsqueeze(0)
    vy_tensor = torch.from_numpy(np.stack(vy_list)).float().unsqueeze(0)

    # Create CNN
    net = EncoderDecoder(len(shots_x), NX, NZ,
                         init_model.vp, init_model.vs, init_model.rho)

    # Forward through CNN
    vp, vs, rho = net(vx_tensor, vy_tensor)

    # Forward through DENISE (get misfit + store gradients)
    misfit = DeniseFWIFunction.apply(vp, vs, rho, di, src, rec, DX)

    # Backward
    misfit.backward()

    # Check that CNN parameters got gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in net.parameters())
    print(f'Misfit: {misfit.item()}, has_grad: {has_grad}')
    assert has_grad, "CNN parameters should have non-zero gradients"
```

Run: `uv run python -m pytest tests/test_denise_fwi.py::test_gradient_flows_through_cnn -v -s`
Expected: PASS (slow — runs DENISE). This is the critical integration test.

- [ ] **Step 7: Commit**

```bash
git add denise_fwi.py tests/test_denise_fwi.py
git commit -m "feat: add DENISE-PyTorch bridge with custom autograd function"
```

---

### Task 4: Training Loop (`train.py`)

Assemble all components into the training pipeline.

**Files:**
- Create: `train.py`

- [ ] **Step 1: Implement training script**

```python
# train.py
"""
Training loop for physics-guided encoder-decoder FWI.

Usage:
    python train.py [--n_iters 350] [--lr 0.0025] [--save_dir checkpoints]
"""

import os
import argparse
import numpy as np
import torch

import pyapi_denise as api
from sth_model import (create_true_model, create_initial_model,
                       create_acquisition, NX, NZ, DX)
from network import EncoderDecoder
from denise_fwi import DeniseInterface, DeniseFWIFunction


def prepare_inputs(denise_interface, downsample=4):
    """Load observed shot gathers and prepare as CNN input tensors.

    Returns:
        vx_tensor, vy_tensor: each (1, n_shots, nrec, nt_down) float32
    """
    shots_x, shots_y = denise_interface.load_observed_shots()

    vx_list = [s[::downsample, :] for s in shots_x]
    vy_list = [s[::downsample, :] for s in shots_y]

    vx = torch.from_numpy(np.stack(vx_list)).float().unsqueeze(0)
    vy = torch.from_numpy(np.stack(vy_list)).float().unsqueeze(0)

    return vx, vy


def compute_model_loss(net, vx, vy, true_model):
    """Compute MSE between predicted and true model (for monitoring only)."""
    with torch.no_grad():
        vp, vs, rho = net(vx, vy)
        vp_true = torch.from_numpy(true_model.vp).float().to(vp.device)
        vs_true = torch.from_numpy(true_model.vs).float().to(vp.device)
        rho_true = torch.from_numpy(true_model.rho).float().to(vp.device)

        mse_vp = ((vp.squeeze() - vp_true) ** 2).mean().item()
        mse_vs = ((vs.squeeze() - vs_true) ** 2).mean().item()
        mse_rho = ((rho.squeeze() - rho_true) ** 2).mean().item()

    return mse_vp, mse_vs, mse_rho


def train(n_iters=350, lr=0.0025, save_dir='checkpoints',
          log_interval=10, save_interval=50):
    os.makedirs(save_dir, exist_ok=True)

    # Models
    true_model = create_true_model()
    init_model = create_initial_model()
    src, rec = create_acquisition()

    # DENISE interface
    di = DeniseInterface(
        denise_root='./',
        save_folder='./outputs_sth/',
        nprocx=7, nprocy=2,
    )

    # Generate observed data (skip if already exists)
    obs_dir = os.path.join(di.save_folder, 'su')
    su_files = [f for f in os.listdir(obs_dir) if f.endswith('.su')] if os.path.exists(obs_dir) else []
    if len(su_files) == 0:
        print('Generating observed data...')
        di.generate_observed_data(true_model, src, rec)
    else:
        print(f'Using existing observed data ({len(su_files)} files)')

    # Prepare CNN inputs
    vx, vy = prepare_inputs(di, downsample=4)
    print(f'CNN input: vx={vx.shape}, vy={vy.shape}')

    # Build CNN
    net = EncoderDecoder(vx.shape[1], NX, NZ,
                         init_model.vp, init_model.vs, init_model.rho)

    # Warm-up forward to initialize lazy linear layers
    with torch.no_grad():
        _ = net(vx, vy)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Logging
    history = {'iter': [], 'data_loss': [], 'mse_vp': [], 'mse_vs': [], 'mse_rho': []}

    print(f'\nStarting training: {n_iters} iterations, lr={lr}')
    print(f'Params: {sum(p.numel() for p in net.parameters()):,}')
    print('-' * 70)

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        # Forward through CNN
        vp, vs, rho = net(vx, vy)

        # Forward through DENISE -> misfit + adjoint gradients
        data_loss = DeniseFWIFunction.apply(vp, vs, rho, di, src, rec, DX)

        # Backward through CNN
        data_loss.backward()

        # Adam step
        optimizer.step()

        # Logging
        dl = data_loss.item()
        if it % log_interval == 0 or it == 1:
            mse_vp, mse_vs, mse_rho = compute_model_loss(net, vx, vy, true_model)
            history['iter'].append(it)
            history['data_loss'].append(dl)
            history['mse_vp'].append(mse_vp)
            history['mse_vs'].append(mse_vs)
            history['mse_rho'].append(mse_rho)
            print(f'[{it:4d}/{n_iters}] data_loss={dl:.6f}  '
                  f'mse_vp={mse_vp:.4f}  mse_vs={mse_vs:.4f}  mse_rho={mse_rho:.4f}')

        # Save checkpoint
        if it % save_interval == 0:
            ckpt_path = os.path.join(save_dir, f'ckpt_{it:04d}.pt')
            torch.save({
                'iter': it,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, ckpt_path)
            print(f'  -> saved {ckpt_path}')

    # Save final
    np.savez(os.path.join(save_dir, 'history.npz'), **history)
    torch.save(net.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    print('\nTraining complete.')
    return net, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iters', type=int, default=350)
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=50)
    args = parser.parse_args()

    train(n_iters=args.n_iters, lr=args.lr, save_dir=args.save_dir,
          log_interval=args.log_interval, save_interval=args.save_interval)
```

- [ ] **Step 2: Smoke test — run 2 iterations**

Run: `uv run python train.py --n_iters 2 --log_interval 1 --save_interval 2`
Expected: Prints 2 iterations with decreasing (or at least changing) data_loss, saves checkpoint. Each iteration takes ~10-30s for DENISE.

- [ ] **Step 3: Verify gradients are updating the model**

```python
# Quick check: the CNN output after 2 iterations should differ from initial model
import torch
import numpy as np
from network import EncoderDecoder
from sth_model import create_initial_model, NX, NZ

init_model = create_initial_model()
state = torch.load('checkpoints/ckpt_0002.pt', weights_only=False)

net = EncoderDecoder(28, NX, NZ, init_model.vp, init_model.vs, init_model.rho)
net.load_state_dict(state['model_state_dict'])

print(f"History: {state['history']}")
print("Training is producing gradient updates." if len(state['history']['data_loss']) > 0 else "ERROR")
```

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat: add training loop with Adam optimizer and DENISE gradient injection"
```

---

### Task 5: Visualization Notebook (`notebook.ipynb`)

Create the experimentation notebook for running training and visualizing results.

**Files:**
- Create: `notebook.ipynb`

- [ ] **Step 1: Create notebook with setup and model visualization cells**

Create `notebook.ipynb` with these cells:

**Cell 1: Setup**
```python
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import os

os.chdir('/home/x/Workspace/7-encoder_decoderFWI')

from sth_model import create_true_model, create_initial_model, create_acquisition, NX, NZ, DX
from network import EncoderDecoder
from denise_fwi import DeniseInterface
from train import train, prepare_inputs, compute_model_loss
```

**Cell 2: Helper functions**
```python
def plot_models(vp, vs, rho, title_prefix='', vp_lim=None, vs_lim=None, rho_lim=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    extent = [0, NX*DX/1000, NZ*DX/1000, 0]
    for ax, arr, name, lim in zip(axes,
        [vp, vs, rho], ['Vp (km/s)', 'Vs (km/s)', 'Rho (g/cm3)'],
        [vp_lim, vs_lim, rho_lim]):
        kwargs = {}
        if lim: kwargs.update(vmin=lim[0], vmax=lim[1])
        im = ax.imshow(arr, cmap='RdBu_r', extent=extent, aspect='auto', **kwargs)
        ax.set_title(f'{title_prefix}{name}')
        ax.set_xlabel('x (km)'); ax.set_ylabel('z (km)')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
    plt.tight_layout()
```

**Cell 3: Visualize true and initial models (Fig. 3)**
```python
true_model = create_true_model()
init_model = create_initial_model()
src, rec = create_acquisition()

plot_models(true_model.vp/1000, true_model.vs/1000, true_model.rho/1000, 'True ')
plot_models(init_model.vp/1000, init_model.vs/1000, init_model.rho/1000, 'Initial ')
```

**Cell 4: Run training (or load checkpoint)**
```python
# Run training (takes ~1-3 hours for 350 iterations)
# net, history = train(n_iters=350, lr=0.0025, log_interval=10, save_interval=50)

# Or load from checkpoint:
# state = torch.load('checkpoints/final_model.pt')
```

**Cell 5: Convergence curve (Fig. 5)**
```python
history = np.load('checkpoints/history.npz')
fig, ax = plt.subplots(figsize=(8, 5))
iters = history['iter']
ax.semilogy(iters, history['data_loss'] / history['data_loss'][0], label='data loss')
ax.semilogy(iters, history['mse_vp'] / history['mse_vp'][0], label='model loss (Vp)')
ax.set_xlabel('Iterations'); ax.set_ylabel('Normalized Loss')
ax.legend(); ax.grid(True)
plt.title('Convergence (STH model)')
```

**Cell 6: Inverted model visualization (Fig. 4a)**
```python
# Load final model and predict
net = EncoderDecoder(28, NX, NZ, init_model.vp, init_model.vs, init_model.rho)
net.load_state_dict(torch.load('checkpoints/final_model.pt', weights_only=True))

di = DeniseInterface('./', './outputs_sth/')
vx, vy = prepare_inputs(di)

with torch.no_grad():
    vp, vs, rho = net(vx, vy)

plot_models(vp.squeeze().numpy()/1000, vs.squeeze().numpy()/1000,
            rho.squeeze().numpy()/1000, 'Inverted ')
```

- [ ] **Step 2: Commit**

```bash
git add notebook.ipynb
git commit -m "feat: add visualization notebook for STH model experiments"
```

---

### Task 6: End-to-end validation run

Run the full pipeline for a small number of iterations to verify everything works together.

**Files:** None (validation only)

- [ ] **Step 1: Run 10 iterations end-to-end**

```bash
uv run python train.py --n_iters 10 --log_interval 1 --save_interval 5
```

Expected: All 10 iterations complete without errors. Data loss should change between iterations (not stuck at same value).

- [ ] **Step 2: Verify convergence direction**

```python
import numpy as np
h = np.load('checkpoints/history.npz')
losses = h['data_loss']
print(f'Loss trend: {losses}')
# Data loss should generally decrease (may not be monotonic with Adam)
assert losses[-1] < losses[0] * 2, "Loss should not diverge"
print('PASS: loss is not diverging')
```

- [ ] **Step 3: Commit all remaining changes and push**

```bash
git add -A
git commit -m "feat: complete encoder-decoder FWI pipeline for STH model"
git push origin main
```
