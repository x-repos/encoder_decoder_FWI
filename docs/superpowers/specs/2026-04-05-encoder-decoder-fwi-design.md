# Encoder-Decoder FWI: Design Specification

Faithful reimplementation of "Elastic Full-Waveform Inversion Using a Physics-Guided Deep Convolutional Encoder-Decoder" (Dhara & Sen, IEEE TGRS 2023) starting with the STH toy model.

## Architecture Overview

A PyTorch encoder-decoder CNN maps multicomponent shot gathers (Vx, Vz) to elastic model parameters (Vp, Vs, rho). The CNN output is fed to DENISE (CPU-based elastic FD solver) which generates synthetic seismic data. The misfit gradient from DENISE's adjoint-state method is injected back into PyTorch's autograd to update CNN weights. Training is completely unsupervised.

```
Shot gathers (Vx, Vz)
        |
  [Encoder-Decoder CNN]  (GPU, PyTorch)
        |
  Vp, Vs, rho  (added to starting model, clamped)
        |
  [DENISE FWI 1-iter]    (CPU, MPI x15)
        |
  d_synth  vs  d_obs  -->  misfit E
        |
  adjoint gradient dE/dm  (read from jacobian/)
        |
  backprop through CNN:  dE/dw = (dm/dw) * (dE/dm)
        |
  Adam update (lr=0.0025)
```

## Module Design

### 1. `sth_model.py` — STH Model & Acquisition

**Responsibilities:** Create the synthetic STH (Square-Triangle-Hourglass) velocity model and acquisition geometry.

**True model:**
- Grid: 294 (x) x 150 (z), dx = 10m
- Two elastic layers, interface at 1.2 km (z=120 grid points)
- Top layer (0-1.2 km): Vp=1500 m/s, Vs=900 m/s, rho=1800 kg/m3
- Bottom layer (1.2-1.5 km): Vp=3000 m/s, Vs=1700 m/s, rho=2300 kg/m3
- Anomaly values in top layer: Vp=2000, Vs=1200, rho=2100
- 4 square anomalies embedded in Vp field
- 4 triangle anomalies embedded in Vs field
- 4 hourglass anomalies embedded in rho field

**Initial model:**
- Same two-layer structure without geometric shapes
- Constant values per layer matching the background
- This serves as the "starting model" (SM) added to CNN output

**Acquisition:**
- 28 sources, spacing 80m, at surface (z=10m)
- 223 receivers, spacing 10m, at 0.1 km depth (z=100m), centered on model
- Source: Ricker wavelet, 10 Hz center frequency
- Recording: 5s, dt determined by CFL condition (~1ms)

**Interface:**
- `create_true_model()` -> `api.Model` (vp, vs, rho arrays + dx)
- `create_initial_model()` -> `api.Model`
- `create_acquisition()` -> `(api.Sources, api.Receivers)`
- `get_grid_params()` -> dict with NX, NZ, dx, dt, etc.

### 2. `network.py` — Encoder-Decoder CNN

**Responsibilities:** Define the convolutional encoder-decoder architecture exactly as specified in Table I of the paper.

**Input processing:**
- Conv2d(N_shots, 1, kernel=3x3, stride=1, padding=1) for Vx component
- Conv2d(N_shots, 1, kernel=3x3, stride=1, padding=1) for Vz component
- Concatenate along channel dim -> 2-channel feature map

**Encoder (4 blocks):**
- Block pattern: Conv2d(3x3, pad=1) -> BatchNorm2d(eps=1e-5, momentum=0.1) -> LeakyReLU(0.1) -> MaxPool2d(2)
- Channels: 2 -> 8 -> 16 -> 32 -> 64

**Latent space:**
- Flatten -> Linear(flattened_size, 8) -> Linear(8, 48640)
- Reshape to (64, h, w) matching decoder input spatial dims

**Decoder (3 independent heads):**
- Vp head: 4 blocks of Conv2d(5x5, pad=2) -> BN -> LeakyReLU(0.1) -> Upsample(2x, bilinear)
- Vs head: same as Vp (5x5 kernels, pad=2)
- rho head: 4 blocks of Conv2d(3x3, pad=1) -> BN -> LeakyReLU(0.1) -> Upsample(2x, bilinear)
- All heads: channels 64 -> 32 -> 16 -> 8 -> 8
- Final per head: Conv2d(8, 8, 1x1) -> Conv2d(8, 1, 1x1)

**Output:**
- CNN output = residual added to initial (starting) model
- Clamped to STH model bounds: Vp [1500, 3000] m/s, Vs [900, 1700] m/s, rho [1800, 2300] kg/m3

**Weight initialization:** Kaiming (He) initialization for ReLU-family activations, per reference [59] in paper.

**Interface:**
- `EncoderDecoder(n_shots, nx, nz, initial_vp, initial_vs, initial_rho)` constructor
- `forward(vx_gathers, vz_gathers)` -> (vp, vs, rho) tensors

### 3. `denise_fwi.py` — PyTorch-DENISE Bridge

**Responsibilities:** Bridge between PyTorch autograd and DENISE's adjoint gradient computation.

**Custom autograd function (`DeniseFWIFunction`):**
- `forward(ctx, vp, vs, rho, denise_wrapper)`:
  1. Detach tensors, move to CPU, convert to numpy
  2. Create `api.Model` and write binary files
  3. Run `d.fwi()` with ITERMAX=1
  4. Read misfit from DENISE log
  5. Read gradients via `d.get_fwi_gradients()` -> dE/dVp, dE/dVs, dE/drho
  6. Store gradients in ctx for backward
  7. Return scalar misfit as tensor
- `backward(ctx, grad_output)`:
  1. Return stored DENISE gradients as upstream gradient for vp, vs, rho

**DENISE wrapper class (`DeniseInterface`):**
- Initialize DENISE API with correct paths and parameters
- Configure for STH model: NPROCX=7, NPROCY=2, PHYSICS=1 (PSV)
- `generate_observed_data(true_model, src, rec)` — run forward on true model, save seismograms as target data
- `compute_gradient(model, src, rec)` — run 1-iter FWI, return (misfit, grad_vp, grad_vs, grad_rho)
- Handle file I/O: write model binaries, read gradient binaries, parse misfit log

**MPI configuration:** `mpirun --mca coll ^hcoll -np 15`

**DENISE config:** A new `.inp` file (`par/DENISE_STH.inp`) must be created for the STH model, derived from the existing Marmousi config but with updated grid (NX=294, NY=150), DH=10m, TIME=5.0s, appropriate DT, and source wavelet set to Ricker (QUELLART=1) at 10 Hz. The `denise_fwi.py` module will modify runtime parameters via the pyapi (NPROCX, NPROCY, MODE, ITERMAX, etc.).

### 4. `train.py` — Training Loop

**Responsibilities:** Orchestrate the full training pipeline.

**Setup:**
1. Create STH true model and initial model via `sth_model.py`
2. Generate observed data by running DENISE forward on true model (one-time)
3. Load shot gathers (Vx, Vz) as CNN input tensors
4. Downsample gathers by 4x along time axis
5. Initialize CNN with Kaiming weights
6. Set up Adam optimizer (lr=0.0025)

**Training iteration:**
1. Forward: CNN(shot_gathers) -> vp, vs, rho
2. Run DENISE 1-iter FWI -> misfit + gradients
3. Inject gradients into autograd backward pass
4. Adam step on CNN weights
5. Log data_loss (misfit) and model_loss (MSE vs true)

**Configuration:**
- Total iterations: ~350 (paper converges by then for STH)
- All 28 shots per iteration (no minibatching at this scale)
- Input shot gather dims after 4x downsample: (28, 223, ~1250)
- Checkpoint saving interval: configurable (e.g. every 50 iterations)

**Logging:**
- Print iteration, data_loss, model_loss per step
- Save convergence history to file for plotting

**Interface:**
- `train(config)` — main entry point with config dict
- `Config` with fields: n_iters, lr, checkpoint_dir, log_interval, etc.

### 5. `notebook.ipynb` — Experimentation & Visualization

**Sections:**
1. Setup — imports, DENISE init, STH model creation
2. Visualize true & initial models — reproduce Fig. 3(a,b)
3. Generate observed data — forward on true model, plot shot gathers
4. Run training — call `train()`, display convergence
5. Visualize results — inverted Vp/Vs/rho (Fig. 4a), convergence curve (Fig. 5), intermediate iterations (Fig. 18)
6. Metrics — MSE, SSIM, MAE tables (Tables II-IV)

Notebook calls into `.py` modules; no duplicated logic.

## File Structure

```
7-encoder_decoderFWI/
├── sth_model.py          # STH true/initial model + acquisition
├── network.py            # Encoder-decoder CNN (Table I)
├── denise_fwi.py         # PyTorch <-> DENISE bridge (custom autograd)
├── train.py              # Training loop (Adam, logging, checkpoints)
├── notebook.ipynb        # Experimentation & visualization
├── pyapi_denise.py       # Existing DENISE Python API (unchanged)
├── bin/                  # DENISE binaries (unchanged)
├── par/                  # DENISE configs + model data
│   └── start/            # Model binaries
├── outputs/              # DENISE runtime outputs
└── checkpoints/          # Saved CNN weights
```

## Key Paper Parameters (STH)

| Parameter | Value |
|-----------|-------|
| Grid | 294 x 150, dx=10m |
| Sources | 28, spacing ~80m |
| Receivers | 223, spacing ~10m |
| Source wavelet | Ricker, 10 Hz |
| Recording time | 5s |
| Time downsample | 4x |
| CNN input | (28, 223, ~1250) per component |
| Optimizer | Adam, lr=0.0025 |
| Iterations | ~350 |
| MPI | 14 cores (7x2) |
| DENISE physics | 2D-PSV (elastic) |
| FD order | 8 |

## Dependencies

All installed in uv environment:
- torch 2.11.0+cu128 (GPU: RTX 5090)
- numpy, matplotlib, segyio, ipykernel
- DENISE binary + pyapi_denise.py
- OpenMPI 4.1.9
