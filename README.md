# Physics-Guided Encoder-Decoder FWI

Reimplementation of **"Elastic Full-Waveform Inversion Using a Physics-Guided Deep Convolutional Encoder-Decoder"** (Dhara & Sen, IEEE TGRS 2023).

A PyTorch encoder-decoder CNN maps multicomponent seismic shot gathers to elastic model parameters (Vp, Vs, rho). The CNN output is fed to [DENISE](https://github.com/daniel-koehn/DENISE-Black-Edition) (elastic finite-difference PDE solver) which generates synthetic data. The adjoint-state gradient from DENISE is injected into PyTorch's autograd to update the CNN weights. Training is completely unsupervised.

```
Shot gathers (Vx, Vz)
        |
  [Encoder-Decoder CNN]   (GPU, PyTorch)
        |
  Vp, Vs, rho             (residual + starting model, clamped)
        |
  [DENISE 1-iter FWI]     (CPU, MPI)
        |
  adjoint gradient dE/dm
        |
  backprop: dE/dw = (dm/dw) * (dE/dm)
        |
  Adam update
```

## Setup

```bash
# Install dependencies
uv sync

# Register Jupyter kernel
uv run python -m ipykernel install --user --name denise-fwi --display-name "DENISE FWI (uv)"
```

Requires:
- DENISE compiled binary (included in `bin/`)
- MPI (`mpirun`, OpenMPI)
- CUDA GPU (tested on RTX 5090, CUDA 12.8)

## Project Structure

| File | Description |
|------|-------------|
| `sth_model.py` | STH (Square-Triangle-Hourglass) toy model creation |
| `network.py` | Encoder-decoder CNN (Table I from paper) |
| `denise_fwi.py` | PyTorch-DENISE bridge with custom autograd |
| `train.py` | Training loop (Adam, checkpointing, logging) |
| `notebook.ipynb` | Visualization and experimentation |
| `pyapi_denise.py` | DENISE Python API (from upstream) |

## Usage

### Train on STH model

```bash
uv run python train.py --n_iters 350 --lr 0.0025
```

Each iteration runs DENISE forward+adjoint (~25s on 14 cores), so 350 iterations takes ~2.5 hours.

### Options

```
--n_iters       Number of training iterations (default: 350)
--lr            Learning rate for Adam (default: 0.0025)
--save_dir      Checkpoint directory (default: checkpoints/)
--log_interval  Print loss every N iterations (default: 10)
--save_interval Save checkpoint every N iterations (default: 50)
```

### Visualize results

Open `notebook.ipynb` with the "DENISE FWI (uv)" kernel to plot models, shot gathers, convergence curves, and metrics.

## STH Model

Synthetic two-layer elastic model (294 x 150, dx=10m) with geometric anomalies:
- Squares in Vp, triangles in Vs, hourglasses in rho
- 28 shots, 223 receivers, Ricker wavelet 10 Hz, 5s recording
- Top layer: Vp=1500, Vs=900, rho=1800
- Bottom layer: Vp=3000, Vs=1700, rho=2300
- Anomaly values: Vp=2000, Vs=1200, rho=2100

## Reference

Dhara, A. and Sen, M.K., "Elastic Full-Waveform Inversion Using a Physics-Guided Deep Convolutional Encoder-Decoder," *IEEE Trans. Geosci. Remote Sens.*, vol. 61, 2023. DOI: 10.1109/TGRS.2023.3294427
