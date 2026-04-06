"""Plot the current CNN-predicted model from the latest checkpoint.

Run in a separate terminal while training is going:
    MPLBACKEND=Agg uv run python plot_current.py
"""

import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from denise_fwi import DeniseInterface
from network import EncoderDecoder
from sth_model import NX, NZ, DX, create_initial_model, create_true_model
from train import prepare_shot_gathers

# Find latest checkpoint
ckpts = sorted(glob.glob("checkpoints/checkpoint_*.pt"))
if not ckpts:
    print("No checkpoints found. Wait for training to save one.")
    sys.exit(1)

ckpt_path = ckpts[-1]
print(f"Loading {ckpt_path}")

ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
iteration = ckpt["iter"]

# Build CNN and load weights
init_model = create_initial_model()
true_model = create_true_model()
net = EncoderDecoder(28, NX, NZ, init_model.vp, init_model.vs, init_model.rho)

di = DeniseInterface(verbose=0)
vx, vy = prepare_shot_gathers(di)

# Warm-up to init lazy layers, then load state
with torch.no_grad():
    _ = net(vx, vy)
net.load_state_dict(ckpt["model_state_dict"])

# Predict
with torch.no_grad():
    vp, vs, rho = net(vx, vy)

vp = vp.squeeze().numpy()
vs = vs.squeeze().numpy()
rho = rho.squeeze().numpy()

# Plot: true, inverted, difference
extent = [0, NX * DX / 1000, NZ * DX / 1000, 0]
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

for col, (name, inv, true, unit) in enumerate([
    ("Vp", vp, true_model.vp, "m/s"),
    ("Vs", vs, true_model.vs, "m/s"),
    ("Rho", rho, true_model.rho, "kg/m3"),
]):
    for row, (arr, title) in enumerate([
        (true, f"True {name}"),
        (inv, f"Inverted {name} (iter {iteration})"),
        (inv - true, f"Diff {name}"),
    ]):
        ax = axes[row, col]
        cmap = "seismic" if row == 2 else "RdBu_r"
        kwargs = {}
        if row == 2:
            vmax = max(abs(arr.min()), abs(arr.max()))
            kwargs = {"vmin": -vmax, "vmax": vmax}
        im = ax.imshow(arr, cmap=cmap, extent=extent, aspect="auto", **kwargs)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label=unit)

plt.suptitle(f"Checkpoint: iteration {iteration}", fontsize=14)
plt.tight_layout()
plt.savefig("current_model.png", dpi=150)
print(f"Saved current_model.png (iteration {iteration})")
