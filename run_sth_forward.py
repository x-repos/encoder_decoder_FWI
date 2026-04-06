"""
Task 1: Verify DENISE forward modeling on STH model.

Runs DENISE forward on the STH true model, loads resulting shot gathers,
prints dimensions, and saves a verification plot.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import matplotlib.pyplot as plt
import pyapi_denise as api
import sth_model

# ---- Create model and acquisition ----
model = sth_model.create_true_model()
src, rec = sth_model.create_acquisition()

print(f"Model shape: vp={model.vp.shape}, vs={model.vs.shape}, rho={model.rho.shape}")
print(f"Sources: {len(src)}, Receivers: {len(rec)}")

# ---- Configure DENISE ----
d = api.Denise('./', verbose=1)

d.NPROCX = 7
d.NPROCY = 2
d.PHYSICS = 1       # 2D-PSV elastic
d.FREE_SURF = 1     # free surface on top
d.FW = 10           # PML width in grid points
d.QUELLART = 1      # Ricker wavelet
d.SNAP = 0          # no snapshots (faster)
d.TIME = 5.0        # 5s recording
d.DT = None         # auto-compute from CFL
d.save_folder = './outputs_sth/'
d.set_paths()

print(f"\nDomain decomposition check:")
print(f"  NX=294 / NPROCX=7 = {294 // 7}")
print(f"  NY=150 / NPROCY=2 = {150 // 2}")

# ---- Run forward modeling ----
run_cmd = 'mpirun --mca coll ^hcoll -np 14'
print(f"\nRunning DENISE forward with: {run_cmd}")
d.forward(model, src, rec, run_command=run_cmd)

# ---- Load shot gathers ----
print("\n" + "="*60)
print("Loading shot gathers...")
print("="*60)

shots_x = d.get_shots(keys=['_x'])
shots_y = d.get_shots(keys=['_y'])

print(f"\n{'='*60}")
print(f"SHOT GATHER DIMENSIONS")
print(f"{'='*60}")
print(f"Vx component: {len(shots_x)} shots")
if len(shots_x) > 0:
    print(f"  Each shot shape (nt, nrec): {shots_x[0].shape}")
print(f"Vy component: {len(shots_y)} shots")
if len(shots_y) > 0:
    print(f"  Each shot shape (nt, nrec): {shots_y[0].shape}")
print(f"DT used: {d.DT}")
print(f"NT used: {d.NT}")
print(f"{'='*60}")

# ---- Verification plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

shot_idx = 0  # plot first shot

for ax, data, comp in zip(axes, [shots_x, shots_y], ['Vx', 'Vy']):
    shot = data[shot_idx]
    clip = 0.1 * np.abs(shot).max()
    ax.imshow(shot, aspect='auto', cmap='gray',
              vmin=-clip, vmax=clip,
              extent=[0, shot.shape[1], shot.shape[0] * d.DT, 0])
    ax.set_title(f'Shot {shot_idx+1} - {comp} ({shot.shape[0]}x{shot.shape[1]})')
    ax.set_xlabel('Receiver index')
    ax.set_ylabel('Time (s)')

plt.suptitle(f'STH Forward Model - Shot Gathers (DT={d.DT}, NT={d.NT})')
plt.tight_layout()
plt.savefig('sth_shot_test.png', dpi=150, bbox_inches='tight')
print(f"\nVerification plot saved to: sth_shot_test.png")
