"""
STH (Square-Triangle-Hourglass) toy model for encoder-decoder FWI.

Based on Fig. 3(a) of Dhara & Sen (IEEE TGRS 2023).
Grid: 294 (x) x 150 (z), dx = 10m
"""

import numpy as np
import pyapi_denise as api


# Grid parameters
NX = 294
NZ = 150
DX = 10.0  # meters

# Layer interface at 1.2 km depth (in grid points)
INTERFACE_Z = 120  # 1.2 km

# Top layer (0 - 1.2 km): low velocity
TOP_VP = 1500.0
TOP_VS = 900.0
TOP_RHO = 1800.0

# Bottom layer (1.2 - 1.5 km): high velocity
BOT_VP = 3000.0
BOT_VS = 1700.0
BOT_RHO = 2300.0

# Anomaly values (embedded in top layer)
ANOMALY_VP = 2000.0
ANOMALY_VS = 1200.0
ANOMALY_RHO = 2100.0


def _fill_squares(field, value, positions, size):
    """Embed square anomalies into field."""
    for (cx, cz) in positions:
        z1 = cz - size // 2
        z2 = cz + size // 2
        x1 = cx - size // 2
        x2 = cx + size // 2
        field[z1:z2, x1:x2] = value


def _fill_triangles(field, value, positions, size):
    """Embed upward-pointing triangle anomalies into field."""
    for (cx, cz) in positions:
        for row in range(size):
            # Triangle narrows toward the top
            half_width = int(size / 2 * (size - row) / size)
            z = cz + size // 2 - row
            x1 = cx - half_width
            x2 = cx + half_width
            if 0 <= z < field.shape[0]:
                field[z, max(0, x1):min(field.shape[1], x2)] = value


def _fill_hourglasses(field, value, positions, size):
    """Embed hourglass (bowtie/X) anomalies into field."""
    for (cx, cz) in positions:
        half = size // 2
        for row in range(size):
            # Two triangles mirrored: narrow at center, wide at top/bottom
            dist_from_center = abs(row - half)
            half_width = int(half * dist_from_center / half) + 1
            z = cz - half + row
            x1 = cx - half_width
            x2 = cx + half_width
            if 0 <= z < field.shape[0]:
                field[z, max(0, x1):min(field.shape[1], x2)] = value


def create_true_model():
    """Create the STH true velocity model.

    Returns:
        api.Model with vp, vs, rho arrays (nz, nx) and dx
    """
    # Two-layer elastic model
    vp = np.full((NZ, NX), TOP_VP, dtype=np.float32)
    vs = np.full((NZ, NX), TOP_VS, dtype=np.float32)
    rho = np.full((NZ, NX), TOP_RHO, dtype=np.float32)

    # Bottom layer (high velocity)
    vp[INTERFACE_Z:, :] = BOT_VP
    vs[INTERFACE_Z:, :] = BOT_VS
    rho[INTERFACE_Z:, :] = BOT_RHO

    # Anomaly positions (cx, cz) in grid points
    # 4 shapes arranged in top layer, evenly spaced
    shape_size = 20  # grid points per shape
    x_positions = [60, 130, 200, 250]

    # Row 1 and Row 2 in the top layer
    row1_z = 35
    row2_z = 75

    # Squares in Vp (4 squares, 2 rows x 2 columns)
    square_positions = [
        (x_positions[0], row1_z), (x_positions[1], row1_z),
        (x_positions[2], row2_z), (x_positions[3], row2_z),
    ]
    _fill_squares(vp, ANOMALY_VP, square_positions, shape_size)

    # Triangles in Vs (4 triangles, offset pattern)
    triangle_positions = [
        (x_positions[2], row1_z), (x_positions[3], row1_z),
        (x_positions[0], row2_z), (x_positions[1], row2_z),
    ]
    _fill_triangles(vs, ANOMALY_VS, triangle_positions, shape_size)

    # Hourglasses in rho (4 hourglasses, different pattern)
    hourglass_positions = [
        (x_positions[0], row1_z + shape_size // 2),
        (x_positions[3], row1_z + shape_size // 2),
        (x_positions[1], row2_z + shape_size // 2),
        (x_positions[2], row2_z + shape_size // 2),
    ]
    _fill_hourglasses(rho, ANOMALY_RHO, hourglass_positions, shape_size)

    return api.Model(vp, vs, rho, DX)


def create_initial_model():
    """Create the initial (starting) model — flat layers, no anomalies.

    Returns:
        api.Model with vp, vs, rho arrays (nz, nx) and dx
    """
    vp = np.full((NZ, NX), TOP_VP, dtype=np.float32)
    vs = np.full((NZ, NX), TOP_VS, dtype=np.float32)
    rho = np.full((NZ, NX), TOP_RHO, dtype=np.float32)

    # Bottom layer (high velocity)
    vp[INTERFACE_Z:, :] = BOT_VP
    vs[INTERFACE_Z:, :] = BOT_VS
    rho[INTERFACE_Z:, :] = BOT_RHO

    return api.Model(vp, vs, rho, DX)


def create_acquisition():
    """Create source and receiver geometry for STH model.

    Returns:
        (api.Sources, api.Receivers)
    """
    # Receivers: 223, covering the surface
    # Model width = 2940m, use dx spacing, centered
    nrec = 223
    rec_spacing = DX  # 10m
    total_rec_span = (nrec - 1) * rec_spacing  # 2220m
    xrec1 = (NX * DX - total_rec_span) / 2  # center on model
    xrec = np.arange(nrec) * rec_spacing + xrec1
    yrec = np.full_like(xrec, 100.0)  # 0.1 km depth

    # Sources: 28, spacing 0.08 km = 80m
    nsrc = 28
    src_spacing = 80.0
    total_src_span = (nsrc - 1) * src_spacing  # 2160m
    xsrc1 = (NX * DX - total_src_span) / 2  # center on model
    xsrc = np.arange(nsrc) * src_spacing + xsrc1
    ysrc = np.full_like(xsrc, DX)  # 1 grid point below free surface

    rec = api.Receivers(xrec, yrec)
    src = api.Sources(xsrc, ysrc)
    src.src_type[:] = 1  # explosive source

    return src, rec


def get_grid_params():
    """Return grid parameters dict."""
    return {
        'NX': NX,
        'NZ': NZ,
        'dx': DX,
        'interface_z': INTERFACE_Z,
    }


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    model = create_true_model()
    model_init = create_initial_model()
    src, rec = create_acquisition()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for col, (name, true_arr, init_arr) in enumerate([
        ('Vp (m/s)', model.vp, model_init.vp),
        ('Vs (m/s)', model.vs, model_init.vs),
        ('Rho (kg/m3)', model.rho, model_init.rho),
    ]):
        for row, (arr, label) in enumerate([
            (true_arr, f'True {name}'),
            (init_arr, f'Initial {name}'),
        ]):
            ax = axes[row, col]
            extent = [0, NX * DX / 1000, NZ * DX / 1000, 0]
            im = ax.imshow(arr, cmap='RdBu_r', extent=extent, aspect='auto')
            if row == 0:
                ax.scatter(src.x / 1000, src.y / 1000, 3, color='w', label='src')
                ax.scatter(rec.x / 1000, rec.y / 1000, 1, color='m', label='rec')
            ax.set_title(label)
            ax.set_xlabel('x (km)')
            ax.set_ylabel('z (km)')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig('sth_model_preview.png', dpi=150)
    plt.show()
    print('Saved sth_model_preview.png')
