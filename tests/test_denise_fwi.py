"""Integration tests for the DENISE-PyTorch bridge (denise_fwi.py).

These tests run the actual DENISE binary via MPI and take several minutes each.
Run with:
    uv run python -m pytest tests/test_denise_fwi.py -v -s
"""

import os
import sys

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sth_model
from denise_fwi import DeniseInterface


ROOT = os.path.join(os.path.dirname(__file__), "..")
OBS_FOLDER = os.path.join(ROOT, "outputs_sth_obs")
FWI_FOLDER = os.path.join(ROOT, "outputs_sth_fwi")


@pytest.fixture(scope="module")
def interface():
    """Shared DeniseInterface for the whole test module."""
    return DeniseInterface(
        root=ROOT,
        obs_folder=OBS_FOLDER,
        fwi_folder=FWI_FOLDER,
        verbose=1,
    )


@pytest.fixture(scope="module")
def acquisition():
    """Source and receiver geometry."""
    return sth_model.create_acquisition()


# --------------------------------------------------------------------------
# Test 1: Generate observed data from the true model
# --------------------------------------------------------------------------

def test_generate_observed_data(interface, acquisition):
    """Run DENISE forward on the true model and verify 28 Vx shots are loaded."""
    true_model = sth_model.create_true_model()
    src, rec = acquisition

    shots = interface.generate_observed_data(true_model, src, rec)

    assert len(shots) == 28, f"Expected 28 shots, got {len(shots)}"
    for i, shot in enumerate(shots):
        assert shot.ndim == 2, f"Shot {i} should be 2-D (nrec, nt)"
        assert shot.shape[0] == 223, f"Shot {i} nrec={shot.shape[0]}, expected 223"
        assert np.any(shot != 0.0), f"Shot {i} is all zeros"

    print(f"\n[PASS] Generated {len(shots)} observed shots, "
          f"each shape {shots[0].shape}")


# --------------------------------------------------------------------------
# Test 2: Compute gradient via single-iteration FWI
# --------------------------------------------------------------------------

def test_compute_gradient(interface, acquisition):
    """Run 1-iter FWI on the initial model; verify non-zero misfit and gradients."""
    initial_model = sth_model.create_initial_model()
    src, rec = acquisition

    misfit, grad_vp, grad_vs, grad_rho = interface.compute_gradient(
        initial_model.vp,
        initial_model.vs,
        initial_model.rho,
        sth_model.DX,
        src,
        rec,
    )

    # Misfit must be a positive number
    assert np.isfinite(misfit), f"Misfit is not finite: {misfit}"
    assert misfit > 0.0, f"Misfit should be > 0 for wrong model, got {misfit}"

    # Gradients must have the right shape and be non-zero
    expected_shape = (sth_model.NZ, sth_model.NX)
    for name, grad in [("vp", grad_vp), ("vs", grad_vs), ("rho", grad_rho)]:
        assert grad.shape == expected_shape, (
            f"grad_{name} shape {grad.shape} != {expected_shape}"
        )
        assert np.any(grad != 0.0), f"grad_{name} is all zeros"

    print(f"\n[PASS] misfit = {misfit:.6e}")
    print(f"  grad_vp  range: [{grad_vp.min():.4e}, {grad_vp.max():.4e}]")
    print(f"  grad_vs  range: [{grad_vs.min():.4e}, {grad_vs.max():.4e}]")
    print(f"  grad_rho range: [{grad_rho.min():.4e}, {grad_rho.max():.4e}]")
