"""Tests for the encoder-decoder CNN (network.py)."""

import numpy as np
import pytest
import torch

from network import EncoderDecoder

# ---------------------------------------------------------------------------
# Shared constants matching the project dimensions
# ---------------------------------------------------------------------------
N_SHOTS = 28
NREC = 223
NT_DOWN = 1250
NZ = 150
NX = 294


def _make_model():
    """Instantiate EncoderDecoder with random initial models."""
    rng = np.random.RandomState(42)
    initial_vp = rng.uniform(1500, 3000, (NZ, NX)).astype(np.float32)
    initial_vs = rng.uniform(900, 1700, (NZ, NX)).astype(np.float32)
    initial_rho = rng.uniform(1800, 2300, (NZ, NX)).astype(np.float32)
    model = EncoderDecoder(N_SHOTS, NX, NZ, initial_vp, initial_vs, initial_rho)
    return model


# ---------------------------------------------------------------------------
# Test 1: output shapes
# ---------------------------------------------------------------------------
def test_encoder_decoder_output_shapes():
    """Each output (vp, vs, rho) must be (1, 1, nz, nx)."""
    model = _make_model()
    model.eval()
    with torch.no_grad():
        vx = torch.randn(1, N_SHOTS, NREC, NT_DOWN)
        vz = torch.randn(1, N_SHOTS, NREC, NT_DOWN)
        vp, vs, rho = model(vx, vz)

    assert vp.shape == (1, 1, NZ, NX), f"vp shape {vp.shape}"
    assert vs.shape == (1, 1, NZ, NX), f"vs shape {vs.shape}"
    assert rho.shape == (1, 1, NZ, NX), f"rho shape {rho.shape}"


# ---------------------------------------------------------------------------
# Test 2: with zeroed weights the residual is zero -> output == clamped init
# ---------------------------------------------------------------------------
def test_output_includes_initial_model():
    """When all trainable weights are zero the CNN residual is zero,
    so the output must equal the clamped initial model."""
    model = _make_model()
    # Zero every parameter
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    model.eval()

    with torch.no_grad():
        vx = torch.randn(1, N_SHOTS, NREC, NT_DOWN)
        vz = torch.randn(1, N_SHOTS, NREC, NT_DOWN)
        vp, vs, rho = model(vx, vz)

    # Expected: clamped initial models
    expected_vp = torch.clamp(model.initial_vp, 1500.0, 3000.0)
    expected_vs = torch.clamp(model.initial_vs, 900.0, 1700.0)
    expected_rho = torch.clamp(model.initial_rho, 1800.0, 2300.0)

    torch.testing.assert_close(vp, expected_vp.unsqueeze(0).unsqueeze(0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(vs, expected_vs.unsqueeze(0).unsqueeze(0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rho, expected_rho.unsqueeze(0).unsqueeze(0), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Kaiming init produces non-zero weights
# ---------------------------------------------------------------------------
def test_kaiming_init_nonzero():
    """After Kaiming initialisation every conv/linear weight tensor must be
    non-zero (i.e. the init actually ran)."""
    model = _make_model()
    for name, param in model.named_parameters():
        if "weight" in name:
            assert param.abs().sum().item() > 0.0, (
                f"Parameter {name} is all zeros after Kaiming init"
            )
