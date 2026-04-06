"""Encoder-Decoder CNN for elastic FWI.

Architecture from Table I of Dhara & Sen (IEEE TGRS 2023).
Maps multicomponent shot gathers (Vx, Vz) to elastic model parameters
(Vp, Vs, rho).
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ── helpers ────────────────────────────────────────────────────────────────

def _encoder_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d(3x3) -> BN -> LeakyReLU(0.1) -> MaxPool2d(2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=True),
        nn.LeakyReLU(0.1),
        nn.MaxPool2d(2),
    )


def _decoder_block_vp_vs(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d(5x5) -> BN -> LeakyReLU(0.1) -> UpsamplingBilinear2d(2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=True),
        nn.LeakyReLU(0.1),
        nn.UpsamplingBilinear2d(scale_factor=2.0),
    )


def _decoder_block_rho(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d(3x3) -> BN -> LeakyReLU(0.1) -> UpsamplingBilinear2d(2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1, affine=True),
        nn.LeakyReLU(0.1),
        nn.UpsamplingBilinear2d(scale_factor=2.0),
    )


def _make_decoder_head(block_fn) -> nn.Sequential:
    """4 up-blocks  (64->32->16->8->8)  +  two 1x1 convs  (8->8->1)."""
    return nn.Sequential(
        block_fn(64, 32),
        block_fn(32, 16),
        block_fn(16, 8),
        block_fn(8, 8),
        nn.Conv2d(8, 8, kernel_size=1),
        nn.Conv2d(8, 1, kernel_size=1),
    )


# ── main module ────────────────────────────────────────────────────────────

class EncoderDecoder(nn.Module):
    """Encoder-decoder CNN for elastic FWI (Dhara & Sen, IEEE TGRS 2023).

    Parameters
    ----------
    n_shots : int
        Number of shots (input channels per component).
    nx, nz : int
        Spatial grid dimensions of the output model.
    initial_vp, initial_vs, initial_rho : np.ndarray
        Starting models, each shape ``(nz, nx)``, dtype float32.
    """

    def __init__(
        self,
        n_shots: int,
        nx: int,
        nz: int,
        initial_vp: np.ndarray,
        initial_vs: np.ndarray,
        initial_rho: np.ndarray,
    ):
        super().__init__()
        self.nx = nx
        self.nz = nz

        # ---- non-trainable initial models (buffers) ----
        self.register_buffer(
            "initial_vp", torch.from_numpy(initial_vp.copy())
        )
        self.register_buffer(
            "initial_vs", torch.from_numpy(initial_vs.copy())
        )
        self.register_buffer(
            "initial_rho", torch.from_numpy(initial_rho.copy())
        )

        # ---- input convolutions (reduce n_shots -> 1 channel each) ----
        self.conv_vx = nn.Conv2d(
            n_shots, 1, kernel_size=3, stride=1, padding=1
        )
        self.conv_vz = nn.Conv2d(
            n_shots, 1, kernel_size=3, stride=1, padding=1
        )

        # ---- encoder ----
        self.encoder = nn.Sequential(
            _encoder_block(2, 8),
            _encoder_block(8, 16),
            _encoder_block(16, 32),
            _encoder_block(32, 64),
        )

        # ---- latent (flatten -> FC -> reshape) ----
        # Compute flat_size dynamically with a dummy forward pass.
        # Use the actual spatial dims the encoder will see (nrec, nt_down
        # are arbitrary at construction time -- we only need to know them
        # for flat_size).  We derive flat_size at the first real forward
        # call instead, *unless* a pair of dummy inputs is passed.
        # For a clean __init__ we compute decoder spatial dims analytically.
        self.dec_h = math.ceil(nz / 16)
        self.dec_w = math.ceil(nx / 16)
        self.dec_size = 64 * self.dec_h * self.dec_w

        # flat_size is data-dependent; will be set on first forward().
        self._flat_size: int | None = None
        self.fc1: nn.Linear | None = None
        self.fc2: nn.Linear | None = None

        # ---- decoder heads ----
        self.head_vp = _make_decoder_head(_decoder_block_vp_vs)
        self.head_vs = _make_decoder_head(_decoder_block_vp_vs)
        self.head_rho = _make_decoder_head(_decoder_block_rho)

        # ---- clamping bounds ----
        self.bounds = {
            "vp": (1500.0, 3000.0),
            "vs": (900.0, 1700.0),
            "rho": (1800.0, 2300.0),
        }

        # Apply Kaiming init to everything defined so far.
        self._apply_kaiming_init()

    # ------------------------------------------------------------------ #
    # weight initialisation
    # ------------------------------------------------------------------ #
    def _apply_kaiming_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # lazy creation of FC layers (first forward)
    # ------------------------------------------------------------------ #
    def _build_fc(self, flat_size: int):
        self._flat_size = flat_size
        device = self.conv_vx.weight.device
        self.fc1 = nn.Linear(flat_size, 8).to(device)
        self.fc2 = nn.Linear(8, self.dec_size).to(device)
        # Kaiming init for the new layers
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(
                m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
            )
            nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, vx: torch.Tensor, vz: torch.Tensor):
        """
        Parameters
        ----------
        vx, vz : Tensor of shape ``(batch, n_shots, nrec, nt_down)``

        Returns
        -------
        vp, vs, rho : Tensors of shape ``(batch, 1, nz, nx)``
        """
        batch = vx.shape[0]

        # ---- input processing ----
        x_vx = self.conv_vx(vx)  # (B, 1, nrec, nt_down)
        x_vz = self.conv_vz(vz)  # (B, 1, nrec, nt_down)
        x = torch.cat([x_vx, x_vz], dim=1)  # (B, 2, nrec, nt_down)

        # ---- encoder ----
        z = self.encoder(x)  # (B, 64, H', W')

        # ---- latent ----
        z_flat = z.reshape(batch, -1)
        if self._flat_size is None:
            self._build_fc(z_flat.shape[1])
        z_fc = self.fc2(self.fc1(z_flat))  # (B, dec_size)
        z_dec = z_fc.reshape(batch, 64, self.dec_h, self.dec_w)

        # ---- decoder heads ----
        out_vp = self.head_vp(z_dec)[:, :, : self.nz, : self.nx]
        out_vs = self.head_vs(z_dec)[:, :, : self.nz, : self.nx]
        out_rho = self.head_rho(z_dec)[:, :, : self.nz, : self.nx]

        # ---- residual + clamp ----
        init_vp = self.initial_vp.unsqueeze(0).unsqueeze(0)
        init_vs = self.initial_vs.unsqueeze(0).unsqueeze(0)
        init_rho = self.initial_rho.unsqueeze(0).unsqueeze(0)

        vp = torch.clamp(out_vp + init_vp, *self.bounds["vp"])
        vs = torch.clamp(out_vs + init_vs, *self.bounds["vs"])
        rho = torch.clamp(out_rho + init_rho, *self.bounds["rho"])

        return vp, vs, rho
