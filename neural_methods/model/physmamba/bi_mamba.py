"""The bidirectional Mamba that replaces the old fork's ``bimamba=True``."""

import torch.nn as nn

from neural_methods.model.physmamba.mamba_backend import mamba_block


class BiMamba(nn.Module):
    """Bidirectional Mamba built from two vanilla Mamba blocks.

    forward direction processes the sequence as-is; backward direction
    processes the time-reversed sequence and un-reverses its output:
        y = fwd(x) + flip(bwd(flip(x)))
    x: (B, L, D) -> y: (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.fwd = mamba_block(d_model, d_state, d_conv, expand, **kwargs)
        self.bwd = mamba_block(d_model, d_state, d_conv, expand, **kwargs)

    def forward(self, x):
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)
