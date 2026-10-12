"""One residual bidirectional Mamba3 block of a CardioHydra trunk."""

import torch
import torch.nn as nn

from neural_methods.model.physmamba3.bi_mamba3 import BiMamba3


class MambaBlock(nn.Module):
    """``x + BiMamba3(LayerNorm(x))`` over ``(B, T, D)``.

    The CardioHydra repository's block ran two Mamba1 layers, concatenated
    the directions and projected them back; this one is PhysMamba3's
    ``BiMamba3``, which sums them.
    """

    def __init__(self, d_model: int, d_state: int, expand: int, headdim: int, dt_init: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = BiMamba3(d_model, d_state=d_state, expand=expand, headdim=headdim,
                            dt_init=dt_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))
