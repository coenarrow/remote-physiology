"""The bidirectional Mamba3 of a PhysMamba3 temporal-difference block."""

import torch.nn as nn

try:
    from mamba_ssm import Mamba3
except ImportError:  # pragma: no cover - exercised only where the package is absent
    Mamba3 = None


class BiMamba3(nn.Module):
    """Bidirectional Mamba3 built from two ``mamba_ssm.Mamba3`` blocks.

    The same composition as PhysMamba's ``BiMamba``: the forward direction
    processes the sequence as-is, the backward direction the reversed
    sequence, un-reversed and summed:
        y = fwd(x) + flip(bwd(flip(x)))
    x: (B, L, D) -> y: (B, L, D)

    The blocks are the SISO Mamba3 (``is_mimo=False``), whose kernels are
    Triton, so this runs on CUDA only; there is no pure-PyTorch stand-in as
    ``MambaRef`` is for Mamba1. Mamba3 has no causal convolution, so unlike
    ``BiMamba2`` there is no ``d_conv`` and no ``causal_conv1d`` path.
    """

    def __init__(self, d_model, d_state, expand, headdim):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "PhysMamba3 needs mamba_ssm's Mamba3 (mamba-ssm >= 2.3.2 on CUDA); "
                "it has no pure-PyTorch fallback. PhysMamba runs without it.")
        if (expand * d_model) % headdim:
            raise ValueError(
                f"Mamba3 splits expand * d_model = {expand * d_model} channels into "
                f"heads of {headdim}, which does not divide it.")
        self.fwd = self._block(d_model, d_state, expand, headdim)
        self.bwd = self._block(d_model, d_state, expand, headdim)

    @staticmethod
    def _block(d_model, d_state, expand, headdim):
        return Mamba3(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                      is_mimo=False)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError(
                "PhysMamba3 runs on CUDA only: Mamba3's SISO kernels are Triton. "
                "Drop --no-gpu, or use PhysMamba, which has a CPU path.")
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)
