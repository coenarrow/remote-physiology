"""The bidirectional Mamba2 of a PhysMamba2 temporal-difference block."""

import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:  # pragma: no cover - exercised only where the package is absent
    Mamba2 = None

try:
    import causal_conv1d  # noqa: F401 - only probing that the fused conv loads
    HAS_CAUSAL_CONV1D = True
except ImportError:  # pragma: no cover - a missing or broken build
    HAS_CAUSAL_CONV1D = False


class BiMamba2(nn.Module):
    """Bidirectional Mamba2 built from two ``mamba_ssm.Mamba2`` blocks.

    The same composition as PhysMamba's ``BiMamba``: the forward direction
    processes the sequence as-is, the backward direction the reversed
    sequence, un-reversed and summed:
        y = fwd(x) + flip(bwd(flip(x)))
    x: (B, L, D) -> y: (B, L, D)

    Mamba2's SSD kernels are Triton, so this runs on CUDA only; there is no
    pure-PyTorch stand-in as ``MambaRef`` is for Mamba1. The fused
    conv-and-scan path needs ``causal_conv1d``; without it each block takes
    mamba_ssm's unfused path, the same arithmetic, slower.

    Two departures from mamba_ssm's defaults put the block where PhysMamba's
    Mamba1 is, so the SSD layer is the only difference between the models:

    - ``dt_init`` is the time step every head starts at, in place of the draw
      from [0.001, 0.1].
    - ``rmsnorm=False``: Mamba1 has no norm inside the block. Mamba2's gated
      RMSNorm rescales the branch to full size however small its weights are,
      so at ``MambaLayer``'s 0.02 re-init it starts at about 0.3 of its input
      where Mamba1 starts at about 0.003, and weight decay cannot shrink it.
      With the norm in, PhysMamba2 never learned ABP on the Neckflix
      interface; without it, it follows PhysMamba epoch for epoch.
    """

    def __init__(self, d_model, d_state, d_conv, expand, headdim, dt_init):
        super().__init__()
        if Mamba2 is None:
            raise ImportError(
                "PhysMamba2 needs mamba_ssm's Mamba2 (mamba-ssm >= 2.0 on CUDA); "
                "it has no pure-PyTorch fallback. PhysMamba runs without it.")
        if (expand * d_model) % headdim:
            raise ValueError(
                f"Mamba2 splits expand * d_model = {expand * d_model} channels into "
                f"heads of {headdim}, which does not divide it.")
        self.fwd = self._block(d_model, d_state, d_conv, expand, headdim, dt_init)
        self.bwd = self._block(d_model, d_state, d_conv, expand, headdim, dt_init)

    @staticmethod
    def _block(d_model, d_state, d_conv, expand, headdim, dt_init):
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                      headdim=headdim, dt_min=dt_init, dt_max=dt_init, rmsnorm=False,
                      use_mem_eff_path=HAS_CAUSAL_CONV1D)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError(
                "PhysMamba2 runs on CUDA only: Mamba2's SSD kernels are Triton. "
                "Drop --no-gpu, or use PhysMamba, which has a CPU path.")
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)
