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

    Two departures from mamba_ssm's defaults put the block where PhysMamba's
    Mamba1 is, so the Mamba3 layer is the only difference between the models:

    - ``dt_init`` is the time step every head starts at, in place of the draw
      from [0.001, 0.1].
    - B and C are unnormalised and unbiased, as Mamba1's are. Mamba3
      RMS-normalises both and adds biases that start at one, which has no
      switch, so the norms are replaced by identity and the biases start at
      zero (they stay trainable). Normalised, the branch starts at about 0.05
      of its input where Mamba1 starts at about 0.003, and PhysMamba3's ABP
      copy never learned on the Neckflix interface; like this, it follows
      PhysMamba epoch for epoch.
    """

    def __init__(self, d_model, d_state, expand, headdim, dt_init):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "PhysMamba3 needs mamba_ssm's Mamba3 (mamba-ssm >= 2.3.2 on CUDA); "
                "it has no pure-PyTorch fallback. PhysMamba runs without it.")
        if (expand * d_model) % headdim:
            raise ValueError(
                f"Mamba3 splits expand * d_model = {expand * d_model} channels into "
                f"heads of {headdim}, which does not divide it.")
        self.fwd = self._block(d_model, d_state, expand, headdim, dt_init)
        self.bwd = self._block(d_model, d_state, expand, headdim, dt_init)

    @staticmethod
    def _block(d_model, d_state, expand, headdim, dt_init):
        block = Mamba3(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                       dt_min=dt_init, dt_max=dt_init, is_mimo=False)
        block.B_norm = nn.Identity()
        block.C_norm = nn.Identity()
        nn.init.zeros_(block.B_bias)
        nn.init.zeros_(block.C_bias)
        return block

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError(
                "PhysMamba3 runs on CUDA only: Mamba3's SISO kernels are Triton. "
                "Drop --no-gpu, or use PhysMamba, which has a CPU path.")
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)
