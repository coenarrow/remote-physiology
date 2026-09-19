"""PhysMamba3: PhysMamba with Mamba3 (SISO) layers in place of Mamba1.

Not a published network. It is PhysMamba (Luo et al.,
https://doi.org/10.48550/arXiv.2409.12031) with one substitution: the
bidirectional Mamba1 inside every temporal-difference block becomes a
bidirectional Mamba3 (Lahoti et al., https://doi.org/10.48550/arXiv.2603.15569),
the single-input single-output variant. The stem, the slow and fast streams,
the lateral fusion, the head, the input normalisation and the any-window,
any-frame stages are PhysMamba's own, inherited, so a comparison of the two
models, or of either with PhysMamba2, is a comparison of the SSM layer and
nothing else.

The swap is minimal: Mamba3 keeps PhysMamba's state size and expansion, and
PhysMamba2's head size, 16, which gives the slow stream (dim 64) eight heads
and the fast stream (dim 32) four. PhysMamba's convolution width has nowhere
to go: Mamba3 has no causal convolution. Everything else is Mamba3's default,
which puts rotary angles on half of the 16 state channels.

CUDA only: see ``BiMamba3``.
"""

from neural_methods.model.physmamba.physmamba import MIN_FRAME, PhysMamba
from neural_methods.model.physmamba3.bi_mamba3 import BiMamba3

__all__ = ["MIN_FRAME", "PhysMamba3"]

#: PhysMamba's ``MambaLayer`` defaults, kept so only the layer type changes.
D_STATE = 16
EXPAND = 2

#: Channels per Mamba3 head, PhysMamba2's; must divide ``EXPAND * dim`` of
#: both streams.
HEADDIM = 16


class PhysMamba3(PhysMamba):
    def _build_ssm(self, channels):
        return BiMamba3(channels, d_state=D_STATE, expand=EXPAND, headdim=HEADDIM)
