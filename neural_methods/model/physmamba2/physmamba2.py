"""PhysMamba2: PhysMamba with Mamba2 (SSD) layers in place of Mamba1.

Not a published network. It is PhysMamba (Luo et al.,
https://doi.org/10.48550/arXiv.2409.12031) with one substitution: the
bidirectional Mamba1 inside every temporal-difference block becomes a
bidirectional Mamba2 (Dao & Gu, https://doi.org/10.48550/arXiv.2405.21060).
The stem, the slow and fast streams, the lateral fusion, the head, the input
normalisation and the any-window, any-frame stages are PhysMamba's own,
inherited, so a comparison of the two models is a comparison of the SSM layer
and nothing else.

The swap is minimal: Mamba2 keeps PhysMamba's state size, convolution width
and expansion. The one new constant is the head size Mamba2 splits its
``expand * dim`` inner channels into; 16 gives the slow stream (dim 64) eight
heads and the fast stream (dim 32) four.

CUDA only: see ``BiMamba2``.
"""

import math

from neural_methods.model.physmamba.physmamba import MIN_FRAME, PhysMamba
from neural_methods.model.physmamba2.bi_mamba2 import BiMamba2

__all__ = ["MIN_FRAME", "PhysMamba2"]

#: PhysMamba's ``MambaLayer`` defaults, kept so only the layer type changes.
D_STATE = 16
D_CONV = 4
EXPAND = 2

#: Channels per Mamba2 head; must divide ``EXPAND * dim`` of both streams.
HEADDIM = 16

#: The time step every head starts at. PhysMamba's ``MambaLayer`` re-initialises
#: its ``nn.Linear`` layers and zeroes their biases, Mamba1's ``dt_proj`` bias
#: among them, so the published network starts at softplus(0) = ln 2, a
#: short-memory SSM. Mamba2's ``dt_bias`` is a bare parameter that re-init never
#: reaches, and mamba_ssm's own draw from [0.001, 0.1] starts it 7 to 700 times
#: slower; this puts it where PhysMamba is.
DT_INIT = math.log(2)


class PhysMamba2(PhysMamba):
    def _build_ssm(self, channels):
        return BiMamba2(channels, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND,
                        headdim=HEADDIM, dt_init=DT_INIT)
