"""FSAM: the factorized self-attention module of FactorizePhys.

Joshi, Agaian and Cho, "FactorizePhys: Matrix Factorization for
Multidimensional Attention in Remote Physiological Sensing", NeurIPS 2024.

What survives here is the one path FactorizePhys takes: a 3-D non-negative
matrix factorisation (``MD_TYPE: NMF``) of the ``T_KAB`` transform, with
bases initialised afresh on every forward (``RAND_INIT``). The vector-
quantisation decomposition, the 1-D / 2-D / TSM variants, the two other
transforms, the online-updated persistent bases and the debug plumbing were
unreachable from FactorizePhys and are deleted with the rest of the legacy
code rather than carried as dead branches.

Two things the module no longer carries: a ``device`` (it follows its
parameters, and the bases follow the tensor being factorised) and the
approximation error it used to return beside the attention mask, which the
upstream trainer only logged.
"""

import math

import torch.nn as nn

from neural_methods.model.factorizephys.conv_relu_3d import ConvReLU3D
from neural_methods.model.factorizephys.nmf import NMF


class FeaturesFactorizationModule(nn.Module):
    """Voxel embeddings in, an attention mask of the same shape out.

    A 1x1x1 convolution aligns the embedding to ``align_channels`` and makes
    it non-negative, :class:`NMF` factorises it, and a second pair of 1x1x1
    convolutions puts it back at ``in_channels``.
    """

    def __init__(self, in_channels, align_channels, rank: int = 1,
                 splits: int = 1, steps: int = 3):
        super().__init__()
        self.pre_conv_block = nn.Sequential(
            nn.Conv3d(in_channels, align_channels, (1, 1, 1)),
            nn.ReLU(inplace=True))

        self.md_block = NMF(rank=rank, splits=splits, steps=steps)

        self.post_conv_block = nn.Sequential(
            ConvReLU3D(align_channels, align_channels),
            nn.Conv3d(align_channels, in_channels, 1, bias=False))

        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                fan_out = (module.kernel_size[0] * module.kernel_size[1]
                           * module.kernel_size[2] * module.out_channels)
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))

    def forward(self, x):
        x = self.pre_conv_block(x)
        return self.post_conv_block(self.md_block(x))
