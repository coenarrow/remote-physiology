"""FactorizePhys: matrix factorization as multidimensional attention for rPPG.

Joshi, Agaian and Cho, "FactorizePhys: Matrix Factorization for
Multidimensional Attention in Remote Physiological Sensing", NeurIPS 2024.

The architecture is the published ``FSAM_Res`` one: a 3-D convolutional
feature extractor over the temporal difference of the clip, a head whose
voxel embeddings are multiplied by the factorized attention mask of
:mod:`neural_methods.model.factorizephys.features_factorization_module` and added back to themselves,
and a single readout plane. At ``in_channels=3`` and the defaults it is that
network layer for layer; the differences from the upstream file are the ones
this contract asks of every model:

- the stem takes ``in_channels`` inputs instead of the 1 / 3 / 4 if-chain,
  through one ``InstanceNorm3d(in_channels)`` (the norm carries no
  parameters, so nothing is added or renamed);
- the in-network ``torch.diff`` keeps a zero frame appended, so a clip of T
  frames leaves T rows. Upstream repeated the last frame in its trainer
  before the forward, which is the same arithmetic (``x_T - x_T = 0``) and
  the same convention as ``src.frame_transforms.diff_normalized``;
- the ``frames``, ``md_config``, ``device`` and ``debug`` arguments are gone:
  the window is read off the input, the published factorisation values are
  constructor defaults, modules follow their parameters, and the debug
  branches went with the rest of the legacy code. ``MD_INFERENCE: True`` in
  the paper's config, so FSAM ran in eval too and there is nothing left to
  switch on;
- the readout returns one trace, ``(B, 1, T)``, and gains a bias (upstream
  had none) because the trainer seeds each readout's bias with its trace's
  physiological prior. A multi-signal run is one complete copy of this
  network per trace (``MultiTraceModel``), never a widened readout;
- the approximation error of the factorisation is no longer returned: the
  upstream trainer only logged it, never adding it to the loss.

Any window length is accepted as it is: nothing in the network strides or
pools in time. Frames may be any size from ``MIN_FRAME``: the head's valid
convolutions take the feature map from 13x13 down to a point, so the
extractor's output is resampled spatially to 13x13 before it. At the
paper's 72x72 frames the extractor already emits 13x13 and the pool is
skipped, so the paper path is the published forward pass.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` in,
``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by the
trainer. All reshaping is einops.
"""

import torch
import torch.nn as nn

from neural_methods.model.factorizephys.bvp_head import BVPHead
from neural_methods.model.factorizephys.rppg_feature_extractor import RPPGFeatureExtractor
from neural_methods.model._shared_modules.utils import require_min_frame

#: The published filter counts of the three stages.
FILTERS = [8, 12, 16]

#: What the head's chain of valid 3x3x3 convolutions consumes: three in the
#: conv block and three in the final layer, two pixels each, ending at 1x1.
HEAD_SPATIAL = 13

#: The extractor is five valid convolutions, two of them strided: a frame
#: runs 72 -> 35 -> 33 -> 31 -> 15 -> 13, and 23 -> 11 -> 9 -> 7 -> 3 -> 1 is
#: the smallest frame it leaves anything of.
MIN_FRAME = 23


class FactorizePhys(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.1, use_fsam: bool = True):
        """Definition of FactorizePhys.

        Args:
          in_channels: the number of input channels. Default: 3.
          dropout: the published DROP_RATE. Default: 0.1.
          use_fsam: run the factorized attention module, the ablation the
            paper reports. Default: True, the FSAM_Res path.
        """
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.InstanceNorm3d(in_channels)
        self.rppg_feature_extractor = RPPGFeatureExtractor(in_channels, FILTERS, dropout_rate=dropout)
        self.rppg_head = BVPHead(FILTERS, HEAD_SPATIAL, dropout_rate=dropout, use_fsam=use_fsam)

    def output_layers(self):
        """The activation-free readout of this one copy."""
        return self.rppg_head.output_layers()

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``."""
        height, width = x.shape[3:]
        require_min_frame("FactorizePhys", MIN_FRAME, height, width)

        # The temporal difference the network is built on, one frame short of
        # the window; the zero frame puts it back, as upstream's repeated last
        # frame did.
        x = torch.diff(x, dim=2)
        x = torch.cat([x, torch.zeros_like(x[:, :, :1])], dim=2)
        x = self.norm(x)

        return self.rppg_head(self.rppg_feature_extractor(x))
