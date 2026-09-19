"""RhythmFormer: hierarchical temporal periodic transformer for rPPG.

Zou et al., "RhythmFormer: Extracting Patterned rPPG Signals based on
Hierarchical Temporal Periodic Transformer",
https://arxiv.org/abs/2402.12788 — the official implementation
(https://github.com/zizheng-guo/RhythmFormer), with its bi-level routing
attention helpers vendored from BiFormer
(https://github.com/rayleizhu/BiFormer).

The architecture is unchanged from the original: the same two-path fusion
stem over the frame and its four temporal differences, the same 1x4x4 patch
embedding, the same three TPT stages (temporal periodic transformer blocks at
2-, 4- and 8-frame patches with top-40 region routing), the same single-plane
readout. One thing differs from the published network: the stem takes
``in_channels`` inputs (the interface's channel count) instead of 3, and its
difference path takes ``4 * in_channels`` — the hard-coded ``12`` upstream is
four frame differences of three channels. The readout (``conv_block_last``)
stays a single plane, a bare ``Conv1d`` with no activation, so an
absolute-class signal like ABP can be predicted directly in mmHg; a
multi-signal run is one complete copy of this network per trace
(``MultiTraceModel``), never a widened readout on a shared trunk. The loss is
the trainer's: RhythmFormer's published criterion adds frequency-domain
cross-entropy and label-distribution terms to negative Pearson, and those
belong in the interface's ``LOSS`` block, not here.

On ``configs/original_model_config/rhythmformer_FS30_W5.33S5.33_RGB_PPG_H128W128.yaml``
(128x128 frames, 160-frame windows) the forward pass is the published one, bit for bit:
loaded from the same weights, this module and the pre-migration one return
identical tensors on a paper-shaped clip.

Any other frame size from 16x16 and any window length are accepted through
three adaptive stages, each an exact no-op at the paper's shape:

* the original read the token grid as ``view(N, D, 64, H//4, W//4)``, true
  only where the stem's arithmetic happens to land on ``H//4``. Here the grid
  is read off the stem output's actual shape with einops;
* the TPT stages patch time by 2, 4 and 8 frames (stride-2 time convolutions
  undone by 2x upsamples), so the window has to divide by 8. The patch
  embedding's output is average-pooled in time to the nearest multiple of 8
  and the prediction is linearly interpolated back to the window length;
  at 160 frames neither runs;
* the bi-level routing attention cuts the token grid into ``REGION_SPLIT``
  regions per spatial axis, which only tiles the grid when the region size
  divides it. The grid is replicate-padded up to a whole number of regions
  and the attention output cropped back, and the routing ``topk`` is clamped
  to the number of regions there actually are. At the paper's 8x8 grid the
  padding is empty and 40 of 320 regions are routed, so neither fires.

No parameter is added or resized. A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)``
raw frames in, ``(B, 1, T)`` out, the loss owned by the trainer. The input
preprocessing is the network's own first stage: the raw clip is z-scored
(``Standardize``, the toolbox's Standardized block, with the clip's own
statistics) before the stem, which is what the published network was fed.
All reshaping is einops.
"""

import torch
from einops import rearrange, reduce
from timm.layers import trunc_normal_
from torch import nn
from torch.nn import functional as F

from neural_methods.model._shared_modules.standardize import Standardize
from neural_methods.model._shared_modules.utils import nearest_multiple, require_min_frame
from neural_methods.model.rhythmformer.fusion_stem import FusionStem
from neural_methods.model.rhythmformer.tpt_block import TPTBlock

#: ``FusionStem``'s stride-2 convolution and stride-2 max pool.
STEM_SPATIAL_STRIDE = 4

#: The patch embedding's ``(1, 4, 4)`` kernel and stride.
PATCH_SPATIAL = 4

#: The smallest frame the stem and the patch embedding leave a 1x1 token grid
#: of; anything smaller tokenizes to nothing.
MIN_FRAME = STEM_SPATIAL_STRIDE * PATCH_SPATIAL

#: Regions per spatial axis in the bi-level routing attention: the region size
#: is the token grid divided by this, i.e. 2x2 on the paper's 8x8 grid.
REGION_SPLIT = 4


class RhythmFormer(nn.Module):
    """RhythmFormer as a clip backbone: ``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``.

    ``in_channels`` is the number of input channels (default 3); every other
    argument is a published hyperparameter at its published value.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_dim: int = 64,
        head_dim: int = 16,
        embed_dim=(64, 64, 64),
        mlp_ratios=(1.5, 1.5, 1.5),
        depth=(2, 2, 2),
        t_patchs=(2, 4, 8),
        topks=(40, 40, 40),
        side_dwconv: int = 3,
        drop_path_rate: float = 0.,
    ):
        super().__init__()

        self.in_channels = in_channels
        #: The deepest stage patches time by 8, so the window must divide by it.
        self.temporal_stride = max(t_patchs)

        # Input preprocessing: raw clip -> standardised clip
        self.input_norm = Standardize()
        self.fusion_stem = FusionStem(in_channels=in_channels, dim=stem_dim)
        self.patch_embedding = nn.Conv3d(
            stem_dim, embed_dim[0], kernel_size=(1, PATCH_SPATIAL, PATCH_SPATIAL),
            stride=(1, PATCH_SPATIAL, PATCH_SPATIAL))
        # The readout: one plane, bias kept (the trainer seeds it with the
        # trace's physiological prior) and deliberately no activation, so an
        # absolute-class signal is expressible in its own units.
        self.conv_block_last = nn.Conv1d(embed_dim[-1], 1, kernel_size=1, stride=1, padding=0)

        # -------------------------------------------------------------------
        self.stages = nn.ModuleList()
        nheads = [dim // head_dim for dim in embed_dim]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        for i in range(len(embed_dim)):
            stage = TPTBlock(dim=embed_dim[i],
                             depth=depth[i],
                             num_heads=nheads[i],
                             mlp_ratio=mlp_ratios[i],
                             drop_path=dp_rates[sum(depth[:i]):sum(depth[:i + 1])],
                             t_patch=t_patchs[i], topk=topks[i],
                             region_split=REGION_SPLIT, side_dwconv=side_dwconv
                             )
            self.stages.append(stage)
        # -------------------------------------------------------------------

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def output_layers(self):
        """The activation-free readout: the final 1x1 conv."""
        return (self.conv_block_last,)

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` raw clip -> ``(B, 1, T)``."""
        frames, height, width = x.shape[2:]
        require_min_frame("RhythmFormer", MIN_FRAME, height, width)

        x = self.fusion_stem(rearrange(self.input_norm(x), "b c t h w -> b t c h w"))
        # The token grid is read off the stem's output, not asserted to be H/4.
        x = rearrange(x, "(b t) c h w -> b c t h w", t=frames)
        x = self.patch_embedding(x)            # [B, dim, T, H/16, W/16]

        # Any window length: the stages patch time by up to 8. At the paper's
        # 160-frame windows the target equals T and this is skipped.
        patched = nearest_multiple(frames, self.temporal_stride)
        if patched != frames:
            x = F.adaptive_avg_pool3d(x, (patched, x.shape[3], x.shape[4]))

        for stage in self.stages:
            x = stage(x)                       # [B, dim, T, gh, gw]

        # Pool the token grid away, row by row then across, the order
        # upstream's two ``torch.mean(., 3)`` calls average in; time survives.
        features_last = reduce(reduce(x, "b c t gh gw -> b c t gw", "mean"),
                               "b c t gw -> b c t", "mean")
        rppg = self.conv_block_last(features_last)             # [B, 1, T']

        # Back to the window length if the temporal patches did not divide it.
        if rppg.shape[-1] != frames:
            rppg = F.interpolate(rppg, size=frames, mode="linear", align_corners=False)
        return rppg
