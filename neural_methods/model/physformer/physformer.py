"""PhysFormer: temporal-difference transformer for physiological measurement.

Yu et al., https://arxiv.org/abs/2111.12082 — a combination of ``Physformer.py``
and ``transformer_layer.py`` from the official implementation
(https://github.com/ZitongYu/PhysFormer).

The architecture is unchanged from the original. One thing differs from the
published network: the first layer (``stem_0``'s conv) takes ``in_channels``
inputs (the interface's channel count) instead of 3. The readout
(``conv_block_last``) stays a single plane, a bare ``Conv1d`` with no
activation, so an absolute-class signal like ABP can be predicted directly in
mmHg; a multi-signal run is one complete copy of this network per trace
(``MultiTraceModel``), never a widened readout on a shared trunk. The loss is
the trainer's, not the original's, and is whatever the interface's ``LOSS``
block states; the published DLDL frequency/KL term is not carried yet, which
``configs/original_model_config/physformer_FS30_W5.33S5.33_RGB_PPG_H128W128.yaml``
notes beside its negative Pearson.

Everything between those two layers — the 3-D stem, the (4,4,4) tube
tokenization, the three temporal-difference transformer stages, the temporal
upsampling head — is the published network, and on
``configs/original_model_config/physformer_FS30_W5.33S5.33_RGB_PPG_H128W128.yaml``
(128x128 frames, 160-frame windows) the forward pass is numerically the
published one.

Any other frame size or window length is accepted through two adaptive
stages around that network, both exact no-ops at the paper's shape:

* the original wrote the token grid as ``view(B, C, P//16, 4, 4)``, true only
  for 128x128 frames. Here the stem's output is average-pooled to the nearest
  multiple of the 4-pixel patch and the grid is read off the result: 128x128
  frames still tokenize to the paper's 4x4 (a pool from 16x16 to 16x16 is the
  identity, bin for bin), 72x72 frames to 2x2, and so on down to 8x8 frames,
  the smallest the stem's three 2x pools leave anything of;
* the 4-frame tubes and the head's two 2x temporal upsamples need a window
  that divides by 4. The stem's output is average-pooled in time to the
  nearest multiple of 4 and the prediction is linearly interpolated back to
  the window length; when the window already divides by 4 neither runs.

No parameter is added or resized. A clip backbone on the multi-signal
contract: ``(B, C_in, T, H, W)`` raw frames in, ``(B, 1, T)`` out, the loss owned by the trainer. The input
preprocessing is the network's own first stage: the raw clip is
difference-normalised (``DiffNormalize``, the toolbox's DiffNormalized block,
with the clip's own statistics) before the stem, which is what the published
network was fed. All reshaping is einops.
"""

from torch import nn

from neural_methods.model._shared_modules.diffnormalize import DiffNormalize
from neural_methods.model.physformer.vit_compact3_tdc import ViTCompact3TDC

#: The stem's three ``MaxPool3d((1, 2, 2))`` stages, i.e. the spatial factor
#: the tube patch embedding sees on top of its own patch size.
STEM_SPATIAL_STRIDE = 8

#: The smallest frame the stem leaves a 1x1 map of; anything smaller pools
#: to nothing.
MIN_FRAME = STEM_SPATIAL_STRIDE

#: The head's two ``Upsample(scale_factor=(2, 1, 1))`` stages. The temporal
#: patch size has to match it for the output to come back at the input length.
HEAD_TEMPORAL_UPSAMPLE = 4


class PhysFormer(nn.Module):
    """PhysFormer as a clip backbone: ``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``.

    Wraps :class:`ViTCompact3TDC` rather than merging with it,
    so the published network stays a self-contained module that can be checked
    against the original at ``in_channels=3``.

    ``gra_sharp`` is a forward-time argument of the original network, held here
    as the constant the paper and every published trainer use (2.0). It is the
    softmax temperature of the attention, so it belongs to the architecture, not
    to the training loop.
    """

    def __init__(self, in_channels=3, patches=4, dim=96, ff_dim=144, num_heads=4,
                 num_layers=12, theta=0.7, dropout_rate=0.1, gra_sharp=2.0):
        """``in_channels`` is the number of input channels (default 3); every
        other argument is a published hyperparameter at its published value."""
        super().__init__()
        self.in_channels = in_channels
        self.gra_sharp = float(gra_sharp)
        # Input preprocessing: raw clip -> difference-normalised clip
        self.input_norm = DiffNormalize()
        self.backbone = ViTCompact3TDC(
            MIN_FRAME, HEAD_TEMPORAL_UPSAMPLE, patches=patches, dim=dim, ff_dim=ff_dim, num_heads=num_heads,
            num_layers=num_layers, dropout_rate=dropout_rate, theta=theta,
            in_channels=in_channels,
        )

    def forward(self, video):
        """``(B, in_channels, T, H, W)`` raw clip -> ``(B, 1, T)``; attention maps are dropped."""
        rppg, *_ = self.backbone(self.input_norm(video), self.gra_sharp)
        return rppg

    def output_layers(self):
        """The activation-free readout: the final 1x1 conv."""
        return (self.backbone.conv_block_last,)

    def extra_repr(self):
        return f"gra_sharp={self.gra_sharp}"
