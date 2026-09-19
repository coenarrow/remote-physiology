"""iBVPNet: a 3D convolutional encoder-decoder for rPPG, proposed with the
iBVP dataset.

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG
Dataset with High Resolution Signal Quality Labels." Electronics 13, no. 7:
1334. https://doi.org/10.3390/electronics13071334

Two things differ from the published network. First, the stem takes
``in_channels`` inputs (the interface's channel count) instead of the
upstream 1/3/4-channel branching; instance norm is per (sample, channel), so
one ``InstanceNorm3d(in_channels)`` over every channel is numerically
identical to upstream's split RGB / thermal norms. Second, upstream fed the
model ``T + 1`` frames (the trainer duplicated the last frame) so that an
internal ``torch.diff`` landed back on the window length ``T``; here the
model takes exactly ``T`` frames, takes the diff itself (``T - 1`` rows), and
appends one zero frame so the trunk always sees ``T`` rows, matching the
window length without help from the caller.

Any window length is accepted: ``temporal_encoder`` halves time twice, so the
spatio-temporal encoder's output is average-pooled in time to the nearest
multiple of 4 before it (a no-op when it already divides by 4), and the
final adaptive max-pool already targets the window length directly, so no
interpolation is needed afterward. At ``T = 160`` (the upstream
``CHUNK_LENGTH``, 30 fps) this is the published network, bit for bit. Frames
may be any size from 64x64, the smallest the spatial pools and strided
decoder convs leave anything of.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` in,
``(B, 1, T)`` out, preprocessing done by the dataset, the loss owned by the
trainer. All reshaping is einops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_methods.model._shared_modules.utils import require_min_frame
from neural_methods.model.ibvpnet.decoder_block import DecoderBlock
from neural_methods.model.ibvpnet.encoder_block import EncoderBlock

#: The two spatial-only ``MaxPool3d`` stages in ``spatio_temporal_encoder``,
#: the stride-1 spatial ``MaxPool3d`` in ``temporal_encoder``, and the two
#: stride-2 spatial convs in ``DecoderBlock``; a smaller frame pools to
#: nothing (empirically, since the strides do not divide evenly).
MIN_FRAME = 64

#: ``temporal_encoder``'s two 2x temporal halvings: the spatio-temporal
#: encoder's output is pooled to the nearest multiple of this before them.
TEMPORAL_STRIDE = 4

#: num_filters
FILTERS = [8, 16, 24, 40, 64]


class IBVPNet(nn.Module):
    def __init__(self, in_channels=3):
        """Definition of iBVPNet.

        Args:
          in_channels: the number of input channels. Default: 3.
        """
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.InstanceNorm3d(in_channels)
        self.encoder = EncoderBlock(in_channels, FILTERS, TEMPORAL_STRIDE)
        self.decoder = DecoderBlock(FILTERS)
        self.readout = nn.Conv3d(FILTERS[2], 1, [1, 1, 1], stride=1, padding=0)

    def output_layers(self):
        """The activation-free readout: the final 1x1x1 conv."""
        return (self.readout,)

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` -> ``(B, 1, T)``."""
        frames, height, width = x.shape[2:]
        require_min_frame("iBVPNet", MIN_FRAME, height, width)

        # Diff along time, T -> T - 1, then append a zero frame so the trunk
        # sees T rows, same as the window length in and out.
        x = torch.diff(x, dim=2)
        zero_frame = torch.zeros_like(x[:, :, :1, :, :])
        x = torch.cat([x, zero_frame], dim=2)

        x = self.norm(x)
        x = self.encoder(x)
        x = self.decoder(x)

        # Spatial adaptive pooling to a point, time straight to the window
        # length: the frame the diff and zero frame produced.
        x = F.adaptive_max_pool3d(x, (frames, 1, 1))
        x = self.readout(x)
        return rearrange(x, "b 1 t 1 1 -> b 1 t")
