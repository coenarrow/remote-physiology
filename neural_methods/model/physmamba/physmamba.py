"""PhysMamba: slow-fast temporal-difference Mamba for physiological measurement.

Luo et al., https://doi.org/10.48550/arXiv.2409.12031

The architecture is unchanged from the original: the same convolutional stem,
the same two-stream slow/fast temporal-difference Mamba blocks with lateral
fusion, the same upsampling head. One thing differs from the published
network: the stem takes ``in_channels`` inputs (the interface's channel
count) instead of 3. The readout stays a single output plane; a multi-signal
run is one complete copy of this network per trace (``MultiTraceModel``),
never a widened readout on a shared trunk. At ``3`` this is exactly the
original, layer for layer.

Any window length is accepted: the slow stream strides time by 4 and the
head's two 2x upsamples put it back, so the stem's output is average-pooled
in time to the nearest multiple of 4 and the prediction is linearly
interpolated back to the window length. Both are skipped when the window
already divides by 4, so on
``configs/original_model_config/physmamba_FS30_W4.27S4.27_RGB_PPG_H128W128.yaml``
(128-frame windows) the forward pass is the published one. Frames may be any
size from 16x16, the smallest the four 2x spatial pools leave anything of.

A clip backbone on the multi-signal contract: ``(B, C_in, T, H, W)`` raw
frames in, ``(B, 1, T)`` out, the loss owned by the trainer. The input
preprocessing is the network's own first stage: the raw clip is
difference-normalised (``DiffNormalize``, the toolbox's DiffNormalized block,
with the clip's own statistics) before the stem, which is what the published
network was fed. All reshaping is einops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_methods.model._shared_modules.cdc_t import CDCT
from neural_methods.model._shared_modules.diffnormalize import DiffNormalize
from neural_methods.model._shared_modules.utils import nearest_multiple, require_min_frame
from neural_methods.model.physmamba.channel_attention_3d import ChannelAttention3D
from neural_methods.model.physmamba.lateral_connection import LateralConnection
from neural_methods.model.physmamba.mamba_layer import MambaLayer
from neural_methods.model.physmamba.utils import conv_block

#: Four ``MaxPool3d((1, 2, 2))`` stages between the input and the last Mamba
#: block; a smaller frame pools to nothing.
MIN_FRAME = 16

#: The slow stream's temporal stride, which the head's upsampling undoes.
TEMPORAL_STRIDE = 4


class PhysMamba(nn.Module):
    def __init__(self, in_channels=3, theta=0.5, drop_rate1=0.25, drop_rate2=0.5):
        """Definition of PhysMamba.

        Args:
          in_channels: the number of input channels. Default: 3.
          theta: the central-difference weight of every CDCT conv.
          drop_rate1, drop_rate2: dropout after the first and later blocks.
        """
        super().__init__()
        self.in_channels = in_channels

        # Input preprocessing: raw clip -> difference-normalised clip
        self.input_norm = DiffNormalize()
        self.conv_block_1 = conv_block(in_channels, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])
        self.conv_block_2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.conv_block_3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.conv_block_4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.conv_block_5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)
        self.conv_block_6 = conv_block(32, 32, [3, 1, 1], stride=1, padding=[1, 0, 0], activation='elu')

        # Temporal Difference Mamba Blocks
        # Slow Stream
        self.block_1 = self._build_block(64, theta)
        self.block_2 = self._build_block(64, theta)
        self.block_3 = self._build_block(64, theta)
        # Fast Stream
        self.block_4 = self._build_block(32, theta)
        self.block_5 = self._build_block(32, theta)
        self.block_6 = self._build_block(32, theta)

        # Upsampling
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,1,1)),
            nn.Conv3d(96, 48, [3, 1, 1], stride=1, padding=(1,0,0)),
            nn.BatchNorm3d(48),
            nn.ELU(),
        )

        self.conv_block_last = nn.Conv3d(48, 1, [1, 1, 1], stride=1, padding=0)
        self.maxpool_spa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.maxpool_spa_tem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.fuse_1 = LateralConnection(fast_channels=32, slow_channels=64)
        self.fuse_2 = LateralConnection(fast_channels=32, slow_channels=64)

        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)

        # Spatial-only pooling: ``None`` keeps the temporal axis at whatever
        # length the window happens to be, so one model serves any window.
        self.poolspa = nn.AdaptiveAvgPool3d((None, 1, 1))

    def output_layers(self):
        """The activation-free readout: the final 1x1x1 conv."""
        return (self.conv_block_last,)

    def _build_block(self, channels, theta):
        return nn.Sequential(
            CDCT(channels, channels, theta=theta),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            MambaLayer(dim=channels, mamba=self._build_ssm(channels)),
            ChannelAttention3D(in_channels=channels, reduction=2),
        )

    def _build_ssm(self, channels):
        """The sequence module of one block, ``(B, L, channels)`` in and out.

        ``None`` is ``MambaLayer``'s own, the published bidirectional Mamba1;
        PhysMamba2 overrides this and nothing else.
        """
        return None

    def forward(self, x):
        """``(B, in_channels, T, H, W)`` raw clip -> ``(B, 1, T)``."""
        frames, height, width = x.shape[2:]
        require_min_frame(type(self).__name__, MIN_FRAME, height, width)

        x = self.conv_block_1(self.input_norm(x))
        x = self.maxpool_spa(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.maxpool_spa(x)

        # Any window length: the slow stream needs T to divide by 4. At the
        # paper's 128-frame windows the target equals T and this is skipped.
        t4 = nearest_multiple(frames, TEMPORAL_STRIDE)
        if t4 != frames:
            x = F.adaptive_avg_pool3d(x, (t4, x.shape[3], x.shape[4]))

        # Process streams
        s_x = self.conv_block_4(x) # Slow stream
        f_x = self.conv_block_5(x) # Fast stream

        # First set of blocks and fusion
        s_x1 = self.block_1(s_x)
        s_x1 = self.maxpool_spa(s_x1)
        s_x1 = self.drop_1(s_x1)

        f_x1 = self.block_4(f_x)
        f_x1 = self.maxpool_spa(f_x1)
        f_x1 = self.drop_2(f_x1)

        s_x1 = self.fuse_1(s_x1,f_x1) # LateralConnection

        # Second set of blocks and fusion
        s_x2 = self.block_2(s_x1)
        s_x2 = self.maxpool_spa(s_x2)
        s_x2 = self.drop_3(s_x2)

        f_x2 = self.block_5(f_x1)
        f_x2 = self.maxpool_spa(f_x2)
        f_x2 = self.drop_4(f_x2)

        s_x2 = self.fuse_2(s_x2,f_x2) # LateralConnection

        # Third blocks and upsampling
        s_x3 = self.block_3(s_x2)
        s_x3 = self.upsample1(s_x3)
        s_x3 = self.drop_5(s_x3)

        f_x3 = self.block_6(f_x2)
        f_x3 = self.conv_block_6(f_x3)
        f_x3 = self.drop_6(f_x3)

        # Final fusion and upsampling
        x_fusion = torch.cat((f_x3, s_x3), dim=1)
        x_final = self.upsample2(x_fusion)

        x_final = self.poolspa(x_final)
        x_final = self.conv_block_last(x_final)
        out = rearrange(x_final, "b 1 t 1 1 -> b 1 t")

        # Back to the window length if the stride did not divide it.
        if out.shape[-1] != frames:
            out = F.interpolate(out, size=frames, mode="linear", align_corners=False)
        return out
