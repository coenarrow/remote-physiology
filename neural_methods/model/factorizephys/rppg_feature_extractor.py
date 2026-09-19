"""The feature extractor of FactorizePhys."""

import torch.nn as nn

from neural_methods.model._shared_modules.conv_block_3d import ConvBlock3D


class RPPGFeatureExtractor(nn.Module):
    """The 3-D convolutional stem. Shapes below are the paper's 72x72 frames."""

    def __init__(self, in_channels, filters, dropout_rate=0.1):
        super().__init__()
        # in_channels, out_channel, kernel_size, stride, padding, bias
        #                                                                    Input: #B, in_channels, 160, 72, 72
        self.feature_extractor = nn.Sequential(
            ConvBlock3D(in_channels, filters[0], [3, 3, 3], [1, 1, 1], [1, 1, 1], bias=False),  #B, filters[0], 160, 72, 72
            ConvBlock3D(filters[0], filters[1], [3, 3, 3], [1, 2, 2], [1, 0, 0], bias=False), #B, filters[1], 160, 35, 35
            ConvBlock3D(filters[1], filters[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False), #B, filters[1], 160, 33, 33
            nn.Dropout3d(p=dropout_rate),

            ConvBlock3D(filters[1], filters[1], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False), #B, filters[1], 160, 31, 31
            ConvBlock3D(filters[1], filters[2], [3, 3, 3], [1, 2, 2], [1, 0, 0], bias=False), #B, filters[2], 160, 15, 15
            ConvBlock3D(filters[2], filters[2], [3, 3, 3], [1, 1, 1], [1, 0, 0], bias=False), #B, filters[2], 160, 13, 13
            nn.Dropout3d(p=dropout_rate),
        )

    def forward(self, x):
        return self.feature_extractor(x)
