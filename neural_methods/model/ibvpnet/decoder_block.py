"""The decoder of iBVPNet: two deconv blocks back up to the window length."""

import torch.nn as nn

from neural_methods.model.ibvpnet.deconv_block_3d import DeconvBlock3D


class DecoderBlock(nn.Module):
    """Two ``DeconvBlock3D``, each doubling time and halving space."""

    def __init__(self, filters):
        """``filters`` are the five stages' filter counts; they are iBVPNet's,
        passed in."""
        super().__init__()
        self.decoder_block = nn.Sequential(
            DeconvBlock3D(filters[4], filters[3], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
            DeconvBlock3D(filters[3], filters[2], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
        )

    def forward(self, x):
        return self.decoder_block(x)
