from torch import nn


class ConvBlock3D(nn.Module):
    """``Conv3d`` -> ``Tanh`` -> ``InstanceNorm3d``.

    Used by iBVPNet and FactorizePhys. ``bias`` has no default because the two
    published networks disagree on it: iBVPNet's convolutions carry a bias,
    FactorizePhys's do not, and the ``Tanh`` sits between the convolution and
    the norm, so the norm does not cancel it.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, *,
                 bias: bool):
        super().__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding, bias=bias),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)
