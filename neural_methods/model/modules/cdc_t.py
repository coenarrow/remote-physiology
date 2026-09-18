import math

from einops import rearrange, reduce
from torch import Tensor, nn
from torch.nn import functional as F


def sum_spatial(weight: Tensor) -> Tensor:
    """``(co, ci, kh, kw)`` summed over the kernel plane, row by row then across.

    Two reductions rather than one over both axes: in float32 the accumulation
    order is part of the answer, and this is the order the original summed in.
    """
    return reduce(reduce(weight, "co ci kh kw -> co ci kw", "sum"),
                  "co ci kw -> co ci", "sum")


class CDC_T(nn.Module):
    """Temporal center-difference 3-D convolution.

    From https://github.com/ZitongYu/PhysFormer/model/transformer_layer.py.
    ``theta`` mixes the plain convolution with the central-difference one;
    ``theta = 0`` is an ordinary ``Conv3d``. Used by PhysFormer, RhythmFormer
    and PhysMamba (and so PhysHydra), each of which passes its own ``theta``.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.6):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal

        # Only the central difference over a temporal kernel > 1 is meaningful.
        if self.conv.weight.shape[2] > 1:
            # Summed over kh and then over kw, the order upstream's
            # ``.sum(2).sum(2)`` accumulates in: a single joint reduction over
            # both axes is the same number in exact arithmetic but not in
            # float32, and this kernel goes on to weight a convolution.
            kernel_diff = (sum_spatial(self.conv.weight[:, :, 0])
                           + sum_spatial(self.conv.weight[:, :, 2]))
            kernel_diff = rearrange(kernel_diff, "cout cin -> cout cin 1 1 1")
            out_diff = F.conv3d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0,
                                dilation=self.conv.dilation, groups=self.conv.groups)
            return out_normal - self.theta * out_diff

        return out_normal
