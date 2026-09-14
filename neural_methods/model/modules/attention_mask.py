## The attention_mask is used by DeepPhys, EfficientPhys and TS_CAN

import torch
from einops import reduce
from torch import Tensor, nn

class Attention_mask(nn.Module):
    """The CAN family's soft attention: a map normalised to sum to ``H * W / 2``."""

    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5