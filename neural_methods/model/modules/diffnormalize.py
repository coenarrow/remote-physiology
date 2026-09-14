import torch
import torch.nn as nn
from einops import reduce


class DiffNormalize(nn.Module):
    """Frame-to-frame difference normalised by its own std, zero-padded to T.

    d_t = (x_{t+1} - x_t) / (x_{t+1} + x_t + eps), divided by the std over
    that (sample, channel) block, with a zero frame appended so the temporal
    length is unchanged. Input and output are ``(B, C, T, H, W)``.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        later, earlier = x[:, :, 1:], x[:, :, :-1]
        diff = (later - earlier) / (later + earlier + self.eps)
        std = reduce(diff, "b c t h w -> b c 1 1 1", torch.std)
        diff = torch.nan_to_num(diff / std.clamp_min(self.eps))
        pad = torch.zeros_like(x[:, :, :1])
        return torch.cat([diff, pad], dim=2)