"""Wavelet power -> one soft pulsatility mask per channel."""

import torch
import torch.nn as nn
from einops import rearrange, reduce

from neural_methods.model.cardioconv.utils import present_mean


class PulsatilityMasker(nn.Module):
    """Where the clip pulses at the heart rate, as a weight in ``[0, 1]`` per
    pixel and frame.

    A separable 3-D convolution over the wavelet power: the scales are mixed
    pointwise (which scales matter), then smoothed in space, then in time.
    Nothing is downsampled. Time is replicate-padded so the ends of the
    window do not blend with zeros. Every channel goes through the same
    network on its own.

    The mask is supervised only through what is predicted from it, so three
    priors keep it from the trivial answers: ``sparsity`` (few pixels overlie
    a vessel), ``pulsatility_tv`` (a vessel has spatial extent) and
    ``pulsatility_smoothness`` (it does not flicker).

    Args:
      n_scales: the wavelet field's scale count, the input width.
      hidden: channels after the scale-mixing layer.
      temporal_kernel: frames the temporal smoothing sees; odd.
    """

    def __init__(self, n_scales: int, hidden: int = 16, temporal_kernel: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(n_scales, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, power: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, h, w, K)`` power and ``(B, C)`` presence -> the mask
        ``(B, C, T, h, w)`` and its priors ``{term: ()}``."""
        b = power.shape[0]
        mask = self.net(rearrange(power, "b c t h w k -> (b c) k t h w"))
        mask = rearrange(mask, "(b c) 1 t h w -> b c t h w", b=b)
        tv = (reduce((mask[..., 1:, :] - mask[..., :-1, :]).abs(), "b c t h w -> b c", "mean")
              + reduce((mask[..., 1:] - mask[..., :-1]).abs(), "b c t h w -> b c", "mean"))
        smoothness = reduce((mask[:, :, 1:] - mask[:, :, :-1]).abs(), "b c t h w -> b c", "mean")
        priors = {
            "sparsity": present_mean(reduce(mask, "b c t h w -> b c", "mean"), present),
            "pulsatility_tv": present_mean(tv, present),
            "pulsatility_smoothness": present_mean(smoothness, present),
        }
        return mask, priors
