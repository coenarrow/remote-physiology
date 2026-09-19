"""Masked clip -> one candidate waveform per channel per trace."""

import torch
import torch.nn as nn
from einops import rearrange, reduce


class ChannelPredictor(nn.Module):
    """Each channel's pixels, weighted by the masks and summed, as a waveform.

    The masks own the spatial selection, so the spatial collapse is a plain
    sum of the masked pixel mass; what is learned sits on the time axis only,
    one temporal filter shared by every (channel, trace) waveform. The channel
    axis is kept: each channel has its own polarity and phase, which the
    mixer weighs. Each channel is min-max scaled over the clip first, so 8-bit
    colour and 16-bit infrared or depth arrive on one scale.

    Args:
      temporal_kernel: taps of the temporal filter; odd.
    """

    def __init__(self, temporal_kernel: int = 15):
        super().__init__()
        self.temporal = nn.Conv1d(1, 1, kernel_size=temporal_kernel,
                                  padding=temporal_kernel // 2, padding_mode="replicate")

    def forward(self, small: torch.Tensor, pulsatility: torch.Tensor,
                phase_masks: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, h, w)`` shrunk clip, ``(B, C, T, h, w)`` pulsatility mask
        and ``(B, C, T, h, w, S)`` trace masks -> ``(B, C, T, S)``."""
        b, c = small.shape[:2]
        low = reduce(small, "b c t h w -> b c 1 1 1", "min")
        high = reduce(small, "b c t h w -> b c 1 1 1", "max")
        scaled = (small - low) / (high - low).clamp_min(1e-8)
        weight = rearrange(scaled * pulsatility, "b c t h w -> b c t h w 1") * phase_masks
        signals = reduce(weight, "b c t h w s -> (b c s) 1 t", "sum")
        return rearrange(self.temporal(signals), "(b c s) 1 t -> b c t s", b=b, c=c)
