"""Masked clip -> one candidate waveform per channel per trace."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model.cardioconv.utils import EPS


class ChannelPredictor(nn.Module):
    """Each channel's pixels under the masks, averaged, as a waveform.

    The masks own the spatial selection, so the spatial collapse is the
    weighted mean of the pixels under ``pulsatility * trace mask``, normalised
    by the mass under the mask at every frame: what the waveform depends on is
    *which* pixels the masks pick, never how much mask there is. The channel
    axis is kept: each channel has its own polarity and phase, which the mixer
    weighs. Each channel is min-max scaled over the clip first, so 8-bit
    colour and 16-bit infrared or depth arrive on one scale. Nothing is
    learned here.
    """

    def forward(self, small: torch.Tensor, pulsatility: torch.Tensor,
                phase_masks: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, h, w)`` shrunk clip, ``(B, C, T, h, w)`` pulsatility mask
        and ``(B, C, T, h, w, S)`` trace masks -> ``(B, C, T, S)``."""
        low = reduce(small, "b c t h w -> b c 1 1 1", "min")
        high = reduce(small, "b c t h w -> b c 1 1 1", "max")
        scaled = (small - low) / (high - low).clamp_min(1e-8)
        weight = rearrange(pulsatility, "b c t h w -> b c t h w 1") * phase_masks
        area = reduce(weight, "b c t h w s -> b c t s", "sum").clamp_min(EPS)
        return einsum(scaled, weight, "b c t h w, b c t h w s -> b c t s") / area
