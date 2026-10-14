"""Masked clip and kinematics -> candidate waveforms per channel per trace."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model._shared_modules.utils import EPS


class ChannelPredictor(nn.Module):
    """Each channel's pixels under the masks, averaged, as a waveform.

    The masks own the spatial selection, so the spatial collapse is the
    weighted mean of the pixels under ``pulsatility * trace mask``, normalised
    by the mass under the mask at every frame: what the waveform depends on is
    *which* pixels the masks pick, never how much mask there is. The channel
    axis is kept: each channel has its own polarity and phase, which the mixer
    weighs. Each channel is min-max scaled over the clip first, so 8-bit
    colour and 16-bit infrared or depth arrive on one scale. Any per-pixel
    feature ``planes`` (the kinematics) are pooled under the same weights,
    each a candidate of its own beside the intensity. Nothing is learned
    here.
    """

    def forward(self, clip: torch.Tensor, pulsatility: torch.Tensor,
                phase_masks: torch.Tensor, planes: torch.Tensor = None) -> torch.Tensor:
        """``(B, C, T, h, w)`` clip, ``(B, C, T, h, w)`` pulsatility mask,
        ``(B, C, T, h, w, S)`` trace masks and ``(B, C, T, h, w, F)`` feature
        planes -> ``(B, C, 1 + F, T, S)``, the intensity candidate first."""
        low = reduce(clip, "b c t h w -> b c 1 1 1", "min")
        high = reduce(clip, "b c t h w -> b c 1 1 1", "max")
        scaled = (clip - low) / (high - low).clamp_min(1e-8)
        weight = rearrange(pulsatility, "b c t h w -> b c t h w 1") * phase_masks
        area = reduce(weight, "b c t h w s -> b c 1 t s", "sum").clamp_min(EPS)
        candidates = rearrange(einsum(scaled, weight, "b c t h w, b c t h w s -> b c t s"),
                               "b c t s -> b c 1 t s")
        if planes is not None:
            candidates = torch.cat(
                [candidates, einsum(planes, weight, "b c t h w f, b c t h w s -> b c f t s")], dim=2)
        return candidates / area
