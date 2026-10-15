"""Masked clip and kinematics -> candidate waveforms per channel per trace."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model._shared_modules.utils import mask_area


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

    def forward(self, clip: torch.Tensor, weight_fine: torch.Tensor,
                planes: torch.Tensor = None, weight: torch.Tensor = None) -> torch.Tensor:
        """``(B, C, T, H, W)`` clip with the pooling weight ``(B, C, H, W, S)``
        on its grid, and ``(B, C, T, h, w, F)`` feature planes with the
        weight ``(B, C, h, w, S)`` on theirs -> ``(B, C, 1 + F, T, S)``, the
        intensity candidate first."""
        low = reduce(clip, "b c t h w -> b c 1 1 1", "min")
        high = reduce(clip, "b c t h w -> b c 1 1 1", "max")
        scaled = (clip - low) / (high - low).clamp_min(1e-8)
        candidates = [rearrange(einsum(scaled, weight_fine, "b c t h w, b c h w s -> b c t s")
                                / rearrange(mask_area(weight_fine), "b c s -> b c 1 s"), "b c t s -> b c 1 t s")]
        if planes is not None:
            candidates.append(einsum(planes, weight, "b c t h w f, b c h w s -> b c f t s")
                              / rearrange(mask_area(weight), "b c s -> b c 1 1 s"))
        return torch.cat(candidates, dim=2)
