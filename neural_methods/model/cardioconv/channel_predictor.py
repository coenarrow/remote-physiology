"""Masked clip and kinematics -> candidate waveforms per channel per trace."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model._shared_modules.utils import EPS, mask_area


def coherent_candidate(harmonics: torch.Tensor, pulsatility: torch.Tensor,
                       weight: torch.Tensor) -> torch.Tensor:
    """The pulse under each mask, pooled coherently in the wavelet domain:
    ``(B, C, T, h, w, N)`` complex harmonics, ``(B, C, h, w)`` pulsatility and
    the ``(B, C, h, w, S)`` pooling weight -> ``(B, C, T, S)`` waveform.

    A masked mean of the intensity averages pixels whose pulses sit at
    different phases across the mask's bin, so they partly cancel, worst at
    the harmonics that carry the beat's shape. Here each pixel's harmonic
    ``n`` is first rotated back by ``n`` times the pixel's own relative phase
    at the fundamental (its window-mean phase against the channel's
    pulsatility-weighted pulse: a delay shifts harmonic ``n`` by ``n`` times
    the fundamental's shift), so every pixel's pulse is in step before the
    mean; the real parts of the pooled harmonics summed are the band-passed
    waveform. In-phase signal adds, noise does not. Nothing is learned; the
    weight carries the masks' gradient as the intensity pool does.
    """
    reference = einsum(harmonics, pulsatility.to(harmonics.dtype), "b c t h w n, b c h w -> b c t n")
    reference = reference / (reference.abs() + EPS)
    relative = harmonics[..., 0] * rearrange(reference[..., 0].conj(), "b c t -> b c t 1 1")
    relative = relative / (relative.abs() + EPS)
    phase = torch.angle(relative.mean(dim=2))                                     # (B, C, h, w)
    order = torch.arange(1, harmonics.shape[-1] + 1, device=phase.device, dtype=phase.dtype)
    angles = -phase[..., None] * order                                             # (B, C, h, w, N)
    align = torch.polar(torch.ones_like(angles), angles)
    aligned = harmonics * rearrange(align, "b c h w n -> b c 1 h w n")
    pooled = einsum(aligned, weight.to(harmonics.dtype), "b c t h w n, b c h w s -> b c t n s")
    pooled = pooled / rearrange(mask_area(weight), "b c s -> b c 1 1 s")
    return pooled.real.sum(dim=3)


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
                planes: torch.Tensor = None, weight: torch.Tensor = None,
                coherent: torch.Tensor = None) -> torch.Tensor:
        """``(B, C, T, H, W)`` clip with the pooling weight ``(B, C, H, W, S)``
        on its grid, and ``(B, C, T, h, w, F)`` feature planes with the
        weight ``(B, C, h, w, S)`` on theirs -> ``(B, C, 1 + F, T, S)``, the
        intensity candidate first. A ``(B, C, T, S)`` ``coherent`` waveform
        (:func:`coherent_candidate`) takes the intensity candidate's place."""
        if coherent is not None:
            candidates = [rearrange(coherent, "b c t s -> b c 1 t s")]
        else:
            low = reduce(clip, "b c t h w -> b c 1 1 1", "min")
            high = reduce(clip, "b c t h w -> b c 1 1 1", "max")
            scaled = (clip - low) / (high - low).clamp_min(1e-8)
            candidates = [rearrange(einsum(scaled, weight_fine, "b c t h w, b c h w s -> b c t s")
                                    / rearrange(mask_area(weight_fine), "b c s -> b c 1 s"), "b c t s -> b c 1 t s")]
        if planes is not None:
            candidates.append(einsum(planes, weight, "b c t h w f, b c h w s -> b c f t s")
                              / rearrange(mask_area(weight), "b c s -> b c 1 1 s"))
        return torch.cat(candidates, dim=2)
