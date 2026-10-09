"""Functions CardioHydra's modules share: the per-harmonic collapse of the
wavelet field, the phase gradient of a complex field, and the wave speed
pooled under the masks."""

import torch
from einops import einsum, rearrange, reduce

from neural_methods.model.cardioconv.utils import EPS


def collapse_harmonics(field: torch.Tensor, n_harmonics: int, sharpness: float) -> torch.Tensor:
    """``(B, C, T, h, w, K)`` complex, harmonic-major -> ``(B, C, T, h, w, N)``
    complex: each harmonic's scales blended into one field.

    A power-weighted complex mean, weights ``|W| ** sharpness`` normalised
    per harmonic. Per pixel ``|W|`` is a noisy estimate, so a soft blend
    avoids the jitter of picking the peak scale, and phase, nearly constant
    across the narrow buffer, is kept. The magnitudes are scaled to their
    per-harmonic maximum first, so the power cannot overflow."""
    grouped = rearrange(field, "b c t h w (n j) -> b c t h w n j", n=n_harmonics)
    magnitude = grouped.abs()
    magnitude = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = magnitude ** sharpness
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (grouped * weights).sum(dim=-1)


def phase_gradient(field: torch.Tensor, dim: int) -> torch.Tensor:
    """The derivative of a complex field's phase along ``dim``, in radians per
    step: ``Im(dW conj(W)) / |W|^2``, which needs no unwrapping. Central
    differences inside, one-sided at the two ends."""
    (difference,) = torch.gradient(field, dim=dim)
    return (difference * field.conj()).imag / (field.real ** 2 + field.imag ** 2 + 1e-9)


def pooled_speed(speed: torch.Tensor, pulsatility: torch.Tensor, phase_masks: torch.Tensor,
                 present: torch.Tensor, supervise_masks: bool) -> torch.Tensor:
    """``(B, C, T, h, w, N)`` phase speed -> ``(B, C * S * N)``: ``log1p`` of its
    magnitude, averaged under each trace's mask per channel and harmonic,
    zero for an absent channel. Normalised by the mask's area, like the
    statistics head's pooled power, and detached from the masks with it."""
    weight = rearrange(pulsatility, "b c t h w -> b c t h w 1") * phase_masks
    if not supervise_masks:
        weight = weight.detach()
    area = reduce(weight, "b c t h w s -> b c s 1", "sum").clamp_min(EPS)
    pooled = einsum(torch.log1p(speed.abs()), weight, "b c t h w n, b c t h w s -> b c s n") / area
    pooled = pooled * rearrange(present.to(pooled.dtype), "b c -> b c 1 1")
    return rearrange(pooled, "b c s n -> b (c s n)")
