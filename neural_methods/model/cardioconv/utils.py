"""Functions CardioConv's modules share: the wavelet centre frequencies of a
heart rate, the spatial shrink and its inverse, circular statistics of a complex field, and
presence-masked reduction."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from neural_methods.model._shared_modules.utils import EPS


def buffer_steps(buffer_bpm: float, step_bpm: float) -> int:
    """Scales per harmonic: the rates ``-buffer .. +buffer`` in ``step`` steps."""
    return int(round(2.0 * buffer_bpm / step_bpm)) + 1


def centre_frequencies(rate: torch.Tensor, fs: float, n_harmonics: int, buffer_bpm: float,
                       step_bpm: float) -> torch.Tensor:
    """``(B,)`` bpm -> ``(B, K)`` wavelet centre frequencies in cycles per frame.

    Harmonic ``k`` contributes the rates ``k * rate +- buffer_bpm`` in
    ``step_bpm`` steps. Ordered by harmonic, the fundamental first, so the
    last axis of the transform folds into ``(harmonic, step)``. A torch
    function of the rate, so a field computed with autograd on would carry
    ``d(field) / d(rate)``; the models compute theirs without it, the rate
    training on the estimator's own terms.
    """
    offsets = torch.linspace(-buffer_bpm, buffer_bpm, buffer_steps(buffer_bpm, step_bpm),
                             device=rate.device, dtype=rate.dtype)
    rates = torch.cat([k * rate[:, None] + offsets for k in range(1, n_harmonics + 1)], dim=1)
    return rates / 60.0 / fs


def shrink(video: torch.Tensor, factor: int) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B, C, T, H // factor, W // factor)``, each output
    pixel the mean of its ``factor x factor`` block, in float32. An area
    average, not a sample: pulse is weak and averaging the block is the
    gain a shrink is for. A factor of 1 is the clip itself."""
    if factor == 1:
        return video.float()
    b = video.shape[0]
    flat = rearrange(video.float(), "b c t h w -> (b c) t h w")
    small = F.interpolate(flat, scale_factor=1.0 / factor, mode="area")
    return rearrange(small, "(b c) t h w -> b c t h w", b=b)


def upscale(field: torch.Tensor, size: tuple) -> torch.Tensor:
    """``(B, C, T, h, w, K)`` complex -> ``(B, C, T, H, W, K)``: the field
    resampled to ``size`` bilinearly, real and imaginary parts alike, one
    harmonic group of scales at a time to bound the peak. The inverse of
    :func:`shrink` for a smooth field; a field already at ``size`` is
    returned as it is."""
    b, c, t, h, w, k = field.shape
    if (h, w) == tuple(size):
        return field
    parts = []
    for part in torch.view_as_real(field).split(max(1, k // 3), dim=-2):
        flat = rearrange(part, "b c t h w k r -> (b c t) (k r) h w")
        big = F.interpolate(flat, size=tuple(size), mode="bilinear", align_corners=False)
        parts.append(rearrange(big, "(b c t) (k r) h w -> b c t h w k r", b=b, c=c, r=2))
    return torch.view_as_complex(torch.cat(parts, dim=-2).contiguous())


def pearson(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` prediction and label -> ``(B,)`` Pearson correlation."""
    pred = pred - pred.mean(dim=-1, keepdim=True)
    label = label - label.mean(dim=-1, keepdim=True)
    return (pred * label).sum(dim=-1) / (pred.norm(dim=-1) * label.norm(dim=-1) + EPS)


def softplus_inverse(y: float) -> float:
    """``x`` such that ``softplus(x) == y``, for ``y > 0``."""
    return math.log(math.expm1(y))


def present_mean(values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """``(B, C)`` per-sample, per-channel values -> ``()``: each channel's mean
    over the samples that carry it, summed over channels. A channel no sample
    carries contributes zero, never NaN."""
    weight = present.to(values.dtype)
    per_channel = (values * weight).sum(dim=0) / weight.sum(dim=0).clamp_min(1.0)
    return per_channel.sum()


# ---------------------------------------------------------------------------
# Circular statistics of a complex field: every statistic goes through the
# unit vector, so no phase is ever unwrapped and no wrap is ever crossed.
# ---------------------------------------------------------------------------
def unit(z: torch.Tensor) -> torch.Tensor:
    """A complex tensor on the unit circle; zero stays near zero."""
    return z / z.abs().clamp_min(EPS)


def temporal_resultant(units: torch.Tensor, window: int) -> torch.Tensor:
    """``(B, C, T, h, w, N)`` unit complex -> the same, real in ``[0, 1]``: the
    length of the mean unit vector over the ``window`` frames centred on each
    frame, which is one minus the circular variance there. The ends are
    replicate-padded, so any window and any ``T`` give exactly ``T`` values."""
    b, c, _, h, w, n = units.shape
    flat = rearrange(torch.stack([units.real, units.imag], dim=-1),
                     "b c t h w n r -> (b c h w n) r t")
    left = (window - 1) // 2
    mean = F.avg_pool1d(F.pad(flat, [left, window - 1 - left], mode="replicate"), window, stride=1)
    return rearrange(mean.norm(dim=1), "(b c h w n) t -> b c t h w n", b=b, c=c, h=h, w=w, n=n)


def spatial_resultant(units: torch.Tensor, kernel: int) -> torch.Tensor:
    """``(B, C, T, h, w, N)`` unit complex -> the same, real in ``[0, 1]``: the
    length of the mean unit vector over each pixel's ``kernel x kernel``
    neighbourhood, real pixels only, so a border pixel is judged on the
    neighbours it has."""
    b, c, t, _, _, n = units.shape
    flat = rearrange(torch.stack([units.real, units.imag], dim=-1),
                     "b c t h w n r -> (b c t n) r h w")
    mean = F.avg_pool2d(flat, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    return rearrange(mean.norm(dim=1), "(b c t n) h w -> b c t h w n", b=b, c=c, t=t, n=n)
