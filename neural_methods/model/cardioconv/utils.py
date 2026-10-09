"""Functions CardioConv's modules share: the wavelet centre frequencies of a
heart rate, the spatial shrink, circular statistics of wrapped phase, and
presence-masked reduction."""

import math

import torch
import torch.nn.functional as F
from einops import rearrange

EPS = 1e-6


def buffer_steps(buffer_bpm: float, step_bpm: float) -> int:
    """Scales per harmonic: the rates ``-buffer .. +buffer`` in ``step`` steps."""
    return int(round(2.0 * buffer_bpm / step_bpm)) + 1


def centre_frequencies(rate: torch.Tensor, fs: float, n_harmonics: int, buffer_bpm: float,
                       step_bpm: float) -> torch.Tensor:
    """``(B,)`` bpm -> ``(B, K)`` wavelet centre frequencies in cycles per frame.

    Harmonic ``k`` contributes the rates ``k * rate +- buffer_bpm`` in
    ``step_bpm`` steps. Ordered by harmonic, the fundamental first, so the
    last axis of the transform folds into ``(harmonic, step)``. A torch
    function of the rate, so the field carries ``d(field) / d(rate)``.
    """
    offsets = torch.linspace(-buffer_bpm, buffer_bpm, buffer_steps(buffer_bpm, step_bpm),
                             device=rate.device, dtype=rate.dtype)
    rates = torch.cat([k * rate[:, None] + offsets for k in range(1, n_harmonics + 1)], dim=1)
    return rates / 60.0 / fs


def shrink(video: torch.Tensor, factor: int) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> ``(B, C, T, H // factor, W // factor)``, each output
    pixel the mean of its ``factor x factor`` block, in float32. An area
    average, not a sample: pulse is weak and averaging the block is the
    gain a shrink is for."""
    b = video.shape[0]
    flat = rearrange(video.float(), "b c t h w -> (b c) t h w")
    small = F.interpolate(flat, scale_factor=1.0 / factor, mode="area")
    return rearrange(small, "(b c) t h w -> b c t h w", b=b)


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
# Circular statistics: phase is wrapped to [-pi, pi], so every statistic goes
# through the unit vector and no unwrapping is needed.
# ---------------------------------------------------------------------------
def circular_mean(angles: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.atan2(torch.sin(angles).mean(dim=dim), torch.cos(angles).mean(dim=dim))


def circular_variance(angles: torch.Tensor, dim: int) -> torch.Tensor:
    """``1 - R`` in ``[0, 1]``: 0 when every angle agrees, 1 when they are spread evenly."""
    resultant = torch.sqrt(torch.sin(angles).mean(dim=dim) ** 2 + torch.cos(angles).mean(dim=dim) ** 2)
    return 1.0 - resultant


def _neighbourhood_weight(height: int, width: int, kernel_size: int, like: torch.Tensor) -> torch.Tensor:
    """``(1, 1, H, W, k, k)``: 1 for a real, non-centre neighbour, 0 for padding and the centre."""
    pad = kernel_size // 2
    valid = F.pad(like.new_ones(1, 1, height, width), [pad, pad, pad, pad])
    valid = valid.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    centre = like.new_ones(kernel_size, kernel_size)
    centre[pad, pad] = 0.0
    return valid * centre


def spatial_confidence(height: int, width: int, kernel_size: int, like: torch.Tensor) -> torch.Tensor:
    """``(H, W)``: the fraction of its neighbourhood each pixel really has, 1 in
    the interior and less on the border (5/8 on an edge, 3/8 in a corner at
    ``kernel_size=3``). A border pixel's coherence is an estimate over fewer
    neighbours, so it is trusted that much less."""
    weight = _neighbourhood_weight(height, width, kernel_size, like)
    return (weight.sum(dim=(-2, -1)) / (kernel_size ** 2 - 1))[0, 0]


def neighbour_circular_difference(phase: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> the same: each pixel's mean absolute circular
    difference, in ``[0, pi]``, from the real neighbours in its ``k x k``
    neighbourhood. The phase is replicate-padded only to keep the lookups
    finite; padded positions carry no weight, so a border pixel is never
    compared with a copy of itself."""
    b, c = phase.shape[:2]
    height, width = phase.shape[-2:]
    pad = kernel_size // 2
    flat = rearrange(phase, "b c t h w -> (b c t) 1 h w")
    patches = F.pad(flat, [pad, pad, pad, pad], mode="replicate")
    patches = patches.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)     # (N, 1, H, W, k, k)
    delta = patches - rearrange(flat, "n 1 h w -> n 1 h w 1 1")
    difference = torch.atan2(torch.sin(delta), torch.cos(delta)).abs()
    weight = _neighbourhood_weight(height, width, kernel_size, flat)
    mean = (difference * weight).sum(dim=(-2, -1)) / weight.sum(dim=(-2, -1)).clamp_min(1.0)
    return rearrange(mean, "(b c t) 1 h w -> b c t h w", b=b, c=c)


def sliding_circular_variance(phase: torch.Tensor, window: int) -> torch.Tensor:
    """``(B, C, T, H, W)`` -> the same: the circular variance of the ``window``
    frames centred on each frame, the ends replicate-padded, so any window and
    any ``T`` give exactly ``T`` values."""
    b, height = phase.shape[0], phase.shape[3]
    left = (window - 1) // 2
    flat = rearrange(phase, "b c t h w -> (b c) (h w) t")
    windows = F.pad(flat, [left, window - 1 - left], mode="replicate").unfold(-1, window, 1)
    return rearrange(circular_variance(windows, dim=-1), "(b c) (h w) t -> b c t h w", b=b, h=height)
