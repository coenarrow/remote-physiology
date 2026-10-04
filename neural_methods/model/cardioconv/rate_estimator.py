"""Shrunk clip -> the heart rate, as a softmax over 1 bpm bins."""

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce

from neural_methods.model.cardioconv.utils import EPS

#: The two per-pixel statistics the spatial attention reads.
N_STATISTICS = 2


def interpolate_bins(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``(..., n)`` values on a uniform grid -> ``(..., m)`` at fractional
    indices ``positions`` ``(m,)``, by linear interpolation; positions past
    either end take the end value."""
    low = positions.floor().long().clamp(0, values.shape[-1] - 2)
    frac = (positions - low).clamp(0.0, 1.0)
    return values[..., low] * (1.0 - frac) + values[..., low + 1] * frac


def rate_terms(logits: torch.Tensor, rate: torch.Tensor, target: torch.Tensor, bins: torch.Tensor,
               sigma_bpm: float = 2.0) -> dict:
    """The estimator's two terms over the samples with a finite target.

    ``rate`` is the cross-entropy between the bin distribution and a Gaussian
    of ``sigma_bpm`` around the target, normalised over the bins: it makes
    the distribution unimodal and sharp. ``rate_error`` is the mean absolute
    error of the readout in bpm: it trains the readout's position directly,
    and weighted it logs as ``weight * bpm``. Both are zero when no sample
    has a target.
    """
    has = torch.isfinite(target)
    zero = logits.new_zeros(())
    if not bool(has.any()):
        return {"rate": zero, "rate_error": zero}
    goal = target[has]
    prior = torch.exp(-0.5 * ((bins[None, :] - goal[:, None]) / sigma_bpm) ** 2)
    prior = prior / prior.sum(dim=-1, keepdim=True)
    cross_entropy = -(prior * logits[has].log_softmax(dim=-1)).sum(dim=-1).mean()
    return {"rate": cross_entropy, "rate_error": (rate[has] - goal).abs().mean()}


class RateEstimator(nn.Module):
    """Which rate the clip pulses at, read off a learned fusion of per-channel spectra.

    Three parts. *Attention*: each pixel's trace loses its mean; the log of
    its mean intensity and of its temporal standard deviation go through a
    2-D convolution shared by every channel to one logit per pixel, softmaxed
    over the pixels, and the weighted sum of the traces is one trace per
    channel. *Spectra*: each trace is Hann-windowed, FFT'd at the next power
    of two, and its power resampled onto a fixed grid of 1 bpm bins,
    normalised to unit sum over the grid and ``log1p``-compressed with the
    bin count as gain, so a flat spectrum reads ``log 2`` everywhere and a
    peaked one is O(1) at its peak whatever the channel's bit depth. The
    fixed grid is what makes the head independent of window length and
    frame rate. An absent channel is a zero plane. *Head*: a 1-D convolution
    over frequency with the channels as input planes, a learned weighted sum
    of log spectra, which is a learned geometric mean, the same fusion the
    label side does by hand. The rate is the softmax-weighted bin centre:
    bounded by construction, differentiable, and the only thing the rest of
    the model receives.

    Args:
      in_channels: video channels ``C``.
      fs: frame rate, in Hz.
      low_bpm, high_bpm: the grid, inclusive, 1 bpm apart.
      hidden: width of the head.
      attention_hidden: width of the attention's hidden layer.
    """

    def __init__(self, in_channels: int, fs: float, low_bpm: float = 30.0, high_bpm: float = 200.0,
                 hidden: int = 16, attention_hidden: int = 8):
        super().__init__()
        self.fs = fs
        self.register_buffer("bins", torch.arange(low_bpm, high_bpm + 1.0, dtype=torch.float32))
        self.attention = nn.Sequential(
            nn.Conv2d(N_STATISTICS, attention_hidden, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_hidden, 1, kernel_size=1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def traces(self, small: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, h, w)`` -> ``(B, C, T)`` attention-pooled, mean-removed traces."""
        b = small.shape[0]
        clip = small.float()
        centred = clip - reduce(clip, "b c t h w -> b c 1 h w", "mean")
        statistics = torch.stack([
            torch.log1p(reduce(clip, "b c t h w -> b c h w", "mean").clamp_min(0.0)),
            torch.log1p(centred.std(dim=2)),
        ], dim=2)                                                                   # (B, C, 2, h, w)
        logits = self.attention(rearrange(statistics, "b c f h w -> (b c) f h w"))
        weights = rearrange(logits, "(b c) 1 h w -> b c (h w)", b=b).softmax(dim=-1)
        return einsum(rearrange(centred, "b c t h w -> b c t (h w)"), weights, "b c t p, b c p -> b c t")

    def spectra(self, traces: torch.Tensor) -> torch.Tensor:
        """``(B, C, T)`` -> ``(B, C, n_bins)`` log power on the bpm grid."""
        t = traces.shape[-1]
        window = torch.hann_window(t, device=traces.device, dtype=traces.dtype)
        n = 1 << (t - 1).bit_length()
        power = torch.fft.rfft(traces * window, n=n, dim=-1).abs() ** 2         # (B, C, n // 2 + 1)
        bin_hz = self.fs / n
        positions = self.bins / 60.0 / bin_hz                                   # fractional rfft bins
        on_grid = interpolate_bins(power, positions)
        on_grid = on_grid / on_grid.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return torch.log1p(on_grid * self.bins.numel())

    def forward(self, small: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, h, w)`` shrunk clip and ``(B, C)`` presence -> the rate
        ``(B,)`` in bpm and the logits ``(B, n_bins)`` it was read from."""
        spectra = self.spectra(self.traces(small)) * rearrange(present.float(), "b c -> b c 1")
        logits = rearrange(self.head(spectra), "b 1 k -> b k")
        return logits.softmax(dim=-1) @ self.bins, logits
