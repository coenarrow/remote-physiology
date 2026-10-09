"""Clip -> the heart rate, as the spectral peak of a waveform a network predicts."""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.loss.ccc import MAX_LAG_SECONDS, lag_frames, lagged_pair
from neural_methods.model.cardioconv.rate_estimator import grid_power
from neural_methods.model.cardioconv.utils import EPS, pearson

#: The exponent the spectrum is raised to before its centroid is taken. At 8
#: a second harmonic of half the fundamental's amplitude weighs 4^-8 of it,
#: so the centroid is the largest peak's, interpolated between bins.
PEAK_SHARPNESS = 8.0


class WaveformRateEstimator(nn.Module):
    """Which rate the clip pulses at, read off the waveform of a single-trace network.

    ``network`` is one copy of any single-trace architecture of this
    repository: raw clip ``(B, C, T, H, W)`` in, one waveform ``(B, 1, T)``
    out. Its waveform is trained against the window's fused cardiac waveform
    (``src.evaluation.rate.fused_waveform``) by :meth:`waveform_term`; the
    rate is the centroid of that waveform's power on a fixed grid of 1 bpm
    bins, the power raised to ``PEAK_SHARPNESS``: the largest spectral peak,
    which is how a pulse network's rate is always read, made differentiable
    and finer than a bin. Bounded by the grid by construction.

    The interface is :class:`~neural_methods.model.cardioconv.rate_estimator.RateEstimator`'s,
    so either is the model's ``rate_estimator``.

    Args:
      network: the single-trace network.
      fs: frame rate, in Hz.
      low_bpm, high_bpm: the grid, inclusive, 1 bpm apart.
    """

    def __init__(self, network: nn.Module, fs: float, low_bpm: float = 30.0, high_bpm: float = 200.0):
        super().__init__()
        self.network = network
        self.fs = fs
        self.max_lag = lag_frames(MAX_LAG_SECONDS, fs)
        self.register_buffer("bins", torch.arange(low_bpm, high_bpm + 1.0, dtype=torch.float32))
        self.waveform = None

    def waveform_term(self, target: torch.Tensor) -> torch.Tensor:
        """``1 - r`` between the last forward's waveform and ``target``
        ``(B, T)``, at the best lag within ``MAX_LAG_SECONDS``, averaged over
        the samples whose target is finite; zero when none is. The lag is
        searched because the neck pulses a transit time after the trace the
        target took its phase from."""
        has = torch.isfinite(target).all(dim=-1)
        if not bool(has.any()):
            return target.new_zeros(())
        waveform, goal = self.waveform[has], target[has].to(self.waveform.dtype)
        max_lag = min(self.max_lag, waveform.shape[-1] - 2)
        correlations = torch.stack([
            pearson(*lagged_pair(waveform, goal, lag))
            for lag in range(-max_lag, max_lag + 1)])                 # (lags, b)
        return (1.0 - correlations.max(dim=0).values).mean()

    def forward(self, video: torch.Tensor, small: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, H, W)`` raw clip -> the rate ``(B,)`` in bpm and the
        logits ``(B, n_bins)`` it was read from. The shrunk clip and the
        presence are not read: the network takes the clip as it is."""
        # Float32 whatever autocast made of the network: the rate should not
        # be quantised by half precision.
        self.waveform = rearrange(self.network(video), "b 1 t -> b t").float()
        centred = self.waveform - self.waveform.mean(dim=-1, keepdim=True)
        logits = PEAK_SHARPNESS * grid_power(centred, self.fs, self.bins).clamp_min(EPS).log()
        rate = (logits.softmax(dim=-1) * self.bins).sum(dim=-1)
        return rate, logits
