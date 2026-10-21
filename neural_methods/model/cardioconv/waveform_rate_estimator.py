"""Clip -> the heart rate, as the spectral peak of a waveform a network predicts."""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.loss.lagged_neg_pearson import LaggedNegPearson
from neural_methods.model.cardioconv.rate_estimator import grid_power
from neural_methods.model.cardioconv.utils import shrink
from neural_methods.model._shared_modules.utils import EPS

#: The exponent the spectrum is raised to before its centroid is taken. At 8
#: a second harmonic of half the fundamental's amplitude weighs 4^-8 of it,
#: so the centroid is the largest peak's, interpolated between bins.
PEAK_SHARPNESS = 8.0


class WaveformRateEstimator(nn.Module):
    """Which rate the clip pulses at, read off the waveform of a single-trace network.

    ``network`` is one copy of any single-trace architecture of this
    repository: the clip shrunk by ``downscale``, ``(B, C, T, h, w)``, in, one waveform ``(B, 1, T)``
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
      downscale: the area shrink of the clip the network reads; 1 is none.
    """

    def __init__(self, network: nn.Module, fs: float, low_bpm: float = 30.0, high_bpm: float = 200.0,
                 downscale: int = 1):
        super().__init__()
        self.network = network
        self.fs = fs
        self.downscale = downscale
        self.loss = LaggedNegPearson(fs)
        self.register_buffer("bins", torch.arange(low_bpm, high_bpm + 1.0, dtype=torch.float32))
        self.waveform = None

    def waveform_term(self, target: torch.Tensor) -> torch.Tensor:
        """``LaggedNegPearson`` between the last forward's waveform and
        ``target`` ``(B, T)``, averaged over the samples whose target is
        finite; zero when none is."""
        has = torch.isfinite(target).all(dim=-1)
        if not bool(has.any()):
            return 0.0 * self.waveform.sum()          # zero, with the network kept in the graph
        return self.loss(self.waveform[has], target[has].to(self.waveform.dtype)).mean()

    def forward(self, video: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, H, W)`` clip -> the rate ``(B,)`` in bpm and the logits
        ``(B, n_bins)`` it was read from. The network reads the clip shrunk
        by ``downscale``; the presence is not read."""
        # Float32 whatever autocast made of the network: the rate should not
        # be quantised by half precision.
        self.waveform = rearrange(self.network(shrink(video, self.downscale)), "b 1 t -> b t").float()
        centred = self.waveform - self.waveform.mean(dim=-1, keepdim=True)
        logits = PEAK_SHARPNESS * grid_power(centred, self.fs, self.bins).clamp_min(EPS).log()
        rate = (logits.softmax(dim=-1) * self.bins).sum(dim=-1)
        return rate, logits
