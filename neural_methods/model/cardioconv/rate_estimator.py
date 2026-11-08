"""Clip -> the heart rate, read off the waveform a network predicts as the evaluation reads one."""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.loss.lagged_neg_pearson import LaggedNegPearson
from neural_methods.model.cardioconv.utils import grid_power, shrink
from src.evaluation.rate import BAND, rate_of


class RateEstimator(nn.Module):
    """Which rate the clip pulses at, read off the waveform of a single-trace network.

    ``network`` is one copy of any single-trace architecture of this
    repository: the clip shrunk by ``downscale``, ``(B, C, T, h, w)``, in, one
    waveform ``(B, 1, T)`` out. Its waveform is trained against the window's
    fused cardiac waveform (``src.evaluation.rate.fused_waveform``) by the
    one term in :data:`TERMS`; the rate is ``src.evaluation.rate.rate_of`` on
    the waveform's power over a grid of 1 bpm bins spanning the heart-rate
    band: the largest bin unless half it explains the spectrum better, the
    very reading the label's rate target and the evaluation's score use, so
    the model is gated on the quantity it is scored on. Not a gradient path:
    the network trains through its waveform alone.

    Args:
      network: the single-trace network.
      fs: frame rate, in Hz.
      low_bpm, high_bpm: the grid, inclusive, 1 bpm apart; the band of
        ``src.evaluation.rate`` by default.
      downscale: the area shrink of the clip the network reads; 1 is none.
    """

    TERMS = ("rate_waveform",)

    def __init__(self, network: nn.Module, fs: float, low_bpm: float = BAND[0] * 60,
                 high_bpm: float = BAND[1] * 60, downscale: int = 1):
        super().__init__()
        self.network = network
        self.fs = fs
        self.downscale = downscale
        self.loss = LaggedNegPearson(fs)
        self.register_buffer("bins", torch.arange(low_bpm, high_bpm + 1.0, dtype=torch.float32))
        self.waveform = None

    def terms(self, target_rate: torch.Tensor, target_waveform: torch.Tensor) -> dict:
        """``rate_waveform``: ``LaggedNegPearson`` between the last forward's
        waveform and ``target_waveform`` ``(B, T)``, averaged over the samples
        whose target is finite; zero when none is. The target rate is not read."""
        has = torch.isfinite(target_waveform).all(dim=-1)
        if not bool(has.any()):
            term = 0.0 * self.waveform.sum()          # zero, with the network kept in the graph
        else:
            term = self.loss(self.waveform[has], target_waveform[has].to(self.waveform.dtype)).mean()
        return {"rate_waveform": term}

    def forward(self, video: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W)`` clip -> the rate ``(B,)`` in bpm. The network
        reads the clip shrunk by ``downscale``; the presence is not read."""
        # Float32 whatever autocast made of the network: the rate should not
        # be quantised by half precision.
        self.waveform = rearrange(self.network(shrink(video, self.downscale)), "b 1 t -> b t").float()
        centred = self.waveform - self.waveform.mean(dim=-1, keepdim=True)
        return rate_of(self.bins / 60.0, grid_power(centred, self.fs, self.bins))
