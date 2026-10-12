"""Per-channel candidate waveforms -> one waveform per trace, by temporal convolution."""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.model.cardioconv.utils import EPS


class ChannelMixer(nn.Module):
    """A temporal convolution stack per trace over its channels' candidates.

    Each candidate is standardised over the window first, so the mix reads
    shape and timing and not the channels' very different pulse amplitudes
    (the statistics head puts the level back). A ``kernel``-tap temporal
    layer per trace, then a pointwise readout, give the trace its shape-only
    waveform: a linear filter bank, so CardioConv's prediction module is
    convolutional end to end and keeps every half-wave (a ReLU between two
    temporal layers rectified the zero-mean candidates and the predicted
    waveform's rate came out 20 bpm low on the synthetic test set,
    2026-10-10 18:05).

    Args:
      in_channels: video channels ``C`` of the candidates.
      n_traces: traces ``S``.
      trace_independent: ``True`` gives each trace its own stack over its own
        candidates and lets nothing cross between traces; ``False`` is one
        dense stack in which every trace draws on every candidate.
      hidden: channels of each trace's temporal layer.
      kernel: taps of the temporal layer; odd.
    """

    def __init__(self, in_channels: int, n_traces: int, trace_independent: bool = True,
                 hidden: int = 16, kernel: int = 15):
        super().__init__()
        groups = n_traces if trace_independent else 1
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels * n_traces, hidden * n_traces, kernel, padding=pad,
                      padding_mode="replicate", groups=groups),
            nn.Conv1d(hidden * n_traces, n_traces, kernel_size=1, groups=groups),
        )

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, S)`` -> ``(B, S, T)``. Trace-major, so each trace's
        channels are one contiguous group of the grouped convolutions."""
        centred = candidates - candidates.mean(dim=2, keepdim=True)
        standardised = centred / (centred.std(dim=2, keepdim=True) + EPS)
        return self.net(rearrange(standardised, "b c t s -> b (s c) t"))
