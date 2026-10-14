"""Per-channel candidate waveforms -> one waveform per trace, by temporal convolution."""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.model._shared_modules.utils import EPS


class ChannelMixer(nn.Module):
    """A temporal convolution stack per trace over its channels' candidates.

    Each candidate is standardised over the window first, so the mix reads
    shape and timing and not the channels' very different pulse amplitudes
    (the statistics head puts the level back). A ``kernel``-tap temporal
    layer per trace over the intensity candidates, plus one over the
    kinematic candidates, then a pointwise readout, give the trace its
    shape-only waveform. The kinematic layer starts at zero, so the mixer
    starts as the intensity filter bank and learns the kinematics in. It is a linear filter bank, so CardioConv's prediction module is
    convolutional end to end and keeps every half-wave (a ReLU between two
    temporal layers rectified the zero-mean candidates and the predicted
    waveform's rate came out 20 bpm low on the synthetic test set,
    2026-10-10 18:05).

    Args:
      in_channels: video channels ``C`` of the candidates.
      n_traces: traces ``S``.
      n_kinematic: kinematic candidates per channel and trace, ``F``.
      trace_independent: ``True`` gives each trace its own stack over its own
        candidates and lets nothing cross between traces; ``False`` is one
        dense stack in which every trace draws on every candidate.
      hidden: channels of each trace's temporal layer.
      kernel: taps of the temporal layer; odd.
    """

    def __init__(self, in_channels: int, n_traces: int, n_kinematic: int,
                 trace_independent: bool = True, hidden: int = 16, kernel: int = 15):
        super().__init__()
        groups = n_traces if trace_independent else 1
        pad = kernel // 2
        self.temporal = nn.Conv1d(in_channels * n_traces, hidden * n_traces, kernel, padding=pad,
                                  padding_mode="replicate", groups=groups)
        self.kinematic = nn.Conv1d(in_channels * n_kinematic * n_traces, hidden * n_traces, kernel,
                                   padding=pad, padding_mode="replicate", groups=groups, bias=False)
        nn.init.zeros_(self.kinematic.weight)
        self.readout = nn.Conv1d(hidden * n_traces, n_traces, kernel_size=1, groups=groups)

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        """``(B, C, 1 + F, T, S)``, the intensity candidate first -> ``(B, S, T)``.
        Trace-major, so each trace's candidates are one contiguous group of
        the grouped convolutions."""
        centred = candidates - candidates.mean(dim=3, keepdim=True)
        standardised = centred / (centred.std(dim=3, keepdim=True) + EPS)
        hidden = (self.temporal(rearrange(standardised[:, :, 0], "b c t s -> b (s c) t"))
                  + self.kinematic(rearrange(standardised[:, :, 1:], "b c f t s -> b (s c f) t")))
        return self.readout(hidden)
