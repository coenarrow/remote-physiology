"""Per-channel candidate waveforms -> one waveform per trace."""

import torch
import torch.nn as nn
from einops import rearrange


class ChannelMixer(nn.Module):
    """A pointwise mix that collapses the channel axis at every frame.

    Args:
      in_channels: video channels ``C`` of the candidates.
      n_traces: traces ``S``.
      trace_independent: ``True`` gives each trace its own ``C -> 1`` weights
        and lets nothing cross between traces; ``False`` is one dense
        ``C * S -> S`` mix in which every trace draws on every candidate.
    """

    def __init__(self, in_channels: int, n_traces: int, trace_independent: bool = True):
        super().__init__()
        self.mix = nn.Conv1d(in_channels * n_traces, n_traces, kernel_size=1,
                             groups=n_traces if trace_independent else 1)

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, S)`` -> ``(B, S, T)``. Trace-major, so each trace's
        channels are one contiguous group of the grouped convolution."""
        return self.mix(rearrange(candidates, "b c t s -> b (s c) t"))
