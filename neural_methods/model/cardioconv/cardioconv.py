"""CardioConv: cardiac pressure waveforms from neck video, by wavelet masks.

The neck shows two cardiac pulsations, arterial and venous. They share the
heart rate and differ in phase, so the clip is taken to the wavelet domain at
the harmonics of the heart rate and read there:

1. :class:`~neural_methods.model.cardioconv.wavelet_transformer.WaveletTransformer`
   gives the complex wavelet field of every pixel (not learned);
2. :class:`~neural_methods.model.cardioconv.pulsatility_masker.PulsatilityMasker`
   turns its power into a mask of where the clip pulses at all;
3. :class:`~neural_methods.model.cardioconv.phase_masker.PhaseMasker` turns
   its phase into one mask per trace plus background, in one softmax;
4. :class:`~neural_methods.model.cardioconv.channel_predictor.ChannelPredictor`
   sums each channel's pixels under the masks into a candidate waveform per
   channel per trace;
5. :class:`~neural_methods.model.cardioconv.channel_mixer.ChannelMixer` mixes
   the channels into one shape-only waveform per trace;
6. :class:`~neural_methods.model.cardioconv.statistics_head.StatisticsHead`
   predicts each trace's level and spread and rescales the waveform to them.

A multi-trace architecture (``docs/adding_a_model.md``): it is built once and
returns every trace, because the traces compete for pixels in the phase
masker's softmax, which no pair of copies could do. It is for cardiac traces
only. The traces are the interface's, in its order; nothing here knows their
names.

**The heart rate is given, not estimated.** ``NEEDS_HEART_RATE`` makes the
wrapper pass ``heart_rate``, which it reads from the *label*. That is label
leakage at test time and the numbers of a run carry it; it stands in for an
estimator from the video (a frozen rPPG network) that will take this stage's
place inside the model.

Differences from the CardioHydra repository's CardioConv, all asked by the
contract: the batch dict and its per-channel dicts are one stacked tensor,
every channel still processed on its own; the stages' supervised losses are
the interface's ``LOSS`` on the output, and only the label-free mask priors
stay, as ``REGULARISERS``; the statistics head's min and max outputs are gone
(the ``MAX`` and ``MIN`` loss components supervise the waveform's own), its
mean is the single-bias readout the trainer seeds, and its spread is relative
to the mean instead of seeded from per-label constants; the phase masker's
lazy first layer is sized outright; a channel's presence is read off the clip
(an absent channel is a zero-filled plane) instead of a mask in the batch; and
the channel predictor reads the wavelet stage's shrunk clip instead of
shrinking the clip a second time.

Any window length is accepted. Frames may be any size from ``MIN_FRAME``.
All reshaping is einops.
"""

import torch
import torch.nn as nn
from einops import reduce

from neural_methods.model.cardioconv.channel_mixer import ChannelMixer
from neural_methods.model.cardioconv.channel_predictor import ChannelPredictor
from neural_methods.model.cardioconv.phase_masker import PhaseMasker
from neural_methods.model.cardioconv.pulsatility_masker import PulsatilityMasker
from neural_methods.model.cardioconv.statistics_head import StatisticsHead
from neural_methods.model.cardioconv.wavelet_transformer import WaveletTransformer
from neural_methods.model._shared_modules.utils import require_min_frame

#: Harmonics of the heart rate the wavelet scales cover.
N_HARMONICS = 3

#: The spatial shrink before the wavelet transform; every mask lives on this grid.
DOWNSCALE = 5

#: The smallest frame that leaves the masks a 3x3 grid, the spatial
#: neighbourhood their convolutions and coherence features read.
MIN_FRAME = 3 * DOWNSCALE

#: Plausible heart rates, in bpm; a given rate is clamped to them, and a rate
#: nobody could give (a sample with no label to read it from, whose loss is
#: masked out anyway) becomes the lower one.
RATE_RANGE = (30.0, 200.0)

PULSATILITY_PRIORS = ("sparsity", "pulsatility_tv", "pulsatility_smoothness")
PHASE_PRIORS = ("harmonic_coherence", "spatial_coherence", "temporal_consistency",
                "phase_tv", "phase_smoothness")


class CardioConv(nn.Module):
    REGULARISERS = PULSATILITY_PRIORS + PHASE_PRIORS
    NEEDS_HEART_RATE = True

    def __init__(self, in_channels: int, traces: tuple, fs: float, pulsatility_masker: bool = True,
                 phase_masker: bool = True, trace_independent: bool = True,
                 supervise_masks: bool = True):
        """Definition of CardioConv.

        Args:
          in_channels: the number of input channels.
          traces: the traces to predict, in the order of the output's axis 1.
          fs: frame rate of the clip, in Hz.
          pulsatility_masker: learn the pulsatility mask; ``False`` is the
            ablation, a mask of ones.
          phase_masker: learn the per-trace masks; ``False`` is the ablation,
            masks of ones, which leaves the traces nothing spatial to differ by.
          trace_independent: mix each trace from its own candidates only.
          supervise_masks: let the statistics head's pooled power train the masks.
        """
        super().__init__()
        self.traces = tuple(traces)
        self.fs = fs
        n_traces = len(self.traces)
        self.wavelet_transformer = WaveletTransformer(fs, N_HARMONICS, DOWNSCALE)
        self.pulsatility_masker = (
            PulsatilityMasker(self.wavelet_transformer.n_scales) if pulsatility_masker else None)
        self.phase_masker = PhaseMasker(n_traces) if phase_masker else None
        self.channel_predictor = ChannelPredictor()
        self.channel_mixer = ChannelMixer(in_channels, n_traces, trace_independent)
        self.statistics_head = StatisticsHead(in_channels, n_traces, N_HARMONICS,
                                              supervise_masks=supervise_masks)

    def output_layers(self):
        """Each trace's activation-free mean readout, in traces order."""
        return tuple(self.statistics_head.means)

    def regularisers(self) -> dict:
        return self._regularisers

    def forward(self, video: torch.Tensor, heart_rate: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` raw clip and ``(B,)`` bpm -> ``(B, S, T)``."""
        require_min_frame("CardioConv", MIN_FRAME, video.shape[-2], video.shape[-1])
        present = reduce(video.abs(), "b c t h w -> b c", "max") > 0
        heart_rate = torch.nan_to_num(heart_rate, nan=RATE_RANGE[0]).clamp(*RATE_RANGE)
        field, small = self.wavelet_transformer(video, heart_rate)
        power = field.abs()
        zero = video.new_zeros(())

        if self.pulsatility_masker is None:
            pulsatility = torch.ones_like(small)
            shared = dict.fromkeys(PULSATILITY_PRIORS, zero)
        else:
            pulsatility, shared = self.pulsatility_masker(power, present)
        if self.phase_masker is None:
            phase_masks = small.new_ones(*small.shape, len(self.traces))
            per_trace = [dict.fromkeys(PHASE_PRIORS, zero) for _ in self.traces]
        else:
            phase_masks, per_trace = self.phase_masker(field.angle(), present)
        # The pulsatility priors belong to every trace alike, so each trace
        # reports them: the trainer's total is the mean over traces.
        self._regularisers = {trace: {**shared, **own}
                              for trace, own in zip(self.traces, per_trace)}

        candidates = self.channel_predictor(small, pulsatility, phase_masks)
        mixed = self.channel_mixer(candidates)
        return self.statistics_head(mixed, power, small, pulsatility, phase_masks,
                                    heart_rate, present)
