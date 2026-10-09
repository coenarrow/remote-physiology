"""CardioConv: cardiac pressure waveforms from neck video, by wavelet masks.

The neck shows two cardiac pulsations, arterial and venous. They share the
heart rate and differ in phase, so the clip is taken to the wavelet domain at
the harmonics of the heart rate and read there:

1. :class:`~neural_methods.model.cardioconv.rate_estimator.RateEstimator`
   reads the heart rate off a learned fusion of the channels' spectra, or
   :class:`~neural_methods.model.cardioconv.waveform_rate_estimator.WaveformRateEstimator`
   reads it off the waveform of a single-trace network (``RATE_ESTIMATOR``);
2. :class:`~neural_methods.model.cardioconv.wavelet_transformer.WaveletTransformer`
   gives the complex wavelet field of every pixel at the harmonics of that
   rate, in closed form, so the rate receives the waveform loss's gradient
   through the field (not learned itself);
3. :class:`~neural_methods.model.cardioconv.pulsatility_masker.PulsatilityMasker`
   turns its power into a mask of where the clip pulses at all;
4. :class:`~neural_methods.model.cardioconv.phase_masker.PhaseMasker` turns
   its phase into one mask per trace plus background, in one softmax;
5. :class:`~neural_methods.model.cardioconv.channel_predictor.ChannelPredictor`
   sums each channel's pixels under the masks into a candidate waveform per
   channel per trace;
6. :class:`~neural_methods.model.cardioconv.channel_mixer.ChannelMixer` mixes
   the channels into one shape-only waveform per trace;
7. :class:`~neural_methods.model.cardioconv.statistics_head.StatisticsHead`
   predicts each trace's level and spread and rescales the waveform to them.

A multi-trace architecture (``docs/adding_a_model.md``): it is built once and
returns every trace, because the traces compete for pixels in the phase
masker's softmax, which no pair of copies could do. It is for cardiac traces
only. The traces are the interface's, in its order; nothing here knows their
names.

**The rate is learned, and training gates on it.** The wrapper hands the
model, in training mode only, the rate of each sample's fused cardiac
traces, the ECG among them (``src.inputs.window_rate``), and one waveform
fused from the same traces that beats at it (``src.inputs.window_waveform``).
The estimator's terms are computed against them: ``rate`` and ``rate_error``
against the rate, and ``rate_waveform``, which only the waveform estimator
has, against the waveform. A sample whose estimate is more than ``rate_tolerance_bpm``
from it does not go through the bank or anything after: its output is zero
and ``sample_mask`` tells the wrapper to mask it out of the loss. Until the
estimator is close, the field would be noise and the masks would learn
from noise. One sample always runs, so every parameter stays in the graph.
At test time nothing from the label reaches the model and every sample runs.

Differences from the CardioHydra repository's CardioConv, all asked by the
contract: the batch dict and its per-channel dicts are one stacked tensor,
every channel still processed on its own; the stages' supervised losses are
the interface's ``LOSS`` on the output, and only the label-free mask priors
stay, as ``REGULARISERS``, beside the estimator's terms; the statistics
head's min and max outputs are gone (the ``MAX`` and ``MIN`` loss components
supervise the waveform's own), its mean is the single-bias readout the
trainer seeds, and its spread is relative to the mean instead of seeded from
per-label constants; the phase masker's lazy first layer is sized outright; a
channel's presence is read off the clip (an absent channel is a zero-filled
plane) instead of a mask in the batch; and the clip is shrunk once, at the
top, for every stage that reads it.

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
from neural_methods.model.cardioconv.rate_estimator import RateEstimator, rate_terms
from neural_methods.model.cardioconv.statistics_head import StatisticsHead
from neural_methods.model.cardioconv.utils import shrink
from neural_methods.model.cardioconv.waveform_rate_estimator import WaveformRateEstimator
from neural_methods.model.cardioconv.wavelet_transformer import WaveletTransformer
from neural_methods.model._shared_modules.utils import require_min_frame

#: Harmonics of the heart rate the wavelet scales cover.
N_HARMONICS = 3

#: The wavelet scales sit this far either side of each harmonic, in bpm. The
#: training gate's tolerance must stay inside it.
BUFFER_BPM = 10.0

#: The spatial shrink before the estimator and the wavelet; every mask lives on this grid.
DOWNSCALE = 5

#: The smallest frame that leaves the masks a 3x3 grid, the spatial
#: neighbourhood their convolutions and coherence features read.
MIN_FRAME = 3 * DOWNSCALE

PULSATILITY_PRIORS = ("sparsity", "pulsatility_tv", "pulsatility_smoothness")
PHASE_PRIORS = ("harmonic_coherence", "spatial_coherence", "temporal_consistency",
                "phase_tv", "phase_smoothness")
RATE_TERMS = ("rate", "rate_error", "rate_waveform")


class CardioConv(nn.Module):
    REGULARISERS = PULSATILITY_PRIORS + PHASE_PRIORS + RATE_TERMS
    NEEDS_RATE_TARGET = True

    def __init__(self, in_channels: int, traces: tuple, fs: float, pulsatility_masker: bool = True,
                 phase_masker: bool = True, trace_independent: bool = True,
                 supervise_masks: bool = True, rate_tolerance_bpm: float = 6.0,
                 rate_network: nn.Module = None):
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
          rate_tolerance_bpm: in training, a sample whose estimated rate is
            further than this from its target skips everything after the
            estimator. Must be inside ``BUFFER_BPM``.
          rate_network: one copy of a single-trace architecture to read the
            rate off (``WaveformRateEstimator``); ``None`` is the spectral
            ``RateEstimator``.
        """
        super().__init__()
        self.traces = tuple(traces)
        self.fs = fs
        self.rate_tolerance_bpm = rate_tolerance_bpm
        n_traces = len(self.traces)
        self.rate_estimator = (RateEstimator(in_channels, fs) if rate_network is None
                               else WaveformRateEstimator(rate_network, fs))
        # The bank's pad covers the slowest kernel the estimator can ask for.
        self.wavelet_transformer = WaveletTransformer(
            fs, N_HARMONICS, buffer_bpm=BUFFER_BPM, min_bpm=float(self.rate_estimator.bins[0]))
        self.pulsatility_masker = (
            PulsatilityMasker(self.wavelet_transformer.n_scales) if pulsatility_masker else None)
        self.phase_masker = PhaseMasker(n_traces) if phase_masker else None
        self.channel_predictor = ChannelPredictor()
        self.channel_mixer = ChannelMixer(in_channels, n_traces, trace_independent)
        self.statistics_head = StatisticsHead(in_channels, n_traces, N_HARMONICS,
                                              supervise_masks=supervise_masks)
        self.sample_mask = None

    def output_layers(self):
        """Each trace's activation-free mean readout, in traces order."""
        return tuple(self.statistics_head.means)

    def regularisers(self) -> dict:
        return self._regularisers

    def forward(self, video: torch.Tensor, target_rate: torch.Tensor = None,
                target_waveform: torch.Tensor = None) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` raw clip, and in training the ``(B,)`` target
        bpm and the ``(B, T)`` waveform beating at it (NaN where a sample has
        none) -> ``(B, S, T)``. Sets ``sample_mask``: the samples whose
        output counts."""
        require_min_frame("CardioConv", MIN_FRAME, video.shape[-2], video.shape[-1])
        present = reduce(video.abs(), "b c t h w -> b c", "max") > 0
        small = shrink(video, DOWNSCALE)
        rate, logits = self.rate_estimator(video, small, present)
        zero = small.new_zeros(())

        keep = 1.0                                # weight of the mask priors
        if target_rate is None:
            passed = torch.ones_like(rate, dtype=torch.bool)
            run = passed
            terms = dict.fromkeys(RATE_TERMS, zero)
        else:
            error = (rate - target_rate).abs()
            passed = torch.isfinite(target_rate) & (error <= self.rate_tolerance_bpm)
            run = passed
            if not bool(passed.any()):
                # Nothing passed: run the closest sample anyway, so every
                # parameter is in the graph on every step, but let nothing of
                # it count: its waveform leaves through sample_mask and its
                # mask priors, taken on a field at the wrong rate, are zeroed.
                run = torch.zeros_like(passed)
                run[torch.nan_to_num(error, nan=float("inf")).argmin()] = True
                keep = 0.0
            terms = rate_terms(logits, rate, target_rate, self.rate_estimator.bins)
            terms["rate_waveform"] = self.rate_estimator.waveform_term(target_waveform)
        self.sample_mask = passed

        index = run.nonzero().squeeze(1)
        subset, priors = self.predict(small[index], present[index], rate[index])
        out = small.new_zeros(video.shape[0], len(self.traces), video.shape[2])
        out[index] = subset.to(out.dtype)       # autocast may have left the head in half precision
        self._regularisers = {trace: {**{term: value * keep for term, value in own.items()}, **terms}
                              for trace, own in zip(self.traces, priors)}
        return out

    def predict(self, small: torch.Tensor, present: torch.Tensor, rate: torch.Tensor) -> tuple:
        """Everything after the estimator, on the samples that run: ``(b, C, T, h, w)``
        shrunk clip, ``(b, C)`` presence and ``(b,)`` bpm -> ``(b, S, T)`` and
        each trace's mask priors, ``[{term: ()}]`` in traces order."""
        field = self.wavelet_transformer(small, rate)
        power = field.abs()
        zero = small.new_zeros(())

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
        priors = [{**shared, **own} for own in per_trace]

        candidates = self.channel_predictor(small, pulsatility, phase_masks)
        mixed = self.channel_mixer(candidates)
        out = self.statistics_head(mixed, power, small, pulsatility, phase_masks, rate, present)
        return out, priors
