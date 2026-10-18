"""CardioConv: cardiac pressure waveforms from neck video, by wavelet masks.

The neck shows two cardiac pulsations, arterial and venous. They share the
heart rate and differ in phase, so the clip is taken to the wavelet domain at
the harmonics of the heart rate and read there:

1. :class:`~neural_methods.model.cardioconv.waveform_rate_estimator.WaveformRateEstimator`
   reads the heart rate off the waveform of a single-trace network
   (``RATE_ESTIMATOR``, RhythmFormer by default), or
   :class:`~neural_methods.model.cardioconv.rate_estimator.RateEstimator`
   off a learned fusion of the channels' spectra;
2. :class:`~neural_methods.model.cardioconv.wavelet_transformer.WaveletTransformer`
   gives the complex wavelet field of every pixel at the harmonics of that
   rate, in closed form (not learned, and not a gradient path: the rate
   trains on the estimator's own terms);
3. :class:`~neural_methods.model.cardioconv.pulsatility_masker.PulsatilityMasker`
   turns its magnitude into one distribution per window over the pixels of
   each channel: where the clip pulses at all;
4. :class:`~neural_methods.model.cardioconv.phase_masker.PhaseMasker` turns
   its phase, relative to the channel's own pulse, into one mask per trace
   plus background, per window, in one softmax;
5. :class:`~neural_methods.model._shared_modules.kinematics.Kinematics`
   reads how the pulse travels off the field's phase gradient: per pixel,
   its wavenumber, speed and heading (not learned; CardioHydra's stage 5);
6. :class:`~neural_methods.model.cardioconv.channel_predictor.ChannelPredictor`
   averages each channel's pixels, and its kinematics, under the masks into
   candidate waveforms per channel per trace;
7. :class:`~neural_methods.model.cardioconv.channel_mixer.ChannelMixer` runs
   each trace's candidates through a temporal convolution stack, to one
   shape-only waveform per trace;
8. :class:`~neural_methods.model.cardioconv.statistics_head.StatisticsHead`
   predicts each trace's level and spread, from the wave speed under the
   masks as well, and rescales the waveform to them.

The kinematics reach the prediction by CardioHydra's two routes: per frame,
pooled under each trace's mask beside the intensity candidate, and per
window, as the speed under each mask (``pooled_speed``) the statistics head
reads. By the Bramwell-Hill relation a pulse travels faster in a vessel at
higher pressure, so the speed is a level cue; the per-frame kinematics are
standardised over the window by the mixer and carry only their changes.

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

**What the model looked at** is kept when ``capture`` is set: after a
forward, ``captured`` holds the clip, the rate, the pulsatility
distribution, the trace and background masks, the per-harmonic magnitude,
the kinematics' speed and heading and the pooled speed of the samples that
ran, detached, for ``scripts/visualise.py``.

Differences from the CardioHydra repository's CardioConv, all asked by the
contract: the batch dict and its per-channel dicts are one stacked tensor,
every channel still processed on its own; the stages' supervised losses are
the interface's ``LOSS`` on the output, and only the label-free mask priors
stay, as ``REGULARISERS``, beside the estimator's terms; the statistics
head's min and max outputs are gone (the ``MAX`` and ``MIN`` loss components
supervise the waveform's own), its mean is the single-bias readout the
trainer seeds, and its spread is relative to the mean instead of seeded from
per-label constants; a channel's presence is read off the clip (an absent
channel is a zero-filled plane) instead of a mask in the batch; and the clip
is shrunk only where it is expensive to read, by a factor each (the rate
estimator's ``rate_downscale``, the wavelet transform's
``wavelet_downscale``); the masks, the kinematics and every reading of the
field work on the wavelet's grid, and only the masks are read at the input's
resolution, upscaled bilinearly, to pool the clip (2026-10-14; from 2026-10-12 the field itself was upscaled and every later
stage ran at the input's resolution, which put the step's memory peak in
the mask networks; until then one shrink at the top put every stage on the
shrunk grid). The masks were
redesigned on 2026-10-10 after both had collapsed in every training run
(``cardioconv_cardiohydra_development_notes/``): the pulsatility mask is a
softmax over pixels instead of a sigmoid, the phase masker reads the phase
relative to the channel's pulse and answers to a prior that makes the traces
claim what pulses, and the wavelet bank keeps three scales per harmonic
instead of twenty-one. The repository's CardioConv had no kinematics; they
were CardioHydra's, and joined CardioConv on 2026-10-12. On 2026-10-11 the redesigned masks collapsed in turn
under long training: one trace's share of the pulsatile pixels fell under
float32 resolution and the phase masker froze, and the pulsatility map
shrank to a few pixels, both because the mask networks' corrections grew
without bound. Both networks now add a bounded correction
(``PULSATILITY_BOUND``, ``PHASE_BOUND``) to a fixed physical logit (the
pulsatility index; the phase bins, with one pair of centres per channel),
with a prior that pulls the raw correction back inside its bound; the phase
masker's correction cannot prefer a class wholesale; the background may keep
what pulses at neither trace's phase; and the phase priors are taken on each
trace's claimed distribution, so no share can be emptied.

Any window length is accepted. Frames may be any size from
``min_frame(wavelet_downscale)``, and from whatever floor the rate
network's own stem has on its shrunk clip (the builder checks it there). All reshaping is
einops.
"""

import torch
import torch.nn as nn
from einops import rearrange, reduce

from neural_methods.model.cardioconv.channel_mixer import ChannelMixer
from neural_methods.model.cardioconv.channel_predictor import ChannelPredictor, coherent_candidate
from neural_methods.model.cardioconv.phase_masker import PhaseMasker
from neural_methods.model.cardioconv.pulsatility_masker import PulsatilityMasker
from neural_methods.model.cardioconv.rate_estimator import RateEstimator, rate_terms
from neural_methods.model.cardioconv.statistics_head import StatisticsHead
from neural_methods.model.cardioconv.utils import shrink, upscale
from neural_methods.model.cardioconv.waveform_rate_estimator import WaveformRateEstimator
from neural_methods.model.cardioconv.wavelet_transformer import WaveletTransformer
from neural_methods.model._shared_modules.kinematics import Kinematics
from neural_methods.model._shared_modules.utils import (
    EPS, KINEMATIC_FEATURES, kinematic_features, pooled_speed, require_min_frame, window_weight,
)

#: Harmonics of the heart rate the wavelet scales cover.
N_HARMONICS = 3

#: The wavelet scales sit this far either side of each harmonic, in bpm, this
#: far apart: three scales per harmonic, at -10, 0 and +10 bpm. The Morlet
#: kernel is about 0.18 x its centre wide (1 sigma), so they overlap and
#: cover the band; the training gate's tolerance must stay inside the buffer.
BUFFER_BPM = 10.0
STEP_BPM = 10.0

#: The smallest grid the wavelet transform may run on, the spatial
#: neighbourhood the masks' convolutions and coherence features read.
MIN_GRID = 3

#: The largest move each mask network may make to a logit, in nats.
#: Unbounded, the corrections grew to 44 and 120 nats and the masks collapsed
#: (2026-10-11, experiments 5, 6a, 6b). The pulsatility map's: two pixels'
#: weights move against each other by at most e^(2 x 1.5), about 20, relative
#: to the pulsatility index's. The phase masks': two classes' logits move
#: against each other by at most 2 x 0.9 nats, under the phase bins' 4 nats
#: between a coherent trace pixel and the background (``PHASE_GAIN`` each
#: way, since the background's bin falls with coherence), so where the phase
#: holds the phase decides.
PULSATILITY_BOUND = 1.5
PHASE_BOUND = 0.9

PULSATILITY_PRIORS = ("pulsatility_tv", "pulsatility_saturation")
PHASE_PRIORS = ("phase_saturation", "harmonic_coherence", "spatial_coherence",
                "temporal_consistency", "phase_tv")
RATE_TERMS = ("rate", "rate_error", "rate_waveform")


def min_frame(downscale: int) -> int:
    """The smallest frame that leaves the wavelet transform a ``MIN_GRID`` x
    ``MIN_GRID`` grid at a shrink of ``downscale``."""
    return MIN_GRID * downscale


class CardioConv(nn.Module):
    REGULARISERS = PULSATILITY_PRIORS + PHASE_PRIORS + RATE_TERMS
    NEEDS_RATE_TARGET = True

    def __init__(self, in_channels: int, traces: tuple, fs: float, pulsatility_masker: bool = True,
                 phase_masker: bool = True, trace_independent: bool = True,
                 supervise_masks: bool = True, rate_tolerance_bpm: float = 8.0,
                 rate_downscale: int = 2, wavelet_downscale: int = 2,
                 rate_network: nn.Module = None, mixer_hidden: int = 16, mixer_taps: int = 15,
                 coherent_pooling: bool = False):
        """Definition of CardioConv.

        Args:
          in_channels: the number of input channels.
          traces: the traces to predict, in the order of the output's axis 1.
          fs: frame rate of the clip, in Hz.
          pulsatility_masker: learn the pulsatility mask; ``False`` is the
            ablation, a uniform distribution.
          phase_masker: learn the per-trace masks; ``False`` is the ablation,
            masks of ones, which leaves the traces nothing spatial to differ by.
          trace_independent: mix each trace from its own candidates only.
          supervise_masks: let the statistics head's pooled power and speed
            train the masks.
          rate_tolerance_bpm: in training, a sample whose estimated rate is
            further than this from its target skips everything after the
            estimator. Must be inside ``BUFFER_BPM``.
          rate_downscale: the area shrink of the clip the rate estimator
            reads. 1 is none.
          wavelet_downscale: the area shrink of the clip the wavelet
            transform reads; the masks, the kinematics and every reading of
            the field work on that grid, and the masks are read at the
            input's resolution only to pool the clip. 1 is none.
          rate_network: one copy of a single-trace architecture to read the
            rate off (``WaveformRateEstimator``); ``None`` is the spectral
            ``RateEstimator``.
          mixer_hidden: channels of each trace's temporal layer in the
            channel mixer.
          mixer_taps: taps of that layer; odd. 15 is half a beat at 30 fps.
          coherent_pooling: the intensity candidate is the pulse pooled
            coherently in the wavelet domain (``coherent_candidate``)
            instead of the masked mean of the intensity.
        """
        super().__init__()
        self.traces = tuple(traces)
        self.fs = fs
        self.rate_tolerance_bpm = rate_tolerance_bpm
        self.wavelet_downscale = wavelet_downscale
        self.coherent_pooling = coherent_pooling
        self.min_frame = min_frame(wavelet_downscale)
        n_traces = len(self.traces)
        self.rate_estimator = (RateEstimator(in_channels, fs, downscale=rate_downscale)
                               if rate_network is None
                               else WaveformRateEstimator(rate_network, fs, downscale=rate_downscale))
        # The bank's pad covers the slowest kernel the estimator can ask for.
        self.wavelet_transformer = WaveletTransformer(
            fs, N_HARMONICS, buffer_bpm=BUFFER_BPM, step_bpm=STEP_BPM,
            min_bpm=float(self.rate_estimator.bins[0]))
        self.pulsatility_masker = (
            PulsatilityMasker(self.wavelet_transformer.n_scales, PULSATILITY_BOUND)
            if pulsatility_masker else None)
        self.phase_masker = (PhaseMasker(in_channels, n_traces, N_HARMONICS, PHASE_BOUND,
                                         self.wavelet_transformer.bandwidth)
                             if phase_masker else None)
        self.kinematics = Kinematics(N_HARMONICS, pixel=wavelet_downscale)
        self.channel_predictor = ChannelPredictor()
        self.channel_mixer = ChannelMixer(in_channels, n_traces,
                                          N_HARMONICS * len(KINEMATIC_FEATURES), trace_independent,
                                          hidden=mixer_hidden, kernel=mixer_taps)
        self.statistics_head = StatisticsHead(in_channels, n_traces, N_HARMONICS,
                                              supervise_masks=supervise_masks,
                                              n_extra=in_channels * n_traces * N_HARMONICS)
        self.sample_mask = None
        self.capture = False
        self.captured = None

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
        require_min_frame("CardioConv", self.min_frame, video.shape[-2], video.shape[-1])
        present = reduce(video.abs(), "b c t h w -> b c", "max") > 0
        clip = video.float()
        rate, logits = self.rate_estimator(clip, present)
        zero = clip.new_zeros(())

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
        subset, priors = self.predict(clip[index], present[index], rate[index])
        out = clip.new_zeros(video.shape[0], len(self.traces), video.shape[2])
        out[index] = subset.to(out.dtype)       # autocast may have left the head in half precision
        self._regularisers = {trace: {**{term: value * keep for term, value in own.items()}, **terms}
                              for trace, own in zip(self.traces, priors)}
        return out

    def masks(self, small: torch.Tensor, field: torch.Tensor, power: torch.Tensor,
              present: torch.Tensor, size: tuple, rate: torch.Tensor) -> tuple:
        """The two mask stages on the field and its power, on the field's grid
        ``(h, w)``: the pulsatility distribution ``(b, C, h, w)``, the trace
        masks ``(b, C, h, w, S)`` and the background mask ``(b, C, h, w)``;
        the same three read at the input's grid ``size``, upscaled
        bilinearly (the distribution renormalised; the classes a convex
        blend, so they still sum to one), which is what pools the clip; and
        the priors shared by every trace and each trace's own,
        ``({term: ()}, [{term: ()}])``. An ablated stage returns its neutral
        mask and zero priors."""
        b, c, _, h, w = small.shape
        zero = small.new_zeros(())
        if self.pulsatility_masker is None:
            pulsatility = small.new_full((b, c, h, w), 1.0 / (h * w))
            shared = dict.fromkeys(PULSATILITY_PRIORS, zero)
        else:
            pulsatility, shared = self.pulsatility_masker(power, small, present)
        fine = upscale(rearrange(pulsatility, "b c h w -> b c h w 1"), size)[..., 0]
        pulsatility_fine = fine / reduce(fine, "b c h w -> b c 1 1", "sum").clamp_min(EPS)
        if self.phase_masker is None:
            phase_masks = small.new_ones(b, c, h, w, len(self.traces))
            background = small.new_zeros(b, c, h, w)
            phase_fine = small.new_ones(b, c, *size, len(self.traces))
            background_fine = small.new_zeros(b, c, *size)
            per_trace = [dict.fromkeys(PHASE_PRIORS, zero) for _ in self.traces]
        else:
            # Beats in the window at the estimated rate, a feature not a path:
            # the rate network trains on its own terms (see predict).
            beats = field.shape[2] * rate.detach() / (60.0 * self.fs)    # (b,)
            phase_masks, background, per_trace, also = self.phase_masker(field, pulsatility, present, beats)
            shared = {**shared, **also}
            classes = upscale(torch.cat([phase_masks, background[..., None]], dim=-1), size)
            phase_fine, background_fine = classes[..., :-1], classes[..., -1]
        return ((pulsatility, phase_masks, background), (pulsatility_fine, phase_fine, background_fine),
                shared, per_trace)

    def predict(self, clip: torch.Tensor, present: torch.Tensor, rate: torch.Tensor) -> tuple:
        """Everything after the estimator, on the samples that run: ``(b, C, T, H, W)``
        clip, ``(b, C)`` presence and ``(b,)`` bpm -> ``(b, S, T)`` and
        each trace's mask priors, ``[{term: ()}]`` in traces order."""
        # The field is a feature of the clip at the estimated rate, not a path:
        # the rate network trains on its own terms, and nothing of the field
        # is kept for the backward pass (it is the largest tensor in the model).
        with torch.no_grad():
            small = shrink(clip, self.wavelet_downscale)
            field = self.wavelet_transformer(small, rate)
            power = field.abs()
        coarse, fine, shared, per_trace = self.masks(small, field, power, present, clip.shape[-2:], rate)
        pulsatility, phase_masks, _ = coarse
        weight = window_weight(pulsatility, phase_masks)          # (b, C, h, w, S): pools what the field gives
        weight_fine = window_weight(fine[0], fine[1])             # (b, C, H, W, S): pools the clip
        # The shared priors belong to every trace alike, so each trace
        # reports them: the trainer's total is the mean over traces.
        priors = [{**shared, **own} for own in per_trace]

        kinematics = self.kinematics(field)
        planes = torch.cat(kinematic_features(kinematics), dim=-1)          # (b, C, T, h, w, N F)
        coherent = (coherent_candidate(kinematics["harmonics"], pulsatility, weight)
                    if self.coherent_pooling else None)
        candidates = self.channel_predictor(clip, weight_fine, planes, weight, coherent)
        mixed = self.channel_mixer(candidates)
        speed = pooled_speed(kinematics["phase_speed"], weight, present,
                             self.statistics_head.supervise_masks)
        out = self.statistics_head(mixed, power, clip, weight, rate, present, extra=speed)
        if self.capture:
            self.captured = capture(clip, rate, power, coarse, fine, N_HARMONICS, kinematics, speed)
        return out, priors


def capture(clip, rate, power, coarse: tuple, fine: tuple, n_harmonics: int,
            kinematics: dict, speed: torch.Tensor) -> dict:
    """What a forward looked at, detached, for the visualiser: the clip and
    the masks read at its grid (one map per window, expanded over the frames
    as views), the same masks on the field's grid (``*_coarse``, one map per
    window), the per-harmonic magnitude, the kinematics and the fundamental's
    blended complex field on the field's grid."""
    pulsatility, phase_masks, background = fine
    t = clip.shape[2]
    return {
        "clip": clip.detach(), "rate": rate.detach(),
        "power": reduce(power, "b c t h w (n j) -> b c t h w n", "mean", n=n_harmonics).detach(),
        "pulsatility": rearrange(pulsatility.detach(), "b c h w -> b c 1 h w").expand(-1, -1, t, -1, -1),
        "phase_masks": rearrange(phase_masks.detach(), "b c h w s -> b c 1 h w s").expand(-1, -1, t, -1, -1, -1),
        "background": rearrange(background.detach(), "b c h w -> b c 1 h w").expand(-1, -1, t, -1, -1),
        "pulsatility_coarse": coarse[0].detach(),
        "phase_masks_coarse": coarse[1].detach(),
        "background_coarse": coarse[2].detach(),
        "wavenumber": kinematics["wavenumber"].detach(),
        "phase_speed": kinematics["phase_speed"].detach(),
        "direction_x": kinematics["direction_x"].detach(),
        "direction_y": kinematics["direction_y"].detach(),
        "harmonic": kinematics["harmonics"][..., 0].detach(),
        "pooled_speed": speed.detach(),
    }
