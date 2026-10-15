"""CardioHydra: cardiac pressure waveforms from neck video, by wavelet masks
and a state-space trunk per trace.

CardioConv's front end with a different reading of what the masks select.
The first four stages are CardioConv's own modules, loaded from its package,
and the fifth is the kinematics both models share:

1. :class:`~neural_methods.model.cardioconv.waveform_rate_estimator.WaveformRateEstimator`
   reads the heart rate off the waveform of a single-trace network
   (``RATE_ESTIMATOR``, RhythmFormer by default), or a ``RateEstimator`` off
   a learned fusion of the channels' spectra;
2. :class:`~neural_methods.model.cardioconv.wavelet_transformer.WaveletTransformer`
   gives the complex wavelet field of every pixel at the harmonics of that
   rate, in closed form;
3. :class:`~neural_methods.model.cardioconv.pulsatility_masker.PulsatilityMasker`
   turns its magnitude into one distribution per window over the pixels of
   each channel: where the clip pulses at all;
4. :class:`~neural_methods.model.cardioconv.phase_masker.PhaseMasker` turns
   its phase, relative to the channel's own pulse, into one mask per trace
   plus background, per window, in one softmax;
5. :class:`~neural_methods.model._shared_modules.kinematics.Kinematics` reads how
   the pulse travels off the field's phase gradient: per pixel, its
   wavenumber, speed and heading (not learned);
6. :class:`~neural_methods.model.cardiohydra.ssm_predictor.SSMPredictor`
   averages the field, the kinematics and the clip under each trace's mask
   at every frame and runs each trace's features through its own
   bidirectional Mamba3 trunk, to one shape-only waveform per trace;
7. :class:`~neural_methods.model.cardioconv.statistics_head.StatisticsHead`
   predicts each trace's level and spread, here from the wave speed under
   the masks as well, and rescales the waveform to them.

A multi-trace architecture (``docs/adding_a_model.md``), built once, for the
reason CardioConv is: the traces compete for pixels in the phase masker's
softmax. Cardiac traces only.

The rate is learned and training gates on it, exactly as in CardioConv; see
that module's docstring, which also has the mask redesign of 2026-10-10 and
the ``capture`` switch for the visualiser. The regularisers are CardioConv's too:
the mask priors and the estimator's terms.

Differences from the CardioHydra repository's model, beside those CardioConv
already lists: the trunk's blocks are Mamba3 and sum their two directions
(``MambaBlock``); the predictor's own Pearson and variance-anchor losses are
gone, the interface's ``LOSS`` supervises the output; and the phase gradient
is one-sided at the borders of the grid and the window instead of wrapping
around them.

Any window length is accepted. Frames may be any size from
``min_frame(wavelet_downscale)``. CUDA only: Mamba3's kernels are Triton. All
reshaping is einops.
"""

import math

import torch
import torch.nn as nn
from einops import reduce

from neural_methods.model.cardiohydra.ssm_predictor import SSMPredictor
from neural_methods.model.cardioconv.cardioconv import (
    BUFFER_BPM, N_HARMONICS, PHASE_BOUND, PHASE_PRIORS, PULSATILITY_BOUND, PULSATILITY_PRIORS, RATE_TERMS,
    STEP_BPM, CardioConv, capture, min_frame,
)
from neural_methods.model.cardioconv.phase_masker import PhaseMasker
from neural_methods.model.cardioconv.pulsatility_masker import PulsatilityMasker
from neural_methods.model.cardioconv.rate_estimator import RateEstimator, rate_terms
from neural_methods.model.cardioconv.statistics_head import StatisticsHead
from neural_methods.model.cardioconv.utils import shrink
from neural_methods.model.cardioconv.waveform_rate_estimator import WaveformRateEstimator
from neural_methods.model.cardioconv.wavelet_transformer import WaveletTransformer
from neural_methods.model._shared_modules.kinematics import Kinematics
from neural_methods.model._shared_modules.utils import pooled_speed, require_min_frame, window_weight

__all__ = ["BUFFER_BPM", "CardioHydra", "min_frame"]

#: Width of each trace's trunk, and its Mamba blocks.
D_MODEL = 128
N_BLOCKS = 2

#: Each block's Mamba3: the state size and expansion of the Mamba1 the trunk
#: was designed with, in heads of PhysMamba3's size.
D_STATE = 16
EXPAND = 2
HEADDIM = 16

#: The time step every head starts at; PhysMamba3's, the one Mamba3 start
#: that has trained on this data.
DT_INIT = math.log(2)


class CardioHydra(nn.Module):
    REGULARISERS = PULSATILITY_PRIORS + PHASE_PRIORS + RATE_TERMS
    NEEDS_RATE_TARGET = True

    def __init__(self, in_channels: int, traces: tuple, fs: float, pulsatility_masker: bool = True,
                 phase_masker: bool = True, supervise_masks: bool = True,
                 rate_tolerance_bpm: float = 8.0, rate_downscale: int = 2,
                 wavelet_downscale: int = 2,
                 mamba_lr_scale: float = 0.1, rate_network: nn.Module = None):
        """Definition of CardioHydra.

        Args:
          in_channels: the number of input channels.
          traces: the traces to predict, in the order of the output's axis 1.
          fs: frame rate of the clip, in Hz.
          pulsatility_masker: learn the pulsatility mask; ``False`` is the
            ablation, a uniform distribution.
          phase_masker: learn the per-trace masks; ``False`` is the ablation,
            masks of ones, which leaves the traces nothing spatial to differ by.
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
          mamba_lr_scale: the trunks' Mamba blocks train at this multiple of
            the recipe's learning rate (the trainer reads the ``_lr_scale``
            flag), as PhysMamba's layers do. On Neckflix the CVP phase mask
            died after about 4,000 steps at the full rate and about 8,500 at
            a tenth (2026-10-11, experiments 5 and 6a; one run each).
          rate_network: one copy of a single-trace architecture to read the
            rate off (``WaveformRateEstimator``); ``None`` is the spectral
            ``RateEstimator``.
        """
        super().__init__()
        self.traces = tuple(traces)
        self.fs = fs
        self.rate_tolerance_bpm = rate_tolerance_bpm
        self.supervise_masks = supervise_masks
        self.wavelet_downscale = wavelet_downscale
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
        self.ssm_predictor = SSMPredictor(in_channels, n_traces, N_HARMONICS, d_model=D_MODEL,
                                          n_blocks=N_BLOCKS, d_state=D_STATE, expand=EXPAND,
                                          headdim=HEADDIM, dt_init=DT_INIT)
        for parameter in self.ssm_predictor.trunks.parameters():
            parameter._lr_scale = mamba_lr_scale
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

    #: The two mask stages are CardioConv's, run the same way.
    masks = CardioConv.masks

    def forward(self, video: torch.Tensor, target_rate: torch.Tensor = None,
                target_waveform: torch.Tensor = None) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` raw clip, and in training the ``(B,)`` target
        bpm and the ``(B, T)`` waveform beating at it (NaN where a sample has
        none) -> ``(B, S, T)``. Sets ``sample_mask``: the samples whose
        output counts."""
        require_min_frame("CardioHydra", self.min_frame, video.shape[-2], video.shape[-1])
        if not video.is_cuda:
            raise RuntimeError(
                "CardioHydra runs on CUDA only: Mamba3's kernels are Triton. "
                "Drop --no-gpu, or use CardioConv, which has a CPU path.")
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

    def predict(self, clip: torch.Tensor, present: torch.Tensor, rate: torch.Tensor) -> tuple:
        """Everything after the estimator, on the samples that run: ``(b, C, T, H, W)``
        clip, ``(b, C)`` presence and ``(b,)`` bpm -> ``(b, S, T)`` and
        each trace's mask priors, ``[{term: ()}]`` in traces order."""
        with torch.no_grad():                     # a feature, not a path (see CardioConv.predict)
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
        shapes = self.ssm_predictor(field, power, clip, weight, weight_fine, kinematics, present)
        speed = pooled_speed(kinematics["phase_speed"], weight, present, self.supervise_masks)
        out = self.statistics_head(shapes, power, clip, weight, rate, present, extra=speed)
        if self.capture:
            self.captured = capture(clip, rate, power, *fine, N_HARMONICS, kinematics, speed)
        return out, priors
