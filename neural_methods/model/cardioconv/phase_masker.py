"""Wavelet phase, relative to each channel's own pulse -> one mask per trace, plus background."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, reduce, repeat
from torch.utils.checkpoint import checkpoint

from neural_methods.model.cardioconv.utils import present_mean, spatial_resultant, temporal_resultant, unit

#: Per harmonic: cos and sin of the relative phase, coherence over the
#: harmonic's scales, temporal stability and spatial coherence of the
#: relative phase.
FEATURES_PER_HARMONIC = 5

#: The gain of each trace's cosine phase bin, in logits: a pixel whose
#: relative phase holds at a trace's centre gets ``+PHASE_GAIN``, one in
#: antiphase ``-PHASE_GAIN``. Fixed, so the bound on the network's
#: correction is a bound on everything that is learned beside the centres.
#: Above the network's pairwise reach (2 x ``PHASE_BOUND``), so where the
#: phase holds it decides: with the network able to overrule it (gain 1,
#: bound 1.5), held-out CVP shape fell from 0.39 to 0.19 at epoch 8
#: (experiment 7, 7b against 7c).
PHASE_GAIN = 2.0

#: Prior odds that a pulsatile pixel with no phase evidence is noise rather
#: than a given trace: the vessels are a few percent of the neck's projected
#: area (2.3 % of the grid in the synthetic neck), so about 40 to 1. The
#: background's bin starts here, ``ln 40 = 3.7`` nats, and the Rayleigh
#: evidence ``n R^2`` is taken off it; at ``R = 1`` on a 10 s window it ends
#: near ``-2.7``, beside the antiphase trace's ``-PHASE_GAIN``.
BACKGROUND_PRIOR_ODDS = 40.0


class PhaseMasker(nn.Module):
    """Which pulsing pixels belong to which trace, told apart by phase.

    The traces share a heart rate and differ in when their pulse arrives, so
    the pixels of each sit at their own phase. The absolute phase of a pixel
    turns with time and says nothing by itself; what tells an artery from a
    vein is the phase *relative* to the channel's pulse. So at each harmonic
    the field is referred to the pulsatility-weighted mean field of its
    channel and frame (a constant, not a path for gradients), which gives a
    unit vector per pixel that stands still over a beat. Five features per
    harmonic go to the network, every one circular-aware: the relative
    phase's cosine and sine; coherence over the harmonic's scales (a true
    periodic signal holds its phase across them); the relative phase's
    stability over ``stability_window`` frames; and its agreement with the
    ``coherence_kernel`` neighbourhood.

    Phase decides, the network nudges. Each trace's logit is a cosine bin of
    the relative phase at the fundamental, ``PHASE_GAIN * cos(phase -
    centre)``, averaged over the window's frames: ``PHASE_GAIN * R *
    cos(mean phase - centre)``, where ``R`` is the frame-mean resultant, 1 for
    a pixel whose relative phase holds and 0 for one whose phase wanders.
    The centres are learned per channel and trace (they start evenly spaced
    around the circle; per channel, because which vessel dominates a
    channel's pulse, and so sits at relative phase zero, differs between
    colour, infrared and depth). The background's bin is the log-odds of
    "no coherent source here": ``ln BACKGROUND_PRIOR_ODDS - n R^2``. ``R`` is
    the mean resultant length of the relative phase over the window, the
    circular measure of how steadily it holds and a monotone function of the
    pixel's band SNR (phase jitter variance about ``1 / (2 SNR)``, so ``R``
    about ``exp(-1 / (4 SNR))``); ``n R^2`` is the Rayleigh statistic, the
    log-likelihood ratio of a coherent source against uniform phase over
    ``n`` independent samples; and ``n`` is the window's beats over the
    wavelet's correlation length, because the Morlet's time support makes
    neighbouring coefficients of even a noise pixel agree, so a 10 s window
    holds about six independent phases, not 300, and noise ``R`` sits near
    0.3 with a 95th percentile near 0.6 (measured, notes experiment 13). The
    prior is the base rate: the vessels are a few percent of the neck, so a
    pulsatile pixel with no phase evidence is noise 40 to 1 against each
    trace. On a 10 s window the background therefore wins below ``R`` of
    about 0.6, which is above 95 % of noise pixels and below the neck's
    coherent spots (``R`` 0.8-1.0), and at ``R = 1`` its bin is near
    ``-PHASE_GAIN``, the antiphase trace's. Until 2026-10-14 the bin was zero,
    which taxed a perfectly coherent pixel 12 % for the background and split
    a wandering one evenly three ways; the background then held a quarter of
    the pulsatile mass in every run as a floor, not as a region. A separable
    3-D convolution over all the features adds a correction to every class,
    from which two things are removed before it is bounded: each class's
    pulsatility-weighted mean, so no class can be preferred wholesale (the
    network can move pulsatile mass between classes only pixel against
    pixel), and the mean over the classes. It is then bounded to
    ``+-correction_bound`` nats by a tanh, so two classes' logits move
    against each other by at most twice that: under the bins' separation on
    a coherent pixel (``2 R`` between a trace at its centre and the
    background, ``4 R`` between traces in antiphase) once ``R`` is above
    ``correction_bound`` and ``correction_bound / 2`` respectively. So where
    the phase holds, the phase decides; where it wanders, the bins shrink
    and the network decides, background included. With every logit bounded
    no class can take a pixel outright: each keeps a few thousandths of
    every pixel, so no trace can be starved to zero. Every channel goes
    through the same network on its own. Nothing labels a mask: which one
    becomes which trace is decided by the waveform predicted through it.

    Before 2026-10-11 the correction was unbounded and the gains learned: the
    network grew a nearly uniform offset of 30 to 120 nats for one trace,
    the other trace's share fell under float32 resolution and the phase
    masker froze (experiments 5 and 6a, CardioHydra's CVP); the background
    died within four epochs of every run. A first bounded version (bound
    1.5, the class means kept) let the same offset return, pinned at the
    bound: CVP held 70-83 % of the pulsatile mass in every channel and the
    background its floor (experiment 7a).

    Priors, all with the pulsatility map detached, so phase never shapes
    the phase-blind map. None is on the background: it is what the traces
    did not claim, and with its bin set by coherence it needs no prior to
    hold or to yield pulsatile mass (the ``background_pulsatile`` hinge above
    an even share, removed 2026-10-14, read zero in every recorded run and
    would have fought the coherence bin on a noisy window).
    ``phase_saturation`` is the mean of ``relu(|c| / bound - 1) ** 2`` over
    the raw correction ``c``: zero inside the bound, and a pull back towards
    it beyond, where the tanh has no gradient left (in experiment 7a the raw
    correction ran to 12-18 nats and the network froze). Each trace's
    ``harmonic_coherence``, ``spatial_coherence`` and ``temporal_consistency``
    are the incoherence under its claimed distribution (pulsatility times
    mask, normalised over the pixels in log space): independent of the
    trace's share, so neither shrinking nor emptying a mask pays.
    ``phase_tv`` is on the mask itself. The masks are one map per window,
    for the reason the pulsatility mask is.

    Args:
      in_channels: video channels ``C``, each with its own trace centres.
      n_traces: masks to produce, background not counted.
      n_harmonics: harmonic groups the scale axis folds into.
      correction_bound: the network's largest move of a class logit, in nats.
      hidden: channels of the separable convolution.
      temporal_kernel: frames the temporal smoothing sees; odd.
      coherence_kernel: the spatial neighbourhood of the coherence feature; odd.
      stability_window: frames in the sliding window of the stability feature.
    """

    def __init__(self, in_channels: int, n_traces: int, n_harmonics: int, correction_bound: float,
                 wavelet_bandwidth: float, hidden: int = 16, temporal_kernel: int = 15,
                 coherence_kernel: int = 3, stability_window: int = 30):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.correction_bound = correction_bound
        # The Morlet's envelope is exp(-t^2 / B) in periods, so neighbouring
        # coefficients decorrelate over about 2 sigma = sqrt(2 B) beats; a
        # window of ``beats`` beats holds ``beats / sqrt(2 B)`` independent
        # phase samples. Checked against white noise through the transform
        # (notes, experiment 13): median R 0.32 on a 10 s window at 67 bpm,
        # the Rayleigh null's sqrt(ln 2 / n) at n = 6.4.
        self.correlation_beats = math.sqrt(2.0 * wavelet_bandwidth)
        self.coherence_kernel = coherence_kernel
        self.stability_window = stability_window
        # Replicate padding in space and time: zeros blended in at the border
        # left a bright ring in the softmaxed masks.
        self.net = nn.Sequential(
            nn.Conv3d(FEATURES_PER_HARMONIC * n_harmonics, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, n_traces + 1, kernel_size=1),
        )
        centres = torch.arange(n_traces).float() * (2 * math.pi / n_traces)
        self.phase_centre = nn.Parameter(repeat(centres, "s -> c s", c=in_channels).clone())  # own memory per channel

    def features(self, field: torch.Tensor, pulsatility: torch.Tensor) -> tuple:
        """``(B, C, T, h, w, K)`` complex field and ``(B, C, h, w)`` pulsatility
        -> the ``(B, C, T, h, w, 5N)`` features and the three ``(B, C, T, h, w, N)``
        coherences they include (over scales, in time, in space)."""
        # Features, not a path (see PulsatilityMasker.forward): the masks train
        # through the pooling weights downstream.
        with torch.no_grad():
            grouped = rearrange(field, "b c t h w (n j) -> b c t h w n j", n=self.n_harmonics)
            centre = grouped[..., grouped.shape[-1] // 2]                             # (B, C, T, h, w, N)
            scale_coherence = unit(grouped).mean(dim=-1).abs()
            reference = unit(einsum(centre, pulsatility.to(centre.dtype),
                                    "b c t h w n, b c h w -> b c t n"))
            relative = unit(centre * rearrange(reference.conj(), "b c t n -> b c t 1 1 n"))
            stability = temporal_resultant(relative, self.stability_window)
            spatial = spatial_resultant(relative, self.coherence_kernel)
            features = torch.cat([relative.real, relative.imag, scale_coherence, stability, spatial],
                                 dim=-1)
        return features, scale_coherence, stability, spatial

    def forward(self, field: torch.Tensor, pulsatility: torch.Tensor, present: torch.Tensor,
                beats: torch.Tensor) -> tuple:
        """``(B, C, T, h, w, K)`` complex field, ``(B, C, h, w)`` pulsatility,
        ``(B, C)`` presence and ``(B,)`` beats in the window at the estimated
        rate -> the trace masks ``(B, C, h, w, S)`` and
        the background mask ``(B, C, h, w)``, one map per window each, each
        trace's priors ``[{term: ()}]`` in the masks' order, and the shared
        priors ``{term: ()}``."""
        b = field.shape[0]
        features, scale_coherence, stability, spatial = self.features(field, pulsatility)
        net = checkpoint(self.net, rearrange(features, "b c t h w f -> (b c) f t h w"),
                         use_reentrant=False)
        # One set of masks per window, from the frames' mean logit (see
        # PulsatilityMasker.forward). No class preferred wholesale: each
        # class's pulsatility-weighted mean correction is removed, then the
        # mean over the classes (which the softmax ignores anyway).
        weight = pulsatility.detach()                                             # (B, C, h, w), one map per window
        correction = reduce(net, "(b c) m t h w -> b c h w m", "mean", b=b)
        correction = correction - reduce(rearrange(weight, "b c h w -> b c h w 1") * correction,
                                         "b c h w m -> b c 1 1 m", "sum")
        correction = correction - reduce(correction, "b c h w m -> b c h w 1", "mean")
        bound = self.correction_bound
        n = self.n_harmonics
        cos = reduce(features[..., 0], "b c t h w -> b c h w 1", "mean")         # the fundamental's unit vector
        sin = reduce(features[..., n], "b c t h w -> b c h w 1", "mean")
        resultant = torch.sqrt(cos ** 2 + sin ** 2)                               # R: 1 holds, 0 wanders
        centre = rearrange(self.phase_centre, "c s -> 1 c 1 1 s")
        bins = PHASE_GAIN * (cos * centre.cos() + sin * centre.sin())             # (B, C, h, w, S)
        # The background's bin: the prior odds of noise, less the Rayleigh
        # evidence for a coherent source, ``n R^2`` over the window's ``n``
        # independent phase samples (its beats over the wavelet's correlation
        # length). See the class docstring.
        independent = rearrange(beats / self.correlation_beats, "b -> b 1 1 1 1").clamp_min(1.0)
        background_bin = math.log(BACKGROUND_PRIOR_ODDS) - independent * resultant ** 2
        logits = torch.cat([bins, background_bin], dim=-1) + bound * torch.tanh(correction / bound)
        classes = logits.softmax(dim=-1)                                          # (B, C, h, w, M)
        maps, background_map = classes[..., :-1], classes[..., -1]

        shared = {"phase_saturation": present_mean(
                      reduce(F.relu(correction.abs() / bound - 1.0) ** 2, "b c h w m -> b c", "mean"), present)}
        # Each trace's claimed distribution, pulsatility times mask normalised
        # over the pixels, in log space: no clamp, so it is the same
        # distribution at any share.
        log_claimed = rearrange(rearrange(weight.log(), "b c h w -> b c h w 1") + logits.log_softmax(dim=-1)[..., :-1],
                                "b c h w s -> b c s (h w)")
        claimed = (log_claimed - torch.logsumexp(log_claimed, dim=-1, keepdim=True)).exp()
        incoherent = torch.stack([1.0 - scale_coherence.mean(dim=-1), 1.0 - spatial.mean(dim=-1),
                                  1.0 - stability.mean(dim=-1)], dim=-1)            # (B, C, T, h, w, 3)
        incoherent = reduce(incoherent, "b c t h w k -> b c (h w) k", "mean")
        incoherence = einsum(claimed, incoherent, "b c s n, b c n k -> b c s k")   # (B, C, S, 3)
        priors = []
        for index, mask_map in enumerate(maps.unbind(dim=-1)):
            tv = (reduce((mask_map[..., 1:, :] - mask_map[..., :-1, :]).abs(), "b c h w -> b c", "mean")
                  + reduce((mask_map[..., 1:] - mask_map[..., :-1]).abs(), "b c h w -> b c", "mean"))
            terms = {name: present_mean(incoherence[:, :, index, k], present)
                     for k, name in enumerate(("harmonic_coherence", "spatial_coherence",
                                               "temporal_consistency"))}
            terms["phase_tv"] = present_mean(tv, present)
            priors.append(terms)
        return maps, background_map, priors, shared
