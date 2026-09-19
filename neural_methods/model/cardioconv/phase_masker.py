"""Wavelet phase -> one soft mask per trace, plus background, per channel."""

import math

import torch
import torch.nn as nn
from einops import rearrange, reduce

from neural_methods.model.cardioconv.utils import (
    circular_mean, circular_variance, neighbour_circular_difference, present_mean,
    sliding_circular_variance, spatial_confidence,
)

#: The explicit per-pixel phase features the network reads.
N_FEATURES = 4


class PhaseMasker(nn.Module):
    """Which pulsing pixels belong to which trace, told apart by phase.

    The traces share a heart rate and differ in when their pulse arrives, so
    the pixels of each sit at their own phase. Four per-pixel features are
    taken from the wrapped phase, every one circular-aware so no unwrapping
    is needed:

    1. the circular mean over the scales, the pixel's dominant phase;
    2. harmonic coherence, ``1 -`` the circular variance over the scales: a
       true periodic signal holds its phase across harmonics;
    3. spatial coherence, ``1 -`` the normalised difference from the
       neighbours, shrunk toward the no-information value 0.5 by how much of
       its neighbourhood the pixel really has, so border pixels do not read
       as coherent on the strength of fewer neighbours;
    4. temporal stability, ``1 -`` the circular variance inside a sliding
       window of about a second.

    A separable 3-D convolution maps them to one logit per trace and one for
    background, and the softmax across those is where the traces compete for
    each pixel: background absorbs what pulses but is incoherent (clothing
    edges, hard boundaries). Every channel goes through the same network on
    its own.

    Nothing labels a mask: which one becomes which trace is decided by the
    waveform predicted through it. The priors each trace's mask answers to are
    the features again, weighted by the mask — ``harmonic_coherence``,
    ``spatial_coherence`` and ``temporal_consistency`` penalise weight on
    incoherent pixels — with ``phase_tv`` and ``phase_smoothness`` on the mask
    itself.

    Args:
      n_traces: masks to produce, background not counted.
      hidden: channels of the separable convolution.
      temporal_kernel: frames the temporal smoothing sees; odd.
      coherence_kernel: the spatial neighbourhood of feature 3; odd.
      stability_window: frames in the sliding window of feature 4.
    """

    def __init__(self, n_traces: int, hidden: int = 16, temporal_kernel: int = 15,
                 coherence_kernel: int = 3, stability_window: int = 30):
        super().__init__()
        self.coherence_kernel = coherence_kernel
        self.stability_window = stability_window
        # Replicate padding in space and time: zeros blended in at the border
        # left a bright ring in the softmaxed masks.
        self.net = nn.Sequential(
            nn.Conv3d(N_FEATURES, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0), padding_mode="replicate"),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, n_traces + 1, kernel_size=1),
        )

    def forward(self, phase: torch.Tensor, present: torch.Tensor) -> tuple:
        """``(B, C, T, h, w, K)`` wrapped phase and ``(B, C)`` presence -> the
        trace masks ``(B, C, T, h, w, S)``, background dropped, and each
        trace's priors, ``[{term: ()}]`` in the masks' order."""
        b = phase.shape[0]
        height, width = phase.shape[3:5]
        mean = circular_mean(phase, dim=-1)
        harmonic_variance = circular_variance(phase, dim=-1)
        neighbour_difference = neighbour_circular_difference(mean, self.coherence_kernel)
        window_variance = sliding_circular_variance(mean, self.stability_window)
        confidence = spatial_confidence(height, width, self.coherence_kernel, mean)

        spatial_coherence = 0.5 + confidence * (0.5 - neighbour_difference / math.pi)
        features = torch.stack(
            [mean, 1.0 - harmonic_variance, spatial_coherence, 1.0 - window_variance], dim=1)
        logits = self.net(rearrange(features, "b f c t h w -> (b c) f t h w"))
        masks = rearrange(logits, "(b c) m t h w -> b c t h w m", b=b).softmax(dim=-1)[..., :-1]

        priors = []
        for mask in masks.unbind(dim=-1):
            tv = (reduce((mask[..., 1:, :] - mask[..., :-1, :]).abs(), "b c t h w -> b c", "mean")
                  + reduce((mask[..., 1:] - mask[..., :-1]).abs(), "b c t h w -> b c", "mean"))
            smoothness = reduce((mask[:, :, 1:] - mask[:, :, :-1]).abs(), "b c t h w -> b c", "mean")
            weighted = {
                "harmonic_coherence": harmonic_variance * mask,
                "spatial_coherence": neighbour_difference * mask * confidence,
                "temporal_consistency": window_variance * mask,
            }
            terms = {name: present_mean(reduce(value, "b c t h w -> b c", "mean"), present)
                     for name, value in weighted.items()}
            terms["phase_tv"] = present_mean(tv, present)
            terms["phase_smoothness"] = present_mean(smoothness, present)
            priors.append(terms)
        return masks, priors
