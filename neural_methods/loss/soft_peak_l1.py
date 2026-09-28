"""Absolute error of the soft systolic (``max``) or diastolic (``min``) value.

The statistic is derived from the predicted waveform itself (a soft
local-extrema map and a temperature softmax, after the legacy
``PhysHydraLoss``), not from a second head, so a waveform can never disagree
with its own systolic and diastolic values.

The evaluation's counterparts are ``err_max`` and ``err_min``
(``src/evaluation/recording.py``), the per-beat extremes of the hard beat
detector averaged over the beats. This is a surrogate for them, not the same
number: at ``TEMPERATURE = 0.03`` the softmax is close to the window's single
most extreme beat. Measured on cached Neckflix labels in 10 s windows, the
soft systolic sits 5.6 mmHg above the beat-mean systolic on ABP (the hard
window maximum is 7.5 above it) and 1.5 mmHg on CVP. Prediction and label are
measured with the same ruler, so the term is consistent with itself, but its
gradient falls on the tallest beat or two of a window. A higher temperature
does not fix it: it closes the systolic gap while the diastolic one opens,
the trough being broad.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

EPS = 1e-8

#: Softness of the local-extrema map and of the peak softmax, both as a
#: fraction of the window's own range — see :func:`soft_peak_stat`.
TAU = 0.2
TEMPERATURE = 0.03

#: Peak-search neighbourhood, as a fraction of the window: a local maximum has
#: to be the largest sample within +/- T/10, which at any plausible heart rate
#: is one beat's worth of context.
PEAK_WIDTH_FRACTION = 5


def soft_peak_stat(signal, scale, kind):
    """Differentiable soft maximum (or minimum) of ``(B, T)``, as ``(B,)``.

    Two soft steps: a local-extrema map (how close each sample is to the
    largest value in its neighbourhood) and a temperature-weighted average of
    the signal at those extrema, in place of a peak detector's
    non-differentiable argmax.

    ``scale`` makes both softness parameters unit-free. ``PhysHydraLoss`` used
    absolute ones, which only work on a normalised signal: at ``tau=0.2`` on a
    raw mmHg trace every sample within 0.2 mmHg of the local max counts as a
    peak and nothing else does, and a ``temperature=0.03`` softmax over
    logits of ~90/0.03 saturates to a hard argmax. Expressed as a fraction of
    the window's own range, the same constants behave identically on a
    z-scored ECG and on a 40 mmHg pulse pressure.
    """
    work = -signal if kind == "min" else signal
    padded = rearrange(work, "b t -> b 1 t")

    width = max(3, (work.shape[-1] // PEAK_WIDTH_FRACTION) | 1)   # odd, >= 3
    local_max = F.max_pool1d(padded, kernel_size=width, stride=1, padding=width // 2)
    local_max = rearrange(local_max, "b 1 t -> b t")

    # Distance below the local maximum, in units of the window's range.
    distance = ((local_max - work) / scale).clamp_min(-1.0)
    peak_map = torch.exp(-distance / TAU)

    logits = work / (TEMPERATURE * scale) + (peak_map + EPS).log()
    weights = torch.softmax(logits, dim=-1)
    peak = (weights * work).sum(dim=-1)
    return -peak if kind == "min" else peak


class SoftPeakL1(nn.Module):
    def __init__(self, kind: str):
        """``kind``: ``'max'`` for the systolic value, ``'min'`` for the diastolic."""
        super().__init__()
        self.kind = kind

    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``.

        The softness scale comes from the *label*, detached: both sides are
        then measured with the same ruler, and the ruler itself carries no
        gradient.
        """
        scale = (label.amax(dim=-1) - label.amin(dim=-1)).detach().clamp_min(EPS)
        scale = rearrange(scale, "b -> b 1")
        return (soft_peak_stat(pred, scale, self.kind)
                - soft_peak_stat(label, scale, self.kind)).abs()

    def extra_repr(self):
        return f"kind={self.kind!r}"
