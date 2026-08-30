"""Per-window label normalisation and the statistics that generate it.

Ported from CardioHydra src/transforms/labels.py (see the design spec at
docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md). The dataset
normalises each label window at load time and stamps the generating statistics
into the batch, so the model learns waveform shape and absolute scale as
separate problems. Both normalisations share one signature and return the same
four-key stats dict, so the batch layout is identical whichever is used and
either is exactly invertible.
"""

from torch import Tensor

# Canonical stat keys and their order. Consumers iterate or validate against
# this rather than hardcoding key lists.
STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max")

# Numerical guard only: a constant trace yields 0/EPS == 0 instead of 0/0 ==
# NaN. That matters because masked losses compute ``values * mask`` and
# NaN * 0 is still NaN, so a NaN could not be masked away after the fact.
EPS = 1e-8


def _stats(trace: Tensor) -> dict[str, Tensor]:
    """Per-window statistics of one ``(T,)`` label trace, as 0-dim tensors."""
    return {
        "mean": trace.mean(),                # ()
        "std": trace.std(correction=1),      # () unbiased
        "min": trace.amin(),                 # ()
        "max": trace.amax(),                 # ()
    }


def _align(stat: Tensor, sig: Tensor) -> Tensor:
    """Right-pad ``stat`` with singleton dims so it broadcasts against ``sig``.

    Handles both the per-sample case (0-dim stat vs ``(T,)`` signal) and the
    collated case (``(B,)`` stat vs ``(B, T)`` signal).
    """
    return stat.reshape(stat.shape + (1,) * (sig.dim() - stat.dim()))


def zscore(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Z-score one ``(T,)`` trace; returns ``(normed, stats)`` in physical units."""
    stats = _stats(trace)                                             # 4 x ()
    normed = (trace - stats["mean"]) / stats["std"].clamp_min(EPS)    # (T,)
    return normed, stats


def zscore_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Map a z-scored signal back to physical units: ``sig * std + mean``."""
    # Must clamp identically to zscore's forward division, or the round-trip
    # is inexact in the 0 < std < EPS band.
    return sig * _align(stats["std"].clamp_min(EPS), sig) + _align(stats["mean"], sig)


def minmax(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Min-max one ``(T,)`` trace to ``[0, 1]``; same stats dict as ``zscore``."""
    stats = _stats(trace)                                                     # 4 x ()
    span = (stats["max"] - stats["min"]).clamp_min(EPS)                       # ()
    normed = (trace - stats["min"]) / span                                    # (T,)
    return normed, stats


def minmax_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Map a min-maxed signal back to physical units: ``sig * (max - min) + min``."""
    # Must clamp identically to minmax's forward division (see zscore_inverse).
    span = (_align(stats["max"], sig) - _align(stats["min"], sig)).clamp_min(EPS)
    return sig * span + _align(stats["min"], sig)
