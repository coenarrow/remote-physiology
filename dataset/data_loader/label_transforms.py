"""Per-window label normalisation and the statistics that generate it.

The dataset
normalises each label window at load time and stamps the generating statistics
into the batch, so the model learns waveform shape and absolute scale as
separate problems. Both normalisations share one signature and return the same
four-key stats dict, so the batch layout is identical whichever is used and
either is exactly invertible.
"""

import torch
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


def finite_stats(trace: Tensor) -> dict[str, Tensor]:
    """``_stats`` over the finite entries of ``trace`` only (deviation 4).

    All-NaN/inf input yields all-zero 0-dim stats (the absent-label
    convention); a single finite entry yields ``std == 0`` rather than the
    NaN that ``std(correction=1)`` would produce. On all-finite input this is
    bit-identical to ``_stats``.
    """
    finite = trace[torch.isfinite(trace)]
    if finite.numel() == 0:
        zero = trace.new_zeros(())
        return {name: zero.clone() for name in STAT_NAMES}
    std = finite.std(correction=1) if finite.numel() > 1 else trace.new_zeros(())
    return {
        "mean": finite.mean(),   # ()
        "std": std,              # ()
        "min": finite.amin(),    # ()
        "max": finite.amax(),    # ()
    }


def apply_norm(trace: Tensor, stats: dict[str, Tensor], mode: str) -> Tensor:
    """Forward zscore/minmax using precomputed ``stats`` — no recomputation.

    The dataset normalises with finite-only stats and emits those same stats,
    so the inverse round-trip is exact at finite positions (deviation 4).
    """
    if mode == "zscore":
        return (trace - stats["mean"]) / stats["std"].clamp_min(EPS)
    if mode == "minmax":
        span = (stats["max"] - stats["min"]).clamp_min(EPS)
        return (trace - stats["min"]) / span
    raise ValueError(f"mode must be 'zscore' or 'minmax', got {mode!r}")
