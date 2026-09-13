"""One beat detector for every cardiac trace, on the label and on the prediction.

The beat is the atomic unit of the evaluation. This module finds the beats
of one trace, reads each beat's max / mean / min off the raw waveform, and
pairs the predicted beats with the reference ones so misses and spurious
beats are counted, not hidden.

Detection: the trace is cleaned as the heart-rate estimator cleans it
(detrend, zero-phase bandpass to the heart-rate band), multiplied by the
signal's polarity from the registry, and ``scipy.signal.find_peaks`` runs
with the minimum distance between beats set by the top of the band and the
prominence a registry fraction of the cleaned range. Each candidate is then
moved to the extremum of the raw trace within a quarter of the median beat
interval, so the peak is read off the real waveform, not the filtered one.

Levels: a beat spans trough to trough around its peak, the troughs being
the minima of the polarity-signed trace between neighbouring peaks. Over
that span ``max`` is the peak, ``min`` the lower trough and ``mean`` the
area under the curve divided by the duration. The beat's trough time is
its foot: the trough before the peak.

Matching: each predicted beat goes to the nearest reference beat within 40
percent of the median reference interval, closest pairs first, one to one.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.evaluation.rate import BAND, MIN_FRAMES, clean
from src.signal_transforms import beat_config, is_absolute

LEVELS = ("max", "mean", "min")
#: A predicted beat matches a reference beat within this fraction of the
#: median reference inter-beat interval.
MATCH_FRACTION = 0.4
#: A candidate peak is moved to the raw extremum within this fraction of the
#: median inter-beat interval.
REFINE_FRACTION = 0.25
BEAT_COLUMNS = ("beat", "t_ref_peak", "t_ref_trough", "t_pred_peak", "t_pred_trough",
                *(f"{side}_{s}" for side in ("ref", "pred") for s in LEVELS))
_NAN = float("nan")


def _median_interval(peaks: np.ndarray, fs: float) -> float:
    """Samples between beats; one second when there are too few beats to tell."""
    return float(np.median(np.diff(peaks))) if peaks.size > 1 else float(fs)


def detect_beats(trace, fs: float, sig: str) -> np.ndarray:
    """Sample indices of the beats of one finite trace, sorted, unique."""
    trace = np.asarray(trace, dtype=np.float64)
    if trace.size < MIN_FRAMES or not np.all(np.isfinite(trace)):
        raise ValueError(
            f"detect_beats needs a finite trace of at least {MIN_FRAMES} samples")
    config = beat_config(sig)
    signed = trace * config["polarity"]
    cleaned = clean(signed, fs)
    span = float(np.ptp(cleaned))
    if span <= 0:
        return np.array([], dtype=int)
    candidates, _ = find_peaks(cleaned, distance=max(1, int(fs / BAND[1])),
                               prominence=config["prominence"] * span)
    if candidates.size == 0:
        return candidates.astype(int)
    radius = max(1, int(round(REFINE_FRACTION * _median_interval(candidates, fs))))
    refined = []
    for c in candidates:
        lo, hi = max(0, c - radius), min(signed.size, c + radius + 1)
        refined.append(lo + int(np.argmax(signed[lo:hi])))
    return np.unique(np.asarray(refined, dtype=int))


def beat_levels(trace, peaks: np.ndarray, fs: float, polarity: int) -> tuple:
    """``(levels, troughs)``: ``(n_beats, 3)`` of max, mean, min over each
    beat's trough-to-trough span, read off the raw trace, and the sample
    index of each beat's foot, the trough before its peak. Both empty when
    there are no beats."""
    trace = np.asarray(trace, dtype=np.float64)
    if peaks.size == 0:
        return np.empty((0, len(LEVELS))), np.array([], dtype=int)
    signed = trace * polarity
    interval = int(round(_median_interval(peaks, fs)))
    starts = np.concatenate(([max(0, peaks[0] - interval)], peaks[:-1]))
    ends = np.concatenate((peaks[1:], [min(trace.size - 1, peaks[-1] + interval)]))
    rows, feet = [], []
    for start, peak, end in zip(starts, peaks, ends):
        lo = start + int(np.argmin(signed[start:peak + 1]))
        hi = peak + int(np.argmin(signed[peak:end + 1]))
        span = trace[lo:hi + 1]
        rows.append((float(span.max()), float(span.mean()), float(span.min())))
        feet.append(lo)
    return np.asarray(rows, dtype=np.float64), np.asarray(feet, dtype=int)


def match_beats(ref_peaks: np.ndarray, pred_peaks: np.ndarray, fs: float) -> np.ndarray:
    """For every reference beat the index of its predicted beat, or -1."""
    match = np.full(ref_peaks.size, -1, dtype=int)
    if ref_peaks.size == 0 or pred_peaks.size == 0:
        return match
    tolerance = MATCH_FRACTION * _median_interval(ref_peaks, fs)
    gaps = np.abs(ref_peaks[:, None] - pred_peaks[None, :]).astype(np.float64)
    pairs = np.argwhere(gaps <= tolerance)
    if pairs.size == 0:
        return match
    order = np.argsort(gaps[pairs[:, 0], pairs[:, 1]], kind="stable")
    used = np.zeros(pred_peaks.size, dtype=bool)
    for i, j in pairs[order]:
        if match[i] < 0 and not used[j]:
            match[i], used[j] = j, True
    return match


@dataclass
class Beats:
    """The beats of one recording: reference and predicted peaks and troughs
    (sample indices), their levels, and which predicted beat each reference
    beat matched (-1 for a miss)."""
    ref_peaks: np.ndarray
    pred_peaks: np.ndarray
    ref_troughs: np.ndarray
    pred_troughs: np.ndarray
    ref_levels: np.ndarray
    pred_levels: np.ndarray
    match: np.ndarray

    @property
    def n_matched(self) -> int:
        return int((self.match >= 0).sum())


def analyse(label, pred, fs: float, sig: str) -> Beats:
    """Detect, level and match the beats of one finite label / prediction pair."""
    polarity = beat_config(sig)["polarity"]
    ref_peaks, pred_peaks = detect_beats(label, fs, sig), detect_beats(pred, fs, sig)
    ref_levels, ref_troughs = beat_levels(label, ref_peaks, fs, polarity)
    pred_levels, pred_troughs = beat_levels(pred, pred_peaks, fs, polarity)
    return Beats(ref_peaks, pred_peaks, ref_troughs, pred_troughs,
                 ref_levels, pred_levels, match_beats(ref_peaks, pred_peaks, fs))


def beat_rows(beats: Beats, fs: float, t0: float, sig: str) -> pd.DataFrame:
    """One row per reference beat: its peak and trough times, its matched
    predicted beat's (blank on a miss) and, for absolute-class signals, both
    beats' levels. ``t0`` is the scored stretch's start in seconds on the
    trace table's time axis, so the times line up with it."""
    levels = is_absolute(sig)
    rows = []
    for i, (peak, trough, j) in enumerate(zip(beats.ref_peaks, beats.ref_troughs, beats.match)):
        row = {"beat": i,
               "t_ref_peak": t0 + peak / fs, "t_ref_trough": t0 + trough / fs,
               "t_pred_peak": t0 + beats.pred_peaks[j] / fs if j >= 0 else _NAN,
               "t_pred_trough": t0 + beats.pred_troughs[j] / fs if j >= 0 else _NAN}
        for k, s in enumerate(LEVELS):
            row[f"ref_{s}"] = float(beats.ref_levels[i, k]) if levels else _NAN
            row[f"pred_{s}"] = (float(beats.pred_levels[j, k])
                                if levels and j >= 0 else _NAN)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(BEAT_COLUMNS))
