"""One paired measurement scored: the waveform metrics of one trace, and the
derived parameters read off it.

A scorer takes one measurement's ``(label, pred, fs)`` arrays and returns a
flat dict; :func:`signal_measurements` runs one over every measurement of a
signal, :func:`rate_measurements` runs :func:`score_rate` over every signal
of each measurement together. :data:`PARAMETERS` lists the derived
parameters, each with its scorer; a new one is a scorer and a line there.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from neural_methods.loss.ccc import MAX_LAG_SECONDS, lag_frames, lagged_pair
from src.evaluation import progress
from src.evaluation.beat_metrics import detect_beats
from src.evaluation.pooling import MEASUREMENT_KEY
from src.evaluation.rate import MIN_FRAMES, clean, fuse, rate_of, snr, spectrum
from src.signal_transforms import is_cardiac

#: The waveform metrics summarised across measurements; ``lag`` is not one,
#: its mean across measurements meaning nothing.
WAVEFORM_METRICS = ("mae", "rmse", "ccc")
_NAN = float("nan")


# ---------------------------------------------------------------------------
# Agreement between two sequences
# ---------------------------------------------------------------------------
def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return _NAN
    return float(np.corrcoef(a, b)[0, 1])


def ccc(a, b) -> float:
    """Lin's concordance: correlation penalised by disagreement in level."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    r = pearson(a, b)
    if not np.isfinite(r):
        return _NAN
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denominator = va + vb + (a.mean() - b.mean()) ** 2
    return float(2 * r * np.sqrt(va * vb) / denominator) if denominator > 0 else _NAN


def lagged_ccc(ref: np.ndarray, pred: np.ndarray, fs: float) -> tuple:
    """``(ccc, lag)``: Lin's concordance at the best lag within
    ``MAX_LAG_SECONDS``, and that lag in seconds, positive when the
    prediction is delayed against the reference. The training loss's
    counterpart (``neural_methods.loss.ccc``): the label is measured at a
    different site from the one the camera sees, and the transit delay
    between them is not the model's error. Non-finite pairs are dropped after
    the shift, so the search sees the traces contiguous."""
    max_lag = min(lag_frames(MAX_LAG_SECONDS, fs), ref.size - 2)
    best, best_lag = _NAN, _NAN
    for lag in range(-max_lag, max_lag + 1):
        p, r = lagged_pair(pred, ref, lag)
        ok = np.isfinite(p) & np.isfinite(r)
        value = ccc(p[ok], r[ok])
        if np.isfinite(value) and not value <= best:
            best, best_lag = value, lag / fs
    return best, best_lag


# ---------------------------------------------------------------------------
# Running a scorer over the measurements
# ---------------------------------------------------------------------------
def signal_measurements(measurements: pd.DataFrame, signal: str, score,
                        fs: float, desc: str) -> pd.DataFrame:
    """One row per measurement of ``signal``: the key and what ``score``
    returns for its ``(label, pred, fs)``; ``desc`` labels the progress."""
    groups = measurements[measurements["signal"] == signal].groupby(MEASUREMENT_KEY)
    rows = []
    for key, g in progress(groups, desc, total=groups.ngroups):
        rows.append({**dict(zip(MEASUREMENT_KEY, key)),
                     **score(g["label"].to_numpy(), g["mean"].to_numpy(), fs)})
    return pd.DataFrame(rows)


def rate_measurements(measurements: pd.DataFrame, fs: float, desc: str) -> pd.DataFrame:
    """One row per measurement: the key and :func:`score_rate` over every
    signal the measurement carries; ``desc`` labels the progress."""
    groups = measurements.groupby(MEASUREMENT_KEY)
    rows = []
    for key, g in progress(groups, desc, total=groups.ngroups):
        traces = {sig: (part["label"].to_numpy(), part["mean"].to_numpy())
                  for sig, part in g.groupby("signal")}
        rows.append({**dict(zip(MEASUREMENT_KEY, key)), **score_rate(traces, fs)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The waveform metrics
# ---------------------------------------------------------------------------
def score_waveform(label: np.ndarray, pred: np.ndarray, fs: float) -> dict:
    """Sample-wise agreement over one measurement: Lin's concordance at its
    best lag within half a second (:func:`lagged_ccc`),
    that lag in seconds, and the mean absolute error and RMSE of the
    prediction against the reference once shifted by that lag, over the
    samples both still cover."""
    concordance, lag = lagged_ccc(label, pred, fs)
    shift = int(round(lag * fs)) if np.isfinite(lag) else 0
    aligned_pred, aligned_label = lagged_pair(pred, label, shift)
    error = aligned_pred - aligned_label
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "ccc": concordance, "lag": lag}


def waveform_measurements(measurements: pd.DataFrame, fs: float) -> dict:
    """``{signal: frame}``: one row per measurement of each signal present,
    scored by :func:`score_waveform`."""
    return {sig: signal_measurements(
                measurements, sig, score_waveform, fs,
                f"{sig} waveform metrics (CCC searched over ±{MAX_LAG_SECONDS:g} s of lags)")
            for sig in sorted(measurements["signal"].unique())}


# ---------------------------------------------------------------------------
# The derived parameters
# ---------------------------------------------------------------------------
def score_mean(label: np.ndarray, pred: np.ndarray, fs: float) -> dict:
    """The mean level over one measurement, reference and prediction, and
    the error (prediction minus reference)."""
    ref_mean, pred_mean = float(label.mean()), float(pred.mean())
    return {"ref_mean": ref_mean, "pred_mean": pred_mean,
            "err_mean": pred_mean - ref_mean}


def _extremum_level(label: np.ndarray, pred: np.ndarray, fs: float,
                    sign: int, name: str) -> dict:
    """The mean level at one measurement's beats on each side, the beats
    detected on that side's own trace, as peaks (``sign`` +1) or troughs
    (-1, the trace negated for the detector); the error is prediction
    minus reference. A side with no beat falls back to its single extreme
    sample (the maximum for peaks, the minimum for troughs), and
    ``<side>_<name>_fallback`` says so."""
    row = {}
    for side, trace in (("ref", label), ("pred", pred)):
        beats = detect_beats(sign * trace, fs, "ABP")
        fallback = beats.size == 0
        level = float((sign * trace).max() * sign) if fallback else float(trace[beats].mean())
        row[f"n_{side}_beats"] = int(beats.size)
        row[f"{side}_{name}"] = level
        row[f"{side}_{name}_fallback"] = fallback
    row[f"err_{name}"] = row[f"pred_{name}"] - row[f"ref_{name}"]
    return row


def score_systolic(label: np.ndarray, pred: np.ndarray, fs: float) -> dict:
    """The mean peak level over one measurement's beats."""
    return _extremum_level(label, pred, fs, +1, "systolic")


def score_diastolic(label: np.ndarray, pred: np.ndarray, fs: float) -> dict:
    """The mean trough level over one measurement's beats."""
    return _extremum_level(label, pred, fs, -1, "diastolic")


def score_rate(traces: dict, fs: float) -> dict:
    """The heart rate of one measurement from its cardiac traces fused in
    the spectrum (``src.evaluation.rate.fuse``), each side from its own
    traces: ``{signal: (label, pred)}`` in, reference and predicted rate in
    bpm, the error, and the SNR of the fused prediction around the
    reference rate."""
    freqs, ref_powers, pred_powers = None, [], []
    for sig, (label, pred) in traces.items():
        if not is_cardiac(sig) or label.size < MIN_FRAMES:
            continue
        freqs, ref_power = spectrum(clean(label, fs), fs)
        _, pred_power = spectrum(clean(pred, fs), fs)
        ref_powers.append(ref_power)
        pred_powers.append(pred_power)
    if not ref_powers:
        return {"n_traces": 0, "ref_hr": np.nan, "pred_hr": np.nan,
                "err_hr": np.nan, "snr": np.nan}
    ref_fused, pred_fused = fuse(freqs, ref_powers), fuse(freqs, pred_powers)
    ref_hr, pred_hr = rate_of(freqs, ref_fused), rate_of(freqs, pred_fused)
    return {"n_traces": len(ref_powers), "ref_hr": ref_hr, "pred_hr": pred_hr,
            "err_hr": pred_hr - ref_hr, "snr": snr(freqs, pred_fused, ref_hr)}


class Parameter(NamedTuple):
    """One derived parameter: its label, file-name key and unit; the signal
    and scorer of its measurement frame (``None`` for the rate, which fuses
    every signal); the suffix of its ``ref_`` / ``pred_`` / ``err_``
    columns; and how it is read, for the progress bar."""
    name: str
    key: str
    unit: str
    signal: str | None
    score: object
    column: str
    how: str


PARAMETERS = (
    Parameter("CVP mean", "CVP_mean", "mmHg", "CVP", score_mean, "mean", "sample mean"),
    Parameter("ABP mean", "ABP_mean", "mmHg", "ABP", score_mean, "mean", "sample mean"),
    Parameter("ABP systolic", "ABP_systolic", "mmHg", "ABP", score_systolic, "systolic",
              "peaks detected on both sides"),
    Parameter("ABP diastolic", "ABP_diastolic", "mmHg", "ABP", score_diastolic, "diastolic",
              "troughs detected on both sides"),
    Parameter("heart rate", "rate", "bpm", None, None, "hr",
              "every cardiac trace's spectrum, fused"),
)


def score_measurements(measurements: pd.DataFrame, fs: float) -> dict:
    """``{parameter name: frame}`` for every entry of :data:`PARAMETERS`."""
    frames = {}
    for p in PARAMETERS:
        if p.signal is None:
            frames[p.name] = rate_measurements(measurements, fs, f"{p.name} ({p.how})")
        else:
            frames[p.name] = signal_measurements(
                measurements, p.signal, p.score, fs, f"{p.name} ({p.how})")
    return frames


def pairs_of(frame: pd.DataFrame, p: Parameter) -> pd.DataFrame:
    """The measurements of ``p`` that have an error, as the generic
    ``subject`` / ``ref`` / ``pred`` / ``error`` columns the agreement
    statistics and figures read, plus ``fallback``: whether either side was
    read off a single extreme sample for want of beats (never, for a
    parameter without beats)."""
    columns = {"participant": "subject", f"ref_{p.column}": "ref",
               f"pred_{p.column}": "pred", f"err_{p.column}": "error"}
    pairs = frame.dropna(subset=[f"err_{p.column}"]).rename(columns=columns)
    flags = [c for c in (f"ref_{p.column}_fallback", f"pred_{p.column}_fallback")
             if c in pairs.columns]
    pairs["fallback"] = pairs[flags].any(axis=1) if flags else False
    return pairs
