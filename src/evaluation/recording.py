"""The evaluation of one recording and camera: its beats, per-signal
metrics and heart rates, written beside the trace tables it read as one
``<TRACE>_beats.csv`` per cardiac trace, ``signals.csv`` and ``rates.csv``,
plus one ``<TRACE>.png`` per trace showing the label, the combined
prediction with its spread across windows, and the detected beats.

The whole covered stretch of the combined trace, from the first covered
frame to the last, is scored once. Per signal the row carries the beat
counts on both sides and how many matched; for absolute-class signals the
mean and SD over the beats of each level (max / mean / min: systolic / MAP
/ diastolic for ABP), the error of the means (prediction minus reference,
the sign every standard uses) and the ISO 81060-2 clause 6.2.5, p. 22,
dead-band error (zero inside the reference mean ± SD, else the distance to
the nearer limit); and the per-sample agreement of the combined prediction
with the label over the stretch, the IEEE 1708 waveform metrics (equations
(3) and (4), p. 28). The prediction is the trace table's ``mean`` column:
the average of every strided window covering the frame.

These files carry no dataset / participant / recording / perspective
columns: the folder's place in the records directory says where it sits,
so this module never needs to know.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.beat_metrics import BEAT_COLUMNS, LEVELS, analyse, beat_rows
from src.evaluation.plots import recording_figure
from src.evaluation.rate import MIN_FRAMES, RATE_METRICS, recording_rates
from src.outputs import FLOAT_FORMAT
from src.signal_transforms import is_absolute, is_cardiac

SIGNALS_NAME, RATES_NAME = "signals.csv", "rates.csv"
FIGURE_DPI = 150
#: The beat times the figure marks, as ``Beats`` attributes.
MARKS = ("ref_peaks", "ref_troughs", "pred_peaks", "pred_troughs")
WAVEFORM_METRICS = ("mad", "rmse", "r", "ccc")
SIGNAL_COLUMNS = (
    "signal", "t_start", "t_end",
    "n_ref_beats", "n_pred_beats", "n_matched",
    *(f"{side}_{s}_{stat}" for s in LEVELS for side in ("ref", "pred")
      for stat in ("mean", "sd")),
    *(f"err_{s}" for s in LEVELS),
    *(f"err_{s}_deadband" for s in LEVELS),
    *(f"waveform_{m}" for m in WAVEFORM_METRICS),
)
RATE_COLUMNS = ("source", *RATE_METRICS)
TRACE_COLUMNS = ["frame", "t", "label", "mean", "std", "n"]
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


def _waveform(ref: np.ndarray, pred: np.ndarray) -> dict:
    error = pred - ref
    return {"waveform_mad": float(np.abs(error).mean()),
            "waveform_rmse": float(np.sqrt((error ** 2).mean())),
            "waveform_r": pearson(pred, ref), "waveform_ccc": ccc(pred, ref)}


# ---------------------------------------------------------------------------
# The trace tables and the stretch they cover
# ---------------------------------------------------------------------------
def read_trace(folder: Path, sig: str) -> pd.DataFrame:
    """The fixed columns of one trace table (``src/outputs.py``)."""
    return pd.read_csv(Path(folder) / f"{sig}.csv", usecols=TRACE_COLUMNS)[TRACE_COLUMNS]


def beats_name(sig: str) -> str:
    """``<TRACE>_beats.csv``: one beats table per cardiac trace."""
    return f"{sig}_beats.csv"


def covered_span(covered: np.ndarray) -> tuple | None:
    """``(start, end)`` sample slice from the first covered frame to the
    last, the one stretch every metric is scored over; ``None`` when no
    window covered anything."""
    frames = np.flatnonzero(covered)
    if frames.size == 0:
        return None
    return int(frames[0]), int(frames[-1]) + 1


def _filled(x: np.ndarray) -> np.ndarray:
    """NaN samples replaced by the finite mean, for the filters."""
    finite = np.isfinite(x)
    return np.where(finite, x, x[finite].mean())


def _errors(s: str, row: dict) -> dict:
    err = row[f"pred_{s}_mean"] - row[f"ref_{s}_mean"]
    sd = row[f"ref_{s}_sd"]
    if np.isfinite(err) and np.isfinite(sd):
        dead = 0.0 if abs(err) <= sd else err - np.sign(err) * sd
    else:
        dead = err
    return {f"err_{s}": err, f"err_{s}_deadband": dead}


def _beat_level_columns(ref_levels: np.ndarray, pred_levels: np.ndarray) -> dict:
    """Mean and SD over the beats of each level, each side, and the errors."""
    row = {}
    for k, s in enumerate(LEVELS):
        for side, levels in (("ref", ref_levels), ("pred", pred_levels)):
            values = levels[:, k]
            row[f"{side}_{s}_mean"] = float(values.mean()) if values.size else _NAN
            row[f"{side}_{s}_sd"] = float(values.std(ddof=1)) if values.size > 1 else _NAN
        row.update(_errors(s, row))
    return row


def _sample_level_columns(ref: np.ndarray, pred: np.ndarray) -> dict:
    """For an absolute signal without beats (SpO2): the level is the sample
    mean, its SD the sample SD, max and min the sample extremes."""
    row = {}
    for side, values in (("ref", ref), ("pred", pred)):
        row[f"{side}_max_mean"], row[f"{side}_max_sd"] = float(values.max()), _NAN
        row[f"{side}_mean_mean"] = float(values.mean())
        row[f"{side}_mean_sd"] = float(values.std(ddof=1)) if values.size > 1 else _NAN
        row[f"{side}_min_mean"], row[f"{side}_min_sd"] = float(values.min()), _NAN
    for s in LEVELS:
        row.update(_errors(s, row))
    return row


# ---------------------------------------------------------------------------
# One folder to its three files
# ---------------------------------------------------------------------------
def score_recording(folder, meta: dict) -> dict:
    """Score every trace of one recording-and-camera folder over the whole
    covered stretch; write and return ``{"beats": {trace: table},
    "signals", "rates"}``."""
    folder = Path(folder)
    fs, traces = float(meta["fs"]), [str(sig) for sig in meta["traces"]]
    tables = {sig: read_trace(folder, sig) for sig in traces}
    # The tables' own time axis, not the row index: under --limit-windows the
    # first covered frame need not be frame 0, so index / fs and the table's
    # "t" column disagree by the covered window's start offset. t_start,
    # t_end and the beat times are on the "t" axis so they line up with the
    # trace tables they sit beside.
    times = tables[traces[0]]["t"].to_numpy(dtype=np.float64)
    covered = np.zeros(len(tables[traces[0]]), dtype=bool)
    for table in tables.values():
        covered |= table["n"].to_numpy() > 0
    signals, rates, cardiac = [], [], {}
    beats = {sig: pd.DataFrame(columns=list(BEAT_COLUMNS))
             for sig in traces if is_cardiac(sig)}
    marks = {sig: {key: [] for key in MARKS} for sig in beats}
    span = covered_span(covered)
    if span is not None:
        start, end = span
        t0, t_end = float(times[start]), float(times[end - 1]) + 1 / fs
        for sig, table in tables.items():
            label = table["label"].to_numpy(dtype=np.float64)[start:end]
            pred = table["mean"].to_numpy(dtype=np.float64)[start:end]
            row = {"signal": sig, "t_start": t0, "t_end": t_end}
            ok = np.isfinite(label) & np.isfinite(pred)
            if ok.sum() >= MIN_FRAMES:
                row.update(_waveform(label[ok], pred[ok]))
                if is_cardiac(sig):
                    label_f, pred_f = _filled(label), _filled(pred)
                    cardiac[sig] = (label_f, pred_f)
                    found = analyse(label_f, pred_f, fs, sig)
                    row.update({"n_ref_beats": found.ref_peaks.size,
                                "n_pred_beats": found.pred_peaks.size,
                                "n_matched": found.n_matched})
                    beats[sig] = beat_rows(found, fs, t0, sig)
                    for key in MARKS:
                        marks[sig][key].extend((t0 + getattr(found, key) / fs).tolist())
                    if is_absolute(sig):
                        row.update(_beat_level_columns(found.ref_levels, found.pred_levels))
                elif is_absolute(sig):
                    row.update(_sample_level_columns(label[ok], pred[ok]))
            signals.append(row)
        rates = recording_rates(cardiac, fs)
    frames = {
        "beats": beats,
        "signals": pd.DataFrame(signals, columns=list(SIGNAL_COLUMNS)),
        "rates": pd.DataFrame(rates, columns=list(RATE_COLUMNS)),
    }
    for sig, table in frames["beats"].items():
        table.to_csv(folder / beats_name(sig), index=False, float_format=FLOAT_FORMAT)
    for name, key in ((SIGNALS_NAME, "signals"), (RATES_NAME, "rates")):
        frames[key].to_csv(folder / name, index=False, float_format=FLOAT_FORMAT)
    where = f"{folder.parent.name} camera {folder.name}"
    for sig, table in tables.items():
        figure = recording_figure(table, marks.get(sig), sig, f"{sig} {where}")
        figure.savefig(folder / f"{sig}.png", dpi=FIGURE_DPI)
        plt.close(figure)
    return frames
