"""The sanity picture of one recording and camera: one ``<TRACE>.png`` per
trace beside its trace table, showing the label, the combined prediction
with its spread across the overlapping windows, and the beats
``detect_beats`` finds on each. ``scripts/run.py`` draws it the moment the
records are written, so a running fold can be looked at; nothing is scored
here. ``scripts/evaluate.py`` scores a finished run from the trace tables.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.beat_metrics import detect_beats
from src.evaluation.plots import recording_figure
from src.evaluation.rate import MIN_FRAMES
from src.signal_transforms import is_cardiac

FIGURE_DPI = 150
TRACE_COLUMNS = ["frame", "t", "label", "mean", "std", "n"]


def beat_marks(table: pd.DataFrame, fs: float, sig: str) -> dict:
    """``{"ref_peaks": times, "pred_peaks": times}``: the beats detected on
    the label and on the prediction, each over the stretch from its first
    finite sample to its last (gaps inside it filled with the finite mean,
    for the filters), in seconds on the table's time axis."""
    t = table["t"].to_numpy(dtype=np.float64)
    marks = {}
    for side, column in (("ref", "label"), ("pred", "mean")):
        values = table[column].to_numpy(dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        times = np.array([], dtype=np.float64)
        if finite.size >= MIN_FRAMES:
            start, end = int(finite[0]), int(finite[-1]) + 1
            stretch = values[start:end]
            stretch = np.where(np.isfinite(stretch), stretch, values[finite].mean())
            times = t[start + detect_beats(stretch, fs, sig)]
        marks[f"{side}_peaks"] = times
    return marks


def plot_recording(folder, meta: dict) -> list:
    """Draw every trace of one recording-and-camera folder from its trace
    tables alone; returns the figure paths written."""
    folder = Path(folder)
    fs, where = float(meta["fs"]), f"{folder.parent.name} camera {folder.name}"
    written = []
    for sig in (str(sig) for sig in meta["traces"]):
        table = pd.read_csv(folder / f"{sig}.csv", usecols=TRACE_COLUMNS)[TRACE_COLUMNS]
        marks = beat_marks(table, fs, sig) if is_cardiac(sig) else None
        figure = recording_figure(table, marks, sig, f"{sig} {where}")
        path = folder / f"{sig}.png"
        figure.savefig(path, dpi=FIGURE_DPI)
        plt.close(figure)
        written.append(path)
    return written
