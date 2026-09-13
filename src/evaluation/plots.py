"""The per-recording figure of one trace: the label, the combined prediction
with its spread across the overlapping windows, and the detected beats.

Seaborn on the Agg backend. ``recording_figure`` draws and returns the
figure; the caller (``recording.py``) saves and closes it.
"""

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.signal_transforms import signal_unit

sns.set_theme(style="whitegrid", context="paper")
LABEL_COLOUR, PRED_COLOUR = "0.25", "C0"
#: Inches of figure width per second of recording, between the two bounds.
WIDTH_PER_SECOND, MIN_WIDTH, MAX_WIDTH = 0.4, 8.0, 20.0
#: The beat marks, by the ``Beats`` attribute suffix they come from.
MARK_STYLES = {"peaks": ("^", "peak"), "troughs": ("v", "trough")}


def _at(t: np.ndarray, values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """``values`` interpolated at ``times`` over the finite samples, so a
    mark sits on the curve it belongs to."""
    ok = np.isfinite(values)
    if ok.sum() < 2 or times.size == 0:
        return np.full(times.size, np.nan)
    return np.interp(times, t[ok], values[ok])


def recording_figure(trace: pd.DataFrame, marks: dict | None, sig: str, title: str) -> plt.Figure:
    """The whole recording: the label, the combined prediction (the mean over
    the windows covering each frame) with a ± 1 SD band across those
    windows, and, when ``marks`` is given, every detected peak (up
    triangle) and trough (down triangle) on both traces. ``marks`` maps
    ``ref_peaks`` / ``ref_troughs`` / ``pred_peaks`` / ``pred_troughs`` to
    times in seconds."""
    t = trace["t"].to_numpy(dtype=np.float64)
    label = trace["label"].to_numpy(dtype=np.float64)
    mean = trace["mean"].to_numpy(dtype=np.float64)
    std = trace["std"].to_numpy(dtype=np.float64)
    width = min(MAX_WIDTH, max(MIN_WIDTH, WIDTH_PER_SECOND * (t[-1] - t[0])))
    figure, axis = plt.subplots(figsize=(width, 3.6))
    sns.lineplot(x=t, y=label, ax=axis, color=LABEL_COLOUR, linewidth=1.0, label="label")
    sns.lineplot(x=t, y=mean, ax=axis, color=PRED_COLOUR, linewidth=1.0,
                 label="prediction (mean over windows)")
    axis.fill_between(t, mean - std, mean + std, color=PRED_COLOUR, alpha=0.25,
                      linewidth=0, label="± 1 SD across windows")
    for side, word, values, colour in (("ref", "label", label, LABEL_COLOUR),
                                       ("pred", "predicted", mean, PRED_COLOUR)):
        for kind, (marker, name) in MARK_STYLES.items():
            times = np.asarray((marks or {}).get(f"{side}_{kind}", []), dtype=np.float64)
            if times.size == 0:
                continue
            sns.scatterplot(x=times, y=_at(t, values, times), ax=axis, marker=marker, s=36,
                            color=colour, edgecolor="white", linewidth=0.4, zorder=3,
                            label=f"{word} {name}s ({times.size})")
    axis.set_xlabel("time (s)")
    axis.set_ylabel(signal_unit(sig))
    axis.set_title(title, fontsize=10)
    # Below the axes, not on them: a legend inside would sit on the traces.
    axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=7,
                frameon=False)
    figure.tight_layout()
    return figure
