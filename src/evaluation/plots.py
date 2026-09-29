"""The figures: one trace of one recording (the label, the combined
prediction with its spread across the overlapping windows, and the detected
beats), and the pooled figures of a run's evaluation report.

Seaborn on the Agg backend. Every function draws and returns the figure; the
caller saves and closes it. Report figures are ``REPORT_WIDTH`` inches wide,
the text width of A4 portrait, so one file serves Markdown, HTML and print.
"""

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

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


# ---------------------------------------------------------------------------
# The pooled figures of a run's evaluation report
# ---------------------------------------------------------------------------
REPORT_WIDTH = 6.3
#: Subjects are told apart by colour up to this many; beyond it no palette
#: keeps them distinct and every point wears the one colour.
MAX_SUBJECT_COLOURS = 8
LINE_COLOUR = "0.35"


def _report_axes(height: float = 3.6, columns: int = 1):
    figure, axes = plt.subplots(1, columns, figsize=(REPORT_WIDTH, height), squeeze=False)
    return figure, axes[0]


def _points(pairs: pd.DataFrame, x, y, axis) -> None:
    """One point per measurement, coloured by subject while they are few."""
    style = {"ax": axis, "s": 30, "edgecolor": "white", "linewidth": 0.5}
    if pairs["subject"].nunique() > MAX_SUBJECT_COLOURS:
        sns.scatterplot(x=x, y=y, color=PRED_COLOUR, **style)
        return
    sns.scatterplot(x=x, y=y, hue=pairs["subject"], palette="colorblind", **style)
    axis.legend(title="subject", fontsize=7, title_fontsize=7, loc="upper center",
                bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)


def bland_altman_figure(pairs: pd.DataFrame, stats: dict, limits: tuple, unit: str,
                        title: str) -> plt.Figure:
    """Error against the mean of reference and prediction, with the mean
    error and the limits of agreement."""
    figure, (axis,) = _report_axes()
    _points(pairs, (pairs["ref"] + pairs["pred"]) / 2, pairs["error"], axis)
    for value, style, name in ((stats["mean_error"], "-", "mean error"),
                               (limits[0], "--", "lower limit"),
                               (limits[1], "--", "upper limit")):
        if np.isfinite(value):
            axis.axhline(value, color=LINE_COLOUR, linestyle=style, linewidth=0.9)
            axis.annotate(f"{name} {value:.3g}", xy=(1.0, value),
                          xycoords=("axes fraction", "data"), xytext=(-2, 2),
                          textcoords="offset points", ha="right", va="bottom",
                          fontsize=7, color=LINE_COLOUR)
    axis.set_xlabel(f"mean of reference and prediction ({unit})")
    axis.set_ylabel(f"prediction − reference ({unit})")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def agreement_figure(pairs: pd.DataFrame, unit: str, title: str) -> plt.Figure:
    """Prediction against reference with the line of identity."""
    figure, (axis,) = _report_axes(height=4.2)
    _points(pairs, pairs["ref"], pairs["pred"], axis)
    low = float(min(pairs["ref"].min(), pairs["pred"].min()))
    high = float(max(pairs["ref"].max(), pairs["pred"].max()))
    pad = 0.05 * (high - low) or 1.0
    axis.plot([low - pad, high + pad], [low - pad, high + pad], color=LINE_COLOUR,
              linestyle="--", linewidth=0.9, zorder=0)
    axis.set_xlabel(f"reference ({unit})")
    axis.set_ylabel(f"prediction ({unit})")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def histogram_figure(values, label: str, title: str, edges=()) -> plt.Figure:
    """The spread of one quantity; ``edges`` are drawn as vertical lines (the
    band edges of ISO 81060-3 clause 4.3.3)."""
    figure, (axis,) = _report_axes(height=3.0)
    sns.histplot(x=np.asarray(values, dtype=np.float64), ax=axis, color=PRED_COLOUR,
                 edgecolor="white")
    for edge in edges:
        axis.axvline(edge, color=LINE_COLOUR, linestyle="--", linewidth=0.9)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel(label)
    axis.set_ylabel("count")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def counts_figure(columns: dict, title: str) -> plt.Figure:
    """One bar chart of counts per value for each ``{label: values}``."""
    figure, axes = _report_axes(height=3.0, columns=len(columns))
    for axis, (label, values) in zip(axes, columns.items()):
        counts = values.dropna().map(
            lambda v: f"{v:g}" if isinstance(v, float) else str(v)).value_counts().sort_index()
        sns.barplot(x=counts.index.to_numpy(), y=counts.to_numpy(), ax=axis,
                    color=PRED_COLOUR, width=0.6)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel(label)
        axis.set_ylabel("count" if axis is axes[0] else "")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def box_figure(frame: pd.DataFrame, value: str, label: str, title: str) -> plt.Figure:
    """One box per trace of a per-measurement metric, with every measurement
    as a point."""
    figure, (axis,) = _report_axes(height=3.0)
    sns.boxplot(data=frame, x="signal", y=value, ax=axis, color=PRED_COLOUR,
                width=0.5, fliersize=0, boxprops={"alpha": 0.4})
    sns.stripplot(data=frame, x="signal", y=value, ax=axis, color=LABEL_COLOUR, size=3)
    axis.set_xlabel("trace")
    axis.set_ylabel(label)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def curves_figure(frame: pd.DataFrame, value: str, label: str, title: str) -> plt.Figure:
    """``value`` against the epoch, one panel per trace (their units differ);
    the line is the mean and the band ± 1 SD over folds and recordings."""
    signals = list(dict.fromkeys(frame["signal"]))
    figure, axes = _report_axes(height=2.8, columns=len(signals))
    for axis, sig in zip(axes, signals):
        sns.lineplot(data=frame[frame["signal"] == sig], x="epoch", y=value, ax=axis,
                     color=PRED_COLOUR, errorbar="sd")
        axis.set_title(sig, fontsize=9)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("epoch")
        axis.set_ylabel(label if axis is axes[0] else "")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure
