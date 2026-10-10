"""The figures: one trace of one recording (the label, the combined
prediction with its spread across the overlapping windows, and the detected
beats), the Bland-Altman plot of a pooled parameter, and the sets of them a
run's evaluation writes (the best measurement of each waveform metric per
signal, one Bland-Altman plot per parameter), each with its numbers in a
panel beside the axes.

Seaborn on the Agg backend. Every function draws and returns the figure; the
caller saves and closes it. Pooled figures are ``REPORT_WIDTH`` inches wide,
the text width of A4 portrait.
"""

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.agreement import limits, repeated_measures
from src.evaluation.measurements import PARAMETERS, pairs_of
from src.evaluation.pooling import MEASUREMENT_KEY
from src.signal_transforms import signal_unit

sns.set_theme(style="whitegrid", context="paper")
LABEL_COLOUR, PRED_COLOUR = "0.25", "C0"
#: Inches of figure width per second of recording, between the two bounds.
WIDTH_PER_SECOND, MIN_WIDTH, MAX_WIDTH = 0.4, 8.0, 20.0


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
    windows, and, when ``marks`` is given, every detected beat on both
    traces. ``marks`` maps ``ref_peaks`` / ``pred_peaks`` to times in
    seconds."""
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
        times = np.asarray((marks or {}).get(f"{side}_peaks", []), dtype=np.float64)
        if times.size == 0:
            continue
        sns.scatterplot(x=times, y=_at(t, values, times), ax=axis, marker="^", s=36,
                        color=colour, edgecolor="white", linewidth=0.4, zorder=3,
                        label=f"{word} beats ({times.size})")
    axis.set_xlabel("time (s)")
    axis.set_ylabel(signal_unit(sig))
    axis.set_title(title, fontsize=10)
    # Below the axes, not on them: a legend inside would sit on the traces.
    axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=7,
                frameon=False)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# The pooled figures
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
    """One point per measurement, coloured by subject while they are few; a
    measurement read off an extreme sample for want of beats
    (``pairs["fallback"]``) is a cross."""
    style = {"ax": axis, "s": 30, "edgecolor": "white", "linewidth": 0.5}
    fallback = pairs["fallback"].to_numpy(dtype=bool)
    few = pairs["subject"].nunique() <= MAX_SUBJECT_COLOURS
    for marker, rows, word in (("o", ~fallback, "beats"), ("X", fallback, "no beats, extreme sample")):
        if not rows.any():
            continue
        if few:
            sns.scatterplot(x=x[rows], y=y[rows], hue=pairs["subject"][rows],
                            palette="colorblind", marker=marker, **style)
        else:
            sns.scatterplot(x=x[rows], y=y[rows], color=PRED_COLOUR, marker=marker,
                            label=f"{word} ({rows.sum()})", **style)
    if few:
        axis.legend(title="subject (crosses: no beats, extreme sample)", fontsize=7,
                    title_fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                    ncol=4, frameon=False)
    elif fallback.any():
        axis.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
                    frameon=False)


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


# ---------------------------------------------------------------------------
# The figures of a run's evaluation
# ---------------------------------------------------------------------------
#: The share of the figure width the side panel takes.
PANEL_WIDTH = 0.24
#: The waveform metrics a larger value is better for; the rest are errors.
HIGHER_IS_BETTER = frozenset({"ccc", "r", "cb"})
#: The waveform metrics a best-measurement figure is drawn for.
BEST_WAVEFORM_METRICS = ("mae", "rmse", "ccc")


def with_panel(figure, lines: list):
    """``figure`` with the axes narrowed and ``lines`` of text in a box to
    their right."""
    figure.subplots_adjust(right=1 - PANEL_WIDTH - 0.02)
    figure.text(1 - PANEL_WIDTH, 0.5, "\n".join(lines), fontsize=8, family="monospace",
                va="center", ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7"})
    return figure


def best_waveform_figures(measurements: pd.DataFrame, frames: dict) -> dict:
    """``{figure name: Figure}``: per signal, the measurement each metric in
    :data:`BEST_WAVEFORM_METRICS` rates best (lowest MAE or RMSE, highest
    CCC), drawn as its label and prediction over time."""
    figures = {}
    for sig, frame in frames.items():
        for metric in BEST_WAVEFORM_METRICS:
            scored = frame.dropna(subset=[metric])
            if scored.empty:
                continue
            best = scored.loc[scored[metric].idxmax() if metric in HIGHER_IS_BETTER
                              else scored[metric].idxmin()]
            where = (measurements["signal"] == sig)
            for name in MEASUREMENT_KEY:
                where &= measurements[name] == best[name]
            trace = measurements[where].assign(std=np.nan)
            title = (f"{sig}, best {metric}: {best['recording']} camera "
                     f"{best['perspective']} segment {best['segment']}")
            figure = recording_figure(trace, None, sig, title)
            figures[f"waveform_{sig}_best_{metric}"] = with_panel(figure, [
                f"mae  {best['mae']:.2f}",
                f"rmse {best['rmse']:.2f}",
                f"ccc  {best['ccc']:.3f}",
                f"r    {best['r']:.3f}",
                f"cb   {best['cb']:.3f}",
                f"lag  {best['lag']:.2f} s",
            ])
    return figures


def bland_altman_figures(frames: dict) -> dict:
    """``{figure name: Figure}``: one Bland-Altman plot per parameter, the
    error against the mean of reference and prediction, with the mean error
    and the limits of agreement on the corrected SD."""
    figures = {}
    for p in PARAMETERS:
        pairs = pairs_of(frames[p.name], p)
        if pairs.empty:
            continue
        stats = repeated_measures(pairs["error"], pairs["subject"])
        figure = bland_altman_figure(pairs, stats, limits(stats), p.unit,
                                     f"{p.name}: Bland-Altman")
        figures[f"bland_altman_{p.key}"] = with_panel(figure, [
            f"n          {stats['n']}",
            f"subjects   {stats['k']}",
            f"mean error {stats['mean_error']:.2f} {p.unit}",
            f"s_corr     {stats['s_corr']:.2f} {p.unit}",
            f"ICC        {stats['icc']:.2f}",
            f"N_ind      {stats['n_ind']:.0f}",
            f"fallback   {int(pairs['fallback'].sum())}",
        ])
    return figures
