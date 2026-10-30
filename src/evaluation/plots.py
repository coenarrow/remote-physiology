"""The figures: one trace of one recording (the label, the combined
prediction with its spread across the overlapping windows, and the detected
beats), the Bland-Altman plot of a pooled parameter, and the sets of them a
run's evaluation writes (the best-CCC measurement per signal, one
Bland-Altman plot per parameter), each with its numbers in a panel beside
the axes.

Seaborn on the Agg backend. Every function draws and returns the figure; the
caller saves and closes it. The evaluation's figures share one layout: the
axes on the left, a panel of numbers and the legend in a column on the right.
"""

import matplotlib
matplotlib.use("Agg")            # write files; never open a window on a cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.agreement import limits, repeated_measures
from src.evaluation.beat_metrics import detect_beats
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
# The figures of a run's evaluation: one layout
# ---------------------------------------------------------------------------
# Each is a seaborn figure-level plot, its axes on the left, and on the
# right a column holding a panel of numbers hung from the top of the axes
# with the legend in a framed box below it, two columns wide.
#: The axes' height in inches and width-to-height ratio.
FIGURE_HEIGHT, FIGURE_ASPECT = 2.8, 2.3
#: Where the axes end and the right column starts and how wide it is, in
#: figure fractions; the top edge both hang from.
AXES_RIGHT, COLUMN_LEFT, COLUMN_WIDTH, TOP = 0.72, 0.745, 0.245, 0.96
#: The panel's row height, block gap and padding, in figure fractions.
PANEL_ROW, PANEL_GAP, PANEL_PAD = 0.055, 0.03, 0.03
PANEL_FONT, PANEL_SMALL_FONT = 7, 6.5
#: seaborn's default palette: the reference wears its first colour, the
#: prediction its second, a pooled measurement the first.
PALETTE = sns.color_palette()
LINE_COLOUR = "0.3"


def _panel(figure: plt.Figure, blocks: list) -> float:
    """A rounded panel at the top of the right column holding ``blocks``,
    each a list of rows; a row is ``(label, value)`` right-aligned, or
    ``(label, ref, pred)`` under a small ``ref`` / ``pred`` header that a
    block's leading ``(unit,)`` row supplies. Returns the panel's height."""
    rows = sum(len(block) for block in blocks)
    height = PANEL_PAD * 2 + PANEL_ROW * rows + PANEL_GAP * (len(blocks) - 1)
    panel = figure.add_axes((COLUMN_LEFT, TOP - height, COLUMN_WIDTH, height))
    panel.set_axis_off()
    panel.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                   facecolor="white", edgecolor="0.8", linewidth=0.8,
                                   transform=panel.transAxes, clip_on=False))
    left, mid, right = 0.08, 0.68, 0.92
    step = PANEL_ROW / height
    y = 1 - (PANEL_PAD + PANEL_ROW / 2) / height
    for index, block in enumerate(blocks):
        if index:
            y -= PANEL_GAP / height
        for row in block:
            if len(row) == 1:
                for x, word, align in ((left, row[0], "left"), (mid, "ref", "right"),
                                       (right, "pred", "right")):
                    panel.text(x, y, word, fontsize=PANEL_SMALL_FONT, color="0.4",
                               va="center", ha=align)
            else:
                panel.text(left, y, row[0], fontsize=PANEL_FONT, va="center")
                for x, value in zip((mid, right)[-(len(row) - 1):], row[1:]):
                    panel.text(x, y, value, fontsize=PANEL_FONT, va="center", ha="right")
            y -= step
    return height


def _legend(figure: plt.Figure, left_column: list, right_column: list, below: float) -> None:
    """A framed legend of two columns under a panel of height ``below``. A
    legend fills column by column, so the shorter column is padded with
    blank entries to the longer's length."""
    blank = Line2D([], [], linestyle="", label="")
    length = max(len(left_column), len(right_column))
    handles = (left_column + [blank] * (length - len(left_column))
               + right_column + [blank] * (length - len(right_column)))
    figure.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.8,
                  handletextpad=0.5, handlelength=1.4, borderaxespad=0,
                  bbox_to_anchor=(COLUMN_LEFT, TOP - below - PANEL_GAP),
                  bbox_transform=figure.transFigure, frameon=True, fancybox=True,
                  fontsize=PANEL_FONT)


def _figure(grid: sns.FacetGrid) -> plt.Figure:
    """The grid's figure with its axes narrowed to leave the right column."""
    figure = grid.figure
    figure.subplots_adjust(top=TOP, bottom=0.16, left=0.10, right=AXES_RIGHT)
    return figure


# ---------------------------------------------------------------------------
# The best-CCC waveform
# ---------------------------------------------------------------------------
BEST_SERIES = {"reference": PALETTE[0], "prediction": PALETTE[1]}
BEST_LINE_WIDTH = 1.1
#: Per signal, the derived parameters the panel lists as a reference /
#: prediction pair: ``(label, PARAMETERS name, column suffix)``.
PANEL_PARAMETERS = {
    "ABP": (("systolic", "ABP systolic", "systolic"),
            ("diastolic", "ABP diastolic", "diastolic"),
            ("mean", "ABP mean", "mean")),
    "CVP": (("mean", "CVP mean", "mean"),),
}
#: The signals whose extremes are read at detected beats, drawn as markers.
BEAT_SIGNALS = frozenset({"ABP"})


def _row_of(frame: pd.DataFrame, key: pd.Series) -> pd.Series:
    """The one row of ``frame`` at the measurement ``key`` names."""
    where = np.ones(len(frame), dtype=bool)
    for name in MEASUREMENT_KEY:
        where &= (frame[name].astype(str) == str(key[name])).to_numpy()
    return frame[where].iloc[0]


def best_ccc_figure(trace: pd.DataFrame, best: pd.Series, parameters: dict,
                    sig: str, fs: float) -> plt.Figure:
    """One measurement over time: the reference and, shifted to the lag its
    metrics were scored at (a positive lag delays it against the reference,
    so it moves earlier), the prediction; each side's mean dotted in its
    colour and, for a :data:`BEAT_SIGNALS` signal, the beats its systolic
    and diastolic were averaged over as up and down triangles. The panel
    holds the measurement's CCC, MAE, RMSE and lag and its
    :data:`PANEL_PARAMETERS` as reference / prediction pairs; the legend
    has the sides left and the marks right. ``parameters`` maps a parameter
    name to its scored frame."""
    unit = signal_unit(sig)
    t0 = trace["t"].min()
    lag = float(best["lag"]) if np.isfinite(best["lag"]) else 0.0
    sides = {"reference": (trace["t"].to_numpy() - t0, trace["label"].to_numpy()),
             "prediction": (trace["t"].to_numpy() - t0 - lag, trace["mean"].to_numpy())}
    span = (0.0, float(sides["reference"][0].max()))
    long = pd.concat([pd.DataFrame({"t": t, "value": v, "series": s})
                      for s, (t, v) in sides.items()], ignore_index=True)
    long = long[long["t"].between(*span)]

    grid = sns.relplot(data=long, x="t", y="value", hue="series", kind="line",
                       palette=BEST_SERIES, hue_order=list(BEST_SERIES),
                       linewidth=BEST_LINE_WIDTH, height=FIGURE_HEIGHT, aspect=FIGURE_ASPECT,
                       legend=False)
    axis = grid.ax
    axis.set_xlim(*span)
    axis.set_xlabel("time (s)")
    axis.set_ylabel(f"{sig} ({unit})")
    for side, (t, values) in sides.items():
        colour = BEST_SERIES[side]
        axis.axhline(values.mean(), color=colour, linestyle=":", linewidth=1.0)
        if sig in BEAT_SIGNALS:
            for sign, marker in ((+1, "^"), (-1, "v")):
                beats = detect_beats(sign * values, fs, sig)
                keep = (t[beats] >= span[0]) & (t[beats] <= span[1])
                axis.scatter(t[beats][keep], values[beats][keep], marker=marker, s=22,
                             color=colour, edgecolor="white", linewidth=0.5, zorder=3)
    figure = _figure(grid)

    stats = [("CCC", f"{best['ccc']:.3f}"), ("MAE", f"{best['mae']:.2f} {unit}"),
             ("RMSE", f"{best['rmse']:.2f} {unit}"), ("lag", f"{lag:+.2f} s")]
    pressures = [(unit,)]
    for label, name, column in PANEL_PARAMETERS.get(sig, ()):
        row = _row_of(parameters[name], best)
        pressures.append((label, f"{row[f'ref_{column}']:.1f}", f"{row[f'pred_{column}']:.1f}"))
    height = _panel(figure, [stats, pressures] if len(pressures) > 1 else [stats])

    sides_column = [Line2D([], [], color=c, linewidth=BEST_LINE_WIDTH, label=s)
                    for s, c in BEST_SERIES.items()]
    marks_column = [Line2D([], [], color="0.4", linestyle=":", linewidth=1.0, label="mean")]
    if sig in BEAT_SIGNALS:
        marks_column += [Line2D([], [], marker="^", linestyle="", color="0.4", markersize=5,
                                label="systolic"),
                         Line2D([], [], marker="v", linestyle="", color="0.4", markersize=5,
                                label="diastolic")]
    _legend(figure, sides_column, marks_column, height)
    return figure


def best_waveform_figures(measurements: pd.DataFrame, frames: dict, parameters: dict,
                          fs: float) -> dict:
    """``{figure name: Figure}``: per signal, the measurement with the
    highest CCC drawn by :func:`best_ccc_figure`, named after the signal
    and the recording, camera and segment it is."""
    figures = {}
    for sig, frame in frames.items():
        scored = frame.dropna(subset=["ccc"])
        if scored.empty:
            continue
        best = scored.loc[scored["ccc"].idxmax()]
        where = (measurements["signal"] == sig)
        for name in MEASUREMENT_KEY:
            where &= measurements[name] == best[name]
        name = (f"waveform_{sig}_best_ccc_{best['recording']}_cam{best['perspective']}"
                f"_seg{int(best['segment'])}")
        figures[name] = best_ccc_figure(measurements[where], best, parameters, sig, fs)
    return figures


# ---------------------------------------------------------------------------
# The Bland-Altman plots
# ---------------------------------------------------------------------------
#: The parameters a Bland-Altman plot is drawn for, by ``PARAMETERS`` key.
BLAND_ALTMAN_PARAMETERS = ("CVP_mean", "ABP_mean", "ABP_systolic", "ABP_diastolic", "rate")
POINT_COLOUR = PALETTE[0]


def bland_altman_figure(pairs: pd.DataFrame, unit: str) -> plt.Figure:
    """Error against the mean of reference and prediction, one point per
    measurement (a cross where a side was read off an extreme sample for
    want of beats), with the mean error solid, the 95 % limits of
    agreement on the corrected SD dashed and zero dotted. The panel holds
    the repeated-measures statistics; the legend has the points left and
    the lines right."""
    stats = repeated_measures(pairs["error"], pairs["subject"])
    lower, upper = limits(stats)
    data = pairs.assign(average=(pairs["ref"] + pairs["pred"]) / 2,
                        marker=np.where(pairs["fallback"], "extreme sample", "beats"))
    grid = sns.relplot(data=data, x="average", y="error", kind="scatter",
                       style="marker", style_order=["beats", "extreme sample"],
                       markers={"beats": "o", "extreme sample": "X"},
                       color=POINT_COLOUR, s=18, alpha=0.6, edgecolor="white", linewidth=0.3,
                       height=FIGURE_HEIGHT, aspect=FIGURE_ASPECT, legend=False)
    axis = grid.ax
    for value, style in ((stats["mean_error"], "-"), (lower, "--"), (upper, "--")):
        if np.isfinite(value):
            axis.axhline(value, color=LINE_COLOUR, linestyle=style, linewidth=0.9)
    axis.axhline(0, color="0.6", linestyle=":", linewidth=0.8)
    axis.set_xlabel(f"mean of reference and prediction ({unit})")
    axis.set_ylabel(f"prediction − reference ({unit})")
    figure = _figure(grid)

    fallback = int(pairs["fallback"].sum())
    rows = [("n", f"{stats['n']}"), ("subjects", f"{stats['k']}"),
            ("mean error", f"{stats['mean_error']:.2f} {unit}"),
            ("s_corr", f"{stats['s_corr']:.2f} {unit}"),
            ("LoA", f"{lower:.1f} to {upper:.1f}"),
            ("ICC", f"{stats['icc']:.2f}"), ("N_ind", f"{stats['n_ind']:.0f}")]
    if fallback:
        rows.append(("fallback", f"{fallback}"))
    height = _panel(figure, [rows])

    points_column = [Line2D([], [], marker="o", linestyle="", color=POINT_COLOUR,
                            markersize=4, label="measurement")]
    if fallback:
        points_column.append(Line2D([], [], marker="X", linestyle="", color=POINT_COLOUR,
                                    markersize=4, label="extreme sample"))
    lines_column = [Line2D([], [], color=LINE_COLOUR, linestyle="-", linewidth=0.9, label="mean"),
                    Line2D([], [], color=LINE_COLOUR, linestyle="--", linewidth=0.9,
                           label="95 % LoA")]
    _legend(figure, points_column, lines_column, height)
    return figure


def bland_altman_figures(frames: dict) -> dict:
    """``{figure name: Figure}``: one Bland-Altman plot per
    :data:`BLAND_ALTMAN_PARAMETERS` entry with any measurement."""
    figures = {}
    for p in PARAMETERS:
        if p.key not in BLAND_ALTMAN_PARAMETERS:
            continue
        pairs = pairs_of(frames[p.name], p)
        if pairs.empty:
            continue
        figures[f"bland_altman_{p.key}"] = bland_altman_figure(pairs, p.unit)
    return figures
