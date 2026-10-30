"""The figures: one trace of one recording (the label, the combined
prediction with its spread across the overlapping windows, and the detected
beats), the Bland-Altman plot of a pooled parameter, and the sets of them a
run's evaluation writes (the best, median and worst measurement per
signal, rainclouds of the waveform metrics and of the parameters'
errors, one Bland-Altman plot per parameter), each with its numbers in a panel beside
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
import ptitprince
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
# right a column holding a panel of numbers hung from the top of each axes
# with the legend in a framed box below the last, two columns wide.
#: One axes' height in inches and width-to-height ratio.
FIGURE_HEIGHT, FIGURE_ASPECT = 2.8, 2.3
#: Where the axes end and the right column starts and how wide it is, in
#: figure fractions; the top edge both hang from.
AXES_RIGHT, COLUMN_LEFT, COLUMN_WIDTH, TOP = 0.72, 0.745, 0.245, 0.96
#: The panel's row height, block gap and padding, as fractions of a figure
#: ``FIGURE_HEIGHT`` tall; a taller figure scales them down to keep them
#: the same size on paper.
PANEL_ROW, PANEL_GAP, PANEL_PAD = 0.055, 0.03, 0.03
PANEL_FONT, PANEL_SMALL_FONT = 7, 6.5
#: seaborn's default palette: the reference wears its first colour, the
#: prediction its second, a pooled measurement the first.
PALETTE = sns.color_palette()
LINE_COLOUR = "0.3"


def _panel(figure: plt.Figure, blocks: list, top: float = TOP) -> float:
    """A rounded panel in the right column hung from ``top`` holding
    ``blocks``, each a list of rows; a row is ``(label, value)``
    right-aligned, ``(label, ref, pred)``, or a one-word ``(header,)`` in
    small muted type that also carries the ``ref`` / ``pred`` column
    heads when its block has paired rows. Returns the panel's bottom
    edge."""
    scale = FIGURE_HEIGHT / figure.get_figheight()
    row_height, gap, pad = PANEL_ROW * scale, PANEL_GAP * scale, PANEL_PAD * scale
    rows = sum(len(block) for block in blocks)
    height = pad * 2 + row_height * rows + gap * (len(blocks) - 1)
    panel = figure.add_axes((COLUMN_LEFT, top - height, COLUMN_WIDTH, height))
    panel.set_axis_off()
    panel.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                   facecolor="white", edgecolor="0.8", linewidth=0.8,
                                   transform=panel.transAxes, clip_on=False))
    left, mid, right = 0.08, 0.68, 0.92
    step = row_height / height
    y = 1 - (pad + row_height / 2) / height
    for index, block in enumerate(blocks):
        if index:
            y -= gap / height
        paired = any(len(row) == 3 for row in block)
        for row in block:
            if len(row) == 1:
                header = [(left, row[0], "left")]
                if paired:
                    header += [(mid, "ref", "right"), (right, "pred", "right")]
                for x, word, align in header:
                    panel.text(x, y, word, fontsize=PANEL_SMALL_FONT, color="0.4",
                               va="center", ha=align)
            else:
                panel.text(left, y, row[0], fontsize=PANEL_FONT, va="center")
                for x, value in zip((mid, right)[-(len(row) - 1):], row[1:]):
                    panel.text(x, y, value, fontsize=PANEL_FONT, va="center", ha="right")
            y -= step
    return top - height


def _legend(figure: plt.Figure, left_column: list, right_column: list, below: float) -> None:
    """A framed legend of two columns in the right column, hung a gap under
    the edge ``below``. A legend fills column by column, so the shorter
    column is padded with blank entries to the longer's length."""
    blank = Line2D([], [], linestyle="", label="")
    length = max(len(left_column), len(right_column))
    handles = (left_column + [blank] * (length - len(left_column))
               + right_column + [blank] * (length - len(right_column)))
    gap = PANEL_GAP * FIGURE_HEIGHT / figure.get_figheight()
    figure.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.8,
                  handletextpad=0.5, handlelength=1.4, borderaxespad=0,
                  bbox_to_anchor=(COLUMN_LEFT, below - gap),
                  bbox_transform=figure.transFigure, frameon=True, fancybox=True,
                  fontsize=PANEL_FONT)


def _figure(grid: sns.FacetGrid, bottom: float = 0.16, hspace: float = 0.25) -> plt.Figure:
    """The grid's figure with its axes narrowed to leave the right column."""
    figure = grid.figure
    figure.subplots_adjust(top=TOP, bottom=bottom, left=0.10, right=AXES_RIGHT, hspace=hspace)
    return figure


# ---------------------------------------------------------------------------
# The best, median and worst waveform
# ---------------------------------------------------------------------------
RANKS = ("best", "median", "worst")
#: One row's height in inches; the rows share the single figure's width.
RANK_ROW_HEIGHT = 2.0
SERIES_COLOURS = {"reference": PALETTE[0], "prediction": PALETTE[1]}
SERIES_LINE_WIDTH = 1.1
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


def ranked_by_ccc(scored: pd.DataFrame) -> dict:
    """``{rank: row}``: the measurement of highest, median and lowest CCC."""
    scored = scored.dropna(subset=["ccc"]).sort_values("ccc")
    return {"best": scored.iloc[-1], "median": scored.iloc[len(scored) // 2],
            "worst": scored.iloc[0]}


def ranked_waveform_figure(traces: dict, rows: dict, parameters: dict, sig: str,
                           fs: float) -> plt.Figure:
    """The :data:`RANKS` measurements of one signal over time, one row each
    on a shared scale: the reference and, shifted to the lag its metrics
    were scored at (a positive lag delays it against the reference, so it
    moves earlier), the prediction; each side's mean dotted in its colour
    and, for a :data:`BEAT_SIGNALS` signal, the beats its systolic and
    diastolic were averaged over as up and down triangles. Beside each row
    a panel of that measurement's CCC, MAE, RMSE and lag and its
    :data:`PANEL_PARAMETERS` as reference / prediction pairs; under the
    last the legend, the sides left and the marks right. ``traces`` and
    ``rows`` map a rank to the measurement's frames and its scored row,
    ``parameters`` a parameter name to its scored frame."""
    unit = signal_unit(sig)
    sides_by_rank, long = {}, []
    for rank in RANKS:
        trace, row = traces[rank], rows[rank]
        t0 = trace["t"].min()
        lag = float(row["lag"]) if np.isfinite(row["lag"]) else 0.0
        sides = {"reference": (trace["t"].to_numpy() - t0, trace["label"].to_numpy()),
                 "prediction": (trace["t"].to_numpy() - t0 - lag, trace["mean"].to_numpy())}
        sides_by_rank[rank] = sides
        long += [pd.DataFrame({"t": t, "value": v, "series": s, "rank": rank})
                 for s, (t, v) in sides.items()]
    long = pd.concat(long, ignore_index=True)
    span = (0.0, float(long.loc[long["series"] == "reference", "t"].max()))
    long = long[long["t"].between(*span)]

    grid = sns.relplot(data=long, x="t", y="value", hue="series", kind="line",
                       row="rank", row_order=list(RANKS),
                       palette=SERIES_COLOURS, hue_order=list(SERIES_COLOURS),
                       linewidth=SERIES_LINE_WIDTH, height=RANK_ROW_HEIGHT,
                       aspect=FIGURE_HEIGHT * FIGURE_ASPECT / RANK_ROW_HEIGHT,
                       legend=False, facet_kws={"sharey": True})
    grid.set_titles("")
    figure = _figure(grid, bottom=0.10)
    figure.suptitle(f"{sig} ({unit})", x=0.10, ha="left", fontsize=10)
    below = TOP
    for rank, axis in zip(RANKS, grid.axes.flat):
        axis.set_xlim(*span)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(rank.capitalize())
        for side, (t, values) in sides_by_rank[rank].items():
            colour = SERIES_COLOURS[side]
            axis.axhline(values.mean(), color=colour, linestyle=":", linewidth=1.0)
            if sig in BEAT_SIGNALS:
                for sign, marker in ((+1, "^"), (-1, "v")):
                    beats = detect_beats(sign * values, fs, sig)
                    keep = (t[beats] >= span[0]) & (t[beats] <= span[1])
                    axis.scatter(t[beats][keep], values[beats][keep], marker=marker, s=16,
                                 color=colour, edgecolor="white", linewidth=0.5, zorder=3)
        row = rows[rank]
        lag = float(row["lag"]) if np.isfinite(row["lag"]) else 0.0
        stats = [("CCC", f"{row['ccc']:.3f}"), ("MAE", f"{row['mae']:.2f} {unit}"),
                 ("RMSE", f"{row['rmse']:.2f} {unit}"), ("lag", f"{lag:+.2f} s")]
        pressures = [(unit,)]
        for label, name, column in PANEL_PARAMETERS.get(sig, ()):
            value = _row_of(parameters[name], row)
            pressures.append((label, f"{value[f'ref_{column}']:.1f}",
                              f"{value[f'pred_{column}']:.1f}"))
        blocks = [stats, pressures] if len(pressures) > 1 else [stats]
        below = _panel(figure, blocks, top=axis.get_position().y1)

    sides_column = [Line2D([], [], color=c, linewidth=SERIES_LINE_WIDTH, label=s)
                    for s, c in SERIES_COLOURS.items()]
    marks_column = [Line2D([], [], color="0.4", linestyle=":", linewidth=1.0, label="mean")]
    if sig in BEAT_SIGNALS:
        marks_column += [Line2D([], [], marker="^", linestyle="", color="0.4", markersize=5,
                                label="systolic"),
                         Line2D([], [], marker="v", linestyle="", color="0.4", markersize=5,
                                label="diastolic")]
    _legend(figure, sides_column, marks_column, below)
    return figure


def waveform_figures(measurements: pd.DataFrame, frames: dict, parameters: dict,
                     fs: float) -> dict:
    """``{figure name: Figure}``: per signal, its best, median and worst
    measurement by CCC drawn by :func:`ranked_waveform_figure`."""
    figures = {}
    for sig, frame in frames.items():
        if frame["ccc"].notna().sum() < len(RANKS):
            continue
        rows = ranked_by_ccc(frame)
        traces = {}
        for rank, row in rows.items():
            where = (measurements["signal"] == sig)
            for name in MEASUREMENT_KEY:
                where &= measurements[name] == row[name]
            traces[rank] = measurements[where]
        figures[f"waveform_{sig}_best_median_worst"] = ranked_waveform_figure(
            traces, rows, parameters, sig, fs)
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
    below = _panel(figure, [rows])

    points_column = [Line2D([], [], marker="o", linestyle="", color=POINT_COLOUR,
                            markersize=4, label="measurement")]
    if fallback:
        points_column.append(Line2D([], [], marker="X", linestyle="", color=POINT_COLOUR,
                                    markersize=4, label="extreme sample"))
    lines_column = [Line2D([], [], color=LINE_COLOUR, linestyle="-", linewidth=0.9, label="mean"),
                    Line2D([], [], color=LINE_COLOUR, linestyle="--", linewidth=0.9,
                           label="95 % LoA")]
    _legend(figure, points_column, lines_column, below)
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


# ---------------------------------------------------------------------------
# The rainclouds: the waveform metrics per signal, the parameters' errors
# ---------------------------------------------------------------------------
#: The cloud's and box's widths in category units, and the rain's opacity.
CLOUD_WIDTH, BOX_WIDTH, RAIN_ALPHA = 0.9, 0.15, 0.15
#: The x range of one raincloud's axes, in category units about its box at
#: 0; a figure with fewer columns than ``REFERENCE_COLUMNS`` widens it so
#: every raincloud is the same size on paper.
X_RANGE, REFERENCE_COLUMNS = (-0.75, 0.35), 4
#: One colour per signal, in seaborn's default palette order; the rate's
#: is the third.
SIGNAL_COLOURS = {"CVP": PALETTE[0], "ABP": PALETTE[1], "rate": PALETTE[2]}
#: The waveform metrics drawn, with their labels.
SHOWN_METRICS = (("mae", "MAE"), ("rmse", "RMSE"), ("ccc", "CCC"))
#: The parameters whose errors are drawn, by ``PARAMETERS`` key, in order.
ERROR_PARAMETERS = ("CVP_mean", "ABP_mean", "ABP_systolic", "ABP_diastolic", "rate")


def raincloud(axis, values, label: str, columns: int, colour) -> None:
    """One column: the density as a half-violin to the left of the box
    position, every value as a faint jittered point under the box, and the
    box itself white-faced with a bold median and no outlier markers (the
    points show them). ``columns`` is how many share the figure."""
    data = pd.DataFrame({"x": label, "y": np.asarray(values, dtype=np.float64)})
    ptitprince.half_violinplot(x="x", y="y", data=data, ax=axis, color=colour,
                               width=CLOUD_WIDTH, cut=0, bw=0.2, linewidth=0.8,
                               inner=None, offset=BOX_WIDTH / 1.8 + 0.05)
    sns.stripplot(x="x", y="y", data=data, ax=axis, color=colour, size=1.5,
                  jitter=BOX_WIDTH / 2, alpha=RAIN_ALPHA, zorder=1)
    sns.boxplot(x="x", y="y", data=data, ax=axis, width=BOX_WIDTH, showfliers=False,
                boxprops={"facecolor": "white", "alpha": 0.9, "zorder": 10},
                whiskerprops={"linewidth": 1.5, "zorder": 10},
                capprops={"linewidth": 1.5, "zorder": 10},
                medianprops={"linewidth": 1.6, "color": "0.2", "zorder": 12})
    stretch = max(1.0, REFERENCE_COLUMNS / columns)
    centre, half = (X_RANGE[0] + X_RANGE[1]) / 2, (X_RANGE[1] - X_RANGE[0]) / 2 * stretch
    axis.set_xlim(centre - half, centre + half)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="y", labelsize=7)


def raincloud_figure(columns: list, title: str, blocks: list) -> plt.Figure:
    """A row of rainclouds, each on its own scale, with ``title`` over the
    axes and ``blocks`` in the panel. A column is ``(label, values, zero
    line?, y limits or None, colour)``."""
    labels = [column[0] for column in columns]
    grid = sns.FacetGrid(pd.DataFrame({"column": labels}), col="column", col_order=labels,
                         sharey=False, sharex=False, height=FIGURE_HEIGHT,
                         aspect=FIGURE_ASPECT / len(columns))
    grid.set_titles("")
    for (label, values, zero, limits, colour), axis in zip(columns, grid.axes.flat):
        raincloud(axis, values, label, len(columns), colour)
        if zero:
            axis.axhline(0, color="0.6", linestyle=":", linewidth=0.8)
        if limits is not None:
            axis.set_ylim(*limits)
    figure = _figure(grid, bottom=0.20)
    figure.subplots_adjust(top=0.88, wspace=0.7)
    figure.suptitle(title, x=0.10, y=0.97, ha="left", fontsize=10)
    _panel(figure, blocks)
    return figure


def waveform_metrics_figure(scored: pd.DataFrame, pooled: pd.DataFrame, sig: str) -> plt.Figure:
    """The :data:`SHOWN_METRICS` of one signal, a raincloud each; the panel
    holds n, subjects, and each metric's median with interquartile range
    and ICC from ``pooled``, the signal's rows of the waveform agreement
    table."""
    unit, colour = signal_unit(sig), SIGNAL_COLOURS[sig]
    columns = []
    for metric, label in SHOWN_METRICS:
        values = scored[metric].dropna()
        bounded = metric == "ccc"
        columns.append((label if bounded else f"{label} ({unit})", values, bounded,
                        (min(-0.05, float(values.min())), 1.0) if bounded else None, colour))
    rows = pooled.set_index("metric")
    first = rows.iloc[0]
    blocks = [[("n", f"{int(first['n'])}"), ("subjects", f"{int(first['k'])}")],
              [("median [IQR]",)] + [
                  (label, f"{rows.loc[m, 'median']:.2f} [{rows.loc[m, 'q1']:.2f}, "
                          f"{rows.loc[m, 'q3']:.2f}]") for m, label in SHOWN_METRICS],
              [("ICC",)] + [(label, f"{rows.loc[m, 'icc']:.2f}") for m, label in SHOWN_METRICS]]
    return raincloud_figure(columns, f"{sig} waveform metrics per measurement", blocks)


def waveform_metrics_figures(frames: dict, pooled: pd.DataFrame) -> dict:
    """``{figure name: Figure}``: one :func:`waveform_metrics_figure` per
    signal in ``frames``, from the waveform agreement table ``pooled``."""
    return {f"waveform_metrics_{sig}": waveform_metrics_figure(
                frame, pooled[pooled["signal"] == sig], sig)
            for sig, frame in frames.items() if frame["ccc"].notna().any()}


def parameter_errors_figure(frames: dict, pooled: pd.DataFrame) -> plt.Figure:
    """The :data:`ERROR_PARAMETERS` errors (prediction minus reference), a
    raincloud each in its signal's colour; the panel holds each one's mean
    error with corrected SD and its ICC from ``pooled``, the agreement
    table."""
    by_key = {p.key: p for p in PARAMETERS}
    rows = pooled.set_index("parameter")
    columns, error_block, icc_block = [], [("mean error ± s_corr",)], [("ICC",)]
    for key in ERROR_PARAMETERS:
        p = by_key[key]
        pairs = pairs_of(frames[p.name], p)
        colour = SIGNAL_COLOURS[p.signal or "rate"]
        columns.append((f"{p.name}\n({p.unit})", pairs["error"], True, None, colour))
        row = rows.loc[p.name]
        short = p.name.removeprefix("ABP ") if p.column in ("systolic", "diastolic") else p.name
        error_block.append((short, f"{row['mean_error']:+.1f} ± {row['s_corr']:.1f} {p.unit}"))
        icc_block.append((short, f"{row['icc']:.2f}"))
    return raincloud_figure(columns, "Parameter errors per measurement",
                            [error_block, icc_block])
