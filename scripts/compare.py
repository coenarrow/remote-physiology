"""Compare the evaluations of several runs in one table set.

    uv run scripts/compare.py Z:/runs/benchmark/comparison_5s \\
        FactorizePhys=Z:/runs/benchmark/neckflix_hpc_factorizephys_H200W200/evaluation_5s \\
        PhysNet=Z:/runs/benchmark/neckflix_hpc_physnet_H200W200/evaluation_5s ...

Each ``NAME=DIR`` is a directory ``scripts/evaluate.py`` wrote. Nothing is
re-scored: the pooled tables are stacked with a ``model`` column, and one
row is derived per model, ``heart rate (arterial line)``: the heart-rate
measurements of the participants an ABP measurement exists for, pooled
with the same repeated-measures statistics. On Neckflix the heart-rate
reference of a participant without an arterial line comes from the CVP
trace alone, whose largest spectral bin is often a harmonic, so this row
is the one to trust.

Written to ``OUT_DIR``, replacing what was there::

    tables/agreement.csv            every run's derived-parameter rows, stacked
    tables/waveform_agreement.csv   every run's waveform-metric rows, stacked
    tables/summary.csv              the thesis columns per model and parameter
    figures/<PARAMETER>.png         mean error and 95 % limits per model
    figures/tracking.png            between- and within-subject slope per model
    figures/shared_<SIGNAL>.png     one measurement, every model's prediction of it
    figures/shared_waveforms.png    the same for every signal, one column each, A4

The shared-measurement figure takes the measurement of each signal with the
highest mean concordance across the models (among those every model
scored), reads that stretch of each run's trace table at the epoch the
evaluation used, and draws the reference with each model's prediction in
physical units, so the shape and the level can be judged apart.
``--measurement-duration`` and ``--measurement-tolerance`` must be what the
evaluations were cut with.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.colors                                           # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402

from src.evaluation.agreement import limits, repeated_measures, tracking  # noqa: E402
from src.evaluation.measurements import PARAMETERS                 # noqa: E402
from src.evaluation.pooling import (                               # noqa: E402
    MEASUREMENT_KEY, WAVEFORM_COLUMNS, epoch_dir, find_folds, segment,
)
from src.outputs import FLOAT_FORMAT, META_NAME, RECORDS_DIR       # noqa: E402
from src.signal_transforms import signal_unit                      # noqa: E402

TABLES_DIR, FIGURES_DIR, MEASUREMENTS_DIR = "tables", "figures", "measurements"
ARTERIAL_RATE = "heart rate (arterial line)"
RATE = next(p for p in PARAMETERS if p.signal is None)
ABP_MEAN = next(p for p in PARAMETERS if p.name == "ABP mean")
UNITS = {p.name: p.unit for p in PARAMETERS} | {ARTERIAL_RATE: RATE.unit}
SUMMARY_COLUMNS = ["model", "parameter", "unit", "n", "k", "mean_error", "s_corr",
                   "lower_limit", "upper_limit", "icc", "n_ind", "ref_sd", "pred_sd",
                   "slope_between", "slope_within", "r_within"]
FIGURE_DPI = 150
#: Figure width and height in inches: the text area of A4 portrait.
REPORT_WIDTH, A4_HEIGHT = 6.3, 9.3
MARK_COLOUR, ZERO_COLOUR, GRID_COLOUR = "#2a78d6", "0.35", "0.88"
REF_COLOUR = "0.3"
#: Diverging scale of the tracking grids: orange below zero, blue above.
SLOPE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "slope", ["#eb6834", "#f3f2ef", "#2a78d6"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stack several runs' evaluations into one comparison.")
    parser.add_argument("out_dir", metavar="OUT_DIR", help="where the comparison is written")
    parser.add_argument("runs", metavar="NAME=DIR", nargs="+",
                        help="a model's name and the evaluation directory of its run")
    parser.add_argument(
        "--measurement-duration", type=float, default=10.0, metavar="SECONDS",
        help="the measurement length the evaluations were cut with (default: 10)")
    parser.add_argument(
        "--measurement-tolerance", type=float, default=1.0, metavar="SECONDS",
        help="the tolerance the evaluations were cut with (default: 1)")
    return parser


def parse_runs(items: list) -> dict:
    runs = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise ValueError(f"{item!r} is not NAME=DIR")
        folder = Path(path)
        if not (folder / TABLES_DIR / "agreement.csv").is_file():
            raise ValueError(f"{folder} holds no tables/agreement.csv")
        runs[name] = folder
    return runs


def arterial_rate_row(folder: Path) -> dict:
    """The heart-rate measurements of the participants with an ABP
    measurement, pooled like every other parameter."""
    rate = pd.read_csv(folder / MEASUREMENTS_DIR / f"{RATE.key}.csv",
                       dtype={"participant": str})
    abp = pd.read_csv(folder / MEASUREMENTS_DIR / f"{ABP_MEAN.key}.csv",
                      dtype={"participant": str})
    arterial = abp.dropna(subset=[f"err_{ABP_MEAN.column}"])["participant"].unique()
    rows = rate[rate["participant"].isin(arterial)].dropna(subset=[f"err_{RATE.column}"])
    return {"parameter": ARTERIAL_RATE,
            **repeated_measures(rows[f"err_{RATE.column}"], rows["participant"]),
            **tracking(rows[f"ref_{RATE.column}"], rows[f"pred_{RATE.column}"],
                       rows["participant"])}


def stacked(runs: dict, table: str, derive=None) -> pd.DataFrame:
    frames = []
    for name, folder in runs.items():
        frame = pd.read_csv(folder / TABLES_DIR / f"{table}.csv")
        if derive is not None:
            frame = pd.concat([frame, pd.DataFrame([derive(folder)])], ignore_index=True)
        frame.insert(0, "model", name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summary(agreement: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in agreement.iterrows():
        lower, upper = limits(row)
        rows.append({**row, "unit": UNITS[row["parameter"]],
                     "lower_limit": lower, "upper_limit": upper})
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _axes(height: float, columns: int = 1):
    figure, axes = plt.subplots(1, columns, figsize=(REPORT_WIDTH, height), squeeze=False)
    for axis in axes[0]:
        axis.grid(True, axis="y", color=GRID_COLOUR, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(labelsize=8)
    return figure, axes[0]


def limits_figure(rows: pd.DataFrame, parameter: str) -> plt.Figure:
    """Each model's mean error as a dot with its 95 % limits of agreement
    as a bar, the models along the x axis, zero error marked."""
    figure, (axis,) = _axes(3.2)
    x = np.arange(len(rows))
    axis.axhline(0, color=ZERO_COLOUR, linewidth=0.8, linestyle="--")
    axis.vlines(x, rows["lower_limit"], rows["upper_limit"], color=MARK_COLOUR,
                linewidth=2, capstyle="round")
    axis.scatter(x, rows["mean_error"], s=36, color=MARK_COLOUR, edgecolor="white",
                 linewidth=0.8, zorder=3)
    for xi, (_, row) in zip(x, rows.iterrows()):
        axis.annotate(f"{row['mean_error']:+.1f}", (xi, row["mean_error"]),
                      xytext=(6, 0), textcoords="offset points", fontsize=7,
                      va="center", color="0.2")
    axis.set_xticks(x, rows["model"], fontsize=8)
    axis.set_xlim(-0.6, len(rows) - 0.4)
    unit = rows["unit"].iloc[0]
    axis.set_ylabel(f"prediction − reference ({unit})", fontsize=8)
    n, k = int(rows["n"].min()), int(rows["k"].min())
    axis.set_title(f"{parameter}: mean error and 95 % limits of agreement\n"
                   f"{n} measurements of {k} subjects per model", fontsize=9)
    figure.tight_layout()
    return figure


def tracking_figure(table: pd.DataFrame, parameters: list) -> plt.Figure:
    """Two grids, parameters down and models across: the between-subject
    and the within-subject slope of prediction on reference, printed in
    each cell and shaded on a diverging scale about zero, where 1 tracks
    the reference level and 0 is a constant predictor."""
    models = list(dict.fromkeys(table["model"]))
    figure, axes = plt.subplots(1, 2, figsize=(REPORT_WIDTH, 0.32 * len(parameters) + 1.2),
                                sharey=True)
    for axis, column, title in zip(axes, ("slope_between", "slope_within"),
                                   ("between subjects", "within subjects")):
        grid = (table.pivot(index="parameter", columns="model", values=column)
                .reindex(index=parameters, columns=models))
        axis.imshow(grid.to_numpy(dtype=float), cmap=SLOPE_CMAP, vmin=-1, vmax=1,
                    aspect="auto")
        for i in range(len(parameters)):
            for j in range(len(models)):
                value = grid.iat[i, j]
                axis.text(j, i, f"{value:+.2f}" if np.isfinite(value) else "–",
                          ha="center", va="center", fontsize=6.5, color="0.1")
        axis.set_xticks(range(len(models)), models, fontsize=6.5, rotation=30, ha="right")
        axis.set_yticks(range(len(parameters)), parameters, fontsize=7)
        axis.set_title(title, fontsize=8)
        axis.tick_params(length=0)
        for side in axis.spines.values():
            side.set_visible(False)
    figure.suptitle("Level tracking: slope of prediction on reference "
                    "(1 follows the reference, 0 is a constant predictor)", fontsize=8.5)
    figure.tight_layout()
    return figure


# ---------------------------------------------------------------------------
# One measurement, every model's prediction of it
# ---------------------------------------------------------------------------
def shared_measurement(runs: dict, signal: str) -> tuple:
    """``(key, scores)``: the measurement of ``signal`` every run scored
    whose concordance averaged over the runs is highest, and each run's
    ``ccc`` and ``mae`` on it; ``(None, {})`` when no run scored it."""
    frames = {}
    for name, folder in runs.items():
        path = folder / MEASUREMENTS_DIR / f"waveform_{signal}.csv"
        if not path.is_file():
            return None, {}
        frames[name] = (pd.read_csv(path, dtype={"participant": str, "recording": str})
                        .set_index(MEASUREMENT_KEY)[["ccc", "mae"]])
    shared = pd.concat(frames, axis=1, join="inner").dropna()
    if shared.empty:
        return None, {}
    key = shared.xs("ccc", axis=1, level=1).mean(axis=1).idxmax()
    return (dict(zip(MEASUREMENT_KEY, key)),
            {name: shared.loc[key, name].to_dict() for name in runs})


def measurement_trace(run_dir: Path, key: dict, signal: str, duration: float,
                      tolerance: float) -> pd.DataFrame | None:
    """The frames of measurement ``key`` of ``signal``, read from the fold
    of ``run_dir`` that holds its recording at that fold's last epoch, cut
    as the evaluation cut it."""
    for fold in find_folds(run_dir):
        _, folder = epoch_dir(fold, None)
        for path in (folder / RECORDS_DIR).glob(
                f"*/{key['recording']}/{key['perspective']}/{signal}.csv"):
            meta = json.loads((path.parents[2] / META_NAME).read_text(encoding="utf-8"))
            table = pd.read_csv(path, usecols=WAVEFORM_COLUMNS)
            for name in ("participant", "recording", "perspective"):
                table[name] = key[name]
            table["signal"] = signal
            cut = segment(table, float(meta["fs"]), duration, tolerance)
            return cut[cut["segment"] == key["segment"]]
    return None


def shared_traces(runs: dict, signal: str, duration: float, tolerance: float) -> dict:
    """``{model: frames}`` of the shared measurement of ``signal``, one
    entry per run that holds it; empty when no run scored the signal."""
    key, _ = shared_measurement(runs, signal)
    if key is None:
        return {}
    traces = {name: measurement_trace(folder.parent, key, signal, duration, tolerance)
              for name, folder in runs.items()}
    return {name: t for name, t in traces.items() if t is not None and not t.empty}


def _waveform_panel(axis, trace: pd.DataFrame, title: str) -> None:
    """The reference in grey and the prediction in colour against time
    from the measurement's start, titled at the top left."""
    t = trace["t"] - trace["t"].iloc[0]
    axis.plot(t, trace["label"], color=REF_COLOUR, linewidth=1.0, label="reference")
    axis.plot(t, trace["mean"], color=MARK_COLOUR, linewidth=1.2, label="prediction")
    axis.set_title(title, fontsize=8, loc="left", pad=2)
    axis.tick_params(labelsize=7)
    axis.grid(True, axis="y", color=GRID_COLOUR, linewidth=0.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def shared_waveform_figure(runs: dict, signal: str, duration: float,
                           tolerance: float) -> plt.Figure | None:
    """One panel per model, titled with its name: the reference of the
    shared measurement in grey and the model's prediction in colour, in
    physical units on one shared y scale. The measurement's identity and
    the scores are in the tables, not on the figure."""
    traces = shared_traces(runs, signal, duration, tolerance)
    if not traces:
        return None
    figure, axes = plt.subplots(len(traces), 1,
                                figsize=(REPORT_WIDTH, 1.4 * len(traces) + 0.8),
                                sharex=True, sharey=True, squeeze=False)
    for axis, (name, trace) in zip(axes[:, 0], traces.items()):
        _waveform_panel(axis, trace, name)
    axes[0, 0].legend(fontsize=6.5, loc="upper left", frameon=False, ncol=2)
    axes[-1, 0].set_xlabel("time (s)", fontsize=8)
    figure.supylabel(f"{signal} ({signal_unit(signal)})", fontsize=8)
    figure.suptitle(f"{signal}: example waveform prediction for each model", fontsize=9)
    figure.tight_layout()
    return figure


def shared_waveforms_figure(runs: dict, signals: list, duration: float,
                            tolerance: float) -> plt.Figure | None:
    """Every signal's shared measurement side by side: one row per model,
    one column per signal, each column on its own y scale, sized to the
    text area of an A4 page."""
    columns = {sig: shared_traces(runs, sig, duration, tolerance) for sig in signals}
    columns = {sig: t for sig, t in columns.items() if t}
    if not columns:
        return None
    models = [name for name in runs if all(name in t for t in columns.values())]
    if not models:
        return None
    figure, axes = plt.subplots(len(models), len(columns),
                                figsize=(REPORT_WIDTH, A4_HEIGHT),
                                sharex=True, sharey="col", squeeze=False)
    for j, (sig, traces) in enumerate(columns.items()):
        for i, name in enumerate(models):
            _waveform_panel(axes[i, j], traces[name], name if j == 0 else "")
        axes[-1, j].set_xlabel("time (s)", fontsize=8)
    figure.suptitle("Example waveform prediction for each model", fontsize=9)
    figure.tight_layout(h_pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.965))
    for j, sig in enumerate(columns):          # column headers above the top row
        box = axes[0, j].get_position()
        figure.text((box.x0 + box.x1) / 2, box.y1 + 0.022, f"{sig} ({signal_unit(sig)})",
                    ha="center", va="bottom", fontsize=8.5)
    # One legend for the whole grid, centred on the row of the time labels.
    figure.canvas.draw()
    label = axes[-1, 0].xaxis.label.get_window_extent()
    y = figure.transFigure.inverted().transform((0, (label.y0 + label.y1) / 2))[1]
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, y), ncol=2,
                  fontsize=7, frameon=False, columnspacing=1.5)
    return figure


def main(argv=None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        runs = parse_runs(args.runs)
    except ValueError as err:
        parser.error(str(err))
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for name in (TABLES_DIR, FIGURES_DIR):
        (out_dir / name).mkdir(parents=True)

    def write(name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(out_dir / TABLES_DIR / f"{name}.csv", index=False,
                     float_format=FLOAT_FORMAT)

    agreement = stacked(runs, "agreement", arterial_rate_row)
    write("agreement", agreement)
    write("waveform_agreement", stacked(runs, "waveform_agreement"))
    table = summary(agreement)
    write("summary", table)

    parameters = list(dict.fromkeys(table["parameter"]))
    for parameter in parameters:
        figure = limits_figure(table[table["parameter"] == parameter], parameter)
        figure.savefig(out_dir / FIGURES_DIR / f"{parameter.replace(' ', '_')}.png",
                       dpi=FIGURE_DPI)
        plt.close(figure)
    figure = tracking_figure(table, parameters)
    figure.savefig(out_dir / FIGURES_DIR / "tracking.png", dpi=FIGURE_DPI)
    plt.close(figure)
    signals = sorted({p.signal for p in PARAMETERS if p.signal})
    for signal in signals:
        figure = shared_waveform_figure(runs, signal, args.measurement_duration,
                                        args.measurement_tolerance)
        if figure is not None:
            figure.savefig(out_dir / FIGURES_DIR / f"shared_{signal}.png", dpi=FIGURE_DPI)
            plt.close(figure)
    figure = shared_waveforms_figure(runs, signals, args.measurement_duration,
                                     args.measurement_tolerance)
    if figure is not None:
        figure.savefig(out_dir / FIGURES_DIR / "shared_waveforms.png", dpi=FIGURE_DPI)
        plt.close(figure)
    print(f"written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
