"""Evaluate a finished run from its waveforms.

    uv run scripts/evaluate.py runs/neckflix_hpc_factorizephys_H200W200

The run directory is one ``scripts/run.py`` wrote, or one ``main.py`` wrote
a fold per participant into. Nothing is trained or inferred here; the
per-trace tables each fold wrote (label and mean prediction per frame) are
read and everything is scored from them. The last epoch of each fold is
evaluated, or the one ``--epoch`` names; it is never chosen by test error.

Each trace of each recording and camera is cut into paired measurements of
``--measurement-duration`` seconds (a trailing one kept while within
``--measurement-tolerance`` of that). Per measurement the waveform metrics
(lag-aware MAE and RMSE, Lin's concordance at its best lag and its two
factors, Pearson's r and the bias correction C_b) and the derived
parameters (CVP mean; ABP mean, systolic, diastolic; the heart rate of the
spectrally fused cardiac traces) are scored, then pooled over all subjects
with the ISO 81060-3 repeated-measures statistics: mean error, corrected SD,
ICC and the number of independent measurements. The pooled waveform table
also carries each metric's median and interquartile range.

Written to ``RUN_DIR/evaluation/``, replacing what was there::

    measurements/waveform_<SIGNAL>.csv   one row per measurement and signal
    measurements/<PARAMETER>.csv         one row per measurement and parameter
    tables/demographics_<SIGNAL>.csv     who each signal was measured on
    tables/waveform_agreement.csv        the waveform metrics pooled, with median/IQR
    tables/agreement.csv                 the derived parameters pooled
    figures/*.png                        best waveforms, Bland-Altman plots

Every measurement row ends with its recording's sex, age, skin tone and
posture, for comparisons across them. Each stage reports a progress bar;
the slow ones are the waveform metrics (a concordance per lag per
measurement) and the beat detection behind systolic and diastolic.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt                                    # noqa: E402
import pandas as pd                                                # noqa: E402

from src.evaluation import progress                               # noqa: E402
from src.evaluation.agreement import agreement, waveform_agreement  # noqa: E402
from src.evaluation.demographics import demographics, with_markers  # noqa: E402
from src.evaluation.measurements import (                          # noqa: E402
    PARAMETERS, score_measurements, waveform_measurements,
)
from src.evaluation.plots import (                                 # noqa: E402
    best_waveform_figures, bland_altman_figures,
)
from src.evaluation.pooling import (                               # noqa: E402
    MEASUREMENT_KEY, collect_waveforms, segment,
)
from src.outputs import FLOAT_FORMAT                               # noqa: E402

OUT_DIR = "evaluation"
MEASUREMENTS_DIR, TABLES_DIR, FIGURES_DIR = "measurements", "tables", "figures"
FIGURE_DPI = 150


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a finished run's waveforms as paired measurements.")
    parser.add_argument(
        "run_dir", metavar="RUN_DIR",
        help="a run directory written by scripts/run.py, or the directory "
             "main.py wrote its folds into")
    parser.add_argument(
        "--epoch", type=int, metavar="N",
        help="which epoch of every fold to evaluate (default: each fold's last)")
    parser.add_argument(
        "--measurement-duration", type=float, default=10.0, metavar="SECONDS",
        help="nominal length of one paired measurement (default: 10)")
    parser.add_argument(
        "--measurement-tolerance", type=float, default=1.0, metavar="SECONDS",
        help="how far from the nominal length a measurement may be (default: 1)")
    return parser


def main(argv=None) -> Path:
    """Write the evaluation; returns its directory."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")
    try:
        waveforms, attrs, meta = collect_waveforms(run_dir, args.epoch)
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))
    fs = float(meta["fs"])
    measurements = segment(waveforms, fs, args.measurement_duration,
                           args.measurement_tolerance)
    if measurements.empty:
        parser.error(f"no trace has a paired stretch of {args.measurement_duration:g} "
                     f"± {args.measurement_tolerance:g} s")
    n = len(measurements.drop_duplicates(MEASUREMENT_KEY))
    print(f"{n} measurement(s) of {args.measurement_duration:g} ± "
          f"{args.measurement_tolerance:g} s over "
          f"{measurements['participant'].nunique()} participant(s)")

    out_dir = run_dir / OUT_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for name in (MEASUREMENTS_DIR, TABLES_DIR, FIGURES_DIR):
        (out_dir / name).mkdir(parents=True)

    def write(folder: str, name: str, frame: pd.DataFrame) -> None:
        frame.to_csv(out_dir / folder / f"{name}.csv", index=False,
                     float_format=FLOAT_FORMAT)

    for name, table in demographics(measurements, attrs).items():
        write(TABLES_DIR, f"demographics_{name}", table)

    waveform_frames = {sig: with_markers(frame, attrs) for sig, frame
                       in waveform_measurements(measurements, fs).items()}
    for sig, frame in waveform_frames.items():
        write(MEASUREMENTS_DIR, f"waveform_{sig}", frame)
    write(TABLES_DIR, "waveform_agreement", waveform_agreement(waveform_frames))

    parameter_frames = {name: with_markers(frame, attrs) for name, frame
                        in score_measurements(measurements, fs).items()}
    for p in PARAMETERS:
        write(MEASUREMENTS_DIR, p.key, parameter_frames[p.name])
    write(TABLES_DIR, "agreement", agreement(parameter_frames))

    figures = {**best_waveform_figures(measurements, waveform_frames),
               **bland_altman_figures(parameter_frames)}
    for name, figure in progress(figures.items(), "writing figures"):
        figure.savefig(out_dir / FIGURES_DIR / f"{name}.png", dpi=FIGURE_DPI)
        plt.close(figure)

    print(f"written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
