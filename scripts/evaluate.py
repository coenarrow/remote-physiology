"""Evaluate a finished run: pool its per-recording tables into one report.

    uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba

The run directory is one ``scripts/run.py`` wrote, or one ``main.py`` wrote
a fold per participant into. Its records are already scored
(``docs/evaluation.md``); nothing is trained, inferred or re-scored here,
and nothing but the run directory is read. One recording and camera is one
paired measurement, eligible when its covered stretch is within
``--segment-min`` and ``--segment-max`` seconds. The last epoch of each fold
is evaluated, or the one ``--epoch`` names; it is never chosen by test error.

Written to ``RUN_DIR/evaluation/``, replacing what was there::

    measurements.csv   one row per fold, recording, camera and signal
    exclusions.csv     what was left out, and why
    tables/*.csv       every table of the report
    figures/*.png      every figure
    report.md          the report
    report.html        the same, as one file, A4 when printed
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.demographics import with_core                  # noqa: E402
from src.evaluation.pooling import (                               # noqa: E402
    additional_params, collect, collect_epochs, exclusions, mark_eligible,
    parameters, read_losses,
)
from src.evaluation.report import MEASUREMENT_KEY, build           # noqa: E402
from src.outputs import FLOAT_FORMAT                               # noqa: E402

OUT_DIR = "evaluation"
MEASUREMENTS_NAME, EXCLUSIONS_NAME = "measurements.csv", "exclusions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pool a finished run's per-recording tables into one report.")
    parser.add_argument(
        "run_dir", metavar="RUN_DIR",
        help="a run directory written by scripts/run.py, or the directory "
             "main.py wrote its folds into")
    parser.add_argument(
        "--epoch", type=int, metavar="N",
        help="which epoch of every fold to evaluate (default: each fold's last)")
    parser.add_argument(
        "--segment-min", type=float, default=20.0, metavar="SECONDS",
        help="shortest covered stretch that counts as a measurement (default: 20)")
    parser.add_argument(
        "--segment-max", type=float, default=30.0, metavar="SECONDS",
        help="longest covered stretch that counts as a measurement (default: 30)")
    parser.add_argument(
        "--additional-params", nargs="+", default=[], metavar="ATTR",
        help="root attrs to report beside the core five and the datasets' "
             "ADDITIONAL_PARAMS, as recordings.csv names them (dotted when nested)")
    return parser


def main(argv=None) -> Path:
    """Write the evaluation; returns ``report.md``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.segment_min > args.segment_max:
        parser.error("--segment-min is above --segment-max")
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")
    bounds = (args.segment_min, args.segment_max)
    try:
        pooled, meta = collect(run_dir, args.epoch)
        pooled = mark_eligible(with_core(pooled), *bounds)
        curves = mark_eligible(collect_epochs(run_dir), *bounds)
        losses = read_losses(run_dir)
        additional = list(dict.fromkeys(
            [*additional_params(run_dir), *args.additional_params]))
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))

    out_dir = run_dir / OUT_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    excluded = exclusions(pooled)
    excluded.to_csv(out_dir / EXCLUSIONS_NAME, index=False, float_format=FLOAT_FORMAT)
    measurements = pooled[pooled["eligible"]].drop(columns=["eligible", "reason"])
    if measurements.empty:
        parser.error(f"no recording has a covered stretch within {bounds[0]:g} to "
                     f"{bounds[1]:g} s; {out_dir / EXCLUSIONS_NAME} lists them")
    measurements.to_csv(out_dir / MEASUREMENTS_NAME, index=False,
                        float_format=FLOAT_FORMAT)

    params = parameters(meta["traces"], measurements)
    path = build(out_dir, run_dir, pooled, measurements, excluded, params, meta,
                 curves, losses, additional, bounds)
    print(f"{len(measurements.drop_duplicates(MEASUREMENT_KEY))} measurement(s) of "
          f"{measurements['subject'].nunique()} subject(s), {len(excluded)} excluded")
    print(f"report: {path}")
    return path


if __name__ == "__main__":
    main()
