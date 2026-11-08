"""Draw the ROC curves of a finished evaluation.

    uv run scripts/roc.py runs/neckflix_hpc_factorizephys_H200W200

Nothing is scored here: ``scripts/evaluate.py`` wrote the confusion counts
at every candidate threshold (``tables/roc.csv``) and the clinical
thresholds' results (``tables/classification.csv``); this reads them and
writes one figure per signal, ``figures/roc_<SIGNAL>.png``, beside the
evaluation's other figures. ``--out-dir`` names the evaluation folder
under the run directory when it is not the default.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt                                    # noqa: E402
import pandas as pd                                                # noqa: E402

from scripts.evaluate import FIGURE_DPI, FIGURES_DIR, OUT_DIR, TABLES_DIR  # noqa: E402
from src.evaluation import progress                               # noqa: E402
from src.evaluation.plots import roc_figures                      # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw the ROC curves of a finished evaluation.")
    parser.add_argument(
        "run_dir", metavar="RUN_DIR",
        help="the run directory scripts/evaluate.py evaluated")
    parser.add_argument(
        "--out-dir", default=OUT_DIR, metavar="NAME",
        help=f"the evaluation folder under RUN_DIR (default: {OUT_DIR})")
    return parser


def main(argv=None) -> Path:
    """Write the figures; returns the figures directory."""
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluation = Path(args.run_dir) / args.out_dir
    tables = evaluation / TABLES_DIR
    for name in ("roc", "classification"):
        if not (tables / f"{name}.csv").is_file():
            parser.error(f"{tables / f'{name}.csv'} is missing: run scripts/evaluate.py first")
    sweeps = pd.read_csv(tables / "roc.csv")
    classes = pd.read_csv(tables / "classification.csv")
    figures_dir = evaluation / FIGURES_DIR
    figures_dir.mkdir(exist_ok=True)
    for name, figure in progress(roc_figures(sweeps, classes).items(), "writing figures"):
        figure.savefig(figures_dir / f"{name}.png", dpi=FIGURE_DPI)
        plt.close(figure)
    print(f"written to {figures_dir}")
    return figures_dir


if __name__ == "__main__":
    main()
