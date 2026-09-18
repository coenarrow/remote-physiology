"""Train one model, run it over the held-out participant and score the records.

    uv run python scripts/run.py --datasets pure \\
        --test-participant-dataset pure --test-participant-id 01 \\
        --config configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml \\
        --epochs 30 --batch-size 4 --num-workers 8

One command is one fold. It fits the model by the recipe, for the epochs,
batch size and workers the flags say (``--no-gpu`` keeps it off the GPU;
otherwise a GPU is used when there is one), and writes the run
directory — ``runs/<MODEL>_<DATASET>.<held-out participant or all>-..._<YYYYMMDDHHMM>``,
e.g. ``runs/PHYSMAMBA_PURE.all-NECKFLIX.24_202609091000``, or
``MODEL_FILE_NAME`` if the recipe names it — holding ``config.yaml``
(everything the run ran on), ``losses.csv`` and ``model.pt``, the latest
epoch's weights. With a participant held out, after every epoch it runs
that epoch's model over every strided window of that participant and
writes ``RUN_DIR/epoch_NN/``: that epoch's ``model.pt`` and its
``test_records/`` (``meta.json``, ``windows.csv``, and per recording and
camera one ``<TRACE>.csv`` with the time axis, the label, the mean and
spread of the overlapping window predictions and one column per window,
all in physical units; ``src/outputs.py``), then scores every
``<recording>/<perspective>/`` folder from those files alone
(``src/evaluation/recording.py``), writing one ``<TRACE>_beats.csv`` per
cardiac trace, ``signals.csv``, ``rates.csv`` and one ``<TRACE>.png`` per
trace beside its trace tables; ``docs/evaluation.md`` lists every column.
A killed run keeps every finished epoch. With nobody held out it trains on
every admitted store and stops there.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.dataset_config import load_dataset_configs              # noqa: E402
from src.datasets import load_stores                             # noqa: E402
from src.distributed import init_runtime, shutdown               # noqa: E402
from src.evaluation.recording import score_recording             # noqa: E402
from src.experiment import (                                     # noqa: E402
    add_config_arguments, add_limit_argument, add_run_arguments,
    add_split_arguments, check_split_arguments, compile_config, limit_windows,
    print_model, print_runtime, print_setup, print_split, print_stores,
    run_name, run_settings, split_stores, test_windows, train_windows,
)
from src.model_config import load_config                         # noqa: E402
from src.models import build_model                               # noqa: E402
from src.outputs import META_NAME, RECORDS_DIR, write_records    # noqa: E402
from src.trainer import CHECKPOINT_NAME, DEFAULT_RUNS_DIR, Trainer  # noqa: E402

SCRIPT = "scripts/run.py"
#: One directory per epoch under the run directory, 1-based like ``losses.csv``.
EPOCH_DIR = "epoch_{:02d}"


def build_parser() -> argparse.ArgumentParser:
    parser = add_config_arguments(argparse.ArgumentParser(
        description="Train one model holding one participant out or none, run "
                    "it over the held-out participant and score the records."))
    add_run_arguments(parser)
    add_split_arguments(parser)
    parser.add_argument(
        "--runs-dir", metavar="PATH", default=DEFAULT_RUNS_DIR,
        help="where run directories land (default: runs/)")
    parser.add_argument(
        "--run-dir", metavar="PATH",
        help="the exact run directory to write, for a launcher that names "
             "the run itself (default: --runs-dir/<derived name>)")
    add_limit_argument(parser)
    return parser


def score_records(records_dir: Path) -> list:
    """Score every ``<recording>/<perspective>/`` folder under the records
    directory from its files alone, found by the first trace's table;
    returns the folders scored, in order."""
    meta = json.loads((records_dir / META_NAME).read_text(encoding="utf-8"))
    first = str(meta["traces"][0])
    folders = sorted(path.parent for path in records_dir.glob(f"*/*/{first}.csv"))
    for folder in folders:
        score_recording(folder, meta)
    return folders


def main(argv=None) -> Path:
    """Fit, record and score the run; returns its directory."""
    parser = build_parser()
    args = parser.parse_args(argv)
    check_split_arguments(parser, args)
    run = run_settings(parser, args)

    try:
        interface, model_config, training = load_config(args.config)
        print_setup(interface, model_config, training, run)
        # Name the run now, before the process group exists: the name
        # carries the start minute, and every rank stamps it here, within
        # milliseconds of the launch, so they all land in one directory.
        run_dir = (Path(args.run_dir) if args.run_dir
                   else Path(args.runs_dir) / run_name(args, model_config, training))
        # Resolve device / precision / process group against this machine
        # first: under a launch that cannot run distributed, only rank 0
        # continues past this line.
        runtime = init_runtime(training, run.gpu)
        print_runtime(runtime, run)
        configs = load_dataset_configs(args.datasets)
        stores = load_stores(configs)
        print_stores(configs, stores)
        split = split_stores(stores, args.test_participant_dataset,
                             args.test_participant_id)
        print_split(split)
        train_dataset = limit_windows(train_windows(split, interface), args.limit_windows)
        test_dataset = (limit_windows(test_windows(split, interface), args.limit_windows)
                        if split.test else None)
    except ValueError as err:          # ConfigError is a ValueError
        parser.error(str(err))

    # The model: one copy of the architecture per trace, widths from the
    # interface, dict in and dict out. The loss is the trainer's, not its.
    model = build_model(model_config, interface)
    print_model(model_config, model)

    print(f"run: {run_dir}")
    config = compile_config(SCRIPT, argv, args, interface, model_config, training,
                            run, runtime, configs, split, run_dir)
    try:
        trainer = Trainer(model, interface, model_config, training, run, runtime, run_dir, config)
    except ConfigError as err:
        parser.error(str(err))
    meta = {"dataset": args.test_participant_dataset,
            "participant": args.test_participant_id, "run_dir": str(run_dir),
            "command": config["command"], "git": config["git"]}

    def after_epoch(epoch: int) -> None:
        """That epoch's model over the held-out participant: every rank runs
        its shard, the main rank gets every record and alone keeps the
        weights, writes the records and scores them under ``epoch_NN/``."""
        records = trainer.test(test_dataset)
        if not runtime.is_main:
            return
        epoch_dir = run_dir / EPOCH_DIR.format(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.checkpoint(), epoch_dir / CHECKPOINT_NAME)
        out_dir = epoch_dir / RECORDS_DIR
        write_records(records, out_dir, interface, {**meta, "epoch": epoch})
        folders = score_records(out_dir)
        print(f"epoch {epoch}: {len(records)} windows written to {out_dir}, "
              f"{len(folders)} recording(s) scored")

    try:
        trainer.fit(train_dataset, after_epoch if test_dataset is not None else None)
        return run_dir
    finally:
        shutdown(runtime)


if __name__ == "__main__":
    main()
