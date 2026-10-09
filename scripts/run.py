"""Train one model, run it over the held-out participant and record it.

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
``test_records/<dataset>/`` (``meta.json``, ``windows.csv``, and per recording and
camera one ``<TRACE>.csv`` with the time axis, the label, the mean and
spread of the overlapping window predictions and one column per window,
all in physical units; ``src/outputs.py``), then draws every
``<recording>/<perspective>/`` folder from those files alone
(``src/evaluation/recording.py``): one ``<TRACE>.png`` per trace beside its
trace table, the sanity picture of the fold. Nothing is scored until
``scripts/evaluate.py`` runs over the finished run.
A killed run keeps every finished epoch. With nobody held out it trains on
every admitted store and ``RUN_DIR/epoch_NN/`` holds that epoch's
``model.pt`` alone: every run keeps every epoch's weights.
``scripts/test.py`` runs a finished epoch over any other dataset.

``--init-from`` starts from a trained model instead of a config file:

    uv run scripts/run.py --init-from runs/PHYSMAMBA_PURE.01_202609091000 --epochs 10

It names a run directory (its latest epoch), one of its ``epoch_NN`` folders
or a ``model.pt``. The interface, model and recipe are the checkpoint's, so
there is no ``--config``; ``--lr`` alone overrides the recipe's rate. The
``--epochs`` are new ones, numbered after the checkpoint's — continued from
epoch 20 for 10, the run writes ``epoch_21/`` to ``epoch_30/`` and a
``losses.csv`` of those rows — and the schedule starts afresh over them;
``main.py --init-from`` does this for every fold of an experiment. Without ``--datasets`` it trains
on the datasets that run trained on, exactly as it admitted them; it holds
out the participant that run held out, unless the flags name another or the
dataset is no longer among those trained on. It always writes a new run
directory, whose ``config.yaml`` lists under ``history`` every run the model
came through, oldest first: the checkpoint taken, and the datasets, split
and recipe that run trained on.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.dataset_config import load_dataset_configs              # noqa: E402
from src.datasets import load_stores                             # noqa: E402
from src.distributed import init_runtime, shutdown               # noqa: E402
from src.experiment import (                                     # noqa: E402
    EPOCH_DIR, add_config_arguments, add_limit_argument, add_run_arguments,
    add_split_arguments, check_config_arguments, check_split_arguments,
    compile_config, limit_windows, load_start,
    plot_records, print_model, print_runtime, print_setup, print_split,
    print_stores, records_dir, run_name, run_settings, split_stores,
    test_windows, train_windows,
)
from src.model_config import load_config                         # noqa: E402
from src.models import build_model                               # noqa: E402
from src.outputs import write_records                            # noqa: E402
from src.trainer import CHECKPOINT_NAME, DEFAULT_RUNS_DIR, Trainer  # noqa: E402

SCRIPT = "scripts/run.py"


def build_parser() -> argparse.ArgumentParser:
    parser = add_config_arguments(argparse.ArgumentParser(
        description="Train one model holding one participant out or none, run "
                    "it over the held-out participant and record it."),
        init_from=True)
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


def inherit(args, start) -> None:
    """Fill the flags a run started from a checkpoint left out: the datasets
    that run trained on, and the participant it held out while its dataset
    is still among those trained on."""
    setup = start.setup
    if not args.datasets:
        args.datasets = list(setup.datasets)
    held_out = setup.test_participant_dataset
    if args.test_participant_dataset is None and held_out in args.datasets:
        args.test_participant_dataset = held_out
        args.test_participant_id = setup.test_participant_id
        print(f"held out, as in {start.path}: {held_out} "
              f"participant {setup.test_participant_id}")


def main(argv=None) -> Path:
    """Fit and record the run; returns its directory."""
    parser = build_parser()
    args = parser.parse_args(argv)
    check_config_arguments(parser, args)
    check_split_arguments(parser, args)
    run = run_settings(parser, args)

    try:
        start = None
        if args.init_from:
            start = load_start(args.init_from, args.datasets, args.lr)
            inherit(args, start)
            print(f"init from: {start.path} (epoch {start.checkpoint.get('epoch')})")
            interface, model_config, training = (
                start.setup.interface, start.setup.model, start.setup.training)
        else:
            interface, model_config, training = load_config(args.config)
        print_setup(interface, model_config, training, run)
        # Name the run now, before the process group exists: the name
        # carries the start minute, and every rank stamps it here, within
        # milliseconds of the launch, so they all land in one directory.
        run_dir = (Path(args.run_dir) if args.run_dir
                   else Path(args.runs_dir) / run_name(args, model_config, training))
        if start and run_dir.resolve() in start.path.parents:
            raise ConfigError(f"{run_dir} is the run {start.path} belongs to; a "
                              f"run started from a checkpoint writes a new directory")
        # Resolve device / precision / process group against this machine
        # first: under a launch that cannot run distributed, only rank 0
        # continues past this line.
        runtime = init_runtime(training, run.gpu)
        print_runtime(runtime, run)
        configs = ({name: start.setup.datasets[name] for name in start.inherited}
                   if start and start.inherited else load_dataset_configs(args.datasets))
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

    # The model: one prediction per trace (a single-trace architecture is
    # copied per trace, a multi-trace one built once), widths from the
    # interface, dict in and dict out. The loss is the trainer's, not its.
    model = build_model(model_config, interface)
    print_model(model_config, model)

    print(f"run: {run_dir}")
    config = compile_config(SCRIPT, argv, args, interface, model_config, training,
                            run, runtime, configs, split, run_dir, start)
    try:
        trainer = Trainer(model, interface, model_config, training, run, runtime, run_dir, config)
    except ConfigError as err:
        parser.error(str(err))
    if start:
        # After the trainer, which sets the readout biases of a new model.
        try:
            trainer.restore(start.checkpoint)
        except RuntimeError as err:      # a state dict the model does not fit
            parser.error(f"{start.path} does not fit the model it describes: {err}")
        except ConfigError as err:
            parser.error(f"{start.path}: {err}")
    meta = {"dataset": args.test_participant_dataset,
            "participant": args.test_participant_id, "run_dir": str(run_dir),
            "command": config["command"], "git": config["git"]}

    def after_epoch(epoch: int) -> None:
        """That epoch's model over the held-out participant: every rank runs
        its shard, the main rank gets every record and alone keeps the
        weights, writes the records and draws them under ``epoch_NN/``.
        With nobody held out it keeps the weights and stops there."""
        records = trainer.test(test_dataset) if test_dataset is not None else None
        if not runtime.is_main:
            return
        epoch_dir = run_dir / EPOCH_DIR.format(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.checkpoint(), epoch_dir / CHECKPOINT_NAME)
        if records is None:
            return
        out_dir = records_dir(epoch_dir, args.test_participant_dataset)
        write_records(records, out_dir, interface, {**meta, "epoch": epoch},
                      split.test[args.test_participant_dataset])
        folders = plot_records(out_dir)
        print(f"epoch {epoch}: {len(records)} windows written to {out_dir}, "
              f"{len(folders)} recording(s) drawn")

    try:
        trainer.fit(train_dataset, after_epoch)
        return run_dir
    finally:
        shutdown(runtime)


if __name__ == "__main__":
    main()
