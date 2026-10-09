"""Run a trained model over datasets it has not seen and record it.

    uv run scripts/test.py --checkpoint runs/PHYSMAMBA_PURE.all_202609091000 \\
        --test-dataset neckflix synthetic_neck

``--checkpoint`` names a run directory written by ``scripts/run.py``: its
latest epoch is tested, or the one ``--epoch`` names. An ``epoch_NN`` folder
or a ``model.pt`` is tested as it is. The interface and the model are the
checkpoint's; nothing is trained.

Every admitted store of each ``--test-dataset`` is run, every strided
window of it, or one participant's stores with ``--test-participant-id``
(one dataset only: ids are a dataset's own). A store the model has trained
on, in its own run or in any run of its ``history``, is refused.

The records land beside the weights that made them, one directory per
dataset, in the layout a held-out run writes and drawn the same way::

    RUN_DIR/epoch_NN/test_records/<dataset>/
      meta.json  windows.csv  recordings.csv
      <recording>/<perspective>/<TRACE>.csv, <TRACE>.png

``src/outputs.py`` describes the files; ``scripts/evaluate.py`` scores
them. A dataset already tested at that epoch is refused, not overwritten.
"""

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.dataset_config import load_dataset_configs              # noqa: E402
from src.datasets import Split, hold_out_participant, load_stores  # noqa: E402
from src.distributed import init_runtime, shutdown               # noqa: E402
from src.experiment import (                                     # noqa: E402
    add_limit_argument, add_run_arguments, epoch_checkpoint, git_state,
    limit_windows, load_checkpoint, plot_records, print_model, print_runtime,
    print_stores, rebuild, records_dir, run_settings, test_windows,
    trained_stores,
)
from src.models import build_model                               # noqa: E402
from src.outputs import write_records                            # noqa: E402
from src.trainer import Trainer                                  # noqa: E402

SCRIPT = "scripts/test.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained model over datasets it has not seen and "
                    "record it.")
    parser.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="the trained model: a run directory written by scripts/run.py "
             "(its latest epoch, or --epoch), one of its epoch_NN folders, "
             "or a model.pt")
    parser.add_argument(
        "--epoch", type=int, metavar="N",
        help="which epoch of the run directory to test (default: the latest)")
    parser.add_argument(
        "--test-dataset", nargs="+", required=True, metavar="NAME",
        help="dataset config name(s) to test on, each resolved to "
             "configs/datasets/<NAME>.yaml and written to its own directory")
    parser.add_argument(
        "--test-participant-id", metavar="ID",
        help="test this participant's stores only, exactly as the stores "
             "write the id (one --test-dataset; default: every admitted store)")
    add_run_arguments(parser, epochs=False)
    add_limit_argument(parser)
    return parser


def test_stores(stores: dict, participant: str | None, trained: set) -> dict:
    """``{dataset: stores}`` to test: every admitted store, or the one
    participant's; refused where the model trained on any of them."""
    if participant is not None:
        (dataset,) = stores
        stores = hold_out_participant(stores, dataset, participant).test
    for dataset, kept in stores.items():
        if not kept:
            raise ConfigError(f"{dataset} admitted no store to test on")
        seen = sorted(path.name for path in kept if path.resolve() in trained)
        if seen:
            raise ConfigError(
                f"the model trained on {len(seen)} of the {len(kept)} stores of "
                f"{dataset} to test ({', '.join(seen[:5])}"
                f"{', ...' if len(seen) > 5 else ''}); a test is of stores it "
                f"has not seen. --test-participant-id picks one participant's")
    return stores


def main(argv=None) -> list:
    """Record and draw each dataset; returns the records directories."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test_participant_id is not None and len(args.test_dataset) > 1:
        parser.error("--test-participant-id goes with one --test-dataset; "
                     "participant ids are a dataset's own")
    run = run_settings(parser, args)

    try:
        path = epoch_checkpoint(args.checkpoint, args.epoch)
        checkpoint = load_checkpoint(path)
        setup = rebuild(checkpoint["config"], str(path))
        interface, model_config, training = setup.interface, setup.model, setup.training
        print(f"checkpoint: {path} (epoch {checkpoint.get('epoch')})")
        out_dirs = {name: records_dir(path.parent, name) for name in args.test_dataset}
        for out_dir in out_dirs.values():
            if out_dir.exists():
                raise ConfigError(f"{out_dir} is already there; a test does not "
                                  f"overwrite one. Delete it to test again")
        runtime = init_runtime(training, run.gpu)
        print_runtime(runtime, run)
        configs = load_dataset_configs(args.test_dataset)
        stores = load_stores(configs)
        print_stores(configs, stores)
        stores = test_stores(stores, args.test_participant_id,
                             trained_stores(checkpoint["config"]))
        datasets = {name: limit_windows(
                        test_windows(Split(train={}, test={name: kept}), interface),
                        args.limit_windows)
                    for name, kept in stores.items()}
    except (ValueError, FileNotFoundError) as err:      # ConfigError is a ValueError
        parser.error(str(err))

    model = build_model(model_config, interface)
    print_model(model_config, model)
    try:
        trainer = Trainer(model, interface, model_config, training, run, runtime,
                          path.parent, checkpoint["config"])
        # After the trainer, which sets the readout biases of a new model.
        trainer.restore(checkpoint)
    except ConfigError as err:
        parser.error(str(err))
    except RuntimeError as err:          # a state dict the model does not fit
        parser.error(f"{path} does not fit the model it describes: {err}")

    meta = {"participant": args.test_participant_id,
            "run_dir": checkpoint["config"].get("run_dir"), "checkpoint": str(path),
            "epoch": checkpoint.get("epoch"),
            "command": shlex.join([SCRIPT, *(sys.argv[1:] if argv is None else argv)]),
            "git": git_state()}
    try:
        for name, dataset in datasets.items():
            records = trainer.test(dataset)
            if not runtime.is_main:
                continue
            write_records(records, out_dirs[name], interface,
                          {"dataset": name, **meta}, stores[name])
            folders = plot_records(out_dirs[name])
            print(f"{name}: {len(records)} windows written to {out_dirs[name]}, "
                  f"{len(folders)} recording(s) drawn")
        return list(out_dirs.values())
    finally:
        shutdown(runtime)


if __name__ == "__main__":
    main()
