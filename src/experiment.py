"""What ``scripts/run.py`` is made of beyond the trainer: a run's arguments,
its compiled config, and rebuilding a run from that config.

``scripts/run.py`` fits a model, writes a run directory, runs the fitted
model over the held-out participant and scores the records;
``tools/memory_report.py`` takes the same setup without running it. This
module is written once for both:

* the argparse groups they have in common — the dataset files and the one
  config file a run is made of (``src.model_config.load_config`` reads it),
  the run settings (epochs, batch size, workers, GPU or not), and the
  held-out participant;
* ``compile_config``, the one mapping of everything a run ran on, which the
  run writes as ``config.yaml`` and carries inside ``model.pt``;
* ``rebuild``, the typed reading of that mapping back into the interface,
  model, recipe and datasets — through the same parsers the files go
  through (``src.model_config``, ``src.dataset_config``), so a run rebuilt
  from its checkpoint is checked exactly as a loaded one is;
* the windowed datasets each side of the split becomes, and the progress
  lines the script prints on the way.
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset

from src.config import ConfigError, build
from src.dataset_config import (
    DatasetConfig, parse_dataset_config, resolve_dataset_configs,
)
from src.datasets import Split, hold_out_participant
from src.distributed import Runtime
from src.model_config import (
    InterfaceConfig, RunSettings, TrainingConfig, parse_interface,
    parse_model_config, parse_training,
)
from src.trainer import CHECKPOINT_NAME, CONFIG_NAME
from src.inputs import WindowedDataset

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The config files every run is made of: the datasets, and the one file
    that holds the model, its interface and its training recipe. Shared with
    the tools that take a run's setup without running it
    (``tools/memory_report.py``)."""
    parser.add_argument(
        "--datasets", nargs="+", required=True, metavar="NAME",
        help="dataset config name(s), each resolved to "
             "configs/datasets/<NAME>.yaml (e.g. --datasets neckflix pure)")
    parser.add_argument(
        "--config", required=True, metavar="PATH",
        help="the config file: MODEL, INTERFACE and TRAIN sections "
             "(e.g. configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml)")
    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The run settings (:class:`RunSettings`), defaults from the dataclass."""
    d = RunSettings()
    parser.add_argument(
        "--epochs", type=int, default=d.epochs, metavar="N",
        help=f"training epochs (default: {d.epochs})")
    parser.add_argument(
        "--batch-size", type=int, default=d.batch_size, metavar="N",
        help=f"windows per batch, per process, train and test alike "
             f"(default: {d.batch_size})")
    parser.add_argument(
        "--num-workers", type=int, default=d.num_workers, metavar="N",
        help=f"DataLoader worker processes (default: {d.num_workers})")
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="train on the CPU even where a GPU exists (default: use CUDA or "
             "Apple MPS if present, else fall back to the CPU with a warning)")
    return parser


def run_settings(parser: argparse.ArgumentParser, args) -> RunSettings:
    """The parsed run flags as one :class:`RunSettings`, checked."""
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size are at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers is at least 0")
    return RunSettings(epochs=args.epochs, batch_size=args.batch_size,
                       num_workers=args.num_workers, gpu=not args.no_gpu)


def run_argv(run: RunSettings) -> list[str]:
    """The flags that reproduce ``run`` on another launcher's command line."""
    argv = ["--epochs", str(run.epochs), "--batch-size", str(run.batch_size),
            "--num-workers", str(run.num_workers)]
    if not run.gpu:
        argv.append("--no-gpu")
    return argv


def add_split_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The held-out participant, as a pair of flags that go together."""
    parser.add_argument(
        "--test-participant-dataset", metavar="NAME",
        help="which loaded dataset the test participant comes from")
    parser.add_argument(
        "--test-participant-id", metavar="ID",
        help="the participant held out of that dataset, exactly as its "
             "stores write it (e.g. '1' for Neckflix, '01' for PURE)")
    return parser


def add_limit_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--limit-windows", type=int, default=0, metavar="N",
        help="smoke runs: keep N evenly spaced windows of each split")
    return parser


def check_split_arguments(parser: argparse.ArgumentParser, args) -> None:
    if (args.test_participant_dataset is None) != (args.test_participant_id is None):
        parser.error("--test-participant-dataset and --test-participant-id go together")


# ---------------------------------------------------------------------------
# The split and its windows
# ---------------------------------------------------------------------------
def split_stores(stores: dict, dataset: str | None, participant: str | None) -> Split:
    """Hold ``participant`` out of ``dataset``, or — with neither named —
    train on every admitted store and hold nobody out."""
    if dataset is None:
        return Split(train=stores, test={})
    return hold_out_participant(stores, dataset, participant)


def limit_windows(dataset, limit: int):
    """Every k-th window so a smoke run touches the whole set, not its head."""
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    step = len(dataset) / limit
    return Subset(dataset, [int(i * step) for i in range(limit)])


def train_windows(split: Split, interface: InterfaceConfig) -> ConcatDataset:
    """One WindowedDataset per training dataset, concatenated; training draws
    one random window per (recording, perspective)."""
    parts = [WindowedDataset(name, kept, interface, mode="random")
             for name, kept in split.train.items() if kept]
    if not parts:
        raise ConfigError("no dataset admitted a store to train on")
    for part in parts:
        print(f"train {part.name}: {len(part)} windows (random, one per sample)")
    dataset = ConcatDataset(parts)
    print(f"train total: {len(dataset)} windows")
    return dataset


def test_windows(split: Split, interface: InterfaceConfig) -> WindowedDataset:
    """The held-out participant's windows, strided."""
    (name, stores), = split.test.items()
    dataset = WindowedDataset(name, stores, interface, mode="strided")
    print(f"test  {dataset.name}: {len(dataset)} windows (strided)")
    return dataset


# ---------------------------------------------------------------------------
# The compiled config, and a run rebuilt from it
# ---------------------------------------------------------------------------
def git_state() -> dict:
    """``{commit, dirty}`` of the checkout, or ``None`` values outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def run_name(args, model_config, training: TrainingConfig,
             now: datetime | None = None) -> str:
    """``MODEL_FILE_NAME`` if the recipe names the run; otherwise
    ``<MODEL>_<DATASET>.<held-out participant or all>-..._<YYYYMMDDHHMM>``,
    e.g. ``PHYSMAMBA_PURE.all-NECKFLIX.24_202609091000`` for PhysMamba
    trained on all of PURE and on Neckflix minus participant 24."""
    if training.MODEL_FILE_NAME:
        return training.MODEL_FILE_NAME
    stamp = (now or datetime.now()).strftime("%Y%m%d%H%M")
    datasets = "-".join(
        f"{name.upper()}."
        f"{args.test_participant_id if name == args.test_participant_dataset else 'all'}"
        for name in args.datasets)
    return f"{model_config.NAME.upper()}_{datasets}_{stamp}"


def compile_config(script: str, argv, args, interface: InterfaceConfig, model_config,
                   training: TrainingConfig, run: RunSettings, runtime: Runtime,
                   datasets: dict, split: Split, run_dir: Path) -> dict:
    """Everything this run ran on, as one plain mapping.

    The four config sections (``datasets``, ``interface``, ``model``,
    ``training``) are exactly what their files loaded as after ``BASE``
    merging and validation, so each feeds back to its parser (``rebuild``);
    ``sources`` says which files those were: the dataset files, and the one
    config file the other three sections came from. ``run`` is the run
    settings the flags gave. ``split`` lists the stores on
    each side of the hold-out (``test`` is empty when nobody is held out),
    ``runtime`` the device and precision actually used (after any
    downgrade), ``git`` the code the run executed. Written to the run
    directory as ``config.yaml`` and carried inside the checkpoint.
    """
    return {
        "command": shlex.join([script, *(sys.argv[1:] if argv is None else argv)]),
        "git": git_state(),
        "run_dir": str(run_dir),
        "sources": {
            "datasets": {name: str(path.resolve())
                         for name, path in resolve_dataset_configs(args.datasets).items()},
            "config": str(Path(args.config).resolve()),
        },
        "datasets": {name: asdict(cfg) for name, cfg in datasets.items()},
        "split": {
            "test_participant_dataset": args.test_participant_dataset,
            "test_participant_id": args.test_participant_id,
            "train": {name: sorted(p.name for p in kept) for name, kept in split.train.items()},
            "test": {name: sorted(p.name for p in kept) for name, kept in split.test.items()},
        },
        "interface": asdict(interface),
        "model": asdict(model_config),
        "training": asdict(training),
        "run": asdict(run),
        "runtime": {
            "device": str(runtime.device),
            "precision": runtime.precision,
            "world_size": runtime.world_size,
        },
        "limit_windows": args.limit_windows,
    }


@dataclass
class Setup:
    """A run's configs, typed again from its compiled config."""

    interface: InterfaceConfig
    model: object                        # one of ``src.model_config.MODEL_CONFIGS``
    training: TrainingConfig
    run: RunSettings
    datasets: dict[str, DatasetConfig]
    test_participant_dataset: str | None
    test_participant_id: str | None


def rebuild(config: dict, where: str = CONFIG_NAME) -> Setup:
    """The typed reading of a compiled config, through the parsers the files
    went through: a rebuilt run is validated exactly as a loaded one was."""
    for key in ("interface", "model", "training", "run", "datasets"):
        if key not in config:
            raise ConfigError(f"{where} has no {key} section; was it written by "
                              f"scripts/run.py?")
    interface = parse_interface(config["interface"], f"{where}: interface")
    model = parse_model_config(config["model"], interface, f"{where}: model")
    training = parse_training(config["training"], f"{where}: training")
    run = build(RunSettings, config["run"], f"{where}: run")
    datasets = {name: parse_dataset_config(mapping, f"{where}: datasets.{name}")
                for name, mapping in config["datasets"].items()}
    split = config.get("split") or {}
    return Setup(interface, model, training, run, datasets,
                 split.get("test_participant_dataset"), split.get("test_participant_id"))


def load_checkpoint(run_dir: Path) -> dict:
    """``{"model_state", "config"}`` from a run directory's ``model.pt``."""
    path = Path(run_dir) / CHECKPOINT_NAME
    if not path.is_file():
        raise ConfigError(f"{run_dir} has no {CHECKPOINT_NAME}; is it a run "
                          f"directory written by scripts/run.py?")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("config"):
        raise ConfigError(f"{path} carries no compiled config; nothing to rebuild "
                          f"the run from")
    return checkpoint


# ---------------------------------------------------------------------------
# What every script prints on the way
# ---------------------------------------------------------------------------
def print_setup(interface: InterfaceConfig, model_config, training: TrainingConfig,
                run: RunSettings) -> None:
    print(f"interface: {interface.window_frames} frames per window at "
          f"{interface.FS} fps, test stride {interface.stride_frames} "
          f"frames, channels {interface.CHANNELS}, traces {interface.TRACES}")
    print(f"model: {model_config}")
    print(f"training: {training}")
    print(f"run: {run.epochs} epoch(s), batch {run.batch_size}, "
          f"{run.num_workers} worker(s), {'gpu' if run.gpu else 'no gpu'}")


def print_runtime(runtime: Runtime, run: RunSettings) -> None:
    if runtime.distributed:
        print(f"runtime: rank {runtime.rank} of {runtime.world_size} on "
              f"{runtime.device}, global batch "
              f"{run.batch_size * runtime.world_size}")
    else:
        print(f"runtime: single process on {runtime.device}, "
              f"precision {runtime.precision}")


def print_stores(configs: dict, stores: dict) -> None:
    for name, kept in stores.items():
        print(f"{name}: {len(kept)} stores admitted from {configs[name].CACHED_PATH}")


def print_split(split: Split) -> None:
    for name, kept in split.train.items():
        print(f"train {name}: {len(kept)} stores")
    if not split.test:
        print("test: nobody held out")
    for name, kept in split.test.items():
        print(f"test  {name}: {len(kept)} stores ({', '.join(p.name for p in kept)})")


def print_model(model_config, model) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model_config.NAME} x {len(model.traces)} traces, "
          f"{model.in_channels} input channels, {n_params:,} parameters")
