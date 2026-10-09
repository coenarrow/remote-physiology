"""One experiment: a model on a set of datasets, holding out one participant
after another, each fold a ``scripts/run.py`` run — several folds at a time,
each on its own GPUs.

    uv run python main.py --datasets pure --test-participant-dataset pure \\
        --config configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml \\
        --epochs 30
    uv run python main.py --datasets neckflix --test-participant-dataset neckflix \\
        --test-participant-id 32 33 34 --parallel 2 --nproc-per-node 2 \\
        --config configs/combined_model_config/physmamba_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml \\
        --epochs 20 --batch-size 4
    uv run python main.py --init-from runs/neckflix_physmamba --epochs 10 --batch-size 4

The folds are the participants named, in that order, or with none named
every participant the test dataset admits, in the order their ids sort;
every id is checked against the cache before the first fold starts. Each
fold is a ``scripts/run.py`` subprocess with the same flags (config, epochs,
batch size, workers, ``--no-gpu``) and one participant, with its own run
directory under ``runs/<experiment>/`` —
``<test dataset>_<model>``, e.g. ``runs/pure_physmamba/``, unless
``--experiment`` says otherwise — named as run.py names it, stamped when
the fold starts, and its own ``log.txt`` inside it. Nothing is written at
the experiment level. On a terminal each running fold shows a progress bar
over its epochs with the latest training loss, read from its ``losses.csv``;
in a log file the bars are silent and only the fold lines print.

``--init-from runs/<experiment>`` continues an experiment this script wrote
instead of starting one from a config file: every fold directory in it (one
``config.yaml`` and ``model.pt`` each) is continued from its latest epoch for
``--epochs`` more, numbered after the ones it finished, by ``scripts/run.py
--init-from``. The new folds keep their names with a new stamp, under
``runs/<experiment>_continued/`` unless ``--experiment`` says otherwise — a
directory of their own, because the evaluation pools every fold directory of
an experiment. ``--test-participant-id`` picks the folds; the datasets and
the held-out participant are each fold's own, unless ``--datasets`` names
others (the dev-box copy of a cache); ``--lr`` replaces the recipe's rate.

``--parallel K`` runs K folds at a time and ``--nproc-per-node N`` gives each
fold N processes, one per GPU, under ``torch.distributed.run``. The visible
CUDA devices are dealt to the running folds through ``CUDA_VISIBLE_DEVICES``:
with N of 1 a fold takes one device and folds may share it (the memory
report tool says how many fit); with more, each fold takes a disjoint group
of N, so K times N is at most the devices visible. Each fold's rendezvous
port is the job's (from the SLURM job id, else this process id) plus its
slot, so concurrent launches never collide. Both numbers default to 1, one
fold at a time in one process: the same command on the dev box and on the
cluster, where the SLURM file adds only the allocation and the two numbers.
On Windows N stays 1: that torch build has no libuv and the launcher's
rendezvous store cannot start without it. A fold that fails stops new folds
from starting; the running ones finish, and the experiment exits non-zero
naming the failures.
"""

import argparse
import csv
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from tqdm import tqdm

from src.config import ConfigError
from src.dataset_config import load_dataset_configs
from src.datasets import id_order, load_stores, participants
from src.evaluation.pooling import find_folds
from src.experiment import (
    add_config_arguments, add_limit_argument, add_run_arguments,
    check_config_arguments, run_argv, run_name, run_settings,
)
from src.model_config import load_config
from src.trainer import CHECKPOINT_NAME, CONFIG_NAME, DEFAULT_RUNS_DIR, LOSS_LOG_NAME

REPO_ROOT = Path(__file__).resolve().parent
RUN_SCRIPT = REPO_ROOT / "scripts" / "run.py"
LOG_NAME = "log.txt"
BASE_PORT = 29500
POLL_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Where a fold runs
# ---------------------------------------------------------------------------
def visible_gpus() -> list[str]:
    """The CUDA device ids this process may use, as the driver numbers them:
    ``CUDA_VISIBLE_DEVICES`` when set, else every device torch sees."""
    named = os.environ.get("CUDA_VISIBLE_DEVICES")
    if named is not None:
        return [s.strip() for s in named.split(",") if s.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def gpu_group(slot: int, gpus: list[str], nproc: int) -> list[str]:
    """The devices the fold in ``slot`` gets: one, shared round-robin, when
    a fold is one process; a disjoint group of ``nproc`` otherwise."""
    if not gpus:
        return []
    if nproc <= 1:
        return [gpus[slot % len(gpus)]]
    return gpus[slot * nproc:(slot + 1) * nproc]


def base_port() -> int:
    """The job's rendezvous port: from the SLURM job id under SLURM, else
    from this process id, so jobs sharing a node never collide. Each fold
    adds its slot."""
    seed = int(os.environ.get("SLURM_JOB_ID", os.getpid()))
    return BASE_PORT + seed % 1000


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = add_config_arguments(argparse.ArgumentParser(
        description="Run one experiment: hold out one participant after "
                    "another, each fold a scripts/run.py run, several at a time; "
                    "or continue every fold of an experiment already run."),
        init_from=True)
    add_run_arguments(parser)
    # The split flags of scripts/run.py, as an experiment takes them: the
    # dataset goes with --config, and the ids are several or none for all.
    parser.add_argument(
        "--test-participant-dataset", metavar="NAME",
        help="which loaded dataset the held-out participants come from "
             "(with --config; a continued experiment's folds say their own)")
    parser.add_argument(
        "--test-participant-id", nargs="+", metavar="ID",
        help="the participants to hold out, one fold each in this order, "
             "exactly as the stores write them (default: every participant "
             "the dataset admits); with --init-from, the folds to continue")
    parser.add_argument(
        "--experiment", metavar="NAME",
        help="the directory under runs/ the fold directories land in "
             "(default: <test dataset>_<model>, e.g. pure_physmamba; "
             "<experiment>_continued with --init-from)")
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="K",
        help="folds running at a time (default: 1)")
    parser.add_argument(
        "--nproc-per-node", type=int, default=1, metavar="N",
        help="processes per fold, one per GPU; more than one launches the fold "
             "under torch.distributed.run (default: 1)")
    add_limit_argument(parser)
    return parser


def continued_folds(source: Path, wanted: list[str] | None) -> list[tuple[str, Path]]:
    """``(participant, fold directory)`` for the folds of ``source`` to
    continue: every fold directory in it (one per ``config.yaml``) in id
    order, or those holding out the ``wanted`` participants in that order,
    each with a ``model.pt`` to continue from. Read from the folds'
    ``config.yaml`` alone; the checkpoints are ``scripts/run.py``'s to check."""
    folds = {}
    for fold in find_folds(source):
        with open(fold / CONFIG_NAME, encoding="utf-8") as handle:
            split = (yaml.safe_load(handle) or {}).get("split") or {}
        participant = split.get("test_participant_id")
        if participant is None:
            raise ConfigError(f"{fold} holds nobody out; only a fold with a "
                              f"held-out participant is continued")
        participant = str(participant)
        if participant in folds:
            raise ConfigError(f"{source} has two folds holding out {participant}: "
                              f"{folds[participant].name} and {fold.name}")
        folds[participant] = fold
    if wanted:
        missing = [p for p in wanted if p not in folds]
        if missing:
            raise ConfigError(f"--test-participant-id {missing} match no fold of "
                              f"{source}; its folds hold out {sorted(folds, key=id_order)}")
        chosen = [(p, folds[p]) for p in wanted]
    else:
        chosen = sorted(folds.items(), key=lambda item: id_order(item[0]))
    weightless = [fold.name for _, fold in chosen if not (fold / CHECKPOINT_NAME).is_file()]
    if weightless:
        raise ConfigError(f"no {CHECKPOINT_NAME} to continue from in {weightless}")
    return chosen


def fold_argv(args, participant: str, run_dir: Path, source: Path | None) -> list[str]:
    """The ``scripts/run.py`` arguments of one fold, into this run directory:
    the experiment's flags with this one participant held out — or, continuing
    ``source``, its checkpoint in place of the config and the split, which
    run.py inherits from it."""
    if source is None:
        argv = ["--datasets", *args.datasets, "--config", str(args.config),
                "--test-participant-dataset", args.test_participant_dataset,
                "--test-participant-id", participant]
    else:
        argv = ["--init-from", str(source)]
        if args.datasets:
            argv += ["--datasets", *args.datasets]
        if args.lr is not None:
            argv += ["--lr", str(args.lr)]
    argv += [*run_argv(args.run), "--run-dir", str(run_dir)]
    if args.limit_windows:
        argv += ["--limit-windows", str(args.limit_windows)]
    return argv


def fold_command(argv: list[str], nproc: int, port: int) -> list[str]:
    """The fold's process: ``scripts/run.py`` itself, or under
    ``torch.distributed.run`` with ``nproc`` processes on ``port``."""
    if nproc <= 1:
        return [sys.executable, str(RUN_SCRIPT), *argv]
    return [sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}", f"--master_port={port}",
            str(RUN_SCRIPT), *argv]


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------
@dataclass
class Fold:
    index: int
    participant: str
    run_dir: Path
    process: subprocess.Popen
    log: object
    started: float
    bar: tqdm
    epochs_seen: int = 0


def say(line: str) -> None:
    """A line of the launcher's own output, kept clear of the bars."""
    tqdm.write(line)
    sys.stdout.flush()


def epochs_logged(run_dir: Path) -> tuple:
    """``(epochs, last total loss)`` from the fold's ``losses.csv``;
    ``(0, None)`` before the first epoch, or while the trainer is rewriting
    the file."""
    path = run_dir / LOSS_LOG_NAME
    if not path.is_file():
        return 0, None
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return len(rows), float(rows[-1]["total"])
    except (OSError, csv.Error, IndexError, KeyError, TypeError, ValueError):
        return 0, None


def launch(args, model_config, training, runs_dir: Path, index: int, total: int,
           participant: str, source: Path | None, slot: int, gpus: list[str]) -> Fold:
    """Start one fold in ``slot``: its run directory and log under
    ``runs_dir``, its GPUs, its process, and its progress bar over the
    run's epochs. A fold continued from ``source`` keeps the source's name
    with a new stamp."""
    now = datetime.now()
    if source is None:
        setup = SimpleNamespace(datasets=args.datasets,
                                test_participant_dataset=args.test_participant_dataset,
                                test_participant_id=participant)
        name = run_name(setup, model_config, training, now)
    else:
        name = f"{source.name.rsplit('_', 1)[0]}_{now:%Y%m%d%H%M}"
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    group = gpu_group(slot, gpus, args.nproc_per_node)
    env = dict(os.environ)
    if group:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(group)
    command = fold_command(fold_argv(args, participant, run_dir, source),
                           args.nproc_per_node, base_port() + slot)
    log = open(run_dir / LOG_NAME, "w", encoding="utf-8")
    log.write(f"{shlex.join(command)}\nCUDA_VISIBLE_DEVICES={','.join(group) or '(unset)'}\n\n")
    log.flush()
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    what = (f"continuing {source.name} ({epochs_logged(source)[0]} epochs)" if source
            else f"holding out {args.test_participant_dataset} {participant}")
    say(f"fold {index}/{total}: {what} on GPU(s) {','.join(group) or 'none'} -> {run_dir}")
    # disable=None: a bar on a terminal, nothing in a log file.
    bar = tqdm(total=args.run.epochs, desc=f"fold {index}/{total} {participant}",
               unit="epoch", position=slot, leave=False, disable=None,
               dynamic_ncols=True)
    return Fold(index, participant, run_dir, process, log, time.time(), bar)


def run_pool(args, model_config, training, runs_dir: Path,
             folds: list[tuple[str, Path | None]], gpus: list[str]) -> tuple[list, list, list]:
    """Run every fold — ``(participant, source directory or None)`` —
    ``args.parallel`` at a time, under ``runs_dir``; returns the run
    directories finished, the folds that failed, and the participants never
    started because one failed."""
    total, queue = len(folds), list(enumerate(folds, 1))
    running: dict[int, Fold] = {}
    done, failed = [], []
    try:
        while queue or running:
            while queue and not failed and len(running) < args.parallel:
                index, (participant, source) = queue.pop(0)
                slot = next(s for s in range(args.parallel) if s not in running)
                running[slot] = launch(args, model_config, training, runs_dir,
                                       index, total, participant, source, slot, gpus)
            if not running:
                break
            time.sleep(POLL_SECONDS)
            for slot, fold in list(running.items()):
                logged, loss = epochs_logged(fold.run_dir)
                if logged > fold.epochs_seen:
                    fold.bar.update(logged - fold.epochs_seen)
                    fold.bar.set_postfix(loss=f"{loss:.4g}")
                    fold.epochs_seen = logged
                code = fold.process.poll()
                if code is None:
                    continue
                fold.bar.close()
                fold.log.close()
                del running[slot]
                elapsed = time.time() - fold.started
                if code == 0:
                    done.append(fold.run_dir)
                    say(f"fold {fold.index}/{total}: done in {elapsed:.0f}s, "
                        f"final loss {loss:.4g}" if loss is not None else
                        f"fold {fold.index}/{total}: done in {elapsed:.0f}s")
                else:
                    failed.append(fold)
                    say(f"fold {fold.index}/{total}: failed with exit status {code} "
                        f"after {elapsed:.0f}s; see {fold.run_dir / LOG_NAME}")
    except KeyboardInterrupt:
        for fold in running.values():
            fold.bar.close()
            fold.process.terminate()
        raise
    return done, failed, [participant for _, (participant, _) in queue]


def main(argv=None) -> list[Path]:
    """Run every fold; returns their run directories, in the order finished."""
    parser = build_parser()
    args = parser.parse_args(argv)
    check_config_arguments(parser, args)
    if (args.test_participant_dataset is None) == (args.init_from is None):
        parser.error("--config needs --test-participant-dataset; --init-from "
                     "takes the held-out dataset from the folds it continues")
    if args.parallel < 1 or args.nproc_per_node < 1:
        parser.error("--parallel and --nproc-per-node are at least 1")
    if args.nproc_per_node > 1 and platform.system() == "Windows":
        # torch.distributed.run opens its rendezvous TCPStore with libuv,
        # which the Windows torch build lacks, and it never passes
        # use_libuv=False; the launch fails before any worker starts.
        parser.error(f"--nproc-per-node {args.nproc_per_node} launches the fold "
                     f"under torch.distributed.run, whose rendezvous store cannot "
                     f"start on the Windows torch build (no libuv); use one "
                     f"process per fold here, several on Linux")
    args.run = run_settings(parser, args)
    model_config = training = None
    try:
        gpus = visible_gpus() if args.run.gpu else []
        need = args.parallel * args.nproc_per_node
        if args.nproc_per_node > 1 and gpus and need > len(gpus):
            raise ConfigError(
                f"--parallel {args.parallel} x --nproc-per-node {args.nproc_per_node} "
                f"needs {need} GPUs and {len(gpus)} are visible ({','.join(gpus)})")
        if args.init_from:
            source = Path(args.init_from)
            folds = continued_folds(source, args.test_participant_id)
            runs_dir = Path(DEFAULT_RUNS_DIR) / (args.experiment or f"{source.name}_continued")
            what = f"continuing {len(folds)} fold(s) of {source}"
        else:
            dataset = args.test_participant_dataset
            _, model_config, training = load_config(args.config)
            present = participants(load_stores(load_dataset_configs(args.datasets)), dataset)
            ids = args.test_participant_id or present
            missing = [p for p in ids if p not in present]
            if missing:
                raise ConfigError(
                    f"--test-participant-id {missing} match no admitted store in "
                    f"{dataset!r}; its participants are {present}")
            if training.MODEL_FILE_NAME and len(ids) > 1:
                raise ConfigError(
                    f"{args.config} names the run {training.MODEL_FILE_NAME!r}, so "
                    f"every fold would write the same directory; unset MODEL_FILE_NAME")
            folds = [(participant, None) for participant in ids]
            model = model_config.NAME.lower()
            runs_dir = Path(DEFAULT_RUNS_DIR) / (args.experiment or f"{dataset}_{model}")
            what = (f"{model} on {args.datasets}, {len(folds)} fold(s) holding out "
                    f"{dataset} {ids}")
    except ValueError as err:          # ConfigError is a ValueError
        parser.error(str(err))

    print(f"experiment: {what}, {args.parallel} at a time, "
          f"{args.nproc_per_node} process(es) each, GPUs {','.join(gpus) or 'none'}, "
          f"into {runs_dir}", flush=True)
    done, failed, skipped = run_pool(args, model_config, training, runs_dir, folds, gpus)
    if failed:
        sys.exit(f"{len(failed)} fold(s) failed ({', '.join(f.participant for f in failed)}); "
                 f"{len(done)} finished, {len(skipped)} never started "
                 f"({', '.join(skipped) or 'none'})")
    print(f"experiment: {len(done)} fold(s) done")
    return done


if __name__ == "__main__":
    main()
