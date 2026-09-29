"""What a sweep needs to be tuned to a node, measured rather than guessed.

Two commands, one for each half of the question.

``convergence`` reads finished runs and says how many epochs they needed:
per run, the training loss and the held-out level error (the mean of
``|err_mean|``, ``|err_max|`` and ``|err_min|`` over the scored recordings,
in the trace's own units) at the epochs asked for, at the last epoch and at
the best one. It runs anywhere the run directories are.

    uv run tools/tuning_report.py convergence runs --at 10 20

``probe`` measures what one step of each config costs on the GPU it runs on
and turns that into the launcher's numbers. Per config and per batch size,
doubling until the card runs out: the peak memory of a training step, the
seconds a step takes, the seconds a forward pass takes, and the seconds one
core takes to load one window. From those, for a node of ``--gpus`` cards
and ``--cpus`` cores: the ``--batch-size``, ``--parallel``,
``--nproc-per-node`` and ``--num-workers`` that keep the recipe's global
batch, whether the cards or the loaders set the pace, and the hours the
sweep takes. The steps are real ones on real windows, as in
``tools/memory_report.py``: it needs the cache mounted and the GPU the sweep
would use (``.slurm_scripts/Neckflix_Tuning.slurm``).

    uv run tools/tuning_report.py probe --datasets neckflix_hpc \\
        --configs configs/hpc_configs --gpus 4 --cpus 48 --epochs 20

``--resize H W`` measures the same configs at another frame size, in place of
the one their ``INTERFACE`` states, so a size can be costed before a
directory of configs is written for it.

The hours leave out what happens after each epoch's test pass (writing the
records, scoring them, the plots), which no single step can measure, and the
cost of synchronising gradients between cards.
"""

import argparse
import gc
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_config import load_dataset_configs          # noqa: E402
from src.datasets import load_stores, participants           # noqa: E402
from src.distributed import init_runtime, shutdown           # noqa: E402
from src.inputs import WindowedDataset                       # noqa: E402
from src.memory import device_memory, human, peak_memory, reset_peak  # noqa: E402
from src.model_config import ResizeConfig, RunSettings, load_config  # noqa: E402
from src.models import build_model                           # noqa: E402
from src.trainer import LOSS_LOG_NAME, Trainer, move_to_device  # noqa: E402
from tools.memory_report import CUDA_CONTEXT_BYTES           # noqa: E402

LEVEL_ERRORS = ("err_mean", "err_max", "err_min")
BATCH_SIZES = (1, 2, 4, 8, 16)


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------
def find_runs(paths: list[str]) -> list[Path]:
    """Every run directory (a folder holding ``losses.csv``) at or below ``paths``."""
    runs = {log.parent for path in paths for log in Path(path).rglob(LOSS_LOG_NAME)}
    return sorted(runs)


def held_out_errors(run: Path) -> pd.DataFrame:
    """``epoch x trace`` level error of the held-out recordings, from every
    ``epoch_NN/**/signals.csv`` the run scored; empty where it scored none."""
    rows = []
    for epoch_dir in sorted(run.glob("epoch_*")):
        tables = [pd.read_csv(path) for path in epoch_dir.rglob("signals.csv")]
        if not tables:
            continue
        table = pd.concat(tables)
        table["level"] = table[list(LEVEL_ERRORS)].abs().mean(axis=1)
        means = table.groupby("signal")["level"].mean()
        rows.append({"epoch": int(epoch_dir.name.split("_")[1]), **means.to_dict()})
    return pd.DataFrame(rows).set_index("epoch") if rows else pd.DataFrame()


def convergence_row(run: Path, at: list[int]) -> dict:
    """One run's line: its loss and held-out level error at the epochs ``at``,
    at its last epoch and at its best."""
    losses = pd.read_csv(run / LOSS_LOG_NAME).set_index("epoch")
    last = int(losses.index.max())
    row = {"experiment": run.parent.name, "run": run.name, "epochs": last,
           "s/epoch": losses["seconds"].median()}
    for epoch in [*at, last]:
        if epoch in losses.index:
            row[f"loss@{epoch}"] = losses.loc[epoch, "total"]
    errors = held_out_errors(run)
    for trace in errors.columns:
        series = errors[trace].dropna()
        if series.empty:
            continue
        for epoch in [*at, last]:
            if epoch in series.index:
                row[f"{trace}@{epoch}"] = series[epoch]
        row[f"{trace} best"] = series.min()
        row[f"{trace} best epoch"] = int(series.idxmin())
    return row


def convergence(args) -> pd.DataFrame:
    runs = find_runs(args.runs)
    if not runs:
        raise SystemExit(f"no {LOSS_LOG_NAME} at or below {args.runs}")
    at = sorted(set(args.at))
    table = pd.DataFrame([convergence_row(run, at) for run in runs])
    print(f"{len(table)} run(s); loss is the training loss, a trace's column is the "
          f"held-out level error (mean of |{'|, |'.join(LEVEL_ERRORS)}|) in its own units")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    return table


# ---------------------------------------------------------------------------
# probe: measuring
# ---------------------------------------------------------------------------
def usable_cpus() -> int:
    """The cores this process may run on, which a SLURM binding can make
    fewer than the node has."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def config_files(paths: list[str]) -> list[Path]:
    """The config files named, a directory standing for every YAML in it."""
    files = []
    for path in map(Path, paths):
        files += sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    return files


def load_setup(path: Path, resize) -> tuple:
    """``(interface, model, training)`` of the config at ``path``, its frame
    size replaced by ``resize = (H, W)`` when one is given."""
    interface, model_config, training = load_config(path)
    if resize:
        interface.RESIZE = ResizeConfig(*resize)
    return interface, model_config, training


def timed_batch(dataset, batch_size: int, offset: int):
    """One collated batch of ``batch_size`` windows spread over the dataset,
    loaded in this process on one thread as a DataLoader worker loads them,
    and the seconds each window took."""
    step = max(len(dataset) // batch_size, 1)
    indices = [(offset + i * step) % len(dataset) for i in range(batch_size)]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=0)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        started = time.perf_counter()
        batch = next(iter(loader))
        seconds = time.perf_counter() - started
    finally:
        torch.set_num_threads(threads)
    return batch, seconds / batch_size


def timed(device: torch.device, call, repeats: int) -> float:
    """Median seconds of ``call()`` over ``repeats``, the device's queue drained."""
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds.append(time.perf_counter() - started)
    return statistics.median(seconds)


def measure_batch(setup, runtime, batch, batch_size: int, steps: int) -> dict:
    """Peak memory and seconds of a training step, and seconds of a forward
    pass, for one batch on a newly built model."""
    interface, model_config, training = setup
    run = RunSettings(epochs=1, batch_size=batch_size, num_workers=0)
    trainer = Trainer(build_model(model_config, interface), interface, model_config,
                      training, run, runtime, Path(tempfile.gettempdir()) / "tuning_report")
    device = runtime.device
    reset_peak(device)
    trainer.step(batch)                      # the first step also picks the kernels
    peak = peak_memory(device)
    step_seconds = timed(device, lambda: trainer.step(batch), steps)

    def forward():
        with torch.no_grad(), trainer._autocast():
            trainer.net(move_to_device(batch, device))

    trainer.net.eval()
    forward()
    return {"peak_reserved": peak["reserved"] if peak else 0,
            "peak_allocated": peak["allocated"] if peak else 0,
            "step_seconds": step_seconds,
            "forward_seconds": timed(device, forward, steps)}


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def probe_config(path: Path, stores: dict, args) -> list[dict]:
    """One row per batch size that was tried for the config at ``path``; the
    last row says ``fits`` False when the card ran out."""
    setup = load_setup(path, args.resize)
    interface, model_config, training = setup
    runtime = init_runtime(training, True)
    dataset = ConcatDataset([WindowedDataset(name, kept, interface, mode="random")
                             for name, kept in stores.items() if kept])
    memory = device_memory(runtime.device)
    rows = []
    try:
        for n, batch_size in enumerate(size for size in BATCH_SIZES
                                       if size <= args.max_batch):
            batch, load_seconds = timed_batch(dataset, batch_size, offset=n)
            row = {"config": path.name, "model": model_config.NAME,
                   "frame": f"{interface.RESIZE.H}x{interface.RESIZE.W}",
                   "precision": runtime.precision, "batch": batch_size,
                   "load_seconds": load_seconds,
                   "card_total": memory["total"] if memory else 0, "fits": True}
            try:
                row.update(measure_batch(setup, runtime, batch, batch_size, args.steps))
            except RuntimeError as err:          # torch.OutOfMemoryError is one
                if "out of memory" not in str(err).lower():
                    raise
                row["fits"] = False
            rows.append(row)
            print(describe_row(row), flush=True)
            del batch
            release(runtime.device)
            if not row["fits"]:
                break
    finally:
        shutdown(runtime)
    return rows


def describe_row(row: dict) -> str:
    head = f"  {row['model']} batch {row['batch']}: "
    if not row["fits"]:
        return head + "out of memory"
    return (head + f"peak {human(row['peak_reserved'])}, step {row['step_seconds']:.2f} s, "
            f"forward {row['forward_seconds']:.2f} s, load {row['load_seconds']:.2f} s/window")


# ---------------------------------------------------------------------------
# probe: the launcher's numbers
# ---------------------------------------------------------------------------
def recommend(rows: list[dict], node: dict) -> dict:
    """The launcher's flags for one model on ``node``, and the hours they cost.

    The card takes the largest measured batch, no larger than the recipe's
    global batch, whose step stays under ``headroom`` of the card; the
    recipe's batch is then made up of cards, and what cards are left run
    folds side by side. The workers are what the card can consume, capped by
    the cores each process has.
    """
    model = rows[0]["model"]
    limit = node["headroom"] * rows[0]["card_total"]
    fitting = [r for r in rows if r["fits"] and r["batch"] <= node["batch"]
               and r["peak_reserved"] + CUDA_CONTEXT_BYTES <= limit]
    if not fitting:
        return {"model": model, "fits": False}
    chosen = max(fitting, key=lambda r: r["batch"])
    batch = chosen["batch"]
    nproc = min(math.ceil(node["batch"] / batch), node["gpus"])
    parallel = max(node["gpus"] // nproc, 1)
    cores = node["cpus"] // (parallel * nproc)
    load = statistics.median(r["load_seconds"] for r in rows)

    card_rate = batch / chosen["step_seconds"]            # windows a second, one card
    wanted = math.ceil(1.2 * card_rate * load)
    workers = max(min(wanted, cores - 1), 1)
    rate = min(card_rate, workers / load)
    test_rate = min(batch / chosen["forward_seconds"], workers / load)
    epoch_seconds = (node["train_windows"] / (rate * nproc)
                     + node["test_windows"] / (test_rate * nproc))
    fold_hours = node["epochs"] * epoch_seconds / 3600
    return {
        "model": model, "fits": True, "precision": chosen["precision"],
        "--batch-size": batch, "--parallel": parallel, "--nproc-per-node": nproc,
        "--num-workers": workers, "global batch": batch * nproc,
        "card GB": (chosen["peak_reserved"] + CUDA_CONTEXT_BYTES) / 2 ** 30,
        "card %": 100 * (chosen["peak_reserved"] + CUDA_CONTEXT_BYTES) / chosen["card_total"],
        "largest batch": max(r["batch"] for r in rows if r["fits"]),
        "paced by": "cards" if card_rate <= workers / load else "loaders",
        "workers wanted": wanted,
        "epoch min": epoch_seconds / 60, "fold h": fold_hours,
        "sweep h": node["folds"] * fold_hours / parallel,
    }


def probe(args) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise SystemExit("probe measures a CUDA card and none is visible")
    configs = load_dataset_configs(args.datasets)
    stores = load_stores(configs)
    files = config_files(args.configs)
    interface = load_setup(files[0], args.resize)[0]
    folds = args.folds or len(participants(stores, args.datasets[0]))
    windows = sum(len(WindowedDataset(name, kept, interface, mode="random"))
                  for name, kept in stores.items() if kept)
    strided = sum(len(WindowedDataset(name, kept, interface, mode="strided"))
                  for name, kept in stores.items() if kept)
    node = {"gpus": args.gpus or torch.cuda.device_count(),
            "cpus": args.cpus or usable_cpus(), "batch": args.batch,
            "headroom": args.headroom, "epochs": args.epochs, "folds": folds,
            "train_windows": windows * (folds - 1) / folds,
            "test_windows": strided / folds}
    print(f"this process may run on {usable_cpus()} core(s); "
          f"{torch.cuda.device_count()} card(s) visible")
    print(f"planning for {node['gpus']} card(s), {node['cpus']} core(s), global batch "
          f"{node['batch']}, {node['epochs']} epochs, {folds} folds of about "
          f"{node['train_windows']:.0f} training and {node['test_windows']:.0f} test "
          f"windows of {interface.RESIZE.H}x{interface.RESIZE.W} frames "
          f"(from {files[0].name})")

    measured, plans = [], []
    for path in files:
        print(path.name, flush=True)
        rows = probe_config(path, stores, args)
        measured += rows
        plans.append(recommend(rows, node))
    plan = pd.DataFrame(plans)
    print("\nthe launcher's numbers, fastest sweep first:")
    print(plan.sort_values("sweep h", na_position="last").to_string(
        index=False, float_format=lambda v: f"{v:.3g}"))
    if args.out:
        out = Path(args.out)
        pd.DataFrame(measured).to_csv(out.with_name(out.stem + "_measured.csv"), index=False)
        plan.to_csv(out, index=False)
        print(f"written: {out} and {out.with_name(out.stem + '_measured.csv')}")
    return plan


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure what a sweep needs to be tuned to a node.")
    commands = parser.add_subparsers(dest="command", required=True)

    c = commands.add_parser("convergence", help="how many epochs finished runs needed")
    c.add_argument("runs", nargs="+", metavar="PATH",
                   help="run directories, or folders holding them")
    c.add_argument("--at", nargs="+", type=int, default=[10, 20], metavar="EPOCH",
                   help="the epochs to read off beside the last (default: 10 20)")
    c.set_defaults(call=convergence)

    p = commands.add_parser("probe", help="what one step of each config costs here")
    p.add_argument("--datasets", nargs="+", required=True, metavar="NAME",
                   help="dataset config name(s); folds are the first one's participants")
    p.add_argument("--configs", nargs="+", required=True, metavar="PATH",
                   help="config files, or a directory of them")
    p.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"),
                   help="frame size to measure at (default: each config's own)")
    p.add_argument("--gpus", type=int, metavar="N",
                   help="cards the sweep's node has (default: the cards visible)")
    p.add_argument("--cpus", type=int, metavar="N",
                   help="cores the sweep's node has (default: the cores usable here)")
    p.add_argument("--batch", type=int, default=4, metavar="N",
                   help="the recipe's global batch (default: 4)")
    p.add_argument("--epochs", type=int, default=30, metavar="N",
                   help="epochs a fold trains for (default: 30)")
    p.add_argument("--folds", type=int, metavar="N",
                   help="folds in the sweep (default: the participants admitted)")
    p.add_argument("--headroom", type=float, default=0.9, metavar="FRACTION",
                   help="share of a card a step may take (default: 0.9)")
    p.add_argument("--max-batch", type=int, default=BATCH_SIZES[-1], metavar="N",
                   help=f"largest batch to try (default: {BATCH_SIZES[-1]})")
    p.add_argument("--steps", type=int, default=3, metavar="N",
                   help="timed steps per batch size, after one warm-up (default: 3)")
    p.add_argument("--out", metavar="PATH", help="write the plan to this CSV, and "
                   "the measurements beside it as <name>_measured.csv")
    p.set_defaults(call=probe)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.call(args)
    except ValueError as err:          # ConfigError is a ValueError
        raise SystemExit(str(err))


if __name__ == "__main__":
    main()
