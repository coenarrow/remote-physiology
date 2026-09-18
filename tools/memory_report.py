"""What one training run of a setup costs in memory, without running it.

Takes the same config files as ``scripts/run.py`` (the datasets and the one
model/interface/training file), builds the model and optimiser exactly as the
trainer does, draws one real batch of windows from the loaded stores, takes
exactly one training step on it, and reports:

* the static footprint that follows from the parameter count — weights,
  gradients and optimiser state, in the recipe's precision;
* the bytes one collated batch takes;
* on a GPU, the peak the CUDA allocator reached during that step, against the
  card's total and free memory, and how many runs of this size fit side by
  side — the number a SLURM ``--mem``/``--gres`` line or a parallel sweep on
  the dev box needs.

Activation memory is the term nobody can compute by hand for a video model,
which is why this measures a step rather than estimating one. The step is a
real one on a real batch: it needs the cache mounted, and it needs the GPU
the run would use (an ``salloc`` on the target partition, or the dev box).

    uv run tools/memory_report.py --datasets pure \\
        --config configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml \\
        --batch-size 4
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                # noqa: E402
from src.dataset_config import load_dataset_configs          # noqa: E402
from src.datasets import load_stores                         # noqa: E402
from src.distributed import init_runtime, shutdown           # noqa: E402
from src.experiment import (                      # noqa: E402
    add_config_arguments, add_run_arguments, run_settings,
)
from src.memory import (                          # noqa: E402
    device_memory, human, peak_memory, reset_peak, tensor_bytes,
)
from src.model_config import load_config          # noqa: E402
from src.models import build_model                # noqa: E402
from src.trainer import Trainer                   # noqa: E402
from src.inputs import WindowedDataset            # noqa: E402

#: A CUDA context is created per process outside the allocator's books, so
#: neither ``max_memory_reserved`` nor a warm ``mem_get_info`` delta sees it.
#: This is a working allowance for the driver, kernels and cuBLAS workspace.
CUDA_CONTEXT_BYTES = 500 * 2 ** 20


def one_batch(datasets: dict, stores: dict, interface, batch_size: int):
    """One collated batch of random training windows over every loaded
    dataset that admitted a store, drawn in-process (no loader workers)."""
    parts = [WindowedDataset(name, kept, interface, mode="random")
             for name, kept in stores.items() if kept]
    if not parts:
        raise ConfigError("no dataset admitted a store; nothing to draw a batch from")
    dataset = ConcatDataset(parts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return next(iter(loader)), len(dataset)


def measure(device: torch.device, probe) -> dict | None:
    """Run ``probe()`` and return what the device's allocator saw, or None on
    a device with nothing to read (CPU)."""
    memory = device_memory(device)
    reset_peak(device)
    probe()
    peak = peak_memory(device)
    if memory is None:
        return None
    return {
        "name": memory["name"],
        "peak_allocated": peak["allocated"],
        "peak_reserved": peak["reserved"],
        "context": CUDA_CONTEXT_BYTES if device.type == "cuda" else 0,
        "total": memory["total"],
        "free_before": memory["free"],
    }


def report(args) -> dict:
    run = args.run
    interface, model_config, training = load_config(args.config)
    runtime = init_runtime(training, run.gpu)
    configs = load_dataset_configs(args.datasets)
    stores = load_stores(configs)
    batch, n_windows = one_batch(configs, stores, interface, run.batch_size)
    model = build_model(model_config, interface)
    n_params = sum(p.numel() for p in model.parameters())

    # The trainer builds the model on the device and the optimiser over it
    # exactly as a run does; the run directory is never touched by one step.
    state = {}

    def probe():
        trainer = Trainer(model, interface, model_config, training, run, runtime,
                          Path(tempfile.gettempdir()) / "memory_report")
        state["loss"], _ = trainer.step(batch)
        state["optimiser"] = trainer.optimizer

    try:
        measured = measure(runtime.device, probe)
    finally:
        shutdown(runtime)

    optimiser_state = sum(tensor_bytes(s) for s in state["optimiser"].state.values())
    return {
        "model": model_config.NAME,
        "traces": list(model.traces),
        "in_channels": model.in_channels,
        "parameters": n_params,
        "precision": runtime.precision,
        "optimiser": training.OPTIMIZER,
        "batch_size": run.batch_size,
        "batch_windows": next(iter(batch["label_mask"].values())).shape[0],
        "train_windows": n_windows,
        "num_workers": run.num_workers,
        "window_frames": interface.window_frames,
        "frame_hw": _frame_hw(batch),
        "bytes": {
            "weights": tensor_bytes(list(model.parameters())),
            "gradients": sum(tensor_bytes(p.grad) for p in model.parameters()),
            "optimiser_state": optimiser_state,
            "batch": tensor_bytes(batch),
        },
        "device": {"type": runtime.device.type, "index": runtime.device.index,
                   "measured": measured},
        "loss": state["loss"],
    }


def _frame_hw(batch) -> tuple[int, int]:
    plane = next(iter(batch["frames"].values()))
    return tuple(plane.shape[-2:])


def print_report(r: dict) -> None:
    b, dev = r["bytes"], r["device"]
    h, w = r["frame_hw"]
    print(f"model: {r['model']} x {len(r['traces'])} traces, {r['in_channels']} input "
          f"channels, {r['parameters']:,} parameters")
    print(f"  weights          {human(b['weights']):>10}   {r['precision']} master weights")
    print(f"  gradients        {human(b['gradients']):>10}")
    print(f"  optimiser state  {human(b['optimiser_state']):>10}   {r['optimiser']}")
    print(f"  one batch        {human(b['batch']):>10}   {r['batch_windows']} windows x "
          f"{r['in_channels']} ch x {r['window_frames']} x {h} x {w} frames + labels")
    if r["batch_windows"] < r["batch_size"]:
        print(f"  NOTE: only {r['train_windows']} training windows, so this batch is "
              f"smaller than --batch-size {r['batch_size']}; the step below understates "
              f"a full one")
    static = b["weights"] + b["gradients"] + b["optimiser_state"]
    print(f"  static total     {human(static):>10}   weights + gradients + optimiser state")

    where = f"{dev['type']}" + (f":{dev['index']}" if dev["index"] is not None else "")
    m = dev["measured"]
    if m is None:
        print(f"one training step at {r['precision']} on {where}: loss {r['loss']:.4g}; "
              f"no device memory measurement on {dev['type']}, the static footprint "
              f"above is what the parameters alone cost")
        return
    per_run = m["peak_reserved"] + m["context"]
    fit = math.floor(m["free_before"] / per_run) if per_run else 0
    print(f"one training step at {r['precision']} on {where} ({m['name']}): "
          f"loss {r['loss']:.4g}")
    print(f"  peak allocated   {human(m['peak_allocated']):>10}   tensors live at once "
          f"(activations are the gap above the static total)")
    print(f"  peak reserved    {human(m['peak_reserved']):>10}   the allocator's footprint "
          f"for this process")
    if m["context"]:
        print(f"  + context        {human(m['context']):>10}   CUDA context allowance, "
              f"not measured")
    print(f"  = per run        {human(per_run):>10}   {100 * per_run / m['total']:.0f}% of "
          f"{human(m['total'])}; {human(m['free_before'])} was free before this probe")
    print(f"  side by side     {fit:>10}   run(s) of this size fit in that free memory")
    in_flight = (2 * r["num_workers"] + 1) * b["batch"]
    print(f"host: DataLoader keeps ~{2 * r['num_workers'] + 1} batches in flight "
          f"({human(in_flight)}) plus one Python process per worker "
          f"(--num-workers {r['num_workers']})")


def main(argv=None) -> dict:
    parser = add_run_arguments(add_config_arguments(argparse.ArgumentParser(
        description="Measure what one training run of this setup takes in "
                    "memory — one real step on one real batch — without running it.")))
    args = parser.parse_args(argv)
    args.run = run_settings(parser, args)
    try:
        r = report(args)
    except ValueError as err:          # ConfigError is a ValueError
        parser.error(str(err))
    print_report(r)
    return r


if __name__ == "__main__":
    main()
