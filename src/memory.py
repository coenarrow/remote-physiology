"""What a run costs in device memory, read off the allocator.

Written once for the two places that report it: the trainer prints the
device's memory before the first epoch and the peak the first training step
reached, and ``tools/memory_report.py`` measures one step without running a
fold. CUDA and Apple MPS have allocators that can be read; a CPU device has
nothing to read and every reader returns ``None`` for it.
"""

import torch


def tensor_bytes(obj) -> int:
    """Bytes of every tensor in a nested dict/list/tuple of tensors."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def human(n_bytes: float) -> str:
    """``n_bytes`` in the largest binary unit that keeps the number readable."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{n_bytes:.0f} B"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def device_memory(device: torch.device) -> dict | None:
    """``{name, free, total}`` bytes of the device right now, or ``None`` on a
    device with nothing to read (CPU). On MPS the memory is unified, so
    ``free`` is the recommended working set rather than a separate figure."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return {"name": torch.cuda.get_device_name(device), "free": free, "total": total}
    if device.type == "mps":
        total = torch.mps.recommended_max_memory()
        return {"name": "Apple MPS", "free": total, "total": total}
    return None


def reset_peak(device: torch.device) -> None:
    """Start counting the allocator's peak from now."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device: torch.device) -> dict | None:
    """``{allocated, reserved}`` peak bytes since :func:`reset_peak` — tensors
    live at once, and the allocator's footprint for this process — after
    waiting for the device's queued work; ``None`` on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return {"allocated": torch.cuda.max_memory_allocated(device),
                "reserved": torch.cuda.max_memory_reserved(device)}
    if device.type == "mps":
        torch.mps.synchronize()
        return {"allocated": torch.mps.current_allocated_memory(),
                "reserved": torch.mps.driver_allocated_memory()}
    return None


def describe_device(memory: dict) -> str:
    """One line: the device, what is free of what."""
    return (f"{memory['name']}, {human(memory['free'])} free of "
            f"{human(memory['total'])}")


def describe_peak(peak: dict, memory: dict) -> str:
    """One line: the peak a step reached, against the device."""
    return (f"peak {human(peak['reserved'])} reserved for this process "
            f"({100 * peak['reserved'] / memory['total']:.0f}% of "
            f"{human(memory['total'])}; {human(peak['allocated'])} in tensors at once)")
