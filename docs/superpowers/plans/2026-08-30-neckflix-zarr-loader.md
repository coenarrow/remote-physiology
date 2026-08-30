# Neckflix Zarr Loader Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the HDF5-based `NeckflixLoader` with a lazy zarr-backed dataset stack ported from CardioHydra: `BaseZarrDataset` + `NeckflixDataset` + `label_transforms`, fully unit-tested against synthetic zarr stores.

**Architecture:** A plain-dict-configured, metadata-only-construction `torch.utils.data.Dataset` that scans `*.zarr` stores produced by the Neckflix preprocessor (`ghcr.io/coenarrow/neckflix` ≥1.0.0), gates them on `complete`/`tool_version` root attrs, filters samples by generic attribute include/exclude (the LOSO mechanism), builds a strided-or-random window index, and slices frames/traces lazily per `__getitem__`, emitting the CardioHydra nested-dict batch contract. The old loader, its collate helper, and its tests are deleted; a two-line `main.py` edit keeps the other datasets working.

**Tech Stack:** Python ≥3.13, PyTorch, zarr-python ≥3.3 <4, numpy 2.x, pytest ≥8.4.2, `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md` — read it before starting any task. Where this plan and the spec disagree, the spec wins. The port reference is `/Users/20759193/repos/CardioHydra/src/dataset/base.py`, `src/dataset/neckflix.py`, `src/transforms/labels.py` — parity with it is the rule; the spec's deviations 1–7 are the only sanctioned differences.

## Global Constraints

- Dependency: `zarr>=3.3,<4` in `pyproject.toml` ONLY. Never touch `requirements.txt` (stale; pins numpy 1.24.3).
- All commands run from the repo root `/Users/20759193/repos/rPPG-Toolbox` with `uv run ...`. Tests: `uv run python -m pytest ...` (the `python -m` form puts the repo root on `sys.path`; bare `pytest` breaks project imports).
- Branch: `multisignal-pilot` (already checked out). Commit at the end of every task; end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- The loader takes a **plain dict** cfg — no yacs/CfgNode anywhere in the new files.
- No changes to `neckflix_main.py`, trainers, models, `signals.py`, `SignalDictWrapper`, `MaskedMultiSignalLoss`, docs, or SLURM scripts (all follow-up tasks).
- Frames are emitted as **raw pixel values cast to float32** — no `/255`, no DiffNormalized/Standardized in the loader.
- The full existing test suite must stay green after every task: `uv run python -m pytest tests/ -q`.

---

### Task 1: zarr dependency

**Files:**
- Modify: `pyproject.toml` (dependency list)
- Modify: `uv.lock` (via `uv add`, never by hand)

**Interfaces:**
- Consumes: nothing.
- Produces: importable `zarr` (v3 API: `zarr.open_group`, `Group.create_group`, `Group.create_array(name, data=...)`, `Group.groups()`, `.attrs`) for every later task.

- [ ] **Step 1: Add the dependency**

```bash
uv add "zarr>=3.3,<4"
```

- [ ] **Step 2: Verify the import and version floor**

Run: `uv run python -c "import zarr; print(zarr.__version__)"`
Expected: prints a `3.x` version (≥3.3, <4). If `uv add` changed the pin format in `pyproject.toml`, confirm the line reads exactly `"zarr>=3.3,<4"` and fix it if not, then `uv sync`.

- [ ] **Step 3: Confirm existing suite still green**

Run: `uv run python -m pytest tests/ -q`
Expected: all pass (same count as before this task).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add zarr>=3.3,<4 for the Neckflix zarr loader

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: label_transforms — verbatim port

**Files:**
- Create: `dataset/data_loader/label_transforms.py`
- Test: `tests/test_label_transforms.py`

**Interfaces:**
- Consumes: nothing project-side (torch only).
- Produces (used by Tasks 3, 10, 11):
  - `STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max")`
  - `EPS = 1e-8`
  - `zscore(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]` — trace `(T,)`; returns normed `(T,)` + stats of 0-dim tensors in physical units
  - `minmax(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]` — normed to `[0, 1]`
  - `zscore_inverse(sig: Tensor, stats: dict) -> Tensor` / `minmax_inverse(sig, stats)` — accept `(T,)` sig with 0-dim stats, or `(B, T)` sig with `(B,)` stats

- [ ] **Step 1: Write the failing tests**

Create `tests/test_label_transforms.py`:

```python
import pytest
import torch

from dataset.data_loader.label_transforms import (
    EPS,
    STAT_NAMES,
    minmax,
    minmax_inverse,
    zscore,
    zscore_inverse,
)


def _trace():
    torch.manual_seed(0)
    return 80.0 + 15.0 * torch.randn(64)


def test_stat_names_canonical():
    assert STAT_NAMES == ("mean", "std", "min", "max")


def test_zscore_stats_values_and_unbiased_std():
    t = _trace()
    _, stats = zscore(t)
    assert set(stats) == set(STAT_NAMES)
    assert all(v.dim() == 0 for v in stats.values())
    assert torch.equal(stats["mean"], t.mean())
    assert torch.equal(stats["std"], t.std(correction=1))
    assert torch.equal(stats["min"], t.amin())
    assert torch.equal(stats["max"], t.amax())


def test_zscore_round_trip_exact():
    t = _trace()
    normed, stats = zscore(t)
    # float32 at ~80-magnitude: a few ulps of slack; exactness in the sub-EPS
    # band has its own dedicated test below.
    assert torch.allclose(zscore_inverse(normed, stats), t, atol=1e-3)


def test_minmax_range_and_round_trip():
    t = _trace()
    normed, stats = minmax(t)
    assert normed.min().item() == pytest.approx(0.0)
    assert normed.max().item() == pytest.approx(1.0)
    assert torch.allclose(minmax_inverse(normed, stats), t, atol=1e-3)


def test_constant_trace_never_nan():
    t = torch.full((16,), 7.0)
    for fn in (zscore, minmax):
        normed, _ = fn(t)
        assert torch.isfinite(normed).all()
        assert torch.equal(normed, torch.zeros(16))


def test_zero_trace_zero_stats():
    # The absent-label invariant: zero-filled traces emit zeros with zero stats.
    t = torch.zeros(16)
    for fn in (zscore, minmax):
        normed, stats = fn(t)
        assert torch.equal(normed, torch.zeros(16))
        assert all(v.item() == 0.0 for v in stats.values())


def test_inverse_broadcasts_collated_batch():
    t = _trace()
    normed, stats = zscore(t)
    sig_b = torch.stack([normed, normed])                    # (2, T)
    stats_b = {k: torch.stack([v, v]) for k, v in stats.items()}  # (2,)
    out = zscore_inverse(sig_b, stats_b)
    assert out.shape == (2, t.shape[0])
    assert torch.allclose(out[0], t, atol=1e-3)


def test_round_trip_exact_in_sub_eps_band():
    # Forward and inverse must clamp identically for 0 < std < EPS.
    t = torch.full((8,), 3.0) + torch.linspace(0, EPS / 10, 8)
    normed, stats = zscore(t)
    assert torch.allclose(zscore_inverse(normed, stats), t, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_label_transforms.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'dataset.data_loader.label_transforms'`

- [ ] **Step 3: Write the implementation**

Create `dataset/data_loader/label_transforms.py` — a verbatim port of CardioHydra `src/transforms/labels.py` (same docstring intent, same formulas, same clamps):

```python
"""Per-window label normalisation and the statistics that generate it.

Ported from CardioHydra src/transforms/labels.py (see the design spec at
docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md). The dataset
normalises each label window at load time and stamps the generating statistics
into the batch, so the model learns waveform shape and absolute scale as
separate problems. Both normalisations share one signature and return the same
four-key stats dict, so the batch layout is identical whichever is used and
either is exactly invertible.
"""

from torch import Tensor

# Canonical stat keys and their order. Consumers iterate or validate against
# this rather than hardcoding key lists.
STAT_NAMES: tuple[str, ...] = ("mean", "std", "min", "max")

# Numerical guard only: a constant trace yields 0/EPS == 0 instead of 0/0 ==
# NaN. That matters because masked losses compute ``values * mask`` and
# NaN * 0 is still NaN, so a NaN could not be masked away after the fact.
EPS = 1e-8


def _stats(trace: Tensor) -> dict[str, Tensor]:
    """Per-window statistics of one ``(T,)`` label trace, as 0-dim tensors."""
    return {
        "mean": trace.mean(),                # ()
        "std": trace.std(correction=1),      # () unbiased
        "min": trace.amin(),                 # ()
        "max": trace.amax(),                 # ()
    }


def _align(stat: Tensor, sig: Tensor) -> Tensor:
    """Right-pad ``stat`` with singleton dims so it broadcasts against ``sig``.

    Handles both the per-sample case (0-dim stat vs ``(T,)`` signal) and the
    collated case (``(B,)`` stat vs ``(B, T)`` signal).
    """
    return stat.reshape(stat.shape + (1,) * (sig.dim() - stat.dim()))


def zscore(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Z-score one ``(T,)`` trace; returns ``(normed, stats)`` in physical units."""
    stats = _stats(trace)                                             # 4 x ()
    normed = (trace - stats["mean"]) / stats["std"].clamp_min(EPS)    # (T,)
    return normed, stats


def zscore_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Map a z-scored signal back to physical units: ``sig * std + mean``."""
    # Must clamp identically to zscore's forward division, or the round-trip
    # is inexact in the 0 < std < EPS band.
    return sig * _align(stats["std"].clamp_min(EPS), sig) + _align(stats["mean"], sig)


def minmax(trace: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Min-max one ``(T,)`` trace to ``[0, 1]``; same stats dict as ``zscore``."""
    stats = _stats(trace)                                                     # 4 x ()
    span = (stats["max"] - stats["min"]).clamp_min(EPS)                       # ()
    normed = (trace - stats["min"]) / span                                    # (T,)
    return normed, stats


def minmax_inverse(sig: Tensor, stats: dict[str, Tensor]) -> Tensor:
    """Map a min-maxed signal back to physical units: ``sig * (max - min) + min``."""
    # Must clamp identically to minmax's forward division (see zscore_inverse).
    span = (_align(stats["max"], sig) - _align(stats["min"], sig)).clamp_min(EPS)
    return sig * span + _align(stats["min"], sig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_label_transforms.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/label_transforms.py tests/test_label_transforms.py
git commit -m "feat: port CardioHydra label transforms (zscore/minmax + exact inverses)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: label_transforms — finite-stats path (deviation 4 groundwork)

**Files:**
- Modify: `dataset/data_loader/label_transforms.py`
- Test: `tests/test_label_transforms.py` (append)

**Interfaces:**
- Consumes: Task 2's `STAT_NAMES`, `EPS`, `zscore`, `minmax`.
- Produces (used by Tasks 10, 11):
  - `finite_stats(trace: Tensor) -> dict[str, Tensor]` — the four stats over finite entries only; all-NaN input ⇒ all-zero stats; a single finite entry ⇒ `std == 0` (not NaN).
  - `apply_norm(trace: Tensor, stats: dict[str, Tensor], mode: str) -> Tensor` — forward zscore/minmax formulas using the *given* stats (no recomputation); `mode` not in `("zscore", "minmax")` ⇒ `ValueError`.
  - Guarantee: on all-finite input with ≥2 elements, `apply_norm(t, finite_stats(t), mode)` is bit-identical to `zscore(t)[0]` / `minmax(t)[0]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_label_transforms.py`:

```python
from dataset.data_loader.label_transforms import apply_norm, finite_stats


def test_finite_stats_all_finite_bit_identical_to_stats():
    t = _trace()
    fs = finite_stats(t)
    _, zs = zscore(t)
    for key in STAT_NAMES:
        assert torch.equal(fs[key], zs[key])


def test_apply_norm_bit_identical_to_verbatim_transforms():
    t = _trace()
    assert torch.equal(apply_norm(t, finite_stats(t), "zscore"), zscore(t)[0])
    assert torch.equal(apply_norm(t, finite_stats(t), "minmax"), minmax(t)[0])


def test_finite_stats_ignores_nans():
    t = _trace()
    dirty = t.clone()
    dirty[3] = float("nan")
    dirty[40] = float("inf")
    keep = torch.ones_like(t, dtype=torch.bool)
    keep[3] = keep[40] = False
    fs = finite_stats(dirty)
    clean = t[keep]
    assert torch.equal(fs["mean"], clean.mean())
    assert torch.equal(fs["std"], clean.std(correction=1))
    assert torch.equal(fs["min"], clean.amin())
    assert torch.equal(fs["max"], clean.amax())


def test_finite_stats_all_nan_gives_zero_stats():
    t = torch.full((8,), float("nan"))
    fs = finite_stats(t)
    assert all(v.item() == 0.0 for v in fs.values())
    assert all(v.dim() == 0 for v in fs.values())


def test_finite_stats_single_finite_entry_std_zero():
    t = torch.full((8,), float("nan"))
    t[2] = 42.0
    fs = finite_stats(t)
    assert fs["mean"].item() == 42.0
    assert fs["std"].item() == 0.0          # not NaN
    assert fs["min"].item() == 42.0
    assert fs["max"].item() == 42.0


def test_apply_norm_rejects_bad_mode():
    t = _trace()
    with pytest.raises(ValueError, match="mode"):
        apply_norm(t, finite_stats(t), "fixed")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_label_transforms.py -v`
Expected: the six new tests FAIL at import (`ImportError: cannot import name 'apply_norm'`); Task 2 tests still pass.

- [ ] **Step 3: Write the implementation**

Append to `dataset/data_loader/label_transforms.py`:

```python
import torch


def finite_stats(trace: Tensor) -> dict[str, Tensor]:
    """``_stats`` over the finite entries of ``trace`` only (deviation 4).

    All-NaN/inf input yields all-zero 0-dim stats (the absent-label
    convention); a single finite entry yields ``std == 0`` rather than the
    NaN that ``std(correction=1)`` would produce. On all-finite input this is
    bit-identical to ``_stats``.
    """
    finite = trace[torch.isfinite(trace)]
    if finite.numel() == 0:
        zero = trace.new_zeros(())
        return {name: zero.clone() for name in STAT_NAMES}
    std = finite.std(correction=1) if finite.numel() > 1 else trace.new_zeros(())
    return {
        "mean": finite.mean(),   # ()
        "std": std,              # ()
        "min": finite.amin(),    # ()
        "max": finite.amax(),    # ()
    }


def apply_norm(trace: Tensor, stats: dict[str, Tensor], mode: str) -> Tensor:
    """Forward zscore/minmax using precomputed ``stats`` — no recomputation.

    The dataset normalises with finite-only stats and emits those same stats,
    so the inverse round-trip is exact at finite positions (deviation 4).
    """
    if mode == "zscore":
        return (trace - stats["mean"]) / stats["std"].clamp_min(EPS)
    if mode == "minmax":
        span = (stats["max"] - stats["min"]).clamp_min(EPS)
        return (trace - stats["min"]) / span
    raise ValueError(f"mode must be 'zscore' or 'minmax', got {mode!r}")
```

(Note: `import torch` goes at the top of the file with the existing `from torch import Tensor`, not mid-file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_label_transforms.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/label_transforms.py tests/test_label_transforms.py
git commit -m "feat: finite-only stats + apply_norm for the interior-NaN guard

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: synthetic zarr store fixtures

**Files:**
- Create: `tests/zarr_fixtures.py`
- Test: `tests/test_zarr_fixtures.py`

**Interfaces:**
- Consumes: `zarr` (Task 1).
- Produces (used by every later task's tests):
  - `TOOL_VERSION = "1.0.0"`
  - `make_store(cache_dir, name="P030_S01_R1_0_D", *, attrs=None, perspectives=("1",), streams=("rgb", "ir", "depth"), traces=("abp", "cvp"), num_frames=12, hw=(8, 8), frame_fill=None, trace_values=None, trace_lengths=None, events=False, extra_groups=()) -> Path`
  - `make_unreadable_store(cache_dir, name="P099_S01_R1_0_D") -> Path`
  - `base_cfg(cache_dir, **overrides) -> dict` — a valid loader cfg dict with all keys.

- [ ] **Step 1: Write the failing self-test**

Create `tests/test_zarr_fixtures.py`:

```python
import numpy as np
import zarr

from tests.zarr_fixtures import TOOL_VERSION, base_cfg, make_store


def test_store_schema_matches_preprocessor_output(tmp_path):
    path = make_store(tmp_path, "P030_S01_R1_0_D", num_frames=10, hw=(6, 6))
    root = zarr.open_group(str(path), mode="r")
    attrs = dict(root.attrs)
    assert attrs["participant"] == "030"          # unprefixed (spec: value format)
    assert attrs["posture"] == "0"
    assert attrs["recording"] == "P030_S01_R1_0_D"
    assert attrs["complete"] is True
    assert attrs["tool_version"] == TOOL_VERSION
    rgb = root["1"]["rgb"]["video"]["frames"]
    assert rgb.shape == (3, 10, 6, 6) and rgb.dtype == np.uint8
    ir = root["1"]["ir"]["video"]["frames"]
    assert ir.shape == (1, 10, 6, 6) and ir.dtype == np.uint16
    assert root["1"]["rgb"]["video"].attrs["num_frames"] == 10
    abp = root["1"]["rgb"]["abp"]["data"]
    assert abp.shape == (10,) and abp.dtype == np.float64


def test_attr_override_and_removal(tmp_path):
    path = make_store(tmp_path, "P031_S01_R1_45_D",
                      attrs={"complete": None, "tool_version": "0.9.0"})
    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    assert "complete" not in attrs
    assert attrs["tool_version"] == "0.9.0"
    assert attrs["posture"] == "45"


def test_per_stream_num_frames_and_short_trace(tmp_path):
    path = make_store(tmp_path, "P032_S01_R1_0_D",
                      num_frames={"rgb": 12, "ir": 8, "depth": 12},
                      trace_lengths={("1", "rgb", "abp"): 7})
    root = zarr.open_group(str(path), mode="r")
    assert root["1"]["ir"]["video"].attrs["num_frames"] == 8
    assert root["1"]["rgb"]["abp"]["data"].shape == (7,)
    assert root["1"]["rgb"]["cvp"]["data"].shape == (12,)


def test_events_and_extra_groups(tmp_path):
    path = make_store(tmp_path, "P033_S01_R1_0_D",
                      events=True, extra_groups=(("1", "notes"),))
    root = zarr.open_group(str(path), mode="r")
    assert "events" in root
    assert "video" not in root["events"]
    assert "video" not in root["1"]["notes"]


def test_base_cfg_shape(tmp_path):
    cfg = base_cfg(tmp_path, window_size=8, labels=["ABP"])
    assert cfg["cache_dir"] == str(tmp_path)
    assert cfg["window_size"] == 8
    assert cfg["labels"] == ["ABP"]
    assert cfg["label_norm"] == "zscore"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m pytest tests/test_zarr_fixtures.py -v`
Expected: FAIL at import — no module `tests.zarr_fixtures`.

- [ ] **Step 3: Write the fixture module**

Create `tests/zarr_fixtures.py`:

```python
"""Synthetic Neckflix zarr-v3 stores for loader tests.

Mirrors the store schema written by the Neckflix preprocessor
(ghcr.io/coenarrow/neckflix >= 1.0.0); see the design spec at
docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md.
"""

import numpy as np
import zarr

TOOL_VERSION = "1.0.0"

# stream group -> (channel count, dtype), matching the preprocessor output.
STREAM_SPECS = {
    "rgb": (3, np.uint8),
    "ir": (1, np.uint16),
    "depth": (1, np.uint16),
}

# Distinct, deterministic base offsets so traces are tellable-apart in tests.
TRACE_OFFSETS = {"abp": 100.0, "cvp": 5.0, "ecg": 0.5}


def default_attrs(name):
    """Root attrs derived from a recording name like ``P030_S01_R1_0_D``.

    ``posture`` is the second-to-last underscore token; the trailing token is
    part of the name only and maps to no attr.
    """
    parts = name.split("_")
    return {
        "recording": name,
        "participant": parts[0][1:],   # unprefixed, e.g. "030"
        "session": parts[1],
        "repeat": parts[2],
        "posture": parts[-2],
        "source_resolution": [650, 650],
        "resized_to": None,
        "tool_version": TOOL_VERSION,
        "complete": True,
    }


def make_store(
    cache_dir,
    name="P030_S01_R1_0_D",
    *,
    attrs=None,
    perspectives=("1",),
    streams=("rgb", "ir", "depth"),
    traces=("abp", "cvp"),
    num_frames=12,
    hw=(8, 8),
    frame_fill=None,
    trace_values=None,
    trace_lengths=None,
    events=False,
    extra_groups=(),
):
    """Write one synthetic store under ``cache_dir``; return its path.

    attrs          : dict merged over the defaults; a value of None REMOVES the key.
    num_frames     : int, or {stream: int} for per-stream frame counts.
    frame_fill     : {(persp, stream): int} constant pixel value (default: a
                     deterministic arange pattern).
    trace_values   : {(persp, stream, trace): np.ndarray} explicit trace data.
    trace_lengths  : {(persp, stream, trace): int} truncates that trace copy.
    events         : also write a root-level events/ group (arrays, no video child).
    extra_groups   : iterable of (persp, group_name) video-less groups placed
                     INSIDE an existing perspective.
    """
    path = cache_dir / f"{name}.zarr"
    root = zarr.open_group(str(path), mode="w")
    merged = default_attrs(name)
    for key, value in (attrs or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    root.attrs.update(merged)

    frames_per_stream = (
        dict(num_frames) if isinstance(num_frames, dict)
        else {s: num_frames for s in streams}
    )
    h, w = hw
    for persp in perspectives:
        pgroup = root.create_group(persp)
        for stream in streams:
            n_ch, dtype = STREAM_SPECS[stream]
            t = frames_per_stream.get(stream, 12)
            sgroup = pgroup.create_group(stream)
            video = sgroup.create_group("video")
            fill = (frame_fill or {}).get((persp, stream))
            if fill is None:
                data = (np.arange(n_ch * t * h * w) % 251).reshape(n_ch, t, h, w)
                data = data.astype(dtype)
            else:
                data = np.full((n_ch, t, h, w), fill, dtype=dtype)
            video.create_array("frames", data=data)
            video.create_array(
                "timestamps_us", data=(np.arange(t) * 33_333).astype(np.int64)
            )
            video.attrs.update({"fps": 30.0, "num_frames": int(t)})
            for trace in traces:
                length = (trace_lengths or {}).get((persp, stream, trace), t)
                values = (trace_values or {}).get((persp, stream, trace))
                if values is None:
                    values = TRACE_OFFSETS.get(trace, 1.0) + np.arange(
                        length, dtype=np.float64
                    )
                tgroup = sgroup.create_group(trace)
                tgroup.create_array("data", data=np.asarray(values, dtype=np.float64))

    if events:
        egroup = root.create_group("events")
        egroup.create_array("x", data=np.zeros(4, dtype=np.uint16))
        egroup.create_array("y", data=np.zeros(4, dtype=np.uint16))
        egroup.create_array("p", data=np.zeros(4, dtype=np.int8))
        egroup.create_array("t", data=np.zeros(4, dtype=np.int64))
    for persp, gname in extra_groups:
        root[persp].create_group(gname)
    return path


def make_unreadable_store(cache_dir, name="P099_S01_R1_0_D"):
    """A directory that globs as *.zarr but fails zarr.open_group."""
    path = cache_dir / f"{name}.zarr"
    path.mkdir()
    (path / "zarr.json").write_text("this is not json")
    return path


def base_cfg(cache_dir, **overrides):
    """A complete, valid loader cfg dict; override any key per test."""
    cfg = {
        "cache_dir": str(cache_dir),
        "channels": ["R", "G", "B", "I", "D"],
        "labels": ["ABP", "CVP"],
        "window_size": 4,
        "window_stride": 4,
        "random_windows": False,
        "filters": {},
        "label_norm": "zscore",
        "allow_missing": False,
        "min_channels": 1,
        "min_labels": 1,
    }
    cfg.update(overrides)
    return cfg
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m pytest tests/test_zarr_fixtures.py -v`
Expected: all PASS. If `create_array(name, data=...)` raises a signature error under the installed zarr version, consult `uv run python -c "import zarr, inspect; print(inspect.signature(zarr.Group.create_array))"` and adapt the two call sites in the fixture only (the API is `create_array` in zarr 3 — never `create_dataset`).

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add tests/zarr_fixtures.py tests/test_zarr_fixtures.py
git commit -m "test: synthetic Neckflix zarr store fixtures

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: clean break — skeleton, NeckflixDataset, main.py edit, deletions

**Files:**
- Create: `dataset/data_loader/zarr_dataset.py` (validation skeleton; pipeline arrives in Tasks 6–10)
- Modify: `dataset/data_loader/NeckflixLoader.py` (full rewrite — old content deleted)
- Modify: `main.py:74` and `main.py:131-136`
- Delete: `dataset/data_loader/multisignal_collate.py`, `tests/test_neckflix_dict.py`
- Test: `tests/test_neckflix_zarr.py` (new)

**Interfaces:**
- Consumes: nothing yet (label_transforms imports arrive in Task 10).
- Produces (relied on by all later tasks):
  - `BaseZarrDataset(cfg: dict)` in `dataset.data_loader.zarr_dataset` — ABC with abstract `channel_map` property; `__init__` validates `label_norm`, `filters`, `channels` (in that order) then raises `NotImplementedError` until Task 6.
  - `_validate_filters(filters: dict) -> None` module function — overlap ⇒ `ValueError`.
  - `_resolve_streams(self, channels: list[str]) -> list[tuple[str, int]]` — unknown channel ⇒ `ValueError`.
  - `NeckflixDataset(BaseZarrDataset)` in `dataset.data_loader.NeckflixLoader` with `_CHANNEL_MAP = {"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2), "I": ("ir", 0), "D": ("depth", 0)}` and `channel_map` property returning it.
  - `main.py` imports cleanly with no Neckflix registry entry.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_neckflix_zarr.py`:

```python
import pytest
import torch

from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.zarr_dataset import BaseZarrDataset
from tests.zarr_fixtures import base_cfg, make_store, make_unreadable_store


# --------------------------------------------------------------------------
# Task 5: class shape + config validation
# --------------------------------------------------------------------------
def test_neckflix_channel_map():
    ds = object.__new__(NeckflixDataset)   # property needs no __init__
    assert ds.channel_map == {
        "R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2),
        "I": ("ir", 0), "D": ("depth", 0),
    }


def test_is_torch_dataset_subclass():
    assert issubclass(NeckflixDataset, BaseZarrDataset)
    assert issubclass(BaseZarrDataset, torch.utils.data.Dataset)


def test_bad_label_norm_raises(tmp_path):
    with pytest.raises(ValueError, match="label_norm"):
        NeckflixDataset(base_cfg(tmp_path, label_norm="fixed"))


def test_filter_overlap_raises_upfront_even_with_empty_cache(tmp_path):
    # Deviation 6: validation is unconditional, before any store is scanned.
    cfg = base_cfg(
        tmp_path,
        filters={"participant": {"include": ["030"], "exclude": ["030"]}},
    )
    with pytest.raises(ValueError, match="Overlapping include/exclude"):
        NeckflixDataset(cfg)


def test_unknown_channel_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown channel"):
        NeckflixDataset(base_cfg(tmp_path, channels=["R", "X"]))


def test_main_still_imports_and_neckflix_unregistered():
    import main
    with pytest.raises(ValueError, match="Unsupported dataset"):
        main.get_loader_class("Neckflix")
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: FAIL at import — no module `dataset.data_loader.zarr_dataset` (and `NeckflixDataset` not in `NeckflixLoader`).

- [ ] **Step 3: Create the skeleton `dataset/data_loader/zarr_dataset.py`**

```python
"""Lazy torch datasets over external Neckflix-preprocessor zarr caches.

Ported from CardioHydra src/dataset/base.py; see the design spec at
docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md for the
contract and the sanctioned deviations (1-7). The cache is an external input
produced by ghcr.io/coenarrow/neckflix (>= 1.0.0); this module never writes it.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

# Sentinel for root attrs absent from a store (deviation 2).
_MISSING = object()


def _validate_filters(filters):
    """Reject overlapping include/exclude upfront (deviation 6).

    CardioHydra checks lazily inside the per-sample loop, where an overlap can
    pass silently when no sample reaches that attribute; this port validates
    unconditionally at construction.
    """
    for attribute, spec in (filters or {}).items():
        overlap = set(spec.get("include", [])) & set(spec.get("exclude", []))
        if overlap:
            raise ValueError(
                f"Overlapping include/exclude for '{attribute}': {overlap}"
            )


class BaseZarrDataset(ABC, torch.utils.data.Dataset):
    """Abstract lazy dataset over Neckflix-preprocessor zarr stores.

    Subclasses provide only ``channel_map``. Construction is metadata-only:
    no pixel data is read until ``__getitem__``.
    """

    MIN_TOOL_VERSION = (1, 0, 0)  # raw-frame cache format floor

    @property
    @abstractmethod
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """Map channel names to ``(stream_group, channel_index)`` pairs."""
        ...

    def _resolve_streams(self, channels):
        """Map config channel names through ``channel_map``, preserving order."""
        cmap = self.channel_map
        plan = []
        for ch in channels:
            if ch not in cmap:
                raise ValueError(f"Unknown channel: {ch}. Valid: {list(cmap)}")
            plan.append(cmap[ch])
        return plan

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cache_root = Path(cfg["cache_dir"])
        self.channels = cfg["channels"]
        self.labels = cfg["labels"]
        self.window_size = cfg["window_size"]
        self.window_stride = cfg.get("window_stride", self.window_size)
        self.random_windows = cfg.get("random_windows", False)
        self.filters = cfg.get("filters", {})
        self.label_norm = cfg.get("label_norm", "zscore")
        if self.label_norm not in ("zscore", "minmax"):
            raise ValueError(
                f"label_norm must be 'zscore' or 'minmax', got {self.label_norm!r}"
            )
        _validate_filters(self.filters)

        self.stream_plan = self._resolve_streams(self.channels)
        self.required_streams = sorted({s[0].lower() for s in self.stream_plan})

        self.allow_missing = cfg.get("allow_missing", False)
        self.min_channels = cfg.get("min_channels", 1)
        self.min_labels = cfg.get("min_labels", 1)
        self.present_streams: dict[tuple[str, str], list[str]] = {}
        self.present_labels: dict[tuple[str, str], list[str]] = {}
        self.stream_hw: dict[str, tuple[int, int]] | None = None

        raise NotImplementedError("cache pipeline arrives in Task 6")
```

- [ ] **Step 4: Rewrite `dataset/data_loader/NeckflixLoader.py`**

Replace the ENTIRE file content with:

```python
"""Neckflix dataset over the preprocessor zarr cache.

Provides only the Neckflix channel map; all loading logic lives in
``BaseZarrDataset``. Replaces the retired HDF5-cache loader — see the design
spec at docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md.
"""

from dataset.data_loader.zarr_dataset import BaseZarrDataset


class NeckflixDataset(BaseZarrDataset):
    """Dataset for Neckflix zarr stores."""

    _CHANNEL_MAP = {
        "R": ("rgb", 0),
        "G": ("rgb", 1),
        "B": ("rgb", 2),
        "I": ("ir", 0),
        "D": ("depth", 0),
    }

    @property
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """Neckflix channel name to (stream_group, channel_index) mapping."""
        return self._CHANNEL_MAP
```

- [ ] **Step 5: Edit `main.py` (two locations)**

Remove the registry line (currently line 74):

```python
    "Neckflix": data_loader.NeckflixLoader.NeckflixLoader,
```

Replace in `create_dataset` (currently lines 131-136):

```python
    # Neckflix-specific parameters
    if loader_class == data_loader.NeckflixLoader.NeckflixLoader:
        if test_participants is not None:
            kwargs["test_participants"] = test_participants
        if name == "test":
            kwargs["get_raw_resized"] = get_raw_resized
    return loader_class(**kwargs)
```

with:

```python
    return loader_class(**kwargs)
```

(`create_dataset`'s `test_participants`/`get_raw_resized` parameters stay — callers still pass them.)

- [ ] **Step 6: Delete the dead contract's files**

```bash
git rm dataset/data_loader/multisignal_collate.py tests/test_neckflix_dict.py
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all 6 PASS.

- [ ] **Step 8: Full suite + commit**

Run: `uv run python -m pytest tests/ -q`
Expected: all green (`test_neckflix_dict.py` no longer collected; the other suites — signals, masked loss, deepphys — untouched and passing).

```bash
git add -A
git commit -m "feat!: clean break to zarr-based NeckflixDataset skeleton

Old HDF5 NeckflixLoader (tuple + dict modes), multisignal_collate, and their
tests removed; main.py keeps working for all other datasets ('Neckflix' now
raises Unsupported dataset). neckflix_main.py is broken until the wiring
follow-up, per the approved spec.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `_scan_cache` — store gating

**Files:**
- Modify: `dataset/data_loader/zarr_dataset.py`
- Test: `tests/test_neckflix_zarr.py` (append)

**Interfaces:**
- Consumes: Task 4 fixtures; Task 5 skeleton.
- Produces (used by Tasks 7–11):
  - `self.dataset_dict: dict` — `{recording_stem: {"attrs": {...}, "<persp>": {"<stream>": {entry: {}}}}}` for admitted stores.
  - After this task `__init__` completes on a valid cache; `self.samples == []` and `self.windows == []` (stubs replaced in Tasks 7 and 9); `__len__` returns `len(self.windows)`.
  - Gate behavior per spec: unreadable ⇒ warn+skip; `attrs.get("complete") is not True` ⇒ warn+skip; unparseable/short `tool_version` ⇒ `(0,)` ⇒ warn+skip; missing dir ⇒ `FileNotFoundError`; no admitted store with ≥1 video-bearing perspective ⇒ `RuntimeError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neckflix_zarr.py`:

```python
# --------------------------------------------------------------------------
# Task 6: _scan_cache gating
# --------------------------------------------------------------------------
def test_missing_cache_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cache directory not found"):
        NeckflixDataset(base_cfg(tmp_path / "nope"))


def test_empty_cache_dir_raises_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="No usable zarr stores"):
        NeckflixDataset(base_cfg(tmp_path))


@pytest.mark.parametrize("bad_attrs, warning_match", [
    ({"complete": None}, "complete"),          # attr absent
    ({"complete": False}, "complete"),
    ({"complete": 1}, "complete"),             # identity check: 1 is not True
    ({"tool_version": "0.9.0"}, "tool_version"),
    ({"tool_version": "abc"}, "tool_version"),  # unparseable -> (0,)
    ({"tool_version": None}, "tool_version"),   # absent -> "0" -> (0,)
])
def test_gate_skips_bad_store_with_warning(tmp_path, bad_attrs, warning_match):
    make_store(tmp_path, "P030_S01_R1_0_D", attrs=bad_attrs)
    make_store(tmp_path, "P031_S01_R1_0_D")     # one good store keeps init alive
    with pytest.warns(UserWarning, match=warning_match):
        ds = NeckflixDataset(base_cfg(tmp_path))
    assert list(ds.dataset_dict) == ["P031_S01_R1_0_D"]


def test_unreadable_store_skipped_with_warning(tmp_path):
    make_unreadable_store(tmp_path)
    make_store(tmp_path, "P031_S01_R1_0_D")
    with pytest.warns(UserWarning, match="unreadable"):
        ds = NeckflixDataset(base_cfg(tmp_path))
    assert list(ds.dataset_dict) == ["P031_S01_R1_0_D"]


def test_sole_store_with_only_events_raises(tmp_path):
    # Admitted by the attr gate but contributes no video-bearing perspective.
    make_store(tmp_path, "P030_S01_R1_0_D", streams=(), traces=(), events=True)
    with pytest.raises(RuntimeError, match="No usable zarr stores"):
        NeckflixDataset(base_cfg(tmp_path))


def test_events_and_videoless_groups_ignored(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D",
               events=True, extra_groups=(("1", "notes"),))
    ds = NeckflixDataset(base_cfg(tmp_path))
    rec = ds.dataset_dict["P030_S01_R1_0_D"]
    assert "events" not in rec                       # root-level placement
    assert "notes" not in rec["1"]                   # in-perspective placement
    assert set(rec["1"]) == {"rgb", "ir", "depth"}
    assert set(rec["1"]["rgb"]) == {"video", "abp", "cvp"}


def test_dataset_dict_structure_and_attrs(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", perspectives=("1", "2"))
    ds = NeckflixDataset(base_cfg(tmp_path))
    rec = ds.dataset_dict["P030_S01_R1_0_D"]
    assert rec["attrs"]["participant"] == "030"
    assert set(rec) == {"attrs", "1", "2"}
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: new tests FAIL with `NotImplementedError: cache pipeline arrives in Task 6`; Task 5 tests still pass.

- [ ] **Step 3: Implement `_scan_cache` and complete `__init__`**

In `dataset/data_loader/zarr_dataset.py`, add to the module imports:

```python
import warnings

import zarr
```

Replace the `raise NotImplementedError(...)` line at the end of `__init__` with:

```python
        # Pipeline: scan -> discover -> filter -> window
        self.dataset_dict = self._scan_cache()
        self.samples: list[tuple[str, str]] = []   # populated in Task 7
        self.windows: list[tuple[str, str, int | None]] = []  # populated in Task 9
```

Add these methods to `BaseZarrDataset` (ported verbatim from the reference, `base.py:127-204`):

```python
    def _scan_cache(self) -> dict:
        """Walk external zarr stores in ``cache_root`` into the recording dict.

        Admission gate (see spec): unreadable stores, stores whose root attrs
        lack ``complete is True`` (identity — JSON boolean true only), and
        stores whose ``tool_version`` parses below 1.0.0 (unparseable values
        count as 0) are skipped with a warning. Groups without stream
        sub-groups (root ``events/``) and stream groups without a ``video``
        child are ignored.
        """
        if not self.cache_root.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_root}. The zarr cache "
                "is an external input — generate it with the Neckflix "
                "preprocessor (ghcr.io/coenarrow/neckflix) first."
            )

        cache_dict: dict = {}
        for store_path in sorted(self.cache_root.glob("*.zarr")):
            try:
                root = zarr.open_group(str(store_path), mode="r")
            except Exception as err:
                warnings.warn(
                    f"Skipping {store_path.name}: unreadable store ({err})"
                )
                continue
            attrs = dict(root.attrs)
            if attrs.get("complete") is not True:
                warnings.warn(
                    f"Skipping {store_path.name}: no 'complete: true' root attr "
                    "(partial run or pre-Neckflix store); regenerate it."
                )
                continue
            version = str(attrs.get("tool_version", "0"))
            try:
                version_tuple = tuple(int(p) for p in version.split("."))
            except ValueError:
                version_tuple = (0,)
            if version_tuple < self.MIN_TOOL_VERSION:
                warnings.warn(
                    f"Skipping {store_path.name}: tool_version {version!r} predates "
                    "the raw-frame format (needs >= 1.0.0); regenerate it."
                )
                continue

            recording: dict = {"attrs": attrs}
            for perspective_name, perspective_group in root.groups():
                perspective: dict = {}
                for stream_name, stream_group in perspective_group.groups():
                    entries = {name for name, _ in stream_group.groups()}
                    if "video" not in entries:
                        continue  # non-stream group, e.g. a bare trace group
                    perspective[stream_name] = {name: {} for name in entries}
                if perspective:
                    recording[perspective_name] = perspective
            cache_dict[store_path.stem] = recording

        if not any(len(rec) > 1 for rec in cache_dict.values()):
            raise RuntimeError(
                f"No usable zarr stores under {self.cache_root} — every store "
                "was missing, incomplete, or pre-1.0.0. Regenerate the cache "
                "with the Neckflix preprocessor (ghcr.io/coenarrow/neckflix)."
            )
        return cache_dict

    def __len__(self) -> int:
        return len(self.windows)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/zarr_dataset.py tests/test_neckflix_zarr.py
git commit -m "feat: zarr cache scan with complete/tool_version gating

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `discover_samples`

**Files:**
- Modify: `dataset/data_loader/zarr_dataset.py`
- Test: `tests/test_neckflix_zarr.py` (append)

**Interfaces:**
- Consumes: `self.dataset_dict` (Task 6), `self.required_streams`, `self.labels`, `self.allow_missing`, `self.min_channels`, `self.min_labels`.
- Produces (used by Tasks 8–11):
  - `discover_samples(self) -> list[tuple[str, str]]` — sorted `(recording, perspective)` pairs; populates `self.present_streams[(rec, persp)]: list[str]` (canonical required-stream order) and `self.present_labels[(rec, persp)]: list[str]` (canonical config-label order).
  - `__init__` now sets `self.samples = self.discover_samples()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neckflix_zarr.py`:

```python
# --------------------------------------------------------------------------
# Task 7: discover_samples
# --------------------------------------------------------------------------
def test_strict_mode_requires_all_streams_and_labels(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D")                       # full
    make_store(tmp_path, "P031_S01_R1_0_D", streams=("rgb",))     # missing ir/depth
    make_store(tmp_path, "P032_S01_R1_0_D", traces=("abp",))      # missing cvp
    ds = NeckflixDataset(base_cfg(tmp_path))
    assert ds.samples == [("P030_S01_R1_0_D", "1")]


def test_strict_mode_every_stream_must_carry_every_label(tmp_path):
    # The fixture always writes symmetric traces, so build the asymmetric case
    # by appending an ir stream that carries only abp (no cvp) to a store.
    import numpy as _np
    import zarr as _z
    make_store(tmp_path, "P030_S01_R1_0_D")
    path = make_store(tmp_path, "P031_S01_R1_0_D", streams=("rgb", "depth"))
    root = _z.open_group(str(path), mode="a")
    ir = root["1"].create_group("ir")
    video = ir.create_group("video")
    video.create_array("frames", data=_np.zeros((1, 12, 8, 8), dtype=_np.uint16))
    video.attrs.update({"fps": 30.0, "num_frames": 12})
    ir.create_group("abp").create_array(
        "data", data=_np.arange(12, dtype=_np.float64))
    # Strict mode drops P031: its ir stream lacks cvp.
    ds = NeckflixDataset(base_cfg(tmp_path))
    assert ds.samples == [("P030_S01_R1_0_D", "1")]


def test_allow_missing_thresholds(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", streams=("rgb",), traces=("abp",))
    cfg = base_cfg(tmp_path, allow_missing=True, min_channels=1, min_labels=1)
    ds = NeckflixDataset(cfg)
    assert ds.samples == [("P030_S01_R1_0_D", "1")]
    assert ds.present_streams[("P030_S01_R1_0_D", "1")] == ["rgb"]
    assert ds.present_labels[("P030_S01_R1_0_D", "1")] == ["ABP"]

    # Raising min_channels above what the store offers empties the dataset
    # (zero samples is legal — the RuntimeError gate is about stores, not samples).
    cfg2 = base_cfg(tmp_path, allow_missing=True, min_channels=2)
    ds2 = NeckflixDataset(cfg2)
    assert ds2.samples == []


def test_samples_sorted_across_recordings_and_perspectives(tmp_path):
    make_store(tmp_path, "P031_S01_R1_0_D", perspectives=("2", "1"))
    make_store(tmp_path, "P030_S01_R1_0_D")
    ds = NeckflixDataset(base_cfg(tmp_path))
    assert ds.samples == [
        ("P030_S01_R1_0_D", "1"),
        ("P031_S01_R1_0_D", "1"),
        ("P031_S01_R1_0_D", "2"),
    ]
```

(Note the second test builds an asymmetric store by appending an `ir` group carrying only `abp` to a fixture store — the fixture itself always writes symmetric traces.)

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: new tests FAIL — `ds.samples == []` everywhere (stub from Task 6).

- [ ] **Step 3: Implement**

Add to `BaseZarrDataset` (verbatim port of `base.py:72-123`):

```python
    def discover_samples(self) -> list[tuple[str, str]]:
        """Build the sample list, retaining partial samples when allowed.

        Records, for every admitted ``(recording, perspective)``, the streams
        and labels actually present (canonical order) in
        ``self.present_streams`` / ``self.present_labels``. With
        ``allow_missing`` a sample is kept when it has >= ``min_channels``
        present streams and >= ``min_labels`` present labels; otherwise the
        strict all-present rule applies.
        """
        samples: list[tuple[str, str]] = []
        for rec_name, rec_data in sorted(self.dataset_dict.items()):
            for perspective, perspective_data in sorted(rec_data.items()):
                if perspective == "attrs":
                    continue

                streams = perspective_data.keys()
                present_streams = [s for s in self.required_streams if s in streams]
                present_labels = [
                    lab for lab in self.labels
                    if any(lab.lower() in perspective_data[s] for s in present_streams)
                ]

                if self.allow_missing:
                    keep = (len(present_streams) >= self.min_channels
                            and len(present_labels) >= self.min_labels)
                else:
                    # Strict: every required stream present AND every stream in
                    # the perspective carries every configured label.
                    keep = (
                        len(present_streams) == len(self.required_streams)
                        and all(
                            all(lab.lower() in perspective_data[s] for lab in self.labels)
                            for s in streams
                        )
                    )
                if not keep:
                    continue

                self.present_streams[(rec_name, perspective)] = present_streams
                self.present_labels[(rec_name, perspective)] = present_labels
                samples.append((rec_name, perspective))
        return samples
```

In `__init__`, replace:

```python
        self.samples: list[tuple[str, str]] = []   # populated in Task 7
```

with:

```python
        self.samples = self.discover_samples()
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all PASS. (Clean up the second test's scaffolding if it needed adjustment — the assertion set is what matters.)

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/zarr_dataset.py tests/test_neckflix_zarr.py
git commit -m "feat: sample discovery with strict and allow_missing modes

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: attribute filters + `attribute_values`

**Files:**
- Modify: `dataset/data_loader/zarr_dataset.py`
- Test: `tests/test_neckflix_zarr.py` (append)

**Interfaces:**
- Consumes: `self.samples`, `self.dataset_dict` (Tasks 6–7), `_MISSING` (Task 5).
- Produces (used by Tasks 9–11 and future wiring):
  - `_filter_by_attribute(self, filters: dict | None = None) -> None` — mutates `self.samples` in place; `"perspective"` pseudo-attribute compared as `str()`-coerced values against the sample's perspective key (deviation 3); missing root attr ⇒ fails non-empty include, passes exclude-only, and after the pass one `UserWarning` per affected attribute listing store names (deviation 2).
  - `attribute_values(self, attribute: str) -> list[str]` — sorted, `str()`-coerced unique values over the current (filtered) `self.samples`; samples missing the attr silently skipped (deviation 5).
  - `__init__` now calls `self._filter_by_attribute(self.filters)` after `discover_samples()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neckflix_zarr.py`:

```python
import warnings as _warnings

# --------------------------------------------------------------------------
# Task 8: attribute filters + attribute_values
# --------------------------------------------------------------------------
def _three_participant_cache(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D")
    make_store(tmp_path, "P031_S01_R1_45_D")
    make_store(tmp_path, "P032_S01_R1_90_D", perspectives=("1", "2"))


def test_include_filter(tmp_path):
    _three_participant_cache(tmp_path)
    cfg = base_cfg(tmp_path, filters={"participant": {"include": ["031"], "exclude": []}})
    ds = NeckflixDataset(cfg)
    assert ds.samples == [("P031_S01_R1_45_D", "1")]


def test_exclude_filter(tmp_path):
    _three_participant_cache(tmp_path)
    cfg = base_cfg(tmp_path, filters={"participant": {"include": [], "exclude": ["031"]}})
    ds = NeckflixDataset(cfg)
    assert {rec for rec, _ in ds.samples} == {"P030_S01_R1_0_D", "P032_S01_R1_90_D"}


def test_loso_split_is_disjoint(tmp_path):
    _three_participant_cache(tmp_path)
    fold = "032"
    train = NeckflixDataset(base_cfg(
        tmp_path, filters={"participant": {"include": [], "exclude": [fold]}}))
    test = NeckflixDataset(base_cfg(
        tmp_path, filters={"participant": {"include": [fold], "exclude": []}}))
    train_p = {train.dataset_dict[r]["attrs"]["participant"] for r, _ in train.samples}
    test_p = {test.dataset_dict[r]["attrs"]["participant"] for r, _ in test.samples}
    assert fold not in train_p and test_p == {fold}


def test_posture_filter(tmp_path):
    _three_participant_cache(tmp_path)
    cfg = base_cfg(tmp_path, filters={"posture": {"include": ["0", "45"], "exclude": []}})
    ds = NeckflixDataset(cfg)
    assert {rec for rec, _ in ds.samples} == {"P030_S01_R1_0_D", "P031_S01_R1_45_D"}


def test_perspective_filter_with_int_coercion(tmp_path):
    _three_participant_cache(tmp_path)
    cfg = base_cfg(tmp_path, filters={"perspective": {"include": [2], "exclude": []}})
    ds = NeckflixDataset(cfg)   # YAML-natural int 2 must match key "2"
    assert ds.samples == [("P032_S01_R1_90_D", "2")]


def test_missing_attr_fails_include_passes_exclude_and_warns(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", attrs={"posture": None})
    make_store(tmp_path, "P031_S01_R1_45_D")
    # include: sample without the attr is dropped (membership unprovable)
    with pytest.warns(UserWarning, match="posture.*P030_S01_R1_0_D"):
        ds = NeckflixDataset(base_cfg(
            tmp_path, filters={"posture": {"include": ["0", "45"], "exclude": []}}))
    assert ds.samples == [("P031_S01_R1_45_D", "1")]
    # exclude-only: sample without the attr passes
    with pytest.warns(UserWarning, match="posture"):
        ds2 = NeckflixDataset(base_cfg(
            tmp_path, filters={"posture": {"include": [], "exclude": ["45"]}}))
    assert ds2.samples == [("P030_S01_R1_0_D", "1")]


def test_no_warning_when_attrs_complete(tmp_path):
    _three_participant_cache(tmp_path)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        NeckflixDataset(base_cfg(
            tmp_path, filters={"posture": {"include": [], "exclude": ["45"]}}))


def test_attribute_values(tmp_path):
    _three_participant_cache(tmp_path)
    ds = NeckflixDataset(base_cfg(tmp_path))
    assert ds.attribute_values("participant") == ["030", "031", "032"]
    assert ds.attribute_values("posture") == ["0", "45", "90"]
    assert ds.attribute_values("perspective") == ["1", "2"]


def test_attribute_values_respects_filters_and_skips_missing(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", attrs={"session": None})
    make_store(tmp_path, "P031_S01_R1_45_D")
    ds = NeckflixDataset(base_cfg(
        tmp_path, filters={"participant": {"include": [], "exclude": ["031"]}}))
    assert ds.attribute_values("participant") == ["030"]
    assert ds.attribute_values("session") == []    # missing attr silently skipped
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: new tests FAIL (`AttributeError: ... no attribute 'attribute_values'`; filter tests see unfiltered samples).

- [ ] **Step 3: Implement**

Add to `BaseZarrDataset`:

```python
    def _filter_by_attribute(self, filters=None) -> None:
        """Filter ``self.samples`` in place by attribute include/exclude.

        ``filters`` is ``{attribute: {"include": [...], "exclude": [...]}}``:
        a value in ``exclude`` drops the sample; a non-empty ``include``
        whitelists. The pseudo-attribute ``"perspective"`` compares
        ``str()``-coerced values against the sample's perspective key
        (deviation 3). A sample whose store lacks a root attr fails any
        non-empty include and passes an exclude-only filter; one UserWarning
        per affected attribute is emitted after the pass (deviation 2 —
        CardioHydra raises KeyError instead). Overlap validation already
        happened upfront in ``__init__`` (deviation 6).
        """
        filters = filters or {}
        missing: dict[str, set[str]] = {}
        filtered_samples = []
        for recording, perspective in self.samples:
            attrs = self.dataset_dict[recording]["attrs"]
            passed = True
            for attribute, spec in filters.items():
                include = spec.get("include", [])
                exclude = spec.get("exclude", [])
                if attribute == "perspective":
                    value = str(perspective)
                    include = [str(v) for v in include]
                    exclude = [str(v) for v in exclude]
                else:
                    value = attrs.get(attribute, _MISSING)
                    if value is _MISSING:
                        missing.setdefault(attribute, set()).add(recording)
                        if include:            # membership unprovable
                            passed = False
                            break
                        continue               # exclude-only: passes
                if value in exclude or (include and value not in include):
                    passed = False
                    break
            if passed:
                filtered_samples.append((recording, perspective))

        for attribute, stores in sorted(missing.items()):
            warnings.warn(
                f"Filter attribute '{attribute}' missing from store root attrs "
                f"of: {', '.join(sorted(stores))}",
                UserWarning,
            )
        self.samples = filtered_samples

    def attribute_values(self, attribute: str) -> list[str]:
        """Sorted unique values of a root attr (or 'perspective') over the
        current samples — the LOSO fold-enumeration primitive (deviation 5).

        Values are ``str()``-coerced before sorting; samples whose store lacks
        the attribute are silently skipped.
        """
        values: set[str] = set()
        for recording, perspective in self.samples:
            if attribute == "perspective":
                values.add(str(perspective))
                continue
            attrs = self.dataset_dict[recording]["attrs"]
            if attribute in attrs:
                values.add(str(attrs[attribute]))
        return sorted(values)
```

In `__init__`, after `self.samples = self.discover_samples()`, add:

```python
        self._filter_by_attribute(self.filters)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/zarr_dataset.py tests/test_neckflix_zarr.py
git commit -m "feat: attribute filters (LOSO mechanism) + attribute_values helper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: window index

**Files:**
- Modify: `dataset/data_loader/zarr_dataset.py`
- Test: `tests/test_neckflix_zarr.py` (append)

**Interfaces:**
- Consumes: `self.samples`, `self.present_streams` (Task 7–8), `self.window_size`, `self.window_stride`, `self.random_windows`.
- Produces (used by Tasks 10–11):
  - `_get_frame_count(self, recording_name: str, perspective: str) -> int` — min `num_frames` video attr across the sample's present streams; missing attr ⇒ `RuntimeError`.
  - `_load_windows(self) -> None` — sets `self.windows: list[tuple[str, str, int | None]]`; strided starts are exactly `range(0, frame_count - window_size + 1, window_stride)`; random mode one `(rec, persp, None)` per sample; samples shorter than `window_size` skipped in both modes.
  - `_sample_streams(self, rec, persp) -> list[str]` and `_sample_traces(self, rec, persp) -> list[str]` helpers.
  - `__init__` ends with `self._load_windows()`; `len(ds)` now meaningful.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neckflix_zarr.py`:

```python
# --------------------------------------------------------------------------
# Task 9: window index
# --------------------------------------------------------------------------
def test_strided_starts_exact_and_non_multiple(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=10)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4, window_stride=3))
    # range(0, 10 - 4 + 1, 3) -> 0, 3, 6
    assert ds.windows == [("P030_S01_R1_0_D", "1", 0),
                          ("P030_S01_R1_0_D", "1", 3),
                          ("P030_S01_R1_0_D", "1", 6)]
    assert len(ds) == 3


def test_stride_defaults_to_window_size(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    cfg = base_cfg(tmp_path, window_size=4)
    del cfg["window_stride"]
    ds = NeckflixDataset(cfg)
    assert [start for _, _, start in ds.windows] == [0, 4, 8]


def test_short_samples_skipped_both_modes(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=3)   # < window_size
    make_store(tmp_path, "P031_S01_R1_0_D", num_frames=8)
    for random_windows in (False, True):
        ds = NeckflixDataset(base_cfg(
            tmp_path, window_size=4, random_windows=random_windows))
        assert {rec for rec, _, _ in ds.windows} == {"P031_S01_R1_0_D"}


def test_random_mode_one_window_per_sample(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", perspectives=("1", "2"), num_frames=20)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4, random_windows=True))
    assert ds.windows == [("P030_S01_R1_0_D", "1", None),
                          ("P030_S01_R1_0_D", "2", None)]
    assert len(ds) == 2


def test_frame_count_is_min_across_present_streams(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D",
               num_frames={"rgb": 12, "ir": 6, "depth": 12})
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4, window_stride=2))
    # frame_count = 6 -> starts range(0, 3, 2) -> 0, 2
    assert [start for _, _, start in ds.windows] == [0, 2]
    assert ds._get_frame_count("P030_S01_R1_0_D", "1") == 6


def test_windows_deterministic_across_constructions(tmp_path):
    make_store(tmp_path, "P031_S01_R1_45_D", num_frames=10)
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=10)
    a = NeckflixDataset(base_cfg(tmp_path))
    b = NeckflixDataset(base_cfg(tmp_path))
    assert a.windows == b.windows
    assert a.windows[0][0] == "P030_S01_R1_0_D"   # sorted recording order
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: new tests FAIL — `ds.windows == []` (Task 6 stub).

- [ ] **Step 3: Implement**

Add to `BaseZarrDataset` (verbatim port of `base.py:245-269, 310-356`):

```python
    def _sample_streams(self, recording_name: str, perspective: str) -> list[str]:
        """Per-sample present streams; full required set if sample unknown."""
        return self.present_streams.get(
            (recording_name, perspective), list(self.required_streams)
        )

    def _sample_traces(self, recording_name: str, perspective: str) -> list[str]:
        """Per-sample present labels lowercased; all labels if sample unknown."""
        labels = self.present_labels.get((recording_name, perspective), self.labels)
        return [lab.lower() for lab in labels]

    def _get_frame_count(self, recording_name: str, perspective: str) -> int:
        """Shortest aligned frame count for a sample, from ``num_frames`` attrs."""
        store_path = self.cache_root / f"{recording_name}.zarr"
        cam = zarr.open_group(str(store_path), mode="r")[perspective]
        length: int | None = None
        for stream_name in self._sample_streams(recording_name, perspective):
            try:
                video = cam[stream_name]["video"]
                n = int(video.attrs["num_frames"])
            except KeyError as err:
                raise RuntimeError(
                    f"{store_path.name}/{perspective}/{stream_name}: missing "
                    "video group or 'num_frames' attr; regenerate this store "
                    "with the Neckflix preprocessor."
                ) from err
            length = n if length is None else min(length, n)
        assert length is not None, f"no streams for {recording_name}/{perspective}"
        return int(length)

    def _load_windows(self) -> None:
        """Build the window index from samples (frame units).

        Strided mode emits ``range(0, frame_count - window_size + 1,
        window_stride)`` starts; random mode emits a single ``None``-start
        entry per sample (start chosen at access time). Samples shorter than
        ``window_size`` are skipped in both modes.
        """
        windows: list[tuple[str, str, int | None]] = []
        for recording_name, perspective in self.samples:
            frame_count = self._get_frame_count(recording_name, perspective)
            if frame_count < self.window_size:
                continue
            if self.random_windows:
                windows.append((recording_name, perspective, None))
                continue
            for start in range(0, frame_count - self.window_size + 1, self.window_stride):
                windows.append((recording_name, perspective, int(start)))
        self.windows = windows
```

In `__init__`, replace:

```python
        self.windows: list[tuple[str, str, int | None]] = []  # populated in Task 9
```

with:

```python
        self._load_windows()
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all PASS. (Note the earlier `test_allow_missing_thresholds` constructs with zero samples — `_load_windows` over an empty list yields `windows == []`, which stays valid.)

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/zarr_dataset.py tests/test_neckflix_zarr.py
git commit -m "feat: strided/random window index over admitted samples

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: `__getitem__` — the nested item contract

**Files:**
- Modify: `dataset/data_loader/zarr_dataset.py`
- Test: `tests/test_neckflix_zarr.py` (append)

**Interfaces:**
- Consumes: Tasks 3 (`finite_stats`, `apply_norm`, `STAT_NAMES`), 6–9 (pipeline state).
- Produces (the loader's public batch contract; Task 11 refines internals only):
  - `_ensure_stream_shapes(self) -> None` — populates `self.stream_hw: dict[str, (H, W)]` lazily.
  - `__getitem__(self, idx: int) -> dict` with exactly the keys `"frames"`, `"labels"`, `"label_stats"`, `"channel_mask"`, `"label_mask"`, `"metadata"` per the spec: frames `{ch: (1, T, H, W) float32}` raw pixels zero-filled where absent; labels `{sig: (T,) float32}` normalised; label_stats `{sig: {STAT_NAMES: () float32}}` physical units; masks scalar bools; metadata `{"recording_id": str, "camera_id": str, "start_frame": int}` with `recording_id` = root attr `recording` falling back to the store stem.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neckflix_zarr.py`:

```python
import numpy as np
import zarr as _zarr
from torch.utils.data import default_collate

from dataset.data_loader.label_transforms import (
    STAT_NAMES, minmax_inverse, zscore_inverse,
)

# --------------------------------------------------------------------------
# Task 10: __getitem__ contract
# --------------------------------------------------------------------------
def _item_ds(tmp_path, **cfg_overrides):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    return NeckflixDataset(base_cfg(tmp_path, window_size=4, **cfg_overrides))


def test_item_keys_shapes_dtypes(tmp_path):
    ds = _item_ds(tmp_path)
    item = ds[0]
    assert set(item) == {"frames", "labels", "label_stats",
                         "channel_mask", "label_mask", "metadata"}
    assert set(item["frames"]) == {"R", "G", "B", "I", "D"}
    for ch in ("R", "G", "B", "I", "D"):
        t = item["frames"][ch]
        assert t.shape == (1, 4, 8, 8) and t.dtype == torch.float32
        assert item["channel_mask"][ch].dtype == torch.bool
        assert item["channel_mask"][ch].dim() == 0
    for sig in ("ABP", "CVP"):
        assert item["labels"][sig].shape == (4,)
        assert item["labels"][sig].dtype == torch.float32
        assert set(item["label_stats"][sig]) == set(STAT_NAMES)
        assert all(v.dim() == 0 for v in item["label_stats"][sig].values())
        assert item["label_mask"][sig].item() is True
    meta = item["metadata"]
    assert meta["recording_id"] == "P030_S01_R1_0_D"
    assert meta["camera_id"] == "1"
    assert meta["start_frame"] == 0


def test_frames_are_raw_pixels(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8,
               frame_fill={("1", "rgb"): 200, ("1", "ir"): 1000,
                           ("1", "depth"): 3000})
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4))
    item = ds[0]
    assert torch.all(item["frames"]["R"] == 200.0)     # no /255, no zstand
    assert torch.all(item["frames"]["I"] == 1000.0)
    assert torch.all(item["frames"]["D"] == 3000.0)


def test_window_slices_match_store_content(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=10)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4, window_stride=3))
    root = _zarr.open_group(str(tmp_path / "P030_S01_R1_0_D.zarr"), mode="r")
    raw = np.asarray(root["1"]["rgb"]["video"]["frames"][:, 3:7])
    item = ds[1]                                        # start == 3
    assert torch.equal(item["frames"]["R"],
                       torch.from_numpy(raw[0][np.newaxis].copy()).float())
    assert item["metadata"]["start_frame"] == 3


def test_absent_channel_zero_filled_at_canonical_shape(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", streams=("rgb",), hw=(8, 8))
    make_store(tmp_path, "P031_S01_R1_0_D", streams=("rgb", "ir"), hw=(6, 6))
    ds = NeckflixDataset(base_cfg(
        tmp_path, channels=["R", "I"], window_size=4,
        allow_missing=True, min_channels=1))
    idx = next(i for i, (rec, _, _) in enumerate(ds.windows)
               if rec == "P030_S01_R1_0_D")
    item = ds[idx]
    assert item["channel_mask"]["R"].item() is True
    assert item["channel_mask"]["I"].item() is False
    assert item["frames"]["I"].shape == (1, 4, 6, 6)    # ir's real shape, from P031
    assert torch.all(item["frames"]["I"] == 0.0)


def test_labels_round_trip_and_averaging(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4))
    item = ds[0]
    root = _zarr.open_group(str(tmp_path / "P030_S01_R1_0_D.zarr"), mode="r")
    copies = [np.asarray(root["1"][s]["abp"]["data"][0:4])
              for s in ("rgb", "ir", "depth")]
    raw_window = torch.from_numpy(np.mean(copies, axis=0)).float()
    recovered = zscore_inverse(item["labels"]["ABP"], item["label_stats"]["ABP"])
    assert torch.allclose(recovered, raw_window, atol=1e-3)


def test_minmax_labels_in_unit_interval(tmp_path):
    ds = _item_ds(tmp_path, label_norm="minmax")
    item = ds[0]
    for sig in ("ABP", "CVP"):
        assert item["labels"][sig].min().item() >= 0.0
        assert item["labels"][sig].max().item() <= 1.0
        root = _zarr.open_group(str(ds.cache_root / "P030_S01_R1_0_D.zarr"), mode="r")
        copies = [np.asarray(root["1"][s][sig.lower()]["data"][0:4])
                  for s in ("rgb", "ir", "depth")]
        raw_window = torch.from_numpy(np.mean(copies, axis=0)).float()
        recovered = minmax_inverse(item["labels"][sig], item["label_stats"][sig])
        assert torch.allclose(recovered, raw_window, atol=1e-3)


def test_absent_label_exact_zeros_and_mask_false(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", traces=("abp",))
    ds = NeckflixDataset(base_cfg(
        tmp_path, window_size=4, allow_missing=True, min_labels=1))
    item = ds[0]
    assert item["label_mask"]["ABP"].item() is True
    assert item["label_mask"]["CVP"].item() is False
    assert torch.equal(item["labels"]["CVP"], torch.zeros(4))
    assert all(v.item() == 0.0 for v in item["label_stats"]["CVP"].values())


def test_random_window_start_within_bounds_and_varies(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=64)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4, random_windows=True))
    torch.manual_seed(0)
    starts = {ds[0]["metadata"]["start_frame"] for _ in range(20)}
    assert all(0 <= s <= 60 for s in starts)
    assert len(starts) >= 2                      # start re-drawn per access


def test_default_collate_batches_the_nested_dict(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4))
    batch = default_collate([ds[0], ds[1]])
    assert batch["frames"]["R"].shape == (2, 1, 4, 8, 8)
    assert batch["labels"]["ABP"].shape == (2, 4)
    assert batch["label_stats"]["ABP"]["mean"].shape == (2,)
    assert batch["channel_mask"]["R"].shape == (2,)
    assert batch["channel_mask"]["R"].dtype == torch.bool
    assert batch["metadata"]["recording_id"] == ["P030_S01_R1_0_D"] * 2
    assert batch["metadata"]["camera_id"] == ["1", "1"]
    assert batch["metadata"]["start_frame"].dtype == torch.int64
    assert batch["metadata"]["start_frame"].tolist() == [0, 4]
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: new tests FAIL — `TypeError`/`NotImplementedError`: `__getitem__` not defined (torch Dataset raises).

- [ ] **Step 3: Implement**

Add to the imports of `zarr_dataset.py`:

```python
import numpy as np

from dataset.data_loader.label_transforms import STAT_NAMES, apply_norm, finite_stats
```

Add to `BaseZarrDataset` (port of `base.py:358-528` with the deviation-4 label path; deviation 7 lands in Task 11):

```python
    def _ensure_stream_shapes(self) -> None:
        """Populate ``self.stream_hw`` (canonical (H, W) per required stream).

        Scans cached stores for each required stream's frame shape so absent
        channels can be zero-filled to a size consistent across samples.
        Freezes only once every required stream has a real shape; a stream
        never seen in any store keeps the fallback (first real shape found).
        """
        if getattr(self, "_stream_hw_complete", False):
            return
        found: dict[str, tuple[int, int]] = dict(getattr(self, "_stream_hw_found", {}))
        for rec_name, perspective in self.samples:
            store_path = self.cache_root / f"{rec_name}.zarr"
            if not store_path.exists():
                continue
            root = zarr.open_group(str(store_path), mode="r")
            try:
                cam = root[str(perspective)]
            except KeyError:
                continue
            for stream_name in self.required_streams:
                if stream_name in found:
                    continue
                try:
                    frames = cam[stream_name]["video"]["frames"]  # (C, T, H, W)
                    found[stream_name] = (int(frames.shape[-2]), int(frames.shape[-1]))
                except KeyError:
                    pass
        if not found:
            raise RuntimeError(
                "No cached streams found to infer fill shapes; check cache_dir stores"
            )
        self._stream_hw_found = found
        fallback = next(iter(found.values()))
        self.stream_hw = {s: found.get(s, fallback) for s in self.required_streams}
        self._stream_hw_complete = True

    def __getitem__(self, idx: int) -> dict:
        """Load one window as the spec's nested dense dict (see class docstring).

        frames: {channel: (1, T, H, W) float32} raw pixels, zeros where the
        stream is absent; labels: {label: (T,) float32} normalised per
        ``label_norm`` with finite-only stats (deviation 4); label_stats:
        physical-unit stats that normalised each window; channel_mask /
        label_mask: scalar bools; metadata: recording_id / camera_id /
        start_frame.
        """
        rec_name, camera_id, start = self.windows[idx]
        self._ensure_stream_shapes()

        present_streams = self._sample_streams(rec_name, str(camera_id))
        present_labels = set(
            self.present_labels.get((rec_name, str(camera_id)), self.labels)
        )

        if start is None:  # random window
            n_frames = self._get_frame_count(rec_name, camera_id)
            max_start = n_frames - self.window_size
            start = (
                int(torch.randint(0, max_start + 1, (1,)).item())
                if max_start > 0
                else 0
            )
        end = start + self.window_size

        store_path = self.cache_root / f"{rec_name}.zarr"
        root = zarr.open_group(str(store_path), mode="r")
        cam_group = root[str(camera_id)]

        # --- Load present streams' frames + label trace copies ---
        stream_frames: dict[str, np.ndarray] = {}
        label_accumulators: dict[str, list[np.ndarray]] = {
            name: [] for name in self.labels
        }
        for stream_name in present_streams:
            stream = cam_group[stream_name]
            video = stream["video"]["frames"]  # (C, T, H, W) raw frames
            stream_frames[stream_name] = np.asarray(video[:, start:end])
            for label_name in self.labels:
                trace_key = label_name.lower()
                if trace_key in stream:
                    label_accumulators[label_name].append(
                        np.asarray(stream[trace_key]["data"][start:end], dtype=np.float64)
                    )

        # --- Dense frames: every channel, zeros where its stream is absent ---
        frames: dict[str, torch.Tensor] = {}
        channel_mask: dict[str, torch.Tensor] = {}
        for ch_name, (s_name, ch_idx) in zip(self.channels, self.stream_plan):
            present = s_name in stream_frames
            if present:
                arr = stream_frames[s_name][ch_idx][np.newaxis]  # (1, T, H, W)
                frames[ch_name] = torch.from_numpy(arr.copy()).float()
            else:
                h, w = self.stream_hw[s_name]
                frames[ch_name] = torch.zeros(
                    (1, self.window_size, h, w), dtype=torch.float32
                )
            channel_mask[ch_name] = torch.tensor(present, dtype=torch.bool)

        # --- Dense labels: finite-only stats + post-norm NaN zeroing (dev. 4) ---
        labels: dict[str, torch.Tensor] = {}
        label_stats: dict[str, dict[str, torch.Tensor]] = {}
        label_mask: dict[str, torch.Tensor] = {}
        for label_name in self.labels:
            arrays = label_accumulators[label_name]
            has_data = bool(arrays) and label_name in present_labels
            if has_data:
                raw = torch.from_numpy(np.mean(arrays, axis=0)).float()  # (T,)
                finite = torch.isfinite(raw)
                present = bool(finite.any())
            else:
                present = False
            if present:
                stats = finite_stats(raw)
                normed = torch.where(
                    finite, apply_norm(raw, stats, self.label_norm),
                    raw.new_zeros(()),
                )
            else:
                normed = torch.zeros(self.window_size, dtype=torch.float32)
                stats = {
                    name: torch.zeros((), dtype=torch.float32)
                    for name in STAT_NAMES
                }
            labels[label_name] = normed
            label_stats[label_name] = stats
            label_mask[label_name] = torch.tensor(present, dtype=torch.bool)

        recording_id = root.attrs.get("recording", rec_name)

        return {
            "frames": frames,
            "labels": labels,
            "label_stats": label_stats,
            "channel_mask": channel_mask,
            "label_mask": label_mask,
            "metadata": {
                "recording_id": recording_id,
                "camera_id": camera_id,
                "start_frame": start,
            },
        }
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/zarr_dataset.py tests/test_neckflix_zarr.py
git commit -m "feat: lazy __getitem__ emitting the nested multichannel item contract

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: deviations 4 + 7 edge cases — NaN guard and short traces

**Files:**
- Modify: `dataset/data_loader/zarr_dataset.py`
- Test: `tests/test_neckflix_zarr.py` (append)

**Interfaces:**
- Consumes: Task 10's `__getitem__`.
- Produces:
  - `_window_trace(self, stream, trace_key: str, start: int, end: int) -> np.ndarray` — `(end-start,)` float64, NaN-right-padded when the stored trace is shorter than `end`.
  - `_finite_mean(arrays: list[np.ndarray]) -> np.ndarray` (staticmethod) — position-wise mean over finite values across copies; all-NaN positions stay NaN; no RuntimeWarning.
  - `__getitem__` label path uses both; its public contract is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_neckflix_zarr.py`:

```python
# --------------------------------------------------------------------------
# Task 11: NaN guard + short-trace tolerance (deviations 4 and 7)
# --------------------------------------------------------------------------
def test_interior_nan_zeroed_under_both_norms(tmp_path):
    values = np.arange(8, dtype=np.float64) + 100.0
    values[2] = np.nan
    tv = {("1", s, "abp"): values for s in ("rgb", "ir", "depth")}
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8,
               traces=("abp",), trace_values=tv)
    for norm in ("zscore", "minmax"):
        ds = NeckflixDataset(base_cfg(
            tmp_path, labels=["ABP"], window_size=8, label_norm=norm))
        item = ds[0]
        sig = item["labels"]["ABP"]
        assert torch.isfinite(sig).all()
        assert sig[2].item() == 0.0                      # exact zero, both norms
        assert item["label_mask"]["ABP"].item() is True
        # stats computed over the 7 finite entries only
        finite = torch.tensor(np.delete(values, 2), dtype=torch.float32)
        assert item["label_stats"]["ABP"]["mean"].item() == pytest.approx(
            finite.mean().item())


def test_all_nan_window_treated_as_absent(tmp_path):
    values = np.full(8, np.nan)
    tv = {("1", s, "abp"): values for s in ("rgb", "ir", "depth")}
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8,
               traces=("abp",), trace_values=tv)
    ds = NeckflixDataset(base_cfg(tmp_path, labels=["ABP"], window_size=8))
    item = ds[0]
    assert item["label_mask"]["ABP"].item() is False
    assert torch.equal(item["labels"]["ABP"], torch.zeros(8))
    assert all(v.item() == 0.0 for v in item["label_stats"]["ABP"].values())


def test_nan_in_one_copy_uses_other_copies_at_that_position(tmp_path):
    a = np.arange(8, dtype=np.float64) + 100.0
    b = a.copy()
    b[3] = np.nan
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8, traces=("abp",),
               trace_values={("1", "rgb", "abp"): a,
                             ("1", "ir", "abp"): b,
                             ("1", "depth", "abp"): a})
    ds = NeckflixDataset(base_cfg(tmp_path, labels=["ABP"], window_size=8))
    item = ds[0]
    recovered = zscore_inverse(item["labels"]["ABP"], item["label_stats"]["ABP"])
    # Position 3 averages the two finite copies (both == a[3]) — not zeroed.
    assert recovered[3].item() == pytest.approx(a[3], abs=1e-3)
    assert torch.isfinite(item["labels"]["ABP"]).all()


def test_short_trace_copy_padded_not_crashing(tmp_path):
    # abp under rgb trimmed to 5 of 8 frames; window overlaps the tail.
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8, traces=("abp",),
               trace_lengths={("1", "rgb", "abp"): 5})
    ds = NeckflixDataset(base_cfg(tmp_path, labels=["ABP"], window_size=8))
    item = ds[0]                                        # must not raise
    sig = item["labels"]["ABP"]
    assert sig.shape == (8,)
    assert torch.isfinite(sig).all()
    # Tail positions still covered by the full-length ir/depth copies.
    recovered = zscore_inverse(sig, item["label_stats"]["ABP"])
    root = _zarr.open_group(str(tmp_path / "P030_S01_R1_0_D.zarr"), mode="r")
    full = np.asarray(root["1"]["ir"]["abp"]["data"][:])
    assert recovered[6].item() == pytest.approx(full[6], abs=1e-3)


def test_all_copies_short_tail_zeroed_and_masked_true(tmp_path):
    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=8, traces=("abp",),
               trace_lengths={("1", s, "abp"): 5 for s in ("rgb", "ir", "depth")})
    ds = NeckflixDataset(base_cfg(tmp_path, labels=["ABP"], window_size=8))
    item = ds[0]
    sig = item["labels"]["ABP"]
    assert torch.isfinite(sig).all()
    assert torch.equal(sig[5:], torch.zeros(3))          # padded tail -> zeros
    assert item["label_mask"]["ABP"].item() is True      # finite head exists
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: `test_short_trace_copy_padded_not_crashing` and `test_all_copies_short_tail_zeroed...` FAIL (shape mismatch in `np.mean` over unequal-length copies); `test_nan_in_one_copy...` FAILS (`np.mean` propagates the NaN, so position 3 is zeroed instead of averaged). The two pure-NaN tests may already pass — keep them; they pin the contract.

- [ ] **Step 3: Implement**

Add to `BaseZarrDataset`:

```python
    def _window_trace(self, stream, trace_key: str, start: int, end: int) -> np.ndarray:
        """Slice one trace copy, NaN-right-padded to the window (deviation 7).

        Per-stream trailing-NaN trimming can leave a trace shorter than its
        stream's ``num_frames``; a window overlapping that tail yields a short
        slice, padded here so copies always align. CardioHydra crashes on this.
        """
        data = stream[trace_key]["data"]
        stop = min(end, int(data.shape[0]))
        sliced = np.asarray(data[start:stop], dtype=np.float64)
        if sliced.shape[0] < (end - start):
            pad = np.full((end - start) - sliced.shape[0], np.nan)
            sliced = np.concatenate([sliced, pad])
        return sliced

    @staticmethod
    def _finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
        """Position-wise mean over finite values across trace copies.

        Positions where every copy is non-finite stay NaN (absorbed by the
        deviation-4 guard downstream). Warning-free equivalent of np.nanmean.
        """
        stacked = np.stack(arrays)                       # (n_copies, T)
        finite = np.isfinite(stacked)
        counts = finite.sum(axis=0)                      # (T,)
        sums = np.where(finite, stacked, 0.0).sum(axis=0)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
```

In `__getitem__`, replace the trace-slicing line inside the stream loop:

```python
                if trace_key in stream:
                    label_accumulators[label_name].append(
                        np.asarray(stream[trace_key]["data"][start:end], dtype=np.float64)
                    )
```

with:

```python
                if trace_key in stream:
                    label_accumulators[label_name].append(
                        self._window_trace(stream, trace_key, start, end)
                    )
```

and replace the averaging line:

```python
                raw = torch.from_numpy(np.mean(arrays, axis=0)).float()  # (T,)
```

with:

```python
                raw = torch.from_numpy(self._finite_mean(arrays)).float()  # (T,)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run python -m pytest tests/test_neckflix_zarr.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite + commit**

Run: `uv run python -m pytest tests/ -q` — all green.

```bash
git add dataset/data_loader/zarr_dataset.py tests/test_neckflix_zarr.py
git commit -m "feat: interior-NaN guard and short-trace tolerance in the label path

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: final verification sweep

**Files:**
- Test: whole suite; no production edits expected.

**Interfaces:**
- Consumes: everything above.
- Produces: a verified, spec-complete loader on `multisignal-pilot`.

- [ ] **Step 1: Full test suite, verbose**

Run: `uv run python -m pytest tests/ -v`
Expected: every test passes; `tests/test_neckflix_dict.py` absent from collection.

- [ ] **Step 2: Spec coverage checklist**

Open `docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md` and confirm each is implemented and tested (fix + amend tests in this task if any gap is found):

- [ ] Admission gate: unreadable / `complete` identity / `tool_version` parse — Task 6.
- [ ] `FileNotFoundError` / `RuntimeError` conditions incl. events-only store — Task 6.
- [ ] Strict + `allow_missing` discovery, presence bookkeeping — Task 7.
- [ ] Upfront overlap `ValueError` (dev. 6), missing-attr warning semantics (dev. 2), perspective filter with coercion (dev. 3), unprefixed participant LOSO — Tasks 5, 8.
- [ ] `attribute_values` coercion + skip (dev. 5) — Task 8.
- [ ] Strided formula, stride default, random-mode epoch semantics, short-sample skip, min-across-streams frame count, determinism — Task 9.
- [ ] Item contract: keys/shapes/dtypes, raw pixels, zero-fill + canonical shapes, mask semantics, stats round-trip, minmax `[0,1]`, absent-label zeros, `recording_id` source, collate incl. `start_frame` int64 — Task 10.
- [ ] NaN guard uniform zeros both norms, all-NaN ⇒ absent (dev. 4); short-trace padding + finite-mean (dev. 7) — Task 11.
- [ ] Config table defaults (`window_stride`, `random_windows`, `label_norm`, `allow_missing`, `min_channels`, `min_labels`) exercised; no `fps` key anywhere (dev. 1).
- [ ] `main.py` imports; `"Neckflix"` unsupported — Task 5.
- [ ] `requirements.txt` untouched: `git diff main..HEAD --stat -- requirements.txt` shows nothing.

- [ ] **Step 3: Construction-cost sanity (metadata-only init)**

Run:

```bash
uv run python - <<'EOF'
import pathlib, tempfile, time
import sys
sys.path.insert(0, ".")
from tests.zarr_fixtures import base_cfg, make_store
from dataset.data_loader.NeckflixLoader import NeckflixDataset

with tempfile.TemporaryDirectory() as d:
    root = pathlib.Path(d)
    for i in range(20):
        make_store(root, f"P{i:03d}_S01_R1_0_D", num_frames=64)
    t0 = time.perf_counter()
    ds = NeckflixDataset(base_cfg(root, window_size=16))
    dt = time.perf_counter() - t0
    print(f"init over 20 stores: {dt:.3f}s, {len(ds)} windows, "
          f"participants={ds.attribute_values('participant')[:3]}...")
    assert dt < 5.0, "metadata-only init should be fast"
EOF
```

Expected: prints timing well under 5 s and a non-zero window count.

- [ ] **Step 4: Commit any checklist fixes; otherwise no-op**

```bash
git status   # clean, or commit fixes with a descriptive message
```

---

## Follow-ups (explicitly NOT in this plan, per spec)

1. Wire `neckflix_main.py`: yacs → cfg-dict translation, two-instance LOSO construction from `--test_participants` (strip the `P` prefix!), DataLoaders/DDP, trainer dispatch.
2. Consumer rework: adapt `SignalDictWrapper` / `MaskedMultiSignalLoss` / a trainer to the nested contract; frame preprocessing (DiffNormalized etc.) consumer-side.
3. Regenerate the zarr cache with `ghcr.io/coenarrow/neckflix:1.0.0` (docker locally; Apptainer or native uv on HPC) and point configs at it.
4. Update `docs/architecture.md`, `docs/changelog.md`, and CLAUDE.md's Neckflix/HDF5 sections.
