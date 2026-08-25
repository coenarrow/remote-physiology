# Multi-Signal Pipeline (DeepPhys Pilot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** DeepPhys trains/infers on Neckflix through a dict-based multi-signal pipeline: up to 5 zero-filled+masked input channels, dict predictions/labels over the canonical signal vocabulary, masked multi-signal loss.

**Architecture:** Approach B from the spec — backbones stay dict-free (`in_channels`, `out_signals` → raw tensor); all dict/mask machinery lives in shared modules (`signals.py` registry, `SignalDictWrapper`, `MaskedMultiSignalLoss`, `MultiSignalTrainer` + model registry). NeckflixLoader gains a `dict_output` mode emitting the batch-dict contract.

**Tech Stack:** PyTorch, YACS config, h5py, pytest. Runs on cpu/mps/cuda (trainer picks).

**Spec:** `docs/superpowers/specs/2026-08-25-multisignal-pipeline-design.md`

## Global Constraints

- Signal vocabulary: `PPG, ECG, ABP, CVP, RESP, EDA, SPO2` (waveforms) + `HR` eval-only. Loader names "BVP"/"Pulse" map to `PPG`.
- Channel slots: `('R','G','B','I','D')` canonical order; config `CHANNELS` picks an ordered subset; missing channels are zero-filled with `channel_mask=0`.
- Missing labels are masked (`label_mask=0`), never dropped; loss denominators clamped ≥1 (no NaN).
- Normalization: clip to `(lo, hi)` then min-max to `[-1, 1]` (matches existing `normalise_trace`).
- Legacy paths (main.py HR pipeline, existing trainers, non-dict loader mode) must keep working: all changes are additive or default-off (`dict_output=False`).
- Run all tests with: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/<file> -q`
- Every commit message ends with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailer.

---

### Task 1: Signal & channel registries (`neural_methods/signals.py`)

**Files:**
- Create: `neural_methods/signals.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Consumes: nothing (leaf module; no repo imports).
- Produces (used by every later task):
  - `CHANNELS: tuple[str] = ('R','G','B','I','D')`
  - `SIGNALS: dict[str, dict]` with key `'norm': (float, float)` per signal
  - `EVAL_ONLY: tuple[str] = ('HR',)`
  - `canonical_signal(name: str) -> str`
  - `validate_traces(traces: list[str]) -> list[str]`
  - `validate_channels(channels: list[str]) -> list[str]`
  - `norm_range(sig: str, overrides: dict | None = None) -> tuple[float, float]`
  - `normalize_signal(x: np.ndarray, sig: str, overrides=None) -> np.ndarray`
  - `denormalize_signal(x: np.ndarray, sig: str, overrides=None) -> np.ndarray`
  - `resolve_channels(config_data) -> list[str]`, `resolve_traces(config_data) -> list[str]` (new global keys with `NECKFLIX.*` fallback)
  - `signal_norm_overrides(config_data) -> dict[str, tuple]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signals.py
import numpy as np
import pytest

from neural_methods import signals as S


def test_registry_contents():
    assert S.CHANNELS == ('R', 'G', 'B', 'I', 'D')
    assert set(S.SIGNALS) == {'PPG', 'ECG', 'ABP', 'CVP', 'RESP', 'EDA', 'SPO2'}
    assert S.EVAL_ONLY == ('HR',)
    for sig, meta in S.SIGNALS.items():
        lo, hi = meta['norm']
        assert lo < hi


def test_canonical_signal_aliases():
    assert S.canonical_signal('BVP') == 'PPG'
    assert S.canonical_signal('bvp') == 'PPG'
    assert S.canonical_signal('Pulse') == 'PPG'
    assert S.canonical_signal('abp') == 'ABP'
    with pytest.raises(KeyError):
        S.canonical_signal('THERMISTOR')


def test_validate_traces():
    assert S.validate_traces(['abp', 'BVP']) == ['ABP', 'PPG']
    with pytest.raises(ValueError):
        S.validate_traces(['HR'])          # eval-only is not a training trace
    with pytest.raises(ValueError):
        S.validate_traces([])


def test_validate_channels():
    assert S.validate_channels(['R', 'G', 'B']) == ['R', 'G', 'B']
    with pytest.raises(ValueError):
        S.validate_channels(['R', 'X'])
    with pytest.raises(ValueError):
        S.validate_channels([])


def test_normalize_roundtrip_and_clip():
    x = np.array([-100.0, 0.0, 100.0, 300.0])   # ABP norm (0, 200)
    n = S.normalize_signal(x, 'ABP')
    assert n.min() >= -1.0 and n.max() <= 1.0
    assert n[0] == -1.0        # clipped low
    assert n[3] == 1.0         # clipped high
    assert n[2] == 0.0         # midpoint of (0,200)
    d = S.denormalize_signal(n, 'ABP')
    assert np.allclose(d, [0.0, 0.0, 100.0, 200.0])


def test_norm_override():
    lo, hi = S.norm_range('ABP', overrides={'ABP': (0.0, 100.0)})
    assert (lo, hi) == (0.0, 100.0)
    lo, hi = S.norm_range('CVP', overrides={'ABP': (0.0, 100.0)})
    assert (lo, hi) == S.SIGNALS['CVP']['norm']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_signals.py -q`
Expected: FAIL / error — `No module named 'neural_methods.signals'`

- [ ] **Step 3: Implement `neural_methods/signals.py`**

```python
"""Canonical physiological signal and camera-channel registries.

Single source of truth for the multi-signal pipeline (see
docs/superpowers/specs/2026-08-25-multisignal-pipeline-design.md).
Signal names cover exactly what this repo's dataloaders provide.
"""
import numpy as np

CHANNELS = ('R', 'G', 'B', 'I', 'D')

SIGNALS = {
    'PPG':  {'norm': (-3.0, 3.0)},       # a.k.a. BVP; standardized units
    'ECG':  {'norm': (-1500.0, 1500.0)},
    'ABP':  {'norm': (0.0, 200.0)},
    'CVP':  {'norm': (-20.0, 30.0)},
    'RESP': {'norm': (0.0, 10.0)},       # BP4D Resp_Volts scale; override per dataset
    'EDA':  {'norm': (0.0, 40.0)},       # microsiemens; override per dataset
    'SPO2': {'norm': (0.0, 100.0)},
}

EVAL_ONLY = ('HR',)

_ALIASES = {'BVP': 'PPG', 'PULSE': 'PPG'}


def canonical_signal(name):
    """Map any loader-side name to the canonical vocabulary (KeyError if unknown)."""
    up = str(name).upper()
    up = _ALIASES.get(up, up)
    if up in SIGNALS or up in EVAL_ONLY:
        return up
    raise KeyError(
        f"Unknown signal {name!r}; known: {sorted(SIGNALS)} + eval-only {list(EVAL_ONLY)}")


def validate_traces(traces):
    """Canonicalize a config TRACES list; reject empty, unknown, or eval-only."""
    if not traces:
        raise ValueError("TRACES must name at least one signal")
    out = []
    for t in traces:
        c = canonical_signal(t)
        if c in EVAL_ONLY:
            raise ValueError(f"{c} is eval-only and cannot be a training trace")
        out.append(c)
    return out


def validate_channels(channels):
    """Validate a config CHANNELS list against the canonical slots."""
    if not channels:
        raise ValueError("CHANNELS must name at least one channel")
    bad = [c for c in channels if c not in CHANNELS]
    if bad:
        raise ValueError(f"Unknown channels {bad}; known: {list(CHANNELS)}")
    return list(channels)


def norm_range(sig, overrides=None):
    if overrides and sig in overrides:
        lo, hi = overrides[sig]
    else:
        lo, hi = SIGNALS[sig]['norm']
    return float(lo), float(hi)


def normalize_signal(x, sig, overrides=None):
    """Clip to the signal's range then min-max to [-1, 1]."""
    lo, hi = norm_range(sig, overrides)
    clipped = np.clip(x, lo, hi)
    return (clipped - lo) / (hi - lo) * 2.0 - 1.0


def denormalize_signal(x, sig, overrides=None):
    lo, hi = norm_range(sig, overrides)
    return (np.asarray(x) + 1.0) / 2.0 * (hi - lo) + lo


def resolve_channels(config_data):
    """Global PREPROCESS.CHANNELS, falling back to legacy NECKFLIX.CHANNELS."""
    chs = list(getattr(config_data.PREPROCESS, 'CHANNELS', []) or [])
    if not chs:
        chs = list(config_data.PREPROCESS.NECKFLIX.CHANNELS)
    return validate_channels(chs)


def resolve_traces(config_data):
    """Global PREPROCESS.TRACES, falling back to legacy NECKFLIX.TRACES."""
    trs = list(getattr(config_data.PREPROCESS, 'TRACES', []) or [])
    if not trs:
        trs = list(config_data.PREPROCESS.NECKFLIX.TRACES)
    return validate_traces(trs)


def signal_norm_overrides(config_data):
    """Read PREPROCESS.SIGNAL_NORMS (if configured) into {sig: (lo, hi)}."""
    out = {}
    norms = getattr(config_data.PREPROCESS, 'SIGNAL_NORMS', None)
    if norms is not None:
        for sig in SIGNALS:
            val = getattr(norms, sig, None)
            if val:
                out[sig] = (float(val[0]), float(val[1]))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_signals.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add neural_methods/signals.py tests/test_signals.py
git commit -m "feat: canonical signal/channel registries for multi-signal pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Config keys (`config.py`)

**Files:**
- Modify: `config.py` (TRAIN/VALID/TEST `DATA.PREPROCESS` sections — find them by searching `PREPROCESS.NECKFLIX = CN()`, they exist at roughly lines 96, 169, 244)
- Test: `tests/test_config_keys.py`

**Interfaces:**
- Consumes: `neural_methods.signals.SIGNALS` for defaults.
- Produces: config keys read by later tasks: `<SECTION>.DATA.PREPROCESS.CHANNELS` (list, default `[]`), `<SECTION>.DATA.PREPROCESS.TRACES` (list, default `[]`), `<SECTION>.DATA.PREPROCESS.SIGNAL_NORMS.<SIG>` (2-list, defaults from registry) for SECTION in TRAIN/VALID/TEST.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_keys.py
import argparse
import textwrap


def test_new_preprocess_keys(tmp_path):
    from config import get_config
    yaml = tmp_path / "c.yaml"
    yaml.write_text(textwrap.dedent("""\
        BASE: ['']
        TRAIN:
          DATA:
            PREPROCESS:
              CHANNELS: ['R','G','B','I']
              TRACES: ['ABP','CVP']
              SIGNAL_NORMS:
                ABP: [0, 180]
    """))
    cfg = get_config(argparse.Namespace(config_file=str(yaml)))
    assert cfg.TRAIN.DATA.PREPROCESS.CHANNELS == ['R', 'G', 'B', 'I']
    assert cfg.TRAIN.DATA.PREPROCESS.TRACES == ['ABP', 'CVP']
    assert list(cfg.TRAIN.DATA.PREPROCESS.SIGNAL_NORMS.ABP) == [0, 180]
    # defaults intact elsewhere
    assert cfg.TEST.DATA.PREPROCESS.CHANNELS == []
    assert list(cfg.TEST.DATA.PREPROCESS.SIGNAL_NORMS.CVP) == [-20.0, 30.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_config_keys.py -q`
Expected: FAIL — `Non-existent config key: TRAIN.DATA.PREPROCESS.CHANNELS`

- [ ] **Step 3: Add the keys**

In `config.py`, add near the top (after existing imports):

```python
from neural_methods.signals import SIGNALS as _SIGNAL_REGISTRY
```

Then, for EACH of the three sections (search for the three occurrences of
`.DATA.PREPROCESS.NECKFLIX = CN()` — TRAIN, VALID, TEST) insert IMMEDIATELY
BEFORE that NECKFLIX line (replace `_C.TRAIN` with the section's prefix):

```python
_C.TRAIN.DATA.PREPROCESS.CHANNELS = []      # global channel slots (subset of R,G,B,I,D); [] -> legacy NECKFLIX.CHANNELS
_C.TRAIN.DATA.PREPROCESS.TRACES = []        # global signal targets; [] -> legacy NECKFLIX.TRACES
_C.TRAIN.DATA.PREPROCESS.SIGNAL_NORMS = CN()
for _sig, _meta in _SIGNAL_REGISTRY.items():
    _C.TRAIN.DATA.PREPROCESS.SIGNAL_NORMS[_sig] = list(_meta['norm'])
```

- [ ] **Step 4: Run test + existing suites to verify nothing broke**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/ -q`
Expected: all pass (new test + existing test_mamba_compat suite)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config_keys.py
git commit -m "feat: global CHANNELS/TRACES/SIGNAL_NORMS preprocess config keys

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `MaskedMultiSignalLoss`

**Files:**
- Create: `neural_methods/loss/MaskedMultiSignalLoss.py`
- Test: `tests/test_masked_loss.py`

**Interfaces:**
- Consumes: `signals.validate_traces` (Task 1).
- Produces: `MaskedMultiSignalLoss(traces: list[str], base: str = 'mse')`;
  `forward(preds: dict[str, (B,T)], labels: dict[str, (B,T)], label_mask: dict[str, (B,)]) -> scalar Tensor`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_masked_loss.py
import torch

from neural_methods.loss.MaskedMultiSignalLoss import MaskedMultiSignalLoss


def _mk(B=4, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    preds = {'ABP': torch.randn(B, T, generator=g), 'CVP': torch.randn(B, T, generator=g)}
    labels = {'ABP': torch.randn(B, T, generator=g), 'CVP': torch.randn(B, T, generator=g)}
    return preds, labels


def test_hand_computed_masked_mse():
    loss_fn = MaskedMultiSignalLoss(['ABP', 'CVP'], base='mse')
    preds = {'ABP': torch.zeros(2, 4), 'CVP': torch.zeros(2, 4)}
    labels = {'ABP': torch.ones(2, 4), 'CVP': torch.full((2, 4), 2.0)}
    mask = {'ABP': torch.tensor([1.0, 1.0]), 'CVP': torch.tensor([1.0, 0.0])}
    # ABP: per-sample MSE = 1.0, both present -> 1.0
    # CVP: per-sample MSE = 4.0, one present -> 4.0
    out = loss_fn(preds, labels, mask)
    assert torch.isclose(out, torch.tensor((1.0 + 4.0) / 2))


def test_fully_masked_signal_contributes_zero_no_nan():
    loss_fn = MaskedMultiSignalLoss(['ABP', 'CVP'], base='mse')
    preds, labels = _mk()
    mask = {'ABP': torch.ones(4), 'CVP': torch.zeros(4)}
    out = loss_fn(preds, labels, mask)
    assert torch.isfinite(out)
    # equals half the ABP-only term (CVP contributes exactly 0)
    only_abp = MaskedMultiSignalLoss(['ABP'], base='mse')(
        {'ABP': preds['ABP']}, {'ABP': labels['ABP']}, {'ABP': mask['ABP']})
    assert torch.isclose(out, only_abp / 2)


def test_negpearson_base_and_gradients():
    loss_fn = MaskedMultiSignalLoss(['ABP'], base='negpearson')
    p = torch.randn(3, 32, requires_grad=True)
    labels = {'ABP': torch.randn(3, 32)}
    out = loss_fn({'ABP': p}, labels, {'ABP': torch.ones(3)})
    out.backward()
    assert torch.isfinite(out)
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_unknown_base_raises():
    import pytest
    with pytest.raises(ValueError):
        MaskedMultiSignalLoss(['ABP'], base='huber')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_masked_loss.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
# neural_methods/loss/MaskedMultiSignalLoss.py
"""Masked multi-signal loss over dict predictions/labels.

Per signal: per-sample base loss, masked mean over the batch (denominator
clamped to >= 1 so a fully-absent signal contributes exactly 0, never NaN);
then the plain mean over the configured trace list.
"""
import torch
import torch.nn as nn

from neural_methods.signals import validate_traces

_EPS = 1e-8


def _mse_per_sample(pred, label):
    return ((pred - label) ** 2).mean(dim=-1)


def _negpearson_per_sample(pred, label):
    p = pred - pred.mean(dim=-1, keepdim=True)
    l = label - label.mean(dim=-1, keepdim=True)
    num = (p * l).sum(dim=-1)
    den = torch.sqrt((p ** 2).sum(dim=-1) * (l ** 2).sum(dim=-1) + _EPS)
    return 1.0 - num / den


_BASES = {'mse': _mse_per_sample, 'negpearson': _negpearson_per_sample}


class MaskedMultiSignalLoss(nn.Module):
    def __init__(self, traces, base='mse'):
        super().__init__()
        self.traces = validate_traces(traces)
        if base not in _BASES:
            raise ValueError(f"Unknown base loss {base!r}; known: {sorted(_BASES)}")
        self.base = base
        self._fn = _BASES[base]

    def forward(self, preds, labels, label_mask):
        terms = []
        for sig in self.traces:
            per_sample = self._fn(preds[sig], labels[sig])          # (B,)
            m = label_mask[sig].to(per_sample.dtype)                # (B,)
            terms.append((per_sample * m).sum() / m.sum().clamp(min=1.0))
        return torch.stack(terms).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_masked_loss.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add neural_methods/loss/MaskedMultiSignalLoss.py tests/test_masked_loss.py
git commit -m "feat: masked multi-signal loss (mse/negpearson bases)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Parameterize DeepPhys (`in_channels` respected, `out_signals`, computed dense size)

**Files:**
- Modify: `neural_methods/model/DeepPhys.py`
- Test: `tests/test_deepphys_multisignal.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `DeepPhys(in_channels=3, out_signals=1, ..., img_size=36)`;
  forward input `(N, 2*in_channels, H, W)` (diff block then raw block);
  output `(N, out_signals)`. Existing defaults keep legacy behavior byte-compatible.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deepphys_multisignal.py
import torch

from neural_methods.model.DeepPhys import DeepPhys


def test_legacy_default_contract_unchanged():
    m = DeepPhys(img_size=72)
    out = m(torch.randn(8, 6, 72, 72))
    assert out.shape == (8, 1)


def test_five_channels_two_signals():
    m = DeepPhys(in_channels=5, out_signals=2, img_size=72)
    out = m(torch.randn(8, 10, 72, 72))
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


def test_arbitrary_img_size():
    # previously only 36/72/96 were supported via a lookup table
    m = DeepPhys(in_channels=3, out_signals=2, img_size=64)
    out = m(torch.randn(4, 6, 64, 64))
    assert out.shape == (4, 2)


def test_gradients_flow_from_both_blocks():
    m = DeepPhys(in_channels=5, out_signals=2, img_size=36)
    x = torch.randn(4, 10, 36, 36, requires_grad=True)
    m(x).sum().backward()
    g = x.grad.abs().sum(dim=(0, 2, 3))
    assert (g[:5] > 0).all()      # diff block used
    assert (g[5:] > 0).all()      # raw block used
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_deepphys_multisignal.py -q`
Expected: `test_legacy_default_contract_unchanged` PASSES; the other three FAIL (`unexpected keyword argument 'out_signals'` / `Unsupported image size`)

- [ ] **Step 3: Modify DeepPhys**

In `neural_methods/model/DeepPhys.py`:

1. Constructor signature: add `out_signals=1` after `in_channels=3`.
2. Replace the img_size lookup block:

```python
        # OLD (delete):
        # if img_size == 36:
        #     self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        # elif img_size == 72: ... elif img_size == 96: ... else: raise
        # NEW:
        h1 = (img_size - 2) // 2          # conv2 (valid) then pool /2
        h2 = (h1 - 2) // 2                # conv4 (valid) then pool /2
        self.final_dense_1 = nn.Linear(self.nb_filters2 * h2 * h2, self.nb_dense, bias=True)
```

(Formula reproduces the old table exactly: 36→3136, 72→16384, 96→30976.)

3. `self.final_dense_2 = nn.Linear(self.nb_dense, out_signals, bias=True)`
4. Forward split respects `in_channels`:

```python
        diff_input = inputs[:, :self.in_channels, :, :]
        raw_input = inputs[:, self.in_channels:2 * self.in_channels, :, :]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_deepphys_multisignal.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add neural_methods/model/DeepPhys.py tests/test_deepphys_multisignal.py
git commit -m "feat: DeepPhys respects in_channels, adds out_signals, computed dense size

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `SignalDictWrapper`

**Files:**
- Create: `neural_methods/model/SignalDictWrapper.py`
- Test: append to `tests/test_deepphys_multisignal.py`

**Interfaces:**
- Consumes: `signals.validate_traces` (Task 1); any backbone from Task 4's contract.
- Produces: `SignalDictWrapper(backbone: nn.Module, traces: list[str], input_mode: str)`;
  `forward(frames: (B, C, T, H, W)) -> dict[str, (B, T)]`.
  `input_mode='frames2d'`: flattens to `(B*T, C, H, W)`, expects backbone output `(B*T, S)`.
  `input_mode='video3d'`: passes `(B, C, T, H, W)` through, expects `(B, S, T)` (reserved for the 3D-model phase; implemented now because it is 3 lines).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_deepphys_multisignal.py`)

```python
from neural_methods.model.SignalDictWrapper import SignalDictWrapper


def test_wrapper_frames2d_dict_output():
    from neural_methods.model.DeepPhys import DeepPhys
    backbone = DeepPhys(in_channels=3, out_signals=2, img_size=36)
    w = SignalDictWrapper(backbone, ['ABP', 'CVP'], input_mode='frames2d')
    out = w(torch.randn(2, 6, 8, 36, 36))     # (B, 2*C, T, H, W)
    assert set(out) == {'ABP', 'CVP'}
    assert out['ABP'].shape == (2, 8)
    assert out['CVP'].shape == (2, 8)


def test_wrapper_video3d_dict_output():
    class Fake3D(torch.nn.Module):
        def forward(self, x):                  # (B, C, T, H, W) -> (B, S, T)
            return x.mean(dim=(3, 4))[:, :2, :]
    w = SignalDictWrapper(Fake3D(), ['ABP', 'CVP'], input_mode='video3d')
    out = w(torch.randn(2, 3, 8, 16, 16))
    assert out['ABP'].shape == (2, 8) and out['CVP'].shape == (2, 8)


def test_wrapper_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        SignalDictWrapper(torch.nn.Identity(), ['ABP'], input_mode='pointcloud')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_deepphys_multisignal.py -q`
Expected: 3 new FAIL (module not found), 4 old PASS

- [ ] **Step 3: Implement**

```python
# neural_methods/model/SignalDictWrapper.py
"""Shared dict boundary: raw backbone tensors in, {signal: (B, T)} out."""
import torch.nn as nn

from neural_methods.signals import validate_traces

INPUT_MODES = ('frames2d', 'video3d')


class SignalDictWrapper(nn.Module):
    def __init__(self, backbone, traces, input_mode):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"Unknown input_mode {input_mode!r}; known: {INPUT_MODES}")
        self.backbone = backbone
        self.traces = validate_traces(traces)
        self.input_mode = input_mode

    def forward(self, frames):
        B, C, T, H, W = frames.shape
        if self.input_mode == 'frames2d':
            x = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            out = self.backbone(x)                       # (B*T, S)
            out = out.view(B, T, -1).permute(0, 2, 1)    # (B, S, T)
        else:  # video3d
            out = self.backbone(frames)                  # (B, S, T)
        return {sig: out[:, i, :] for i, sig in enumerate(self.traces)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_deepphys_multisignal.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add neural_methods/model/SignalDictWrapper.py tests/test_deepphys_multisignal.py
git commit -m "feat: SignalDictWrapper - dict boundary for frames2d/video3d backbones

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: NeckflixLoader dict mode + collate

**Files:**
- Modify: `dataset/data_loader/NeckflixLoader.py`
- Create: `dataset/data_loader/multisignal_collate.py`
- Test: `tests/test_neckflix_dict.py`

**Interfaces:**
- Consumes: Task 1 helpers (`resolve_channels`, `resolve_traces`, `normalize_signal`, `signal_norm_overrides`, `CHANNELS`).
- Produces:
  - `NeckflixLoader(..., dict_output=False)` — when True, `__getitem__` returns the spec's dict:
    `{'frames': (C_total, T, H, W) f32, 'channel_mask': (n_slots,) f32, 'labels': {sig: (T,) f32}, 'label_mask': {sig: () f32}, 'filename': str, 'chunk_id': int}`
    where `n_slots = len(resolved CHANNELS)`, `C_total = n_slots * len([t for t in DATA_TYPE if t])`.
  - `multisignal_collate(batch: list[dict]) -> dict` (stacks tensors, lists strings).
  - In dict mode `get_cached_file_list` filters by participant/posture and requires at least one configured channel present; it does NOT filter on traces.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_neckflix_dict.py
import argparse
import textwrap

import h5py
import numpy as np
import pytest
import torch

T_FRAMES, H, W = 32, 16, 16


def _write_h5(path, channels, traces, T=T_FRAMES):
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        for ch in channels:
            g = f.create_group(ch)
            g.create_dataset('frames', data=rng.integers(0, 255, (T, H, W), dtype=np.uint8))
            g.create_dataset('timestamps', data=np.arange(T, dtype=np.int32))
            for tr in traces:
                g.create_dataset(tr, data=rng.normal(50, 10, T).astype(np.float32))


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    # P001: full channels + all traces; P002: RGB only, no CVP
    _write_h5(d / "P001_S01_R1_0_D_K1.hdf5", ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP', 'ECG'])
    _write_h5(d / "P002_S01_R1_0_D_K1.hdf5", ['R', 'G', 'B'], ['ABP', 'ECG'])
    return d


def _make_loader(cache_dir, tmp_path, channels, traces):
    from config import get_config
    from dataset.data_loader.NeckflixLoader import NeckflixLoader
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(textwrap.dedent(f"""\
        BASE: ['']
        TRAIN:
          DATA:
            CACHED_PATH: "{cache_dir}"
            PREPROCESS:
              CHUNK_LENGTH: 16
              DATA_TYPE: ['DiffNormalized','Standardized']
              RESIZE:
                H: 8
                W: 8
              CHANNELS: {channels}
              TRACES: {traces}
              NECKFLIX:
                RANDOM_CHUNK: False
    """))
    cfg = get_config(argparse.Namespace(config_file=str(yaml)))
    return NeckflixLoader(name="train", data_path=str(cache_dir),
                          config_data=cfg.TRAIN.DATA, dict_output=True)


def test_dict_contract_full_recording(cache_dir, tmp_path):
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP'])
    items = [dl[i] for i in range(len(dl))]
    full = [it for it in items if it['filename'].startswith('P001')][0]
    assert full['frames'].shape == (10, 16, 8, 8)          # 2 blocks * 5 slots
    assert full['channel_mask'].tolist() == [1, 1, 1, 1, 1]
    assert set(full['labels']) == {'ABP', 'CVP'}
    assert full['labels']['ABP'].shape == (16,)
    assert float(full['label_mask']['ABP']) == 1.0
    assert float(full['label_mask']['CVP']) == 1.0
    assert np.isfinite(full['frames']).all()


def test_zero_fill_and_masks_partial_recording(cache_dir, tmp_path):
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP'])
    partial = [dl[i] for i in range(len(dl)) if dl[i]['filename'].startswith('P002')][0]
    assert partial['channel_mask'].tolist() == [1, 1, 1, 0, 0]
    # zero-filled slots: I and D blocks are all-zero in both transform blocks
    f = partial['frames']
    assert np.abs(f[3:5]).sum() == 0 and np.abs(f[8:10]).sum() == 0
    assert np.abs(f[0:3]).sum() > 0
    # CVP missing -> zeros + mask 0
    assert float(partial['label_mask']['CVP']) == 0.0
    assert np.abs(partial['labels']['CVP']).sum() == 0
    assert float(partial['label_mask']['ABP']) == 1.0
    assert np.isfinite(f).all()


def test_partial_recordings_are_kept_not_filtered(cache_dir, tmp_path):
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP'])
    names = {dl[i]['filename'][:4] for i in range(len(dl))}
    assert names == {'P001', 'P002'}


def test_collate(cache_dir, tmp_path):
    from dataset.data_loader.multisignal_collate import multisignal_collate
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B'], ['ABP'])
    batch = multisignal_collate([dl[0], dl[1]])
    assert batch['frames'].shape[0] == 2
    assert isinstance(batch['frames'], torch.Tensor)
    assert batch['labels']['ABP'].shape[0] == 2
    assert batch['label_mask']['ABP'].shape == (2,)
    assert isinstance(batch['filename'], list) and len(batch['filename']) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_neckflix_dict.py -q`
Expected: FAIL — `unexpected keyword argument 'dict_output'`

- [ ] **Step 3: Implement dict mode**

3a. In `NeckflixLoader.__init__`, add `dict_output: bool = False` to the signature and, after `self.get_raw_resized = get_raw_resized`, insert:

```python
        self.dict_output = dict_output
        if dict_output:
            from neural_methods.signals import (resolve_channels, resolve_traces,
                                                signal_norm_overrides)
            self.slot_channels = resolve_channels(config_data)
            self.traces = resolve_traces(config_data)
            self.norm_overrides = signal_norm_overrides(config_data)
```

3b. In `get_cached_file_list`, wrap the existing channel/trace availability checks (the `with h5py.File... available_channels ... available_traces ...` block) so they only run when `not self.dict_output`; in dict mode replace them with:

```python
            if self.dict_output:
                with h5py.File(file, 'r') as f:
                    if not any(ch in f for ch in self.slot_channels):
                        continue
                selected_files.append(file)
                continue
```

(participant and posture filters above stay shared.)

3c. In `load()`, the length probe uses `CHANNELS[0]/TRACES[0]`; make it dict-aware:

```python
            with h5py.File(file, 'r') as f:
                if self.dict_output:
                    ch0 = next(ch for ch in self.slot_channels if ch in f)
                    keys = f[ch0]
                    probe = keys['timestamps'] if 'timestamps' in keys else \
                        keys[next(k for k in keys if k != 'frames')]
                    ids = probe.shape[0]
                else:
                    first_trace = f[self.config_data.PREPROCESS.NECKFLIX.CHANNELS[0]][
                        self.config_data.PREPROCESS.NECKFLIX.TRACES[0]]
                    ids = len(first_trace[~np.isnan(first_trace[:])])
```

3d. Add the two new methods (place after `load_recording`):

```python
    def load_recording_dict(self, index):
        """Dict-mode read: zero-filled channel slots + per-signal labels/masks."""
        from neural_methods.signals import normalize_signal
        h5_filepath, chunk_start_idx = self.inputs[index]
        chunk_size = self.config_data.PREPROCESS.CHUNK_LENGTH
        random_chunk = self.config_data.PREPROCESS.NECKFLIX.RANDOM_CHUNK
        slots = self.slot_channels

        with h5py.File(h5_filepath, 'r') as f:
            present = [ch in f for ch in slots]
            ch0 = slots[present.index(True)]
            keys = f[ch0]
            probe = keys['timestamps'] if 'timestamps' in keys else \
                keys[next(k for k in keys if k != 'frames')]
            n_total = probe.shape[0]

            # ----- labels over full length, averaged over channels that carry them -----
            labels_full = {}
            label_mask = {}
            for tr in self.traces:
                acc, n_carrier = None, 0
                for ch, ok in zip(slots, present):
                    if ok and tr in f[ch]:
                        arr = f[ch][tr][...].astype(np.float32)
                        acc = arr if acc is None else acc + arr
                        n_carrier += 1
                if n_carrier:
                    labels_full[tr] = acc / n_carrier
                    label_mask[tr] = 1.0
                else:
                    labels_full[tr] = np.zeros(n_total, dtype=np.float32)
                    label_mask[tr] = 0.0

            # ----- valid frames: finite across PRESENT traces only -----
            present_traces = [tr for tr in self.traces if label_mask[tr] > 0]
            if present_traces:
                finite = np.all([np.isfinite(labels_full[tr]) for tr in present_traces], axis=0)
            else:
                finite = np.ones(n_total, dtype=bool)
            valid_indices = np.flatnonzero(finite)
            n_valid = valid_indices.shape[0]
            if n_valid == 0:
                raise ValueError(f"No finite labels in {h5_filepath}.")
            chunk_len = min(chunk_size, n_valid)

            if random_chunk:
                max_start = n_valid - chunk_len
                start_pos = 0 if max_start <= 0 else np.random.randint(0, max_start + 1)
            else:
                start_pos = chunk_start_idx * chunk_size
                if start_pos >= n_valid:
                    raise IndexError(
                        f"chunk_start_idx {chunk_start_idx} out of range for {n_valid} valid frames.")
            end_pos = min(start_pos + chunk_len, n_valid)
            sel_idx = valid_indices[start_pos:end_pos]

            # ----- frames per slot (zeros where absent) -----
            frames_list = []
            for ch, ok in zip(slots, present):
                if ok:
                    frames_list.append(f[ch]['frames'][sel_idx, ...].astype(np.float32))
                else:
                    hh, ww = f[ch0]['frames'].shape[1:3]
                    frames_list.append(np.zeros((len(sel_idx), hh, ww), dtype=np.float32))

        np_input = np.stack(frames_list, axis=-1)                # (T, H, W, n_slots)
        labels = {tr: normalize_signal(labels_full[tr][sel_idx], tr, self.norm_overrides)
                       .astype(np.float32) if label_mask[tr] > 0
                  else np.zeros(len(sel_idx), dtype=np.float32)
                  for tr in self.traces}
        ch_mask = np.array(present, dtype=np.float32)
        return np_input, labels, {tr: np.float32(label_mask[tr]) for tr in self.traces}, ch_mask

    def process_item_dict(self, np_input, ch_mask):
        """Resize then CONCATENATE each DATA_TYPE transform along channels.

        np_input: (T, H, W, n_slots) -> frames: (n_types * n_slots, T, H, W)
        Zero-filled slots are re-zeroed after each transform (diffnorm/zstand
        of a constant-zero channel would otherwise produce 0/0 artifacts).
        """
        resized = self.resize_frames(np_input)                   # (T, C, H, W) torch
        mask = torch.from_numpy(ch_mask).view(1, -1, 1, 1)
        blocks = []
        for process in self.config_data.PREPROCESS.DATA_TYPE:
            if process == 'Standardized':
                block = self.zstand(resized.clone(), exclude_mask=True)
            elif process == 'DiffNormalized':
                block = self.diffnorm(resized.clone(), exclude_mask=True)
            elif process == '':
                continue
            else:
                raise ValueError(f"Unsupported preprocessing type {process}")
            blocks.append(torch.nan_to_num(block) * mask)
        frames = torch.cat(blocks, dim=1)                        # (T, C_total, H, W)
        return frames.permute(1, 0, 2, 3).contiguous().numpy()   # (C_total, T, H, W)
```

3e. At the very top of `__getitem__`, add the dict-mode branch:

```python
        if self.dict_output:
            filename = Path(self.inputs[index][0]).name
            chunk_id = self.inputs[index][1]
            np_input, labels, label_mask, ch_mask = self.load_recording_dict(index)
            frames = self.process_item_dict(np_input, ch_mask)
            return {'frames': frames, 'channel_mask': ch_mask,
                    'labels': labels, 'label_mask': label_mask,
                    'filename': filename, 'chunk_id': chunk_id}
```

3f. Create `dataset/data_loader/multisignal_collate.py`:

```python
"""Collate for the multi-signal dict batch contract."""
from torch.utils.data import default_collate

_REQUIRED = ('frames', 'channel_mask', 'labels', 'label_mask', 'filename', 'chunk_id')


def multisignal_collate(batch):
    for item in batch:
        missing = [k for k in _REQUIRED if k not in item]
        if missing:
            raise KeyError(f"multisignal batch item missing keys: {missing}")
    return default_collate(batch)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_neckflix_dict.py tests/ -q`
Expected: all pass (including the pre-existing suites — legacy loader mode untouched)

- [ ] **Step 5: Commit**

```bash
git add dataset/data_loader/NeckflixLoader.py dataset/data_loader/multisignal_collate.py tests/test_neckflix_dict.py
git commit -m "feat: NeckflixLoader dict mode - zero-filled channel slots, per-signal labels/masks, concat DATA_TYPE blocks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `MultiSignalTrainer` + model registry

**Files:**
- Create: `neural_methods/trainer/MultiSignalTrainer.py`
- Test: `tests/test_multisignal_trainer.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4, 5 (`resolve_*`, `MaskedMultiSignalLoss`, `DeepPhys(in_channels, out_signals)`, `SignalDictWrapper`).
- Produces:
  - `ModelEntry = namedtuple('ModelEntry', ['build', 'input_mode'])`
  - `MODEL_REGISTRY: dict[str, ModelEntry]` containing `'DeepPhys'`
  - `MultiSignalTrainer(config, data_loader_dict, rank=0, world_size=1, debug=False)` with `train(data_loader_dict)`, `valid(data_loader_dict) -> float`, `test(data_loader_dict)`, `save_model(index)`.
  - Test outputs: pickle at `TEST.OUTPUT_SAVE_DIR/multisignal_outputs.pickle` shaped `{sig: {'predictions': {filename: {chunk_id: np.ndarray(T,)}}, 'labels': {...same...}, 'masks': {filename: {chunk_id: float}}}}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multisignal_trainer.py
import argparse
import textwrap

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDictDataset(Dataset):
    """Emits the multisignal batch contract without touching disk."""
    def __init__(self, n=8, slots=3, blocks=2, T=16, hw=36):
        self.n, self.slots, self.blocks, self.T, self.hw = n, slots, blocks, T, hw

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        return {
            'frames': torch.randn(self.slots * self.blocks, self.T, self.hw, self.hw,
                                  generator=g),
            'channel_mask': torch.ones(self.slots),
            'labels': {'ABP': torch.randn(self.T, generator=g),
                       'CVP': torch.randn(self.T, generator=g)},
            'label_mask': {'ABP': torch.tensor(1.0),
                           'CVP': torch.tensor(1.0 if i % 2 == 0 else 0.0)},
            'filename': f"P{i:03d}_S01_R1_0_D_K1.hdf5",
            'chunk_id': 0,
        }


@pytest.fixture
def config(tmp_path):
    from config import get_config
    yaml = tmp_path / "c.yaml"
    yaml.write_text(textwrap.dedent(f"""\
        BASE: ['']
        TOOLBOX_MODE: "train_and_test"
        DEVICE: cpu
        LOG:
          PATH: "{tmp_path}/runs"
        MODEL:
          NAME: DeepPhys
          MODEL_DIR: "{tmp_path}/runs/models"
        TRAIN:
          BATCH_SIZE: 4
          EPOCHS: 2
          LR: 1e-3
          MODEL_FILE_NAME: pilot_test
          DATA:
            PREPROCESS:
              CHUNK_LENGTH: 16
              DATA_TYPE: ['DiffNormalized','Standardized']
              RESIZE:
                H: 36
                W: 36
              CHANNELS: ['R','G','B']
              TRACES: ['ABP','CVP']
        TEST:
          USE_LAST_EPOCH: True
          OUTPUT_SAVE_DIR: "{tmp_path}/runs/out"
          DATA:
            PREPROCESS:
              CHUNK_LENGTH: 16
              DATA_TYPE: ['DiffNormalized','Standardized']
              RESIZE:
                H: 36
                W: 36
              CHANNELS: ['R','G','B']
              TRACES: ['ABP','CVP']
    """))
    return get_config(argparse.Namespace(config_file=str(yaml)))


def _loaders():
    from dataset.data_loader.multisignal_collate import multisignal_collate
    mk = lambda n: DataLoader(SyntheticDictDataset(n=n), batch_size=4,
                              collate_fn=multisignal_collate)
    return {'train': mk(8), 'valid': mk(4), 'test': mk(4)}


def test_registry_has_deepphys():
    from neural_methods.trainer.MultiSignalTrainer import MODEL_REGISTRY
    assert 'DeepPhys' in MODEL_REGISTRY
    assert MODEL_REGISTRY['DeepPhys'].input_mode == 'frames2d'


def test_train_and_test_end_to_end(config):
    import os
    import pickle
    from neural_methods.trainer.MultiSignalTrainer import MultiSignalTrainer
    loaders = _loaders()
    tr = MultiSignalTrainer(config, loaders)
    tr.train(loaders)
    assert np.isfinite(tr.last_train_loss)
    model_file = os.path.join(config.MODEL.MODEL_DIR,
                              "pilot_test_Epoch1.pth")
    assert os.path.exists(model_file)
    tr.test(loaders)
    out_file = os.path.join(config.TEST.OUTPUT_SAVE_DIR, "multisignal_outputs.pickle")
    assert os.path.exists(out_file)
    with open(out_file, 'rb') as fh:
        out = pickle.load(fh)
    assert set(out) == {'ABP', 'CVP'}
    fname = next(iter(out['ABP']['predictions']))
    pred = out['ABP']['predictions'][fname][0]
    assert pred.shape == (16,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_multisignal_trainer.py -q`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```python
# neural_methods/trainer/MultiSignalTrainer.py
"""One trainer for all dict-contract (multi-signal) models.

Model-specific knowledge lives in MODEL_REGISTRY entries; everything else
(loss masking, epochs, checkpointing, saving outputs) is shared.
"""
import os
import pickle
from collections import namedtuple

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from neural_methods.loss.MaskedMultiSignalLoss import MaskedMultiSignalLoss
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.SignalDictWrapper import SignalDictWrapper
from neural_methods.signals import resolve_channels, resolve_traces
from neural_methods.trainer.BaseTrainer import BaseTrainer

ModelEntry = namedtuple('ModelEntry', ['build', 'input_mode'])


def _build_deepphys(config, in_channels, out_signals):
    return DeepPhys(in_channels=in_channels, out_signals=out_signals,
                    img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H)


MODEL_REGISTRY = {
    'DeepPhys': ModelEntry(_build_deepphys, 'frames2d'),
}


class MultiSignalTrainer(BaseTrainer):
    def __init__(self, config, data_loader_dict, rank=0, world_size=1, debug=False):
        super().__init__(rank=rank, world_size=world_size)
        self.config = config
        self.debug = debug
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        entry = MODEL_REGISTRY[config.MODEL.NAME]
        data_cfg = config.TRAIN.DATA
        self.traces = resolve_traces(data_cfg)
        self.channels = resolve_channels(data_cfg)
        backbone = entry.build(config, in_channels=len(self.channels),
                               out_signals=len(self.traces))
        self.model = SignalDictWrapper(backbone, self.traces, entry.input_mode).to(self.device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank)

        self.loss_fn = MaskedMultiSignalLoss(self.traces, base='mse').to(self.device)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.min_valid_loss = None
        self.best_epoch = 0
        self.last_train_loss = float('nan')

        if config.TOOLBOX_MODE == "train_and_test":
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=max(1, len(data_loader_dict["train"])))

    def _to_device(self, batch):
        frames = batch['frames'].to(self.device).float()
        labels = {s: batch['labels'][s].to(self.device).float() for s in self.traces}
        mask = {s: batch['label_mask'][s].to(self.device).float() for s in self.traces}
        return frames, labels, mask

    def train(self, data_loader):
        for epoch in range(self.max_epoch_num):
            self.model.train()
            losses = []
            for batch in tqdm(data_loader["train"], ncols=80,
                              desc=f"Train epoch {epoch}", disable=not self.is_main):
                frames, labels, mask = self._to_device(batch)
                self.optimizer.zero_grad()
                preds = self.model(frames)
                loss = self.loss_fn(preds, labels, mask)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                losses.append(loss.item())
            self.last_train_loss = float(np.mean(losses)) if losses else float('nan')
            if self.is_main:
                tqdm.write(f"epoch {epoch}: train loss {self.last_train_loss:.4f}")
                self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH and data_loader.get("valid"):
                valid_loss = self.valid(data_loader)
                if self.is_main:
                    tqdm.write(f"epoch {epoch}: valid loss {valid_loss:.4f}")
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
        if self.is_main and not self.config.TEST.USE_LAST_EPOCH:
            tqdm.write(f"best epoch {self.best_epoch} (valid loss {self.min_valid_loss:.4f})")

    def valid(self, data_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in data_loader["valid"]:
                frames, labels, mask = self._to_device(batch)
                preds = self.model(frames)
                losses.append(self.loss_fn(preds, labels, mask).item())
        return float(np.mean(losses)) if losses else float('nan')

    def test(self, data_loader):
        if self.config.TOOLBOX_MODE == "only_test":
            path = self.config.INFERENCE.MODEL_PATH
        elif self.config.TEST.USE_LAST_EPOCH:
            path = os.path.join(self.model_dir,
                                f"{self.model_file_name}_Epoch{self.max_epoch_num - 1}.pth")
        else:
            path = os.path.join(self.model_dir,
                                f"{self.model_file_name}_Epoch{self.best_epoch}.pth")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

        out = {s: {'predictions': {}, 'labels': {}, 'masks': {}} for s in self.traces}
        with torch.no_grad():
            for batch in data_loader["test"]:
                frames, labels, mask = self._to_device(batch)
                preds = self.model(frames)
                for i, fname in enumerate(batch['filename']):
                    chunk = int(batch['chunk_id'][i])
                    for s in self.traces:
                        out[s]['predictions'].setdefault(fname, {})[chunk] = \
                            preds[s][i].cpu().numpy()
                        out[s]['labels'].setdefault(fname, {})[chunk] = \
                            labels[s][i].cpu().numpy()
                        out[s]['masks'].setdefault(fname, {})[chunk] = \
                            float(mask[s][i])
        if self.is_main:
            os.makedirs(self.config.TEST.OUTPUT_SAVE_DIR, exist_ok=True)
            out_path = os.path.join(self.config.TEST.OUTPUT_SAVE_DIR,
                                    "multisignal_outputs.pickle")
            with open(out_path, 'wb') as fh:
                pickle.dump(out, fh)
            tqdm.write(f"saved multisignal outputs to {out_path}")

    def save_model(self, index):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{index}.pth")
        torch.save(self.model.state_dict(), path)
```

Note: `BaseTrainer.__init__(rank, world_size)` already exists (DDP migration) and sets `self.is_main`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/test_multisignal_trainer.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add neural_methods/trainer/MultiSignalTrainer.py tests/test_multisignal_trainer.py
git commit -m "feat: MultiSignalTrainer + model registry (DeepPhys, frames2d)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Wire `neckflix_main.py`, pilot config, end-to-end smoke

**Files:**
- Modify: `neckflix_main.py`
- Create: `physhydra_configs/deepphys_neckflix.yaml`
- Modify: `docs/changelog.md`

**Interfaces:**
- Consumes: Tasks 6-7 (`dict_output`, `multisignal_collate`, `MODEL_REGISTRY`, `MultiSignalTrainer`), Task 1 resolvers.
- Produces: `uv run python neckflix_main.py --config_file physhydra_configs/deepphys_neckflix.yaml --test_participants P030` trains DeepPhys on the local subset.

- [ ] **Step 1: Modify `neckflix_main.py`**

1a. Add imports (after the existing `from neural_methods import trainer` line):

```python
from neural_methods.trainer.MultiSignalTrainer import MODEL_REGISTRY, MultiSignalTrainer
from neural_methods.signals import resolve_channels, resolve_traces
from dataset.data_loader.multisignal_collate import multisignal_collate
```

1b. In `train_and_test(...)` and `test(...)`, add BEFORE the `if config.MODEL.NAME == "Physnet":` chain:

```python
    if config.MODEL.NAME in MODEL_REGISTRY:
        model_trainer = MultiSignalTrainer(config, data_loader_dict, rank=rank,
                                           world_size=world_size, debug=config.DEBUG)
        model_trainer.train(data_loader_dict)   # train_and_test variant only
        model_trainer.test(data_loader_dict)
        return
```

(in the `test(...)` function, only the `.test(...)` call.)

1c. In `main()`, the channels/traces/postures naming block currently reads
`config.TRAIN.DATA.PREPROCESS.NECKFLIX.CHANNELS` / `.TRACES`. Replace those reads
with the resolvers so both old and new configs work:

```python
    channels = ''.join(resolve_channels(config.TRAIN.DATA))
    traces = '-'.join(resolve_traces(config.TRAIN.DATA))
```

(keep the equality checks by comparing `resolve_*(config.TRAIN.DATA)` with
`resolve_*(config.TEST.DATA)`).

1d. Where the three `NeckflixLoader(...)` instances are constructed, add:

```python
    dict_mode = config.MODEL.NAME in MODEL_REGISTRY
```

and pass `dict_output=dict_mode` to each loader. Where the train/valid/test
`DataLoader(...)`s are constructed, pass `collate_fn=multisignal_collate if dict_mode else None`.

- [ ] **Step 2: Create the pilot config**

```yaml
# physhydra_configs/deepphys_neckflix.yaml
BASE: ['']
TOOLBOX_MODE: "train_and_test"

TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 5
  LR: 9e-3
  MODEL_FILE_NAME: neckflix_deepphys_multisignal
  PLOT_LOSSES_AND_LR: False
  DATA:
    BEGIN: 0.0
    END: 0.8
    FS: 30
    DATASET: Neckflix
    DATA_FORMAT: NCDHW
    DATA_PATH: "/Users/20759193/repos/Neckflix/dataset"
    CACHED_PATH: "/Users/20759193/repos/Neckflix/cached_dir"
    EXP_DATA_NAME: "deepphys_pilot"
    PREPROCESS:
      CHUNK_LENGTH: 128
      DATA_TYPE: ['DiffNormalized','Standardized']
      RESIZE:
        W: 72
        H: 72
      CHANNELS: ['R','G','B']
      TRACES: ['ABP','CVP']
      NECKFLIX:
        RANDOM_CHUNK: True

VALID:
  DATA:
    BEGIN: 0.8
    END: 1.0
    FS: 30
    DATASET: Neckflix
    DATA_FORMAT: NCDHW
    DATA_PATH: "/Users/20759193/repos/Neckflix/dataset"
    CACHED_PATH: "/Users/20759193/repos/Neckflix/cached_dir"
    EXP_DATA_NAME: "deepphys_pilot"
    PREPROCESS:
      CHUNK_LENGTH: 128
      DATA_TYPE: ['DiffNormalized','Standardized']
      RESIZE:
        W: 72
        H: 72
      CHANNELS: ['R','G','B']
      TRACES: ['ABP','CVP']
      NECKFLIX:
        RANDOM_CHUNK: True

TEST:
  METRICS: []
  USE_LAST_EPOCH: False
  DATA:
    BEGIN: 0.0
    END: 1.0
    FS: 30
    DATASET: Neckflix
    DO_PREPROCESS: False
    DATA_FORMAT: NCDHW
    DATA_PATH: "/Users/20759193/repos/Neckflix/dataset"
    CACHED_PATH: "/Users/20759193/repos/Neckflix/cached_dir"
    EXP_DATA_NAME: "deepphys_pilot"
    PREPROCESS:
      CHUNK_LENGTH: 128
      DATA_TYPE: ['DiffNormalized','Standardized']
      RESIZE:
        W: 72
        H: 72
      CHANNELS: ['R','G','B']
      TRACES: ['ABP','CVP']
      NECKFLIX:
        RANDOM_CHUNK: False

DEVICE: cpu
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: runs/
MODEL:
  NAME: DeepPhys

INFERENCE:
  BATCH_SIZE: 4
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
  MODEL_PATH: ""
```

- [ ] **Step 3: Full test suite**

Run: `PYTHONPATH=/Users/20759193/repos/rPPG-Toolbox uv run python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 4: End-to-end smoke on the real subset (short)**

Run (expect several minutes; watch first epoch only, then Ctrl-C is acceptable —
the acceptance bar is: loaders build, first training batches step with finite
decreasing loss, checkpoint file appears):

```bash
uv run python neckflix_main.py --config_file physhydra_configs/deepphys_neckflix.yaml --test_participants P030
```

Expected console: dataset sizes for train/valid/test; per-epoch train loss finite.

- [ ] **Step 5: Update `docs/changelog.md`** — add under a new `## 2026-08-25 — Branch: main (multi-signal pilot)` heading:

```markdown
### Changes
- Multi-signal dict pipeline (DeepPhys pilot): canonical signal registry
  (PPG/ECG/ABP/CVP/RESP/EDA/SPO2 + HR eval-only), zero-fill+mask channels,
  masked multi-signal loss, SignalDictWrapper, MultiSignalTrainer + registry,
  NeckflixLoader dict mode with DATA_TYPE concatenation.
  Spec: docs/superpowers/specs/2026-08-25-multisignal-pipeline-design.md
```

- [ ] **Step 6: Commit**

```bash
git add neckflix_main.py physhydra_configs/deepphys_neckflix.yaml docs/changelog.md
git commit -m "feat: wire multi-signal pipeline into neckflix_main; DeepPhys pilot config

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
