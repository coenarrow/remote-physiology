# Model Regularisation Terms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a backbone contribute its own loss terms, weighted from an optional `REGULARISATION` mapping in the `MODEL` section, with "left out means off".

**Architecture:** A backbone declares `REGULARISERS` and exposes `regularisers()` beside `output_layers()`; the wrapper forwards every term per trace in its output dict; the trainer, now handed the model config, keeps every weight and merges only the named terms into the per-trace raw loss dict before `weight_losses`. No new class, loop or log path.

**Tech Stack:** Python 3, PyTorch, dataclass configs (`src/config.py:build`), `uv run`.

**Spec:** `docs/plans/2026-09-18-model-regularisation-design.md`

## Global Constraints

- No tests: this repo has no `tests/` directory by decision. Each task's check is a `uv run python -c` probe or the README run command, never a new test file.
- Reshaping uses einops; none is needed here.
- `REGULARISATION` is the only optional key of `MODEL`; every other key stays required and unknown keys stay refused.
- Term names are lower-case internally (like loss components) and upper-case in YAML.
- The nine migrated backbones are not edited.
- Commit after each task with the attribution line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

### Task 1: `REGULARISATION` on `ModelConfig`

**Files:**
- Modify: `src/model_config.py:63-70` (`ModelConfig`), `:77-80` (`TemporalShiftConfig.validate`), `:104-114` (`parse_model_config`)
- Modify: `configs/_model_config_template.yaml:14-26` (the `MODEL` comment block)

**Interfaces:**
- Produces: `ModelConfig.REGULARISATION: dict` — `{term_lower: float > 0}` after `validate`; `normalise_regularisation(mapping, where) -> dict` in `src/model_config.py`.

- [ ] **Step 1: Add the field and the normaliser**

Replace lines 63-70 of `src/model_config.py` with:

```python
@dataclass
class ModelConfig:
    """An architecture's section: ``NAME`` plus the switches an experiment may
    flip. ``REGULARISATION`` is the one optional key: the terms of the
    backbone's own regularisers that count and their weights, ``{}`` or
    absent for none (a term left out is off, never zeroed)."""
    NAME: str = ""
    REGULARISATION: dict = field(default_factory=dict)   # {TERM: weight > 0}

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        self.REGULARISATION = normalise_regularisation(self.REGULARISATION, where)


#: The one key of the MODEL section a file may leave out.
OPTIONAL_MODEL_KEYS = ("REGULARISATION",)


def normalise_regularisation(weights, where: str) -> dict:
    """``{TERM: weight}`` (YAML spelling) -> ``{term: float}``, lower-case keys,
    every weight a positive number. Which terms exist is the backbone's to
    say; ``src.models`` checks the names when it builds the model."""
    if weights is None:
        weights = {}
    if not isinstance(weights, dict):
        raise ConfigError(
            f"{where}: REGULARISATION must be a mapping of term to weight, got "
            f"{weights!r}")
    out = {}
    for term, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or weight <= 0:
            raise ConfigError(
                f"{where}: REGULARISATION.{term} must be a positive number "
                f"(a term that should not count is left out, not zeroed), got "
                f"{weight!r}")
        out[str(term).lower()] = float(weight)
    return out
```

- [ ] **Step 2: Make the subclass validate call the base**

Replace `TemporalShiftConfig.validate` (lines 77-80 before the edit) with:

```python
    def validate(self, interface: InterfaceConfig, where: str) -> None:
        super().validate(interface, where)
        if self.FRAME_DEPTH <= 0:
            raise ConfigError(
                f"{where}: FRAME_DEPTH must be positive, got {self.FRAME_DEPTH}")
```

`FactorizePhysConfig` does not override `validate`, so it inherits the new one.

- [ ] **Step 3: Make the key optional in the parser**

In `parse_model_config`, change

```python
    cfg = build(MODEL_CONFIGS[arch], mapping, where)
```

to

```python
    cfg = build(MODEL_CONFIGS[arch], mapping, where, optional=OPTIONAL_MODEL_KEYS)
```

- [ ] **Step 4: Document the key in the template**

In `configs/_model_config_template.yaml`, after the `FactorizePhys ... FSAM` comment lines and before `MODEL:`, add:

```yaml
# One key is optional for every NAME:
#   REGULARISATION: {<TERM>: <number > 0>, ...}   the backbone's own loss terms
#                                                 (penalties on its internals, not
#                                                 on the prediction) that count,
#                                                 and their weights; absent or {}
#                                                 means none. A term left out is
#                                                 off, never zeroed. The terms a
#                                                 backbone has are its REGULARISERS
#                                                 (none of the nine above has any).
```

and inside the `MODEL:` block, after the `NAME:` line, add:

```yaml
  REGULARISATION: {}             # optional; see above
```

- [ ] **Step 5: Check it loads**

Run:

```bash
uv run python -c "
from src.model_config import load_config
i, m, t = load_config('configs/original_model_config/physmamba_FS30_W4.27S4.27_RGB_PPG_H128W128.yaml')
print(m)
from src.model_config import parse_model_config
print(parse_model_config({'NAME': 'PhysMamba', 'REGULARISATION': {'SPARSITY': 0.1}}, i, 'x'))
try:
    parse_model_config({'NAME': 'PhysMamba', 'REGULARISATION': {'SPARSITY': 0}}, i, 'x')
except Exception as e: print('refused:', e)
"
```

(Use whatever the PhysMamba paper file is actually named under `configs/original_model_config/`; `ls` it first.)
Expected: the first print shows `REGULARISATION={}`; the second `{'sparsity': 0.1}`; the third prints `refused: x: REGULARISATION.SPARSITY must be a positive number ...`.

- [ ] **Step 6: Commit**

```bash
git add src/model_config.py configs/_model_config_template.yaml
git commit -m "feat(config): optional REGULARISATION mapping on the MODEL section

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The wrapper forwards regularisers and the builder checks the names

**Files:**
- Modify: `src/models.py:36-45` (add `_require_regularisers` beside `_require_min_frame`), `:106-110` (`MultiTraceModel.forward`), `:119-121` (`_multi_trace`) and every `_multi_trace(...)` call in the builders at `:127-184`

**Interfaces:**
- Consumes: `ModelConfig.REGULARISATION` from Task 1.
- Produces: `out["regularisers"]: {trace: {term: () tensor}}` in `MultiTraceModel.forward`; backbone convention `REGULARISERS: tuple[str, ...]` class attribute and `regularisers() -> dict` method; `_multi_trace(make_copy, interface, cfg)`.

- [ ] **Step 1: Add the name check**

After `_require_frame_size` (ends at line 55) add:

```python
def _require_regularisers(cfg: ModelConfig, copy: nn.Module) -> None:
    """The terms the config weights must be terms the backbone computes.

    A backbone declares them as a class attribute ``REGULARISERS`` and returns
    them from ``regularisers()`` after each forward; a backbone with neither
    has none, so any non-empty mapping is refused by name.
    """
    known = tuple(getattr(type(copy), "REGULARISERS", ()))
    unknown = sorted(t for t in cfg.REGULARISATION if t not in known)
    if unknown:
        have = f"has {list(known)}" if known else "has no regularisers"
        raise ConfigError(
            f"{cfg.NAME}: REGULARISATION names {[u.upper() for u in unknown]} "
            f"but the backbone {have}")
```

- [ ] **Step 2: Forward the terms in the wrapper**

Replace `MultiTraceModel.forward`:

```python
    def forward(self, batch: dict) -> dict:
        out = self.forward_video(self.prepare_frames(batch))
        predictions = {trace: out[:, i] for i, trace in enumerate(self.traces)}
        return {**batch, "predictions": predictions,
                "regularisers": self.collect_regularisers()}

    def collect_regularisers(self) -> dict:
        """``{trace: {term: () tensor}}`` each copy computed on its last forward,
        every term it declares; which ones count is the trainer's, from the
        model config. ``{}`` per trace for a backbone with none."""
        return {trace: dict(copy.regularisers()) if hasattr(copy, "regularisers") else {}
                for trace, copy in self.copies.items()}
```

Also extend the class docstring's contract paragraph: after "returns ``(B, 1, T)``;" add "a backbone with regularisers also declares ``REGULARISERS`` and returns them from ``regularisers()`` after each forward;".

- [ ] **Step 3: Route the config through `_multi_trace`**

Replace `_multi_trace`:

```python
def _multi_trace(make_copy, interface: InterfaceConfig, cfg: ModelConfig) -> MultiTraceModel:
    model = MultiTraceModel(make_copy=make_copy, channels=interface.CHANNELS,
                            traces=interface.TRACES)
    _require_regularisers(cfg, next(iter(model.copies.values())))
    return model
```

Then add `cfg` as the last argument to all nine `_multi_trace(...)` calls, e.g.

```python
    return _multi_trace(lambda: physmamba.PhysMamba(in_channels=width), interface, cfg)
```

and for the multi-line calls (EfficientPhys, FactorizePhys, TS-CAN) change the trailing `interface)` to `interface, cfg)`.

- [ ] **Step 4: Check the dict and the refusal**

Run:

```bash
uv run python -c "
import torch
from src.model_config import load_config, parse_model_config
from src.models import build_model
i, m, t = load_config('configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml')
model = build_model(m, i)
B, T = 2, i.window_frames
batch = {'frames': {ch: torch.rand(B, T, 72, 72) for ch in i.CHANNELS}}
out = model(batch)
print(out['regularisers'])
try:
    build_model(parse_model_config({'NAME': 'DeepPhys', 'REGULARISATION': {'SPARSITY': 0.1}}, i, 'x'), i)
except Exception as e: print('refused:', e)
"
```

Expected: `{'PPG': {}}` then `refused: DeepPhys: REGULARISATION names ['SPARSITY'] but the backbone has no regularisers`.

- [ ] **Step 5: Commit**

```bash
git add src/models.py
git commit -m "feat(models): wrapper forwards backbone regularisers, builder checks their names

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The trainer owns the weights and merges the named terms

**Files:**
- Modify: `neural_methods/loss/PerSignalLoss.py:296-327` (`weight_losses`)
- Modify: `src/trainer.py:1-44` (docstring), `:62` (import), `:223-244` (`__init__`), `:264-268` (`_losses`)
- Modify: `scripts/run.py:130`

**Interfaces:**
- Consumes: `out["regularisers"]` from Task 2, `ModelConfig.REGULARISATION` from Task 1.
- Produces: `Trainer(model, interface, model_config, training, run, runtime, run_dir, config=None)`; `Trainer.weights: {trace: {component_or_term: float}}`.

- [ ] **Step 1: Drop the 1.0 default in `weight_losses`**

In `neural_methods/loss/PerSignalLoss.py`, change the docstring's "Reads" entry to:

```
    Reads    : ``raw`` = {module: {component: () tensor}} (unweighted,
               graph-attached), ``weights`` = {module: {component: float}};
               every component in ``raw`` must have a weight — the callers
               compute only what the config names, so a missing weight is a
               programming error, not a default.
```

and change

```python
            term = module_weights.get(component, 1.0) * value
```

to

```python
            term = module_weights[component] * value
```

- [ ] **Step 2: Hand the trainer the model config**

In `src/trainer.py` line 62 change the import to:

```python
from src.model_config import InterfaceConfig, ModelConfig, RunSettings, TrainingConfig
```

Change the constructor signature to:

```python
    def __init__(self, model: MultiTraceModel, interface: InterfaceConfig,
                 model_config: ModelConfig, training: TrainingConfig,
                 run: RunSettings, runtime: Runtime, run_dir: Path,
                 config: dict | None = None):
        check_window(model, interface)
        self.interface = interface
        self.model_config = model_config
```

and directly after the `self.criterion = PerSignalLoss(...)` line add:

```python
        # Every weight, loss and regulariser, per trace: the interface's LOSS
        # block plus the model config's REGULARISATION. A term not named here
        # is not merged in _losses, so it is off.
        self.weights = {trace: {**components, **model_config.REGULARISATION}
                        for trace, components in self.criterion.weights.items()}
```

- [ ] **Step 3: Merge the named terms in `_losses`**

Replace `_losses`:

```python
    def _losses(self, out: dict) -> tuple:
        """``(total, weighted)``: the scalar to backpropagate and the per-trace
        per-component floats for logging. The loss components come from the
        criterion, weighted by the interface; the backbone's regularisers that
        the model config names are merged in beside them, per trace, and
        weighted by it. A term the config leaves out is dropped here."""
        raw = self.criterion(out["predictions"], out["labels"], out["label_mask"])
        named = self.model_config.REGULARISATION
        for trace, terms in out["regularisers"].items():
            raw[trace].update({term: value for term, value in terms.items()
                               if term in named})
        return weight_losses(raw, self.weights)
```

- [ ] **Step 4: Update the module docstring**

In the bullet list at the top of `src/trainer.py`, replace

```
* the **model** says nothing — it is a function from frames to predictions.
```

with

```
* the **model config** says which of the backbone's own regularisers count
  and their weights (``REGULARISATION``); the model itself is a function
  from frames to predictions that may also report those terms.
```

- [ ] **Step 5: Update the caller**

In `scripts/run.py` line 130:

```python
        trainer = Trainer(model, interface, model_config, training, run, runtime, run_dir, config)
```

- [ ] **Step 6: Check one step end to end**

Run:

```bash
uv run python -c "
import torch
from src.model_config import load_config, parse_model_config
from src.models import build_model
from src.trainer import Trainer
from neural_methods.loss.PerSignalLoss import weight_losses
raw = {'PPG': {'mse': torch.tensor(1.0)}}
print(weight_losses(raw, {'PPG': {'mse': 2.0}}))
try: weight_losses(raw, {'PPG': {}})
except KeyError as e: print('KeyError as intended:', e)
"
```

Expected: `(tensor(2.), {'PPG': {'mse': 2.0, 'total': 2.0}})` then the KeyError line.

Then hand the user the README PhysMamba command with `--limit-windows 8` appended (the user runs training themselves). Expected: the run reaches evaluation and `losses.csv` has the same columns as a run before this change (`epoch, total, PPG/negpearson, PPG/total, seconds`).

- [ ] **Step 7: Commit**

```bash
git add neural_methods/loss/PerSignalLoss.py src/trainer.py scripts/run.py
git commit -m "feat(trainer): weight and merge the model config's regularisers per trace

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Document the contract

**Files:**
- Modify: `docs/adding_a_model.md:132-142` (after the `output_layers()` subsection), `:228-245` (the config class subsection)

- [ ] **Step 1: Add the backbone-side subsection**

After the `output_layers()` subsection (ends "The trainer refuses a model whose readouts do not match its traces one to one.") add:

````markdown
### `regularisers()` (only if the model has any)

A regulariser is a scalar the backbone computes during `forward` from its
own internals — an attention map's sparsity, say — that the loss cannot
see. Most models have none. One that does declares the names and returns
the values beside its readouts:

```python
class MyNet(nn.Module):
    REGULARISERS = ("sparsity",)     # lower-case; the YAML keys of REGULARISATION

    def forward(self, x):
        ...
        self._regularisers = {"sparsity": sparsity}     # () tensors, graph-attached
        return out

    def regularisers(self):
        return self._regularisers
```

`forward` still returns the one tensor. The wrapper reads the terms back
per copy, so each trace's copy has its own values and `losses.csv` logs
them per trace (`ABP/sparsity`). The backbone computes every term it
declares; which ones count is the config's `REGULARISATION` (step 2), and
a term left out is off.
````

- [ ] **Step 2: Add the config-side paragraph**

In "The config class (only if the model has a switch)", after the bullet list, add:

```markdown
Every `MODEL` section also accepts one optional key, `REGULARISATION`,
without any class of its own: `{TERM: weight > 0}` naming which of the
backbone's `REGULARISERS` count and how much. Absent or `{}` means none;
a term left out is off, never zeroed, so an ablation is deleting a key.
The builder refuses a name the backbone does not declare. The weights are
the trainer's: they are merged beside the interface's `LOSS` weights per
trace, and the total stays the mean over traces of each trace's weighted
sum.
```

- [ ] **Step 3: Commit**

```bash
git add docs/adding_a_model.md
git commit -m "docs: regularisers in the backbone and config contract

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
