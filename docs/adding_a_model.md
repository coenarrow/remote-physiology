# Adding a model

How to put a new architecture on the multi-signal contract, whether you are
writing it from scratch or migrating one of the upstream rPPG-Toolbox models.
DeepPhys is the worked example throughout: it is the simplest of the nine
models on the contract today (DeepPhys, EfficientPhys, FactorizePhys,
PhysFormer, PhysMamba, PhysNet, RhythmFormer, TS-CAN, iBVPNet), and every
file it touches is the file yours will touch.

The authority on *how* a model is run is [`scripts/run.py`](../scripts/run.py)
and the modules it imports from [`src/`](../src/). `main.py` and the
`neural_methods/trainer/` package are legacy and are not what this guide
describes.

## What a model is here

A model is **one complete copy of a single-trace architecture per predicted
trace**, wrapped in `MultiTraceModel` ([`src/models.py`](../src/models.py)),
speaking the batch dict: frames in, predictions out, nothing else.

- **Every width comes from the interface**, never from the model's YAML. The
  first layer takes the interface's channel count, and there is one copy of
  the network per entry of `TRACES`. Layer sizes are defined once, in the
  architecture's module, at their published values.
- **The input is raw and the model normalises it.** The dataset resizes and
  nothing else; whatever the paper fed its network (standardised frames,
  frame differences) is the backbone's own first stage, taken from
  [`neural_methods/model/modules/`](../neural_methods/model/modules/). There
  is no input-preprocessing switch anywhere in config.
- **The loss is the trainer's, not the model's.** The interface's `LOSS` block
  states it per trace; a model that computes its own loss is wrong.
- **The trainer reaches into the model in two places**: the readouts
  returned by `output_layers()`, whose bias it seeds with the trace's
  physiological prior and which it exempts from weight decay; and, for a
  model that declares `REGULARISERS`, the terms returned by `regularisers()`,
  which it weights from the model config's `REGULARISATION`.

Because the wrapper owns channel order, trace order and the dict, an
architecture never sees a dict at all. It sees a tensor and returns a tensor.

## The six things you touch

| # | What | Where | Exists for DeepPhys as |
| --- | ------ | ------- | ------------------------ |
| 1 | The backbone, a plain `nn.Module` | `neural_methods/model/<Name>.py` | [`neural_methods/model/DeepPhys.py`](../neural_methods/model/DeepPhys.py) |
| 2 | One registry line (+ a config class only if the model has a switch) | `src/model_config.py` | the `"DeepPhys"` entry of `MODEL_CONFIGS`, mapped to the shared `ModelConfig` |
| 3 | Builder + one registry line | `src/models.py` | `_build_deepphys`, the `"DeepPhys"` entry of `MODEL_BUILDERS` |
| 4 | The paper config: `MODEL`, `INTERFACE` and `TRAIN` in one file | `configs/original_model_config/<name>_<interface>.yaml` | [`deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml`](../configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml) |
| 5 | One smoke test | `tests/test_<name>.py` | (not yet written; see step 4) |
| 6 | The PURE command that proves it runs | `README.md`, "Algorithms" | the DeepPhys line |

Nothing else. No new trainer, loader, loss, dataset or plot. If your model
needs something the shared pieces almost do, extend the shared piece for
every model rather than adding a parallel copy beside it.

Copyable starting points:

- [`neural_methods/model/_template.py`](../neural_methods/model/_template.py)
  for the backbone (step 1);
- [`configs/_model_config_template.yaml`](../configs/_model_config_template.yaml)
  for the paper config (step 3): every key with its type and the values
  the parser accepts; DeepPhys's
  [config file](../configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml)
  is the filled-in example;
- the code blocks in steps 2 and 4 below for the registration and the test.

## Step 1: the backbone

**File:** `neural_methods/model/<Name>.py`, one architecture per file, named
after the architecture as its paper names it.

The backbone is an `nn.Module` with no knowledge of dicts, traces or
channels. Its whole contract is:

### Constructor

```python
def __init__(self, in_channels: int = 3, ...published sizes as defaults...):
```

- `in_channels` is **required** and is the only width the wrapper always
  passes. It is `len(interface.CHANNELS)`: the interface's channels stacked
  in order, raw. A two-branch network (DeepPhys, TS-CAN) does not receive two
  blocks; it takes the one raw clip and derives its motion and appearance
  inputs itself.
- Any other width the interface determines (`img_size` for DeepPhys, whose
  dense layer is sized from the frame) is a constructor argument too, passed
  by the builder.
- Every published hyperparameter (filter counts, dropout, kernel sizes) is a
  default in the signature and is **not** exposed through YAML. At the
  defaults the module is the published network, layer for layer.

### `forward`

One shape: `(B, C_in, T, H, W)`, a whole raw clip, in; `(B, 1, T)` out.

A published network that ran on single frames (DeepPhys) still takes the
clip, because its frame difference needs the time axis; it normalises the
clip, folds `(b t)` into the batch axis for its 2D layers and unfolds the
prediction back. The temporal-shift models (TS-CAN, EfficientPhys) need the
clip for the same reason and also so the shift knows where each clip starts
and ends. The shift itself is the shared `TSM` in
[`neural_methods/model/shared.py`](../neural_methods/model/shared.py). That
module is where every piece more than one backbone needs lives —
`nearest_multiple`, `dense_width`, `min_frame_message`,
`require_min_frame`, `Attention_mask`, `TSM` — and it is where a new shared
piece belongs.

The output is one trace, width one. Never widen the readout to several
signals and never add per-signal heads on a shared trunk: the wrapper makes
the copies. Return three dimensions, `(B, 1, T)`, not `(B, T)`; the wrapper
concatenates copies on axis 1.

### Input normalisation

The input is raw. The dataset resizes to `RESIZE` and hands over pixel
values; the backbone applies the paper's `DATA_TYPE` itself as its first
stage, one `nn.Module` from
[`neural_methods/model/modules/`](../neural_methods/model/modules/):

| Paper `DATA_TYPE` | First stage | Who |
| ------------------- | ------------- | ----- |
| `DiffNormalized` | `DiffNormalize()` | PhysFormer, PhysMamba, PhysNet |
| `Standardized` | `Standardize()` | EfficientPhys, RhythmFormer |
| `DiffNormalized` + `Standardized` | both, one per branch | DeepPhys, TS-CAN |
| `Raw` | nothing (the network differences internally) | FactorizePhys, iBVPNet |

Both modules take `(B, C, T, H, W)` and normalise each (sample, channel)
block with its own statistics over the clip, the upstream toolbox's
`diff_normalize_data` and `standardized_data` formulas. Store the stage as
an attribute (`self.input_norm`, or `self.motion_norm` / `self.appearance_norm`
for two branches) and apply it as the first line of `forward`.

### `output_layers()`

```python
def output_layers(self):
    return (self.final_layer,)
```

The activation-free readout(s) of this one copy, as a tuple. Each must be a
layer whose `bias` has exactly one element (an `nn.Linear(k, 1)` or an
`nn.Conv1d(k, 1, ...)`). The trainer refuses a model whose readouts do not
match its traces one to one.

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

One caution for a distributed run: the trainer wraps the model in
`DistributedDataParallel` with `find_unused_parameters=False`, so a
parameter that feeds only a regulariser and nothing else has no gradient
when that term is left out of `REGULARISATION`, and DDP refuses the step.
Route every such parameter into the prediction too, or keep its term
weighted.

### Any frame size, any window length

Every model must accept whatever `RESIZE` and `WINDOW_SECONDS` an interface
states, because the point of a standard interface is that every model runs
on the same one. Upstream code hard-codes things that break this: a token
grid written as `view(B, C, P // 16, 4, 4)` (true only at 128x128), a
`frame=160` baked into a head, a temporal stride the window must divide.
Each of those becomes one of:

- a **constructor argument derived from the interface** by the builder
  (`img_size` for DeepPhys's dense layer), so the module is sized for the
  frames it will actually see;
- a **config switch** when the paper leaves it free (a patch size);
- an **adaptive stage around the published network** that is an exact no-op
  at the paper's shape. PhysFormer average-pools the stem's output to the
  nearest whole tube and reads the token grid off the result; PhysFormer and
  PhysMamba pool time to a multiple of their stride before the trunk and
  interpolate the prediction back to the window length after it. At the
  paper's frame size and window every one of those is skipped, so the paper
  path is the published network, bit for bit, and no parameter is added or
  renamed.

Prefer the adaptive stage for anything the architecture cannot derive: it
keeps one module serving every interface. What is never acceptable is a
silent crop, truncation or reinterpretation.

The one refusal that remains is a frame the stem pools to nothing. Each
pooling backbone states its own floor as a module-level `MIN_FRAME`:

| Model | `MIN_FRAME` | Why |
| ------- | ------------- | ----- |
| PhysFormer | 8 | three 2x spatial pools in the stem |
| PhysMamba | 16 | the stem's two spatial pools before the streams |
| PhysNet | 16 | four 2x spatial pools before the bottleneck |
| RhythmFormer | 16 | a 4x stem and a 4x patch embedding |
| FactorizePhys | 23 | five valid convolutions, two of them strided |
| iBVPNet | 64 | the encoder's pools, the stride-1 pool and two strided convs |

The builder names it via `_require_min_frame` when the interface resizes;
the module raises the same sentence at forward time, both taking it from
`neural_methods.model.shared.min_frame_message`.

The other refusal that stays is for a dense-layer model. DeepPhys, TS-CAN
and EfficientPhys size their dense head from the frame, so they need an
interface that states a `RESIZE`; a run with no resize cannot tell them how
wide that layer is. The frame need not be square — the width is derived per
axis by `neural_methods.model.shared.dense_width`.

Two interim class attributes exist for a migration that is not there yet:

```python
temporal_divisor = 4     # the window length must be a multiple of this
temporal_length = 128    # the window length must be exactly this
```

These are an interim stop for a migration in progress, not a destination, and
none of the nine migrated models declares either any more — every one reached
the adaptive stage described above. `src/trainer.py` still honours them: it
reads them off the first copy and refuses a mismatched `WINDOW_SECONDS`
rather than truncating.

### House rules

- Tensor reshaping uses einops (`rearrange`, `reduce`, `einsum`), not
  `view` / `permute` / `reshape`. This applies to migrated code too.
- No `params` argument, no loss, no `get_config`, no device handling.
- Nothing about traces or channels by name. A backbone cannot tell ABP from
  CVP and must not try.

## Step 2: registration in `src/model_config.py` and `src/models.py`

Two additions, sometimes three, each beside its DeepPhys counterpart: a
registry line (and, only if the model has a switch, a config class) in
[`src/model_config.py`](../src/model_config.py), the schema of the config
file; and a builder with its registry line in
[`src/models.py`](../src/models.py), which turns a parsed config into a
network.

### The config class (only if the model has a switch)

Most models have none: their `MODEL` section is `NAME` alone and they map to
the shared `ModelConfig`. Write a class only when the paper leaves something
free:

```python
@dataclass
class MyNetConfig(ModelConfig):
    FSAM: bool = True        # a flag the paper ablates

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        ...                  # raise ConfigError for anything the interface cannot satisfy
```

- **Fields are the YAML keys.** Every field is required in the file, unknown
  keys are refused, so the dataclass *is* the schema of the `MODEL` section.
- Fields are **experiment switches**, not sizes: a head variant, a flag the
  paper ablates, the temporal-shift segment length. If you find yourself
  adding `HIDDEN_DIM`, stop; it belongs in the module's defaults. Input
  preprocessing is never a switch; it is the backbone's first stage.
- `validate(interface, where)` is called at load, after the interface is
  loaded. `TemporalShiftConfig` (EfficientPhys, TS-CAN) and
  `FactorizePhysConfig` are the two that exist.

Every `MODEL` section also accepts one optional key, `REGULARISATION`,
without any class of its own: `{TERM: weight > 0}` naming which of the
backbone's `REGULARISERS` count and how much. Absent or `{}` means none;
a term left out is off, never zeroed, so an ablation is deleting a key.
The builder refuses a name the backbone does not declare. The weights are
the trainer's: they are merged beside the interface's `LOSS` weights per
trace, and the total stays the mean over traces of each trace's weighted
sum. A term may not share a loss component's name (MSE, CCC and the rest);
config load refuses it.

### The builder

```python
def _build_mynet(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: MyNet(in_channels=width), interface)
```

Inputs: the loaded config and the loaded interface. Output: a
`MultiTraceModel`. The builder is where every width is derived and where any
interface requirement beyond the config's is enforced (DeepPhys refuses an
interface with no `RESIZE` here, because its dense layer is sized from the
frame; the pooling backbones refuse a frame below `MIN_FRAME`).
`_multi_trace` fills in the wrapper's arguments:

| Argument | Meaning |
| ---------- | --------- |
| `make_copy` | zero-argument callable returning one fresh backbone; called once per trace |
| `channels` | `interface.CHANNELS`, the order the channel axis is stacked in; `C_in = len(channels)` |
| `traces` | `interface.TRACES`, the order copies are made and predictions keyed in |

### The two registry lines

One in each file, the same key in both:

```python
# src/model_config.py
MODEL_CONFIGS = {
    "DeepPhys": ModelConfig,
    "MyNet": ModelConfig,        # or MyNetConfig if it has a switch
}

# src/models.py
MODEL_BUILDERS = {
    "DeepPhys": _build_deepphys,
    "MyNet": _build_mynet,
}
```

The key is what `NAME:` must say in the YAML. Use the architecture's proper
name, matching the module's class.

Also add the import at the top of `src/models.py` beside `DeepPhys`'s:

```python
from neural_methods.model.MyNet import MyNet
```

## Step 3: the paper config file

**File:** `configs/original_model_config/<name>_<interface>.yaml`, named
after the architecture and the interface it encodes, as DeepPhys's
`deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml` is. When migrating a model, this
file **is** the paper's configuration of it: the rPPG-Toolbox definition of
the model in the `INTERFACE` section and its training recipe in `TRAIN`.
Nothing in code declares the paper setup. Copy
[`configs/_model_config_template.yaml`](../configs/_model_config_template.yaml),
which lists every key of the three sections with its type and the values
the parser accepts, and fill it in with DeepPhys's file beside it as the
worked example. `src/model_config.py` is the schema, in the same order,
and every key is required.

### `MODEL`

```yaml
MODEL:
  NAME: MyNet        # a key of MODEL_CONFIGS in src/model_config.py; every other
                     # key is a field of the config class, and most have none
```

Parsed by `NAME` into the config class, every field required, unknown keys
refused, then `validate`d against the interface. The instance is what
`build_model` hands to your builder and what the run writes into its
`config.yaml` (and the checkpoint) alongside every other config it ran on.
Several experiments on one architecture are several files with the same
`NAME` and different switches.

### `INTERFACE`

Set every key to the rPPG-Toolbox definition of the model:

| Key | Upstream source |
| ----- | ----------------- |
| `FS` | the rate the published config trains at (30 for the UBFC recipes) |
| `WINDOW_SECONDS` | `CHUNK_LENGTH / FS`, written to six decimals so it snaps to a whole frame |
| `RESIZE` | the published `RESIZE.H` / `RESIZE.W` |
| `TRACES` | `[PPG]`: the upstream models predict BVP and nothing else |
| `LOSS` | the published criterion (`MSE` for DeepPhys, `NEGPEARSON` for PhysMamba and PhysFormer) |

Neither published preprocessing key survives as an interface key. The
`DATA_TYPE` is the backbone's first stage (step 1, "Input normalisation").
The `LABEL_TYPE` is fixed by the signal: PPG, ECG and respiration labels are
z-scored over the window and every other trace is left raw, in physical
units (`label_mode` in `src/signal_transforms.py`, read off the signal's
class in the `SIGNALS` table). Every upstream PPG recipe z-scores or
difference-normalises its label, so z-score is what a migrated model gets.
Nothing Neckflix-specific goes in this file. The pressure traces and their
loss belong to the standard interface every model is compared on. Where the loss module lacks a published term (the
DLDL frequency loss of PhysFormer, say), note it in the file rather than
substituting something.

### `TRAIN`

The paper's training recipe, filled from two upstream sources, because the
upstream YAML alone does not say how the model was optimised:

| Key | Upstream source |
| ----- | ----------------- |
| `LR` | `TRAIN` block of the `train_configs/` file |
| `OPTIMIZER`, `WEIGHT_DECAY` | the `optim.*` call in `neural_methods/trainer/<Name>Trainer.py` at the `pre-overhaul` git tag (the legacy per-model trainers are deleted from the working tree; `git show pre-overhaul:neural_methods/trainer/<Name>Trainer.py` is where they still live) |
| `SCHEDULER` | the `lr_scheduler.*` call in the same trainer; a `StepLR` that never fires inside the paper's epochs is `Constant` |
| `PRECISION` | `float32` unless the trainer autocasts |

The paper's epochs and batch size are not in the file: they are the
launcher's `--epochs` and `--batch-size` flags (with `--num-workers` and
`--no-gpu`, the machine's), so note them in a comment and pass them in the
README command. If the
trainer needs an optimiser or schedule the shared `src/trainer.py` lacks,
add it there for every model (one line in its `OPTIMIZERS` or `SCHEDULERS`
and one name in `src/model_config.py`), never a per-model training loop.
Note in the file any trainer-side difference that remains, such as the
readout being exempt from weight decay here when upstream decayed
everything.

Nothing in code checks a run against this file; passing it as `--config`
is what makes a run the paper's configuration. Model comparisons then run
every model on one standard interface instead (the files under
`configs/combined_model_config/`), which is why step 1 insists the model
accepts any size. The nine paper interfaces today (the "Input" column is
the backbone's own first stage, not an interface key):

| Model | Frames | Window | Input | Loss | Recipe |
| ------- | -------- | -------- | ------- | ------ | -------- |
| DeepPhys | 72x72 | 180 frames | DiffNormalized + Standardized | MSE | AdamW 9e-3, OneCycle, 30 epochs |
| EfficientPhys | 72x72 | 180 frames | Standardized | MSE | AdamW 9e-3, OneCycle, 30 epochs |
| FactorizePhys | 72x72 | 160 frames | Raw | negative Pearson | Adam 1e-3, OneCycle, 10 epochs |
| PhysFormer | 128x128 | 160 frames | DiffNormalized | negative Pearson | Adam 1e-4, decay 5e-5, constant, 10 epochs |
| PhysMamba | 128x128 | 128 frames | DiffNormalized | negative Pearson | Adam 3e-3, decay 5e-4, OneCycle, 20 epochs |
| PhysNet | 72x72 | 128 frames | DiffNormalized | negative Pearson | Adam 9e-3, OneCycle, 30 epochs |
| RhythmFormer | 128x128 | 160 frames | Standardized | negative Pearson | AdamW 9e-3, OneCycle, 30 epochs |
| TS-CAN | 72x72 | 180 frames | DiffNormalized + Standardized | MSE | AdamW 9e-3, OneCycle, 30 epochs |
| iBVPNet | 72x72 | 160 frames | Raw | negative Pearson | Adam 1e-3, OneCycle, 30 epochs |

## Step 4: one smoke test

**File:** `tests/test_<name>.py`. One test, and only one: build the model
from its paper config and push one synthetic batch through. This is
the ceiling for a migration; do not add tests opportunistically.

```python
import torch

from src.model_config import load_config
from src.models import build_model

CONFIG = "configs/original_model_config/mynet_<interface>.yaml"


def test_mynet_forward_matches_the_contract():
    interface, model_config, _ = load_config(CONFIG)
    model = build_model(model_config, interface)
    B, T = 2, interface.window_frames
    H, W = interface.RESIZE.H, interface.RESIZE.W
    batch = {"frames": {ch: torch.rand(B, T, H, W) for ch in interface.CHANNELS}}
    out = model(batch)
    assert set(out["predictions"]) == set(interface.TRACES)
    assert all(p.shape == (B, T) for p in out["predictions"].values())
    assert len(model.output_layers()) == len(interface.TRACES)
```

Run it with `uv run pytest tests/test_mynet.py`. If the architecture cannot
yet take the interface's window, the test is where the `temporal_divisor`
refusal will surface first.

## Running it

A fold is one command, which trains and, after every epoch, runs that
epoch's model over the held-out participant and scores the records,
printing the run directory it writes:

```bash
uv run python scripts/run.py --datasets neckflix --test-participant-dataset neckflix --test-participant-id 1 --config configs/original_model_config/mynet_<interface>.yaml --limit-windows 8
```

`--config` is one file holding the model config, the interface and the
training recipe described above as its `MODEL`, `INTERFACE` and `TRAIN`
sections (see
[`deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml`](../configs/original_model_config/deepphys_FS30_W6S6_RGB_PPG_H72W72.yaml)).
`--limit-windows N` keeps N evenly spaced windows per split for a wiring
check; drop it for a real fold. Run the paper config first to check the
migration against the paper, then the standard interface every model is
compared on.

What happens, in order:

1. Training: `load_config` (`src/model_config.py`) reads the file's three
   sections, validating the model section against the interface;
   `build_model(model_config, interface)` calls your builder;
   `Trainer(...)` checks the window against `temporal_divisor` /
   `temporal_length`, seeds the readout biases, exempts the readouts from
   weight decay, wraps in DDP if distributed, then `fit` writes the run
   directory. What the script is made of beyond the trainer (the
   arguments, the compiled config, the windowed datasets) is
   `src/experiment.py`.
2. Inference, after every epoch: `test` on the same trainer records the
   held-out participant's strided windows with that epoch's weights, and
   `src/outputs.py` writes them as `epoch_NN/test_records/` beside a copy
   of the weights.
3. Evaluation, straight after: `src/evaluation/recording.py` reads that
   `test_records/` alone
   and writes one `<TRACE>_beats.csv` per cardiac trace, `signals.csv`,
   `rates.csv` and one `<TRACE>.png` per trace beside each recording's
   trace tables (`docs/evaluation.md`).

Outputs land in `runs/<MODEL>_<DATASET>.<participant or all>-..._<YYYYMMDDHHMM>/`
(one `<DATASET>.<...>` per `--datasets` entry, the held-out participant on the
dataset it came from and `all` on the rest):

| File | When | Contents |
| ------ | ----------- | ---------- |
| `config.yaml` | before the first epoch | everything the run ran on in one mapping: the command, git commit, every dataset / interface / model / training config as loaded, the files they came from, the stores on each side of the hold-out, and the resolved device and precision |
| `model.pt` | every epoch, overwritten | the latest epoch's state dict plus the same compiled config (`src.experiment.rebuild` reads a run back from it) |
| `losses.csv` | every epoch | per-epoch loss, per trace and component; `epoch` is 1-based like the directories below |
| `epoch_NN/model.pt` | after epoch NN | that epoch's state dict, in the same form |
| `epoch_NN/test_records/` | after epoch NN | `meta.json` (with the epoch), `windows.csv` (one row per window with its position and presence flags), and per recording and camera one `<TRACE>.csv`: frame, time, label, mean / std / n over the overlapping windows, then one column per window, all in physical units |
| `epoch_NN/test_records/<recording>/<camera>/<TRACE>_beats.csv`, `signals.csv`, `rates.csv`, `<TRACE>.png` | after the records | per reference beat its peak and trough times, its matched predicted beat's and both beats' levels; per signal, over the whole covered stretch, the beat counts, the level means / SDs / errors and the waveform agreement; a heart rate per source; per trace the figure of label, prediction with its spread, and beats |

## Migrating an upstream rPPG-Toolbox model

Nine upstream rPPG-Toolbox models are migrated and registered in
`MODEL_CONFIGS` today (see the intro). One upstream model in
`neural_methods/model/` remains unmigrated —
[`PhysHydra.py`](../neural_methods/model/PhysHydra.py) — and it stays there
deliberately, out of scope rather than pending. An upstream model, before
migration, predicts one BVP trace from RGB and carries conventions this
contract drops. Migrating one is editing the module in place, not writing a
new one beside it, and the diff is almost always these items:

1. **Widen the first layer.** Replace the hard-coded `3` on the first conv
   with an `in_channels` constructor argument defaulting to `3`.
2. **One readout, width one, exposed.** Keep the final layer emitting a
   single trace and return it from `output_layers()`. Delete any
   multi-signal head, `ParallelSignals` or `DictModel` wrapper from the
   legacy attempt; the wrapper in `src/models.py` replaces all of them.
3. **Fix the output shape.** Clip models often emit `(B, T)`; return
   `(B, 1, T)` (`rearrange(x, "b t -> b 1 t")`). A per-frame network folds
   `(b t)` inside and unfolds its `(N, 1)` back to `(B, 1, T)`.
4. **Own the input preprocessing, drop the loss.** The upstream `DATA_TYPE`
   becomes the network's first stage: `DiffNormalize()` and/or
   `Standardize()` from `neural_methods/model/modules/`, applied to the raw
   clip on the first line of `forward` (step 1, "Input normalisation"). Any
   loss lives in the interface's `LOSS` block.
5. **Drop `params`, `get_config`, and dead imports** (`pdb`, `math` for
   nothing). Delete rather than keep for compatibility.
6. **einops for every reshape.** `x.view(B, -1)` becomes
   `rearrange(x, "b c h w -> b (c h w)")`, and so on.
7. **Free the frame and the window.** Derive what the interface can supply,
   and wrap what it cannot in an adaptive stage that is the identity at the
   paper's shape (step 1, "Any frame size, any window length"). Only as an
   interim, declare `temporal_divisor` or `temporal_length` instead.
8. **Take the clip.** Every backbone receives `(B, C_in, T, H, W)`. If the
   paper fed single frames with `T` hidden in the batch axis, normalise the
   clip first, then fold `(b t)` for the 2D layers and unfold at the end, as
   DeepPhys does.

Then steps 2 to 4 above. DeepPhys shows the finished form: compare
`neural_methods/model/DeepPhys.py` against the upstream file to see exactly
how small the diff is.

## Checklist

- [ ] `neural_methods/model/<Name>.py`: `in_channels` argument, published
      sizes as defaults, the paper's input normalisation as the first stage,
      `(B, C_in, T, H, W) -> (B, 1, T)`, `output_layers()`, einops, no loss.
- [ ] `src/model_config.py`: one line in `MODEL_CONFIGS`; a config class
      only if there is a switch.
- [ ] `src/models.py`: import, `_build_<name>`, one line in `MODEL_BUILDERS`.
- [ ] `configs/original_model_config/<name>_<interface>.yaml`, from
      `configs/_model_config_template.yaml`: `MODEL`
      (`NAME` plus one key per config field), `INTERFACE` (the paper's rate,
      window, resize, single PPG trace and loss; nothing Neckflix-specific)
      and `TRAIN` (the paper's optimiser, rate, decay, schedule and
      precision, from the upstream config and trainer class; its epochs and
      batch size go in the README command).
- [ ] The model builds and runs on the standard interface too; any size the
      paper did not use goes through an adaptive stage, not a refusal.
- [ ] `tests/test_<name>.py`: one build-and-forward test, passing.
- [ ] A smoke run with `--limit-windows` reaches `evaluate` and writes the
      run directory.
- [ ] `README.md`, "Algorithms": add the model to the list on the contract,
      and the exact command that trains and tests it on PURE with only the
      first participant held out (`--datasets pure --test-participant-dataset
      pure --test-participant-id 01`) on its paper interface and paper
      training recipe. A migration is not finished until that command is
      there.
