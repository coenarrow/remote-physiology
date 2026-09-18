# Model regularisation terms

Design for letting a backbone contribute its own loss terms — penalties on
its internals, not on the prediction against the label — weighted from the
`MODEL` section of the config file, so an ablation is a YAML edit.

First user: the PhysHydra migration (its interpretability branch carries a
sparsity and a smoothness penalty). That migration is a separate task on
top of this one.

## The split

| Term | Depends on | Owned by | Config key |
| ------ | ------------ | ---------- | ------------ |
| Loss | prediction and label | the trace, shared by every model | `INTERFACE.LOSS.<TRACE>` |
| Regulariser | a model's own intermediate tensors | the architecture | `MODEL.REGULARISATION` |

The loss stays exactly what it is. A regulariser is a scalar a backbone
computes during `forward` from something the loss cannot see (an attention
map, a gate). Its weight sits beside the switch that creates the stage it
penalises, in the same file, so the run's `config.yaml` records both.

**Left out means off.** `REGULARISATION` names the terms that count and
their positive weights; a term the mapping does not name contributes
nothing. This is the `LOSS` convention ("a component that should not count
is left out, not zeroed") applied to the model section. An empty mapping is
the no-regularisation ablation.

## Config

`REGULARISATION` is a field of the base `ModelConfig`, so every model has
it and no new class exists for it. It is the one **optional** key of the
`MODEL` section (the dataclass builder in `src/config.py` already takes an
`optional` tuple): absent or `{}` means no regularisation, so none of the
existing config files changes.

```yaml
MODEL:
  NAME: PhysHydra
  INTERPRETABLE: true
  REGULARISATION: {SPARSITY: 0.1, SMOOTHNESS: 0.01}   # optional; absent or {} means none
```

In `src/model_config.py`:

```python
@dataclass
class ModelConfig:
    NAME: str = ""
    REGULARISATION: dict = field(default_factory=dict)   # {TERM: weight > 0}, optional

    def validate(self, interface, where):
        self.REGULARISATION = normalise_regularisation(self.REGULARISATION, where)
```

`parse_model_config` passes `optional=("REGULARISATION",)` to `build`.
A subclass that overrides `validate` (`TemporalShiftConfig`) calls
`super().validate` first.

`normalise_regularisation` mirrors `normalise_loss_weights`: a mapping,
every weight a positive number (not a bool), keys lower-cased so the log
columns match the loss components' spelling. It does **not** know the
term names; those belong to the module, and `src/models.py` is the one
file that imports every module. The builder refuses, by name, a term the
module does not declare, which for the nine models with no `REGULARISERS`
means any non-empty mapping. It also refuses a term whose name is a loss
component (MSE, CCC and the rest), because the trainer merges both into
one per-trace dict.

`INTERPRETABLE` is not a regulariser: it is PhysHydra's structural switch
(the attention branch exists or not) and follows the existing pattern of
`FSAM` and `FRAME_DEPTH`, a small per-model class:

```python
@dataclass
class PhysHydraConfig(ModelConfig):
    INTERPRETABLE: bool = True
```

## Backbone contract

Two additions beside `output_layers()`, both optional. A backbone without
them is unchanged, so the nine migrated models are not touched.

```python
class PhysHydra(nn.Module):
    #: The terms this backbone can compute, lower-case, the keys of REGULARISATION.
    REGULARISERS = ("sparsity", "smoothness")

    def forward(self, x):            # (B, C_in, T, H, W) -> (B, 1, T), as today
        ...
        self._regularisers = {"sparsity": sparsity, "smoothness": smoothness}
        return out

    def regularisers(self) -> dict:
        """``{term: () tensor}`` from the last forward, graph-attached."""
        return self._regularisers
```

`forward` still returns one tensor. The terms are read back the way the
readouts are, by a method on the copy, so the wrapper's `forward_video`
signature and every other backbone stay as they are. A backbone computes
every term it declares on every forward; which ones count is not its
concern.

## Wrapper

`MultiTraceModel` knows nothing about weights. Its `forward` adds
`regularisers` to the dict it returns, keyed like `predictions`, holding
every term each copy computed:

```python
def collect_regularisers(self) -> dict:
    collected = {}
    for trace, copy in self.copies.items():
        declared = tuple(getattr(type(copy), "REGULARISERS", ()))
        if not declared:
            collected[trace] = {}
            continue
        returned = copy.regularisers() if hasattr(copy, "regularisers") else {}
        missing = [t for t in declared if t not in returned]
        if missing:
            raise RuntimeError(
                f"{type(copy).__name__} declares REGULARISERS {list(declared)} "
                f"but regularisers() did not return {missing}")
        collected[trace] = {t: returned[t] for t in declared}
    return collected
```

One copy per trace means one set of terms per trace: the sparsity of the
ABP copy's attention and of the CVP copy's are separate scalars and are
logged separately. For a backbone that declares none the entry is
`{trace: {}}` and nothing downstream changes. For a backbone that declares
a term and does not return it from `regularisers()`, the wrapper refuses
at the first forward, naming the missing term.

`_multi_trace` checks `cfg.REGULARISATION` against the backbone's
`REGULARISERS` (a class attribute, read off one copy, `()` when absent) through a
shared `_require_regularisers` and refuses an unknown term by name, next
to where builders refuse a frame below `MIN_FRAME`. Every builder gets the
check without repeating it.

## Trainer

`Trainer` is handed the model config as one more constructor argument
(`scripts/run.py` and `tools/memory_report.py` are the two callers;
`src.experiment.rebuild` never constructs a trainer). It then owns every weight: `self.weights` is built
once in `__init__` as `{trace: {**criterion.weights[trace],
**model_config.REGULARISATION}}`.

`_losses` picks the named terms out of what the wrapper emitted and merges
them into the per-trace raw dict before weighting. This is where "left out
means off" lives:

```python
raw = self.criterion(out["predictions"], out["labels"], out["label_mask"])
named = self.model_config.REGULARISATION
for trace, terms in out["regularisers"].items():
    raw[trace].update({k: v for k, v in terms.items() if k in named})
return weight_losses(raw, self.weights)
```

The total stays the mean over traces of each trace's weighted sum, so
adding a term to a trace does not rescale the others, and `losses.csv`
gains `ABP/sparsity`, `CVP/sparsity`, … columns through the existing loop.
`test` ignores the terms.

`weight_losses` loses its "unnamed component is weighted 1.0" default.
Nothing relies on it: `PerSignalLoss` computes only the components the
interface names, and `_losses` merges only the terms the model config
names. An unweighted component is now a programming error (`KeyError`),
not a silent contribution.

The trainer's docstring line "the model says nothing" becomes "the model
says nothing about the loss; it may contribute regularisers, weighted from
its own config".

## Files

| File | Change |
| ------ | -------- |
| `src/model_config.py` | `REGULARISATION` on `ModelConfig` (optional key), `normalise_regularisation` |
| `src/models.py` | `regularisers` in the wrapper's dict, `_require_regularisers` called from `_multi_trace` |
| `src/trainer.py` | `model_config` argument, merged weights, `_losses` filter and merge, docstring |
| `scripts/run.py` | pass the model config to `Trainer` |
| `tools/memory_report.py` | pass the model config to `Trainer` |
| `neural_methods/loss/PerSignalLoss.py` | `weight_losses` without the 1.0 default |
| `configs/_model_config_template.yaml` | `REGULARISATION` under the switches comment |
| `docs/adding_a_model.md` | "Regularisers" under step 1 and step 2 |

No new module, class, loop, loss class or log path.

## Verification

No tests (repo policy). The check is that an existing model is unchanged
and that the mechanism carries a term end to end:

1. The README PhysMamba command with `--limit-windows 8` runs to
   `evaluate` and `losses.csv` has the same columns as before.
2. The PhysHydra migration, the first user, runs with
   `REGULARISATION: {SPARSITY: 0.1, SMOOTHNESS: 0.01}` and the two columns
   appear per trace, and with `{}` and they do not.

## Out of scope

- The PhysHydra migration itself (next task; this spec is its prerequisite).
- Regularisers on the wrapper rather than per copy (a cross-trace term).
  No model needs one; the per-trace dict would take it as a pseudo-trace
  key if one ever does.
- Saving the intermediate tensors (attention maps) for inspection at test
  time. That is an outputs question, separate from the loss.
