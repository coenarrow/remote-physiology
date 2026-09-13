# run_experiment.py: one command per experiment, scored at every epoch

**Goal:** one command that trains a model on a set of datasets, holding out
one participant after another, and after **every epoch** runs the held-out
participant through the model, writes its records and scores them, so an
experiment ends as one self-contained directory under `runs/` holding every
fold, every epoch's weights, records, scores and figures, and a learning
curve per fold.

**Worked example:** PhysMamba on Neckflix, pressure traces only, recumbent
recordings only, participants 32, 33 and 34 held out in turn.

**Why now:** the three scripts (`scripts/train.py`, `scripts/infer.py`,
`scripts/eval.py`) each do one thing well but chain by hand, infer only the
last epoch, and leave one flat run directory per fold with a timestamp for
a name. Comparing configurations means remembering which timestamp was
which.

## Constraints

Binding, from `CLAUDE.md` and the questions answered while designing this:

1. Everything shared is written once. The per-epoch pass reuses the
   trainer's test pass, the records writer `scripts/infer.py` uses and the
   scorer `scripts/eval.py` uses. `scripts/train.py` stays the one-fold
   worker; `scripts/run_experiment.py` calls it once per fold and adds
   nothing but naming and looping.
2. Legacy code is deleted, not adapted: the name-to-directory lookups for
   dataset and model configs go, and the dead default paths for the
   interface and training files with them.
3. No new tests. The existing chain test keeps passing because
   `scripts/train.py`'s default path is unchanged. Validation is the two
   end-to-end runs at the end of this document.
4. Filters live in dataset YAML files only. There is no command-line
   filter override; "only recumbent" is a dataset file built on the base
   one. Trace selection lives in the interface file.

## The command and the tree it leaves

The recumbent filter is a dataset file, `configs/datasets/neckflix_recumbent.yaml`:

```yaml
# Neckflix, recumbent recordings only. Keys and grammar: neckflix.yaml.
BASE: neckflix.yaml
FILTERS:
  posture:
    include: [recumbent]
    exclude: []
```

The command:

```bash
uv run python scripts/run_experiment.py \
    --datasets configs/datasets/neckflix_recumbent.yaml \
    --test-participant-dataset configs/datasets/neckflix_recumbent.yaml \
    --test-participant-id 32 33 34 \
    --model configs/models/physmamba.yaml \
    --interface configs/interfaces/interface_neckflix_bp.yaml \
    --training configs/training/physmamba_3ep.yaml \
    --reading-seconds 10
```

The tree:

```text
runs/physmamba_datasets-neckflix_recumbent_filters-posture=recumbent/
  neckflix_recumbent.32_202609131200/
    config.yaml        everything the fold ran on (as train.py writes today)
    losses.csv         training loss per epoch (as today; epoch column now 1-based)
    model.pt           the latest epoch (as today; infer.py still works on the fold)
    epochs.csv         NEW: the learning curve on the held-out participant
    epoch_01/
      model.pt         that epoch's weights and compiled config; infer.py accepts this directory
      test_records/    exactly what infer.py writes, then exactly what eval.py writes beside it
        meta.json, windows.csv
        <recording>/<perspective>/<TRACE>.csv, <TRACE>_beats.csv, readings.csv, rates.csv, <TRACE>.png
    epoch_02/ ...
    epoch_03/ ...
  neckflix_recumbent.33_202609131418/ ...
  neckflix_recumbent.34_202609131631/ ...
```

Nothing is written at the experiment level. Each fold directory is what
`scripts/train.py` writes today plus the epoch directories and the curve,
so `scripts/infer.py` and `scripts/eval.py` work on a fold, and on an epoch
directory, exactly as they work on a run directory now.

### The experiment directory name

`<model>_datasets-<datasets>_filters-<filters>`, where

* `<model>` is the model config file's stem (`physmamba`);
* `<datasets>` is every dataset file's stem, in command-line order, joined
  with `+`;
* `<filters>` renders each dataset's merged `FILTERS` **as written in its
  file**, in file order: every attr with a non-empty include list as
  `<attr>=<values>`, every attr with a non-empty exclude list as
  `<attr>-not=<values>`, values joined with `+`, entries joined with `.`.
  With more than one dataset each entry is prefixed `<dataset>-`. When
  every list is empty the field reads `filters-none`. Any character in a
  value other than a letter, digit, `.`, `-` or `_` becomes `-`. Every
  character used is legal in a Windows, Linux and macOS file name and
  needs no quoting in PowerShell or bash.

So `--datasets configs/datasets/neckflix.yaml` (whose base file spells out
all three postures) renders as
`physmamba_datasets-neckflix_filters-posture=supine+recumbent+sitting`,
`configs/datasets/neckflix_p32_36.yaml` as
`physmamba_datasets-neckflix_p32_36_filters-posture=supine+recumbent+sitting.participant=32+33+34+35+36`,
and the worked example as
`physmamba_datasets-neckflix_recumbent_filters-posture=recumbent`.
The base YAMLs are left alone: what the file says is what the name says.

### The fold directory name

`<dataset stem>.<participant>_<YYYYMMDDHHMM>`, stamped when the fold
starts. Which interface and recipe the fold ran on is in its
`config.yaml`; reruns of the same fold sit side by side.

## The command-line change: every config flag is a file path

Today `--datasets` and `--model` take names resolved under
`configs/datasets/` and `configs/models/`, while `--interface` and
`--training` take paths. After this change **every config flag takes a
path**, and a dataset or model is known by its file's stem:

| flag | takes | the name it is known by |
| --- | --- | --- |
| `--datasets PATH [PATH ...]` | dataset config files | each file's stem (`neckflix`, `neckflix_recumbent`, `pure`) |
| `--test-participant-dataset PATH` | one of those files | matched to the loaded datasets **by stem** |
| `--model PATH` | the model config file | its stem (`physmamba`) |
| `--interface PATH` | the interface file | — |
| `--training PATH` | the training recipe | — |

The four config flags are required; `--interface` and `--training` lose
their defaults (`configs/interface.yaml` and `configs/training.yaml`,
neither of which exists). The test-participant pair stays optional in
`scripts/train.py` (nobody held out) and `scripts/infer.py` (the run's
own participant) and is required by `scripts/run_experiment.py`. Two
`--datasets` files sharing a stem are refused, naming both.
A `--test-participant-dataset` whose stem matches no loaded dataset is
refused, listing the loaded stems. `scripts/infer.py` matches the same
way, whether its datasets come from the checkpoint or from its own
`--datasets`. The stem is what the run name, the experiment name, the
compiled config's `datasets` and `split` sections, the store dictionaries
and the records' `dataset` metadata already key on, so nothing downstream
changes shape.

`src/datasets.py` loses `DATASET_CONFIG_DIR` and the name lookup;
`resolve_dataset_configs(paths)` returns `{stem: path}` with the two
refusals above, and `load_dataset_configs(paths)` follows. `src/models.py`
loses `MODEL_CONFIG_DIR` and `resolve_model_config`;
`load_model_config(path, interface)` reads the file it is given. The
compiled config's `sources` section records the resolved paths exactly as
it does today.

## Components

### `scripts/train.py`: three new flags, default behaviour unchanged

| flag | meaning |
| --- | --- |
| `--run-dir PATH` | the exact run directory; wins over `--runs-dir` and the derived name |
| `--eval-each-epoch` | after every epoch, run the held-out participant and score it; refused when nobody is held out |
| `--reading-seconds S` | passed to the scorer; shares `scripts/eval.py`'s default of 30 |

Under `--eval-each-epoch` the script builds the held-out participant's
strided windows once, honouring `--limit-windows`, and hands
`Trainer.fit` the after-epoch hook below. Without the flag the script does
exactly what it does today, so the README's three-step chain and the
chain test are untouched.

One new refusal, on every rank before the runtime is initialised: a run
directory that already holds `config.yaml` is an error naming it. Today
the directory is silently reused and the first run's `config.yaml` kept,
which a smoke run repeated inside one minute would otherwise hit.

### `src/trainer.py`: one optional parameter

`Trainer.fit(train_dataset, after_epoch=None)`. Once the main rank has
printed the epoch, rewritten `losses.csv` and saved `model.pt`, every rank
calls `after_epoch(epoch)` with the 1-based epoch number. That is the only
trainer change; the trainer still writes no records. The loss log's
`epoch` column becomes 1-based to join the epoch directories and the
curve; the printed progress line already counts from 1.

### The hook: `src/experiment.py`

One callable built by `scripts/train.py` from the trainer, the held-out
windows, the interface, the run directory, the records metadata (dataset,
participant, run directory, command, git state) and the reading length.
Per epoch:

1. on every rank, `trainer.test(test_dataset)` (sharded under DDP,
   gathered on the main rank);
2. on the main rank only: create `epoch_NN/`, save
   `trainer.checkpoint()` there as `model.pt`, write the records to
   `epoch_NN/test_records/` through `src.outputs.write_records` with
   `"epoch": NN` added to the metadata, score them through the shared
   scorer below, and rewrite the fold's `epochs.csv`.

A killed run keeps every finished epoch and a partial curve. The epoch
directory's name and glob live in `src/outputs.py` beside `RECORDS_DIR`,
so the hook that writes them and the curve that reads them share one
definition.

### The scorer moves into the package: `src/evaluation/recording.py`

Today `scripts/eval.py` itself reads `meta.json`, finds every
`<recording>/<perspective>/` folder and scores each. That loop becomes
`score_records(records_dir, reading_seconds) -> list[Path]` in the
package, with `recording_folders` beside it, raising when a records
directory holds no recording. `scripts/eval.py` keeps its path shorthand
(`find_records_dirs`) and the argparse and calls `score_records`; the hook
calls the same function.

### The curve: `src/evaluation/curve.py`

`epochs.csv` at the fold root, one row per epoch and per signal or rate
source, rebuilt from whatever epoch directories exist:

| column | meaning |
| --- | --- |
| `epoch` | 1-based, matching `epoch_NN/` and `losses.csv` |
| `signal` | a trace (`ABP`, `CVP`, ...) or a rate source (`FUSED`, `MEDIAN`) |
| `n_readings` | readings pooled from every recording and camera of the held-out participant |
| every numeric `readings.csv` column | its mean over those readings (`n_ref_beats`, ..., `err_max`, ..., `waveform_ccc`) |
| `mae_max`, `mae_mean`, `mae_min` | mean of the absolute signed errors, beside each `err_<s>` |
| `ref_hr`, `pred_hr`, `err_hr`, `mae_hr`, `snr`, `macc` | means of the `rates.csv` columns for that source, plus the absolute rate error |

A cardiac trace's row carries both its reading means and its rate means;
`FUSED` and `MEDIAN` rows carry only the rate columns; a non-cardiac
trace's rate columns are blank. Blank cells are ignored by the means. The
module exposes `epochs_table(run_dir) -> DataFrame` and
`write_epochs_table(run_dir) -> Path`; being a pure read of the epoch
directories, it is also the one-call repair after re-scoring an epoch by
hand with `scripts/eval.py`.

### `scripts/run_experiment.py`: naming and looping

```text
scripts/run_experiment.py --datasets PATH [PATH ...]
    --test-participant-dataset PATH --test-participant-id ID [ID ...]
    --model PATH --interface PATH --training PATH
    [--runs-dir DIR] [--reading-seconds S] [--limit-windows N]
```

1. Parse the shared config group, the participant dataset (required) and
   one or more participant ids (required), plus the three pass-through
   flags.
2. Load the dataset configs and derive the experiment name
   (`experiment_name(model_stem, configs)` in `src/experiment.py`, beside
   `run_name`).
3. Scan the stores once and check every id with `hold_out_participant`,
   so a typo fails before the first epoch rather than after the fifth fold.
4. For each id in order: stamp the fold directory, call `train.main` with
   the forwarded flags plus `--run-dir` and `--eval-each-epoch`, print the
   fold directory when it finishes.
5. Return the list of fold directories.

Because each fold runs `train.main`, its `config.yaml` records a
`scripts/train.py` command that reproduces that fold alone. Under a
distributed launch every rank runs the same loop; each fold initialises
and shuts down the process group as `scripts/train.py` does today, which
torch supports.

## Data flow, per fold

```text
train.main(argv)
  load interface, model config, training recipe, dataset configs  (all by path)
  init runtime; scan stores; hold the participant out
  train windows (random)            test windows (strided, once)
  Trainer(model, ...).fit(train, after_epoch=hook)
    per epoch: step ... -> losses.csv, model.pt      (main rank)
               hook(epoch):
                 trainer.test(test)                  (all ranks)
                 epoch_NN/model.pt                   (main rank)
                 epoch_NN/test_records/  <- write_records
                 score_records(epoch_NN/test_records) -> beats, readings, rates, figures
                 epochs.csv              <- write_epochs_table(run_dir)
```

## Error handling

* Every config path is checked when loaded, with the file named in the
  error, as today.
* Duplicate dataset stems, an unmatched `--test-participant-dataset`, an
  unknown participant id and `--eval-each-epoch` with nobody held out are
  argparse errors before any store is opened for windows.
* A run directory already holding `config.yaml` is refused before the
  runtime starts.
* A fold that fails stops the sweep; the finished folds keep their
  directories, and the failing fold keeps its finished epochs.

## Documentation

* `README.md`: the ten "Algorithms" commands take paths; a new
  "Experiments" section with the command and tree above.
* `CLAUDE.md`: the migration rule's quoted flags become paths.
* `docs/adding_a_model.md`: the commands and the sentence about
  `--model <name>` become paths.
* `docs/evaluation.md`: `score_records` and the curve's columns.
* Module docstrings of `scripts/train.py`, `scripts/infer.py`,
  `scripts/eval.py`, `src/datasets.py`, `src/models.py`,
  `src/experiment.py`, `tools/memory_report.py`.

## Tests

No new tests. Three invocation adaptations, assertions untouched, listed
so a human can veto them:

* the eight per-model smoke tests (`tests/test_bigsmall.py` and siblings)
  pass the model YAML's path instead of its name;
* `tests/test_scripts.py` and `tests/test_memory_report.py` pass the
  dataset YAML's path and drop their `DATASET_CONFIG_DIR` monkeypatch.

## Validation

1. `uv run pytest tests/test_scripts.py tests/test_memory_report.py tests/test_physnet.py -p no:faulthandler`:
   the chain still leaves every promised file, and a model smoke test
   builds from a path. The other seven model smoke tests change the same
   one line and are run once at the end.
2. A wiring run on the five-participant cut, two folds, eight windows:
   ```bash
   uv run python scripts/run_experiment.py --datasets configs/datasets/neckflix_p32_36.yaml --test-participant-dataset configs/datasets/neckflix_p32_36.yaml --test-participant-id 32 33 --model configs/models/physmamba.yaml --interface configs/interfaces/interface_neckflix_bp.yaml --training configs/training/physmamba_3ep.yaml --limit-windows 8 --reading-seconds 10
   ```
   Expected: two fold directories, three epoch directories each with
   `model.pt` and scored `test_records/`, and an `epochs.csv` of up to
   twelve rows per fold (ABP, CVP, FUSED and MEDIAN per epoch; a source
   with no rate row in any reading is absent).
3. The worked example above, one fold, on the full recumbent cut.

## Files

| file | change |
| --- | --- |
| `scripts/run_experiment.py` | new |
| `scripts/train.py` | `--run-dir`, `--eval-each-epoch`, `--reading-seconds`; the hook; the existing-directory refusal |
| `scripts/infer.py` | paths on `--datasets`; stem matching |
| `scripts/eval.py` | calls `score_records` |
| `src/experiment.py` | path-taking argument group; `experiment_name`; the hook; `run_name` on stems |
| `src/trainer.py` | `after_epoch`; 1-based loss log |
| `src/datasets.py` | path-based `resolve_dataset_configs`; name lookup deleted |
| `src/models.py` | path-based `load_model_config`; name lookup deleted |
| `src/outputs.py` | the epoch directory name |
| `src/evaluation/recording.py` | `recording_folders`, `score_records` |
| `src/evaluation/curve.py` | new |
| `src/interface.py`, `src/training.py` | dead default paths deleted |
| `tools/memory_report.py` | paths |
| `configs/datasets/neckflix_recumbent.yaml` | new, the worked example |
| `tests/test_*.py` (ten files) | invocation adaptations above |
| `README.md`, `CLAUDE.md`, `docs/adding_a_model.md`, `docs/evaluation.md` | as above |

## Decisions, and who made them

| decision | by |
| --- | --- |
| One invocation loops over one or more held-out participants in-process | user |
| Experiment specified by CLI flags; filters only in dataset YAMLs; no trace override | user |
| Per-epoch inference and scoring as an in-process hook, behind `--eval-each-epoch` on `train.py` | user |
| Experiment name is model, datasets, filters as written; YAMLs untouched | user |
| Fold directories timestamped; one `model.pt` per epoch; figures every epoch | user |
| Per-fold `epochs.csv`; nothing at the experiment level | user |
| `train.py` stays the one-fold worker, called by `run_experiment.py` | user |
| Every config flag takes a path; datasets and models known by stem; test dataset matched by stem | user |
| `losses.csv` epoch column 1-based | Claude, approved |
| Every participant id checked before the first fold | Claude, approved |
| `--interface` and `--training` required, dead defaults deleted | Claude, for review |
| A run directory already holding `config.yaml` is refused | Claude, for review |
| `epochs.csv` columns as tabled above | Claude, for review |
| `configs/datasets/neckflix_recumbent.yaml` added as the worked example | Claude, for review |
