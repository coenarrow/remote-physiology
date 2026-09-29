# Run Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `scripts/evaluate.py <run-dir>` pools a finished run's per-recording tables into `<run-dir>/evaluation/`: demographics, ISO 81060-3 accuracy statistics and deep-learning metrics, as `report.md` and `report.html`.

**Architecture:** The record writer gains `recordings.csv` (each test recording's cache attributes). A pooling module stacks every fold's `signals.csv` and `rates.csv` into one table of paired measurements; statistics, demographics and figures are computed from that table alone; a small report builder writes the tables, figures, Markdown and HTML. The script is a thin caller and is never invoked by `run.py`, `test.py` or `main.py`.

**Tech Stack:** Python, pandas, numpy, matplotlib and seaborn (all present), `markdown-it-py` (new direct dependency).

**Spec:** `docs/plans/2026-09-29-run-evaluation-design.md`

**Covers:** pieces 2 and 3 of the spec. Piece 1 (core attribute standard in the caches) is Part C at the end: it waits on one decision and gets its own plan. Nothing here depends on it.

## Global Constraints

- Dependencies are added with `uv add`, never pip.
- **No tests are written.** `CLAUDE.md` overrides this skill's test steps: verification of each task is a command, run on `runs/synthetic_benchmark_physmamba`.
- **No commits** unless the user asks. Each task ends at a checkpoint where the user runs the command.
- The evaluation reads the run directory only: no cache, no checkpoint, no file under `configs/`.
- Do not inspect `runs/neckflix_*` or the Neckflix data. The reference input is `runs/synthetic_benchmark_physmamba`.
- Error is prediction minus reference everywhere.
- Eligibility defaults: `--segment-min 20`, `--segment-max 30`, seconds, inclusive.
- Epoch default: the last epoch of each fold. Never chosen by test error.
- Subject is the pair (dataset, participant), written `dataset/participant`.
- Identity columns are read from CSV as strings, so participant `01` stays `01`.
- ISO 81060-3 thresholds, verbatim: mean error within ±6.0 mmHg; `s_corr` ≤ 10.0 mmHg; `N_ind` ≥ 278; subjects ≥ 30.
- Markdown uses pipe tables, headings and image links only. Figures are 6.3 inches wide (A4 portrait text width).
- Output folder is `<run-dir>/evaluation/`, deleted and rewritten on every run.
- Names follow PEP 8; `uv run ruff check` passes on every file touched.
- Before writing any chart code (Task 5), load the `dataviz` skill.

## Review Focus

No tests pin these; each is checked by eye in the named task's verification step.

1. **The run directory is itself one fold** (`config.yaml` at its root, as `scripts/run.py` writes without `main.py`): it is evaluated as a single fold, not refused. Task 2.
2. **Every subject has one measurement**, as in the reference run: `s_corr` is the plain SD, ICC is blank, `N_ind = n`, with a note, and no division warning. Task 3.
3. **`recordings.csv` is absent** (every run made before Task 1): the report is written and each characteristic reads "not recorded". Task 4.
4. **A parameter has no paired values** (for example no predicted CVP beats, so every predicted level is blank): its subsection says so and the rest of the report is written. Task 6.
5. **The reference does not vary** (one measurement, or identical references): the spread ratio and Pearson r are blank, not an error or `inf`. Task 3.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/outputs.py` | Modify | Also writes `recordings.csv` |
| `src/dataset_config.py` | Modify | `ADDITIONAL_PARAMS`, optional list |
| `scripts/run.py`, `scripts/test.py` | Modify | Pass the test stores to `write_records` |
| `src/evaluation/pooling.py` | Create | Folds, epoch, stacking, eligibility, parameters |
| `src/evaluation/agreement.py` | Create | Statistics, pure functions on arrays |
| `src/evaluation/demographics.py` | Create | Core attributes, descriptive tables, ISO population checks |
| `src/evaluation/plots.py` | Modify | Report figures beside `recording_figure` |
| `src/evaluation/report.py` | Create | Report builder, the six sections, HTML |
| `scripts/evaluate.py` | Create | Arguments and the calls in order |
| `README.md`, `docs/evaluation.md`, `docs/adding_a_model.md`, `src/evaluation/__init__.py` | Modify | Documentation |

---

## Part A: `recordings.csv`

### Task 1: Write each test recording's attributes beside its records

**Files:**
- Modify: `src/outputs.py`
- Modify: `src/dataset_config.py`
- Modify: `scripts/run.py:189`
- Modify: `scripts/test.py:152`
- Modify: every `configs/datasets/*.yaml`
- Modify: `docs/adding_a_model.md:634`

**Interfaces:**
- Consumes: `load_stores` returns `{dataset: {store path: root attrs}}`; `Split.test` has the same shape.
- Produces:
  - `src.outputs.RECORDINGS_NAME = "recordings.csv"`
  - `src.outputs.recording_rows(stores: dict) -> pd.DataFrame`
  - `src.outputs.write_records(records, out_dir, interface, meta, stores) -> Path` (one new positional argument)
  - `DatasetConfig.ADDITIONAL_PARAMS: list`, which reaches each fold's `config.yaml` as `datasets.<name>.ADDITIONAL_PARAMS`

- [ ] **Step 1: Add the flattening and the table to `src/outputs.py`**

Below `WINDOWS_NAME`:

```python
RECORDINGS_NAME = "recordings.csv"
```

Below `window_rows`:

```python
def _flatten(attrs: dict, prefix: str = "") -> dict:
    """Scalar attrs keyed by dotted path (``skin_tone.clinician``); lists and
    other containers are left out."""
    out = {}
    for key, value in attrs.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{name}."))
        elif value is None or isinstance(value, (str, bool, int, float)):
            out[name] = value
    return out


def recording_rows(stores: dict) -> pd.DataFrame:
    """``recordings.csv``: one row per store with every scalar root attr, as
    the cache wrote it. Nothing is selected or renamed here."""
    rows = []
    for store, attrs in stores.items():
        flat = _flatten(attrs)
        flat.pop("recording", None)
        flat.pop("participant", None)
        rows.append({"recording": str(attrs.get("recording", Path(store).stem)),
                     "participant": str(attrs.get("participant", "")), **flat})
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Write the file from `write_records`**

Change the signature and docstring, and write the table after `windows.csv`:

```python
def write_records(records, out_dir, interface: InterfaceConfig, meta: dict,
                  stores: dict) -> Path:
    """Write the directory described in the module docstring; returns it.

    ``meta`` is what the caller knows and the records do not (the dataset and
    participant, the checkpoint's run directory, the command, the git state);
    the interface's rate, window, channels and traces are added here.
    ``stores`` is ``{store path: root attrs}`` of the dataset tested; the
    rows of the recordings the records cover go to ``recordings.csv``.
    """
    if not records:
        raise ValueError("no records to write")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window_rows(records, interface).to_csv(out_dir / WINDOWS_NAME, index=False)
    covered = {str(record["metadata"]["recording"]) for record in records}
    recordings = recording_rows(stores)
    recordings[recordings["recording"].isin(covered)].to_csv(
        out_dir / RECORDINGS_NAME, index=False)
```

The rest of the function is unchanged. Add one line to the module docstring's layout, under `windows.csv`:

```
      recordings.csv                   one row per recording: the cache's root
                                       attrs (posture, sex, age, ...)
```

- [ ] **Step 3: Pass the stores from both callers**

`scripts/run.py`, in `after_epoch`:

```python
        write_records(records, out_dir, interface, {**meta, "epoch": epoch},
                      split.test[args.test_participant_dataset])
```

`scripts/test.py`, in the loop over `datasets`:

```python
            write_records(records, out_dirs[name], interface,
                          {"dataset": name, **meta}, stores[name])
```

Add `recordings.csv` to the layout in the module docstring of `scripts/test.py` (the line `meta.json  windows.csv`).

- [ ] **Step 4: Add `ADDITIONAL_PARAMS` to the dataset config**

`src/dataset_config.py`:

```python
@dataclass
class DatasetConfig:
    CACHED_PATH: str = ""
    FILTERS: dict = field(default_factory=dict)   # {attr: {include, exclude}}
    ADDITIONAL_PARAMS: list = field(default_factory=list)   # root attrs the evaluation also reports
```

In `parse_dataset_config`, so existing files and existing runs' `config.yaml` still load:

```python
    cfg = build(DatasetConfig, mapping, where, optional=("ADDITIONAL_PARAMS",))
```

Replace the last paragraph of the module docstring:

```
``CACHED_PATH`` and ``FILTERS`` are required (``FILTERS: {}`` admits every
store); ``ADDITIONAL_PARAMS`` is optional, a list of root attrs (dotted when
nested) that ``scripts/evaluate.py`` reports beside the five core ones.
Unknown keys are refused.
```

- [ ] **Step 5: Add the empty list to every dataset file**

At the end of each file matched by `configs/datasets/*.yaml`:

```yaml

# Root attrs scripts/evaluate.py reports beside the core five (age_years, sex,
# skin_tone, posture, neck_circumference_cm); dotted when nested.
ADDITIONAL_PARAMS: []
```

- [ ] **Step 6: Document the file**

In `docs/adding_a_model.md`, in the row for `epoch_NN/test_records/<dataset>/`, change "`meta.json` (with the epoch), `windows.csv` (...)" to also name "`recordings.csv` (one row per recording with the cache's root attributes)".

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/outputs.py src/dataset_config.py scripts/run.py scripts/test.py`
Expected: `All checks passed!`

- [ ] **Step 8: Checkpoint: hand the user the verification command**

```bash
uv run scripts/run.py \
--datasets synthetic_neck synthetic_neck_high_snr synthetic_neck_demo \
--test-participant-dataset synthetic_neck \
--test-participant-id 1 \
--config configs/default_configs/physmamba_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
--epochs 1 --batch-size 2 --num-workers 4 \
--run-dir runs/recordings_check
```

Expected: `runs/recordings_check/epoch_01/test_records/synthetic_neck/recordings.csv` exists with one row, `recording` and `participant` both `1`, and columns including `posture`, `monk_tone` and `synthetic_neck.traces.heart_rate_bpm`.

---

## Part B: the evaluation script

### Task 2: Pooling

**Files:**
- Create: `src/evaluation/pooling.py`

**Interfaces:**
- Consumes: `src.outputs.META_NAME`, `RECORDINGS_NAME`, `RECORDS_DIR`, `WINDOWS_NAME`; `src.evaluation.recording.SIGNALS_NAME`, `RATES_NAME`; `src.config.load_yaml(path) -> dict`; `src.signal_transforms.is_absolute`, `is_cardiac`, `beat_labels`, `signal_unit`.
- Produces:
  - `IDENTITY = ("fold", "dataset", "participant", "recording", "perspective")`
  - `find_folds(run_dir) -> list[Path]`
  - `collect(run_dir, epoch: int | None) -> tuple[pd.DataFrame, dict]`: the pooled table and the first records directory's `meta.json`
  - `collect_epochs(run_dir) -> pd.DataFrame`: the same table over every epoch
  - `read_losses(run_dir) -> pd.DataFrame`: every fold's `losses.csv` with a `fold` column
  - `additional_params(run_dir) -> list[str]`
  - `mark_eligible(pooled, minimum: float, maximum: float) -> pd.DataFrame`: adds `eligible` and `reason`
  - `exclusions(pooled) -> pd.DataFrame`
  - `Parameter` (frozen dataclass: `key, label, unit, signal, ref, pred, kind, level, criteria`), `kind` is `"rate"` or `"level"`
  - `parameters(traces, pooled) -> list[Parameter]`
  - `paired(frame, parameter) -> pd.DataFrame`: identity columns, `subject`, `ref`, `pred`, `error`
- Pooled table columns: `IDENTITY`, `epoch`, `subject`, `signal`, `duration`, every column of `signals.csv`, `ref_hr pred_hr err_hr snr macc`, then every column of `recordings.csv` that does not collide.

- [ ] **Step 1: Write the module**

```python
"""One table of paired measurements from a run's per-recording tables.

A run directory holds one fold per held-out participant (or is itself one
fold). Under the chosen epoch of each, every
``test_records/<dataset>/<recording>/<perspective>/`` folder contributes its
``signals.csv`` and ``rates.csv`` rows, tagged with where they sit and joined
to the recording's row of ``recordings.csv``. One (recording, perspective)
is one paired measurement; it is eligible when its covered stretch is within
the segment bounds. Nothing here reads a waveform.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_yaml
from src.evaluation.recording import RATES_NAME, SIGNALS_NAME
from src.outputs import META_NAME, RECORDINGS_NAME, RECORDS_DIR, WINDOWS_NAME
from src.signal_transforms import beat_labels, is_absolute, is_cardiac, signal_unit

CONFIG_NAME, LOSSES_NAME = "config.yaml", "losses.csv"
EPOCH_PATTERN = re.compile(r"epoch_(\d+)")
IDENTITY = ("fold", "dataset", "participant", "recording", "perspective")
FUSED = "FUSED"
#: Seconds of slack on the segment bounds: the tables are written to six digits.
TOLERANCE = 1e-6
#: What one records directory must share with every other to be pooled.
SHAPE_KEYS = ("traces", "fs", "window_frames", "stride_frames")
#: The levels of an absolute signal the report headlines; the mean otherwise.
HEADLINE_LEVELS = {"ABP": ("max", "min", "mean")}
#: The signals ISO 81060-3 sets criteria for.
ISO_SIGNALS = ("ABP",)
#: Label and unit of a rate read off a trace that does not beat with the heart.
RATE_LABELS = {"RESP": ("Respiratory rate", "breaths/min")}


# ---------------------------------------------------------------------------
# Folds and epochs
# ---------------------------------------------------------------------------
def find_folds(run_dir) -> list:
    """The fold directories: the run directory itself when it carries a
    ``config.yaml``, else every subfolder that does."""
    run_dir = Path(run_dir)
    if (run_dir / CONFIG_NAME).is_file():
        return [run_dir]
    found = sorted(path.parent for path in run_dir.glob(f"*/{CONFIG_NAME}"))
    if not found:
        raise ValueError(f"{run_dir} holds no fold: no {CONFIG_NAME} in it or "
                         f"one level below")
    return found


def epochs_of(fold: Path) -> dict:
    """``{epoch: folder}`` for the epochs of a fold that carry test records."""
    found = {}
    for path in fold.glob("epoch_*"):
        match = EPOCH_PATTERN.fullmatch(path.name)
        if match and (path / RECORDS_DIR).is_dir():
            found[int(match.group(1))] = path
    return found


def epoch_dir(fold: Path, epoch: int | None) -> tuple:
    """``(epoch, folder)``: the epoch asked for, or the fold's last."""
    found = epochs_of(fold)
    if not found:
        raise ValueError(f"{fold} has no epoch with test records")
    if epoch is None:
        epoch = max(found)
    if epoch not in found:
        raise ValueError(f"{fold} has no test records for epoch {epoch}; it "
                         f"has {sorted(found)}")
    return epoch, found[epoch]


def records_dirs(epoch_folder: Path) -> list:
    """One dataset's records directory per entry, found by its ``meta.json``."""
    return sorted(path.parent for path in (epoch_folder / RECORDS_DIR).glob(f"*/{META_NAME}"))


# ---------------------------------------------------------------------------
# One records directory to rows
# ---------------------------------------------------------------------------
def folder_rows(folder: Path) -> pd.DataFrame:
    """The signal rows of one recording-and-camera folder with the rate of
    each trace beside them, and one more row per rate source that is not a
    trace (``FUSED``). A folder no window covered gives one blank row, so it
    can be listed as excluded."""
    signals = pd.read_csv(folder / SIGNALS_NAME)
    rates = pd.read_csv(folder / RATES_NAME).rename(columns={"source": "signal"})
    table = signals.merge(rates, on="signal", how="outer")
    if table.empty:
        return pd.DataFrame([{"signal": None, "t_start": np.nan, "t_end": np.nan}])
    span = signals[["t_start", "t_end"]].dropna()
    if len(span):
        table["t_start"] = table["t_start"].fillna(span["t_start"].iloc[0])
        table["t_end"] = table["t_end"].fillna(span["t_end"].iloc[0])
    return table


def dataset_rows(records: Path, fold: str, epoch: int) -> pd.DataFrame:
    """Every row of one dataset's records directory, tagged and joined to
    ``recordings.csv`` where there is one."""
    windows = pd.read_csv(records / WINDOWS_NAME, dtype=str)
    owner = windows.drop_duplicates("recording").set_index("recording")["participant"]
    frames = []
    for path in sorted(records.glob(f"*/*/{SIGNALS_NAME}")):
        folder = path.parent
        table = folder_rows(folder)
        recording = folder.parent.name
        tags = {"fold": fold, "dataset": records.name,
                "participant": str(owner.get(recording, "")),
                "recording": recording, "perspective": folder.name, "epoch": epoch}
        for position, (name, value) in enumerate(tags.items()):
            table.insert(position, name, value)
        frames.append(table)
    if not frames:
        raise ValueError(f"{records} holds no scored recording ({SIGNALS_NAME})")
    pooled = pd.concat(frames, ignore_index=True)
    pooled.insert(len(tags), "subject", pooled["dataset"] + "/" + pooled["participant"])
    pooled["duration"] = pooled["t_end"] - pooled["t_start"]
    if (records / RECORDINGS_NAME).is_file():
        attrs = pd.read_csv(records / RECORDINGS_NAME,
                            dtype={"recording": str, "participant": str})
        keep = ["recording", *(c for c in attrs.columns if c not in pooled.columns)]
        pooled = pooled.merge(attrs[keep].drop_duplicates("recording"),
                              on="recording", how="left")
    return pooled


def _shape(meta: dict) -> tuple:
    return tuple(str(meta[key]) for key in SHAPE_KEYS)


def collect(run_dir, epoch: int | None = None) -> tuple:
    """``(pooled, meta)``: every fold's rows at the chosen epoch, and the
    ``meta.json`` of the first records directory. Folds that disagree on the
    traces or the interface are refused."""
    frames, shapes, first = [], {}, None
    for fold in find_folds(run_dir):
        number, folder = epoch_dir(fold, epoch)
        found = records_dirs(folder)
        if not found:
            raise ValueError(f"{folder / RECORDS_DIR} holds no dataset directory "
                             f"with a {META_NAME}")
        for records in found:
            meta = json.loads((records / META_NAME).read_text(encoding="utf-8"))
            first = first or meta
            shapes[f"{fold.name}/{records.name}"] = _shape(meta)
            frames.append(dataset_rows(records, fold.name, number))
    if len(set(shapes.values())) > 1:
        listed = "; ".join(f"{name}: {dict(zip(SHAPE_KEYS, shape))}"
                           for name, shape in shapes.items())
        raise ValueError(f"the folds disagree on the traces or the interface and "
                         f"cannot be pooled: {listed}")
    return pd.concat(frames, ignore_index=True), first


def collect_epochs(run_dir) -> pd.DataFrame:
    """The pooled table over every epoch of every fold, for the curves."""
    frames = []
    for fold in find_folds(run_dir):
        for number, folder in sorted(epochs_of(fold).items()):
            for records in records_dirs(folder):
                frames.append(dataset_rows(records, fold.name, number))
    return pd.concat(frames, ignore_index=True)


def read_losses(run_dir) -> pd.DataFrame:
    """Every fold's ``losses.csv`` with the fold beside each row; empty when
    no fold has one."""
    frames = []
    for fold in find_folds(run_dir):
        path = fold / LOSSES_NAME
        if path.is_file():
            table = pd.read_csv(path)
            table.insert(0, "fold", fold.name)
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def additional_params(run_dir) -> list:
    """The ``ADDITIONAL_PARAMS`` of every dataset in every fold's
    ``config.yaml``, in first-seen order."""
    names = []
    for fold in find_folds(run_dir):
        datasets = (load_yaml(fold / CONFIG_NAME) or {}).get("datasets") or {}
        for mapping in datasets.values():
            for name in (mapping or {}).get("ADDITIONAL_PARAMS") or []:
                if str(name) not in names:
                    names.append(str(name))
    return names


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def mark_eligible(pooled: pd.DataFrame, minimum: float, maximum: float) -> pd.DataFrame:
    """``eligible`` and ``reason`` per row: the covered stretch is within the
    bounds, both inclusive."""
    pooled = pooled.copy()
    duration = pooled["duration"]
    short, long = duration < minimum - TOLERANCE, duration > maximum + TOLERANCE
    pooled["eligible"] = duration.notna() & ~short & ~long
    pooled["reason"] = np.select(
        [duration.isna(), short, long],
        ["no window covered the recording",
         f"covered stretch shorter than {minimum:g} s",
         f"covered stretch longer than {maximum:g} s"], default="")
    return pooled


def exclusions(pooled: pd.DataFrame) -> pd.DataFrame:
    """One row per excluded (recording, perspective)."""
    columns = [*IDENTITY, "duration", "reason"]
    out = pooled.loc[~pooled["eligible"], columns].drop_duplicates(list(IDENTITY))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The parameters a run is scored on
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameter:
    key: str          # file-safe name: hr_ABP, ABP_max
    label: str        # as the report prints it
    unit: str
    signal: str       # the row of the pooled table it is read from
    ref: str          # reference column
    pred: str         # prediction column
    kind: str         # "rate" | "level"
    level: str        # max | mean | min for a level, "" for a rate
    criteria: bool    # ISO 81060-3 sets pass / fail thresholds for it


def parameters(traces, pooled: pd.DataFrame) -> list:
    """What the run is scored on, from the traces it predicted: a rate per
    source ``rates.csv`` carried, and the headline levels of each
    absolute-class trace."""
    out = []
    rated = pooled.loc[pooled["ref_hr"].notna(), "signal"].dropna().unique().tolist()
    for source in [*(s for s in traces if s in rated), *(s for s in rated if s not in traces)]:
        beats = source == FUSED or is_cardiac(source)
        label, unit = (("Heart rate", "bpm") if beats
                       else RATE_LABELS.get(source, ("Rate", "1/min")))
        out.append(Parameter(f"rate_{source}", f"{label} ({source})", unit, source,
                             "ref_hr", "pred_hr", "rate", "", False))
    for sig in traces:
        if not is_absolute(sig):
            continue
        names = beat_labels(sig)
        for level in HEADLINE_LEVELS.get(sig, ("mean",)):
            out.append(Parameter(f"{sig}_{level}", f"{sig} {names[level]}",
                                 signal_unit(sig), sig, f"ref_{level}_mean",
                                 f"pred_{level}_mean", "level", level,
                                 sig in ISO_SIGNALS))
    return out


def paired(frame: pd.DataFrame, parameter: Parameter) -> pd.DataFrame:
    """The finite (reference, prediction) pairs of one parameter, with the
    error and where each pair sits."""
    rows = frame[frame["signal"] == parameter.signal]
    out = rows[[*IDENTITY, "subject"]].copy()
    out["ref"] = pd.to_numeric(rows[parameter.ref], errors="coerce")
    out["pred"] = pd.to_numeric(rows[parameter.pred], errors="coerce")
    out = out.dropna(subset=["ref", "pred"])
    out["error"] = out["pred"] - out["ref"]
    return out.reset_index(drop=True)
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/evaluation/pooling.py`
Expected: `All checks passed!`

- [ ] **Step 3: Checkpoint: hand the user the verification command**

```bash
uv run python -c "
from src.evaluation.pooling import collect, mark_eligible, exclusions, parameters
pooled, meta = collect('runs/synthetic_benchmark_physmamba')
pooled = mark_eligible(pooled, 20, 30)
print(pooled[['fold','subject','recording','perspective','signal','duration','eligible']].to_string())
print(exclusions(pooled))
print([p.label for p in parameters(meta['traces'], pooled)])
"
```

Expected: 16 rows (4 folds × ABP, CVP, PPG, FUSED), every `duration` 30 and `eligible` True, subjects `synthetic_neck/1` to `synthetic_neck/4`, no exclusions, and the labels `Heart rate (ABP)`, `Heart rate (CVP)`, `Heart rate (PPG)`, `Heart rate (FUSED)`, `ABP systolic`, `ABP diastolic`, `ABP MAP`, `CVP mean`.

Review Focus 1: repeat with one fold's own path in place of the run directory; expected 4 rows and no error.

### Task 3: Agreement statistics

**Files:**
- Create: `src/evaluation/agreement.py`

**Interfaces:**
- Consumes: `src.evaluation.recording.pearson(a, b) -> float`
- Produces, all pure functions on arrays:
  - `repeated_measures(error, subject) -> dict` with keys `n k m_min m_median m_max mean_error sd f_ba ms_between ms_within s_corr icc n_ind note`
  - `limits(stats: dict) -> tuple[float, float]`
  - `spread(error) -> dict` with keys `mae rmse`
  - `within_shares(error) -> dict` with keys `within_5 within_10 within_15` (percent)
  - `criteria(stats: dict) -> list[dict]` with keys `criterion value threshold met`
  - `rate_metrics(ref, pred) -> dict` with keys `mae rmse mape r`
  - `baseline(ref, pred) -> dict` with keys `model_mae baseline_mae model_rmse baseline_rmse spread_ratio`

- [ ] **Step 1: Write the module**

```python
"""Agreement between predictions and references over pooled measurements.

The accuracy statistics are ISO 81060-3:2022 clauses 4.5.2 and 5.1.3
(printed pp. 12 and 15). With ``n`` paired measurements of ``k`` subjects,
``m_i`` of them from subject ``i``:

    f_BA  = (n^2 - sum m_i^2) / ((k - 1) n)                          (10)
    MS_B  = sum m_i (mean_i - mean)^2 / (k - 1)                      (11)
    MS_W  = sum (m_i - 1) var_i / (n - k)                            (12)
    s_corr = sqrt((MS_B - MS_W) / f_BA + MS_W)                        (9)
    ICC   = ((MS_B - MS_W) / f_BA) / ((MS_B - MS_W) / f_BA + MS_W)    (5)
    N_ind = k (1 + (1 - ICC)(f_BA - 1))                               (6)

The standard prints (5) and (9) to (12) for unequal ``m_i``; (6) it prints
with ``r`` measurements per subject, for which ``f_BA`` stands here and to
which it reduces when every subject contributes equally. Errors are
prediction minus reference. Every function takes arrays and returns numbers.
"""

import numpy as np

from src.evaluation.recording import pearson

#: Limits of agreement are the mean error plus and minus this many s_corr.
LIMIT_Z = 1.96
#: ISO 81060-3 clause 5.1.4 a) to c) and clause 4.5.1 b) 3).
ISO_MEAN_ERROR, ISO_S_CORR, ISO_N_IND, ISO_SUBJECTS = 6.0, 10.0, 278, 30
#: Cumulative accuracy: the share of absolute errors within each, in the
#: parameter's unit (the BHS grading's 5 / 10 / 15 mmHg).
WITHIN = (5, 10, 15)
_NAN = float("nan")


def repeated_measures(error, subject) -> dict:
    """The accuracy statistics of the module docstring. Where they cannot be
    estimated the plain SD stands for ``s_corr`` and ``note`` says why."""
    error = np.asarray(error, dtype=np.float64)
    ids, index = np.unique(np.asarray(subject, dtype=str), return_inverse=True)
    n, k = error.size, ids.size
    counts = np.bincount(index, minlength=k).astype(np.float64)
    sd = float(error.std(ddof=1)) if n > 1 else _NAN
    out = {"n": n, "k": k,
           "m_min": float(counts.min()) if k else _NAN,
           "m_median": float(np.median(counts)) if k else _NAN,
           "m_max": float(counts.max()) if k else _NAN,
           "mean_error": float(error.mean()) if n else _NAN, "sd": sd,
           "f_ba": _NAN, "ms_between": _NAN, "ms_within": _NAN,
           "s_corr": sd, "icc": _NAN, "n_ind": float(n), "note": ""}
    if n == 0:
        out["note"] = "no paired measurements"
        return out
    if k < 2 or n == k:
        why = "one subject" if k < 2 else "one measurement per subject"
        out["note"] = (f"{why}: s_corr is the plain SD, the ICC is not "
                       f"estimable and N_ind is n")
        return out
    subject_mean = np.bincount(index, weights=error, minlength=k) / counts
    f_ba = (n ** 2 - (counts ** 2).sum()) / ((k - 1) * n)
    ms_between = float((counts * (subject_mean - error.mean()) ** 2).sum() / (k - 1))
    ms_within = float(((error - subject_mean[index]) ** 2).sum() / (n - k))
    between = (ms_between - ms_within) / f_ba
    if between < 0:
        between = 0.0
        out["note"] = ("the between-subject mean square is below the "
                       "within-subject one: the between-subject component is "
                       "zero, s_corr is the within-subject SD and the ICC is 0")
    total = between + ms_within
    icc = between / total if total > 0 else _NAN
    out.update({"f_ba": float(f_ba), "ms_between": ms_between, "ms_within": ms_within,
                "s_corr": float(np.sqrt(total)), "icc": float(icc),
                "n_ind": float(k * (1 + (1 - icc) * (f_ba - 1))) if np.isfinite(icc)
                         else float(n)})
    return out


def limits(stats: dict) -> tuple:
    """The 95 % limits of agreement, on the corrected SD."""
    half = LIMIT_Z * stats["s_corr"]
    return stats["mean_error"] - half, stats["mean_error"] + half


def spread(error) -> dict:
    error = np.asarray(error, dtype=np.float64)
    if error.size == 0:
        return {"mae": _NAN, "rmse": _NAN}
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean()))}


def within_shares(error) -> dict:
    """Percent of absolute errors within each of ``WITHIN``."""
    error = np.abs(np.asarray(error, dtype=np.float64))
    return {f"within_{bound}": float((error <= bound).mean() * 100) if error.size else _NAN
            for bound in WITHIN}


def criteria(stats: dict) -> list:
    """The ISO 81060-3 accuracy criteria against the statistics."""
    rows = (
        ("mean error (5.1.4 a)", abs(stats["mean_error"]), f"within ±{ISO_MEAN_ERROR:.1f}",
         abs(stats["mean_error"]) <= ISO_MEAN_ERROR),
        ("s_corr (5.1.4 b)", stats["s_corr"], f"≤ {ISO_S_CORR:.1f}",
         stats["s_corr"] <= ISO_S_CORR),
        ("N_ind (5.1.4 c)", stats["n_ind"], f"≥ {ISO_N_IND}", stats["n_ind"] >= ISO_N_IND),
        ("subjects (4.5.1 b)", stats["k"], f"≥ {ISO_SUBJECTS}", stats["k"] >= ISO_SUBJECTS),
    )
    return [{"criterion": name, "value": float(value), "threshold": threshold,
             "met": bool(met) if np.isfinite(value) else False}
            for name, value, threshold, met in rows]


def rate_metrics(ref, pred) -> dict:
    """What rPPG papers report of a rate: MAE, RMSE, MAPE (percent of the
    reference) and Pearson r."""
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if ref.size == 0:
        return {"mae": _NAN, "rmse": _NAN, "mape": _NAN, "r": _NAN}
    error = pred - ref
    return {**spread(error),
            "mape": float((np.abs(error) / np.abs(ref)).mean() * 100) if (ref != 0).all()
                    else _NAN,
            "r": pearson(pred, ref)}


def baseline(ref, pred) -> dict:
    """The model beside a constant predictor, the mean of the references
    themselves (which favours the constant: it has seen them), and the ratio
    of the predictions' SD to the references'. A ratio near 0 is a model that
    answers the same level for everyone."""
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if ref.size == 0:
        return {"model_mae": _NAN, "baseline_mae": _NAN, "model_rmse": _NAN,
                "baseline_rmse": _NAN, "spread_ratio": _NAN}
    model, constant = spread(pred - ref), spread(ref.mean() - ref)
    ref_sd = ref.std(ddof=1) if ref.size > 1 else 0.0
    return {"model_mae": model["mae"], "baseline_mae": constant["mae"],
            "model_rmse": model["rmse"], "baseline_rmse": constant["rmse"],
            "spread_ratio": float(pred.std(ddof=1) / ref_sd) if ref_sd > 0 else _NAN}
```

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/evaluation/agreement.py`
Expected: `All checks passed!`

- [ ] **Step 3: Checkpoint: hand the user the verification command**

```bash
uv run python -W error -c "
from src.evaluation.agreement import repeated_measures, baseline
# equal counts: f_BA is r
s = repeated_measures([1,2,3,5,6,7], ['a','a','a','b','b','b'])
print(s['f_ba'], round(s['s_corr'], 4), round(s['icc'], 4), round(s['n_ind'], 4))
# unequal counts
print(repeated_measures([1,2,3,5,6], ['a','a','a','b','b'])['f_ba'])
# one measurement per subject
print(repeated_measures([1,2,3,4], ['a','b','c','d']))
# a reference that does not vary
print(baseline([80,80,80], [70,80,90]))
"
```

Expected, with no warning raised (`-W error`):
- first line `3.0 2.9439 0.8846 2.4615`
- second line `2.4`
- third: `s_corr` equal to `sd` (1.29099), `icc` nan, `n_ind` 4.0, and the note (Review Focus 2)
- fourth: `spread_ratio` nan (Review Focus 5)

Working for the first line: subject means 2 and 6, grand mean 4, `MS_B = 3·4 + 3·4 = 24`, `MS_W = (2 + 2)/4 = 1`, between `= 23/3 = 7.6667`, `s_corr = sqrt(8.6667) = 2.9439`, `ICC = 7.6667/8.6667 = 0.8846`, `N_ind = 2(1 + 0.1154·2) = 2.4615`.

### Task 4: Demographics

**Files:**
- Create: `src/evaluation/demographics.py`

**Interfaces:**
- Consumes: `pooling.paired`, `pooling.Parameter`; `agreement.repeated_measures`
- Produces:
  - `CORE: dict[str, str]` (attribute name to printed label), `NOT_RECORDED = "not recorded"`
  - `with_core(pooled) -> pd.DataFrame`: the five core columns resolved
  - `characteristics(pooled, additional: list[str]) -> pd.DataFrame`, columns `level characteristic n_all all n_eligible eligible`
  - `reference_spread(pooled, params) -> pd.DataFrame`, same columns
  - `population_checks(pooled) -> pd.DataFrame`, columns `requirement clause n share_percent required_percent met`
  - `bp_distribution(pooled, params) -> pd.DataFrame`, columns `parameter band n share_percent required_percent met`
  - `band_edges(parameter) -> tuple[float, ...]`
  - `stratified(measurements, params) -> pd.DataFrame`, columns `parameter characteristic group n subjects mean_error sd small`

- [ ] **Step 1: Write the module**

```python
"""Who was tested: the spread of the test population and of the reference
values, against the population requirements of ISO 81060-3:2022.

Five root attrs are core and carry the same name in every cache:
``age_years``, ``sex``, ``skin_tone`` (Monk 1 to 10), ``posture`` and
``neck_circumference_cm``. Where a cache holds several measurements of one
(``skin_tone.self``, ``skin_tone.clinician``, ...) the value is their
median. Anything absent reads "not recorded".

Three levels, because the facts sit at three: a subject's characteristics
are counted once per subject, a recording's once per recording, and a
reference value once per recording, whatever the number of cameras.
"""

import numpy as np
import pandas as pd

from src.evaluation.agreement import repeated_measures
from src.evaluation.pooling import paired

CORE = {"age_years": "Age (years)", "sex": "Sex", "skin_tone": "Skin tone (Monk)",
        "posture": "Posture", "neck_circumference_cm": "Neck circumference (cm)"}
SUBJECT_LEVEL = ("age_years", "sex", "skin_tone", "neck_circumference_cm")
RECORDING_LEVEL = ("posture",)
#: Reported as counts per value whatever their dtype.
CATEGORICAL = ("sex", "skin_tone", "posture")
STRATA = ("posture", "skin_tone", "sex")
NOT_RECORDED = "not recorded"
#: A stratum of fewer measurements than this is flagged as too small to read.
SMALL_GROUP = 5
RECORDING_KEY = ["dataset", "recording"]

#: ISO 81060-3 clause 4.3.2.2 and 4.3.2.3.2: (requirement, clause, column,
#: test on the recorded values, percent required).
POPULATION = (
    ("male", "4.3.2.2", "sex", lambda v: v.astype(str) == "M", 30),
    ("female", "4.3.2.2", "sex", lambda v: v.astype(str) == "F", 30),
    ("age ≥ 50 y", "4.3.2.3.2", "age_years", lambda v: v >= 50, 40),
    ("age ≥ 60 y", "4.3.2.3.2", "age_years", lambda v: v >= 60, 25),
    ("age ≥ 70 y", "4.3.2.3.2", "age_years", lambda v: v >= 70, 10),
)
#: ISO 81060-3 clause 4.3.3, in mmHg, per level of ABP: (band, test, percent
#: of reference readings required).
BP_BANDS = {
    "max": (("≤ 90", lambda v: v <= 90, 5), ("≤ 110", lambda v: v <= 110, 20),
            ("> 110 and < 140", lambda v: (v > 110) & (v < 140), 20),
            ("≥ 140", lambda v: v >= 140, 20), ("≥ 160", lambda v: v >= 160, 5)),
    "min": (("≤ 50", lambda v: v <= 50, 5), ("≤ 60", lambda v: v <= 60, 20),
            ("> 60 and < 80", lambda v: (v > 60) & (v < 80), 20),
            ("≥ 80", lambda v: v >= 80, 20), ("≥ 90", lambda v: v >= 90, 5)),
    "mean": (("≤ 65", lambda v: v <= 65, 5), ("≤ 75", lambda v: v <= 75, 20),
             ("> 75 and < 100", lambda v: (v > 75) & (v < 100), 20),
             ("≥ 100", lambda v: v >= 100, 20), ("≥ 115", lambda v: v >= 115, 5)),
}
BAND_EDGES = {"max": (90, 110, 140, 160), "min": (50, 60, 80, 90),
              "mean": (65, 75, 100, 115)}


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------
def attribute(frame: pd.DataFrame, name: str) -> pd.Series:
    """The column of that name; failing that the median of the numeric
    columns under it (``name.*``); failing that blank."""
    if name in frame.columns and frame[name].notna().any():
        return frame[name]
    members = [c for c in frame.columns if c.startswith(f"{name}.")]
    if members:
        numeric = frame[members].apply(pd.to_numeric, errors="coerce")
        if numeric.notna().any().any():
            return numeric.median(axis=1, skipna=True)
    return pd.Series(np.nan, index=frame.index, dtype=object)


def with_core(pooled: pd.DataFrame) -> pd.DataFrame:
    """The pooled table with the five core attributes as columns."""
    pooled = pooled.copy()
    for name in CORE:
        pooled[name] = attribute(pooled, name)
    return pooled


def describe(values: pd.Series, categorical: bool) -> tuple:
    """``(n recorded, summary)``: counts per value, or mean ± SD and range."""
    values = values.dropna()
    if values.empty:
        return 0, NOT_RECORDED
    numeric = pd.to_numeric(values, errors="coerce")
    if categorical or numeric.isna().any() or pd.api.types.is_bool_dtype(values):
        counts = values.map(lambda v: f"{v:g}" if isinstance(v, float) else str(v)).value_counts()
        parts = [f"{value}: {count} ({count / len(values) * 100:.0f} %)"
                 for value, count in counts.sort_index().items()]
        return len(values), ", ".join(parts)
    sd = f" ± {numeric.std(ddof=1):.3g}" if len(numeric) > 1 else ""
    return len(numeric), (f"{numeric.mean():.3g}{sd} "
                          f"(range {numeric.min():.3g} to {numeric.max():.3g})")


def _row(level: str, label: str, everyone: pd.Series, eligible: pd.Series,
         categorical: bool) -> dict:
    n_all, text_all = describe(everyone, categorical)
    n_eligible, text_eligible = describe(eligible, categorical)
    return {"level": level, "characteristic": label, "n_all": n_all, "all": text_all,
            "n_eligible": n_eligible, "eligible": text_eligible}


def characteristics(pooled: pd.DataFrame, additional: list) -> pd.DataFrame:
    """Subject and recording characteristics, over every test recording and
    over the eligible ones."""
    kept = pooled[pooled["eligible"]]
    rows = []
    for level, key, names in (("subject", ["subject"], SUBJECT_LEVEL),
                              ("recording", RECORDING_KEY, RECORDING_LEVEL),
                              ("recording", RECORDING_KEY, tuple(additional))):
        everyone, eligible = pooled.drop_duplicates(key), kept.drop_duplicates(key)
        for name in names:
            rows.append(_row(level, CORE.get(name, name), attribute(everyone, name),
                             attribute(eligible, name), name in CATEGORICAL))
    return pd.DataFrame(rows)


def _references(frame: pd.DataFrame, parameter) -> pd.Series:
    """One reference value per recording: the first camera's."""
    pairs = paired(frame, parameter).sort_values("perspective")
    return pairs.drop_duplicates(RECORDING_KEY)["ref"]


def reference_spread(pooled: pd.DataFrame, params: list) -> pd.DataFrame:
    kept = pooled[pooled["eligible"]]
    return pd.DataFrame([
        _row("reference", f"{p.label} ({p.unit})", _references(pooled, p),
             _references(kept, p), False) for p in params])


# ---------------------------------------------------------------------------
# ISO 81060-3 population requirements
# ---------------------------------------------------------------------------
def _check(values: pd.Series, test, required: float) -> dict:
    values = values.dropna()
    if values.empty:
        return {"n": 0, "share_percent": np.nan, "required_percent": required,
                "met": NOT_RECORDED}
    share = float(test(values).mean() * 100)
    return {"n": len(values), "share_percent": share, "required_percent": required,
            "met": "yes" if share >= required else "no"}


def population_checks(pooled: pd.DataFrame) -> pd.DataFrame:
    """Sex and age shares of the eligible subjects, and the postures seen."""
    kept = pooled[pooled["eligible"]]
    subjects = kept.drop_duplicates("subject")
    rows = [{"requirement": name, "clause": clause,
             **_check(pd.to_numeric(subjects[column], errors="coerce")
                      if column == "age_years" else subjects[column], test, required)}
            for name, clause, column, test, required in POPULATION]
    postures = kept.drop_duplicates(RECORDING_KEY)["posture"].dropna()
    rows.append({"requirement": "different postures evaluated", "clause": "4.2 b)",
                 "n": len(postures), "share_percent": np.nan, "required_percent": np.nan,
                 "met": NOT_RECORDED if postures.empty
                        else ("yes" if postures.nunique() > 1 else "no")})
    return pd.DataFrame(rows)


def bp_distribution(pooled: pd.DataFrame, params: list) -> pd.DataFrame:
    """The reference readings of each ISO parameter against the bands of
    clause 4.3.3, over the eligible recordings."""
    kept = pooled[pooled["eligible"]]
    rows = []
    for p in params:
        if not p.criteria:
            continue
        values = _references(kept, p)
        for band, test, required in BP_BANDS[p.level]:
            rows.append({"parameter": p.label, "band": band,
                         **_check(values, test, required)})
    return pd.DataFrame(rows)


def band_edges(parameter) -> tuple:
    return BAND_EDGES[parameter.level] if parameter.criteria else ()


# ---------------------------------------------------------------------------
# Accuracy by group
# ---------------------------------------------------------------------------
def stratified(measurements: pd.DataFrame, params: list) -> pd.DataFrame:
    """Mean error and SD of each parameter within each posture, skin tone and
    sex; groups under ``SMALL_GROUP`` measurements are flagged."""
    rows = []
    for p in params:
        pairs = paired(measurements, p)
        source = measurements[measurements["signal"] == p.signal].dropna(
            subset=[p.ref, p.pred]).reset_index(drop=True)
        for name in STRATA:
            groups = source[name].dropna()
            for group, index in groups.groupby(groups).groups.items():
                chosen = pairs.loc[index]
                stats = repeated_measures(chosen["error"], chosen["subject"])
                rows.append({"parameter": p.label, "characteristic": CORE[name],
                             "group": f"{group:g}" if isinstance(group, float) else str(group),
                             "n": stats["n"], "subjects": stats["k"],
                             "mean_error": stats["mean_error"], "sd": stats["sd"],
                             "small": "yes" if stats["n"] < SMALL_GROUP else ""})
    return pd.DataFrame(rows, columns=["parameter", "characteristic", "group", "n",
                                       "subjects", "mean_error", "sd", "small"])
```

`stratified` relies on `paired` and `source` having the same rows in the same order: both take the rows of `p.signal` and drop those with a blank reference or prediction. `paired` coerces to numeric first; the columns are numeric already, so the two agree.

- [ ] **Step 2: Lint**

Run: `uv run ruff check src/evaluation/demographics.py`
Expected: `All checks passed!`

- [ ] **Step 3: Checkpoint: hand the user the verification command**

```bash
uv run python -c "
from src.evaluation.pooling import collect, mark_eligible, parameters
from src.evaluation.demographics import with_core, characteristics, reference_spread, population_checks, bp_distribution
pooled, meta = collect('runs/synthetic_benchmark_physmamba')
pooled = mark_eligible(with_core(pooled), 20, 30)
params = parameters(meta['traces'], pooled)
for table in (characteristics(pooled, []), reference_spread(pooled, params), population_checks(pooled), bp_distribution(pooled, params)):
    print(table.to_string(), end='\n\n')
"
```

Expected on the run as it stands (no `recordings.csv`): every characteristic reads `not recorded` with `n_all` 0 (Review Focus 3); the reference rows show 4 values each; the population checks read `not recorded`; the BP distribution has 15 rows with `n` 4.

### Task 5: Report figures

**Files:**
- Modify: `src/evaluation/plots.py`

**Interfaces:**
- Consumes: pair tables from `pooling.paired` (columns `subject ref pred error`); `agreement.limits`
- Produces, each returning a `plt.Figure` the caller saves and closes:
  - `REPORT_WIDTH = 6.3`
  - `bland_altman_figure(pairs, stats: dict, unit: str, title: str)`
  - `agreement_figure(pairs, unit: str, title: str)`
  - `histogram_figure(values, label: str, title: str, edges=())`
  - `counts_figure(columns: dict, title: str)`: `{label: pd.Series}`, one bar chart each
  - `box_figure(frame, value: str, label: str, title: str)`: one box per `signal`
  - `curves_figure(frame, value: str, label: str, title: str)`: `value` against `epoch`, one panel per `signal`

- [ ] **Step 1: Load the `dataviz` skill** and read it before writing the chart code. Keep the module's existing theme (`sns.set_theme(style="whitegrid", context="paper")`) and its colours (`LABEL_COLOUR`, `PRED_COLOUR`).

- [ ] **Step 2: Update the module docstring**

```python
"""The figures: one trace of one recording (the label, the combined
prediction with its spread across the overlapping windows, and the detected
beats), and the pooled figures of a run's evaluation report.

Seaborn on the Agg backend. Every function draws and returns the figure; the
caller saves and closes it. Report figures are ``REPORT_WIDTH`` inches wide,
the text width of A4 portrait, so one file serves Markdown, HTML and print.
"""
```

- [ ] **Step 3: Add the report figures at the end of the file**

```python
# ---------------------------------------------------------------------------
# The pooled figures of a run's evaluation report
# ---------------------------------------------------------------------------
REPORT_WIDTH = 6.3
#: Above this many subjects the legend would not fit; the colours stay.
MAX_LEGEND_SUBJECTS = 10
LINE_COLOUR = "0.35"


def _report_axes(height: float = 3.6, columns: int = 1):
    figure, axes = plt.subplots(1, columns, figsize=(REPORT_WIDTH, height), squeeze=False)
    return figure, axes[0]


def _by_subject(pairs: pd.DataFrame, x, y, axis) -> None:
    legend = "brief" if pairs["subject"].nunique() <= MAX_LEGEND_SUBJECTS else False
    sns.scatterplot(x=x, y=y, hue=pairs["subject"], ax=axis, s=28, edgecolor="white",
                    linewidth=0.4, legend=legend)
    if legend:
        axis.legend(title="subject", fontsize=7, title_fontsize=7, loc="upper center",
                    bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False)


def bland_altman_figure(pairs: pd.DataFrame, stats: dict, unit: str, title: str) -> plt.Figure:
    """Error against the mean of reference and prediction, with the mean
    error and the 95 % limits of agreement on the corrected SD."""
    figure, (axis,) = _report_axes()
    _by_subject(pairs, (pairs["ref"] + pairs["pred"]) / 2, pairs["error"], axis)
    half = 1.96 * stats["s_corr"]
    for value, style, name in ((stats["mean_error"], "-", "mean error"),
                               (stats["mean_error"] - half, "--", "lower limit"),
                               (stats["mean_error"] + half, "--", "upper limit")):
        if np.isfinite(value):
            axis.axhline(value, color=LINE_COLOUR, linestyle=style, linewidth=0.9)
            axis.annotate(f"{name} {value:.3g}", xy=(1.0, value), xycoords=("axes fraction", "data"),
                          ha="right", va="bottom", fontsize=7, color=LINE_COLOUR)
    axis.set_xlabel(f"mean of reference and prediction ({unit})")
    axis.set_ylabel(f"prediction − reference ({unit})")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def agreement_figure(pairs: pd.DataFrame, unit: str, title: str) -> plt.Figure:
    """Prediction against reference with the line of identity."""
    figure, (axis,) = _report_axes(height=4.2)
    _by_subject(pairs, pairs["ref"], pairs["pred"], axis)
    low = float(min(pairs["ref"].min(), pairs["pred"].min()))
    high = float(max(pairs["ref"].max(), pairs["pred"].max()))
    pad = 0.05 * (high - low) or 1.0
    axis.plot([low - pad, high + pad], [low - pad, high + pad], color=LINE_COLOUR,
              linestyle="--", linewidth=0.9)
    axis.set_xlabel(f"reference ({unit})")
    axis.set_ylabel(f"prediction ({unit})")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def histogram_figure(values, label: str, title: str, edges=()) -> plt.Figure:
    """The spread of one quantity; ``edges`` are drawn as vertical lines (the
    band edges of ISO 81060-3 clause 4.3.3)."""
    figure, (axis,) = _report_axes(height=3.0)
    sns.histplot(x=np.asarray(values, dtype=np.float64), ax=axis, color=PRED_COLOUR,
                 edgecolor="white")
    for edge in edges:
        axis.axvline(edge, color=LINE_COLOUR, linestyle="--", linewidth=0.9)
    axis.set_xlabel(label)
    axis.set_ylabel("count")
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def counts_figure(columns: dict, title: str) -> plt.Figure:
    """One bar chart of counts per value for each ``{label: values}``."""
    figure, axes = _report_axes(height=3.0, columns=len(columns))
    for axis, (label, values) in zip(axes, columns.items()):
        counts = values.dropna().astype(str).value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.to_numpy(), ax=axis, color=PRED_COLOUR)
        axis.set_xlabel(label)
        axis.set_ylabel("count")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure


def box_figure(frame: pd.DataFrame, value: str, label: str, title: str) -> plt.Figure:
    """One box per trace of a per-measurement metric, with every measurement
    as a point."""
    figure, (axis,) = _report_axes(height=3.0)
    sns.boxplot(data=frame, x="signal", y=value, ax=axis, color=PRED_COLOUR,
                width=0.5, fliersize=0, boxprops={"alpha": 0.4})
    sns.stripplot(data=frame, x="signal", y=value, ax=axis, color=LABEL_COLOUR, size=3)
    axis.set_xlabel("trace")
    axis.set_ylabel(label)
    axis.set_title(title, fontsize=10)
    figure.tight_layout()
    return figure


def curves_figure(frame: pd.DataFrame, value: str, label: str, title: str) -> plt.Figure:
    """``value`` against the epoch, one panel per trace (their units differ);
    the line is the mean and the band the spread over folds and recordings."""
    signals = list(dict.fromkeys(frame["signal"]))
    figure, axes = _report_axes(height=2.8, columns=len(signals))
    for axis, sig in zip(axes, signals):
        sns.lineplot(data=frame[frame["signal"] == sig], x="epoch", y=value, ax=axis,
                     color=PRED_COLOUR, errorbar="sd")
        axis.set_title(sig, fontsize=9)
        axis.set_xlabel("epoch")
        axis.set_ylabel(label if axis is axes[0] else "")
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check src/evaluation/plots.py`
Expected: `All checks passed!`

The figures are verified by eye in Task 6, in the report.

### Task 6: Report builder, script and documentation

**Files:**
- Create: `src/evaluation/report.py`
- Create: `scripts/evaluate.py`
- Modify: `pyproject.toml`, `uv.lock` (through `uv add`)
- Modify: `README.md` (after "Leave one out folds (Experiments)", before "Train on dataset(s) A, B..., test on dataset X")
- Modify: `docs/evaluation.md`
- Modify: `src/evaluation/__init__.py`

**Interfaces:**
- Consumes: everything Tasks 2 to 5 produce, under the names listed there.
- Produces:
  - `src.evaluation.report.Report(folder)` with `heading(text, level=2)`, `text(paragraph)`, `table(name, frame)`, `figure(name, figure, caption)`, `write(title) -> Path`
  - `src.evaluation.report.build(out_dir, run_dir, pooled, measurements, excluded, params, meta, curves, losses, additional, bounds) -> Path` (the path of `report.md`)
  - `scripts/evaluate.py` with `main(argv=None) -> Path`

- [ ] **Step 1: Add the dependency**

Run: `uv add markdown-it-py`
Expected: `pyproject.toml` lists `markdown-it-py` and `uv.lock` is updated. It is pure Python, so nothing differs between Windows, Linux and macOS.

- [ ] **Step 2: Write `src/evaluation/report.py`**

```python
"""The evaluation report of a run: every table as a CSV, every figure as a
PNG, and the document that reads them, as Markdown and as one HTML file.

``Report`` collects the document while it writes the files beside it;
``build`` fills it, section by section, from the pooled measurements. The
Markdown holds headings, paragraphs, pipe tables and image links and nothing
else, so any renderer reads it. The HTML is the same document with the
figures embedded and an A4 print stylesheet.
"""

import base64
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt

from src.evaluation import demographics
from src.evaluation.agreement import (
    ISO_SUBJECTS, baseline, criteria, limits, rate_metrics, repeated_measures,
    spread, within_shares,
)
from src.evaluation.plots import (
    agreement_figure, bland_altman_figure, box_figure, counts_figure,
    curves_figure, histogram_figure,
)
from src.evaluation.pooling import FUSED, paired
from src.evaluation.recording import WAVEFORM_METRICS
from src.outputs import FLOAT_FORMAT
from src.signal_transforms import signal_unit

TABLES, FIGURES = "tables", "figures"
MARKDOWN_NAME, HTML_NAME = "report.md", "report.html"
REPORT_DPI = 200
PRESSURE_UNIT = "mmHg"

DEPARTURES = (
    ("Segments matching the device output period (5.1.3)",
     "One measurement per recording, over its covered stretch"),
    ("Equal r for every subject (4.5.1 b)",
     "Unequal counts, with Formulas (5) and (9) to (12) in their general form"),
    ("N_ind from r (Formula 6)", "f_BA of Formula (10) in place of r"),
    ("One device reading per reference reading",
     "Each camera view is a reading against the same reference"),
    ("Method for stability (5.2)", "Not assessable on recordings of this length"),
    ("Method for blood pressure changes (5.3)",
     "Not assessable on recordings of this length"),
    ("Reference against time since re-initialization (5.1.3 f)", "Not applicable"),
    ("Scope", "The standard covers systolic, diastolic and mean arterial "
              "pressure only; every other parameter is outside it"),
)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<style>
@page { size: A4; margin: 20mm; }
body { font-family: system-ui, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1a1a; background: #ffffff; max-width: 170mm; margin: 0 auto;
       padding: 16px; }
h1 { font-size: 18pt; } h2 { font-size: 14pt; margin-top: 1.6em; }
h3 { font-size: 11.5pt; margin-top: 1.3em; }
h2, h3 { break-after: avoid; }
img { max-width: 100%; height: auto; display: block; margin: 0.6em 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 0.6em 0; }
th, td { border: 1px solid #c8c8c8; padding: 3px 5px; text-align: left;
         vertical-align: top; }
th { background: #efefef; }
table, img { break-inside: avoid; }
code { font-size: 9pt; overflow-wrap: anywhere; }
@media print { body { max-width: none; padding: 0; } }
</style>
</head>
<body>
{{body}}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def _cell(value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def pipe_table(frame: pd.DataFrame) -> list:
    """The lines of a Markdown pipe table."""
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "|".join(" --- " for _ in frame.columns) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return lines


class Report:
    """The document, and the files it links to, under one folder."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.lines = []
        for name in (TABLES, FIGURES):
            (self.folder / name).mkdir(parents=True, exist_ok=True)

    def heading(self, text: str, level: int = 2) -> None:
        self.lines += [f"{'#' * level} {text}", ""]

    def text(self, paragraph: str) -> None:
        self.lines += [paragraph, ""]

    def table(self, name: str, frame: pd.DataFrame) -> None:
        """Write ``tables/<name>.csv`` and print the table with a link to it."""
        path = f"{TABLES}/{name}.csv"
        frame.to_csv(self.folder / path, index=False, float_format=FLOAT_FORMAT)
        if frame.empty:
            self.text("Nothing to report.")
            return
        self.lines += [*pipe_table(frame), "", f"Data: [{path}]({path})", ""]

    def figure(self, name: str, figure: plt.Figure, caption: str) -> None:
        """Save ``figures/<name>.png``, close the figure and link it."""
        path = f"{FIGURES}/{name}.png"
        figure.savefig(self.folder / path, dpi=REPORT_DPI)
        plt.close(figure)
        self.lines += [f"![{caption}]({path})", "", f"*{caption}*", ""]

    def write(self, title: str) -> Path:
        markdown = "\n".join([f"# {title}", "", *self.lines])
        path = self.folder / MARKDOWN_NAME
        path.write_text(markdown, encoding="utf-8")
        (self.folder / HTML_NAME).write_text(
            to_html(markdown, self.folder, title), encoding="utf-8")
        return path


def to_html(markdown: str, folder: Path, title: str) -> str:
    """The document as one HTML file: figures embedded, A4 when printed."""
    body = MarkdownIt("commonmark").enable("table").render(markdown)

    def embed(match) -> str:
        data = base64.b64encode((folder / match.group(1)).read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(rf'src="({FIGURES}/[^"]+\.png)"', embed, body)
    return PAGE.replace("{{title}}", title).replace("{{body}}", body)


# ---------------------------------------------------------------------------
# The sections
# ---------------------------------------------------------------------------
def summary_section(report: Report, run_dir: Path, pooled: pd.DataFrame,
                    measurements: pd.DataFrame, excluded: pd.DataFrame, meta: dict,
                    bounds: tuple) -> None:
    report.heading("Summary")
    git = meta.get("git") or {}
    commit = f"{git.get('commit', '')}{' (dirty)' if git.get('dirty') else ''}"
    keys = ["fold", "dataset", "recording", "perspective"]
    rows = [
        ("Run directory", str(run_dir)),
        ("Folds", pooled["fold"].nunique()),
        ("Epoch evaluated", ", ".join(str(e) for e in sorted(pooled["epoch"].unique()))),
        ("Datasets tested", ", ".join(sorted(pooled["dataset"].unique()))),
        ("Subjects", measurements["subject"].nunique()),
        ("Paired measurements (recording and camera)",
         len(measurements.drop_duplicates(keys))),
        ("Excluded", len(excluded)),
        ("Eligible covered stretch", f"{bounds[0]:g} to {bounds[1]:g} s"),
        ("Traces", ", ".join(meta["traces"])),
        ("Channels", ", ".join(meta["channels"])),
        ("Rate and window", f"{meta['fs']:g} Hz, {meta['window_frames']} frames, "
                            f"stride {meta['stride_frames']}"),
        ("Git commit", commit),
        ("Command of the first fold", f"`{meta.get('command', '')}`"),
    ]
    report.table("summary", pd.DataFrame(rows, columns=["item", "value"]).astype(str))


def demographics_section(report: Report, pooled: pd.DataFrame, params: list,
                         additional: list) -> None:
    report.heading("Demographics")
    report.text("Subject characteristics are counted once per subject, recording "
                "characteristics and reference values once per recording. 'All' "
                "is every test recording; 'eligible' those inside the covered "
                "stretch bounds. A characteristic the run directory does not "
                f"carry reads '{demographics.NOT_RECORDED}'.")
    report.heading("Population", 3)
    report.table("characteristics", demographics.characteristics(pooled, additional))
    report.heading("Reference values", 3)
    report.table("reference_spread", demographics.reference_spread(pooled, params))
    report.heading("ISO 81060-3 population requirements", 3)
    report.text("Clauses 4.2, 4.3.2 and 4.3.3, over the eligible recordings. The "
                "standard sets no requirement on skin tone, neck circumference, "
                "heart rate or venous pressure.")
    report.table("population_checks", demographics.population_checks(pooled))
    report.table("bp_distribution", demographics.bp_distribution(pooled, params))

    kept = pooled[pooled["eligible"]]
    report.heading("Distributions", 3)
    for p in params:
        values = paired(kept, p).sort_values("perspective").drop_duplicates(
            demographics.RECORDING_KEY)["ref"]
        if len(values):
            report.figure(f"reference_{p.key}",
                          histogram_figure(values, f"{p.label} ({p.unit})",
                                           f"Reference {p.label}",
                                           demographics.band_edges(p)),
                          f"Reference {p.label} over the eligible recordings"
                          + (", with the band edges of ISO 81060-3 clause 4.3.3"
                             if p.criteria else ""))
    subjects = kept.drop_duplicates("subject")
    recordings = kept.drop_duplicates(demographics.RECORDING_KEY)
    counted = {demographics.CORE[name]: frame[name]
               for name, frame in (("posture", recordings), ("sex", subjects),
                                   ("skin_tone", subjects))
               if frame[name].notna().any()}
    if counted:
        report.figure("counts", counts_figure(counted, "Eligible test population"),
                      "Counts per posture (recordings), sex and skin tone (subjects)")
    ages = pd.to_numeric(subjects["age_years"], errors="coerce").dropna()
    if len(ages):
        report.figure("age", histogram_figure(ages, "age (years)", "Age", (50, 60, 70)),
                      "Age of the eligible subjects, with the thresholds of ISO "
                      "81060-3 clause 4.3.2.3.2")

    report.heading("Accuracy by group", 3)
    report.text(f"Mean error and SD within each group. Groups of fewer than "
                f"{demographics.SMALL_GROUP} measurements are marked small and "
                f"are too few to read.")
    report.table("stratified", demographics.stratified(kept, params))


def clinical_section(report: Report, measurements: pd.DataFrame, params: list) -> None:
    report.heading("Clinical accuracy")
    report.text("Error is prediction minus reference. s_corr is the standard "
                "deviation corrected for repeated measurements of a subject "
                "(ISO 81060-3 Formula 9); the limits of agreement are the mean "
                "error ± 1.96 s_corr.")
    found = []
    for p in params:
        pairs = paired(measurements, p)
        if len(pairs):
            found.append((p, pairs, repeated_measures(pairs["error"], pairs["subject"])))
    agreement, sample, cumulative, passes = [], [], [], []
    for p, pairs, stats in found:
        low, high = limits(stats)
        agreement.append({"parameter": p.label, "unit": p.unit,
                          "mean_error": stats["mean_error"], "s_corr": stats["s_corr"],
                          "loa_low": low, "loa_high": high, "sd": stats["sd"],
                          **spread(pairs["error"])})
        sample.append({"parameter": p.label, "n": stats["n"], "subjects": stats["k"],
                       "per_subject_min": stats["m_min"],
                       "per_subject_median": stats["m_median"],
                       "per_subject_max": stats["m_max"], "f_BA": stats["f_ba"],
                       "ICC": stats["icc"], "N_ind": stats["n_ind"]})
        if p.unit == PRESSURE_UNIT:
            cumulative.append({"parameter": p.label, **within_shares(pairs["error"])})
        if p.criteria:
            passes += [{"parameter": p.label, **row} for row in criteria(stats)]
    report.heading("Agreement", 3)
    report.table("agreement", pd.DataFrame(agreement))
    report.heading("Sample", 3)
    report.table("sample", pd.DataFrame(sample))
    report.heading("Cumulative accuracy", 3)
    report.text("Percent of absolute errors within 5, 10 and 15 mmHg.")
    report.table("cumulative_accuracy", pd.DataFrame(cumulative))
    report.heading("ISO 81060-3 accuracy criteria", 3)
    report.text(f"Clause 5.1.4, and the {ISO_SUBJECTS} subjects of clause 4.5.1. "
                f"The standard sets criteria for arterial pressure only. See "
                f"'Departures from ISO 81060-3' for how this evaluation differs "
                f"from the clinical investigation the standard describes.")
    report.table("criteria", pd.DataFrame(passes))

    for p in params:
        report.heading(p.label, 3)
        match = [(pairs, stats) for q, pairs, stats in found if q.key == p.key]
        if not match:
            report.text("No paired values: the reference or the prediction is "
                        "blank in every eligible measurement.")
            continue
        (pairs, stats), = match
        if stats["note"]:
            report.text(f"Note: {stats['note']}.")
        report.figure(f"bland_altman_{p.key}",
                      bland_altman_figure(pairs, stats, p.unit, f"{p.label}: Bland-Altman"),
                      f"{p.label}: error against the mean of reference and "
                      f"prediction, with the mean error and the limits of agreement")
        report.figure(f"agreement_{p.key}",
                      agreement_figure(pairs, p.unit, f"{p.label}: prediction against reference"),
                      f"{p.label}: prediction against reference, with the line of identity")


def _mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    sd = f" ± {values.std(ddof=1):.3g}" if len(values) > 1 else ""
    return f"{values.mean():.3g}{sd}"


def learning_section(report: Report, measurements: pd.DataFrame, params: list,
                     traces: list, curves: pd.DataFrame, losses: pd.DataFrame) -> None:
    report.heading("Deep-learning metrics")
    rows = measurements[measurements["signal"].isin(traces)]

    report.heading("Waveform fidelity", 3)
    report.text("Sample by sample over each measurement's covered stretch, as "
                "mean ± SD over the measurements. The concordance is at its best "
                "lag within half a second; the lag is positive when the "
                "prediction is delayed.")
    report.table("waveform", pd.DataFrame([
        {"trace": sig, "unit": signal_unit(sig), "n": len(group),
         **{metric: _mean_sd(group[f"waveform_{metric}"]) for metric in WAVEFORM_METRICS}}
        for sig, group in rows.groupby("signal", sort=False)]))
    if rows["waveform_ccc"].notna().any():
        report.figure("waveform_ccc",
                      box_figure(rows, "waveform_ccc", "concordance (CCC)",
                                 "Waveform concordance per trace"),
                      "Concordance of the predicted with the reference waveform, "
                      "one point per measurement")

    report.heading("Rate", 3)
    rates = []
    for p in params:
        if p.kind != "rate":
            continue
        pairs = paired(measurements, p)
        source = measurements[measurements["signal"] == p.signal]
        rates.append({"source": p.signal, "unit": p.unit, "n": len(pairs),
                      **rate_metrics(pairs["ref"], pairs["pred"]),
                      "snr_db": source["snr"].mean(), "macc": source["macc"].mean()})
    report.table("rate", pd.DataFrame(rates))

    report.heading("Beat detection", 3)
    report.text("Predicted beats matched to reference beats, summed over the "
                "measurements: recall is matched over reference, precision "
                "matched over predicted.")
    beats = []
    for sig, group in rows.dropna(subset=["n_ref_beats"]).groupby("signal", sort=False):
        ref, pred, matched = (group[c].sum() for c in ("n_ref_beats", "n_pred_beats",
                                                       "n_matched"))
        beats.append({"trace": sig, "reference_beats": int(ref), "predicted_beats": int(pred),
                      "matched": int(matched),
                      "precision": matched / pred if pred else np.nan,
                      "recall": matched / ref if ref else np.nan})
    report.table("beats", pd.DataFrame(beats))

    report.heading("Constant-predictor baseline", 3)
    report.text("The model beside a predictor that answers the mean reference "
                "for every measurement. That mean is of the test references, "
                "which favours the constant. The spread ratio is the SD of the "
                "predictions over the SD of the references: near 0 the model "
                "answers one level for everyone, near 1 it follows the range.")
    report.table("baseline", pd.DataFrame([
        {"parameter": p.label, "unit": p.unit, **baseline(pairs["ref"], pairs["pred"])}
        for p in params if p.kind == "level"
        for pairs in [paired(measurements, p)] if len(pairs)]))

    report.heading("Training behaviour", 3)
    totals = [c for c in losses.columns if c.endswith("/total")] if len(losses) else []
    if totals:
        long = losses.melt(id_vars=["fold", "epoch"], value_vars=totals,
                           var_name="signal", value_name="loss")
        long["signal"] = long["signal"].str.removesuffix("/total")
        report.figure("training_loss",
                      curves_figure(long, "loss", "training loss", "Training loss per trace"),
                      "Training loss per trace against the epoch; the band is the "
                      "spread over folds")
    tested = curves[curves["eligible"] & curves["signal"].isin(traces)]
    if tested["waveform_rmse"].notna().any():
        report.figure("test_rmse",
                      curves_figure(tested, "waveform_rmse", "waveform RMSE",
                                    "Test error per trace"),
                      "Waveform RMSE on the held-out recordings against the "
                      "epoch; the band is the spread over folds and recordings")
    report.text("The epoch evaluated is fixed before the test error is seen; "
                "these curves describe training and select nothing.")


def departures_section(report: Report) -> None:
    report.heading("Departures from ISO 81060-3")
    report.text("This evaluation borrows the accuracy statistics of ISO "
                "81060-3:2022. It is not the clinical investigation the standard "
                "describes, and differs from it as follows.")
    report.table("departures", pd.DataFrame(
        DEPARTURES, columns=["requirement", "this evaluation"]))


def exclusions_section(report: Report, excluded: pd.DataFrame) -> None:
    report.heading("Exclusions")
    if excluded.empty:
        report.text("No recording was excluded.")
    report.table("exclusions", excluded)


def build(out_dir, run_dir, pooled: pd.DataFrame, measurements: pd.DataFrame,
          excluded: pd.DataFrame, params: list, meta: dict, curves: pd.DataFrame,
          losses: pd.DataFrame, additional: list, bounds: tuple) -> Path:
    """Write the report under ``out_dir``; returns ``report.md``."""
    report = Report(out_dir)
    traces = [str(sig) for sig in meta["traces"]]
    summary_section(report, Path(run_dir), pooled, measurements, excluded, meta, bounds)
    demographics_section(report, pooled, params, additional)
    clinical_section(report, measurements, params)
    learning_section(report, measurements, params, traces, curves, losses)
    departures_section(report)
    exclusions_section(report, excluded)
    return report.write(f"Evaluation of {Path(run_dir).name}")
```

`FUSED` is imported for ruff only if used; remove the import if ruff reports it unused.

- [ ] **Step 3: Write `scripts/evaluate.py`**

```python
"""Evaluate a finished run: pool its per-recording tables into one report.

    uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba

The run directory is one ``scripts/run.py`` wrote, or one ``main.py`` wrote
a fold per participant into. Its records are already scored
(``docs/evaluation.md``); nothing is trained, inferred or re-scored here,
and nothing but the run directory is read. One recording and camera is one
paired measurement, eligible when its covered stretch is within
``--segment-min`` and ``--segment-max`` seconds. The last epoch of each fold
is evaluated, or the one ``--epoch`` names; it is never chosen by test error.

Written to ``RUN_DIR/evaluation/``, replacing what was there::

    measurements.csv   one row per fold, recording, camera and signal
    exclusions.csv     what was left out, and why
    tables/*.csv       every table of the report
    figures/*.png      every figure
    report.md          the report
    report.html        the same, as one file, A4 when printed
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.demographics import with_core                  # noqa: E402
from src.evaluation.pooling import (                               # noqa: E402
    additional_params, collect, collect_epochs, exclusions, mark_eligible,
    parameters, read_losses,
)
from src.evaluation.report import build                            # noqa: E402
from src.outputs import FLOAT_FORMAT                               # noqa: E402

OUT_DIR = "evaluation"
MEASUREMENTS_NAME, EXCLUSIONS_NAME = "measurements.csv", "exclusions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pool a finished run's per-recording tables into one report.")
    parser.add_argument(
        "run_dir", metavar="RUN_DIR",
        help="a run directory written by scripts/run.py, or the directory "
             "main.py wrote its folds into")
    parser.add_argument(
        "--epoch", type=int, metavar="N",
        help="which epoch of every fold to evaluate (default: each fold's last)")
    parser.add_argument(
        "--segment-min", type=float, default=20.0, metavar="SECONDS",
        help="shortest covered stretch that counts as a measurement (default: 20)")
    parser.add_argument(
        "--segment-max", type=float, default=30.0, metavar="SECONDS",
        help="longest covered stretch that counts as a measurement (default: 30)")
    parser.add_argument(
        "--additional-params", nargs="+", default=[], metavar="ATTR",
        help="root attrs to report beside the core five and the datasets' "
             "ADDITIONAL_PARAMS, as recordings.csv names them (dotted when nested)")
    return parser


def main(argv=None) -> Path:
    """Write the evaluation; returns ``report.md``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.segment_min > args.segment_max:
        parser.error("--segment-min is above --segment-max")
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        parser.error(f"{run_dir} is not a directory")
    bounds = (args.segment_min, args.segment_max)
    try:
        pooled, meta = collect(run_dir, args.epoch)
        pooled = mark_eligible(with_core(pooled), *bounds)
        curves = mark_eligible(collect_epochs(run_dir), *bounds)
        losses = read_losses(run_dir)
        additional = [*additional_params(run_dir),
                      *(a for a in args.additional_params)]
    except (ValueError, FileNotFoundError) as err:
        parser.error(str(err))
    additional = list(dict.fromkeys(additional))

    out_dir = run_dir / OUT_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    excluded = exclusions(pooled)
    excluded.to_csv(out_dir / EXCLUSIONS_NAME, index=False, float_format=FLOAT_FORMAT)
    measurements = pooled[pooled["eligible"]].drop(columns=["eligible", "reason"])
    if measurements.empty:
        parser.error(f"no recording has a covered stretch within {bounds[0]:g} to "
                     f"{bounds[1]:g} s; {out_dir / EXCLUSIONS_NAME} lists them")
    measurements.to_csv(out_dir / MEASUREMENTS_NAME, index=False, float_format=FLOAT_FORMAT)

    params = parameters(meta["traces"], measurements)
    path = build(out_dir, run_dir, pooled, measurements, excluded, params, meta,
                 curves, losses, additional, bounds)
    print(f"{len(measurements.drop_duplicates(['fold', 'dataset', 'recording', 'perspective']))} "
          f"measurement(s) of {measurements['subject'].nunique()} subject(s), "
          f"{len(excluded)} excluded")
    print(f"report: {path}")
    return path


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Lint**

Run: `uv run ruff check src/evaluation/report.py scripts/evaluate.py`
Expected: `All checks passed!`

- [ ] **Step 5: Add the command to `README.md`**

After the "Leave one out folds (Experiments)" section and before "Train on dataset(s) A, B..., test on dataset X":

````markdown
### Evaluating a finished run

Once the folds have trained and scored their held-out participants, pool them into one report:

```bash
# the last epoch of every fold; recordings covered for 20 to 30 s count
uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba

# or a specific epoch, other bounds, and further attributes to describe
uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba \
--epoch 8 \
--segment-min 20 --segment-max 30 \
--additional-params abp_site synthetic_neck.traces.heart_rate_bpm
```

The report lands in `runs/synthetic_benchmark_physmamba/evaluation/` as `report.md` and `report.html`, with every table and figure beside it. It covers who was tested, the clinical accuracy of each parameter (ISO 81060-3 statistics, Bland-Altman plots) and the usual deep-learning metrics; `docs/evaluation.md` lists what each table holds.
````

- [ ] **Step 6: Document the pooled layer in `docs/evaluation.md`**

Insert before `## Not covered`:

```markdown
## Per run

`scripts/evaluate.py RUN_DIR` pools the tables above over every fold of a
run into `RUN_DIR/evaluation/`. It is run by hand after the run has
finished, reads the run directory and nothing else, and re-scores nothing.

**The paired measurement** (`src/evaluation/pooling.py`). One recording and
camera, over its whole covered stretch. It is eligible when that stretch is
within `--segment-min` and `--segment-max` seconds (20 and 30, both
inclusive); the rest are listed in `exclusions.csv` and enter no statistic.
The subject is the dataset and participant together. The epoch is each
fold's last, or `--epoch`; it is never chosen by test error.

`measurements.csv`, one row per fold, recording, camera and signal: `fold`,
`dataset`, `participant`, `recording`, `perspective`, `epoch`, `subject`,
`signal`, `duration`, the columns of `signals.csv`, the rate columns of
`rates.csv` for that source (`FUSED` has a row of its own), then the
columns of `recordings.csv` and the five core attributes.

**Demographics** (`src/evaluation/demographics.py`). `recordings.csv`,
written beside `meta.json` by `src/outputs.py`, carries each recording's
root attrs. Five are core: `age_years`, `sex`, `skin_tone`, `posture`,
`neck_circumference_cm`; several measurements of one are reported as their
median. Further attrs come from a dataset's `ADDITIONAL_PARAMS` and from
`--additional-params`. The sex and age shares are checked against ISO
81060-3 clauses 4.3.2.2 and 4.3.2.3.2 (p. 6), the reference pressures
against the bands of clause 4.3.3 (pp. 7-8). An attr the run does not carry
reads "not recorded".

**Accuracy** (`src/evaluation/agreement.py`). Per parameter (a rate per
source; systolic, diastolic and mean ABP; mean CVP): the mean error, the
corrected standard deviation, the intra-class correlation and the number of
independent measurements of ISO 81060-3 Formulas (5), (6) and (8) to (12)
(pp. 12 and 15), in the general form the standard prints for unequal counts
per subject, with `f_BA` standing for `r` in Formula (6). Limits of
agreement are the mean error ± 1.96 `s_corr`. For pressures, the percent of
absolute errors within 5, 10 and 15 mmHg. The criteria of clause 5.1.4
(p. 16) are applied to arterial pressure only.

**Deep-learning metrics.** Waveform MAD, RMSE, r, CCC and lag as mean ± SD
over measurements; rate MAE, RMSE, MAPE and r per source; beat precision
and recall; the model beside a constant predictor, and the ratio of the
predictions' SD to the references'; training loss and test error against
the epoch.

The report's "Departures from ISO 81060-3" section lists where this differs
from the clinical investigation the standard describes.
```

In `## Not covered`, replace the sentence beginning "Pooling recordings into per-participant or per-dataset summaries" with:

```markdown
The methods for stability and for blood pressure changes of ISO 81060-3
(clauses 5.2 and 5.3), which need hours and half-hours of recording. The
criteria of IEEE 1708. Comparison across runs. A respiratory rate.
```

- [ ] **Step 7: Update `src/evaluation/__init__.py`**

Append to the module docstring:

```
Over a whole run: ``pooling.py`` stacks those tables into one of paired
measurements, ``agreement.py`` and ``demographics.py`` compute from it, and
``report.py`` writes the tables, the figures (``plots.py``) and the report.
``scripts/evaluate.py`` runs them, by hand, after the run has finished.
```

- [ ] **Step 8: Checkpoint: hand the user the verification command**

```bash
uv run scripts/evaluate.py runs/synthetic_benchmark_physmamba
```

Expected output: `4 measurement(s) of 4 subject(s), 0 excluded` and the path of `report.md`.

Expected in `runs/synthetic_benchmark_physmamba/evaluation/`:
- `measurements.csv` with 16 rows, `exclusions.csv` with none
- `report.md` and `report.html`, six sections each
- Demographics read "not recorded" (this run predates `recordings.csv`)
- Every clinical parameter carries the note "one measurement per subject" (Review Focus 2)
- ABP criteria: `N_ind` and subjects are not met (4 of each)
- `report.html` opens with its figures when moved out of the folder on its own, and prints to A4 without clipped tables

Review Focus 4: run with `--segment-min 31 --segment-max 40`. Expected: an error naming `exclusions.csv`, which lists all four recordings as shorter than 31 s.

To see the demographics filled, the test records of each fold must be regenerated with Task 1 in place. `scripts/test.py` refuses to overwrite, so per fold: delete `epoch_10/test_records/synthetic_neck/`, then run

```bash
uv run scripts/test.py --checkpoint runs/synthetic_benchmark_physmamba/<fold> \
--test-dataset synthetic_neck --test-participant-id <the fold's participant>
```

This deletes generated files only; the same command recreates them. Leave it to the user to decide.

---

## Part C: core attribute standard (separate plan, one decision pending)

The spec's piece 1 standardises five root attributes in the caches. Reading the cachers showed the ground differs from what the spec assumed:

| Attribute | Neckflix cache today | Synthetic cache today |
|---|---|---|
| `age_years` | Present, integer | Absent |
| `sex` | Present, `M` / `F` | Absent |
| `posture` | Present | Present |
| `skin_tone` | Present **as a group**: `scale`, `self`, `recorder`, `clinician` | Absent; `monk_tone` instead |
| `neck_circumference_cm` | Present **as a group**: `lower`, `mid`, `upper` | Absent; drawn by the generator, stored only as a radius in its scene settings |

So for Neckflix the two names are already taken by the groups of measurements. One attribute cannot be both a number and a group.

Part B already handles this without touching any cache: `demographics.attribute` reads the column of that name, and failing that the median of the numeric columns under it. Neckflix therefore reports correctly as it is.

**Decision needed before Part C is planned:**

1. **The standard allows either form (recommended).** A core attribute is a single value, or a group of measurements whose median is the value. Neckflix needs no change and no store is patched. Only the synthetic generator changes: it writes `skin_tone` (its `monk_tone`) and `neck_circumference_cm` (the value it already draws). `docs/cache-contract.md` states the rule.
2. **The standard requires a single value.** The Neckflix cacher writes the medians as `skin_tone` and `neck_circumference_cm` and moves the groups to new names (`skin_tone_ratings`, `neck_circumference_sites_cm`); every existing store is patched; filters that name `skin_tone.clinician` are updated.

Both cachers are git submodules with their own repositories (`dataset/cachers/neckflix`, `tools/synthetic_neck`), so Part C is committed there, not here.
