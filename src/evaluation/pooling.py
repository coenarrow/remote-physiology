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
    """``{epoch: folder}`` for the epochs of a fold that carry scored test
    records."""
    found = {}
    for path in fold.glob("epoch_*"):
        match = EPOCH_PATTERN.fullmatch(path.name)
        if match and (path / RECORDS_DIR).is_dir() and records_dirs(path):
            found[int(match.group(1))] = path
    return found


def epoch_dir(fold: Path, epoch: int | None) -> tuple:
    """``(epoch, folder)``: the epoch asked for, or the fold's last."""
    found = epochs_of(fold)
    if not found:
        raise ValueError(f"{fold} has no epoch with scored test records")
    if epoch is None:
        epoch = max(found)
    if epoch not in found:
        raise ValueError(f"{fold} has no test records for epoch {epoch}; it "
                         f"has {sorted(found)}")
    return epoch, found[epoch]


def records_dirs(epoch_folder: Path) -> list:
    """One dataset's records directory per entry, found by its ``meta.json``;
    one that is written but not yet scored is not there yet."""
    return sorted(path.parent for path in (epoch_folder / RECORDS_DIR).glob(f"*/{META_NAME}")
                  if any(path.parent.glob(f"*/*/{SIGNALS_NAME}")))


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
    table = signals.merge(rates, on="signal", how="outer", sort=False)
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
    pooled.insert(len(IDENTITY) + 1, "subject",
                  pooled["dataset"] + "/" + pooled["participant"])
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
        datasets = (load_yaml(str(fold / CONFIG_NAME)) or {}).get("datasets") or {}
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
    key: str          # file-safe name: rate_ABP, ABP_max
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
