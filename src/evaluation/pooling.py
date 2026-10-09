"""A run's waveforms as one table of paired measurements.

A run directory holds one fold per held-out participant (or is itself one
fold). Under the chosen epoch of each, every
``test_records/<dataset>/`` directory with a ``meta.json`` and at least one
recording written is a records directory; each ``<recording>/<camera>/
<TRACE>.csv`` under it is one trace's label and mean prediction per frame.
:func:`collect_waveforms` stacks them all; :func:`segment` cuts each trace
into paired measurements of a fixed length.
"""

import json
import re
from pathlib import Path

import pandas as pd

from src.evaluation import progress
from src.evaluation.demographics import with_core
from src.outputs import META_NAME, RECORDINGS_NAME, RECORDS_DIR, WINDOWS_NAME

CONFIG_NAME = "config.yaml"
EPOCH_PATTERN = re.compile(r"epoch_(\d+)")
WAVEFORM_COLUMNS = ["frame", "t", "label", "mean"]
#: What names one trace, and what names one measurement of it.
TRACE_KEY = ["participant", "recording", "perspective", "signal"]
MEASUREMENT_KEY = ["participant", "recording", "perspective", "segment"]


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
    """``{epoch: folder}`` for the epochs of a fold that carry test
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
        raise ValueError(f"{fold} has no epoch with test records")
    if epoch is None:
        epoch = max(found)
    if epoch not in found:
        raise ValueError(f"{fold} has no test records for epoch {epoch}; it "
                         f"has {sorted(found)}")
    return epoch, found[epoch]


def records_dirs(epoch_folder: Path) -> list:
    """One dataset's records directory per entry, found by its ``meta.json``
    and at least one trace table under it."""
    return sorted(path.parent for path in (epoch_folder / RECORDS_DIR).glob(f"*/{META_NAME}")
                  if any(path.parent.glob("*/*/*.csv")))


# ---------------------------------------------------------------------------
# The waveforms, and the measurements cut from them
# ---------------------------------------------------------------------------
def collect_waveforms(run_dir, epoch: int | None = None) -> tuple:
    """``(waveforms, attrs, meta)``: every trace of every recording and
    camera at the chosen epoch of every fold, one row per frame, with the
    reference (``label``) and the mean prediction over the windows that
    covered the frame; one row per recording of ``recordings.csv`` with the
    core attributes as columns (``src.evaluation.demographics.with_core``);
    and the ``meta.json`` of the first records directory."""
    frames, attrs, first, tables = [], [], None, []
    for fold in find_folds(run_dir):
        number, folder = epoch_dir(fold, epoch)
        for records in records_dirs(folder):
            meta = json.loads((records / META_NAME).read_text(encoding="utf-8"))
            first = first or meta
            if (records / RECORDINGS_NAME).is_file():
                attrs.append(pd.read_csv(records / RECORDINGS_NAME,
                                         dtype={"recording": str, "participant": str}))
            windows = pd.read_csv(records / WINDOWS_NAME, dtype=str)
            owner = windows.drop_duplicates("recording").set_index("recording")["participant"]
            for trace in meta["traces"]:
                for path in sorted(records.glob(f"*/*/{trace}.csv")):
                    tables.append((path, number, trace, owner))
    if not tables:
        raise ValueError(f"{run_dir} holds no waveform")
    for path, number, trace, owner in progress(
            tables, "reading trace tables (one CSV per recording, camera and signal)"):
        recording, perspective = path.parent.parent.name, path.parent.name
        table = pd.read_csv(path, usecols=WAVEFORM_COLUMNS)
        tags = {"participant": str(owner.get(recording, "")),
                "recording": recording, "perspective": perspective,
                "epoch": number, "signal": trace}
        for position, (name, value) in enumerate(tags.items()):
            table.insert(position, name, value)
        frames.append(table)
    attrs = (pd.concat(attrs, ignore_index=True).drop_duplicates("recording")
             if attrs else pd.DataFrame(columns=["recording", "participant"]))
    return pd.concat(frames, ignore_index=True), with_core(attrs), first


def segment(waveforms: pd.DataFrame, fs: float, duration: float,
            tolerance: float) -> pd.DataFrame:
    """The paired measurements: each trace's frames that carry both a
    reference and a prediction, cut at fixed ``duration``-second boundaries
    from its first such frame into ``segment`` 0, 1, ...; a segment whose
    length is not within ``tolerance`` seconds of ``duration`` is dropped."""
    paired = waveforms.dropna(subset=["label", "mean"]).copy()
    offset = paired["frame"] - paired.groupby(TRACE_KEY)["frame"].transform("min")
    paired["segment"] = offset // round(duration * fs)
    key = [*TRACE_KEY, "segment"]
    length = paired.groupby(key)["frame"].transform("size") / fs
    return paired[(length - duration).abs() <= tolerance].reset_index(drop=True)
