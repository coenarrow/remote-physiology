"""Who each signal was measured on: the characteristics of a trace's
population, read off ``recordings.csv``.

Subject-level characteristics are counted once per participant,
recording-level ones once per recording, whatever the number of cameras.
Where a cache holds several measurements of one attribute
(``skin_tone.self``, ``skin_tone.clinician``, ...) the value is their median.
Anything absent reads "not recorded".
"""

import numpy as np
import pandas as pd

from src.signal_transforms import is_cardiac

#: The root attrs every cache names the same way.
CORE = ("age_years", "sex", "skin_tone", "posture", "neck_circumference_cm")
#: Reported as counts per value whatever their dtype.
CATEGORICAL = ("sex", "skin_tone", "posture", "arrhythmia")
NOT_RECORDED = "not recorded"
#: The characteristics of each trace's population, by level.
SUBJECT_CHARACTERISTICS = {
    "sex": "Sex", "age_years": "Age (years)", "skin_tone": "Skin tone (Monk)",
    "neck_circumference_cm": "Neck circumference (cm)", "bmi": "BMI (kg/m²)",
    "arrhythmia": "Arrhythmia",
}
RECORDING_CHARACTERISTICS = {"posture": "Posture"}
#: Where the reference line of a trace sits, reported for that trace only.
LINE_SITE = {"ABP": ("arterial_line.site", "Arterial line site"),
             "CVP": ("central_line.site", "Central line site")}
#: The name of the rate's population, every cardiac trace's together.
RATE_POPULATION = "rate"
#: The characteristics every measurement row carries, so measurements can
#: be compared across them.
MARKERS = ("sex", "age_years", "skin_tone", "posture")


def attribute(frame: pd.DataFrame, name: str) -> pd.Series:
    """Row by row, the column of that name; where it is blank the median of
    the numeric columns under it (``name.*``); failing both, blank."""
    values = (frame[name].astype(object) if name in frame.columns
              else pd.Series(np.nan, index=frame.index, dtype=object))
    members = [c for c in frame.columns if c.startswith(f"{name}.")]
    if members:
        numeric = frame[members].apply(pd.to_numeric, errors="coerce")
        median = numeric.median(axis=1, skipna=True).astype(object)
        values = values.where(values.notna(), median)
    return values.where(values.notna(), np.nan)


def with_core(attrs: pd.DataFrame) -> pd.DataFrame:
    """``attrs`` with every ``CORE`` attribute as a column."""
    attrs = attrs.copy()
    for name in CORE:
        attrs[name] = attribute(attrs, name)
    return attrs


def with_markers(frame: pd.DataFrame, attrs: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the ``MARKERS`` of each row's recording appended,
    matched on participant and recording; blank where ``attrs`` has no row
    for it."""
    columns = ["participant", "recording", *MARKERS]
    return frame.merge(attrs[columns], on=["participant", "recording"], how="left")


def _word(value) -> str:
    return f"{value:g}" if isinstance(value, (float, np.floating)) else str(value)


def describe(values: pd.Series, categorical: bool) -> tuple:
    """``(n recorded, summary)``: counts per value, or mean ± SD and range."""
    values = values.dropna()
    if values.empty:
        return 0, NOT_RECORDED
    numeric = pd.to_numeric(values, errors="coerce")
    flags = values.map(lambda v: isinstance(v, (bool, np.bool_))).all()
    if categorical or flags or numeric.isna().any():
        counts = values.map(_word).value_counts().sort_index()
        parts = [f"{value}: {count} ({count / len(values) * 100:.0f} %)"
                 for value, count in counts.items()]
        return len(values), ", ".join(parts)
    sd = f" ± {numeric.std(ddof=1):.3g}" if len(numeric) > 1 else ""
    return len(numeric), (f"{numeric.mean():.3g}{sd} "
                          f"(range {numeric.min():.3g} to {numeric.max():.3g})")


def population_of(measurements: pd.DataFrame, signal: str | None) -> pd.DataFrame:
    """The ``participant`` / ``recording`` pairs with a measurement of
    ``signal``, or of any cardiac trace when it is ``None`` (the rate's)."""
    rows = (measurements["signal"] == signal if signal is not None
            else measurements["signal"].map(is_cardiac))
    return measurements.loc[rows, ["participant", "recording"]].drop_duplicates()


def demographics(measurements: pd.DataFrame, attrs: pd.DataFrame) -> dict:
    """``{signal: table}``: the characteristics of the population each
    signal was measured on, and of the rate's, one row per characteristic:
    its level, how many had it recorded, and counts per value or
    mean ± SD and range (:func:`describe`)."""
    tables = {}
    signals = [*sorted(measurements["signal"].unique()), None]
    for signal in signals:
        recordings = population_of(measurements, signal).merge(
            attrs, on=["participant", "recording"], how="left")
        subjects = recordings.drop_duplicates("participant")
        rows = []
        for name, label in SUBJECT_CHARACTERISTICS.items():
            n, text = describe(attribute(subjects, name), name in CATEGORICAL)
            rows.append({"level": "participant", "characteristic": label,
                         "n_recorded": n, "value": text})
        extra = {**RECORDING_CHARACTERISTICS, **dict([LINE_SITE[signal]]
                                                     if signal in LINE_SITE else [])}
        for name, label in extra.items():
            n, text = describe(attribute(recordings, name), True)
            rows.append({"level": "recording", "characteristic": label,
                         "n_recorded": n, "value": text})
        table = pd.DataFrame(rows)
        table.insert(0, "n_recordings", len(recordings))
        table.insert(0, "n_participants", len(subjects))
        tables[signal if signal is not None else RATE_POPULATION] = table
    return tables
