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
    """Row by row, the column of that name; where it is blank the median of
    the numeric columns under it (``name.*``); failing both, blank. Datasets
    pooled into one table may each carry it in either form."""
    values = (frame[name].astype(object) if name in frame.columns
              else pd.Series(np.nan, index=frame.index, dtype=object))
    members = [c for c in frame.columns if c.startswith(f"{name}.")]
    if members:
        numeric = frame[members].apply(pd.to_numeric, errors="coerce")
        median = numeric.median(axis=1, skipna=True).astype(object)
        values = values.where(values.notna(), median)
    return values.where(values.notna(), np.nan)


def with_core(pooled: pd.DataFrame) -> pd.DataFrame:
    """The pooled table with the five core attributes as columns."""
    pooled = pooled.copy()
    for name in CORE:
        pooled[name] = attribute(pooled, name)
    return pooled


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


def references(frame: pd.DataFrame, parameter) -> pd.Series:
    """One reference value per recording, the first camera's that has one,
    whether or not there is a prediction beside it."""
    rows = frame[frame["signal"] == parameter.signal]
    rows = rows.assign(ref=pd.to_numeric(rows[parameter.ref], errors="coerce"))
    rows = rows.dropna(subset=["ref"]).sort_values("perspective")
    return rows.drop_duplicates(RECORDING_KEY)["ref"]


def reference_spread(pooled: pd.DataFrame, params: list) -> pd.DataFrame:
    kept = pooled[pooled["eligible"]]
    return pd.DataFrame([
        _row("reference", f"{p.label} ({p.unit})", references(pooled, p),
             references(kept, p), False) for p in params])


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
    rows = []
    for name, clause, column, test, required in POPULATION:
        values = (pd.to_numeric(subjects[column], errors="coerce")
                  if column == "age_years" else subjects[column])
        rows.append({"requirement": name, "clause": clause,
                     **_check(values, test, required)})
    postures = kept.drop_duplicates(RECORDING_KEY)["posture"].dropna()
    met = NOT_RECORDED if postures.empty else ("yes" if postures.nunique() > 1 else "no")
    rows.append({"requirement": "different postures evaluated", "clause": "4.2 b)",
                 "n": len(postures), "share_percent": np.nan,
                 "required_percent": np.nan, "met": met})
    return pd.DataFrame(rows)


def bp_distribution(pooled: pd.DataFrame, params: list) -> pd.DataFrame:
    """The reference readings of each ISO parameter against the bands of
    clause 4.3.3, over the eligible recordings."""
    kept = pooled[pooled["eligible"]]
    rows = []
    for p in params:
        if not p.criteria:
            continue
        values = references(kept, p)
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
        source = measurements[measurements["signal"] == p.signal].dropna(
            subset=[p.ref, p.pred])
        error = source[p.pred] - source[p.ref]
        for name in STRATA:
            for group, chosen in source.groupby(source[name].map(_word, na_action="ignore")):
                stats = repeated_measures(error[chosen.index], chosen["subject"])
                rows.append({"parameter": p.label, "characteristic": CORE[name],
                             "group": group, "n": stats["n"], "subjects": stats["k"],
                             "mean_error": stats["mean_error"], "sd": stats["sd"],
                             "small": "yes" if stats["n"] < SMALL_GROUP else ""})
    return pd.DataFrame(rows, columns=["parameter", "characteristic", "group", "n",
                                       "subjects", "mean_error", "sd", "small"])
