"""The derived parameters against clinical thresholds: does the prediction
call the same measurements elevated (or low) as the reference?

Each :data:`CONDITIONS` entry names a parameter of
``src.evaluation.measurements.PARAMETERS``, a threshold and a side.
Applied to the reference it gives each measurement's true class, applied
to the prediction its predicted class, and the two give a confusion matrix
and the usual rates. The unit is the measurement, as everywhere else: a
participant's state differs between postures, so a participant is not one
case. Beside the measurement counts the table says how many recordings and
participants are positive in the reference, so the number of distinct
states behind the counts is visible.

:func:`sweep` is the ROC's data, not its figure: the confusion counts at
every distinct predicted value with the reference fixed, the clinical
threshold's row flagged, and ``±inf`` rows so the curve runs from (0, 0)
to (1, 1). The AUC in the main table is the trapezoid over that sweep.
Nothing is drawn here.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.evaluation.measurements import PARAMETERS, pairs_of

#: The thresholds, in the parameter's unit.
SYSTOLIC_THRESHOLD_MMHG = 130.0
DIASTOLIC_THRESHOLD_MMHG = 80.0
CVP_MEAN_THRESHOLD_MMHG = 8.0
TACHYCARDIA_BPM = 100.0
BRADYCARDIA_BPM = 60.0
_NAN = float("nan")


class Condition(NamedTuple):
    """One clinical condition: its label, the ``PARAMETERS`` entry it reads,
    the threshold, and ``above`` (positive at or above it) or not (positive
    strictly below it)."""
    name: str
    parameter: str
    threshold: float
    above: bool


CONDITIONS = (
    Condition("elevated systolic", "ABP systolic", SYSTOLIC_THRESHOLD_MMHG, True),
    Condition("elevated diastolic", "ABP diastolic", DIASTOLIC_THRESHOLD_MMHG, True),
    Condition("elevated CVP", "CVP mean", CVP_MEAN_THRESHOLD_MMHG, True),
    Condition("tachycardia", "heart rate", TACHYCARDIA_BPM, True),
    Condition("bradycardia", "heart rate", BRADYCARDIA_BPM, False),
)


def positive(values, threshold: float, above: bool) -> np.ndarray:
    """Which values the condition calls positive."""
    values = np.asarray(values, dtype=np.float64)
    return values >= threshold if above else values < threshold


def confusion(ref_positive: np.ndarray, pred_positive: np.ndarray) -> dict:
    """The four counts and the rates read off them; a rate whose denominator
    is zero is NaN."""
    tp = int((ref_positive & pred_positive).sum())
    fp = int((~ref_positive & pred_positive).sum())
    tn = int((~ref_positive & ~pred_positive).sum())
    fn = int((ref_positive & ~pred_positive).sum())

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else _NAN

    sensitivity, specificity = rate(tp, tp + fn), rate(tn, tn + fp)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sensitivity, "specificity": specificity,
            "accuracy": rate(tp + tn, tp + fp + tn + fn),
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "ppv": rate(tp, tp + fp), "npv": rate(tn, tn + fn)}


def sweep(ref_positive: np.ndarray, pred: np.ndarray, condition: Condition) -> pd.DataFrame:
    """One row per candidate threshold on the prediction (its distinct
    values, the clinical threshold, and ``±inf``): the confusion counts,
    sensitivity and specificity of calling positive every measurement on
    the condition's side of it, with ``clinical`` marking the row at the
    condition's own threshold."""
    thresholds = np.union1d(np.unique(pred), [condition.threshold, -np.inf, np.inf])
    rows = []
    for threshold in thresholds:
        counts = confusion(ref_positive, positive(pred, threshold, condition.above))
        rows.append({"condition": condition.name, "threshold": float(threshold),
                     "clinical": bool(threshold == condition.threshold),
                     **{k: counts[k] for k in ("tp", "fp", "tn", "fn",
                                               "sensitivity", "specificity")}})
    return pd.DataFrame(rows)


def auc(swept: pd.DataFrame) -> float:
    """The area under the ROC the sweep traces: sensitivity against
    1 - specificity, by the trapezoid rule; NaN when either is undefined
    (a class absent from the reference)."""
    points = swept[["sensitivity", "specificity"]].to_numpy(dtype=np.float64)
    if not np.isfinite(points).all():
        return _NAN
    fpr, tpr = 1 - points[:, 1], points[:, 0]
    order = np.lexsort((tpr, fpr))
    fpr, tpr = fpr[order], tpr[order]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def classify(pairs: pd.DataFrame, condition: Condition) -> tuple:
    """``(row, sweep)``: the condition's row of the classification table,
    and its ROC sweep."""
    ref_positive = positive(pairs["ref"], condition.threshold, condition.above)
    pred_positive = positive(pairs["pred"], condition.threshold, condition.above)
    swept = sweep(ref_positive, pairs["pred"].to_numpy(dtype=np.float64), condition)
    n = len(pairs)
    note = ("no paired measurements" if n == 0
            else "no positive reference measurement" if not ref_positive.any()
            else "no negative reference measurement" if ref_positive.all() else "")
    row = {"condition": condition.name, "parameter": condition.parameter,
           "threshold": condition.threshold,
           "side": "at or above" if condition.above else "below",
           "n": n, "n_positive": int(ref_positive.sum()),
           "n_positive_recordings": int(pairs.loc[ref_positive, "recording"].nunique()),
           "n_positive_subjects": int(pairs.loc[ref_positive, "subject"].nunique()),
           "n_recordings": int(pairs["recording"].nunique()),
           "n_subjects": int(pairs["subject"].nunique()),
           "prevalence": float(ref_positive.mean()) if n else _NAN,
           **confusion(ref_positive, pred_positive),
           "auc": auc(swept),
           "n_fallback": int(pairs["fallback"].sum()),
           "note": note}
    return row, swept


def classification(frames: dict) -> tuple:
    """``(table, sweeps)``: one row per :data:`CONDITIONS` entry over the
    measurements of its parameter that have an error, and every condition's
    ROC sweep stacked, from the scored parameter frames."""
    parameters = {p.name: p for p in PARAMETERS}
    rows, sweeps = [], []
    for condition in CONDITIONS:
        pairs = pairs_of(frames[condition.parameter], parameters[condition.parameter])
        row, swept = classify(pairs, condition)
        rows.append(row)
        sweeps.append(swept)
    return pd.DataFrame(rows), pd.concat(sweeps, ignore_index=True)
