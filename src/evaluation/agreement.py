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
prediction minus reference. :func:`repeated_measures` and :func:`limits`
take arrays and return numbers; :func:`agreement` and
:func:`waveform_agreement` run them over the scored frames into one table.

Beside the ISO statistics, which see only the error, :func:`tracking` says
whether the prediction follows the reference's level at all. A model that
predicts the population average scores a fair mean error and SD on a narrow
test set while carrying no information about the subject. These columns are
descriptive, not ISO statistics:

- ``ref_sd`` and ``pred_sd``: the spread of each side across measurements. A
  prediction far narrower than its reference is shrinking towards a mean.
- ``r_between`` and ``slope_between``: Pearson's r and the least-squares
  slope of the per-subject mean prediction on the per-subject mean
  reference, one point per subject. They need three subjects.
- ``r_within`` and ``slope_within``: the same on values centred within each
  subject, which is how each subject's own changes are followed.

A slope of 1 tracks the level, 0 is a constant predictor, and in between
the prediction shrinks towards the mean. ``tracking_note`` says why a value
is not estimable.
"""

import numpy as np
import pandas as pd

from src.evaluation.measurements import PARAMETERS, WAVEFORM_METRICS, pairs_of, pearson

#: Limits of agreement are the mean error plus and minus this many s_corr.
LIMIT_Z = 1.96
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
    n_ind = k * (1 + (1 - icc) * (f_ba - 1)) if np.isfinite(icc) else n
    out.update({"f_ba": float(f_ba), "ms_between": ms_between, "ms_within": ms_within,
                "s_corr": float(np.sqrt(total)), "icc": float(icc),
                "n_ind": float(n_ind)})
    return out


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """The least-squares slope of ``y`` on ``x``, both already centred."""
    denominator = float((x ** 2).sum())
    return float((x * y).sum() / denominator) if denominator > 0 else _NAN


def tracking(ref, pred, subject) -> dict:
    """The level-tracking columns of the module docstring."""
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ids, index = np.unique(np.asarray(subject, dtype=str), return_inverse=True)
    n, k = ref.size, ids.size
    out = {"ref_sd": float(ref.std(ddof=1)) if n > 1 else _NAN,
           "pred_sd": float(pred.std(ddof=1)) if n > 1 else _NAN,
           "r_between": _NAN, "slope_between": _NAN,
           "r_within": _NAN, "slope_within": _NAN}
    notes = []
    if n == 0:
        out["tracking_note"] = "no paired measurements"
        return out
    counts = np.bincount(index, minlength=k).astype(np.float64)
    ref_mean = np.bincount(index, weights=ref, minlength=k) / counts
    pred_mean = np.bincount(index, weights=pred, minlength=k) / counts
    if k < 3:
        notes.append("fewer than three subjects: no between-subject tracking")
    else:
        out["r_between"] = pearson(ref_mean, pred_mean)
        out["slope_between"] = _slope(ref_mean - ref_mean.mean(), pred_mean - pred_mean.mean())
        if not np.isfinite(out["slope_between"]):
            notes.append("the subjects' mean references do not vary")
    if n == k:
        notes.append("one measurement per subject: no within-subject tracking")
    else:
        ref_centred, pred_centred = ref - ref_mean[index], pred - pred_mean[index]
        out["r_within"] = pearson(ref_centred, pred_centred)
        out["slope_within"] = _slope(ref_centred, pred_centred)
        if not np.isfinite(out["slope_within"]):
            notes.append("the reference does not vary within any subject")
    out["tracking_note"] = "; ".join(notes)
    return out


def limits(stats: dict) -> tuple:
    """The 95 % limits of agreement, on the corrected SD."""
    half = LIMIT_Z * stats["s_corr"]
    return stats["mean_error"] - half, stats["mean_error"] + half


def agreement(frames: dict) -> pd.DataFrame:
    """One row per parameter: the repeated-measures accuracy statistics of
    its errors over the measurements that have one, the participant as the
    subject, and the level-tracking columns of :func:`tracking`."""
    rows = []
    for p in PARAMETERS:
        pairs = pairs_of(frames[p.name], p)
        rows.append({"parameter": p.name,
                     **repeated_measures(pairs["error"], pairs["subject"]),
                     **tracking(pairs["ref"], pairs["pred"], pairs["subject"])})
    return pd.DataFrame(rows)


def waveform_agreement(frames: dict) -> pd.DataFrame:
    """One row per signal and waveform metric: the repeated-measures
    statistics of the metric across measurements, the participant as the
    subject, with the median and quartiles beside the mean because a few
    participants skew the per-measurement distribution. Descriptive only: a
    score is not a signed error, so ``s_corr`` is its corrected spread, not
    a limit of agreement, and no criterion applies."""
    rows = []
    for sig, frame in frames.items():
        for metric in WAVEFORM_METRICS:
            scored = frame.dropna(subset=[metric])
            stats = repeated_measures(scored[metric], scored["participant"])
            values = scored[metric].to_numpy(dtype=np.float64)
            q1, median, q3 = (np.percentile(values, (25, 50, 75)) if values.size
                              else (_NAN, _NAN, _NAN))
            row = {"signal": sig, "metric": metric}
            for k, v in stats.items():
                row["mean" if k == "mean_error" else k] = v
                if k == "mean_error":
                    row.update({"median": float(median), "q1": float(q1), "q3": float(q3)})
            rows.append(row)
    return pd.DataFrame(rows)
