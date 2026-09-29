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
    n_ind = k * (1 + (1 - icc) * (f_ba - 1)) if np.isfinite(icc) else n
    out.update({"f_ba": float(f_ba), "ms_between": ms_between, "ms_within": ms_within,
                "s_corr": float(np.sqrt(total)), "icc": float(icc),
                "n_ind": float(n_ind)})
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
             "met": "yes" if np.isfinite(value) and met else "no"}
            for name, value, threshold, met in rows]


def rate_metrics(ref, pred) -> dict:
    """What rPPG papers report of a rate: MAE, RMSE, MAPE (percent of the
    reference) and Pearson r."""
    ref, pred = np.asarray(ref, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if ref.size == 0:
        return {"mae": _NAN, "rmse": _NAN, "mape": _NAN, "r": _NAN}
    error = pred - ref
    mape = float((np.abs(error) / np.abs(ref)).mean() * 100) if (ref != 0).all() else _NAN
    return {**spread(error), "mape": mape, "r": pearson(pred, ref)}


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
