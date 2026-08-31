"""Shared aggregation and printing of window-level HR metrics.

``evaluation.metrics`` (supervised) and ``unsupervised_methods`` (traditional)
both reduce a pile of per-window ``(ground-truth HR, predicted HR, SNR, MACC)``
tuples to the same handful of numbers and print them the same way. That
reduction lives here once, so a new evaluation path — like the per-signal
Neckflix one, which reports the same table for ABP, CVP and ECG separately —
gets it for free and stays comparable.
"""

import numpy as np

from evaluation.BlandAltmanPy import BlandAltman

#: Metric names understood by :func:`report_hr_metrics`.
SUPPORTED_METRICS = ("MAE", "RMSE", "MAPE", "Pearson", "SNR", "MACC", "BA")


def _standard_error(values, count):
    return float(np.std(values) / np.sqrt(count)) if count else float("nan")


def report_hr_metrics(gt_hr, pred_hr, snr, macc, *, metrics, config, filename_id,
                      hr_method="FFT", scope="", printer=print):
    """Compute, print and return the configured HR metrics for one group.

    ``scope`` labels the group in the printed lines (e.g. the signal a
    Neckflix run derived its reference HR from); it is empty for the
    single-group datasets. Returns ``{metric: value}`` plus ``n`` so callers can
    tabulate results without re-parsing stdout.
    """
    gt_hr = np.asarray(gt_hr, dtype=np.float64)
    pred_hr = np.asarray(pred_hr, dtype=np.float64)
    snr = np.asarray(snr, dtype=np.float64)
    macc = np.asarray(macc, dtype=np.float64)

    tag = f"[{scope}] " if scope else ""
    n = len(pred_hr)
    results = {"n": n}
    if n == 0:
        printer(f"{tag}no evaluable windows — nothing to report")
        return results

    errors = pred_hr - gt_hr
    for metric in metrics:
        if metric == "MAE":
            value = float(np.mean(np.abs(errors)))
            printer(f"{tag}{hr_method} MAE: {value} +/- {_standard_error(np.abs(errors), n)}")
        elif metric == "RMSE":
            # Standard error is taken on the squared errors, then rooted, so an
            # unusual error distribution cannot distort it.
            squared = np.square(errors)
            value = float(np.sqrt(np.mean(squared)))
            printer(f"{tag}{hr_method} RMSE: {value} +/- {float(np.sqrt(np.std(squared) / np.sqrt(n)))}")
        elif metric == "MAPE":
            with np.errstate(divide="ignore", invalid="ignore"):
                relative = np.abs(errors / gt_hr)
            relative = relative[np.isfinite(relative)]
            value = float(np.mean(relative) * 100) if relative.size else float("nan")
            printer(f"{tag}{hr_method} MAPE: {value} +/- {_standard_error(relative, relative.size) * 100}")
        elif metric == "Pearson":
            # Correlation is undefined for < 3 points or a constant series (which
            # a short evaluation split, or a model predicting one rate, produces).
            # Say so rather than printing "nan +/- nan".
            if n < 3:
                value = float("nan")
                printer(f"{tag}{hr_method} Pearson: undefined, needs >= 3 windows (got {n})")
            elif np.std(pred_hr) == 0 or np.std(gt_hr) == 0:
                value = float("nan")
                which = "predicted" if np.std(pred_hr) == 0 else "ground-truth"
                printer(f"{tag}{hr_method} Pearson: undefined, {which} HR is constant "
                        f"across all {n} windows")
            else:
                value = float(np.corrcoef(pred_hr, gt_hr)[0][1])
                printer(f"{tag}{hr_method} Pearson: {value} "
                        f"+/- {float(np.sqrt(max(1 - value ** 2, 0.0) / (n - 2)))}")
        elif metric == "SNR":
            value = float(np.mean(snr))
            printer(f"{tag}{hr_method} SNR: {value} +/- {_standard_error(snr, n)} (dB)")
        elif metric == "MACC":
            value = float(np.mean(macc))
            printer(f"{tag}MACC: {value} +/- {_standard_error(macc, n)}")
        elif "BA" in metric:
            _bland_altman_plots(gt_hr, pred_hr, config, filename_id, hr_method, scope)
            value = None
        else:
            raise ValueError(
                f"Unsupported metric {metric!r}; known: {', '.join(SUPPORTED_METRICS)}"
            )
        results[metric] = value
    return results


def _bland_altman_plots(gt_hr, pred_hr, config, filename_id, hr_method, scope):
    """Write the scatter and difference plots for one group."""
    suffix = f"_{scope}" if scope else ""
    compare = BlandAltman(gt_hr, pred_hr, config, averaged=True)
    stem = f"{filename_id}{suffix}_{hr_method}_BlandAltman"
    compare.scatter_plot(
        x_label='GT HR [bpm]',
        y_label='rPPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{stem}_ScatterPlot',
        file_name=f'{stem}_ScatterPlot.pdf')
    compare.difference_plot(
        x_label='Difference between rPPG HR and GT HR [bpm]',
        y_label='Average of rPPG HR and GT HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{stem}_DifferencePlot',
        file_name=f'{stem}_DifferencePlot.pdf')
