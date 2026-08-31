"""Evaluation driver for the traditional (unsupervised) rPPG methods.

Handles both dataset contracts:

* the legacy tuple ``(frames, labels, filename, chunk_id)`` the upstream
  loaders emit, evaluated against its single label; and
* the Neckflix nested batch dict, evaluated **per label signal** -- ABP, CVP and
  ECG each carry the cardiac rhythm, so one video yields one HR estimate scored
  against every reference trace that recording actually has. ``label_mask``
  decides which those are, so a recording missing ABP simply contributes
  nothing to the ABP row instead of being dropped.

Both paths funnel into the same per-window HR comparison and the same metric
report (:mod:`evaluation.metrics_report`).
"""

from collections import defaultdict

import numpy as np
from tqdm import tqdm

from evaluation.metrics_report import report_hr_metrics
from evaluation.post_process import calculate_metric_per_video
from neural_methods.batch import (
    CHANNEL_MASK, FRAMES, LABEL_MASK, LABELS, METADATA,
    frames_to_rgb_trace, is_batch_dict, iter_samples,
)
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.LGI import LGI
from unsupervised_methods.methods.OMIT import OMIT
from unsupervised_methods.methods.PBV import PBV
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.utils import rgb_trace

#: Every supported method, as ``(clip_or_trace, fs) -> BVP``. The rate-free
#: methods ignore ``fs``; wrapping them here keeps the dispatch a lookup, not a
#: chain.
BVP_ESTIMATORS = {
    "POS": lambda video, fs: POS_WANG(video, fs),
    "CHROM": lambda video, fs: CHROME_DEHAAN(video, fs),
    "ICA": lambda video, fs: ICA_POH(video, fs),
    "GREEN": lambda video, fs: GREEN(video),
    "LGI": lambda video, fs: LGI(video),
    "PBV": lambda video, fs: PBV(video),
    "OMIT": lambda video, fs: OMIT(video),
}

#: The three channels every traditional method consumes.
RGB_CHANNELS = ("R", "G", "B")

#: Shortest window ``calculate_metric_per_video`` can filter (filtfilt padlen).
MIN_WINDOW = 9

#: Signal name reported for the legacy single-label datasets.
LEGACY_SIGNAL = "PPG"


def estimate_bvp(method_name, video, fs):
    """Run one named method over a ``(T, H, W, 3)`` clip or a ``(T, 3)`` trace."""
    try:
        estimator = BVP_ESTIMATORS[method_name]
    except KeyError:
        raise ValueError(
            f"unsupervised method name wrong: {method_name!r}; "
            f"known: {', '.join(sorted(BVP_ESTIMATORS))}"
        ) from None
    return np.asarray(estimator(video, fs))


def _dict_windows(batch):
    """Yield ``(rgb_trace, {signal: reference_trace}, name)`` from a Neckflix batch.

    The clip is reduced to its per-frame mean RGB here, once, because that is
    all any of the methods consume.
    """
    for sample in iter_samples(batch):
        frames = sample[FRAMES]
        missing = [ch for ch in RGB_CHANNELS if ch not in frames]
        if missing:
            raise ValueError(
                f"The unsupervised methods need channels {list(RGB_CHANNELS)}; the "
                f"batch is missing {missing}. Set PREPROCESS.CHANNELS to include them."
            )
        absent = [ch for ch in RGB_CHANNELS if not bool(sample[CHANNEL_MASK][ch])]
        if absent:
            continue  # zero-filled RGB carries no pulse; scoring it is meaningless
        references = {
            signal: trace.numpy()
            for signal, trace in sample[LABELS].items()
            if bool(sample[LABEL_MASK][signal])
        }
        if not references:
            continue
        meta = sample[METADATA]
        name = "{}_cam{}@{}".format(
            meta["recording_id"], meta["camera_id"], int(meta["start_frame"]))
        yield frames_to_rgb_trace(frames, RGB_CHANNELS), references, name


def _legacy_windows(batch):
    """Yield the same triples from the upstream ``(frames, labels, ...)`` tuple."""
    frames, labels = batch[0], batch[1]
    for idx in range(frames.shape[0]):
        video = rgb_trace(frames[idx].cpu().numpy()[..., :3])
        reference = labels[idx].cpu().numpy()
        name = str(batch[2][idx]) if len(batch) > 2 else str(idx)
        yield video, {LEGACY_SIGNAL: reference}, name


def _window_size(config, n_frames):
    """Evaluation window length in frames, clipped to what the clip provides."""
    window_cfg = config.INFERENCE.EVALUATION_WINDOW
    if not window_cfg.USE_SMALLER_WINDOW:
        return n_frames
    return min(window_cfg.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS, n_frames)


def _hr_method(config):
    method = config.INFERENCE.EVALUATION_METHOD
    if method == "peak detection":
        return "Peak"
    if method == "FFT":
        return "FFT"
    raise ValueError(f"Inference evaluation method name wrong: {method!r}")


def _accumulate(config, data_loader, method_names):
    """One pass over the data, scoring every named method on every label signal.

    Loading a Neckflix window means decompressing a few hundred frames out of
    zarr, so the seven methods share a single pass rather than each triggering
    its own: the spatial means they consume come from the same decoded clip.

    Returns ``{method: {signal: {"gt"/"pred"/"snr"/"macc": [...]}}}``.
    """
    hr_method = _hr_method(config)
    fs = config.UNSUPERVISED.DATA.FS
    groups = {method: defaultdict(lambda: defaultdict(list)) for method in method_names}

    for test_batch in tqdm(data_loader, ncols=80):
        windows = _dict_windows(test_batch) if is_batch_dict(test_batch) \
            else _legacy_windows(test_batch)
        for video, references, _name in windows:
            window_frame_size = _window_size(config, video.shape[0])
            for method_name in method_names:
                bvp = estimate_bvp(method_name, video, fs)
                for start in range(0, len(bvp), window_frame_size):
                    bvp_window = bvp[start:start + window_frame_size]
                    if len(bvp_window) < MIN_WINDOW:
                        print(f"Window frame size of {len(bvp_window)} is smaller than "
                              f"minimum pad length of {MIN_WINDOW}. Window ignored!")
                        continue
                    for signal, reference in references.items():
                        label_window = reference[start:start + len(bvp_window)]
                        # A method whose output is shorter than the video (CHROM
                        # drops a partial final window) must not be compared
                        # against a longer label slice.
                        usable = min(len(bvp_window), len(label_window))
                        if usable < MIN_WINDOW:
                            continue
                        gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
                            bvp_window[:usable], label_window[:usable],
                            diff_flag=False, fs=fs, hr_method=hr_method)
                        group = groups[method_name][signal]
                        group["gt"].append(gt_hr)
                        group["pred"].append(pred_hr)
                        group["snr"].append(snr)
                        group["macc"].append(macc)
    return groups


def _report(config, method_name, signal_groups):
    """Print and return the metric table for one method."""
    print("Used Unsupervised Method: " + method_name)
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE != "unsupervised_method":
        raise ValueError(
            "unsupervised_predictor.py evaluation only supports unsupervised_method!")
    filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET

    if not signal_groups:
        print("No evaluable windows found - check the label masks and channels.")
        return {}

    hr_method = _hr_method(config)
    multi_signal = set(signal_groups) != {LEGACY_SIGNAL}
    report = {}
    for signal in sorted(signal_groups):
        group = signal_groups[signal]
        report[signal] = report_hr_metrics(
            group["gt"], group["pred"], group["snr"], group["macc"],
            metrics=config.UNSUPERVISED.METRICS, config=config,
            filename_id=filename_id, hr_method=hr_method,
            scope=signal if multi_signal else "")
    return report


def unsupervised_predict(config, data_loader, method_name):
    """Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    groups = _accumulate(config, data_loader["unsupervised"], [method_name])
    return _report(config, method_name, groups[method_name])


def unsupervised_predict_many(config, data_loader, method_names):
    """Evaluate several methods in a single pass over the data.

    Equivalent to calling :func:`unsupervised_predict` once per method, but
    decodes each video once instead of once per method.
    """
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    method_names = list(method_names)
    print("===Unsupervised Methods ( " + ", ".join(method_names) + " ) Predicting ===")
    groups = _accumulate(config, data_loader["unsupervised"], method_names)
    return {method: _report(config, method, groups[method]) for method in method_names}
