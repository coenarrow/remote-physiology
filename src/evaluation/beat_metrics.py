"""One beat detector for every cardiac trace, on the label and on the prediction.

The trace is cleaned as the heart-rate estimator cleans it (detrend,
zero-phase bandpass to the heart-rate band), multiplied by the signal's
polarity from the registry, and ``scipy.signal.find_peaks`` runs with the
minimum distance between beats set by the top of the band and the
prominence a registry fraction of the cleaned range. Each candidate is then
moved to the extremum of the raw trace within a quarter of the median beat
interval, so the peak is read off the real waveform, not the filtered one.
Troughs are the peaks of the negated trace.
"""

import numpy as np
from scipy.signal import find_peaks

from src.evaluation.rate import BAND, MIN_FRAMES, clean
from src.signal_transforms import beat_config

#: A candidate peak is moved to the raw extremum within this fraction of the
#: median inter-beat interval.
REFINE_FRACTION = 0.25


def _median_interval(peaks: np.ndarray, fs: float) -> float:
    """Samples between beats; one second when there are too few beats to tell."""
    return float(np.median(np.diff(peaks))) if peaks.size > 1 else float(fs)


def detect_beats(trace, fs: float, sig: str) -> np.ndarray:
    """Sample indices of the beats of one finite trace, sorted, unique, the
    first and last detected dropped."""
    trace = np.asarray(trace, dtype=np.float64)
    if trace.size < MIN_FRAMES or not np.all(np.isfinite(trace)):
        raise ValueError(
            f"detect_beats needs a finite trace of at least {MIN_FRAMES} samples")
    config = beat_config(sig)
    signed = trace * config["polarity"]
    cleaned = clean(signed, fs)
    span = float(np.ptp(cleaned))
    if span <= 0:
        return np.array([], dtype=int)
    candidates, _ = find_peaks(cleaned, distance=max(1, int(fs / BAND[1])),
                               prominence=config["prominence"] * span)
    if candidates.size == 0:
        return candidates.astype(int)
    radius = max(1, int(round(REFINE_FRACTION * _median_interval(candidates, fs))))
    refined = []
    for c in candidates:
        lo, hi = max(0, c - radius), min(signed.size, c + radius + 1)
        refined.append(lo + int(np.argmax(signed[lo:hi])))
    # The first and last beats sit on the filter's edge transients and may
    # be cut by the stretch's ends, so neither is trusted.
    return np.unique(np.asarray(refined, dtype=int))[1:-1]
