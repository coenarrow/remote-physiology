"""Heart rate from every cardiac trace, and from all of them fused.

For every trace the registry marks cardiac (PPG, ECG, ABP, CVP), the
estimate the upstream toolbox made: detrend, bandpass to the heart-rate
band, periodogram, the largest in-band bin, in beats per minute
(:func:`clean`, :func:`spectrum`, :func:`rate_of`). Run on the label and
on the prediction, so each side has its own rate. Where a stretch carries
several cardiac traces they are fused (:func:`fuse`):

    The traces' power spectra, each normalised to unit power in the band,
    combined as their weighted geometric mean — a product of spectra, so
    the frequency the traces agree on wins and a peak only one of them has
    (CVP's respiratory harmonic, say) is suppressed. A trace's weight is its
    *auto-SNR*, the SNR around its own spectral peak rather than around a
    reference rate, in dB and clipped at zero: a measure of how peaky the
    spectrum is, so a trace that splits its power between the fundamental
    and its harmonics (CVP, whose largest bin is often twice the heart
    rate) counts for little and one whose peak holds less power than the
    rest of the band counts for nothing. Power spectra rather than complex
    ones: the traces are phase-shifted against each other by transit time
    and morphology, and a complex sum would cancel at the very fundamental
    it is after. Nothing goes back to a waveform, because a rate needs no
    phase. The prediction side fuses exactly the traces the label side has,
    each side weighted by its own auto-SNRs and blind to the other, so the
    fused prediction is compared with a fused label built the same way.

No config. The detrender is the upstream one, the band is wider than
upstream's 36–198 bpm, and the trace tables are already in physical units,
so nothing here needs to know how a trace's label was normalised.
The two upstream helpers this needs (the smoothness-prior detrender and
the FFT length) live at the top of this module; the rest of the toolbox's
post-processing is gone.
"""

import functools

import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import butter, filtfilt, periodogram
from scipy.sparse import diags as sparse_diags

#: The heart-rate band in Hz, 30–240 bpm.
BAND = (0.5, 4.0)
DETREND_LAMBDA = 100
#: Half-width of the harmonic bins counted as signal in the SNR, in bpm.
SNR_DEVIATION_BPM = 6
#: ``filtfilt`` pads ``3 * max(len(a), len(b))`` = 9 samples for the
#: first-order bandpass and needs strictly more than that to work on.
MIN_FRAMES = 10
#: Floor under a normalised spectrum before its log, so one trace's empty bin
#: cannot veto the frequency every other trace favours.
SPECTRUM_FLOOR = 1e-12
_NAN = float("nan")


# ---------------------------------------------------------------------------
# The upstream toolbox's helpers
# ---------------------------------------------------------------------------
def next_power_of_2(x: int) -> int:
    """The smallest power of two at or above ``x``."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@functools.lru_cache(maxsize=32)
def _detrend_bands(signal_length: int, lambda_value: float) -> np.ndarray:
    """Upper-banded form of ``I + lambda^2 * D'D`` for the smoothness prior.

    ``D`` is the second-difference operator, so ``D'D`` is symmetric and
    pentadiagonal: only three diagonals are needed, and they depend on nothing
    but the length, hence the cache.
    """
    second_difference = sparse_diags(
        [1.0, -2.0, 1.0], [0, 1, 2], shape=(signal_length - 2, signal_length))
    gram = (second_difference.T @ second_difference).tocsr()
    banded = np.zeros((3, signal_length))
    banded[2] = 1.0 + lambda_value ** 2 * gram.diagonal(0)
    banded[1, 1:] = lambda_value ** 2 * gram.diagonal(1)
    banded[0, 2:] = lambda_value ** 2 * gram.diagonal(2)
    return banded


def detrend(trace, lambda_value: float) -> np.ndarray:
    """Tarvainen's smoothness-prior detrender: removes the smooth trend ``z``
    that solves ``(I + lambda^2 D'D) z = x`` and returns ``x - z``, solved as
    the banded system it is (O(n), not a dense inverse). A trace too short
    to difference is returned as is."""
    trace = np.asarray(trace, dtype=np.float64)
    length = trace.shape[0]
    if length < 3:
        return trace
    flat = trace.reshape(length, -1)
    trend = solveh_banded(_detrend_bands(length, float(lambda_value)), flat, lower=False)
    return (flat - trend).reshape(trace.shape)


# ---------------------------------------------------------------------------
# One trace to one spectrum to one rate
# ---------------------------------------------------------------------------
def clean(trace, fs: float) -> np.ndarray:
    """Detrended and zero-phase bandpassed to the heart-rate band."""
    detrended = detrend(np.asarray(trace, dtype=np.float64), DETREND_LAMBDA)
    b, a = butter(1, [BAND[0] / fs * 2, BAND[1] / fs * 2], btype="bandpass")
    return filtfilt(b, a, detrended)


def spectrum(trace, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, power)`` of a cleaned trace, zero-padded to a power of two."""
    trace = np.asarray(trace, dtype=np.float64)
    return periodogram(trace, fs=fs, nfft=next_power_of_2(trace.size), detrend=False)


def in_band(freqs) -> np.ndarray:
    return (freqs >= BAND[0]) & (freqs <= BAND[1])


def rate_of(freqs, power) -> float:
    """The largest in-band bin, in beats per minute."""
    band = in_band(freqs)
    return float(freqs[band][np.argmax(power[band])] * 60)


def normalised(freqs, power) -> np.ndarray:
    """The spectrum scaled to unit power inside the band, so every trace
    weighs the same regardless of its units and however much power sits at
    DC or in the respiratory range."""
    total = power[in_band(freqs)].sum()
    return power / total if total > 0 else power


def fuse(freqs, powers) -> np.ndarray:
    """Geometric mean of the traces' normalised spectra, bin by bin, each
    weighted by its auto-SNR in dB clipped at zero (a NaN counts as zero).
    Uniform when no trace has a positive one."""
    stacked = np.stack([normalised(freqs, p) for p in powers])
    weights = np.nan_to_num([max(snr(freqs, p), 0.0) for p in powers])
    if weights.sum() <= 0:
        weights = np.ones(len(powers))
    weights = weights / weights.sum()
    return np.exp(weights @ np.log(np.maximum(stacked, SPECTRUM_FLOOR)))


def snr(freqs, power, hr_bpm: float | None = None) -> float:
    """Power within +/- 6 bpm of the reference rate and its second harmonic,
    over the rest of the band, in dB (upstream's definition). Without a
    reference rate it is the *auto-SNR*, centred on the spectrum's own rate:
    how peaky the spectrum is, whatever the peak is of."""
    if hr_bpm is None:
        hr_bpm = rate_of(freqs, power)
    deviation = SNR_DEVIATION_BPM / 60
    harmonic = np.zeros_like(freqs, dtype=bool)
    for centre in (hr_bpm / 60, 2 * hr_bpm / 60):
        harmonic |= (freqs >= centre - deviation) & (freqs <= centre + deviation)
    remainder = in_band(freqs) & ~harmonic
    signal_power, noise_power = power[harmonic].sum(), power[remainder].sum()
    if noise_power <= 0 or signal_power <= 0:
        return _NAN
    return float(10 * np.log10(signal_power / noise_power))
