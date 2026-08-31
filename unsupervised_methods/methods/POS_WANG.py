"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic principles of remote PPG.
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

Implementation notes: the original used ``np.mat``, removed in NumPy 2.0, and
ran its overlap-add in a Python loop over every frame. This computes the same
quantity with a strided sliding window and einops reshapes;
``tests/test_unsupervised_methods.py`` pins it to the frozen pre-NumPy-2
original.
"""

import math

import numpy as np
from einops import einsum, rearrange
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from unsupervised_methods import utils

# The POS projection: rows are the two chrominance combinations of normalised RGB.
_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def _process_video(frames):
    """Spatial mean of each frame: ``(T, H, W, 3)`` -> ``(T, 3)``."""
    return utils.rgb_trace(frames)


def pos_signal(frames, fs, WinSec=1.6):
    """The detrended POS overlap-add signal, before any bandpass.

    Split out because the loaders' pseudo-PPG label generation wants exactly
    this and then applies its own (differently tuned) filter — previously each
    had its own copy of the algorithm.
    """
    RGB = _process_video(frames)                     # (T, 3)
    n_frames = RGB.shape[0]
    window = math.ceil(WinSec * fs)
    if n_frames <= window:
        return np.zeros(n_frames)

    # One (start, length) block per window end in [window, n_frames): the same
    # blocks the original loop visited, stacked instead of iterated.
    blocks = sliding_window_view(RGB, window, axis=0)[:n_frames - window]   # (M, 3, L)
    normalised = blocks / blocks.mean(axis=2, keepdims=True)
    projected = einsum(_PROJECTION, normalised, "k c, m c l -> m k l")      # (M, 2, L)
    alpha = projected[:, 0].std(axis=1) / projected[:, 1].std(axis=1)       # (M,)
    h = projected[:, 0] + rearrange(alpha, "m -> m 1") * projected[:, 1]    # (M, L)
    h = h - h.mean(axis=1, keepdims=True)

    # Overlap-add: block m contributes to frames [m, m + L).
    starts = rearrange(np.arange(h.shape[0]), "m -> m 1")
    offsets = rearrange(np.arange(window), "l -> 1 l")
    overlap_added = np.bincount((starts + offsets).ravel(), weights=h.ravel(),
                                minlength=n_frames)
    return utils.detrend(overlap_added, 100)


def POS_WANG(frames, fs):
    BVP = pos_signal(frames, fs)
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    return signal.filtfilt(b, a, BVP.astype(np.double))
