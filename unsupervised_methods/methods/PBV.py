"""PBV
Improved motion robustness of remote-ppg by using the blood volume pulse signature.
De Haan, G. & Van Leest, A.
Physiol. measurement 35, 1913 (2014)

Implementation note: the original relied on ``np.linalg.solve`` treating a 2-D
right-hand side as a stack of vectors. NumPy 2.0 made that a matrix solve, so
the right-hand side is now given an explicit trailing axis; the chained
``swapaxes``/``transpose`` shuffles are einops rearranges. Same arithmetic.
"""

import numpy as np
from einops import rearrange

from unsupervised_methods import utils


def _pbv_signature(rgb):
    """Blood-volume signature and the mean-normalised traces it is built from.

    ``rgb`` is ``(1, 3, T)``; returns ``(norm, pbv)`` shaped ``(1, 3, T)`` and
    ``(1, 3)``.
    """
    norm = rgb / rearrange(rgb.mean(axis=2), "b c -> b c 1")
    spread = np.sqrt(norm.var(axis=2).sum(axis=1))            # (1,)
    pbv = norm.std(axis=2) / rearrange(spread, "b -> b 1")    # (1, 3)
    return norm, pbv


def PBV(frames):
    rgb = utils.process_video(frames)                          # (1, 3, T)
    norm, pbv = _pbv_signature(rgb)
    channels_last = rearrange(norm, "b c t -> b t c")          # (1, T, 3)
    covariance = norm @ channels_last                          # (1, 3, 3)
    weights = np.linalg.solve(covariance, rearrange(pbv, "b c -> b c 1"))   # (1, 3, 1)
    numerator = channels_last @ weights                        # (1, T, 1)
    denominator = rearrange(pbv, "b c -> b 1 c") @ weights     # (1, 1, 1)
    return rearrange(numerator / denominator, "b t 1 -> (b t)")


def PBV2(frames):
    """Variant kept from upstream; identical output to :func:`PBV`."""
    return PBV(frames)
