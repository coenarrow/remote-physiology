"""LGI
Local group invariance for heart rate estimation from face videos.
Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V.
In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 1254–1262
(2018).
"""

import numpy as np
from einops import rearrange

from unsupervised_methods import utils


def LGI(frames):
    rgb = utils.process_video(frames)                        # (1, 3, T)
    U, _, _ = np.linalg.svd(rgb)
    principal = rearrange(U[:, :, 0], "b c -> b c 1")        # leading left singular vector
    projector = np.identity(3) - principal @ rearrange(principal, "b c one -> b one c")
    Y = projector @ rgb                                      # (1, 3, T)
    return rearrange(Y[:, 1, :], "b t -> (b t)")
