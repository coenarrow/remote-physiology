"""OMIT
Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.
Álvarez Casado, C., & Bordallo López, M.
IEEE Journal of Biomedical and Health Informatics.
(2023).
"""

import numpy as np
from einops import rearrange

from unsupervised_methods import utils


def OMIT(frames):
    rgb = utils.process_video(frames)[0]                  # (3, T)
    Q, _ = np.linalg.qr(rgb)
    leading = rearrange(Q[:, 0], "c -> 1 c")              # first orthonormal direction
    projector = np.identity(3) - leading.T @ leading
    Y = projector @ rgb                                   # (3, T)
    return Y[1, :]
