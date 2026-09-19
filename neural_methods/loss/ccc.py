"""``1 - CCC``: the well-conditioned base term for an absolute signal.

Lin's concordance is dimensionless and O(1) whatever the units, yet unlike a
correlation it penalises a wrong mean and a wrong amplitude, which is exactly
the part of an absolute-class prediction that negpearson would throw away.
The evaluation's counterpart is ``waveform_ccc`` (``src/evaluation/recording.py``).
"""

import torch.nn as nn
from einops import rearrange

EPS = 1e-8


class CCC(nn.Module):
    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        mx = pred.mean(dim=-1, keepdim=True)
        my = label.mean(dim=-1, keepdim=True)
        vx = ((pred - mx) ** 2).mean(dim=-1, keepdim=True)
        vy = ((label - my) ** 2).mean(dim=-1, keepdim=True)
        cov = ((pred - mx) * (label - my)).mean(dim=-1, keepdim=True)
        coefficient = (2 * cov) / (vx + vy + (mx - my) ** 2 + EPS)
        return rearrange(1.0 - coefficient, "b 1 -> b")
