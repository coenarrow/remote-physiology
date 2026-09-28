"""``1 - r``: waveform shape only, blind to level and amplitude.

The loss of a shape-class signal (PPG, ECG, RESP), which arrives per-window
z-scored, and the one upstream trains PhysNet, iBVPNet and FactorizePhys with.
The evaluation's counterpart is ``waveform_r`` (``src/evaluation/recording.py``).
"""

import torch
import torch.nn as nn

EPS = 1e-8


class NegPearson(nn.Module):
    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        pre = pred - pred.mean(dim=-1, keepdim=True)
        lab = label - label.mean(dim=-1, keepdim=True)
        num = (pre * lab).sum(dim=-1)
        den = torch.sqrt((pre ** 2).sum(dim=-1) * (lab ** 2).sum(dim=-1) + EPS)
        return 1.0 - num / den
