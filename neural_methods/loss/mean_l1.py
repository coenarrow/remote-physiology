"""Absolute error of the window mean: mean arterial pressure, in mmHg.

The evaluation's counterparts are ABP mean and CVP mean
(``src/evaluation/measurements.py``), the sample mean of each 10 s
measurement; the two differ only in the span they average over.
"""

import torch.nn as nn


class MeanL1(nn.Module):
    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        return (pred.mean(dim=-1) - label.mean(dim=-1)).abs()
