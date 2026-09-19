"""Absolute error of the window mean: mean arterial pressure, in mmHg.

The evaluation's counterpart is ``err_mean`` (``src/evaluation/recording.py``),
the per-beat mean averaged over the recording's beats; over a window of many
beats the two differ only by the partial beats at its edges.
"""

import torch.nn as nn


class MeanL1(nn.Module):
    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        return (pred.mean(dim=-1) - label.mean(dim=-1)).abs()
