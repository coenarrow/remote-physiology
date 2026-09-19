"""Pointwise squared error, the loss upstream trains DeepPhys, TS-CAN and
EfficientPhys with.

The evaluation's counterpart is ``waveform_rmse`` (``src/evaluation/recording.py``),
its square root over the whole recording.
"""

import torch.nn as nn


class MSE(nn.Module):
    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        return ((pred - label) ** 2).mean(dim=-1)
