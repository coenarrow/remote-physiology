"""``1 - r`` at the best lag: waveform shape only, tolerant of a transit delay.

``NegPearson`` scored where the label was measured somewhere else than the
camera looks: the neck pulses a transit time after the trace the label took
its phase from, so the correlation is searched over the lags within
``MAX_LAG_SECONDS`` (``ccc.best_lag``) and scored at the best one. The
evaluation's counterpart is ``waveform_r`` (``src/evaluation/recording.py``).
"""

import torch.nn as nn

from neural_methods.loss.ccc import MAX_LAG_SECONDS, best_lag, lag_frames
from neural_methods.loss.neg_pearson import NegPearson


class LaggedNegPearson(nn.Module):
    def __init__(self, fs: float, max_lag_seconds: float = MAX_LAG_SECONDS):
        """``fs``: the interface's frame rate, which turns the lag bound from
        seconds into frames."""
        super().__init__()
        self.max_lag = lag_frames(max_lag_seconds, fs)
        self.neg_pearson = NegPearson()

    def pearson(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)`` Pearson correlation."""
        return 1.0 - self.neg_pearson(pred, label)

    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        return 1.0 - best_lag(self.pearson, pred, label, self.max_lag)
