"""``1 - r`` at the best lag: waveform shape, tolerant of a transit delay.

A pulse seen in the video and the same pulse at the pressure transducer are
apart by the time it takes to travel between them, which differs between
patients and between traces. ``NegPearson`` charges that delay as a shape
error; this component searches the lags within ``max_lag`` frames and scores
the correlation at the best one. The lag is picked by a detached argmax, so
the gradient flows only through the correlation at the chosen alignment.
"""

import torch
import torch.nn as nn

EPS = 1e-8


class LagPearson(nn.Module):
    def __init__(self, max_lag: int = 15):
        """``max_lag``: half-width of the lag search, in frames (half a second
        at 30 fps); clipped to what the window leaves room for."""
        super().__init__()
        self.max_lag = max_lag

    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        frames = pred.shape[-1]
        max_lag = min(self.max_lag, frames - 2)
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            # A positive lag delays the prediction against the label; the
            # samples without a partner are dropped from both.
            pre = pred[:, max(lag, 0):frames + min(lag, 0)]
            lab = label[:, max(-lag, 0):frames - max(lag, 0)]
            pre = pre - pre.mean(dim=-1, keepdim=True)
            lab = lab - lab.mean(dim=-1, keepdim=True)
            num = (pre * lab).sum(dim=-1)
            # EPS inside the root: a flat segment would otherwise give the
            # root an infinite gradient that no later clamp can undo.
            den = torch.sqrt((pre ** 2).sum(dim=-1) + EPS) * torch.sqrt((lab ** 2).sum(dim=-1) + EPS)
            correlations.append(num / den)
        correlations = torch.stack(correlations)                       # (lags, B)
        best = correlations.argmax(dim=0, keepdim=True).detach()
        return 1.0 - correlations.gather(0, best)[0]
