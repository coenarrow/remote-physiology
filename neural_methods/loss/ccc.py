"""``1 - CCC`` at the best lag: level, amplitude and shape, tolerant of a
transit delay.

Lin's concordance is dimensionless and O(1) whatever the units, yet unlike a
correlation it penalises a wrong mean and a wrong amplitude, which is exactly
the part of an absolute-class prediction that negpearson would throw away.

The label is measured somewhere else than the camera looks: a pulse seen in
the video and the same pulse at the pressure transducer are apart by the time
it takes to travel between them, which differs between patients and between
traces. Scored at lag zero, that delay would be charged as a shape error, so
this component searches the lags within ``MAX_LAG_SECONDS`` and scores the
concordance at the best one. The lag is picked by a detached argmax, so the
gradient flows only through the concordance at the chosen alignment.
The evaluation's counterpart is ``waveform_ccc`` (``src/evaluation/recording.py``).
"""

import torch
import torch.nn as nn

EPS = 1e-8
#: Half-width of the lag search. Half a second covers the pulse transit time
#: from the heart to any peripheral site with margin to spare.
MAX_LAG_SECONDS = 0.5


def lag_frames(max_lag_seconds: float, fs: float) -> int:
    """The lag search's half-width in frames at ``fs``."""
    return int(round(max_lag_seconds * fs))


def lagged_pair(pred, label, lag: int):
    """The overlapping stretches of ``pred`` and ``label`` at ``lag``.

    A positive lag delays the prediction against the label; the samples
    without a partner are dropped from both. Works on the last axis of any
    array-like that slices, so the evaluation shares the convention.
    """
    frames = pred.shape[-1]
    return (pred[..., max(lag, 0):frames + min(lag, 0)],
            label[..., max(-lag, 0):frames - max(lag, 0)])


class CCC(nn.Module):
    def __init__(self, fs: float, max_lag_seconds: float = MAX_LAG_SECONDS):
        """``fs``: the interface's frame rate, which turns the lag bound from
        seconds into frames. The search is clipped to what the window leaves
        room for."""
        super().__init__()
        self.max_lag = lag_frames(max_lag_seconds, fs)

    @staticmethod
    def concordance(pred, label):
        """``(B, T)`` prediction and label -> ``(B,)`` Lin's concordance."""
        mx = pred.mean(dim=-1)
        my = label.mean(dim=-1)
        vx = ((pred - mx[:, None]) ** 2).mean(dim=-1)
        vy = ((label - my[:, None]) ** 2).mean(dim=-1)
        cov = ((pred - mx[:, None]) * (label - my[:, None])).mean(dim=-1)
        return (2 * cov) / (vx + vy + (mx - my) ** 2 + EPS)

    def forward(self, pred, label):
        """``(B, T)`` prediction and label -> ``(B,)``."""
        max_lag = min(self.max_lag, pred.shape[-1] - 2)
        coefficients = torch.stack([
            self.concordance(*lagged_pair(pred, label, lag))
            for lag in range(-max_lag, max_lag + 1)])                 # (lags, B)
        best = coefficients.argmax(dim=0, keepdim=True).detach()
        return 1.0 - coefficients.gather(0, best)[0]
