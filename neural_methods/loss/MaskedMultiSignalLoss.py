"""Masked multi-signal loss over dict predictions/labels.

Per signal: per-sample base loss, masked mean over the batch (denominator
clamped to >= 1 so a fully-absent signal contributes exactly 0, never NaN);
then the plain mean over the configured trace list.
"""
import torch
import torch.nn as nn

from neural_methods.signals import validate_traces

_EPS = 1e-8


def _mse_per_sample(pred, label):
    return ((pred - label) ** 2).mean(dim=-1)


def _negpearson_per_sample(pred, label):
    p = pred - pred.mean(dim=-1, keepdim=True)
    l = label - label.mean(dim=-1, keepdim=True)
    num = (p * l).sum(dim=-1)
    den = torch.sqrt((p ** 2).sum(dim=-1) * (l ** 2).sum(dim=-1) + _EPS)
    return 1.0 - num / den


_BASES = {'mse': _mse_per_sample, 'negpearson': _negpearson_per_sample}


class MaskedMultiSignalLoss(nn.Module):
    def __init__(self, traces, base='mse'):
        super().__init__()
        self.traces = validate_traces(traces)
        if base not in _BASES:
            raise ValueError(f"Unknown base loss {base!r}; known: {sorted(_BASES)}")
        self.base = base
        self._fn = _BASES[base]

    def forward(self, preds, labels, label_mask):
        terms = []
        for sig in self.traces:
            per_sample = self._fn(preds[sig], labels[sig])          # (B,)
            m = label_mask[sig].to(per_sample.dtype)                # (B,)
            terms.append((per_sample * m).sum() / m.sum().clamp(min=1.0))
        return torch.stack(terms).mean()
