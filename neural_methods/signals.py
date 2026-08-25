"""Canonical physiological signal and camera-channel registries.

Single source of truth for the multi-signal pipeline (see
docs/superpowers/specs/2026-08-25-multisignal-pipeline-design.md).
Signal names cover exactly what this repo's dataloaders provide.
"""
import numpy as np

CHANNELS = ('R', 'G', 'B', 'I', 'D')

SIGNALS = {
    'PPG':  {'norm': (-3.0, 3.0)},       # a.k.a. BVP; standardized units
    'ECG':  {'norm': (-1500.0, 1500.0)},
    'ABP':  {'norm': (0.0, 200.0)},
    'CVP':  {'norm': (-20.0, 30.0)},
    'RESP': {'norm': (0.0, 10.0)},       # BP4D Resp_Volts scale; override per dataset
    'EDA':  {'norm': (0.0, 40.0)},       # microsiemens; override per dataset
    'SPO2': {'norm': (0.0, 100.0)},
}

EVAL_ONLY = ('HR',)

_ALIASES = {'BVP': 'PPG', 'PULSE': 'PPG'}


def canonical_signal(name):
    """Map any loader-side name to the canonical vocabulary (KeyError if unknown)."""
    up = str(name).upper()
    up = _ALIASES.get(up, up)
    if up in SIGNALS or up in EVAL_ONLY:
        return up
    raise KeyError(
        f"Unknown signal {name!r}; known: {sorted(SIGNALS)} + eval-only {list(EVAL_ONLY)}")


def validate_traces(traces):
    """Canonicalize a config TRACES list; reject empty, unknown, or eval-only."""
    if not traces:
        raise ValueError("TRACES must name at least one signal")
    out = []
    for t in traces:
        c = canonical_signal(t)
        if c in EVAL_ONLY:
            raise ValueError(f"{c} is eval-only and cannot be a training trace")
        out.append(c)
    return out


def validate_channels(channels):
    """Validate a config CHANNELS list against the canonical slots."""
    if not channels:
        raise ValueError("CHANNELS must name at least one channel")
    bad = [c for c in channels if c not in CHANNELS]
    if bad:
        raise ValueError(f"Unknown channels {bad}; known: {list(CHANNELS)}")
    return list(channels)


def norm_range(sig, overrides=None):
    if overrides and sig in overrides:
        lo, hi = overrides[sig]
    else:
        lo, hi = SIGNALS[sig]['norm']
    return float(lo), float(hi)


def normalize_signal(x, sig, overrides=None):
    """Clip to the signal's range then min-max to [-1, 1]."""
    lo, hi = norm_range(sig, overrides)
    clipped = np.clip(x, lo, hi)
    return (clipped - lo) / (hi - lo) * 2.0 - 1.0


def denormalize_signal(x, sig, overrides=None):
    lo, hi = norm_range(sig, overrides)
    return (np.asarray(x) + 1.0) / 2.0 * (hi - lo) + lo


def resolve_channels(config_data):
    """Global PREPROCESS.CHANNELS, falling back to legacy NECKFLIX.CHANNELS."""
    chs = list(getattr(config_data.PREPROCESS, 'CHANNELS', []) or [])
    if not chs:
        chs = list(config_data.PREPROCESS.NECKFLIX.CHANNELS)
    return validate_channels(chs)


def resolve_traces(config_data):
    """Global PREPROCESS.TRACES, falling back to legacy NECKFLIX.TRACES."""
    trs = list(getattr(config_data.PREPROCESS, 'TRACES', []) or [])
    if not trs:
        trs = list(config_data.PREPROCESS.NECKFLIX.TRACES)
    return validate_traces(trs)


def signal_norm_overrides(config_data):
    """Read PREPROCESS.SIGNAL_NORMS (if configured) into {sig: (lo, hi)}."""
    out = {}
    norms = getattr(config_data.PREPROCESS, 'SIGNAL_NORMS', None)
    if norms is not None:
        for sig in SIGNALS:
            val = getattr(norms, sig, None)
            if val:
                out[sig] = (float(val[0]), float(val[1]))
    return out
