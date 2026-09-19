"""The loss components a config may name, and the check of its ``LOSS`` block.

Every component is one ``nn.Module`` in its own file whose ``forward`` maps a
``(B, T)`` prediction and a ``(B, T)`` label to a ``(B,)`` per-sample loss.
That per-sample reduction is the whole contract: it is what lets
``PerSignalLoss`` mask every component the same way. A new component is a file
and a line here, not a redesign.
"""

from functools import partial

from neural_methods.loss.ccc import CCC
from neural_methods.loss.mean_l1 import MeanL1
from neural_methods.loss.mse import MSE
from neural_methods.loss.neg_pearson import NegPearson
from neural_methods.loss.soft_peak_l1 import SoftPeakL1
from src.signal_transforms import canonical_signal, validate_traces

#: Component name (the config's key, lower-cased) -> what builds its module.
COMPONENTS = {
    'ccc': CCC,
    'mean': MeanL1,
    'max': partial(SoftPeakL1, 'max'),
    'min': partial(SoftPeakL1, 'min'),
    'negpearson': NegPearson,
    'mse': MSE,
}


def normalise_loss_weights(traces, weights) -> dict:
    """``{trace: {COMPONENT: weight}}`` (YAML spelling) -> ``{signal: {component: float}}``.

    Exactly one entry per trace, no more and no fewer; every component known;
    every weight positive (a component that should not count is left out, not
    zeroed). Returned in ``traces`` order with lower-case component keys.
    """
    traces = validate_traces(traces)
    if not isinstance(weights, dict):
        raise ValueError(
            f"LOSS must be a mapping of trace to {{component: weight}}, got "
            f"{weights!r}")
    resolved = {}
    for name, entry in weights.items():
        signal = canonical_signal(name)
        if not isinstance(entry, dict) or not entry:
            raise ValueError(
                f"LOSS.{name} must be a non-empty mapping of component to "
                f"weight, components {sorted(c.upper() for c in COMPONENTS)}; "
                f"got {entry!r}")
        out = {}
        for component, weight in entry.items():
            key = str(component).lower()
            if key not in COMPONENTS:
                raise ValueError(
                    f"LOSS.{name}: unknown component {component!r}; known "
                    f"{sorted(c.upper() for c in COMPONENTS)}")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                    or weight <= 0:
                raise ValueError(
                    f"LOSS.{name}.{component} must be a positive number, got "
                    f"{weight!r}")
            out[key] = float(weight)
        resolved[signal] = out
    missing = [t for t in traces if t not in resolved]
    extra = [s for s in resolved if s not in traces]
    if missing or extra:
        raise ValueError(
            f"LOSS must name exactly the TRACES {traces}; missing {missing}, "
            f"not in TRACES {extra}")
    return {t: resolved[t] for t in traces}
