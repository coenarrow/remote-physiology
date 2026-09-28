"""The per-signal composite loss for the multi-signal batch-dict contract.

Two things vary per signal, and they vary together (migration contract §3):

* an **absolute-class** signal (ABP, CVP) arrives in physical units and its
  *level* is part of the prediction, so it is scored with CCC plus L1 terms on
  the window mean and on the soft systolic/diastolic peaks — all in mmHg;
* a **shape-class** signal (PPG, ECG, RESP) arrives per-window z-scored and
  only its waveform means anything, so it is scored with negpearson.

CCC searches a lag range before scoring, because the label is measured at a
different site from the one the camera sees and the transit delay between
them is not the model's error (``neural_methods.loss.ccc``).

So the loss is stated per trace, outright: which components, at what weight
(``INTERFACE.LOSS``, ``{ABP: {CCC: 1.0, MEAN: 0.05, ...}}``). There are no
presets and no class-implied defaults — the weights *are* the loss. That is
also where the per-signal scale factors live — raw ABP error is O(10 mmHg),
CVP O(1 mmHg), and a CCC term is O(1) in any units, so an unweighted sum would
let ABP own every gradient. There are deliberately no global or dataset-wide
normalisation constants: the model predicts physical units off an
activation-free readout, and the weights are the one place the units are
reconciled.

This file only accumulates. Each component is one module in its own file
(``neural_methods.loss.registry`` lists them) and reduces **per sample** to
``(B,)``, which is what lets the masking compose here, once: each is averaged
over the batch with the denominator clamped to >= 1, so a signal no window in
the batch carries contributes exactly 0 — never NaN, never a sentinel.

Contract splits the two halves: :class:`PerSignalLoss` produces the
*unweighted* components, and :func:`weight_losses` applies the config weights
and reduces them to the scalar to backpropagate — the mean over modules, as it
has always been over traces. Keeping them apart is what lets a run plot a
component's raw magnitude against its weighted contribution, which is how a
drowned or dominating term is spotted.
"""

import torch
import torch.nn as nn

from neural_methods.loss.registry import COMPONENTS, normalise_loss_weights
from src.signal_transforms import validate_traces


class PerSignalLoss(nn.Module):
    """Masked composite loss components per signal — contract v2's raw_losses.

    ``forward`` returns ``{signal: {component: () tensor}}``: every masked
    component value, **unweighted** and graph-attached, keyed by signal. Which
    term dominates is the first question debugging a multi-signal run raises,
    and a per-signal breakdown is what answers whether one signal is drowning
    the others — so the components, not a scalar, are the return value.

    ``weights`` is the interface's ``LOSS`` block; it decides *which*
    components are computed. Applying the weights, and producing the single
    scalar to backpropagate, is :func:`weight_losses`'s job.
    """

    def __init__(self, traces, weights, fs: float):
        """``fs`` is the interface's frame rate: every window in a batch sits
        on that one time base, and it is what lets a component state a bound
        in seconds."""
        super().__init__()
        self.traces = validate_traces(traces)
        self.weights = normalise_loss_weights(self.traces, weights)
        # The components carry no state, so the signals share one of each.
        named = {c for components in self.weights.values() for c in components}
        self.components = nn.ModuleDict(
            {name: build(fs) for name, build in COMPONENTS.items() if name in named})

    def forward(self, preds, labels, label_mask):
        """Unweighted masked components per signal — contract v2's raw_losses.

        Reads    : preds, labels, label_mask (all keyed by signal)
        Returns  : {signal: {component: () tensor}}, graph-attached.
        Weighting is the trainer's job — see :func:`weight_losses`.
        """
        raw = {}
        for signal in self.traces:
            pred, label = preds[signal], labels[signal]
            mask = label_mask[signal].to(pred.dtype)                  # (B,)
            # Clamped denominator: a signal absent from every window in the
            # batch contributes exactly 0 instead of 0/0.
            denominator = mask.sum().clamp(min=1.0)
            raw[signal] = {
                component: (self.components[component](pred, label) * mask).sum()
                           / denominator
                for component in self.weights[signal]
            }
        return raw

    def extra_repr(self):
        return "\n".join(
            f"{signal}: " + ", ".join(f"{c}={w:g}" for c, w in weights.items())
            for signal, weights in self.weights.items())


def weight_losses(raw, weights):
    """Apply config weights to a model's ``raw_losses`` dict.

    Reads    : ``raw`` = {module: {component: () tensor}} (unweighted,
               graph-attached), ``weights`` = {module: {component: float}};
               every component in ``raw`` must have a weight — the callers
               compute only what the config names, so a missing weight is a
               programming error, not a default.
    Returns  : ``(total, weighted)``. ``total`` is the scalar to
               backpropagate — the mean over modules of each module's weighted
               component sum, which is exactly the old mean-over-signals when
               the modules are the signals. ``weighted`` mirrors ``raw`` as
               detached floats, plus a ``'total'`` per module, for logging.

    A module whose spec zeroed every component still contributes its zero to
    the mean, so the denominator is the module count either way — dropping it
    would silently rescale every other module's gradient.
    """
    zero = next((torch.zeros_like(value) for components in raw.values()
                 for value in components.values()), torch.zeros(()))
    module_totals, weighted = [], {}
    for module, components in raw.items():
        module_weights = weights[module]
        module_total, entries = zero, {}
        for component, value in components.items():
            term = module_weights[component] * value
            entries[component] = float(term.detach())
            module_total = module_total + term
        entries['total'] = float(module_total.detach())
        weighted[module] = entries
        module_totals.append(module_total)
    return torch.stack(module_totals).mean(), weighted
