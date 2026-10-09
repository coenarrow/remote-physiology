"""The model stage: build the model a parsed ``MODEL`` section describes.

The section's schema — ``NAME`` plus the switches an experiment may flip —
is ``src.model_config``; this module is what turns a parsed one into a
network. Layer sizes are not config: an architecture is defined once, in its
module, at its published values. Every width is derived from the interface
(first layer from ``CHANNELS``, one prediction per entry of
``TRACES``), so nothing is said in two places. Input preprocessing is not a
switch either: every backbone takes the raw clip and normalises it itself
(``neural_methods.model._shared_modules``).

The built model is a :class:`MultiTraceModel`, speaking the batch dict: an
architecture designed to predict a single trace is copied, one complete copy
per trace (:class:`PerTraceCopies`); one designed from the start to predict
several is built once. A model is a function from frames to predictions and
nothing else — the loss is the trainer's.
"""

import torch
import torch.nn as nn

from src.config import ConfigError
from neural_methods.model.cardioconv import cardioconv
from neural_methods.model.cardiohydra import cardiohydra
from neural_methods.model.deepphys import deepphys
from neural_methods.model.efficientphys import efficientphys
from neural_methods.model.factorizephys import factorizephys
from neural_methods.model.ibvpnet import ibvpnet
from neural_methods.model.physformer import physformer
from neural_methods.model.physmamba import physmamba
from neural_methods.model.physmamba2 import physmamba2
from neural_methods.model.physmamba3 import physmamba3
from neural_methods.model.physnet import physnet
from neural_methods.model.rhythmformer import rhythmformer
from neural_methods.model.tscan import tscan
from neural_methods.model._shared_modules.utils import min_frame_message
from src.model_config import (
    MODEL_CONFIGS, SPECTRAL_RATE, CardioConvConfig, CardioHydraConfig, FactorizePhysConfig,
    InterfaceConfig, ModelConfig, PhysMambaConfig, TemporalShiftConfig,
)


# ---------------------------------------------------------------------------
# What a builder may demand of the interface
# ---------------------------------------------------------------------------
def _require_min_frame(interface: InterfaceConfig, name: str, minimum: int) -> None:
    """A backbone whose stem pools spatially needs a frame it leaves something of.
    Only checkable here when the interface resizes; otherwise the backbone
    refuses at forward time with the same sentence, which both sides take from
    ``neural_methods.model._shared_modules.utils.min_frame_message``."""
    if interface.resizes and min(interface.RESIZE.H, interface.RESIZE.W) < minimum:
        raise ConfigError(
            f"{min_frame_message(name, minimum)}; the interface RESIZE is "
            f"{{H: {interface.RESIZE.H}, W: {interface.RESIZE.W}}}")


def _require_frame_size(interface: InterfaceConfig, name: str) -> tuple:
    """``(H, W)`` for a backbone whose dense layer is sized from the frame.

    The dense width is derived per axis, so a non-square frame is fine; what
    cannot be derived is a frame size the interface does not state.
    """
    if not interface.resizes:
        raise ConfigError(
            f"{name} sizes its dense layer from the frame, so it needs an "
            f"interface RESIZE; this one does not resize")
    return (interface.RESIZE.H, interface.RESIZE.W)


def _require_regularisers(cfg: ModelConfig, backbone: nn.Module) -> None:
    """The terms the config weights must be terms the backbone computes.

    A backbone declares them as ``REGULARISERS`` and returns them from
    ``regularisers()`` after each forward; a backbone with neither has none,
    so any non-empty mapping is refused by name.
    """
    known = tuple(getattr(backbone, "REGULARISERS", ()))
    unknown = sorted(t for t in cfg.REGULARISATION if t not in known)
    if unknown:
        have = f"has {list(known)}" if known else "has no regularisers"
        raise ConfigError(
            f"{cfg.NAME}: REGULARISATION names {[u.upper() for u in unknown]} "
            f"but the backbone {have}")


# ---------------------------------------------------------------------------
# The multi-trace model
# ---------------------------------------------------------------------------
class PerTraceCopies(nn.Module):
    """S complete copies of a single-trace architecture, as one multi-trace backbone.

    Each trace gets its own untouched copy of the published architecture, its
    first layer widened to the interface's channels. Signals such as ABP and
    CVP come from different regions of the frame, and a network that was not
    designed to tell them apart shares no trunk between them; the cost is
    parameters, by design.

    A copy is any ``nn.Module`` with ``forward(x)`` and ``output_layers()``
    that takes a raw clip ``(B, C_in, T, H, W)`` and returns ``(B, 1, T)``; a
    copy with regularisers also declares ``REGULARISERS`` and returns them
    from ``regularisers()`` after each forward. Every copy is the same
    network, so what the architecture declares is read off the first.
    """

    def __init__(self, make_copy, traces):
        super().__init__()
        self.traces = tuple(traces)
        self.copies = nn.ModuleDict({trace: make_copy() for trace in self.traces})

    @property
    def _first(self) -> nn.Module:
        return self.copies[self.traces[0]]

    @property
    def architecture(self) -> str:
        return type(self._first).__name__

    @property
    def REGULARISERS(self) -> tuple:  # noqa: N802 - the backbone contract's name
        return tuple(getattr(self._first, "REGULARISERS", ()))

    @property
    def temporal_divisor(self):
        return getattr(self._first, "temporal_divisor", None)

    @property
    def temporal_length(self):
        return getattr(self._first, "temporal_length", None)

    def output_layers(self):
        """Each copy's activation-free readout, in traces order."""
        return [layer for trace in self.traces
                for layer in self.copies[trace].output_layers()]

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` -> ``(B, S, T)``, traces order."""
        return torch.cat([self.copies[trace](video) for trace in self.traces], dim=1)

    def regularisers(self) -> dict:
        """``{trace: {term: () tensor}}``, each copy's own values."""
        return {trace: copy.regularisers() if hasattr(copy, "regularisers") else {}
                for trace, copy in self.copies.items()}


class MultiTraceModel(nn.Module):
    """One multi-trace backbone, dict in, dict out.

    A backbone is any ``nn.Module`` that takes a raw clip
    ``(B, C_in, T, H, W)`` and returns ``(B, S, T)`` in traces order, with
    ``output_layers()`` giving one activation-free readout per trace in the
    same order; one with regularisers also declares ``REGULARISERS`` and
    returns ``{trace: {term: () tensor}}`` from ``regularisers()`` after each
    forward. A backbone that declares ``NEEDS_RATE_TARGET`` takes
    ``forward(video, target_rate=None, target_waveform=None)`` and is handed,
    in training mode only, each sample's heart rate as the dataset read it
    off the store's cardiac traces and the waveform it fused from them
    (``batch["heart_rate"]`` and ``batch["rate_waveform"]``,
    ``src.inputs.window_rate`` and ``window_waveform``); one that sets a
    ``sample_mask`` ``(B,)`` after its forward has that mask joined into
    every trace's ``label_mask``, so a sample it chose not to predict is
    out of the loss. Whatever preprocessing the published network was fed,
    the backbone applies itself. An architecture designed to predict several
    traces is such a backbone as it stands; one designed for a single trace
    becomes one through :class:`PerTraceCopies`. ``C_in = len(channels)``:
    the interface's channels stacked in order. Channel and trace order is
    owned here, never inferred from dict iteration.

    ``forward(batch)`` returns the same dict with ``predictions`` and
    ``regularisers`` added, ``{trace: (B, T)}`` and ``{trace: {term: () tensor}}``.
    Nothing is dropped in transit and nothing else is computed: the loss
    belongs to the trainer.
    """

    def __init__(self, backbone, channels, traces):
        super().__init__()
        self.channels = tuple(channels)
        self.traces = tuple(traces)
        self.backbone = backbone

    @property
    def in_channels(self) -> int:
        return len(self.channels)

    @property
    def architecture(self) -> str:
        """The architecture's class name, whether copied or built once."""
        return getattr(self.backbone, "architecture", type(self.backbone).__name__)

    def output_layers(self):
        """The backbone's activation-free readouts, one per trace, in traces order."""
        return list(self.backbone.output_layers())

    def prepare_frames(self, batch) -> torch.Tensor:
        """``batch['frames'][ch]`` ``(B, T, H, W)`` -> ``(B, C_in, T, H, W)``."""
        frames = batch["frames"]
        return torch.stack([frames[ch] for ch in self.channels], dim=1)

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` -> ``(B, S, T)``, traces order."""
        return self.backbone(video)

    def forward(self, batch: dict) -> dict:
        video = self.prepare_frames(batch)
        if self.training and getattr(self.backbone, "NEEDS_RATE_TARGET", False):
            out = self.backbone(video, target_rate=batch["heart_rate"].to(video.device),
                                target_waveform=batch["rate_waveform"].to(video.device))
        else:
            out = self.forward_video(video)
        predictions = {trace: out[:, i] for i, trace in enumerate(self.traces)}
        result = {**batch, "predictions": predictions,
                  "regularisers": self.collect_regularisers()}
        # A backbone that gates samples says which outputs count; the loss
        # already masks a trace per sample, so the gate joins that mask.
        sample_mask = getattr(self.backbone, "sample_mask", None)
        if sample_mask is not None:
            result["label_mask"] = {trace: mask & sample_mask
                                    for trace, mask in batch["label_mask"].items()}
        return result

    def collect_regularisers(self) -> dict:
        """``{trace: {term: () tensor}}``: every term the backbone declares in
        ``REGULARISERS``, read from its ``regularisers()`` after the forward.
        Which ones count is the trainer's, from the model config. ``{}`` per
        trace for a backbone that declares none. A backbone that declares a
        term and does not return it for a trace is refused here, at the first
        forward."""
        declared = tuple(getattr(self.backbone, "REGULARISERS", ()))
        if not declared:
            return {trace: {} for trace in self.traces}
        returned = self.backbone.regularisers() if hasattr(self.backbone, "regularisers") else {}
        collected = {}
        for trace in self.traces:
            terms = returned.get(trace, {})
            missing = [t for t in declared if t not in terms]
            if missing:
                raise RuntimeError(
                    f"{self.architecture} declares REGULARISERS {list(declared)} "
                    f"but regularisers() did not return {missing} for {trace}")
            collected[trace] = {t: terms[t] for t in declared}
        return collected

    def extra_repr(self) -> str:
        return f"channels={list(self.channels)}, traces={list(self.traces)}"


# ---------------------------------------------------------------------------
# Builders: (model config, interface) -> MultiTraceModel
# ---------------------------------------------------------------------------
def _wrap(backbone: nn.Module, interface: InterfaceConfig, cfg: ModelConfig) -> MultiTraceModel:
    """A multi-trace backbone, wrapped and checked against the config."""
    _require_regularisers(cfg, backbone)
    return MultiTraceModel(backbone=backbone, channels=interface.CHANNELS,
                           traces=interface.TRACES)


def _require_rate_tolerance(cfg: CardioHydraConfig, buffer_bpm: float) -> None:
    if not cfg.RATE_TOLERANCE_BPM < buffer_bpm:
        raise ConfigError(
            f"{cfg.NAME}: RATE_TOLERANCE_BPM must be inside the wavelet buffer of "
            f"{buffer_bpm:g} bpm either side of each harmonic, got "
            f"{cfg.RATE_TOLERANCE_BPM:g}")


# -- single-trace architectures: (model config, interface) -> one copy --------
def _make_deepphys(cfg: ModelConfig, interface: InterfaceConfig) -> nn.Module:
    size = _require_frame_size(interface, "DeepPhys")
    return deepphys.DeepPhys(in_channels=len(interface.CHANNELS), img_size=size)


def _make_efficientphys(cfg: TemporalShiftConfig, interface: InterfaceConfig) -> nn.Module:
    size = _require_frame_size(interface, "EfficientPhys")
    return efficientphys.EfficientPhys(in_channels=len(interface.CHANNELS), img_size=size,
                                       frame_depth=cfg.FRAME_DEPTH)


def _make_factorizephys(cfg: FactorizePhysConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "FactorizePhys", factorizephys.MIN_FRAME)
    return factorizephys.FactorizePhys(in_channels=len(interface.CHANNELS), use_fsam=cfg.FSAM)


def _make_physformer(cfg: ModelConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "PhysFormer", physformer.MIN_FRAME)
    return physformer.PhysFormer(in_channels=len(interface.CHANNELS))


def _make_physmamba(cfg: PhysMambaConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "PhysMamba", physmamba.MIN_FRAME)
    return physmamba.PhysMamba(in_channels=len(interface.CHANNELS),
                               mamba_lr_scale=cfg.MAMBA_LR_SCALE)


def _make_physmamba2(cfg: PhysMambaConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "PhysMamba2", physmamba2.MIN_FRAME)
    return physmamba2.PhysMamba2(in_channels=len(interface.CHANNELS),
                                 mamba_lr_scale=cfg.MAMBA_LR_SCALE)


def _make_physmamba3(cfg: PhysMambaConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "PhysMamba3", physmamba3.MIN_FRAME)
    return physmamba3.PhysMamba3(in_channels=len(interface.CHANNELS),
                                 mamba_lr_scale=cfg.MAMBA_LR_SCALE)


def _make_physnet(cfg: ModelConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "PhysNet", physnet.MIN_FRAME)
    return physnet.PhysNet(in_channels=len(interface.CHANNELS))


def _make_rhythmformer(cfg: ModelConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "RhythmFormer", rhythmformer.MIN_FRAME)
    return rhythmformer.RhythmFormer(in_channels=len(interface.CHANNELS))


def _make_tscan(cfg: TemporalShiftConfig, interface: InterfaceConfig) -> nn.Module:
    size = _require_frame_size(interface, "TSCAN")
    return tscan.TSCAN(in_channels=len(interface.CHANNELS), img_size=size,
                       frame_depth=cfg.FRAME_DEPTH)


def _make_ibvpnet(cfg: ModelConfig, interface: InterfaceConfig) -> nn.Module:
    _require_min_frame(interface, "iBVPNet", ibvpnet.MIN_FRAME)
    return ibvpnet.IBVPNet(in_channels=len(interface.CHANNELS))


#: ``NAME`` -> what makes one copy of a single-trace architecture. One line
#: per architecture, matching its line in ``src.model_config.MODEL_CONFIGS``.
SINGLE_TRACE = {
    "DeepPhys": _make_deepphys,
    "EfficientPhys": _make_efficientphys,
    "FactorizePhys": _make_factorizephys,
    "PhysFormer": _make_physformer,
    "PhysMamba": _make_physmamba,
    "PhysMamba2": _make_physmamba2,
    "PhysMamba3": _make_physmamba3,
    "PhysNet": _make_physnet,
    "RhythmFormer": _make_rhythmformer,
    "TSCAN": _make_tscan,
    "iBVPNet": _make_ibvpnet,
}


def _build_single_trace(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    """A single-trace architecture: one copy per trace."""
    make = SINGLE_TRACE[cfg.NAME]
    return _wrap(PerTraceCopies(lambda: make(cfg, interface), interface.TRACES), interface, cfg)


# -- multi-trace architectures ------------------------------------------------
def _rate_network(cfg: CardioHydraConfig, interface: InterfaceConfig) -> nn.Module | None:
    """One copy of the architecture ``RATE_ESTIMATOR`` names, at its paper's
    switches (its config's defaults), for the model to read the heart rate
    off; ``None`` for the model's own spectral estimator."""
    name = cfg.RATE_ESTIMATOR
    if name == SPECTRAL_RATE:
        return None
    return SINGLE_TRACE[name](MODEL_CONFIGS[name](NAME=name), interface)


def _build_cardioconv(cfg: CardioConvConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "CardioConv", cardioconv.MIN_FRAME)
    _require_rate_tolerance(cfg, cardioconv.BUFFER_BPM)
    backbone = cardioconv.CardioConv(
        in_channels=len(interface.CHANNELS), traces=interface.TRACES, fs=interface.FS,
        pulsatility_masker=cfg.PULSATILITY_MASKER, phase_masker=cfg.PHASE_MASKER,
        trace_independent=cfg.TRACE_INDEPENDENT, supervise_masks=cfg.SUPERVISE_MASKS,
        rate_tolerance_bpm=cfg.RATE_TOLERANCE_BPM, rate_network=_rate_network(cfg, interface))
    return _wrap(backbone, interface, cfg)


def _build_cardiohydra(cfg: CardioHydraConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "CardioHydra", cardiohydra.MIN_FRAME)
    _require_rate_tolerance(cfg, cardiohydra.BUFFER_BPM)
    backbone = cardiohydra.CardioHydra(
        in_channels=len(interface.CHANNELS), traces=interface.TRACES, fs=interface.FS,
        pulsatility_masker=cfg.PULSATILITY_MASKER, phase_masker=cfg.PHASE_MASKER,
        supervise_masks=cfg.SUPERVISE_MASKS, rate_tolerance_bpm=cfg.RATE_TOLERANCE_BPM,
        rate_network=_rate_network(cfg, interface))
    return _wrap(backbone, interface, cfg)


#: ``NAME`` -> builder: the multi-trace architectures' own, and one shared
#: builder for every line of ``SINGLE_TRACE``.
MODEL_BUILDERS = {
    "CardioConv": _build_cardioconv,
    "CardioHydra": _build_cardiohydra,
    **dict.fromkeys(SINGLE_TRACE, _build_single_trace),
}


def build_model(cfg, interface: InterfaceConfig) -> MultiTraceModel:
    """The model a loaded config describes, every width taken from ``interface``."""
    return MODEL_BUILDERS[cfg.NAME](cfg, interface)
