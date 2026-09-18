"""The model stage: build the model a parsed ``MODEL`` section describes.

The section's schema — ``NAME`` plus the switches an experiment may flip —
is ``src.model_config``; this module is what turns a parsed one into a
network. Layer sizes are not config: an architecture is defined once, in its
module, at its published values. Every width is derived from the interface
(first layer from ``CHANNELS``, one copy of the network per entry of
``TRACES``), so nothing is said in two places. Input preprocessing is not a
switch either: every backbone takes the raw clip and normalises it itself
(``neural_methods.model.modules``).

The built model is a :class:`MultiTraceModel`: one complete copy of the
architecture per trace, speaking the batch dict. A model is a function from
frames to predictions and nothing else — the loss is the trainer's.
"""

import torch
import torch.nn as nn

from src.config import ConfigError
from neural_methods.model import (DeepPhys as deepphys, EfficientPhys as efficientphys,
    PhysFormer as physformer, PhysMamba as physmamba, PhysNet as physnet,
    RhythmFormer as rhythmformer, TS_CAN as tscan, iBVPNet as ibvpnet,
)
# The same idiom; FactorizePhys is a package, so its module needs its own line.
from neural_methods.model.FactorizePhys import FactorizePhys as factorizephys
from neural_methods.model.shared import min_frame_message
from src.model_config import (
    FactorizePhysConfig, InterfaceConfig, ModelConfig, TemporalShiftConfig,
)


# ---------------------------------------------------------------------------
# What a builder may demand of the interface
# ---------------------------------------------------------------------------
def _require_min_frame(interface: InterfaceConfig, name: str, minimum: int) -> None:
    """A backbone whose stem pools spatially needs a frame it leaves something of.
    Only checkable here when the interface resizes; otherwise the backbone
    refuses at forward time with the same sentence, which both sides take from
    ``neural_methods.model.shared.min_frame_message``."""
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


def _require_regularisers(cfg: ModelConfig, copy: nn.Module) -> None:
    """The terms the config weights must be terms the backbone computes.

    A backbone declares them as a class attribute ``REGULARISERS`` and returns
    them from ``regularisers()`` after each forward; a backbone with neither
    has none, so any non-empty mapping is refused by name.
    """
    known = tuple(getattr(type(copy), "REGULARISERS", ()))
    unknown = sorted(t for t in cfg.REGULARISATION if t not in known)
    if unknown:
        have = f"has {list(known)}" if known else "has no regularisers"
        raise ConfigError(
            f"{cfg.NAME}: REGULARISATION names {[u.upper() for u in unknown]} "
            f"but the backbone {have}")


# ---------------------------------------------------------------------------
# The multi-trace model
# ---------------------------------------------------------------------------
class MultiTraceModel(nn.Module):
    """S complete copies of a single-trace architecture, dict in, dict out.

    Each trace gets its own untouched copy of the published architecture, its
    first layer widened to the interface's channels. Signals such as ABP and
    CVP come from different regions of the frame, so they share no trunk; the
    cost is parameters, by design.

    A backbone is any ``nn.Module`` with ``forward(x)`` and
    ``output_layers()`` that takes a raw clip ``(B, C_in, T, H, W)`` and
    returns ``(B, 1, T)``; a backbone with regularisers also declares
    ``REGULARISERS`` and returns them from ``regularisers()`` after each
    forward; whatever preprocessing the published network was fed, the
    backbone applies itself. ``C_in = len(channels)``: the interface's
    channels stacked in order. Channel and trace order is owned here, never
    inferred from dict iteration.

    ``forward(batch)`` returns the same dict with ``predictions`` and
    ``regularisers`` added, ``{trace: (B, T)}`` and ``{trace: {term: () tensor}}``.
    Nothing is dropped in transit and nothing else is computed: the loss
    belongs to the trainer.
    """

    def __init__(self, make_copy, channels, traces):
        super().__init__()
        self.channels = tuple(channels)
        self.traces = tuple(traces)
        self.copies = nn.ModuleDict({trace: make_copy() for trace in self.traces})

    @property
    def in_channels(self) -> int:
        return len(self.channels)

    def output_layers(self):
        """Each copy's activation-free readout, in traces order."""
        return [layer for trace in self.traces
                for layer in self.copies[trace].output_layers()]

    def prepare_frames(self, batch) -> torch.Tensor:
        """``batch['frames'][ch]`` ``(B, T, H, W)`` -> ``(B, C_in, T, H, W)``."""
        frames = batch["frames"]
        return torch.stack([frames[ch] for ch in self.channels], dim=1)

    def forward_video(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W)`` -> ``(B, S, T)``, traces order."""
        return torch.cat([self.copies[trace](video) for trace in self.traces], dim=1)

    def forward(self, batch: dict) -> dict:
        out = self.forward_video(self.prepare_frames(batch))
        predictions = {trace: out[:, i] for i, trace in enumerate(self.traces)}
        return {**batch, "predictions": predictions,
                "regularisers": self.collect_regularisers()}

    def collect_regularisers(self) -> dict:
        """``{trace: {term: () tensor}}``: every term each copy declares in
        ``REGULARISERS``, read from its ``regularisers()`` after the forward.
        Which ones count is the trainer's, from the model config. ``{}`` per
        trace for a backbone that declares none. A backbone that declares a
        term and does not return it is refused here, at the first forward."""
        collected = {}
        for trace, copy in self.copies.items():
            declared = tuple(getattr(type(copy), "REGULARISERS", ()))
            if not declared:
                collected[trace] = {}
                continue
            returned = copy.regularisers() if hasattr(copy, "regularisers") else {}
            missing = [t for t in declared if t not in returned]
            if missing:
                raise RuntimeError(
                    f"{type(copy).__name__} declares REGULARISERS {list(declared)} "
                    f"but regularisers() did not return {missing}")
            collected[trace] = {t: returned[t] for t in declared}
        return collected

    def extra_repr(self) -> str:
        return f"channels={list(self.channels)}, traces={list(self.traces)}"


# ---------------------------------------------------------------------------
# Builders: (model config, interface) -> MultiTraceModel
# ---------------------------------------------------------------------------
def _multi_trace(make_copy, interface: InterfaceConfig, cfg: ModelConfig) -> MultiTraceModel:
    model = MultiTraceModel(make_copy=make_copy, channels=interface.CHANNELS,
                            traces=interface.TRACES)
    _require_regularisers(cfg, next(iter(model.copies.values())))
    return model


def _build_deepphys(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    size = _require_frame_size(interface, "DeepPhys")
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: deepphys.DeepPhys(in_channels=width, img_size=size),
                        interface, cfg)


def _build_efficientphys(cfg: TemporalShiftConfig, interface: InterfaceConfig) -> MultiTraceModel:
    size = _require_frame_size(interface, "EfficientPhys")
    width = len(interface.CHANNELS)
    return _multi_trace(
        lambda: efficientphys.EfficientPhys(in_channels=width, img_size=size,
                                            frame_depth=cfg.FRAME_DEPTH),
        interface, cfg)


def _build_factorizephys(cfg: FactorizePhysConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "FactorizePhys", factorizephys.MIN_FRAME)
    width = len(interface.CHANNELS)
    return _multi_trace(
        lambda: factorizephys.FactorizePhys(in_channels=width, use_fsam=cfg.FSAM),
        interface, cfg)


def _build_physformer(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "PhysFormer", physformer.MIN_FRAME)
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: physformer.PhysFormer(in_channels=width), interface, cfg)


def _build_physmamba(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "PhysMamba", physmamba.MIN_FRAME)
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: physmamba.PhysMamba(in_channels=width), interface, cfg)


def _build_physnet(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "PhysNet", physnet.MIN_FRAME)
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: physnet.PhysNet(in_channels=width), interface, cfg)


def _build_rhythmformer(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "RhythmFormer", rhythmformer.MIN_FRAME)
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: rhythmformer.RhythmFormer(in_channels=width), interface, cfg)


def _build_tscan(cfg: TemporalShiftConfig, interface: InterfaceConfig) -> MultiTraceModel:
    size = _require_frame_size(interface, "TSCAN")
    width = len(interface.CHANNELS)
    return _multi_trace(
        lambda: tscan.TSCAN(in_channels=width, img_size=size,
                            frame_depth=cfg.FRAME_DEPTH),
        interface, cfg)


def _build_ibvpnet(cfg: ModelConfig, interface: InterfaceConfig) -> MultiTraceModel:
    _require_min_frame(interface, "iBVPNet", ibvpnet.MIN_FRAME)
    width = len(interface.CHANNELS)
    return _multi_trace(lambda: ibvpnet.iBVPNet(in_channels=width), interface, cfg)


#: ``NAME`` -> builder. One line per architecture, matching its line in
#: ``src.model_config.MODEL_CONFIGS``.
MODEL_BUILDERS = {
    "DeepPhys": _build_deepphys,
    "EfficientPhys": _build_efficientphys,
    "FactorizePhys": _build_factorizephys,
    "PhysFormer": _build_physformer,
    "PhysMamba": _build_physmamba,
    "PhysNet": _build_physnet,
    "RhythmFormer": _build_rhythmformer,
    "TSCAN": _build_tscan,
    "iBVPNet": _build_ibvpnet,
}


def build_model(cfg, interface: InterfaceConfig) -> MultiTraceModel:
    """The model a loaded config describes, every width taken from ``interface``."""
    return MODEL_BUILDERS[cfg.NAME](cfg, interface)
