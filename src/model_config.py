"""The model config file: the one YAML a run is made of, section by section.

A run's ``--config`` file has exactly three sections, and this module is
their schema in the file's order:

* ``MODEL`` — ``NAME``, the architecture, plus the switches an experiment may
  flip for it. Layer sizes are not config: an architecture is defined once,
  in its module, at its published values, and every width is derived from
  the interface (first layer from ``CHANNELS``, one prediction per
  entry of ``TRACES``). Most architectures have no switch, so their section
  is ``NAME`` alone.
* ``INTERFACE`` — the model's demand on the data pipeline: what every sample
  looks like when it reaches the model (rate, window, channels, traces,
  resolution) and the loss per trace. Preprocessing is not in it: frames
  reach every backbone raw and each normalises its own input, and a label
  is z-scored or left raw by which signal it is
  (``src.signal_transforms.label_mode``).
* ``TRAIN`` — the paper's recipe: optimiser, rate, decay, schedule,
  precision. Names are admitted only when the trainer implements them.

Every key of every section is required (``TRAIN.MODEL_FILE_NAME`` is the
one exception) and unknown keys are refused, so a run's config is exactly
what its file says. How long and how wide a run is (epochs, batch size) and
what machine it uses (workers, GPU or not) are not in the file at all: they
are the launcher's flags, :class:`RunSettings`.

The order of operations the interface describes, for every store of every
dataset:

1. resample the store to ``FS`` (decimate, or interpolate if opted in);
2. cut a window of ``window_frames`` frames;
3. normalise the traces the store *has*, by signal;
4. only then pad: a channel or trace in ``CHANNELS`` / ``TRACES`` the store
   lacks becomes zeros with a False mask.

Padding after normalising is what keeps a padded trace exactly zero — a
zero-filled trace pushed through zscore would not be.

Each section has a ``parse_*`` taking the raw mapping and a ``where`` naming
the source in errors; ``load_config`` reads a file through all three, and
``src.experiment.rebuild`` feeds a run's compiled ``config.yaml`` back
through the same three, so a rebuilt run is checked exactly as a loaded one.
Dataset config files are a different kind of file with their own schema,
``src.dataset_config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from neural_methods.loss.registry import COMPONENTS, normalise_loss_weights
from src.config import ConfigError, build, load_yaml
from src.signal_transforms import is_cardiac, validate_channels, validate_traces

#: The sections of a config file, and the compiled-config key each becomes.
CONFIG_SECTIONS = {"MODEL": "model", "INTERFACE": "interface", "TRAIN": "training"}


# ---------------------------------------------------------------------------
# MODEL: ``NAME`` plus the switches an experiment may flip
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """An architecture's section: ``NAME`` plus the switches an experiment may
    flip. ``REGULARISATION`` is the one optional key: the terms of the
    backbone's own regularisers that count and their weights, ``{}`` or
    absent for none (a term left out is off, never zeroed)."""
    NAME: str = ""
    REGULARISATION: dict = field(default_factory=dict)   # {TERM: weight > 0}

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        self.REGULARISATION = normalise_regularisation(self.REGULARISATION, where)


#: The one key of the MODEL section a file may leave out.
OPTIONAL_MODEL_KEYS = ("REGULARISATION",)


def normalise_regularisation(weights, where: str) -> dict:
    """``{TERM: weight}`` (YAML spelling) -> ``{term: float}``, lower-case keys,
    every weight a positive number. Which terms exist is the backbone's to
    say; ``src.models`` checks the names when it builds the model. A
    regulariser name may not be a loss component name, because the trainer
    merges both into one per-trace dict and a collision would silently
    overwrite the loss component's tensor."""
    if weights is None:
        weights = {}
    if not isinstance(weights, dict):
        raise ConfigError(
            f"{where}: REGULARISATION must be a mapping of term to weight, got "
            f"{weights!r}")
    out = {}
    for term, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) \
                or weight <= 0:
            raise ConfigError(
                f"{where}: REGULARISATION.{term} must be a positive number "
                f"(a term that should not count is left out, not zeroed), got "
                f"{weight!r}")
        key = str(term).lower()
        if key in COMPONENTS:
            raise ConfigError(
                f"{where}: REGULARISATION.{term} is a loss component name; a "
                f"regulariser must not share a name with "
                f"{sorted(c.upper() for c in COMPONENTS)}")
        out[key] = float(weight)
    return out


@dataclass
class TemporalShiftConfig(ModelConfig):
    """EfficientPhys and TS-CAN: the segment length the temporal shift shifts within."""
    FRAME_DEPTH: int = 0

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        super().validate(interface, where)
        if self.FRAME_DEPTH <= 0:
            raise ConfigError(
                f"{where}: FRAME_DEPTH must be positive, got {self.FRAME_DEPTH}")


@dataclass
class PhysMambaConfig(ModelConfig):
    """PhysMamba, PhysMamba2 and PhysMamba3: the Mamba layers' learning rate as
    a multiple of the recipe's; 1 is the paper."""
    MAMBA_LR_SCALE: float = 0.0

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        super().validate(interface, where)
        if self.MAMBA_LR_SCALE <= 0:
            raise ConfigError(
                f"{where}: MAMBA_LR_SCALE must be positive, got {self.MAMBA_LR_SCALE}")


@dataclass
class FactorizePhysConfig(ModelConfig):
    FSAM: bool = True             # run the factorized attention module. For ablation testing


@dataclass
class CardioConvConfig(ModelConfig):
    """CardioConv's ablations; every one is ``True`` in the model as designed."""
    PULSATILITY_MASKER: bool = True   # learn the pulsatility mask; false is a mask of ones
    PHASE_MASKER: bool = True         # learn the per-trace masks; false is masks of ones
    TRACE_INDEPENDENT: bool = True    # mix each trace from its own candidates only
    SUPERVISE_MASKS: bool = True      # the statistics head's pooled power trains the masks

    def validate(self, interface: InterfaceConfig, where: str) -> None:
        super().validate(interface, where)
        other = [t for t in interface.TRACES if not is_cardiac(t)]
        if other:
            raise ConfigError(
                f"{where}: CardioConv reads the clip at the harmonics of the heart "
                f"rate, so it predicts cardiac traces only; TRACES has {other}")


#: ``NAME`` -> the dataclass its section is parsed into. One line per
#: architecture; its builder is the matching line of ``src.models.MODEL_BUILDERS``.
MODEL_CONFIGS = {
    "CardioConv": CardioConvConfig,
    "DeepPhys": ModelConfig,
    "EfficientPhys": TemporalShiftConfig,
    "FactorizePhys": FactorizePhysConfig,
    "PhysFormer": ModelConfig,
    "PhysMamba": PhysMambaConfig,
    "PhysMamba2": PhysMambaConfig,
    "PhysMamba3": PhysMambaConfig,
    "PhysNet": ModelConfig,
    "RhythmFormer": ModelConfig,
    "TSCAN": TemporalShiftConfig,
    "iBVPNet": ModelConfig,
}


def parse_model_config(mapping: dict, interface: InterfaceConfig, where: str):
    """One ``MODEL`` mapping, typed by its ``NAME`` and checked against
    ``interface``."""
    if not isinstance(mapping, dict) or not mapping:
        raise ConfigError(f"{where}: the model section must be a non-empty mapping")
    arch = mapping.get("NAME")
    if arch not in MODEL_CONFIGS:
        raise ConfigError(
            f"{where}: NAME must be one of {sorted(MODEL_CONFIGS)}, got {arch!r}")
    cfg = build(MODEL_CONFIGS[arch], mapping, where, optional=OPTIONAL_MODEL_KEYS)
    cfg.validate(interface, where)
    return cfg


# ---------------------------------------------------------------------------
# INTERFACE: the model's demand on the data pipeline
# ---------------------------------------------------------------------------
UPSAMPLING_MODES = ("refuse", "interpolate")


@dataclass
class ResizeConfig:
    H: int = 0
    W: int = 0


@dataclass
class InterfaceConfig:
    FS: float = 0.0
    UPSAMPLING: str = ""
    WINDOW_SECONDS: float = 0.0
    WINDOW_STRIDE: float = 0.0
    CHANNELS: list = field(default_factory=list)
    TRACES: list = field(default_factory=list)
    RESIZE: ResizeConfig = field(default_factory=ResizeConfig)
    LOSS: dict = field(default_factory=dict)      # {trace: {component: weight}}

    # Derived, never written in YAML.
    @property
    def window_frames(self) -> int:
        return _frames(self.WINDOW_SECONDS, self.FS)

    @property
    def stride_frames(self) -> int:
        return _frames(self.WINDOW_STRIDE, self.FS)

    @property
    def resizes(self) -> bool:
        return self.RESIZE.H > 0


def _frames(seconds: float, fs: float) -> int:
    return int(round(seconds * fs))


def validate_interface(cfg: InterfaceConfig, where: str) -> InterfaceConfig:
    """Every rule the interface carries, applied in place; returns ``cfg``."""
    if cfg.FS <= 0:
        raise ConfigError(f"{where}: FS must be a positive frame rate, got {cfg.FS}")
    if cfg.UPSAMPLING not in UPSAMPLING_MODES:
        raise ConfigError(
            f"{where}: UPSAMPLING must be one of {list(UPSAMPLING_MODES)}, "
            f"got {cfg.UPSAMPLING!r}")
    if cfg.WINDOW_SECONDS <= 0:
        raise ConfigError(
            f"{where}: WINDOW_SECONDS must be a positive duration, got "
            f"{cfg.WINDOW_SECONDS}")
    if cfg.stride_frames < 1:
        raise ConfigError(
            f"{where}: WINDOW_STRIDE must be at least one frame "
            f"(1 / FS = {1 / cfg.FS:.4f} s), got {cfg.WINDOW_STRIDE}")

    try:
        cfg.CHANNELS = validate_channels(cfg.CHANNELS)
        cfg.TRACES = validate_traces(cfg.TRACES)
    except ValueError as err:
        raise ConfigError(f"{where}: {err}") from err
    if len(set(cfg.CHANNELS)) != len(cfg.CHANNELS):
        raise ConfigError(f"{where}: CHANNELS repeats a channel: {cfg.CHANNELS}")
    if len(set(cfg.TRACES)) != len(cfg.TRACES):
        raise ConfigError(f"{where}: TRACES repeats a signal: {cfg.TRACES}")

    h, w = cfg.RESIZE.H, cfg.RESIZE.W
    if h < 0 or w < 0 or (h == 0) != (w == 0):
        raise ConfigError(
            f"{where}: RESIZE must be {{H: >0, W: >0}} or {{H: 0, W: 0}} for no "
            f"resize, got {{H: {h}, W: {w}}}")

    # The loss is stated outright per trace: which components, at what
    # weight. No presets — the weights are the loss, and they are also where
    # the per-signal scale factors live. The rules are the loss module's.
    try:
        cfg.LOSS = normalise_loss_weights(cfg.TRACES, cfg.LOSS)
    except (ValueError, KeyError) as err:
        raise ConfigError(f"{where}: {err}") from err
    return cfg


def parse_interface(mapping: dict, where: str) -> InterfaceConfig:
    """One ``INTERFACE`` mapping, typed and every rule checked."""
    return validate_interface(build(InterfaceConfig, mapping, where), where)


# ---------------------------------------------------------------------------
# TRAIN: how the train set is optimised
# ---------------------------------------------------------------------------
#: The names the trainer implements (``src.trainer.OPTIMIZERS`` / ``SCHEDULERS``
#: / ``PRECISION_DTYPES``); adding one is a line there and a name here.
OPTIMIZERS = ("Adam", "AdamW")
SCHEDULERS = ("OneCycle", "Constant")
PRECISIONS = ("float32", "bfloat16", "float16")


@dataclass
class TrainingConfig:
    OPTIMIZER: str = ""
    LR: float = 0.0
    WEIGHT_DECAY: float = -1.0
    SCHEDULER: str = ""
    PRECISION: str = ""
    MODEL_FILE_NAME: str = ""     # optional; "" = derive the run name from the run


def _one_of(value, allowed, key: str, where: str) -> None:
    if value not in allowed:
        raise ConfigError(
            f"{where}: {key} must be one of {list(allowed)}, got {value!r}")


def validate_training(cfg: TrainingConfig, where: str) -> TrainingConfig:
    """Every rule the recipe carries, applied in place; returns ``cfg``."""
    if cfg.LR <= 0:
        raise ConfigError(f"{where}: LR must be positive, got {cfg.LR}")
    if cfg.WEIGHT_DECAY < 0:
        raise ConfigError(
            f"{where}: WEIGHT_DECAY must be non-negative, got {cfg.WEIGHT_DECAY}")
    _one_of(cfg.OPTIMIZER, OPTIMIZERS, "OPTIMIZER", where)
    _one_of(cfg.SCHEDULER, SCHEDULERS, "SCHEDULER", where)
    _one_of(cfg.PRECISION, PRECISIONS, "PRECISION", where)
    # A reduced precision on a CPU run (no accelerator, or --no-gpu) drops
    # to float32 with a warning at runtime (src.distributed.init_runtime).
    return cfg


def parse_training(mapping: dict, where: str) -> TrainingConfig:
    """One ``TRAIN`` mapping, typed and every rule checked."""
    cfg = build(TrainingConfig, mapping, where, optional=("MODEL_FILE_NAME",))
    return validate_training(cfg, where)


# ---------------------------------------------------------------------------
# The launcher's flags: not in the file
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSettings:
    """How long, how wide and on what a run trains: the launcher's flags,
    not the paper's recipe. ``batch_size`` is per process, train and test
    alike; ``gpu`` False is ``--no-gpu``, CPU even where a GPU exists."""

    epochs: int = 1
    batch_size: int = 4
    num_workers: int = 8
    gpu: bool = True


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------
def load_config(path) -> tuple:
    """``(interface, model, training)`` from one config file: its
    ``INTERFACE``, ``MODEL`` and ``TRAIN`` sections, each typed and checked,
    with the model checked against the interface. Every section is required
    and nothing else is admitted."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"No config at {path}")
    raw = load_yaml(str(path))
    missing = sorted(key for key in CONFIG_SECTIONS if key not in raw)
    unknown = sorted(str(key) for key in raw if key not in CONFIG_SECTIONS)
    if missing or unknown:
        raise ConfigError(
            f"{path.name}: a config has exactly the sections "
            f"{sorted(CONFIG_SECTIONS)}; missing {missing}, unknown {unknown}")
    interface = parse_interface(raw["INTERFACE"], f"{path.name}: INTERFACE")
    model = parse_model_config(raw["MODEL"], interface, f"{path.name}: MODEL")
    training = parse_training(raw["TRAIN"], f"{path.name}: TRAIN")
    return interface, model, training
