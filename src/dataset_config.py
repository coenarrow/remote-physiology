"""The dataset config file: which stores of one cache participate.

A dataset is named on the command line (``--datasets neckflix pure``) and
each name resolves to ``configs/datasets/<name>.yaml``, which carries only
the cache location and include/exclude filters over the stores' root attrs.
It says nothing about channels, traces, windows or rates — that is the
model config file's ``INTERFACE`` section (``src.model_config``), a
different kind of file. Enumerating and filtering the stores a loaded
config names is ``src.datasets``.

Both keys are required (``FILTERS: {}`` admits every store) and unknown
keys are refused.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.config import ConfigError, build, load_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "datasets"


@dataclass
class DatasetConfig:
    CACHED_PATH: str = ""
    FILTERS: dict = field(default_factory=dict)   # {attr: {include, exclude}}


def normalise_filters(filters: dict, where: str) -> dict:
    """``{attr: {"include": [...], "exclude": [...]}}`` with values as strings.

    Every entry must be a mapping with exactly the keys ``include`` and
    ``exclude``, each a list (empty = no constraint). An attr may be nested
    (dotted path). Overlapping include/exclude is a config error.
    """
    out = {}
    for attr, spec in (filters or {}).items():
        if not isinstance(spec, dict) or set(spec) != {"include", "exclude"}:
            raise ConfigError(
                f"{where}: FILTERS.{attr} must be a mapping with exactly the "
                f"keys include and exclude (each a list, [] = no constraint), "
                f"got {spec!r}")
        for key in ("include", "exclude"):
            if spec[key] is not None and not isinstance(spec[key], list):
                raise ConfigError(
                    f"{where}: FILTERS.{attr}.{key} must be a list, got "
                    f"{spec[key]!r}")
        include = [str(v) for v in (spec["include"] or [])]
        exclude = [str(v) for v in (spec["exclude"] or [])]
        overlap = sorted(set(include) & set(exclude))
        if overlap:
            raise ConfigError(
                f"{where}: FILTERS.{attr} both includes and excludes {overlap}")
        out[str(attr)] = {"include": include, "exclude": exclude}
    return out


def resolve_dataset_configs(names: list[str]) -> dict[str, Path]:
    """``{name: path}`` for each dataset name, or a ValueError naming the gap.

    A name resolves to ``configs/datasets/<name>.yaml``; nothing else is
    searched. The name doubles as the label a test participant is qualified
    with later (``--test-participant-dataset``), so it has to be exact.
    """
    resolved = {}
    for name in names:
        path = DATASET_CONFIG_DIR / f"{name}.yaml"
        if not path.is_file():
            available = sorted(p.stem for p in DATASET_CONFIG_DIR.glob("*.yaml"))
            raise ValueError(
                f"No dataset config for {name!r}: expected {path}. "
                f"Create it, or pick one of {available}.")
        resolved[name] = path
    return resolved


def parse_dataset_config(mapping: dict, where: str) -> DatasetConfig:
    """One dataset mapping — a loaded file, or an entry of the ``datasets``
    section a run's compiled config carries — typed and checked. ``where``
    names the source in errors."""
    cfg = build(DatasetConfig, mapping, where)
    if not cfg.CACHED_PATH:
        raise ConfigError(f"{where}: CACHED_PATH must name the cache directory")
    cfg.FILTERS = normalise_filters(cfg.FILTERS, where)
    return cfg


def load_dataset_config(path: Path) -> DatasetConfig:
    """One dataset file, typed and checked."""
    return parse_dataset_config(load_yaml(str(path)), path.stem)


def load_dataset_configs(names: list[str]) -> dict[str, DatasetConfig]:
    """``{name: DatasetConfig}`` for the names given on the command line."""
    return {name: load_dataset_config(path)
            for name, path in resolve_dataset_configs(names).items()}
