"""The dataset stage: enumerate a dataset's stores, and hold a participant out.

A loaded dataset config (``src.dataset_config``) names a cache and filters
over the stores' root attrs. This stage loads every store in the cache and
then drops what the filters exclude; it knows nothing about channels,
traces, windows or rates — that is the interface's business, and where
padding and masking happen.
"""

from dataclasses import dataclass
from pathlib import Path

import zarr

from src.dataset_config import DatasetConfig


# ---------------------------------------------------------------------------
# Stores: enumerate every store, then drop what the filters exclude
# ---------------------------------------------------------------------------
def _lookup(attrs: dict, dotted: str):
    """The value at a dotted path, or ``None`` when any step is absent."""
    node = attrs
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def admitted(attrs: dict, filters: dict) -> bool:
    """Include/exclude matching on (possibly nested) root attrs.

    A store lacking the attr, or carrying null, fails an include on it and
    passes an exclude: absence is never a reason to keep under include, and
    never a reason to drop under exclude.
    """
    for attr, spec in filters.items():
        value = _lookup(attrs, attr)
        text = None if value is None else str(value)
        if spec["include"] and text not in spec["include"]:
            return False
        if spec["exclude"] and text in spec["exclude"]:
            return False
    return True


def scan_stores(cfg: DatasetConfig) -> tuple[dict[Path, dict], dict[Path, dict]]:
    """``(kept, dropped)`` as ``{store path: root attrs}``. Metadata only."""
    root = Path(cfg.CACHED_PATH)
    if not root.is_dir():
        raise FileNotFoundError(f"CACHED_PATH {root} is not a directory")
    kept, dropped = {}, {}
    for store in sorted(root.glob("*.zarr")):
        attrs = dict(zarr.open_group(str(store), mode="r").attrs)
        (kept if admitted(attrs, cfg.FILTERS) else dropped)[store] = attrs
    return kept, dropped


def load_stores(configs: dict[str, DatasetConfig]) -> dict[str, dict[Path, dict]]:
    """``{dataset name: {store path: root attrs}}`` for every admitted store."""
    return {name: scan_stores(cfg)[0] for name, cfg in configs.items()}


# ---------------------------------------------------------------------------
# Test participant: held out of its own dataset before anything is concatenated
# ---------------------------------------------------------------------------
@dataclass
class Split:
    """Stores per dataset name for training, and for the held-out participant."""

    train: dict[str, dict[Path, dict]]
    test: dict[str, dict[Path, dict]]


def participants(stores: dict[str, dict[Path, dict]], dataset: str) -> list[str]:
    """The participant ids the named dataset's admitted stores carry, sorted
    as ids (``'2'`` before ``'10'``). Naming a dataset that was not loaded
    is an error listing what is."""
    if dataset not in stores:
        raise ValueError(
            f"--test-participant-dataset {dataset!r} is not one of the loaded "
            f"datasets {sorted(stores)}")
    ids = {str(_lookup(a, "participant")) for a in stores[dataset].values()
           if _lookup(a, "participant") is not None}
    return sorted(ids, key=lambda s: (len(s), s))


def hold_out_participant(stores: dict[str, dict[Path, dict]],
                         dataset: str, participant: str) -> Split:
    """Drop one participant from one dataset; that participant is the test set.

    Ids are the dataset's own (``'1'`` in Neckflix and ``'01'`` in PURE are
    unrelated people), so the participant is removed from the named dataset
    only. Every other dataset trains whole. Naming a dataset that was not
    loaded, or a participant it does not carry, is an error listing what is.
    """
    present = participants(stores, dataset)
    participant = str(participant)
    pool = stores[dataset]
    test = {p: a for p, a in pool.items()
            if _lookup(a, "participant") is not None
            and str(_lookup(a, "participant")) == participant}
    if not test:
        raise ValueError(
            f"--test-participant-id {participant!r} matches no admitted store in "
            f"{dataset!r}; its participants are {present}")
    train = {name: dict(pool) for name, pool in stores.items()}
    train[dataset] = {p: a for p, a in pool.items() if p not in test}
    return Split(train=train, test={dataset: test})
