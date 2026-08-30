"""Lazy torch datasets over external Neckflix-preprocessor zarr caches.

Ported from CardioHydra src/dataset/base.py; see the design spec at
docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md for the
contract and the sanctioned deviations (1-7). The cache is an external input
produced by ghcr.io/coenarrow/neckflix (>= 1.0.0); this module never writes it.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

# Sentinel for root attrs absent from a store (deviation 2).
_MISSING = object()


def _validate_filters(filters):
    """Reject overlapping include/exclude upfront (deviation 6).

    CardioHydra checks lazily inside the per-sample loop, where an overlap can
    pass silently when no sample reaches that attribute; this port validates
    unconditionally at construction.
    """
    for attribute, spec in (filters or {}).items():
        overlap = set(spec.get("include", [])) & set(spec.get("exclude", []))
        if overlap:
            raise ValueError(
                f"Overlapping include/exclude for '{attribute}': {overlap}"
            )


class BaseZarrDataset(ABC, torch.utils.data.Dataset):
    """Abstract lazy dataset over Neckflix-preprocessor zarr stores.

    Subclasses provide only ``channel_map``. Construction is metadata-only:
    no pixel data is read until ``__getitem__``.
    """

    MIN_TOOL_VERSION = (1, 0, 0)  # raw-frame cache format floor

    @property
    @abstractmethod
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """Map channel names to ``(stream_group, channel_index)`` pairs."""
        ...

    def _resolve_streams(self, channels):
        """Map config channel names through ``channel_map``, preserving order."""
        cmap = self.channel_map
        plan = []
        for ch in channels:
            if ch not in cmap:
                raise ValueError(f"Unknown channel: {ch}. Valid: {list(cmap)}")
            plan.append(cmap[ch])
        return plan

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cache_root = Path(cfg["cache_dir"])
        self.channels = cfg["channels"]
        self.labels = cfg["labels"]
        self.window_size = cfg["window_size"]
        self.window_stride = cfg.get("window_stride", self.window_size)
        self.random_windows = cfg.get("random_windows", False)
        self.filters = cfg.get("filters", {})
        self.label_norm = cfg.get("label_norm", "zscore")
        if self.label_norm not in ("zscore", "minmax"):
            raise ValueError(
                f"label_norm must be 'zscore' or 'minmax', got {self.label_norm!r}"
            )
        _validate_filters(self.filters)

        self.stream_plan = self._resolve_streams(self.channels)
        self.required_streams = sorted({s[0].lower() for s in self.stream_plan})

        self.allow_missing = cfg.get("allow_missing", False)
        self.min_channels = cfg.get("min_channels", 1)
        self.min_labels = cfg.get("min_labels", 1)
        self.present_streams: dict[tuple[str, str], list[str]] = {}
        self.present_labels: dict[tuple[str, str], list[str]] = {}
        self.stream_hw: dict[str, tuple[int, int]] | None = None

        raise NotImplementedError("cache pipeline arrives in Task 6")
