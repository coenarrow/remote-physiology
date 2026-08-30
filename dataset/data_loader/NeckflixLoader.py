"""Neckflix dataset over the preprocessor zarr cache.

Provides only the Neckflix channel map; all loading logic lives in
``BaseZarrDataset``. Replaces the retired HDF5-cache loader — see the design
spec at docs/superpowers/specs/2026-08-30-neckflix-zarr-loader-design.md.
"""

from dataset.data_loader.zarr_dataset import BaseZarrDataset


class NeckflixDataset(BaseZarrDataset):
    """Dataset for Neckflix zarr stores."""

    _CHANNEL_MAP = {
        "R": ("rgb", 0),
        "G": ("rgb", 1),
        "B": ("rgb", 2),
        "I": ("ir", 0),
        "D": ("depth", 0),
    }

    @property
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """Neckflix channel name to (stream_group, channel_index) mapping."""
        return self._CHANNEL_MAP
