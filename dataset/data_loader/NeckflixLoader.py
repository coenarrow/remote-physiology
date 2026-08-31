"""Neckflix dataset over the preprocessor zarr cache.

Provides only the Neckflix channel map; all loading logic lives in
``BaseZarrDataset``. The cache is written by ghcr.io/coenarrow/neckflix
(>= 1.0.0); the store contract is documented in docs/architecture.md.
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
