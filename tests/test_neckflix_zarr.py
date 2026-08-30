import pytest
import torch

from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.zarr_dataset import BaseZarrDataset
from tests.zarr_fixtures import base_cfg, make_store, make_unreadable_store


# --------------------------------------------------------------------------
# Task 5: class shape + config validation
# --------------------------------------------------------------------------
def test_neckflix_channel_map():
    ds = object.__new__(NeckflixDataset)   # property needs no __init__
    assert ds.channel_map == {
        "R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2),
        "I": ("ir", 0), "D": ("depth", 0),
    }


def test_is_torch_dataset_subclass():
    assert issubclass(NeckflixDataset, BaseZarrDataset)
    assert issubclass(BaseZarrDataset, torch.utils.data.Dataset)


def test_bad_label_norm_raises(tmp_path):
    with pytest.raises(ValueError, match="label_norm"):
        NeckflixDataset(base_cfg(tmp_path, label_norm="fixed"))


def test_filter_overlap_raises_upfront_even_with_empty_cache(tmp_path):
    # Deviation 6: validation is unconditional, before any store is scanned.
    cfg = base_cfg(
        tmp_path,
        filters={"participant": {"include": ["030"], "exclude": ["030"]}},
    )
    with pytest.raises(ValueError, match="Overlapping include/exclude"):
        NeckflixDataset(cfg)


def test_unknown_channel_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown channel"):
        NeckflixDataset(base_cfg(tmp_path, channels=["R", "X"]))


def test_main_still_imports_and_neckflix_unregistered():
    import main
    with pytest.raises(ValueError, match="Unsupported dataset"):
        main.get_loader_class("Neckflix")
