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


# --------------------------------------------------------------------------
# End-to-end smoke: data loads through a real DataLoader
# --------------------------------------------------------------------------
def test_dataloader_end_to_end_smoke(tmp_path):
    from torch.utils.data import DataLoader

    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    make_store(tmp_path, "P031_S01_R1_45_D", num_frames=12)
    ds = NeckflixDataset(base_cfg(tmp_path, window_size=4))
    assert len(ds) == 6                                  # 2 recordings x 3 windows

    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
    assert set(batch) == {"frames", "labels", "label_stats",
                          "channel_mask", "label_mask", "metadata"}
    for ch in ("R", "G", "B", "I", "D"):
        assert batch["frames"][ch].shape == (4, 1, 4, 8, 8)
        assert batch["frames"][ch].dtype == torch.float32
        assert batch["channel_mask"][ch].dtype == torch.bool
    for sig in ("ABP", "CVP"):
        assert batch["labels"][sig].shape == (4, 4)
        assert torch.isfinite(batch["labels"][sig]).all()
        assert batch["label_stats"][sig]["mean"].shape == (4,)
        assert batch["label_mask"][sig].all()
    assert batch["metadata"]["recording_id"][0] == "P030_S01_R1_0_D"
    assert batch["metadata"]["start_frame"].dtype == torch.int64
