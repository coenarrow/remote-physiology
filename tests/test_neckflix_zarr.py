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


def test_dataset_keys_are_exactly_the_declared_contract(tmp_path):
    """The dataset and neural_methods.batch must not drift apart."""
    from neural_methods.batch import (
        CAMERA_ID, LOADER_KEYS, METADATA, RECORDING_ID, START_FRAME,
    )

    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=12)
    item = NeckflixDataset(base_cfg(tmp_path, window_size=4))[0]
    assert set(item) == set(LOADER_KEYS)
    assert set(item[METADATA]) == {RECORDING_ID, CAMERA_ID, START_FRAME}


def test_model_consumes_the_dataset_output_unchanged(tmp_path):
    """Loader -> collate -> model, with no adapter in between."""
    from torch.utils.data import DataLoader

    from neural_methods.batch import PREDICTIONS
    from neural_methods.frame_transforms import FrameTransform
    from neural_methods.model.PhysMamba import PhysMamba

    make_store(tmp_path, "P030_S01_R1_0_D", num_frames=64, hw=(32, 32))
    dataset = NeckflixDataset(base_cfg(tmp_path, window_size=16, channels=["R", "G", "B"]))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    model = PhysMamba(channels=("R", "G", "B"), traces=("ABP", "CVP"),
                      frame_transform=FrameTransform(("DiffNormalized",), size=(32, 32)))
    out = model(batch)
    assert set(out) == set(batch) | {PREDICTIONS}
    assert out[PREDICTIONS]["ABP"].shape == (2, 16)
    assert torch.isfinite(out[PREDICTIONS]["CVP"]).all()
