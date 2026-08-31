"""The dict-contract trainer, end to end over a synthetic zarr cache.

Small stores, tiny frames, one epoch: this is a wiring test, not a learning
test. What it pins down is that the batch dict survives the whole round trip —
loader, collate, model, masked loss, per-signal metrics, saved outputs — and
that a recording missing a trace is scored on the traces it does have.
"""
import argparse
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.neckflix_config import zarr_config
from neural_methods.batch import PREDICTIONS, move_to_device
from neural_methods.trainer.MultiSignalTrainer import (
    MODEL_REGISTRY, MultiSignalTrainer, build_model,
)
from tests.zarr_fixtures import make_store

SMOKE_CONFIG = "configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml"
WINDOW = 16
FRAME_SIZE = 32


@pytest.fixture
def cache(tmp_path):
    """Three recordings; the third carries no ABP, exercising label_mask."""
    make_store(tmp_path, name="P001_S01_R1_0_D", streams=("rgb",),
               traces=("abp", "cvp"), num_frames=40, hw=(FRAME_SIZE, FRAME_SIZE))
    make_store(tmp_path, name="P002_S01_R1_0_D", streams=("rgb",),
               traces=("abp", "cvp"), num_frames=40, hw=(FRAME_SIZE, FRAME_SIZE))
    make_store(tmp_path, name="P003_S01_R1_45_D", streams=("rgb",),
               traces=("cvp",), num_frames=40, hw=(FRAME_SIZE, FRAME_SIZE))
    return tmp_path


@pytest.fixture
def config(cache):
    cfg = get_config(argparse.Namespace(config_file=SMOKE_CONFIG))
    cfg.defrost()
    for block in (cfg.TRAIN.DATA, cfg.VALID.DATA, cfg.TEST.DATA):
        block.CACHED_PATH = str(cache)
        block.PREPROCESS.CHUNK_LENGTH = WINDOW
        block.PREPROCESS.CHUNK_STRIDE = WINDOW
        block.PREPROCESS.TRACES = ["ABP", "CVP"]
        block.PREPROCESS.CHANNELS = ["R", "G", "B"]
        block.PREPROCESS.RESIZE.H = FRAME_SIZE
        block.PREPROCESS.RESIZE.W = FRAME_SIZE
    cfg.TRAIN.EPOCHS = 1
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.INFERENCE.BATCH_SIZE = 2
    cfg.TEST.USE_LAST_EPOCH = True
    cfg.TEST.METRICS = ['MAE', 'RMSE', 'MACC']       # no BA: skip plot writing
    cfg.MODEL.MODEL_DIR = str(cache / "models")
    cfg.TEST.OUTPUT_SAVE_DIR = str(cache / "outputs")
    cfg.freeze()
    return cfg


def loaders_for(config, *, train_exclude=("P003",), test_include=("P003",)):
    def loader(block, batch_size, **kwargs):
        dataset = NeckflixDataset(zarr_config(block, random_windows=False, **kwargs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, drop_last=False)
    return {
        "train": loader(config.TRAIN.DATA, config.TRAIN.BATCH_SIZE,
                        exclude_participants=train_exclude),
        "valid": None,
        "test": loader(config.TEST.DATA, config.INFERENCE.BATCH_SIZE,
                       include_participants=test_include),
    }


# --- registry / construction --------------------------------------------
def test_registry_holds_the_dict_contract_models():
    assert {"PhysMamba", "DeepPhys"} <= set(MODEL_REGISTRY)


def test_build_model_reads_channels_traces_and_transform(config):
    model = build_model(config, config.TRAIN.DATA)
    assert model.channels == ("R", "G", "B")
    assert model.traces == ("ABP", "CVP")
    assert model.frame_transform.size == (FRAME_SIZE, FRAME_SIZE)
    assert model.frame_transform.data_types == ("DiffNormalized",)


def test_build_model_rejects_an_unregistered_model(config):
    config.defrost()
    config.MODEL.NAME = "RhythmFormer"
    config.freeze()
    with pytest.raises(ValueError, match="does not speak the Neckflix dict contract"):
        build_model(config, config.TRAIN.DATA)


# --- end to end ----------------------------------------------------------
def test_train_then_test_round_trip(config, cache, capsys):
    loaders = loaders_for(config)
    assert len(loaders["train"].dataset) > 0
    assert len(loaders["test"].dataset) > 0

    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    assert (cache / "models" / "neckflix_physmamba_smoke_Epoch0.pth").exists()

    report = trainer.test(loaders)
    # P003 has no ABP, so only CVP is scored on the held-out split.
    assert set(report) == {"CVP"}
    assert report["CVP"]["n"] > 0
    assert report["CVP"]["MAE"] == pytest.approx(report["CVP"]["MAE"])  # finite

    printed = capsys.readouterr().out
    assert "[CVP] waveform Pearson" in printed
    assert "no windows carried this label" in printed      # the ABP row


def test_saved_outputs_carry_signal_keyed_windows(config, cache):
    loaders = loaders_for(config, train_exclude=("P003",), test_include=("P001",))
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    trainer.test(loaders)

    path = cache / "outputs" / "neckflix_physmamba_smoke_outputs.pickle"
    payload = pickle.loads(path.read_bytes())
    assert payload["traces"] == ["ABP", "CVP"]
    assert payload["channels"] == ["R", "G", "B"]
    assert payload["label_norm"] == "zscore"
    signals = {record["signal"] for record in payload["windows"]}
    assert signals == {"ABP", "CVP"}
    record = payload["windows"][0]
    assert record["recording_id"].startswith("P001")
    assert len(record["prediction"]) == WINDOW == len(record["label"])
    assert set(record["label_stats"]) == {"mean", "std", "min", "max"}


def test_one_training_step_moves_the_weights(config):
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    before = trainer.model.ConvBlock1[0].weight.detach().clone()
    trainer.train(loaders)
    after = trainer.model.ConvBlock1[0].weight.detach()
    assert not torch.allclose(before, after)


def test_model_output_still_carries_the_loader_keys(config):
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    batch = move_to_device(next(iter(loaders["train"])), trainer.device)
    out = trainer.model(batch)
    assert set(out) == set(batch) | {PREDICTIONS}
    assert set(out[PREDICTIONS]) == {"ABP", "CVP"}
    assert out["metadata"]["recording_id"] == batch["metadata"]["recording_id"]


def test_validation_split_drives_best_epoch_selection(config, cache):
    """USE_LAST_EPOCH False + a held-out valid split exercises valid()."""
    config.defrost()
    config.TEST.USE_LAST_EPOCH = False
    config.TRAIN.EPOCHS = 2
    config.freeze()

    def loader(block, batch_size, **kwargs):
        dataset = NeckflixDataset(zarr_config(block, random_windows=False, **kwargs))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, drop_last=False)

    loaders = {
        "train": loader(config.TRAIN.DATA, 2, exclude_participants=("P002", "P003")),
        "valid": loader(config.VALID.DATA, 2, include_participants=("P002",)),
        "test": loader(config.TEST.DATA, 2, include_participants=("P003",)),
    }
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)

    assert trainer.min_valid_loss is not None
    assert np.isfinite(trainer.min_valid_loss)
    assert trainer.best_epoch in (0, 1)
    # test() then loads the *best* epoch, not the last one.
    assert (cache / "models"
            / f"neckflix_physmamba_smoke_Epoch{trainer.best_epoch}.pth").exists()
    assert trainer.test(loaders) is not None


def test_valid_without_a_valid_loader_is_an_explicit_error(config):
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    with pytest.raises(ValueError, match="No data for valid"):
        trainer.valid({"valid": None})


def test_only_test_mode_needs_no_train_loader(config, cache):
    """Train once to produce a checkpoint, then reload it through only_test."""
    loaders = loaders_for(config, test_include=("P001",))
    MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False).train(loaders)

    config.defrost()
    config.TOOLBOX_MODE = "only_test"
    config.INFERENCE.MODEL_PATH = str(
        cache / "models" / "neckflix_physmamba_smoke_Epoch0.pth")
    config.freeze()
    trainer = MultiSignalTrainer(config, {"test": loaders["test"]},
                                 rank=0, world_size=1, debug=False)
    assert trainer.test({"test": loaders["test"]}) is not None


def test_physical_unit_error_is_the_normalised_error_times_the_window_scale(config):
    """z-score inverse is linear, so the mmHg error is MAE x that window's std."""
    import torch as _torch

    loaders = loaders_for(config, test_include=("P001",))
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    sample = {
        "predictions": {"ABP": _torch.tensor([0.0, 1.0, -1.0, 0.5])},
        "labels": {"ABP": _torch.tensor([0.0, 0.0, 0.0, 0.0])},
        "label_stats": {"ABP": {"mean": _torch.tensor(90.0), "std": _torch.tensor(12.0),
                                "min": _torch.tensor(70.0), "max": _torch.tensor(130.0)}},
    }
    physical_pred, physical_label = trainer._to_physical(sample, "ABP")
    assert np.allclose(physical_label, 90.0)
    assert np.allclose(physical_pred, [90.0, 102.0, 78.0, 96.0])
    normalised_mae = float(np.mean(np.abs(
        sample["predictions"]["ABP"].numpy() - sample["labels"]["ABP"].numpy())))
    physical_mae = float(np.mean(np.abs(physical_pred - physical_label)))
    assert physical_mae == pytest.approx(normalised_mae * 12.0)


def test_test_report_prints_both_normalised_and_physical_errors(config, capsys):
    loaders = loaders_for(config, test_include=("P001",))
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    trainer.test(loaders)
    printed = capsys.readouterr().out
    assert "(normalised units)" in printed
    assert "(mmHg, at the window's own scale)" in printed


def test_unknown_label_norm_is_rejected_at_construction(config):
    """The loader rejects it first; the trainer guards the same key independently,
    since it has to pick the matching inverse for the physical-unit report."""
    loaders = loaders_for(config)                      # built while the norm is valid
    config.defrost()
    config.TEST.DATA.PREPROCESS.NECKFLIX.LABEL_NORM = "robust"
    config.freeze()
    with pytest.raises(ValueError, match="Unknown LABEL_NORM"):
        MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    with pytest.raises(ValueError, match="label_norm must be"):
        NeckflixDataset(zarr_config(config.TEST.DATA))


def test_epoch_mean_loss_is_printed(config, capsys):
    """The progress bar shows only the last batch; the mean is what matters."""
    loaders = loaders_for(config)
    trainer = MultiSignalTrainer(config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    printed = capsys.readouterr().out
    assert "mean training loss:" in printed
