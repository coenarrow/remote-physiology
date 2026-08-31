"""Entry-point wiring: split construction, naming, and the misconfiguration guards."""
import argparse

import numpy as np
import pytest

import neckflix_main
from config import get_config
from tests.zarr_fixtures import make_store

PHYSMAMBA_CONFIG = "configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml"
UNSUPERVISED_CONFIG = "configs/neckflix/NECKFLIX_UNSUPERVISED.yaml"


@pytest.fixture
def cache(tmp_path):
    for name in ("P001_S01_R1_0_D", "P002_S01_R1_0_D", "P003_S01_R1_45_D"):
        make_store(tmp_path, name=name, streams=("rgb",), traces=("abp", "cvp"),
                   num_frames=48, hw=(16, 16))
    return tmp_path


def _config(config_file, cache, **overrides):
    config = get_config(argparse.Namespace(config_file=config_file))
    config.defrost()
    for block in (config.TRAIN.DATA, config.VALID.DATA, config.TEST.DATA,
                  config.UNSUPERVISED.DATA):
        block.CACHED_PATH = str(cache)
        block.PREPROCESS.CHUNK_LENGTH = 16
        block.PREPROCESS.CHUNK_STRIDE = 16
        block.PREPROCESS.CHANNELS = ["R", "G", "B"]
        block.PREPROCESS.TRACES = ["ABP", "CVP"]
    for key, value in overrides.items():
        setattr(config.TEST, key, value)
    config.freeze()
    return config


def _args(**overrides):
    defaults = dict(config_file=PHYSMAMBA_CONFIG, test_participants=None,
                    valid_participants=None, limit_windows=0)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# --- splits --------------------------------------------------------------
def test_loso_splits_are_disjoint_by_participant(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    loaders = neckflix_main.build_data_loaders(
        config, _args(test_participants=["P003"]), rank=0, world_size=1, is_main=True)

    def participants(loader):
        dataset = loader.dataset
        return {rec.split("_")[0] for rec, _ in dataset.samples}

    assert participants(loaders["test"]) == {"P003"}
    assert participants(loaders["train"]) == {"P001", "P002"}
    assert loaders["valid"] is None


def test_valid_participants_are_held_out_of_training(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=False)
    loaders = neckflix_main.build_data_loaders(
        config, _args(test_participants=["P003"], valid_participants=["P002"]),
        rank=0, world_size=1, is_main=True)
    train = {rec.split("_")[0] for rec, _ in loaders["train"].dataset.samples}
    valid = {rec.split("_")[0] for rec, _ in loaders["valid"].dataset.samples}
    assert train == {"P001"}
    assert valid == {"P002"}


def test_model_selection_without_a_valid_split_is_refused(cache):
    """USE_LAST_EPOCH False with no valid split would silently test epoch 0."""
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=False)
    with pytest.raises(ValueError, match="no --valid_participants were given"):
        neckflix_main.build_data_loaders(config, _args(test_participants=["P003"]),
                                         rank=0, world_size=1, is_main=True)


def test_empty_split_names_what_to_check(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    with pytest.raises(ValueError, match="dataset is empty"):
        neckflix_main.build_data_loaders(config, _args(test_participants=["P999"]),
                                         rank=0, world_size=1, is_main=True)


def test_limit_windows_subsamples_evenly(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    full = neckflix_main.build_data_loaders(
        config, _args(test_participants=["P003"]), rank=0, world_size=1, is_main=True)
    limited = neckflix_main.build_data_loaders(
        config, _args(test_participants=["P003"], limit_windows=2),
        rank=0, world_size=1, is_main=True)
    assert len(limited["test"].dataset) == 2 < len(full["test"].dataset)
    # spans the split rather than taking a prefix
    assert limited["test"].dataset.indices[-1] == len(full["test"].dataset) - 1


def test_unsupervised_mode_builds_only_its_own_loader(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    loaders = neckflix_main.build_data_loaders(
        config, _args(config_file=UNSUPERVISED_CONFIG), rank=0, world_size=1,
        is_main=True)
    assert set(loaders) == {"unsupervised"}
    assert len(loaders["unsupervised"].dataset) > 0


# --- naming ---------------------------------------------------------------
def test_experiment_name_records_what_varies(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    named = neckflix_main.apply_experiment_naming(
        config, _args(test_participants=["P015"]))
    name = named.TRAIN.DATA.EXP_DATA_NAME
    assert "TRACES-ABP-CVP" in name
    assert "CHANNELS-RGB" in name
    assert "tested_on_015" in name.replace("\\", "/")
    assert named.MODEL.MODEL_DIR.endswith("PreTrainedModels")


def test_unsupervised_naming_uses_the_unsupervised_block(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    named = neckflix_main.apply_experiment_naming(
        config, _args(config_file=UNSUPERVISED_CONFIG))
    assert "CHANNELS-RGB_" in named.UNSUPERVISED.DATA.EXP_DATA_NAME
    assert named.UNSUPERVISED.OUTPUT_SAVE_DIR.endswith("saved_outputs")


def test_train_test_channel_mismatch_is_rejected(cache):
    config = _config(PHYSMAMBA_CONFIG, cache, USE_LAST_EPOCH=True)
    config.defrost()
    config.TEST.DATA.PREPROCESS.CHANNELS = ["R", "G"]
    config.freeze()
    with pytest.raises(ValueError, match="Train and test channels"):
        neckflix_main.apply_experiment_naming(config, _args())


# --- unsupervised dispatch -------------------------------------------------
def test_unknown_unsupervised_method_is_rejected(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    config.defrost()
    config.UNSUPERVISED.METHOD = ["POS", "MAGIC"]
    config.freeze()
    with pytest.raises(ValueError, match="Not supported unsupervised method"):
        neckflix_main.run_unsupervised(config, {"unsupervised": []})


def test_unsupervised_is_a_no_op_off_rank_zero(cache):
    config = _config(UNSUPERVISED_CONFIG, cache)
    assert neckflix_main.run_unsupervised(config, {"unsupervised": []},
                                          is_main=False) is None
