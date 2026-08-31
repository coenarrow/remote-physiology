"""yacs config -> zarr-loader plain-dict translation, and the LOSO wiring."""
import argparse

import pytest

from config import get_config
from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.neckflix_config import (
    build_filters, frame_size, normalise_participant, participant_filter, zarr_config,
)
from tests.zarr_fixtures import make_store

UNSUPERVISED_CONFIG = "configs/neckflix/NECKFLIX_UNSUPERVISED.yaml"
PHYSMAMBA_CONFIG = "configs/neckflix/NECKFLIX_PHYSMAMBA.yaml"


def load(config_file):
    return get_config(argparse.Namespace(config_file=config_file))


# --- participant ids -----------------------------------------------------
@pytest.mark.parametrize("given,expected", [
    ("P015", "015"), ("015", "015"), (15, "015"), ("p007", "007"),
    (" P030 ", "030"), ("control", "control"),
])
def test_normalise_participant(given, expected):
    """The CLI says P015; the store's root attr says 015."""
    assert normalise_participant(given) == expected


def test_participant_filter_normalises_both_sides():
    spec = participant_filter(include=["P015"], exclude=[7, "P030"])
    assert spec == {"include": ["015"], "exclude": ["007", "030"]}


# --- config translation --------------------------------------------------
def test_unsupervised_config_translates_to_the_loader_contract():
    cfg = zarr_config(load(UNSUPERVISED_CONFIG).UNSUPERVISED.DATA)
    assert cfg["channels"] == ["R", "G", "B"]
    assert cfg["labels"] == ["ABP", "CVP", "ECG"]
    assert cfg["window_size"] == 300
    assert cfg["window_stride"] == 300
    assert cfg["random_windows"] is False
    assert cfg["label_norm"] == "zscore"
    assert cfg["allow_missing"] is True
    assert cfg["filters"]["posture"] == {"include": ["0", "45", "90"], "exclude": []}


def test_chunk_stride_defaults_to_a_whole_window():
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    config.UNSUPERVISED.DATA.PREPROCESS.CHUNK_STRIDE = 0
    assert zarr_config(config.UNSUPERVISED.DATA)["window_stride"] == 300


def test_physmamba_train_block_uses_overlapping_windows():
    config = load(PHYSMAMBA_CONFIG)
    train = zarr_config(config.TRAIN.DATA)
    test = zarr_config(config.TEST.DATA)
    assert train["window_size"] == 128 and train["window_stride"] == 64
    assert test["window_stride"] == 128, "evaluation windows should not overlap"


def test_loso_filters_are_disjoint():
    data = load(PHYSMAMBA_CONFIG).TRAIN.DATA
    train = zarr_config(data, exclude_participants=["P015"])
    held_out = zarr_config(data, include_participants=["P015"])
    assert train["filters"]["participant"] == {"include": [], "exclude": ["015"]}
    assert held_out["filters"]["participant"] == {"include": ["015"], "exclude": []}


def test_random_windows_override_wins_over_the_config():
    data = load(PHYSMAMBA_CONFIG).TRAIN.DATA
    data.defrost()
    data.PREPROCESS.NECKFLIX.RANDOM_CHUNK = True
    assert zarr_config(data)["random_windows"] is True
    assert zarr_config(data, random_windows=False)["random_windows"] is False


def test_empty_config_lists_mean_no_filter():
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    neckflix = config.UNSUPERVISED.DATA.PREPROCESS.NECKFLIX
    neckflix.FILTERS.posture = []
    assert build_filters(config.UNSUPERVISED.DATA) == {}


def test_configured_attribute_lists_become_include_filters():
    """FILTERS keys are store attrs verbatim — no fixed list, any attr works."""
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    neckflix = config.UNSUPERVISED.DATA.PREPROCESS.NECKFLIX
    neckflix.FILTERS.perspective = [1]
    neckflix.FILTERS.light = ["D"]
    neckflix.FILTERS.site = ["A"]      # an attr the old fixed list never knew
    filters = build_filters(config.UNSUPERVISED.DATA)
    assert filters["perspective"]["include"] == [1]
    assert filters["light"]["include"] == ["D"]
    assert filters["site"]["include"] == ["A"]


def test_participant_key_in_filters_is_refused():
    """Participants go through PARTICIPANTS/CLI so their ids get normalised."""
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    config.UNSUPERVISED.DATA.PREPROCESS.NECKFLIX.FILTERS.participant = ["P015"]
    with pytest.raises(ValueError, match="PARTICIPANTS"):
        build_filters(config.UNSUPERVISED.DATA)


def test_explicit_participants_list_is_used_when_no_cli_argument():
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    config.UNSUPERVISED.DATA.PREPROCESS.NECKFLIX.PARTICIPANTS = ["P002"]
    filters = build_filters(config.UNSUPERVISED.DATA)
    assert filters["participant"]["include"] == ["002"]


def test_frame_size_reads_the_resize_block():
    config = load(PHYSMAMBA_CONFIG)
    assert frame_size(config.TRAIN.DATA) == (128, 128)
    config.defrost()
    config.TRAIN.DATA.PREPROCESS.RESIZE.H = 0
    assert frame_size(config.TRAIN.DATA) is None


# --- against a real (synthetic) cache ------------------------------------
def test_translated_config_drives_a_loso_split(tmp_path):
    for name in ("P015_S01_R1_0_D", "P016_S01_R1_0_D", "P017_S01_R1_45_N"):
        make_store(tmp_path, name=name, streams=("rgb",), traces=("abp", "cvp"),
                   num_frames=16, hw=(6, 6))
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    data = config.UNSUPERVISED.DATA
    data.CACHED_PATH = str(tmp_path)
    data.PREPROCESS.CHUNK_LENGTH = 8
    data.PREPROCESS.CHUNK_STRIDE = 8
    data.PREPROCESS.TRACES = ["ABP", "CVP"]

    held_out = NeckflixDataset(zarr_config(data, include_participants=["P015"]))
    rest = NeckflixDataset(zarr_config(data, exclude_participants=["P015"]))
    assert held_out.attribute_values("participant") == ["015"]
    assert rest.attribute_values("participant") == ["016", "017"]
    assert len(held_out) and len(rest)


def test_posture_filter_from_the_config_reaches_the_loader(tmp_path):
    make_store(tmp_path, name="P020_S01_R1_0_D", streams=("rgb",),
               traces=("abp",), num_frames=16, hw=(6, 6))
    make_store(tmp_path, name="P020_S01_R2_45_D", streams=("rgb",),
               traces=("abp",), num_frames=16, hw=(6, 6))
    config = load(UNSUPERVISED_CONFIG)
    config.defrost()
    data = config.UNSUPERVISED.DATA
    data.CACHED_PATH = str(tmp_path)
    data.PREPROCESS.CHUNK_LENGTH = 8
    data.PREPROCESS.CHUNK_STRIDE = 8
    data.PREPROCESS.TRACES = ["ABP"]
    data.PREPROCESS.NECKFLIX.FILTERS.posture = ["45"]
    dataset = NeckflixDataset(zarr_config(data))
    assert {rec for rec, _ in dataset.samples} == {"P020_S01_R2_45_D"}
