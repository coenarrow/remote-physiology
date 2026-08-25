import argparse
import textwrap

import h5py
import numpy as np
import pytest
import torch

T_FRAMES, H, W = 32, 16, 16


def _write_h5(path, channels, traces, T=T_FRAMES):
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        for ch in channels:
            g = f.create_group(ch)
            g.create_dataset('frames', data=rng.integers(0, 255, (T, H, W), dtype=np.uint8))
            g.create_dataset('timestamps', data=np.arange(T, dtype=np.int32))
            for tr in traces:
                g.create_dataset(tr, data=rng.normal(50, 10, T).astype(np.float32))


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    # P001: full channels + all traces; P002: RGB only, no CVP
    _write_h5(d / "P001_S01_R1_0_D_K1.hdf5", ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP', 'ECG'])
    _write_h5(d / "P002_S01_R1_0_D_K1.hdf5", ['R', 'G', 'B'], ['ABP', 'ECG'])
    return d


def _make_loader(cache_dir, tmp_path, channels, traces):
    from config import get_config
    from dataset.data_loader.NeckflixLoader import NeckflixLoader
    yaml = tmp_path / "cfg.yaml"
    yaml.write_text(textwrap.dedent(f"""\
        BASE: ['']
        TOOLBOX_MODE: 'train_and_test'
        TRAIN:
          DATA:
            CACHED_PATH: "{cache_dir}"
            PREPROCESS:
              CHUNK_LENGTH: 16
              DATA_TYPE: ['DiffNormalized','Standardized']
              RESIZE:
                H: 8
                W: 8
              CHANNELS: {channels}
              TRACES: {traces}
              NECKFLIX:
                RANDOM_CHUNK: False
    """))
    cfg = get_config(argparse.Namespace(config_file=str(yaml)))
    return NeckflixLoader(name="train", data_path=str(cache_dir),
                          config_data=cfg.TRAIN.DATA, dict_output=True)


def test_dict_contract_full_recording(cache_dir, tmp_path):
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP'])
    items = [dl[i] for i in range(len(dl))]
    full = [it for it in items if it['filename'].startswith('P001')][0]
    assert full['frames'].shape == (10, 16, 8, 8)          # 2 blocks * 5 slots
    assert full['channel_mask'].tolist() == [1, 1, 1, 1, 1]
    assert set(full['labels']) == {'ABP', 'CVP'}
    assert full['labels']['ABP'].shape == (16,)
    assert float(full['label_mask']['ABP']) == 1.0
    assert float(full['label_mask']['CVP']) == 1.0
    assert np.isfinite(full['frames']).all()


def test_zero_fill_and_masks_partial_recording(cache_dir, tmp_path):
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP'])
    partial = [dl[i] for i in range(len(dl)) if dl[i]['filename'].startswith('P002')][0]
    assert partial['channel_mask'].tolist() == [1, 1, 1, 0, 0]
    # zero-filled slots: I and D blocks are all-zero in both transform blocks
    f = partial['frames']
    assert np.abs(f[3:5]).sum() == 0 and np.abs(f[8:10]).sum() == 0
    assert np.abs(f[0:3]).sum() > 0
    # CVP missing -> zeros + mask 0
    assert float(partial['label_mask']['CVP']) == 0.0
    assert np.abs(partial['labels']['CVP']).sum() == 0
    assert float(partial['label_mask']['ABP']) == 1.0
    assert np.isfinite(f).all()


def test_partial_recordings_are_kept_not_filtered(cache_dir, tmp_path):
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B', 'I', 'D'], ['ABP', 'CVP'])
    names = {dl[i]['filename'][:4] for i in range(len(dl))}
    assert names == {'P001', 'P002'}


def test_collate(cache_dir, tmp_path):
    from dataset.data_loader.multisignal_collate import multisignal_collate
    dl = _make_loader(cache_dir, tmp_path, ['R', 'G', 'B'], ['ABP'])
    batch = multisignal_collate([dl[0], dl[1]])
    assert batch['frames'].shape[0] == 2
    assert isinstance(batch['frames'], torch.Tensor)
    assert batch['labels']['ABP'].shape[0] == 2
    assert batch['label_mask']['ABP'].shape == (2,)
    assert isinstance(batch['filename'], list) and len(batch['filename']) == 2
