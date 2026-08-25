import argparse
import textwrap


def test_new_preprocess_keys(tmp_path):
    from config import get_config
    yaml = tmp_path / "c.yaml"
    yaml.write_text(textwrap.dedent("""\
        BASE: ['']
        TOOLBOX_MODE: 'train_and_test'
        TRAIN:
          DATA:
            DATASET: 'TestDataset'
            PREPROCESS:
              CHANNELS: ['R','G','B','I']
              TRACES: ['ABP','CVP']
              SIGNAL_NORMS:
                ABP: [0, 180]
        TEST:
          DATA:
            DATASET: 'TestDataset'
    """))
    cfg = get_config(argparse.Namespace(config_file=str(yaml)))
    assert cfg.TRAIN.DATA.PREPROCESS.CHANNELS == ['R', 'G', 'B', 'I']
    assert cfg.TRAIN.DATA.PREPROCESS.TRACES == ['ABP', 'CVP']
    assert list(cfg.TRAIN.DATA.PREPROCESS.SIGNAL_NORMS.ABP) == [0, 180]
    # defaults intact elsewhere
    assert cfg.TEST.DATA.PREPROCESS.CHANNELS == []
    assert list(cfg.TEST.DATA.PREPROCESS.SIGNAL_NORMS.CVP) == [-20.0, 30.0]
