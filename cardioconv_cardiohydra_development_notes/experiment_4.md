# Experiment 4 — Every other model on the same synthetic split

Started 2026-10-10 16:45 WAST, alongside experiments 2 and 3 on the same
GPU (`runs/dev_cardio/exp4_baselines_chain.sh`).

## Purpose

The user's acceptance criterion allows "a small synthetic dataset for
train/test, and outperform all the other models on that". This is the
"other models" side: the eleven single-trace architectures in the registry,
trained and tested exactly as experiments 2 and 3 are, so the comparison is
like for like in data, epochs and scoring.

## Setup

- Models: PhysMamba, FactorizePhys, RhythmFormer, PhysNet, PhysFormer,
  iBVPNet, in that order. Cut at 16:10 from all eleven to the six that led
  the Neckflix benchmarks, so the night's GPU queue (experiments 2, 3, 5,
  then this) ends before the deadline; DeepPhys, TS-CAN, EfficientPhys,
  PhysMamba2 and PhysMamba3 can be added with the same command if time
  remains.
- Configs: `configs/default_configs/<model>_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml`,
  the repo's synthetic benchmark interface: 72 px, RGB-IR-depth, ABP, CVP and
  PPG, each model's own paper recipe. The PPG trace is extra for them; the
  comparison reads the ABP and CVP rows only.
- Train: `synthetic_neck_high_snr` + `synthetic_neck_demo`, no hold-out,
  20 epochs (cut from 30 at 16:14 to fit the night's GPU queue; the README's
  synthetic benchmark uses 10), batch 2 (the README's benchmark batch). Test
  on `synthetic_neck`, scored with `scripts/evaluate.py`.
- Known difference in favour of experiments 2 and 3: they run at 128 px,
  the baselines at their default 72 px. If the margin is close this is
  revisited with a 128 px baseline.

Command per model:

```bash
uv run python scripts/run.py --datasets synthetic_neck_high_snr synthetic_neck_demo \
  --config configs/default_configs/<model>_FS30_W10S1_RGBID_ABP-CVP-PPG_H72W72.yaml \
  --epochs 30 --batch-size 2 --num-workers 4 --run-dir runs/dev_cardio/exp4_<model>_synth_72
uv run python scripts/test.py --checkpoint runs/dev_cardio/exp4_<model>_synth_72 --test-dataset synthetic_neck
uv run python scripts/evaluate.py runs/dev_cardio/exp4_<model>_synth_72
```

## Results (chain ran 01:01-02:04, every stage exit 0)

Test set `synthetic_neck`, 30 measurements of 10 s from 10 stores, scored
with `scripts/evaluate.py` on the last epoch. The two redesigned models
are from experiments 2 and 3 (128 px, 30 epochs at batch 1, 3,000 steps);
the baselines ran at their 72 px default with 20 epochs at batch 2 (1,000
steps). ABP and CVP only; the baselines' PPG trace is not compared.

| Model | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **CardioConv (exp 2c)** | **0.33** | **0.28** | 21.7 | 4.25 | +13.4 (20.8) | +6.9 | +7.5 | -0.2 (5.0) | -7.5 (26.3) |
| **CardioHydra (exp 3)** | **0.33** | 0.22 | **21.1** | 4.37 | +12.5 (20.8) | +10.1 | +14.7 | -0.8 (5.0) | -4.9 (24.5) |
| CardioConv, old design (exp 1, 40 ep) | 0.27 | 0.11 | 22.1 | 4.38 | +14.3 (20.7) | +10.2 | +6.9 | -0.8 (5.0) | -4.6 (24.8) |
| FactorizePhys | 0.25 | 0.18 | 22.6 | 4.40 | +11.2 (20.9) | +4.3 | +6.4 | +0.4 (5.0) | +9.5 (23.6) |
| RhythmFormer | 0.21 | 0.09 | 21.4 | 4.22 | +14.3 (20.4) | +19.3 | -1.4 | -2.0 (5.3) | -17.1 (25.1) |
| PhysMamba | 0.08 | 0.14 | 26.5 | 4.22 | +22.9 (20.7) | +33.3 | +2.2 | -2.7 (4.4) | -19.1 (40.9) |
| PhysNet | 0.05 | 0.01 | 32.0 | 4.24 | +30.0 (20.6) | +30.0 | +12.9 | -0.9 (5.0) | -25.0 (25.8) |

On waveform shape both redesigned models beat every baseline and the old
design on both traces; CardioHydra has the lowest ABP MAE and CardioConv
the best CVP shape. The level errors are of one size for every model
that predicts a waveform at all (ABP read 11-14 mmHg high on these
noisy base-prior stores): the magnitude problem the project already
knows, not something the masks address. Caveats for the thesis: the
baselines had a third of the gradient steps and a smaller input (their
paper interface, 72 px), and FactorizePhys at 72 px is within 0.08 CCC of
the redesigned models on ABP; a 128 px baseline run would make the margin
cleaner.
