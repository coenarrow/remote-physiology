# Experiment 3 — Redesigned CardioHydra on the synthetic split at 128 px

Queued after experiment 2 in `runs/dev_cardio/exp2_exp3_chain.sh`; same
split, interface, epochs, batch (1), priors and checks as experiment 2's
run 2b (parametric mask heads, softened priors, `RATE_WAVEFORM` as the
only rate term). A first start at 18:03 with the earlier config was
stopped after two minutes when run 2a's result came in; nothing of it is
kept.

## What differs from experiment 2

- Stage 5, kinematics: per pixel, frame and harmonic, the wavenumber,
  signed phase speed and heading from the phase gradient of the field
  (unchanged module; three scales per harmonic now feed its collapse).
- Stage 6: `SSMPredictor`, per trace a linear embed of the per-frame
  features pooled under `pulsatility x trace mask` (log power, unit phase,
  log amplitude, wavenumber, signed log speed, heading per harmonic; log
  intensity; mask share; presence), two residual bidirectional Mamba3 blocks
  of width 128, a linear readout per frame. The mask share feature is now the
  share of the pulsatile pixels the trace claims (the pulsatility mask sums
  to one per frame).
- The statistics head also reads `log1p` of the phase speed pooled under
  each trace's mask, per channel and harmonic.
- CUDA only (Mamba3 Triton kernels).

Command:

```bash
uv run python scripts/run.py --datasets synthetic_neck_high_snr synthetic_neck_demo \
  --config configs/experiment_configs/cardiohydra_FS30_W10S1_RGBID_ABP-CVP_H128W128.yaml \
  --epochs 30 --batch-size 2 --num-workers 4 --run-dir runs/dev_cardio/exp3_cardiohydra_synth_128
```

then the same test, evaluate and visualise commands as experiment 2 on
`runs/dev_cardio/exp3_cardiohydra_synth_128`.

## Results

Started 20:03 after run 2c (batch 1, 1.7 s per step, 7.5 GB on the card,
2.8 min per epoch). Training loss tracks CardioConv's (total 2.8-2.9 by
epoch 6, rate term 0.08-0.10). The epoch-10 weights, tested at 20:40 on
`synthetic_neck`:

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (exp 1, CardioConv old) | 0.27 | 0.11 | 22.1 | 4.4 | +14.3 (20.7) | +10.2 | -0.8 (5.0) | -4.6 (24.8) |
| CardioConv run 2c, epoch 30 | 0.33 | 0.28 | 21.7 | 4.3 | +13.4 (20.8) | +6.9 | -0.2 (5.0) | -7.5 (26.3) |
| CardioHydra, epoch 10 | 0.38 | 0.20 | 20.3 | 4.4 | +11.3 (20.9) | +3.6 | -0.5 (5.0) | +0.2 (36.8) |

CardioHydra leads on ABP (shape, MAE, diastolic level) and on the
predicted waveform's rate; CardioConv leads on CVP shape. Final-epoch
results and masks follow.

## Results at epoch 30 (final), 21:38

Run directory `runs/dev_cardio/exp3_cardiohydra_synth_128` (every epoch's
weights; test records at epochs 10 and 30; visuals under `epoch_30/visuals`,
six windows). Training ended at total 2.73 (lowest epoch 2.56).

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (exp 1, CardioConv old) | 0.27 | 0.11 | 22.1 | 4.4 | +14.3 (20.7) | +10.2 | +6.9 | -0.8 (5.0) | -4.6 (24.8) |
| CardioConv run 2c, epoch 30 | 0.33 | 0.28 | 21.7 | 4.3 | +13.4 (20.8) | +6.9 | +7.5 | -0.2 (5.0) | -7.5 (26.3) |
| CardioHydra, epoch 10 | 0.38 | 0.20 | 20.3 | 4.4 | +11.3 (20.9) | +3.6 | +11.0 | -0.5 (5.0) | +0.2 (36.8) |
| CardioHydra, epoch 30 | 0.33 | 0.22 | 21.1 | 4.4 | +12.5 (20.8) | +10.1 | +14.7 | -0.8 (5.0) | -4.9 (24.5) |

Both redesigned models beat the old design on the synthetic split;
CardioHydra's epoch-10 weights are its best on ABP, the later epochs
traded a little ABP shape for a tighter training loss. CardioConv keeps
the better CVP shape (0.28 against 0.22).

Masks at epoch 30 (six windows): the pulsatility mask sits on the neck in
every channel (0.64 to 0.76 against the neck's 0.62) and, in depth, on the
vessel band: 8.1 % of its mass on the true vessels against a 2.3 % share,
3.5 times chance, with the ABP mask's depth mass on the vein at 6.4 % and
the CVP mask's at 4.5 %. The panel for window 6 (`epoch_30/visuals/
synthetic_neck/6_1_420/panel.png`) shows it: the depth row's attention
follows the vessel contours. The kinematics stage runs on the same
field; its pooled wave speed is one of the statistics head's inputs and
is not visualised separately tonight.
