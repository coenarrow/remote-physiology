# Experiment 5 — Both models on Neckflix, participant 33 held out

Planned for after experiments 2 and 3 (queued once they finish and the
synthetic results have been read).

## Purpose

The user's other acceptance route: beat the best of the existing models on
participant 33. Every baseline has a 30-epoch run in `runs/neckflix_<model>`
(72 px, batch 4, participant 33 held out); their evaluation tables are
produced tonight with `scripts/evaluate.py` where missing
(`runs/dev_cardio/neckflix_baseline_evals.log`).

## Setup

- Local Neckflix cache: 128 px, 332 stores, 50 participants, two camera
  perspectives per recording. Participant 33 held out, every other store
  trains (one random 10 s window per store and perspective per epoch).
- Interface: the 72 px experiment configs, so the comparison with the
  existing runs is like for like in resolution and the epoch is affordable
  (DOWNSCALE 2 gives a 36x36 mask grid; a 128 px epoch over ~650 windows
  at 3 s per step would take 15-25 minutes, too long for tonight). Batch 4,
  15 epochs (cut from 20 at 16:10: a Neckflix epoch is about 650 windows,
  twice the pixel-work of a synthetic epoch, so 15 epochs is about two
  hours per model).
- Known weakness here: Neckflix windows without an ECG fuse only ABP and
  CVP for the rate target, and CVP's second harmonic flips the fused peak
  more often than on the synthetic stores (experiment 1, "The spectral-peak
  rate target"). Such a window trains the rate network towards a doubled
  rate; the gate then keeps that window out of the mask and waveform losses.

Commands:

```bash
uv run python scripts/run.py --datasets neckflix \
  --config configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml \
  --epochs 20 --batch-size 4 --num-workers 4 \
  --test-participant-dataset neckflix --test-participant-id 33 \
  --run-dir runs/dev_cardio/exp5_cardioconv_neckflix_p33
uv run python scripts/evaluate.py runs/dev_cardio/exp5_cardioconv_neckflix_p33
uv run python scripts/visualise.py --checkpoint runs/dev_cardio/exp5_cardioconv_neckflix_p33 \
  --dataset neckflix --test-participant-id 33 --windows 6
```

and the same with `cardiohydra` for `exp5_cardiohydra_neckflix_p33`.

## The bar: the September baselines, re-scored tonight

Their records predate the current evaluator's layout, so each latest
checkpoint (epoch 30, 72 px, batch 4) was run once more over participant 33
with `scripts/test.py` into `runs/dev_cardio/neckflix_p33_baselines/<model>/`
and scored there (inference only; the weights are the September ones).
Pooled over the 10 s measurements of participant 33:

| Model | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FactorizePhys | 0.82 | 0.39 | 11.1 | 2.9 | -3.9 (2.7) | -13.9 | -2.3 (1.4) | -0.2 (0.9) |
| PhysNet | 0.72 | 0.40 | 14.5 | 2.8 | -13.4 (4.7) | -20.9 | -1.0 (2.5) | -0.1 (1.1) |
| RhythmFormer | 0.69 | 0.42 | 16.8 | 2.5 | -16.0 (3.0) | -22.8 | -0.1 (2.4) | -0.3 (1.0) |
| PhysFormer | 0.57 | 0.44 | 16.1 | 2.5 | -14.7 (4.2) | -35.8 | -0.0 (2.5) | -0.3 (1.0) |
| iBVPNet | 0.07 | 0.06 | 20.6 | 3.6 | -17.1 (3.3) | -42.5 | -2.2 (2.0) | -7.1 (1.4) |
| DeepPhys | 0.13 | 0.04 | 24.3 | 3.4 | -18.0 (3.1) | -23.8 | -2.7 (2.0) | +113 (49) |
| TS-CAN | 0.00 | 0.01 | 18.1 | 3.2 | -9.6 (2.8) | -54.3 | -2.6 (1.6) | -31.9 (0.9) |
| EfficientPhys | 0.00 | 0.00 | 18.0 | 3.4 | -9.4 (2.8) | -54.1 | -2.8 (2.0) | -31.9 (0.9) |

(The PhysMamba family needs CUDA and was dropped at the user's request.)
The CAN family predicts a flat waveform here (CCC 0, heart rate off by
32 bpm), which matches the earlier finding that its tanh readout saturates
at the paper learning rate. Every baseline under-reads ABP by 4-18 mmHg;
CVP shape is poor for all of them. The honest target for tonight's models:
ABP CCC above 0.82 or a smaller ABP level error, and a CVP CCC above 0.44.

## Log

- 21:38 CardioConv started (batch 2, not 4, for memory headroom): 628
  training windows per epoch (314 steps), 1.2 steps per second, 5.7 GB on
  the card; the held-out participant's strided windows are run after every
  epoch, so about six minutes per epoch and 1.5 hours for 15 epochs.
  CardioHydra follows in the same chain, then the synthetic baselines
  (experiment 4).
- 22:05, epoch 4 scored from the per-epoch records (CPU): ABP CCC 0.54,
  CVP CCC 0.30, ABP MAE 14.0, ABP diastolic error +2.3 (2.6), mean -8.8
  (2.7), systolic -31.3 (4.0), CVP mean -2.3 (1.9), heart rate -0.2 (0.9).
  After four epochs it already matches PhysNet's ABP MAE and beats every
  baseline on the diastolic level; the systolic is read far too low (the
  predicted spread is too small), and FactorizePhys's ABP CCC 0.82 is the
  bar still to reach.
- 22:25, epoch 7 scored: ABP CCC 0.54, CVP CCC 0.23, ABP MAE 15.8, ABP
  mean -12.3, systolic -33.6, CVP mean -3.2, heart rate -0.1 (0.6). No
  better than epoch 4 on the held-out participant while the training total
  keeps falling (2.97 -> 2.80): the model fits the 49 training participants
  further without transferring more to participant 33. The evaluation
  scores the last epoch by design (never chosen by test error), so the
  epoch-15 numbers are the ones reported; the trajectory is noted here.
- 22:30, masks from the epoch-7 weights on four participant-33 windows
  (`epoch_07/visuals/neckflix`, drawn on the CPU). On real video the
  masks are sharp and anatomical: in recording R3 (45 degrees, camera 2)
  the pulsatility attention in every channel, depth included, lies on a
  thin diagonal band running across the neck where the great vessels run,
  and the ABP and CVP bins take different ends of it; in R1 (supine,
  camera 1) the attention is a short horizontal band mid-neck with the ABP
  mask on its brightest point. The rate estimate matches the label (67.0
  for 66.8; 66.8 for 66.8). The predicted ABP follows the label beat for
  beat with too small an amplitude (systolic under-read), the predicted
  CVP has the right rhythm about 3 mmHg low. No vessel ground truth exists
  here; the correspondence is to where the neck pulses in the frame.

## Results: CardioConv, epoch 15 (final), 23:14

| Model | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FactorizePhys (Sep, 30 ep) | 0.82 | 0.39 | 11.1 | 2.9 | -3.9 (2.7) | -6.5 | -13.9 | -2.3 (1.4) | -0.2 (0.9) |
| PhysNet (Sep, 30 ep) | 0.72 | 0.40 | 14.5 | 2.8 | -13.4 (4.7) | -4.5 | -20.9 | -1.0 (2.5) | -0.1 (1.1) |
| RhythmFormer (Sep, 30 ep) | 0.69 | 0.42 | 16.8 | 2.5 | -16.0 (3.0) | -8.4 | -22.8 | -0.1 (2.4) | -0.3 (1.0) |
| PhysFormer (Sep, 30 ep) | 0.57 | 0.44 | 16.1 | 2.5 | -14.7 (4.2) | -1.2 | -35.8 | -0.0 (2.5) | -0.3 (1.0) |
| **CardioConv (exp 5, 15 ep)** | **0.56** | **0.29** | **16.2** | **3.2** | -11.3 (3.9) | -7.3 | -29.2 | -2.5 (2.0) | -0.2 (0.9) |

Mid-pack on ABP shape (level with PhysFormer, under PhysNet and
RhythmFormer, well under FactorizePhys), the weakest of the five on CVP
shape, the same heart-rate agreement as the best, and the same under-read
of ABP as every model. The held-out agreement did not improve after epoch
4 (ABP CCC 0.54 then, 0.56 now) while the training loss fell throughout,
so more epochs at this setting would not close the gap. The user's
Neckflix acceptance route (beat the best existing model on participant 33)
is not met by this run; the synthetic route is (experiments 2-4).

What the masks look like on this participant is in the log above and in
`epoch_15/visuals/neckflix`.

## Results: CardioHydra

- 23:15 started (same settings as CardioConv; 6.2 min first epoch, then
  4.4 min per epoch).
- 23:35, epoch 3 scored from the per-epoch records: ABP CCC 0.45, CVP CCC
  0.42, ABP MAE 16.9, CVP MAE 2.70, ABP diastolic -1.7 (3.3), mean -13.4
  (3.6), systolic -42.4 (6.9), CVP mean -2.0 (1.9), heart rate -0.4 (1.2).
  The CVP shape and MAE are the best of any model on this participant
  (FactorizePhys 0.39 / 2.88, RhythmFormer 0.42 / 2.54, PhysFormer
  0.44 / 2.47) after three epochs; ABP shape trails.
- 00:05, epoch 6 scored: ABP CCC 0.54, CVP CCC 0.39, ABP MAE 15.7, CVP
  MAE 2.76, ABP diastolic -3.9 (2.8), mean -12.2 (3.6), systolic -38.8
  (5.5), CVP mean -1.8 (2.0), heart rate -0.2 (0.9). Still improving on
  ABP, unlike CardioConv at the same point; CVP MAE better than
  FactorizePhys's.

- 01:01, epoch 15 (final, the epoch the protocol scores): ABP CCC 0.49,
  CVP CCC **0.02**, ABP MAE 17.8, CVP MAE 3.6, ABP mean -15.8 (3.2),
  systolic -43.2 (5.1), CVP mean -2.8 (2.3), heart rate -3.9 (2.1).

The CVP shape collapsed in the last two epochs. The training log shows
it: the CVP CCC term sat at 0.73-0.75 through epoch 13 and jumped to 0.95
at epochs 14 and 15 while the CVP level terms did not move, and no step was
refused as non-finite. Scored per epoch from the held-out records: CVP CCC
0.42 at epoch 3, 0.39 at 6, 0.30 at 11, 0.33 at 13, 0.02 at 15; ABP CCC
0.45, 0.54, 0.53, 0.52, 0.49. So up to epoch 13 CardioHydra was the best
model on this participant for CVP shape and MAE and level with the others
on ABP; its CVP trunk then diverged. The likely cause: CardioHydra's
Mamba3 blocks train at the recipe's full rate (1e-3, constant), whereas
every PhysMamba config in the repo trains its Mamba layers at a tenth of
the rate (`MAMBA_LR_SCALE 0.1`) for exactly this reason, and the 128 px
synthetic run, which never collapsed, took 3,000 steps against this run's
4,700. Fix queued (01:10): the same scale for CardioHydra's trunk, and a
rerun once the baseline chain releases the GPU.

| Model (participant 33, last epoch) | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FactorizePhys (Sep, 30 ep) | 0.82 | 0.39 | 11.1 | 2.9 | -3.9 (2.7) | -2.3 (1.4) | -0.2 (0.9) |
| PhysNet (Sep, 30 ep) | 0.72 | 0.40 | 14.5 | 2.8 | -13.4 (4.7) | -1.0 (2.5) | -0.1 (1.1) |
| RhythmFormer (Sep, 30 ep) | 0.69 | 0.42 | 16.8 | 2.5 | -16.0 (3.0) | -0.1 (2.4) | -0.3 (1.0) |
| PhysFormer (Sep, 30 ep) | 0.57 | 0.44 | 16.1 | 2.5 | -14.7 (4.2) | -0.0 (2.5) | -0.3 (1.0) |
| CardioConv (exp 5, 15 ep) | 0.56 | 0.29 | 16.2 | 3.2 | -11.3 (3.9) | -2.5 (2.0) | -0.2 (0.9) |
| CardioHydra (exp 5, 15 ep) | 0.49 | 0.02 | 17.8 | 3.6 | -15.8 (3.2) | -2.8 (2.3) | -3.9 (2.1) |
| CardioHydra, epoch 13 of the same run (for the record) | 0.52 | 0.33 | 16.6 | 3.2 | -14.9 (2.9) | -2.6 (1.9) | -0.2 (0.9) |

## Experiment 5b: CardioHydra rerun with the Mamba trunks at 0.1x rate

Started 02:04 (`runs/dev_cardio/exp5b_cardiohydra_rerun.sh`, run directory
`exp5b_cardiohydra_neckflix_p33_lr01`); identical to the first run except
`MAMBA_LR_SCALE: 0.1`. Epoch 4 scored from the held-out records at 02:30:
ABP CCC 0.58, CVP CCC 0.35, ABP MAE 18.5, CVP MAE 2.84, ABP diastolic
-11.1 (1.9), mean -16.2 (2.8), systolic -28.2 (6.3), CVP mean -2.1 (1.9),
heart rate -0.1 (0.6): the best ABP shape of any of tonight's Neckflix
runs at the same point, and a CVP error below FactorizePhys's.
Epoch 8 (02:50): ABP CCC 0.62, CVP CCC 0.34, ABP MAE 17.8, CVP MAE 2.87,
ABP mean -16.7 (2.8), systolic -30.4 (5.9), CVP mean -2.2 (1.9), heart
rate -0.2 (0.9); the training CVP term is steady at 0.74-0.76 where the
first run's jumped, so the reduced rate holds so far.

Final, epoch 15 (03:50): the training CVP term stayed at 0.74-0.76 to the
end; no divergence. Held out:

| Model (participant 33, last epoch) | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FactorizePhys (Sep, 30 ep) | 0.82 | 0.39 | 11.1 | 2.9 | -3.9 (2.7) | -6.5 | -13.9 | -2.3 (1.4) | -0.2 (0.9) |
| PhysNet (Sep, 30 ep) | 0.72 | 0.40 | 14.5 | 2.8 | -13.4 (4.7) | -4.5 | -20.9 | -1.0 (2.5) | -0.1 (1.1) |
| RhythmFormer (Sep, 30 ep) | 0.69 | 0.42 | 16.8 | 2.5 | -16.0 (3.0) | -8.4 | -22.8 | -0.1 (2.4) | -0.3 (1.0) |
| **CardioHydra, Mamba at 0.1x rate (exp 5b, 15 ep)** | **0.62** | **0.34** | 18.4 | 3.0 | -17.5 (3.0) | -8.1 | -31.7 | -2.4 (1.9) | -0.2 (0.9) |
| PhysFormer (Sep, 30 ep) | 0.57 | 0.44 | 16.1 | 2.5 | -14.7 (4.2) | -1.2 | -35.8 | -0.0 (2.5) | -0.3 (1.0) |
| CardioConv (exp 5, 15 ep) | 0.56 | 0.29 | 16.2 | 3.2 | -11.3 (3.9) | -7.3 | -29.2 | -2.5 (2.0) | -0.2 (0.9) |
| CardioHydra, full rate (exp 5, 15 ep, diverged) | 0.49 | 0.02 | 17.8 | 3.6 | -15.8 (3.2) | -4.5 | -43.2 | -2.8 (2.3) | -3.9 (2.1) |

The reduced rate fixed the divergence outright. The stable CardioHydra is
the third model on ABP shape on this participant, ahead of PhysFormer and
CardioConv and behind FactorizePhys, PhysNet and RhythmFormer; on CVP
shape it is behind the four baselines and ahead of CardioConv; its
heart-rate agreement equals the best. Its level errors are of the baselines'
size except the systolic, which every model under-reads and these two
more than most. The Neckflix acceptance route (beat the best existing
model on participant 33) is therefore still not met after one night; the
gap is in ABP shape (0.62 against 0.82) and in the systolic level.

Masks of the rerun (`epoch_15/visuals/neckflix/P033_S01_R3_45_D_2_240`):
the pulsatility attention lies on the same diagonal vessel band as before
in every channel including depth, the ABP bin takes the band, and the
CVP bin now sits on a few distinct spots at the ends of the band instead
of leaving the neck; on that window the predicted CVP follows the label
beat for beat at the right amplitude and the predicted ABP the rhythm
with the systolic under-read.

How the collapse looks (`exp5_cardiohydra_neckflix_p33/epoch_15/visuals/
neckflix/P033_S01_R3_45_D_2_240`): the pulsatility attention lies on the
same diagonal vessel band CardioConv found and the ABP bin takes all of
it, but the CVP bin and the background bin have moved onto the static
region outside the neck in the top-right corner, in every channel, so the
CVP candidates are the mean of pixels that do not pulse and the predicted
CVP is a slow ramp. The mask moved when the CVP trunk diverged; which of
the two moved first is not resolved by the logs.

## Correction (2026-10-11 10:20)

The "CVP trunk divergence" above was a misreading. The training log shows
the CVP phase mask dying at epochs 13-15: its three coherence priors fell
from 0.0036 / 0.037 / 0.020 (epoch 12) to 7e-23 (epoch 15), so its
pulsatile mass fell below the priors' 1e-6 clamp; the CVP CCC training
term jumped to 0.95 and the background prior was already near zero. The
CVP level terms barely moved (0.74-0.78). The panels' "CVP bin on the
static region outside the neck" is the vanished map's residue drawn at its
own colour scale (`scripts/visualise.py` scales every map to its own
range). The same death happened at epochs 27-28 of the 30-epoch
continuation at a tenth of the Mamba rate (experiment 6a). So the reduced
rate did not "fix the divergence outright": in one run each, the mask
lasted to about 4,000 steps at the full rate and about 8,500 at a tenth.
Mechanism and remedies: "Why the masks collapse" in experiment_6.md.
