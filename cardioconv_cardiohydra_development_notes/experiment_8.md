# Experiment 8 — Level checks, kinematics in CardioConv, full-resolution masks, fit probe (2026-10-12)

Asked by the user on 2026-10-12, in four ordered steps:
1. add level checks to the evaluator;
2. add kinematics to CardioConv, always on;
3. give the rate estimator and the wavelet transformer a downscale factor each,
   with the field upscaled back;
4. run one fit probe ("literally not even 1 epoch") to find the minimum
   downsizing each model can use on the 16 GB dev card, shrinking only the rate
   estimator and leaving the rest at input resolution.

Nothing was trained. The code is on branch `cardio-kinematics-fullres` off main
1db060e, uncommitted. The probe scripts, configs and logs are in
`runs/dev_cardio/exp8_probes/`.

## What changed

| Change | Files |
| --- | --- |
| **Level tracking** in `agreement.csv`, beside the ISO statistics: `ref_sd`, `pred_sd`, `r_between`, `slope_between` (per-subject means, k >= 3), `r_within`, `slope_within` (values centred per subject) and `tracking_note`. Descriptive only. A slope of 1 tracks the level; 0 is a constant predictor. | `src/evaluation/agreement.py` |
| **CVP max and CVP min**, the 95th and 5th percentiles of each 10 s measurement. A CVP beat has three peaks (the a, c and v waves), so ABP's beat detection does not transfer. The CSVs and Bland-Altman figures come for free. | `src/evaluation/measurements.py` |
| **Kinematics moved to `_shared_modules/`**: `Kinematics`, `collapse_harmonics`, `phase_gradient`, `pooled_speed`, the new `kinematic_features` and `EPS`. `cardiohydra/kinematics.py` and `cardiohydra/utils.py` are deleted. | `_shared_modules/{kinematics,utils}.py` |
| **CardioConv runs the kinematics, always**, by CardioHydra's two routes. (1) `pooled_speed` goes to the statistics head (`n_extra = C*S*N = 30`, head input 61 -> 91). (2) Per frame, the 15 kinematic features per channel are pooled under each trace's mask beside the intensity candidate (`ChannelPredictor` returns `(B, C, 1+F, T, S)`) and enter the mixer through a new bias-free `kinematic` conv, initialised to zero, so the model starts as the old filter bank. +39,840 parameters. | `cardioconv/{cardioconv,channel_predictor,channel_mixer}.py` |
| **`RATE_DOWNSCALE` and `WAVELET_DOWNSCALE` replace `DOWNSCALE`** (both default 2). Each estimator shrinks the clip it reads. The wavelet transformer shrinks, transforms and upscales the field back (bilinear on the real and imaginary parts, `cardioconv.utils.upscale`). Both maskers, the kinematics, the predictors and the statistics head now read the full clip and the full-size field. The capture key `small` is now `clip`. | `cardioconv/*`, `cardiohydra/cardiohydra.py`, `src/model_config.py`, `src/models.py`, the 12 cardio YAMLs, the template, the README, `scripts/visualise.py` |

## Verification

All checks below passed. The scripts are in `runs/dev_cardio/exp8_probes/`.

- **Lint.** Ruff passes on the touched files. The repo-wide count is still 196,
  all in the legacy PhysHydra files.
- **Configs.** All 12 cardio configs load with both keys. A config that names
  `DOWNSCALE` is refused: "MODEL has unknown key(s) ['DOWNSCALE']".
- **CardioHydra parity across the kinematics move.** Exp7b epoch 12 ran on 3
  P33 windows, checked by SHA-1 of every output and captured tensor:
  - The two runs on the untouched tree were identical.
  - After the move, all 33 tensors were bitwise identical; the only addition
    is the new `pooled_speed` capture.
  - This check was taken before step 3, after which 7b no longer rebuilds.
- **CPU smoke, CardioConv `(2, 5, 150, 16, 16)` at factors 2/2 and 1/1.**
  - The output is `(2, 2, 150)`, and all 44 gradient tensors are finite.
  - The head input is 91.
  - The kinematic weight is all zero at the start, with a nonzero gradient.
  - The masks are at the full 16x16 size.
  - The captured keys include `clip`, `phase_speed`, `direction_x`,
    `direction_y` and `pooled_speed`.
- **Evaluator,** run on scratch copies of exp2 (CardioConv synthetic, epoch 30)
  and exp7b (CardioHydra Neckflix P33, epoch 12).
  - Every existing column of `agreement.csv` matches the stored table exactly
    (max diff 0.0).
  - `CVP_max.csv`, `CVP_min.csv` and their Bland-Altman figures are written.
  - Exp2 shows the "predicts the average" failure the columns were built for
    (table below).
  - On P33, every between-subject value is NaN with the note "fewer than three
    subjects".

| exp2, synthetic test set (n 30, k 10) | ref_sd | pred_sd | r_between | slope_between | r_within | slope_within |
| --- | --- | --- | --- | --- | --- | --- |
| ABP mean | 20.85 | 0.34 | 0.19 | 0.002 | 0.76 | 0.61 |
| ABP systolic | 25.99 | 2.19 | 0.42 | 0.029 | 0.05 | 0.03 |
| ABP diastolic | 19.25 | 2.67 | -0.09 | -0.010 | 0.20 | 0.05 |
| CVP mean | 5.03 | 0.07 | 0.37 | 0.003 | 0.57 | 0.60 |
| CVP max | 5.10 | 0.24 | 0.62 | 0.027 | -0.39 | -0.49 |
| CVP min | 5.16 | 0.18 | 0.27 | 0.008 | 0.07 | 0.07 |

The between-subject slopes are about 0: the model predicts one level for
everyone. The synthetic frames carry no absolute pressure (`render.py`
renders `compliance * (P - mean)`), and the test subjects share seeds and
traces with training stores, so no synthetic result can show level tracking.

**End-to-end smoke.** Each model ran `scripts/run.py` for 1 epoch on 2
windows (Neckflix 72 px, P33 held out), then `scripts/visualise.py` on 1
window.
- Both exited 0. `losses.csv` is finite and there is no `nonfinite_steps/`.
- The waveform terms are 0 because the untrained rate gate let no window
  through, as expected.
- The panels are drawn at the full 72x72 grid.
- Untrained, the pulsatility map is a speckle of single pixels and edges.
  This is the full-resolution consequence below: the index divides a smooth,
  upscaled band power by a noisy per-pixel variance.

## Fit probe

One training step per probe (`tools/memory_report.py`, wrapped by
`fit_probe.py`). The wrapper opens the rate gate (`rate_tolerance_bpm = inf`),
so every sample reaches the masks: the worst case. An untrained gate passes
one sample, which would understate a batch-2 step by about half.

**"Fits"** means peak reserved plus the 0.5 GiB CUDA-context allowance is
below the card's free memory before the probe (14.9 GB of 16.0). On Windows
an overfull step does not fail: WDDM pages it to system RAM. That is why
every probe below exited 0 and none ran out of memory.

**Settings.**
- Neckflix: 72 px, batch 2, the P33 setting of every comparison.
- Synthetic: 128 px, batch 1, experiments 2-4.

Rung 0 is today's design (one `DOWNSCALE` 2 shrink, masks on the shrunk
grid), measured on the untouched tree before any edit. Every other rung has
the masks at input resolution.

| Model | Setting | Rate / wavelet | Params | Peak alloc | Peak reserved | Per run | Fits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CardioConv | Neckflix 72, b2 | rung 0 (DOWNSCALE 2) | 3,365,541 | 2.6 GB | 3.8 GB | 4.3 GB | yes |
| CardioConv | Neckflix 72, b2 | 2 / 1 | 3,405,381 | 8.7 GB | 13.2 GB | 13.7 GB | **yes** |
| CardioConv | Neckflix 72, b2 | 1 / 1 | 3,405,381 | 10.2 GB | 14.8 GB | 15.3 GB | no |
| CardioConv | Synthetic 128, b1 | rung 0 (DOWNSCALE 2) | 3,365,541 | 4.2 GB | 6.1 GB | 6.6 GB | yes |
| CardioConv | Synthetic 128, b1 | 2 / 1 | 3,405,381 | 13.7 GB | 18.0 GB | 18.5 GB | no |
| CardioConv | Synthetic 128, b1 | 4 / 1 | 3,405,381 | 13.1 GB | 17.4 GB | 17.8 GB | no |
| CardioConv | Synthetic 128, b1 | 2 / 2 | 3,405,381 | 13.7 GB | 18.9 GB | 19.4 GB | no |
| CardioHydra | Neckflix 72, b2 | rung 0 (DOWNSCALE 2) | 4,279,813 | 2.6 GB | 4.1 GB | 4.6 GB | yes |
| CardioHydra | Neckflix 72, b2 | 2 / 1 | 4,279,813 | 8.7 GB | 14.3 GB | 14.8 GB | **yes, by 0.1 GB** |
| CardioHydra | Neckflix 72, b2 | 1 / 1 | 4,279,813 | 10.2 GB | 15.9 GB | 16.4 GB | no |
| CardioHydra | Synthetic 128, b1 | rung 0 (DOWNSCALE 2) | 4,279,813 | 4.2 GB | 6.5 GB | 7.0 GB | yes |
| CardioHydra | Synthetic 128, b1 | 2 / 1 | 4,279,813 | 13.7 GB | 19.7 GB | 20.2 GB | no |
| CardioHydra | Synthetic 128, b1 | 4 / 1 | 4,279,813 | 13.1 GB | 19.0 GB | 19.5 GB | no |
| CardioHydra | Synthetic 128, b1 | 2 / 2 | 4,279,813 | 13.7 GB | 20.6 GB | 21.1 GB | no |

### Answer

**Neckflix at 72 px, batch 2: the minimum is rate 2 / wavelet 1, for both
models.**
- With the rate estimator shrunk by 2 and everything else at input
  resolution, CardioConv fits with 1.2 GB to spare.
- CardioHydra fits with 0.1 GB to spare, which is too tight to trust: any
  other process on the card would push it into paging.
- Not shrinking the rate estimator either (1/1) costs another 1.5 GB, and
  neither model fits.

**Synthetic at 128 px, batch 1: nothing fits.**
- No factor helps, because with the field upscaled back the stages after
  the transform run at 128x128 whatever the factors are.
- Shrinking the rate network further (4/1) saves only 0.7 GB.
- Shrinking the transform (2/2) does not help at all. Its upscaled field is
  the same size, and the shrink-and-upscale adds a little.
- The memory is in the full-size masks, kinematics and pooling: allocations
  went 4.2 -> 13.7 GB against rung 0.
- At 128 px, the levers left are:
  1. a smaller input size;
  2. keeping the masks on a shrunk grid (today's design);
  3. a leaner full-size path, for example computing the kinematics' 15
     feature planes per harmonic instead of all at once, or not keeping
     `power` beside the field.

  Batch is already 1.

**Cost against today's design.** At 72 px the full-resolution masks take about 3.2x
the per-run memory of rung 0. A 4x larger
grid explains most of that. Step time was not measured: these probes take one
step and do not time it.

## Consequences for the first full-size training run (not acted on)

- **Both models' numbers change.** The masks and kinematics now live at input
  size, and no earlier run is comparable as-is.
- **The spatial priors cover a smaller area.** `PULSATILITY_TV`, `PHASE_TV`
  and the 3x3 coherence kernel now cover half the physical neighbourhood they
  did at factor 2. Their weights may need retuning.
- **The pulsatility index changes balance.** It divides the upscaled, smooth
  band power by the noisier per-pixel variance of the full clip.
- **The old checkpoints no longer rebuild.** Every checkpoint before this
  change (7b/7c and every CardioConv run) needs a 1db060e checkout; see
  `runs/dev_cardio/code_snapshots/README.md`.
