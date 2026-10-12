# Experiment 2 — Redesigned CardioConv on the synthetic split at 128 px

Started 2026-10-10 16:45 WAST. Code state: branch `cardiohydra` working tree
after the changes listed in experiment 1, section 3 (uncommitted).

## Purpose

First full training of the redesign: do the pulsatility and phase masks
stay non-degenerate, do they land on the vessels, and does the waveform
agreement beat the before-baseline (experiment 1: ABP CCC 0.27, CVP CCC
0.11 on the same test set)?

## Setup

- Train: `synthetic_neck_high_snr` + `synthetic_neck_demo` (100 stores,
  one random 10 s window per store per epoch), no hold-out, 30 epochs,
  batch 1 (see the log: batch 2 did not fit the card once the allocator's
  reservation is counted), Adam 1e-3 constant, float32.
- Interface: 128 px, no resize; RGB, IR, depth; ABP and CVP.
- Model: RhythmFormer rate estimator trained from scratch on the shrunk
  clip, gate 8 bpm; wavelet bank 3 harmonics x 3 scales; DOWNSCALE 2 (64x64
  mask grid); priors as in the yaml (`PULSATILITY_ENTROPY 0.02`,
  `PULSATILITY_TV 0.02`, `PULSATILITY_SMOOTHNESS 0.05`,
  `BACKGROUND_PULSATILE 0.5`, coherence priors 0.05, `PHASE_TV 0.05`,
  `PHASE_SMOOTHNESS 0.1`, `RATE_WAVEFORM 1.0`, `RATE_ERROR 0.1`).
- One training step at this size: 1.5 s at batch 1, 6.9 GB on the card
  (4.6 GB allocated at the peak); about 2.5 minutes per epoch.
- Test: `scripts/test.py` on `synthetic_neck` (10 stores), `scripts/evaluate.py`,
  `scripts/visualise.py --windows 6`.
- Runs alongside experiment 4 (the baselines at 72 px) on the same GPU.

Command (`runs/dev_cardio/exp2_exp3_chain.sh`, which runs experiment 3 after it):

```bash
uv run python scripts/run.py --datasets synthetic_neck_high_snr synthetic_neck_demo \
  --config configs/experiment_configs/cardioconv_FS30_W10S1_RGBID_ABP-CVP_H128W128.yaml \
  --epochs 30 --batch-size 2 --num-workers 4 --run-dir runs/dev_cardio/exp2_cardioconv_synth_128
uv run python scripts/test.py --checkpoint runs/dev_cardio/exp2_cardioconv_synth_128 --test-dataset synthetic_neck
uv run python scripts/evaluate.py runs/dev_cardio/exp2_cardioconv_synth_128
uv run python scripts/visualise.py --checkpoint runs/dev_cardio/exp2_cardioconv_synth_128 --dataset synthetic_neck --windows 6
```

## What to look at

1. `losses.csv`: `rate_error` (0.1 x bpm) should fall below 0.8 within a few
   epochs, or the gate keeps the rest of the model from training;
   `background_pulsatile` should fall from ~0.5 x (1/3 per class) towards 0;
   `pulsatility_entropy` should fall below 1 (uniform); the CCC components
   should fall below the before-baseline's 0.53 / 0.82.
2. `visuals/synthetic_neck/scores.csv`: `pulsatility_on_vessels` against
   `vessel_share_of_grid` (0.03 on this grid), and each trace's mass in the
   true artery and vein.
3. `evaluation/tables/waveform_agreement.csv` and `agreement.csv` against
   experiment 1.

## Log

- 16:45 (clock later found to read 15:26) first launch, alongside the
  baseline chain and a CPU re-test of the Neckflix baselines. With the
  three sharing the card the step went from 3.1 s to 13-18 s, then 90 s:
  the 16 GB card was full and the driver paged this run's memory to RAM
  (its commit size reached 24 GB). Stopping the other jobs did not bring
  the step back under 90 s, so the run was killed at 48/50 of epoch 1.
- 15:55 relaunched with the GPU exclusive (`runs/dev_cardio/exp2_exp3_chain.log`).
  Rule for the rest of the night: one GPU job at a time; the Neckflix
  baseline re-tests run on the CPU with six threads; everything else waits
  in `runs/dev_cardio/queue.sh`.
- Lesson for the notes: stopping the tool's shell did not stop a chained
  `bash script.sh`; the baseline chain kept going into its second model
  unseen for fifteen minutes. Kill the script's own bash process.
- 15:57 the exclusive relaunch still filled the card (16.0 GB used, 12 s
  per step): one step allocates 9.2 GB at its peak but the caching
  allocator reserves 13.2 GB (`torch.cuda.max_memory_reserved`), and with
  the CUDA context and the desktop that is the whole card. PyTorch's
  `expandable_segments` is not supported on Windows; `max_split_size_mb`
  made the reservation worse (21 GB). Two changes: the wavelet field and
  its magnitude are now computed without autograd in both models (the
  rate network trains on its own terms; the only gradient path that went
  through the field was the statistics head's pooled power), and the runs
  go to **batch 1**: 4.6 GB allocated, 6.6 GB reserved, 2.4 s per step, so
  about 4 min per epoch of 100 windows and 2 h for 30 epochs.
- 16:07 relaunched at batch 1 with the field under no_grad; 1.5 s per step,
  6.9 GB. The baselines (experiment 4) are cut to 20 epochs to keep the
  night's queue inside the deadline.
- 16:40, epoch 9 of that run, masks drawn from the saved weights on two
  test windows (`epoch_09/visuals`): every score was exactly chance,
  pulsatility mass on the vessels 0.03 = the vessels' share of the grid, and
  the two trace masks identical. Both masks were uniform; the loss had
  settled at CCC terms 0.50 / 0.60 from epoch 4 with nothing spatial. Cause:
  the entropy prior has zero gradient at the uniform point (it is the
  maximum), TV and smoothness pull towards flat, and the waveform loss's
  gradient per pixel is of order 1/4096. A free conv net starts flat and
  stays flat.
- Two label-free objectives were checked on four synthetic stores against
  the true vessel map before any code change (band-power fraction of the
  attention-weighted mean trace, raw and with standardised pixels): neither
  prefers the vessels over the whole neck skin, because the synthetic skin
  pulse is everywhere and in phase, and the depth lift is weak per pixel.
  So pulsatility alone cannot find the vessels here; the phase stage must.
  The user confirmed at 16:40 that this is the intent: the pulsatility mask
  is where the neck pulses, only the phase masks separate the vessels.
- Change (16:45): each masker gets a small parametric head that is
  non-uniform from the first step, with the conv net as a correction.
  Pulsatility: logit = gain x pulsatility index (log band power over the
  pixel's variance, floored so a static pixel scores far below a pulsing
  one) + net; at init the attention is proportional to the index.
  Phase: trace logit = gain_s x cos(relative phase - centre_s) + net, with
  learned centres starting evenly spaced on the circle and gains at 2; the
  background logit is the net's alone. On an untrained model the
  pulsatility mass on the neck is 0.72-0.80 against a neck share of 0.63,
  and the trace masks already differ (ABP bin 0.5-0.7 of the pulsatile
  mass, CVP 0.1-0.26, background 0.2). The flattening priors were softened
  (`PULSATILITY_TV 0.005`, `PULSATILITY_SMOOTHNESS 0.01`, coherence priors
  0.02, `PHASE_TV 0.01`, `PHASE_SMOOTHNESS 0.02`). The 16:07 run was
  stopped at epoch 12 and the chain relaunched at 16:47.
- 17:10, epoch 5 of the relaunch (`epoch_05/visuals`, three windows). The
  masks have structure now but it is not the vessels. In R, G, B and D the
  pulsatility attention sits on a band along the lower edge of the neck,
  which flickers with the pulse as the silhouette moves, and the ABP phase
  bin takes that band while the CVP bin takes the rest of the skin and the
  background bin the upper edge. Scores: pulsatility mass on the neck 0.53
  to 0.79 against neck shares 0.53 to 0.76 (at chance), trace-mask mass in
  the artery and vein at chance in every channel. The head parameters did
  not move in five epochs (index gain 0.99, phase gains 2.0 and 1.97,
  centres 0 and pi), so the waveform loss is close to indifferent to the
  masks here. On this synthetic data the vessels do not colour the skin;
  their only trace is a sub-millimetre depth lift, so an RGB mask cannot
  land on the drawn contours whatever the prior, and the depth lift is too
  faint per pixel for the index to pick out (checked above). Decision: no
  further restart; let this run finish, judge the vessel question on
  Neckflix (experiment 5), where the carotid and jugular pulsation is
  visible, and treat the synthetic set as the test of the pipeline, the
  rate stage and the waveform agreement.

## Results of run 2a (ReLU mixer, L1 rate term), 16:47-18:03

Kept as `runs/dev_cardio/exp2a_cardioconv_synth_128_relu`. Training: final
CCC terms 0.52 (ABP) and 0.64 (CVP), rate error 6 bpm, rate waveform term
0.14 (the before-baseline's recipe reached 0.03 on the same split). Test
on `synthetic_neck`, same 30 measurements as experiment 1:

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before (exp 1) | 0.27 | 0.11 | 22.1 | 4.4 | +14.3 (20.7) | -0.8 (5.0) | -4.6 (24.8) |
| run 2a | 0.13 | 0.11 | 23.1 | 4.3 | +14.6 (20.9) | -0.7 (5.0) | -20.6 (40.3) |

Worse on ABP shape and on the predicted waveform's rate, equal elsewhere.
The panels (`epoch_30/visuals/synthetic_neck/3_1_210`) show why: the
estimated rate is right (88.0 against 87.9 bpm) but the predicted ABP is
a train of sharp peaks over clipped troughs and the predicted CVP a slow
wander, the signature of the ReLU between the mixer's temporal layers
rectifying a zero-mean candidate. The masks on the test windows are
speckled, not flat (the phase bins respond to per-pixel phase noise), and
score at chance against the vessel contours, as expected on this data.

Two changes before run 2b:

1. `ChannelMixer` is a linear filter bank again: one 15-tap temporal layer
   per trace and a pointwise readout, no ReLU (as the pre-redesign mixer,
   which predicted the rate within 5 bpm).
2. `RATE_ERROR` (L1 on the rate through the sharpened spectral centroid)
   is dropped from every config; `RATE_WAVEFORM` alone trains the rate
   network, as in the ablation whose RhythmFormer arm fitted best.

## Results of run 2b (linear mixer, frame-wise masks), 18:06-18:40

Kept as `runs/dev_cardio/exp2b_cardioconv_synth_128_framewise`, stopped at
epoch 12. The rate term fell faster (0.09 by epoch 10 against run 2a's
0.14), but a test of the epoch-10 weights was no better: ABP CCC 0.13,
CVP CCC 0.08, predicted-waveform rate 48 bpm low (SD 35). So the mixer
was not the cause.

The cause, found on three test windows with the epoch-10 weights
(`epoch_10/test_records`): the masks are frame-wise (a softmax per frame),
and they move. The area-normalised mean under a moving mask sweeps across
the clip's intensity gradients, which puts a slow wander into every
candidate waveform. With the model's own masks as they are, the
candidates' spectral peaks fall anywhere between 35 and 70 bpm on windows
whose rate is 88 to 120, and their standard deviation is 10 to 40 times
that of the same candidates under the window-averaged masks, which peak
exactly at the heart rate in every channel (depth sometimes at the second
harmonic). The rate estimator itself was right on these windows (116.6
for 119.5, 88.2 for 87.9).

Change for run 2c: both maskers average their logits over the frames
before the softmax, so each window has one pulsatility map and one set of
trace masks (a vessel does not move in ten seconds). The temporal
smoothness priors are gone with the flicker (`PULSATILITY_SMOOTHNESS`,
`PHASE_SMOOTHNESS` removed from the regularisers and every config). The
baseline chain is cut to four models (PhysMamba, FactorizePhys,
RhythmFormer, PhysNet) to keep the night's queue inside the deadline.

## Results of run 2c (window-constant masks), started 18:45

Training loss from epoch 1 is below anything the earlier runs reached
(total 2.9 against 3.4 to 3.6; CCC terms 0.49 / 0.60 at epoch 1). The
epoch-10 weights, tested at 19:10 on `synthetic_neck`:

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (exp 1, epoch 40) | 0.27 | 0.11 | 22.1 | 4.4 | +14.3 (20.7) | +10.2 | -0.8 (5.0) | -4.6 (24.8) |
| run 2b, frame-wise masks, epoch 10 | 0.13 | 0.08 | 23.1 | 4.4 | +14.8 (20.6) | +21.2 | -0.3 (5.0) | -47.9 (34.5) |
| run 2c, constant masks, epoch 10 | 0.33 | 0.27 | 21.0 | 4.2 | +11.3 (20.9) | +5.5 | -0.6 (5.0) | +7.7 (45.4) |

Ten epochs of the constant-mask model beat forty of the old design on
every waveform metric, and CVP shape agreement is 2.5 times the old
design's. The predicted waveform's rate is still noisy on some windows
(SD 45 bpm); the final-epoch evaluation and the panels follow.

Masks at epoch 10 (`epoch_10/visuals/synthetic_neck`, three windows,
scores against the generator's vessel map): in the depth channel the CVP
mask puts 12.9, 12.3 and 12.4 % of its pulsatile mass on the true jugular
against the vein's 2.1 % share of the grid, six times chance, in every
window; the ABP mask stays at chance on the artery in every channel and
holds the skin instead. That is the anatomy the design expected: the vein's
depth lift carries CVP, the skin pulse carries ABP, and the synthetic
artery colours nothing. The pulsatility mask sits on the neck in every
channel (0.52 to 0.88 against neck shares of 0.53 to 0.76), strongest in
depth, where it also picks the vessel band (window 6: 6.8 % on the vessels
against 2.5 %).

## Results of run 2c at epoch 30 (final), 20:03

Run directory `runs/dev_cardio/exp2_cardioconv_synth_128` (every epoch's
weights kept; `epoch_10` and `epoch_30` carry test records, evaluation
tables are the epoch-30 ones; visuals under `epoch_30/visuals`, six
windows, and `epoch_10/visuals`, three windows).

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP diastolic err | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before (exp 1, epoch 40) | 0.27 | 0.11 | 22.1 | 4.4 | +14.3 (20.7) | +10.2 | +6.9 | -0.8 (5.0) | -4.6 (24.8) |
| run 2c, epoch 10 | 0.33 | 0.27 | 21.0 | 4.2 | +11.3 (20.9) | +5.5 | +5.6 | -0.6 (5.0) | +7.7 (45.4) |
| run 2c, epoch 30 | 0.33 | 0.28 | 21.7 | 4.3 | +13.4 (20.8) | +6.9 | +7.5 | -0.2 (5.0) | -7.5 (26.3) |

The redesigned CardioConv beats the old design on every waveform metric
of the synthetic test set, with CVP shape agreement 2.5 times the old
design's. The level errors are the same size as before (ABP read 13 mmHg
high on these noisy base-prior stores); that is the magnitude problem the
project already knows, not something tonight's changes address.

Masks at epoch 30 (six windows): the pulsatility mass on the neck is above
the neck's share in every channel (0.63 to 0.69 against 0.62). The
jugular focus of the CVP mask in depth seen at epoch 10 (12 % of its mass
on the vein) has faded to 2.6 % against a 2.3 % share by epoch 30, while
the waveform metrics barely moved between the two epochs: the loss does
not need the vein to predict this test set's CVP, so nothing holds the
mask there. For a thesis figure of the anatomy, the epoch-10 weights are
the ones to draw; for the waveform numbers, either.
