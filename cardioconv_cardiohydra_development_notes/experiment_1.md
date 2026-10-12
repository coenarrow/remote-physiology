# Experiment 1 — Brief, starting point, and the "before" baseline

Date: 2026-10-10, 14:45 WAST. Branch `cardiohydra` at cf88a59, working tree
clean. Deadline for working CardioConv and CardioHydra: 2026-10-11 07:00.

## 1. Brief (agreed with the user, 2026-10-10)

Both models keep six stages in this order: rate estimator, wavelet bank,
pulsatility mask, phase masks, kinematics, prediction module (a seventh,
the statistics head, sets each trace's level and spread). CardioConv's
predictor must be convolutional; CardioHydra's must be mamba-ssm (Mamba3).

Fixed by the user:

- Rate estimator: RhythmFormer, trained jointly from scratch. No
  pretraining or checkpoint initialisation (leakage).
- Kinematics stays as it is.
- Training rate target: the fused spectral peak, the same quantity the
  evaluation scores (`src/evaluation/rate.py: rate_of`), not the periodicity
  rate. The evaluation itself is not touched.
- Git: work on branch `cardiohydra`, no worktrees, no commits unless
  essential, keep the diff small and readable; the user merges to main.

Flexible (within the stage's principle): the frame downscale (try none at
128 px); the wavelet scales; the mask networks and every regulariser, as
long as pulsatility reads band power only and the phase stage separates
artery, vein and background; the predictor architectures.

Acceptance:

1. every stage demonstrably working (masks that are not degenerate);
2. pressure predictions that beat the other models, either on Neckflix
   participant 33 against the existing runs or on a synthetic train/test
   split against the same models trained the same way;
3. a visualisation of what the model looked at, checked against the
   anatomical vessel positions, which the synthetic data provides as a
   per-pixel vessel map.

## 2. Starting point (verified this afternoon)

Code: the full pipeline exists only on this branch. The RATE_ESTIMATOR
switch defaults to the spectral estimator; RhythmFormer is one CardioConv
ablation arm. CardioHydra has only ever trained with the spectral estimator.
Every training run so far predates cf88a59, so the committed tree is untested.

Failures, reproduced from the saved checkpoints and loss logs
(runs/neckflix_cardioconv, neckflix_cardioconv_rate, neckflix_cardiohydra_rate,
neckflix_cardiohydra_periodicity):

| Stage | Observed | Cause in the code |
| --- | --- | --- |
| Pulsatility mask | flat constant ~0.05 everywhere; `pulsatility_tv` falls to 1e-8 | the waveform is invariant to the mask's level (masked sum, affine convs, z-score in the head; pooled power area-normalised; mass detached), so only the sparsity prior acts on the level and drives it to zero. Input is raw magnitude: uint8 RGB and uint16 IR/depth land on different scales. |
| Phase masks | >= 0.999 of every pixel is background in every checkpoint; phase priors reach 1e-5 by epoch 16 | every phase prior is minimised by empty trace masks and the background class carries no prior; the one phase-identity feature is the absolute wrapped phase, which rotates in time, so no time-stable arterial/venous distinction exists; coherence features pool all 63 scales across three harmonics. |
| Rate estimator | spectral arm trains to ~10 bpm error; RhythmFormer arm best on the ablation | the ablation (`runs/rate_estimator_ablation`) compares training terms of different kinds; the test-side rate is the predicted waveform's spectral peak, not the estimator's. |

Memory: the field is `(B, C, T, h, w, 63)` complex; at 14x14 that is 0.04 GB
per sample. The Morlet `cmor1.5-1.0` kernel has a 1-sigma half-width of about
0.18 x centre (13 bpm at 70 bpm), so 21 scales 1 bpm apart per harmonic were
redundant.

## 3. Design changes (what will be built tonight)

1. **Rate estimator.** `RATE_ESTIMATOR` defaults to `RhythmFormer` in the
   config class and in every CardioConv/CardioHydra yaml. Rate = sharpened
   spectral centroid of its waveform, trained by `rate_waveform` (1 - lagged
   Pearson to the fused label waveform). Training gate widened from 6 to 8 bpm.
   Rate target = fused spectral peak (`window_rate` calls the evaluation's
   `clean / spectrum / fuse / rate_of`); `periodicity_rate` is deleted.
2. **Wavelet bank.** 3 scales per harmonic at -10, 0, +10 bpm (9 total,
   was 63); coverage is unchanged because of the kernel width, memory is 7x
   smaller. `DOWNSCALE` becomes a config key (default 2 at 128 px input, so a
   64x64 mask grid; 1 is tried if it fits).
3. **Pulsatility mask.** Input magnitude normalised per sample and channel by
   its mean (scale-free across bit depths), `log1p`. Output is a per-frame
   spatial attention: softmax over the pixels of each channel and frame, so it
   sums to one, cannot vanish, and is literally "where the model looks".
   Priors: entropy (concentration; replaces sparsity), spatial TV, temporal
   smoothness.
4. **Phase masks.** Features per harmonic from the complex field: relative
   phase to the pulsatility-weighted mean field of that channel (cos, sin),
   which is time-stable for a periodic pixel and encodes arterial versus
   venous timing; within-harmonic scale coherence; temporal stability and
   spatial coherence of the relative phase. Same separable conv, softmax over
   traces plus background. Priors: `background_pulsatile`, the fraction of
   pulsatile mass assigned to background (the missing counter-force);
   coherence priors mass-normalised under each trace mask; TV and smoothness
   kept.
5. **Kinematics.** Unchanged.
6. **Predictors.** CardioConv: masked area-normalised mean per channel and
   trace, then a per-trace temporal conv stack (two 15-tap layers with ReLU
   and a pointwise readout) replacing the single tap plus pointwise mix.
   CardioHydra: SSMPredictor unchanged. Statistics head unchanged.
7. **Interpretability.** `scripts/visualise.py`: loads a run, runs windows,
   and writes per-channel panels of pulsatility, artery, vein and background
   masks, kinematics speed and heading, predicted versus label waveforms, and
   an animated overlay; on synthetic data it overlays the true vessel map and
   reports the fraction of each mask's pulsatile mass inside artery, vein and
   background. The model keeps its last intermediates only when asked.

## 4. Protocol

- **Synthetic split.** Train on `synthetic_neck_high_snr` + `synthetic_neck_demo`
  (100 stores, 128 px, 30 s each), no hold-out; test on `synthetic_neck`
  (10 stores, base priors, the noisiest set) with `scripts/test.py`, score
  with `scripts/evaluate.py`. One random window per store per epoch, so
  epochs are cheap and many are needed (40 for development runs).
- **Baselines on the same split.** PhysMamba, FactorizePhys, RhythmFormer,
  PhysNet from `configs/default_configs/` (72 px, ABP-CVP-PPG), same epochs.
- **Neckflix.** Participant 33 held out, 20 epochs, both models, compared with
  `runs/neckflix_*` (30 epochs). Local cache is 128 px; the HPC cache is larger,
  so weaker masks here are not proof of failure.
- **Metrics.** Waveform CCC and level errors from the evaluation tables;
  heart-rate agreement; mask-versus-vessel-map mass fractions on synthetic.

## 5. Time plan (WAST)

| When | What |
| --- | --- |
| 14:45 | this brief; "before" baseline started on the GPU |
| 15:00-18:00 | design changes 1-6, smoke tests on a few windows |
| 18:00-19:30 | CardioConv on the synthetic split; visualiser |
| 19:30-21:00 | CardioHydra on the synthetic split |
| 21:00-00:00 | baselines on the synthetic split; fixes |
| 00:00-05:00 | Neckflix P33, both models |
| 05:00-07:00 | evaluation, notes, summary |

## 6. Experiment 1: the "before" baseline

Purpose: the unchanged branch code, with the RhythmFormer estimator the
user chose, on the synthetic split, so the redesign has a before/after and
the mask collapse is shown on data where the vessels are known.

Command:

```bash
uv run python scripts/run.py --datasets synthetic_neck_high_snr synthetic_neck_demo \
  --config configs/experiment_configs/cardioconv-rhythmformer_FS30_W10S1_RGBID_ABP-CVP_H72W72.yaml \
  --epochs 40 --batch-size 4 --num-workers 4 \
  --run-dir runs/dev_cardio/exp1_before_cardioconv_rf_synth
uv run scripts/test.py --checkpoint runs/dev_cardio/exp1_before_cardioconv_rf_synth --test-dataset synthetic_neck
uv run scripts/evaluate.py runs/dev_cardio/exp1_before_cardioconv_rf_synth
```

Results. Training finished 15:05 (40 epochs, ~23 s each, exit 0). Weighted
training losses: ABP CCC 0.72 -> 0.53, CVP CCC 0.98 -> 0.82, rate_waveform
0.20 -> 0.03 (RhythmFormer locks on within five epochs on synthetic data).
The old mask priors move the way the Neckflix runs did: sparsity 0.054 ->
0.040 (per-channel mean mask 0.49 -> 0.36 and falling), pulsatility TV
0.023 -> 0.006, phase TV 0.017 -> 0.005. Test records and scores are run
with the committed code, in a scratch copy of the tree, since the new
modules cannot load this checkpoint; see "Before-baseline test" below.

### Before-baseline test (synthetic_neck, 10 stores, 30 measurements of 10 s)

Run with the committed code from a scratch copy of the tree
(`scripts/test.py` then `scripts/evaluate.py`), epoch 40:

| Signal | CCC | MAE (mmHg) | Mean error of the level (mmHg) |
| --- | --- | --- | --- |
| ABP | 0.27 | 22.1 | +14.3 (systolic +6.9, diastolic +10.2) |
| CVP | 0.11 | 4.4 | -0.8 |

Heart rate of the predicted waveform against the label's: mean error
-4.6 bpm, SD 24.8 bpm. The test set is the base-priors set, noisier than
the two training sets; the model that fit CVP at CCC-loss 0.82 in training
transfers a CVP shape of CCC 0.11 to it. This is the number to beat, with
the masks shown rather than assumed.

### Memory probe of the redesign at 128 px (15:40)

One training step at the full interface (5 channels, 300 frames, 128 px),
random clip, gate forced open, measured with `torch.cuda.max_memory_allocated`:

| Model | Batch | DOWNSCALE | Grid | Peak | Step |
| --- | --- | --- | --- | --- | --- |
| CardioConv (first version) | 2 | 2 | 64x64 | 19.8 GB | 378 s (spilled to host memory) |
| CardioConv (first version) | 4 | 2 | 64x64 | 39.5 GB | 192 s |

The estimate in the brief (0.7 GB per sample for the field) left out what
autograd keeps: both mask networks' hidden activations (16 planes over
``(B C, T, h, w)`` per layer) and every intermediate of the phase features.
Two changes, both made before any training run of the redesign:

1. The mask networks' inputs are computed without autograd. The masks are
   trained by the waveform loss through the pooling weights downstream and by
   their priors; the rate keeps its gradient from the field through the
   statistics head's pooled power. Nothing is lost that the design needs.
2. Both mask networks run under `torch.utils.checkpoint`, so their hidden
   activations are recomputed in the backward pass instead of being kept.

Also decided here: the rate network reads the shrunk clip (what the masks
read), not the raw one; at DOWNSCALE 1 the two are the same.

After the fixes: CardioConv, batch 2, DOWNSCALE 2, 128 px: 3.1 s per step,
10.8 GB peak. Batch 4 and DOWNSCALE 1 both exceed the card and stay out of
tonight's runs.

### The spectral-peak rate target, checked on the synthetic stores (16:40)

The user chose the fused spectral peak as the training target, the quantity
the evaluation scores. Against the generator's true heart rate, on the first
window of every store: fused over ABP, CVP, ECG and PPG (what the loader
builds, since the synthetic stores carry all four) the peak is off by more
than 8 bpm on 2 of 50 stores in each training set and 0 of 10 in the test
set, both failures second-harmonic flips at 42-46 bpm. Fused over ABP and
CVP alone it would be off on 14 of 50. CVP alone reads its second harmonic on
37 of 50. So the target is sound here because the ECG and PPG are fused in;
on Neckflix windows without an ECG the flip rate will be higher.

## 7. Decision log

- 2026-10-10 14:30 user: RhythmFormer default, no pretraining; try no
  downscale at 128 px; "decent" = beat the other models on P33 or on a
  synthetic split; visualisation must be checkable against vessel anatomy
  (synthetic); spectral-peak rate target to match the evaluation; stay on
  branch `cardiohydra`, minimal diff, no commits unless required.
- 2026-10-10 14:45 me: "features in cardiohydra to port" read as the
  kinematic features, already ported; nothing else to port.
