# CardioConv and CardioHydra, 2026-10-10 overnight: summary

**2026-10-15 (experiment 14, `experiment_14.md`):** the final local tuning
for the 200 px cluster run, 24 h on the dev card, P033 held out, 10 epochs
an arm on the `hpc_configs_200` recipe. CardioHydra ships as it stood: four
arms (ABP max/min weights 0.2, Mamba trunks 0.3x, pulsatility TV 0.02, LR
5e-4) all scored below the baseline, whose best P033 reading is ABP CCC
0.68 / r 0.85 and CVP 0.40 / 0.61 at epoch 4 (r 0.89 by epoch 8), second
to FactorizePhys's 0.78 / 0.43 at 200 px on this fold and ahead of every
other sweep model, which predict nothing on most of P030-P039 at 200 px.
CardioConv ships with `TRACE_INDEPENDENT: false` (every trace mixed from
every candidate): +0.06 ABP r and +0.05 CVP r on P033, where a wider and
longer mixer (new `MIXER_TAPS` / `MIXER_HIDDEN` switches) changed nothing.
The ABP level is still read 15-18 mmHg low with the pulse 20 mmHg too
narrow; heavier amplitude weights lift the mean, not the spread, and cost
shape (experiment 6b again). Masks were the baseline's in every arm.
On the confirmation fold P035, where every sweep model scores ABP CCC
under 0.10, CardioHydra reaches ABP CCC 0.86 / r 0.93 (MAE 7.3 mmHg, level
+6) and CVP 0.40 / 0.61 at epoch 10: the best ABP reading of any model on
any fold in this work.

**2026-10-14, later (experiment 13, `experiment_13.md`):** local tuning on
the HPC recipe (200 px, rate 4 / wavelet 3), P033, 10 epochs an arm. The
phase masker's background bin now follows coherence, `PHASE_GAIN (1 - 2 R)`
instead of 0, so a wandering-phase pixel is background by default and a
coherent one is not taxed for it; `background_pulsatile` removed. The
CardioConv control under the old rule is run 1; both baselines rerun under
the new rule. Result on P033 at 10 epochs: CardioConv ABP CCC 0.28 / r
0.52, CVP 0.18 / 0.31; **CardioHydra ABP 0.64 / r 0.90, CVP 0.40 / 0.60**,
the best cardio result so far, on the same masks.

**2026-10-14 (experiment 12, `experiment_12.md`):** the coarse-grid design
(masks on the `WAVELET_DOWNSCALE` grid) scores the 11b weights on P033 at
0.507 / 0.316 against 0.51 / 0.31 at full resolution, with no retraining;
at 200 px it fits an A100 at batch 2 a card. The P033 tuning job
(`.slurm_scripts/Neckflix_CardioTuning_P033_4GPU.slurm`,
`configs/hpc_tuning_p33/`) smoked locally for both models.

**2026-10-12 (experiment 8, `experiment_8.md`):**
- **Changes.**
  - The evaluator scores level tracking and CVP max/min.
  - CardioConv runs the kinematics.
  - `RATE_DOWNSCALE` and `WAVELET_DOWNSCALE` replace `DOWNSCALE`, and the
    masks work at input resolution.
- **Fit probe.** At Neckflix 72 px, batch 2, both models fit with rate 2 /
  wavelet 1, CardioHydra only by 0.1 GB. At 128 px nothing fits.
- **Not trained yet.**

**Superseded in part on 2026-10-11 afternoon (experiment 7):** the masks
described below collapsed under long training (end of `experiment_6.md`)
and were redesigned with bounded corrections (`experiment_7.md`): no mask
can vanish or shrink to a point any more, and CardioHydra trained on the
new masks keeps every trace alive; at 12 epochs it trades ABP shape for CVP
shape against the old design before its collapse (the last four rows of
the Neckflix table). The
working tree now holds that design; the code that produced every result in
this summary is archived in `runs/dev_cardio/code_snapshots/v0_*` (it was
never committed), and the "What was built" rows for the two masks and the
diff guide describe it, not the current tree.

Written for the thesis record; the per-experiment notes hold the detail
and every number here is copied from an `evaluation/tables/` file or a
`scores.csv` under `runs/dev_cardio/`. Code: branch `cardiohydra`,
uncommitted working tree on top of cf88a59 (the diff is 26 tracked files,
about +570/-480 lines, plus this notes directory, two 128 px configs, two
experiment-6 configs and `scripts/visualise.py`; the staged edit inside the
Neckflix cacher submodule predates this work and is not part of it).

## What was built

Both models keep the six stages in order (rate estimator, wavelet bank,
pulsatility mask, phase masks, kinematics [CardioHydra only], prediction
module) plus the statistics head for level and spread. Changes made
tonight, each after a measurement that showed the previous form failing:

| Stage | Change | Why (experiment) |
| --- | --- | --- |
| Rate estimator | RhythmFormer by default, trained from scratch on the shrunk clip with the waveform-correlation term only | user's choice; the L1 rate term left the fit worse (exp 2a vs 2b) |
| Wavelet bank | 3 scales per harmonic (was 21); field computed without autograd | memory: the Morlet kernel's width makes 1 bpm steps redundant; the field was the largest retained tensor (exp 2 memory probes) |
| Pulsatility mask | softmax over pixels, **one map per window**, logit = gain x pulsatility index + conv correction; entropy and TV priors | the sigmoid mask collapsed to a constant in every earlier run; a free net stayed flat; frame-wise masks put a slow wander into the masked mean (exp 2b -> 2c) |
| Phase masks | relative phase to the channel's pulse; trace logit = gain x cos(phase - centre) + conv correction, one map per window; `background_pulsatile` prior makes the traces claim what pulses; coherence priors mass-normalised | the earlier phase masks were all background in every checkpoint, every prior being minimised by empty masks |
| Kinematics | unchanged | user's instruction |
| CardioConv predictor | area-normalised masked mean per channel and trace, then a linear temporal conv filter bank per trace (no ReLU) | a ReLU rectified the zero-mean candidates (exp 2a) |
| CardioHydra predictor | unchanged Mamba3 trunks over mask-pooled features (mask share now the share of pulsatile pixels); the trunks train at `MAMBA_LR_SCALE` 0.1 of the rate | at the full rate CardioHydra's CVP mask died after about 4,000 Neckflix steps (exp 5, first misread as a trunk divergence), at 0.1x after about 8,500 (exp 6a) |
| Training target | the fused spectral peak, as the evaluation scores it | user's choice; flips on 2 of 50 synthetic stores, all at 42-46 bpm (exp 1) |
| Interpretability | `scripts/visualise.py`: per-channel panels of the masks with the synthetic vessel map as contours and mask-versus-vessel scores; an animated overlay per window | acceptance criterion 3 |

Memory on the 16 GB card at 128 px, 64x64 mask grid: batch 1, 4.6 GB
allocated, 6.6 GB reserved, 1.5 s per step (CardioConv), 1.7 s
(CardioHydra); batch 2 reserves 13 GB and pages. On Neckflix at 72 px
(36x36 grid), batch 2, 5.7 GB.

## Synthetic split (train high_snr + demo, 100 stores; test synthetic_neck, 10 stores)

| Model | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CardioConv, old design (exp 1, 72 px, 40 epochs) | 0.27 | 0.11 | 22.1 | 4.4 | +14.3 (20.7) | -0.8 (5.0) | -4.6 (24.8) |
| **CardioConv, redesign (exp 2c, 128 px, 30 epochs)** | **0.33** | **0.28** | 21.7 | 4.3 | +13.4 (20.8) | -0.2 (5.0) | -7.5 (26.3) |
| **CardioHydra, redesign (exp 3, 128 px, 30 epochs)** | **0.33** | 0.22 | **21.1** | 4.4 | +12.5 (20.8) | -0.8 (5.0) | -4.9 (24.5) |
| CardioHydra, epoch 10 of the same run | 0.38 | 0.20 | 20.3 | 4.4 | +11.3 (20.9) | -0.5 (5.0) | +0.2 (36.8) |
| FactorizePhys (exp 4, 72 px, 20 epochs) | 0.25 | 0.18 | 22.6 | 4.4 | +11.2 (20.9) | +0.4 (5.0) | +9.5 (23.6) |
| RhythmFormer (exp 4) | 0.21 | 0.09 | 21.4 | 4.2 | +14.3 (20.4) | -2.0 (5.3) | -17.1 (25.1) |
| PhysMamba (exp 4) | 0.08 | 0.14 | 26.5 | 4.2 | +22.9 (20.7) | -2.7 (4.4) | -19.1 (40.9) |
| PhysNet (exp 4) | 0.05 | 0.01 | 32.0 | 4.2 | +30.0 (20.6) | -0.9 (5.0) | -25.0 (25.8) |

Both redesigned models beat every baseline and the old design on the
shape of both traces, and CardioHydra has the lowest ABP error; the level
errors are of one size for every model (the magnitude problem the project
already knows). Caveat: the baselines ran at their 72 px paper interface
with a third of the gradient steps (experiment 4 has the details).

Masks on the synthetic test windows: the pulsatility mask sits on the neck
above the neck's share of the grid in every channel for both models
(CardioHydra 0.64-0.76 against 0.62). In the depth channel CardioHydra's
attention puts 8 % of its mass on the true vessels against a 2.3 % share,
and CardioConv's CVP bin at epoch 10 put 12 % of its pulsatile mass on the
true jugular against 2 %, in every window drawn, with the ABP bin on the
skin. On this generator the vessels do not colour the skin and lift it only
sub-millimetre in depth, so no RGB mask can find them (checked against the
vessel map before any training, experiment 2); the anatomy question is
answered on Neckflix.

## Neckflix, participant 33 held out (72 px, 15 epochs, batch 2; baselines 30 epochs, batch 4)

| Model | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FactorizePhys (Sep) | **0.82** | 0.39 | **11.1** | 2.9 | **-3.9 (2.7)** | **-13.9** | -2.3 (1.4) | -0.2 (0.9) |
| PhysNet (Sep) | 0.72 | 0.40 | 14.5 | 2.8 | -13.4 (4.7) | -20.9 | -1.0 (2.5) | -0.1 (1.1) |
| RhythmFormer (Sep) | 0.69 | 0.42 | 16.8 | 2.5 | -16.0 (3.0) | -22.8 | -0.1 (2.4) | -0.3 (1.0) |
| PhysFormer (Sep) | 0.57 | **0.44** | 16.1 | **2.5** | -14.7 (4.2) | -35.8 | -0.0 (2.5) | -0.3 (1.0) |
| CardioConv (exp 5) | 0.56 | 0.29 | 16.2 | 3.2 | -11.3 (3.9) | -29.2 | -2.5 (2.0) | -0.2 (0.9) |
| CardioHydra (exp 5, last epoch; the CVP mask died at epochs 13-15) | 0.49 | 0.02 | 17.8 | 3.6 | -15.8 (3.2) | -43.2 | -2.8 (2.3) | -3.9 (2.1) |
| CardioHydra, epoch 13 of that run (for the record) | 0.52 | 0.33 | 16.6 | 3.2 | -14.9 (2.9) | -40.3 | -2.6 (1.9) | -0.2 (0.9) |
| **CardioHydra rerun, Mamba trunks at 0.1x rate (exp 5b, 15 epochs)** | **0.62** | 0.34 | 18.4 | 3.0 | -17.5 (3.0) | -31.7 | -2.4 (1.9) | -0.2 (0.9) |
| CardioHydra continued to 30 epochs (exp 6a, last epoch; the CVP mask vanished at epochs 27-28) | 0.69 | 0.01 | 17.5 | 3.4 | -17.0 (2.9) | -16.3 | -1.6 (2.9) | -4.2 (2.2) |
| **CardioHydra continued, epoch 26 of that run (the last before the collapse; chosen on the test participant)** | 0.72 | 0.38 | 15.5 | 2.6 | -15.0 (4.1) | -15.9 | -1.5 (1.8) | -0.2 (0.9) |
| CardioHydra, ABP level terms at 0.2 instead of 0.05 (exp 6b, 15 epochs) | 0.36 | 0.38 | 19.6 | 2.7 | -17.1 (4.8) | -48.1 | -1.4 (2.1) | +0.0 (0.9) |
| CardioConv, ABP level terms at 0.2 instead of 0.05 (exp 6b, 15 epochs) | 0.28 | 0.28 | 17.5 | 3.4 | -9.7 (2.4) | -36.1 | -3.1 (1.9) | +26.3 (33.4) |
| FactorizePhys, ABP level terms at 0.2 (exp 6c, the control, 15 epochs) | 0.01 | 0.42 | 26.7 | 3.0 | -3.2 (2.8) | -11.6 | -2.3 (1.8) | +5.5 (19.0) |
| CardioHydra, bounded masks v1, Mamba at the full rate (exp 7a, epoch 10) | 0.53 | 0.34 | 19.5 | 3.0 | -15.9 (3.1) | -23.8 | -2.3 (1.9) | -0.4 (1.2) |
| CardioHydra rerun at 0.1x, epoch 12 of that run (exp 5b, for the A/B) | 0.66 | 0.31 | 15.3 | 2.8 | -13.2 (3.0) | -27.2 | -2.0 (1.9) | -0.3 (1.0) |
| CardioHydra, bounded masks v2, phase decides (exp 7b, epoch 12) | 0.59 | 0.34 | 20.7 | 3.0 | -20.2 (3.8) | -28.2 | -2.4 (2.0) | -0.3 (1.0) |
| CardioHydra, bounded masks v2, network decides (exp 7c, epoch 12) | 0.66 | 0.24 | 19.0 | 2.9 | -17.8 (3.6) | -18.2 | -1.9 (1.9) | -0.2 (0.9) |

The CAN family (DeepPhys, TS-CAN, EfficientPhys) and iBVPNet predict flat
waveforms on this participant (CCC 0.0-0.13; experiment 5 has their rows).
The Neckflix acceptance route, beating the best existing model on this
participant, is not met: FactorizePhys leads on ABP by a wide margin.
CardioConv's held-out agreement stopped improving after epoch 4 while its
training loss kept falling, so more epochs at this setting would not have
closed the gap; CardioHydra was the best model on this participant for CVP
shape and error through epoch 13 (CCC 0.33-0.42, MAE 2.7-3.2) and then
its CVP mask died. The heart-rate agreement of both is as good as the best
baseline's. The systolic under-read (29 to 43 mmHg) is the known magnitude
problem and is shared, in varying degree, by every model.

Masks on real video (`exp5_*/epoch_*/visuals/neckflix`): sharp and
anatomical. The pulsatility attention lies on a thin diagonal band running
across the neck where the great vessels run, in every channel, depth
included, and the ABP and CVP bins take different ends of it; in the
supine recording it is a short horizontal band mid-neck. When CardioHydra's
CVP mask died at the full Mamba rate, the panels showed its CVP bin on the
static region outside the neck; that is the vanished map's residue at its
own colour scale (the visualiser scales every map to its own range); in the
stable rerun (`exp5b_*/epoch_15/visuals/neckflix`) the CVP bin sits on a
few distinct spots at the ends of the band and the predicted CVP follows
the label beat for beat on that window. There is no vessel ground truth on
Neckflix; the correspondence is to where the neck pulses in the frame.

## Against the three acceptance criteria

1. **Every stage working.** Yes, on both models and both datasets: the rate
   estimator matches the label rate on held-out windows (66.8 for 66.8), the
   wavelet bank, the two masks and the kinematics run and are non-degenerate
   (the masks were flat or constant in every run before tonight), and both
   predictors train. The masks of this design collapsed under long training:
   CardioHydra's CVP phase mask vanished at epochs 13-15 at the full Mamba
   rate (experiment 5, first misread as a trunk divergence) and at epochs
   27-28 at a tenth of it (experiment 6a; held-out CVP shape 0.38 to 0.01),
   and CardioConv's pulsatility map collapsed to about two effective pixels
   under the heavier ABP weights (experiment 6b); the analysis is at the end
   of experiment_6.md. **Fixed in experiment 7:** both mask networks now add
   a bounded correction to a fixed physical logit, no class can be preferred
   wholesale, and a prior keeps each network out of saturation; every class
   keeps at least 2.7e-3 of every pixel by construction, and in three
   CardioHydra runs (one at the full Mamba rate) every trace held at least
   a fifth of the pulsatile mass in every channel at every epoch drawn.
2. **Decent pressure predictions.** On the synthetic split both models beat
   every other model and the old design on waveform shape. On Neckflix
   participant 33 the stable CardioHydra is third on ABP shape at 15
   epochs (0.62, behind FactorizePhys 0.82, PhysNet 0.72 and RhythmFormer
   0.69) and second at epoch 26 of the continuation (0.72, level with
   PhysNet, with CVP 0.38 against FactorizePhys's 0.39 and ABP MAE 15.5
   against 11.1; chosen on the test participant); CardioConv is sixth
   (0.56). Neither beats FactorizePhys, so that acceptance route is not
   met. Levels are read low on Neckflix and high on the synthetic set by
   every model alike, and the ABP level is not an under-weighted term:
   four times the weight on it (experiment 6b) left CardioHydra's
   held-out mean error at -17 mmHg and CardioConv's at -10, and cost
   them ABP shape (0.62 to 0.36 and 0.56 to 0.28), the pulse flattening
   to meet the window statistics; FactorizePhys under the same weights
   (experiment 6c, the control) never learned ABP's shape at all (0.82 to
   0.01) and its level error did not move either. The shared weights
   stand. On the redesigned masks (experiment 7, 12 epochs) CardioHydra
   scores 0.59 / 0.34 with the phase deciding the split (the tree) and
   0.66 / 0.24 with the network overruling it, against 0.66 / 0.31 for the
   old masks at the same epoch: the collapse is gone, the accuracy has not
   moved past the old design's pre-collapse level.
3. **Interpretable.** `scripts/visualise.py` draws what each model looked at
   per channel and trace; on synthetic data the depth masks sit on the
   vessel band and the CVP bin on the jugular, on Neckflix the attention
   follows the vessel course along the neck. Since experiment 7 each map's
   title carries its share of the pulsatile mass and the pulsatility map's
   effective pixel count (every map is drawn at its own colour scale, which
   is how a vanished mask was first read as a one-pixel one). On the new
   masks the ABP and CVP masks sit on different stretches or bands of the
   vessels in most channels, but which band goes to which trace flips
   between channels: the phase reference follows each channel's dominant
   vessel (`experiment_7.md`, "Open issues").

## Reproduction

The commands are in the README under "CardioConv and CardioHydra" and in
each experiment note; every run directory is under `runs/dev_cardio/`,
every chain script beside it. The experiment notes record every decision
with its time and the measurement behind it.

## Guide to the diff (for the review before merging)

| File | What changed |
| --- | --- |
| `neural_methods/model/cardioconv/pulsatility_masker.py` | rewritten: pulsatility-index head plus conv correction, softmax over pixels, one map per window, entropy and TV priors; checkpointed, features without autograd |
| `neural_methods/model/cardioconv/phase_masker.py` | rewritten: relative-phase features (unit vectors, scale coherence, temporal and spatial resultants), cosine phase bins with learned centres plus conv correction, one map per window, `background_pulsatile` and mass-normalised coherence priors |
| `neural_methods/model/cardioconv/channel_predictor.py` | area-normalised masked mean per channel and trace; no parameters |
| `neural_methods/model/cardioconv/channel_mixer.py` | standardised candidates, a 15-tap temporal layer and a pointwise readout per trace, no ReLU |
| `neural_methods/model/cardioconv/utils.py` | `unit`, `temporal_resultant`, `spatial_resultant` replace the angle-based circular helpers; `shrink(video, 1)` is the clip |
| `neural_methods/model/cardioconv/cardioconv.py` | 3 scales per harmonic, `DOWNSCALE` as a constructor argument, field without autograd, `masks()` shared with CardioHydra, `capture` for the visualiser, prior names |
| `neural_methods/model/cardioconv/waveform_rate_estimator.py` | the rate network reads the shrunk clip |
| `neural_methods/model/cardioconv/statistics_head.py` | mask mass is the share of pulsatile pixels (one line) |
| `neural_methods/model/cardiohydra/cardiohydra.py` | mirrors CardioConv's changes; `mamba_lr_scale` flags the trunks' parameters |
| `neural_methods/model/cardiohydra/ssm_predictor.py` | mask mass as above (one line) |
| `src/model_config.py` | `CardioBaseConfig` shared by `CardioConvConfig` and `CardioHydraConfig`; `RATE_ESTIMATOR` defaults to RhythmFormer; new `DOWNSCALE` and (CardioHydra) `MAMBA_LR_SCALE` |
| `src/models.py` | the builders pass `downscale` and `mamba_lr_scale`; min-frame check from the shrink |
| `src/inputs.py` | `window_rate` is the fused spectral peak |
| `src/evaluation/rate.py` | `periodicity_rate` and its constants removed |
| `configs/experiment_configs/*cardio*` and `hpc_configs_*` | new `MODEL` blocks (estimator, gate 8 bpm, shrink, Mamba rate, prior names and weights); the redundant `cardioconv-rhythmformer` variant deleted; two new 128 px configs; two `_abplevel` configs for experiment 6 |
| `configs/_model_config_template.yaml`, `README.md` | the new keys documented; run commands for both models |
| `scripts/visualise.py` | new: the interpretability script |
| `cardioconv_cardiohydra_development_notes/` | new: these notes |
