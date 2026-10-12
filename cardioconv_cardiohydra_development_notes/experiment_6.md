# Experiment 6 — Two follow-ups on Neckflix participant 33 (morning of 2026-10-11)

Asked for by the user at 03:55 after the night's summary; deadline 10:00.
Chain: `runs/dev_cardio/exp6_chain.sh`, log `exp6_chain.log`, one GPU job
at a time.

## 6a. Longer training at the reduced Mamba rate

CardioHydra's stable run (experiment 5b, `MAMBA_LR_SCALE` 0.1) was still
improving on the held-out participant at epoch 8 where CardioConv had
plateaued at epoch 4, and its training loss was still falling at epoch 15.
The run is continued from its epoch-15 weights for 15 more epochs with
`scripts/run.py --init-from` (epochs 16-30, a fresh constant schedule at
the recipe's rate, the same split), into
`runs/dev_cardio/exp6a_cardiohydra_neckflix_p33_lr01_ep30`, then scored
on the last epoch and drawn. Comparison: experiment 5b at epoch 15
(ABP CCC 0.62, CVP 0.34, ABP MAE 18.4, CVP MAE 3.0).

## 6b. ABP level terms at CVP's weight

The shared interface weights ABP's level terms at 0.05 (mean, max, min)
against CVP's 0.2, and every model under-reads the ABP systolic by 14 to
43 mmHg on this participant. Two variant configs raise ABP's three level
weights to 0.2 and change nothing else
(`configs/experiment_configs/<model>_FS30_W10S1_RGBID_ABP-CVP_H72W72_abplevel.yaml`).
CardioHydra first, then CardioConv, 15 epochs each at batch 2, so each
compares directly with its experiment-5 run (CardioHydra 5b: 0.62 / 0.34,
ABP mean -17.5, systolic -31.7; CardioConv 5: 0.56 / 0.29, ABP mean
-11.3, systolic -29.2). The shared interface file is untouched, so the
baselines' numbers stand as they are; a like-for-like comparison with
them under the heavier weights would need their runs repeated.

## 6c. The control for 6b: FactorizePhys under the same weights

Added at 05:05, once 6a's epoch time put the chain's end near 07:50 and
left the card idle before the deadline. Without it, a better ABP level
from 6b could not be told apart from a better loss: the baselines were
trained under the shared weights. FactorizePhys, the leading model on this
participant, is trained under the same heavier ABP weights at its
September setting (30 epochs, batch 4, about 50 minutes), from a scratch
copy of its experiment config with only the three ABP level weights
changed (`runs/dev_cardio/factorizephys_..._abplevel.yaml`; the repo's
config directories are untouched). Script
`runs/dev_cardio/exp6c_factorizephys_abplevel.sh`. Comparison: its
September run under the shared weights (ABP CCC 0.82, MAE 11.1, mean
-3.9, systolic -13.9).

Changed at 07:55: `scripts/run.py`'s end-of-run test pass over the held-out
participant takes 33 minutes after a CardioConv or CardioHydra training
(19 s per batch of two windows, in every run of experiments 5 and 6; the
same 217 windows take 25 s in `scripts/test.py` in a fresh process, so
the likely cause is the training's reserved memory pushing the test pass
into WDDM paging, to be confirmed after the deadline). With that tail the
chain would end near 09:30. The control was started beside 6b CardioConv
at 07:52 (the card has the memory) and stopped five minutes later: the two
trainings slowed each other three-fold (CardioConv 0.65 to 2.0 s per step,
FactorizePhys 0.6 to 1.7), which would have put the user's run past the
deadline. The sequence is now `runs/dev_cardio/exp6_tail.sh`: when
CardioConv's epoch-15 checkpoint is on disk, kill its crawling test pass,
test and evaluate the epoch in a fresh process (what the chain would have
produced), then the control for 15 epochs, the 6b runs' length, with the
same treatment of its test pass. Its comparison with the 30-epoch
September run therefore carries an epoch-count caveat, and it is dropped
from the report if it is not scored by 09:40.

## Results

### 6a, during training

Epochs 16 to 26 trained as 5b's last epochs did: the total loss drifted
from 2.65 to 2.46, the ABP CCC term from 0.29 to 0.25, the waveform-rate
term from 0.20 to 0.15, the CVP CCC term flat at 0.72 to 0.75. At epoch
27 the CVP CCC term rose to 0.91 while the CVP mask's three per-trace
priors fell ten-fold in the same epoch (harmonic coherence 0.0038 to
0.0004, spatial coherence 0.044 to 0.005, temporal consistency 0.023 to
0.002); every ABP term, the pulsatility priors, the rate term and the CVP
level terms were unchanged, and no step was non-finite. That is the
CVP mask vanishing: by epoch 28 its pulsatile mass was below the priors'
1e-6 clamp (the three terms read 1e-13 to 1e-15) and the ABP class held
every pixel. It is the same event as experiment 5's epochs 13-15, which
that note first read as a trunk divergence (corrected 10:20, see the end
of experiment_5.md). The
epochs before it are scored on the held-out participant for the record
(`exp6a_epochs.sh`: epochs 26 and 20, tables in `evaluation_epoch_NN/`),
beside the protocol's last epoch.

### 6b CardioHydra, scored on participant 33 (07:50)

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5b, shared weights (ABP level 0.05) | 0.62 | 0.34 | 18.4 | 3.0 | -17.5 (3.0) | -31.7 | -8.1 | -2.4 (1.9) | -0.2 (0.9) |
| 6b, ABP level at 0.2 | 0.36 | 0.38 | 19.6 | 2.7 | -17.1 (4.8) | -48.1 | -3.6 | -1.4 (2.1) | +0.0 (0.9) |

Four times the weight on ABP's level terms made the held-out ABP worse:
shape fell from 0.62 to 0.36, the systolic under-read grew from 32 to
48 mmHg, and the mean error did not move (-17 mmHg either way) while
the training-set mean term sat where 6a's did (about 8 mmHg raw). The
diastolic error improved and CVP was unchanged within noise. Reading:
the ABP level is a participant offset the model cannot read off this
participant's video, not an under-weighted term; pushing the weight
flattens the pulse (the window mean and extremes are cheaper to hit with
a smaller pulse) and costs shape. The shared weights stand.

### 6b CardioConv, scored on participant 33 (09:28)

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5, shared weights (ABP level 0.05) | 0.56 | 0.29 | 16.2 | 3.2 | -11.3 (3.9) | -29.2 | -7.3 | -2.5 (2.0) | -0.2 (0.9) |
| 6b, ABP level at 0.2 | 0.28 | 0.28 | 17.5 | 3.4 | -9.7 (2.4) | -36.1 | +2.1 | -3.1 (1.9) | +26.3 (33.4) |

The same picture as CardioHydra's, sharper: ABP shape halved (0.56 to
0.28), the systolic under-read grew (29 to 36 mmHg), the mean error moved
2 mmHg (-11.3 to -9.7) and the diastolic came level; the heart rate read
off the predicted waveforms is 26 bpm off with a 33 bpm spread, so the
ABP waveform has lost its pulse on many windows (the shape term is one
of five now and the level terms are cheaper to meet with a flat trace).
CVP, whose weights did not change, is unchanged within noise. The
run's own training terms were healthy throughout (ABP CCC term 0.58 to
0.62, no non-finite step); the training-set ABP mean term ended near 9.5
mmHg raw, so here too the held-out level error is an offset the model
cannot read, not an under-weighted term.

The epoch-15 panels (`exp6b_cardioconv_.../epoch_15/visuals/neckflix/`)
show why the heart rate went: under the heavier level weights the
pulsatility map itself collapsed to one to three pixels per channel (a
corner pixel in the infrared, a single pixel in the depth), every trace
mask with it, and the ABP waveform read off a few pixels is noise with a
pulse somewhere in it. CardioHydra's 6b masks did not collapse; its ABP
flattened instead. Two different models, the same lever: a mask that
can sharpen without limit.

**6b conclusion.** Weighting ABP's level terms like CVP's hurts both
models on the held-out participant and does not recover the level. The
shared interface weights stand; the level problem needs another input
(a per-participant calibration, or a feature that carries absolute
pressure), not another loss weight.

### 6c, the control, scored on participant 33 (10:30)

Started on the freed card at 09:29, after the deadline (the CardioConv
run's end-of-run test pass crawled for its full 33 minutes: `run.py`
writes the last epoch's `model.pt` only after that pass, so the tail
script's trigger fired at 09:28 instead of 08:55). FactorizePhys at its
experiment config's recipe (Adam 1e-3, OneCycle, float32), batch 4, 15
epochs, ABP level weights 0.2; losses finite throughout.

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| September, shared weights (30 epochs) | 0.82 | 0.39 | 11.1 | 2.9 | -3.9 (2.7) | -13.9 | -6.5 | -2.3 (1.4) | -0.2 (0.9) |
| 6c, ABP level at 0.2 (15 epochs) | 0.01 | 0.42 | 26.7 | 3.0 | -3.2 (2.8) | -11.6 | -10.0 | -2.3 (1.8) | +5.5 (19.0) |

The heavier level weights stopped FactorizePhys learning ABP's shape at
all: its training ABP CCC term stayed at 0.97-0.99 for the whole run
(the September run's was 0.69 by epoch 4 and 0.46 at the end), while its
level terms fit (ABP mean term 3.2 -> 2.9, i.e. about 15 mmHg raw). The
model's ABP copy outputs a pulse-less trace at the window's level; the
held-out mean error is what it was under the shared weights (-3.2 against
-3.9). CVP, whose weights did not change, is unchanged. The epoch counts
differ (15 against 30), but the September run had learned most of its
shape by epoch 7, so the count does not explain a shape term that never
left 0.97.

**6c conclusion.** The control settles 6b: weighting ABP's level terms
like CVP's costs ABP shape in all three models, worst in the strongest
baseline, and moves no model's held-out level error. It is the loss
weighting, not a property of the wavelet masks.

### 6a, scored on participant 33 (06:07)

| Epoch | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15 (experiment 5b, the start) | 0.62 | 0.34 | 18.4 | 3.0 | -17.5 (3.0) | -31.7 | -8.1 | -2.4 (1.9) | -0.2 (0.9) |
| 20 | 0.70 | 0.35 | 18.5 | 2.9 | -18.0 (3.2) | -20.1 | -13.2 | -2.2 (1.9) | -0.1 (0.6) |
| 26, the last before the CVP mask went | 0.72 | 0.38 | 15.5 | 2.6 | -15.0 (4.1) | -15.9 | -4.5 | -1.5 (1.8) | -0.2 (0.9) |
| 30, the protocol's last epoch | 0.69 | 0.01 | 17.5 | 3.4 | -17.0 (2.9) | -16.3 | -7.4 | -1.6 (2.9) | -4.2 (2.2) |

Longer training at the reduced Mamba rate kept paying until the CVP mask
collapsed: by epoch 26 ABP shape had risen from 0.62 to 0.72 (level with
PhysNet, second to FactorizePhys's 0.82), CVP shape from 0.34 to 0.38
(FactorizePhys 0.39, the best model 0.44), ABP MAE from 18.4 to 15.5 mmHg
and the systolic under-read from 32 to 16 mmHg. The epoch-30 panels
(`epoch_30/visuals/neckflix/*/panel.png`) show what the training terms
said: the pulsatility map has sharpened to a few pixels on the vessel band
in every channel, the ABP mask takes all of them, the background mask is
zero and the CVP map has vanished (corrected 10:20: what the panel draws
as one or two bright pixels is the vanished map's residue, because
`scripts/visualise.py` scales every map to its own range), so its
waveform is a drift (hence CVP CCC 0.01 and the 4 bpm rate error, which
is read off both traces). The ABP trace at epoch 30 follows the systolic
peaks closely; its score fell only a little from epoch 26.

Reading (corrected 10:20): the per-pixel trace softmax lets a trace's
share of the pulsatile mass go to zero and nothing in the priors stops it;
the coherence priors are mass-normalised, so a vanishing mask scores zero
on them, and once a class is saturated its gradient is exactly zero, so it
cannot come back. "Why the masks collapse" at the end of this note has the
mechanism and the remedies. The epoch-26 number is
chosen on the test participant and is reported as such; the epoch-30
number is the protocol's. The fix is a design change left for after the
deadline: a floor on each trace's share of the pulsatile pixels (or an
entropy prior on the trace softmax), which would also bound the
pulsatility map's sharpening.

## Why the masks collapse (analysis, 2026-10-11 10:00-11:45)

Asked by the user after the report: how might we prevent the mask from
collapsing? Analysis only; no code was changed. Method: four analyses
(gradient incentives, regularisers, structural floors, training protocol),
each read the code, the checkpoints of every epoch of experiments 5, 5b, 6a
and 6b, and `losses.csv`, and each was checked by an adversarial verifier
that re-ran its probes; a synthesis kept what survived (workflow
`wf_3f41b1a1-db8`; probe scripts, their outputs, the verdicts and the
synthesis in `runs/dev_cardio/mask_collapse_probes/`, outside git).

How far it is verified (11:55): every analysis claim was attacked by a
second agent that re-ran the probes; the trapdoor was found independently
by two of them; the vanished CVP masks (experiments 5 and 6a), the dead
background, the entropy prior's sign and the clamp were also confirmed by
hand. Checked by one probe only: the minority shares of the healthy runs
and the MAX/MIN gradient dominance (measured on held-out windows in eval
mode; 6b CardioHydra had the same dominance without collapsing). Not
checked by a second agent: the synthesis's own final probes (the Adam-step
decomposition over 28 training windows, the 0.60 gradient of F1 at 120
nats, the healthy A2 range, R2's 1.2e-7 agreement) and R2 as a proposal.
F1 was rated sound by its checker. F2 was rated "needs change": it stops
the map shrinking below the floor but does not prevent a collapse down to
it (6b's maps ended at 1-2 px, so a 4 px floor would have held them at 4),
and it touches one healthy depth window (A2 3.45).

### Mechanism, ranked

1. **No restoring force, and absorbing vertices.** Neither softmax has a
   floor (`pulsatility_masker.py:92`, `phase_masker.py:130-131`), and every
   readout is area-normalised (`channel_predictor.py:30-32`,
   `ssm_predictor.py:79-81`), so no loss sees a trace's share of the
   pulsatile mass. One trace is starved in every run, healthy ones
   included: the minority held 1e-3 to 1e-2 of the mass (CardioConv's ABP
   1e-4 to 1e-2 in experiment 5; CardioHydra's CVP 0.2 % to 11 % at epoch
   16 of 6a). The panels hid it because each map is drawn at its own
   scale. The logit scale grows in the conv corrections, not the learned
   gains (index gain 0.86-1.26, phase gain 1.89-2.12 in every epoch of
   every run): the pulsatility net's spread went from 0.1 to 44 nats (6b
   CardioConv, epochs 1 to 13), the ABP-CVP class gap from 6.5 to 33 to
   120 nats (6a, epochs 16, 26, 27). The recipe has no restoring force
   (Adam, constant 1e-3, no weight decay). At a vertex every gradient
   carries a factor of the probability and dies; the phase masker froze
   from 6a's epoch 28.
2. **The push: the soft-peak MAX/MIN level terms**, reaching the masks
   through the z-scored shape (`statistics_head.py:117-120`,
   `soft_peak_l1.py:85-88`), not through pooled power: at 6b CardioConv
   epoch 9 their gradient on the pulsatility logits is 0.33 and 0.35
   against at most 1.8e-3 for any prior, unchanged with SUPERVISE_MASKS
   off. Their sign flips between epochs; the heavier ABP weights made the
   pull 6-7 times experiment 5's.
3. **The drift.** `pulsatility_entropy` at a positive weight is a sparsity
   prior: it shrinks the map's effective area in every window, and with
   the TV prior the pair is cheapest at a 2x2 corner block (0.0090 per
   channel against 0.0200 for a uniform map), which is where 6b's infrared
   map landed. In 6a the spatial and temporal coherence priors and
   `phase_tv` lowered the minority's share in 86-100 % of windows.
4. **A trapdoor at share EPS/T = 3.3e-9** (`phase_masker.py:143` clamps the
   mass summed over 300 frames at 1e-6). Above it each coherence prior is a
   weighted mean with no net force on the share; below it each prior equals
   T s v / EPS, linear in the share, so gradient descent empties the trace.
   Every dip that stayed above it recovered (experiment 5 CardioConv's ABP
   went as low as 1e-7.3); at 6a epoch 26, with CVP shares of 1e-6.7 to
   1e-9.3, the coherence priors lowered the share in 29 of 29 windows, and
   at epoch 27 the share fell to 1e-29 to 1e-35.

6a: items 1-3 slid CVP's share from 1e-2.9 (epoch 20) to 1e-7.7 (epoch 26)
and item 4 shut it. 6b: item 2 at four times the weight acted on item 1's
unbounded logits, and item 3 decided where the map landed. The background
class is dead by epoch 3-4 of every run (`background_pulsatile` at 0.5, 25
times each coherence weight), so the phase softmax has been two-way
throughout: today the masks separate artery from vein in relative terms
only, and nothing from background.

### What would stop it (proposed, not implemented)

Floors, both in log space, so they keep an O(1) gradient at any class gap
and are exactly zero above the floor:

- **F1 `trace_share`**, after `phase_masker.py:131`, a new entry of
  `PHASE_PRIORS` with weight 1.0; about 6 lines:

      log_claimed = rearrange(pulsatility[:, :, 0].detach().log()[..., None]
                              + logits.log_softmax(-1)[..., :-1], "b c h w s -> b c s (h w)")
      log_share = torch.logsumexp(log_claimed, dim=-1)                  # (B, C, S)
      trace_share = (math.log(SHARE_FLOOR) - log_share).clamp_min(0)    # per trace k: [..., k]

  SHARE_FLOOR = 10 EPS = 1e-5, a fraction, so independent of grid and
  window, a decade above the readouts' clamp. Gradient 0.60 at a 120-nat
  gap, where a linear hinge gives 0. Zero on the median channel of the
  healthy epochs (5b and experiment 5 epoch 15, 6a epochs 20-25); active
  from 6a epoch 26.
- **F2 `pulsatility_area`**, replacing `pulsatility_entropy`; weight 1.0
  (2.0 under the heavier ABP weights); about 5 lines:

      flat = rearrange(logits, "b c h w -> b c (h w)")
      log_conc = torch.logsumexp(2 * flat, -1) - 2 * torch.logsumexp(flat, -1)   # -log A2
      pulsatility_area = (math.log(max(AREA_FLOOR * h * w, 1.0)) + log_conc).clamp_min(0)

  AREA_FLOOR = (2/36)^2, one vessel cross-section: a grid pixel at 72 px
  and DOWNSCALE 2 is 4.7-9.4 mm of neck, the jugular and carotid 6-14 mm,
  so 1-3 px across. 4 px at 36x36, 69 at 150x150, off below 18x18. Healthy
  Renyi-2 areas are 9-379 px (one experiment-5 depth window 3.45). It
  prevents a collapse but cannot undo one (its gradient vanishes on a
  flat-topped patch), so it needs fresh runs; if 4 px proves too low, the
  next value is sqrt(hw).

Removing the incentives:

- **R1**: F2 deletes the entropy prior, the only pulsatility sharpener
  whose sign never changes; it also contradicts "the pulsatility mask may
  cover all pulsatile skin".
- **R2**: the coherence priors on the claimed distribution with the
  pulsatility map detached (`phase_masker.py:142-147`, 3 lines):

      q = (log_claimed - log_share[..., None]).exp()
      prior_k = einsum(q[:, :, k], rearrange(value.mean(2), "b c h w -> b c (h w)"), "b c n, b c n -> b c")

  Equal to today's value wherever the share is above EPS/T (to 1.2e-7),
  so a no-op on healthy runs; it closes the trapdoor and stops phase-based
  priors training the phase-blind map through the self-referenced phase
  (`phase_masker.py:102-104`).

Ship F1 + F2 + R2: no forward value of a healthy run changes, the new loss
columns read "nats below floor", and configs naming `PULSATILITY_ENTROPY`
stop loading. Backstop if F2 is overrun: `index + B tanh(c / B)` at
`pulsatility_masker.py:91` (not a no-op on healthy runs). SUPERVISE_MASKS
false is the ablation of the pooled-power path, not a fix.

### What to measure

Per epoch, the median and minimum over channels of each trace's and the
background's log10 share, log A2, the pulsatility-weighted class gap
(healthy 8-12 nats; 35 at 6a epoch 26, 120 at 27) and the net-versus-index
logit spread (3 / 3 healthy, 44 / 4 at 6b epoch 13); offline, the update
norm of each stage between checkpoints. Runs are unseeded: repeat an arm
that survives. Experiments, cheapest first:

- E0 (minutes): the probes on every epoch; expected: a median share below
  1e-8.5 dies within an epoch, dips above it recover.
- E1 (~70 min): fresh CardioHydra at MAMBA_LR_SCALE 1.0 (experiment 5's
  setting, CVP dead at epochs 13-15) with F1 + F2 + R2; expected: CVP share
  at or above 1e-5 throughout, held-out CVP CCC about 0.33 at epoch 15
  (experiment 5: 0.02).
- E2 (2 x ~47 min): CardioConv under the heavier ABP weights for 10 epochs,
  SUPERVISE_MASKS false (expected to collapse anyway) against F2 (expected:
  A2 at least 4 and a waveform heart-rate error near -0.2 bpm, not +26).

### What not to do

Choose epochs on participant 33 (epoch 26 of 6a is an oracle); maximise
entropy or set one area target (healthy maps span 9-379 px); floors or
balance losses in probability space (their gradient is 5e-6 at 6a epoch
26); clamp the gains, apply global weight decay, cap the class logits (it
forces a 50/50 split) or give each trace its own pixel softmax; read mask
size off the panels; continue a dead run (`--init-from` keeps the old
priors).
