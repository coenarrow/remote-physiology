# Experiment 7 — Bounded masks (afternoon of 2026-10-11)

Asked by the user at 12:00 after the mask-collapse analysis ("let's see if we
can make those fixes"); deadline 18:00. The analysis is at the end of
`experiment_6.md`; this note is what was built from it and how it was tested.

## Design decisions taken without the user's answer

The analysis left two choices open. The user did not answer them, so the
readings that follow from the original requirements were taken ("the phase
masks must separate vein / artery / background"; "the pulsatility mask should
not isolate the vessels, we only expect the phase masks to do that"):

1. **Phase decides artery against vein.** The network may nudge a class by a
   bounded amount but cannot override the phase bins on a coherent pixel
   (v1 did not achieve this; v2 does; the A/B of 7b against 7c kept it, see
   the verdict). The
   trace centres are learned per channel, because the phase reference is the
   channel's own pulsatility-weighted pulse, so whichever vessel dominates a
   channel sits at relative phase zero, and that differs between colour,
   infrared and depth. Known limit: a vessel dominance that flips from window
   to window within a channel is not captured.
2. **The background means pulsatile pixels that match neither trace's
   phase.** Its prior is a hinge: zero while the background holds at most an
   even share, 1/(S+1), of a channel's pulsatile mass.

## What changed (12:05-12:20)

| File | Change |
| --- | --- |
| `cardioconv/phase_masker.py` | logits = fixed bins `PHASE_GAIN cos(phase - centre)` (frame mean; centres `(C, S)`, gains no longer learned; background bin 0) + `CORRECTION_BOUND tanh(c / CORRECTION_BOUND)` of the network's frame-mean correction, centred over the classes; `background_pulsatile` a hinge above 1/(S+1); the three coherence priors taken on each trace's claimed distribution computed in log space (share-invariant, no clamp); every prior reads the pulsatility map detached |
| `cardioconv/pulsatility_masker.py` | logits = frame-mean pulsatility index (unit gain, `index_gain` gone) + `CORRECTION_BOUND tanh(c / CORRECTION_BOUND)` of the network's correction, centred over the pixels; `pulsatility_entropy` deleted |
| `cardioconv/cardioconv.py`, `cardiohydra/cardiohydra.py` | `CORRECTION_BOUND = 1.5`; the maskers get the channel count and the bound; `PULSATILITY_PRIORS = ("pulsatility_tv",)`; module docstring |
| 12 cardio configs | `PULSATILITY_ENTROPY` removed; comments |
| `scripts/visualise.py` | each map's title and `scores.csv` carry the pulsatility map's effective pixel count and each trace's and the background's share of the pulsatile mass, for every dataset (the panels draw each map at its own colour scale, which is how a vanished mask was read as a one-pixel mask this morning) |

The share floor and the area floor the analysis proposed were not built: with
every class logit bounded, each trace keeps at least 8e-4 of every pixel and
the pulsatility map's top pixel at most a fraction set by the index, so both
floors would be dead code.

Checks before training (CPU and GPU smoke, `check_bounds.py` in the session
scratchpad): with the mask networks' last layers scaled by 1000, the
pulsatility map's top pixel held at most 13 % of the mass (effective 14 px
on a 12x12 grid) and every phase class kept at least 8.0e-4 of each pixel;
the coherence priors and `phase_tv` reach the phase network and the
centres and do not reach the pulsatility network; the waveform path reaches
both networks and the centres; all 12 cardio configs build; ruff clean.

## Review of v1, and v2 (13:00-13:30)

An adversarial review of the patch (workflow `wf_d65c85f7-14f`: math,
codebase contract and training dynamics, each finding checked by a second
agent that re-ran the probes on 7a's checkpoints) confirmed four defects
while 7a trained:

1. **Phase did not decide.** The class-centred tanh bounds each class's
   correction to 1.5 nats, but two classes can then differ by 3 nats, more
   than the bins give on real pixels (the pulsatility-weighted frame-mean
   resultant R is 0.24-0.66 on Neckflix, so the bins span 2R-4R nats). In
   7a the network had learned a uniform class offset within one epoch,
   pulsatility-weighted means of about (-1.5, +1.5, -1.46) nats for (ABP,
   CVP, background) after the tanh, which turned the bins' own split of
   about (0.39-0.54, 0.20-0.30, 0.25-0.31) into (0.08-0.22, 0.71-0.86,
   0.06-0.07). The visualiser at epoch 6 agrees: CVP held 70-83 % of the
   pulsatile mass in every channel, ABP 12-24 %, the background its 5-7 %
   floor. Experiment 6a's mechanism, capped at 3 nats.
2. **The tanh saturated.** The raw correction was beyond three bounds on
   94-100 % of the pulsatile mass by epoch 2 and reached 12-18 nats by
   epoch 4, where the tanh's gradient is under 1e-4 or exactly zero: the
   frozen state moved from the softmax into the tanh.
3. **The per-channel centres were a zero-stride view** (einops `repeat`
   returns an expanded view and the Parameter kept it). Harmless on the GPU,
   where the move to CUDA copies it (7a has distinct per-channel centres),
   but every CPU load of a checkpoint failed, which would have crashed
   `scripts/visualise.py`. Fixed with a `.clone()` at 13:08.
4. **The pre-patch code was in no commit**, so experiments 1-6 could no
   longer be rebuilt. Archived without a commit in
   `runs/dev_cardio/code_snapshots/` (v0 from Claude Code's file history, v1
   reconstructed), with a README on how to reload an old run.

Two low ones were also fixed: the documented pulsatility guarantee was
e^(+-B) where the true pairwise factor on the normalised map is e^(+-2B), and
one reshape was not einops.

**v2** (13:15, in the tree when 7b and 7c start; 7a keeps v1):

- each class's pulsatility-weighted mean correction is removed before the
  class-centring and the tanh, so no class can be preferred wholesale; a
  +-100-nat class offset now leaves every share exactly where the bins put
  it (`check_v2.py`);
- the phase bound is its own constant, `PHASE_BOUND = 0.9`: two classes then
  move against each other by at most 1.8 nats, under the bins' 2R between a
  coherent trace pixel and the background once R > 0.9 and under the 4R
  between antiphase traces once R > 0.45; every class keeps at least 2.7e-3
  of every pixel (measured 2.66e-3 with the networks' outputs scaled by
  1000); the pulsatility map keeps `PULSATILITY_BOUND = 1.5`;
- a restoring prior on each raw correction, `relu(|c| / bound - 1) ** 2`
  averaged over the pixels (`pulsatility_saturation`, `phase_saturation`,
  weight 1.0 in every config): zero inside the bound, so a no-op on a
  healthy network, and a gradient back towards the bound where the tanh has
  none (measured nonzero with saturated networks).

Open issues the review found that v2 does not address (decided at 13:30 to
test the approved "phase decides" design as it stands rather than change it
again before 7b):

- **The phase reference moves with the dominant vessel.** Relative phase
  zero is each channel's pulsatility-weighted mean pulse in each window; on
  real windows the G channel's reference sits 4-44 degrees from the ABP
  label, while the ABP-CVP phase gap at the fundamental ranges from 46
  degrees (P027 supine, P052 at 45 degrees) to 166 (P044), catheter delays
  included. With fixed centres, the vein then falls in ABP's bin in some
  windows; per-channel centres can learn a channel's average gap, not a
  window's. The principled fixes: a per-window centre predicted from the
  window's pooled features, or a reference taken from one trace's own
  claimed pulse instead of the whole pulsatility map.
- **The background has no phase meaning.** Its bin is zero, so it wins
  pixels only where both traces' bins are low (low resultant), and in v1 the
  class-centring dragged it down with every push on a trace. v2 removes the
  wholesale push; where the background sits is still set by the resultant
  and the network, not by "pulses at neither trace's phase".

## Runs

Neckflix, participant 33 held out, 72 px, batch 2, the recipe's Adam at 1e-3
constant. Each run trains 16 epochs and is stopped once epoch 15's checkpoint
and held-out records are written (`runs/dev_cardio/exp7_chain.sh`): the last
epoch's test pass crawls for about 33 minutes on this box and its checkpoint
is written after it, while every earlier epoch's pass takes two minutes.

- **7a** CardioHydra at the full Mamba rate (`MAMBA_LR_SCALE` 1.0, scratch
  config `runs/dev_cardio/cardiohydra_..._fullrate.yaml`): experiment 5's
  setting, where the old masks lost CVP at epochs 13-15. Expected: no
  trace's share below about 1e-3 at any epoch, held-out CVP CCC at epoch 15
  near experiment 5's pre-collapse 0.33 or better (experiment 5 at epoch 15:
  0.02).
Revised at 13:40, after the review found that the phase reference locks
each channel's coherent core to ABP (the depth channel's core has venous
timing) and that the centres start at a stationary point: the CardioConv
run and the queued 7c were cancelled and replaced by an A/B on the v2 masks
(`runs/dev_cardio/exp7_ab.sh`, which also stops and scores 7a), both arms
CardioHydra at the repo's `MAMBA_LR_SCALE` 0.1, against experiment 5b (same
setting, old masks: ABP 0.62, CVP 0.34 at epoch 15):

- **7b, phase decides:** v2 as built, `PHASE_GAIN` 2, `PHASE_BOUND` 0.9.
- **7c, network decides:** the review's interim fix, `PHASE_GAIN` 1,
  `PHASE_BOUND` 1.5, so the network can overrule the bins (pairwise reach 3
  nats against the bins' 2R and R), keeping the removal of the class means,
  the restoring priors and the floors (every class at least about 5e-3 of
  every pixel). The script switches the two constants on disk after 7b's
  panels are drawn and before 7c starts.

Revised again at 13:45: an epoch takes about 7.5 minutes with its held-out
test pass and plots (not the 4.6 minutes of training alone), so 7a was
stopped after epoch 10 (v1 is superseded, and a vanished trace is impossible
by construction) and each arm trains 12 epochs, compared with experiment 5b
scored at its own epoch 12 (`runs/dev_cardio/exp7_ab12.sh`).

CardioConv is not re-run today: it shares both mask modules, its own
collapse (the pulsatility map) is ruled out by construction, and the A/B
on the phase design is the open question.

## Results

### Verdict (16:45)

**The collapse is fixed, by construction and in training.** No mask can
vanish or shrink to a point any more: every phase class keeps at least
2.7e-3 of every pixel, the pulsatility map is anchored to its index, no
class can be preferred wholesale, and the restoring priors keep the
networks out of saturation. In all three runs today every trace held at
least a fifth of the pulsatile mass in every channel at every epoch drawn,
and the pulsatility map covered 600-840 of 1,296 pixels.

**Accuracy at 12 epochs is close to, not above, the old design before its
collapse, with a trade-off set by who decides the split:**

| Run, epoch 12 | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5b, old masks (collapses later: CVP died at epoch 27 of its continuation) | 0.66 | 0.31 | 15.3 | 2.8 | -13.2 (3.0) | -27.2 | -2.0 (1.9) | -0.3 (1.0) |
| **7b, v2, phase decides (the tree)** | 0.59 | **0.34** | 20.7 | 3.0 | -20.2 (3.8) | -28.2 | -2.4 (2.0) | -0.3 (1.0) |
| 7c, v2, network decides | **0.66** | 0.24 | 19.0 | 2.9 | -17.8 (3.6) | **-18.2** | -1.9 (1.9) | -0.2 (0.9) |

The trade-off held at every epoch scored (held-out CCC, ABP / CVP):

| Epoch | 5b | 7b phase decides | 7c network decides |
| --- | --- | --- | --- |
| 4 | 0.58 / 0.35 | 0.54 / 0.33 | 0.58 / 0.20 |
| 8 | 0.62 / 0.34 | 0.55 / 0.39 | 0.58 / 0.19 |
| 12 | 0.66 / 0.31 | 0.59 / 0.34 | 0.66 / 0.24 |

Letting the phase decide gives the best CVP shape of the three and costs
ABP shape and level; letting the network overrule the phase recovers ABP
(and the best systolic, -18 mmHg) but loses 0.1-0.2 of CVP shape. The tree
keeps **phase decides**: it is the design the user set out (the phase masks
separate the vessels), CVP is the trace the masks exist for, and its masks
read as phase. 7c's setting is two constants (`PHASE_GAIN` 1.0,
`PHASE_BOUND` 1.5). One seed per arm and one held-out participant: the
differences in CVP (0.1-0.2) are larger than the epoch-to-epoch wobble of
either arm (0.05), the ABP ones (0.03-0.07) are not by much.

**What is left.** (1) The phase reference: relative phase zero follows the
dominant vessel of each channel and window, so the trace assignment flips
between channels (visible in 7b's panels) and depth's venous core goes to
ABP; a per-window reference is the next design change. (2) The ABP level
is still read 13-20 mmHg low by every CardioHydra run. (3) CardioConv was
not re-run on the new masks today.

### 7c, network decides, epoch 12 (16:36)

Masks at epoch 12, range over channels: ABP 38-48 % of the pulsatile mass,
CVP 21-30 %, background 31-32 % (at its cap: the background prior engaged
from epoch 7, at 6e-5 to 4e-4 weighted), pulsatility map 600-833 effective
pixels. The panels (`epoch_04/visuals/`, `epoch_12/visuals/`) look much like
7b's at epoch 4: the network kept most of the phase bins' assignment and
moved pixels near the boundaries, and that is where CVP lost its signal.

### 7b, phase decides, epoch 12 (15:08)

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5b, old masks, epoch 12 | 0.66 | 0.31 | 15.3 | 2.8 | -13.2 (3.0) | -27.2 | -3.1 | -2.0 (1.9) | -0.3 (1.0) |
| 7b, v2 phase decides, epoch 4 | 0.54 | 0.33 | 17.9 | 2.9 | | | | | |
| 7b, epoch 8 | 0.55 | 0.39 | 19.7 | 2.9 | | | | | |
| **7b, epoch 12** | 0.59 | 0.34 | 20.7 | 3.0 | -20.2 (3.8) | -28.2 | -13.2 | -2.4 (2.0) | -0.3 (1.0) |

Against 5b at the same epoch: CVP shape better at every epoch scored (0.33,
0.39, 0.34 against 0.35, 0.34, 0.31), ABP shape behind (0.59 against 0.66)
and the ABP level 7 mmHg lower. The masks held all twelve epochs: at epoch 12
ABP 40-56 % of the pulsatile mass per channel, CVP 20-32 %, background
24-29 %, the pulsatility map 635-840 effective pixels; the restoring priors
stayed small (pulsatility 1-5e-3, phase under 2e-3, weighted). The panels
(`epoch_12/visuals/neckflix/`) show the traces on separate structures in
most channels (in the supine recording, R puts ABP on the lower and CVP on
the upper of two parallel bands), but the assignment flips between channels
(B puts ABP on the upper band): the per-channel reference problem the
review predicted, visible.

### 7b, phase decides, at epoch 4 (14:10)

The v2 masks did what they were built for. Per channel (mean over four
held-out windows, `epoch_04/visuals/neckflix/scores.csv`): ABP held 40-51 %
of the pulsatile mass, CVP 24-32 %, the background 25-31 % (under its 1/3
cap, so its prior is still zero), and the pulsatility map covered 680-760
of 1,296 pixels. In the colour channels the two trace masks sit on different
stretches of the diagonal vessel band (ABP the lower-left, CVP the
upper-right, in R and B), the background holds the broad skin remainder,
and in depth the two traces overlap on the band (the review's depth-core
issue). Both restoring priors were active at epochs 1-2 and fell steadily
(pulsatility 1.35e-2 to 3.6e-3, phase 1.7e-3 to 3.9e-4, weighted, epochs 2
to 4). Held out at epoch 4: ABP CCC 0.54, CVP 0.33 (5b at epoch 4: 0.58,
0.35; at 8: 0.62, 0.34; at 10: 0.66, 0.36; at 12: 0.66, 0.31).

### 7a, v1 at the full Mamba rate, stopped after epoch 10 (13:35)

Held out, participant 33, each run at the epoch named (the 5b and
experiment-5 epochs scored from their own per-epoch records, kept in
`evaluation_epoch_NN/`):

| Run | ABP CCC | CVP CCC | ABP MAE | CVP MAE | ABP mean err (SD) | ABP systolic err | ABP diastolic err | CVP mean err (SD) | HR err (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiment 5, old masks, full rate, epoch 10 | 0.50 | 0.31 | 18.4 | 3.1 | -16.3 (3.3) | -42.0 | -8.3 | -2.5 (2.0) | -0.2 (0.9) |
| **7a, v1 bounded, full rate, epoch 10** | 0.53 | 0.34 | 19.5 | 3.0 | -15.9 (3.1) | -23.8 | -10.9 | -2.3 (1.9) | -0.4 (1.2) |
| 5b, old masks, Mamba at 0.1, epoch 10 | 0.66 | 0.36 | 15.8 | 3.0 | -13.8 (2.9) | -28.7 | -4.9 | -2.4 (1.9) | -0.1 (0.6) |
| 5b, epoch 12 | 0.66 | 0.31 | 15.3 | 2.8 | -13.2 (3.0) | -27.2 | -3.1 | -2.0 (1.9) | -0.3 (1.0) |

At the same setting and epoch, the bounded masks beat the old ones on both
shapes and read the systolic 18 mmHg closer; the reduced Mamba rate is
still worth 0.12 of ABP shape. The masks stayed alive (every trace above
its 8e-4 floor by construction, CVP coherence terms steady at 4.7e-3 to
4.8e-3 for ten epochs), but the phase did not decide the split: at epoch 6
CVP held 70-83 % of the pulsatile mass in every channel through the
bounded class offset, and the pulsatility map covered 700-780 of 1,296
pixels (`epoch_06/visuals/`, drawn with the v1 code; the share and pixel
count are in each panel's titles).
