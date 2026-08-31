# Neckflix Dict Contract — Design

**Date:** 2026-08-31
**Status:** implemented
**Supersedes (in part):** the "Out of scope (follow-up tasks)" list of
`2026-08-30-neckflix-zarr-loader-design.md` — consumer rework, `neckflix_main.py`
wiring, and consumer-side frame preprocessing are now done.

## Context

The zarr rebuild left `NeckflixDataset` emitting a nested dict while every
consumer downstream still spoke the upstream tuple contract
`(frames, labels, filename, chunk_id)`. `neckflix_main.py` constructed a class
that no longer existed, so nothing past the loader ran. Separately, three of the
seven traditional rPPG methods — and the pseudo-PPG label generation every
dataset shares — were broken by NumPy 2 regardless of dataset.

This design carries the loader's dicts all the way through models, losses and
evaluation, and gets the seven unsupervised methods plus PhysMamba running on
the cache.

## Decisions

1. **The loader's dict is the pipeline currency.** A model takes the batch dict
   and returns *the same dict plus* `predictions`. Nothing is dropped in
   transit, so at any point one object carries frames, labels, stats, masks,
   metadata and predictions — each identifiable by key.
2. **Order lives on the model, not in the dict.** `model.channels` and
   `model.traces` are the canonical orders; dict iteration order is never
   load-bearing.
3. **Every reshape is einops.** New and touched code uses `rearrange` /
   `reduce` / `einsum` rather than `view` / `permute` / `reshape`.
4. **Architectures are unchanged.** Only signatures and the two boundary shape
   moves changed. PhysMamba is layer-for-layer the same network.
5. **The legacy tensor path survives.** `DictModel.forward` accepts a plain
   `(B, C, T, H, W)` tensor and returns a plain tensor, so the upstream
   tuple-contract trainers (PURE, UBFC-rPPG, ...) keep working unchanged.
6. **Per-signal evaluation.** The cache has no PPG; ABP, CVP and ECG each carry
   the cardiac rhythm, so a prediction is scored against every reference the
   recording actually has, reported as a separate row per signal.
7. **No mamba_ssm, no problem.** A pure-PyTorch `MambaRef` stands in where the
   package has no wheel, so PhysMamba is runnable and testable everywhere.

## The contract

Per sample (dataset `__getitem__`) — unchanged from the loader spec:

```python
{"frames":       {ch:  (1, T, H, W) float32},   # raw pixel values
 "labels":       {sig: (T,)         float32},   # per-window normalised
 "label_stats":  {sig: {stat: ()    float32}},  # physical units
 "channel_mask": {ch:  ()           bool},
 "label_mask":   {sig: ()           bool},
 "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int}}
```

`default_collate` adds a leading batch axis to every tensor and turns the
metadata strings into lists. A model then adds:

```python
out = model(batch)
out["predictions"]      # {signal: (B, T)}
out["frames"] is batch["frames"]     # everything else passes straight through
```

`neural_methods/batch.py` owns the key names and exactly two shape moves:

- `stack_frames(frames, channels)` — `{ch: (B,1,T,H,W)}` → `(B, C, T, H, W)`
- `split_signals(raw, traces)` — `(B, S, T)` → `{sig: (B, T)}`

plus their inverses, traversal helpers (`map_tensors`, `move_to_device`,
`detach_to_cpu`, `iter_samples`) and the RGB reductions the traditional methods
need.

## Components

### `neural_methods/frame_transforms.py`

The loader emits raw pixels, so `DATA_TYPE` moved consumer-side. `Raw` /
`Standardized` / `DiffNormalized` on `(B, C, T, H, W)` tensors, matching
`BaseLoader` semantics (statistics per sample rather than per cached chunk), plus
a spatial resize so one cache resolution serves models that want another.
Multiple `DATA_TYPE` entries concatenate along the channel axis — upstream
semantics, and what DeepPhys/TS-CAN split back apart.

### `neural_methods/model/DictModel.py`

Base class owning `channels`, `traces` and the frame transform. A subclass
implements only `forward_video(video) -> (B, S, T)`. `in_channels` is
`len(channels) * frame_transform.channel_multiplier`.

### `PhysMamba`

Now a `DictModel`. Three changes, no architectural ones:
- `ConvBlock1` input width and `ConvBlockLast` output width are configurable
  (`len(channels)` blocks in, one plane per trace out);
- `poolspa` is `AdaptiveAvgPool3d((None, 1, 1))` — spatial-only, so one model
  serves any `CHUNK_LENGTH` instead of resampling to a construction-time
  `frames`;
- `.view` / `.reshape` / index-gymnastics replaced by `rearrange`.

### `SignalDictWrapper`

Also a `DictModel`; folds the temporal axis into the batch axis for per-frame
2-D backbones (`frames2d`) or passes a clip straight through (`video3d`). This
is how DeepPhys joins the contract without being modified.

### `neural_methods/trainer/MultiSignalTrainer.py`

One trainer for every dict-contract model, driven by `MODEL_REGISTRY`
(`{name: builder}`). Owns DDP, AMP, `MaskedMultiSignalLoss`, checkpointing, and
per-signal evaluation. Adding a model is a builder function and a registry line.

Evaluation reports, per signal: waveform Pearson/MAE/RMSE in normalised units;
the same MAE/RMSE **in physical units** (mmHg for ABP/CVP), obtained by running
each window's own `label_stats` back through the exact inverse of the
normalisation the loader applied; and the inherited HR metrics. The physical
figure is the error you would see given a perfect estimate of that window's
scale — it measures *shape*, expressed in the signal's units. Absolute level is
a separate problem, which per-window normalisation deliberately removes and
which nothing here yet attempts.

### `dataset/data_loader/neckflix_config.py`

The only place that knows how YAML maps onto the loader's plain-dict config,
including the participant-id convention mismatch (`P015` on the CLI,
`"015"` in the store's root attrs).

### `neckflix_main.py`

Rebuilt: LOSO splits by participant filter (test *includes* the held-out
participants, train *excludes* them), optional DDP, and the three toolbox modes.
`--limit_windows` subsamples evenly for smoke runs. `USE_LAST_EPOCH: False` with
no `--valid_participants` is refused rather than silently testing epoch 0.

### `tools/`

`list_neckflix_folds.py` enumerates LOSO folds from a config (metadata-only, so
it is safe on a login node) — the input to the SLURM job array.
`summarise_neckflix_outputs.py` turns a saved test pickle into a per-window
table and a per-signal summary; because every record carries its own
`label_stats`, it reproduces the physical-unit numbers without the zarr cache.

## Unsupervised methods

Three were broken independently of any dataset:

| method | breakage | fix |
|---|---|---|
| POS | `np.mat` (removed in NumPy 2.0) | ndarray + einops; also vectorised the per-frame overlap-add with a sliding window (~960x) |
| ICA | `np.mat` throughout JADE | ndarray + explicit `conj().T`; einops reshapes |
| PBV | NumPy 2 made `linalg.solve` treat a 2-D RHS as a matrix, not a vector stack | explicit trailing axis; einops instead of chained `swapaxes` |

All three are pinned in tests against frozen pre-NumPy-2 originals in
`tests/reference/`, matching to 1e-10 (POS/PBV) and 1e-8 (ICA).

A fourth `np.mat` break lived outside `unsupervised_methods/` entirely:
`BaseLoader.generate_pos_psuedo_labels` and its verbatim twin in
`BP4DPlusBigSmallLoader` each carried their own inline copy of POS, so
`USE_PSUEDO_PPG_LABEL: True` was broken for *every* dataset. Both now call
`POS_WANG.pos_signal` — the detrended overlap-add signal before any bandpass,
split out precisely because each call site applies its own filter afterwards.
One implementation instead of three, pinned to the loaders' original to 1e-10.

`unsupervised_predict` accepts both contracts. For dict batches it evaluates per
label signal and honours `label_mask`. `unsupervised_predict_many` scores all
seven methods in **one pass** over the data — a Neckflix window costs a few
hundred zarr frame decodes, and every method consumes only the per-frame spatial
mean, so the clip is decoded and reduced once.

## Shared post-processing (exact-equivalent rewrites)

Both are hot: they run once per signal per window per method.

- `_detrend` — the smoothness-priors detrend built a dense `n x n` matrix and
  inverted it. `I + lambda^2 D'D` is symmetric pentadiagonal, so it is now a
  banded Cholesky solve: `x - solveh_banded(bands, x)`. O(n) instead of O(n^3);
  0.75 s → 0.09 ms at n=300 on the dev box, matching the dense form to 5e-11.
- `_compute_macc` — the per-lag `corrcoef` loop is one circular
  cross-correlation; computed by FFT it is bit-identical and ~350x faster.

## Robustness fixes surfaced by running the thing

- `BlandAltmanPy` coloured its scatter points by `gaussian_kde` density, which
  raises `LinAlgError` on a singular covariance — a short evaluation split, or a
  model predicting a near-constant rate, both produce one. The plot is still
  worth writing, so density colouring now degrades to a constant.
- `report_hr_metrics` printed `Pearson: nan +/- nan` when the ground-truth or
  predicted HR was constant across the split. It now says which side is
  constant, and why the statistic is undefined.
- `MultiSignalTrainer` indexes its GPU by `LOCAL_RANK`, not the global rank:
  on a multi-node job rank 5 is local GPU 1 of node 1, not a sixth GPU here.
  (The legacy per-model trainers still have the global-rank version.)
- `neckflix_main` picks the DDP backend from where the model will actually live,
  so `DEVICE: cpu` on a GPU machine uses gloo rather than failing in nccl. The
  trainer honours `DEVICE: cpu` for the same reason.

## Testing

`tests/` (235 passing, 1 skipped where mamba_ssm is absent):

- `test_batch_contract.py` — key recognition, stack/split round trips, collation
  and un-collation, frame transforms vs `BaseLoader` semantics.
- `test_unsupervised_methods.py` — equivalence with the frozen originals, HR
  recovery through the real evaluation path for all seven methods, the dict and
  legacy window iterators, mask handling.
- `test_post_process.py` — detrend and MACC pinned to the implementations they
  replaced.
- `test_physmamba_dict.py` — dict in/out with nothing dropped, channel/trace
  arity, window-length independence, gradients per signal, masked-loss safety.
- `test_deepphys_multisignal.py` — the wrapper path.
- `test_mamba_compat.py` — `MambaRef` causality, gradients, scan-vs-sequential.
- `test_neckflix_config.py` — YAML → loader dict, LOSO disjointness, filters.
- `test_multisignal_trainer.py` — train → checkpoint → test → saved outputs over
  a synthetic zarr cache, including a recording missing a trace, the
  validation/best-epoch path, and `only_test` reloading a checkpoint.
- `test_metrics_report.py` — metric definitions, degenerate inputs, per-signal
  scoping, and that Bland-Altman plots still land on degenerate data.
- `test_neckflix_main.py` — LOSO split disjointness, `--valid_participants`
  hold-out, experiment naming, `--limit_windows`, and the misconfiguration
  guards (empty split; model selection with nothing to select on).
- `test_legacy_contract.py` — the upstream tuple contract still trains and
  tests through the real `PhysMambaTrainer`, and `evaluation.metrics` still
  works over the rewritten post-processing.
- `test_summarise_outputs.py` — the offline summariser reproduces the
  physical-unit numbers from the saved pickle alone.

## Verified on the real cache

`C:/Users/20759193/neckflix_cache/rgb128` (still being generated; 1243 windows of
300 frames across 44 participants at the time of the run).

**All seven unsupervised methods**, HR agreement per reference trace. The
per-signal split is the point: ABP is a pressure pulse and makes a good HR
reference, while an FFT-derived "HR" from CVP (respiratory and a/c/v waves) or
from bandpassed ECG (QRS structure removed by the 0.6-3.3 Hz filter) is much
weaker — and the table says so rather than averaging them together.

| method | ABP MAE (bpm) | ABP Pearson | CVP MAE | ECG MAE |
|---|---|---|---|---|
| CHROM | **3.83** | **0.637** | 23.17 | 20.00 |
| POS   | 4.12 | 0.622 | 22.19 | 19.22 |
| ICA   | 5.79 | 0.477 | 25.89 | 23.34 |
| LGI   | 6.40 | 0.456 | 25.57 | 22.49 |
| OMIT  | 6.42 | 0.453 | 25.58 | 22.51 |
| PBV   | 9.40 | 0.327 | 28.47 | 25.46 |
| GREEN | 9.54 | 0.362 | 31.22 | 28.34 |

The whole sweep takes ~4 minutes on CPU: one pass over the cache scores all
seven, and the post-processing rewrites removed what was previously the
dominant cost.

**PhysMamba** trains and tests end to end predicting ABP + CVP + ECG together —
on CPU at 32x32, and on GPU at the production 128x128x128 (2.5 s/step at batch 1
on an RTX 5000 with the pure-torch Mamba fallback; the fused CUDA kernels on HPC
are both faster and far lighter on memory).

A deliberately small learning check — 8 epochs over 600 training windows at
72x72, P015 held out — confirms the masked multi-signal loss is wired to the
right tensors. Training loss falls from ~0.99 to ~0.40, and on the held-out
subject:

| signal | windows | waveform Pearson | MACC | MAE (normalised) | MAE (physical) | HR MAE (bpm) |
|---|---|---|---|---|---|---|
| ABP | 84 | 0.756 | 0.867 | 0.521 | 12.9 mmHg | 8.71 |
| CVP | 84 | 0.521 | 0.643 | 0.721 | 1.49 mmHg | 29.8 |
| ECG | 72 | 0.241 | 0.433 | 0.782 | 110 uV | 49.2 |

That is not a tuned result — it is 600 windows and eight epochs — but ABP
waveform correlation of 0.76 on an unseen subject is well clear of chance, and
the ordering (ABP > CVP > ECG) matches what the unsupervised baselines say about
which trace the camera can actually see. Note the physical-unit column scales
with each window's own dynamic range, so it compares runs on one signal, not
signals against each other.
