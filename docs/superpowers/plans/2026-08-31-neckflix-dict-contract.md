# Neckflix Dict Contract: Unsupervised Methods + PhysMamba

**Date:** 2026-08-30 (planned) / 2026-08-31 (delivered)
**Status:** done. The as-built design, including the fixes this plan did not
anticipate, is `docs/superpowers/specs/2026-08-31-neckflix-dict-contract-design.md`.
**Goal:** Run *all seven* unsupervised methods and PhysMamba against the zarr
Neckflix cache at `C:/Users/20759193/neckflix_cache/rgb128`, with the loader's
nested dicts persisting end-to-end and every reshape expressed in einops.

## Findings from the survey

1. `NeckflixDataset` (zarr) already emits the nested dict contract and works.
   `neckflix_main.py` still constructs the *deleted* `NeckflixLoader` class, so
   nothing downstream of the loader currently runs.
2. `POS_WANG` and `ICA_POH` call `np.mat`, **removed in NumPy 2.0** (the repo
   pins numpy 2.3.2). Those two unsupervised methods are hard-broken today,
   independent of the dataset.
3. `unsupervised_predictor.py` indexes `test_batch[0]`/`[1]` — the legacy tuple
   contract. It also assumes exactly one label signal.
4. Every model/trainer takes `(data, label)` tensors, not dicts.
5. `mamba_ssm` is Linux/macOS-only, so PhysMamba cannot even be imported on this
   Windows box — no way to verify it locally without a pure-torch fallback.
6. The loader emits **raw pixels**; DiffNormalized/Standardized moved
   consumer-side and nothing implements them yet.
7. The cache carries `rgb` only, 128x128, ~893 frames/recording, and the
   available traces vary per recording (`cvp+ecg`, `abp+ecg`, `abp+cvp`,
   sometimes all three) — so `allow_missing` + `label_mask` are load-bearing.

## Contract

The loader dict travels unchanged through the whole stack. A model consumes a
batch dict and returns *the same dict plus* a `predictions` entry:

```python
batch = {
  "frames":       {ch:  (B, 1, T, H, W) float32},   # raw pixels
  "labels":       {sig: (B, T) float32},            # per-window normalised
  "label_stats":  {sig: {stat: (B,) float32}},      # physical units
  "channel_mask": {ch:  (B,) bool},
  "label_mask":   {sig: (B,) bool},
  "metadata":     {"recording_id": [str], "camera_id": [str], "start_frame": (B,)},
}
out = model(batch)          # out is batch + {"predictions": {sig: (B, T)}}
```

Rules:
- Keys are canonical signal/channel names, so any tensor is identifiable by its
  key at any point in the pipeline.
- Every reshape/transpose/stack in new or touched code uses `einops`.
- Channel and trace *order* lives on the model (`model.channels`,
  `model.traces`); dict iteration order is never relied upon.

## Work items

### A. Shared contract layer
- `neural_methods/batch.py` — key constants, `stack_frames`, `split_signals`,
  `iter_samples`, `move_to_device`, `frames_to_rgb_video`. All einops.
- `neural_methods/frame_transforms.py` — `Raw`/`Standardized`/`DiffNormalized`
  + optional spatial resize, on `(B, C, T, H, W)` torch tensors, ported from
  `BaseLoader` semantics.
- `neural_methods/model/DictModel.py` — base class owning
  channels/traces/transform, `forward(batch) -> batch+predictions`.

### B. Unsupervised path
- Port `POS_WANG` and `ICA_POH` off `np.mat` (ndarray + einops), verified
  numerically against an `np.asmatrix` reference in tests.
- `evaluation/metrics_report.py` — extract the duplicated HR-metric
  aggregation/printing so both predictors share it.
- `unsupervised_methods/unsupervised_predictor.py` — accept both contracts;
  for dict batches evaluate **per label signal** (ECG/ABP/CVP all give a valid
  cardiac reference) and honour `label_mask`.

### C. PhysMamba
- `mamba_compat`: pure-torch `MambaRef` fallback when `mamba_ssm` is absent, so
  the model runs on Windows/CPU. CUDA fast path untouched.
- `PhysMamba`: keep architecture identical; parameterise `in_channels` /
  `out_signals`, swap `.view`/`.permute` for einops, take/return dicts.
- `neural_methods/trainer/MultiSignalTrainer.py` — dict-contract trainer
  (masked multi-signal loss, DDP-aware, per-signal metrics).

### D. Wiring
- `config.py`: keys the zarr loader needs (`CACHE_DIR`, `CHUNK_STRIDE`,
  `LABEL_NORM`, `ALLOW_MISSING`, `MIN_CHANNELS`, `MIN_LABELS`, `PERSPECTIVES`,
  `LIGHT`, unsupervised-side `CHANNELS`/`TRACES`/`NECKFLIX`).
- `dataset/data_loader/neckflix_config.py` — yacs -> plain-dict translation,
  incl. `P015` -> `015` participant translation for LOSO.
- `neckflix_main.py` — rebuilt entry point: train_and_test / only_test /
  unsupervised_method over the zarr dataset.
- `configs/neckflix/*.yaml` — unsupervised + PhysMamba configs pointed at the
  local cache.

### E. Verification
- pytest suite green on Windows (mamba fallback, no `np.mat`).
- All 7 unsupervised methods run over the real cache and print metrics.
- PhysMamba trains a few steps on the real cache and runs test/metrics.
