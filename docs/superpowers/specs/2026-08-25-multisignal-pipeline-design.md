# Multi-Signal / Multi-Channel Pipeline — Design (DeepPhys pilot)

Date: 2026-08-25
Status: approved design, pilot scope
Author: Coen Arrow (with Claude)

## Goal

Make the toolbox train/infer on the Neckflix dataset with:
- up to 5 input channels (R, G, B, I, D), where recordings may lack I/D
- multiple predicted physiological signals per model, drawn from a fixed
  vocabulary derived from what this repo's dataloaders actually provide

Pilot: DeepPhys end-to-end on Neckflix (ABP + CVP), learning the shape of the
work before rolling out to the 3D video models (PhysNet, PhysMamba, PhysHydra,
PhysFormer, RhythmFormer, iBVPNet, FactorizePhys, EfficientPhys), then the
remaining 2D models (TS-CAN, BigSmall).

## Decisions (locked with user)

| Decision | Choice |
|---|---|
| Missing input channels | Zero-fill to 5 canonical slots + availability mask; one model covers the whole dataset |
| Pred/label flow | Dicts end-to-end, keyed by canonical signal name |
| Signal vocabulary | PPG, ECG, ABP, CVP, RESP, EDA, SPO2 (waveforms; loaders' "BVP" maps to PPG) + HR as eval-only |
| Missing labels | Masked loss: model predicts every configured signal; loss counted only where labels exist |
| Rollout order | DeepPhys pilot -> 3D models -> remaining 2D models |
| Retrofit style | Approach B: thin backbone contract (in_channels, out_signals -> raw (B,S,T)); all dict/mask logic in shared code |
| Checkpoint compat | Not required (fresh training) |

## Components

### 1. `neural_methods/signals.py` — canonical registries

```python
CHANNELS = ('R', 'G', 'B', 'I', 'D')          # canonical order = slot order
SIGNALS = {
    'PPG':  {'norm': (-2.0, 2.0)},            # a.k.a. BVP in several loaders
    'ECG':  {'norm': (-1500, 1500)},
    'ABP':  {'norm': (0, 200)},
    'CVP':  {'norm': (-20, 30)},
    'RESP': {'norm': (0, 10)},     # BP4D Resp_Volts scale; override per dataset
    'EDA':  {'norm': (0, 40)},     # microsiemens; override per dataset
    'SPO2': {'norm': (0, 100)},
}
EVAL_ONLY = ('HR',)
```
Norm ranges are defaults; config can override per signal. Loader-side names
("BVP", "bvp", "Pulse") normalize to 'PPG' at the loader boundary.

### 2. Loader contract (NeckflixLoader first)

`__getitem__` returns:
```python
{
  'frames': FloatTensor (C_total, T, H, W),
      # DATA_TYPE transforms CONCATENATED along channels (upstream semantics;
      # fixes NeckflixLoader's sequential-transform quirk).
      # Each transform block is zero-filled to the configured channel slots.
      # e.g. DATA_TYPE=[DiffNormalized, Standardized], CHANNELS=RGBID
      #      -> C_total = 2 * 5 = 10
  'channel_mask': FloatTensor (n_slots,),     # 1 = channel present in recording
  'labels': {sig: FloatTensor (T,)},          # configured TRACES only, registry-normalized
  'label_mask': {sig: FloatTensor scalar},    # 1 = label present
  'filename': str, 'chunk_id': int,
}
```
A custom `collate_fn` stacks this dict shape. Recordings lacking a configured
label are KEPT (mask=0) rather than filtered out.

### 3. Backbone contract

A conforming model:
- constructor accepts `in_channels` (per transform block) and `out_signals`
- forward returns a raw tensor; per-frame models: (N*T, S); video models: (B, S, T)
- no dicts inside models

DeepPhys changes: respect `in_channels` in the `[:, :C]` / `[:, C:]`
diff/appearance split; `final_dense_2 -> Linear(nb_dense, out_signals)`;
compute dense-1 input size from img_size instead of the 36/72/96 lookup.

### 4. `SignalDictWrapper` (shared)

Wraps a backbone + configured trace list; reshapes raw output to
`{sig: (B, T)}`. Owns nothing else.

### 5. `MaskedMultiSignalLoss` (shared)

```python
# per signal: masked mean over batch, then mean over signals
per_sig = (base_loss_elementwise(pred[sig], label[sig]) * label_mask[sig]).sum() \
          / label_mask[sig].sum().clamp(min=1)
loss = mean(per_sig for sig in traces)
```
Base loss pluggable per signal (MSE, NegPearson initially; PhysHydraLoss
components later). A signal with no present labels in the batch contributes 0.

### 6. `MultiSignalTrainer` (shared)

One trainer for the Neckflix pipeline, skeleton from PhysHydraTrainer
(DDP, AMP guard, cuda->mps->cpu, checkpointing, config snapshot). Driven by a
model registry:
```python
MODEL_REGISTRY = {'DeepPhys': ModelEntry(builder, input_mode='frames2d')}
# input_mode 'frames2d': flatten (B,T,C,H,W)->(B*T,C,H,W) before forward,
#                        unflatten predictions after
# input_mode 'video3d':  pass (B,C,T,H,W) straight through (next phase)
```
`neckflix_main.py` uses the registry when the model name is in it; the legacy
elif chain and per-model trainers stay untouched for the HR pipeline.

### 7. Config schema

```yaml
PREPROCESS:
  CHANNELS: ['R','G','B']        # subset of canonical slots, any of R,G,B,I,D
  TRACES:   ['ABP','CVP']        # subset of SIGNALS
  DATA_TYPE: ['DiffNormalized','Standardized']
```
TRACES/CHANNELS become global preprocess keys (NECKFLIX.* keys deprecated but
readable during transition). Per-signal norm override: `SIGNAL_NORMS: {ABP: [0,200]}`.

## Error handling

- Config TRACES not in SIGNALS -> config error at startup.
- Config CHANNELS not in CHANNELS registry -> config error at startup.
- A batch where every label_mask for a signal is 0 -> that signal's loss term
  is exactly 0 (no NaN from 0/0; denominator clamped).
- Loader never silently drops recordings for missing labels; only missing
  *requested channels* that are also unmaskable (none, by design) would.

## Testing

Unit (pytest, `tests/`):
- loader: dict shape, concat semantics (DATA_TYPE blocks), zero-fill +
  channel_mask correctness on a synthetic HDF5 with/without I,D
- loss: hand-computed masked case (2 signals, one fully masked); no-NaN when
  a signal is absent from the whole batch
- DeepPhys contract: shapes for C in {3,5}, S in {1,2}; gradients flow
- wrapper: dict keys match TRACES, shapes (B,T)

End-to-end (manual/MPS): short train on the local subset, ABP+CVP, loss
descends; saved test outputs keyed by signal name.

## Out of scope (this pilot)

- 3D video models and remaining 2D models (next phases, same contracts)
- Other datasets' loaders emitting the dict contract (they keep legacy format
  until their phase; the HR pipeline in main.py is untouched)
- Channel-mask *consumption inside models* (mask is provided; models may
  ignore it — zero-filled channels are learnable-ignorable)
- HR derivation/eval integration for PPG
