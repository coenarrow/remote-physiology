# Neckflix Zarr Loader Rebuild — Design

**Date:** 2026-08-30
**Status:** Approved in session (deviations 1–5 accepted); revised after adversarial
spec review. Items flagged **[review-gate]** below extend the literal approval and
need a nod at spec review: the two-line `main.py` preservation edit, the
`zarr` dependency addition, and the `multisignal_collate.py` deletion.
**Branch:** `multisignal-pilot`

## Context

The current `dataset/data_loader/NeckflixLoader.py` reads a hand-rolled per-recording
HDF5 cache format and carries two output modes (legacy tuple mode feeding
`PhysHydraTrainer`, and an unwired flat-dict "multisignal" mode). The Neckflix repo
(github.com/coenarrow/Neckflix, v1.0.0) has since become a preprocessing CLI
distributed as a Docker image (`ghcr.io/coenarrow/neckflix`) that converts raw
recordings into **one zarr-v3 store per recording**. Downstream consumers do not
import the `neckflix` package; the contract is the zarr store schema plus two
root-attr gates. CardioHydra (`/Users/20759193/repos/CardioHydra`) already consumes
this contract with a lazy, metadata-only-construction dataset class, and is the
reference implementation this rebuild ports.

This rebuild replaces the Neckflix loader with a direct port of CardioHydra's
dataset stack (`src/dataset/base.py`, `src/dataset/neckflix.py`,
`src/transforms/labels.py`), with the approved deviations listed below.

## Decisions (user-confirmed)

1. **Batch contract:** CardioHydra nested dict, ported verbatim (not the
   multisignal-pilot flat dict).
2. **Legacy path:** clean break. Old loader (both modes) deleted; PhysHydra/
   `neckflix_main.py` stop working until a follow-up wiring task.
   **[review-gate]** Corollary discovered in review: `main.py` references the old
   class at module import time ([main.py:74](../../../main.py) registry entry,
   `:132` branch), so without a two-line edit `main.py` breaks for *every*
   dataset — which decision 2 did not intend. In scope: remove the `"Neckflix"`
   `LOADER_REGISTRY` entry and the Neckflix branch in `create_dataset`, so
   `main.py` keeps working for all other datasets ("Neckflix" then raises the
   existing "Unsupported dataset" `ValueError`).
3. **Cache:** regenerated with the ≥1.0.0 container. The loader gates strictly
   (see admission gate below). The 62 stores currently at
   `CardioHydra/cached_data/neckflix/` predate the gate and will be skipped until
   regenerated.
4. **Splits:** attribute filters only; no valid split. LOSO = two instances
   (train: `participant.exclude=[fold]`; test: `participant.include=[fold]`).
   `BEGIN`/`END` percentage slicing is gone.
5. **Label normalisation:** CardioHydra parity — per-window `zscore` or `minmax`
   chosen by `label_norm`, stats emitted in `label_stats`, exact inverses ported.
6. **Windowing:** CardioHydra parity — `random_windows: true` ⇒ one fresh
   random-start window per (recording, perspective) per epoch; `false` ⇒
   deterministic strided index over `window_size`/`window_stride`.
7. **Perspectives:** independent samples AND filterable (deviation 3).
8. **Scope:** loader + label transforms + tests only — plus, from review:
   the two-line `main.py` edit (decision 2 corollary) and the `zarr` dependency
   addition **[review-gate]** (unavoidable: the loader imports it; added to
   `pyproject.toml` only, see Dependency note).

## Scope

**In scope**

- `dataset/data_loader/zarr_dataset.py` — new `BaseZarrDataset`.
- `dataset/data_loader/NeckflixLoader.py` — rewritten to hold only
  `NeckflixDataset(BaseZarrDataset)`.
- `dataset/data_loader/label_transforms.py` — port of CardioHydra
  `src/transforms/labels.py` plus the finite-stats path (deviation 4).
- Deletion of the old loader code and `tests/test_neckflix_dict.py`;
  deletion of `dataset/data_loader/multisignal_collate.py` **[review-gate]**
  (extends decision 2's literal wording; it is dead code — its only consumers
  are itself and the deleted test).
- Two-line `main.py` preservation edit (decision 2 corollary).
- New pytest suite against synthetic zarr stores.
- Dependency: add `zarr>=3.3,<4` to `pyproject.toml` (not `requirements.txt` —
  see Dependency note).

**Out of scope (follow-up tasks)**

- Wiring `neckflix_main.py` (yacs → plain-dict translation, loader construction,
  DDP samplers, trainer dispatch, P-prefix → bare participant-id translation for
  `--test_participants`).
- Consumer rework: `SignalDictWrapper`, `MaskedMultiSignalLoss`, `signals.py`,
  and any trainer speak the old contracts; they stay in-tree untouched.
- Frame preprocessing (`DiffNormalized`/`Standardized`): the loader emits **raw
  pixel values as float32**; DATA_TYPE-style transforms move consumer-side.
- Cache regeneration itself (docker locally; Apptainer or native `uv` path on
  HPC), and choosing `--resize` / cache locations.
- Events (`EV`) support. ECG needs no code: list `"ECG"` in `labels`.
- Doc updates that the rebuild makes stale: `docs/architecture.md` (NeckflixLoader
  described as the HDF5 loader), `docs/changelog.md`, and CLAUDE.md's
  Neckflix/HDF5 sections.

## Architecture

Three modules, mirroring CardioHydra's split:

```text
dataset/data_loader/
  zarr_dataset.py       BaseZarrDataset (ABC, torch.utils.data.Dataset)
  NeckflixLoader.py     NeckflixDataset(BaseZarrDataset)  — channel map only
  label_transforms.py   zscore/minmax + inverses, finite-stats path, STAT_NAMES, EPS
```

`BaseZarrDataset` declares an abstract `channel_map` property — the single
subclass obligation. `NeckflixDataset` returns its class-level map from it:

```python
_CHANNEL_MAP = {
    "R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2),
    "I": ("ir", 0), "D": ("depth", 0),
}
```

### Consumed zarr store contract

`{cache_dir}/{recording}.zarr`, zarr v3 (written by `neckflix-preprocess` ≥1.0.0):

- root attrs: `recording`, `participant`, `session`, `repeat`, `posture`,
  `light`, `source_resolution`, `resized_to`, `tool_version`, `complete`.
  **Value format:** `participant` is an *unprefixed* zero-padded string
  (e.g. `"030"`, from `P030_...` — the writer strips the `P`); the repo's
  P-prefixed convention (`--test_participants P030`) must be translated by
  wiring code. `attribute_values("participant")` is the source of truth for
  fold ids.
- perspective groups `"1"`/`"2"`, each holding stream groups `rgb`/`ir`/`depth`:
  - `.../video/frames` `(C, T, H, W)` — uint8 (rgb, C=3) or uint16 (ir/depth, C=1),
    chunked `(C, 32, H, W)`.
  - `.../video` attrs: `fps`, `num_frames`.
  - `.../{abp,cvp,ecg}/data` `(T,)` float64, physical units, index-aligned 1:1
    with frames, duplicated per stream. Copies may differ in length (per-stream
    trailing-NaN trimming), may be **shorter than the stream's `num_frames`**,
    and may contain interior NaNs — deviations 4 and 7 handle both.
- optional root `events/` group. It is ignored for two distinct reasons at two
  levels: iterated as a perspective-level group it yields no stream sub-groups,
  so the empty-perspective check drops it; separately, any group *inside* a
  perspective lacking a `video` child (e.g. a bare trace group) is skipped.
  Tests cover both placements.

**Admission gate** per store, during `_scan_cache()`:

- Store unreadable (`zarr.open_group` raises) ⇒ skip with `warnings.warn`.
- `attrs.get("complete") is not True` ⇒ skip with warning (identity check:
  JSON boolean `true` only; `complete: 1` fails).
- `tool_version` parsed as a dot-separated int tuple; an unparseable value
  (e.g. `"abc"`, `"1.0.0-rc1"`) is treated as `(0,)`; anything
  `< (1, 0, 0)` ⇒ skip with warning (pre-1.0.0 frames are temporal-delta
  encoded and would silently decode to garbage).
- `FileNotFoundError` if `cache_dir` does not exist. `RuntimeError` if no
  admitted store contributes **at least one perspective with a video-bearing
  stream group** (an admitted store holding only `events/` does not count).

### Config dict

`BaseZarrDataset(cfg: dict)` consumes a plain dict (no yacs coupling):

| key | type | default | meaning |
|---|---|---|---|
| `cache_dir` | str | required | directory of `*.zarr` stores |
| `channels` | list[str] | required | ordered channel names, resolved via `channel_map` |
| `labels` | list[str] | required | trace names, e.g. `["ABP", "CVP"]` (`"ECG"` allowed) |
| `window_size` | int | required | window length in frames |
| `window_stride` | int | `window_size` | strided-mode stride |
| `random_windows` | bool | `False` | one random-start window per sample per epoch |
| `filters` | dict | `{}` | `{attribute: {"include": [...], "exclude": [...]}}` |
| `label_norm` | str | `"zscore"` | `"zscore"` or `"minmax"` (else `ValueError`) |
| `allow_missing` | bool | `False` | tolerate partial-modality samples |
| `min_channels` | int | `1` | with `allow_missing`: min present streams |
| `min_labels` | int | `1` | with `allow_missing`: min present labels |

**Deviation 1:** no `fps` key (CardioHydra stores it unused; store attrs are
authoritative).

### Construction pipeline (metadata-only)

`__init__` never reads pixel data — cheap enough to instantiate on a SLURM login
node for fold enumeration:

1. Validate `label_norm` (`ValueError` otherwise) and **validate `filters`
   upfront** (deviation 6): for every attribute, overlapping
   `include`/`exclude` ⇒ `ValueError`, raised unconditionally in `__init__`
   before any sample iteration. (CardioHydra checks lazily inside the per-sample
   loop, so an overlap can pass silently when no sample reaches it — this port
   is deliberately stricter.)
2. `_resolve_streams(channels)` → ordered `(stream_group, channel_index)` plan;
   unknown channel ⇒ `ValueError`. `required_streams` = sorted unique lowercase
   stream groups.
3. `_scan_cache()` → `{recording: {"attrs": {...}, perspective: {stream: {entries}}}}`
   over sorted `*.zarr`, applying the admission gate above.
4. `discover_samples()` → sorted `(recording, perspective)` pairs, recording
   `present_streams` / `present_labels` per sample. `allow_missing` keeps a
   sample with ≥ `min_channels` present streams and ≥ `min_labels` present
   labels; strict mode requires every required stream present and every stream
   in the perspective carrying every configured label (CardioHydra parity).
5. `_filter_by_attribute(filters)` — see below.
6. `_load_windows()` → `windows: list[(recording, perspective, start | None)]`.
   Strided: `starts = range(0, frame_count - window_size + 1, window_stride)`
   (e.g. `frame_count=10, window_size=4, window_stride=3` → starts 0, 3, 6);
   random: one `(rec, persp, None)` entry per sample. Frame count = min
   `num_frames` attr across the sample's present streams; samples shorter than
   `window_size` are skipped in both modes.

`__len__` = `len(windows)`. Determinism: `sorted()` at every enumeration point.

### Filtering semantics

`filters = {attribute: {"include": [...], "exclude": [...]}}`, evaluated per
sample:

- A value in `exclude` drops the sample; non-empty `include` whitelists.
- Overlap validation is upfront (deviation 6, above).
- **Deviation 2 (missing-attr safety):** attribute values come from
  `attrs.get(attribute, MISSING)` instead of `attrs[...]`. A sample whose store
  lacks the attribute *fails* any non-empty `include` (membership unprovable)
  and *passes* an exclude-only filter. After the filter pass completes, one
  `UserWarning` per attribute that had ≥1 missing-attr sample (regardless of
  include/exclude outcome), listing the affected store names. (CardioHydra
  raises `KeyError`.)
- **Deviation 3 (perspective filterable):** the pseudo-attribute
  `"perspective"` is evaluated against the sample's perspective key
  (`"1"`/`"2"`), not root attrs. Filter values for `perspective` are
  `str()`-coerced before comparison, so the natural YAML spelling
  `include: [1]` works. All other attributes read root attrs (`participant`,
  `posture`, `light`, `session`, `repeat`, ...) and compare as stored. Note the
  store attr is named `posture` — configs filter on `posture`, not `position`.

LOSO usage (by the future wiring code, and by tests) — note the unprefixed
participant format:

```python
train_cfg["filters"]["participant"] = {"include": [], "exclude": ["030"]}
test_cfg["filters"]["participant"]  = {"include": ["030"], "exclude": []}
```

### `__getitem__` contract

Per access: resolve a `None` start via `torch.randint(0, max_start + 1, (1,))`;
open the store with `zarr.open_group(..., mode="r")` (no held handles — safe for
`num_workers > 0`); slice `frames[:, start:end]` per present stream and
`{trace}/data[start:end]` per present label; close implicitly. Returns:

```python
{
  "frames":       {ch: Tensor (1, T, H, W) float32},   # every configured channel;
                                                       # raw pixel values, zero-filled
                                                       # where the stream is absent
  "labels":       {sig: Tensor (T,) float32},          # per-window normalised
  "label_stats":  {sig: {"mean"|"std"|"min"|"max": Tensor ()}},  # physical units
  "channel_mask": {ch: Tensor () bool},                # True = real data
  "label_mask":   {sig: Tensor () bool},
  "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int},
}
```

Details (parity unless marked):

- Zero-fill shapes come from `_ensure_stream_shapes()`: canonical `(H, W)` per
  required stream, inferred lazily by scanning stores; a stream never seen
  anywhere falls back to the first real shape found.
- **Deviation 7 (short-trace tolerance):** each per-stream trace slice
  `data[start:end]` shorter than `window_size` (trailing-NaN trim made the trace
  shorter than the video) is right-padded with NaN to `window_size`. Copies are
  then averaged **position-wise over finite values** (`np.nanmean` semantics,
  with the all-NaN-position warning suppressed); positions where every copy is
  NaN remain NaN and are absorbed by deviation 4. (CardioHydra crashes with
  `ValueError` on unequal copy lengths and violates the `(T,)` contract on a
  single short copy — this port is deliberately robust.)
- **Deviation 4 (interior-NaN guard):** per label window, on the averaged trace
  `raw`: compute stats over **finite entries only** (`finite_stats`); normalise
  `raw` with those precomputed stats (`apply_norm` — the dataset never re-calls
  the stat-recomputing `zscore()`/`minmax()`); then set every non-finite
  position of the normalised output to **exactly 0** — uniformly for both
  `label_norm` modes. The finite-only stats are what `label_stats` emits, so the
  inverse round-trip is exact at finite positions; a zeroed NaN position maps
  back to `mean` (zscore) or `min` (minmax). A window with **no** finite entries
  is treated as absent (`label_mask=False`, zero trace, zero stats). On an
  all-finite window this path is bit-identical to the verbatim
  `zscore()`/`minmax()` (tested). (CardioHydra would propagate NaN into the
  loss.)
- Absent labels emit exact zeros with zero stats; both transforms clamp their
  denominator with `EPS = 1e-8` so `0/EPS == 0` (never NaN — load-bearing for
  masked losses, where `NaN * 0` is still NaN).
- `recording_id` = root attr `recording`, falling back to the store filename
  stem.
- Default `torch.utils.data.default_collate` handles the nesting: per-sample
  `(1,T,H,W)` → `(B,1,T,H,W)`, `()` bools → `(B,)`, stats `()` → `(B,)`,
  metadata strings → lists, `start_frame` ints → `(B,)` int64 tensor. No custom
  collate_fn exists or is needed.
- **Deviation 5 (`attribute_values` helper):**
  `attribute_values(attribute: str) -> list[str]` returns sorted unique values
  of a root attr (or `"perspective"`) over the *admitted, filtered* samples —
  the fold-enumeration primitive for SLURM fan-out. Values are `str()`-coerced
  before sorting (mixed-type attrs sort safely); samples whose store lacks the
  attribute are silently skipped.

### Label transforms (`label_transforms.py`)

Port of CardioHydra `src/transforms/labels.py`:

- `STAT_NAMES = ("mean", "std", "min", "max")`, `EPS = 1e-8`.
- Verbatim: `zscore(trace) -> (normed, stats)`; `minmax(trace) -> (normed, stats)`
  (to `[0, 1]`); `zscore_inverse(sig, stats)` / `minmax_inverse(sig, stats)`
  with identical `clamp_min(EPS)` in forward and inverse (exact round-trip);
  `_align` right-pads stats for both `(T,)` and collated `(B, T)` inputs;
  `std` is the unbiased (`correction=1`) estimate.
- Additions for deviation 4 (the verbatim functions above are untouched):
  - `finite_stats(trace) -> stats` — the four stats over finite entries only;
    all-NaN input yields zero stats.
  - `apply_norm(trace, stats, mode) -> normed` — the forward zscore/minmax
    formulas using the *given* stats (no recomputation), same `EPS` clamps.
  - On all-finite input, `apply_norm(t, finite_stats(t), mode)` is bit-identical
    to `zscore(t)`/`minmax(t)`.

## Deletions and accepted breakage

- Old `NeckflixLoader` implementation (tuple mode, flat-dict mode,
  `get_cached_file_list`, `resize_frames`, `normalise_trace`, `diffnorm`,
  `zstand`, HDF5 reading) — deleted.
- `dataset/data_loader/multisignal_collate.py` and
  `tests/test_neckflix_dict.py` — deleted (dead code / tests of the dead
  contract; **[review-gate]** extends decision 2's literal wording).
- `main.py` — two-line edit (registry entry + `create_dataset` branch) so it
  keeps working for all non-Neckflix datasets; `"Neckflix"` there now raises the
  existing "Unsupported dataset" `ValueError`.
- `neckflix_main.py` imports `NeckflixLoader` and constructs it with the old
  signature: it will fail until the follow-up wiring task. This is accepted.
- `SignalDictWrapper`, `MaskedMultiSignalLoss`, `neural_methods/signals.py`,
  all trainers and models: untouched, awaiting consumer rework.
- The old externally-produced HDF5 cache is no longer readable by this repo.

## Testing

Pytest suite (new `tests/test_neckflix_zarr.py` + a synthetic-store builder
fixture, e.g. `tests/zarr_fixtures.py`) writing small zarr-v3 stores into
`tmp_path` with realistic attrs/dtypes (uint8 RGB C=3, uint16 IR/depth C=1,
float64 traces, per-stream trace copies, `num_frames` attrs):

- **Gating:** missing `complete`, `complete: false`, `complete: 1` (identity,
  not equality), `tool_version` `"0.9.0"` / malformed / missing, unreadable
  store ⇒ skipped with warning; empty cache dir and sole-store-with-only-events
  ⇒ `RuntimeError`; missing dir ⇒ `FileNotFoundError`; root `events/` group and
  a video-less group inside a perspective both ignored (two placements, two code
  paths).
- **Config validation:** unknown channel name ⇒ `ValueError`;
  `label_norm: "foo"` ⇒ `ValueError`.
- **Discovery:** strict all-present rule vs `allow_missing` with
  `min_channels`/`min_labels` thresholds; `present_streams`/`present_labels`
  bookkeeping; sorted sample order.
- **Filters:** include, exclude, both across attributes; overlap ⇒ `ValueError`
  upfront even when zero samples would reach the attribute (deviation 6);
  missing-attr semantics incl. the post-pass `UserWarning` contract
  (deviation 2); `perspective` filtering incl. int-value coercion
  (deviation 3); LOSO include/exclude construction with unprefixed ids yields
  disjoint participant sets.
- **Windowing:** strided starts match
  `range(0, frame_count - window_size + 1, window_stride)` for exact and
  non-multiple lengths; short-sample skip in both modes; random mode = one
  window per sample, start within bounds, varies across accesses; frame count =
  min across present streams.
- **Item contract:** exact keys, shapes, dtypes; every configured channel/label
  present; zero-fill exactness and canonical fill shapes; mask semantics; raw
  pixel passthrough (no scaling); trace averaging across differing stream
  copies; `label_stats` round-trips through the matching inverse to the raw
  window at finite positions; `minmax` output in `[0, 1]`.
- **NaN guard (deviation 4):** interior NaNs ⇒ finite-only stats, **exactly 0**
  at NaN positions post-normalisation under *both* `label_norm` modes, mask
  stays True; all-NaN window ⇒ mask False, zero trace/stats; no NaN anywhere in
  the emitted item; all-finite window bit-identical to verbatim
  `zscore`/`minmax`.
- **Short traces (deviation 7):** a trace copy shorter than `num_frames` with a
  window overlapping the trimmed tail ⇒ NaN-padded, position-wise finite-mean
  across copies, guard absorbs the tail; no crash, `(T,)` shape holds.
- **Helper (deviation 5):** `attribute_values("participant")` /
  `("perspective")` over filtered samples; str-coercion; missing-attr samples
  skipped.
- **Collate:** `default_collate` over a batch of items produces the documented
  batched shapes, including `start_frame` → `(B,)` int64.
- **Determinism:** two constructions over the same cache yield identical
  `windows`.
- **main.py preservation:** `import main` succeeds post-rebuild and
  `get_loader_class("Neckflix")` raises the "Unsupported dataset" `ValueError`
  (smoke-level; no training run).

## Dependency note

`zarr>=3.3,<4` added to `pyproject.toml` only. `requirements.txt` is stale for
this repo (pins `numpy==1.24.3`, incompatible with zarr 3.x's numpy ≥1.26 floor,
while `pyproject.toml` already pins `numpy==2.3.2`; the project runs under `uv`)
and is left untouched. `h5py`/`hdf5plugin` remain (other loaders use them). No
dependency on the `neckflix` package — the zarr schema is the only coupling,
matching CardioHydra.
