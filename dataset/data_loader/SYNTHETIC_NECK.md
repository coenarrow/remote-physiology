# Synthetic Neck Cache Spec

Synthetic neck videos with ground-truth arterial (ABP) and central venous
(CVP) pressure, ECG, finger PPG and respiration, rendered by
`synthetic-neck`, the generator carried as a
submodule at `tools/synthetic_datasets/synthetic_neck`. There is no raw
dataset and no separate cacher: `synthetic-neck generate --zarr` renders
straight into stores that satisfy [the cache contract](../../docs/cache-contract.md).
The validator is the acceptance test. Generator design:
`tools/synthetic_datasets/synthetic_neck/docs/superpowers/specs/2026-09-15-zarr-output-design.md`.

## Building

```bash
uv run --project tools/synthetic_datasets/synthetic_neck synthetic-neck generate --zarr \
    --preset neckflix --n 200 --jobs 8 --out <cache dir>
uv run python tools/validate_cache.py <cache dir>
```

Presets: `lesson` (large pulse, flat lighting), `benchmark` (moderate pulse,
controlled degradations), `neckflix` (ranges calibrated from Neckflix).
`--set block.field=value` overrides any generator field. `--seed` is the base
seed (sample `i` uses `seed + i`), so a cache can be rebuilt from its
`dataset.json`.

**Clip length versus the model window.** The default clip is
`video.duration_s` = 10 s at 30 fps, i.e. 300 frames. A model window of 10 s
at FS 30 is also 300 frames, so each sample yields exactly one window:
random-window training sees a fixed crop, not a resampled one. A config
whose window is longer than the clip skips every synthetic store, with a
warning — `src/inputs.py`'s per-store frame-count check (around line 197)
skips any sample whose `frame_count` is shorter than the window's native
span. If you need longer or more varied clips, set
`--set video.duration_s=<seconds>` to at least the window length, and longer
if you want window variety.

## What the generator writes

```text
{out}/
|-- dataset.json                    preset, overrides, seeds, generator version and commit, "format": "zarr"
`-- 1.zarr                          one store per sample; the name is the sample index
    |-- attrs                       see "Root attributes"
    |-- vessel_ids   (H, W) uint8   0 background, 1 artery, 2 vein; attrs: labels
    `-- 1/                          attrs: fps = video.fps (30 by default)
        |-- rgb/
        |   |-- timestamps_us/data  (T,) int64          round(i * 1e6 / fps)
        |   |-- video/data          (3, T, H, W) uint8   Delta + blosc-zstd
        |   |-- abp/data            (T,) float64        attrs: units="mmHg"
        |   |-- cvp/data            (T,) float64        attrs: units="mmHg"
        |   |-- ecg/data            (T,) float64        attrs: units="mV"
        |   |-- ppg/data            (T,) float64        attrs: units="arb"
        |   `-- rr/data             (T,) float64        attrs: units="arb"
        |-- ir/                     only when the sample drew IR + depth
        |   `-- video/data          (1, T, H, W) uint8, plus rgb's three siblings
        `-- depth/                  only when the sample drew IR + depth
            `-- video/data          (1, T, H, W) float32 mm, blosc-zstd without Delta, plus rgb's three siblings
```

- **Frames**: rendered square at `video.frame_size` (300 by default), with
  no resize at write time; resizing is consumer-side as usual. Chunks are
  `(C, min(32, T), H, W)`. Every stored signal has a footprint: the carotid
  and jugular masks carry the delayed ABP and CVP (darkening in RGB and IR,
  a depth lift); every skin pixel carries the PPG at neck timing (a uniform
  darkening, weaker than the carotid, no depth lift); the whole frame
  brightens and moves nearer with `rr`; ECG is present as rate only. The
  submodule's `docs/superpowers/specs/2026-09-16-frame-physiology-design.md`
  gives the terms.
- **Timestamps**: synthesised from the nominal rate, starting at 0 and
  identical in every modality.
- **`abp`, `cvp`, `ecg`, `ppg`, `rr`**: the generator's 1 kHz traces, linearly
  interpolated at the frame times, identical under every modality. ABP is
  stored at a drawn catheter site (`abp_site`, radial or brachial); the
  carotid pixels are rendered from it shifted back to central timing. ECG is
  the McSharry ECGSYN waveform in mV. PPG is a finger pulse and `rr` a chest
  excursion in [0, 1], both `arb`. All five share one cardiac timeline and
  one respiratory waveform; the generator's spec
  (`docs/superpowers/specs/2026-09-16-physiological-traces-design.md` in the
  submodule) tabulates the delays and their sources.
- **`depth`**: float32 millimetres. That is Neckflix's unit but not its
  integer dtype, so the pulse's 0.3-0.5 mm skin lift survives. Kinect-like
  noise (1.6 mm sd at 1 m, growing with distance squared) is already in it.
- **`ir`**: uint8, not Neckflix's uint16 Kinect IR; the two scales are not
  comparable.
- **`vessel_ids`**: one static map per sample (nothing in the scene moves
  in-plane), on the frames' pixel grid. It is not part of the contract, and the
  validator and the reader only walk root groups, so neither sees it. If
  frames are resized consumer-side, resize this map nearest-neighbour to
  match.
- **Guaranteed pulse**: the generator redraws a sample, up to 20 times,
  until the green channel's region-averaged cardiac SNR is at least 10 for
  the artery, the vein and the skin. IR and depth carry no such guarantee.

### Root attributes

```python
{
  "participant": "1",          # the sample index; every sample is its own participant
  "recording": "1",            # same value
  "posture": "recumbent",      # supine | recumbent | sitting, from trace.posture_deg 0 | 45 | 90
  "abp_site": "radial",        # radial | brachial, from trace.abp_site
  "monk_tone": 6,              # Monk skin tone 1-10 when the preset draws one, else null
  "preset": "neckflix",
  "seed": 2027,                # this sample's seed: base seed 2026 + index 1
  "synthetic_neck": {...},     # full per-sample metadata: config, drawn params, geometry_px,
                               # derived quantities, visibility (SNR, attempts)
}
```

Configs filter on these as written, e.g.
`posture: {include: [supine], exclude: []}`. Drawn values are reachable by
dotted path (`synthetic_neck.params.trace.heart_rate_bpm`), but filters match
exact strings, so they select exact values, not ranges.

## Mixing with other datasets

The ids `"1"`, `"2"`, ... collide with other datasets' ids. Hold out a
participant together with its dataset:
`--test-participant-dataset synthetic_neck --test-participant-id 3`. The
dataset name is the stem of `configs/datasets/synthetic_neck.yaml`.

**LOSO granularity.** `participant` is the sample index, so a
leave-one-subject-out sweep (`main.py`) with no `--test-participant-id` runs
one fold per unique participant id admitted for the dataset — one fold per
sample, since every sample is its own participant. A 200-sample cache means
200 folds.

## Liberties taken

- Timestamps are synthesised, not measured.
- Traces are interpolated from 1 kHz onto the frame grid.
- Depth is float32 rather than a sensor's integer millimetres.
- `vessel_ids` is an uncontracted root array.
- The `synthetic_neck` metadata keeps the generator's folder-layout fields
  verbatim (`vessel_ids.file`, `depth_units_mm`, `rgbid_streams`). Inside a
  store they describe the folder layout, not the store.
