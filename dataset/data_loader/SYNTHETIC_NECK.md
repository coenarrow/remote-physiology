# Synthetic Neck Cache Spec

Synthetic neck videos with ground-truth arterial (ABP) and central venous
(CVP) pressure, ECG, finger PPG and respiration, rendered by
`synthetic-neck`, the generator carried as a
submodule at `tools/synthetic_neck`. There is no raw
dataset and no separate cacher: `synthetic-neck generate` renders
straight into stores that satisfy [the cache contract](../../docs/cache-contract.md).
The validator is the acceptance test. Generator design, in the submodule:
`docs/trace_generation.md` (the traces) and `docs/frame_rendering.md` (camera,
geometry, appearance, distension, optics, sensor).

## Building

```bash
# data/synthetic_neck is what configs/datasets/synthetic_neck.yaml reads; data/ is gitignored
uv run python tools/synthetic_neck_cli.py generate \
    --n 200 --size 128 --jobs 1 --out data/synthetic_neck
uv run python tools/validate_cache.py data/synthetic_neck
```

Every prior is drawn once per sample from a priors YAML, by default the
submodule's `priors/base.yaml`; each value is a fixed number, a uniform
`[lo, hi]` or a clipped normal `{MEAN, SD, MIN, MAX}`. To change a prior,
copy that file and pass the copy with `--priors`. `--seed` is the base seed
(sample `i` uses `seed + i`) and `--start` the first sample index, so a
second run with a later `--start` adds samples to an existing cache.
`--size` area-averages the frames down from the native 650 px crop; without
it they are stored at 650 px.

`dataset.json` records the priors file and every value in it, the base seed,
`start`, `n`, the generator version and git commit, and any samples that
failed, so a cache can be rebuilt from it.

**Clip length versus the model window.** Every clip is `DURATION_S` = 30 s
at `SAMPLE_RATE_HZ` = 30 fps, i.e. 900 frames, the length of a Neckflix
recording. A 10 s window at FS 30 is 300 frames, so each sample yields
several windows. A config whose window is longer than the clip skips every
synthetic store, with a warning — `src/inputs.py`'s per-store frame-count
check skips any sample whose `frame_count` is shorter than the window's
native span. For longer clips, raise `DURATION_S` in a copied priors file.

## What the generator writes

```text
{out}/
|-- dataset.json                    priors path and values, seeds, start, n, generator version and commit, failures
`-- 1.zarr                          one store per sample; the name is the sample index
    |-- attrs                       see "Root attributes"
    |-- vessel_ids   (H, W) uint8   0 background, 1 artery, 2 vein; attrs: labels
    |-- neck_mask    (H, W) uint8   1 where a pixel sees the neck, else 0
    `-- 1/                          attrs: fps = 30
        |-- rgb/
        |   |-- timestamps_us/data  (T,) int64          round(i * 1e6 / fps)
        |   |-- video/data          (3, T, H, W) uint8   Delta + blosc-zstd
        |   |-- abp/data            (T,) float64        attrs: units="mmHg"
        |   |-- cvp/data            (T,) float64        attrs: units="mmHg"
        |   |-- ecg/data            (T,) float64        attrs: units="mV"
        |   |-- ppg/data            (T,) float64        attrs: units="arb"
        |   `-- rr/data             (T,) float64        attrs: units="arb"
        |-- ir/
        |   `-- video/data          (1, T, H, W) uint16, plus rgb's six siblings
        `-- depth/
            `-- video/data          (1, T, H, W) uint16 mm, plus rgb's six siblings
```

- **Frames**: square, `--size` pixels (650 without it). Chunks are
  `(C, min(32, T), H, W)`, Delta + blosc-zstd for all three modalities.
  Every sample carries rgb, ir and depth. The carotid and jugular do not
  colour the skin; they lift it by an amount set by the carotid pressure and
  the CVP (noise-free, delayed along each vessel) and their compliances,
  which the camera sees through shading and the depth stream. Every skin
  pixel also carries a uniform skin pulse generated as the finger PPG is,
  with its respiratory modulation. The submodule's `docs/frame_rendering.md`
  gives the terms.
- **Timestamps**: synthesised from the nominal rate, starting at 0 and
  identical in every modality.
- **`abp`, `cvp`, `ecg`, `ppg`, `rr`**: the generator's 1 kHz traces,
  resampled to the frame rate so that trace sample `j` and frame `j` share a
  timestamp, identical under every modality. ABP is stored at a drawn
  catheter site (`abp_site`, radial or brachial) with catheter noise; the
  carotid pixels are rendered from the carotid pressure instead. CVP carries
  its a, c, x, v, y landmarks and catheter noise. ECG is the ECGSYN beat in
  mV. PPG is a finger pleth, systole up; `rr` the respiratory waveform in
  [-1, 1], +1 at end-inspiration; both `arb`. All five share one cardiac
  timeline and one respiratory waveform; the submodule's
  `docs/trace_generation.md` gives the delays and their sources.
- **`depth`**: uint16 integer millimetres, Neckflix's unit and dtype. The
  sub-millimetre skin lift survives quantisation because the generator rounds
  stochastically: a 0.3 mm lift moves 30 % of the pixels over the vessel by
  one millimetre. Sensor noise is added before rounding.
- **`ir`**: uint16, on the level of Neckflix's Kinect IR (skin around 2200,
  drawn from `APPEARANCE.IR_LEVEL`), lit from the camera.
- **`vessel_ids`, `neck_mask`**: one static map each per sample (nothing in
  the scene moves in-plane), on the frames' pixel grid, resampled from the
  native crop by area majority. Neither is part of the contract, and the
  validator and the reader only walk root groups, so neither sees them. If
  frames are resized consumer-side, resize these maps nearest-neighbour to
  match.
- **No redraws**: every sample is kept as drawn. The jugular lifts the skin in
  every sample, by an amount set by its CVP and compliance and recorded in
  `synthetic_neck.derived`; nothing guarantees a minimum pulse SNR in any
  modality.

### Root attributes

```python
{
  "participant": "1",          # the sample index; every sample is its own participant
  "recording": "1",            # same value
  "posture": "supine",         # supine < 30 deg <= recumbent <= 60 deg < sitting, from the drawn scene.posture_deg
  "abp_site": "radial",        # radial | brachial
  "skin_tone": 3,              # Monk skin tone 1-10 (core attr, docs/cache-contract.md)
  "neck_circumference_cm": 38.2,   # from the drawn scene.neck_radius_mm (core attr)
  "seed": 2027,                # this sample's seed: base seed 2026 + index 1
  "synthetic_neck": {...},     # seed, output_px, n_frames, abp_site, posture, version, priors path,
                               # and every drawn value by stage: traces, camera, scene, propagation,
                               # distension, appearance, optics, sensor; plus derived quantities
}
```

Configs filter on these as written, e.g.
`posture: {include: [supine], exclude: []}`. Drawn values are reachable by
dotted path (`synthetic_neck.traces.heart_rate_bpm`), but filters match
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
- Traces are resampled from 1 kHz onto the frame grid.
- `vessel_ids` and `neck_mask` are uncontracted root arrays.
- `synthetic_neck.priors` is the absolute path of the priors file on the
  machine that generated the cache; `dataset.json` keeps its values.
