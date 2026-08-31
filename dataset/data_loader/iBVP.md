# iBVP Cache Spec

iBVP is an RGB-thermal rPPG dataset with high-resolution signal-quality
labels: paired 3-channel RGB and radiometric thermal face videos with a
PhysioKit-acquired BVP ground truth and per-sample signal-quality vectors.
Citations: Joshi & Cho 2024, "iBVP Dataset: RGB-Thermal rPPG Dataset with
High Resolution Signal Quality Labels", Electronics 13(7):1334,
<https://doi.org/10.3390/electronics13071334>; Joshi, Wang & Cho 2023,
"PhysioKit", Sensors 23(19):8244, <https://doi.org/10.3390/s23198244>.
Dataset repository: <https://github.com/PhysiologicAILab/iBVP-Dataset>.
This spec describes how to write `{recording}.zarr` stores for iBVP
conforming to the cache contract in docs/architecture.md; it replaces the
deleted legacy loader (`iBVPLoader.py`, retrievable at the `pre-overhaul`
git tag).

## Raw layout

```
iBVP_Dataset/
|-- p01_a/
|   |-- p01_a_rgb/          *.bmp frame sequence (RGB camera)
|   |-- p01_a_t/            *.raw frame sequence (thermal camera)
|   |-- p01_a_bvp.csv       BVP + signal-quality columns
|-- p01_b/
|   |-- p01_b_rgb/
|   |-- p01_b_t/
|   |-- p01_b_bvp.csv
|...
|-- pii_x/
```

The legacy loader discovered recordings with
`glob.glob(data_path + os.sep + "*_*")` — any directory whose name
contains an underscore. Within a recording directory named `{name}`, it
built paths as `{name}_rgb/`, `{name}_t/`, and `{name}_bvp.csv` (the
frame-folder globs were `{dir}/{name}_rgb/*.bmp` and `{dir}/{name}_t/*.raw`,
each `sorted()` lexicographically — frame order therefore depends on
zero-padded filenames; the filename pattern inside the folders is not
recorded in the loader).

## Video

**RGB** (`{name}_rgb/*.bmp`): one BMP per frame, read with `cv2.imread`
(8-bit BGR) then `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. Result per
recording: `(T, H, W, 3)` uint8, values 0–255. The loader's own comment
states `rgb_height = 480`; the RGBT concatenation (below) only works if the
RGB width equals the thermal width, so RGB frames are 640x480. No fps or
timestamp data exists in the raw layout; the only in-repo configs declare
`FS: 30` for iBVP.

**Thermal** (`{name}_t/*.raw`): one headerless binary file per frame, read
as

```python
np.fromfile(raw_path, dtype=np.uint16, count=640 * 512).reshape(512, 640)
```

i.e. `im_width = 640`, `im_height = 512`, row-major uint16. The loader then
converted to degrees Celsius as float32:

```python
celsius = counts.astype(np.float32) * 0.04 - 273.15
```

(0.04 K per count, Kelvin-to-Celsius offset 273.15), and appended a
trailing channel axis, giving `(T, 512, 640, 1)` float32.

The loader's `PREPROCESS.IBVP.DATA_MODE` selected `RGB` (default in
config.py), `T`, or `RGBT`. In `RGBT` mode both sequences were truncated to
`min(rgb_length, thermal_length)` frames and the thermal frames were
cropped to the first 480 rows (`thermal[:, :480, :, :]`) so they could be
channel-concatenated with RGB. Note the `T` mode called the *BMP* reader on
the `_t` folder (globbing `*.raw` never happens on that path), which
returns an empty array if the thermal folders hold only `.raw` files —
either the released thermal folders also contain BMP renders, or T-only
mode was simply broken; the code does not disambiguate.

## Physiological traces

**`bvp`** — from `{name}_bvp.csv`, read via `pd.read_csv(f).to_numpy()`
(the first CSV row is consumed as a header and discarded). The loader used
columns strictly by position:

- column 0 → the BVP waveform (`waves`);
- column 3 → a signal-quality vector, commented `#SQ2` in the code
  (`sq_vec`);
- columns 1 and 2 exist (the file has at least 4 columns) but were
  **discarded** — per the dataset publication they are additional
  signal-quality measures, but the loader never named them.

The CSV carries no timestamp column that the loader used. The native BVP
sampling rate is not recorded anywhere in the loader; alignment was a pure
linear stretch of the whole trace onto the whole frame sequence
(`BaseLoader.resample_ppg`, reproduced exactly):

```python
np.interp(np.linspace(1, N, target_length),
          np.linspace(1, N, N), signal)      # N = len(signal)
```

where `target_length` was the frame count (in `RGBT` mode, the truncated
common length `min(rgb_length, thermal_length)`). This assumes the BVP
record and the frame sequence span the same wall-clock interval at uniform
rates. The quality vector was resampled with the same function.

**`sq2`** — the column-3 quality vector, proposed here as its own trace so
the legacy quality gate stays reproducible (see Quirks). The legacy
pipeline discarded it after using it; the cache keeps it.

## Identity and attributes

Recording directories are named `p{NN}_{c}`: a participant token `pNN`
(two-digit, zero-padded) and a single-letter condition suffix after the
underscore. The loader formed its index by deleting the underscore
(`p01_a` → `p01a`) and took the subject as the **first three characters**
of that (`p01`) — a parse that silently breaks for participant numbers
wider than two digits. It never interpreted the condition letter; the
docstring shows `a` and `b` (and a generic `x`), and the dataset
publication describes the letters as experimental conditions — the exact
letter set is not enumerable from the code.

Proposed root attrs:

- `recording`: the raw directory name, e.g. `"p01_a"`;
- `participant`: `"01"` — zero-padded, unprefixed, matching the Neckflix
  convention (design choice: the legacy loader keyed subjects as `"p01"`;
  the constant `p` prefix carries no information);
- `condition`: the underscore suffix, e.g. `"a"` (needed for
  condition-based filtering; the legacy loader never exposed it).

## Proposed store mapping

One store per recording directory: `p01_a.zarr`.

RGB and thermal are co-located cameras filming the same face — one view,
two modalities — so they are **streams under a single perspective**
(design choice; nothing in the raw data forces this reading):

```
p01_a.zarr
  attrs: recording, participant, condition, complete: true,
         tool_version (>= "1.0.0")
  1/
    rgb/
      video/frames        (3, T_rgb, 480, 640) uint8   attrs: num_frames, fps
      bvp/data            (T_rgb,) float64
      sq2/data            (T_rgb,) float64
    t/
      video/frames        (1, T_t, 512, 640) uint16    attrs: num_frames, fps
      bvp/data            (T_t,) float64
      sq2/data            (T_t,) float64
```

- Channel order in `rgb` is R, G, B (post BGR→RGB conversion).
- **Thermal dtype is a design choice**: store the raw uint16 sensor counts
  and record the calibration (`celsius = counts * 0.04 - 273.15`) in the
  stream attrs, rather than baking in the float conversion. Consumers that
  z-score are unaffected (the mapping is affine), but `DiffNormalized` is
  *not* affine-invariant, so a consumer wanting the legacy Celsius values
  must apply the conversion itself. Storing float32 Celsius instead would
  reproduce the loader byte-for-byte at the cost of doubling size and
  leaving the raw-counts domain.
- Store both streams at **full native length and resolution** — do not
  replicate the legacy min-length truncation or the 512→480 thermal crop.
  The zarr loader windows each sample over the minimum `num_frames` across
  requested streams, which reproduces the truncation, and resizing is
  consumer-side.
- Write `bvp` and `sq2` under **both** streams, each linearly resampled
  (formula above) to that stream's own frame count, so thermal-only
  configurations still see the traces. The legacy loader resampled once to
  the truncated common length; per-stream resampling differs from it by at
  most the tail frames of the longer stream (design choice, flagged).
- A suggested `channel_map` for the future dataset class:
  `{"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2), "T": ("t", 0)}`.

## Quirks

- **Signal-quality frame gate**: after resampling, the loader dropped every
  frame (and BVP/quality sample) where `sq_vec <= 0.3`
  (`np.delete(..., axis=0)`), creating temporal discontinuities *before*
  chunking. The cache does not replicate this — stores keep all frames and
  the `sq2` trace so consumers can reapply the exact `<= 0.3` rule (or
  not). Any comparison against legacy results must account for this gate.
- Thermal calibration constants: `* 0.04 - 273.15` (uint16 counts →
  Celsius), frame geometry 640x512 uint16 row-major, no file header.
- RGBT thermal crop kept rows `[:480]` (top rows) to match the RGB height.
- `DATA_MODE == "T"` used the BMP reader on the `_t` folder (see Video) —
  likely broken; do not treat it as evidence of thermal BMPs.
- `USE_PSUEDO_PPG_LABEL` mode was broken for iBVP: `sq_vec` is only bound
  by `read_wave`, but the code unconditionally resampled and thresholded it
  afterwards, so pseudo-label runs would raise `NameError`.
- The BVP CSV's header row and columns 1–2 are silently discarded; column
  indices, not names, are load-bearing.
- Subject parse `subject_trail_val[0:3]` assumes exactly two digits after
  `p`.
- Frame ordering relies on lexicographic `sorted()` of filenames in both
  frame folders.
- The only in-repo iBVP configs set `FS: 30` and `IBVP.DATA_MODE: RGB`;
  `config.py`'s default `DATA_MODE` is also `'RGB'`.

## Not replicated

Face cropping (Haar/YOLO5Face), spatial resizing, chunking into
`CHUNK_LENGTH` windows, `DiffNormalized`/`Standardized` pixel and label
normalisation, POS pseudo-labels, the `BEGIN`/`END` subject-fraction split
(sorted-subject-list slicing) and the `.npy`/file-list cache are all
consumer-side or obsolete in the new pipeline. The store carries raw
full-length frames and physical-unit traces only; splits become root-attr
filters.
