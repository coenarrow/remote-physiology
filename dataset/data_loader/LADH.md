# LADH Cache Spec

LADH ("Non-Contact Health Monitoring During Daily Personal Care Routines")
is a multi-camera dataset of paired RGB and IR face videos recorded across
multiple days and daily-care scenarios, with BVP, HR, RR and SpO2 ground
truth. Citation: Xulin Ma, Jiankai Tang, Zhang Jiang, Songqin Cheng,
Yuanchun Shi, Dong Li, Xin Liu, Daniel McDuff, Xiaojing Liu, Yuntao Wang,
"Non-Contact Health Monitoring During Daily Personal Care Routines",
IEEE-EMBS BSN, 2025. Dataset repository:
<https://github.com/McJackTang/FusionVitals/>. This spec describes how to
write `{recording}.zarr` stores for LADH conforming to the cache contract
in docs/architecture.md; it replaces the deleted legacy loader
(`LADHLoader.py`, retrievable at the `pre-overhaul` git tag).

## Raw layout

```
data/LADH/
|-- 12_05/                        date directory (MM_DD)
|   |-- p_12_05_caip/             participant-on-date directory
|   |   |-- v01/                  session/scenario directory (v01..v05)
|   |   |   |-- BVP.csv
|   |   |   |-- HR.csv
|   |   |   |-- RR.csv
|   |   |   |-- SpO2.csv
|   |   |   |-- frames_timestamp_IR.csv
|   |   |   |-- frames_timestamp_RGB.csv
|   |   |   |-- video_RGB_H264.avi
|   |   |   |-- video_IR_H264.avi
|   |   |-- v02/ .. v05/
|   |-- p_12_05_huangxj/ ...
|-- 12_06/
|   |-- p_12_06_caip/ ...
|...
```

The legacy loader's docstring says `DATA_PATH` should be
`"On-Road-rPPG/*"` — i.e. the configured path **embeds a glob wildcard**
for the date level, because discovery was
`glob.glob(data_path + os.sep + 'p_*')`. It then `os.listdir`-ed each
`p_*` directory for session dirs (`v01`..`v05`) and listed each session
dir, keeping every filename containing `"H264.avi"` (both the RGB and the
IR video). Per video it recorded `index = dir[1:]` (e.g. `"01"`),
`subject = "p_12_05_caip"` (the whole directory name, **date included**),
and `type = '_'.join(item.split('_')[-2:]).split('.')[0]` (`"RGB_H264"` or
`"IR_H264"`).

## Video

Both videos are H.264-encoded `.avi` containers read with
`cv2.VideoCapture`: a `VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)` seek to zero,
then a sequential read loop with per-frame
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. Result: `(T, H, W, 3)` uint8,
0–255. Resolution and nominal fps are not recorded in the loader; the
in-repo LADH configs declare `FS: 30`, and actual per-frame timing lives in
the timestamp CSVs (`frames_timestamp_RGB.csv` / `frames_timestamp_IR.csv`,
one row per frame, column `timestamp`).

The IR video decodes to 3 channels through this reader like any other AVI.
The legacy loader read IR files but **never used them** — only paths
containing `"RGB_H264"` were preprocessed and saved, and
`frames_timestamp_IR.csv` was never opened — so nothing in the code
verifies that the three decoded IR channels are identical replicas.

## Physiological traces

**`bvp`** — `BVP.csv`, read with `pd.read_csv`; columns `timestamp` and
`bvp` (physical PPG units as recorded; the native sampling rate is not
recorded in the loader). Alignment to frames was timestamp-based:

```python
interp1d(bvp_timestamps, bvp_values,
         bounds_error=False, fill_value="extrapolate")(frame_timestamps)
```

i.e. scipy linear interpolation evaluated at the video's per-frame
timestamps, with **linear extrapolation** beyond both ends of the BVP
record. `frame_timestamps` came from `frames_timestamp_RGB.csv` (column
`timestamp`); the units of both timestamp columns are not stated in the
code — the loader only relies on the two files sharing one clock.

**`hr`, `rr`, `spo2`** — `HR.csv`, `RR.csv`, `SpO2.csv` exist in every
session directory but were **never read by the legacy loader**; their
column layout is therefore unknown from the code and must be verified
against the raw files (presumably `timestamp` + value, like `BVP.csv`).
Proposed: align each to frame timestamps with the same
linear-interpolate-and-extrapolate mechanism and store under the lowercase
keys `hr`, `rr`, `spo2`. Flagged as an extension beyond loader fidelity:
the cache adds traces the legacy pipeline discarded.

## Identity and attributes

The loader's recording unit was one session directory; its saved filename
was `f"{subject_id}_{experiment_id}"`, e.g. `p_12_05_caip_v01`, where
`subject_id` is the `p_*` directory name and `experiment_id` the session
directory name. Naming convention: `p_{MM}_{DD}_{name}` — a `p_` prefix,
the recording date, and a short participant name token (`caip`, `huangxj`,
`liutj`, `lujg`, ...). Session codes are `v01`..`v05`; the loader never
interpreted what each scenario is.

Important for LOSO: the legacy loader's split key was the **whole**
`p_12_05_caip` string, so the same person recorded on two dates counted as
two distinct "subjects" and could leak across percentage splits. Proposed
root attrs separate the axes:

- `recording`: `"p_12_05_caip_v01"` (the legacy filename);
- `participant`: the name token, e.g. `"caip"` — so LOSO folds hold a
  *person* out across all dates (design choice; filter on
  `participant` + `date` together to reproduce the legacy grouping);
- `date`: `"12_05"` (MM_DD; no year is present anywhere in the layout);
- `session`: `"v01"`.

## Proposed store mapping

One store per session directory: `p_12_05_caip_v01.zarr`.

RGB and IR are co-located cameras filming the same face — one view, two
modalities — so they are **streams under a single perspective** (design
choice; the raw data would also admit treating them as two perspectives,
but the Neckflix precedent keeps co-located modalities as streams):

```
p_12_05_caip_v01.zarr
  attrs: recording, participant, date, session, complete: true,
         tool_version (>= "1.0.0")
  1/
    rgb/
      video/frames      (3, T_rgb, H, W) uint8    attrs: num_frames, fps
      bvp/data          (T_rgb,) float64
      hr/data           (T_rgb,) float64
      rr/data           (T_rgb,) float64
      spo2/data         (T_rgb,) float64
    ir/
      video/frames      (1, T_ir, H, W) uint8     attrs: num_frames, fps
      bvp/data          (T_ir,) float64
      hr/data           (T_ir,) float64
      rr/data           (T_ir,) float64
      spo2/data         (T_ir,) float64
```

- `rgb` channel order is R, G, B (post BGR→RGB conversion).
- `ir` is stored single-channel by taking one decoded channel — a design
  choice matching the Neckflix `ir` convention; verify on real files that
  the three decoded channels are identical before collapsing (the legacy
  loader gives no evidence either way).
- Each stream's traces are interpolated onto **that stream's own**
  timestamp file (`frames_timestamp_RGB.csv` for `rgb`,
  `frames_timestamp_IR.csv` for `ir`), so traces stay index-aligned to
  their frames. Note the legacy loader only ever produced the RGB
  alignment.
- RGB and IR frame counts need not match; the zarr loader windows over the
  minimum `num_frames` across requested streams.
- Suggested `channel_map`:
  `{"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2), "I": ("ir", 0)}`.

## Quirks

- `DATA_PATH` must include the date-level glob wildcard
  (`.../LADH/*`) or discovery finds nothing — the loader globs `p_*`
  directly under the configured path.
- The IR videos were fully decoded during preprocessing and then thrown
  away (`if "RGB_H264" in video_file:` gated saving); the substring test
  runs on the **full path**, so a path containing `RGB_H264` elsewhere
  would misfire.
- `frames_timestamp_IR.csv`, `HR.csv`, `RR.csv` and `SpO2.csv` were never
  read; only `BVP.csv` + `frames_timestamp_RGB.csv` fed the labels.
- Interpolation extrapolates linearly beyond both ends of the BVP record
  (`fill_value="extrapolate"`), so frames outside the BVP time span get
  fabricated values rather than NaN. A cache-writer may prefer NaN outside
  the recorded span (the zarr loader NaN-pads and masks correctly), but
  that deviates from the legacy values — flagged as a design choice.
- The loader never checked that the decoded frame count equals the
  timestamp row count; it indexed both assuming 1:1. A cache-writer must
  verify and reconcile (the zarr contract requires `num_frames` to match
  the frames array).
- The same person appears under different `p_MM_DD_name` directories on
  different dates — see Identity for the LOSO implication.
- `self.info = config_data.INFO` was read but never used (dead config —
  no LIGHT/MOTION/etc. filtering ever happened despite the config
  surface). The `get_raw_data` docstring says "suitable for the THUSPO2
  dataset" — the loader is a copy-paste derivative; ignore the name.
- Legacy-pipeline detail: the recorded `index` (`dir[1:]`, e.g. `"01"`)
  never matched the saved filenames (`p_12_05_caip_v01_input0.npy`), so
  `build_file_list_retroactive` could not find any files for LADH.

## Not replicated

Face cropping, spatial resizing, chunking, `DiffNormalized`/`Standardized`
pixel and label normalisation, POS pseudo-labels
(`USE_PSUEDO_PPG_LABEL`), the `BEGIN`/`END` sorted-subject percentage
split, and the `.npy`/file-list cache are consumer-side or obsolete in the
new pipeline. The store carries raw full-length frames and physical-unit
traces only; splits become root-attr filters.
