# SUMS Cache Spec

SUMS ("Summit Vitals: Multi-Camera and Multi-Signal Biosensing at High
Altitudes") pairs a face camera and a finger camera per session with BVP,
HR and RR ground truth, recorded at high altitude. Citation: Ke Liu*,
Jiankai Tang* (*co-first authors), Zhang Jiang, Yuntao Wang, Xiaojing Liu,
Dong Li, Yuanchun Shi, "Summit Vitals: Multi-Camera and Multi-Signal
Biosensing at High Altitudes", IEEE UIC, 2024. Dataset repository:
<https://github.com/thuhci/SUMS/>. This spec describes how to write
`{recording}.zarr` stores for SUMS conforming to the cache contract in
docs/architecture.md; it replaces the deleted legacy loader
(`SUMSLoader.py`, retrievable at the `pre-overhaul` git tag).

## Raw layout

```
data/SUMS/
|-- 060200/                       subject directory (glob: 0602*)
|   |-- v01/                      session directory (v01..v04)
|   |   |-- BVP.csv
|   |   |-- frames_timestamp.csv
|   |   |-- HR.csv
|   |   |-- RR.csv
|   |   |-- video_ZIP_H264_face.avi
|   |   |-- video_ZIP_H264_finger.avi
|   |-- v02/ v03/ v04/
|-- 060201/
|   |-- v01/ v02/ ...
|...
|-- 0602mn/
|   |-- v01/ v02/ ...
```

Discovery was `glob.glob(data_path + os.sep + '0602*')` — subject
directory names share the literal prefix `0602` followed by a two-character
suffix (digits in `060200`, `060201`, ...; letters in `0602mn`). Each
subject directory was `os.listdir`-ed for session dirs, and every filename
containing `"avi"` was kept (both face and finger videos), recording
`index = dir[1:]` (e.g. `"01"`), `subject = int(dirname)`, and
`type = item.split('_')[-1].split('.')[0]` (`"face"` or `"finger"`).

## Video

Both videos are H.264 `.avi` containers (named `video_ZIP_H264_face.avi`
and `video_ZIP_H264_finger.avi`) read with `cv2.VideoCapture`: a
`VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)` seek to zero, then a sequential
read loop with per-frame `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.
Result per video: `(T, H, W, 3)` uint8, 0–255. Resolution and nominal fps
are not recorded in the loader; the only in-repo config that declares SUMS
sets `FS: 60` (contrast LADH's 30 in the same file). Per-frame timing
lives in `frames_timestamp.csv` — a **single** timestamp file per session
(column `timestamp`, one row per frame).

## Physiological traces

**`bvp`** — `BVP.csv`, read with `pd.read_csv`; columns `timestamp` and
`bvp` (physical PPG units as recorded; native sampling rate not recorded
in the loader). Alignment to frames was timestamp-based:

```python
interp1d(bvp_timestamps, bvp_values,
         bounds_error=False, fill_value="extrapolate")(frame_timestamps)
```

scipy linear interpolation evaluated at the frame timestamps from
`frames_timestamp.csv`, with **linear extrapolation** beyond both ends of
the BVP record. Timestamp units are not stated in the code; the loader
relies only on the two files sharing one clock. The loader aligned BVP to
these timestamps and paired the result with the **face** video only.
Whether `frames_timestamp.csv` timestamps the face video, the finger
video, or both (synchronized capture) is not determinable from the code —
it is the only timing source available, so this spec uses it for both.

**`hr`, `rr`** — `HR.csv` and `RR.csv` exist in every session directory
but were **never read by the legacy loader**; their column layout is
unknown from the code and must be verified against raw files (presumably
`timestamp` + value, like `BVP.csv`). Proposed: align each to frame
timestamps with the same mechanism and store under lowercase keys `hr`,
`rr`. Flagged as an extension beyond loader fidelity. A code comment
mentions "SpO2 signals", but no SpO2 file appears in the layout and none
is read — there is no `spo2` trace.

## Identity and attributes

The loader's recording unit was one session directory; its saved filename
was `f"{subject_id}_{experiment_id}"`, e.g. `060200_v01`, where
`subject_id` is the subject directory name **as a string** (leading zeros
kept) and `experiment_id` is `v01`..`v04`. For split grouping, however,
the loader cast `subject = int(dirname)` — which drops leading zeros
(`"060200"` → `60200`) and **crashes with ValueError on the documented
`0602mn` directory**. Either that directory does not exist in real copies
of the dataset or the loader was never run against it; a cache-writer must
keep subject ids as strings.

Proposed root attrs:

- `recording`: `"060200_v01"` (the legacy filename);
- `participant`: `"060200"` — the full directory name as a string,
  leading zeros preserved (design choice: the literal `0602` prefix is
  common to all subjects but is retained so ids match the raw layout);
- `session`: `"v01"` (codes `v01`..`v04`; the loader never interpreted
  what each session is).

## Proposed store mapping

One store per session directory: `060200_v01.zarr`.

Face and finger are **distinct camera views of different body sites**, so
they map onto **two perspectives**, each with a single `rgb` stream —
following the rule that distinct views are perspectives and matching the
Neckflix treatment of perspectives as independent samples. Design choice,
explicitly flagged: the raw data equally admits one perspective with
streams `face` and `finger`, which is what a joint face+finger fusion
model (the SUMS paper's own setting) would need, since the zarr loader
draws each sample from a single perspective. This spec proposes the
two-perspective reading; revisit before writing the cache if fusion
experiments are planned.

```
060200_v01.zarr
  attrs: recording, participant, session, complete: true,
         tool_version (>= "1.0.0")
  1/                              face camera
    rgb/
      video/frames      (3, T_face, H, W) uint8    attrs: num_frames, fps
      bvp/data          (T_face,) float64
      hr/data           (T_face,) float64
      rr/data           (T_face,) float64
  2/                              finger camera
    rgb/
      video/frames      (3, T_finger, H, W) uint8  attrs: num_frames, fps
      bvp/data          (T_finger,) float64
      hr/data           (T_finger,) float64
      rr/data           (T_finger,) float64
```

- Channel order is R, G, B (post BGR→RGB conversion).
- Traces are duplicated under both perspectives, interpolated onto each
  video's frame timeline. With only one `frames_timestamp.csv`, both use
  the same timestamps; if the decoded finger frame count differs from the
  timestamp row count, the discrepancy must be resolved at write time (see
  Quirks).
- An attr distinguishing the views (e.g. `view: "face"` / `"finger"` on
  each perspective, or the perspective-number convention documented here)
  should be fixed at write time; the contract's filter surface offers the
  `perspective` pseudo-attribute (`"1"` = face, `"2"` = finger under this
  spec).
- Suggested `channel_map`:
  `{"R": ("rgb", 0), "G": ("rgb", 1), "B": ("rgb", 2)}` — the same map
  serves both perspectives.

## Quirks

- `int(subject_dirname)` crashes on `0602mn` (a directory the loader's own
  docstring documents) and silently collapses `"060200"` to `60200`
  elsewhere; keep participant ids as strings.
- The finger videos were fully decoded during preprocessing and then
  thrown away (`if "face" in video_file:` gated saving); the substring
  test runs on the **full path**, so any path containing `face` elsewhere
  would misfire.
- `HR.csv` and `RR.csv` were never read; only `BVP.csv` +
  `frames_timestamp.csv` fed the labels.
- One timestamp file serves both videos; nothing in the code establishes
  which camera it belongs to (see Physiological traces).
- Interpolation extrapolates linearly beyond both ends of the BVP record
  (`fill_value="extrapolate"`), fabricating values for frames outside the
  BVP time span rather than NaN. A cache-writer may prefer NaN outside the
  recorded span (the zarr loader NaN-pads and masks correctly), but that
  deviates from the legacy values — flagged as a design choice.
- The loader never checked that decoded frame count equals timestamp row
  count; it assumed 1:1. Verify at write time (the zarr contract requires
  `num_frames` to match the frames array).
- The only in-repo SUMS config declares `FS: 60`; nothing in the loader
  itself fixes the frame rate.
- `self.info = config_data.INFO` was read but never used (dead config).
  The `get_raw_data` docstring says "suitable for the THUSPO2 dataset" —
  the loader is a copy-paste derivative; ignore the name.
- Legacy-pipeline detail: the recorded `index` (`dir[1:]`, e.g. `"01"`)
  never matched the saved filenames (`060200_v01_input0.npy`), so
  `build_file_list_retroactive` could not find any files for SUMS.

## Not replicated

Face cropping, spatial resizing, chunking, `DiffNormalized`/`Standardized`
pixel and label normalisation, POS pseudo-labels
(`USE_PSUEDO_PPG_LABEL`), the `BEGIN`/`END` sorted-subject percentage
split, and the `.npy`/file-list cache are consumer-side or obsolete in the
new pipeline. The store carries raw full-length frames and physical-unit
traces only; splits become root-attr filters.
