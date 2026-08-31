# BP4D+ Cache Spec

BP4D+ (BP4D-Spontaneous extended) is a multimodal spontaneous emotion corpus
from Binghamton University: high-resolution 2D/3D facial video of 140 subjects
performing 10 emotion-elicitation tasks (T1..T10), with synchronized
physiological recordings (continuous blood pressure waveform, systolic /
diastolic / mean BP, pulse rate, respiration waveform and rate, EDA), thermal
video, and facial action unit (AU) coding. See
https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html. Cite
Zhang et al., "BP4D-Spontaneous: A high resolution spontaneous 3D dynamic
facial expression database", Image and Vision Computing 32 (2014) 692-706,
and Zhang et al., FG 2013 (see loader docstring for full citations; the CVPR
2016 "Multimodal Spontaneous Emotion Corpus" paper covers the BP4D+
extension). This spec describes how to write `{recording}.zarr` stores for
BP4D+ conforming to the cache contract in docs/architecture.md; it replaces
the deleted legacy loader (`BP4DPlusLoader.py`, retrievable at the
`pre-overhaul` git tag).

The BigSmall variant of this dataset (AU subset, 3-fold splits) is specified
in [BP4DPlusBigSmall.md](BP4DPlusBigSmall.md); it shares this raw layout and
should share the same stores — see that spec for the AU trace additions.

## Raw layout

`DATA_PATH` pointed at the `RawData/` directory (shipped configs used
`/gscratch/ubicomp/xliu0/data3/mnt/Datasets/BP4DPlus/RawData`):

```
RawData/
|-- 2D+3D/
|   |-- F001.zip            one zip per subject; entries <top>/<task>/<frame>.jpg
|   |-- F002.zip
|   |-- ... M001.zip ...
|-- 2DFeatures/             F001_T1.mat ...      (never read by the loader)
|-- 3DFeatures/             F001_T1.mat ...      (never read by the loader)
|-- AUCoding/
|   |-- AU_OCC/             F001_T1.csv ...      (read by BigSmall variant)
|   |-- AU_INT/AU06/        F001_T1_AU06.csv ... (read by BigSmall variant)
|-- IRFeatures/             F001_T1.txt ...      (never read by the loader)
|-- Physiology/
|   |-- F001/
|   |   |-- T1/
|   |   |   |-- BP_mmHg.txt
|   |   |   |-- LA Mean BP_mmHg.txt
|   |   |   |-- LA Systolic BP_mmHg.txt
|   |   |   |-- BP Dia_mmHg.txt
|   |   |   |-- Pulse Rate_BPM.txt
|   |   |   |-- Resp_Volts.txt
|   |   |   |-- Respiration Rate_BPM.txt
|   |   |   |-- EDA_microsiemens.txt   (see ambiguity note below)
|   |   |-- T2/ ... T10/
|   |-- ... M058/ ...
|-- Thermal/                F001/T1.mv ...       (never read by the loader)
|-- BP4D+UserGuide_v0.2.pdf
```

Recording enumeration globbed the Physiology tree, not the zips:
`glob(DATA_PATH/Physiology/F*/T*)` plus `glob(DATA_PATH/Physiology/M*/T*)`.
Each hit is one recording (subject x task).

**Ambiguity — EDA filename**: the loader docstring's tree lists
`microsiemens.txt`, but the only code that actually reads EDA (the BigSmall
variant, `read_raw_phys_labels`) opens `EDA_microsiemens.txt`. Trust the
executed code; verify against the raw dataset before writing.

## Video

- Source: `2D+3D/{subject}.zip` (e.g. `F001.zip`), opened with Python
  `zipfile.ZipFile`. Entries are selected where the file extension is `.jpg`
  **and** the second `/`-separated path component equals the task id
  (`str(ele).split('/')[1] == trial`, with `trial` like `"T8"`), i.e. entry
  paths are `<top>/<task>/<frame>.jpg`. The loader never inspected the top
  component or the frame filename.
- **Frame order**: the loader iterated `zipfile.namelist()` in archive order
  and never sorted. A cache-writer must sort entries numerically by frame
  filename rather than trusting archive order (ambiguity: the legacy loader
  implicitly assumed the archive lists frames in temporal order).
- Decode: `cv2.imdecode(..., cv2.IMREAD_COLOR)` then
  `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` — store RGB channel order, uint8,
  values 0-255.
- Resolution: the loader decoded at native resolution, then immediately
  downsampled as a *processing-time optimisation* ("otherwise processing time
  becomes WAY TOO LONG"): `dim_w = min(2*config_preprocess.W, frame.shape[1])`,
  `dim_h = int(dim_w * frame.shape[0]/frame.shape[1])` (aspect-preserving),
  `cv2.INTER_AREA`. Note: `PREPROCESS.W` was never defined in `config.py`
  (only `PREPROCESS.RESIZE.W = 128`), so this line as committed would raise
  `AttributeError`; the evident intent was twice the model resize width. The
  cache must NOT replicate this: **store frames at native resolution**. The
  native resolution is not recorded anywhere in the code — read it from the
  jpgs (the user guide documents the 2D texture video resolution).
- fps: not read from any file. All shipped BP4D+ configs set `FS: 25`, and
  the loader hard-codes `fs=25` in its POS pseudo-label branch
  (`self.generate_pos_psuedo_labels(frames, fs=25)`), so 25 fps is the
  operative value. Write `fps: 25` (verify against the user guide).
- The loader raised `ValueError('EMPTY VIDEO', index)` if a zip contained no
  matching frames.

## Physiological traces

All eight signals live under `Physiology/{subject}/{task}/` as single-column
text files, one sample per line, no timestamps. The legacy plain loader read
**only** `BP_mmHg.txt` (used directly as the training label in place of a PPG
signal); the BigSmall variant read all eight — parse details below are the
union, and every file should go into the cache.

| Raw file | Signal | Units | Proposed key |
| --- | --- | --- | --- |
| `BP_mmHg.txt` | continuous blood-pressure waveform | mmHg | `abp` |
| `LA Systolic BP_mmHg.txt` | systolic BP (beat-updated series) | mmHg | `systolic_bp` |
| `BP Dia_mmHg.txt` | diastolic BP (beat-updated series) | mmHg | `diastolic_bp` |
| `LA Mean BP_mmHg.txt` | mean BP (beat-updated series) | mmHg | `mean_bp` |
| `Pulse Rate_BPM.txt` | pulse rate | BPM | `hr` |
| `Resp_Volts.txt` | respiration waveform | volts | `resp` |
| `Respiration Rate_BPM.txt` | respiration rate | breaths/min | `rr` |
| `EDA_microsiemens.txt` | electrodermal activity | microsiemens | `eda` |

Key names are a proposal (design choice): `abp` matches the contract's
lowercase examples and this repo's mission vocabulary; the scalar-BP keys
mirror the legacy dict names (`systolic_bp`, `diastolic_bp`, `mean_bp`).

**What BP_mmHg.txt actually is (ambiguity)**: a continuous noninvasive
arterial-pressure waveform in mmHg. The measurement site/device is not
recorded in the code; the `LA` prefix on the summary files reads as "left
arm". Confirm the acquisition device and site from
`BP4D+UserGuide_v0.2.pdf` before describing it as ABP in publications; the
`abp` key is proposed on the mmHg units and waveform character alone.

**Native sampling rate (ambiguity)**: not encoded anywhere in the loader —
it aligned purely by length ratio (below). The BP4D+ user guide documents
the physiology sampling rate (commonly cited as 1000 Hz); verify there and
record it in the store attrs if desired.

**Parse**: `pd.read_csv(path).to_numpy().flatten()` — note pandas' default
`header='infer'` consumed the **first line of each file as a column name**,
silently dropping one sample. The files (ambiguity — verify) carry no header
row, so a faithful-to-the-data writer should read every line
(`header=None`); a faithful-to-the-loader writer would drop line one. This
spec recommends `header=None` and flags the one-sample offset as a known
difference from the legacy pipeline.

**Alignment to frames** — the contract requires `(T,)` traces index-aligned
to the frames, so this resampling moves into the cache-writer. The plain
loader did, for its single label:

```python
target_length = frames.shape[0]
bvps = BaseLoader.resample_ppg(bvps, target_length)
# where resample_ppg is exactly:
np.interp(np.linspace(1, input_signal.shape[0], target_length),
          np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
          input_signal)
```

i.e. pure linear interpolation over sample index, no timestamps, no
anti-alias filtering (the BigSmall variant used a slightly different but
equally naive `np.interp` call — see its spec). Design choice, flagged: a
cache-writer decimating a ~1 kHz waveform to a 25 Hz frame timeline with
bare linear interpolation aliases; replicating the legacy interp is the
faithful option, low-pass filtering first is the better one. Whichever is
chosen must be recorded in the store's provenance attrs.

## Identity and attributes

- Recording id: `{subject}{task}` concatenated from the Physiology path
  components, e.g. `F008T8` (`trial_data[-2] + trial_data[-1]`).
- Subject id: first four characters, e.g. `F008` — sex letter (`F`/`M`) plus
  a 3-digit number. **Keep the sex prefix in the participant attr**: the
  numeric part alone collides between female and male subjects (F001 vs
  M001).
- Sex: `index[0]` (`F` or `M`).
- Task: `T1`..`T10` (loader comment: "trial number (1-10)").
- Subject-disjoint splitting grouped recordings by the 4-char subject code
  and sliced the sorted subject list by `BEGIN`/`END` fractions —
  obsolete; LOSO in the new pipeline filters on the `participant` root attr.

Root attrs: `recording` (e.g. `"F008T8"`), `participant` (e.g. `"F008"`),
`sex`, `task`, plus the mandatory `complete: true` and `tool_version`.

## Proposed store mapping

One store per subject x task, single camera perspective:

```
F008T8.zarr
  attrs: complete: true, tool_version: ">=1.0.0", recording: "F008T8",
         participant: "F008", sex: "F", task: "T8"
  1/
    rgb/
      video/frames        (3, T, H, W) uint8, RGB, native resolution
      video/  attrs: num_frames (required), fps: 25
      abp/data            (T,) float64 mmHg, index-aligned to frames
      systolic_bp/data    (T,) float64 mmHg
      diastolic_bp/data   (T,) float64 mmHg
      mean_bp/data        (T,) float64 mmHg
      hr/data             (T,) float64 BPM
      resp/data           (T,) float64 volts
      rr/data             (T,) float64 breaths/min
      eda/data            (T,) float64 microsiemens
```

Thermal video (`Thermal/{subject}/{task}.mv`) exists in the raw data but was
never read by any loader; adding it as a second stream group is possible
later without breaking this schema. AU traces are specified in
[BP4DPlusBigSmall.md](BP4DPlusBigSmall.md) and belong in these same stores.

## Quirks

- **`F042T11` is excluded**: the loader skipped it with the comment "this
  filename exists but the file does not... weird..." — a Physiology
  directory exists for a task outside T1..T10 with no matching data. Do not
  write a store for it.
- Loader docstring: "There are 5 videos in BP4D+ with length of less than
  180 frames." They still get stores; short recordings simply yield fewer
  (or zero) windows consumer-side.
- Frame/label length reconciliation was nothing more than the linear
  resample of the label to the frame count — there is no timestamp
  alignment anywhere.
- The zip is the unit of video storage (one per subject, all tasks inside);
  the Physiology tree is the unit of enumeration. A subject zip missing a
  task's frames surfaced as `ValueError('EMPTY VIDEO')`.
- `pd.read_csv` header consumption drops the first sample of every
  physiology file (see Physiological traces).
- Frame order relied on zip archive order (see Video).
- The loader's `adjust_data_dirs` (skip already-preprocessed trials) exists
  but its call site is commented out — irrelevant to the cache.
- A stray `open(video_file)` handle was leaked per zip read — harmless,
  not behavior.

## Not replicated

Face cropping (Haar/YOLO5Face backends), the decode-time 2x-resize
downsample, `DiffNormalized`/`Standardized` frame and label transforms,
180-frame chunking, `BEGIN`/`END` percentage splits, file-list CSVs, and the
POS-based pseudo-PPG labels (`generate_pos_psuedo_labels`, fs=25, 0.70-3 Hz
2nd-order Butterworth + Hilbert envelope normalisation) are all consumer-side
or derived and are deliberately absent from the cache. The store holds raw
full-length RGB frames and physical-unit traces only.
