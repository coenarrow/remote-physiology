# MMPD Cache Spec

MMPD (Multi-domain Mobile Video Physiology Dataset) is a mobile-camera rPPG
dataset with controlled variation in lighting, motion, exercise state, and
Fitzpatrick skin type, distributed as one MATLAB `.mat` file per trial
(<https://github.com/McJackTang/MMPD_rPPG_dataset>). Citation: Jiankai Tang,
Kequan Chen, Yuntao Wang, Yuanchun Shi, Shwetak Patel, Daniel McDuff, Xin
Liu, "MMPD: Multi-Domain Mobile Video Physiology Dataset", IEEE EMBC 2023.
This spec describes how to write `{recording}.zarr` stores for MMPD
conforming to the cache contract in docs/architecture.md; it replaces the
deleted legacy loader (`MMPDLoader.py`, retrievable at the `pre-overhaul`
git tag). Error strings in the loader covered both "MMPD" and "Mini-MMPD";
the parse below applies to both.

## Raw layout

```text
mat_dataset/
 |-- subject1/
 |    |-- p1_0.mat
 |    |-- p1_1.mat
 |    |...
 |    |-- p1_19.mat        (pre-overhaul README shows trials 0..19)
 |-- subject2/
 |    |-- p2_0.mat
 |...
 |-- subjectn/
```

Discovery: `glob.glob(raw_data_path + os.sep + 'subject*')`; subject number
parsed as `int(os.path.split(data_dir)[-1][7:])` (characters after the
literal `subject` prefix). Files per subject came from `os.listdir(data_dir)`
with **no extension filter** — any stray non-.mat file in a subject folder
would have been swept in; a cache writer should filter to `p*_*.mat`. The
loader's per-file index was the token after the last `_`, before the
extension: `mat_dir.split('_')[-1].split('.')[0]` (the trial number, e.g.
`'0'` from `p1_0.mat`).

## Video

- Reader: `scipy.io.loadmat(mat_file)` (`sio.loadmat` — classic v5/v7 .mat,
  not v7.3/mat73).
- Field: `mat['video']`, shape `(T, H, W, 3)` channel-last. The loader
  applied **no channel-order conversion**; the toolbox consumed the stored
  order as RGB downstream, so write it as RGB unchanged.
- Dtype/range: float in `[0, 1]` — the loader converted with exactly
  `frames = (np.round(frames * 255)).astype(np.uint8)` (round-then-cast).
  The cache writer must do the same to produce uint8 frames.
- Resolution: not asserted anywhere in the loader or configs; take `(H, W)`
  from the array (the MMPD repo documents 320x240, but that figure is not
  from this repo's code).
- fps: not read from the .mat. Every MMPD config block in this repo pins
  `FS: 30`; write `fps: 30` noting it comes from config convention.

## Physiological traces

- **bvp** (zarr key `bvp`): field `mat['GT_ppg']`, flattened as
  `np.array(mat['GT_ppg']).T.reshape(-1)` (transpose then flatten — handles
  the row-vector shape loadmat produces). Units: not asserted in code
  (PPG waveform, arbitrary units).
- Frame alignment: the loader resampled to the frame count with
  `BaseLoader.resample_ppg(bvps, frames.shape[0])`, which is exactly:

  ```python
  np.interp(np.linspace(1, len(sig), target_length),
            np.linspace(1, len(sig), len(sig)),
            sig)
  ```

  i.e. linear interpolation onto `T` evenly spaced points spanning the whole
  trace (endpoints included). Replicate this to produce the index-aligned
  `(T,)` trace (`BaseLoader` is being deleted; the quote is the whole
  mechanism).

## Identity and attributes

**Identity.** Participant id for LOSO is the integer from the `subject{N}`
directory name. `split_raw_data` grouped strictly by that number (sorted
ascending) so subjects never straddled splits. The trial number comes from
the filename (`p{N}_{trial}.mat`).

Legacy bug worth knowing: the .npy cache filename was
`'subject' + str(subject) + f'_L{light}_MO{motion}_E{exercise}_S{skin_color}_GE{gender}_GL{glasser}_H{hair_cover}_MA{makeup}'`
— it did **not** include the trial number, so two trials of one subject with
an identical attribute octuple overwrote each other in the cache. The zarr
store name must include the trial to avoid replicating that collision.

**Attribute fields.** The .mat carries eight metadata fields the loader read
and mapped through `get_information`; the resulting integer codes were
embedded in filenames and matched against the `INFO` config lists
(`INFO.LIGHT`, `INFO.MOTION`, `INFO.EXERCISE`, `INFO.SKIN_COLOR`,
`INFO.GENDER`, `INFO.GLASSER`, `INFO.HAIR_COVER`, `INFO.MAKEUP`) at load
time. Exact translations (any other raw value raised `ValueError`):

| Field | Raw value in .mat | Code |
| --- | --- | --- |
| `light` | `'LED-low'` | 1 |
| | `'LED-high'` | 2 |
| | `'Incandescent'` | 3 |
| | `'Nature'` | 4 |
| `motion` | `'Stationary'` or `'Stationary (after exercise)'` | 1 |
| | `'Rotation'` | 2 |
| | `'Talking'` | 3 |
| | `'Walking'` or `'Watching Videos'` | 4 |
| `exercise` | `'True'` | 1 |
| | `'False'` | 2 |
| `skin_color` | numeric, read as `information[3][0][0]` | kept as-is; only 3, 4, 5, 6 accepted |
| `gender` | `'male'` | 1 |
| | `'female'` | 2 |
| `glasser` | `'True'` | 1 |
| | `'False'` | 2 |
| `hair_cover` | `'True'` | 1 |
| | `'False'` | 2 |
| `makeup` | `'True'` | 1 |
| | `'False'` | 2 |

Notes:

- `'Watching Videos'` is an erroneous label from older MMPD versions; the
  loader's comment says it "should be handled as 'Walking'", hence code 4.
- `skin_color` is stored as a nested numeric array (hence the `[0][0]`
  indexing) holding a Fitzpatrick scale type; config comments confirm
  "Fitzpatrick Scale Skin Types - 3, 4, 5, 6". Types 1 and 2 do not occur
  (the loader rejects them).
- The boolean-ish fields are the **strings** `'True'`/`'False'` in the .mat,
  not MATLAB logicals. `glasser` means "wears glasses".
- `light`/`gender` etc. comparisons in the loader were against exact strings
  (`'LED-low'`, lowercase `l`; config comments write "LED-Low" but the code
  string is authoritative).
- Filtering semantics to preserve: a recording was included only if **every**
  code was a member of the corresponding `INFO` list (the legacy check
  parsed the last character of each filename token, e.g. `int(info[1][-1])`
  — single-digit codes made that safe). The `config.py` defaults
  (`LIGHT: ['']`, `EXERCISE: [True]`, `SKIN_COLOR: [1]`, ...) match nothing;
  real configs always set numeric lists. In the zarr pipeline these become
  root-attr include filters.

## Proposed store mapping

```text
subject{N}_p{N}_{trial}.zarr      e.g. subject1_p1_0.zarr
  attrs: complete: true, tool_version: ">= 1.0.0",
         recording: "subject{N}_p{N}_{trial}", participant: "{N}",
         trial: {trial},
         light: {1..4}, motion: {1..4}, exercise: {1,2},
         skin_color: {3..6}, gender: {1,2}, glasser: {1,2},
         hair_cover: {1,2}, makeup: {1,2}
  1/
    rgb/
      video/frames                (3, T, H, W) uint8, RGB channel order
      video/  attrs: num_frames: T, fps: 30   (fps from config convention)
      bvp/data                    (T,) float64, linearly interpolated to T
```

Store the loader's integer codes (they are what every existing config and
result refers to); optionally also store the raw strings under separate
attrs (e.g. `light_raw`) for readability.

## Quirks

- `sio.loadmat` was wrapped in a bare `try/except` that printed the file
  path 20 times and then crashed anyway (`mat` undefined -> `NameError`).
  Corrupt files were not tolerated, only noisily reported; a cache writer
  should fail cleanly instead.
- Round-then-cast pixel conversion (`np.round(x * 255).astype(np.uint8)`),
  not truncation.
- `GT_ppg` needs the `.T.reshape(-1)` flatten; a naive `ravel()` of the
  loadmat output happens to agree for 1-D row vectors but the transpose is
  what the code did.
- Unknown attribute strings raise — do not default or skip silently; new
  raw values mean a dataset revision this spec has not seen.
- The legacy cache-name collision (missing trial number) described above.
- `USE_PSUEDO_PPG_LABEL` generated POS pseudo-labels *inside* `read_mat`,
  on the float `[0,1]` frames before uint8 conversion — a training-time
  option, not cached.

## Not replicated

Face cropping, `RESIZE`, `DiffNormalized`/`Standardized` pixel and label
transforms, `CHUNK_LENGTH` chunking, POS pseudo-labels, `BEGIN`/`END`
subject-fraction splits, and the filename-token `INFO` filtering are
consumer-side or obsolete in the new pipeline: the cache stores raw
full-length frames and the physical-unit trace only, and the eight
attribute codes become root attrs driving load-time include filters
instead.
