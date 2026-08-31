# COHFACE Cache Spec

COHFACE is a webcam remote-heart-rate dataset from Idiap: 40 subjects, four
one-minute trials each, RGB video plus a contact blood-volume-pulse trace
(<https://www.idiap.ch/en/dataset/cohface>). Citation: Guillaume Heusch, Andre
Anjos, Sebastien Marcel, "A reproducible study on remote heart rate
measurement", arXiv 2016
(<http://publications.idiap.ch/index.php/publications/show/3688>).
This spec describes how to write `{recording}.zarr` stores for COHFACE
conforming to the cache contract in docs/architecture.md; it replaces the
deleted legacy loader (`COHFACELoader.py`, retrievable at the `pre-overhaul`
git tag).

Caveat on provenance: the COHFACE loader in this repo was **vestigial** — it
was not registered in `main.py`'s `LOADER_REGISTRY`, no COHFACE config
exists, and its `preprocess_dataset(self, data_dirs, config_preprocess)`
signature no longer matched the four-argument call
(`data_dirs, PREPROCESS, BEGIN, END`) that `BaseLoader.__init__` makes, so
running it would raise `TypeError`. It also never implemented
`split_raw_data`, so the retroactive file-list path would raise too. This
spec records the parse the code *describes*, i.e. its last working upstream
form.

## Raw layout

From the loader docstring (COHFACE is not in the pre-overhaul README's
dataset trees):

```text
RawData/
 |-- 1/                       one directory per subject
 |    |-- 0/                  trials, exactly 0..3
 |    |    |-- data.avi
 |    |    |-- data.hdf5
 |    |...
 |    |-- 3/
 |         |-- data.avi
 |         |-- data.hdf5
 |...
 |-- n/
```

Discovery: `glob.glob(data_path + os.sep + "*")` — every entry under the
root was treated as a subject directory (no name filter; stray files or
extra folders like protocol lists would be swept in), then trials were
**hardcoded** as `for i in range(4)`, with paths
`os.path.join(data_dir, str(i))/data.avi` and `.../data.hdf5`. A cache
writer should verify each of the four trial directories actually exists
rather than assume.

## Video

- Reader: `cv2.VideoCapture(video_file)` on `data.avi`; the loader first
  called `VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)`, then looped `VidObj.read()`
  to exhaustion.
- Per frame it applied `cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)` —
  so frames are stored/consumed in **RGB** order; the cache must write RGB.
- It also ran `frame[np.isnan(frame)] = 0` (a no-op on uint8 data; carried a
  `# TODO: maybe change into avg` comment). Nothing to replicate.
- Dtype: uint8 `(T, H, W, 3)` as decoded by OpenCV. The code asserts nothing
  about codec, resolution, or fps; take resolution from the decoded frames
  and fps from the container (`cv2.CAP_PROP_FPS`). Idiap documents 640x480
  at 20 fps, but that figure is from the dataset page, **not** this repo's
  code — verify against the files.

## Physiological traces

- **bvp** (zarr key `bvp`): read as
  `h5py.File(bvp_file, 'r')["pulse"][:]` from `data.hdf5` — the full
  `pulse` dataset, nothing else. Units: not asserted anywhere in the code
  (a contact BVP sensor trace; treat as arbitrary units). Native sampling
  rate: **not read by the loader** — Idiap documents 256 Hz and the hdf5
  also carries `respiration` and `time` datasets per the dataset page, but
  none of that appears in this repo's code; verify from the files.
- Frame alignment: the loader resampled the full pulse trace to the frame
  count with `BaseLoader.resample_ppg(bvps, frames.shape[0])`, which is
  exactly:

  ```python
  np.interp(np.linspace(1, len(sig), target_length),
            np.linspace(1, len(sig), len(sig)),
            sig)
  ```

  i.e. linear interpolation onto `T` evenly spaced points spanning the
  whole trace (endpoints included). The cache writer must replicate this to
  produce the index-aligned `(T,)` trace the contract requires
  (`BaseLoader` is being deleted; the quote above is the whole mechanism).
- Optional extension: a `resp` trace from the hdf5 `respiration` dataset,
  aligned the same way — not legacy behavior, flagged here because the
  repo's mission is multi-signal.

## Identity and attributes

- The loader's recording index was
  `int('{0}0{1}'.format(subject, i))` — subject dir name, a literal `'0'`,
  then trial number: subject `1` trial 2 -> `102`, subject 12 trial 3 ->
  `1203`. Unambiguous for 40 subjects x trials 0-3, but opaque; prefer an
  explicit id and keep the parts as attrs.
- Participant id: the subject directory name (`1` .. `n`) — the only
  identity in the layout; needed for LOSO.
- Trial number (0-3) should be a root attr. The COHFACE protocol split
  (studio vs natural lighting per trial) never appears in the loader; if
  wanted as a `light` attr it must come from the dataset documentation, not
  from this spec.

## Proposed store mapping

```text
{subject}_{trial}.zarr            e.g. 1_0.zarr
  attrs: complete: true, tool_version: ">= 1.0.0",
         recording: "{subject}_{trial}", participant: "{subject}",
         trial: {0..3}
  1/
    rgb/
      video/frames                (3, T, H, W) uint8, RGB channel order
      video/  attrs: num_frames: T, fps: <from cv2.CAP_PROP_FPS>
      bvp/data                    (T,) float64, linearly interpolated to T
      resp/data                   (T,) float64, optional extension
```

## Quirks

- The loader is dead code in the current tree (unregistered, stale
  signature, `split_raw_data` unimplemented) — see the caveat at the top.
  There is no COHFACE yaml anywhere in `configs/`, and `config.py`'s `FS`
  default is `0`, so no sampling rate was ever pinned in this repo; fps
  must come from the AVI container.
- Exactly four trials per subject are assumed (`range(4)`), and the root
  glob matches *everything* — both need defensive handling.
- `cv2.VideoCapture` silently yields zero frames on a missing/corrupt file
  (no exception); the legacy code would have produced an empty array. The
  cache writer should treat zero decoded frames as an error.
- BGR->RGB conversion happened per frame in the loader; writing raw
  OpenCV output without the conversion would flip R and B.
- The NaN-zeroing line and the `CAP_PROP_POS_MSEC` seek are no-ops to a
  cache writer; listed only so their absence is not mistaken for an
  omission.

## Not replicated

Face cropping, `RESIZE`, `DiffNormalized`/`Standardized` pixel and label
transforms, `CHUNK_LENGTH` chunking, POS pseudo-labels
(`USE_PSUEDO_PPG_LABEL`), and `BEGIN`/`END` fractional splits with file-list
CSVs are consumer-side or obsolete in the new pipeline. The cache stores raw
full-length RGB frames and the physical trace (interpolated to frame count)
only.
