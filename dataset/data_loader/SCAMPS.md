# SCAMPS Cache Spec

SCAMPS is a fully synthetic dataset of 72x72 avatar face videos with
ground-truth physiological waveforms, distributed as one MATLAB v7.3 `.mat`
file per clip (<https://github.com/danmcduff/scampsdataset>). Citation: McDuff,
Wander, Liu, Hill, Hernandez, Lester, Baltrusaitis, "SCAMPS: Synthetics for
Camera Measurement of Physiological Signals", NeurIPS 2022
(<https://arxiv.org/abs/2206.04197>). This spec describes how to write
`{recording}.zarr` stores for SCAMPS conforming to the cache contract in
docs/architecture.md; it replaces the deleted legacy loader
(`SCAMPSLoader.py`, retrievable at the `pre-overhaul` git tag).

## Raw layout

The loader globbed one flat directory, non-recursively:
`glob.glob(data_path + os.sep + "*.mat")`. The upstream distribution is split
into three such directories (pre-overhaul README):

```text
data/SCAMPS/Train/
   |-- P00001.mat
   |-- P00002.mat
   ...
data/SCAMPS/Val/
data/SCAMPS/Test/
```

`DATA_PATH` pointed at exactly one of the three folders per config block; the
loader never saw the split structure itself.

Ambiguity: the loader docstring shows seven-character names (`P000001.mat`)
while the pre-overhaul README tree shows six (`P00001.mat`). The loader never
parsed the name, so either works; verify the digit width against the actual
download rather than this spec.

## Video

- Reader: `mat73.loadmat(video_file)` — the files are MATLAB **v7.3**
  (HDF5-based); `scipy.io.loadmat` cannot open them.
- Field: `mat['Xsub']`, shape `(T, H, W, 3)` channel-last, RGB order (no
  channel conversion was applied anywhere in the pipeline), resolution
  **72x72** per the loader docstring.
- Dtype/range: float in `[0, 1]`. The loader converted with exactly
  `frames = (np.round(frames * 255)).astype(np.uint8)` — round-then-cast,
  not truncation. The cache writer must do the same to produce uint8 frames.
- fps: not stored anywhere the loader read. Every SCAMPS config in this repo
  pins `FS: 30`; write `fps: 30` but note it comes from config convention,
  not from the .mat.

The loader docstring also names `dXsub` ("raw/diffnormalized data") but the
code reads only `Xsub`; `dXsub` is a preprocessed variant and must not be
used as the frame source.

## Physiological traces

Signals present in the .mat per the loader docstring: `d_ppg` (pulse) and
`d_br` (respiration). The .mat files carry further synthetic signals not
named in this repo's code; consult the SCAMPS repository if you want them.

- **bvp** (zarr key `bvp`): field `mat['d_ppg']` read via
  `mat73.loadmat`, converted with `np.asarray(ppg)`. Units: arbitrary
  (synthetic PPG; the code asserts nothing about units). Alignment: used
  **as-is** — the loader performed *no* resampling for SCAMPS (there is no
  `resample_ppg` call in `SCAMPSLoader`), i.e. `d_ppg` is already
  one-sample-per-frame, same length `T` as `Xsub`. Write it unchanged.
- **resp** (optional, zarr key `resp`): `d_br` exists per the docstring but
  the legacy loader never read it. Writing it is an extension consistent
  with the multi-signal mission, not a replication of legacy behavior.

## Identity and attributes

- The loader's per-recording index was the raw **filename including the
  `.mat` extension**: `subject = os.path.split(data_dir)[-1]` (e.g.
  `"P000001.mat"`). Use the stem (`P000001`) as the recording id.
- Each .mat is an independent synthetic clip/avatar; the loader had no
  subject grouping at all (`split_raw_data` sliced the file list by
  fraction directly). Treat the file stem as the participant id — there is
  no finer identity in the data as the loader saw it.
- The Train/Val/Test folder a file came from is worth preserving as a root
  attr (`split`), since the folder is the upstream split mechanism. The
  loader never recorded it; this is a proposed addition, not legacy
  behavior.

## Proposed store mapping

```text
{stem}.zarr                       e.g. P000001.zarr
  attrs: complete: true, tool_version: ">= 1.0.0",
         recording: {stem}, participant: {stem}, split: Train|Val|Test
  1/
    rgb/
      video/frames                (3, T, 72, 72) uint8, RGB channel order
      video/  attrs: num_frames: T, fps: 30   (fps from config convention)
      bvp/data                    (T,) float64, as stored in d_ppg
      resp/data                   (T,) float64, optional, from d_br
```

## Quirks

- v7.3 .mat files: `mat73` (or h5py) is mandatory; `scipy.io.loadmat`
  raises on them.
- Pixel conversion is `np.round(frames * 255).astype(np.uint8)` exactly. A
  dead alternate path (`preprocess_dataset_backup`) skipped the conversion
  and fed float frames; it was not the active code path — replicate the
  round-and-cast.
- Docstring/code mismatch: the docstring advertises `dXsub`, the code reads
  `Xsub`. The code is authoritative.
- `get_raw_data`'s docstring says "(For COHFACE dataset)" — a copy-paste
  artifact with no behavioral meaning.
- No trace resampling: unlike COHFACE/MMPD, SCAMPS labels were used at
  their stored length; if a file's `d_ppg` length ever differed from `T`
  the legacy chunker would have silently misaligned them. The cache writer
  should assert `len(d_ppg) == T` and fail loudly otherwise.
- The loader appended `"_" + dataset_name` to `CACHED_PATH` and spliced the
  dataset name into `FILE_LIST_PATH` — .npy-cache bookkeeping with no zarr
  equivalent.

## Not replicated

Face cropping (SCAMPS configs ran the Haar/YOLO5Face crop on the 72x72
frames), 72x72 resize, `DiffNormalized`/`Standardized` pixel and label
transforms, `CHUNK_LENGTH` chunking, POS pseudo-labels
(`USE_PSUEDO_PPG_LABEL`), and `BEGIN`/`END` fractional splits with file-list
CSVs are all consumer-side or obsolete in the new pipeline. The cache stores
raw full-length frames and unmodified traces only.
