# Evaluation

`scripts/run.py` scores each epoch's `epoch_NN/test_records/` the moment
it has written them. The evaluation reads
`meta.json` and the per-recording trace tables and nothing else: no
config file, no checkpoint. It is not yet torch-free, though — a known
limitation, not the design's goal: no file under `src/evaluation/`
mentions torch, but the signal registry it needs
(`src/signal_transforms.py`) shares a module with torch-dependent label
normalisation, reached transitively through `src/outputs.py` ->
`src/model_config.py`. Importing `src.evaluation` still
pulls in torch and `src.model_config`, so a laptop with no
torch install cannot run it yet. Every `<recording>/<perspective>/`
folder under an epoch's `test_records/` is scored afresh, and its
tables are written beside its trace tables. Clause and page numbers
below are the printed ones in `standards/`.

## The prediction being scored

The trace table's `mean` column: the average of every strided window
covering the frame. Its `std` is the repeatability across windows and
`label` the reference in physical units. Averaging overlapping windows
smooths the prediction; the per-window columns stay in the trace tables
for anyone who wants the unsmoothed one. Errors are prediction minus
reference everywhere, the sign every standard uses.

## Per recording and camera

Written beside the trace tables: one `<TRACE>_beats.csv` per cardiac
trace, `signals.csv`, `rates.csv` and one `<TRACE>.png` per trace.
Nothing is cached; a run overwrites whatever an earlier one left. These
files carry no dataset, participant, recording or perspective columns: a
folder's place in the records directory says where it sits. Every metric
is over the whole covered stretch, from the first frame any window
covered to the last, scored once per recording.

**Beats** (`src/evaluation/beat_metrics.py`). One detector for every cardiac
trace: detrend and bandpass to 0.6 to 3.3 Hz, `find_peaks` with the
minimum beat distance from the top of the band and the prominence a
registry fraction of the cleaned range (`beat` in
`src/signal_transforms.py`), each candidate moved to the raw extremum
within a quarter of the median beat interval. A beat spans trough to
trough: `max` is its peak, `min` the lower trough, `mean` the area under
the curve over the duration, the MAP definition of ISO 81060-2 clause
6.2.4 e), p. 22. Predicted beats match the nearest reference beat within
40 percent of the median reference interval, closest first, one to one.

`<TRACE>_beats.csv`, one per cardiac trace: `beat`, `t_ref_peak`,
`t_ref_trough` (the beat's foot, the trough before its peak),
`t_pred_peak`, `t_pred_trough` (blank on a miss), `ref_max`, `ref_mean`,
`ref_min`, `pred_max`, `pred_mean`, `pred_min` (blank for shape-class
signals). One row per *reference* beat; a predicted beat with no
reference match has no row here at all, but is counted in
`signals.csv`'s `n_pred_beats`.

`signals.csv`, one row per signal over the covered stretch: `t_start`,
`t_end` (its bounds on the trace table's time axis); `n_ref_beats`,
`n_pred_beats`, `n_matched` (recall is matched over reference, precision
matched over predicted); for absolute signals `ref_<s>_mean`,
`ref_<s>_sd`, `pred_<s>_mean`, `pred_<s>_sd` over the beats for `s` in
`max`, `mean`, `min` (systolic, MAP, diastolic for ABP; peak, mean, trough
for CVP; for a non-cardiac absolute signal the sample mean, SD, max and
min), `err_<s>` and `err_<s>_deadband` (ISO 81060-2 clause 6.2.5, p. 22:
zero inside the reference mean ± SD, else the distance to the nearer
limit); `waveform_mad`, `waveform_rmse`, `waveform_r`, `waveform_ccc`
over the stretch's samples (IEEE 1708 equations (3) and (4), p. 28).

`rates.csv` (`src/evaluation/rate.py`), one row per source: `ref_hr`,
`pred_hr`, `err_hr`, `snr`, `macc`. Each cardiac trace is cleaned
(smoothness-prior detrend, then a zero-phase first-order bandpass to
0.6 to 3.3 Hz), its plain periodogram taken, zero-padded to a power of
two, and the largest in-band bin read as the rate, on the label and on
the prediction. `snr` is the power within 6 bpm of the reference rate
and its second harmonic over the rest of the band, in dB; `macc` the
maximum amplitude of cross-correlation over every lag. When the recording
carries two or more cardiac traces two sources join them: `FUSED`, the
rate of the geometric mean of the traces' spectra, each normalised to
unit in-band power (so the frequency every trace agrees on wins, and a
peak only one trace has is suppressed), and `MEDIAN`, the median of the
per-trace rates. SNR and MACC are on the combined trace and are not
comparable with the upstream toolbox's per-window numbers.

`<TRACE>.png` (`src/evaluation/plots.py`), one per trace: the whole
recording with the label, the combined prediction (the `mean` column)
inside a band of ± 1 SD across the overlapping windows (the `std`
column), and for cardiac traces every detected peak (up triangle) and
trough (down triangle) on both curves. Every predicted beat is marked,
matched or not, so a spurious predicted beat is visible here even though
it has no row in `<TRACE>_beats.csv`.

## Not covered

Cutting a recording into standard-length readings and scoring each: the
ISO 81060-2 invasive reference interval of 30 s (clause 6.2.4 b), p. 21),
the ISO 81060-3 device segment of 5 to 10 s (clause 5.1.3, p. 14, and
A.2, p. 27), the IEEE 1708 60 s recordings (clause 4.4.2, p. 24).
Pooling recordings into per-participant or per-dataset summaries, the
clinical standards' criteria and coverage tables, figures and the
report: to be rebuilt on top of the tables above. ESH 2023 and ISO
81060-1. Calibration and time-since-initialisation. CVP beats are
detected with the shared detector and are expected to be unreliable;
`signals.csv`'s `n_ref_beats`, `n_pred_beats` and `n_matched` say
whether they are.
