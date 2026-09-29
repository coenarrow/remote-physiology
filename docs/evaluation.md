# Evaluation

`scripts/run.py` scores each epoch's `epoch_NN/test_records/<dataset>/`
the moment it has written them, and `scripts/test.py` does the same for a
dataset tested afterwards. The evaluation reads
`meta.json` and the per-recording trace tables and nothing else: no
config file, no checkpoint. It is not yet torch-free, though — a known
limitation, not the design's goal: no file under `src/evaluation/`
mentions torch, but the signal registry it needs
(`src/signal_transforms.py`) shares a module with torch-dependent label
normalisation, reached transitively through `src/outputs.py` ->
`src/model_config.py`. Importing `src.evaluation` still
pulls in torch and `src.model_config`, so a laptop with no
torch install cannot run it yet. Every `<recording>/<perspective>/`
folder under an epoch's `test_records/<dataset>/` is scored afresh, and its
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
trace: detrend and bandpass to 0.5 to 4 Hz, `find_peaks` with the
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
`waveform_ccc` is taken at the best lag within half a second, as the
training loss is, because the reference is measured at a different site
from the one the camera sees and the transit delay between them is not the
model's error; `waveform_lag` is that lag in seconds, positive when the
prediction is delayed against the reference.

`rates.csv` (`src/evaluation/rate.py`), one row per source: `ref_hr`,
`pred_hr`, `err_hr`, `snr`, `macc`. Each cardiac trace is cleaned
(smoothness-prior detrend, then a zero-phase first-order bandpass to
0.5 to 4 Hz, 30 to 240 bpm), its plain periodogram taken, zero-padded to
a power of two, and the largest in-band bin read as the rate, on the
label and on the prediction. `snr` is the power within 6 bpm of the
reference rate and its second harmonic over the rest of the band, in dB;
`macc` the maximum amplitude of cross-correlation over every lag. When
the recording carries two or more cardiac traces one source joins them:
`FUSED`, the rate of the weighted geometric mean of the traces' spectra,
each normalised to unit in-band power (so the frequency every trace
agrees on wins, and a peak only one trace has is suppressed). A trace's
weight is its auto-SNR, the same SNR centred on the spectrum's own peak
instead of the reference rate, in dB and clipped at zero, so a trace
whose power is split between the fundamental and its harmonics (CVP)
counts for little; the label side and the prediction side are each
weighted by their own spectra, and the weights are uniform when no trace
has a positive auto-SNR. SNR and MACC are on the combined trace and are
not comparable with the upstream toolbox's per-window numbers.

`<TRACE>.png` (`src/evaluation/plots.py`), one per trace: the whole
recording with the label, the combined prediction (the `mean` column)
inside a band of ± 1 SD across the overlapping windows (the `std`
column), and for cardiac traces every detected peak (up triangle) and
trough (down triangle) on both curves. Every predicted beat is marked,
matched or not, so a spurious predicted beat is visible here even though
it has no row in `<TRACE>_beats.csv`.

## Per run

`scripts/evaluate.py RUN_DIR` pools the tables above over every fold of a
run into `RUN_DIR/evaluation/`. It is run by hand after the run has
finished, reads the run directory and nothing else, and re-scores nothing.

**The paired measurement** (`src/evaluation/pooling.py`). One recording and
camera, over its whole covered stretch. It is eligible when that stretch is
within `--segment-min` and `--segment-max` seconds (20 and 30, both
inclusive); the rest are listed in `exclusions.csv` and enter no statistic.
The subject is the dataset and participant together. The epoch is each
fold's last, or `--epoch`; it is never chosen by test error.

`measurements.csv`, one row per fold, recording, camera and signal: `fold`,
`dataset`, `participant`, `recording`, `perspective`, `epoch`, `subject`,
`signal`, the columns of `signals.csv`, the rate columns of `rates.csv` for
that source (`FUSED` has a row of its own), `duration`, then the columns of
`recordings.csv` and the five core attributes.

**Demographics** (`src/evaluation/demographics.py`). `recordings.csv`,
written beside `meta.json` by `src/outputs.py`, carries each recording's
root attrs. Five are core: `age_years`, `sex`, `skin_tone`, `posture`,
`neck_circumference_cm`; several measurements of one are reported as their
median. Further attrs come from a dataset's `ADDITIONAL_PARAMS` and from
`--additional-params`. The sex and age shares are checked against ISO
81060-3 clauses 4.3.2.2 and 4.3.2.3.2 (p. 6), the reference pressures
against the bands of clause 4.3.3 (pp. 7-8). An attr the run does not carry
reads "not recorded".

**Accuracy** (`src/evaluation/agreement.py`). Per parameter (a rate per
source; systolic, diastolic and mean ABP; mean CVP): the mean error, the
corrected standard deviation, the intra-class correlation and the number of
independent measurements of ISO 81060-3 Formulas (5), (6) and (8) to (12)
(pp. 12 and 15), in the general form the standard prints for unequal counts
per subject, with `f_BA` standing for `r` in Formula (6). Limits of
agreement are the mean error ± 1.96 `s_corr`. For pressures, the percent of
absolute errors within 5, 10 and 15 mmHg. The criteria of clause 5.1.4
(p. 16) are applied to arterial pressure only.

**Deep-learning metrics.** Waveform MAD, RMSE, r, CCC and lag as mean ± SD
over measurements; rate MAE, RMSE, MAPE and r per source; beat precision
and recall; the model beside a constant predictor, and the ratio of the
predictions' SD to the references'; training loss and test error against
the epoch.

The report's "Departures from ISO 81060-3" section lists where this differs
from the clinical investigation the standard describes.

## Not covered

Cutting a recording into standard-length readings and scoring each: the
ISO 81060-2 invasive reference interval of 30 s (clause 6.2.4 b), p. 21),
the ISO 81060-3 device segment of 5 to 10 s (clause 5.1.3, p. 14, and
A.2, p. 27), the IEEE 1708 60 s recordings (clause 4.4.2, p. 24).
The methods for stability and for blood pressure changes of ISO 81060-3
(clauses 5.2 and 5.3), which need hours and half-hours of recording. The
criteria of IEEE 1708. Comparison across runs. A respiratory rate. ESH
2023 and ISO 81060-1. Calibration and time-since-initialisation. CVP beats are
detected with the shared detector and are expected to be unreliable;
`signals.csv`'s `n_ref_beats`, `n_pred_beats` and `n_matched` say
whether they are.
