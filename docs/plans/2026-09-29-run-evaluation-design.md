# Run evaluation: design

Date: 2026-09-29. Status: awaiting review.

## Purpose

`scripts/evaluate.py` takes a finished run directory and writes a pooled
evaluation of the model into it. It answers three questions:

1. **Who was tested?** The spread of the test population and of the
   reference values, against the population requirements of ISO 81060-3.
2. **Is it clinically accurate?** Agreement between prediction and
   reference for each clinical parameter, with the ISO 81060-3 accuracy
   statistics.
3. **How does it score as a deep-learning model?** The waveform and rate
   metrics that rPPG papers report, plus checks for collapse to a constant.

It is run by hand after training and scoring have finished. Nothing in
`scripts/run.py`, `scripts/test.py` or `main.py` calls it.

## Scope

Three pieces of work. Pieces 2 and 3 can be built and used before piece 1
lands.

| Piece | Touches | Depends on |
|---|---|---|
| 1. Core attribute standard | `docs/cache-contract.md`, the Neckflix cacher, the synthetic generator, a tool that patches attributes on existing stores | Nothing |
| 2. `recordings.csv` | `src/outputs.py` | Nothing |
| 3. Evaluation script | `scripts/evaluate.py`, new modules in `src/evaluation/` | 2; 1 for complete demographics |

Out of scope: IEEE 1708-2025, comparison across runs, a respiratory rate
estimator, PDF output, model size and speed, cutting recordings into
segments.

## Input

The reference layout is `runs/synthetic_benchmark_physmamba`:

```
<run>/
  <fold>/                         one per held-out participant
    config.yaml  losses.csv  log.txt  model.pt
    epoch_NN/test_records/<dataset>/
      meta.json  windows.csv  recordings.csv
      <recording>/<perspective>/
        signals.csv  rates.csv  <TRACE>_beats.csv  <TRACE>.csv  <TRACE>.png
```

The script reads the run directory and nothing else: no cache, no
checkpoint, no file under `configs/`. It computes nothing from waveforms;
it pools the per-recording tables that `docs/evaluation.md` describes.

## Decisions

| Topic | Decision |
|---|---|
| Paired measurement | One per (recording, perspective), over its whole covered stretch |
| Eligibility | Covered stretch (`t_end - t_start`) within `--segment-min 20` and `--segment-max 30` seconds, inclusive |
| Ineligible measurements | Left out of every pooled statistic and listed in `exclusions.csv` with the reason |
| Epoch | `--epoch N`; the default is the last epoch of each fold. Never chosen by test error |
| Subject | The pair (dataset, participant) |
| Camera views | Each perspective is its own measurement, grouped by subject. Reference distributions count each recording once |
| Parameters reported | Follow the traces in `meta.json`; a trace the run lacks has no section |
| Respiratory rate | Reported when `rates.csv` carries it. The estimator is separate, later work |
| Output | `report.md` and a self-contained `report.html` with an A4 print stylesheet |
| Output folder | `<run>/evaluation/`, overwritten on every run |
| Missing demographics | Shown as "not recorded", including when `recordings.csv` is absent |

## Piece 1: core attribute standard

Five root attributes carry the same name and values in every cache.

| Attribute | Type and values |
|---|---|
| `age_years` | Integer |
| `sex` | `M` or `F` |
| `skin_tone` | Integer 1 to 10, Monk scale |
| `posture` | `supine`, `recumbent` or `sitting` |
| `neck_circumference_cm` | Float |

Rules:

- An attribute a dataset cannot supply is null.
- Where a cache holds several measurements of one attribute, the standard
  value is their median. The individual measurements stay under their
  existing names (for example the three `skin_tone.*` ratings).
- `docs/cache-contract.md` lists the five. Each cacher writes them. A
  small tool patches them onto existing stores so nothing is re-cached.

Additional parameters come from two places and are treated alike:

| Source | Form |
|---|---|
| Dataset YAML | A list of attribute names in `configs/datasets/<name>.yaml`, empty for now. It reaches the run directory through the fold's `config.yaml` |
| Command line | `--additional-params name1 name2 ...`, dotted names of `recordings.csv` columns |

## Piece 2: `recordings.csv`

Written by `write_records` in `src/outputs.py`, beside `meta.json` and
`windows.csv`. One row per recording. Columns are `recording`,
`participant`, then every scalar root attribute, nested ones flattened
with dots. The writer selects nothing and knows nothing about
demographics.

Runs made before this change lack the file. Regenerating their test
records with `scripts/test.py` adds it.

## Piece 3: the evaluation script

### Command

```
uv run python scripts/evaluate.py <run-dir>
    [--epoch N]
    [--segment-min 20] [--segment-max 30]
    [--additional-params a b ...]
```

### Modules

| File | Purpose |
|---|---|
| `scripts/evaluate.py` | Parses arguments and calls the steps in order |
| `src/evaluation/pooling.py` | Finds folds, picks the epoch, stacks `signals.csv` and `rates.csv`, joins `recordings.csv`, applies eligibility |
| `src/evaluation/agreement.py` | Accuracy statistics as pure functions on arrays |
| `src/evaluation/demographics.py` | Descriptive tables, ISO distribution checks, stratified accuracy |
| `src/evaluation/plots.py` | Report figures, added beside `recording_figure` |
| `src/evaluation/report.py` | Assembles the Markdown and converts it to HTML |

### Steps

1. A fold is a subfolder holding `config.yaml`.
2. Take the requested epoch from each fold.
3. Stack every `signals.csv` and `rates.csv` under that epoch into one
   table, tagged with fold, dataset, participant, recording and
   perspective.
4. Join each row to its recording's row in `recordings.csv`.
5. Split by eligibility into `measurements.csv` and `exclusions.csv`.
6. Compute the tables from `measurements.csv`.
7. Draw the figures; write `report.md` and `report.html`.

Two further reads: `signals.csv` of every epoch and each fold's
`losses.csv`, for the curves in the deep-learning section.

### Output

```
<run>/evaluation/
  measurements.csv
  exclusions.csv
  tables/*.csv
  figures/*.png
  report.md
  report.html
```

Everything after step 5 depends on `measurements.csv` alone. A later
comparison across runs reads that file and `tables/`, never the report.

`measurements.csv` has one row per (fold, dataset, participant,
recording, perspective, signal). Its columns are the identity columns,
`duration`, the columns of `signals.csv`, the heart rate columns of
`rates.csv` for that trace, and the columns of `recordings.csv`. `FUSED`
heart rate rows carry the identity and rate columns only.

### Report sections

1. Summary
2. Demographics
3. Clinical accuracy
4. Deep-learning metrics
5. Departures from ISO 81060-3
6. Exclusions

The Markdown uses pipe tables, headings and image links only. Figures
are sized to the text width of A4 portrait.

## Section: demographics

Three tables, each given for all test recordings and for eligible ones.

| Level | Characteristics | Counted per |
|---|---|---|
| Subject | Age, sex, skin tone, neck circumference | Subject |
| Recording | Posture | Recording |
| Reference reading | Heart rate, systolic, diastolic and mean ABP, mean CVP | Recording |

Additional parameters are added at the level of the recording.

Checks against ISO 81060-3, each shown as a share and met or not met:

| Requirement | Clause |
|---|---|
| At least 30 % male and at least 30 % female | 4.3.2.2 |
| Age: 40 % at least 50 y, 25 % at least 60 y, 10 % at least 70 y | 4.3.2.3.2 |
| Reference BP distribution, below | 4.3.3 |
| Different postures evaluated | 4.2 b) |

Reference BP distribution, in mmHg, as shares of reference readings:

| | ≥ 5 % | ≥ 20 % | ≥ 20 % | ≥ 20 % | ≥ 5 % |
|---|---|---|---|---|---|
| Systolic | ≤ 90 | ≤ 110 | > 110 and < 140 | ≥ 140 | ≥ 160 |
| Diastolic | ≤ 50 | ≤ 60 | > 60 and < 80 | ≥ 80 | ≥ 90 |
| Mean | ≤ 65 | ≤ 75 | > 75 and < 100 | ≥ 100 | ≥ 115 |

Skin tone, neck circumference, heart rate and CVP are described with no
criterion; the report says the standard sets none.

Figures: a histogram of each reference parameter (the ABP ones with the
band edges drawn), bar charts of posture, sex and skin tone, a histogram
of age.

Stratified accuracy: mean error and SD of each clinical parameter by
posture, skin tone and sex, as one table. Groups of fewer than five
measurements are flagged.

## Section: clinical accuracy

Parameters: heart rate per cardiac trace and `FUSED`; systolic, diastolic
and mean ABP; mean CVP; respiratory rate when present. Error is
prediction minus reference.

### Statistics

With `n` measurements, `k` subjects and `m_i` measurements of subject
`i`, from ISO 81060-3 clauses 4.5.2 and 5.1.3:

| Statistic | Formula |
|---|---|
| Mean error | (8) |
| Bland-Altman factor `f_BA = (n² − Σ m_i²) / ((k − 1) n)` | (10) |
| Between-subject mean square | (11), general form |
| Within-subject mean square | (12), general form |
| Corrected SD `s_corr` | (9) |
| Intra-class correlation | (5) |
| Independent measurements `N_ind = k (1 + (1 − ICC)(f_BA − 1))` | (6), with `f_BA` in place of `r` |

Formulas (5) and (9) to (12) are printed in the standard for unequal
`m_i`. Formula (6) is not; `f_BA` replaces `r` and equals it when every
subject contributes equally.

Also reported: `n`, `k`, the minimum, median and maximum of `m_i`, plain
SD, MAE, RMSE, 95 % limits of agreement (mean error ± 1.96 `s_corr`), and
the share of absolute errors within 5, 10 and 15 mmHg.

Edge cases, each reported with a note:

| Situation | Handling |
|---|---|
| Every subject has one measurement | `s_corr` is the plain SD; ICC is not estimable; `N_ind = n` |
| Between-subject mean square below within-subject | Between-subject component is zero; `s_corr` is the within-subject SD; ICC is 0 |
| One subject | As the first row |

### Criteria

Pass or fail for systolic, diastolic and mean ABP (clauses 4.5.1 and
5.1.4):

| Criterion | Threshold |
|---|---|
| Mean error | Within ±6.0 mmHg |
| `s_corr` | ≤ 10.0 mmHg |
| `N_ind` | ≥ 278 |
| Subjects | ≥ 30 |

Other parameters get the statistics with no criterion.

### Figures, per parameter

- Bland-Altman plot: error against the mean of reference and prediction,
  with the mean error and limits of agreement, points coloured by subject.
- Prediction against reference, with the identity line.

## Section: deep-learning metrics

| Group | Content | Source |
|---|---|---|
| Waveform fidelity | MAE, RMSE, Pearson r, CCC and lag per trace, as mean ± SD over measurements, with a box plot | `signals.csv` |
| Rate | Heart rate MAE, RMSE, MAPE and Pearson r per source; SNR and MACC | `rates.csv` |
| Beat detection | Precision and recall per trace | Beat counts in `signals.csv` |
| Training behaviour | Training loss per trace against epoch; test error against epoch over folds | `losses.csv`, every epoch's `signals.csv` |
| Constant-predictor baseline | Error of predicting the pooled reference mean for every measurement, beside the model's error, for systolic, diastolic and mean ABP and mean CVP | `measurements.csv` |
| Spread ratio | SD of predictions over SD of references, same parameters | `measurements.csv` |

The baseline uses the mean of the test references, which favours the
baseline slightly; the report says so.

## Section: departures from ISO 81060-3

A fixed table in every report.

| Requirement | What the evaluation does |
|---|---|
| Segments matching the device output period (5.1.3) | One measurement of 20 to 30 s per recording |
| Equal `r` for every subject (4.5.1 b) | Unequal counts, general form of the formulas |
| `N_ind` from `r` (Formula 6) | `f_BA` in place of `r` |
| One device reading per reference reading | Each camera view is a reading against the same reference |
| Stability method (5.2) | Not assessable on recordings of this length |
| BP change method (5.3) | Not assessable on recordings of this length |
| Reference against time since re-initialisation (5.1.3 f) | Not applicable |
| Scope | The standard covers systolic, diastolic and mean arterial pressure only |

## Errors

| Condition | Behaviour |
|---|---|
| No folds under the run directory | Error |
| A fold lacks the requested epoch | Error naming the fold |
| Folds disagree on traces or interface | Error naming the folds |
| No eligible measurements | `exclusions.csv` is written, then error |
| `recordings.csv` absent or an attribute missing | "Not recorded" in the report |
| A trace absent from the run | Its section is omitted |

## Dependencies

One addition through `uv add`: a pure-Python Markdown converter
(`markdown-it-py`). Figures use matplotlib and seaborn, already present.

## Verification

The command on `runs/synthetic_benchmark_physmamba`, added to
`README.md`. That run has four subjects with one recording each, so it
exercises the first edge case above. `docs/evaluation.md` gains a section
describing the pooled layer and loses the matching lines under "Not
covered". No tests are added.
