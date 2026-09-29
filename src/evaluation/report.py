"""The evaluation report of a run: every table as a CSV, every figure as a
PNG, and the document that reads them, as Markdown and as one HTML file.

``Report`` collects the document while it writes the files beside it;
``build`` fills it, section by section, from the pooled measurements. The
Markdown holds headings, paragraphs, pipe tables and image links and nothing
else, so any renderer reads it. The HTML is the same document with the
figures embedded and an A4 print stylesheet.
"""

import base64
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from markdown_it import MarkdownIt

from src.evaluation import demographics
from src.evaluation.agreement import (
    ISO_SUBJECTS, baseline, criteria, limits, rate_metrics, repeated_measures,
    spread, within_shares,
)
from src.evaluation.plots import (
    agreement_figure, bland_altman_figure, box_figure, counts_figure,
    curves_figure, histogram_figure,
)
from src.evaluation.pooling import IDENTITY, paired
from src.evaluation.recording import WAVEFORM_METRICS
from src.outputs import FLOAT_FORMAT
from src.signal_transforms import signal_unit

TABLES, FIGURES = "tables", "figures"
MARKDOWN_NAME, HTML_NAME = "report.md", "report.html"
REPORT_DPI = 200
PRESSURE_UNIT = "mmHg"
#: One paired measurement: a recording and camera of a fold.
MEASUREMENT_KEY = [name for name in IDENTITY if name != "participant"]
#: What Markdown would read as markup in a table cell.
MARKDOWN_SPECIAL = re.compile(r"[\\`*_\[\]<>]")

DEPARTURES = (
    ("Segments matching the device output period (5.1.3)",
     "One measurement per recording, over its covered stretch"),
    ("Equal r for every subject (4.5.1 b)",
     "Unequal counts, with Formulas (5) and (9) to (12) in their general form"),
    ("N_ind from r (Formula 6)", "f_BA of Formula (10) in place of r"),
    ("One device reading per reference reading",
     "Each camera view is a reading against the same reference"),
    ("Method for stability (5.2)", "Not assessable on recordings of this length"),
    ("Method for blood pressure changes (5.3)",
     "Not assessable on recordings of this length"),
    ("Reference against time since re-initialization (5.1.3 f)", "Not applicable"),
    ("Scope", "The standard covers systolic, diastolic and mean arterial "
              "pressure only; every other parameter is outside it"),
)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<style>
@page { size: A4; margin: 20mm; }
body { font-family: system-ui, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1a1a; background: #ffffff; max-width: 170mm; margin: 0 auto;
       padding: 16px; }
h1 { font-size: 18pt; } h2 { font-size: 14pt; margin-top: 1.6em; }
h3 { font-size: 11.5pt; margin-top: 1.3em; }
h2, h3 { break-after: avoid; }
img { max-width: 100%; height: auto; display: block; margin: 0.6em 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 0.6em 0; }
th, td { border: 1px solid #c8c8c8; padding: 3px 5px; text-align: left;
         vertical-align: top; overflow-wrap: anywhere; }
th { background: #efefef; }
tr, img { break-inside: avoid; }
code { font-size: 9pt; overflow-wrap: anywhere; }
@media print { body { max-width: none; padding: 0; } }
</style>
</head>
<body>
{{body}}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def _cell(value) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3g}"
    text = str(value).replace("\n", " ")
    if not text.startswith("`"):          # a code span is literal already
        text = MARKDOWN_SPECIAL.sub(r"\\\g<0>", text)
    return text.replace("|", "\\|")


def code(text) -> str:
    """``text`` as a Markdown code span, whatever backticks it holds."""
    text = str(text)
    fence = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def pipe_table(frame: pd.DataFrame) -> list:
    """The lines of a Markdown pipe table."""
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |",
             "|" + "|".join(" --- " for _ in frame.columns) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return lines


class Report:
    """The document, and the files it links to, under one folder."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.lines = []
        for name in (TABLES, FIGURES):
            (self.folder / name).mkdir(parents=True, exist_ok=True)

    def heading(self, text: str, level: int = 2) -> None:
        self.lines += [f"{'#' * level} {text}", ""]

    def text(self, paragraph: str) -> None:
        self.lines += [paragraph, ""]

    def table(self, name: str, frame: pd.DataFrame) -> None:
        """Write ``tables/<name>.csv`` and print the table with a link to it."""
        path = f"{TABLES}/{name}.csv"
        frame.to_csv(self.folder / path, index=False, float_format=FLOAT_FORMAT)
        if frame.empty:
            self.text("Nothing to report.")
            return
        self.lines += [*pipe_table(frame), "", f"Data: [{path}]({path})", ""]

    def figure(self, name: str, figure: plt.Figure, caption: str) -> None:
        """Save ``figures/<name>.png``, close the figure and link it."""
        path = f"{FIGURES}/{name}.png"
        figure.savefig(self.folder / path, dpi=REPORT_DPI)
        plt.close(figure)
        self.lines += [f"![{caption}]({path})", "", f"*{caption}*", ""]

    def write(self, title: str) -> Path:
        markdown = "\n".join([f"# {title}", "", *self.lines])
        path = self.folder / MARKDOWN_NAME
        path.write_text(markdown, encoding="utf-8")
        (self.folder / HTML_NAME).write_text(
            to_html(markdown, self.folder, title), encoding="utf-8")
        return path


def to_html(markdown: str, folder: Path, title: str) -> str:
    """The document as one HTML file: figures embedded, A4 when printed."""
    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(markdown)

    def embed(match) -> str:
        data = base64.b64encode((folder / match.group(1)).read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(rf'src="({FIGURES}/[^"]+\.png)"', embed, body)
    return PAGE.replace("{{title}}", html.escape(title)).replace("{{body}}", body)


# ---------------------------------------------------------------------------
# The sections
# ---------------------------------------------------------------------------
def summary_section(report: Report, run_dir: Path, pooled: pd.DataFrame,
                    measurements: pd.DataFrame, excluded: pd.DataFrame, meta: dict,
                    bounds: tuple) -> None:
    report.heading("Summary")
    git = meta.get("git") or {}
    commit = f"{git.get('commit', '')}{' (dirty)' if git.get('dirty') else ''}"
    rows = [
        ("Run directory", code(run_dir)),
        ("Folds", pooled["fold"].nunique()),
        ("Epoch evaluated", ", ".join(str(e) for e in sorted(pooled["epoch"].unique()))),
        ("Datasets tested", ", ".join(sorted(pooled["dataset"].unique()))),
        ("Subjects", measurements["subject"].nunique()),
        ("Paired measurements (recording and camera)",
         len(measurements.drop_duplicates(MEASUREMENT_KEY))),
        ("Excluded", len(excluded)),
        ("Eligible covered stretch", f"{bounds[0]:g} to {bounds[1]:g} s"),
        ("Traces", ", ".join(meta["traces"])),
        ("Channels", ", ".join(meta["channels"])),
        ("Rate and window", f"{meta['fs']:g} Hz, {meta['window_frames']} frames, "
                            f"stride {meta['stride_frames']}"),
        ("Git commit", commit),
        ("Command of the first fold", code(meta.get("command", ""))),
    ]
    report.table("summary", pd.DataFrame(rows, columns=["item", "value"]).astype(str))


def demographics_section(report: Report, pooled: pd.DataFrame, params: list,
                         additional: list) -> None:
    report.heading("Demographics")
    report.text("Subject characteristics are counted once per subject, recording "
                "characteristics and reference values once per recording. 'All' "
                "is every test recording; 'eligible' those inside the covered "
                "stretch bounds. A characteristic the run directory does not "
                f"carry reads '{demographics.NOT_RECORDED}'.")
    report.heading("Population", 3)
    report.table("characteristics", demographics.characteristics(pooled, additional))
    report.heading("Reference values", 3)
    report.table("reference_spread", demographics.reference_spread(pooled, params))
    report.heading("ISO 81060-3 population requirements", 3)
    report.text("Clauses 4.2, 4.3.2 and 4.3.3, over the eligible recordings. The "
                "standard sets no requirement on skin tone, neck circumference, "
                "heart rate or venous pressure.")
    report.table("population_checks", demographics.population_checks(pooled))
    report.table("bp_distribution", demographics.bp_distribution(pooled, params))

    kept = pooled[pooled["eligible"]]
    report.heading("Distributions", 3)
    for p in params:
        values = demographics.references(kept, p)
        if len(values):
            bands = (", with the band edges of ISO 81060-3 clause 4.3.3"
                     if p.criteria else "")
            report.figure(f"reference_{p.key}",
                          histogram_figure(values, f"{p.label} ({p.unit})",
                                           f"Reference {p.label}",
                                           demographics.band_edges(p)),
                          f"Reference {p.label} over the eligible recordings{bands}")
    subjects = kept.drop_duplicates("subject")
    recordings = kept.drop_duplicates(demographics.RECORDING_KEY)
    counted = {demographics.CORE[name]: frame[name]
               for name, frame in (("posture", recordings), ("sex", subjects),
                                   ("skin_tone", subjects))
               if frame[name].notna().any()}
    if counted:
        report.figure("counts", counts_figure(counted, "Eligible test population"),
                      "Counts per posture (recordings), sex and skin tone (subjects)")
    ages = pd.to_numeric(subjects["age_years"], errors="coerce").dropna()
    if len(ages):
        report.figure("age", histogram_figure(ages, "age (years)", "Age", (50, 60, 70)),
                      "Age of the eligible subjects, with the thresholds of ISO "
                      "81060-3 clause 4.3.2.3.2")

    report.heading("Accuracy by group", 3)
    report.text(f"Mean error and SD within each group. Groups of fewer than "
                f"{demographics.SMALL_GROUP} measurements are marked small and "
                f"are too few to read.")
    report.table("stratified", demographics.stratified(kept, params))


def clinical_section(report: Report, measurements: pd.DataFrame, params: list) -> None:
    report.heading("Clinical accuracy")
    report.text("Error is prediction minus reference. s_corr is the standard "
                "deviation corrected for repeated measurements of a subject "
                "(ISO 81060-3 Formula 9); the limits of agreement are the mean "
                "error ± 1.96 s_corr.")
    found = {}
    for p in params:
        pairs = paired(measurements, p)
        if len(pairs):
            found[p.key] = (pairs, repeated_measures(pairs["error"], pairs["subject"]))
    agreement, sample, cumulative, passes = [], [], [], []
    for p in params:
        if p.key not in found:
            continue
        pairs, stats = found[p.key]
        low, high = limits(stats)
        agreement.append({"parameter": p.label, "unit": p.unit,
                          "mean_error": stats["mean_error"], "s_corr": stats["s_corr"],
                          "loa_low": low, "loa_high": high, "sd": stats["sd"],
                          **spread(pairs["error"])})
        sample.append({"parameter": p.label, "n": stats["n"], "subjects": stats["k"],
                       "per_subject_min": stats["m_min"],
                       "per_subject_median": stats["m_median"],
                       "per_subject_max": stats["m_max"], "f_BA": stats["f_ba"],
                       "ICC": stats["icc"], "N_ind": stats["n_ind"]})
        if p.unit == PRESSURE_UNIT:
            cumulative.append({"parameter": p.label, **within_shares(pairs["error"])})
        if p.criteria:
            passes += [{"parameter": p.label, **row} for row in criteria(stats)]
    report.heading("Agreement", 3)
    report.table("agreement", pd.DataFrame(agreement))
    report.heading("Sample", 3)
    report.table("sample", pd.DataFrame(sample))
    report.heading("Cumulative accuracy", 3)
    report.text("Percent of absolute errors within 5, 10 and 15 mmHg.")
    report.table("cumulative_accuracy", pd.DataFrame(cumulative))
    report.heading("ISO 81060-3 accuracy criteria", 3)
    report.text(f"Clause 5.1.4, and the {ISO_SUBJECTS} subjects of clause 4.5.1. "
                f"The standard sets criteria for arterial pressure only. See "
                f"'Departures from ISO 81060-3' for how this evaluation differs "
                f"from the clinical investigation the standard describes.")
    report.table("criteria", pd.DataFrame(passes))

    for p in params:
        report.heading(p.label, 3)
        if p.key not in found:
            report.text("No paired values: the reference or the prediction is "
                        "blank in every eligible measurement.")
            continue
        pairs, stats = found[p.key]
        if stats["note"]:
            report.text(f"Note: {stats['note']}.")
        report.figure(f"bland_altman_{p.key}",
                      bland_altman_figure(pairs, stats, limits(stats), p.unit,
                                          f"{p.label}: Bland-Altman"),
                      f"{p.label}: error against the mean of reference and "
                      f"prediction, with the mean error and the limits of agreement")
        report.figure(f"agreement_{p.key}",
                      agreement_figure(pairs, p.unit,
                                       f"{p.label}: prediction against reference"),
                      f"{p.label}: prediction against reference, with the line of "
                      f"identity")


def _mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    sd = f" ± {values.std(ddof=1):.3g}" if len(values) > 1 else ""
    return f"{values.mean():.3g}{sd}"


def learning_section(report: Report, measurements: pd.DataFrame, params: list,
                     traces: list, curves: pd.DataFrame, losses: pd.DataFrame) -> None:
    report.heading("Deep-learning metrics")
    rows = measurements[measurements["signal"].isin(traces)]

    report.heading("Waveform fidelity", 3)
    report.text("Sample by sample over each measurement's covered stretch, as "
                "mean ± SD over the measurements. The concordance is at its best "
                "lag within half a second; the lag is positive when the "
                "prediction is delayed.")
    report.table("waveform", pd.DataFrame([
        {"trace": sig, "unit": signal_unit(sig), "n": len(group),
         **{metric: _mean_sd(group[f"waveform_{metric}"]) for metric in WAVEFORM_METRICS}}
        for sig, group in rows.groupby("signal", sort=False)]))
    if rows["waveform_ccc"].notna().any():
        report.figure("waveform_ccc",
                      box_figure(rows, "waveform_ccc", "concordance (CCC)",
                                 "Waveform concordance per trace"),
                      "Concordance of the predicted with the reference waveform, "
                      "one point per measurement")

    report.heading("Rate", 3)
    rates = []
    for p in params:
        if p.kind != "rate":
            continue
        pairs = paired(measurements, p)
        source = measurements[measurements["signal"] == p.signal]
        rates.append({"source": p.signal, "unit": p.unit, "n": len(pairs),
                      **rate_metrics(pairs["ref"], pairs["pred"]),
                      "snr_db": source["snr"].mean(), "macc": source["macc"].mean()})
    report.table("rate", pd.DataFrame(rates))

    report.heading("Beat detection", 3)
    report.text("Predicted beats matched to reference beats, summed over the "
                "measurements: recall is matched over reference, precision "
                "matched over predicted.")
    beats = []
    for sig, group in rows.dropna(subset=["n_ref_beats"]).groupby("signal", sort=False):
        ref, pred, matched = (group[c].sum() for c in ("n_ref_beats", "n_pred_beats",
                                                       "n_matched"))
        beats.append({"trace": sig, "reference_beats": int(ref),
                      "predicted_beats": int(pred), "matched": int(matched),
                      "precision": matched / pred if pred else np.nan,
                      "recall": matched / ref if ref else np.nan})
    report.table("beats", pd.DataFrame(beats))

    report.heading("Constant-predictor baseline", 3)
    report.text("The model beside a predictor that answers the mean reference "
                "for every measurement. That mean is of the test references, "
                "which favours the constant. The spread ratio is the SD of the "
                "predictions over the SD of the references: near 0 the model "
                "answers one level for everyone, near 1 it follows the range.")
    report.table("baseline", pd.DataFrame([
        {"parameter": p.label, "unit": p.unit, **baseline(pairs["ref"], pairs["pred"])}
        for p in params if p.kind == "level"
        for pairs in [paired(measurements, p)] if len(pairs)]))

    report.heading("Training behaviour", 3)
    totals = [c for c in losses.columns if c.endswith("/total")] if len(losses) else []
    if totals:
        long = losses.melt(id_vars=["fold", "epoch"], value_vars=totals,
                           var_name="signal", value_name="loss")
        long["signal"] = long["signal"].str.removesuffix("/total")
        report.figure("training_loss",
                      curves_figure(long, "loss", "training loss",
                                    "Training loss per trace"),
                      "Training loss per trace against the epoch; the band is ± 1 "
                      "SD over folds")
    tested = curves[curves["eligible"] & curves["signal"].isin(traces)]
    if tested["waveform_rmse"].notna().any():
        report.figure("test_rmse",
                      curves_figure(tested, "waveform_rmse", "waveform RMSE",
                                    "Test error per trace"),
                      "Waveform RMSE on the held-out recordings against the "
                      "epoch; the band is ± 1 SD over folds and recordings")
    report.text("The epoch evaluated is fixed before the test error is seen; "
                "these curves describe training and select nothing.")


def departures_section(report: Report) -> None:
    report.heading("Departures from ISO 81060-3")
    report.text("This evaluation borrows the accuracy statistics of ISO "
                "81060-3:2022. It is not the clinical investigation the standard "
                "describes, and differs from it as follows.")
    report.table("departures", pd.DataFrame(
        DEPARTURES, columns=["requirement", "this evaluation"]))


def exclusions_section(report: Report, excluded: pd.DataFrame) -> None:
    report.heading("Exclusions")
    if excluded.empty:
        report.text("No recording was excluded.")
        excluded.to_csv(report.folder / TABLES / "exclusions.csv", index=False)
        return
    report.table("exclusions", excluded)


def build(out_dir, run_dir, pooled: pd.DataFrame, measurements: pd.DataFrame,
          excluded: pd.DataFrame, params: list, meta: dict, curves: pd.DataFrame,
          losses: pd.DataFrame, additional: list, bounds: tuple) -> Path:
    """Write the report under ``out_dir``; returns ``report.md``."""
    report = Report(out_dir)
    traces = [str(sig) for sig in meta["traces"]]
    summary_section(report, Path(run_dir), pooled, measurements, excluded, meta, bounds)
    demographics_section(report, pooled, params, additional)
    clinical_section(report, measurements, params)
    learning_section(report, measurements, params, traces, curves, losses)
    departures_section(report)
    exclusions_section(report, excluded)
    return report.write(f"Evaluation of {Path(run_dir).name}")
