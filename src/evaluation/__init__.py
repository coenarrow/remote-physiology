"""The evaluation of a run's records, with no config, interface file or
checkpoint — though torch still arrives transitively today.

One recording and camera at a time: ``recording.py`` draws each trace of
a folder (label, prediction, the beats ``beat_metrics.py`` finds) through
``plots.py``, the moment ``scripts/run.py`` has written the records, so a
running fold can be looked at. Nothing is scored there.

Over a whole run: ``scripts/evaluate.py``, by hand, after the run has
finished. ``pooling.py`` finds the folds' records, reads the trace tables
and cuts them into paired measurements of a fixed length;
``measurements.py`` scores each (waveform metrics, derived parameters);
``agreement.py`` pools the scores with the repeated-measures statistics;
``demographics.py`` says who each signal was measured on; ``plots.py``
draws the figures. Every slow loop reports through :func:`progress`.
"""

from tqdm import tqdm

#: Every bar is this wide, so the descriptions line up down the terminal.
BAR_WIDTH = 100


def progress(iterable, desc: str, total: int | None = None):
    """``iterable`` behind a progress bar of the nominal width, ``desc``
    saying what is being done and, where it is slow, why."""
    return tqdm(iterable, desc=desc, total=total, ncols=BAR_WIDTH, leave=True)
