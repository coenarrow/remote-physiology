"""The evaluation of a run's records, with no config, interface file or
checkpoint — though torch still arrives transitively today.

One recording and camera at a time: ``recording.py`` (with
``beat_metrics.py`` and ``rate.py``) scores a folder's trace tables into
beats, per-signal metrics and heart rates, written beside them, and ``plots.py``
draws each trace's figure. ``scripts/run.py`` scores a run's records the
moment it has written them; ``docs/evaluation.md`` is the reference.

Over a whole run: ``pooling.py`` stacks those tables into one of paired
measurements, ``agreement.py`` and ``demographics.py`` compute from it, and
``report.py`` writes the tables, the figures (``plots.py``) and the report.
``scripts/evaluate.py`` runs them, by hand, after the run has finished.
"""
