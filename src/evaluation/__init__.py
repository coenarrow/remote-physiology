"""The evaluation of a run's records, with no config, interface file or
checkpoint — though torch still arrives transitively today.

One recording and camera at a time: ``recording.py`` (with
``beat_metrics.py`` and ``rate.py``) scores a folder's trace tables into
beats, per-signal metrics and heart rates, written beside them, and ``plots.py``
draws each trace's figure. ``scripts/run.py`` scores a run's records the
moment it has written them; ``docs/evaluation.md`` is the reference.
"""
