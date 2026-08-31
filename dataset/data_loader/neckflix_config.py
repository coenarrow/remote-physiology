"""Translate a yacs config block into the zarr loader's plain-dict config.

``BaseZarrDataset`` deliberately takes a plain dict, with no yacs coupling (see
the loader design spec). This module is the one place that knows how the
repo's YAML keys map onto it, so the entry points stay short and the mapping is
testable on its own.

It also owns the participant-id convention mismatch: the repo says ``P015`` on
the command line, while the store's ``participant`` root attr is the unprefixed
``"015"`` the preprocessor writes.
"""

import re

from neural_methods.signals import resolve_channels, resolve_traces

_PARTICIPANT_PREFIX = re.compile(r"^[Pp](?=\d)")


def normalise_participant(participant) -> str:
    """``'P015'`` / ``'015'`` / ``15`` -> ``'015'``, the store's own spelling.

    A bare integer is zero-padded to three digits because that is what the
    preprocessor writes; anything already non-numeric is passed through so an
    unusual id is filtered on verbatim rather than mangled.
    """
    text = str(participant).strip()
    text = _PARTICIPANT_PREFIX.sub("", text)
    return text.zfill(3) if text.isdigit() else text


def participant_filter(include=(), exclude=()) -> dict:
    """Build the ``participant`` filter spec, normalising every id."""
    return {
        "include": [normalise_participant(p) for p in include or ()],
        "exclude": [normalise_participant(p) for p in exclude or ()],
    }


def build_filters(config_data, *, include_participants=(), exclude_participants=()) -> dict:
    """Attribute include/exclude filters from a ``DATA`` block plus LOSO ids.

    ``NECKFLIX.FILTERS`` maps store root attrs (or the ``perspective``
    pseudo-attr) to include whitelists — whatever attrs the cache carries, no
    fixed key list. ``[]`` means "do not filter on this attribute". Values are
    left as configured — the loader ``str()``-coerces where it must
    (``perspective``). Participants stay a separate surface
    (``PARTICIPANTS`` / the LOSO arguments) because their ids are normalised;
    a ``participant`` key in ``FILTERS`` is refused rather than left to
    bypass that normalisation.
    """
    neckflix = config_data.PREPROCESS.NECKFLIX
    filters = {}
    for attribute, values in getattr(neckflix, "FILTERS", {}).items():
        if str(attribute) == "participant":
            raise ValueError(
                "Filter participants with NECKFLIX.PARTICIPANTS or the "
                "participant arguments, not FILTERS.participant — those "
                "paths normalise ids (P015 -> 015); this one would not."
            )
        values = list(values or [])
        if values:
            filters[str(attribute)] = {"include": values, "exclude": []}

    configured = list(getattr(neckflix, "PARTICIPANTS", []) or [])
    include = list(include_participants or ()) or configured
    exclude = list(exclude_participants or ())
    if include or exclude:
        filters["participant"] = participant_filter(include, exclude)
    return filters


def zarr_config(config_data, *, include_participants=(), exclude_participants=(),
                random_windows=None) -> dict:
    """Full plain-dict config for ``NeckflixDataset`` from one yacs ``DATA`` block.

    ``random_windows`` overrides ``NECKFLIX.RANDOM_CHUNK`` when given, which is
    how a validation split reuses the training block but iterates
    deterministically.
    """
    preprocess = config_data.PREPROCESS
    neckflix = preprocess.NECKFLIX
    window_size = int(preprocess.CHUNK_LENGTH)
    stride = int(getattr(preprocess, "CHUNK_STRIDE", 0) or 0)
    return {
        "cache_dir": config_data.CACHED_PATH,
        "channels": resolve_channels(config_data),
        "labels": resolve_traces(config_data),
        "window_size": window_size,
        "window_stride": stride or window_size,
        "random_windows": bool(neckflix.RANDOM_CHUNK if random_windows is None
                               else random_windows),
        "label_norm": neckflix.LABEL_NORM,
        "allow_missing": bool(neckflix.ALLOW_MISSING),
        "min_channels": int(neckflix.MIN_CHANNELS),
        "min_labels": int(neckflix.MIN_LABELS),
        "filters": build_filters(
            config_data,
            include_participants=include_participants,
            exclude_participants=exclude_participants,
        ),
    }


def frame_size(config_data):
    """``(H, W)`` the models should see, or ``None`` to keep the cache's own size."""
    resize = config_data.PREPROCESS.RESIZE
    height, width = int(resize.H), int(resize.W)
    return (height, width) if height > 0 and width > 0 else None
