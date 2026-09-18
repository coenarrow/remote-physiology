"""Pieces more than one backbone needs, written once.

Nothing here is an architecture. Each entry is a fragment several of the
migrated models had a copy of, kept in one place so a fix reaches every
consumer:

* :func:`nearest_multiple` — the adaptive stages round a length to a whole
  number of tubes, patches or temporal strides. Used by PhysFormer, PhysMamba,
  PhysNet, RhythmFormer and iBVPNet.
* :func:`dense_width` — the flattened feature count entering the dense head of
  the CAN family, derived per axis so a non-square frame works. Used by
  DeepPhys, TS-CAN and EfficientPhys.
* :func:`min_frame_message` / :func:`require_min_frame` — the one refusal the
  contract still allows, a frame a stem pools to nothing, in one wording. The
  seven pooling backbones raise it at forward time; ``_require_min_frame`` in
  ``src/models.py`` raises the same sentence from the builder when the
  interface resizes.
* :class:`Attention_mask` — the soft-attention normalisation of the CAN
  family. Used by DeepPhys, TS-CAN and EfficientPhys.
* :class:`TSM` — the temporal shift, adaptive within a clip. Used by TS-CAN,
  EfficientPhys and (with ``wrap=True``, the published WTSM) BigSmall.
"""


def nearest_multiple(n: int, k: int) -> int:
    """The positive multiple of ``k`` nearest to ``n``."""
    return max(round(n / k), 1) * k


def dense_width(height: int, width: int, filters: int) -> int:
    """Features entering the dense head of a CAN backbone, per axis.

    The two branches run a padded 3x3 convolution, a valid 3x3 convolution and
    a 2x2 average pool, twice; each axis therefore goes ``n -> (n - 2) // 2``
    twice, independently of the other. At a square frame this is the published
    ``filters * h * h``; at 66x90 it is ``64 * 15 * 21``.
    """
    def axis(n: int) -> int:
        return ((n - 2) // 2 - 2) // 2
    return filters * axis(height) * axis(width)


def min_frame_message(name: str, minimum: int) -> str:
    """The one refusal the contract allows, in the words both sides use."""
    return f"{name} pools frames down to nothing below {minimum}x{minimum}"


def require_min_frame(name: str, minimum: int, height: int, width: int) -> None:
    """Refuse a frame the stem leaves nothing of, at forward time."""
    if min(height, width) < minimum:
        raise ValueError(
            f"{min_frame_message(name, minimum)}; got {height}x{width}.")