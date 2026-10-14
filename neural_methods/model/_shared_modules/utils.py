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
* :class:`AttentionMask` — the soft-attention normalisation of the CAN
  family. Used by DeepPhys, TS-CAN and EfficientPhys.
* :class:`TSM` — the temporal shift, adaptive within a clip. Used by TS-CAN,
  EfficientPhys and (with ``wrap=True``, the published WTSM) BigSmall.
* :data:`EPS` — the floor under every mask-area and norm division of
  CardioConv and CardioHydra.
* :func:`collapse_harmonics`, :func:`phase_gradient`,
  :func:`kinematic_features` and :func:`pooled_speed` — the wave kinematics
  of a wavelet field (:class:`Kinematics`) and the two ways a model reads
  them: per frame under its masks, and as a speed pooled per window. Used by
  CardioConv and CardioHydra.
"""

import torch
from einops import einsum, rearrange, reduce

EPS = 1e-6

#: The per-pixel features :func:`kinematic_features` gives per harmonic, in order.
KINEMATIC_FEATURES = ("log_amplitude", "wavenumber", "log_speed", "direction_x", "direction_y")


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


def collapse_harmonics(field: torch.Tensor, n_harmonics: int, sharpness: float) -> torch.Tensor:
    """``(B, C, T, h, w, K)`` complex, harmonic-major -> ``(B, C, T, h, w, N)``
    complex: each harmonic's scales blended into one field.

    A power-weighted complex mean, weights ``|W| ** sharpness`` normalised
    per harmonic. Per pixel ``|W|`` is a noisy estimate, so a soft blend
    avoids the jitter of picking the peak scale, and phase, nearly constant
    across the narrow buffer, is kept. The magnitudes are scaled to their
    per-harmonic maximum first, so the power cannot overflow."""
    grouped = rearrange(field, "b c t h w (n j) -> b c t h w n j", n=n_harmonics)
    magnitude = grouped.abs()
    magnitude = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = magnitude ** sharpness
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (grouped * weights).sum(dim=-1)


def phase_gradient(field: torch.Tensor, dim: int) -> torch.Tensor:
    """The derivative of a complex field's phase along ``dim``, in radians per
    step: ``Im(dW conj(W)) / |W|^2``, which needs no unwrapping. Central
    differences inside, one-sided at the two ends."""
    (difference,) = torch.gradient(field, dim=dim)
    return (difference * field.conj()).imag / (field.real ** 2 + field.imag ** 2 + 1e-9)


def kinematic_features(kinematics: dict) -> list:
    """The kinematics ``{name: (B, C, T, h, w, N)}`` as the per-pixel features
    a model pools, in :data:`KINEMATIC_FEATURES` order: ``log1p`` of the
    amplitude, the wavenumber, the signed ``log1p`` of the speed and the
    heading's two components. ``log1p`` puts the amplitude and the speed,
    which span decades, in one band; the speed keeps its sign."""
    speed = kinematics["phase_speed"]
    return [
        torch.log1p(kinematics["amplitude"]),
        kinematics["wavenumber"],
        speed.sign() * torch.log1p(speed.abs()),
        kinematics["direction_x"],
        kinematics["direction_y"],
    ]


def pooled_speed(speed: torch.Tensor, pulsatility: torch.Tensor, phase_masks: torch.Tensor,
                 present: torch.Tensor, supervise_masks: bool) -> torch.Tensor:
    """``(B, C, T, h, w, N)`` phase speed -> ``(B, C * S * N)``: ``log1p`` of its
    magnitude, averaged under each trace's mask per channel and harmonic,
    zero for an absent channel. Normalised by the mask's area, like the
    statistics head's pooled power, and detached from the masks with it."""
    weight = rearrange(pulsatility, "b c t h w -> b c t h w 1") * phase_masks
    if not supervise_masks:
        weight = weight.detach()
    area = reduce(weight, "b c t h w s -> b c s 1", "sum").clamp_min(EPS)
    pooled = einsum(torch.log1p(speed.abs()), weight, "b c t h w n, b c t h w s -> b c s n") / area
    pooled = pooled * rearrange(present.to(pooled.dtype), "b c -> b c 1 1")
    return rearrange(pooled, "b c s n -> b (c s n)")
