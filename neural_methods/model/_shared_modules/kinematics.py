"""Wavelet field -> per-pixel wave kinematics: amplitude, wavenumber, speed, heading."""

import torch
import torch.nn as nn

from neural_methods.model._shared_modules.utils import collapse_harmonics, phase_gradient

#: The axes of the ``(B, C, T, h, w, N)`` field.
TIME, ROWS, COLUMNS = 2, 3, 4


class Kinematics(nn.Module):
    """How the pulse travels across the skin, read off the field's phase.

    A pulse is a travelling wave, so at each harmonic the phase of the field
    falls along the direction of travel. Per pixel, frame and harmonic, the
    spatial phase gradient gives the wavevector ``k = -grad(phase)``, the
    temporal one the angular frequency ``omega``, and their ratio the phase
    speed ``omega / |k|``, in pixels per frame. Speed is the level cue: by
    the Bramwell-Hill relation a pulse travels faster in a vessel at higher
    pressure.

    Nothing here is learned and nothing is differentiated: the outputs are
    features, and the ratio's gradient, where ``|k|`` is near zero, would
    be noise.

    Args:
      n_harmonics: harmonic groups the scale axis folds into.
      sharpness: exponent on ``|W|`` in the per-harmonic collapse; 2 weights
        by power, larger approaches the peak scale, 0 is a plain mean.
    """

    def __init__(self, n_harmonics: int, sharpness: float = 2.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sharpness = sharpness

    @torch.no_grad()
    def forward(self, field: torch.Tensor) -> dict:
        """``(B, C, T, h, w, K)`` complex field -> ``{name: (B, C, T, h, w, N)}``
        real: ``amplitude``, ``wavenumber`` (radians per pixel),
        ``phase_speed`` (signed, pixels per frame) and the unit heading,
        ``direction_x`` and ``direction_y``."""
        harmonics = collapse_harmonics(field, self.n_harmonics, self.sharpness)
        k_x = -phase_gradient(harmonics, COLUMNS)
        k_y = -phase_gradient(harmonics, ROWS)
        wavenumber = torch.sqrt(k_x ** 2 + k_y ** 2).clamp_min(1e-9)
        return {
            "amplitude": harmonics.abs(),
            "wavenumber": wavenumber,
            "phase_speed": phase_gradient(harmonics, TIME) / wavenumber,
            "direction_x": k_x / wavenumber,
            "direction_y": k_y / wavenumber,
        }
