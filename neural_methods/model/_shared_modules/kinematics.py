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
      pixel: the size of the field's grid cell in input pixels (the wavelet
        downscale), so the wavenumber is in radians per input pixel and the
        speed in input pixels per frame whatever grid the field is on.
    """

    def __init__(self, n_harmonics: int, sharpness: float = 2.0, pixel: float = 1.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.sharpness = sharpness
        self.pixel = pixel

    @torch.no_grad()
    def forward(self, field: torch.Tensor) -> dict:
        """``(B, C, T, h, w, K)`` complex field -> ``{name: (B, C, T, h, w, N)}``
        real: ``amplitude``, ``wavenumber`` (radians per pixel),
        ``phase_speed`` (signed, pixels per frame) and the unit heading,
        ``direction_x`` and ``direction_y``; plus ``harmonics``, the blended
        complex field itself, for whoever wants its phase (the visualiser;
        the models read the named features only)."""
        harmonics = collapse_harmonics(field, self.n_harmonics, self.sharpness)    # (B, C, T, h, w, N)
        # One harmonic at a time, written into the outputs: the gradients'
        # temporaries are one harmonic's plane, not N of them.
        out = {name: harmonics.real.new_empty(harmonics.shape)
               for name in ("amplitude", "wavenumber", "phase_speed", "direction_x", "direction_y")}
        for n in range(self.n_harmonics):
            harmonic = harmonics[..., n]
            k_x = -phase_gradient(harmonic, COLUMNS) / self.pixel
            k_y = -phase_gradient(harmonic, ROWS) / self.pixel
            wavenumber = torch.sqrt(k_x ** 2 + k_y ** 2).clamp_min(1e-9)
            out["amplitude"][..., n] = harmonic.abs()
            out["wavenumber"][..., n] = wavenumber
            out["phase_speed"][..., n] = phase_gradient(harmonic, TIME) / wavenumber
            out["direction_x"][..., n] = k_x / wavenumber
            out["direction_y"][..., n] = k_y / wavenumber
        out["harmonics"] = harmonics
        return out
