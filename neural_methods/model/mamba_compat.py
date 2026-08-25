"""Compatibility shim for mamba_ssm variants.

On the HPC we use the PhysMamba fork of mamba-ssm, whose Mamba block accepts
`bimamba=True` (bidirectional SSM). On macOS we install `mamba-ssm-macos`,
which exposes the vanilla Mamba API without `bimamba`. This helper passes
`bimamba` through when supported and falls back to unidirectional Mamba
(with a one-time warning) when it is not.
"""
import inspect
import warnings

from mamba_ssm import Mamba

_MAMBA_PARAMS = inspect.signature(Mamba.__init__).parameters
_SUPPORTS_BIMAMBA = 'bimamba' in _MAMBA_PARAMS
_warned = False


def make_mamba(*args, **kwargs):
    """Instantiate Mamba, dropping unsupported kwargs (e.g. `bimamba`).

    Any kwarg not accepted by the installed Mamba implementation is removed.
    Falling back from bimamba=True changes the block to unidirectional
    processing - results will differ from the HPC fork.
    """
    global _warned
    dropped = [k for k in kwargs if k not in _MAMBA_PARAMS]
    if dropped:
        if 'bimamba' in dropped and not _warned:
            warnings.warn(
                "Installed mamba_ssm does not support `bimamba`; "
                "falling back to unidirectional Mamba. Results will differ "
                "from the bidirectional (HPC fork) implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned = True
        kwargs = {k: v for k, v in kwargs.items() if k in _MAMBA_PARAMS}
    return Mamba(*args, **kwargs)
