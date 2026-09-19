"""Which Mamba implementation is installed, and one unidirectional block of it.

The probing half of :mod:`neural_methods.model.physmamba.mamba_compat` (read
that docstring for the whole picture). It is a file of its own so that
:mod:`neural_methods.model.physmamba.bi_mamba`, which builds its two
directions with :func:`mamba_block`, and ``mamba_compat.make_mamba``, which
builds a ``BiMamba``, do not import each other.

Importing this file also imports
:mod:`neural_methods.model.physmamba.selective_scan`, which patches the
parallel scan into a mamba_ssm build that has no CUDA kernels.
"""

import inspect
import warnings

import neural_methods.model.physmamba.selective_scan  # noqa: F401 - the import applies the scan patch
from neural_methods.model.physmamba.mamba_ref import MambaRef

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:  # pragma: no cover - exercised only where the package is absent
    Mamba = None
    HAS_MAMBA_SSM = False

if HAS_MAMBA_SSM:
    from neural_methods.model.physmamba.portable_mamba import PortableMamba
else:  # pragma: no cover - exercised only where the package is absent
    PortableMamba = None

_MAMBA_PARAMS = inspect.signature(Mamba.__init__).parameters if HAS_MAMBA_SSM else {}


def _filter_kwargs(kwargs):
    """Drop kwargs the installed Mamba implementation does not accept."""
    if not HAS_MAMBA_SSM:
        return {}
    return {k: v for k, v in kwargs.items() if k in _MAMBA_PARAMS}


_warned_about_fallback = False


def _warn_once_about_fallback():
    global _warned_about_fallback
    if not _warned_about_fallback:
        _warned_about_fallback = True
        warnings.warn(
            "mamba_ssm is not installed; using the pure-PyTorch MambaRef fallback. "
            "It is numerically a Mamba block and trains, but it materialises the "
            "hidden state (roughly d_inner x tokens x d_state floats per block), so "
            "expect several GiB at video resolution and no fused-kernel speed. "
            "Install mamba-ssm (Linux/CUDA) or mamba-ssm-macos for the fast path.",
            RuntimeWarning, stacklevel=3)


def mamba_block(d_model, d_state, d_conv, expand, **kwargs):
    """One unidirectional block from whichever implementation is available."""
    if HAS_MAMBA_SSM:
        return PortableMamba(d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, **_filter_kwargs(kwargs))
    _warn_once_about_fallback()
    return MambaRef(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
