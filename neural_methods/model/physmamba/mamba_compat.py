"""Compatibility layer over mamba_ssm variants.

PhysMamba/PhysHydra historically depended on the vendored tools/mamba fork,
whose Mamba block accepted `bimamba=True` (Vim-style bidirectional SSM with
shared in/out projections). That fork is gone: this repo now targets the
vanilla mamba_ssm API only - the newest official `mamba-ssm` on Linux/CUDA
and `mamba-ssm-macos` on macOS.

`make_mamba(..., bimamba=True)` returns a `BiMamba`
(:mod:`neural_methods.model.physmamba.bi_mamba`): two independent vanilla
Mamba blocks, one running on the time-reversed sequence, summed. This is a
composition-based replacement for the fork's bimamba, NOT parameter-compatible
with it (separate in/out projections per direction) - checkpoints trained with
the fork do not load. On CUDA each direction uses the fused fast path; on
macOS both directions pick up the parallel selective scan that
:mod:`neural_methods.model.physmamba.selective_scan` patches in on import.

Where `mamba_ssm` is not installed, `MambaRef`
(:mod:`neural_methods.model.physmamba.mamba_ref`) stands in: the same
selective-SSM block written in plain PyTorch on top of the parallel scan of
``selective_scan``, with the same parameter names and shapes, so a checkpoint
moves either way. It is numerically a Mamba block and trains, but it
materialises the hidden state, so it is a development and CI path rather than
a performance one.

Windows does get the real package: `mamba-ssm` publishes no Windows wheel and
its CUDA sources need three MSVC fixes, so `pyproject.toml` builds the patched
`vendor/mamba-ssm` tree there (see `tools/vendor_mamba_windows.py`). `MambaRef`
remains the fallback wherever that build is unavailable -- no CUDA toolkit, or
a platform nobody ships for.

The pieces, bottom up, each importing only the ones before it:
``selective_scan`` (the scan and the mamba_ssm patch), ``mamba_ref``,
``portable_mamba``, ``mamba_backend`` (the availability probe, the warn-once
fallback and the unidirectional ``mamba_block``), ``bi_mamba``, and this file,
the entry point the models call.
"""

from neural_methods.model.physmamba.bi_mamba import BiMamba
from neural_methods.model.physmamba.mamba_backend import mamba_block


def make_mamba(d_model, d_state=16, d_conv=4, expand=2, **kwargs):
    """Build a (bi)directional Mamba block from the installed vanilla package.

    bimamba=True -> BiMamba (two blocks, one time-reversed); otherwise a
    plain Mamba. Unknown kwargs are dropped for portability across
    mamba-ssm releases.
    """
    if kwargs.pop('bimamba', False):
        return BiMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
    return mamba_block(d_model, d_state, d_conv, expand, **kwargs)
