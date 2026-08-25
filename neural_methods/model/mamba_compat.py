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


# ---------------------------------------------------------------------------
# Parallel selective scan
#
# mamba-ssm-macos has no CUDA kernels; its selective_scan_fn falls back to
# selective_scan_ref, a Python loop over the sequence (thousands of tiny
# kernel launches on MPS - it dominates runtime). The recurrence
# h_t = a_t * h_{t-1} + x_t is associative, so we compute it with a
# Hillis-Steele doubling scan: O(log L) vectorized steps instead of O(L)
# sequential ones. Validated against selective_scan_ref to ~1e-5 in both
# forward outputs and gradients.
# ---------------------------------------------------------------------------
import torch
import torch.nn.functional as _F
from einops import rearrange as _rearrange

try:
    import mamba_ssm.ops.selective_scan_interface as _ssi
except ImportError:  # pragma: no cover
    _ssi = None


def _scan_nograd(a, x):
    """h_t = a_t * h_{t-1} + x_t via Hillis-Steele doubling (no autograd graph).
    a, x: (..., L, N) -> h: (..., L, N)"""
    L = a.shape[-2]
    d = 1
    while d < L:
        x = torch.cat([x[..., :d, :], x[..., d:, :] + a[..., d:, :] * x[..., :-d, :]], dim=-2)
        a = torch.cat([a[..., :d, :], a[..., d:, :] * a[..., :-d, :]], dim=-2)
        d *= 2
    return x


class _PScan(torch.autograd.Function):
    """Linear-recurrence scan with analytical backward.

    Autograd through the doubling loop retains every level of intermediates
    (O(L log L) memory - OOMs at full video resolution). The gradient of a
    linear scan is itself a (reverse) linear scan, so backward runs the same
    parallel primitive without retaining doubling levels:
        G_t = g_t + a_{t+1} * G_{t+1}   (reverse scan)
        dL/dx_t = G_t
        dL/da_t = G_t * h_{t-1}
    """

    @staticmethod
    def forward(ctx, a, x):
        with torch.no_grad():
            h = _scan_nograd(a, x)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        a, h = ctx.saved_tensors
        with torch.no_grad():
            # coefficients a_{t+1}, padded at the end (padding lands unused)
            a_next = torch.cat([a[..., 1:, :], torch.ones_like(a[..., :1, :])], dim=-2)
            G = _scan_nograd(a_next.flip(-2), grad_h.flip(-2)).flip(-2)
            h_prev = torch.cat([torch.zeros_like(h[..., :1, :]), h[..., :-1, :]], dim=-2)
            grad_a = G * h_prev
        return grad_a, G


def _parallel_linear_scan(a, x):
    if a.requires_grad or x.requires_grad:
        return _PScan.apply(a, x)
    return _scan_nograd(a, x)


def selective_scan_parallel(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                            delta_softplus=False, return_last_state=False):
    """Drop-in replacement for selective_scan_ref (standard Mamba case:
    real A, batched 3D B and C). Falls back to the reference implementation
    for complex A or grouped B/C layouts."""
    if _ssi is None or A.is_complex() or B.dim() != 3 or C.dim() != 3:
        return _selective_scan_ref_orig(u, delta, A, B, C, D, z, delta_bias,
                                        delta_softplus, return_last_state)
    dtype_in = u.dtype
    u_f = u.float()
    delta_f = delta.float()
    if delta_bias is not None:
        delta_f = delta_f + delta_bias[..., None].float()
    if delta_softplus:
        delta_f = _F.softplus(delta_f)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta_f, A.float()))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta_f, B.float(), u_f)
    h = _parallel_linear_scan(deltaA, deltaB_u)          # (B, D, L, N)
    y = torch.einsum("bdln,bnl->bdl", h, C.float())
    out = y if D is None else y + u_f * _rearrange(D.float(), "d -> d 1")
    if z is not None:
        out = out * _F.silu(z.float())
    out = out.to(dtype=dtype_in)
    if return_last_state:
        return out, h[:, :, -1]
    return out


# Patch the scan only when the installed package has no CUDA kernels (i.e. it
# would use the slow reference loop anyway). On the HPC fork the fused CUDA
# path is untouched.
_selective_scan_ref_orig = _ssi.selective_scan_ref if _ssi is not None else None
# `has_cuda_support` only exists in mamba-ssm-macos; on the HPC fork (attr
# absent, fused CUDA kernels present) we leave everything untouched.
if _ssi is not None and getattr(_ssi, 'has_cuda_support', True) is False:
    _ssi.selective_scan_ref = selective_scan_parallel
