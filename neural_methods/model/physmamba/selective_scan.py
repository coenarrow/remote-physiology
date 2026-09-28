"""Parallel selective scan, and its patch into CUDA-less mamba_ssm builds.

mamba-ssm-macos has no CUDA kernels; its selective_scan_fn falls back to
selective_scan_ref, a Python loop over the sequence (thousands of tiny
kernel launches on MPS - it dominates runtime). The recurrence
h_t = a_t * h_{t-1} + x_t is associative, so we compute it with a
Hillis-Steele doubling scan: O(log L) vectorized steps instead of O(L)
sequential ones. Validated against selective_scan_ref to ~1e-5 in both
forward outputs and gradients.

This file sits below everything else in the package: it imports no sibling,
so :mod:`neural_methods.model.physmamba.mamba_ref` (which runs its recurrence
through :func:`parallel_linear_scan`) and
:mod:`neural_methods.model.physmamba.mamba_backend` (which imports it for the
patch at the bottom) can both depend on it without a cycle. Importing it is
what applies the patch. All reshaping is einops.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

try:
    import mamba_ssm.ops.selective_scan_interface as _ssi
except ImportError:  # pragma: no cover
    _ssi = None


def _scan_nograd(a, x):
    """h_t = a_t * h_{t-1} + x_t via Hillis-Steele doubling (no autograd graph).
    a, x: (..., L, N) -> h: (..., L, N)"""
    length = a.shape[-2]
    d = 1
    while d < length:
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
            grad_x = _scan_nograd(a_next.flip(-2), grad_h.flip(-2)).flip(-2)
            h_prev = torch.cat([torch.zeros_like(h[..., :1, :]), h[..., :-1, :]], dim=-2)
            grad_a = grad_x * h_prev
        return grad_a, grad_x


def parallel_linear_scan(a, x):
    """h_t = a_t * h_{t-1} + x_t over the second-to-last axis, differentiable."""
    if a.requires_grad or x.requires_grad:
        return _PScan.apply(a, x)
    return _scan_nograd(a, x)


# The argument names are mamba_ssm's own `selective_scan_ref` signature, which
# this function replaces in place, so they stay as that package spells them.
def selective_scan_parallel(u, delta, A, B, C, D=None, z=None, delta_bias=None,  # noqa: N803
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
        delta_f = F.softplus(delta_f)
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta_f, A.float()))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta_f, B.float(), u_f)
    h = parallel_linear_scan(delta_a, delta_b_u)          # (B, D, L, N)
    y = torch.einsum("bdln,bnl->bdl", h, C.float())
    out = y if D is None else y + u_f * rearrange(D.float(), "d -> d 1")
    if z is not None:
        out = out * F.silu(z.float())
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
