"""Compatibility layer over mamba_ssm variants.

PhysMamba/PhysHydra historically depended on the vendored tools/mamba fork,
whose Mamba block accepted `bimamba=True` (Vim-style bidirectional SSM with
shared in/out projections). That fork is gone: this repo now targets the
vanilla mamba_ssm API only - the newest official `mamba-ssm` on Linux/CUDA
and `mamba-ssm-macos` on macOS.

`make_mamba(..., bimamba=True)` returns a `BiMamba`: two independent vanilla
Mamba blocks, one running on the time-reversed sequence, summed. This is a
composition-based replacement for the fork's bimamba, NOT parameter-compatible
with it (separate in/out projections per direction) - checkpoints trained with
the fork do not load. On CUDA each direction uses the fused fast path; on
macOS both directions pick up the parallel selective scan patched in below.

Where `mamba_ssm` is not installed, `MambaRef` below stands in: the same
selective-SSM block written in plain PyTorch on top of the parallel scan in this
module, with the same parameter names and shapes, so a checkpoint moves either
way. It is numerically a Mamba block and trains, but it materialises the hidden
state, so it is a development and CI path rather than a performance one.

Windows does get the real package: `mamba-ssm` publishes no Windows wheel and
its CUDA sources need three MSVC fixes, so `pyproject.toml` builds the patched
`vendor/mamba-ssm` tree there (see `tools/vendor_mamba_windows.py`). `MambaRef`
remains the fallback wherever that build is unavailable -- no CUDA toolkit, or
a platform nobody ships for.
"""
import inspect
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as _F
from einops import einsum, rearrange
from einops import rearrange as _rearrange

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:  # pragma: no cover - exercised only where the package is absent
    Mamba = None
    HAS_MAMBA_SSM = False

_MAMBA_PARAMS = inspect.signature(Mamba.__init__).parameters if HAS_MAMBA_SSM else {}


def _filter_kwargs(kwargs):
    """Drop kwargs the installed Mamba implementation does not accept."""
    if not HAS_MAMBA_SSM:
        return {}
    return {k: v for k, v in kwargs.items() if k in _MAMBA_PARAMS}


class MambaRef(nn.Module):
    """Selective-SSM block in plain PyTorch, for platforms without mamba_ssm.

    Layer-for-layer the vanilla Mamba block: gated input projection, causal
    depthwise conv, input-dependent (delta, B, C), the diagonal selective scan,
    skip term D and an output projection. The recurrence runs through
    :func:`_parallel_linear_scan`, the same primitive the macOS path patches
    into mamba_ssm, so behaviour matches that path rather than approximating it.

    x: (B, L, D) -> y: (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None,
                 dt_min=1e-3, dt_max=1e-1, **_ignored):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # S4D-real initialisation: A_n = -n, held in log space so A stays negative.
        state_index = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(state_index).repeat(self.d_inner, 1).clone())
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Bias the timestep so softplus(bias) starts spread over [dt_min, dt_max],
        # matching the reference initialisation.
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        length = x.shape[1]
        gated = self.in_proj(x)                                  # (B, L, 2*d_inner)
        u, z = gated.chunk(2, dim=-1)

        # Causal depthwise convolution over time.
        u = self.conv1d(rearrange(u, "b l d -> b d l"))[..., :length]
        u = _F.silu(rearrange(u, "b d l -> b l d"))               # (B, L, d_inner)

        delta, B, C = torch.split(
            self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = _F.softplus(self.dt_proj(delta))                  # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                        # (d_inner, d_state)

        # Zero-order-hold discretisation, then the linear recurrence.
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b d l n"))
        deltaB_u = einsum(delta * u, B, "b l d, b l n -> b d l n")
        h = _parallel_linear_scan(deltaA, deltaB_u)               # (B, d_inner, L, n)

        y = einsum(h, C, "b d l n, b l n -> b l d") + u * self.D
        return self.out_proj(y * _F.silu(z))

    def extra_repr(self):
        return (f"d_model={self.d_model}, d_state={self.d_state}, "
                f"d_conv={self.d_conv}, d_inner={self.d_inner}")


if HAS_MAMBA_SSM:
    class PortableMamba(Mamba):
        """Vanilla Mamba that still runs when the tensors are not on a GPU.

        `mamba_ssm`'s selective scan and `causal_conv1d`'s convolution are
        CUDA-only -- both raise `Expected x.is_cuda() to be true` on a CPU
        tensor. Without this, merely *installing* mamba-ssm makes PhysMamba and
        PhysHydra unrunnable on CPU, so every CPU test of them fails on exactly
        the machines that have the fast path.

        `MambaRef` is the same block over the same parameter names and shapes
        (verified: `load_state_dict` succeeds strictly in both directions), so
        off CUDA we run *this module's own weights* through it. One parameter
        set, one checkpoint, two execution paths that agree to fp32 noise.
        """

        def forward(self, hidden_states, inference_params=None):
            if hidden_states.is_cuda:
                return super().forward(hidden_states,
                                       inference_params=inference_params)
            if inference_params is not None:
                raise NotImplementedError(
                    "Stepwise inference uses the fused CUDA kernels; move the "
                    "model to a GPU, or run the whole sequence at once.")
            return MambaRef.forward(self, hidden_states)
else:  # pragma: no cover - exercised only where the package is absent
    PortableMamba = None


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


def _mamba_block(d_model, d_state, d_conv, expand, **kwargs):
    """One unidirectional block from whichever implementation is available."""
    if HAS_MAMBA_SSM:
        return PortableMamba(d_model, d_state=d_state, d_conv=d_conv,
                             expand=expand, **_filter_kwargs(kwargs))
    _warn_once_about_fallback()
    return MambaRef(d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class BiMamba(nn.Module):
    """Bidirectional Mamba built from two vanilla Mamba blocks.

    forward direction processes the sequence as-is; backward direction
    processes the time-reversed sequence and un-reverses its output:
        y = fwd(x) + flip(bwd(flip(x)))
    x: (B, L, D) -> y: (B, L, D)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.fwd = _mamba_block(d_model, d_state, d_conv, expand, **kwargs)
        self.bwd = _mamba_block(d_model, d_state, d_conv, expand, **kwargs)

    def forward(self, x):
        return self.fwd(x) + self.bwd(x.flip(1)).flip(1)


def make_mamba(d_model, d_state=16, d_conv=4, expand=2, **kwargs):
    """Build a (bi)directional Mamba block from the installed vanilla package.

    bimamba=True -> BiMamba (two blocks, one time-reversed); otherwise a
    plain Mamba. Unknown kwargs are dropped for portability across
    mamba-ssm releases.
    """
    if kwargs.pop('bimamba', False):
        return BiMamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs)
    return _mamba_block(d_model, d_state, d_conv, expand, **kwargs)


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
