"""The pure-PyTorch Mamba block, for platforms without mamba_ssm.

See :mod:`neural_methods.model.physmamba.mamba_compat` for where it stands in.
All reshaping is einops.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from neural_methods.model.physmamba.selective_scan import parallel_linear_scan


class MambaRef(nn.Module):
    """Selective-SSM block in plain PyTorch, for platforms without mamba_ssm.

    Layer-for-layer the vanilla Mamba block: gated input projection, causal
    depthwise conv, input-dependent (delta, B, C), the diagonal selective scan,
    skip term D and an output projection. The recurrence runs through
    :func:`neural_methods.model.physmamba.selective_scan.parallel_linear_scan`,
    the same primitive the macOS path patches into mamba_ssm, so behaviour
    matches that path rather than approximating it.

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

        # Every parameter-holding attribute below (`in_proj`, `conv1d`,
        # `x_proj`, `dt_proj`, `out_proj`, `A_log`, `D`) is spelled exactly as
        # in mamba_ssm's `Mamba`, so this block and the CUDA one exchange
        # state_dicts and `PortableMamba` can run its weights through this
        # forward. They are the one exemption from the PEP 8 attribute rule.
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
        u = F.silu(rearrange(u, "b d l -> b l d"))                # (B, L, d_inner)

        delta, input_b, output_c = torch.split(
            self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))                   # (B, L, d_inner)
        state_a = -torch.exp(self.A_log.float())                  # (d_inner, d_state)

        # Zero-order-hold discretisation, then the linear recurrence.
        delta_a = torch.exp(einsum(delta, state_a, "b l d, d n -> b d l n"))
        delta_b_u = einsum(delta * u, input_b, "b l d, b l n -> b d l n")
        h = parallel_linear_scan(delta_a, delta_b_u)              # (B, d_inner, L, n)

        y = einsum(h, output_c, "b d l n, b l n -> b l d") + u * self.D
        return self.out_proj(y * F.silu(z))

    def extra_repr(self):
        return (f"d_model={self.d_model}, d_state={self.d_state}, "
                f"d_conv={self.d_conv}, d_inner={self.d_inner}")
