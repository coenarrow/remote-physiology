"""Tests for the mamba_ssm compatibility layer (BiMamba + parallel scan)."""
import numpy as np
import pytest
import torch

from neural_methods.model import mamba_compat as mc
from neural_methods.model.mamba_compat import BiMamba, make_mamba


def test_make_mamba_plain_returns_mamba():
    from mamba_ssm import Mamba
    m = make_mamba(32)
    assert isinstance(m, Mamba)


def test_make_mamba_bimamba_returns_bimamba():
    m = make_mamba(32, bimamba=True)
    assert isinstance(m, BiMamba)


def test_make_mamba_drops_unknown_kwargs():
    # kwargs from old fork call-sites must not crash vanilla packages
    m = make_mamba(32, bimamba=True, use_fast_path=False, not_a_real_kwarg=1)
    assert isinstance(m, BiMamba)


def test_bimamba_shape_and_finite():
    m = BiMamba(48)
    x = torch.randn(2, 100, 48)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_bimamba_uses_both_directions():
    m = BiMamba(32)
    x = torch.randn(2, 64, 32)
    m(x).sum().backward()
    fwd_grads = [p.grad for p in m.fwd.parameters()]
    bwd_grads = [p.grad for p in m.bwd.parameters()]
    assert all(g is not None for g in fwd_grads)
    assert all(g is not None for g in bwd_grads)
    assert any(g.abs().sum() > 0 for g in fwd_grads)
    assert any(g.abs().sum() > 0 for g in bwd_grads)


def test_bimamba_backward_direction_matters():
    """The bwd block must actually see the reversed sequence: zeroing the
    fwd block's contribution, output at t must depend on inputs later than t."""
    torch.manual_seed(0)
    m = BiMamba(16)
    x = torch.randn(1, 32, 16)
    x2 = x.clone()
    x2[:, -1, :] += 10.0  # perturb the LAST timestep
    with torch.no_grad():
        d = (m(x) - m(x2)).abs().sum(dim=-1).squeeze(0)
    assert d[0] > 0, "earliest output should react to a last-timestep change (backward path)"


@pytest.mark.skipif(mc._selective_scan_ref_orig is None, reason="mamba_ssm ops unavailable")
def test_parallel_scan_matches_reference():
    torch.manual_seed(0)
    Bb, Dd, Ll, Nn = 2, 16, 200, 8  # non-power-of-2 L
    u = torch.randn(Bb, Dd, Ll)
    delta = torch.randn(Bb, Dd, Ll) * 0.5
    A = -torch.exp(torch.randn(Dd, Nn) * 0.5)
    Bv = torch.randn(Bb, Nn, Ll)
    Cv = torch.randn(Bb, Nn, Ll)
    Dv = torch.randn(Dd)
    z = torch.randn(Bb, Dd, Ll)
    db = torch.randn(Dd)

    def mk():
        return [t.clone().requires_grad_() for t in (u, delta, A, Bv, Cv, z)]

    a1, a2 = mk(), mk()
    ref = mc._selective_scan_ref_orig(a1[0], a1[1], a1[2], a1[3], a1[4], Dv, a1[5],
                                      delta_bias=db, delta_softplus=True)
    par = mc.selective_scan_parallel(a2[0], a2[1], a2[2], a2[3], a2[4], Dv, a2[5],
                                     delta_bias=db, delta_softplus=True)
    assert (ref - par).abs().max().item() < 1e-4
    ref.sum().backward()
    par.sum().backward()
    for t1, t2 in zip(a1, a2):
        assert (t1.grad - t2.grad).abs().max().item() < 1e-3


def test_physmamba_forward():
    from neural_methods.model.PhysMamba import PhysMamba
    with torch.no_grad():
        out = PhysMamba(frames=128)(torch.randn(1, 3, 128, 32, 32))
    assert out.shape == (1, 128)
    assert torch.isfinite(out).all()


def test_physhydra_forward():
    from neural_methods.model.PhysHydra import PhysHydra
    with torch.no_grad():
        out = PhysHydra(in_channels=3, out_signals=1, frames=128)(torch.randn(1, 3, 128, 32, 32))
    pred = out[0] if isinstance(out, (tuple, list)) else out
    assert pred.shape == (1, 1, 128)
    assert torch.isfinite(pred).all()
