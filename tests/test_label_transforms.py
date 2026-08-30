import pytest
import torch

from dataset.data_loader.label_transforms import (
    EPS,
    STAT_NAMES,
    minmax,
    minmax_inverse,
    zscore,
    zscore_inverse,
)


def _trace():
    torch.manual_seed(0)
    return 80.0 + 15.0 * torch.randn(64)


def test_stat_names_canonical():
    assert STAT_NAMES == ("mean", "std", "min", "max")


def test_zscore_stats_values_and_unbiased_std():
    t = _trace()
    _, stats = zscore(t)
    assert set(stats) == set(STAT_NAMES)
    assert all(v.dim() == 0 for v in stats.values())
    assert torch.equal(stats["mean"], t.mean())
    assert torch.equal(stats["std"], t.std(correction=1))
    assert torch.equal(stats["min"], t.amin())
    assert torch.equal(stats["max"], t.amax())


def test_zscore_round_trip_exact():
    t = _trace()
    normed, stats = zscore(t)
    # float32 at ~80-magnitude: a few ulps of slack; exactness in the sub-EPS
    # band has its own dedicated test below.
    assert torch.allclose(zscore_inverse(normed, stats), t, atol=1e-3)


def test_minmax_range_and_round_trip():
    t = _trace()
    normed, stats = minmax(t)
    assert normed.min().item() == pytest.approx(0.0)
    assert normed.max().item() == pytest.approx(1.0)
    assert torch.allclose(minmax_inverse(normed, stats), t, atol=1e-3)


def test_constant_trace_never_nan():
    t = torch.full((16,), 7.0)
    for fn in (zscore, minmax):
        normed, _ = fn(t)
        assert torch.isfinite(normed).all()
        assert torch.equal(normed, torch.zeros(16))


def test_zero_trace_zero_stats():
    # The absent-label invariant: zero-filled traces emit zeros with zero stats.
    t = torch.zeros(16)
    for fn in (zscore, minmax):
        normed, stats = fn(t)
        assert torch.equal(normed, torch.zeros(16))
        assert all(v.item() == 0.0 for v in stats.values())


def test_inverse_broadcasts_collated_batch():
    t = _trace()
    normed, stats = zscore(t)
    sig_b = torch.stack([normed, normed])                    # (2, T)
    stats_b = {k: torch.stack([v, v]) for k, v in stats.items()}  # (2,)
    out = zscore_inverse(sig_b, stats_b)
    assert out.shape == (2, t.shape[0])
    assert torch.allclose(out[0], t, atol=1e-3)


def test_round_trip_exact_in_sub_eps_band():
    # Forward and inverse must clamp identically for 0 < std < EPS.
    t = torch.full((8,), 3.0) + torch.linspace(0, EPS / 10, 8)
    normed, stats = zscore(t)
    assert torch.allclose(zscore_inverse(normed, stats), t, atol=1e-6)
