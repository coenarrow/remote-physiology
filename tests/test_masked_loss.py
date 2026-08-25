import torch

from neural_methods.loss.MaskedMultiSignalLoss import MaskedMultiSignalLoss


def _mk(B=4, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    preds = {'ABP': torch.randn(B, T, generator=g), 'CVP': torch.randn(B, T, generator=g)}
    labels = {'ABP': torch.randn(B, T, generator=g), 'CVP': torch.randn(B, T, generator=g)}
    return preds, labels


def test_hand_computed_masked_mse():
    loss_fn = MaskedMultiSignalLoss(['ABP', 'CVP'], base='mse')
    preds = {'ABP': torch.zeros(2, 4), 'CVP': torch.zeros(2, 4)}
    labels = {'ABP': torch.ones(2, 4), 'CVP': torch.full((2, 4), 2.0)}
    mask = {'ABP': torch.tensor([1.0, 1.0]), 'CVP': torch.tensor([1.0, 0.0])}
    # ABP: per-sample MSE = 1.0, both present -> 1.0
    # CVP: per-sample MSE = 4.0, one present -> 4.0
    out = loss_fn(preds, labels, mask)
    assert torch.isclose(out, torch.tensor((1.0 + 4.0) / 2))


def test_fully_masked_signal_contributes_zero_no_nan():
    loss_fn = MaskedMultiSignalLoss(['ABP', 'CVP'], base='mse')
    preds, labels = _mk()
    mask = {'ABP': torch.ones(4), 'CVP': torch.zeros(4)}
    out = loss_fn(preds, labels, mask)
    assert torch.isfinite(out)
    # equals half the ABP-only term (CVP contributes exactly 0)
    only_abp = MaskedMultiSignalLoss(['ABP'], base='mse')(
        {'ABP': preds['ABP']}, {'ABP': labels['ABP']}, {'ABP': mask['ABP']})
    assert torch.isclose(out, only_abp / 2)


def test_negpearson_base_and_gradients():
    loss_fn = MaskedMultiSignalLoss(['ABP'], base='negpearson')
    p = torch.randn(3, 32, requires_grad=True)
    labels = {'ABP': torch.randn(3, 32)}
    out = loss_fn({'ABP': p}, labels, {'ABP': torch.ones(3)})
    out.backward()
    assert torch.isfinite(out)
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_unknown_base_raises():
    import pytest
    with pytest.raises(ValueError):
        MaskedMultiSignalLoss(['ABP'], base='huber')
