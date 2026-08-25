import torch

from neural_methods.model.DeepPhys import DeepPhys


def test_legacy_default_contract_unchanged():
    m = DeepPhys(img_size=72)
    out = m(torch.randn(8, 6, 72, 72))
    assert out.shape == (8, 1)


def test_five_channels_two_signals():
    m = DeepPhys(in_channels=5, out_signals=2, img_size=72)
    out = m(torch.randn(8, 10, 72, 72))
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


def test_arbitrary_img_size():
    # previously only 36/72/96 were supported via a lookup table
    m = DeepPhys(in_channels=3, out_signals=2, img_size=64)
    out = m(torch.randn(4, 6, 64, 64))
    assert out.shape == (4, 2)


def test_gradients_flow_from_both_blocks():
    m = DeepPhys(in_channels=5, out_signals=2, img_size=36)
    x = torch.randn(4, 10, 36, 36, requires_grad=True)
    m(x).sum().backward()
    g = x.grad.abs().sum(dim=(0, 2, 3))
    assert (g[:5] > 0).all()      # diff block used
    assert (g[5:] > 0).all()      # raw block used
