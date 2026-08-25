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


from neural_methods.model.SignalDictWrapper import SignalDictWrapper


def test_wrapper_frames2d_dict_output():
    from neural_methods.model.DeepPhys import DeepPhys
    backbone = DeepPhys(in_channels=3, out_signals=2, img_size=36)
    w = SignalDictWrapper(backbone, ['ABP', 'CVP'], input_mode='frames2d')
    out = w(torch.randn(2, 6, 8, 36, 36))     # (B, 2*C, T, H, W)
    assert set(out) == {'ABP', 'CVP'}
    assert out['ABP'].shape == (2, 8)
    assert out['CVP'].shape == (2, 8)


def test_wrapper_video3d_dict_output():
    class Fake3D(torch.nn.Module):
        def forward(self, x):                  # (B, C, T, H, W) -> (B, S, T)
            return x.mean(dim=(3, 4))[:, :2, :]
    w = SignalDictWrapper(Fake3D(), ['ABP', 'CVP'], input_mode='video3d')
    out = w(torch.randn(2, 3, 8, 16, 16))
    assert out['ABP'].shape == (2, 8) and out['CVP'].shape == (2, 8)


def test_wrapper_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        SignalDictWrapper(torch.nn.Identity(), ['ABP'], input_mode='pointcloud')
