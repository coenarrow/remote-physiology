"""DeepPhys as a multi-channel, multi-signal backbone behind the dict contract."""
import pytest
import torch
from torch.utils.data import default_collate

from neural_methods.batch import PREDICTIONS
from neural_methods.frame_transforms import FrameTransform
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.SignalDictWrapper import SignalDictWrapper
from tests.test_batch_contract import make_sample


# --- the backbone itself -------------------------------------------------
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


# --- wrapped in the dict contract ---------------------------------------
def _wrapped(channels=("R", "G", "B"), traces=("ABP", "CVP"), size=(36, 36)):
    transform = FrameTransform(("DiffNormalized", "Standardized"), size=size)
    backbone = DeepPhys(in_channels=len(channels), out_signals=len(traces),
                        img_size=size[0])
    return SignalDictWrapper(backbone, channels=channels, traces=traces,
                             input_mode='frames2d', frame_transform=transform)


def test_wrapper_declares_the_doubled_input_width():
    model = _wrapped()
    assert model.in_channels == 6     # 3 channels x 2 DATA_TYPE blocks
    assert model.out_signals == 2


def test_wrapper_frames2d_returns_the_batch_plus_predictions():
    model = _wrapped()
    batch = default_collate([make_sample(t=8, hw=(40, 40)),
                             make_sample(t=8, hw=(40, 40))])
    out = model(batch)
    assert set(out) == set(batch) | {PREDICTIONS}
    assert set(out[PREDICTIONS]) == {"ABP", "CVP"}
    assert out[PREDICTIONS]["ABP"].shape == (2, 8)
    assert out["frames"] is batch["frames"]        # nothing dropped on the way through


def test_wrapper_video3d_dict_output():
    class Fake3D(torch.nn.Module):
        def forward(self, x):                  # (B, C, T, H, W) -> (B, S, T)
            return x.mean(dim=(3, 4))[:, :2, :]
    model = SignalDictWrapper(Fake3D(), channels=("R", "G", "B"),
                              traces=("ABP", "CVP"), input_mode='video3d')
    batch = default_collate([make_sample(t=8)])
    out = model(batch)
    assert out[PREDICTIONS]["ABP"].shape == (1, 8)
    assert out[PREDICTIONS]["CVP"].shape == (1, 8)


def test_wrapper_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown input_mode"):
        SignalDictWrapper(torch.nn.Identity(), channels=("R",), traces=("ABP",),
                          input_mode='pointcloud')


def test_wrapper_gradients_reach_the_backbone():
    model = _wrapped()
    batch = default_collate([make_sample(t=6, hw=(36, 36))])
    model(batch)[PREDICTIONS]["ABP"].sum().backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
