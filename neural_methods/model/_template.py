"""A backbone template on the multi-signal contract. Copy, rename, replace.

This is not a real architecture. It is the smallest ``nn.Module`` that
satisfies everything ``src.models.MultiTraceModel`` and ``src.trainer`` ask
of a backbone, with each requirement marked. Copy it to
``neural_methods/model/<Name>.py``, rename the class, and swap the layers for
the published network. Then register it in ``src/models.py`` and write
``configs/models/<name>.yaml`` — the steps are in ``docs/adding_a_model.md``.

What a backbone is, in one paragraph: a function from one raw clip to one
trace. It never sees the batch dict, never knows which trace it is
predicting and never computes a loss. It does normalise its own input: the
dataset hands over raw resized pixels, and whatever the published network
was fed (standardised frames, frame differences) the backbone applies
itself as its first stage, from ``neural_methods.model.modules``. The
wrapper makes one copy per trace and owns the dict; the trainer owns the
loss.
"""

import torch
import torch.nn as nn
from einops import rearrange

from neural_methods.model.modules.diffnormalize import DiffNormalize


class TemplateNet(nn.Module):
    """``(B, C_in, T, H, W) -> (B, 1, T)``: a clip backbone.

    Every backbone takes a whole clip. A published network that ran on single
    frames folds ``T`` into the batch axis itself after normalising the clip;
    see ``DeepPhys.py``.
    """

    # INTERIM ONLY, for a migration that has not got there yet. The
    # destination is an adaptive stage that accepts any window length and is
    # an exact no-op at the paper's; none of the ten migrated models declares
    # either of these any more (docs/adding_a_model.md, "Any frame size, any
    # window length"). Until yours has one, declare that the window must be a
    # multiple of something ...
    temporal_divisor = 1
    # ... or that it must be exactly one clip length. Delete both once the
    # adaptive stage is in; while they are here the trainer reads them off the
    # first copy and refuses a WINDOW_SECONDS that does not fit, naming the
    # fix, rather than truncating.
    # temporal_length = 128

    def __init__(self, in_channels: int = 3, hidden: int = 16):
        """``in_channels`` is REQUIRED and is the only width the wrapper
        always passes: ``len(interface.CHANNELS)``. Anything else the
        interface determines (an ``img_size`` for a dense layer, say) is an
        argument the builder passes. Every published hyperparameter is a
        default here, never a YAML key."""
        super().__init__()
        self.in_channels = in_channels
        # REQUIRED: the input preprocessing the paper's DATA_TYPE named, as
        # the network's own first stage. ``DiffNormalize`` for a
        # DiffNormalized paper, ``Standardize`` for a Standardized one, a
        # bare ``nn.Identity()`` for a paper that took Raw frames.
        self.input_norm = DiffNormalize()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),      # keep T, drop H and W
        )
        # REQUIRED: the readout is activation-free and one wide, with a bias
        # of exactly one element. The trainer seeds that bias with the
        # trace's physiological prior and exempts it from weight decay.
        self.readout = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)

    def output_layers(self):
        """REQUIRED. The activation-free readout(s) of this one copy."""
        return (self.readout,)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, T, H, W) -> (B, 1, T)``.

        The input is raw: the interface's channels stacked in order, resized
        by the dataset and nothing else. Normalise first. A two-branch
        network (DeepPhys, TS-CAN) runs the clip through two normalisers and
        feeds one branch each.
        """
        x = self.features(self.input_norm(video))
        x = rearrange(x, "b c t 1 1 -> b c t")      # einops, never view/permute
        return self.readout(x)                       # (B, 1, T), three dims
