"""mamba_ssm's Mamba block, made runnable off the GPU.

This file needs ``mamba_ssm``: it subclasses its ``Mamba``. Only
:mod:`neural_methods.model.physmamba.mamba_backend` imports it, and only
where the package is installed.
"""

from mamba_ssm import Mamba

from neural_methods.model.physmamba.mamba_ref import MambaRef


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
