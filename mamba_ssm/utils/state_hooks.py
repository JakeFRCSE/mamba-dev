from __future__ import annotations

import torch
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.ops.selective_scan_interface import set_selective_scan_hook


class InferenceParamsCapture:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.conv_states = {}
        self.ssm_states = {}
        self.current_layer_idx = None

    def enable(self):
        set_selective_scan_hook(self._selective_scan_hook)
        for module in self.model.modules():
            if isinstance(module, Mamba) and module.layer_idx is not None:
                self.handles.append(
                    module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
                )
                self.handles.append(
                    module.register_forward_hook(self._post_hook, with_kwargs=True)
                )
        return self

    def disable(self):
        set_selective_scan_hook(None)
        for handle in self.handles:
            handle.remove()
        self.handles = []
        return self

    def build_inference_params(self):
        if not self.conv_states or not self.ssm_states:
            raise ValueError("No captured states. Run a forward pass with hooks enabled.")
        sample_state = next(iter(self.conv_states.values()))
        bs = sample_state.shape[0]
        params = InferenceParams(max_seqlen=0, max_batch_size=bs)
        params.key_value_memory_dict = {
            layer_idx: (conv_state, self.ssm_states[layer_idx])
            for layer_idx, conv_state in self.conv_states.items()
            if layer_idx in self.ssm_states
        }
        return params

    def _selective_scan_hook(self, last_state):
        if self.current_layer_idx is None:
            return
        self.ssm_states[self.current_layer_idx] = last_state.detach()

    def _pre_hook(self, module, args, kwargs):
        if kwargs.get("inference_params") is None:
            self.current_layer_idx = module.layer_idx

    def _post_hook(self, module, args, kwargs, output):
        if kwargs.get("inference_params") is not None:
            return
        if not args:
            return
        hidden_states = args[0]
        if hidden_states is None or hidden_states.dim() != 3:
            return
        with torch.no_grad():
            xz = module.in_proj(hidden_states).transpose(1, 2)
            x, _ = xz.chunk(2, dim=1)
            conv_state = F.pad(x, (module.d_conv - x.shape[-1], 0))
        self.conv_states[module.layer_idx] = conv_state.detach()
