"""Source-only successor for CPU-sampling capture; never alters installed SDK.

Requires new compilation, serial admission and full output/state neutrality.
Not imported during source preparation. Captures full feature vectors at every
original prediction position, 13 boundaries/layer, with a trailing TP-rank axis.
Other prefill tokens, attention probabilities and MLP internals are unobserved.
"""
from types import MethodType

from neuronx_distributed.parallel_layers.mappings import gather_from_tensor_model_parallel_region
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed_inference.models.config import TensorCaptureConfig
from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import Qwen3NeuronConfig, Qwen3InferenceConfig

from deepseek_qwen3_neuron_adapter import NeuronDeepSeekQwen3Model, NeuronDeepSeekQwen3ForCausalLM
import deepseek_cpu_capture_bridge_v1 as bridge


class DeepSeekCaptureNeuronConfig(Qwen3NeuronConfig):
    def __init__(self, **kwargs):
        version = kwargs.pop('deepseek_cpu_capture_version', None)
        requested = kwargs.pop('tensor_capture_config', None)
        bridge.require(version == bridge.VERSION, 'explicit capture artifact version required')
        if isinstance(requested, TensorCaptureConfig):
            requested = vars(requested)
        bridge.require(requested == bridge.capture_config(), 'exact capture schema required')
        # The stock constructor couples capture to on-device sampling. Run its
        # complete ordinary CPU-sampler validation, then install our own output
        # ABI, implemented below at both the core and application boundaries.
        super().__init__(tensor_capture_config=None, **kwargs)
        bridge.check_cpu_config(self)
        self.deepseek_cpu_capture_version = version
        self.tensor_capture_config = TensorCaptureConfig(**requested)
        bridge.require(self.tensor_capture_config.get_offset() == bridge.CAPTURE_COUNT, 'capture offset')


class DeepSeekCaptureInferenceConfig(Qwen3InferenceConfig):
    @classmethod
    def get_neuron_config_cls(cls):
        return DeepSeekCaptureNeuronConfig


class DeepSeekCaptureModel(NeuronDeepSeekQwen3Model):
    def init_model(self, config):
        super().init_model(config)
        bridge.check_cpu_config(config.neuron_config)
        bridge.require(len(self.layers) == bridge.LAYERS, 'all original layers required')
        self._boundary_values = {}
        self._boundary_handles = []
        for index, layer in enumerate(self.layers):
            prefix = f'layers.{index}.'
            bridge.require(not layer.self_attn.post_transpose_layernorm, 'pinned Q/K norm BSHD layout')

            def save(name, value):
                bridge.require(name not in self._boundary_values, 'duplicate boundary: ' + name)
                self._boundary_values[name] = bridge.select_prediction(value, name, self._boundary_indices)

            def pre(module, name):
                def hook(mod, args, kwargs):
                    value = args[0] if args else kwargs['hidden_states']
                    save(name, value)
                self._boundary_handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

            def post(module, name, tuple_index=None):
                def hook(mod, args, output):
                    value = output if tuple_index is None else output[tuple_index]
                    save(name, value)
                self._boundary_handles.append(module.register_forward_hook(hook))

            pre(layer.input_layernorm, prefix + 'layer_input')
            post(layer.input_layernorm, prefix + 'normalized_input')
            post(layer.self_attn.q_layernorm, prefix + 'q_normalized')
            post(layer.self_attn.k_layernorm, prefix + 'k_normalized')
            pre(layer.self_attn.get_o_proj(), prefix + 'attention_heads')
            post(layer.self_attn.get_o_proj(), prefix + 'attention_projected')
            pre(layer.post_attention_layernorm, prefix + 'attention_residual')
            post(layer.post_attention_layernorm, prefix + 'mlp_normalized')
            post(layer.mlp, prefix + 'mlp_output', tuple_index=0)
            post(layer, prefix + 'layer_output', tuple_index=0)
            original = layer.self_attn.prep_qkv_tensors

            def prepared(attention, *args, _original=original, _prefix=prefix, **kwargs):
                result = _original(*args, **kwargs)
                bridge.require(isinstance(result, tuple) and len(result) == 6, 'pinned prep_qkv ABI')
                # Explicitly retain all three tensors. SDK's module-first-output
                # shortcut would lose K and V and is deliberately not used.
                for name, value in zip(('query', 'key', 'value'), result[:3]):
                    save(_prefix + name, value)
                return result
            layer.self_attn.prep_qkv_tensors = MethodType(prepared, layer.self_attn)

    def forward(self, *args, **kwargs):
        bridge.require(not self._boundary_values, 'stale capture values before forward')
        positions = args[2] if len(args) > 2 else kwargs['position_ids']
        # Pinned CTE padding adds position 1; all seven prefixes exceed two
        # tokens, so argmax selects the last real
        # prefix token. TKG contains one query per row and selects column zero.
        self._boundary_indices = positions.argmax(dim=1)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._boundary_values.clear()

    def _get_captured_tensors(self, device):
        bridge.require(set(self._boundary_values) == set(bridge.NAMES), 'every declared boundary must execute')
        group = get_tensor_model_parallel_group()
        bridge.require(group.size() == 2, 'original TP2 required')
        values = []
        for name in bridge.NAMES:
            value = self._boundary_values[name]
            bridge.require(str(value.dtype) == 'torch.bfloat16', name + ': original BF16 required')
            # Gathering a singleton last axis preserves both feature shards;
            # no assumption that Q/K/V rank zero contains every logical head.
            values.append(gather_from_tensor_model_parallel_region(value.unsqueeze(-1), process_group=group))
        return values


class DeepSeekCaptureForCausalLM(NeuronDeepSeekQwen3ForCausalLM):
    _model_cls = DeepSeekCaptureModel

    @staticmethod
    def get_config_cls():
        return DeepSeekCaptureInferenceConfig

    def _copy_past_key_values(self, outputs):
        bridge.check_cpu_config(self.neuron_config)
        _, _, caches = bridge.split_outputs(outputs, device=False)
        for wrapper in (self.token_generation_model, self.context_encoding_model):
            bridge.require(len(wrapper.model.kv_mgr.past_key_values) == bridge.KV_COUNT, 'all 72 KV slots')
        for index, value in enumerate(caches):
            self.token_generation_model.model.kv_mgr.past_key_values[index].data = value
            self.context_encoding_model.model.kv_mgr.past_key_values[index].data = value

    def _get_constructed_outputs(self, outputs, is_run_on_neuron):
        bridge.check_cpu_config(self.neuron_config)
        logits, captures, _ = bridge.split_outputs(outputs, device=is_run_on_neuron)
        result = self._construct_output(logits)
        result.captured_tensors = captures
        result.capture_names = bridge.NAMES
        result.capture_rank_axis = -1
        return result

    def _get_model_outputs(self, input_ids, *args, **kwargs):
        # The original cold schedule uses CTE B1 and TKG B4. Stock wrapper
        # batch splitting/padding cannot concatenate this new output ABI.
        expected = 1 if input_ids.shape[1] > 1 else 4
        bridge.require(input_ids.shape[0] == expected, 'only original cold CTE B1/TKG B4 allowed')
        bridge.require(self.neuron_config.ctx_batch_size == 1 and self.neuron_config.tkg_batch_size == 4,
                       'original wrapper batch sizes required')
        return super()._get_model_outputs(input_ids, *args, **kwargs)
