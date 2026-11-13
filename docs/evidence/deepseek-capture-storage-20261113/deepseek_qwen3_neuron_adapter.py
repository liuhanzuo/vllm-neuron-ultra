"""Native NxDI Qwen3 with only the checkpoint's missing YaRN replaced."""
from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import NeuronQwen3Model, NeuronQwen3ForCausalLM
from deepseek_qwen3_rope import DeepSeekQwen3RotaryEmbedding


class NeuronDeepSeekQwen3Model(NeuronQwen3Model):
    def init_model(self, config):
        super().init_model(config)
        for layer in self.layers:
            layer.self_attn.rotary_emb=DeepSeekQwen3RotaryEmbedding(config)


class NeuronDeepSeekQwen3ForCausalLM(NeuronQwen3ForCausalLM):
    _model_cls=NeuronDeepSeekQwen3Model


def install_nxdi():
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    key='deepseekqwen3neuronyarn'
    if key in MODEL_TYPES and MODEL_TYPES[key]['causal-lm'] is not NeuronDeepSeekQwen3ForCausalLM:
        raise RuntimeError('Refusing to replace an existing DeepSeek Qwen3 YaRN registration')
    MODEL_TYPES[key]={'causal-lm':NeuronDeepSeekQwen3ForCausalLM}
