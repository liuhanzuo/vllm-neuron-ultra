# SPDX-License-Identifier: Apache-2.0
"""Honest, process-local text architecture registration for the custom adapter."""
from transformers import AutoConfig, PretrainedConfig
import torch
from vllm.model_executor.models.registry import ModelRegistry
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
from .modeling import install_nxdi, NeuronQwen35ForCausalLM
from .identity import SCHEMA, require, selected_variant
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm_neuron.worker import neuronx_distributed_model_loader as loader
_INSTALLED = False


class Qwen35NeuronTextConfig(PretrainedConfig):
    model_type = 'qwen35_neuron_text'


class Qwen35NeuronTextForCausalLM(NeuronCausalLM):
    """vLLM interface metadata and executable delegate to its Neuron loader.

    The Neuron worker actually constructs its generic NeuronCausalLM and loads
    the explicit NxDI implementation registered by install_nxdi().
    """
    has_inner_state = True
    is_hybrid = True

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config, use_v1=True):
        # vLLM calls all fixed recurrent caches "mamba". These are the actual
        # Qwen delta-rule shapes, and NxDI owns their allocation and updates.
        c = vllm_config.model_config.hf_config
        tp = vllm_config.parallel_config.tensor_parallel_size
        channels = (2*c.linear_num_key_heads*c.linear_key_head_dim +
                    c.linear_num_value_heads*c.linear_value_head_dim)//tp
        return ((channels,c.linear_conv_kernel_dim),
                (c.linear_num_value_heads//tp,c.linear_key_head_dim,c.linear_value_head_dim))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return (vllm_config.model_config.dtype, torch.float32)

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config.model_config.hf_config)

    def forward(self, input_ids, positions, input_block_ids, **kwargs):
        return super().forward(input_ids, input_block_ids, position_ids=positions, **kwargs)

    def compute_logits(self, hidden_states):
        # NeuronCausalLM.forward already returns last-token vocabulary logits.
        return hidden_states


def install():
    global _INSTALLED
    selected_variant()
    require(getattr(loader, "QWEN35_ARTIFACT_ADMISSION_SCHEMA", None) == SCHEMA,
            "Qwen3.5 requires the package-aware artifact admission loader")
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    if _INSTALLED:
        require(MODEL_TYPES.get("qwen35neurontext") == {"causal-lm": NeuronQwen35ForCausalLM},
                "Installed Qwen3.5 implementation changed")
        return
    # Validate every occupied identifier before process-local registry mutation.
    require("qwen35_neuron_text" not in CONFIG_MAPPING, "Refusing to replace Transformers Qwen3.5 config")
    require("Qwen35NeuronTextForCausalLM" not in ModelRegistry.get_supported_archs(),
            "Refusing to replace vLLM Qwen3.5 architecture")
    require("qwen35neurontext" not in MODEL_TYPES or
            MODEL_TYPES["qwen35neurontext"] == {"causal-lm": NeuronQwen35ForCausalLM},
            "Refusing to replace NxDI Qwen3.5 implementation")
    install_nxdi()
    AutoConfig.register(Qwen35NeuronTextConfig.model_type, Qwen35NeuronTextConfig)
    ModelRegistry.register_model("Qwen35NeuronTextForCausalLM", Qwen35NeuronTextForCausalLM)
    _INSTALLED = True
