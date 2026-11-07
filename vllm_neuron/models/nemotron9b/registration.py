# SPDX-License-Identifier: Apache-2.0
"""Process-local Nemotron9B text identity; actual cache allocation is NxDI's."""
from transformers import AutoConfig,PretrainedConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
from .modeling import install_nxdi, NeuronNemotron9BForCausalLM
from .identity import require, selected_variant
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
_INSTALLED=False


class Nemotron9BNeuronTextConfig(PretrainedConfig):
    model_type='nemotron9b_neuron_text'


class Nemotron9BNeuronTextForCausalLM(NeuronCausalLM):
    has_inner_state=False
    is_hybrid=False

    def __init__(self,*,vllm_config,prefix=''):
        super().__init__(vllm_config.model_config.hf_config)

    def forward(self,input_ids,positions,input_block_ids,**kwargs):
        return super().forward(input_ids,input_block_ids,position_ids=positions,**kwargs)

    def compute_logits(self,hidden_states):return hidden_states


def install():
    global _INSTALLED
    selected_variant()
    if _INSTALLED:
        return
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    from . import text_core, cache
    require('nemotron9b_neuron_text' not in CONFIG_MAPPING, 'Existing Transformers config')
    require('Nemotron9BNeuronTextForCausalLM' not in ModelRegistry.get_supported_archs(), 'Existing vLLM architecture')
    require('nemotron9bneurontext' not in MODEL_TYPES or MODEL_TYPES['nemotron9bneurontext']=={'causal-lm':NeuronNemotron9BForCausalLM}, 'Existing NxDI architecture')
    require(text_core.Mamba.forward is cache.ORIGINAL_FORWARD, 'Existing Mamba implementation')
    cache.install()
    install_nxdi()
    AutoConfig.register('nemotron9b_neuron_text',Nemotron9BNeuronTextConfig)
    ModelRegistry.register_model('Nemotron9BNeuronTextForCausalLM',Nemotron9BNeuronTextForCausalLM)
    _INSTALLED=True
