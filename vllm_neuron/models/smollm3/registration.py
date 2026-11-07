# SPDX-License-Identifier: Apache-2.0
"""Process-local SmolLM3 text identity; actual cache allocation is NxDI's."""
from transformers import AutoConfig,PretrainedConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
from .modeling import install_nxdi
_INSTALLED=False


class SmolLM3NeuronTextConfig(PretrainedConfig):
    model_type='smollm3_neuron_text'


class SmolLM3NeuronTextForCausalLM(NeuronCausalLM):
    has_inner_state=False
    is_hybrid=False

    def __init__(self,*,vllm_config,prefix=''):
        super().__init__(vllm_config.model_config.hf_config)

    def forward(self,input_ids,positions,input_block_ids,**kwargs):
        return super().forward(input_ids,input_block_ids,position_ids=positions,**kwargs)

    def compute_logits(self,hidden_states):return hidden_states


def install():
    global _INSTALLED
    if _INSTALLED:return
    install_nxdi()
    AutoConfig.register('smollm3_neuron_text',SmolLM3NeuronTextConfig)
    arch='SmolLM3NeuronTextForCausalLM'
    if arch in ModelRegistry.get_supported_archs():raise RuntimeError('Refusing to replace existing SmolLM3 registration')
    ModelRegistry.register_model(arch,SmolLM3NeuronTextForCausalLM)
    _INSTALLED=True
