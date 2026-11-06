# SPDX-License-Identifier: Apache-2.0
"""Process-local official LFM3 architecture/config identifier on vLLM-Neuron."""
from transformers import AutoConfig,PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.model_executor.models.registry import ModelRegistry
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
from .modeling import install_nxdi, NeuronLfm3ForCausalLM
from .identity import selected_variant, require
_INSTALLED=False
_VARIANT=None


class Lfm3NeuronConfig(PretrainedConfig):
    model_type='lfm3'


class Lfm3ForCausalLM(NeuronCausalLM):
    has_inner_state=True
    is_hybrid=True

    @classmethod
    def get_mamba_state_shape_from_config(cls,vllm_config,use_v1=True):
        c=vllm_config.model_config.hf_config;tp=vllm_config.parallel_config.tensor_parallel_size
        return ((c.hidden_size//tp,c.conv_L_cache),)

    @classmethod
    def get_mamba_state_dtype_from_config(cls,vllm_config):
        return (vllm_config.model_config.dtype,)

    def __init__(self,*,vllm_config,prefix=''):
        super().__init__(vllm_config.model_config.hf_config)

    def forward(self,input_ids,positions,input_block_ids,**kwargs):
        return super().forward(input_ids,input_block_ids,position_ids=positions,**kwargs)

    def compute_logits(self,hidden_states):return hidden_states


def install():
    global _INSTALLED, _VARIANT
    variant = selected_variant()
    if _INSTALLED:
        require(variant == _VARIANT, "Cannot switch the LFM3 writer within one process")
        return
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    # Check every occupied registration before making process-local mutations.
    require('lfm3' not in CONFIG_MAPPING, "Refusing to replace an existing Transformers LFM3 config")
    require('Lfm3ForCausalLM' not in ModelRegistry.get_supported_archs(),
            "Refusing to replace an existing vLLM LFM3 registration")
    require('lfm3' not in MODEL_TYPES or MODEL_TYPES['lfm3'] == {'causal-lm': NeuronLfm3ForCausalLM},
            "Refusing to replace an existing NxDI LFM3 implementation")
    from . import text_core
    from .helpers import store_slots
    require(text_core.store_slots is store_slots, "Another LFM3 writer is already installed")
    if variant == 'nki_predicated_decode_v4':
        from .cache_writer import install as install_writer
        install_writer()
    install_nxdi()
    AutoConfig.register('lfm3', Lfm3NeuronConfig)
    ModelRegistry.register_model('Lfm3ForCausalLM', Lfm3ForCausalLM)
    _VARIANT, _INSTALLED = variant, True
