"""Process-local Nemotron4BNeuronTextV1 text identity; actual cache allocation is NxDI's."""
from transformers import AutoConfig,PretrainedConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
from nemotron4b_neuron_adapter_v1 import install_nxdi
_INSTALLED=False


class Nemotron4BNeuronTextV1Config(PretrainedConfig):
    model_type='nemotron4b_neuron_text_v1'


class Nemotron4BNeuronTextV1ForCausalLM(NeuronCausalLM):
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
    AutoConfig.register('nemotron4b_neuron_text_v1',Nemotron4BNeuronTextV1Config)
    arch='Nemotron4BNeuronTextV1ForCausalLM'
    if arch in ModelRegistry.get_supported_archs():raise RuntimeError('Refusing to replace existing Nemotron4BNeuronTextV1 registration')
    ModelRegistry.register_model(arch,Nemotron4BNeuronTextV1ForCausalLM)
    _INSTALLED=True
