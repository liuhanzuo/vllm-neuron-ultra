"""Source registration wiring, still closed before any native import or binding."""
from spark26_port_contract_v2 import require_runtime_admission


def install():
    require_runtime_admission()
    from transformers import AutoConfig, PretrainedConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
    from spark26_neuron_adapter_v2 import install_nxdi

    class SparkConfig(PretrainedConfig):
        model_type = 'spark2_5'

    class Spark2_5ForCausalLM(NeuronCausalLM):
        def __init__(self, *, vllm_config, prefix=''):
            super().__init__(vllm_config.model_config.hf_config)

        def forward(self, input_ids, positions, input_block_ids, **kwargs):
            # The pinned NeuronCausalLM loader performs slot sorting and restores
            # output rows; all position rows travel through its same reorder.
            return super().forward(input_ids, input_block_ids, position_ids=positions, **kwargs)

        def compute_logits(self, hidden_states):
            return hidden_states

    if 'Spark2_5ForCausalLM' in ModelRegistry.get_supported_archs():
        raise RuntimeError('Refuse existing Spark registration replacement')
    install_nxdi()
    AutoConfig.register('spark2_5', SparkConfig)
    ModelRegistry.register_model('Spark2_5ForCausalLM', Spark2_5ForCausalLM)
