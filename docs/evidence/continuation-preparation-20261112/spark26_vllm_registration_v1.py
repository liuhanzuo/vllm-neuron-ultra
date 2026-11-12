"""Source-only registration recipe; install fails before importing native modules."""
from spark26_port_contract_v1 import require_runtime_qualification


def install():
    require_runtime_qualification()
    from transformers import AutoConfig, PretrainedConfig
    from vllm.model_executor.models.registry import ModelRegistry
    from vllm_neuron.worker.neuronx_distributed_model_loader import NeuronCausalLM
    from spark26_neuron_adapter_v1 import install_nxdi

    class SparkConfig(PretrainedConfig):
        model_type = 'spark2_5'

    class Spark2_5ForCausalLM(NeuronCausalLM):
        def __init__(self, *, vllm_config, prefix=''):
            require_runtime_qualification()
            super().__init__(vllm_config.model_config.hf_config)

        def forward(self, input_ids, positions, input_block_ids, **kwargs):
            # Neither heterogeneous cache positions nor slot reuse is silently
            # mapped onto the old generic loader before its actual ABI is tested.
            require_runtime_qualification()

        def compute_logits(self, hidden_states):
            return hidden_states

    if 'Spark2_5ForCausalLM' in ModelRegistry.get_supported_archs():
        raise RuntimeError('Refuse existing Spark registration replacement')
    install_nxdi()
    AutoConfig.register('spark2_5', SparkConfig)
    ModelRegistry.register_model('Spark2_5ForCausalLM', Spark2_5ForCausalLM)
