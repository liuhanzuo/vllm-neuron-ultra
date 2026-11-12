"""Unexecuted SDK adapter source. Import is intentionally closed before native imports.

The recipe below specifies TP2 projection and cache output structure. No loader,
trace, cache alias, or tensor-parallel ABI is certified by the stdlib checks.
"""
from spark26_port_contract_v1 import require_runtime_qualification
require_runtime_qualification()

import torch
from torch import nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from spark26_native_contract_v1 import config as official_config, state_schema
from spark26_port_contract_v1 import capacity
from spark26_text_port_v1 import SparkLayer, SparkRMSNorm


class SparkInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        expected = official_config()
        keys = ('hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                'num_hidden_layers', 'head_dim', 'vocab_size', 'hidden_act', 'layer_types',
                'rope_parameters', 'rms_norm_eps', 'sliding_window', 'tie_word_embeddings',
                'attention_bias', 'mlp_bias', 'headwise_attn_output_gate', 'gate_attn_act_mode')
        for key in keys:
            if getattr(self, key, None) != expected[key]:
                raise ValueError('Original Spark config changed: ' + key)
        self.num_cores_per_group = 1
        nc = self.neuron_config
        if nc.tp_degree != 2 or nc.max_batch_size != 4 or nc.max_length != 4096:
            raise ValueError('Source target requires TP2, B4 and max_length4096')
        if nc.padding_side != 'right' or nc.torch_dtype != torch.bfloat16:
            raise ValueError('Source target requires right padding and BF16 projections')
        for flag in ('quantized', 'sequence_parallel_enabled', 'flash_decoding_enabled',
                     'is_prefix_caching', 'is_block_kv_layout', 'enable_fused_speculation',
                     'enable_eagle_speculation', 'is_eagle_draft', 'is_medusa',
                     'qkv_kernel_enabled', 'qkv_nki_kernel_enabled', 'mlp_kernel_enabled',
                     'attn_kernel_enabled', 'attn_tkg_builtin_kernel_enabled',
                     'attn_tkg_nki_kernel_enabled', 'attn_block_tkg_nki_kernel_enabled',
                     'attn_block_cte_nki_kernel_enabled', 'qkv_cte_nki_kernel_fuse_rope'):
            if getattr(nc, flag, False):
                raise ValueError('Unvalidated Spark route: ' + flag)
        if any(getattr(nc, key) != 1 for key in ('cp_degree', 'attention_dp_degree', 'pp_degree')):
            raise ValueError('Only TP2 is described')
        if nc.on_device_sampling_config is not None or nc.lora_config is not None:
            raise ValueError('Sampling and LoRA route unvalidated')
        if nc.speculation_length or nc.medusa_speculation_length:
            raise ValueError('Speculation unvalidated')

    def get_required_attributes(self):
        return ['hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                'num_hidden_layers', 'head_dim', 'vocab_size', 'pad_token_id', 'rms_norm_eps',
                'layer_types', 'sliding_window', 'rope_parameters', 'hidden_act']

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronConfig


class SparkCache(nn.Module):
    def __init__(self, config):
        super().__init__()
        nc = config.neuron_config
        self.past_key_values = nn.ParameterList()
        for kind in config.layer_types:
            shape = (nc.max_batch_size, config.num_key_value_heads // nc.tp_degree,
                     capacity(kind, nc.max_length), config.head_dim)
            for _ in range(2):
                self.past_key_values.append(nn.Parameter(torch.zeros(shape, dtype=nc.torch_dtype), requires_grad=False))


class NeuronSparkModel(NeuronBaseModel):
    def __init__(self, *args, **kwargs):
        require_runtime_qualification()
        super().__init__(*args, **kwargs)

    def setup_attr_for_model(self, config):
        self.on_device_sampling = False
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets
        self.prefill = bool(config.neuron_config.is_prefill_stage)

    def init_model(self, config):
        if not parallel_state.model_parallel_is_initialized():
            raise RuntimeError('TP groups must be initialized by the admitted SDK route')
        cfg = official_config(); dtype = torch.bfloat16
        self.embedding = ParallelEmbedding(cfg['vocab_size'], cfg['hidden_size'],
                                           dtype=dtype, shard_across_embedding=True)
        self.lm_head = ColumnParallelLinear(cfg['hidden_size'], cfg['vocab_size'],
                                            bias=False, gather_output=True, dtype=dtype)
        def col(i, o):
            return ColumnParallelLinear(i, o, bias=False, gather_output=False, dtype=dtype)
        def row(i, o):
            return RowParallelLinear(i, o, bias=False, input_is_parallel=True, dtype=dtype)
        self.layers = nn.ModuleList()
        for kind in cfg['layer_types']:
            layer = SparkLayer(cfg, kind)
            attention = layer.self_attn
            attention.q_proj = col(2560, 4096)
            attention.k_proj = col(2560, 1024)
            attention.v_proj = col(2560, 1024)
            attention.g_proj = col(2560, 16)
            attention.out_proj = row(4096, 2560)
            attention.num_heads //= self.tp_degree
            attention.num_kv_heads //= self.tp_degree
            layer.mlp.gate_proj = col(2560, 10240)
            layer.mlp.up_proj = col(2560, 10240)
            layer.mlp.down_proj = row(10240, 2560)
            self.layers.append(layer)
        self.norm = SparkRMSNorm(2560, cfg['rms_norm_eps'])

    def init_inference_optimization(self, config):
        self.kv_mgr = SparkCache(config)

    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params=None, *args, **kwargs):
        require_runtime_qualification()
        # Explicit cache_positions are deliberately not silently replaced with
        # vLLM position_ids. Their equivalence must be proved for the live ABI.
        cache_positions = kwargs['spark_cache_positions']
        lengths = kwargs['spark_valid_lengths']
        leading_pad = kwargs['spark_leading_pad']
        return self._forward_candidate(input_ids, seq_ids, cache_positions, lengths, leading_pad)

    def _forward_candidate(self, input_ids, seq_ids, cache_positions, lengths, leading_pad):
        x = self.embedding(input_ids).float()
        updated = []
        for i, layer in enumerate(self.layers):
            old_k, old_v = self.kv_mgr.past_key_values[2 * i:2 * i + 2]
            x, k, v = layer(x, cache_positions, lengths, leading_pad, seq_ids, old_k, old_v, self.prefill)
            updated.extend((k, v))
        if self.prefill:
            index = (lengths - 1)[:, None, None].expand(x.shape[0], 1, x.shape[-1])
            x = torch.gather(x, 1, index)
        x = self.norm(x).to(torch.bfloat16)
        return [self.lm_head(x).float()] + updated


def convert_official_state_dict(state):
    """Original HF keys -> unprefixed candidate keys; SDK prefix ABI is unverified."""
    expected = state_schema()
    missing = set(expected) - set(state)
    if missing not in (set(), {'lm_head.weight'}) or set(state) - set(expected):
        raise ValueError('Unexpected original Spark checkpoint schema')
    for key, value in state.items():
        if list(value.shape) != expected[key] or value.dtype != torch.bfloat16:
            raise ValueError('Original shape/dtype mismatch: ' + key)
    embedding = state['model.embedding.weight']
    if 'lm_head.weight' in state and not torch.equal(state['lm_head.weight'], embedding):
        raise ValueError('Tied embedding/head mismatch')
    result = {}
    for key, value in state.items():
        if key == 'lm_head.weight':
            continue
        key = key.removeprefix('model.')
        if key.endswith('self_attn.q_k_v_proj.weight'):
            prefix = key[:-len('q_k_v_proj.weight')]
            for name, part in zip(('q_proj', 'k_proj', 'v_proj'), value.split((4096, 1024, 1024), dim=0)):
                result[prefix + name + '.weight'] = part
        else:
            result[key] = value
    result['lm_head.weight'] = embedding
    result['rank_util.rank'] = torch.arange(2, dtype=torch.int32)
    return result


class NeuronSparkForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronSparkModel
    _STATE_DICT_MODEL_PREFIX = 'model.'

    @classmethod
    def get_config_cls(cls):
        return SparkInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        # No forwarding until actual SDK prefix stripping/tie loading is checked.
        require_runtime_qualification()

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        require_runtime_qualification()


def install_nxdi():
    require_runtime_qualification()
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    if 'spark2_5' in MODEL_TYPES:
        raise RuntimeError('Refuse existing Spark registration replacement')
    MODEL_TYPES['spark2_5'] = {'causal-lm': NeuronSparkForCausalLM}
