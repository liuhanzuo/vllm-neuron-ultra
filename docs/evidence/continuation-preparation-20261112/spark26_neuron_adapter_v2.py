"""Source-only Spark SDK wiring successor. Import refuses before native imports.

Interfaces below follow the pinned SDK source. No native execution, weight load,
tracing, TP2 collectives or Neuron qualification has been performed for v2.
"""
from spark26_port_contract_v2 import require_runtime_admission
require_runtime_admission()

from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed_inference.modules.attention.gqa import GroupQueryAttention_QKV
from spark26_native_contract_v1 import config as official_config
from spark26_port_contract_v2 import capacity, checkpoint_schema, converted_key, validate_host_request
from spark26_text_port_v2 import SparkLayer, SparkRMSNorm


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
        target = {'tp_degree': 2, 'max_batch_size': 4, 'kv_cache_batch_size': 4,
                  'ctx_batch_size': 1, 'tkg_batch_size': 4,
                  'max_length': 4096, 'max_context_length': 4096, 'kv_cache_padding_size': 0}
        for key, value in target.items():
            if getattr(nc, key, None) != value:
                raise ValueError('Spark source target requires ' + key + '=' + str(value))
        if nc.padding_side != 'right' or nc.torch_dtype != torch.bfloat16:
            raise ValueError('Right padding and BF16 projections required')
        if not nc.apply_seq_ids_mask or not nc.fused_qkv:
            raise ValueError('apply_seq_ids_mask=True and fused_qkv=True required')
        for flag in ('quantized', 'sequence_parallel_enabled', 'flash_decoding_enabled',
                     'is_prefix_caching', 'is_block_kv_layout', 'enable_fused_speculation',
                     'enable_eagle_speculation', 'is_eagle_draft', 'is_medusa',
                     'qkv_kernel_enabled', 'qkv_nki_kernel_enabled', 'mlp_kernel_enabled',
                     'attn_kernel_enabled', 'attn_tkg_builtin_kernel_enabled',
                     'attn_tkg_nki_kernel_enabled', 'attn_block_tkg_nki_kernel_enabled',
                     'attn_block_cte_nki_kernel_enabled', 'qkv_cte_nki_kernel_fuse_rope',
                     'async_mode', 'output_logits', 'is_chunked_prefill',
                     'kv_cache_tiling', 'k_cache_transposed'):
            if getattr(nc, flag, False):
                raise ValueError('Unvalidated Spark route: ' + flag)
        for key in ('tensor_capture_config', 'tensor_replacement_config', 'lora_config',
                    'on_device_sampling_config', 'kv_quant_config', 'windowed_context_encoding_size',
                    'token_generation_batches'):
            if getattr(nc, key, None) is not None:
                raise ValueError('Unvalidated Spark configuration: ' + key)
        if any(getattr(nc, key) != 1 for key in ('cp_degree', 'attention_dp_degree', 'pp_degree')):
            raise ValueError('Only TP2 is described')
        if nc.speculation_length or nc.medusa_speculation_length:
            raise ValueError('Speculation is unsupported')

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
            shape = (4, 2, capacity(kind, nc.max_length), 256)
            for _ in range(2):
                self.past_key_values.append(nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16), requires_grad=False))


class NeuronSparkModel(NeuronBaseModel):
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
        self.embedding = ParallelEmbedding(131072, 2560, dtype=dtype, shard_across_embedding=True)
        self.lm_head = ColumnParallelLinear(2560, 131072, bias=False, gather_output=True, dtype=dtype)
        def col(i, o):
            return ColumnParallelLinear(i, o, bias=False, gather_output=False, dtype=dtype)
        def row(i, o):
            return RowParallelLinear(i, o, bias=False, input_is_parallel=True, dtype=dtype)
        self.layers = nn.ModuleList()
        for kind in cfg['layer_types']:
            layer = SparkLayer(cfg, kind)
            attention = layer.self_attn
            attention.qkv_proj = GroupQueryAttention_QKV(
                hidden_size=2560, head_dim=256, num_attention_heads=16,
                num_key_value_heads=4, tp_degree=2, dtype=dtype, bias=False,
                gather_output=False, fused_qkv=True, sequence_parallel_enabled=False,
                qkv_kernel_enabled=False, qkv_nki_kernel_enabled=False)
            attention.g_proj = col(2560, 16)
            attention.out_proj = row(4096, 2560)
            attention.num_heads = 8
            attention.num_kv_heads = 2
            layer.mlp.gate_proj = col(2560, 10240)
            layer.mlp.up_proj = col(2560, 10240)
            layer.mlp.down_proj = row(10240, 2560)
            self.layers.append(layer)
        self.norm = SparkRMSNorm(2560, cfg['rms_norm_eps'])

    def init_inference_optimization(self, config):
        self.kv_mgr = SparkCache(config)

    def forward(self, input_ids, attention_mask, position_ids, seq_ids,
                sampling_params, prev_hidden, adapter_ids):
        # SDK's fixed seven inputs are int32. Cast indices before gather/scatter.
        seq_ids = seq_ids.long()
        if self.prefill:
            cache_positions = torch.arange(input_ids.shape[1], device=input_ids.device,
                                           dtype=torch.long)[None, :].expand_as(input_ids)
            lengths = attention_mask.long().sum(-1)
        else:
            cache_positions = position_ids.long()
            lengths = cache_positions[:, 0] + 1
        leading_pad = torch.zeros_like(lengths)
        x = self.embedding(input_ids).float()
        updated = []
        for i, layer in enumerate(self.layers):
            old_k, old_v = self.kv_mgr.past_key_values[2 * i:2 * i + 2]
            x, k, v = layer(x, cache_positions, lengths, leading_pad,
                            seq_ids, old_k, old_v, self.prefill)
            updated.extend((k, v))
        if self.prefill:
            index = (lengths - 1)[:, None, None].expand(x.shape[0], 1, x.shape[-1])
            x = torch.gather(x, 1, index)
        x = self.norm(x).to(torch.bfloat16)
        # SDK DecoderModelInstance aliases these72 values to past_key_values in
        # identical K,V layer order when output_logits and capture are disabled.
        return [self.lm_head(x).float()] + updated


def convert_unprefixed_state_dict(state):
    expected = checkpoint_schema()
    missing = set(expected) - set(state)
    if missing not in (set(), {'lm_head.weight'}) or set(state) - set(expected):
        raise ValueError('Unexpected SDK-stripped original Spark checkpoint schema')
    for key, value in state.items():
        if list(value.shape) != expected[key] or value.dtype != torch.bfloat16:
            raise ValueError('Original shape/dtype mismatch: ' + key)
    embedding = state['embedding.weight']
    if 'lm_head.weight' in state and not torch.equal(state['lm_head.weight'], embedding):
        raise ValueError('Original tied embedding/head mismatch')
    result = {converted_key(key): value for key, value in state.items() if key != 'lm_head.weight'}
    # GQA preshard reads qkv_proj.Wqkv directly. Its fused_qkv parameter metadata
    # drives SDK create_local_weight_qkv: independent Q/K/V splits, then rank concat.
    result['rank_util.rank'] = torch.arange(2, dtype=torch.int32)
    return result


class NeuronSparkForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronSparkModel
    _STATE_DICT_MODEL_PREFIX = 'model.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spark_seen = {}

    @classmethod
    def get_config_cls(cls):
        return SparkInferenceConfig

    @classmethod
    def get_state_dict(cls, model_name_or_path, config):
        if not Path(model_name_or_path).is_dir():
            raise ValueError('Use the pinned original local checkpoint directory')
        return super().get_state_dict(model_name_or_path, config)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        return convert_unprefixed_state_dict(state_dict)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        # Separate physical TP partitions retain identical original head values.
        state_dict['lm_head.weight'] = state_dict['embedding.weight'].clone()

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        raise ValueError('Remote/native reference loading is a separate pinned workflow; use local get_state_dict')

    def forward(self, input_ids=None, seq_ids=None, attention_mask=None,
                position_ids=None, sampling_params=None, **kwargs):
        for name, tensor in (('input_ids', input_ids), ('seq_ids', seq_ids), ('position_ids', position_ids)):
            if tensor is None or tensor.device.type != 'cpu' or tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError('CPU integer ' + name + ' is required')
        if attention_mask is not None and (attention_mask.device.type != 'cpu' or
                attention_mask.dtype not in (torch.bool, torch.int32, torch.int64)):
            raise ValueError('Attention mask must be a CPU boolean/integer tensor')
        info = validate_host_request(input_ids.tolist(), position_ids.tolist(), seq_ids.tolist(),
                                     None if attention_mask is None else attention_mask.int().tolist())
        slots = seq_ids.tolist()
        if not info['prefill']:
            for slot, row in zip(slots, position_ids.tolist()):
                if self._spark_seen.get(slot) != row[0]:
                    raise ValueError('Decode without matching prefill/history for slot ' + str(slot))
        else:
            attention_mask = torch.tensor(info['attention_rows'], dtype=torch.int32)
            if info['pad_single_prompt']:
                input_ids = F.pad(input_ids, (0, 1), value=self.config.pad_token_id)
                position_ids = F.pad(position_ids, (0, 1), value=0)
                attention_mask = F.pad(attention_mask, (0, 1), value=0)
        if sampling_params is None:
            sampling_params = self.default_sampling_params[:len(slots)]
        output = super().forward(input_ids=input_ids, seq_ids=seq_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, sampling_params=sampling_params, **kwargs)
        for slot, length in zip(slots, info['lengths']):
            self._spark_seen[slot] = length
        return output

    def reset(self):
        super().reset()
        self._spark_seen.clear()


def install_nxdi():
    require_runtime_admission()
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    if 'spark2_5' in MODEL_TYPES:
        raise RuntimeError('Refuse existing Spark registration replacement')
    MODEL_TYPES['spark2_5'] = {'causal-lm': NeuronSparkForCausalLM}
