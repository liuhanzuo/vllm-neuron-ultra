# SPDX-License-Identifier: Apache-2.0
"""Experimental text-only Qwen3.5 adapter, using the independently checked port.

Retains the hybrid delta/full-attention architecture and explicit FP32 states.
Registration is process-local. This implementation is not yet model-qualified.
"""
import torch
from torch import nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, BaseParallelLinear,
)
from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from .text_core import Layer, RMSNorm, LinearAttention, store_slots
from .cache_writer import resolve_cache_writer


class Qwen35InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        nc = self.neuron_config
        for flag in ('quantized', 'sequence_parallel_enabled', 'flash_decoding_enabled',
                     'is_prefix_caching', 'is_block_kv_layout', 'enable_fused_speculation',
                     'enable_eagle_speculation', 'is_eagle_draft', 'output_logits'):
            if getattr(nc, flag, False):
                raise ValueError('Unvalidated Qwen3.5 route: ' + flag)
        if nc.cp_degree != 1 or nc.attention_dp_degree != 1 or nc.pp_degree != 1:
            raise ValueError('Only tensor parallelism is implemented')
        if nc.on_device_sampling_config is not None or nc.lora_config is not None:
            raise ValueError('Only CPU sampling, without LoRA, is implemented')
        if nc.padding_side != 'right' or nc.tp_degree not in (1, 2):
            raise ValueError('Initial adapter supports right padding and TP1/TP2')
        if self.attention_bias or not self.attn_output_gate or not self.tie_word_embeddings:
            raise ValueError('Unsupported Qwen3.5 projection/embedding configuration')
        if self.linear_num_key_heads != self.linear_num_value_heads or self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError('Strided QKV sharding currently requires equal Q/K/V widths')
        for key in ('num_attention_heads', 'num_key_value_heads', 'linear_num_key_heads',
                    'linear_num_value_heads', 'intermediate_size', 'vocab_size', 'hidden_size'):
            if getattr(self, key) % nc.tp_degree:
                raise ValueError('Non-divisible TP dimension: ' + key)
        if len(self.layer_types) != self.num_hidden_layers or not set(self.layer_types) <= {'linear_attention', 'full_attention'}:
            raise ValueError('Invalid hybrid layer layout')
        if self.rope_parameters['rope_type'] != 'default':
            raise ValueError('Only original default partial RoPE is implemented')

    def get_required_attributes(self):
        return ['hidden_size', 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                'num_hidden_layers', 'head_dim', 'pad_token_id', 'vocab_size', 'layer_types',
                'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
                'linear_value_head_dim', 'linear_conv_kernel_dim', 'rms_norm_eps', 'rope_parameters']

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronConfig


def partitioned_parameter(shape, stride=1, dtype=torch.float32, device=None):
    # NxD's custom meta initialization cannot reconstruct a CPU Parameter
    # with TP attributes passed as constructor kwargs. Use the device of an
    # already registered projection so registration keeps this object intact.
    p = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device), requires_grad=False)
    set_tensor_model_parallel_attributes(p, True, 0, stride,
                                        num_partitions=parallel_state.get_tensor_model_parallel_size())
    return p


class ShardedLinearAttention(LinearAttention, BaseParallelLinear):
    """BaseParallelLinear marks direct A_log/dt_bias parameters for NxD sharding.

    NxD's trace sharder intentionally ignores arbitrary nn.Module parameters,
    even when the tensor itself has partition attributes.
    """
    pass


class HybridCache(nn.Module):
    def __init__(self, c, nc):
        super().__init__()
        self.past_key_values = nn.ParameterList()
        for kind in c.layer_types:
            if kind == 'linear_attention':
                h = c.linear_num_value_heads // nc.tp_degree
                channels = 3 * h * c.linear_value_head_dim
                shapes = [(nc.max_batch_size, channels, c.linear_conv_kernel_dim),
                          (nc.max_batch_size, h, c.linear_key_head_dim, c.linear_value_head_dim)]
                dtypes = [nc.torch_dtype, torch.float32]
            else:
                shape = (nc.max_batch_size, c.num_key_value_heads // nc.tp_degree, nc.max_length, c.head_dim)
                shapes, dtypes = [shape, shape], [nc.torch_dtype, nc.torch_dtype]
            self.past_key_values.extend(nn.Parameter(torch.zeros(s, dtype=d), requires_grad=False)
                                        for s, d in zip(shapes, dtypes))


class NeuronQwen35Model(NeuronBaseModel):
    def setup_attr_for_model(self, c):
        self.on_device_sampling = False
        self.tp_degree = c.neuron_config.tp_degree
        self.hidden_size = c.hidden_size
        self.num_attention_heads, self.num_key_value_heads = c.num_attention_heads, c.num_key_value_heads
        self.max_batch_size, self.buckets = c.neuron_config.max_batch_size, c.neuron_config.buckets
        self.prefill = bool(c.neuron_config.is_prefill_stage)

    def init_model(self, c):
        if not parallel_state.model_parallel_is_initialized():
            raise RuntimeError('Initialize NxD model parallel groups before constructing Qwen3.5')
        tp, dtype = self.tp_degree, c.neuron_config.torch_dtype
        self.embed_tokens = ParallelEmbedding(c.vocab_size, c.hidden_size, dtype=dtype,
                                             shard_across_embedding=True)
        self.lm_head = ColumnParallelLinear(c.hidden_size, c.vocab_size, bias=False,
                                           gather_output=True, dtype=dtype)
        def col(a, b, stride=1):
            return ColumnParallelLinear(a, b, bias=False, gather_output=False, dtype=dtype, stride=stride)
        def row(a, b):
            return RowParallelLinear(a, b, bias=False, input_is_parallel=True, dtype=dtype)
        self.layers = nn.ModuleList()
        for kind in c.layer_types:
            layer = Layer(vars(c), kind)
            if kind == 'linear_attention':
                layer.linear_attn = ShardedLinearAttention(vars(c))
                a = layer.linear_attn
                key_width, value_width = a.key_dim, a.value_dim
                a.in_proj_qkv = col(c.hidden_size, 2 * key_width + value_width, stride=3)
                a.in_proj_z = col(c.hidden_size, value_width)
                a.in_proj_b = col(c.hidden_size, a.vh)
                a.in_proj_a = col(c.hidden_size, a.vh)
                a.out_proj = row(value_width, c.hidden_size)
                a.kh //= tp
                a.vh //= tp
                a.key_dim //= tp
                a.value_dim //= tp
                a.conv1d = BaseParallelLinear()
                parameter_device = a.in_proj_qkv.weight.device
                a.conv1d.weight = partitioned_parameter((2 * a.key_dim + a.value_dim, 1, a.width), 3, dtype, parameter_device)
                a.A_log = partitioned_parameter((a.vh,), device=parameter_device)
                a.dt_bias = partitioned_parameter((a.vh,), device=parameter_device)
            else:
                a = layer.self_attn
                a.q_proj = col(c.hidden_size, 2 * a.h * a.dim)
                a.k_proj = col(c.hidden_size, a.kvh * a.dim)
                a.v_proj = col(c.hidden_size, a.kvh * a.dim)
                a.o_proj = row(a.h * a.dim, c.hidden_size)
                a.h //= tp
                a.kvh //= tp
            layer.mlp.gate_proj = col(c.hidden_size, c.intermediate_size)
            layer.mlp.up_proj = col(c.hidden_size, c.intermediate_size)
            layer.mlp.down_proj = row(c.intermediate_size, c.hidden_size)
            self.layers.append(layer)
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        # Resolve once after TP sharding. Prefill remains the original writer.
        writer = resolve_cache_writer(c, self.prefill, store_slots)
        for layer in self.layers:
            attention = layer.linear_attn if layer.kind == 'linear_attention' else layer.self_attn
            attention.cache_writer = writer

    def init_inference_optimization(self, c):
        self.kv_mgr = HybridCache(c, c.neuron_config)

    def _apply(self, fn, recurse=True):
        # DecoderModelInstance applies a blanket BF16 cast. Preserve original FP32
        # values, not BF16-rounded values cast back up, across this operation.
        kept = []
        for i, layer in enumerate(self.layers):
            if layer.kind == 'linear_attention':
                a = layer.linear_attn
                kept.extend((a, name, getattr(a, name).detach()) for name in ('A_log', 'dt_bias'))
                if self.kv_mgr is not None:
                    kept.append((self.kv_mgr.past_key_values, str(2 * i + 1),
                                 self.kv_mgr.past_key_values[2 * i + 1].detach()))
            else:
                a = layer.self_attn
                kept.append((a, 'inverse_frequencies', a.inverse_frequencies))
        super()._apply(fn, recurse=recurse)
        for module, name, original in kept:
            current = getattr(module, name)
            preserved = original.to(device=current.device, dtype=torch.float32)
            if isinstance(current, nn.Parameter):
                current.data = preserved
            else:
                setattr(module, name, preserved)
        return self

    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params=None, *args, **kwargs):
        lengths = attention_mask.sum(-1).clamp_min(1) if self.prefill else position_ids[:, 0] + 1
        x, updated = self.embed_tokens(input_ids), []
        for i, layer in enumerate(self.layers):
            cache = self.kv_mgr.past_key_values[2 * i:2 * i + 2]
            # Static Neuron decode bucket bounds attention work. State aliases
            # keep their maximum allocation across buckets and requests.
            complete_cache = cache
            if not self.prefill and layer.kind == 'full_attention':
                cache = [p[:, :, :self.n_positions, :] for p in cache]
            x, cache = layer(x, cache, seq_ids, position_ids, lengths, self.prefill)
            if not self.prefill and layer.kind == 'full_attention':
                cache = [torch.cat((prefix, previous[:, :, self.n_positions:, :]), dim=2)
                         for prefix, previous in zip(cache, complete_cache)]
            updated.extend(cache)
        if self.prefill:
            index = (lengths - 1)[:, None, None].expand(x.shape[0], 1, x.shape[-1])
            x = torch.gather(x, 1, index)
        logits = self.lm_head(self.norm(x)).float()
        return [logits] + updated


class NeuronQwen35ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronQwen35Model
    _STATE_DICT_MODEL_PREFIX = 'model.language_model.'

    @classmethod
    def get_config_cls(cls):
        return Qwen35InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        result = {}
        for key, tensor in state_dict.items():
            if key.startswith(('model.visual.', 'mtp.')):
                continue
            if not (key.startswith('layers.') or key in ('embed_tokens.weight', 'norm.weight', 'lm_head.weight')):
                raise ValueError('Unexpected Qwen3.5 checkpoint tensor: ' + key)
            result[key] = tensor
        result['rank_util.rank'] = torch.arange(config.neuron_config.tp_degree, dtype=torch.int32)
        return result

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict['lm_head.weight'] = state_dict['embed_tokens.weight'].clone()

    def forward(self, input_ids=None, seq_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None:
            raise ValueError('Explicit positions are required for hybrid-state serving')
        # NxDI otherwise routes a one-token prompt to its decode graph.
        if input_ids.shape[1] == 1 and bool((position_ids == 0).all()):
            input_ids = F.pad(input_ids, (0, 1), value=self.config.pad_token_id)
            position_ids = F.pad(position_ids, (0, 1), value=0)
            attention_mask = torch.tensor([1, 0], dtype=position_ids.dtype).expand(input_ids.shape[0], 2)
        return super().forward(input_ids=input_ids, seq_ids=seq_ids, attention_mask=attention_mask,
                               position_ids=position_ids, **kwargs)


def install_nxdi():
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    name = 'qwen35neurontext'
    if name in MODEL_TYPES and MODEL_TYPES[name]['causal-lm'] is not NeuronQwen35ForCausalLM:
        raise RuntimeError('Refusing to replace an existing Qwen3.5 implementation')
    MODEL_TYPES[name] = {'causal-lm': NeuronQwen35ForCausalLM}
