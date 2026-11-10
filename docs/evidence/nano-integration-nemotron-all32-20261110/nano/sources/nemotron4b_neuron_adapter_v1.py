"""Original BF16 Nemotron4B TP2 baseline; runtime validation remains mandatory."""
import torch
import torch.nn.functional as F
from torch import nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear, BaseParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes
from nemotron4b_baseline_contract_v1 import validate_config, original_inventory, learned_entries, NXDI_KEY
from nemotron4b_text_port_v1 import Layer, Mamba, GatedNorm, RMSNorm


def partitioned_parameter(shape, stride=1, dtype=torch.float32, device=None):
    # NxD's custom meta initialization cannot reconstruct a CPU Parameter
    # with TP attributes passed as constructor kwargs. Use the device of an
    # already registered projection so registration keeps this object intact.
    p = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device), requires_grad=False)
    set_tensor_model_parallel_attributes(p, True, 0, stride,
                                        num_partitions=parallel_state.get_tensor_model_parallel_size())
    return p


class Nemotron4BV1InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        validate_config(self)
        self.num_cores_per_group = 1
        nc = self.neuron_config
        if nc.tp_degree != 2 or nc.torch_dtype != torch.bfloat16:
            raise ValueError('Original 4B baseline requires BF16/TP2')
        if nc.max_batch_size != 4 or nc.max_length != 4096:
            raise ValueError('Initial baseline requires max_batch4/context4096')
        if not nc.apply_seq_ids_mask:
            raise ValueError('Inactive-slot seq_ids mask is required')
        for flag in ('quantized', 'sequence_parallel_enabled', 'flash_decoding_enabled', 'is_prefix_caching',
                     'is_block_kv_layout', 'enable_fused_speculation', 'enable_eagle_speculation', 'is_eagle_draft', 'output_logits'):
            if getattr(nc, flag, False): raise ValueError('Unvalidated Nemotron route: '+flag)
        if nc.cp_degree != 1 or nc.attention_dp_degree != 1 or nc.pp_degree != 1: raise ValueError('Only TP is implemented')
        if nc.on_device_sampling_config is not None or nc.lora_config is not None: raise ValueError('Only CPU sampling without LoRA')
        if nc.padding_side != 'right': raise ValueError('Only right padding')
        if nc.speculation_length or nc.medusa_speculation_length: raise ValueError('Speculation not implemented')
        if len(self.hybrid_override_pattern) != self.num_hidden_layers or not set(self.hybrid_override_pattern) <= set('M*-'): raise ValueError('Invalid dense layer pattern')
        for flag in ('tie_word_embeddings', 'residual_in_fp32', 'attention_bias', 'mlp_bias', 'use_bias', 'mamba_proj_bias'):
            if getattr(self, flag, False): raise ValueError('Unvalidated setting: '+flag)
        if self.mlp_hidden_act != 'relu2' or self.mamba_hidden_act != 'silu': raise ValueError('Unexpected activation')
        if getattr(self, 'sliding_window', None) is not None or getattr(self, 'num_nextn_predict_layers', 0): raise ValueError('Only original dense NoPE attention')
        if tuple(getattr(self, 'time_step_limit', (0., float('inf')))) != (0., float('inf')): raise ValueError('Unexpected runtime dt clamp')
        for key in ('hidden_size', 'intermediate_size', 'vocab_size', 'num_attention_heads', 'num_key_value_heads', 'mamba_num_heads', 'n_groups'):
            if getattr(self, key) % nc.tp_degree: raise ValueError('Nondivisible TP dimension: '+key)
        if self.mamba_num_heads % self.n_groups or self.num_attention_heads % self.num_key_value_heads: raise ValueError('Invalid head grouping')

    def get_required_attributes(self):
        return ['hidden_size', 'intermediate_size', 'vocab_size', 'num_hidden_layers', 'hybrid_override_pattern', 'pad_token_id',
                'num_attention_heads', 'num_key_value_heads', 'head_dim', 'mamba_num_heads', 'mamba_head_dim', 'n_groups',
                'ssm_state_size', 'conv_kernel', 'chunk_size', 'use_conv_bias', 'layer_norm_epsilon', 'mlp_hidden_act', 'mamba_hidden_act']

    @classmethod
    def get_neuron_config_cls(cls): return NeuronConfig


class ShardedMamba(Mamba, BaseParallelLinear):
    """Direct A_log/D/dt_bias must participate in NxD checkpoint sharding."""
    pass


class ShardedGatedNorm(GatedNorm, BaseParallelLinear):
    pass


class PackedInput(nn.Module):
    def __init__(self, hidden, sizes, dtype):
        super().__init__()
        for name, width in zip(('gate', 'x', 'B', 'C', 'dt'), sizes):
            setattr(self, name, ColumnParallelLinear(hidden, width, bias=False, gather_output=False, dtype=dtype))

    def forward(self, x):
        return torch.cat([getattr(self, name)(x) for name in ('gate', 'x', 'B', 'C', 'dt')], dim=-1)


class PackedConv(nn.Module):
    def __init__(self, sizes, kernel, use_bias, tp, dtype, device):
        super().__init__()
        self.use_bias = use_bias
        for name, width in zip(('x', 'B', 'C'), sizes):
            part = BaseParallelLinear()
            part.weight = partitioned_parameter((width//tp, 1, kernel), dtype=dtype, device=device)
            if use_bias: part.bias = partitioned_parameter((width//tp,), dtype=dtype, device=device)
            setattr(self, name, part)

    @property
    def weight(self): return torch.cat([getattr(self, name).weight for name in ('x', 'B', 'C')], dim=0)

    @property
    def bias(self):
        return torch.cat([getattr(self, name).bias for name in ('x', 'B', 'C')], dim=0) if self.use_bias else None


class Nemotron4BV1Cache(nn.Module):
    def __init__(self, c, nc):
        super().__init__()
        self.past_key_values, self.offsets = nn.ParameterList(), []
        for kind in c.hybrid_override_pattern:
            start = len(self.past_key_values)
            if kind == 'M':
                inner = c.mamba_num_heads*c.mamba_head_dim
                conv = inner+2*c.n_groups*c.ssm_state_size
                shapes = [(nc.max_batch_size, conv//nc.tp_degree, c.conv_kernel),
                          (nc.max_batch_size, c.mamba_num_heads//nc.tp_degree, c.mamba_head_dim, c.ssm_state_size)]
                dtypes = [nc.torch_dtype, torch.float32]
            elif kind == '*':
                shape = (nc.max_batch_size, c.num_key_value_heads//nc.tp_degree, nc.max_length, c.head_dim)
                shapes, dtypes = [shape, shape], [nc.torch_dtype, nc.torch_dtype]
            else: shapes, dtypes = [], []
            self.past_key_values.extend(nn.Parameter(torch.zeros(s, dtype=d), requires_grad=False) for s, d in zip(shapes, dtypes))
            self.offsets.append((start, len(self.past_key_values)))


class NeuronNemotron4BV1Model(NeuronBaseModel):
    def setup_attr_for_model(self, c):
        self.on_device_sampling = False
        self.tp_degree, self.hidden_size = c.neuron_config.tp_degree, c.hidden_size
        self.num_attention_heads, self.num_key_value_heads = c.num_attention_heads, c.num_key_value_heads
        self.max_batch_size, self.buckets = c.neuron_config.max_batch_size, c.neuron_config.buckets
        self.prefill = bool(c.neuron_config.is_prefill_stage)

    def init_model(self, c):
        if not parallel_state.model_parallel_is_initialized(): raise RuntimeError('Initialize NxD TP groups first')
        tp, dtype, hidden = self.tp_degree, c.neuron_config.torch_dtype, c.hidden_size
        self.embeddings = ParallelEmbedding(c.vocab_size, hidden, dtype=dtype, shard_across_embedding=True)
        self.lm_head = ColumnParallelLinear(hidden, c.vocab_size, bias=False, gather_output=True, dtype=dtype)
        def col(a, b): return ColumnParallelLinear(a, b, bias=False, gather_output=False, dtype=dtype)
        def row(a, b): return RowParallelLinear(a, b, bias=False, input_is_parallel=True, dtype=dtype)
        self.layers = nn.ModuleList()
        for kind in c.hybrid_override_pattern:
            layer = Layer(vars(c), kind)
            if kind == 'M':
                layer.mixer = ShardedMamba(vars(c))
                a = layer.mixer
                inner, gn, heads = a.inner, a.groups*a.sd, a.heads
                a.in_proj = PackedInput(hidden, [inner, inner, gn, gn, heads], dtype)
                device = a.in_proj.gate.weight.device
                a.conv1d = PackedConv([inner, gn, gn], c.conv_kernel, c.use_conv_bias, tp, dtype, device)
                a.out_proj = row(inner, hidden)
                a.norm = ShardedGatedNorm(inner//tp, a.groups//tp, c.layer_norm_epsilon)
                a.norm.weight = partitioned_parameter((inner//tp,), dtype=dtype, device=device)
                for name in ('A_log', 'D', 'dt_bias'):
                    setattr(a, name, partitioned_parameter((heads//tp,), dtype=dtype, device=device))
                a.heads //= tp
                a.groups //= tp
                a.inner //= tp
                a.conv_dim //= tp
            elif kind == '*':
                a = layer.mixer
                a.q_proj, a.k_proj, a.v_proj = col(hidden, a.heads*a.hd), col(hidden, a.kvh*a.hd), col(hidden, a.kvh*a.hd)
                a.o_proj = row(a.heads*a.hd, hidden)
                a.heads //= tp
                a.kvh //= tp
            else:
                layer.mixer.up_proj = col(hidden, c.intermediate_size)
                layer.mixer.down_proj = row(c.intermediate_size, hidden)
            self.layers.append(layer)
        self.norm_f = RMSNorm(hidden, c.layer_norm_epsilon)

    def init_inference_optimization(self, c): self.kv_mgr = Nemotron4BV1Cache(c, c.neuron_config)

    def _apply(self, fn, recurse=True):
        kept = []
        if getattr(self, 'kv_mgr', None) is not None:
            for layer, (start, end) in zip(self.layers, self.kv_mgr.offsets):
                if layer.kind == 'M': kept.append((start+1, self.kv_mgr.past_key_values[start+1].detach()))
        super()._apply(fn, recurse=recurse)
        for index, original in kept:
            current = self.kv_mgr.past_key_values[index]
            current.data = original.to(device=current.device, dtype=torch.float32)
        return self

    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params=None, *args, **kwargs):
        lengths = attention_mask.sum(-1).clamp_min(1) if self.prefill else position_ids[:, 0]+1
        x, updated = self.embeddings(input_ids), []
        for layer, (start, end) in zip(self.layers, self.kv_mgr.offsets):
            full = self.kv_mgr.past_key_values[start:end]
            cache = [p[:, :, :self.n_positions] for p in full] if not self.prefill and layer.kind == '*' else full
            x, cache = layer(x, cache, seq_ids, position_ids, lengths, self.prefill)
            if not self.prefill and layer.kind == '*': cache = [torch.cat([new, old[:, :, self.n_positions:]], dim=2) for new, old in zip(cache, full)]
            updated.extend(cache)
        if self.prefill: x = x.gather(1, (lengths-1)[:, None, None].expand(x.shape[0], 1, x.shape[-1]))
        return [self.lm_head(self.norm_f(x)).float()]+updated


class NeuronNemotron4BV1ForCausalLM(NeuronBaseForCausalLM):
    _model_cls = NeuronNemotron4BV1Model
    _STATE_DICT_MODEL_PREFIX = 'backbone.'

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

    @classmethod
    def get_config_cls(cls): return Nemotron4BV1InferenceConfig

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        raise ValueError('This checkpoint has independent embeddings and lm_head')

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        validate_config(config)
        inventory = original_inventory()
        normalized = {}
        for raw_key, value in state_dict.items():
            local = raw_key.removeprefix('backbone.').removeprefix('model.')
            official = local if local == 'lm_head.weight' else 'backbone.'+local
            if official in normalized: raise ValueError('Duplicate original key: '+official)
            if official not in inventory or list(value.shape) != inventory[official] or value.dtype != torch.bfloat16:
                raise ValueError('Unexpected original tensor shape/dtype: '+raw_key)
            normalized[official] = value
        if set(normalized) != set(inventory): raise ValueError('Incomplete original263 checkpoint')
        inner, gn = config.mamba_num_heads*config.mamba_head_dim, config.n_groups*config.ssm_state_size
        result = {}
        for raw_key, tensor in state_dict.items():
            key = raw_key.removeprefix('backbone.').removeprefix('model.')
            if not (key.startswith('layers.') or key in ('embeddings.weight', 'norm_f.weight', 'lm_head.weight')):
                raise ValueError('Unexpected Nemotron tensor: '+raw_key)
            if key.endswith('.mixer.in_proj.weight'):
                suffixes = ['gate.weight', 'x.weight', 'B.weight', 'C.weight', 'dt.weight']
                parts = tensor.split([inner, inner, gn, gn, config.mamba_num_heads], dim=0)
                entries = [(key.removesuffix('weight')+name, part) for name, part in zip(suffixes, parts)]
            elif '.mixer.conv1d.' in key:
                stem, param = key.rsplit('.', 1)
                entries = [(stem+'.'+name+'.'+param, part) for name, part in zip(('x', 'B', 'C'), tensor.split([inner, gn, gn], dim=0))]
            else: entries = [(key, tensor)]
            for name, part in entries:
                if name in result: raise ValueError('Duplicate normalized Nemotron tensor: '+name)
                result[name] = part
        if set(result) != set(learned_entries()): raise ValueError('Incorrect431 converted tensor keys')
        for name,spec in learned_entries().items():
            if list(result[name].shape) != spec['shape']: raise ValueError('Incorrect converted shape: '+name)
        result['rank_util.rank'] = torch.arange(config.neuron_config.tp_degree, dtype=torch.int32)
        return result


def install_nxdi():
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    key = NXDI_KEY
    if key in MODEL_TYPES and MODEL_TYPES[key]['causal-lm'] is not NeuronNemotron4BV1ForCausalLM:
        raise RuntimeError('Refusing to replace an existing Nemotron implementation')
    MODEL_TYPES[key] = {'causal-lm': NeuronNemotron4BV1ForCausalLM}
