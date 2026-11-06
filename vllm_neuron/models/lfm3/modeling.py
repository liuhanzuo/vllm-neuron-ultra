# SPDX-License-Identifier: Apache-2.0
"""Experimental LFM3 TP adapter, retaining the independently checked CPU core."""
import torch
from torch import nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding,ColumnParallelLinear,RowParallelLinear,BaseParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig,NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from .helpers import partitioned_parameter
from .identity import validate_scope
from .text_core import Layer,RMSNorm


class Lfm3CausalLMBase(NeuronBaseForCausalLM):
    """Only tied weights and one-token prefill routing are shared."""

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

class Lfm3InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group=1;nc=self.neuron_config
        validate_scope(vars(self),vars(nc))
        for flag in ('quantized','sequence_parallel_enabled','flash_decoding_enabled','is_prefix_caching',
                     'is_block_kv_layout','enable_fused_speculation','enable_eagle_speculation','is_eagle_draft','output_logits'):
            if getattr(nc,flag,False):raise ValueError('Unvalidated LFM3 route: '+flag)
        if nc.cp_degree!=1 or nc.attention_dp_degree!=1 or nc.pp_degree!=1:
            raise ValueError('Only tensor parallelism is implemented')
        if nc.on_device_sampling_config is not None or nc.lora_config is not None:
            raise ValueError('Only CPU sampling without LoRA is implemented')
        if nc.padding_side!='right' or nc.tp_degree not in (1,2):
            raise ValueError('Packaged LFM3 admission requires right padding and TP2')
        if nc.speculation_length or nc.medusa_speculation_length:
            raise ValueError('Speculative decoding is not implemented')
        if self.conv_bias or not self.use_sandwich_norm or not self.tie_word_embeddings:
            raise ValueError('Unsupported LFM3 configuration')
        for key in ('num_attention_heads','num_key_value_heads','intermediate_size','hidden_size','vocab_size'):
            if getattr(self,key)%nc.tp_degree:raise ValueError('Non-divisible dimension: '+key)
        if len(self.layer_types)!=self.num_hidden_layers or not set(self.layer_types)<={'conv','full_attention','sliding_attention'}:
            raise ValueError('Invalid layer layout')

    def get_required_attributes(self):
        return ['hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','num_hidden_layers',
                'head_dim','pad_token_id','vocab_size','layer_types','conv_L_cache','norm_eps','rope_parameters','sliding_window']

    @classmethod
    def get_neuron_config_cls(cls):return NeuronConfig


class Lfm3Cache(nn.Module):
    def __init__(self,c,nc):
        super().__init__();self.past_key_values=nn.ParameterList();self.offsets=[]
        for kind in c.layer_types:
            start=len(self.past_key_values)
            if kind=='conv':shapes=[(nc.max_batch_size,c.hidden_size//nc.tp_degree,c.conv_L_cache)]
            else:shapes=[(nc.max_batch_size,c.num_key_value_heads//nc.tp_degree,nc.max_length,c.head_dim)]*2
            self.past_key_values.extend(nn.Parameter(torch.zeros(s,dtype=nc.torch_dtype),requires_grad=False) for s in shapes)
            self.offsets.append((start,len(self.past_key_values)))


class NeuronLfm3Model(NeuronBaseModel):
    def setup_attr_for_model(self,c):
        self.on_device_sampling=False;self.tp_degree=c.neuron_config.tp_degree;self.hidden_size=c.hidden_size
        self.num_attention_heads,self.num_key_value_heads=c.num_attention_heads,c.num_key_value_heads
        self.max_batch_size,self.buckets=c.neuron_config.max_batch_size,c.neuron_config.buckets
        self.prefill=bool(c.neuron_config.is_prefill_stage)

    def init_model(self,c):
        if not parallel_state.model_parallel_is_initialized():raise RuntimeError('Initialize NxD TP groups first')
        tp,dtype=self.tp_degree,c.neuron_config.torch_dtype
        self.embed_tokens=ParallelEmbedding(c.vocab_size,c.hidden_size,dtype=dtype,shard_across_embedding=True)
        self.lm_head=ColumnParallelLinear(c.hidden_size,c.vocab_size,bias=False,gather_output=True,dtype=dtype)
        def col(a,b,stride=1):return ColumnParallelLinear(a,b,bias=False,gather_output=False,dtype=dtype,stride=stride)
        def row(a,b):return RowParallelLinear(a,b,bias=False,input_is_parallel=True,dtype=dtype)
        self.layers=nn.ModuleList()
        for kind in c.layer_types:
            layer=Layer(vars(c),kind)
            if kind=='conv':
                a=layer.conv
                a.in_proj=col(c.hidden_size,3*c.hidden_size,stride=3)
                a.out_proj=row(c.hidden_size,c.hidden_size)
                a.conv=BaseParallelLinear()
                a.conv.weight=partitioned_parameter((c.hidden_size//tp,1,c.conv_L_cache),dtype=dtype,
                                                    device=a.in_proj.weight.device)
            else:
                a=layer.self_attn
                a.q_proj=col(c.hidden_size,a.h*a.dim)
                a.k_proj=col(c.hidden_size,a.kvh*a.dim)
                a.v_proj=col(c.hidden_size,a.kvh*a.dim)
                a.gate_proj=col(c.hidden_size,a.h)
                a.o_proj=row(a.h*a.dim,c.hidden_size)
                a.h//=tp;a.kvh//=tp
            layer.feed_forward.w1=col(c.hidden_size,c.intermediate_size)
            layer.feed_forward.w3=col(c.hidden_size,c.intermediate_size)
            layer.feed_forward.w2=row(c.intermediate_size,c.hidden_size)
            self.layers.append(layer)
        self.embedding_norm=RMSNorm(c.hidden_size,c.norm_eps)

    def init_inference_optimization(self,c):self.kv_mgr=Lfm3Cache(c,c.neuron_config)

    def _apply(self,fn,recurse=True):
        kept=[(l.self_attn,l.self_attn.inverse_frequencies) for l in self.layers if l.kind!='conv']
        super()._apply(fn,recurse=recurse)
        for a,original in kept:
            a.inverse_frequencies=original.to(device=a.inverse_frequencies.device,dtype=torch.float32)
        return self

    def forward(self,input_ids,attention_mask,position_ids,seq_ids,sampling_params=None,*args,**kwargs):
        lengths=attention_mask.sum(-1).clamp_min(1) if self.prefill else position_ids[:,0]+1
        x=self.embed_tokens(input_ids);updated=[]
        for layer,(start,end) in zip(self.layers,self.kv_mgr.offsets):
            full_cache=self.kv_mgr.past_key_values[start:end];cache=full_cache
            if not self.prefill and layer.kind!='conv':cache=[p[:,:,:self.n_positions,:] for p in cache]
            x,cache=layer(x,cache,seq_ids,position_ids,lengths,self.prefill)
            if not self.prefill and layer.kind!='conv':
                cache=[torch.cat((prefix,old[:,:,self.n_positions:,:]),2) for prefix,old in zip(cache,full_cache)]
            updated.extend(cache)
        if self.prefill:
            index=(lengths-1)[:,None,None].expand(x.shape[0],1,x.shape[-1]);x=torch.gather(x,1,index)
        return [self.lm_head(self.embedding_norm(x)).float()]+updated


class NeuronLfm3ForCausalLM(Lfm3CausalLMBase):
    # Only share the generic single-token prefill route and tied weight loader.
    _model_cls=NeuronLfm3Model
    _STATE_DICT_MODEL_PREFIX='model.'

    @classmethod
    def get_config_cls(cls):return Lfm3InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict,config):
        for key in state_dict:
            if not (key.startswith('layers.') or key in ('embed_tokens.weight','embedding_norm.weight','lm_head.weight')):
                raise ValueError('Unexpected LFM3 tensor: '+key)
        state_dict['rank_util.rank']=torch.arange(config.neuron_config.tp_degree,dtype=torch.int32)
        return state_dict


def install_nxdi():
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    if 'lfm3' in MODEL_TYPES and MODEL_TYPES['lfm3']['causal-lm'] is not NeuronLfm3ForCausalLM:
        raise RuntimeError('Refusing to replace an existing LFM3 implementation')
    MODEL_TYPES['lfm3']={'causal-lm':NeuronLfm3ForCausalLM}
