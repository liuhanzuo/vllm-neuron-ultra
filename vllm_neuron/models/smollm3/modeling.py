# SPDX-License-Identifier: Apache-2.0
"""Experimental original BF16 SmolLM3 text adapter for NxDI TP1/TP2."""
import torch
from torch import nn
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding,ColumnParallelLinear,RowParallelLinear
from neuronx_distributed_inference.models.config import InferenceConfig,NeuronConfig
from neuronx_distributed_inference.models.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from .text_core import Attention, Layer, RMSNorm
from .identity import AdmissionError, validate_scope


class SmolLM3InferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group=1;nc=self.neuron_config
        validate_scope(vars(self), vars(nc))
        self.head_dim=getattr(self,'head_dim',None) or self.hidden_size//self.num_attention_heads
        if len(self.no_rope_layers)!=self.num_hidden_layers:raise ValueError('Incorrect per-layer RoPE pattern')
        if any(v not in (0,1,False,True) for v in self.no_rope_layers):raise ValueError('Invalid RoPE mask')
        for flag in ('quantized','sequence_parallel_enabled','flash_decoding_enabled','is_prefix_caching',
                     'is_block_kv_layout','enable_fused_speculation','enable_eagle_speculation','is_eagle_draft','output_logits'):
            if getattr(nc,flag,False):raise ValueError('Unvalidated SmolLM3 route: '+flag)
        if nc.cp_degree!=1 or nc.attention_dp_degree!=1 or nc.pp_degree!=1:
            raise ValueError('Only tensor parallelism is implemented')
        if nc.on_device_sampling_config is not None or nc.lora_config is not None:
            raise ValueError('Only CPU sampling without LoRA is implemented')
        if nc.padding_side!='right' or nc.tp_degree not in (1,2):
            raise ValueError('Only right padding and TP1/TP2 are implemented')
        if nc.speculation_length or nc.medusa_speculation_length:
            raise ValueError('Speculative decoding is not implemented')
        if self.sliding_window is not None or self.hidden_act!='silu' or self.use_sliding_window:
            raise ValueError('Only original dense full-attention text configuration is implemented')
        if getattr(self,'attention_bias',False) or getattr(self,'mlp_bias',False):
            raise ValueError('Bias is unsupported')
        if getattr(self,'rope_scaling',None) is not None:raise ValueError('Only original unscaled RoPE is implemented')
        if any(t!='full_attention' for t in self.layer_types):raise ValueError('Only full attention is implemented')
        r=getattr(self,'rope_parameters',None)
        if r is not None and r.get('rope_type')!='default':raise ValueError('Only default RoPE is implemented')
        if self.pad_token_id is None:raise ValueError('Explicit unused padding ID required for Neuron static shapes')
        for key in ('num_attention_heads','num_key_value_heads','hidden_size','intermediate_size','vocab_size'):
            if getattr(self,key)%nc.tp_degree:raise ValueError('Non-divisible dimension: '+key)

    def get_required_attributes(self):
        return ['hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','num_hidden_layers',
                'head_dim','pad_token_id','vocab_size','rms_norm_eps','no_rope_layers','layer_types','sliding_window','tie_word_embeddings']

    @classmethod
    def get_neuron_config_cls(cls):return NeuronConfig


class SmolLM3Cache(nn.Module):
    def __init__(self,c,nc):
        super().__init__();shape=(nc.max_batch_size,c.num_key_value_heads//nc.tp_degree,nc.max_length,c.head_dim)
        self.past_key_values=nn.ParameterList(nn.Parameter(torch.zeros(shape,dtype=nc.torch_dtype),requires_grad=False)
                                             for _ in range(2*c.num_hidden_layers))


class NeuronSmolLM3Model(NeuronBaseModel):
    def setup_attr_for_model(self,c):
        self.on_device_sampling=False;self.tp_degree=c.neuron_config.tp_degree;self.hidden_size=c.hidden_size
        self.num_attention_heads,self.num_key_value_heads=c.num_attention_heads,c.num_key_value_heads
        self.max_batch_size,self.buckets=c.neuron_config.max_batch_size,c.neuron_config.buckets
        self.prefill=bool(c.neuron_config.is_prefill_stage)

    def init_model(self,c):
        if not parallel_state.model_parallel_is_initialized():raise RuntimeError('Initialize NxD TP groups first')
        tp,dtype=self.tp_degree,c.neuron_config.torch_dtype;h=c.hidden_size
        self.embed_tokens=ParallelEmbedding(c.vocab_size,h,dtype=dtype,shard_across_embedding=True)
        self.lm_head=ColumnParallelLinear(h,c.vocab_size,bias=False,gather_output=True,dtype=dtype)
        def col(a,b):return ColumnParallelLinear(a,b,bias=False,gather_output=False,dtype=dtype)
        def row(a,b):return RowParallelLinear(a,b,bias=False,input_is_parallel=True,dtype=dtype)
        self.layers=nn.ModuleList()
        identity=(c.metadata or {}).get('smollm3_ultra')
        if not identity or identity['variant'] not in ('reference', 'nki_read_write'):
            raise AdmissionError('SmolLM3 requires an admitted build identity before model construction')
        attention_cls=Attention
        if identity['variant']=='nki_read_write':
            from .cache_attention import NkiAttention
            attention_cls=NkiAttention
        for i in range(c.num_hidden_layers):
            layer=Layer(vars(c),i,attention_cls=attention_cls);a=layer.self_attn
            a.q_proj=col(h,a.h*a.dim);a.k_proj=col(h,a.kvh*a.dim);a.v_proj=col(h,a.kvh*a.dim)
            a.o_proj=row(a.h*a.dim,h);a.h//=tp;a.kvh//=tp
            layer.mlp.gate_proj=col(h,c.intermediate_size);layer.mlp.up_proj=col(h,c.intermediate_size)
            layer.mlp.down_proj=row(c.intermediate_size,h);self.layers.append(layer)
        self.norm=RMSNorm(h,c.rms_norm_eps)

    def init_inference_optimization(self,c):self.kv_mgr=SmolLM3Cache(c,c.neuron_config)

    def _apply(self,fn,recurse=True):
        kept=[(l.self_attn,l.self_attn.inverse_frequencies) for l in self.layers]
        super()._apply(fn,recurse=recurse)
        for a,inv in kept:a.inverse_frequencies=inv.to(device=a.inverse_frequencies.device,dtype=torch.float32)
        return self

    def forward(self,input_ids,attention_mask,position_ids,seq_ids,sampling_params=None,*args,**kwargs):
        lengths=attention_mask.sum(-1).clamp_min(1) if self.prefill else position_ids[:,0]+1
        x=self.embed_tokens(input_ids);updated=[]
        for i,layer in enumerate(self.layers):
            full=self.kv_mgr.past_key_values[2*i:2*i+2]
            cache=full if self.prefill else [p[:,:,:self.n_positions,:] for p in full]
            x,cache=layer(x,cache,seq_ids,position_ids,lengths,self.prefill)
            if not self.prefill:cache=[torch.cat((prefix,old[:,:,self.n_positions:,:]),2) for prefix,old in zip(cache,full)]
            updated.extend(cache)
        if self.prefill:x=torch.gather(x,1,(lengths-1)[:,None,None].expand(x.shape[0],1,x.shape[-1]))
        return [self.lm_head(self.norm(x)).float()]+updated


class NeuronSmolLM3ForCausalLM(NeuronBaseForCausalLM):
    _model_cls=NeuronSmolLM3Model
    _STATE_DICT_MODEL_PREFIX='model.'

    @classmethod
    def get_config_cls(cls):return SmolLM3InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict,config):
        result={}
        for key,tensor in state_dict.items():
            if not (key.startswith('layers.') or key in ('embed_tokens.weight','norm.weight','lm_head.weight')):
                raise ValueError('Unexpected SmolLM3 tensor: '+key)
            result[key]=tensor
        result['rank_util.rank']=torch.arange(config.neuron_config.tp_degree,dtype=torch.int32)
        return result


    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict['lm_head.weight'] = state_dict['embed_tokens.weight'].clone()

    def forward(self, input_ids=None, seq_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None:
            raise ValueError('Explicit positions are required for SmolLM3 serving')
        if seq_ids is None or seq_ids.dtype not in (torch.int32, torch.int64):
            raise AdmissionError('SmolLM3 requires integer serving slot IDs')
        if bool((seq_ids >= 4).any()) or bool((seq_ids < -(2**31)).any()):
            raise AdmissionError('SmolLM3 slot IDs exceed the admitted int32/capacity4 contract')
        if input_ids.shape[1] == 1 and bool((position_ids == 0).all()):
            input_ids = torch.nn.functional.pad(input_ids, (0, 1), value=self.config.pad_token_id)
            position_ids = torch.nn.functional.pad(position_ids, (0, 1), value=0)
            attention_mask = torch.tensor([1, 0], dtype=position_ids.dtype).expand(input_ids.shape[0], 2)
        return super().forward(input_ids=input_ids, seq_ids=seq_ids, attention_mask=attention_mask,
                               position_ids=position_ids, **kwargs)

    def load(self, *args, **kwargs):
        result = super().load(*args, **kwargs)
        wrapper = self.token_generation_model
        if getattr(wrapper, '_smollm3_minimum_decode_bucket', None) is None:
            if 2048 not in wrapper.neuron_config.buckets:
                raise AdmissionError('SmolLM3 artifact has no required decode bucket2048')
            original = wrapper.get_target_bucket
            def select(*inputs, strategy='first_fit'):
                return max(2048, original(*inputs, strategy=strategy))
            wrapper.get_target_bucket = select
            wrapper._smollm3_minimum_decode_bucket = 2048
        elif wrapper._smollm3_minimum_decode_bucket != 2048:
            raise AdmissionError('SmolLM3 decode routing conflicts with the admitted floor')
        return result


def install_nxdi():
    from neuronx_distributed_inference.utils.constants import MODEL_TYPES
    key='smollm3neurontext'
    if key in MODEL_TYPES and MODEL_TYPES[key]['causal-lm'] is not NeuronSmolLM3ForCausalLM:
        raise RuntimeError('Refusing to replace an existing SmolLM3 implementation')
    MODEL_TYPES[key]={'causal-lm':NeuronSmolLM3ForCausalLM}
