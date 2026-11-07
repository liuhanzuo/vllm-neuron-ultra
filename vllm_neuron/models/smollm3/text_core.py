# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified: explicit cache state, NxDI serving slots and selectable attention
# class. Derived from the pinned local SmolLM3 port; see provenance.py.
"""Explicit-state SmolLM3 text core from pinned HF Apache-2.0 reference.

Preserves the original per-layer RoPE / NoPE pattern and GQA.
CPU architecture port only: no Neuron registration, compilation or TP yet.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .helpers import store_slots


class RMSNorm(nn.Module):
    def __init__(self,dim,eps):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps

    def forward(self,x):
        a=x.float();a=a*torch.rsqrt(a.square().mean(-1,keepdim=True)+self.eps)
        return self.weight*a.to(x.dtype)


class Attention(nn.Module):
    def __init__(self,c,layer_index):
        super().__init__();self.h=c['num_attention_heads'];self.kvh=c['num_key_value_heads'];self.dim=c.get('head_dim') or c['hidden_size']//self.h
        h=c['hidden_size'];self.use_rope=bool(c['no_rope_layers'][layer_index])
        self.q_proj=nn.Linear(h,self.h*self.dim,bias=False)
        self.k_proj=nn.Linear(h,self.kvh*self.dim,bias=False)
        self.v_proj=nn.Linear(h,self.kvh*self.dim,bias=False)
        self.o_proj=nn.Linear(self.h*self.dim,h,bias=False)
        r=c.get('rope_parameters') or {'rope_theta':c['rope_theta'],'rope_type':'default'}
        assert r['rope_type']=='default'
        inv=1/(r['rope_theta']**(torch.arange(0,self.dim,2).float()/self.dim))
        self.register_buffer('inverse_frequencies',inv,persistent=False)

    def forward(self,x,cache,slots,positions,lengths,prefill):
        b,t,_=x.shape
        q=self.q_proj(x).reshape(b,t,self.h,self.dim).transpose(1,2)
        k=self.k_proj(x).reshape(b,t,self.kvh,self.dim).transpose(1,2)
        v=self.v_proj(x).reshape(b,t,self.kvh,self.dim).transpose(1,2)
        if self.use_rope:
            phase=positions.float()[...,None]*self.inverse_frequencies.float()
            phase=torch.cat((phase,phase),-1)
            co=phase.cos().to(q.dtype)[:,None]
            si=phase.sin().to(q.dtype)[:,None]
            def rotate(z):return z*co+torch.cat((-z[...,self.dim//2:],z[...,:self.dim//2]),-1)*si
            q,k=rotate(q),rotate(k)
        kc,vc=cache
        if prefill:
            valid=torch.arange(t,device=x.device)[None,:]<lengths[:,None]
            all_k,all_v=k,v
            sk,sv=(F.pad(z*valid[:,None,:,None],(0,0,0,kc.shape[2]-t)) for z in (k,v))
        else:
            idx=positions[:,None,:,None].expand(b,self.kvh,1,self.dim)
            sk=torch.scatter(kc[slots.clamp_min(0)],2,idx,k)
            sv=torch.scatter(vc[slots.clamp_min(0)],2,idx,v)
            all_k,all_v=sk,sv
        keys=torch.arange(all_k.shape[2],device=x.device)[None,None,None,:]
        allowed=(keys<=positions[:,None,:,None]) & (keys<lengths[:,None,None,None])
        all_k,all_v=(z.repeat_interleave(self.h//self.kvh,1) for z in (all_k,all_v))
        scores=(q@all_k.transpose(-1,-2))*self.dim**-.5
        scores=scores+torch.where(allowed,torch.zeros_like(scores),torch.full_like(scores,torch.finfo(scores.dtype).min))
        out=(scores.softmax(-1,dtype=torch.float32).to(q.dtype)@all_v).transpose(1,2).reshape(b,t,-1)
        return self.o_proj(out),(store_slots(kc,sk,slots),store_slots(vc,sv,slots))


class MLP(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden_size'];inner=c['intermediate_size']
        self.gate_proj=nn.Linear(h,inner,bias=False);self.up_proj=nn.Linear(h,inner,bias=False)
        self.down_proj=nn.Linear(inner,h,bias=False)

    def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class Layer(nn.Module):
    def __init__(self,c,layer_index,attention_cls=Attention):
        super().__init__();h=c['hidden_size'];eps=c['rms_norm_eps']
        self.self_attn=attention_cls(c,layer_index);self.mlp=MLP(c)
        self.input_layernorm=RMSNorm(h,eps);self.post_attention_layernorm=RMSNorm(h,eps)

    def forward(self,x,cache,slots,positions,lengths,prefill):
        update,cache=self.self_attn(self.input_layernorm(x),cache,slots,positions,lengths,prefill)
        x=x+update
        return x+self.mlp(self.post_attention_layernorm(x)),cache


class SmolLM3TextPort(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c
        assert not c.get('use_sliding_window',False) and c.get('sliding_window') is None and c['hidden_act']=='silu'
        assert len(c['no_rope_layers'])==c['num_hidden_layers']
        assert all(v in (0,1,False,True) for v in c['no_rope_layers'])
        assert all(v=='full_attention' for v in c['layer_types'])
        assert c.get('rope_scaling') is None
        assert not c.get('attention_bias',False) and not c.get('mlp_bias',False)
        assert c['num_attention_heads']%c['num_key_value_heads']==0
        self.embed_tokens=nn.Embedding(c['vocab_size'],c['hidden_size'],c.get('pad_token_id'))
        self.layers=nn.ModuleList([Layer(c,i) for i in range(c['num_hidden_layers'])])
        self.norm=RMSNorm(c['hidden_size'],c['rms_norm_eps'])
        self.lm_head=nn.Linear(c['hidden_size'],c['vocab_size'],bias=False)
        if c.get('tie_word_embeddings',False):self.lm_head.weight=self.embed_tokens.weight

    def _apply(self,fn,recurse=True):
        kept=[(l.self_attn,l.self_attn.inverse_frequencies) for l in self.layers]
        super()._apply(fn,recurse=recurse)
        for a,inv in kept:a.inverse_frequencies=inv.to(device=a.inverse_frequencies.device,dtype=torch.float32)
        return self

    def empty_cache(self,capacity,length):
        c=self.config
        return [tuple(torch.zeros(capacity,c['num_key_value_heads'],length,c.get('head_dim') or c['hidden_size']//c['num_attention_heads'],
                    dtype=self.embed_tokens.weight.dtype,device=self.embed_tokens.weight.device) for _ in range(2)) for _ in self.layers]

    def forward(self,ids,cache,slots,positions,lengths,prefill):
        x=self.embed_tokens(ids);updated=[]
        for layer,state in zip(self.layers,cache):
            x,state=layer(x,state,slots,positions,lengths,prefill);updated.append(state)
        if prefill:x=torch.gather(x,1,(lengths-1)[:,None,None].expand(x.shape[0],1,x.shape[-1]))
        return self.lm_head(self.norm(x)),updated
