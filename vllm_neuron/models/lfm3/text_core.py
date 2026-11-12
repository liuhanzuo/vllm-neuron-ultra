# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Explicit-state LFM3 architecture, derived from pinned official Apache2 source.

Packaged from the historical v4 experiment; this migration awaits qualification.
Preserves width7 B*x/C gated convolution, sandwich norms, per-head attention
gates, sliding-window masks, and zero-tail/full-dimension partial RoPE.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .helpers import store_slots


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(dim));self.eps=eps

    def forward(self,x):
        normalized=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)
        return self.weight*normalized.to(x.dtype)


def inverse_frequencies(c,kind):
    rc=c['rope_parameters'][kind];assert rc['rope_type']=='default'
    dim=c['head_dim'];n=int(rc.get('partial_rotary_factor',1)*dim)//2
    active=1/(rc['rope_theta']**(torch.arange(0,2*n,2).float()/dim))
    return torch.cat((active,torch.zeros(dim//2-n)))


def apply_rope(x,positions,inverse):
    angles=positions.float()[...,None]*inverse.float()
    angles=torch.cat((angles,angles),-1)
    # Official code rounds cos/sin to input dtype, then multiplies in FP32.
    cos=angles.cos().to(x.dtype).float()[:,None]
    sin=angles.sin().to(x.dtype).float()[:,None]
    a=x.float();h=a.shape[-1]//2
    half=torch.cat((-a[...,h:],a[...,:h]),-1)
    return (a*cos+half*sin).to(x.dtype)


class ShortConv(nn.Module):
    def __init__(self,c):
        super().__init__();h=c['hidden_size'];self.width=c['conv_L_cache']
        assert not c['conv_bias']
        self.in_proj=nn.Linear(h,3*h,bias=False)
        self.conv=nn.Conv1d(h,h,self.width,groups=h,bias=False)
        self.out_proj=nn.Linear(h,h,bias=False)

    def forward(self,x,cache,slots,lengths,prefill):
        b,t,_=x.shape
        if prefill:
            valid=torch.arange(t,device=x.device)[None,:]<lengths[:,None]
            x=x*valid[...,None].to(x.dtype)
        B,C,value=self.in_proj(x).transpose(1,2).chunk(3,dim=1)
        value=B*value
        if prefill:
            value=value*valid[:,None].to(value.dtype)
            padded=F.pad(value,(self.width-1,0))
            out=torch.zeros_like(value,dtype=torch.float32)
            for k in range(self.width):
                out=out+padded[:,:,k:k+t].float()*self.conv.weight[:,0,k][None,:,None].float()
            # index length-1 selects the valid window in the left-padded tensor.
            index=(lengths[:,None,None]-1+torch.arange(self.width,device=x.device)[None,None,:])
            state=torch.gather(padded,2,index.expand(b,value.shape[1],self.width))
        else:
            state=torch.cat((cache[0][slots.clamp_min(0)][:,:,1:],value),dim=2)
            out=(state.float()*self.conv.weight[:,0,:][None].float()).sum(-1,keepdim=True)
        out=out.to(value.dtype)
        return self.out_proj((C*out).transpose(1,2)),(store_slots(cache[0],state,slots),)


class Attention(nn.Module):
    def __init__(self,c,kind):
        super().__init__();hidden=c['hidden_size']
        self.h=c['num_attention_heads'];self.kvh=c['num_key_value_heads'];self.dim=c['head_dim']
        self.q_proj=nn.Linear(hidden,self.h*self.dim,bias=False)
        self.k_proj=nn.Linear(hidden,self.kvh*self.dim,bias=False)
        self.v_proj=nn.Linear(hidden,self.kvh*self.dim,bias=False)
        self.o_proj=nn.Linear(self.h*self.dim,hidden,bias=False)
        self.gate_proj=nn.Linear(hidden,self.h,bias=False)
        self.q_norm,self.k_norm=(RMSNorm(self.dim,c['norm_eps']) for _ in range(2))
        self.window=c['sliding_window'] if kind=='sliding_attention' else None
        self.register_buffer('inverse_frequencies',inverse_frequencies(c,kind),persistent=False)

    def forward(self,x,cache,slots,positions,lengths,prefill):
        b,t,_=x.shape
        q=self.q_norm(self.q_proj(x).reshape(b,t,self.h,self.dim)).transpose(1,2)
        k=self.k_norm(self.k_proj(x).reshape(b,t,self.kvh,self.dim)).transpose(1,2)
        v=self.v_proj(x).reshape(b,t,self.kvh,self.dim).transpose(1,2)
        q,k=(apply_rope(z,positions,self.inverse_frequencies) for z in (q,k))
        kc,vc=cache
        if prefill:
            valid=torch.arange(t,device=x.device)[None,:]<lengths[:,None]
            all_k,all_v=k,v
            sk,sv=(F.pad(z*valid[:,None,:,None],(0,0,0,kc.shape[2]-t)) for z in (k,v))
            keypos=torch.arange(t,device=x.device)[None,None,None,:]
            allowed=(keypos<=positions[:,None,:,None]) & valid[:,None,None,:]
        else:
            index=positions[:,None,:,None].expand(b,self.kvh,1,self.dim)
            sk=torch.scatter(kc[slots.clamp_min(0)],2,index,k)
            sv=torch.scatter(vc[slots.clamp_min(0)],2,index,v)
            all_k,all_v=sk,sv
            keypos=torch.arange(kc.shape[2],device=x.device)[None,None,None,:]
            allowed=keypos<=positions[:,None,:,None]
        if self.window is not None:
            allowed=allowed & (keypos>positions[:,None,:,None]-self.window)
        all_k,all_v=(z.repeat_interleave(self.h//self.kvh,dim=1) for z in (all_k,all_v))
        scores=(q@all_k.transpose(-1,-2))*(self.dim**-.5)
        scores=scores+torch.where(allowed,torch.zeros_like(scores),torch.full_like(scores,torch.finfo(scores.dtype).min))
        output=(scores.softmax(-1,dtype=torch.float32).to(x.dtype)@all_v).transpose(1,2)
        output=output*self.gate_proj(x).sigmoid()[...,None]
        updated=(store_slots(kc,sk,slots),store_slots(vc,sv,slots))
        return self.o_proj(output.reshape(b,t,-1)),updated


class MLP(nn.Module):
    def __init__(self,c):
        super().__init__();h,i=c['hidden_size'],c['intermediate_size']
        self.w1,self.w3=(nn.Linear(h,i,bias=False) for _ in range(2))
        self.w2=nn.Linear(i,h,bias=False)

    def forward(self,x):return self.w2(F.silu(self.w1(x))*self.w3(x))


class Layer(nn.Module):
    def __init__(self,c,kind):
        super().__init__();self.kind=kind;assert c['use_sandwich_norm']
        if kind=='conv':self.conv=ShortConv(c)
        else:
            assert kind in ('full_attention','sliding_attention')
            self.self_attn=Attention(c,kind)
        self.feed_forward=MLP(c)
        self.operator_norm,self.ffn_norm,self.post_operator_norm,self.post_ffn_norm=(
            RMSNorm(c['hidden_size'],c['norm_eps']) for _ in range(4))

    def forward(self,x,cache,slots,positions,lengths,prefill):
        normed=self.operator_norm(x)
        if self.kind=='conv':out,cache=self.conv(normed,cache,slots,lengths,prefill)
        else:out,cache=self.self_attn(normed,cache,slots,positions,lengths,prefill)
        x=x+self.post_operator_norm(out)
        x=x+self.post_ffn_norm(self.feed_forward(self.ffn_norm(x)))
        return x,cache


class Lfm3TextPort(nn.Module):
    def __init__(self,c):
        super().__init__();self.config=c;assert c['tie_word_embeddings']
        self.embed_tokens=nn.Embedding(c['vocab_size'],c['hidden_size'],c['pad_token_id'])
        self.layers=nn.ModuleList([Layer(c,kind) for kind in c['layer_types']])
        self.embedding_norm=RMSNorm(c['hidden_size'],c['norm_eps'])

    def empty_cache(self,capacity,length):
        c=self.config;dtype=self.embed_tokens.weight.dtype;device=self.embed_tokens.weight.device;result=[]
        for kind in c['layer_types']:
            if kind=='conv':shapes=[(capacity,c['hidden_size'],c['conv_L_cache'])]
            else:shapes=[(capacity,c['num_key_value_heads'],length,c['head_dim'])]*2
            result.append(tuple(torch.zeros(s,dtype=dtype,device=device) for s in shapes))
        return result

    def forward(self,ids,cache,slots,positions,lengths,prefill):
        x=self.embed_tokens(ids);updated=[]
        for layer,states in zip(self.layers,cache):
            x,states=layer(x,states,slots,positions,lengths,prefill);updated.append(states)
        if prefill:
            index=(lengths-1)[:,None,None].expand(x.shape[0],1,x.shape[-1]);x=torch.gather(x,1,index)
        return F.linear(self.embedding_norm(x),self.embed_tokens.weight),updated
