"""Explicit-state dense Nemotron-H text model for a Neuron adapter.

Preserves the official 4B dimensions, Mamba/MLP/NoPE attention layer pattern,
FP32 recurrent state, and the pinned native HF CPU arithmetic structure.
This 4B port has not yet passed native or device numerical validation.
"""
import torch
from nemotron4b_baseline_contract_v1 import validate_config
from torch import nn
import torch.nn.functional as F

from nemotron4b_primitives_v1 import store_slots, conv_prefill, conv_decode, ssm_prefill, ssm_decode, grouped_gated_norm


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True)+self.eps) * self.weight.float()).to(x.dtype)


class GatedNorm(nn.Module):
    def __init__(self, width, groups, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.group_size, self.eps = width//groups, eps

    def forward(self, x, gate):
        return grouped_gated_norm(x, gate, self.weight, self.group_size, self.eps)


class Mamba(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.hd, self.groups, self.sd = c['mamba_num_heads'], c['mamba_head_dim'], c['n_groups'], c['ssm_state_size']
        self.inner = self.heads*self.hd
        self.conv_dim = self.inner+2*self.groups*self.sd
        self.chunk_size = c['chunk_size']
        self.in_proj = nn.Linear(c['hidden_size'], self.inner+self.conv_dim+self.heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, c['conv_kernel'], groups=self.conv_dim, bias=c['use_conv_bias'])
        self.dt_bias = nn.Parameter(torch.ones(self.heads))
        self.A_log = nn.Parameter(torch.arange(1., self.heads+1).log())
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = GatedNorm(self.inner, self.groups, c['layer_norm_epsilon'])
        self.out_proj = nn.Linear(self.inner, c['hidden_size'], bias=False)

    def forward(self, x, cache, slots, positions, lengths, prefill):
        b, t, _ = x.shape
        gate, xbc, dt = self.in_proj(x).split([self.inner, self.conv_dim, self.heads], dim=-1)
        conv_state, ssm_state = cache
        if prefill:
            xbc, conv_updated = conv_prefill(xbc.transpose(1, 2), self.conv1d.weight[:, 0], self.conv1d.bias, lengths)
        else:
            xbc, conv_updated = conv_decode(xbc.transpose(1, 2), self.conv1d.weight[:, 0], self.conv1d.bias, conv_state[slots.clamp_min(0)])
        values, B, C = xbc.transpose(1, 2).split([self.inner, self.groups*self.sd, self.groups*self.sd], dim=-1)
        values = values.reshape(b, t, self.heads, self.hd)
        B, C = [z.reshape(b, t, self.groups, self.sd) for z in (B, C)]
        if prefill:
            # Every prefill starts a new request even when its slot is reused.
            initial = torch.zeros_like(ssm_state[slots.clamp_min(0)])
            scan, ssm_updated = ssm_prefill(values, dt, self.A_log, B, C, self.D, self.dt_bias, initial, lengths, self.chunk_size)
        else:
            scan, ssm_updated = ssm_decode(values, dt, self.A_log, B, C, self.D, self.dt_bias, ssm_state[slots.clamp_min(0)])
        normalized = self.norm(scan.reshape(b, t, self.inner), gate).to(x.dtype)
        updated = (store_slots(conv_state, conv_updated, slots), store_slots(ssm_state, ssm_updated, slots))
        return self.out_proj(normalized), updated


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.kvh, self.hd = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        hidden = c['hidden_size']
        self.q_proj = nn.Linear(hidden, self.heads*self.hd, bias=False)
        self.k_proj = nn.Linear(hidden, self.kvh*self.hd, bias=False)
        self.v_proj = nn.Linear(hidden, self.kvh*self.hd, bias=False)
        self.o_proj = nn.Linear(self.heads*self.hd, hidden, bias=False)

    def forward(self, x, cache, slots, positions, lengths, prefill):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.kvh, self.hd).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.kvh, self.hd).transpose(1, 2)
        kc, vc = cache
        if prefill:
            valid = torch.arange(t, device=x.device)[None] < lengths[:, None]
            sk, sv = [F.pad(z*valid[:, None, :, None], (0, 0, 0, kc.shape[2]-t)) for z in (k, v)]
            all_k, all_v = k, v
        else:
            index = positions[:, None, :, None].expand(b, self.kvh, 1, self.hd)
            sk = kc[slots.clamp_min(0)].scatter(2, index, k)
            sv = vc[slots.clamp_min(0)].scatter(2, index, v)
            all_k, all_v = sk, sv
        keys = torch.arange(all_k.shape[2], device=x.device)[None, None, None]
        allowed = (keys <= positions[:, None, :, None]) & (keys < lengths[:, None, None, None])
        all_k, all_v = [z.repeat_interleave(self.heads//self.kvh, dim=1) for z in (all_k, all_v)]
        scores = (q @ all_k.transpose(-1, -2))*self.hd**-0.5
        scores = scores + torch.where(allowed, torch.zeros_like(scores), torch.full_like(scores, torch.finfo(scores.dtype).min))
        out = (scores.softmax(-1, dtype=torch.float32).to(x.dtype) @ all_v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), (store_slots(kc, sk, slots), store_slots(vc, sv, slots))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.up_proj = nn.Linear(c['hidden_size'], c['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(c['intermediate_size'], c['hidden_size'], bias=False)

    def forward(self, x):
        return self.down_proj(F.relu(self.up_proj(x)).square())


class Layer(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        self.norm = RMSNorm(c['hidden_size'], c['layer_norm_epsilon'])
        self.mixer = {'M': Mamba, '*': Attention, '-': MLP}[kind](c)

    def forward(self, x, cache, slots, positions, lengths, prefill):
        normalized = self.norm(x)
        if self.kind == '-':
            return x+self.mixer(normalized), cache
        update, cache = self.mixer(normalized, cache, slots, positions, lengths, prefill)
        return x+update, cache


class Nemotron4BTextPortV1(nn.Module):
    def __init__(self, c):
        super().__init__()
        validate_config(c)
        self.config = c
        assert len(c['hybrid_override_pattern']) == c['num_hidden_layers']
        assert set(c['hybrid_override_pattern']) <= set('M*-')
        assert c['mamba_num_heads'] % c['n_groups'] == 0
        assert c['num_attention_heads'] % c['num_key_value_heads'] == 0
        assert c['mamba_hidden_act'] == 'silu' and c['mlp_hidden_act'] == 'relu2'
        assert not any(c.get(k, False) for k in ['tie_word_embeddings', 'residual_in_fp32', 'attention_bias', 'mlp_bias', 'use_bias', 'mamba_proj_bias'])
        assert not c.get('attention_dropout', 0) and not c.get('hidden_dropout', 0)
        assert not c.get('num_nextn_predict_layers', 0) and c.get('sliding_window') is None
        assert tuple(c.get('time_step_limit', (0., float('inf')))) == (0., float('inf'))
        self.embeddings = nn.Embedding(c['vocab_size'], c['hidden_size'])
        self.layers = nn.ModuleList([Layer(c, k) for k in c['hybrid_override_pattern']])
        self.norm_f = RMSNorm(c['hidden_size'], c['layer_norm_epsilon'])
        self.lm_head = nn.Linear(c['hidden_size'], c['vocab_size'], bias=False)

    def empty_cache(self, capacity, length):
        c = self.config
        dtype, device = self.embeddings.weight.dtype, self.embeddings.weight.device
        cache = []
        for kind in c['hybrid_override_pattern']:
            if kind == 'M':
                inner = c['mamba_num_heads']*c['mamba_head_dim']
                conv_dim = inner+2*c['n_groups']*c['ssm_state_size']
                cache.append((torch.zeros(capacity, conv_dim, c['conv_kernel'], dtype=dtype, device=device),
                              torch.zeros(capacity, c['mamba_num_heads'], c['mamba_head_dim'], c['ssm_state_size'], dtype=torch.float32, device=device)))
            elif kind == '*':
                cache.append(tuple(torch.zeros(capacity, c['num_key_value_heads'], length, c['head_dim'], dtype=dtype, device=device) for _ in range(2)))
            else:
                cache.append(())
        return cache

    def forward(self, ids, cache, slots, positions, lengths, prefill):
        x = self.embeddings(ids)
        updated = []
        for layer, states in zip(self.layers, cache):
            x, states = layer(x, states, slots, positions, lengths, prefill)
            updated.append(states)
        if prefill:
            x = x.gather(1, (lengths-1)[:, None, None].expand(x.shape[0], 1, x.shape[-1]))
        return self.lm_head(self.norm_f(x)), updated
