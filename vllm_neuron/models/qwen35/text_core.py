# SPDX-License-Identifier: Apache-2.0
"""Explicit-state, torch-only Qwen3.5 text architecture for Neuron integration.

Derived from the Apache-2.0 Qwen/Hugging Face reference pinned in research/.
This module has no vLLM registration or tensor-parallel integration yet.
All 24 layers retain their actual type; no substitution with plain Qwen3.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .helpers import split_last_dim
from .primitives import conv_decode, conv_prefill, delta_decode, delta_prefill


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * (1 + self.weight.float())).to(x.dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        # The HF reference rounds normalization before multiplying the weight.
        normalized = self.weight * normalized.to(x.dtype)
        return (normalized * F.silu(gate.float())).to(x.dtype)


def rope(x, positions, inverse_frequencies):
    frequencies = positions.float()[..., None] * inverse_frequencies.float()
    cos = torch.cat((frequencies.cos(), frequencies.cos()), -1).to(x.dtype)[:, None]
    sin = torch.cat((frequencies.sin(), frequencies.sin()), -1).to(x.dtype)[:, None]
    dim = cos.shape[-1]
    part, rest = x[..., :dim], x[..., dim:]
    half = torch.cat((-part[..., dim // 2:], part[..., :dim // 2]), -1)
    return torch.cat((part * cos + half * sin, rest), -1)


def store_slots(cache, updates, slots):
    """Functional writes also preserve inactive slots (ID -1), without collisions."""
    indices = torch.arange(cache.shape[0], device=cache.device)
    for row in range(updates.shape[0]):
        selected = indices == slots[row]
        cache = torch.where(selected.reshape((-1,) + (1,) * (cache.ndim - 1)), updates[row:row + 1], cache)
    return cache


class LinearAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.kh, self.vh = c['linear_num_key_heads'], c['linear_num_value_heads']
        self.kd, self.vd = c['linear_key_head_dim'], c['linear_value_head_dim']
        self.width = c['linear_conv_kernel_dim']
        self.key_dim, self.value_dim = self.kh * self.kd, self.vh * self.vd
        hidden, channels = c['hidden_size'], 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden, channels, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.vh, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.vh, bias=False)
        self.conv1d = nn.Conv1d(channels, channels, self.width, groups=channels, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.vh))
        self.A_log = nn.Parameter(torch.zeros(self.vh))
        self.norm = GatedRMSNorm(self.vd, c['rms_norm_eps'])
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)

    def forward(self, x, cache, slots, lengths, prefill):
        b, t, _ = x.shape
        if prefill:
            valid = torch.arange(t, device=x.device)[None, :] < lengths[:, None]
            x = x * valid[..., None].to(x.dtype)
        projected = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(b, t, self.vh, self.vd)
        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias)
        conv_cache, state_cache = cache
        if prefill:
            projected, conv = conv_prefill(projected, self.conv1d.weight[:, 0], lengths)
            state = torch.zeros((b, self.vh, self.kd, self.vd), dtype=torch.float32, device=x.device)
        else:
            selected = slots.clamp_min(0)
            projected, conv = conv_decode(projected, self.conv1d.weight[:, 0], conv_cache[selected])
            state = state_cache[selected]
        q, k, v = split_last_dim(projected.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim])
        q, k = q.reshape(b, t, self.kh, self.kd), k.reshape(b, t, self.kh, self.kd)
        v = v.reshape(b, t, self.vh, self.vd)
        if self.vh != self.kh:
            q = q.repeat_interleave(self.vh // self.kh, dim=2)
            k = k.repeat_interleave(self.vh // self.kh, dim=2)
        if prefill:
            output, state = delta_prefill(q, k, v, g, beta, state, lengths)
        else:
            output, state = delta_decode(q, k, v, g, beta, state)
        output = self.norm(output, z).reshape(b, t, self.value_dim)
        new_cache = store_slots(conv_cache, conv, slots), store_slots(state_cache, state, slots)
        return self.out_proj(output), new_cache


class FullAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        hidden = c['hidden_size']
        self.h, self.kvh, self.dim = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
        self.q_proj = nn.Linear(hidden, 2 * self.h * self.dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kvh * self.dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kvh * self.dim, bias=False)
        self.o_proj = nn.Linear(self.h * self.dim, hidden, bias=False)
        self.q_norm, self.k_norm = (RMSNorm(self.dim, c['rms_norm_eps']) for _ in range(2))
        rc = c['rope_parameters']
        assert rc['rope_type'] == 'default'
        rotary_dim = int(self.dim * rc['partial_rotary_factor'])
        inverse = 1.0 / (rc['rope_theta'] ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer('inverse_frequencies', inverse, persistent=False)

    def forward(self, x, cache, slots, positions, lengths, prefill):
        b, t, _ = x.shape
        q, gate = self.q_proj(x).reshape(b, t, self.h, 2 * self.dim).chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(b, t, self.kvh, self.dim)).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.kvh, self.dim).transpose(1, 2)
        q, k = (rope(z, positions, self.inverse_frequencies) for z in (q, k))
        kcache, vcache = cache
        if prefill:
            mask = torch.arange(t, device=x.device)[None, :] < lengths[:, None]
            all_k, all_v = k, v
            stored_k, stored_v = (F.pad(z * mask[:, None, :, None], (0, 0, 0, kcache.shape[2] - t)) for z in (k, v))
            allowed = torch.arange(t, device=x.device)[None, None, None, :] <= positions[:, None, :, None]
            allowed = allowed & mask[:, None, None, :]
        else:
            gather = slots.clamp_min(0)
            index = positions[:, None, :, None].expand(b, self.kvh, 1, self.dim)
            stored_k = torch.scatter(kcache[gather], 2, index, k)
            stored_v = torch.scatter(vcache[gather], 2, index, v)
            all_k, all_v = stored_k, stored_v
            allowed = torch.arange(kcache.shape[2], device=x.device)[None, None, None, :] <= positions[:, None, :, None]
        all_k, all_v = (z.repeat_interleave(self.h // self.kvh, dim=1) for z in (all_k, all_v))
        scores = q @ all_k.transpose(-1, -2) * (self.dim ** -0.5)
        scores = scores + torch.where(allowed, torch.zeros_like(scores), torch.full_like(scores, torch.finfo(scores.dtype).min))
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        output = (probabilities @ all_v).transpose(1, 2).reshape(b, t, self.h * self.dim)
        output = output * gate.reshape(b, t, -1).sigmoid()
        new_cache = store_slots(kcache, stored_k, slots), store_slots(vcache, stored_v, slots)
        return self.o_proj(output), new_cache


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(c['hidden_size'], c['intermediate_size'], bias=False)
        self.up_proj = nn.Linear(c['hidden_size'], c['intermediate_size'], bias=False)
        self.down_proj = nn.Linear(c['intermediate_size'], c['hidden_size'], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c, kind):
        super().__init__()
        self.kind = kind
        if kind == 'linear_attention':
            self.linear_attn = LinearAttention(c)
        else:
            assert kind == 'full_attention'
            self.self_attn = FullAttention(c)
        self.mlp = MLP(c)
        self.input_layernorm, self.post_attention_layernorm = (RMSNorm(c['hidden_size'], c['rms_norm_eps']) for _ in range(2))

    def forward(self, x, cache, slots, positions, lengths, prefill):
        normalized = self.input_layernorm(x)
        if self.kind == 'linear_attention':
            update, cache = self.linear_attn(normalized, cache, slots, lengths, prefill)
        else:
            update, cache = self.self_attn(normalized, cache, slots, positions, lengths, prefill)
        x = x + update
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class Qwen35TextPort(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert not config['attention_bias'] and config['attn_output_gate']
        assert config['tie_word_embeddings']
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([Layer(config, kind) for kind in config['layer_types']])
        self.norm = RMSNorm(config['hidden_size'], config['rms_norm_eps'])

    def empty_cache(self, capacity, max_length):
        c, cache = self.config, []
        dtype, device = self.embed_tokens.weight.dtype, self.embed_tokens.weight.device
        for kind in c['layer_types']:
            if kind == 'linear_attention':
                channels = 2 * c['linear_num_key_heads'] * c['linear_key_head_dim'] + c['linear_num_value_heads'] * c['linear_value_head_dim']
                a = torch.zeros(capacity, channels, c['linear_conv_kernel_dim'], dtype=dtype, device=device)
                b = torch.zeros(capacity, c['linear_num_value_heads'], c['linear_key_head_dim'], c['linear_value_head_dim'], dtype=torch.float32, device=device)
            else:
                shape = (capacity, c['num_key_value_heads'], max_length, c['head_dim'])
                a, b = (torch.zeros(shape, dtype=dtype, device=device) for _ in range(2))
            cache.append((a, b))
        return cache

    def forward(self, ids, cache, slots, positions, lengths, prefill):
        x, new_cache = self.embed_tokens(ids), []
        for layer, layer_cache in zip(self.layers, cache):
            x, updated = layer(x, layer_cache, slots, positions, lengths, prefill)
            new_cache.append(updated)
        if prefill:
            index = (lengths - 1)[:, None, None].expand(x.shape[0], 1, x.shape[-1])
            x = torch.gather(x, 1, index)
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight), new_cache
