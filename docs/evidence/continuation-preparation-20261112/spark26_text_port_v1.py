"""Unexecuted tensor source for the original Spark model, not a qualified port.

This file has not been imported with Torch. The serving registration and SDK
construction entry points are unconditionally closed in the adapter. Tensor
operations and BF16/FP32 behavior below require native and Neuron validation.
"""
import torch
from torch import nn
from transformers.activations import ACT2FN


class SparkRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * y).to(dtype)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(q, k, cache_positions, rotary_dim, theta):
    # Build frequencies in FP32 each time so Module.to(BF16) cannot corrupt them.
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=q.device).float() / rotary_dim))
    phase = cache_positions.float().unsqueeze(-1) * inv
    phase = torch.cat((phase, phase), -1).unsqueeze(1)
    cos, sin = phase.cos(), phase.sin()
    def rotate(x):
        active = x[..., :rotary_dim].float()
        active = active * cos + rotate_half(active) * sin
        return torch.cat((active.to(x.dtype), x[..., rotary_dim:]), -1)
    return rotate(q), rotate(k)


def repeat_kv(x, groups):
    b, heads, seq, dim = x.shape
    return x[:, :, None, :, :].expand(b, heads, groups, seq, dim).reshape(b, heads * groups, seq, dim)


def prefill_cache(k, v, old_k, old_v, seq_ids, lengths, sliding):
    """Replace every physical cell of the selected row, including invalid cells."""
    cap = old_k.shape[2]
    physical = torch.arange(cap, device=k.device)[None, :]
    if sliding:
        last = lengths[:, None] - 1
        absolute = last - torch.remainder(last - physical, cap)
    else:
        absolute = physical.expand(k.shape[0], cap)
    valid = (absolute >= 0) & (absolute < lengths[:, None])
    gather = absolute.clamp(0, k.shape[2] - 1)[:, None, :, None].expand(k.shape[0], k.shape[1], cap, k.shape[3])
    def replace(x, old):
        selected = torch.gather(x, 2, gather)
        selected = torch.where(valid[:, None, :, None], selected, torch.zeros_like(selected))
        return torch.index_copy(old, 0, seq_ids.long(), selected)
    return replace(k, old_k), replace(v, old_v)


def decode_cache(k, v, old_k, old_v, seq_ids, cache_positions, sliding):
    """Read past chronologically; append current KV separately; update by slot."""
    cap = old_k.shape[2]
    p = cache_positions[:, :1]
    axis = torch.arange(cap, device=k.device)[None, :]
    absolute = p - cap + axis if sliding else axis.expand(p.shape[0], cap)
    physical = torch.remainder(absolute.clamp_min(0), cap)
    gather = physical[:, None, :, None].expand(k.shape[0], k.shape[1], cap, k.shape[3])
    update = (torch.remainder(p, cap) if sliding else p)[:, None, :, None].expand_as(k)
    def read_update(old, active):
        row = torch.index_select(old, 0, seq_ids.long())
        prior = torch.gather(row, 2, gather)
        next_row = torch.scatter(row, 2, update, active)
        next_all = torch.index_copy(old, 0, seq_ids.long(), next_row)
        return torch.cat((prior, active), 2), next_all
    keys, updated_k = read_update(old_k, k)
    values, updated_v = read_update(old_v, v)
    key_positions = torch.cat((absolute, p), -1)
    # The full cache includes the current cell, which still contains old data.
    # Only the separate appended active KV is visible for position p.
    past_valid = (absolute >= 0) & (absolute < p)
    valid = torch.cat((past_valid, torch.ones_like(past_valid[:, :1])), -1)
    return keys, values, key_positions, valid, updated_k, updated_v


class SparkAttention(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind
        self.num_heads = cfg['num_attention_heads']
        self.num_kv_heads = cfg['num_key_value_heads']
        self.dim = cfg['head_dim']
        hidden = cfg['hidden_size']
        self.q_proj = nn.Linear(hidden, self.num_heads * self.dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.dim, bias=False)
        self.g_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.dim, hidden, bias=False)
        rope = cfg['rope_parameters'][kind]
        self.rotary_dim = int(self.dim * rope['partial_rotary_factor'])
        self.theta = rope['rope_theta']

    def forward(self, x, cache_positions, lengths, leading_pad, seq_ids, old_k, old_v, prefill):
        batch, query, _ = x.shape
        q = self.q_proj(x).view(batch, query, self.num_heads, self.dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, query, self.num_kv_heads, self.dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, query, self.num_kv_heads, self.dim).transpose(1, 2)
        gate_score = self.g_proj(x)
        q, k = apply_rope(q, k, cache_positions, self.rotary_dim, self.theta)
        sliding = self.kind == 'sliding_attention'
        if prefill:
            keys, values = k, v
            key_positions = cache_positions
            valid = key_positions < lengths[:, None]
            updated_k, updated_v = prefill_cache(k, v, old_k, old_v, seq_ids, lengths, sliding)
        else:
            keys, values, key_positions, valid, updated_k, updated_v = decode_cache(
                k, v, old_k, old_v, seq_ids, cache_positions, sliding)
        mask = valid[:, None, :] & (key_positions[:, None, :] >= leading_pad[:, None, None])
        mask = mask & (key_positions[:, None, :] <= cache_positions[:, :, None])
        if sliding:
            mask = mask & (key_positions[:, None, :] > cache_positions[:, :, None] - 512)
        keys = repeat_kv(keys, self.num_heads // self.num_kv_heads)
        values = repeat_kv(values, self.num_heads // self.num_kv_heads)
        score = torch.matmul(q, keys.transpose(2, 3)) * (self.dim ** -0.5)
        additive = torch.where(mask[:, None, :, :], torch.zeros_like(score), torch.full_like(score, torch.finfo(score.dtype).min))
        score = score + additive
        score = score - score.max(dim=-1, keepdim=True).values
        prob = torch.softmax(score, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(prob, values)
        gate = torch.sigmoid(gate_score.float()).to(out.dtype).transpose(1, 2).unsqueeze(-1)
        out = out * gate
        out = out.transpose(1, 2).contiguous().view(batch, query, self.num_heads * self.dim)
        return self.out_proj(out), updated_k, updated_v


class SparkMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, inter = cfg['hidden_size'], cfg['intermediate_size']
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)
        self.act = ACT2FN[cfg['hidden_act']]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class SparkLayer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.self_attn = SparkAttention(cfg, kind)
        self.mlp = SparkMLP(cfg)
        self.input_layernorm = SparkRMSNorm(cfg['hidden_size'], cfg['rms_norm_eps'])
        self.post_attention_layernorm = SparkRMSNorm(cfg['hidden_size'], cfg['rms_norm_eps'])

    def forward(self, x, *args):
        residual = x.float()
        normalized = self.input_layernorm(residual).to(self.mlp.gate_proj.weight.dtype)
        attention, k, v = self.self_attn(normalized, *args)
        x = residual + attention
        normalized = self.post_attention_layernorm(x).to(self.mlp.gate_proj.weight.dtype)
        x = x + self.mlp(normalized)
        return x, k, v
