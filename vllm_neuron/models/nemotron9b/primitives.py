# SPDX-License-Identifier: Apache-2.0
"""Functional Nemotron-H Mamba2 operations, with explicit serving state.

Numerical baseline: pinned native HF CPU chunk/step arithmetic, with runtime
dt_limit=(0, inf) as in the original checkpoint and vLLM. In BF16 this is not
bitwise-equivalent to CUDA's FP32 selective-update arithmetic. State is FP32.
No shared Qwen/other-family source is modified by this implementation.
"""
import torch
import torch.nn.functional as F


def grouped_gated_norm(x, gate, weight, group_size, eps):
    dtype = x.dtype
    gated = x.float() * F.silu(gate.float())
    groups = gated.reshape(*gated.shape[:-1], -1, group_size)
    normalized = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + eps)
    # HF/vLLM native group norm rounds before multiplying its norm weight.
    return normalized.reshape_as(gated).to(dtype) * weight


def conv_prefill(x, weight, bias, lengths):
    """x[B,C,T], weight[C,K]; cache keeps the last VALID projected tokens."""
    b, c, t = x.shape
    width = weight.shape[-1]
    padded = F.pad(x, (width-1, 0)).float()
    out = padded[:, :, :t] * weight[:, 0].float()[None, :, None]
    for j in range(1, width):
        out = out + padded[:, :, j:j+t] * weight[:, j].float()[None, :, None]
    if bias is not None:
        out = out + bias.float()[None, :, None]
    out = F.silu(out.to(x.dtype).float()).to(x.dtype)
    positions = lengths[:, None] - width + torch.arange(width, device=x.device)[None]
    cache = x.gather(2, positions.clamp(0, t-1)[:, None].expand(b, c, width))
    cache = torch.where((positions >= 0)[:, None], cache, torch.zeros_like(cache))
    return out, cache


def conv_decode(x, weight, bias, cache):
    assert x.shape[-1] == 1
    cache = torch.cat([cache[:, :, 1:], x], dim=-1)
    out = (cache.float() * weight.float()[None]).sum(-1, keepdim=True)
    if bias is not None:
        out = out + bias.float()[None, :, None]
    return F.silu(out.to(x.dtype).float()).to(x.dtype), cache


def _group_heads(x, heads):
    b, t, groups, state_dim = x.shape
    return x[:, :, :, None].expand(b, t, groups, heads//groups, state_dim).reshape(b, t, heads, state_dim)


def ssm_prefill(x, dt, a_log, B, C, D, dt_bias, state, lengths, chunk_size=128):
    """Chunked SSD. x[B,T,H,P], B/C[B,T,G,N], state[B,H,P,N].

    Returns FP32 scan output (as HF's CPU chunk path) and FP32 final state.
    Invalid right-padding has zero dt and cannot decay the recurrent state.
    Matmuls avoid materializing B*T*T*H*P*N broadcast products.
    """
    b, t, h, p = x.shape
    valid = torch.arange(t, device=x.device)[None] < lengths[:, None]
    step = F.softplus(dt + dt_bias).float()
    step = torch.where(valid[..., None], step, torch.zeros_like(step))
    xf = x.float()
    values = xf * step[..., None]
    keys, queries = _group_heads(B.float(), h), _group_heads(C.float(), h)
    log_decay = step * -a_log.float().exp()
    pad = (-t) % chunk_size
    values, keys, queries = [F.pad(z.transpose(1, 2), (0, 0, 0, pad)) for z in (values, keys, queries)]
    values, keys, queries = [z.reshape(b, h, -1, chunk_size, z.shape[-1]) for z in (values, keys, queries)]
    decay = F.pad(log_decay.transpose(1, 2), (0, pad)).reshape(b, h, -1, chunk_size)
    cumulative = decay.cumsum(-1)
    pos = torch.arange(chunk_size, device=x.device)
    strict_lower = pos[:, None] > pos[None, :]
    lower = pos[:, None] >= pos[None, :]
    # Stable segment sums, including very negative A. Subtracting two large
    # prefix sums would lose short-distance decay precision.
    segments = torch.where(strict_lower, decay[..., :, None].expand(*decay.shape, chunk_size), 0.).cumsum(-2)
    pair_decay = torch.where(lower, segments, -float('inf')).exp()
    within = (queries @ keys.transpose(-1, -2)) * pair_decay
    running = state.float()
    outputs = []
    end_keys = keys * (cumulative[..., -1:] - cumulative).exp()[..., None]
    prefix_queries = queries * cumulative.exp()[..., None]
    for i in range(values.shape[2]):
        outputs.append(within[:, :, i] @ values[:, :, i] + prefix_queries[:, :, i] @ running.transpose(-1, -2))
        running = running * cumulative[:, :, i, -1].exp()[..., None, None] + values[:, :, i].transpose(-1, -2) @ end_keys[:, :, i]
    out = torch.stack(outputs, dim=2).reshape(b, h, -1, p)[:, :, :t].transpose(1, 2)
    out = out + xf * D.float()[None, None, :, None]
    return torch.where(valid[..., None, None], out, torch.zeros_like(out)), running


def ssm_decode(x, dt, a_log, B, C, D, dt_bias, state):
    """One step preserving the pinned HF CPU BF16 intermediate rounding."""
    assert x.shape[1] == 1
    b, _, h, p = x.shape
    step = F.softplus(dt[:, 0] + dt_bias)
    a = -a_log.float().exp()
    decay = (step.float() * a).exp()[..., None, None]
    keys, queries = _group_heads(B, h)[:, 0], _group_heads(C, h)[:, 0]
    discrete_B = step[..., None] * keys
    addition = discrete_B[:, :, None] * x[:, 0, :, :, None]
    updated = state.float()*decay + addition
    out = torch.bmm(updated.to(C.dtype).reshape(b*h, p, -1), queries.reshape(b*h, -1, 1)).reshape(b, h, p)
    out = out + x[:, 0]*D[None, :, None]
    return out[:, None].to(x.dtype), updated
