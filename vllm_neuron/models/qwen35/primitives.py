# SPDX-License-Identifier: Apache-2.0
"""Functional Qwen3.5 linear-attention primitives for a Neuron port.

Reference: Hugging Face Transformers, revision
66f51e76efd8542232b8bdd40ade7d1edbaac46f, Qwen3.5 (Apache-2.0).
These functions do not register a model or establish full-model compatibility.
State is explicit, so a serving adapter can gather/scatter independent slots.
"""
import torch
import torch.nn.functional as F


def normalize(x):
    x = x.float()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def delta_decode(query, key, value, g, beta, state):
    """One recurrent step; B,1,H,D inputs and B,H,K,V FP32 state."""
    assert query.shape[1] == 1
    q = normalize(query[:, 0]) / (query.shape[-1] ** 0.5)
    k = normalize(key[:, 0])
    v = value[:, 0].float()
    decayed = state.float() * g[:, 0].float().exp()[..., None, None]
    correction = (v - (decayed * k[..., None]).sum(-2)) * beta[:, 0].float()[..., None]
    updated = decayed + k[..., None] * correction[..., None, :]
    output = (updated * q[..., None]).sum(-2)
    return output[:, None].to(value.dtype), updated


def unit_lower_inverse(matrix):
    """Invert a unit lower triangular matrix by block forward substitution.

    No torch.linalg operation or in-place indexed update enters the trace.
    The diagonal is implicitly one, regardless of the supplied diagonal.
    Unlike a geometric series, block substitution avoids enormous cancelling
    powers for aligned keys and near-zero decay.
    """
    n = matrix.shape[-1]
    assert n > 0 and n & (n - 1) == 0
    inverse = torch.ones((*matrix.shape[:-2], n, 1, 1), dtype=matrix.dtype, device=matrix.device)
    width = 1
    while width < n:
        lower = torch.stack(
            [matrix[..., start + width:start + 2 * width, start:start + width]
             for start in range(0, n, 2 * width)], dim=-3)
        left, right = inverse[..., 0::2, :, :], inverse[..., 1::2, :, :]
        lower_inverse = -(right @ lower) @ left
        top = torch.cat((left, torch.zeros_like(left)), dim=-1)
        bottom = torch.cat((lower_inverse, right), dim=-1)
        inverse = torch.cat((top, bottom), dim=-2)
        width *= 2
    return inverse.squeeze(-3)


def delta_prefill(query, key, value, g, beta, state, lengths, chunk_size=64):
    """Chunked delta rule with right-padding-safe final state.

    lengths[B] counts valid prefix tokens. Padding must have zero learning
    rate AND zero log-decay; zero hidden states alone would decay the cache.
    """
    dtype = value.dtype
    b, t, h, d = query.shape
    valid = torch.arange(t, device=query.device)[None, :] < lengths[:, None]
    q = normalize(query).transpose(1, 2) / (d ** 0.5)
    k = normalize(key).transpose(1, 2)
    v = value.float().transpose(1, 2)
    decay = torch.where(valid[..., None], g.float(), torch.zeros_like(g.float())).transpose(1, 2)
    learning = torch.where(valid[..., None], beta.float(), torch.zeros_like(beta.float())).transpose(1, 2)
    pad = (-t) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    decay, learning = (F.pad(x, (0, pad)) for x in (decay, learning))
    k_beta, v_beta = k * learning[..., None], v * learning[..., None]
    q, k, k_beta, v_beta = (
        x.reshape(b, h, -1, chunk_size, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    cumulative = decay.reshape(b, h, -1, chunk_size).cumsum(-1)
    positions = torch.arange(chunk_size, device=query.device)
    causal = positions[:, None] >= positions[None, :]
    pairwise = torch.where(causal, cumulative[..., :, None] - cumulative[..., None, :],
                           torch.full_like(cumulative[..., :, None] - cumulative[..., None, :], -float('inf'))).exp()
    lower = (k_beta @ k.transpose(-1, -2)) * pairwise
    inverse = unit_lower_inverse(lower)
    new_values = inverse @ v_beta
    read_keys = inverse @ (k_beta * cumulative.exp()[..., None])
    within = (q @ k.transpose(-1, -2)) * pairwise
    q = q * cumulative.exp()[..., None]
    end_keys = k * (cumulative[..., -1:] - cumulative).exp()[..., None]
    end_decay = cumulative[..., -1].exp()[..., None, None]
    running = state.float()
    chunks = []
    for i in range(q.shape[2]):
        delta = new_values[:, :, i] - read_keys[:, :, i] @ running
        chunks.append(q[:, :, i] @ running + within[:, :, i] @ delta)
        running = running * end_decay[:, :, i] + end_keys[:, :, i].transpose(-1, -2) @ delta
    output = torch.stack(chunks, dim=2).reshape(b, h, -1, value.shape[-1])[:, :, :t].transpose(1, 2)
    output = torch.where(valid[..., None, None], output, torch.zeros_like(output))
    return output.to(dtype), running


def conv_prefill(x, weight, lengths):
    """Depthwise causal SiLU convolution, plus last VALID projected tokens."""
    b, c, t = x.shape
    width = weight.shape[-1]
    padded = F.pad(x, (width - 1, 0)).float()
    accumulated = padded[:, :, :t] * weight[:, 0].float()[None, :, None]
    for j in range(1, width):
        accumulated = accumulated + padded[:, :, j:j + t] * weight[:, j].float()[None, :, None]
    # HF conv1d rounds to the input dtype before SiLU.
    # Preserve HF's FP32 activation arithmetic after the BF16 convolution
    # rounding. BF16 sigmoid/multiply intermediates on Neuron lose precision.
    output = F.silu(accumulated.to(x.dtype).float()).to(x.dtype)
    indices = lengths[:, None] - width + torch.arange(width, device=x.device)[None, :]
    gathered = torch.gather(x, 2, indices.clamp(0, t - 1)[:, None, :].expand(b, c, width))
    cache = torch.where((indices >= 0)[:, None, :], gathered, torch.zeros_like(gathered))
    return output, cache


def conv_decode(x, weight, cache):
    assert x.shape[-1] == 1
    updated = torch.cat((cache[:, :, 1:], x), dim=-1)
    output = (updated.float() * weight.float()[None, :, :]).sum(-1, keepdim=True).to(x.dtype)
    return F.silu(output.float()).to(x.dtype), updated
