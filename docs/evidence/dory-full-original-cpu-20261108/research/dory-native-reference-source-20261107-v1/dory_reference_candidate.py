# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modified 2026-11-07: standalone TP1 CPU reference candidate and explicit request state.
"""UNVALIDATED source candidate. Do not treat source checks as numeric evidence.

Official mathematical source: vLLM 75fc05a53a366068f534a61ab3ee6b50485419f2,
vllm/model_executor/models/dory.py. Exact line mapping is in provenance.json.
No downloading, checkpoint loader, tokenizer, scheduler, or serving entrypoint.
Only a future serial validation task may import/execute this file.
"""
from types import MappingProxyType

import torch
import torch.nn.functional as F

from dory_contract import DoryContract, RequestCursor

EPS = 1e-6


def justnorm(x, dim=-1):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)).to(x.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    ya, yb = justnorm(a).float(), justnorm(b).float()
    lr = (alpha.float() * alpha_scale).abs()
    return justnorm(ya + lr * (yb - ya)).to(a.dtype)


def neox_rope_candidate(q, k, positions, theta):
    """Default full-dimensional NeoX RoPE; backend rounding requires parity gate.

    Upstream Dory delegates to get_rope; this implementation is a local replacement,
    not a claim that the uninspected upstream rotary backend is bitwise equivalent.
    """
    dim = q.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=q.device) / dim))
    phase = positions.float()[:, None] * inv[None, :]
    cosine, sine = phase.cos().to(q.dtype)[:, None, :], phase.sin().to(q.dtype)[:, None, :]

    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)

    return rotate(q), rotate(k)


class DoryReferenceCandidate:
    """Full forward candidate using supplied unpacked CPU tensors, shared by loops.

    It retains complete histories even for SWA, then masks them. This deliberately
    transparent reference is NOT a memory-optimized adapter and proves no fit.
    Caller must not mutate the supplied tensors during the reference lifetime.
    No sampler is provided: tokenization/generation policy must be qualified separately.
    """
    def __init__(self, config, weights, *, tensor_parallel_size=1, float32_fixture=False):
        self.config = DoryContract.from_mapping(config, tensor_parallel_size=tensor_parallel_size)
        weights = dict(weights)
        if any(not isinstance(t, torch.Tensor) for t in weights.values()):
            raise ValueError("All weights must be supplied tensors")
        self.config.validate_weight_shapes({name: tensor.shape for name, tensor in weights.items()})
        expected_dtype = torch.float32 if float32_fixture else torch.bfloat16
        if any(t.device.type != "cpu" or t.dtype != expected_dtype for t in weights.values()):
            raise ValueError("Require homogeneous CPU BF16 weights, or explicit FP32 fixture mode")
        self.weights = MappingProxyType({name: tensor.detach() for name, tensor in weights.items()})
        self.dtype = expected_dtype
        self.slots, self.epochs = {}, {}

    def start_request(self, slot_id, request_id):
        if type(slot_id) is not int or slot_id < 0:
            raise ValueError("slot_id must be a nonnegative integer")
        epoch = self.epochs.get(slot_id, 0) + 1
        cursor = RequestCursor(self.config, epoch, request_id)
        previous = self.slots.pop(slot_id, None)
        if previous is not None:
            previous[0].abort()
            previous[1].clear()
        self.epochs[slot_id] = epoch
        self.slots[slot_id] = (cursor, {})
        return epoch

    def release_request(self, slot_id, epoch, request_id):
        cursor, histories = self.slots[slot_id]
        cursor.check_owner(epoch, request_id)
        cursor.abort()
        histories.clear()
        del self.slots[slot_id]

    def _attention(self, h, positions, layer, loop, cursor, histories):
        c, w = self.config, self.weights
        p = f"backbone.layers.{layer}.mixer"
        length = h.shape[0]
        q = F.linear(h, w[p + ".q_proj.weight"]).view(length, c.num_attention_heads, c.head_dim)
        k = F.linear(h, w[p + ".k_proj.weight"]).view(length, c.num_key_value_heads, c.head_dim)
        v = F.linear(h, w[p + ".v_proj.weight"]).view(length, c.num_key_value_heads, c.head_dim)
        gate = F.linear(h, w[p + ".g_proj.weight"])
        q, k = neox_rope_candidate(q, k, positions, dict(c.rope_thetas)[c.layer_types[layer]])
        sqk = (w[p + ".sqk"].float() * (c.head_dim ** 0.5 / c.initializer_range)).view(1, c.num_attention_heads, c.head_dim)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype).transpose(0, 1)
        k, v = justnorm(k).transpose(0, 1), v.transpose(0, 1)
        cache_id = loop * c.num_hidden_layers + layer
        cursor.record(cache_id)
        previous = histories.get(cache_id)
        if previous is None:
            if cursor.length != 0:
                raise ValueError("Missing history after committed prefill")
            keys, values = k, v
        else:
            if previous[0].shape[1] != cursor.length or previous[1].shape[1] != cursor.length:
                raise ValueError("History length disagrees with committed cursor")
            keys = torch.cat((previous[0], k), dim=1)
            values = torch.cat((previous[1], v), dim=1)
        histories[cache_id] = (keys, values)
        key_pos = torch.arange(keys.shape[1], device=h.device)
        mask = key_pos[None, :] <= positions[:, None]
        if c.layer_types[layer] == "sliding_attention":
            mask &= key_pos[None, :] > positions[:, None] - c.sliding_window
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask,
                                            scale=1.0 / c.head_dim ** 0.5,
                                            dropout_p=0.0, enable_gqa=True)
        out = out.transpose(0, 1).reshape(length, c.num_attention_heads * c.head_dim)
        gate_init = c.attention_output_gate_logit_scale_init
        if gate_init is None:
            gate_init = c.hidden_size ** 0.5
        gate_scale = (w[p + ".s_gate"].float() * (gate_init ** 0.5 / c.initializer_range)).square()
        out = (out.float() * torch.sigmoid(gate.float() * gate_scale)).to(out.dtype)
        return F.linear(justnorm(out), w[p + ".o_proj.weight"])

    def _layer(self, h, positions, layer, loop, cursor, histories):
        c, w = self.config, self.weights
        base = f"backbone.layers.{layer}"
        if c.layer_types[layer] == "mlp":
            p = base + ".mixer"
            g, u = F.linear(h, w[p + ".gate_proj.weight"]), F.linear(h, w[p + ".up_proj.weight"])
            g = (g.float() * (w[p + ".suv_gate"].float() * c.hidden_size ** 0.5)).to(g.dtype)
            u = (u.float() * (w[p + ".suv_up"].float() * c.hidden_size ** 0.5)).to(u.dtype)
            m = (F.silu(g.float()) * u.float()).to(h.dtype)
            out = F.linear(justnorm(m), w[p + ".down_proj.weight"])
            alpha = w[base + ".mlp_alpha"]
        else:
            out = self._attention(h, positions, layer, loop, cursor, histories)
            alpha = w[base + ".attn_alpha"]
        return slerp_alpha(h, out, alpha, c.ngpt_alpha_init / c.initializer_range)

    @torch.inference_mode()
    def forward(self, slot_id, epoch, request_id, input_ids, positions):
        """Returns (all chunk hidden states, all chunk FP32 logits); no hidden reset.

        A failed transaction invalidates the request and clears all 56 histories.
        The next attempt must explicitly start a fresh request at position zero.
        This serial reference must not be called concurrently.
        """
        cursor, histories = self.slots[slot_id]
        cursor.check_owner(epoch, request_id)
        positions = tuple(positions)
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1 or input_ids.device.type != "cpu" or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a one-dimensional CPU int64 tensor")
        if input_ids.numel() != len(positions) or not positions:
            raise ValueError("Token and position lengths must match and be nonempty")
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.config.vocab_size:
            raise ValueError("Token outside checkpoint vocabulary")
        cursor.begin(epoch, request_id, positions)
        try:
            pos = torch.tensor(positions, dtype=torch.long, device="cpu")
            c = self.config
            h = F.embedding(input_ids, self.weights["backbone.embeddings.weight"])
            for layer in range(c.n_input_layers):
                h = self._layer(h, pos, layer, 0, cursor, histories)
            input_h, state = h, h
            end = c.n_input_layers + c.n_recurrent_layers
            for loop in range(c.n_recurrent_loops):
                h = justnorm(state + input_h)
                for layer in range(c.n_input_layers, end):
                    h = self._layer(h, pos, layer, loop, cursor, histories)
                state = justnorm(h)
            h = state
            for layer in range(end, c.num_hidden_layers):
                h = self._layer(h, pos, layer, 0, cursor, histories)
            logits = F.linear(h, self.weights["lm_head.weight"]).float()
            logits = logits * (self.weights["lm_head.logit_scale"].float() / c.initializer_range)
            cursor.commit()
            return h, logits
        except BaseException:
            cursor.abort()
            histories.clear()
            raise
