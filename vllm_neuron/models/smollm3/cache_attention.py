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
"""Per-model decode cache candidate; prefill keeps the original attention."""
import torch
import torch.nn.functional as F
from .text_core import Attention, store_slots


def read_cache(cache,slots):
    from .nki_read_torch import read_slots_nki
    return read_slots_nki(cache,slots)


def write_cache(cache,updates,slots):
    from .nki_write_torch import store_slots_nki_predicated
    return store_slots_nki_predicated(cache,updates,slots)


def read_reference(cache,slots):
    return cache[slots.clamp_min(0)]


def attention_forward(self, x, cache, slots, positions, lengths, prefill):
    if prefill:
        return Attention.forward(self, x, cache, slots, positions, lengths, prefill)
    assert x.shape[0] == 4 and x.shape[1] == 1 and (x.dtype == torch.bfloat16)
    assert self.h == 8 and self.kvh == 2 and (self.dim == 128) and (slots.shape == (4,))
    assert tuple(cache[0].shape) == tuple(cache[1].shape)
    assert cache[0].shape[0] == 4 and cache[0].shape[1] == 2 and (cache[0].shape[3] == 128)
    assert cache[0].shape[2] in (128, 256, 512, 1024, 2048)
    assert cache[0].dtype == cache[1].dtype == torch.bfloat16
    (b, t, _) = x.shape
    q = self.q_proj(x).reshape(b, t, self.h, self.dim).transpose(1, 2)
    k = self.k_proj(x).reshape(b, t, self.kvh, self.dim).transpose(1, 2)
    v = self.v_proj(x).reshape(b, t, self.kvh, self.dim).transpose(1, 2)
    if self.use_rope:
        phase = positions.float()[..., None] * self.inverse_frequencies.float()
        phase = torch.cat((phase, phase), -1)
        co = phase.cos().to(q.dtype)[:, None]
        si = phase.sin().to(q.dtype)[:, None]

        def rotate(z):
            return z * co + torch.cat((-z[..., self.dim // 2:], z[..., :self.dim // 2]), -1) * si
        (q, k) = (rotate(q), rotate(k))
    (kc, vc) = cache
    if prefill:
        valid = torch.arange(t, device=x.device)[None, :] < lengths[:, None]
        (all_k, all_v) = (k, v)
        (sk, sv) = (F.pad(z * valid[:, None, :, None], (0, 0, 0, kc.shape[2] - t)) for z in (k, v))
    else:
        idx = positions[:, None, :, None].expand(b, self.kvh, 1, self.dim)
        sk = torch.scatter(self.read_cache(kc, slots), 2, idx, k)
        sv = torch.scatter(self.read_cache(vc, slots), 2, idx, v)
        (all_k, all_v) = (sk, sv)
    keys = torch.arange(all_k.shape[2], device=x.device)[None, None, None, :]
    allowed = (keys <= positions[:, None, :, None]) & (keys < lengths[:, None, None, None])
    (all_k, all_v) = (z.repeat_interleave(self.h // self.kvh, 1) for z in (all_k, all_v))
    scores = q @ all_k.transpose(-1, -2) * self.dim ** (-0.5)
    scores = scores + torch.where(allowed, torch.zeros_like(scores), torch.full_like(scores, torch.finfo(scores.dtype).min))
    out = (scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ all_v).transpose(1, 2).reshape(b, t, -1)
    return (self.o_proj(out), (self.write_cache(kc, sk, slots), self.write_cache(vc, sv, slots)))


class NkiAttention(Attention):
    forward = attention_forward
    read_cache = staticmethod(read_cache)
    write_cache = staticmethod(write_cache)


class NkiReadAttention(NkiAttention):
    write_cache = staticmethod(store_slots)


class NkiWriteAttention(NkiAttention):
    read_cache = staticmethod(read_reference)
