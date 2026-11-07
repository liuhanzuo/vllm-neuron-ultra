# SPDX-License-Identifier: Apache-2.0
"""Opt-in decode-only Mamba cache migration; fresh package device proof is pending."""

import torch
from . import text_core as port
from .helpers import split_last_dim

ORIGINAL_FORWARD=port.Mamba.forward


def read_cache(cache,slots):
    from .nki_read_torch import read_slots_nki
    return read_slots_nki(cache,slots)


def write_cache(cache,updates,slots):
    from .nki_write_torch import store_slots_nki_predicated
    return store_slots_nki_predicated(cache,updates,slots)


def mamba_forward(self, x, cache, slots, positions, lengths, prefill):
    if prefill:
        return ORIGINAL_FORWARD(self, x, cache, slots, positions, lengths, prefill)
    (b, t, _) = x.shape
    (conv_state, ssm_state) = cache
    assert b == 4 and t == 1 and (tuple(conv_state.shape) == (4, 6144, 4))
    assert tuple(ssm_state.shape) == (4, 64, 80, 128)
    assert self.heads == 64 and self.hd == 80 and (self.groups == 4) and (self.sd == 128)
    assert x.dtype == conv_state.dtype == torch.bfloat16 and ssm_state.dtype == torch.float32
    assert slots.shape == (4,)
    (gate, xbc, dt) = split_last_dim(self.in_proj(x), [self.inner, self.conv_dim, self.heads])
    (xbc, conv_updated) = port.conv_decode(xbc.transpose(1, 2), self.conv1d.weight[:, 0], self.conv1d.bias, read_cache(conv_state, slots))
    (values, B, C) = split_last_dim(xbc.transpose(1, 2), [self.inner, self.groups * self.sd, self.groups * self.sd])
    values = values.reshape(b, t, self.heads, self.hd)
    (B, C) = [z.reshape(b, t, self.groups, self.sd) for z in (B, C)]
    (scan, ssm_updated) = port.ssm_decode(values, dt, self.A_log, B, C, self.D, self.dt_bias, read_cache(ssm_state, slots))
    normalized = self.norm(scan.reshape(b, t, self.inner), gate).to(x.dtype)
    updated = (write_cache(conv_state, conv_updated, slots), write_cache(ssm_state, ssm_updated, slots))
    return (self.out_proj(normalized), updated)


def install():
    from .identity import require, selected_variant
    selected_variant()
    if port.Mamba.forward is mamba_forward:
        return
    require(port.Mamba.forward is ORIGINAL_FORWARD, "Another Mamba patch is already installed")
    port.Mamba.forward = mamba_forward
