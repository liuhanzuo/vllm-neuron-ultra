# SPDX-License-Identifier: Apache-2.0
"""Trace-only wrapper for conditional-copy NKI cache writes."""
import torch
from .nki_write import slot_cache_gather_predicated


def store_slots_nki_predicated(cache, updates, slots):
    result = slot_cache_gather_predicated(cache.reshape(cache.shape[0], -1),
        updates.reshape(updates.shape[0], -1), slots.to(torch.int32))
    return result.reshape(cache.shape)
