# SPDX-License-Identifier: Apache-2.0
"""Trace-only wrapper for the isolated cache-read candidate."""
import torch
from .nki_read import slot_cache_read


def read_slots_nki(cache, slots):
    output = slot_cache_read(cache.reshape(cache.shape[0], -1), slots.to(torch.int32))
    return output.reshape((slots.shape[0],) + tuple(cache.shape[1:]))
