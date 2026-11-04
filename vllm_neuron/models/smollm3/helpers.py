# SPDX-License-Identifier: Apache-2.0
"""Original functional slot writes, extracted without the Qwen model import."""
import torch


def store_slots(cache, updates, slots):
    """Functional writes also preserve inactive slots (ID -1), without collisions."""
    indices = torch.arange(cache.shape[0], device=cache.device)
    for row in range(updates.shape[0]):
        selected = indices == slots[row]
        cache = torch.where(selected.reshape((-1,) + (1,) * (cache.ndim - 1)), updates[row:row + 1], cache)
    return cache
