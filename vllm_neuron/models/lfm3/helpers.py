# SPDX-License-Identifier: Apache-2.0
"""Generic slot writes and TP/tied-weight helpers extracted from the pinned port.

No Qwen architecture or registration is imported. Source attribution and exact
experiment pins are recorded in provenance.py.
"""
import torch
from torch import nn


def store_slots(cache, updates, slots):
    """Functional writes also preserve inactive slots (ID -1), without collisions."""
    indices = torch.arange(cache.shape[0], device=cache.device)
    for row in range(updates.shape[0]):
        selected = indices == slots[row]
        cache = torch.where(selected.reshape((-1,) + (1,) * (cache.ndim - 1)), updates[row:row + 1], cache)
    return cache


def partitioned_parameter(shape, stride=1, dtype=torch.float32, device=None):
    from neuronx_distributed.parallel_layers import parallel_state
    from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes
    # NxD's custom meta initialization cannot reconstruct a CPU Parameter
    # with TP attributes passed as constructor kwargs. Use the device of an
    # already registered projection so registration keeps this object intact.
    p = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device), requires_grad=False)
    set_tensor_model_parallel_attributes(p, True, 0, stride,
                                        num_partitions=parallel_state.get_tensor_model_parallel_size())
    return p
