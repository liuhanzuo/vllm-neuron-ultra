# SPDX-License-Identifier: Apache-2.0
"""Pinned generic helpers; no Qwen architecture registration or model import."""
import torch
from torch import nn
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM

def store_slots(cache, updates, slots):
    """Functional writes also preserve inactive slots (ID -1), without collisions."""
    indices = torch.arange(cache.shape[0], device=cache.device)
    for row in range(updates.shape[0]):
        selected = indices == slots[row]
        cache = torch.where(selected.reshape((-1,) + (1,) * (cache.ndim - 1)), updates[row:row + 1], cache)
    return cache

def partitioned_parameter(shape, stride=1, dtype=torch.float32, device=None):
    # NxD's custom meta initialization cannot reconstruct a CPU Parameter
    # with TP attributes passed as constructor kwargs. Use the device of an
    # already registered projection so registration keeps this object intact.
    p = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device), requires_grad=False)
    set_tensor_model_parallel_attributes(p, True, 0, stride,
                                        num_partitions=parallel_state.get_tensor_model_parallel_size())
    return p

def boundaries(sizes):
    assert len(sizes) >= 2 and all(type(n) is int and n > 0 for n in sizes)
    total, indices = 0, []
    for width in sizes[:-1]:
        total += width
        indices.append(total)
    return tuple(indices)

def split_last_dim(value, sizes):
    import torch
    sizes = tuple(sizes)
    indices = boundaries(sizes)
    assert sum(sizes) == value.shape[-1]
    return torch.tensor_split(value, indices, dim=-1)

class ExplicitPositionForCausalLM(NeuronBaseForCausalLM):
    def forward(self, input_ids=None, seq_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None:
            raise ValueError('Explicit positions are required for hybrid-state serving')
        if input_ids.shape[1] == 1 and bool((position_ids == 0).all()):
            input_ids = F.pad(input_ids, (0, 1), value=self.config.pad_token_id)
            position_ids = F.pad(position_ids, (0, 1), value=0)
            attention_mask = torch.tensor([1, 0], dtype=position_ids.dtype).expand(input_ids.shape[0], 2)
        return super().forward(input_ids=input_ids, seq_ids=seq_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
