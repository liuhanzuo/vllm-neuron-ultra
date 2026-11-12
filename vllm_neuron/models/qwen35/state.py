# SPDX-License-Identifier: Apache-2.0
"""Clear the existing Qwen state buffers without initializing or loading graphs."""
from collections import Counter
import hashlib
from .identity import require
EXPECTED = {((4, 3072, 4), 'torch.bfloat16'): 18,
    ((4, 8, 128, 128), 'torch.float32'): 18,
    ((4, 1, 4096, 256), 'torch.bfloat16'): 12}


def binding_metadata(nxd_model):
    states = nxd_model.state
    require(len(states) == 2, "Two Qwen3.5 state ranks required")
    records = []
    for rank, tensors in enumerate(states):
        require(set(tensors) == {'kv_mgr.past_key_values.' + str(i) for i in range(48)},
                "Original 48 state names per rank required")
        for name, value in sorted(tensors.items()):
            require(value.device.type == 'privateuseone' and value.device.index == rank,
                    "Original Qwen3.5 state device/rank required")
            records.append(dict(rank=rank, name=name, shape=list(value.shape), dtype=str(value.dtype),
                device=str(value.device), tensor_impl=int(value._cdata)))
    for rank in (0, 1):
        require(Counter((tuple(row['shape']), row['dtype']) for row in records if row['rank'] == rank) == EXPECTED,
                "Qwen3.5 state shapes/dtypes changed")
    require(len(records) == len({row['tensor_impl'] for row in records}) == 96,
            "All 96 distinct state tensor bindings required")
    return records


def clear_states(nxd_model, expected_bindings=None):
    import torch
    before = binding_metadata(nxd_model)
    if expected_bindings is not None:
        require(before == expected_bindings, "State bindings changed before cold clear")
    records = []
    for rank, tensors in enumerate(nxd_model.state):
        for name, value in sorted(tensors.items()):
            old = value.detach().cpu().contiguous()
            nonzero = int(torch.count_nonzero(old.view(torch.uint8)))
            old_digest = hashlib.sha256(old.view(torch.uint8).numpy().tobytes()).hexdigest()
            zeros = torch.zeros(tuple(value.shape), dtype=value.dtype, device='cpu')
            # The installed NxDModel.write_to_neuron_buffer uses this same copy operation.
            value.copy_(zeros)
            require(int(torch.count_nonzero(value.detach().cpu().contiguous().view(torch.uint8))) == 0,
                    "In-place Qwen3.5 clear did not produce raw zero bytes")
            records.append(dict(rank=rank, name=name, shape=list(value.shape), dtype=str(value.dtype),
                bytes_written=zeros.numel() * zeros.element_size(), before_nonzero_bytes=nonzero,
                before_sha256=old_digest))
            del old, zeros
    after = binding_metadata(nxd_model)
    require(before == after and len(records) == 96, "Cold clear changed state tensor bindings")
    return dict(complete=True, states_written=96, bindings_unchanged=True, bindings=after,
        states_nonzero_before_clear=sum(row['before_nonzero_bytes'] > 0 for row in records),
        written_bytes=sum(row['bytes_written'] for row in records), rows=records,
        graph_initialization_called=False)

