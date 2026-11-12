# SPDX-License-Identifier: Apache-2.0
"""The measured last-axis split correction, with no runtime AST rewriting."""


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
