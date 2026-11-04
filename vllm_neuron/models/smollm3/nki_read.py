# SPDX-License-Identifier: Apache-2.0
"""Read serving slots using DMA only; negative slot IDs select slot zero.

Contract: capacity four, one to four rows, width divisible by 128, int32 slots
strictly less than capacity. Positive out-of-range slot IDs are not supported.
This experimental kernel has no model registration or serving integration.
"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit(platform_target='trn1')
def slot_cache_read(cache, slots):
    capacity, width = cache.shape
    rows = slots.shape[0]
    assert capacity == 4 and 1 <= rows <= 4 and width % 128 == 0
    groups = width // 128
    source = cache.reshape((capacity * groups, 128))
    output = nl.ndarray((rows * groups, 128), dtype=cache.dtype, buffer=nl.shared_hbm)
    ip = nl.arange(128)[:, None]
    jf = nl.arange(128)[None, :]
    for row in nl.static_range(rows):
        slot = nl.maximum(nl.load(slots[row + 0 * ip]), 0)
        for block in nl.affine_range((groups + 127) // 128):
            group = block * 128 + ip
            value = nl.load(source[slot * groups + group, jf], mask=group < groups)
            nl.store(output[row * groups + group, jf], value=value, mask=group < groups)
    return output.reshape((rows, width))
