"""Copy selected cache values without floating-point arithmetic on either branch."""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit(platform_target='trn1')
def slot_cache_gather_predicated(cache, updates, slots):
    capacity, width = cache.shape
    rows = updates.shape[0]
    assert 1 <= rows <= 4 and 1 <= capacity <= 4
    assert updates.shape[1] == width and slots.shape[0] == rows
    assert width % 128 == 0
    groups = width // 128
    original = cache.reshape((capacity * groups, 128))
    source = updates.reshape((rows * groups, 128))
    output = nl.ndarray((capacity * groups, 128), dtype=cache.dtype, buffer=nl.shared_hbm)
    ip = nl.arange(128)[:, None]
    jf = nl.arange(128)[None, :]
    for destination in nl.static_range(capacity):
        selected = nl.full((128, 1), 0, dtype=nl.int32, buffer=nl.sbuf)
        for row in nl.static_range(rows):
            mapping = nl.load(slots[row + 0 * ip])
            candidate = nl.full((128, 1), row + 1, dtype=nl.int32, buffer=nl.sbuf)
            selected = nl.where(mapping == destination, candidate, selected, dtype=nl.int32)
        source_row = nl.maximum(selected - 1, 0)
        predicate = nl.ndarray((128, 128), dtype=nl.uint8, buffer=nl.sbuf)
        predicate[ip, jf] = nl.add(selected, nl.zeros((128, 128), dtype=nl.uint8), dtype=nl.uint8)
        for block in nl.affine_range((groups + 127) // 128):
            group = block * 128 + ip
            value = nl.ndarray((128, 128), dtype=cache.dtype, buffer=nl.sbuf)
            new = nl.ndarray((128, 128), dtype=cache.dtype, buffer=nl.sbuf)
            value[ip, jf] = nl.load(original[destination * groups + group, jf], mask=group < groups)
            new[ip, jf] = nl.load(source[source_row * groups + group, jf], mask=group < groups)
            nisa.tensor_copy_predicated(src=new, dst=value, predicate=predicate)
            nl.store(output[destination * groups + group, jf], value=value, mask=group < groups)
    return output.reshape((capacity, width))
