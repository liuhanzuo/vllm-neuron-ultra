"""Observer-free NKI 0.5 selector source, unexecuted and unqualified.

Only make_kernel() imports NKI, after a caller's separate serial admission.
The plain function is suitable for a future nki.simulate call. No JIT/trace
bridge, writeback shim, model installation or execution is provided here.
"""


def make_kernel():
    import nki.language as nl
    import nki.isa as nisa

    def kernel(cache, updates, slots):
        capacity, width = cache.shape
        rows = updates.shape[0]
        assert capacity in (4, 8, 16) and 1 <= rows <= capacity and width % 128 == 0
        assert cache.dtype in (nl.bfloat16, nl.float32) and updates.dtype == cache.dtype
        assert slots.dtype == nl.int32 and slots.shape == (rows,)
        output = nl.ndarray((capacity, width), dtype=cache.dtype, buffer=nl.shared_hbm)
        for destination in nl.static_range(capacity):
            selected = nl.full((1, 1), 0, dtype=nl.int32, buffer=nl.sbuf)
            for row in nl.static_range(rows):
                mapping = nl.load(slots[row:row + 1].reshape((1, 1)))
                candidate = nl.full((1, 1), row + 1, dtype=nl.int32, buffer=nl.sbuf)
                selected = nl.where(nl.equal(mapping, destination, dtype=nl.uint8), candidate, selected, dtype=nl.int32)
            source_row = nl.maximum(nl.subtract(selected, 1, dtype=nl.int32), 0)
            predicate = nl.add(nl.zeros((1, 128), dtype=nl.uint8), selected, dtype=nl.uint8)
            source = destination
            for group in nl.affine_range(width // 128):
                column = group * 128
                loaded = nl.load(cache[source:source + 1, column:column + 128])
                prior = nl.ndarray((1, 128), dtype=cache.dtype, buffer=nl.sbuf)
                prior[:, :] = loaded
                view = updates.ap(pattern=[[width, 1], [1, 128]], offset=column,
                                  scalar_offset=source_row, indirect_dim=0)
                replacement_load = nl.load(view)
                replacement = nl.ndarray((1, 128), dtype=cache.dtype, buffer=nl.sbuf)
                replacement[:, :] = replacement_load
                nisa.tensor_copy_predicated(src=replacement, dst=prior, predicate=predicate)
                nl.store(output[destination:destination + 1, column:column + 128], value=prior)
        return output

    return kernel
