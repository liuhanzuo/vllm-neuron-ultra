"""Inert explicit-control successor for the NKI 0.5 transport kernel.

The two integer control expressions use the installed public tensor APIs.
Payload movement, observer coverage, dtype and byte-oracle remain frozen.
No package import or execution occurs until an admitted caller makes a kernel.
"""
from lfm3_nki05_transport_diagnostic_v1 import *


def make_kernel(operation, observe):
    """New SDK diagnostic source; observer returns no data and never mutates tensors."""
    assert operation in ('identity', 'copy', 'permutation', 'selector')
    import nki.language as nl
    import nki.isa as nisa

    def kernel(cache, updates, slots):
        capacity, width = cache.shape
        rows = updates.shape[0]
        assert capacity in (4, 8, 16) and 1 <= rows <= capacity and width % 128 == 0
        assert cache.dtype in (nl.bfloat16, nl.float32) and updates.dtype == cache.dtype
        assert slots.dtype == nl.int32 and slots.shape == (rows,)
        observe('ingress_cache', cache)
        observe('ingress_updates', updates)
        observe('ingress_slots', slots)
        if operation == 'identity':
            observe('pre_return', cache)
            return cache
        output = nl.ndarray((capacity, width), dtype=cache.dtype, buffer=nl.shared_hbm)
        for destination in nl.static_range(capacity):
            if operation == 'selector':
                selected = nl.full((1, 1), 0, dtype=nl.int32, buffer=nl.sbuf)
                for row in nl.static_range(rows):
                    mapping = nl.load(slots[row:row + 1].reshape((1, 1)))
                    candidate = nl.full((1, 1), row + 1, dtype=nl.int32, buffer=nl.sbuf)
                    selected = nl.where(nl.equal(mapping, destination, dtype=nl.uint8), candidate, selected, dtype=nl.int32)
                source_row = nl.maximum(nl.subtract(selected, 1, dtype=nl.int32), 0)
                predicate = nl.add(nl.zeros((1, 128), dtype=nl.uint8), selected, dtype=nl.uint8)
            source = capacity - 1 - destination if operation == 'permutation' else destination
            for group in nl.affine_range(width // 128):
                column = group * 128
                loaded = nl.load(cache[source:source + 1, column:column + 128])
                observe('load_cache', loaded, destination, group)
                prior = nl.ndarray((1, 128), dtype=cache.dtype, buffer=nl.sbuf)
                prior[:, :] = loaded
                observe('assign_cache', prior, destination, group)
                if operation == 'selector':
                    # The new public API expresses the old typed dynamic gather.
                    view = updates.ap(pattern=[[width, 1], [1, 128]], offset=column,
                                      scalar_offset=source_row, indirect_dim=0)
                    replacement_load = nl.load(view)
                    observe('load_updates', replacement_load, destination, group)
                    replacement = nl.ndarray((1, 128), dtype=cache.dtype, buffer=nl.sbuf)
                    replacement[:, :] = replacement_load
                    observe('assign_updates', replacement, destination, group)
                    nisa.tensor_copy_predicated(src=replacement, dst=prior, predicate=predicate)
                    observe('predicated_copy', prior, destination, group)
                nl.store(output[destination:destination + 1, column:column + 128], value=prior)
        observe('pre_return', output)
        return output

    return kernel


def run_case(nki, name, operation, cache, updates, slots):
    from nki.language.tensor import tensor_ap_view
    host = [x.tobytes() for x in (cache, updates, slots)]
    capacity, width = cache.shape
    itemsize, dtype, mapping = cache.dtype.itemsize, cache.dtype.name, slots.tolist()
    stride = width * itemsize
    if operation == 'selector':
        expected = byte_oracle(host[0], host[1], mapping, capacity, width, itemsize)
    elif operation == 'permutation':
        expected = b''.join(host[0][i * stride:(i + 1) * stride] for i in reversed(range(capacity)))
    else:
        expected = host[0]
    events = []

    def observe(stage, tensor, destination=None, group=None):
        # All observed tensors are complete allocations, not partial views.
        # Compare backing bytes and view extraction separately, without a cast.
        view = tensor_ap_view(tensor).get_data()
        backing = tensor._storage.data
        events.append((stage, destination, group, view.tobytes(), view.dtype.name,
                       tuple(view.shape), backing.tobytes(), backing.dtype.name))

    actual = nki.simulate(make_kernel(operation, observe))(cache, updates, slots)
    stages, first = {}, None
    for stage, destination, group, raw, actual_dtype, shape, backing, backing_dtype in events:
        if stage.startswith('ingress_'):
            index = ('ingress_cache', 'ingress_updates', 'ingress_slots').index(stage)
            want, want_dtype, want_shape = host[index], (cache, updates, slots)[index].dtype.name, (cache, updates, slots)[index].shape
        elif stage == 'pre_return':
            want, want_dtype, want_shape = expected, dtype, cache.shape
        else:
            if stage in ('load_updates', 'assign_updates'):
                winners = [row for row, slot in enumerate(mapping) if slot == destination]
                row = winners[-1] if winners else 0
                source = host[1][row * stride:(row + 1) * stride]
            elif stage == 'predicated_copy':
                source = expected[destination * stride:(destination + 1) * stride]
            else:
                row = capacity - 1 - destination if operation == 'permutation' else destination
                source = host[0][row * stride:(row + 1) * stride]
            want = source[group * 128 * itemsize:(group + 1) * 128 * itemsize]
            want_dtype, want_shape = dtype, (1, 128)
        row = stages.setdefault(stage, dict(observations=0, view_failures=0, backing_failures=0))
        row['observations'] += 1
        view_ok = strict_equal(raw, actual_dtype, shape, want, want_dtype, want_shape)
        backing_ok = backing == want and backing_dtype == want_dtype
        row['view_failures'] += not view_ok
        row['backing_failures'] += not backing_ok
        if not (view_ok and backing_ok) and first is None:
            first = dict(stage=stage, destination=destination, group=group,
                         view_dtype=actual_dtype, backing_dtype=backing_dtype,
                         view_shape=list(shape), expected_dtype=want_dtype,
                         expected_shape=list(want_shape), view=difference(raw, want, itemsize),
                         backing=difference(backing, want, itemsize))
    raw = actual.tobytes()
    unchanged = host == [x.tobytes() for x in (cache, updates, slots)]
    coverage = {key: row['observations'] for key, row in stages.items()} == expected_observations(operation, capacity, width)
    passed = coverage and first is None and unchanged and strict_equal(raw, actual.dtype.name, actual.shape, expected, dtype, cache.shape)
    return dict(name=name, operation=operation, passed=passed, shape=list(cache.shape), dtype=dtype,
                inputs_unchanged=unchanged, observer_coverage=coverage, input_sha256=[sha(x) for x in host],
                expected_sha256=sha(expected), actual_sha256=sha(raw),
                actual_dtype=actual.dtype.name, actual_shape=list(actual.shape),
                returned_difference=difference(raw, expected, itemsize), stages=stages,
                first_boundary_divergence=first)
