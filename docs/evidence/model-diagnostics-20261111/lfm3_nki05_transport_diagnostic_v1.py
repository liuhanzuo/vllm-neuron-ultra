"""Inert, CPU-only NKI 0.5 migration diagnostic; NOT a qualification runner.

Call diagnose(admit) only inside a newly admitted serial worker in the pinned
Python 3.12 environment. admit() must perform the existing root-owned admission
and return True. This module neither acquires/releases ownership nor schedules
work. No native import occurs until after admission. The caller saves the return
value through its existing report writer. There is deliberately no CLI.

The frozen legacy selector is not changed or executed. Its typed movement and
integer last-writer selection are migrated to the new slice/access-pattern API.
Read-only simulator observations never supply values to kernel computation.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent
PREFIX = '/home/ubuntu/venvs/vllm_neuron_053_py312'
SDK = Path(PREFIX) / 'lib/python3.12/site-packages/nki'
PINS = {
    ROOT / 'slot_cache_nki_typed_selector_v2.py': 'bb7ce1c563f0cdfe1e927ed0f82d61e07f76af5de6a2ab53d6bb1b2f98835591',
    ROOT / 'lfm3_batch_slots_extended_v1.py': 'da4bf9aebd9d12f87a841061e15c28d73219ec6b34baa9f0f01006e32a1d2ad8',
    ROOT / 'lfm3_batch_slots_protocol_v1.py': '06400ba9c853c03b73c883cf44b2f0d8e6ee648230d43690258da4097d3c1803',
    ROOT / 'lfm3-typed-writer-cpu-pilot-v5/extended_cpu/00000-00016.json': 'f35cec6fa7835c2aea63ac1698e9facda2389b3c9a64d8a6d3057d75fde76ef8',
    ROOT / 'lfm3-capacity-bit-diagnostic-v1/cases/00000.json': '0f86cd2b55ba9e9c2271c74e3745e5530ea1e415a326930bdd4dd5fd5fb027f8',
    SDK / '_simulator.py': '3c28f10492ef8c6c5413acbd811c059586f7ff16d3ef795430e794c57ed8f119',
    SDK / '_backends/simulator/dtypes.py': '260a01643e5158647503d41a2012b5607523f522885b2a7c6bd4ba8cd3fedd4e',
    SDK / '_backends/simulator/alloc.py': '40abe6512cc6884a8fec8c9ffa6d1a7bfdf088babe54a6873ca4bb622b9a8f8c',
    SDK / '_backends/simulator/copy.py': '399389892965cea919dac81ffce900b3ee4820dc6347aa6d2d0550be7db58e87',
    SDK / '_backends/simulator/tensor_view.py': 'b378f0634c591ec8e48b1eecea2c2a77cde7f5ebdba4b19997098644c63cde81',
    SDK / '_backends/simulator/advanced.py': '8f7bec4e5712467e3c4b22ef323aae1ad688b178b34f0f6bec309f7bbb82d3b4',
    SDK / 'framework/_simulate.py': '187dbb5779c8b57b5012ceb013ad9845ed18b864be9e7c813a92998481a40201',
    SDK / 'language/tensor.py': '66e6c75be3a340475b1affa1b69b820e500fa2953159c13884ce0d4199926691',
}
CASE = dict(id=0, seed=20261103, kind='selector_coverage', capacity=8,
            dtype='bfloat16', width=256, slots=[0])
OLD_HASHES = (
    '5781e3d8081dc2fb9a8d4c3593c858dab77b09ed734bed5662d3d7d2a5dcb98d',
    '94555ec4d65509f60933cab1609a615e0d8e0b73a2c9a8a09c5657e5e6fe3f39',
    'df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119',
    'fd8dea2bf754093760877b1a8cdd6b455b320b268e09bff488826fef4482b4c5',
)
OLD_ACTUAL = '81a3a95b13d0e4e66728054d5c4ecfad0d41f985f2850036e904b54b5e6cf2de'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def byte_oracle(cache, updates, slots, capacity, width, itemsize):
    """Independent raw-byte oracle: no floating-point operations or NumPy."""
    stride = width * itemsize
    assert len(cache) == capacity * stride and len(updates) == len(slots) * stride
    result = bytearray(cache)
    for row, slot in enumerate(slots):
        if 0 <= slot < capacity:
            result[slot * stride:(slot + 1) * stride] = updates[row * stride:(row + 1) * stride]
    return bytes(result)


def strict_equal(raw, dtype, shape, expected, expected_dtype, expected_shape):
    return raw == expected and dtype == expected_dtype and tuple(shape) == tuple(expected_shape)


def expected_observations(operation, capacity, width):
    assert operation in ('identity', 'copy', 'permutation', 'selector')
    result = dict(ingress_cache=1, ingress_updates=1, ingress_slots=1, pre_return=1)
    if operation != 'identity':
        result.update(load_cache=capacity * (width // 128), assign_cache=capacity * (width // 128))
    if operation == 'selector':
        result.update(load_updates=capacity * (width // 128), assign_updates=capacity * (width // 128),
                      predicated_copy=capacity * (width // 128))
    return result


def difference(actual, expected, itemsize):
    changed = [i for i in range(min(len(actual), len(expected)) // itemsize)
               if actual[i * itemsize:(i + 1) * itemsize] != expected[i * itemsize:(i + 1) * itemsize]]
    mask, fraction = (0x7f80, 0x7f) if itemsize == 2 else (0x7f800000, 0x7fffff)
    def isnan(raw):
        word = int.from_bytes(raw, 'little')
        return word & mask == mask and bool(word & fraction)
    nan = sum(isnan(actual[i * itemsize:(i + 1) * itemsize]) and
              isnan(expected[i * itemsize:(i + 1) * itemsize]) for i in changed)
    return dict(actual_bytes=len(actual), expected_bytes=len(expected),
                mismatched_words=len(changed), nan_payload_mismatches=nan,
                other_mismatches=len(changed) - nan,
                first8=[dict(word=i, actual=actual[i * itemsize:(i + 1) * itemsize].hex(),
                             expected=expected[i * itemsize:(i + 1) * itemsize].hex()) for i in changed[:8]])


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
                    selected = nl.where(mapping == destination, candidate, selected, dtype=nl.int32)
                source_row = nl.maximum(selected - 1, 0)
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


def negative_controls():
    # One finite-bit corruption and one same-sign NaN payload corruption.
    cache = struct.pack('<4H', 0x3f80, 0x7fc1, 0x8000, 0xffc3)
    updates = struct.pack('<2H', 0x4000, 0x7fff)
    expected = byte_oracle(cache, updates, [1], 2, 2, 2)
    finite, nan = bytearray(expected), bytearray(expected)
    finite[0] ^= 1
    nan[2] ^= 1
    controls = {'finite_byte': bytes(finite), 'nan_payload': bytes(nan),
                'reverse_copy_direction': cache,
                'wrong_slot': byte_oracle(cache, updates, [0], 2, 2, 2)}
    result = {key: not strict_equal(raw, 'bfloat16', (2, 2), expected, 'bfloat16', (2, 2))
              for key, raw in controls.items()}
    result['wrong_dtype'] = not strict_equal(expected, 'uint16', (2, 2), expected, 'bfloat16', (2, 2))
    result['wrong_shape'] = not strict_equal(expected, 'bfloat16', (1, 4), expected, 'bfloat16', (2, 2))
    return result


def diagnose(admit):
    """Execute bounded diagnostic after caller's existing serial admission.

    Caller must enforce inherited MemoryMax=10GiB/MemoryHigh=8GiB/swap=0,
    TasksMax=32, CPU affinity and wall timeout <=1800s. The established worker
    limits (12GiB address space, 1800 CPU seconds, 256KiB report) are retained.
    A passing return is ONLY this migrated simulator diagnostic's result.
    """
    assert callable(admit) and admit() is True, 'Existing serial admission required'
    assert sys.prefix == PREFIX and sys.version_info[:2] == (3, 12)
    assert sys.byteorder == 'little'
    for key, value in {'NKI_PRECISE_FP': '1', 'NEURON_PLATFORM_TARGET_OVERRIDE': 'trn1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}.items():
        assert os.environ.get(key, value) == value, (key, 'conflicting environment')
        os.environ[key] = value
    for kind, cap in ((resource.RLIMIT_AS, 12 * 2**30), (resource.RLIMIT_CPU, 1800),
                      (resource.RLIMIT_FSIZE, 256 * 2**10)):
        soft, hard = resource.getrlimit(kind)
        resource.setrlimit(kind, (min(cap, soft) if soft != resource.RLIM_INFINITY else cap,
                                 min(cap, hard) if hard != resource.RLIM_INFINITY else cap))
    source_sha256 = {}
    for path, wanted in PINS.items():
        assert path.stat().st_size < 2**20
        source_sha256[str(path)] = sha(path.read_bytes())
        assert source_sha256[str(path)] == wanted, str(path)
    result = dict(schema='lfm3_nki05_transport_diagnostic_v1', complete=False, passed=False,
                  source_sha256=source_sha256, diagnostic_source_sha256=sha(Path(__file__).read_bytes()),
                  legacy_candidate_changed=False, legacy_full3716_passed=False,
                  device_behavior_proven=False, full_model_qualified=False, new_speedup_claim=False,
                  target='trn1', precise_fp=True, checks=[], negative_controls=negative_controls())
    began = time.monotonic()
    try:
        from importlib.metadata import version
        result['versions'] = {name: version(name) for name in ('nki', 'neuronx-cc', 'numpy', 'ml_dtypes')}
        assert result['versions']['nki'] == '0.5.0+28631259367.ga768afa6'
        assert result['versions']['neuronx-cc'] == '2.26.6360.0+6f180f47'
        assert result['versions']['ml_dtypes'] == '0.6.0'
        import numpy as np
        import ml_dtypes
        import nki
        from lfm3_batch_slots_extended_v1 import numpy_inputs

        def check(name, operation, cache, updates, slots):
            row = run_case(nki, name, operation, cache, updates, slots)
            result['checks'].append(row)
            assert row['passed'], (name, row['first_boundary_divergence'], row['returned_difference'])

        cache, updates, slots = numpy_inputs(CASE)
        expected = byte_oracle(cache.tobytes(), updates.tobytes(), CASE['slots'], 8, 256, 2)
        hashes = tuple(sha(x.tobytes()) for x in (cache, updates, slots)) + (sha(expected),)
        assert hashes == OLD_HASHES, ('Original fixture changed; no fallback or normalization', hashes)
        result['exact_fixture_hashes_verified'] = True
        result['historical_failed_actual_sha256'] = OLD_ACTUAL
        for operation in ('identity', 'copy', 'permutation', 'selector'):
            check('exact_v5_first_case', operation, cache.copy(), updates.copy(), slots.copy())

        # Each of the 65,536 encodings traverses every movement control. At most
        # 128 KiB per BF16 operand; kernels use 128-element SBUF tiles.
        raw = b''.join(struct.pack('<H', word) for word in range(65536))
        cache = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).reshape(8, 8192).copy()
        updates = np.frombuffer(raw[::-1], dtype=ml_dtypes.bfloat16).reshape(8, 8192).copy()
        slots = np.asarray(list(reversed(range(8))), dtype=np.int32)
        for operation in ('identity', 'copy', 'permutation', 'selector'):
            check('all65536_bf16_encodings', operation, cache.copy(), updates.copy(), slots.copy())
        result['bf16_encoding_count'] = 65536
        result['bf16_encoding_source_sha256'] = sha(raw)

        for dtype in ('bfloat16', 'float32'):
            for capacity in (4, 8, 16):
                cases = {'inactive': [-1, capacity, -(2**31), 2**31 - 1],
                         'duplicate_last_writer': [0, 0, -1, 0],
                         'last_winner_before_inactive': [capacity - 1, capacity - 1, -1, capacity]}
                for name, mapping in cases.items():
                    case = dict(CASE, capacity=capacity, dtype=dtype, slots=mapping)
                    operands = numpy_inputs(case)
                    check(f'{dtype}_B{capacity}_{name}', 'selector', *operands)
                if capacity == 8:
                    for operation in ('identity', 'copy', 'permutation'):
                        check(dtype + '_movement_controls', operation, *numpy_inputs(case))

        def arithmetic(source):
            import nki.language as nl
            value = nl.load(source)
            out = nl.ndarray(source.shape, dtype=nl.float32, buffer=nl.shared_hbm)
            nl.store(out, value=nl.add(value, 1.0))
            return out

        def cast(source):
            import nki.language as nl
            import nki.isa as nisa
            value = nl.load(source)
            converted = nl.ndarray(source.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=converted, src=value)
            out = nl.ndarray(source.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)
            nl.store(out, value=converted)
            return out

        source = np.frombuffer(struct.pack('<4f', -2, -1, 0, 2), dtype=np.float32).reshape(1, 4).copy()
        actual = nki.simulate(arithmetic)(source)
        result['arithmetic_control'] = strict_equal(actual.tobytes(), actual.dtype.name, actual.shape,
                                                   struct.pack('<4f', -1, 0, 1, 3), 'float32', (1, 4))
        source = np.frombuffer(struct.pack('<4I', 0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000),
                               dtype=np.float32).reshape(1, 4).copy()
        actual = nki.simulate(cast)(source)
        result['round_to_nearest_even_cast_control'] = strict_equal(
            actual.tobytes(), actual.dtype.name, actual.shape,
            struct.pack('<4H', 0x3f80, 0x3f82, 0xbf80, 0xbf82), 'bfloat16', (1, 4))
        assert result['arithmetic_control'] and result['round_to_nearest_even_cast_control']
        assert all(result['negative_controls'].values())
        result.update(complete=True, passed=True)
    except Exception as error:
        result['error'] = repr(error)
    result['seconds'] = time.monotonic() - began
    result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert len(json.dumps(result, allow_nan=False).encode()) <= 256 * 2**10
    return result
