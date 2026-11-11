"""Pure stdlib checks. Never calls diagnose, make_kernel or simulator/native code."""
import ast
import itertools
import json
from pathlib import Path
import struct
import sys

import lfm3_nki05_transport_diagnostic_v1 as d


def check():
    source = Path(d.__file__)
    tree = ast.parse(source.read_text())
    forbidden = {'numpy', 'ml_dtypes', 'nki', 'neuronxcc', 'torch', 'torch_xla'}
    assert not forbidden.intersection(name.split('.')[0] for name in sys.modules)
    assert not any(isinstance(node, ast.If) and '__main__' in ast.unparse(node.test) for node in tree.body)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not forbidden.intersection(alias.name.split('.')[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] not in forbidden
    for path, wanted in d.PINS.items():
        assert d.sha(path.read_bytes()) == wanted, str(path)
    assert all(d.negative_controls().values())
    # Enumerate all 4-row mappings over inactive / OOB / duplicated active slots.
    # Independent expected source-row selection; oracle itself processes writes.
    count = 0
    for capacity in (4, 8, 16):
        cache = b''.join(struct.pack('<2H', row, 0x8000 | row) for row in range(capacity))
        updates = b''.join(struct.pack('<2H', 0x7f81 + row, 0xffc1 + row) for row in range(4))
        for mapping in itertools.product((-1, 0, capacity - 1, capacity), repeat=4):
            expected = bytearray()
            for destination in range(capacity):
                last = next((row for row in range(3, -1, -1) if mapping[row] == destination), None)
                expected.extend(cache[destination * 4:(destination + 1) * 4] if last is None
                                else updates[last * 4:(last + 1) * 4])
            assert d.byte_oracle(cache, updates, mapping, capacity, 2, 2) == bytes(expected)
            count += 1
    raw = b''.join(struct.pack('<H', word) for word in range(65536))
    assert len(raw) == 128 * 1024 and set(struct.unpack('<65536H', raw)) == set(range(65536))
    assert set(struct.unpack('<65536H', raw[::-1])) == set(range(65536))
    assert d.difference(raw, raw, 2)['mismatched_words'] == 0
    corrupted = bytearray(raw)
    corrupted[2 * 0x7fc1] ^= 1
    diff = d.difference(bytes(corrupted), raw, 2)
    assert diff['nan_payload_mismatches'] == 1 and diff['other_mismatches'] == 0
    corrupted[2 * 0x3f80] ^= 1
    diff = d.difference(bytes(corrupted), raw, 2)
    assert diff['nan_payload_mismatches'] == 1 and diff['other_mismatches'] == 1
    assert d.expected_observations('selector', 8, 8192)['predicated_copy'] == 512
    assert d.expected_observations('identity', 8, 8192) == dict(ingress_cache=1, ingress_updates=1, ingress_slots=1, pre_return=1)
    return dict(passed=True, source_only=True, synthetic_mapping_checks=count,
                bf16_encodings=65536, native_imports=False, simulator_execution=False,
                device_execution=False, model_execution=False,
                diagnostic_source_sha256=d.sha(source.read_bytes()),
                checker_source_sha256=d.sha(Path(__file__).read_bytes()),
                checked_source_pins=len(d.PINS), negative_controls=d.negative_controls())


if __name__ == '__main__':
    print(json.dumps(check(), indent=2, sort_keys=True))
