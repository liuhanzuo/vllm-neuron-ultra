"""Small independent alignment controls, no model or observed data reads."""
import copy
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.compare_minicpm5_layer1_native_cache_v1 import aligned_prefix, distance


class Checks(unittest.TestCase):
    def fixture(self):
        frame = dict(index=9, fixture=7, slot=1, step=2)
        case = dict(rank=1, positions=[1, 3, 1, 1], physical_frame_order=[frame])
        row = dict(frame=dict(index=5, fixture=7, slot=1, step=1), cache_length=3,
            cache_metadata=dict(key=dict(shape=[1, 2, 3, 128])))
        cache = bytearray(b'\xff' * (4*4224*128*2))
        target = b''.join(struct.pack('<H', 0x3f80+i) * 128 for i in range(3))
        start = 4224*128*2; cache[start:start+len(target)] = target
        native = b'\x00' * len(target) + target
        return case, frame, row, bytes(cache), native, target

    def test_physical_slot_second_head_and_valid_prefix_only(self):
        case, frame, row, cache, native, expected = self.fixture()
        self.assertEqual(aligned_prefix(case, frame, row, cache, native, 'key'), (expected, expected))

    def test_previous_step_and_fixture_required(self):
        case, frame, row, cache, native, _ = self.fixture()
        for field, value in [('step', 2), ('fixture', 2), ('slot', 3)]:
            bad = copy.deepcopy(row); bad['frame'][field] = value
            with self.assertRaises(RuntimeError): aligned_prefix(case, frame, bad, cache, native, 'key')

    def test_short_or_different_head_geometry_rejected(self):
        case, frame, row, cache, native, _ = self.fixture()
        for badcache, badnative in [(cache[:-2], native), (cache, native[:-2])]:
            with self.assertRaises(RuntimeError): aligned_prefix(case, frame, row, badcache, badnative, 'key')
        row['cache_metadata']['key']['shape'] = [1, 1, 3, 128]
        with self.assertRaises(RuntimeError): aligned_prefix(case, frame, row, cache, native, 'key')

    def test_bf16_values_and_reference_denominator(self):
        result = distance(struct.pack('<HH', 0x4000, 0x4080), struct.pack('<HH', 0x3f80, 0x4000))
        self.assertEqual(result['elements'], 2); self.assertEqual(result['changed_elements'], 2)
        self.assertEqual(result['l2_squared'], 5); self.assertEqual(result['reference_l2_squared'], 5)
        self.assertEqual(result['relative_l2'], 1); self.assertEqual(result['max_abs'], 2)

    def test_nonfinite_and_short_bytes_rejected(self):
        for bad in [b'\x80\x7f', b'\xc0\x7f', b'\x80\xff', b'\x00']:
            with self.assertRaises(RuntimeError): distance(bad, b'\x00\x00')

    def test_empty_appended_span_and_signed_zero(self):
        self.assertEqual(distance(b'', b'')['elements'], 0)
        result = distance(b'\x00\x80', b'\x00\x00')
        self.assertFalse(result['raw_exact']); self.assertEqual(result['changed_elements'], 0)


if __name__ == '__main__': unittest.main(verbosity=2)
