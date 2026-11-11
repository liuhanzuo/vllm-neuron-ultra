"""Independent small stdlib controls; no real tensor, model or filesystem IO."""
import itertools
from pathlib import Path
import random
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'research')]
from research import compare_minicpm5_layer1_native_sdk_v3 as candidate
from research import check_compare_minicpm5_native_sdk_kv_v2 as fixture_source


def tensor(shape, values, dtype='torch.float32'):
    fmt = '<f' if dtype == 'torch.float32' else '<H'
    return dict(shape=shape, dtype=dtype, raw=b''.join(struct.pack(fmt, x) for x in values),
                evidence=[{'metadata': {'source_offset': 8192, 'source_stride': [1] * len(shape)}}])


class Independent(unittest.TestCase):
    def test_packed_selection_matches_coordinate_oracle(self):
        rng = random.Random(11011)
        shape = [4, 3, 5, 7]
        coordinates = list(itertools.product(*(range(n) for n in shape)))
        original = tensor(shape, range(len(coordinates)))
        for _ in range(80):
            ranges = []
            for extent in shape:
                low = rng.randrange(extent + 1)
                ranges.append([low, rng.randrange(low, extent + 1)])
            expected = [float(i) for i, coord in enumerate(coordinates)
                        if all(low <= x < high for x, (low, high) in zip(coord, ranges))]
            selected = candidate.select(original, ranges)
            self.assertEqual(list(candidate.numbers(selected['raw'], 'torch.float32', lambda: None)), expected)
            self.assertEqual(selected['evidence'][0]['metadata']['source_offset'], 8192)

    def test_prior_active_join_preserves_every_batch_and_head(self):
        for axis in range(4):
            shape = [2, 3, 4, 5]
            coordinates = list(itertools.product(*(range(n) for n in shape)))
            source = tensor(shape, range(len(coordinates)))
            a = [[0, n] for n in shape]
            b = [[0, n] for n in shape]
            a[axis] = [0, shape[axis] - 1]
            b[axis] = [shape[axis] - 1, shape[axis]]
            joined = candidate.join(candidate.select(source, a), candidate.select(source, b), axis)
            self.assertEqual(joined['raw'], source['raw'])
            self.assertEqual(joined['shape'], shape)

    def test_original_slot_survives_nested_selection(self):
        source = tensor([4, 2, 5, 3], range(120))
        selected = candidate.select(candidate.row(source, 3), [[0, 1], [1, 2], [2, 5], [0, 3]])
        evidence = selected['evidence'][0]
        self.assertEqual(evidence['logical_selection_chain'][0]['ranges'][0], [3, 4])
        self.assertEqual(evidence['logical_selection_chain'][1]['ranges'][2], [2, 5])
        self.assertEqual(list(candidate.numbers(selected['raw'], 'torch.float32', lambda: None)), list(range(111, 120)))

    def test_signed_zero_is_byte_difference_and_dtype_difference_is_not_byte_comparison(self):
        positive = tensor([1], [0], 'torch.bfloat16')
        negative = tensor([1], [0x8000], 'torch.bfloat16')
        result = candidate.compare_values(positive, negative, lambda: None)
        self.assertEqual(result['status'], 'different')
        self.assertTrue(result['values_equal'])
        self.assertFalse(result['bytes_equal'])
        self.assertEqual(result['numeric_mismatch_elements'], 0)
        drift = candidate.compare_values(tensor([1], [0x3F80], 'torch.bfloat16'),
                                         tensor([1], [1.0009765625]), lambda: None)
        self.assertIsNone(drift['bytes_equal'])
        self.assertEqual(drift['maximum_absolute_error'], 0.0009765625)
        self.assertEqual(drift['relative_l2'], 0.0009765625)

    def test_128_wide_non_prior_events_never_become_padding(self):
        for bucket in (128, 2048):
            for event in candidate.sdk_manifest.events(bucket):
                for leaf in event['tensors']:
                    label, path = event['label'], tuple(leaf['path'])
                    axis = candidate.padding_axis(label, path)
                    if axis is not None:
                        self.assertEqual(leaf['shape'][axis], bucket)
                        self.assertNotIn(label, ('q_projection', 'k_projection', 'v_projection',
                                                'q_after_prep', 'k_after_prep', 'v_after_prep',
                                                'pv_prior', 'pv_active', 'returned_current_kv'))
        self.assertIsNone(candidate.padding_axis('future_unreviewed_score', ()))

    def test_last_valid_key_difference_is_retained_with_original_slot(self):
        native, sdk, shards = fixture_source.fixture()
        value = sdk['events']['effective_prior_kv', (1,)]
        raw = bytearray(value['raw'])
        offset = ((2 * 128 + 4) * 128 + 127) * 2
        raw[offset:offset + 2] = struct.pack('<H', 0x3F80)
        value['raw'] = bytes(raw)
        result = candidate.compare_frame(native, sdk, shards, lambda: None)
        self.assertEqual(result['first_observed_difference'], 'cache_before_value')
        comparison = next(x for x in result['boundaries'] if x['boundary'] == 'cache_before_value')
        self.assertEqual(comparison['numeric_mismatch_elements'], 1)
        self.assertEqual(comparison['first_numeric_difference_flat_index'], 5 * 128 - 1)
        self.assertEqual(comparison['sdk_evidence'][0]['logical_selection_chain'][0]['ranges'][0], [2, 3])
        self.assertFalse(result['first_difference_is_causal_localization'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
