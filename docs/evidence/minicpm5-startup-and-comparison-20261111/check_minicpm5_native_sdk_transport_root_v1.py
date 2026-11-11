"""Independent codec controls on small synthetic values; no real result IO."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import minicpm5_native_sdk_report_transport_v1 as candidate


class IndependentTransport(unittest.TestCase):
    def test_standard_decoder_reproduces_canonical_source_exactly(self):
        value = {'unicode': '完整证据', 'negative_zero': -0.0, 'unknown': None,
                 'nested': [{'rank': r, 'slot': s, 'offsets': list(range(11))}
                            for r in range(2) for s in range(4)]}
        envelope = candidate.packed_physical(value)
        compressed = base64.b64decode(envelope['data'], validate=True)
        raw = zlib.decompress(compressed)
        expected = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        self.assertEqual(raw, expected)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), envelope['sha256'])
        self.assertEqual(candidate.unpack(envelope), value)

    def test_understated_length_never_accepts_large_expansion(self):
        envelope = candidate.packed_physical({'repeated': 'a' * 40000})
        for declared in (1, 64, envelope['uncompressed_bytes'] - 1):
            with self.subTest(declared=declared), self.assertRaises(RuntimeError):
                candidate.unpack(dict(envelope, uncompressed_bytes=declared))

    def test_independently_constructed_second_stream_is_rejected(self):
        envelope = candidate.packed_physical({'record': 1})
        compressed = base64.b64decode(envelope['data']) + zlib.compress(b'{"record":2}')
        forged = dict(envelope, data=base64.b64encode(compressed).decode(), compressed_bytes=len(compressed))
        with self.assertRaises(RuntimeError):
            candidate.unpack(forged)

    def test_complete_envelope_overhead_and_duplicate_keys_are_checked(self):
        # Test the complete stored-file cap, independently of compressed payload limits.
        with patch.object(candidate, 'pack_report', return_value={'data': 'x' * (candidate.FILE_CAP - 1)}):
            with self.assertRaises(RuntimeError):
                candidate.encode_report({})
        with self.assertRaises(RuntimeError):
            candidate.decode_report(b'{"nested":{"duplicate":1,"duplicate":2}}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
