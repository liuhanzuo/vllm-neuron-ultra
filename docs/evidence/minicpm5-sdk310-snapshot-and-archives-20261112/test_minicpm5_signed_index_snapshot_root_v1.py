"""Root independent pure decoder and output-boundary checks; no real XZ reads."""
import copy
import hashlib
import importlib.abc
from pathlib import Path
import sys
import tempfile
import unittest


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'ctypes', 'lzma', '_lzma', 'neuronxcc', 'torch_neuronx'}:
            raise AssertionError('Native/import payload forbidden in this source test: '+fullname)


sys.meta_path.insert(0, NoNative())
import minicpm5_sdk310_signed_index_snapshot_v1 as s


class Decoder:
    def __init__(self, output, eof=True, tail=b''):
        self.output, self.eof, self.unused_data = output, eof, tail
        self.calls = []

    def decompress(self, data, *, max_length):
        self.calls.append((data, max_length))
        return self.output


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.compressed, self.plain = b'synthetic source', b'Package: fixture\nVersion: 1\n'
        self.row = {'compressed': self.ref(self.compressed), 'expected_uncompressed': self.ref(self.plain)}

    @staticmethod
    def ref(raw):
        return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def execute(self, *decoders, row=None, compressed=None):
        self.limits, self.guards = [], []
        iterator = iter(decoders)
        def factory(limit):
            self.limits.append(limit)
            return next(iterator)
        return s.decode_exact(self.compressed if compressed is None else compressed,
                              self.row if row is None else row, factory, lambda: self.guards.append(True))

    def test_valid_exact_single_stream(self):
        d = Decoder(self.plain)
        self.assertEqual(self.execute(d), self.plain)
        self.assertEqual(d.calls, [(self.compressed, len(self.plain)+1)])
        self.assertEqual(self.limits, [64*2**20])
        self.assertGreaterEqual(len(self.guards), 2)

    def test_corrupt_compressed_fails_before_decoder(self):
        for raw in (b'', self.compressed+b'x', b'X'+self.compressed[1:]):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.execute(compressed=raw)
            self.assertEqual(self.limits, [])

    def test_wrong_raw_size_or_hash(self):
        for raw in (self.plain[:-1], self.plain+b'x', b'X'+self.plain[1:]):
            with self.subTest(raw=raw), self.assertRaises(RuntimeError):
                self.execute(Decoder(raw))

    def test_incomplete_first_stream(self):
        with self.assertRaises(RuntimeError):
            self.execute(Decoder(self.plain, eof=False))

    def test_only_exact_reviewed_tail(self):
        for tail in (b'x', s.EMPTY_TAIL[:-1], s.EMPTY_TAIL+b'x', s.EMPTY_TAIL*2):
            with self.subTest(tail=tail), self.assertRaises(RuntimeError):
                self.execute(Decoder(self.plain, tail=tail))
            self.assertEqual(len(self.limits), 1)

    def test_empty_tail_decoded_with_one_byte_cap(self):
        first, second = Decoder(self.plain, tail=s.EMPTY_TAIL), Decoder(b'')
        self.assertEqual(self.execute(first, second), self.plain)
        self.assertEqual(second.calls, [(s.EMPTY_TAIL, 1)])
        self.assertEqual(self.limits, [64*2**20]*2)

    def test_second_stream_must_be_empty_complete_and_final(self):
        for second in (Decoder(b'x'), Decoder(b'', eof=False), Decoder(b'', tail=b'x')):
            with self.subTest(second=second), self.assertRaises(RuntimeError):
                self.execute(Decoder(self.plain, tail=s.EMPTY_TAIL), second)

    def test_guard_failure_prevents_decode(self):
        called = []
        def guard():
            raise RuntimeError('stop')
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            s.decode_exact(self.compressed, self.row, lambda limit: called.append(limit), guard)
        self.assertEqual(called, [])

    def test_exclusive_readonly_output_and_verified_read(self):
        with tempfile.TemporaryDirectory(prefix='mini-snapshot-root-test-') as name:
            path = Path(name)/'fixture'
            expected = self.ref(self.plain)
            ref = s.save_plain(path, self.plain, expected, lambda: None)
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            self.assertEqual(s.read_verified(ref, lambda: None, 1024), self.plain)
            with self.assertRaises(FileExistsError):
                s.save_plain(path, self.plain, expected, lambda: None)
            path.chmod(0o600)
            path.write_bytes(b'X'+self.plain[1:])
            with self.assertRaises(RuntimeError):
                s.read_verified(ref, lambda: None, 1024)

    def test_symlink_input_rejected(self):
        with tempfile.TemporaryDirectory(prefix='mini-snapshot-root-test-') as name:
            target, alias = Path(name)/'target', Path(name)/'alias'
            target.write_bytes(self.plain)
            alias.symlink_to(target)
            with self.assertRaises(RuntimeError):
                s.read_verified(dict(path=str(alias), **self.ref(self.plain)), lambda: None, 1024)

    def test_hardlink_input_rejected(self):
        with tempfile.TemporaryDirectory(prefix='mini-snapshot-root-test-') as name:
            target, alias = Path(name)/'target', Path(name)/'alias'
            target.write_bytes(self.plain)
            alias.hardlink_to(target)
            with self.assertRaises(RuntimeError):
                s.read_verified(dict(path=str(target), **self.ref(self.plain)), lambda: None, 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2)
