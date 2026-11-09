"""Temporary-file failure tests; never opens experiment or journal targets."""
import errno
import gzip
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import completed_storage_files_v1 as f


class FileSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='completed-storage-test-')
        self.root = Path(self.tmp.name).resolve()
        self.source = self.root/'source'
        self.body = b'closed journal contents\x00\xff' * 1000
        self.source.write_bytes(self.body)
        self.expected = f.ordinary(self.source)
        self.target = self.root/'archive.gz'

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, guard=lambda: None, cap=2**20):
        return f.archive_journal(self.source, self.target, self.expected, guard, cap)

    def test_full_hash(self):
        self.assertEqual(f.file_hash(self.source, self.expected, lambda: None),
                         dict(bytes=len(self.body), sha256=hashlib.sha256(self.body).hexdigest()))

    def test_source_mutation_during_hash(self):
        def mutate():
            with self.source.open('ab') as out: out.write(b'changed')
        with self.assertRaises(RuntimeError): f.file_hash(self.source, self.expected, mutate)
        self.assertTrue(self.source.exists())

    def test_archive_roundtrip_is_private_and_keeps_original(self):
        row = self.archive()
        self.assertEqual(gzip.decompress(self.target.read_bytes()), self.body)
        self.assertEqual(self.source.read_bytes(), self.body)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o600)
        self.assertTrue(row['independently_decompressed'])
        self.assertEqual(row['gzip_sha256'], hashlib.sha256(self.target.read_bytes()).hexdigest())

    def test_corrupt_and_truncated_archives_rejected(self):
        raw = gzip.compress(self.body)
        for broken in (raw[:-5], raw[:-8]+b'\x00'*8):
            with self.subTest(length=len(broken)):
                self.target.write_bytes(broken)
                with self.assertRaises((EOFError, gzip.BadGzipFile, RuntimeError)):
                    f.verify_gzip(self.target, len(self.body), hashlib.sha256(self.body).hexdigest(), lambda: None)
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_valid_gzip_wrong_content_rejected(self):
        self.target.write_bytes(gzip.compress(b'x'*len(self.body)))
        with self.assertRaises(RuntimeError):
            f.verify_gzip(self.target, len(self.body), hashlib.sha256(self.body).hexdigest(), lambda: None)

    def test_decompressed_overrun_rejected(self):
        self.target.write_bytes(gzip.compress(self.body))
        with self.assertRaisesRegex(RuntimeError, 'overrun'):
            f.verify_gzip(self.target, len(self.body)-1, hashlib.sha256(self.body).hexdigest(), lambda: None)

    def test_cap_failure_preserves_input_and_partial_output(self):
        with self.assertRaises(RuntimeError): self.archive(cap=1)
        self.assertEqual(self.source.read_bytes(), self.body)
        self.assertTrue(self.target.exists())

    def test_interruption_preserves_input(self):
        def interrupted(): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.archive(guard=interrupted)
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_failed_destination_open_closes_source_fd(self):
        original = os.open
        source_fds = []
        def fail(path, flags, *args, **kwargs):
            if Path(path) == self.target: raise OSError(errno.ENOSPC, 'injected full disk')
            fd = original(path, flags, *args, **kwargs)
            if Path(path) == self.source: source_fds.append(fd)
            return fd
        with patch.object(f.os, 'open', side_effect=fail):
            with self.assertRaises(OSError): self.archive()
        self.assertEqual(len(source_fds), 1)
        with self.assertRaises(OSError): os.fstat(source_fds[0])
        self.assertEqual(self.source.read_bytes(), self.body)

    def test_links_rejected(self):
        link = self.root/'link'
        link.symlink_to(self.source)
        with self.assertRaises(RuntimeError): f.ordinary(link)
        link.unlink()
        os.link(self.source, link)
        with self.assertRaises(RuntimeError): f.ordinary(self.source)

    def test_symlink_parent_rejected(self):
        folder = self.root/'folder'
        folder.mkdir()
        (folder/'data').write_bytes(b'data')
        (self.root/'alias').symlink_to(folder, target_is_directory=True)
        with self.assertRaises(RuntimeError): f.ordinary(self.root/'alias/data')

    def test_unlink_refuses_same_length_replacement(self):
        replacement = self.root/'replacement'
        replacement.write_bytes(self.body)
        replacement.replace(self.source)
        with self.assertRaises(RuntimeError): f.unlink_exact(self.source, self.expected)
        self.assertTrue(self.source.exists())

    def test_exact_unlink_preserves_retained_tree(self):
        retained = self.root/'retained'
        retained.mkdir()
        (retained/'data').write_bytes(b'keep this')
        before = f.snapshot_tree(self.root, [str(self.source)])
        f.unlink_exact(self.source, self.expected)
        self.assertFalse(self.source.exists())
        self.assertEqual(before, f.snapshot_tree(self.root, [str(self.source)]))

    def test_metadata_no_overwrite_and_private_action_journal(self):
        out = self.root/'receipt.json'
        f.once(out, {'completed': False})
        with self.assertRaises(RuntimeError): f.once(out, {'completed': True})
        log = self.root/'actions.jsonl'
        f.append(log, {'phase': 'before'})
        f.append(log, {'phase': 'after'})
        self.assertEqual(len(log.read_text().splitlines()), 2)
        self.assertEqual(log.stat().st_mode & 0o777, 0o600)
        os.link(log, self.root/'shared-log')
        with self.assertRaises(RuntimeError): f.append(log, {'phase': 'forbidden'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
