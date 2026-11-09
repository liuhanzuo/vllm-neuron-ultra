"""Tiny generated file fixtures, including replacement and partial-read races."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import minicpm5_saved_tap_io_v1 as io


class BoundedFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)/'data'
        self.folder.mkdir()
        self.path = self.folder/'fixture.bin'
        self.raw = b'abcd'*20000
        self.path.write_bytes(self.raw)
        self.uid = os.getuid()
        self.ref = dict(path=str(self.path), sha256=hashlib.sha256(self.raw).hexdigest())
        self.entry = io.snapshot(self.ref, len(self.raw), self.uid)
        self.guards = 0

    def guard(self):
        self.guards += 1

    def reader(self, entries=None, **kwargs):
        return io.PinnedReader(entries or [self.entry], kwargs.pop('guard', self.guard),
            uid=self.uid, file_cap=2**20, total_cap=kwargs.pop('total_cap', 2**20), **kwargs)

    def test_snapshot_never_opens_or_reads_payload(self):
        original_open = os.open
        def only_directories(path, flags, **kwargs):
            self.assertTrue(flags & os.O_DIRECTORY)
            return original_open(path, flags, **kwargs)
        with patch.object(io.os, 'open', side_effect=only_directories), patch.object(io.os, 'read') as read:
            self.assertEqual(io.snapshot(self.ref, len(self.raw), self.uid), self.entry)
        read.assert_not_called()

    def test_complete_multichunk_read_and_terminal_accounting(self):
        reader = self.reader()
        original_read = os.read
        requests = []
        def record(fd, count):
            requests.append(count)
            return original_read(fd, count)
        with patch.object(io.os, 'read', side_effect=record):
            self.assertEqual(reader(self.ref, len(self.raw)), self.raw)
        self.assertEqual(requests, [65536, 14464, 1])
        self.assertEqual(reader.finish(), dict(files=1, bytes_read=80000,
            all_pinned_inputs_read_once=True, all_input_hashes_verified=True))
        self.assertFalse(reader.poisoned)
        with self.assertRaisesRegex(ValueError, 'finalized'): reader(self.ref)

    def test_wrong_ref_size_or_outside_path_stops_before_open(self):
        cases = [(dict(self.ref, sha256='0'*64), None),
                 (dict(self.ref, path='/tmp/unadmitted.bin'), None),
                 (self.ref, 1), (self.ref, True)]
        for ref, size in cases:
            reader = self.reader()
            with patch.object(io.os, 'open') as opened, self.assertRaises(ValueError):
                reader(ref, size)
            opened.assert_not_called()
            self.assertTrue(reader.poisoned)

    def test_duplicate_read_rejected_and_reader_poisoned(self):
        reader = self.reader()
        reader(self.ref)
        with self.assertRaisesRegex(ValueError, 'once-only'): reader(self.ref)
        with self.assertRaisesRegex(ValueError, 'Failed'): reader.finish()

    def test_finish_requires_every_input_and_cannot_resume(self):
        reader = self.reader()
        with self.assertRaisesRegex(ValueError, 'Every admitted'): reader.finish()
        with self.assertRaisesRegex(ValueError, 'Failed'): reader(self.ref)

    def test_duplicate_manifest_and_small_budget_rejected_before_open(self):
        for entries, cap in (([self.entry]*2, 2**20), ([self.entry], 79999)):
            with patch.object(io.os, 'open') as opened, self.assertRaises(ValueError):
                self.reader(entries, total_cap=cap)
            opened.assert_not_called()

    def test_noncanonical_paths_rejected(self):
        for path in ('relative/file', '/tmp/../etc/passwd', '/tmp//file', '//tmp/file', '/tmp/./file', '/tmp/file\0', '/'):
            with self.subTest(path=path), self.assertRaises(ValueError): io.path_parts(path)

    def test_hardlink_and_symlink_rejected_during_snapshot(self):
        other = self.folder/'second'
        os.link(self.path, other)
        with self.assertRaisesRegex(ValueError, 'single-link'): io.snapshot(self.ref, len(self.raw), self.uid)
        other.unlink()
        original = self.folder/'original'
        self.path.rename(original)
        self.path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, 'ordinary'): io.snapshot(self.ref, len(self.raw), self.uid)

    def test_file_or_ancestor_symlink_rejected_during_read(self):
        original = self.folder/'original'
        self.path.rename(original)
        self.path.symlink_to(original)
        reader = self.reader()
        with self.assertRaises(OSError): reader(self.ref)
        self.assertTrue(reader.poisoned)
        self.path.unlink()
        original.rename(self.path)
        moved = self.folder.with_name('old-data')
        self.folder.rename(moved)
        self.folder.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(OSError): self.reader()(self.ref)

    def test_replaced_file_before_read_rejected_without_payload_read(self):
        old = self.path.with_name('old')
        self.path.rename(old)
        self.path.write_bytes(self.raw)
        with patch.object(io.os, 'read') as read, self.assertRaisesRegex(ValueError, 'full identity'):
            self.reader()(self.ref)
        read.assert_not_called()

    def test_truncation_or_growth_rejected_before_payload_read(self):
        for raw in (self.raw[:-1], self.raw+b'x'):
            self.path.write_bytes(raw)
            with patch.object(io.os, 'read') as read, self.assertRaisesRegex(ValueError, 'declared size'):
                self.reader()(self.ref)
            read.assert_not_called()

    def test_changed_content_same_size_rejected_by_identity_or_hash(self):
        self.path.write_bytes(b'x'*len(self.raw))
        with self.assertRaisesRegex(ValueError, 'full identity|SHA256'):
            self.reader()(self.ref)

    def test_wrong_payload_hash_poisoned_after_complete_read(self):
        entry = copy.deepcopy(self.entry)
        entry['sha256'] = '0'*64
        reader = self.reader([entry])
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            reader(dict(self.ref, sha256='0'*64))
        self.assertEqual(reader.bytes_read, len(self.raw))
        self.assertTrue(reader.poisoned)

    def test_short_read_poisoned(self):
        reader = self.reader()
        with patch.object(io.os, 'read', return_value=b''), self.assertRaisesRegex(ValueError, 'Complete bounded'):
            reader(self.ref)
        self.assertTrue(reader.poisoned)
        self.assertEqual(reader.bytes_read, 0)

    def test_file_replacement_during_read_detected_even_with_equal_content(self):
        original_read = os.read
        replaced = False
        def replace(fd, size):
            nonlocal replaced
            data = original_read(fd, size)
            if not replaced:
                replaced = True
                self.path.rename(self.path.with_name('old'))
                self.path.write_bytes(self.raw)
            return data
        reader = self.reader()
        with patch.object(io.os, 'read', side_effect=replace), self.assertRaisesRegex(ValueError, 'replaced during'):
            reader(self.ref)
        self.assertTrue(reader.poisoned)

    def test_parent_replacement_during_read_detected(self):
        original_read = os.read
        replaced = False
        def replace(fd, size):
            nonlocal replaced
            data = original_read(fd, size)
            if not replaced:
                replaced = True
                self.folder.rename(self.folder.with_name('old-data'))
                self.folder.mkdir()
                self.path.write_bytes(self.raw)
            return data
        reader = self.reader()
        with patch.object(io.os, 'read', side_effect=replace), self.assertRaisesRegex(ValueError, 'Parent directory replaced'):
            reader(self.ref)
        self.assertTrue(reader.poisoned)

    def test_guard_abort_before_open_or_between_chunks_never_returns_partial_bytes(self):
        def abort(): raise RuntimeError('serial guard lost')
        reader = self.reader(guard=abort)
        with patch.object(io.os, 'open') as opened, self.assertRaisesRegex(RuntimeError, 'serial guard'):
            reader(self.ref)
        opened.assert_not_called()
        def stop_after_first_chunk():
            if reader.bytes_read: raise RuntimeError('resource budget')
        reader = self.reader(guard=stop_after_first_chunk)
        with self.assertRaisesRegex(RuntimeError, 'resource budget'): reader(self.ref)
        self.assertEqual(reader.bytes_read, 65536)
        self.assertTrue(reader.poisoned)

    def test_no_fd_leak_on_success_or_hash_failure(self):
        before = len(list(Path('/proc/self/fd').iterdir()))
        reader = self.reader()
        reader(self.ref)
        reader.finish()
        bad = dict(self.entry, sha256='0'*64)
        with self.assertRaises(ValueError): self.reader([bad])(dict(self.ref, sha256='0'*64))
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())), before)

    def test_missing_or_modified_ancestor_identity_rejected(self):
        for mutate in (lambda e:e['directories'].pop(),
                       lambda e:e['directories'][-1].update(st_ino=0)):
            entry = copy.deepcopy(self.entry)
            mutate(entry)
            with patch.object(io.os, 'read') as read, self.assertRaises(ValueError):
                self.reader([entry])(self.ref)
            read.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
