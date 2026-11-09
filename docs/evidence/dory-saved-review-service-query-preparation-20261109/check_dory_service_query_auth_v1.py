"""Bounded dependency-authentication tests; no service or model execution."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import dory_service_query_auth_v1 as a


def pin(path):
    info = path.stat()
    return dict(bytes=info.st_size, device=info.st_dev, inode=info.st_ino,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Authentication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dory-service-auth-test-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'example.py'
        self.path.write_text('answer = 42\n')
        self.pin = pin(self.path)

    def test_correct_bytes(self):
        self.assertEqual(a.authenticated_bytes(self.path, self.pin), b'answer = 42\n')

    def test_same_size_tampering_rejected(self):
        self.path.write_text('answer = 43\n')
        with self.assertRaisesRegex(RuntimeError, 'stable dependency bytes'):
            a.authenticated_bytes(self.path, self.pin)

    def test_replaced_file_identity_rejected(self):
        replacement = self.path.with_suffix('.new')
        replacement.write_bytes(self.path.read_bytes())
        replacement.replace(self.path)
        with self.assertRaisesRegex(RuntimeError, 'file identity'):
            a.authenticated_bytes(self.path, self.pin)

    def test_symlink_rejected(self):
        link = self.path.with_suffix('.link')
        link.symlink_to(self.path)
        with self.assertRaisesRegex(RuntimeError, 'Canonical'):
            a.authenticated_bytes(link, self.pin)

    def test_oversized_claim_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Bounded'):
            a.authenticated_bytes(self.path, dict(self.pin, bytes=a.MAX_FILE + 1))

    def test_shortened_file_rejected(self):
        self.path.write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError, 'file identity'):
            a.authenticated_bytes(self.path, self.pin)

    def test_source_loader_uses_pinned_bytes_without_cache(self):
        loaded = {}
        loader = a.PinnedSourceLoader('example', str(self.path), self.pin, loaded)
        namespace = {}
        exec(loader.get_code('example'), namespace)
        self.assertEqual(namespace['answer'], 42)
        self.assertEqual(loaded, {'example': str(self.path)})
        self.assertFalse((self.path.parent / '__pycache__').exists())

    def test_changed_source_cannot_execute(self):
        loader = a.PinnedSourceLoader('example', str(self.path), self.pin, {})
        self.path.write_text('answer = 99\n')
        with self.assertRaises(RuntimeError):
            loader.get_code('example')

    def test_unknown_module_rejected_before_execution(self):
        old = list(sys.path)
        try:
            sys.path.insert(0, str(self.path.parent))
            with self.assertRaisesRegex(RuntimeError, 'outside reviewed'):
                a.PinnedFinder({}).find_spec('example')
            self.assertNotIn('example', sys.modules)
        finally:
            sys.path[:] = old

    def test_selected_module_gets_authenticated_source_loader(self):
        old = list(sys.path)
        try:
            sys.path.insert(0, str(self.path.parent))
            finder = a.PinnedFinder({str(self.path): self.pin})
            spec = finder.find_spec('example')
            self.assertIsInstance(spec.loader, a.PinnedSourceLoader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.answer, 42)
        finally:
            sys.path[:] = old

    def test_reviewed_mapping_delta(self):
        baseline = {'/old': {'device': 1, 'inode': 2}}
        addition = {'/new': {'device': 1, 'inode': 3}}
        self.assertEqual(a.check_mapping_delta(baseline, dict(baseline, **addition), addition), addition)

    def test_unexpected_native_mapping_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unreviewed'):
            a.check_mapping_delta({}, {'/unexpected': {'device': 1, 'inode': 3}}, {})

    def test_missing_expected_native_mapping_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'expected query native'):
            a.check_mapping_delta({}, {}, {'/missing': {'device': 1, 'inode': 3}})

    def test_changed_existing_native_mapping_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Original native'):
            a.check_mapping_delta({'/old': {'device': 1, 'inode': 2}},
                                  {'/old': {'device': 1, 'inode': 3}}, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
