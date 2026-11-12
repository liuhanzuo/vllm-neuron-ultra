"""SOURCE/STDLIB MOCK; synthetic bytes only, no real History or native work."""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_source import io, c, HERE


class HostEnvelope(unittest.TestCase):
    @contextmanager
    def synthetic(self, temp):
        path = Path(temp) / 'exact-inherited-envelope.json'
        payload = b' ' * (8804235 - 2) + b'{}'
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        with patch.object(io, 'HOST_SOURCE_ENVELOPE', (str(path), sha, len(payload))):
            yield path, payload, sha

    def test_exact_real_pin_is_literal_and_original_limits_stay_unchanged(self):
        self.assertEqual(io.HOST_SOURCE_ENVELOPE, (
            str(c.Q / 'qwen35-sampler-r10-20261112-spec-v12.json'),
            '86a4c46dafcd3c16c74101caef26604af8e4b4041bcc3c6406faeab60d8d433f', 8804235))
        self.assertEqual(c.LIMITS['spec_envelope_bytes'], 64 * 1024**2)
        self.assertEqual(c.LIMITS['metadata_bytes'], 8 * 1024**2)
        self.assertEqual(io.raw.__defaults__, (8 * 1024**2,))

    def test_complete_envelope_read_and_hash_use_original_stable_reader(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            with patch.object(io, 'raw', wraps=io.raw) as raw:
                self.assertEqual(io.host_source_bytes(str(path), sha), payload)
                raw.assert_called_once_with(str(path), 64 * 1024**2)

    def test_wrong_selected_sha_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            with patch.object(io, 'raw') as raw, patch.object(io, 'identity') as identity:
                with self.assertRaisesRegex(ValueError, 'exact inherited host envelope digest'):
                    io.host_source_bytes(str(path), '0' * 64)
                raw.assert_not_called(); identity.assert_not_called()

    def test_short_or_long_envelope_stat_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            for difference in (-1, 1):
                path.write_bytes(payload + b'x' if difference == 1 else payload[:-1])
                with patch.object(io, 'raw') as raw:
                    with self.assertRaisesRegex(ValueError, 'length required before reading'):
                        io.host_source_bytes(str(path), sha)
                    raw.assert_not_called()

    def test_same_size_changed_body_is_rejected_by_complete_digest(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            path.write_bytes(b'x' + payload[1:])
            with self.assertRaisesRegex(ValueError, 'complete host source digest'):
                io.host_source_bytes(str(path), sha)

    def test_path_alias_or_other_large_json_never_gets_envelope_cap(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            other = Path(temp) / 'other-spec.json'; other.write_bytes(payload)
            (Path(temp) / 'nested').mkdir()
            alias = str(Path(temp) / 'nested') + '/../' + path.name
            with patch.object(io, 'raw', wraps=io.raw) as raw:
                for candidate in (str(other), alias):
                    with self.assertRaises(ValueError): io.host_source_bytes(candidate, sha)
                self.assertEqual([call.args[1] for call in raw.call_args_list], [4 * 1024**2] * 2)

    def test_ordinary_host_source_four_mib_limit_remains(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ordinary.py'; payload = b' ' * (4 * 1024**2)
            path.write_bytes(payload); sha = hashlib.sha256(payload).hexdigest()
            self.assertEqual(io.host_source_bytes(str(path), sha), payload)
            path.write_bytes(payload + b' ')
            with self.assertRaisesRegex(ValueError, 'bounded metadata read'):
                io.host_source_bytes(str(path), hashlib.sha256(payload + b' ').hexdigest())

    def test_default_metadata_eight_mib_is_not_widened_even_for_special_path(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            with self.assertRaisesRegex(ValueError, 'bounded metadata read'): io.raw(path)
            with self.assertRaisesRegex(ValueError, 'bounded metadata read'):
                io.read({'path': str(path), 'sha256': sha})

    def test_symlink_and_hardlink_still_rejected_by_original_identity(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            target = path.with_name('target'); path.rename(target); path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'canonical metadata path'): io.host_source_bytes(str(path), sha)
            path.unlink(); io.os.link(target, path)
            with self.assertRaisesRegex(ValueError, 'hardlink'): io.host_source_bytes(str(path), sha)

    def test_before_and_after_identity_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            before = io.identity(path); after = dict(before, inode=before['inode'] + 1)
            with patch.object(io, 'identity', side_effect=[before, after]), patch.object(io, 'raw', return_value=payload):
                with self.assertRaisesRegex(ValueError, 'length/identity changed'): io.host_source_bytes(str(path), sha)

    def test_reader_returns_no_truncated_payload(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            before = io.identity(path)
            with patch.object(io, 'identity', return_value=before), patch.object(io, 'raw', return_value=payload[:-1]):
                with self.assertRaisesRegex(ValueError, 'length/identity changed'): io.host_source_bytes(str(path), sha)

    def test_reader_changes_during_read_still_fail_original_raw(self):
        with tempfile.TemporaryDirectory() as temp, self.synthetic(temp) as (path, payload, sha):
            before = io.identity(path); changed = dict(before, mtime_ns=before['mtime_ns'] + 1)
            with patch.object(io, 'identity', side_effect=[before, before, changed]):
                with self.assertRaisesRegex(ValueError, 'metadata changed while reading'):
                    io.host_source_bytes(str(path), sha)


class HostCallSitePreservation(unittest.TestCase):
    def test_only_three_host_reads_are_substituted_and_all_other_bodies_unchanged(self):
        predecessor = HERE.parent / 'qwen35_package_serial_runtime_v3'
        class RevertHostReader(ast.NodeTransformer):
            count = 0
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == 'host_source_bytes':
                    self.count += 1
                    return ast.copy_location(ast.parse('raw(path, 4 * 1024**2)', mode='eval').body, node)
                return node
            def visit_ImportFrom(self, node):
                node.names = [name for name in node.names if name.name != 'host_source_bytes']
                return node
        for filename in ('history.py', 'adapter.py', 'final_audit.py'):
            old = ast.parse((predecessor / filename).read_text())
            new = ast.parse((HERE / filename).read_text())
            revert = RevertHostReader(); restored = revert.visit(new)
            self.assertEqual(revert.count, 1, filename)
            self.assertEqual(ast.dump(restored, include_attributes=False), ast.dump(old, include_attributes=False), filename)

    def test_old_io_functions_including_raw_loaded_sources_and_storage_are_exact_ast(self):
        predecessor = HERE.parent / 'qwen35_package_serial_runtime_v3'
        def nodes(path):
            return {item.name: ast.dump(item, include_attributes=False) for item in ast.parse(path.read_text()).body
                    if isinstance(item, (ast.FunctionDef, ast.ClassDef))}
        old, new = nodes(predecessor / 'io.py'), nodes(HERE / 'io.py')
        self.assertEqual(set(new) - set(old), {'host_source_bytes'})
        self.assertEqual(old, {name: new[name] for name in old})

    def test_every_other_production_source_is_byte_identical_and_run_only_relocated(self):
        predecessor = HERE.parent / 'qwen35_package_serial_runtime_v3'
        for name in ('contract.py', 'budget.py', 'service.py', 'readiness.py', 'wiring.py', '__init__.py',
                     'source_files.json', 'budget.example.json'):
            self.assertEqual((HERE / name).read_bytes(), (predecessor / name).read_bytes(), name)
        self.assertEqual((HERE / 'run.py').read_text(), (predecessor / 'run.py').read_text().replace(
            'qwen35_package_serial_runtime_v3', 'qwen35_package_serial_runtime_v4'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
