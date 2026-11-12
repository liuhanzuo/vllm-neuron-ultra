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


if __name__ == '__main__':
    unittest.main(verbosity=2)
