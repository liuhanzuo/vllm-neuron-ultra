"""Lossless archive and exact-target safety checks; no real journal mutation."""
import copy
import gzip
import io
import stat
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import compact_closed_journals_v1 as m


class Checks(unittest.TestCase):
    def plan(self):
        rows = []
        for parent, count in [(m.ROOT_JOURNALS, 4), (m.OLD_ARCHIVES, 12)]:
            for i in range(count):
                original = 1793491200*10**9
                actual = original if parent == m.ROOT_JOURNALS else 1794209998*10**9
                rows.append(dict(path=str(parent/f'user-1000@synthetic-{i}.journal'),
                    stat=dict(st_size=128*2**20,st_mtime_ns=actual,st_nlink=1,st_mode=stat.S_IFREG|0o600),
                    original_closed_mtime_ns=original))
        return dict(schema='closed_journals_compaction_plan_v1', completion_exit=m.completion.EXIT,
                    output='/mnt/nvme/closed-journals-compaction-20261109-r1',files=rows,
                    compressed_cap_bytes=400*2**20,root_floor_bytes=28*2**30,nvme_floor_bytes=30*2**30)

    def test_old_archives_use_original_close_time_instead_of_copy_time(self):
        m.validate(self.plan())

    def test_active_or_recent_root_journal_cannot_be_selected(self):
        for mutation in ('active', 'recent', 'invented_close_time'):
            plan = self.plan()
            row = plan['files'][0]
            if mutation == 'active':
                row['path'] = str(m.ROOT_JOURNALS/'user-1000.journal')
            elif mutation == 'recent':
                row['original_closed_mtime_ns'] = row['stat']['st_mtime_ns'] = 1794209998*10**9
            else:
                row['stat']['st_mtime_ns'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                m.validate(plan)

    def test_complete_roundtrip_keeps_original_until_caller_retires_it(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'input', Path(temp)/'archive.gz'
            data = bytes(range(256))*4097+b'last byte'
            source.write_bytes(data)
            result = m.archive(source, target, m.identity(source), lambda: None)
            self.assertEqual(source.read_bytes(), data)
            self.assertEqual(gzip.decompress(target.read_bytes()), data)
            self.assertEqual(result['original_sha256'], m.sha(data))

    def test_bad_decompressed_bytes_prevent_retirement(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'input', Path(temp)/'archive.gz'
            source.write_bytes(b'original payload')
            with patch.object(m.gzip, 'open', return_value=io.BytesIO(b'bad')):
                with self.assertRaisesRegex(RuntimeError, 'Every decompressed byte'):
                    m.archive(source, target, m.identity(source), lambda: None)
            self.assertEqual(source.read_bytes(), b'original payload')

    def test_changed_source_during_compression_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'input', Path(temp)/'archive.gz'
            source.write_bytes(b'original payload')
            def mutate():
                source.write_bytes(b'changed payload')
            with self.assertRaisesRegex(RuntimeError, 'Stable complete'):
                m.archive(source, target, m.identity(source), mutate)
            self.assertTrue(source.exists())

    def test_existing_archive_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'input', Path(temp)/'archive.gz'
            source.write_bytes(b'original')
            target.write_bytes(b'kept')
            with self.assertRaises(FileExistsError):
                m.archive(source, target, m.identity(source), lambda: None)
            self.assertEqual(target.read_bytes(), b'kept')

    def test_symlink_and_shared_inode_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp)/'input', Path(temp)/'alias'
            source.write_bytes(b'payload')
            target.symlink_to(source)
            with self.assertRaises(RuntimeError):
                m.identity(target)
            target.unlink()
            target.hardlink_to(source)
            with self.assertRaises(RuntimeError):
                m.identity(source)

    def test_original_runtime_spec_uses_existing_bounded_envelope(self):
        spec = m.p.metadata.load(dict(path=str(Q/'nemotron9b-package-runtime-v16-r8-20261109-spec.json'),
                sha256='9cfa93c91e0760a2a994375578476ca65899638923d8e562ab44bb79a2a2f27e'))
        self.assertEqual(len(spec['stages']), 52)


if __name__ == '__main__':
    unittest.main(verbosity=2)
