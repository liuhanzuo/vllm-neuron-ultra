"""Real temporary files exercise pinned source reader's mutation boundaries."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import nemotron4b_pinned_source_reader_v1 as reader


class Reader(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nano4b-hardlink-reader-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'module.py'
        self.path.write_bytes(b'VALUE=211\n')
        self.peer=self.root/'original-peer.py';os.link(self.path,self.peer)
        self.reference=dict(bytes=self.path.stat().st_size,
            sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            fullstat=reader.identity(self.path.stat()),original_nlink=2)

    def test_reads_exact_bytes_without_changing_original_identity(self):
        raw,actual=reader.source(self.path,self.reference)
        self.assertEqual(raw,b'VALUE=211\n')
        self.assertEqual(actual['fullstat'],self.reference['fullstat'])
        self.assertEqual(reader.identity(self.peer.stat()),self.reference['fullstat'])

    def test_wrong_digest_rejected_with_original_identity(self):
        self.reference['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'digest'):reader.source(self.path,self.reference)

    def test_forged_link_count_policy_is_rejected(self):
        self.reference['original_nlink']=1
        with self.assertRaisesRegex(ValueError,'link-count policy'):reader.source(self.path,self.reference)

    def test_same_bytes_new_inode_rejected(self):
        other=self.root/'replacement.py';other.write_bytes(self.path.read_bytes())
        os.replace(other,self.path)
        with self.assertRaisesRegex(ValueError,'identity'):reader.source(self.path,self.reference)

    def test_change_via_other_hardlink_during_read_is_detected(self):
        original=reader.os.read;called=False
        def read(fd,n):
            nonlocal called
            data=original(fd,n)
            if data and not called:
                called=True
                self.peer.write_bytes(b'VALUE=223\n')
            return data
        reader.os.read=read
        try:
            with self.assertRaisesRegex(ValueError,'identity unchanged'):reader.source(self.path,self.reference)
        finally:reader.os.read=original
        self.assertTrue(called)

    def test_additional_link_during_read_is_detected(self):
        original=reader.os.read;called=False
        def read(fd,n):
            nonlocal called
            data=original(fd,n)
            if data and not called:
                called=True;os.link(self.path,self.root/'unreviewed-third.py')
            return data
        reader.os.read=read
        try:
            with self.assertRaisesRegex(ValueError,'identity unchanged'):reader.source(self.path,self.reference)
        finally:reader.os.read=original
        self.assertTrue(called)

    def test_io_exception_closes_exact_opened_descriptor(self):
        original_read,original_open=reader.os.read,reader.os.open;opened=[]
        def open_file(*a,**kw):
            fd=original_open(*a,**kw);opened.append(fd);return fd
        def fail(*a):raise OSError('Synthetic source read failure')
        reader.os.open=open_file;reader.os.read=fail
        try:
            with self.assertRaisesRegex(OSError,'Synthetic'):reader.source(self.path,self.reference)
        finally:reader.os.open=original_open;reader.os.read=original_read
        self.assertEqual(len(opened),1)
        with self.assertRaises(OSError):os.fstat(opened[0])

    def test_native_suffix_never_opened(self):
        path=self.root/'library.so';path.write_bytes(b'not native payload')
        reference=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fullstat=reader.identity(path.stat()))
        with self.assertRaisesRegex(ValueError,'Python source'):reader.source(path,reference)

    def test_unapproved_symlink_rejected(self):
        link=self.root/'alias.py';link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError,'Canonical'):reader.source(link,self.reference)

    def test_singlelink_legacy_record_remains_compatible(self):
        self.peer.unlink()
        self.reference['fullstat']=reader.identity(self.path.stat())
        self.reference.pop('original_nlink')
        raw,actual=reader.source(self.path,self.reference)
        self.assertEqual(raw,b'VALUE=211\n')
        self.assertEqual(actual['original_nlink'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
