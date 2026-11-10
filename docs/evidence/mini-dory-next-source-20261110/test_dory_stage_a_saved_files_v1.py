"""Tiny disposable files only; no model, archives or native imports."""
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import dory_stage_a_saved_files_v1 as core


def fixtures(raw=b'\x80\x3f'*8):
    desc=dict(shape=[2,4],dtype='BF16',finite=True,offset=0,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    archive=dict(complete=True,bytes=len(raw),sha256=desc['sha256'],tensors={'t':desc})
    layout={'t':dict(shape=[2,4],dtype='BF16')}
    return raw,archive,layout


class StableFileChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dory-stage-a-file-check-')
        self.root=Path(self.tmp.name);self.file=self.root/'capture.bf16'
        self.raw,self.archive,self.layout=fixtures();self.file.write_bytes(self.raw)
        self.expected=core.identity(self.file.stat())
    def tearDown(self):self.tmp.cleanup()
    def run_verify(self,**kwargs):
        args=dict(root=self.root,name=self.file.name,producer_identity=self.expected,archive=self.archive,
                  layout=self.layout,max_bytes=16,guard=lambda:None,expected_tensors=self.archive['tensors'])
        args.update(kwargs);return core.verify_file(**args)

    def test_complete_stable_file_binds_identity_and_all_bytes(self):
        r=self.run_verify();self.assertTrue(r['complete']);self.assertTrue(r['whole_file_from_offset_zero'])
        self.assertEqual(r['bytes_verified'],16);self.assertEqual(r['file_identity'],self.expected)
        self.assertFalse(r['full_model_qualified']);self.assertFalse(r['runtime_admitted_here'])

    def test_prefix_and_trailing_bytes_rejected_even_with_fresh_stat(self):
        for raw in (b'xx'+self.raw,self.raw+b'xx'):
            self.file.write_bytes(raw)
            with self.assertRaises(RuntimeError):self.run_verify(producer_identity=core.identity(self.file.stat()))

    def test_nonzero_stream_offset_rejected_before_read(self):
        stream=io.BytesIO(b'prefix'+self.raw);stream.seek(6)
        with self.assertRaises(RuntimeError):core.BoundedReader(stream,16,lambda:None)

    def test_replaced_producer_file_rejected(self):
        other=self.root/'other';other.write_bytes(self.raw);other.replace(self.file)
        with self.assertRaises(RuntimeError):self.run_verify()

    def test_modified_timestamp_rejected(self):
        ns=self.file.stat().st_mtime_ns;os.utime(self.file,ns=(ns,ns+1000000))
        with self.assertRaises(RuntimeError):self.run_verify()

    def test_link_and_hardlink_rejected(self):
        link=self.root/'link';link.symlink_to(self.file)
        with self.assertRaises(OSError):self.run_verify(name='link')
        link.unlink();os.link(self.file,link)
        with self.assertRaises(RuntimeError):self.run_verify()

    def test_fifo_rejected_without_blocking(self):
        fifo=self.root/'fifo';os.mkfifo(fifo)
        with self.assertRaises(RuntimeError):self.run_verify(name='fifo')

    def test_parent_escape_and_symlink_parent_rejected(self):
        for name in ('../capture.bf16','.',str(self.file)):
            with self.assertRaises(RuntimeError):self.run_verify(name=name)
        link=self.root/'linked';link.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(RuntimeError):self.run_verify(root=link)

    def test_bound_and_guard_reject_before_open(self):
        with self.assertRaises(RuntimeError):self.run_verify(max_bytes=15)
        def deny():raise RuntimeError('unadmitted')
        with self.assertRaisesRegex(RuntimeError,'unadmitted'):self.run_verify(guard=deny,name='missing')

    def test_same_byte_replacement_during_review_rejected(self):
        calls=[0]
        def guard():
            calls[0]+=1
            if calls[0]==3:
                other=self.root/'replacement';other.write_bytes(self.raw);other.replace(self.file)
        with self.assertRaises(RuntimeError):self.run_verify(guard=guard)

    def test_same_inode_content_change_during_review_rejected(self):
        calls=[0]
        def guard():
            calls[0]+=1
            if calls[0]==3:self.file.write_bytes(b'\x00\x40'*8)
        with self.assertRaises(RuntimeError):self.run_verify(guard=guard)

    def test_parent_directory_replacement_detected(self):
        calls=[0];moved=self.root.parent/(self.root.name+'-moved')
        def guard():
            calls[0]+=1
            if calls[0]==3:self.root.rename(moved);self.root.mkdir()
        try:
            with self.assertRaises(RuntimeError):self.run_verify(guard=guard)
        finally:
            self.root.rmdir();moved.rename(self.root)

    def test_ancestor_symlink_change_during_read_rejected(self):
        parent=self.root/'parent';parent.mkdir();nested=parent/'nested';nested.mkdir()
        target=nested/'capture.bf16';target.write_bytes(self.raw)
        expected=core.identity(target.stat());moved=self.root/'moved';calls=[0]
        def guard():
            calls[0]+=1
            if calls[0]==3:parent.rename(moved);parent.symlink_to(moved,target_is_directory=True)
        try:
            with self.assertRaisesRegex(RuntimeError,'symlink ancestors'):
                self.run_verify(root=nested,producer_identity=expected,guard=guard)
        finally:
            if parent.is_symlink():parent.unlink();moved.rename(parent)

    def test_fdopen_failure_closes_opened_file_descriptor(self):
        real_open=core.os.open;opened=[]
        def tracking_open(*args,**kwargs):
            fd=real_open(*args,**kwargs);opened.append(fd);return fd
        with patch.object(core.os,'open',tracking_open), patch.object(core.os,'fdopen',side_effect=OSError('fdopen failed')):
            with self.assertRaisesRegex(OSError,'fdopen failed'):self.run_verify()
        self.assertEqual(len(opened),2)
        for fd in opened:
            with self.assertRaises(OSError):os.fstat(fd)

    def test_fdopen_failure_after_close_preserves_primary_error(self):
        def closed_failure(fd,*args,**kwargs):
            os.close(fd)
            raise OSError('original construction error')
        with patch.object(core.os,'fdopen',side_effect=closed_failure):
            with self.assertRaisesRegex(OSError,'original construction error'):self.run_verify()

    def test_nonfinite_or_corrupted_file_fails_even_with_new_identity(self):
        for raw in (b'\x80\x7f'+self.raw[2:],b'\x00\x40'+self.raw[2:]):
            self.file.write_bytes(raw)
            with self.assertRaises(RuntimeError):self.run_verify(producer_identity=core.identity(self.file.stat()))

    def test_bounded_reader_denies_seek_unbounded_or_overread(self):
        reader=core.BoundedReader(io.BytesIO(self.raw),16,lambda:None)
        for size in (-1,0,17,65537):
            with self.assertRaises(RuntimeError):reader.read(size)
        self.assertEqual(reader.read(16),self.raw);self.assertEqual(reader.read(1),b'')
        reader.stream.seek(1)
        with self.assertRaises(RuntimeError):reader.read(1)

    def test_no_native_imports(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
