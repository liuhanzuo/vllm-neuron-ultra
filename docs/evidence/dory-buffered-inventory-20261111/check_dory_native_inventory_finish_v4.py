"""Pure mocked JSON writer tests; no real output/raw/process/lock access."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research import dory_native_inventory_finish_v3 as old
from research import dory_native_inventory_finish_v4 as v


class FakePath:
    def __init__(self,fs,name):self.fs=fs;self.name=str(name)
    def __str__(self):return self.name
    def __eq__(self,other):return isinstance(other,FakePath)and self.name==other.name
    @property
    def parent(self):return FakePath(self.fs,str(Path(self.name).parent))
    def is_absolute(self):return self.name.startswith('/')
    def resolve(self):return self
    def lstat(self):
        info=self.fs.info()
        if self.fs.identity_drift:info.st_ino+=1
        return info
    def open(self,mode):
        assert mode=='xb';self.fs.events.append(('create',self.name))
        if self.fs.exists:raise FileExistsError(self.name)
        self.fs.exists=True;self.fs.writer_open=True;return self.fs


class MemoryFS:
    def __init__(self):
        self.data=bytearray();self.events=[];self.exists=False;self.writer_open=False;self.handles={};self.next_fd=20
        self.short_write=False;self.sync_failure=None;self.read_corruption=False;self.identity_drift=False
        self.requests=[];self.write_sizes=[]
        self.os=SimpleNamespace(**{name:getattr(os,name)for name in ('O_RDONLY','O_DIRECTORY','O_NOFOLLOW','O_NONBLOCK')},
            open=self.open,close=self.close,fsync=self.fsync,read=self.read,fstat=self.fstat)
    def path(self,name):return name if isinstance(name,FakePath)else FakePath(self,name)
    def __enter__(self):return self
    def __exit__(self,*a):self.writer_open=False;self.events.append(('writer-close',))
    def write(self,block):
        self.events.append(('write',len(block)));self.write_sizes.append(len(block))
        count=max(0,len(block)-1)if self.short_write else len(block);self.data.extend(block[:count]);return count
    def flush(self):self.events.append(('flush',))
    def fileno(self):return 10
    def fsync(self,fd):
        kind='file'if fd==10 else'directory';self.events.append(('fsync',kind))
        if self.sync_failure==kind:raise OSError('injected '+kind+' fsync failure')
    def open(self,path,flags):
        directory=bool(flags&os.O_DIRECTORY);self.events.append(('open-directory'if directory else'open-read',str(path),flags))
        assert flags&os.O_NOFOLLOW
        fd=self.next_fd;self.next_fd+=1;self.handles[fd]=dict(directory=directory,offset=0);return fd
    def close(self,fd):self.events.append(('close',fd));del self.handles[fd]
    def info(self):return SimpleNamespace(st_dev=1,st_ino=7,st_mode=stat.S_IFREG|0o600,st_nlink=1,
        st_size=len(self.data),st_mtime_ns=1,st_ctime_ns=1,st_uid=1000)
    def fstat(self,fd):return self.info()
    def read(self,fd,size):
        item=self.handles[fd];assert not item['directory'];self.events.append(('read',size));self.requests.append(size)
        offset=item['offset'];out=bytes(self.data[offset:offset+size]);item['offset']+=len(out)
        if self.read_corruption and offset==0 and out:out=b'X'+out[1:]
        return out


class Run:
    def __init__(self,module):
        self.module=module;self.fs=MemoryFS();self.guard_count=0;self.reject_after=None;self.add_sizes=[]
        def guard():
            self.guard_count+=1
            if self.reject_after==self.guard_count:raise RuntimeError('injected guard loss')
        self.guard=guard
    @contextmanager
    def mocked(self):
        m=self.module
        with patch.object(m,'Path',side_effect=self.fs.path),patch.object(m,'os',self.fs.os),patch.object(m,'time',SimpleNamespace(monotonic=lambda:0)):
            budget=m.Budget(self.guard);original=budget.add
            def add(key,size):self.add_sizes.append(size);return original(key,size)
            budget.add=add;self.budget=budget;yield self
    def write(self,value):return self.module.write_json('/synthetic/result.json',value,self.budget)


def original_bytes(value):return ''.join(json.JSONEncoder(separators=(',',':'),allow_nan=False).iterencode(value)).encode()


class BufferedWriter(unittest.TestCase):
    def test_exact_bytes_against_original_nested_unicode_and_large_strings(self):
        values=[None,True,0,-1.25,{},[],{'nested':[1,{'x':None,'text':'plain'},[False,1e-10]],'empty':''},
            {'unicode':'中文🙂\u2028\ud800','escaped':'\\"\n\t\b\f\r','keys':{'é':'𝄞'}},
            {'large':'汉🙂\\"'*40000,'after':[1,2,3]},
            [{'index':i,'path':'/synthetic/pkg/module'+str(i)+'.py','hash':'a'*64,'bytes':i}for i in range(2000)]]
        for value in values:
            with self.subTest(kind=type(value).__name__):
                previous=Run(old);current=Run(v)
                with previous.mocked():a=previous.write(value)
                with current.mocked():b=current.write(value)
                expected=original_bytes(value)
                self.assertEqual(bytes(previous.fs.data),expected);self.assertEqual(bytes(current.fs.data),expected)
                self.assertEqual(a,b);self.assertEqual(b['sha256'],hashlib.sha256(expected).hexdigest())
                self.assertEqual(current.budget.used,previous.budget.used)
                self.assertTrue(all(0<size<=65536 for size in current.fs.write_sizes))
                self.assertTrue(all(size==65536 for size in current.fs.write_sizes[:-1]))
                self.assertFalse(current.fs.handles);self.assertFalse(current.fs.writer_open)
    def test_budget_guard_and_write_calls_scale_with_64k_chunks_not_encoder_fragments(self):
        value=[{'path':'/synthetic/'+str(i),'bytes':i,'sha256':'a'*64}for i in range(3000)]
        before=Run(old);after=Run(v)
        with before.mocked():before.write(value)
        with after.mocked():after.write(value)
        size=len(original_bytes(value));chunks=(size+65535)//65536
        self.assertEqual(len(after.add_sizes),chunks);self.assertEqual(len(after.fs.write_sizes),chunks)
        self.assertEqual(after.guard_count,1+chunks+1+chunks)
        self.assertGreater(len(before.add_sizes),1000*len(after.add_sizes))
        self.assertEqual(sum(after.add_sizes),size);self.assertEqual(before.fs.data,after.fs.data)
        self.assertEqual(after.fs.requests[-1],1)
    def test_full_file_and_directory_fsync_before_original_complete_readback(self):
        run=Run(v)
        with run.mocked():run.write({'large':'x'*140000})
        events=run.fs.events
        for event in [('flush',),('fsync','file'),('writer-close',),('fsync','directory')]:self.assertIn(event,events)
        self.assertLess(events.index(('flush',)),events.index(('fsync','file')))
        self.assertLess(events.index(('fsync','file')),events.index(('writer-close',)))
        self.assertLess(events.index(('writer-close',)),events.index(('fsync','directory')))
        self.assertLess(events.index(('fsync','directory')),next(i for i,e in enumerate(events)if e[0]=='open-read'))
        self.assertEqual(sum(run.fs.requests[:-1]),len(run.fs.data));self.assertFalse(run.fs.handles)
    def test_per_file_cap_stops_before_next_write_and_preserves_prefix(self):
        value={'large':'x'*200000};run=Run(v)
        with run.mocked(),patch.dict(v.LIMITS,per_output_bytes=100000):
            with self.assertRaisesRegex(RuntimeError,'Single output bound'):run.write(value)
        self.assertTrue(run.fs.exists);self.assertEqual(bytes(run.fs.data),original_bytes(value)[:65536])
        self.assertEqual(run.add_sizes,[65536]);self.assertEqual(run.budget.used['output_bytes'],65536)
        self.assertFalse(run.fs.writer_open);self.assertFalse(any(e[0].startswith('open-')for e in run.fs.events))
    def test_aggregate_cap_stops_before_next_write_without_deleting_partial(self):
        value={'large':'x'*200000};run=Run(v)
        with run.mocked():
            run.budget.used['output_bytes']=v.LIMITS['output_bytes']-65536
            with self.assertRaisesRegex(RuntimeError,'output_bytes cap'):run.write(value)
        self.assertEqual(bytes(run.fs.data),original_bytes(value)[:65536]);self.assertTrue(run.fs.exists)
        self.assertEqual(run.add_sizes,[65536,65536]);self.assertFalse(run.fs.writer_open)
    def test_nonfinite_encoder_error_retains_prior_flushed_prefix_and_original_exception(self):
        for bad in (float('nan'),float('inf'),-float('inf')):
            value=['x'*140000,bad];run=Run(v)
            with run.mocked(),self.assertRaises(ValueError):run.write(value)
            self.assertTrue(run.fs.exists);self.assertEqual(bytes(run.fs.data),b'["'+b'x'*(131072-2))
            self.assertFalse(run.fs.writer_open);self.assertFalse(run.fs.handles)
            self.assertFalse(any(e[0]=='fsync'for e in run.fs.events))
    def test_short_write_guard_loss_and_sync_errors_keep_partial_file_and_exception(self):
        for fault in ('short','guard','file-sync','directory-sync'):
            run=Run(v);value={'large':'x'*140000}
            if fault=='short':run.fs.short_write=True
            elif fault=='guard':run.reject_after=3
            else:run.fs.sync_failure=fault.split('-')[0]
            expected=OSError if fault.endswith('sync')else RuntimeError
            with self.subTest(fault=fault),run.mocked(),self.assertRaises(expected):run.write(value)
            self.assertTrue(run.fs.exists);self.assertGreater(len(run.fs.data),0);self.assertFalse(run.fs.writer_open)
            self.assertFalse(run.fs.handles)
    def test_preencode_guard_rejects_before_any_output_bytes(self):
        run=Run(v);run.reject_after=1
        with run.mocked(),self.assertRaisesRegex(RuntimeError,'guard loss'):run.write({'x':'y'})
        self.assertTrue(run.fs.exists);self.assertEqual(run.fs.data,b'');self.assertFalse(run.fs.writer_open)
    def test_original_complete_readback_hash_and_identity_still_reject_corruption(self):
        for fault in ('hash','identity'):
            run=Run(v)
            if fault=='hash':run.fs.read_corruption=True
            else:run.fs.identity_drift=True
            with self.subTest(fault=fault),run.mocked(),self.assertRaises(RuntimeError):run.write({'x':'y'*100000})
            self.assertTrue(run.fs.exists);self.assertGreater(len(run.fs.data),65536);self.assertFalse(run.fs.handles)
            self.assertIn(('fsync','directory'),run.fs.events)
    def test_exclusive_create_cannot_overwrite_old_partial_or_success(self):
        run=Run(v)
        with run.mocked():
            run.write({'x':1});saved=bytes(run.fs.data)
            with self.assertRaises(FileExistsError):run.write({'x':2})
        self.assertEqual(run.fs.data,saved)
    def test_chunk_generator_boundaries_and_no_empty_blocks(self):
        for length in (0,1,65533,65534,65535,65536,65537,131070,131071):
            value='x'*length;blocks=list(v.json_blocks(value));expected=original_bytes(value)
            self.assertEqual(b''.join(blocks),expected)
            self.assertEqual([len(b)for b in blocks],[65536]*(len(expected)//65536)+([len(expected)%65536]if len(expected)%65536 else[]))
            self.assertTrue(all(type(b)is bytearray and 0<len(b)<=65536 for b in blocks))
    def test_only_buffered_writer_and_source_successor_label_changed(self):
        def parse(module):
            tree=ast.parse(Path(module.__file__).read_text())
            functions={n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
            return functions
        before=parse(old);after=parse(v)
        self.assertEqual(set(after)-set(before),{'json_blocks'})
        for name in before:
            if name not in ('write_json','run'):self.assertEqual(before[name],after[name],name)
        self.assertEqual(before['run'].replace('dory_native_inventory_finish_v3.py','dory_native_inventory_finish_v4.py'),after['run'])
        self.assertEqual(old.LIMITS,v.LIMITS);self.assertEqual(old.PINS,v.PINS)
        self.assertEqual(old.SYSTEM_HARDLINK_PATHS,v.SYSTEM_HARDLINK_PATHS)
        self.assertFalse({'torch','numpy','transformers','torch_neuronx','neuronxcc'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
