"""Stdlib, BytesIO and fake filesystem only; no disk payloads/native imports.

Tiny stream tests run the production NativeTensorAdapter/TensorEmitter/writer
chain on FakeTorch. Complete six-case coordinator tests use real pinned JSON
and synthetic descriptors, never allocate or claim verification of large data.
"""
from copy import deepcopy
import io
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dory_stage_a_artifact_transaction_v1 as core
from dory_stage_a_saved_core_v1 import verify_archive, stage_a_requirements
from dory_stage_a_tensor_adapter_v1 import NativeTensorAdapter
from test_dory_stage_a_tensor_adapter_v1 import FakeTorch, patterned, independent_expected
from test_dory_stage_a_source_core_v1 import metadata_cases, bind_cases


class FakeFile:
    def __init__(self, fs, name, fd):
        self.fs, self.name, self.fd = fs, name, fd
        self.buffer, self.closed = io.BytesIO(), False
        self.data = b''

    def fileno(self):
        assert not self.closed
        return self.fd

    def write(self, block):
        self.fs.action('write', self.name)
        assert not self.closed
        n = self.buffer.write(block)
        self.data = self.buffer.getvalue()
        return n

    def flush(self):
        self.fs.action('flush', self.name)
        assert not self.closed

    def close(self):
        self.fs.action('close_file', self.name)
        self.closed = True
        self.fs.fds.pop(self.fd, None)


class FakeFS:
    # Entire module dependency is replaced, so no actual OS payload path exists.
    def __init__(self):
        for name in ('O_RDONLY','O_DIRECTORY','O_NOFOLLOW','O_CREAT','O_EXCL','O_WRONLY','O_NONBLOCK'):
            setattr(self, name, getattr(os, name))
        self.events, self.entries, self.fds = [], {}, {}
        self.next_fd, self.fail, self.rewrite, self.alias, self.root_inode = 100, None, None, False, 1
        self.uid, self.file_nlink, self.file_mode, self.file_uid = os.getuid(), 1, stat.S_IFREG | 0o600, os.getuid()
        self.fdopen_closes_before_failure = False

    def getuid(self): return self.uid

    def action(self, operation, name):
        self.events.append((operation, name))
        if (operation, name) == self.fail:
            raise OSError('injected '+operation)

    def info(self, name):
        directory = name == '/admitted'
        entry = self.entries.get(name)
        return SimpleNamespace(st_mode=stat.S_IFDIR | 0o700 if directory else self.file_mode,
            st_uid=self.uid if directory else self.file_uid, st_nlink=1 if directory else self.file_nlink,
            st_dev=1, st_ino=self.root_inode if directory else entry.fd,
            st_size=0 if directory else len(entry.data), st_mtime_ns=7, st_ctime_ns=9)

    def Path(self, value):
        fs = self
        class MemoryPath:
            def __init__(self, text): self.text = str(text)
            def __str__(self): return self.text
            def __eq__(self, other): return isinstance(other, MemoryPath) and self.text == other.text
            @property
            def name(self): return PurePosixPath(self.text).name
            def is_absolute(self): return PurePosixPath(self.text).is_absolute()
            def is_dir(self): return self.text == '/admitted'
            def resolve(self): return self if not fs.alias else MemoryPath('/changed')
            def stat(self, follow_symlinks=False): return fs.info(self.text)
        return MemoryPath(value)

    def open(self, path, flags, mode=None, *, dir_fd=None):
        name = str(path)
        self.action('open', name)
        if dir_fd is None:
            assert name == '/admitted' and flags == self.O_RDONLY | self.O_DIRECTORY | self.O_NOFOLLOW
            self.fds[90] = name
            return 90
        assert dir_fd == 90 and mode == 0o600
        assert flags == self.O_CREAT | self.O_EXCL | self.O_WRONLY | self.O_NOFOLLOW | self.O_NONBLOCK
        if name in self.entries:
            raise FileExistsError(name)
        self.next_fd += 1
        entry = FakeFile(self, name, self.next_fd)
        self.entries[name] = entry
        self.fds[entry.fd] = name
        return entry.fd

    def fdopen(self, fd, mode):
        name = self.fds[fd]
        assert mode == 'wb'
        if self.fdopen_closes_before_failure and self.fail == ('fdopen', name):
            self.close(fd)
        self.action('fdopen', name)
        return self.entries[name]

    def fstat(self, fd):
        if fd not in self.fds: raise OSError('closed fd')
        return self.info(self.fds[fd])

    def stat(self, name, *, dir_fd, follow_symlinks):
        assert dir_fd == 90 and follow_symlinks is False
        return self.info(name)

    def fsync(self, fd):
        name = self.fds[fd]
        self.action('fsync', name)
        if name == '/admitted' and self.rewrite:
            self.rewrite()

    def close(self, fd):
        if fd not in self.fds: raise OSError('already closed')
        self.action('close_fd', self.fds[fd])
        self.fds.pop(fd)


class TinyTransaction(unittest.TestCase):
    def setUp(self):
        self.fs = FakeFS()
        self.patches = [patch.object(core, 'os', self.fs), patch.object(core, 'Path', self.fs.Path)]
        for p in self.patches: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])
        self.store = core.ArchiveFileStore('/admitted', guard=lambda: None)
        self.addCleanup(self.store.abort)
        self.api = FakeTorch()
        self.adapter = NativeTensorAdapter(self.api)
        self.tensor = patterned(self.api, [2, 4], stride=[8, 1], offset=1)
        self.case = dict(identity=dict(event=1))
        self.plan = {s: {'t': dict(shape=[1 if s == 'selected' else 2, 4], dtype='BF16')}
                     for s in core.SCOPES}
        self.tokens, self.positions = object(), object()
        self.observer = SimpleNamespace(ops=self.adapter, collect=self.collect)

    def collect(self, case, tokens, positions, emit):
        self.assertIs(case, self.case)
        self.assertIs(tokens, self.tokens)
        self.assertIs(positions, self.positions)
        self.assertIs(emit.iter_bytes.__self__, self.adapter)
        return dict(records={s: {'t': emit(s, 't', self.tensor, [1] if s == 'selected' else None)}
                             for s in core.SCOPES})

    def run_capture(self, guard=lambda: None):
        return core._write_capture(self.case, self.plan, self.observer, self.adapter, self.store,
                                   guard, self.tokens, self.positions)

    def test_real_adapter_emitter_writer_bytes_and_durable_order(self):
        result = self.run_capture()
        for scope in core.SCOPES:
            file = result['files'][scope]
            expected = independent_expected(self.tensor, [1] if scope == 'selected' else None)
            self.assertEqual(self.fs.entries[file['name']].data, expected)
            checked = verify_archive(io.BytesIO(expected), file['archive'], self.plan[scope], max_bytes=len(expected))
            self.assertTrue(checked['complete'])
            self.assertTrue(self.fs.entries[file['name']].closed)
            self.assertEqual(file['producer_identity'], core.identity(self.fs.info(file['name'])))
            events = self.fs.events
            self.assertLess(events.index(('flush',file['name'])), events.index(('fsync',file['name'])))
            self.assertLess(events.index(('fsync',file['name'])), events.index(('close_file',file['name'])))
            self.assertLess(events.index(('close_file',file['name'])), events.index(('fsync','/admitted')))
        self.store.close()
        self.assertEqual(self.fs.fds, {})
        self.assertEqual(sum(len(x.data) for x in self.fs.entries.values()), 40)

    def test_exclusive_existing_file_is_not_replaced(self):
        self.store.open_exclusive('event-000001-target.bin')
        before = self.fs.entries['event-000001-target.bin']
        with self.assertRaises(RuntimeError): self.run_capture()
        self.assertIs(before, self.fs.entries['event-000001-target.bin'])
        self.assertEqual(len(self.fs.entries), 1)

    def test_basename_escape_rejected_without_open(self):
        for name in ('../x', '/x', '.', '', '..'):
            count = len(self.fs.events)
            with self.assertRaises(RuntimeError): self.store.open_exclusive(name)
            self.assertEqual(len(self.fs.events), count)

    def test_file_owner_link_and_type_rejected(self):
        for field, value in [('file_uid', self.fs.uid+1),('file_nlink',2),('file_mode',stat.S_IFIFO | 0o600)]:
            with self.subTest(field=field):
                setattr(self.fs, field, value)
                with self.assertRaises(RuntimeError): self.store.open_exclusive('bad-'+field)
                setattr(self.fs, field, {'file_uid':self.fs.uid,'file_nlink':1,'file_mode':stat.S_IFREG | 0o600}[field])

    def test_open_second_failure_preserves_first_partial(self):
        self.fs.fail = ('open', 'event-000001-selected.bin')
        with self.assertRaisesRegex(OSError, 'open'): self.run_capture()
        self.assertEqual(list(self.fs.entries), ['event-000001-target.bin'])
        self.assertEqual(self.fs.fds, {})

    def test_fdopen_failure_preserves_file_and_closes_fd(self):
        self.fs.fail = ('fdopen', 'event-000001-target.bin')
        with self.assertRaisesRegex(OSError, 'fdopen'): self.run_capture()
        self.assertEqual(len(self.fs.entries), 1)
        self.assertEqual(self.fs.fds, {})

    def test_fdopen_failure_after_close_retains_original_error(self):
        self.fs.fdopen_closes_before_failure = True
        self.test_fdopen_failure_preserves_file_and_closes_fd()

    def test_write_failure_retains_partial_and_no_seal(self):
        self.fs.fail = ('write', 'event-000001-selected.bin')
        with self.assertRaisesRegex(OSError, 'write'): self.run_capture()
        self.assertEqual(len(self.fs.entries), 3)
        self.assertEqual(len(self.fs.entries['event-000001-target.bin'].data), 16)
        self.assertNotIn(('fsync','/admitted'), self.fs.events)
        self.assertEqual(self.fs.fds, {})

    def test_adapter_failure_does_not_complete(self):
        self.api.fail_clone = True
        with self.assertRaises(Exception): self.run_capture()
        self.assertEqual(len(self.fs.entries), 3)
        self.assertNotIn(('fsync','/admitted'), self.fs.events)

    def test_observer_failure_does_not_complete(self):
        def fail(*args): raise ValueError('observer failure')
        self.observer.collect = fail
        with self.assertRaisesRegex(ValueError, 'observer failure'): self.run_capture()
        self.assertEqual(len(self.fs.entries), 3)
        self.assertEqual(self.fs.fds, {})

    def test_observer_descriptor_lie_rejected_before_seal(self):
        original = self.collect
        def corrupt(*args):
            result = original(*args)
            result['records']['target']['t']['sha256'] = '0'*64
            return result
        self.observer.collect = corrupt
        with self.assertRaisesRegex(RuntimeError, 'exactly those emitted'): self.run_capture()
        self.assertNotIn(('fsync','/admitted'), self.fs.events)

    def test_flush_failure_no_receipt(self):
        self.fs.fail = ('flush', 'event-000001-target.bin')
        with self.assertRaisesRegex(OSError, 'flush'): self.run_capture()
        self.assertEqual(self.fs.fds, {})
        self.assertEqual(len(self.fs.entries), 3)

    def test_file_fsync_failure_no_receipt(self):
        self.fs.fail = ('fsync', 'event-000001-selected.bin')
        with self.assertRaisesRegex(OSError, 'fsync'): self.run_capture()
        self.assertNotIn(('fsync','/admitted'), self.fs.events)
        self.assertEqual(self.fs.fds, {})

    def test_close_failure_no_receipt_and_partial_not_removed(self):
        self.fs.fail = ('close_file', 'event-000001-original.bin')
        with self.assertRaisesRegex(OSError, 'close_file'): self.run_capture()
        self.assertEqual(len(self.fs.entries), 3)
        self.assertNotIn(('fsync','/admitted'), self.fs.events)
        self.fs.fail = None
        self.fs.entries['event-000001-original.bin'].close()

    def test_directory_fsync_failure_no_receipt(self):
        self.fs.fail = ('fsync','/admitted')
        with self.assertRaisesRegex(OSError, 'fsync'): self.run_capture()
        self.assertEqual(len(self.fs.entries), 3)
        self.assertEqual(self.fs.fds, {})

    def test_name_replacement_after_close_rejected(self):
        self.fs.rewrite = lambda: setattr(self.fs.entries['event-000001-target.bin'], 'fd', 998)
        with self.assertRaisesRegex(RuntimeError, 'producer file identity'): self.run_capture()

    def test_parent_inode_replacement_rejected(self):
        self.fs.rewrite = lambda: setattr(self.fs, 'root_inode', 99)
        with self.assertRaisesRegex(RuntimeError, 'Same original owned directory'): self.run_capture()

    def test_ancestor_symlink_change_rejected(self):
        self.fs.rewrite = lambda: setattr(self.fs, 'alias', True)
        with self.assertRaisesRegex(RuntimeError, 'canonical ancestors'): self.run_capture()

    def test_guard_abort_before_first_output(self):
        def fail(): raise ValueError('guard expired')
        with self.assertRaisesRegex(ValueError, 'guard expired'): self.run_capture(guard=fail)
        self.assertEqual(self.fs.entries, {})
        self.assertEqual(self.fs.fds, {})

    def test_open_handle_prevents_store_close(self):
        self.store.open_exclusive('unfinished')
        with self.assertRaisesRegex(RuntimeError, 'handles closed'): self.store.close()
        self.assertIn('unfinished', self.fs.entries)


def metadata_result(case, ordinal):
    specs = stage_a_requirements(case)
    records, sources, files = {}, {}, {}
    for si, scope in enumerate(core.SCOPES):
        records[scope], sources[scope], offset = {}, {}, 0
        for label, item in specs[scope]['layout'].items():
            size = core.extent({label:item})
            old = specs[scope].get('expected_tensors', {}).get(label)
            part = old['sha256'] if old else core.digest(label)
            records[scope][label] = dict(**deepcopy(item), finite=True, offset=offset, bytes=size, sha256=part)
            shape = [case['identity']['end'],2560] if scope == 'selected' else item['shape']
            stride, n = [], 1
            for dim in reversed(shape): stride.insert(0,n); n *= dim
            sources[scope][label] = dict(shape=shape, dtype=item['dtype'], stride=stride, storage_offset=0)
            offset += size
        file_id = dict(st_dev=1, st_ino=100+ordinal*3+si, st_size=offset, st_mtime_ns=8, st_ctime_ns=9)
        archive = dict(complete=True, bytes=offset, sha256='1'*64, tensors=deepcopy(records[scope]))
        files[scope] = dict(name=core.name_for(case,scope), producer_identity=file_id, archive=archive)
    observation = dict(identity=deepcopy(case['identity']), selected_rows=list(case['selected_rows']),
        records=records, source_layouts=sources, original_top1=list(case['original_top1']),
        physical_visits=112, onednn_calls=281, original_byte_controls_matched=True,
        **{x:False for x in core.UNPROVEN})
    return dict(observation=observation, files=files)


def summary():
    return dict(stage_a_collector_complete=True, events=[x[0] for x in core.WIDE],
                independent_saved_byte_review_complete=False, all87_marked_calls_qualified=False,
                full_model_qualified=False, new_speedup_claim=False)


class CompleteMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = bind_cases(*metadata_cases())

    def setUp(self):
        self.store = SimpleNamespace(close=lambda:None, abort=lambda:None)
        self.adapter = object()
        self.observer = SimpleNamespace(ops=self.adapter, finish=summary)
        self.batch = core.CaptureArtifactBatch(self.cases, self.observer, self.adapter, self.store, guard=lambda:None)
        self.captures = []
        def mocked(case,planned,observer,adapter,store,guard,tokens,positions):
            ordinal = len(self.captures)
            self.assertEqual(case, self.cases[ordinal])
            self.assertEqual(planned, {s:stage_a_requirements(case)[s]['layout'] for s in core.SCOPES})
            self.captures.append((tokens,positions))
            return metadata_result(case, ordinal)
        self.patcher = patch.object(core, '_write_capture', mocked)
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        self.reads = []

    def manifest(self):
        for i in range(6): self.batch.collect('tokens-'+str(i),'positions-'+str(i))
        return self.batch.finish()

    def verifier(self, root,name,file_id,archive,layout,**kwargs):
        self.reads.append((name,kwargs))
        self.assertEqual(kwargs['max_bytes'], core.extent(layout))
        return dict(complete=True,file=name,file_identity=file_id,bytes_verified=archive['bytes'],
            whole_file_from_offset_zero=True,stable_producer_identity_verified=True,
            byte_verification=dict(complete=True),**core.LIMITS)

    def review(self, manifest):
        with patch.object(core, 'verify_file', self.verifier):
            return core.review_saved_batch('/admitted',self.cases,manifest,guard=lambda:None)

    def reject_before_read(self, mutate):
        manifest = self.manifest(); mutate(manifest)
        with self.assertRaises((RuntimeError,KeyError,TypeError,ValueError)): self.review(manifest)
        self.assertEqual(self.reads, [])

    def test_exact_six_case_18_file_budget_and_false_claims(self):
        manifest = self.manifest()
        self.assertEqual(self.batch.committed_bytes,1467113472)
        self.assertEqual(manifest['scope_bytes'], dict(target=221736960, selected=80281600, original=1165094912))
        self.assertTrue(all(manifest[k] is False for k in core.LIMITS))
        self.assertEqual([x[0] for x in self.captures], ['tokens-'+str(i) for i in range(6)])
        self.assertEqual(core.ALL_THREE_STAGE_BYTES,2046031872)
        self.assertEqual(core.PLANNED_RESERVATION_BYTES,3*1024**3)

    def test_complete_independent_all18_routing(self):
        result = self.review(self.manifest())
        self.assertEqual(len(self.reads),18)
        self.assertEqual(result['bytes_verified'],1467113472)
        self.assertFalse(result['source_weight_and_actual_exit_provenance_verified_here'])
        for i,(_,kwargs) in enumerate(self.reads):
            scope = core.SCOPES[i%3]
            self.assertEqual(kwargs['selected_row_checks'] is not None,scope == 'target')
            self.assertEqual(kwargs['expected_tensors'] is not None,scope != 'target')
            if scope == 'target': self.assertEqual(len(kwargs['selected_row_checks']),6)

    def test_unfinished_cannot_finish(self):
        self.batch.collect(None,None)
        with self.assertRaises(RuntimeError): self.batch.finish()

    def test_complete_cannot_collect_or_finish_again(self):
        self.manifest()
        with self.assertRaises(RuntimeError): self.batch.collect(None,None)
        with self.assertRaises(RuntimeError): self.batch.finish()

    def test_collector_failure_poisoned_cannot_continue(self):
        with patch.object(core,'_write_capture',side_effect=ValueError('capture failure')):
            with self.assertRaisesRegex(ValueError,'capture failure'): self.batch.collect(None,None)
        self.assertTrue(self.batch.poisoned)
        self.assertEqual(self.batch.receipts,[])
        with self.assertRaises(RuntimeError): self.batch.collect(None,None)
        with self.assertRaises(RuntimeError): self.batch.finish()

    def test_close_failure_blocks_final_manifest(self):
        for i in range(6): self.batch.collect(None,None)
        def fail(): raise OSError('directory close')
        self.store.close = fail
        with self.assertRaisesRegex(OSError,'directory close'): self.batch.finish()
        self.assertTrue(self.batch.poisoned)
        self.assertFalse(self.batch.finished)

    def test_observer_incomplete_finish_blocks_manifest(self):
        for i in range(6): self.batch.collect(None,None)
        self.observer.finish = lambda:dict(stage_a_collector_complete=True)
        with self.assertRaises(RuntimeError): self.batch.finish()
        self.assertTrue(self.batch.poisoned)

    def test_adapter_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            core.CaptureArtifactBatch(self.cases,self.observer,object(),self.store,guard=lambda:None)

    def test_reordered_cases_rejected_before_capture(self):
        with self.assertRaises(RuntimeError):
            core.CaptureArtifactBatch(list(reversed(self.cases)),self.observer,self.adapter,self.store,guard=lambda:None)
        self.assertEqual(self.captures,[])

    def test_shrunk_original_control_rejected_before_capture(self):
        cases = deepcopy(self.cases)
        cases[0]['original_archive']['tensors']['hidden']['shape'] = [1,2560]
        with self.assertRaises(RuntimeError):
            core.CaptureArtifactBatch(cases,self.observer,self.adapter,self.store,guard=lambda:None)

    def test_last_case_wrong_original_reference_rejected_before_read(self):
        self.reject_before_read(lambda m:m['cases'][-1]['original_event'].update(sha256='0'*64))

    def test_case_digest_mismatch_rejected_before_read(self):
        self.reject_before_read(lambda m:m['cases'][0].update(case_sha256='0'*64))

    def test_cross_case_owner_swap_rejected_before_read(self):
        def swap(m): m['cases'][0],m['cases'][1] = m['cases'][1],m['cases'][0]
        self.reject_before_read(swap)

    def test_omitted_last_file_rejected_before_read(self):
        self.reject_before_read(lambda m:m['cases'][-1]['files'].pop('original'))

    def test_wrong_last_file_size_rejected_before_read(self):
        self.reject_before_read(lambda m:m['cases'][-1]['files']['original']['producer_identity'].update(st_size=1))

    def test_duplicate_inode_across_cases_rejected_before_read(self):
        def dup(m):
            a=m['cases'][0]['files']['target']['producer_identity']; b=m['cases'][-1]['files']['target']['producer_identity']
            b['st_dev'],b['st_ino'] = a['st_dev'],a['st_ino']
        self.reject_before_read(dup)

    def test_path_escape_rejected_before_read(self):
        self.reject_before_read(lambda m:m['cases'][-1]['files']['original'].update(name='../bad'))

    def test_changed_original_hash_with_matching_observer_rejected_before_read(self):
        def corrupt(m):
            r=m['cases'][-1]
            r['observation']['records']['original']['hidden']['sha256']='0'*64
            r['files']['original']['archive']['tensors']['hidden']['sha256']='0'*64
        self.reject_before_read(corrupt)

    def test_wrong_dense_offset_rejected_before_read(self):
        def corrupt(m):
            r=m['cases'][-1]; label=next(iter(r['observation']['records']['target']))
            r['observation']['records']['target'][label]['offset']=2
            r['files']['target']['archive']['tensors'][label]['offset']=2
        self.reject_before_read(corrupt)

    def test_source_selected_shape_must_be_full_input(self):
        def corrupt(m):
            r=m['cases'][0]; label=next(iter(r['observation']['source_layouts']['selected']))
            r['observation']['source_layouts']['selected'][label]['shape']=[33,2560]
        self.reject_before_read(corrupt)

    def test_promoted_numeric_qualification_rejected_before_read(self):
        self.reject_before_read(lambda m:m.update(original_FP32_acceptance_passed=True))

    def test_promoted_observer_saved_review_rejected_before_read(self):
        self.reject_before_read(lambda m:m['cases'][0]['observation'].update(independent_saved_byte_review_complete=True))

    def test_budget_relaxation_rejected_before_read(self):
        self.reject_before_read(lambda m:m.update(bytes=3*1024**3))

    def test_last_file_verification_failure_returns_no_completion(self):
        manifest=self.manifest(); original=self.verifier
        def fail(*args,**kwargs):
            if len(self.reads)==17: raise ValueError('last archive failure')
            return original(*args,**kwargs)
        with patch.object(core,'verify_file',fail):
            with self.assertRaisesRegex(ValueError,'last archive failure'):
                core.review_saved_batch('/admitted',self.cases,manifest,guard=lambda:None)
        self.assertEqual(len(self.reads),17)

    def test_incomplete_verifier_result_rejected(self):
        manifest=self.manifest(); original=self.verifier
        def incomplete(*args,**kwargs):
            r=original(*args,**kwargs);r['whole_file_from_offset_zero']=False;return r
        with patch.object(core,'verify_file',incomplete):
            with self.assertRaises(RuntimeError):core.review_saved_batch('/admitted',self.cases,manifest,guard=lambda:None)

    def test_no_native_modules_imported(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('numpy',sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
