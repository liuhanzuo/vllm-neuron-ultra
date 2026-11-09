"""Synthetic orchestration only; all resource/lock/model operations are mocked."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_dory_saved_review_retry_v1 as e


class Executor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dory-review-retry-executor-')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)/'out'
        self.trace = []
        self.owner = dict(pid=11,start_ticks='22')
        self.canonical = dict(original='same')
        self.spec_ref = dict(path='/synthetic/spec.json',sha256='1'*64)
        self.review_ref = dict(path='/synthetic/review.json',sha256='2'*64)
        self.query_ref = dict(path='/synthetic/query-observation.json',sha256='3'*64)
        self.failed_ref = dict(path='/synthetic/failed-exit.json',sha256='4'*64)
        self.pins = {'/synthetic/old.py':'5'*64,'/synthetic/dependency.py':'6'*64}
        self.spec = dict(output=str(self.out),executor=str(e.contract.EXECUTOR),
            source_sha256=self.pins,essential_sources=list(self.pins),source_read_cap_bytes=100,
            failed_review_exit=self.failed_ref,query_dependency_observation=self.query_ref,
            producer_spec=dict(path='/synthetic/native.json',sha256='7'*64),
            native_exit=dict(path='/synthetic/native-exit.json',sha256='8'*64),
            budget=dict(root_floor=20,nvme_floor=20,nvme_new_bytes=10,wall_seconds=300,
                cpu_seconds=200,address_space_bytes=1024,single_file_bytes=1024,
                cpus=[4],initial_memory_available_bytes=20,memory_available_floor_bytes=10))
        self.review = dict(reviewer='/root',passed=True,blockers=[],spec_sha256='1'*64,source_sha256=self.pins)
        self.h = dict(canonical_sha256=self.canonical,predecessor_source_sha256={'/synthetic/old.py':'5'*64})
        self.reader_pins = dict(self.pins)
        self.low = SimpleNamespace(user_bus=Mock(),lock_proof=Mock(),clear_window=Mock(),active_heavy=Mock(return_value=[]))
        self.protocol = SimpleNamespace(low=self.low,canonical=Mock(return_value=self.canonical))
        self.query = None
        self.query_error = self.close_error = self.attach_error = False
        self.saved_error = self.freeze_error = False
        self.drain = dict(complete=True,services=1149,processes=348)

    def query_factory(self, observation, *, before_check):
        for _ in range(41):
            before_check()
        q = SimpleNamespace(startup_file_count=41,startup_bytes=4326816,
            requests=0,closed=False,poisoned=False,restored=False,final_audit={'mocked':True})
        def audit():
            if self.attach_error:
                raise RuntimeError('synthetic attach audit failure')
        def check():
            q.requests += 1
            if self.query_error:
                q.poisoned = True
                raise RuntimeError('synthetic query failure')
            return []
        def close():
            if not q.closed:
                q.closed = True
                q.restored = True
                self.trace.append('close')
                if self.close_error:
                    q.poisoned = True
                    raise RuntimeError('synthetic final audit failure')
        q.audit,q.check,q.close = audit,check,close
        self.query = q
        return q

    def run_fixture(self):
        test = self
        class Reader:
            def __init__(self):
                self.pins = dict(test.reader_pins)
            def __call__(self, ref):
                self.pins[ref['path']] = ref['sha256']
                if ref == test.spec_ref:
                    return copy.deepcopy(test.spec)
                if ref == test.review_ref:
                    return copy.deepcopy(test.review)
                raise AssertionError(ref)
        def freeze(spec, native, guard):
            self.assertEqual((spec,native),(self.spec['producer_spec'],self.spec['native_exit']))
            guard()
            self.trace.append('freeze')
            if self.freeze_error:
                raise RuntimeError('synthetic freeze failure')
            return dict(original_frozen=True)
        def review(frozen, guard):
            self.assertEqual(frozen,dict(original_frozen=True))
            guard()
            self.trace.append('full_saved_review')
            if self.saved_error:
                raise RuntimeError('synthetic payload failure')
            return dict(complete=True,original_pointwise_failures=7,full_model_qualified=False,new_speedup_claim=False)
        def release(fd):
            self.assertEqual(fd,123)
            self.trace.append('release')
        patches = [
            patch.object(e.contract.original,'Reader',Reader),
            patch.object(e.contract,'validate_spec',return_value={}),
            patch.object(e.contract,'failed_history',return_value=dict(history=copy.deepcopy(self.h))),
            patch.object(e,'acquire',return_value=(123,self.owner,'synthetic-lock')),
            patch.object(e,'release',side_effect=release),patch.object(e,'p',self.protocol),
            patch.object(e,'free',return_value=100),patch.object(e,'memory_free',return_value=100),
            patch.object(e.os,'sched_setaffinity'),patch.object(e.os,'nice'),
            patch.object(e.resource,'setrlimit'),patch.object(e.signal,'signal',return_value=None),
            patch.object(e,'review_source_pins',return_value=dict(source_and_binary_files_authenticated=True)),
            patch.object(e,'physical_drain',return_value=self.drain),
            patch.object(e,'read_observation',return_value=dict(source_and_binary_pins={'/synthetic/dependency.py':dict(sha256='6'*64)})),
            patch.object(e,'AuthenticatedServiceQuery',side_effect=self.query_factory),
            patch.object(e,'freeze',side_effect=freeze),patch.object(e,'review_complete',side_effect=review),
            patch.object(e.sys,'argv',['synthetic','--spec',self.spec_ref['path'],'--spec-sha256',self.spec_ref['sha256'],
                '--review',self.review_ref['path'],'--review-sha256',self.review_ref['sha256']])]
        with ExitStack() as stack,redirect_stdout(io.StringIO()):
            for item in patches:
                stack.enter_context(item)
            e.main()

    def test_success_keeps_original_raw_reviewer_and_closes_before_unlock(self):
        self.run_fixture()
        self.assertEqual(self.trace,['freeze','full_saved_review','close','release'])
        result=json.loads((self.out/'result.json').read_text())
        self.assertTrue(result['complete'])
        self.assertFalse(result['full_model_qualified'] or result['new_speedup_claim'])
        saved=json.loads((self.out/'saved-residual-review.json').read_text())
        self.assertEqual(saved['original_pointwise_failures'],7)
        accounting=json.loads((self.out/'service-query-authentication.json').read_text())['guard_accounting']
        self.assertEqual(accounting['persistent_service_checks'],self.query.requests)
        self.assertTrue(accounting['closed'])

    def test_unpinned_history_read_prevents_query_and_raw_review(self):
        self.reader_pins['/synthetic/unreviewed.json']='0'*64
        with self.assertRaisesRegex(RuntimeError,'unreviewed historical'):
            self.run_fixture()
        self.assertEqual(self.trace,['release'])
        self.assertFalse(self.out.exists())

    def test_incomplete_drain_prevents_query_and_raw_review(self):
        self.drain['processes']=347
        with self.assertRaisesRegex(RuntimeError,'physically drained'):
            self.run_fixture()
        self.assertEqual(self.trace,['release'])

    def test_query_failure_preserves_failure_and_prevents_payload_reads(self):
        self.query_error=True
        with self.assertRaisesRegex(RuntimeError,'synthetic query failure'):
            self.run_fixture()
        self.assertEqual(self.trace,['close','release'])
        self.assertTrue((self.out/'failure.json').exists())
        self.assertFalse((self.out/'result.json').exists())

    def test_freeze_failure_closes_and_releases(self):
        self.freeze_error=True
        with self.assertRaisesRegex(RuntimeError,'synthetic freeze failure'):
            self.run_fixture()
        self.assertEqual(self.trace,['freeze','close','release'])
        self.assertFalse((self.out/'result.json').exists())

    def test_payload_failure_closes_and_releases(self):
        self.saved_error=True
        with self.assertRaisesRegex(RuntimeError,'synthetic payload failure'):
            self.run_fixture()
        self.assertEqual(self.trace,['freeze','full_saved_review','close','release'])
        self.assertFalse((self.out/'result.json').exists())

    def test_final_dependency_audit_failure_prevents_success(self):
        self.close_error=True
        with self.assertRaisesRegex(RuntimeError,'synthetic final audit failure'):
            self.run_fixture()
        self.assertEqual(self.trace,['freeze','full_saved_review','close','release'])
        self.assertTrue((self.out/'saved-residual-review.json').exists())
        self.assertTrue((self.out/'failure.json').exists())
        self.assertFalse((self.out/'result.json').exists())

    def test_unattached_constructed_query_is_closed(self):
        self.attach_error=True
        with self.assertRaisesRegex(RuntimeError,'synthetic attach audit failure'):
            self.run_fixture()
        self.assertEqual(self.trace,['close','release'])
        self.assertTrue(self.query.closed)

    def test_external_spec_and_review_pins_do_not_create_self_hash_cycle(self):
        self.assertNotIn(self.spec_ref['path'],self.pins)
        self.assertNotIn(self.review_ref['path'],self.pins)
        self.run_fixture()
        self.assertTrue((self.out/'result.json').exists())

    def test_existing_namespace_is_not_reused(self):
        self.out.mkdir()
        with self.assertRaisesRegex(RuntimeError,'Fresh retry namespace'):
            self.run_fixture()
        self.assertEqual(self.trace,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
