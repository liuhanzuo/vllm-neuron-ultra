"""Mocked full executor lifecycle with generated data; no serial/native job."""
from contextlib import ExitStack,redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_minicpm5_saved_tap_serial_v1 as r
import nemotron9b_package_protocol_v23 as p
from check_minicpm5_saved_tap_runtime_contract_v1 import generated_report,spec_fixture


class Guard:
    def __init__(self,events,**kwargs):self.events=events;self.query=None;self.closed=False
    def __call__(self):
        if self.closed:raise ValueError('guard closed')
        self.events.append('guard')
    def attach_authenticated_query(self,query):self.query=query;self.events.append('query-attached')
    def close(self):
        if not self.closed:
            self.closed=True;self.events.append('guard-closed')
            if self.query:self.query.close()
    def accounting(self):return dict(closed=self.closed)


class Query:
    def __init__(self,events,observation,*,before_check,fail_close=False):
        before_check();self.events=events;events.append('query-open')
        self.closed=self.poisoned=False;self.restored=False;self.startup_file_count=41
        self.startup_bytes=4326816;self.final_audit={'synthetic':True};self.fail_close=fail_close
    def close(self):
        self.events.append('query-close')
        self.closed=self.restored=True
        if self.fail_close:self.poisoned=True;raise ValueError('query cleanup failed')


class Execution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=generated_report()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-execution-fixture-');self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'out';self.events=[];self.query=None

    def invoke(self,failure=None):
        spec=spec_fixture();spec['source_sha256']={}
        spec_ref=dict(path='/synthetic/spec.json',sha256='a'*64)
        review_ref=dict(path='/synthetic/review.json',sha256='b'*64)
        owner=dict(pid=12345,start_ticks='123',argv=['/synthetic/runner'],ppid=1,cgroup='0::/fixture')
        canonical={'fixture':'c'*64}
        def step(name,value):
            def call(*args,**kwargs):
                self.events.append(name)
                if failure==name:raise ValueError('synthetic failure: '+name)
                return copy.deepcopy(value)
            return call
        def query(observation,**kwargs):
            self.query=Query(self.events,observation,fail_close=failure=='query-cleanup',**kwargs)
            return self.query
        def analysis(*args,**kwargs):
            self.events.append('analysis')
            if failure=='analysis':raise ValueError('synthetic raw analysis failure')
            value=copy.deepcopy(self.report)
            if failure=='partial-report':value['bundle_inputs'].pop()
            return value
        self.canonical_reads=0
        def state():
            self.canonical_reads+=1
            return {'changed':'d'*64} if failure=='canonical-change' and self.canonical_reads==2 else canonical
        class Reader:
            def __init__(self,*args):self.pins={}
        patches=[patch.object(r.c,'OUT',self.out),patch.object(r,'acquire',step('acquire',(999,owner,{'inode':1}))),
            patch.object(r,'release',step('release',None)),patch.object(r.low,'user_bus',step('user-bus',None)),
            patch.object(r,'free',return_value=100*2**30),patch.object(r,'memory_free',return_value=100*2**30),
            patch.object(r.os,'sched_setaffinity',step('affinity',None)),patch.object(r.os,'setpriority',step('priority',None)),
            patch.object(r.resource,'setrlimit',step('resource-limit',None)),patch.object(r.signal,'signal',return_value=None),
            patch.object(r,'CompleteReviewGuard',lambda **kw:Guard(self.events,**kw)),
            patch.object(r,'review_source_pins',step('source-check',{'files':{},'synthetic':True})),
            patch.object(r.c.history,'completed',step('history-check',dict(history=dict(rows=[{}]*1202,
                processes=[{}]*404,canonical_sha256=canonical),source_sha256={}))),
            patch.object(r.c.history,'Reader',Reader),patch.object(r.c,'input_metadata',step('input-check',({},{}))),
            patch.object(r,'physical_drain',step('drain',dict(complete=True,services=1202,processes=404))),
            patch.object(p,'canonical',side_effect=state),patch.object(r.c,'loaded_local_sources',return_value={'synthetic':{}}),
            patch.object(r,'read_observation',return_value={'source_and_binary_pins':{}}),
            patch.object(r,'AuthenticatedServiceQuery',side_effect=query),patch.object(r.c.inputs,'analyze',side_effect=analysis)]
        error=None
        with ExitStack()as stack,redirect_stdout(io.StringIO()):
            for item in patches:stack.enter_context(item)
            try:r.execute(spec,spec_ref,review_ref)
            except BaseException as e:error=e
        return error

    def test_complete_lifecycle_orders_admission_before_all_raw_analysis(self):
        self.assertIsNone(self.invoke())
        for left,right in (('source-check','history-check'),('history-check','input-check'),('input-check','drain'),
                           ('drain','query-open'),('query-open','analysis'),('analysis','query-close'),('query-close','release')):
            self.assertLess(self.events.index(left),self.events.index(right))
        result=json.loads((self.out/'result.json').read_text())
        self.assertTrue(result['all84bundles_and384comparisons_complete'])
        self.assertFalse(result['actual_self_exit_claimed']);self.assertFalse(result['full_model_qualified'])
        self.assertEqual(self.events.count('release'),1)
        self.assertTrue(self.query.closed and self.query.restored)

    def test_source_history_input_or_drain_failure_never_reads_raw(self):
        for failure in ('source-check','history-check','input-check','drain'):
            self.events=[]
            with self.subTest(failure=failure):
                self.assertIsInstance(self.invoke(failure),ValueError)
                self.assertNotIn('analysis',self.events);self.assertFalse(self.out.exists())
                self.assertEqual(self.events.count('release'),1)

    def test_analysis_failure_kept_and_never_publishes_success(self):
        self.assertIsInstance(self.invoke('analysis'),ValueError)
        self.assertTrue((self.out/'failure.json').exists());self.assertFalse((self.out/'result.json').exists())
        self.assertFalse((self.out/'numeric-report.json').exists());self.assertTrue(self.query.closed)
        self.assertEqual(self.events.count('release'),1)

    def test_partial_report_cannot_be_published(self):
        self.assertIsInstance(self.invoke('partial-report'),ValueError)
        self.assertFalse((self.out/'numeric-report.json').exists());self.assertFalse((self.out/'result.json').exists())

    def test_late_canonical_change_retains_report_but_no_success(self):
        self.assertIsInstance(self.invoke('canonical-change'),ValueError)
        self.assertTrue((self.out/'numeric-report.json').exists())
        self.assertTrue((self.out/'failure.json').exists());self.assertFalse((self.out/'result.json').exists())

    def test_query_cleanup_failure_releases_lock_and_keeps_failure(self):
        self.assertIsInstance(self.invoke('query-cleanup'),ValueError)
        self.assertTrue((self.out/'query-cleanup-failure.json').exists())
        self.assertFalse((self.out/'result.json').exists());self.assertEqual(self.events.count('release'),1)

    def test_existing_output_is_preserved_and_rejected(self):
        self.out.mkdir();marker=self.out/'previous.txt';marker.write_text('retain')
        self.assertIsInstance(self.invoke(),ValueError)
        self.assertEqual(marker.read_text(),'retain');self.assertNotIn('analysis',self.events)
        self.assertEqual(self.events.count('release'),1)

    def test_resource_setup_failure_releases_serial_lock(self):
        self.assertIsInstance(self.invoke('resource-limit'),ValueError)
        self.assertEqual(self.events.count('release'),1);self.assertNotIn('source-check',self.events)


if __name__=='__main__':unittest.main(verbosity=2)
