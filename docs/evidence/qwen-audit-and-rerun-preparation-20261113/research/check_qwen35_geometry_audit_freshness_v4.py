"""Source/mock freshness tests: no real History, locks, directory discovery or native work."""
import ast
import contextlib
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'ctypes','_ctypes','torch','torch_xla','torch_neuronx','neuronxcc',
                'numpy','ml_dtypes','transformers','vllm','libneuronxla','nki'}:
            raise AssertionError('Native import forbidden: '+name)
sys.meta_path.insert(0,RejectNative())
Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
from qwen35_writer_geometry_serial_v1 import test_source as inherited,protocol,contract as c
from qwen35_writer_geometry_serial_v3 import run as s,audit as prior_audit
import qwen35_writer_geometry_audit_v4 as audit
from qwen35_package_preflight_history_v3 import History
inherited.s=s;inherited.audit=audit;inherited.History=History
fixture_spec=inherited.fixture_spec;fixture_report=inherited.fixture_report


def observation():return dict(path='/mock/running.json',identity=dict(inode=7,device=1,mtime_ns=2))


class FreshnessTests(unittest.TestCase):
    def test_first_at_zero_every_lock_and_exact_half_second_boundary(self):
        lock=SimpleNamespace(check=Mock());guard=audit.FreshnessGuard(lock,observation())
        with patch.object(audit.time,'monotonic',side_effect=[0.,.1,.499999,.5,.75,1.]),patch.object(audit,'latest_handoff',return_value=observation()) as latest:
            for _ in range(6):guard()
        self.assertEqual(lock.check.call_count,6);self.assertEqual(latest.call_count,3)

    def test_force_inside_interval_does_full_scan(self):
        lock=SimpleNamespace(check=Mock());guard=audit.FreshnessGuard(lock,observation())
        with patch.object(audit.time,'monotonic',return_value=0.),patch.object(audit,'latest_handoff',return_value=observation()) as latest:
            guard();guard();guard(force=True);guard()
        self.assertEqual(lock.check.call_count,4);self.assertEqual(latest.call_count,2)

    def test_changed_path_and_same_path_identity_rejected(self):
        for changed in [dict(observation(),path='/mock/new.json'),dict(observation(),identity=dict(inode=8))]:
            lock=SimpleNamespace(check=Mock());guard=audit.FreshnessGuard(lock,observation())
            with self.subTest(changed=changed),patch.object(audit.time,'monotonic',side_effect=[0.,.1,.5]),                    patch.object(audit,'latest_handoff',side_effect=[observation(),changed]) as latest:
                guard();guard()
                with self.assertRaisesRegex(ValueError,'Running binding changed'):guard()
            self.assertEqual(latest.call_count,2);self.assertEqual(lock.check.call_count,3)
            self.assertEqual(guard.last_full_started,0.)

    def test_force_rejects_change_before_half_second(self):
        guard=audit.FreshnessGuard(SimpleNamespace(check=Mock()),observation())
        with patch.object(audit.time,'monotonic',return_value=0.),patch.object(audit,'latest_handoff',side_effect=[observation(),dict(observation(),path='/new')]):
            guard()
            with self.assertRaises(ValueError):guard(force=True)

    def test_lock_failure_cannot_hide_in_interval(self):
        lock=SimpleNamespace(check=Mock(side_effect=[None,ValueError('lost lock')]))
        guard=audit.FreshnessGuard(lock,observation())
        with patch.object(audit.time,'monotonic',return_value=0.),patch.object(audit,'latest_handoff',return_value=observation()) as latest:
            guard()
            with self.assertRaisesRegex(ValueError,'lost lock'):guard()
        self.assertEqual(latest.call_count,1)

    def test_slow_scan_uses_start_time_and_forces_next_due_call(self):
        now=[0.];guard=audit.FreshnessGuard(SimpleNamespace(check=Mock()),observation())
        def latest():now[0]+=1.;return observation()
        with patch.object(audit.time,'monotonic',side_effect=lambda:now[0]),patch.object(audit,'latest_handoff',side_effect=latest) as scan:
            guard();guard()
        self.assertEqual(scan.call_count,2);self.assertEqual(guard.last_full_started,1.)

    def test_backward_or_nonfinite_clock_fails_closed(self):
        for bad in [-1.,float('nan'),float('inf')]:
            guard=audit.FreshnessGuard(SimpleNamespace(check=Mock()),observation())
            with self.subTest(bad=bad),patch.object(audit.time,'monotonic',side_effect=[0.,bad]),                    patch.object(audit,'latest_handoff',return_value=observation()):
                guard()
                with self.assertRaisesRegex(ValueError,'Monotonic'):guard()

    def test_caller_cannot_rebind_saved_expected_identity(self):
        expected=observation();guard=audit.FreshnessGuard(SimpleNamespace(check=Mock()),expected)
        expected['identity']['inode']=999
        with patch.object(audit.time,'monotonic',return_value=0.),patch.object(audit,'latest_handoff',return_value=expected):
            with self.assertRaises(ValueError):guard()

    def test_missing_original_ack_observation_fails_closed(self):
        for bad in [{},dict(path='/mock'),dict(path='/mock',identity=None),dict(observation(),extra=True)]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):audit.FreshnessGuard(SimpleNamespace(check=Mock()),bad)

    def test_48627_guard_calls_preserve_all_locks_with_bounded_scans(self):
        lock=SimpleNamespace(check=Mock());guard=audit.FreshnessGuard(lock,observation());now=[0.]
        with patch.object(audit.time,'monotonic',side_effect=lambda:now[0]),patch.object(audit,'latest_handoff',return_value=observation()) as latest:
            for i in range(48627):
                now[0]=i/10000;guard()
        self.assertEqual(lock.check.call_count,48627);self.assertEqual(latest.call_count,10)


class PreservedAuditTests(inherited.IndependentAuditTests):
    def exercise_audit(self,root,exit_code=0):
        spec,spec_ref,review=fixture_spec();spec['output_root']=str(root/'work')
        spec['source_sha256']=protocol.verify_sources()
        running_ref=dict(path='/mock-running.json',sha256='a'*64)
        admission_ref=dict(path='/mock-admission.json',sha256='b'*64)
        outer=dict(pid=111,start_ticks='11'); owner=dict(process=dict(pid=222,start_ticks='22'))
        tool=dict(kind='tool_session',session_id=77,exit_code=exit_code,terminal_chunk='aabbcc')
        terminal=dict(schema='qwen35_writer_geometry_outer_terminal_v1',self_exit_claimed=False,
            actual_external_tool_exit_pending=True,whole_owned_scope_drained=True,admission=admission_ref,
            outer_owner=outer,failed=exit_code!=0,result_present=exit_code==0,storage={},storage_error=None,
            failure=None if exit_code==0 else {'type':'RuntimeError','message':'actual leaf failed'})
        running=dict(qwen35_sampler_current_execution=dict(admission=admission_ref,outer_owner=outer,
                     tool=dict(kind='tool_session',session_id=77)))
        observation=dict(path=running_ref['path'],identity={'inode':7})
        saved=dict(running_observation=observation,root=Path(spec['output_root']),admission_ref=admission_ref,admission=dict(outer_owner=outer),
            terminal=terminal,terminal_ref=dict(path='/mock-terminal.json',sha256='c'*64),running=running,
            running_ref=running_ref,owner=owner,ends={'mock-unit':{}})
        reports={name:fixture_report(name)[0] for name in c.LAYOUTS}
        report_refs={name:dict(path='/mock-reports/'+name+'.json',sha256='d'*64) for name in c.LAYOUTS}
        rows=[dict(layout=name,summary=dict(layout=name),compiled_graphs=4,completed_calls=5000) for name in c.LAYOUTS]
        retained=dict(owners=[],layouts=rows,reports=report_refs)
        lock=SimpleNamespace(acquire=Mock(),check=Mock(),release=Mock())
        history=SimpleNamespace(resolve=lambda:history,physical_pair=lambda authority:{'all_original_history':True},unchanged=Mock())
        api=SimpleNamespace(proc=lambda pid:dict(pid=999,start_ticks='99'),active_heavy=Mock(),gone=Mock(),
            terminal=lambda props:True,empty_group=lambda path:{'recursive_empty':True},GROUP_PREFIX='/mock/')
        real_ref=audit.ref
        def fake_ref(path,*args):
            if Path(path).name=='supervisor-result.json':return dict(path=str(path),sha256='e'*64)
            return real_ref(path,*args)
        def fake_read(ref,*args):
            if Path(ref['path']).name=='supervisor-result.json':return {'layouts':[row['summary'] for row in rows]}
            return reports[next(name for name,row in report_refs.items() if row==ref)]
        with contextlib.ExitStack() as stack:
            patches=[patch.object(audit,'Q',root),patch.object(s,'load_spec',return_value=spec),
                patch.object(audit,'saved_binding',return_value=saved),patch.object(audit,'RetainedLock',return_value=lock),
                patch.object(audit,'latest_handoff',return_value=observation),
                patch.object(audit,'History',return_value=history),patch.object(audit,'observe_layouts',return_value=retained),
                patch.object(audit,'process_api',return_value=api),patch.object(s,'properties',return_value={'terminal':True}),
                patch.object(audit,'verify_current_terminal'),patch.object(s,'audit_results'),
                patch.object(s,'terminal_storage',return_value=({},None,None)),patch.object(s,'loaded_sources'),
                patch.object(audit,'read',side_effect=fake_read),patch.object(audit,'ref',side_effect=fake_ref)]
            for context in patches:stack.enter_context(context)
            result=audit.audit(spec_ref,review['spec'],tool)
        lock.acquire.assert_called_once();lock.release.assert_called_once_with(True)
        actual=json.loads(Path(result['actual_exit']['path']).read_text())
        self.assertTrue(actual['physically_drained']);self.assertFalse(actual['full_model_qualified'])
        self.assertEqual(actual['actual_tool'],tool)
        return result,actual,report_refs,reports


class AuditBoundaryTests(unittest.TestCase):
    def exercise(self,*,passed=True,change_after=None):
        events=[];phase=['initial'];published=[];captured={};expected=observation()
        def event(name):phase[0]=name;events.append(name)
        lock=SimpleNamespace(acquire=lambda:event('acquire'),check=lambda:events.append('lock'),
                             release=lambda drained:event('release'))
        spec=dict(run_name='mock',output_root='/mock/work',source_sha256={'/mock/source.py':'a'*64})
        saved=dict(root=Path('/mock/work'),admission=dict(outer_owner={'pid':1}),terminal=dict(failed=not passed,
            failure=None if passed else {'type':'Failure'},storage_error=None,storage={}),owner=dict(process={'pid':2}),
            running_ref=dict(path=expected['path'],sha256='b'*64),running={},running_observation=expected,
            ends={'mock-unit':{}},admission_ref={},terminal_ref={})
        rows=[dict(layout=name,summary=dict(layout=name),compiled_graphs=4,completed_calls=5000) for name in s.LAYOUTS]
        retained=dict(owners=[{'pid':3}],layouts=rows,reports={name:{'path':'/mock/'+name+'.json','sha256':'c'*64} for name in s.LAYOUTS})
        def latest():
            events.append('latest:'+phase[0])
            return dict(expected,path='/changed') if phase[0]==change_after else copy.deepcopy(expected)
        def construct_history(*args):
            captured['guard']=args[-1];event('history_init');return history
        def resolve():event('resolve');captured['guard']();return history
        def pair(authority):
            event('full_history');self.assertIs(authority.guard,captured['guard']);captured['guard']()
            return {'all_original_874_plus_later_scope':True,'full_physical_passes':2}
        history=SimpleNamespace(resolve=resolve,physical_pair=pair,unchanged=lambda:event('unchanged'))
        def layouts(*args):event('reports');self.assertIs(args[-1],captured['guard']);return retained
        api=SimpleNamespace(proc=lambda pid:{'pid':999},active_heavy=lambda:event('active_heavy'),
            gone=lambda owner:event('gone:'+str(owner['pid'])),terminal=lambda props:True,
            empty_group=lambda path:(event('empty_group') or {'recursive_empty':True}),GROUP_PREFIX='/mock/')
        def publish(path,value):
            name='publish_exit' if path.name.endswith('-root-actual-exit.json') else 'publish_qualification'
            event(name);published.append(name);return audit.predicted(path,value)
        def read_mock(ref,*args):
            if ref['path'].endswith('supervisor-result.json'):return {'layouts':[r['summary'] for r in rows]}
            return {'complete':True}
        with contextlib.ExitStack() as stack:
            patches=[patch.object(audit,'Q',Path('/mock')),patch.object(Path,'exists',return_value=False),
                patch.object(Path,'is_symlink',return_value=False),patch.object(s,'load_spec',return_value=spec),
                patch.object(audit,'saved_binding',return_value=saved),patch.object(audit,'RetainedLock',return_value=lock),
                patch.object(audit,'History',side_effect=construct_history),patch.object(audit,'observe_layouts',side_effect=layouts),
                patch.object(audit,'process_api',return_value=api),patch.object(audit,'latest_handoff',side_effect=latest),
                patch.object(audit.time,'monotonic',return_value=0.),patch.object(s,'properties',return_value={'terminal':True}),
                patch.object(audit,'verify_current_terminal',side_effect=lambda *args:event('verify_terminal')),
                patch.object(s,'reconcile',return_value={'passed':passed}),patch.object(s,'audit_results',side_effect=lambda *args:event('audit_results')),
                patch.object(audit.geometry,'check_complete_geometry',side_effect=lambda *args:event('complete_geometry')),
                patch.object(s,'terminal_storage',side_effect=lambda *args:(event('storage') or ({},None,None))),
                patch.object(s,'loaded_sources',side_effect=lambda *args:event('sources')),
                patch.object(protocol,'verify_sources',side_effect=lambda:(event('protocol_sources') or {'/mock/geometry.py':'d'*64})),
                patch.object(audit,'ref',side_effect=lambda path,*args:dict(path=str(path),sha256='e'*64)),
                patch.object(audit,'read',side_effect=read_mock),patch.object(audit,'once',side_effect=publish)]
            for p in patches:stack.enter_context(p)
            if change_after:
                with self.assertRaisesRegex(ValueError,'Running binding changed'):audit.audit({}, {}, {'exit_code':0 if passed else 1})
                result=None
            else:result=audit.audit({}, {}, {'exit_code':0 if passed else 1})
        return result,events,published

    def test_complete_checks_and_force_boundaries_at_same_clock(self):
        result,events,published=self.exercise()
        self.assertTrue(result['passed']);self.assertEqual(published,['publish_exit','publish_qualification'])
        for name in ['resolve','reports','full_history','unchanged','sources','audit_results','complete_geometry']:
            self.assertEqual(events.count(name),1)
        self.assertEqual(events.count('active_heavy'),2);self.assertEqual(events.count('empty_group'),2)
        self.assertEqual(events.count('verify_terminal'),2)
        self.assertEqual(sum(x.startswith('gone:') for x in events),6)
        for name in ['history_init','active_heavy','full_history','publish_exit','publish_qualification']:
            for i,event in enumerate(events):
                if event==name:self.assertTrue(events[i-1].startswith('latest:'),(name,events))
        for name in ['empty_group','full_history','publish_exit','publish_qualification']:
            for i,event in enumerate(events):
                if event==name:
                    j=i+1
                    while events[j]=='lock':j+=1
                    self.assertTrue(events[j].startswith('latest:'),(name,events))
        self.assertEqual(events[-1],'release')

    def test_unknown_successor_before_publication_prevents_all_writes(self):
        _,events,published=self.exercise(change_after='protocol_sources')
        self.assertEqual(published,[]);self.assertEqual(events[-1],'release')

    def test_changed_handoff_after_exit_publication_blocks_qualification(self):
        _,events,published=self.exercise(change_after='publish_exit')
        self.assertEqual(published,['publish_exit']);self.assertEqual(events[-1],'release')

    def test_changed_handoff_after_physical_pass_fails_before_next_pass(self):
        _,events,published=self.exercise(change_after='empty_group')
        self.assertEqual(events.count('active_heavy'),1);self.assertEqual(published,[])

    def test_failed_actual_exit_stays_failed_with_complete_history(self):
        result,events,published=self.exercise(passed=False)
        self.assertFalse(result['passed']);self.assertIsNone(result['qualification'])
        self.assertEqual(events.count('full_history'),1);self.assertEqual(events.count('active_heavy'),2)
        self.assertEqual(published,['publish_exit']);self.assertNotIn('complete_geometry',events)


class PreservationTests(unittest.TestCase):
    def test_audit_body_keeps_every_original_non_guard_statement(self):
        def function(path,name):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        class NormalizeGuard(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.name=='guard':return None
                return self.generic_visit(node)
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='guard':return None
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='guard':return None
                return self.generic_visit(node)
        old=NormalizeGuard().visit(function(prior_audit.__file__,'audit'))
        new=NormalizeGuard().visit(function(audit.__file__,'audit'))
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
        for name in ['latest_handoff','observe_layouts','predicted','main']:
            self.assertEqual(ast.dump(function(prior_audit.__file__,name),include_attributes=False),
                             ast.dump(function(audit.__file__,name),include_attributes=False))
        before=function(prior_audit.__file__,'saved_binding');after=function(audit.__file__,'saved_binding')
        tail=after.body[-1].value
        self.assertEqual(tail.keywords[-1].arg,'running_observation');tail.keywords.pop()
        self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
        self.assertIs(audit.History,prior_audit.History);self.assertIs(audit.Authority,prior_audit.Authority)
        self.assertIs(audit.stage,prior_audit.stage);self.assertIs(audit.RetainedLock,prior_audit.RetainedLock)


if __name__=='__main__':
    suite=unittest.TestSuite()
    for cls in (FreshnessTests,PreservedAuditTests,AuditBoundaryTests,PreservationTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    for name,value in vars(inherited).items():
        if isinstance(value,type) and issubclass(value,unittest.TestCase) and value.__module__==inherited.__name__ and name!='IndependentAuditTests':
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(value))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
