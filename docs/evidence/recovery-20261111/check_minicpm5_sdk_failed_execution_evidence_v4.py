"""Synthetic failure lifecycle checks; no actual proc/service/model/payload IO."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import minicpm5_sdk_failed_execution_evidence_v4 as m

def ref(path):return dict(path=str(path),sha256='a'*64)
def process(pid,argv,cgroup='0::/session.scope',ppid=1):
    return dict(pid=pid,start_ticks=str(1000+pid),ppid=ppid,argv=argv,cgroup=cgroup)

class Fixture:
    def __init__(self,launched=True):
        self.root=Path('/mnt/nvme/minicpm5-layer1-sdk-batch-layout-test')
        self.spec_ref=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-test-spec.json')
        self.review_ref=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-test-source-review.json')
        self.live_ref=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-test-actual-live.json')
        self.model_exit_ref=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-test-actual-exit.json')
        self.lock=dict(device=10,inode=20)
        self.recorder=process(90,['python3','-B','synthetic-recorder'])
        self.outer=process(10,[m.PYTHON,'-B',str(m.RUNNER),'--mode','outer','--spec',self.spec_ref['path'],
            '--spec-sha256','a'*64,'--review',self.review_ref['path'],'--review-sha256','a'*64])
        self.spec=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2',
            runner=str(m.RUNNER),worker=str(m.WORKER),output_root=str(self.root),
            unit='trn1-mini-layer1-sdk-batch-layout-test',source_sha256={'source.py':'a'*64},
            plan=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-test-plan.json'),
            source_contract=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-source-contract-test.json'),
            latest_completion={'frozen':'synthetic'})
        self.review=dict(schema='minicpm5_layer1_sdk_batch_layout_runtime_source_review_v2',
            reviewer='/root',passed=True,blockers=[],spec=self.spec_ref,source_sha256=self.spec['source_sha256'])
        self.live=dict(schema='minicpm5_layer1_sdk_batch_layout_actual_live_v1',reviewer='/root',
            actual_identity_verified=True,actual_execution_handle={'kind':'tool_session','session_id':100},
            owner=self.outer,spec=self.spec_ref,review=self.review_ref)
        self.data={self.spec_ref['path']:self.spec,self.review_ref['path']:self.review,self.live_ref['path']:self.live}
        self.bad=set();self.versions={};self.reads=[]
        self.p=SimpleNamespace(lock_proof=Mock(),clear_window=Mock(),gone=Mock(),active_heavy=Mock(),
            terminal=lambda p:p.get('terminal') is True)
        self.runner=SimpleNamespace(p=self.p,identity=self.identity,reference=lambda p:ref(p),
            source_auth=SimpleNamespace(load=self.load),argv=self.argv,service_command=self.command,
            verify_owned_drain=Mock(side_effect=self.drain))
        self.admission_ref=ref(self.root/'admission.json')
        self.co=process(11,self.argv('coordinator',self.admission_ref),
            '0::'+m.GROUP+self.spec['unit']+'.service')
        self.co.update(unit=self.spec['unit'],invocation_id='b'*32)
        self.ranks=[process(i,self.argv('rank',self.admission_ref),self.co['cgroup'],11)for i in (12,13)]
        if launched:self.add_launch()

    def argv(self,mode,ad):
        return [m.PYTHON,'-B',str(m.RUNNER),'--mode',mode,'--admission',ad['path'],'--admission-sha256',ad['sha256']]
    def command(self,spec,ad,attempt):
        return ['synthetic-owned-launch',spec['unit'],attempt,*self.argv('coordinator',ad)]
    def add(self,name,value):self.data[str(self.root/name)]=value
    def add_launch(self):
        ad=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2',spec=self.spec_ref,review=self.review_ref,
            plan=self.spec['plan'],source_contract=self.spec['source_contract'],outer_owner=self.outer,
            lock_identity=self.lock,attempt_id='c'*32,latest_completion=self.spec['latest_completion'])
        self.add('admission.json',ad)
        self.add('launch-intent.json',dict(admission=self.admission_ref,command=self.command(self.spec,self.admission_ref,'c'*32)))
        self.add('coordinator-owner.json',self.co)
        for i,r in enumerate(self.ranks):self.add('rank%d-owner.json'%i,r)
        self.add('outer-failure.json',dict(owner=self.outer,error_type='RuntimeError',error='synthetic failure'))
        self.add('children-exits.json',dict(owners=self.ranks,exit_codes=[1,-15],successful=False))
    def identity(self,path):
        path=str(path)
        if path not in self.data:raise FileNotFoundError(path)
        return dict(realpath=path,device=1,inode=2,size=100,mtime_ns=self.versions.get(path,0),ctime_ns=0)
    def load(self,r):
        self.reads.append(r['path'])
        if r['path']in self.bad:raise json.JSONDecodeError('synthetic truncated JSON','{',1)
        return copy.deepcopy(self.data[r['path']])
    def drain(self,spec,ad,attempt,co,launched):
        if co is None and any(str(self.root/('rank%d-owner.json'%i))in self.data for i in (0,1)):
            raise RuntimeError('original missing coordinator')
        owners=([co]if co is not None else[])+[
            copy.deepcopy(self.data[str(self.root/('rank%d-owner.json'%i))])
            for i in (0,1)if str(self.root/('rank%d-owner.json'%i))in self.data]
        return dict(physically_drained=True,properties={'terminal':True,'Job':''},
            group={'group':m.GROUP+spec['unit']+'.service','absent':True},owners_checked=owners)
    def record(self,**kw):
        values=dict(runner=self.runner,phase='model',live_ref=self.live_ref,session=100,chunk='abcdef',
            exit_code=1,recorder=self.recorder,lock=self.lock)
        values.update(kw);return m.record_failure(**values)
    def auditor(self):
        self.audit_ref=ref(m.Q/'minicpm5-layer1-sdk-batch-layout-test-audit-live.json')
        self.audit_output=m.Q/'minicpm5-layer1-sdk-batch-layout-test-post-exit-audit-v1.json'
        self.auditor_owner=process(20,['python3','-B',str(m.AUDITOR),'--model-exit',self.model_exit_ref['path'],
            '--model-exit-sha256','a'*64,'--output',str(self.audit_output)])
        self.data[self.model_exit_ref['path']]=dict(schema='minicpm5_layer1_sdk_batch_layout_actual_exit_v1',
            reviewer='/root',actual_exit_observed=True,actual_exit_code=0,full_model_qualified=False,new_speedup_claim=False,
            live=self.live_ref,actual_execution_handle=self.live['actual_execution_handle'],owner=self.outer,
            spec=self.spec_ref,review=self.review_ref,lock_identity=self.lock,coordinator=self.co,ranks=self.ranks)
        self.audit_live=dict(schema='minicpm5_sdk_batch_layout_audit_live_v1',reviewer='/root',
            actual_identity_verified=True,actual_execution_handle={'kind':'tool_session','session_id':101},
            owner=self.auditor_owner,model_actual_exit=self.model_exit_ref)
        self.data[self.audit_ref['path']]=self.audit_live
        return self.record(phase='auditor',live_ref=self.audit_ref,session=101)

class Failure(unittest.TestCase):
    def test_model_failure_preserves_partial_and_missing_without_qualification(self):
        f=Fixture();result=f.record()
        self.assertTrue(result['failed']);self.assertFalse(result['successful'])
        self.assertFalse(result['full_model_qualified']);self.assertFalse(result['automatic_successor_release'])
        self.assertEqual(result['available_partial_metadata']['outer-result.json']['status'],'missing')
        self.assertEqual(result['available_partial_metadata']['outer-failure.json']['status'],'present')
        self.assertEqual(result['exact_recorded_processes_gone'],[f.outer,f.co,*f.ranks])
    def test_pre_admission_failure_requires_original_drain_with_no_launch(self):
        f=Fixture(False);out=f.record()
        self.assertEqual(out['exact_recorded_processes_gone'],[f.outer])
        self.assertEqual(f.runner.verify_owned_drain.call_args.args[1:],(None,None,None,False))
    def test_signal_exit_is_recorded_without_terminating(self):
        self.assertEqual(Fixture().record(exit_code=-9)['actual_exit_code'],-9)
    def test_invalid_tool_facts_reject_before_any_reads(self):
        for kw in ({'exit_code':0},{'exit_code':False},{'exit_code':256},{'exit_code':-128},
                   {'session':False},{'chunk':''},{'chunk':'zzzzzz'}):
            f=Fixture()
            with self.subTest(kw=kw),self.assertRaises(RuntimeError):f.record(**kw)
            self.assertEqual(f.reads,[])
    def test_wrong_session_rejects_before_bundle(self):
        f=Fixture()
        with self.assertRaises(RuntimeError):f.record(session=99)
        self.assertEqual(f.reads,[f.live_ref['path']])
    def test_live_owner_refuses_before_bundle(self):
        f=Fixture();f.p.gone.side_effect=RuntimeError('still live')
        with self.assertRaises(RuntimeError):f.record()
        self.assertEqual(f.reads,[f.live_ref['path']])
    def test_lock_guard_failure_rejects_before_read(self):
        f=Fixture();f.p.lock_proof.side_effect=RuntimeError('wrong lock')
        with self.assertRaises(RuntimeError):f.record()
        self.assertEqual(f.reads,[])
    def test_recorder_cannot_impersonate_failed_owner(self):
        f=Fixture()
        with self.assertRaises(RuntimeError):f.record(recorder=f.outer)
    def test_wrong_owner_argv_rejects(self):
        f=Fixture();f.outer['argv'][0]='python3'
        with self.assertRaises(RuntimeError):f.record()
    def test_wrong_review_binding_rejects(self):
        f=Fixture();f.review['spec']=ref(m.Q/'different.json')
        with self.assertRaises(RuntimeError):f.record()
    def test_wrong_admission_owner_or_lock_rejects(self):
        for key,value in [('outer_owner',process(30,['unrelated'])),('lock_identity',{'device':1,'inode':1})]:
            f=Fixture();f.data[f.admission_ref['path']][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):f.record()
    def test_wrong_intent_does_not_trigger_service_operations(self):
        f=Fixture();f.data[str(f.root/'launch-intent.json')]['command']=['unrelated']
        with self.assertRaises(RuntimeError):f.record()
        f.runner.verify_owned_drain.assert_not_called()
    def test_rank_without_coordinator_rejects(self):
        f=Fixture();del f.data[str(f.root/'coordinator-owner.json')]
        with self.assertRaises(RuntimeError):f.record()
    def test_rank_wrong_parent_or_group_rejects(self):
        for key,value in [('ppid',999),('cgroup','0::/other.service')]:
            f=Fixture();f.ranks[0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):f.record()
    def test_unrelated_child_receipt_rejects(self):
        f=Fixture();f.data[str(f.root/'children-exits.json')]['owners']=[process(88,['unrelated'])]
        with self.assertRaises(RuntimeError):f.record()
    def test_partial_rank_result_owner_mismatch_rejects(self):
        f=Fixture();f.add('rank0-result.json',dict(process=f.ranks[1]))
        with self.assertRaises(RuntimeError):f.record()
    def test_critical_truncation_rejects(self):
        f=Fixture();f.bad.add(f.admission_ref['path'])
        with self.assertRaises(RuntimeError):f.record()
    def test_present_null_critical_metadata_is_not_treated_as_absent(self):
        for name in ('admission.json','coordinator-owner.json','cleanup-drained.json'):
            f=Fixture(False);f.add(name,None)
            with self.subTest(name=name),self.assertRaises(RuntimeError):f.record()
    def test_noncritical_truncation_retains_digest_without_success(self):
        f=Fixture();f.add('rank0-result.json',{});f.bad.add(str(f.root/'rank0-result.json'))
        row=f.record()['available_partial_metadata']['rank0-result.json']
        self.assertEqual(row['status'],'invalid_json');self.assertEqual(row['ref'],ref(f.root/'rank0-result.json'))
    def test_false_drain_rejects(self):
        f=Fixture();f.runner.verify_owned_drain.side_effect=None
        f.runner.verify_owned_drain.return_value={'physically_drained':False}
        with self.assertRaises(RuntimeError):f.record()
    def test_pending_job_rejects(self):
        f=Fixture();old=f.drain
        def drain(*a):
            out=old(*a);out['properties']['Job']='123';return out
        f.runner.verify_owned_drain.side_effect=drain
        with self.assertRaises(RuntimeError):f.record()
    def test_missing_verifier_owner_rejects(self):
        f=Fixture();old=f.drain
        def drain(*a):
            out=old(*a);out['owners_checked']=[];return out
        f.runner.verify_owned_drain.side_effect=drain
        with self.assertRaises(RuntimeError):f.record()
    def test_unknown_service_is_not_cleaned(self):
        f=Fixture(False);f.runner.verify_owned_drain.side_effect=RuntimeError('original verifier unknown service')
        with self.assertRaises(RuntimeError):f.record()
    def test_active_heavy_refuses_output(self):
        f=Fixture();f.p.active_heavy.side_effect=RuntimeError('competing service')
        with self.assertRaises(RuntimeError):f.record()
    def test_changed_partial_identity_refuses(self):
        f=Fixture();old=f.drain
        def drain(*a):
            f.versions[f.admission_ref['path']]=1;return old(*a)
        f.runner.verify_owned_drain.side_effect=drain
        with self.assertRaises(RuntimeError):f.record()
    def test_previously_missing_receipt_appears_refuses(self):
        f=Fixture();old=f.drain
        def drain(*a):
            f.add('outer-result.json',{});return old(*a)
        f.runner.verify_owned_drain.side_effect=drain
        with self.assertRaises(RuntimeError):f.record()
    def test_auditor_failure_retains_model_and_missing_output(self):
        f=Fixture();out=f.auditor()
        self.assertEqual(out['phase'],'auditor');self.assertEqual(out['model_actual_exit'],f.model_exit_ref)
        self.assertEqual(out['available_partial_metadata']['auditor-output']['status'],'missing')
        self.assertEqual(out['exact_recorded_processes_gone'],[f.auditor_owner,f.outer,f.co,*f.ranks])
    def test_auditor_wrong_interpreter_rejects(self):
        f=Fixture();f.auditor();f.auditor_owner['argv'][0]='/usr/bin/python3'
        with self.assertRaises(RuntimeError):f.record(phase='auditor',live_ref=f.audit_ref,session=101)
    def test_auditor_wrong_original_model_lock_rejects(self):
        f=Fixture();f.auditor();f.data[f.model_exit_ref['path']]['lock_identity']={'device':99,'inode':99}
        with self.assertRaises(RuntimeError):f.record(phase='auditor',live_ref=f.audit_ref,session=101)
    def test_partial_output_has_only_exact_known_metadata_paths(self):
        f=Fixture();f.record()
        self.assertTrue(all(Path(p).suffix=='.json'for p in f.reads))
        self.assertFalse(any(p.endswith(('.raw','.so','.bin','.pt'))for p in f.reads))
    def test_transport_rejects_changed_during_authenticated_read(self):
        f=Fixture();metadata=m.StableMetadata(f.runner,lambda:None);original=f.runner.source_auth.load
        def load(r):
            result=original(r);f.versions[r['path']]=1;return result
        f.runner.source_auth.load=load
        with self.assertRaises(RuntimeError):metadata.load(f.live_ref)
    def test_partial_total_budget(self):
        f=Fixture();metadata=m.StableMetadata(f.runner,lambda:None);metadata.partial_bytes=m.MAX_PARTIAL_BYTES
        with self.assertRaises(RuntimeError):metadata.snapshot(f.root/'admission.json')
    def test_helper_has_no_process_termination_or_lock_mutation_calls(self):
        source=Path(m.__file__).read_text();tree=ast.parse(source)
        attrs={n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)}
        self.assertFalse(attrs & {'kill','terminate','send_signal','acquire_lock','release_lock','cleanup_owned_service','retain_until_drained','Popen','system'})
    def test_import_is_stdlib_only(self):
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))

class ChildOwnerChecks(unittest.TestCase):
    def test_null_child_owner_with_missing_rank_records_rejects(self):
        f=Fixture()
        for index in (0,1):del f.data[str(f.root/('rank%d-owner.json'%index))]
        f.add('children-exits.json',dict(owners=[None],exit_codes=[1],successful=False))
        with self.assertRaisesRegex(RuntimeError,'Exact recorded process identity'):f.record()
        f.runner.verify_owned_drain.assert_not_called()

    def test_incomplete_child_identity_rejects(self):
        for missing in ('pid','start_ticks','ppid','argv','cgroup'):
            f=Fixture();child=copy.deepcopy(f.ranks[0]);del child[missing]
            f.add('children-exits.json',dict(owners=[child],exit_codes=[1],successful=False))
            with self.subTest(missing=missing),self.assertRaisesRegex(RuntimeError,'Exact recorded process identity'):
                f.record()
            f.runner.verify_owned_drain.assert_not_called()

    def test_replaced_complete_child_identity_rejects(self):
        f=Fixture();child=copy.deepcopy(f.ranks[0]);child['start_ticks']='999999'
        f.add('children-exits.json',dict(owners=[child],exit_codes=[1],successful=False))
        with self.assertRaisesRegex(RuntimeError,'No unrelated recorded child owner'):f.record()
        f.runner.verify_owned_drain.assert_not_called()

    def test_empty_children_before_rank_acknowledgement_is_valid_partial(self):
        f=Fixture()
        for index in (0,1):del f.data[str(f.root/('rank%d-owner.json'%index))]
        f.add('children-exits.json',dict(owners=[],exit_codes=[1],successful=False))
        out=f.record()
        self.assertEqual(out['exact_recorded_processes_gone'],[f.outer,f.co])
        self.assertTrue(out['failed']);self.assertFalse(out['successful'])
        self.assertFalse(out['automatic_successor_release']);self.assertFalse(out['full_model_qualified'])

    def test_one_acknowledged_rank_is_valid_partial(self):
        f=Fixture();del f.data[str(f.root/'rank1-owner.json')]
        f.add('children-exits.json',dict(owners=[f.ranks[0]],exit_codes=[1],successful=False))
        out=f.record()
        self.assertEqual(out['exact_recorded_processes_gone'],[f.outer,f.co,f.ranks[0]])
        self.assertEqual(out['available_partial_metadata']['rank1-owner.json']['status'],'missing')
        self.assertTrue(out['failed']);self.assertFalse(out['successful'])
        self.assertFalse(out['automatic_successor_release']);self.assertFalse(out['full_model_qualified'])


if __name__=='__main__':unittest.main(verbosity=2)
