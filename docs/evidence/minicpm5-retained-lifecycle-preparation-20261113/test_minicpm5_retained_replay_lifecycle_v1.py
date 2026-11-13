"""Pure synthetic lifecycle/consumer regressions; no real state or payload IO."""
import ast
import copy
import hashlib
import json
import re
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import minicpm5_retained_replay_contract_v1 as old
import minicpm5_retained_replay_observer_v1 as old_observer
import minicpm5_retained_replay_contract_v2 as c
import minicpm5_retained_replay_caller_v2 as caller
import minicpm5_retained_replay_observer_v2 as observer
import minicpm5_retained_replay_catalog_v1 as composer
import minicpm5_retained_replay_finalizer_v3 as finalizer
import minicpm5_retained_replay_history_v1 as history
import qwen35_package_preflight_serial_v2 as serial
import test_prepare_minicpm5_retained_replay_binding_v2 as preparation


def original_functions(path,names,scope):
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)


# Original consumer function bodies run unchanged. AST loading avoids unrelated
# historical imports whose top-level code reads actual saved metadata. Constants
# and external shape helpers are injected explicitly; no real History is entered.
consumer=ModuleType('synthetic_original_serial_consumer')
consumer.__dict__.update(copy=copy,hashlib=hashlib,re=re,Path=Path,require=c.require,
    prior=SimpleNamespace(Q=Q),MAX_SERIAL_RUNS=256,require_complete_binding=lambda value:None,
    identity=lambda path:(_ for _ in ()).throw(AssertionError('Real identity forbidden')),
    sdk=SimpleNamespace(BINDING=dict(path=str(Q/'synthetic-original-sdk-binding.json'),sha256='b'*64)))
physical_scope=dict(re=re,require=c.require)
original_functions(Q/'minicpm5_sdk_physical_evidence_v3.py',{'unit_name'},physical_scope)
pin_scope=dict(re=re,Path=Path,require=c.require)
original_functions(Q.parent/'historical_b3_payload_partition_v1.py',{'pin_shape'},pin_scope)
metadata_scope=dict(require=c.require,history=SimpleNamespace(pin_shape=pin_scope['pin_shape']))
original_functions(Q.parent/'nemotron9b_package_metadata_v27.py',{'pin_shape'},metadata_scope)
reference_scope=dict(re=re,Path=Path,require=c.require)
original_functions(Q/'minicpm5_layer1_operator_sdk_sources_v1.py',{'reference'},reference_scope)
consumer.sources=SimpleNamespace(reference=reference_scope['reference'],metadata=SimpleNamespace(pin_shape=metadata_scope['pin_shape']),
    text=lambda path:(_ for _ in ()).throw(AssertionError('Real source read forbidden')))
consumer.physical_evidence=SimpleNamespace(unit_name=physical_scope['unit_name'])
original_functions(Q/'minicpm5_layer1_operator_sdk_runtime_completion_v4.py',
    {'serial_catalog','handoff_candidates','latest_handoff','publication_view','require_latest_terminal'},consumer.__dict__)


class Fixture:
    def __init__(self,failed=False):
        prep=preparation.Fixture();out=prep.run();self.spec=out['values']['spec'];self.values={}
        for path,row in prep.saved.items():
            if path.endswith('.json'):self.values[path]=json.loads(row['payload'])
        self.values.update({path:json.loads(data)for path,data in out['files'].items()})
        self.spec.update(storage_plan=c.PLAN,original_plan=c.ORIGINAL_PLAN,source_acceptance=c.ACCEPTED)
        self.spec['source_sha256'][c.ACCEPTED['path']]=c.ACCEPTED['sha256']
        self.values[self.spec['history_overlay']['path']]['source_extension'][c.ACCEPTED['path']]=c.ACCEPTED['sha256']
        self.spec_ref=out['values']['candidate']['spec_ref'];self.review_ref=self.ref('synthetic-runtime-review')
        self.values[self.spec_ref['path']]=self.spec
        self.values[self.review_ref['path']]={}
        self.oldcat=self.values[self.spec['current_serial_catalog']['path']]
        self.oldcat['completed_sdk']=copy.deepcopy(consumer.sdk.BINDING)
        for i,row in enumerate(self.oldcat['runs']):
            row['tool']['terminal_chunk']='abc'+str(i);row['units']=['trn1-synthetic-old-'+str(i)]
        self.initial_ref=self.oldcat['current_handoff'];self.initial=self.read(self.initial_ref)
        self.initial['mini_operator_current']=dict(retained='original scientific failure')
        self.values[self.initial_ref['path']]=self.initial
        self.admission_ref=self.ref('synthetic-admission')
        self.admission=dict(schema='minicpm5_retained_replay_outer_admission_v1',run_name=self.spec['run_name'],
            spec=self.spec_ref,review=self.review_ref,lock_identity=c.LOCK,attempts=dict(pair='1'*32,replay='2'*32),
            outer_owner=dict(pid=101,start_ticks='1010'),current_serial_catalog=self.spec['current_serial_catalog'],previous=self.initial_ref)
        self.actual=dict(kind='tool_session',session_id=111,exit_code=1 if failed else 0,terminal_chunk='abcd01')
        self.observer_tool=dict(kind='tool_session',session_id=222,exit_code=0,terminal_chunk='abcd02')
        self.worker_refs={p:self.ref('synthetic-'+p+'-worker')for p in ('pair','replay')}
        self.workers={p:dict(unit=self.spec['units'][p],nonce=self.admission['attempts'][p],
            process=dict(pid=201+i,start_ticks=str(2010+i),argv=c.phase_argv(p,self.admission_ref)),
            owner=dict(pid=201+i,start_ticks=str(2010+i)))for i,p in enumerate(('pair','replay'))}
        tool={k:self.actual[k]for k in ('kind','session_id')}
        self.pair=c.running_value(self.initial,'pair',tool,self.admission_ref,self.admission,self.workers['pair'],
            self.worker_refs['pair'],'synthetic-time',self.initial_ref)
        self.pair_ref=c.publication_ref(self.spec['running_paths']['pair'],self.pair)
        self.running=c.running_value(self.pair,'replay',tool,self.admission_ref,self.admission,self.workers['replay'],
            self.worker_refs['replay'],'synthetic-time',self.pair_ref)
        self.running_ref=c.publication_ref(self.spec['running_paths']['replay'],self.running)
        self.terminal=dict(schema='minicpm5_retained_replay_outer_terminal_v1',self_exit_claimed=False,
            actual_external_exit_pending=True,full_owned_scope_drained=True,admission=self.admission_ref,
            outer_owner=self.admission['outer_owner'],failed=failed,phase_states=[dict(phase=p,unit=self.spec['units'][p],
                nonce=self.admission['attempts'][p],state={},owner=self.workers[p],exit=dict(actual_service_exit_observed=True,
                owner=self.workers[p],properties=dict(ExecMainCode='1',ExecMainStatus='0',Result='success')))for p in ('pair','replay')])
        self.outer_ref=self.ref('outer-terminal',Path(self.spec['output_root']))
        self.observation_ref=self.ref('observation',Path(self.spec['observer_root']))
        self.live_ref=self.ref('observer-live',Path(self.spec['output_root']))
        self.equality_ref=dict(self.ref('full-replay-equality',Path(self.spec['observer_root'])),bytes=100)
        self.request_ref=self.ref('synthetic-finalize-request')
        self.observer_request=self.ref('synthetic-observer-request')
        self.observer_owner=dict(pid=301,start_ticks='3010')
        self.drain=dict(physically_drained=True,all_declared_units=sorted(self.spec['units'].values()))
        self.observation=dict(schema='minicpm5_retained_replay_independent_exit_observation_v1',
            spec=self.spec_ref,review=self.review_ref,admission=self.admission_ref,outer_terminal=self.outer_ref,
            running=self.running_ref,running_identity=dict(inode=999,size=100),request=self.observer_request,
            observer_owner=self.observer_owner,actual_observer_exit_pending=True,self_exit_claimed=False,
            final_owned=copy.deepcopy(self.drain),full_replay_equality=None if failed else self.equality_ref,
            actual_outer_exit=observer.reconcile(self.terminal,self.running,self.actual),independent_live=self.live_ref,lock_identity=c.LOCK)
        self.live=dict(schema='minicpm5_retained_replay_observer_live_v1',independent_actual_live_observation=True,
            owner=self.observer_owner,request=self.observer_request,actual_tool=dict(kind='tool_session',session_id=222))
        self.final=dict(schema='minicpm5_retained_replay_fresh_final_publication_v1',running=self.running_ref,
            running_identity=self.observation['running_identity'],observer_owner=self.observer_owner,
            actual_observer_tool=self.observer_tool,observer_owner_gone=True,final_owned=copy.deepcopy(self.drain),
            lock_identity=c.LOCK,complete_history=[self.ref('synthetic-full-history')])
        self.equality=dict(schema='minicpm5_retained_raw_complete_replay_equality_v1',complete=True,
            full11859_logical_inputs_preserved=True,full32_replay_and_all_scientific_outputs_equal=True,
            all_original_failures_preserved=True,layouts98_byte_identical=True,scientific_reports14_byte_identical=True,
            full_frames32=32,cache_pairs9408=9408,cpu_hook_pairs5964=5964,stage_vectors13632=13632,
            full_model_qualified=False,new_speedup_claim=False)
        self.request=dict(observation=self.observation_ref,observer_live=self.live_ref,
            actual_observer_tool=self.observer_tool,updated_at='synthetic-time')
        for ref,value in ((self.admission_ref,self.admission),(self.pair_ref,self.pair),(self.running_ref,self.running),
            (self.outer_ref,self.terminal),(self.observation_ref,self.observation),(self.live_ref,self.live),
            (self.equality_ref,self.equality),(self.request_ref,self.request)):
            self.values[ref['path']]=value
        self.resolved=dict(plan=dict(source_sha256=self.spec['source_sha256'],serial_predecessors=self.spec['current_serial_catalog']))
    def ref(self,name,parent=Q):return dict(path=str(parent/(name+'.json')),sha256='a'*64)
    def read(self,ref,*args):return copy.deepcopy(self.values[ref['path']])
    def compose(self):
        with patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':consumer}):
            return composer.compose(self.spec,self.resolved,self.running,self.observation_ref,self.observation,
                self.live_ref,self.live,self.observer_tool,'synthetic-time',self.final,self.admission,self.terminal,self.read,lambda:None)
    def freshness(self,catalog_ref,idle_ref):
        raw=c.publication_bytes(self.values[idle_ref['path']]);ident=dict(mtime_ns=123,inode=99)
        plan=dict(serial_predecessors=catalog_ref,source_sha256=self.spec['source_sha256'],latest_completion={})
        # Only the unrelated historical completion-binding constant is injected;
        # original serial_catalog and the full freshness loop execute unchanged.
        with patch.object(consumer,'require_complete_binding',return_value=None):
            return consumer.require_latest_terminal(plan,self.read,lambda:None,paths=lambda:[Path(idle_ref['path'])],
                stat=lambda path:ident,read_text=lambda path:raw)


class RunningAndOldDefects(unittest.TestCase):
    def test_exact_old_running_bug_reproduced_both_phases(self):
        f=Fixture();previous=copy.deepcopy(f.initial);tool={k:f.actual[k]for k in ('kind','session_id')}
        for phase in ('pair','replay'):
            worker=copy.deepcopy(f.workers[phase]);worker['process']['argv']=old.phase_argv(phase,f.admission_ref)
            previous=old.running_value(previous,phase,tool,f.admission_ref,f.admission,worker,f.worker_refs[phase],'time')
            self.assertFalse(previous['running']);self.assertIsNone(previous['active_serial'])
            self.assertTrue(previous['live_tool_sessions']);self.assertTrue(previous['active_heavy_units'])
    def test_corrected_running_is_consistent_and_chains_exact_refs(self):
        f=Fixture()
        for value,ref in ((f.pair,f.initial_ref),(f.running,f.pair_ref)):
            self.assertTrue(value['running']);self.assertIsNotNone(value['active_serial'])
            self.assertEqual(value['previous'],ref);self.assertFalse(value['publication_only_since_bound_terminal'])
        self.assertFalse(f.initial['running']);self.assertIsNone(f.initial['active_serial'])
    def test_running_rejects_inconsistent_idle_and_wrong_previous(self):
        f=Fixture();tool={k:f.actual[k]for k in ('kind','session_id')}
        for previous,ref in ((dict(f.initial,running=True),f.initial_ref),(f.initial,f.ref('wrong'))):
            with self.assertRaises(ValueError):
                c.running_value(previous,'pair',tool,f.admission_ref,f.admission,f.workers['pair'],f.worker_refs['pair'],'time',ref)
    def test_old_completion_does_not_close_original_freshness(self):
        f=Fixture();running=copy.deepcopy(f.running);running['schema']='minicpm5_retained_replay_running_v1'
        running['running']=False;running['active_serial']=None
        idle=old_observer.render_completed(running,f.observation_ref,f.observation,f.live_ref,f.live,f.observer_tool,'time',f.final)
        idle_ref=c.publication_ref(f.spec['terminal_handoff_path'],idle);f.values[idle_ref['path']]=idle
        with self.assertRaisesRegex(ValueError,'publication-only'):
            f.freshness(f.spec['current_serial_catalog'],idle_ref)
    def test_v1_exact_running_comparison_prevents_ack_only_patch(self):
        tree=ast.parse((Q/'minicpm5_retained_replay_caller_v1.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='read_running')
        self.assertIn('value == expected',ast.unparse(fn))


class CatalogComposition(unittest.TestCase):
    def test_original_consumer_roundtrip_retains_all_rows_and_owners(self):
        f=Fixture();out=f.compose();objects=out['objects'];new=objects[-2][1];idle=objects[-1][1]
        self.assertEqual(new['runs'][:-2],f.oldcat['runs']);self.assertEqual(len(new['runs']),len(f.oldcat['runs'])+2)
        self.assertEqual(new['runs'][-2]['tool'],f.actual);self.assertEqual(new['runs'][-1]['tool'],f.observer_tool)
        self.assertEqual([x['pid']for x in new['runs'][-2]['owners']],[101,201,202])
        self.assertEqual(new['runs'][-2]['units'],sorted(f.spec['units'].values()))
        self.assertEqual(new['runs'][-1]['owners'],[f.observer_owner])
        self.assertFalse(idle['running']);self.assertIsNone(idle['active_serial']);self.assertNotIn('minicpm5_retained_replay_current_execution',idle)
        self.assertFalse(idle['minicpm5_retained_replay_completed']['full_model_qualified'])
        self.assertFalse(out['new_speedup_claim']);self.assertFalse(out['deletion_authorized'])
        for ref,value in objects:f.values[ref['path']]=value
        result=f.freshness(out['catalog'],out['idle_handoff']);self.assertTrue(result['complete'])
        self.assertEqual(result['publication_only_successors'],[])
    def test_metadata_only_failed_outer_is_retained_without_science_success(self):
        f=Fixture(failed=True);out=f.compose();outer=out['objects'][0][1]
        self.assertEqual(outer['actual_tool']['exit_code'],1);self.assertFalse(outer['passed'])
        self.assertFalse(out['retained_replay_passed']);self.assertIsNone(outer['full_replay_equality'])
        self.assertTrue(out['objects'][1][1]['original_outer_failure_preserved'])
    def test_failed_outer_cannot_omit_acknowledged_pair_owner(self):
        f=Fixture(failed=True);f.terminal['phase_states'][0].update(owner=None,exit=None)
        with self.assertRaisesRegex(ValueError,'acknowledged worker'):f.compose()
    def test_failed_outer_cannot_omit_acknowledged_replay_owner(self):
        f=Fixture(failed=True);f.terminal['phase_states'][1].update(owner=None,exit=None)
        with self.assertRaisesRegex(ValueError,'acknowledged worker'):f.compose()
    def test_full_source_delta_reconstructs_complete_map_without_repeating_base(self):
        f=Fixture();out=f.compose();delta=out['source_delta'];base=f.read(f.read(f.spec['history_overlay'])['base_plan'])
        self.assertFalse(set(base['source_sha256'])&set(delta))
        self.assertEqual(dict(base['source_sha256'],**delta),f.spec['source_sha256'])
        prior={p:h for row in f.oldcat['runs']for p,h in row['source_sha256'].items()}
        self.assertEqual(out['objects'][-2][1]['runs'][-2]['source_sha256'],{p:h for p,h in delta.items()if p not in prior})
        self.assertEqual(out['objects'][-2][1]['runs'][-1]['source_sha256'],{})
    def test_missing_owner_unit_or_source_fails_closed(self):
        for mode in ('owner','unit','source','source_conflict','phase'):
            f=Fixture()
            if mode=='owner':f.terminal['phase_states'][1]['owner']['process']['pid']=999
            if mode=='unit':f.final['final_owned']['all_declared_units'].pop()
            if mode=='source':f.values[f.spec['history_overlay']['path']]['source_extension'].pop(c.SUCCESSOR_FILES[1])
            if mode=='source_conflict':f.values[f.spec['history_overlay']['path']]['source_extension'][c.SUCCESSOR_FILES[1]]='0'*64
            if mode=='phase':f.terminal['phase_states'].pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):f.compose()
    def test_wrong_actual_tools_cannot_close_catalog(self):
        for mode in ('outer','observer','same'):
            f=Fixture()
            if mode=='outer':f.observation['actual_outer_exit']['actual_tool']['session_id']=999
            if mode=='observer':f.observer_tool['session_id']=999
            if mode=='same':f.observer_tool['session_id']=111
            with self.subTest(mode=mode),self.assertRaises(ValueError):f.compose()
    def test_all_original_science_cardinalities_remain_required(self):
        for key,value in (('full_frames32',31),('cache_pairs9408',9407),('full11859_logical_inputs_preserved',False),
            ('scientific_reports14_byte_identical',False),('layouts98_byte_identical',False)):
            f=Fixture();f.equality[key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Complete original replay'):f.compose()
    def test_catalog_cap_and_publication_cap_are_unchanged(self):
        f=Fixture();f.oldcat['runs']=f.oldcat['runs']*128
        with self.assertRaises(ValueError):f.compose()
        with self.assertRaisesRegex(ValueError,'8 MiB'):c.publication_bytes({'too_big':'x'*(8<<20)})
    def test_167_row_large_catalog_appends_only_new_pins_under_original_cap(self):
        f=Fixture();oldpins={str(Q/('synthetic-large-historical-%05d.py'%i)):'c'*64 for i in range(7000)}
        f.spec['source_sha256'].update(oldpins)
        f.values[f.spec['history_overlay']['path']]['source_extension'].update(oldpins)
        f.oldcat['runs'][0]['source_sha256'].update(oldpins)
        for i in range(165):
            ref=f.ref('synthetic-old-row-%03d'%i);f.values[ref['path']]=dict(original_failure=True)
            f.oldcat['runs'].append(dict(terminal=ref,tool=dict(kind='tool_session',session_id=3000+i,
                exit_code=1,terminal_chunk='abc012'),owners=[],units=[],source_sha256={}))
        target=(8<<20)-180000
        f.oldcat['runs'][0]['owners'][0]['retained_complete_evidence']='x'*(target-len(c.publication_bytes(f.oldcat))-60)
        before=len(c.publication_bytes(f.oldcat));out=f.compose();new=out['objects'][-2][1]
        after=len(c.publication_bytes(new));self.assertEqual(len(new['runs']),169)
        self.assertLess(after,8<<20);self.assertLess(after-before,180000)
        self.assertFalse(set(oldpins)&set(new['runs'][-2]['source_sha256']))
        duplicated=copy.deepcopy(new)
        for row in duplicated['runs'][-2:]:row['source_sha256']=out['source_delta']
        with self.assertRaisesRegex(ValueError,'8 MiB'):c.publication_bytes(duplicated)


def finalizer_probe(mode='ok'):
    f=Fixture();events=[];writes=[];identities={p:dict(inode=i+1000,size=100)for i,p in enumerate(f.values)}
    identities[f.running_ref['path']]=dict(inode=999,size=100)
    if mode=='oversized_ordinary':identities[f.observation_ref['path']]['size']=(8<<20)+1
    latest=[dict(path=f.running_ref['path'],identity=dict(inode=999,size=100))];drains=[0];changed=[False]
    class Lock:
        def acquire(self):events.append('lock-acquire')
        def check(self):events.append('lock-check')
        def release(self,ok):assert ok is True;events.append('lock-release')
    class History:
        def __init__(self,*args):self.resolved=f.resolved
        def resolve(self):events.append('resolve-complete-history');return self
        def physical_pair(self,*args):events.append('physical-complete-history');return dict(complete=True)
        def unchanged(self):
            events.append('history-unchanged')
            if mode=='history_last_drain'and changed[0]:raise ValueError('Synthetic inherited metadata changed')
    def drain(*args):
        drains[0]+=1;events.append('drain-'+str(drains[0]))
        if drains[0]==3:
            changed[0]=True
            if mode=='latest_last_drain':latest[0]=dict(path=str(Q/'unknown.json'),identity={})
            if mode=='input_last_drain':identities[f.outer_ref['path']]['inode']+=1
            if mode=='running_last_drain':identities[f.running_ref['path']]['inode']+=1
        return copy.deepcopy(f.drain)
    def once(path,value):
        events.append('write-'+Path(path).name)
        if mode=='fail_catalog'and str(path)==f.spec['publication_paths']['successor_catalog']:
            raise ValueError('Synthetic interrupted catalog write')
        ref=c.publication_ref(path,value);writes.append(ref);f.values[str(path)]=copy.deepcopy(value)
        identities[str(path)]=dict(inode=8000+len(writes),size=len(c.publication_bytes(value)))
        if mode=='latest_after_catalog'and str(path)==f.spec['publication_paths']['successor_catalog']:
            latest[0]=dict(path=str(Q/'unknown.json'),identity={})
        if mode=='mutate_receipt_after_catalog'and str(path)==f.spec['publication_paths']['successor_catalog']:
            identities[f.spec['publication_paths']['outer_actual_exit']]['inode']+=1
        return ref
    io=SimpleNamespace(read=f.read,identity=lambda p:copy.deepcopy(identities[p]),
        process_api=lambda:SimpleNamespace(gone=lambda owner:events.append('owner-gone')),
        RetainedLock=Lock,once=once)
    error=None;out=None
    with patch.object(caller,'api',return_value=(io,None)),patch.object(caller,'refs',return_value=f.spec),\
        patch.object(caller,'source_unchanged'),patch.object(caller,'storage'),patch.object(finalizer,'verify_self'),\
        patch.object(caller,'latest_observation',side_effect=lambda:copy.deepcopy(latest[0])),\
        patch.object(finalizer,'drain_phases',side_effect=drain),patch.object(history,'History',History),\
        patch.object(history,'Authority',return_value=None),patch.object(serial,'save_history',return_value=f.final['complete_history']),\
        patch.object(Path,'mkdir'),patch.object(finalizer.os.path,'lexists',return_value=False),\
        patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':consumer}):
        try:out=finalizer.publish_completed(f.request_ref)
        except ValueError as e:error=str(e)
    return dict(fixture=f,error=error,result=out,writes=writes,events=events)


class FinalPublication(unittest.TestCase):
    def test_arbitrary_large_runtime_metadata_rejected_before_lock(self):
        p=finalizer_probe('oversized_ordinary');self.assertIn('no arbitrary',p['error'])
        self.assertNotIn('lock-acquire',p['events']);self.assertEqual(p['writes'],[])
    def test_exact_envelope_requires_all_hash_size_path_and_role_bindings(self):
        ref=dict(path=str(Q/'synthetic-enrolled-large.json'),sha256='d'*64);size=(8<<20)+9
        spec=dict(source_sha256={ref['path']:ref['sha256']},exact_envelopes={ref['path']:
            dict(sha256=ref['sha256'],bytes=size,role='complete_json_metadata_envelope')})
        self.assertEqual(finalizer.metadata_cap(ref,spec),size)
        self.assertEqual(finalizer.metadata_cap(dict(path=str(Q/'ordinary.json'),sha256='a'*64),spec),8<<20)
        for key,value in (('sha256','e'*64),('bytes',65<<20),('role','arbitrary_json')):
            changed=copy.deepcopy(spec);changed['exact_envelopes'][ref['path']][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):finalizer.metadata_cap(ref,changed)
    def test_receipts_catalog_then_idle_after_complete_history_and_postdrain_checks(self):
        p=finalizer_probe();self.assertIsNone(p['error']);f=p['fixture']
        self.assertEqual([r['path']for r in p['writes']],[*f.spec['publication_paths'].values(),f.spec['terminal_handoff_path']])
        pivot=p['events'].index('drain-3');write=next(i for i,x in enumerate(p['events'])if x.startswith('write-'))
        self.assertIn('history-unchanged',p['events'][pivot+1:write])
        self.assertTrue(f.freshness(p['result']['catalog'],p['result']['idle_handoff'])['complete'])
    def test_last_drain_changes_prevent_any_publication(self):
        for mode in ('history_last_drain','latest_last_drain','input_last_drain','running_last_drain'):
            with self.subTest(mode=mode):
                p=finalizer_probe(mode);self.assertTrue(p['error']);self.assertEqual(p['writes'],[])
    def test_interrupted_catalog_never_publishes_idle(self):
        p=finalizer_probe('fail_catalog');self.assertTrue(p['error']);self.assertEqual(len(p['writes']),2)
        self.assertNotIn(p['fixture'].spec['terminal_handoff_path'],p['fixture'].values)
    def test_late_handoff_or_receipt_mutation_after_catalog_prevents_idle(self):
        for mode in ('latest_after_catalog','mutate_receipt_after_catalog'):
            with self.subTest(mode=mode):
                p=finalizer_probe(mode);self.assertTrue(p['error']);self.assertEqual(len(p['writes']),3)
                f=p['fixture'];self.assertNotIn(f.spec['terminal_handoff_path'],f.values)
                catalog_ref=p['writes'][-1]
                with self.assertRaises(KeyError):
                    consumer.serial_catalog(dict(serial_predecessors=catalog_ref,source_sha256=f.spec['source_sha256']),f.read,lambda:None)


if __name__=='__main__':unittest.main(verbosity=2)
