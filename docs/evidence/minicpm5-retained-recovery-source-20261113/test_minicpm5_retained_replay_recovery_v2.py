"""Source-only synthetic recovery tests; no actual metadata or runtime IO."""
import ast
import contextlib
import copy
import hashlib
import io as streams
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_minicpm5_retained_replay_lifecycle_v1 as fixtures
import minicpm5_retained_replay_finalizer_v5 as recovery
import minicpm5_retained_replay_catalog_v2 as catalog
import minicpm5_retained_replay_contract_v2 as c
import minicpm5_retained_replay_caller_v2 as caller

Q=fixtures.Q


def compose(f):
    with patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':fixtures.consumer}):
        return catalog.compose(f.spec,f.resolved,f.running,f.observation_ref,f.observation,f.live_ref,f.live,
            f.observer_tool,'time',f.final,f.admission,f.terminal,f.read,lambda:None)


class RecoveryFixture(fixtures.Fixture):
    def __init__(self):
        super().__init__();self.identities={};self.payloads={};self.loaded=[]
        self.spec=json.loads(json.dumps(self.spec,sort_keys=True))
        self.values[self.spec_ref['path']]=self.spec
        self.resolved['plan']['source_sha256']=self.spec['source_sha256']
        self.values[c.ACCEPTED['path']]=dict(passed=True,source_ready_for_caller_integration=True,
            production_deletion_admitted=False,launch_authorized=False)
        self.values[self.review_ref['path']]=dict(schema='minicpm5_retained_replay_caller_source_review_v2',
            reviewer='/root',passed=True,blockers=[],spec=self.spec_ref,source_sha256=self.spec['source_sha256'],
            launch_authorized=True,deletion_authorized=False)
        for i,(path,value)in enumerate(self.values.items()):
            self.payloads[path]=c.publication_bytes(value)
            self.identities[path]=dict(realpath=path,device=1,inode=i+1000,size=len(self.payloads[path]),mtime_ns=1,ctime_ns=2)
        # Source content reads are allowed; all saved JSON values above and below
        # are synthetic and stay in memory, including the abandoned directory.
        for path in {*c.FILES,*recovery.EXTRA_SOURCES}:
            data=Path(path).read_bytes();self.payloads[path]=data
            self.identities[path]=dict(realpath=path,device=1,inode=len(self.identities)+1000,size=len(data),mtime_ns=1,ctime_ns=2)
        self.supplement={p:hashlib.sha256(self.payloads[p]).hexdigest()for p in recovery.EXTRA_SOURCES}
        self.observation['running_identity']=copy.deepcopy(self.identities[self.running_ref['path']])
        self.final['running_identity']=copy.deepcopy(self.observation['running_identity'])
        self.observation['full_replay_equality'].pop('bytes',None)
        self.request_ref=c.publication_ref(self.request_ref['path'],self.request)
        self.old_root=Path(self.spec['output_root'])/'final-publication'
        self.rows=[]
        for i,name in enumerate(recovery.PARTIAL_NAMES):
            path=str(self.old_root/name);self.put(path,dict(synthetic_partial=i));self.rows.append(self.record(path))
        outer_tool=self.put(str(Q/'synthetic-recovery-outer-tool.json'),dict(schema='root_actual_tool_terminal_v1',
            actual_tool=self.actual,outer_terminal=self.outer_ref))
        observer_tool=self.put(str(Q/'synthetic-recovery-observer-tool.json'),dict(schema='root_actual_tool_terminal_v1',
            actual_tool=self.observer_tool,observation=self.observation_ref,full_replay_equality=self.observation['full_replay_equality']))
        running_record=self.record(self.running_ref['path']);running_record.update(self.running_ref)
        self.interruption=dict(schema='minicpm5_retained_finalizer_incomplete_observation_v1',reviewer='/root',
            request=self.record(self.request_ref['path']),finalizer_source=self.record(recovery.OLD_SELF),
            running_handoff=running_record,original_lock_identity=c.LOCK,actual_exit_not_observed=True,
            termination_cause_unknown=True,actual_exit_code=None,actual_terminal_chunk=None,final_publication_complete=False,
            deletion_performed=False,new_speedup_claim=False,partial_history_preserved=True,replay_completed_and_independently_equal=True,
            observed_lock_rows=[],exact_finalizer_argv_matches=[],tool=dict(kind='tool_session',session_id=333),
            all_four_publication_paths_absent=[self.spec['publication_paths'][key]for key in ('outer_actual_exit','observer_actual_exit','successor_catalog')]+[self.spec['terminal_handoff_path']],
            partial_final_history=self.rows,successful_replay_actual_tool=self.record(outer_tool['path']),
            successful_observer_actual_tool=self.record(observer_tool['path']))
        self.interruption_ref=self.put(str(Q/'synthetic-interruption.json'),self.interruption)
        self.recovery_review=dict(schema='minicpm5_retained_replay_recovery_source_review_v1',reviewer='/root',passed=True,
            blockers=[],original_request=self.request_ref,interruption=self.interruption_ref,original_spec=self.spec_ref,
            original_review=self.review_ref,source_sha256=self.supplement,recovery_name='synthetic-recovery',launch_authorized=True,
            deletion_authorized=False,model_or_raw_replay_authorized=False)
        self.recovery_review_ref=self.put(str(Q/'synthetic-recovery-review.json'),self.recovery_review)
        self.recovery_request=dict(original_request=self.request_ref,interruption=self.interruption_ref,
            recovery_review=self.recovery_review_ref,updated_at='synthetic-recovery-time')
        self.recovery_request_ref=self.put(str(Q/'synthetic-recovery-request.json'),self.recovery_request)
    def put(self,path,value):
        self.values[path]=value;self.payloads[path]=c.publication_bytes(value)
        self.identities[path]=dict(realpath=path,device=1,inode=len(self.identities)+1000,size=len(self.payloads[path]),mtime_ns=1,ctime_ns=2)
        return c.publication_ref(path,value)
    def record(self,path):
        data=self.payloads[path];return dict(path=path,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
            identity={k:self.identities[path][k]for k in recovery.IDENTITY_KEYS})
    def api(self):
        return SimpleNamespace(read=self.read,raw=lambda path,cap:self.payloads[path],
            identity=lambda path:copy.deepcopy(self.identities[str(path)]),
            loaded_local_sources=lambda pins:self.loaded.append(copy.deepcopy(pins)))
    def sources(self):
        return recovery.reviewed_sources(self.spec_ref,self.review_ref,self.recovery_review,self.request_ref,
            self.interruption_ref,self.read,self.api())


class SourceAdmission(unittest.TestCase):
    def test_v5_is_only_self_path_and_two_stable_output_order_changes(self):
        old=(Q/'minicpm5_retained_replay_finalizer_v4.py').read_text()
        before="[*spec['publication_paths'].values(),spec['terminal_handoff_path']]"
        after="[spec['publication_paths'][key]for key in ('outer_actual_exit','observer_actual_exit','successor_catalog')]+[spec['terminal_handoff_path']]"
        self.assertEqual(old.count(before),2)
        expected=old.replace(before,after).replace("SELF=str(c.Q/'minicpm5_retained_replay_finalizer_v4.py')",
            "SELF=str(c.Q/'minicpm5_retained_replay_finalizer_v5.py')")
        self.assertEqual(expected,(Q/'minicpm5_retained_replay_finalizer_v5.py').read_text())
    def test_original_ref_prefix_predicates_are_identical_ast(self):
        old=ast.parse((Q/'minicpm5_retained_replay_caller_v2.py').read_text())
        new=ast.parse((Q/'minicpm5_retained_replay_finalizer_v5.py').read_text())
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef)and n.name=='refs')
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef)and n.name=='reviewed_sources')
        def checks(fn):
            return [ast.dump(n.args[0],include_attributes=False)for n in ast.walk(fn)
                if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='require']
        self.assertEqual(checks(a),checks(b)[:2]+checks(b)[4:5])
    def test_old_spec_unchanged_explicit_two_pin_union_only(self):
        f=RecoveryFixture();original=copy.deepcopy(f.spec);spec,supplement,full=f.sources()
        self.assertEqual(spec,original);self.assertEqual(f.spec,original)
        self.assertEqual(set(full)-set(spec['source_sha256']),set(recovery.EXTRA_SOURCES))
        self.assertEqual(f.loaded,[full]);self.assertEqual(supplement,f.supplement)
    def test_frozen_refs_and_new_prefix_reject_same_old_defects(self):
        for mode in ('review','caller','accepted','spec'):
            f=RecoveryFixture()
            if mode=='review':f.values[f.review_ref['path']]['deletion_authorized']=True
            if mode=='caller':f.payloads[c.FILES[0]]+=b'\n'
            if mode=='accepted':f.values[c.ACCEPTED['path']]['launch_authorized']=True
            if mode=='spec':f.spec['schema']='wrong'
            with self.subTest(mode=mode),patch.object(caller,'api',return_value=(f.api(),None)):
                with self.assertRaises(ValueError):caller.refs(f.spec_ref,f.review_ref)
                with self.assertRaises(ValueError):f.sources()
    def test_recovery_cannot_replace_or_add_unreviewed_sources_or_change_binding(self):
        for mode in ('replace','extra','bad_sha','extra_field','request','old_review','unauthorized','name'):
            f=RecoveryFixture()
            if mode=='replace':f.supplement[c.FILES[0]]=f.spec['source_sha256'][c.FILES[0]]
            if mode=='extra':f.supplement[str(Q/'not-reviewed.py')]='e'*64
            if mode=='bad_sha':f.supplement[recovery.SELF]='e'*64
            if mode=='extra_field':f.recovery_review['pretend']=True
            if mode=='request':f.recovery_review['original_request']=f.ref('wrong')
            if mode=='old_review':f.recovery_review['original_review']=f.ref('wrong')
            if mode=='unauthorized':f.recovery_review['launch_authorized']=False
            if mode=='name':f.recovery_review['recovery_name']='../escape'
            with self.subTest(mode=mode),self.assertRaises(ValueError):f.sources()


class RecoveredCatalog(unittest.TestCase):
    def test_successor_is_mechanically_one_predicate_change(self):
        old=(Q/'minicpm5_retained_replay_catalog_v1.py').read_text()
        expected=old.replace("c.require(all(delta.get(p)==h for p,h in prior_pins.items()),'All earlier catalog source pins remain in the extension')",
            "c.require(all(pins.get(p)==h for p,h in prior_pins.items()),'All earlier catalog source pins remain in the authenticated full map')")
        self.assertNotEqual(old,expected);self.assertEqual(expected,(Q/'minicpm5_retained_replay_catalog_v2.py').read_text())
    def test_recovered_prior_pin_valid_but_old_composer_rejects(self):
        f=fixtures.Fixture();base=f.read(f.read(f.spec['history_overlay'])['base_plan'])
        recovered=next(iter(base['source_sha256']));f.oldcat['runs'][0]['source_sha256'][recovered]=base['source_sha256'][recovered]
        with self.assertRaisesRegex(ValueError,'in the extension'):f.compose()
        out=compose(f);self.assertNotIn(recovered,out['new_source_delta'])
        self.assertEqual(out['objects'][2][1]['runs'][:-2],f.oldcat['runs'])
    def test_recovered_missing_or_conflicting_full_pin_rejected(self):
        for mode in ('missing','conflict'):
            f=fixtures.Fixture();base=f.read(f.read(f.spec['history_overlay'])['base_plan']);path=next(iter(base['source_sha256']))
            f.oldcat['runs'][0]['source_sha256'][path]=base['source_sha256'][path]
            if mode=='missing':f.spec['source_sha256'].pop(path)
            else:f.oldcat['runs'][0]['source_sha256'][path]='f'*64
            with self.subTest(mode=mode),self.assertRaises(ValueError):compose(f)
    def test_realistic_recovered_partition_and_pure_chained_rebinding(self):
        f=RecoveryFixture();old_pins=f.spec['source_sha256'];overlay=f.values[f.spec['history_overlay']['path']]
        base=f.values[overlay['base_plan']['path']];target=3484
        recovered={str(Q/('synthetic-recovered-large-%05d.py'%i)):'a'*64 for i in range(target)}
        old_pins.update(recovered);base['source_sha256'].update(recovered);f.oldcat['runs'][0]['source_sha256'].update(recovered)
        for i in range(165):
            ref=f.ref('synthetic-partition-row-%03d'%i);f.values[ref['path']]=dict(old_failure=True)
            f.oldcat['runs'].append(dict(terminal=ref,tool=dict(kind='tool_session',session_id=10000+i,exit_code=1,terminal_chunk='ab1234'),
                owners=[],units=[],source_sha256={}))
        before=compose(f);unchanged=copy.deepcopy(before)
        provenance=dict(interrupted_finalizer_exit_code=None,interrupted_finalizer_terminal_chunk=None,
            interrupted_finalizer_termination_cause_unknown=True,source_sha256=f.supplement)
        with patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':fixtures.consumer}):
            out=recovery.rebind(before,f.spec,f.supplement,provenance,f.read,lambda:None)
        self.assertEqual(before,unchanged);self.assertEqual(out['catalog_rows'],169)
        self.assertEqual(out['full_sources'],len(old_pins)+2)
        for ref,value in out['objects']:self.assertEqual(ref,c.publication_ref(ref['path'],value));f.values[ref['path']]=value
        outer,observer,new,idle=[v for _,v in out['objects']]
        self.assertEqual(outer['actual_tool'],f.actual);self.assertEqual(observer['actual_tool'],f.observer_tool)
        self.assertEqual(outer['owners'],before['objects'][0][1]['owners'])
        self.assertEqual([owner['pid']for owner in outer['owners']],[101,201,202])
        self.assertEqual(outer['complete_actual_phase_states'],before['objects'][0][1]['complete_actual_phase_states'])
        self.assertEqual(observer['owners'],before['objects'][1][1]['owners'])
        self.assertEqual(new['runs'][:-2],f.oldcat['runs']);self.assertEqual(new['runs'][-1]['source_sha256'],{})
        self.assertEqual(new['runs'][-2]['source_sha256'],dict(before['new_source_delta'],**f.supplement))
        self.assertEqual(observer['actual_outer_exit'],out['outer_actual_exit'])
        completed=idle['minicpm5_retained_replay_completed']
        self.assertEqual(completed['actual_outer_terminal'],out['outer_actual_exit'])
        self.assertEqual(completed['actual_observer_terminal'],out['observer_actual_exit'])
        self.assertEqual(new['current_handoff'],out['idle_handoff'])
        for value in (outer,observer):
            self.assertEqual(value['complete_source_map_sha256'],c.sha(c.canonical(dict(old_pins,**f.supplement))))
            self.assertEqual(value['source_delta_sha256'],c.sha(c.canonical(out['source_delta'])))
            self.assertEqual(value['recovered_sources'],before['objects'][0][1]['recovered_sources'])
            self.assertEqual(value['source_delta_sources'],len(out['source_delta']))
            self.assertEqual(value['newly_recorded_sources'],len(out['new_source_delta']))
            self.assertFalse(value['new_speedup_claim']);self.assertFalse(value['deletion_authorized'])
        self.assertEqual(out['retained_replay_passed'],before['retained_replay_passed'])
        self.assertLess(len(c.publication_bytes(new)),8<<20)
        checked=fixtures.consumer.serial_catalog(dict(serial_predecessors=out['catalog'],
            source_sha256=dict(old_pins,**f.supplement)),f.read,lambda:None)
        self.assertEqual(checked['runs'],new['runs'])


class InterruptedNamespace(unittest.TestCase):
    def test_actual_sorted_json_spec_reproduces_v4_order_error_and_v5_preserves_exact_paths(self):
        import minicpm5_retained_replay_finalizer_v4 as failed
        f=RecoveryFixture()
        self.assertEqual(list(f.spec['publication_paths']),['observer_actual_exit','outer_actual_exit','successor_catalog'])
        with self.assertRaisesRegex(ValueError,'before all four final outputs'):
            failed.validate_interruption(f.interruption,f.request_ref,f.request,f.observation,f.spec)
        self.assertEqual(recovery.validate_interruption(f.interruption,f.request_ref,f.request,f.observation,f.spec),f.rows)
        for permutation in (list(reversed(f.interruption['all_four_publication_paths_absent'])),
                f.interruption['all_four_publication_paths_absent'][:-1]+[f.interruption['all_four_publication_paths_absent'][0]]):
            bad=copy.deepcopy(f.interruption);bad['all_four_publication_paths_absent']=permutation
            with self.assertRaises(ValueError):recovery.validate_interruption(bad,f.request_ref,f.request,f.observation,f.spec)
    def test_unknown_exit_and_exact_three_files_required(self):
        for mode in ('ok','exit','tool','rows','old_source','running','outputs'):
            f=RecoveryFixture();v=f.interruption
            if mode=='exit':v['actual_exit_code']=0
            if mode=='tool':v['tool']['session_id']=f.actual['session_id']
            if mode=='rows':v['partial_final_history'].append(v['partial_final_history'][0])
            if mode=='old_source':v['finalizer_source']['sha256']='e'*64
            if mode=='running':v['running_handoff']['identity']['inode']+=1
            if mode=='outputs':v['all_four_publication_paths_absent'].pop()
            with self.subTest(mode=mode):
                if mode=='ok':self.assertEqual(recovery.validate_interruption(v,f.request_ref,f.request,f.observation,f.spec),f.rows)
                else:
                    with self.assertRaises(ValueError):recovery.validate_interruption(v,f.request_ref,f.request,f.observation,f.spec)
    def test_exact_partial_directory_identity_bytes_names_and_no_transient(self):
        for mode in ('ok','extra','missing','bytes','inode','directory'):
            f=RecoveryFixture();names=list(recovery.PARTIAL_NAMES)
            if mode=='extra':names.append('history-part-002.json.publishing')
            if mode=='missing':names.pop()
            if mode=='bytes':f.payloads[f.rows[0]['path']]=b'changed'
            if mode=='inode':f.identities[f.rows[0]['path']]['inode']+=1
            st=SimpleNamespace(st_mode=stat.S_IFDIR|0o700,st_dev=1,st_ino=2,st_mtime_ns=3,st_ctime_ns=4)
            expected={k:getattr(st,k)for k in ('st_dev','st_ino','st_mode','st_mtime_ns','st_ctime_ns')}
            if mode=='directory':expected['st_ino']+=1
            with self.subTest(mode=mode),patch.object(Path,'resolve',lambda p:p),patch.object(Path,'is_symlink',return_value=False),\
                patch.object(Path,'lstat',return_value=st),patch.object(recovery.os,'scandir',return_value=contextlib.nullcontext(iter(SimpleNamespace(name=n)for n in names))):
                if mode=='ok':self.assertEqual(recovery.partial_directory(f.api(),f.old_root,f.rows,expected),expected)
                else:
                    with self.assertRaises(ValueError):recovery.partial_directory(f.api(),f.old_root,f.rows,expected)
    def test_exact_old_publisher_argument_rejected_and_unrelated_strings_allowed(self):
        for argv,allowed in ((['python',recovery.OLD_SELF,'--request','old'],False),(['python','-c','text '+recovery.OLD_SELF],True)):
            api=SimpleNamespace(process_api=lambda:SimpleNamespace(proc=lambda pid:dict(argv=argv)))
            with patch.object(recovery.os,'scandir',return_value=contextlib.nullcontext(iter([SimpleNamespace(name='123')]))):
                if allowed:recovery.old_publisher_gone(api)
                else:
                    with self.assertRaisesRegex(ValueError,'must be absent'):recovery.old_publisher_gone(api)


def runtime_probe(mode='ok'):
    f=RecoveryFixture();events=[];writes=[];drains=[0];changed=[False];old_changed=[False];present=set()
    identities=f.identities;latest=[dict(path=f.running_ref['path'],identity=copy.deepcopy(f.observation['running_identity']))]
    outputs=[f.spec['publication_paths'][key]for key in ('outer_actual_exit','observer_actual_exit','successor_catalog')]+[f.spec['terminal_handoff_path']]
    if mode=='preexisting':present.add(outputs[0])
    if mode=='transient':present.add(outputs[0]+'.publishing')
    class Lock:
        owner=dict(pid=444,start_ticks='4440',argv=['python',recovery.SELF])
        def acquire(self):events.append('lock-acquire')
        def check(self):events.append('lock-check')
        def release(self,ok):assert ok is True;events.append('lock-release')
    class History:
        def __init__(self,spec,*args):
            original=copy.deepcopy(spec);original['source_sha256']=f.spec['source_sha256'];assert original==f.spec
            assert spec['source_sha256']==dict(f.spec['source_sha256'],**f.supplement);self.resolved=f.resolved
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
            if mode=='old_last_drain':old_changed[0]=True
            if mode=='future_last_drain':present.add(outputs[3])
            if mode=='source_last_drain':f.payloads[recovery.SELF]+=b'\n'
        return copy.deepcopy(f.drain)
    def partial(*args):
        events.append('partial-check')
        if old_changed[0]:raise ValueError('Synthetic old partial changed')
        return dict(synthetic_directory=1)
    def once(path,value):
        events.append('write-'+Path(path).name)
        if mode=='fail_catalog'and str(path)==outputs[2]:raise ValueError('Synthetic failed catalog write')
        ref=f.put(str(path),copy.deepcopy(value));writes.append(ref);present.add(str(path))
        if mode=='old_after_outer'and len(writes)==1:old_changed[0]=True
        if mode=='future_after_outer'and len(writes)==1:present.add(outputs[3])
        if mode=='latest_after_catalog'and str(path)==outputs[2]:latest[0]=dict(path=str(Q/'unknown.json'),identity={})
        if mode=='mutate_receipt_after_catalog'and str(path)==outputs[2]:identities[outputs[0]]['inode']+=1
        if str(path)==outputs[3]:latest[0]=dict(path=outputs[3],identity=copy.deepcopy(identities[outputs[3]]))
        return ref
    api=f.api();api.process_api=lambda:SimpleNamespace(gone=lambda owner:events.append('owner-gone'))
    api.RetainedLock=Lock;api.once=once
    result=None;error=None;stdout=streams.StringIO()
    with patch.object(caller,'api',return_value=(api,None)),patch.object(caller,'source_unchanged'),patch.object(caller,'storage'),\
        patch.object(caller,'latest_observation',side_effect=lambda:copy.deepcopy(latest[0])),\
        patch.object(recovery,'check_recorded'),patch.object(recovery,'partial_directory',side_effect=partial),\
        patch.object(recovery,'old_publisher_gone',side_effect=lambda io:events.append('old-gone')),\
        patch.object(recovery,'drain_phases',side_effect=drain),patch.object(fixtures.history,'History',History),\
        patch.object(fixtures.history,'Authority',return_value=None),patch.object(fixtures.serial,'save_history',return_value=f.final['complete_history']),\
        patch.object(Path,'mkdir'),patch.object(recovery.os.path,'lexists',side_effect=lambda p:str(p)in present),\
        patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':fixtures.consumer}),contextlib.redirect_stdout(stdout):
        try:result=recovery.publish_completed(f.recovery_request_ref)
        except ValueError as exc:error=str(exc)
    return dict(fixture=f,result=result,error=error,writes=writes,events=events,stdout=stdout.getvalue())


class RecoveryPublication(unittest.TestCase):
    def test_full_mock_run_retains_full_history_and_four_ordered_outputs(self):
        p=runtime_probe();self.assertIsNone(p['error']);f=p['fixture']
        self.assertEqual([r['path']for r in p['writes']],[f.spec['publication_paths'][key]for key in ('outer_actual_exit','observer_actual_exit','successor_catalog')]+[f.spec['terminal_handoff_path']])
        event=json.loads(p['stdout']);self.assertEqual(event['owner']['pid'],444);self.assertEqual(event['request'],f.recovery_request_ref)
        self.assertIsNone(event['interrupted_finalizer_exit_code'])
        self.assertLess(p['events'].index('resolve-complete-history'),p['events'].index('physical-complete-history'))
        self.assertEqual(p['result']['full_sources'],len(f.spec['source_sha256'])+2)
        self.assertEqual(p['events'][-1],'lock-release')
    def test_occupied_outputs_or_transients_fail_before_lock(self):
        for mode in ('preexisting','transient'):
            with self.subTest(mode=mode):
                p=runtime_probe(mode);self.assertTrue(p['error']);self.assertNotIn('lock-acquire',p['events']);self.assertEqual(p['writes'],[])
    def test_every_last_drain_change_blocks_all_outputs(self):
        for mode in ('history_last_drain','latest_last_drain','input_last_drain','old_last_drain','future_last_drain','source_last_drain'):
            with self.subTest(mode=mode):
                p=runtime_probe(mode);self.assertTrue(p['error']);self.assertEqual(p['writes'],[]);self.assertEqual(p['events'][-1],'lock-release')
    def test_recheck_after_each_write_preserves_old_directory_and_remaining_namespace(self):
        for mode,count in (('old_after_outer',1),('future_after_outer',1),('fail_catalog',2),
            ('latest_after_catalog',3),('mutate_receipt_after_catalog',3)):
            with self.subTest(mode=mode):
                p=runtime_probe(mode);self.assertTrue(p['error']);self.assertEqual(len(p['writes']),count)
                self.assertNotIn(p['fixture'].spec['terminal_handoff_path'],p['fixture'].values)


if __name__=='__main__':unittest.main(verbosity=2)
