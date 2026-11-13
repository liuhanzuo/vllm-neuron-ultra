"""Pure injected source/metadata tests; never acquire locks, inspect proc or unlink."""
import ast
import copy
import json
import subprocess
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from contextlib import ExitStack

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import minicpm5_retained_retirement_contract_v4 as c
import minicpm5_retained_retirement_caller_v4 as caller
import minicpm5_retained_retirement_holders_v4 as holders
import test_minicpm5_retained_replay_lifecycle_v1 as mini


class Fixture:
    def __init__(self):
        self.mini=mini.Fixture();completed=self.mini.compose();self.values=copy.deepcopy(self.mini.values)
        for ref,value in completed['objects']:self.values[ref['path']]=copy.deepcopy(value)
        self.hs=copy.deepcopy(self.mini.spec);self.hs['current_serial_catalog']=completed['catalog']
        self.hs['source_sha256'].update({path:c.sha(Path(path).read_bytes())for path in c.SOURCES})
        for path,h in c.base.FROZEN.items():self.hs['source_sha256'][path]=h
        self.hsref=self.store('synthetic-retirement-history-spec',self.hs)
        self.hrref=self.store('synthetic-retirement-history-review',dict(reviewer='/root',passed=True,blockers=[],
            spec=self.hsref,source_sha256=self.hs['source_sha256']))
        self.plan=dict(entries=[dict(original=dict(realpath='/mnt/nvme/synthetic-retire-original-%04d.raw'%i),
            keeper=dict(realpath='/mnt/nvme/synthetic-retire-keeper-%04d.raw'%i))for i in range(2352)])
        self.original=dict(source_sha256={p:h for p,h in self.hs['source_sha256'].items()if 'synthetic-original-'in p})
        self.spec_ref=self.ref('synthetic-retirement-spec');self.review_ref=self.ref('synthetic-retirement-review')
        self.exclusion=self.store('synthetic-writer-exclusion',dict(schema='minicpm5_retained_retirement_writer_exclusion_v1',
            reviewer='/root',spec_path=self.spec_ref['path'],run_name='synthetic-retirement',storage_plan=c.PLAN,original_lock_identity=c.LOCK,
            other_task_agents_excluded=True,external_namespace_writers_excluded=True,retained_for_entire_transaction_and_finalization=True))
        self.verification=dict(self.ref('synthetic-pair-verification'),bytes=100)
        self.values[completed['objects'][0][1]['full_replay_equality']['path']].update(
            storage_plan_sha256=c.PLAN['sha256'],verification=self.verification)
        refs=dict(history_spec=self.hsref,history_review=self.hrref,idle=completed['idle_handoff'],
            replay_outer_exit=completed['outer_actual_exit'],replay_observer_exit=completed['observer_actual_exit'],
            replay_completed=completed['idle_handoff'],verification=self.verification,
            replay_equality=completed['objects'][0][1]['full_replay_equality'],writer_exclusion=self.exclusion)
        execution={p:self.hs['source_sha256'][p]for p in (*c.SOURCES,*c.base.FROZEN)}
        self.spec=c.render_spec('synthetic-retirement',refs,self.hs,self.original,{p:self.hs['source_sha256'][p]for p in c.SOURCES},execution)
        self.review=dict(schema='minicpm5_retained_raw_replay_and_removal_review_v1',reviewer='/root',spec=self.spec_ref,
            passed=True,blockers=[],launch_authorized=True,production_deletion_admitted=True,
            source_sha256=self.spec['text_source_sha256'],execution_source_sha256=self.spec['execution_source_sha256'],
            verification=self.verification,replay_equality=refs['replay_equality'],
            full11859_logical_inputs_preserved=True,full32_replay_and_all_scientific_outputs_equal=True,
            all_original_failures_preserved=True,original_weights_precision_and_remaining_gates_unchanged=True,
            plan_sha256=c.PLAN['sha256'])
        for key,stem in (('remover_source','transaction'),('replay_source','legacy')):
            path=c.REMOVER if key=='remover_source'else str(Q/('minicpm5_retained_raw_'+stem+'_v1.py'))
            self.review[key]=dict(path=path,sha256=self.spec['source_sha256'][path]if key=='remover_source'else c.base.FROZEN[path])
        self.values[self.spec_ref['path']]=self.spec;self.values[self.review_ref['path']]=self.review
        self.owner=dict(pid=555,start_ticks='5555',argv=['synthetic-actual-owner'])
        self.tool=dict(kind='tool_session',session_id=777,exit_code=0,terminal_chunk='abc123')
        self.permit=self.ref('synthetic-live-permit')
        self.idle=self.values[refs['idle']['path']]
        self.running=c.running(self.idle,refs['idle'],self.spec_ref,self.spec,self.owner,
            {k:self.tool[k]for k in ('kind','session_id')},self.permit,'synthetic-time')
        self.running_ref=c.base.publication_ref(self.spec['paths']['running'],self.running)
        self.scope_proof=caller.ForegroundScopes(None,SimpleNamespace(owner=self.owner)).summary()
        self.terminal=dict(helper_scope_drain=self.scope_proof,owner=self.owner,self_exit_claimed=False,actual_exit_pending=True,failed=False,
            transaction_result=dict(self.ref('synthetic-transaction-result'),bytes=100))
        self.final_proof=dict(helper_scope_drain=self.scope_proof,owners_gone=True,physically_drained=True,complete_history=[self.ref('synthetic-full-history')])
    def ref(self,name):return dict(path=str(Q/(name+'.json')),sha256='a'*64)
    def store(self,name,value):ref=self.ref(name);self.values[ref['path']]=copy.deepcopy(value);return ref
    def read(self,ref,*args):return copy.deepcopy(self.values[ref['path']])
    def outputs(self,extension_override=None):
        oldcat=self.values[self.hs['current_serial_catalog']['path']]
        seen={p:h for row in oldcat['runs']for p,h in row['source_sha256'].items()}
        extension=dict(seen,**{p:self.hs['source_sha256'][p]for p in c.SOURCES})
        if extension_override is not None:extension=extension_override
        return c.completed_outputs(self.spec,self.hs,extension,oldcat,self.running,self.running_ref,
            self.ref('synthetic-retirement-terminal'),self.terminal,self.tool,self.final_proof,'synthetic-time')


class Contracts(unittest.TestCase):
    def test_actual_shape_3484_recovered_prior_pins_need_not_be_overlay_extension(self):
        f=Fixture();oldcat=f.values[f.hs['current_serial_catalog']['path']]
        recovered={str(Q/('synthetic-recovered-prior-%04d.py'%i)):'c'*64 for i in range(3484)}
        oldcat['runs'][-1]['source_sha256'].update(recovered)
        f.spec['source_sha256'].update(recovered);f.hs['source_sha256'].update(recovered)
        extension={p:f.spec['source_sha256'][p]for p in c.SOURCES}
        self.assertFalse(set(recovered)&set(extension))
        out=f.outputs(extension)
        self.assertEqual(out[1][1]['runs'][:-1],oldcat['runs'])
        self.assertEqual(out[1][1]['runs'][-1]['source_sha256'],extension)
        for ref,value in out:f.values[ref['path']]=value
        actual=mini.consumer.serial_catalog(dict(serial_predecessors=out[1][0],source_sha256=f.spec['source_sha256']),f.read,lambda:None)
        self.assertEqual(actual['current_handoff'],out[2][0])
    def test_global_recovered_pin_rebind_and_missing_new_extension_are_rejected(self):
        f=Fixture();oldcat=f.values[f.hs['current_serial_catalog']['path']]
        path=str(Q/'synthetic-recovered-prior.py');oldcat['runs'][-1]['source_sha256'][path]='c'*64
        f.spec['source_sha256'][path]='d'*64
        extension={p:f.spec['source_sha256'][p]for p in c.SOURCES}
        with self.assertRaisesRegex(ValueError,'prior source pins'):f.outputs(extension)
        f=Fixture();extension={p:f.spec['source_sha256'][p]for p in c.SOURCES[1:]}
        with self.assertRaisesRegex(ValueError,'binds new sources'):f.outputs(extension)

    def test_full_global_maps_and_original_text_sources_preserved(self):
        f=Fixture();resolved=dict(plan=dict(source_sha256=f.hs['source_sha256'],inputs={'/never/read/current.raw':{}}))
        proof=c.compatibility(f.spec,f.hs,resolved,f.original,f.plan)
        self.assertEqual(proof['candidate_current_input_intersection'],0)
        self.assertEqual(proof['complete_global_sources'],len(f.hs['source_sha256']))
        self.assertEqual(len(f.original['source_sha256']),1869)
        self.assertTrue(set(f.original['source_sha256'])<=set(f.spec['text_source_sha256']))
    def test_candidate_overlap_in_any_full_map_rejected(self):
        for mode in ('inputs','extension','global'):
            f=Fixture();target=f.plan['entries'][17]['original']['realpath'];inputs={}
            if mode=='inputs':inputs[target]={}
            if mode=='extension':f.hs['input_extension'][target]={}
            if mode=='global':f.hs['source_sha256'][target]='a'*64;f.spec['source_sha256'][target]='a'*64
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'intersects'):
                c.compatibility(f.spec,f.hs,dict(plan=dict(source_sha256=f.hs['source_sha256'],inputs=inputs)),f.original,f.plan)
    def test_idle_never_substitutes_for_explicit_writer_exclusion(self):
        f=Fixture();value=f.read(f.exclusion);c.validate_exclusion(value,f.spec_ref,f.spec)
        for key in ('other_task_agents_excluded','external_namespace_writers_excluded','retained_for_entire_transaction_and_finalization'):
            changed=dict(value,**{key:False})
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_exclusion(changed,f.spec_ref,f.spec)
    def test_missing_python_closure_or_original_source_rejected(self):
        f=Fixture();f.spec['text_source_sha256'].pop(c.SOURCES[1])
        with self.assertRaises(ValueError):c.validate(f.spec)
        f=Fixture();f.spec['text_source_sha256'].pop(next(iter(f.original['source_sha256'])))
        with self.assertRaisesRegex(ValueError,'1869'):
            c.compatibility(f.spec,f.hs,dict(plan=dict(source_sha256=f.hs['source_sha256'],inputs={})),f.original,f.plan)
    def test_actual_replay_and_observer_exits_required_before_retirement(self):
        f=Fixture();saved=SimpleNamespace(read=f.read);caller.prerequisites(f.spec,f.hs,saved)
        for key in ('replay_outer_exit','replay_observer_exit'):
            changed=Fixture();changed.values[changed.spec['references'][key]['path']]['actual_tool']['exit_code']=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'successful independently drained'):
                caller.prerequisites(changed.spec,changed.hs,SimpleNamespace(read=changed.read))
    def test_complete_science_cannot_be_replaced_by_counts_or_metadata_only(self):
        for key,value in (('full_frames32',31),('full11859_logical_inputs_preserved',False),('layouts98_byte_identical',False)):
            f=Fixture();f.values[f.spec['references']['replay_equality']['path']][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'scientific condition'):
                caller.prerequisites(f.spec,f.hs,SimpleNamespace(read=f.read))
    def test_matching_root_review_admits_exact_sources_only(self):
        f=Fixture()
        with patch.object(caller,'source_check'):
            spec,_,_=caller.load(f.spec_ref,f.review_ref,SimpleNamespace(read=f.read));self.assertEqual(spec,f.spec)
            f.review['production_deletion_admitted']=False
            with self.assertRaisesRegex(ValueError,'Independent root'):caller.load(f.spec_ref,f.review_ref,SimpleNamespace(read=f.read))
    def test_actual_exit_catalog_idle_roundtrip_preserves_original_consumer(self):
        f=Fixture();out=f.outputs();self.assertEqual([x[0]['path']for x in out],
            [f.spec['paths'][k]for k in ('actual_exit','catalog','idle')])
        oldcat=f.values[f.hs['current_serial_catalog']['path']]
        self.assertEqual(out[1][1]['runs'][:-1],oldcat['runs']);self.assertEqual(out[1][1]['runs'][-1]['owners'],[f.owner])
        for ref,value in out:f.values[ref['path']]=value
        actual=mini.consumer.serial_catalog(dict(serial_predecessors=out[1][0],source_sha256=f.spec['source_sha256']),f.read,lambda:None)
        self.assertEqual(actual['current_handoff'],out[2][0]);self.assertFalse(out[2][1]['running'])
        ident=dict(mtime_ns=123,inode=9)
        result=mini.consumer.require_latest_terminal(dict(serial_predecessors=out[1][0],source_sha256=f.spec['source_sha256'],latest_completion={}),
            f.read,lambda:None,paths=lambda:[Path(out[2][0]['path'])],stat=lambda path:ident,read_text=lambda path:c.base.publication_bytes(out[2][1]))
        self.assertTrue(result['complete'])
    def test_wrong_actual_exit_and_missing_drain_cannot_finalize(self):
        f=Fixture();f.tool['session_id']=999
        with self.assertRaises(ValueError):f.outputs()
        f=Fixture();f.final_proof['physically_drained']=False
        with self.assertRaises(ValueError):f.outputs()
    def test_failed_remover_is_not_promoted(self):
        f=Fixture();f.tool['exit_code']=1;f.terminal['failed']=True;f.terminal['transaction_result']=None
        out=f.outputs();self.assertFalse(out[0][1]['passed']);self.assertTrue(out[0][1]['original_failure_preserved'])


class SourceBoundary(unittest.TestCase):
    def test_only_public_frozen_transaction_owns_deletion(self):
        tree=ast.parse(Path(c.SOURCES[1]).read_text());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertEqual(calls.count('transaction.retire'),1)
        self.assertEqual(calls.count('transport.prepare_retirement_intent'),1)
        for forbidden in ('unlink','rename','rmtree','remove','kill','killpg'):
            self.assertFalse(any(name.split('.')[-1]==forbidden for name in calls))
        self.assertNotIn('transaction._retire',calls)
    def test_journal_budget_is_exact_and_does_not_change_replay_budget(self):
        self.assertEqual(c.LIMITS['journal_entries'],2352*2+2)
        self.assertEqual(c.base.LIMITS['max_entries'],512)
        self.assertEqual(c.LIMITS['journal_bytes'],128<<20)
    def test_holder_scanner_is_read_only_and_fails_closed_on_denial(self):
        tree=ast.parse(Path(c.SOURCES[2]).read_text());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertFalse(any(name.split('.')[-1]in ('unlink','kill','chmod','chown','flock','mkdir','write_text')for name in calls))
        root=SimpleNamespace(iterdir=lambda:(_ for _ in ()).throw(PermissionError('all-UID scan denied')))
        with self.assertRaises(PermissionError):holders.inspect(dict(entries=[]),dict(pid=1,start_ticks='1'),root)
    def test_actual_holder_scan_failure_never_stops_an_unrelated_process(self):
        scopes=SimpleNamespace(invoke=lambda ref:(_ for _ in ()).throw(PermissionError('sudo unavailable')))
        with self.assertRaises(PermissionError):caller.holders(dict(path='/synthetic/permit.json',sha256='a'*64),scopes)
    def test_three_field_metadata_ref_preserves_byte_limit(self):
        f=Fixture();ident=dict(size=100,inode=3);io=SimpleNamespace(identity=lambda p:copy.deepcopy(ident),read=lambda ref,cap:{})
        saved=caller.Saved(io);saved.read(f.verification)
        ident['size']=101
        with self.assertRaises(ValueError):saved.read(f.verification)


def run_probe(mode='ok'):
    import minicpm5_retained_replay_history_v1 as history_module
    import qwen35_package_preflight_serial_v2 as serial
    import minicpm5_retained_raw_transport_v1 as transport
    import minicpm5_retained_raw_transaction_v2 as transaction
    f=Fixture();events=[];made=set();identities={};written={};latest=[None];observed_holder_calls=[0]
    f.owner['argv']=[c.base.PYTHON,'-B','-u',c.SOURCES[1],'--mode','run','--spec',f.spec_ref['path'],
        '--spec-sha256',f.spec_ref['sha256'],'--review',f.review_ref['path'],'--review-sha256',f.review_ref['sha256']]
    f.values[c.PLAN['path']]=f.plan;f.values[c.base.ORIGINAL_PLAN['path']]=f.original
    def ident(path):
        if path not in identities:identities[path]=dict(size=100,inode=len(identities)+1)
        return copy.deepcopy(identities[path])
    latest[0]=dict(path=f.spec['references']['idle']['path'],identity=ident(f.spec['references']['idle']['path']))
    class Lock:
        owner=f.owner;fd=42
        def acquire(self):events.append('lock-acquire')
        def check(self):events.append('lock-check')
        def release(self,proof):assert proof is True;events.append('lock-release')
    class History:
        def __init__(self,*args):
            self.resolved=dict(plan=dict(source_sha256=f.hs['source_sha256'],inputs={}))
            self.meta=SimpleNamespace(read=lambda ref:dict(source_extension={p:h for row in f.values[f.hs['current_serial_catalog']['path']]['runs']for p,h in row['source_sha256'].items()}
                |{p:f.spec['source_sha256'][p]for p in c.SOURCES}))
        def resolve(self):events.append('complete-history-resolve');return self
        def physical_pair(self,*args):events.append('complete-history-physical');return {'complete':True}
        def freshness(self):events.append('original-current-freshness');return dict(latest_handoff=f.spec['references']['idle'])
        def unchanged(self):events.append('history-unchanged')
    def once(path,value):
        path=str(path);ref=c.base.publication_ref(path,value);f.values[path]=copy.deepcopy(value);written[path]=ref;made.add(path)
        events.append('write-'+Path(path).name);return ref
    def wait(*args,**kwargs):
        permit_path=str(Path(f.spec['paths']['control'])/'permit.json');permit_ref=written[permit_path]
        running=c.running(f.idle,f.spec['references']['idle'],f.spec_ref,f.spec,f.owner,
            {k:f.tool[k]for k in ('kind','session_id')},permit_ref,'synthetic-time')
        ref=once(f.spec['paths']['running'],running);latest[0]=dict(path=ref['path'],identity=ident(ref['path']))
        events.append('actual-external-ack');return True
    def inspect(ref,scopes):
        observed_holder_calls[0]+=1;events.append('all-uid-holder-scan')
        if mode=='holder_timeout':raise subprocess.TimeoutExpired(['synthetic-sudo'],90)
        if mode=='holder_denied':raise PermissionError('Cannot inspect every user')
        if mode=='late_other_writer'and observed_holder_calls[0]>=4:raise ValueError('Unrelated writer became visible')
        return dict(complete=True,zero_other_holders=True)
    def intent(plan,verification,path,admission,guard):
        self_expected=written.get(f.spec['paths']['running']);assert self_expected is not None
        guard();events.append('prepare-exact-original-intent')
        return once(path,dict(plan=plan,verification=verification,admission=admission))
    def retire(plan,verification,intent_ref,journal,guard):
        value=guard();assert value['intent']==intent_ref and value['exclusive_namespace_write_lease']is True
        events.append('frozen-public-transaction-retire');return once(Path(journal)/'complete.json',dict(complete=True))
    io=SimpleNamespace(identity=ident,read=f.read,once=once,ref=lambda p:written[str(p)],RetainedLock=Lock,
        process_api=lambda:SimpleNamespace(active_heavy=lambda:events.append('all-heavy-idle')),wait_until=wait)
    original_drain=caller.ForegroundScopes.drain
    def drain(manager):events.append('helper-scope-drain');return original_drain(manager)
    with ExitStack()as stack:
        stack.enter_context(patch.object(caller.ForegroundScopes,'drain',autospec=True,side_effect=drain))
        for obj,name,value in ((caller,'api',lambda:io),(caller,'source_check',lambda spec:events.append('source-check')),
            (caller,'loaded_text_sources',lambda spec:events.append('complete-loaded-source-map')),
            (caller,'storage',lambda spec:{}),(caller,'latest',lambda:copy.deepcopy(latest[0])),(caller,'holders',inspect),
            (history_module,'History',History),(history_module,'Authority',lambda *args:None),
            (serial,'save_history',lambda *args:[f.ref('synthetic-actual-full-history')]),
            (transport,'validate_plan',lambda plan:plan),(transport,'prepare_retirement_intent',intent),
            (transaction,'retire',retire)):
            stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.object(Path,'mkdir',lambda self,*a,**k:made.add(str(self))))
        stack.enter_context(patch.object(Path,'exists',lambda self:str(self)in made))
        stack.enter_context(patch.object(caller.os.path,'lexists',side_effect=lambda p:str(p)in made))
        stack.enter_context(patch.object(caller.signal,'signal',return_value=None))
        error=None;result=None
        try:result=caller.run(f.spec_ref,f.review_ref)
        except(PermissionError,ValueError,subprocess.TimeoutExpired)as e:error=str(e)
    return dict(events=events,error=error,result=result,values=f.values,fixture=f,written=written)


class ActualEntrySequence(unittest.TestCase):
    def test_mock_run_requires_full_history_holders_and_actual_ack_before_frozen_retire(self):
        result=run_probe();self.assertIsNone(result['error']);events=result['events']
        pivot=events.index('frozen-public-transaction-retire')
        for event in ('complete-history-resolve','complete-history-physical','original-current-freshness',
            'all-uid-holder-scan','actual-external-ack','prepare-exact-original-intent','complete-loaded-source-map'):
            self.assertLess(events.index(event),pivot)
        terminal=result['values'][result['result']['path']]
        self.assertFalse(terminal['self_exit_claimed']);self.assertTrue(terminal['actual_exit_pending'])
        self.assertFalse(terminal['failed']);self.assertEqual(events[-1],'lock-release')
    def test_all_uid_scan_denial_never_reaches_intent_or_unlink(self):
        result=run_probe('holder_denied');self.assertIn('Cannot inspect',result['error'])
        self.assertNotIn('frozen-public-transaction-retire',result['events'])
        self.assertNotIn('prepare-exact-original-intent',result['events'])


def final_probe(mode='ok'):
    import minicpm5_retained_replay_history_v1 as history_module
    import qwen35_package_preflight_serial_v2 as serial
    import minicpm5_retained_raw_transport_v1 as transport
    f=Fixture();events=[];written=[];identities={};latest=[None];holder_count=[0]
    clock=[0.0];source_bad=[False];storage_bad=[False];signals={};resolve_checks=[None]
    def source_check(spec):
        events.append('source-check')
        if source_bad[0]:raise ValueError('Synthetic source changed')
        if mode=='slow_full_check'and 'resolve-loop'in events:clock[0]+=.6
    def storage(spec):
        events.append('storage-check')
        if storage_bad[0]:raise ValueError('Synthetic storage changed')
        return {}
    def set_signal(sig,handler):
        old=signals.get(sig);signals[sig]=handler;return old
    f.terminal.update(spec=f.spec_ref,review=f.review_ref,running=f.running_ref,intent=dict(f.ref('synthetic-whole-intent'),bytes=100))
    terminal_ref=dict(path=str(Path(f.spec['paths']['control'])/'terminal.json'),sha256='a'*64)
    request=dict(spec=f.spec_ref,review=f.review_ref,terminal=terminal_ref,actual_tool=f.tool,updated_at='synthetic-time')
    request_ref=f.store('synthetic-retirement-final-request',request)
    f.values[terminal_ref['path']]=f.terminal;f.values[f.running_ref['path']]=f.running;f.values[c.PLAN['path']]=f.plan
    f.values[f.terminal['transaction_result']['path']]=dict(schema='minicpm5_retained_raw_transaction_complete_v1',complete=True,
        plan_sha256=c.PLAN['sha256'],intent=f.terminal['intent'],pairs=[dict(path=str(Path(f.spec['paths']['journal'])/('%04d-done.json'%i)),sha256='a'*64,bytes=100)for i in range(2352)],
        all_keeper_bytes_retained=True,original_capture_identities_preserved=True,original_allocated_bytes_retired=10173284352)
    def ident(path):
        if path not in identities:identities[path]=dict(inode=len(identities)+1,size=100)
        return copy.deepcopy(identities[path])
    latest[0]=dict(path=f.running_ref['path'],identity=ident(f.running_ref['path']))
    extension={p:h for row in f.values[f.hs['current_serial_catalog']['path']]['runs']for p,h in row['source_sha256'].items()}
    extension.update({p:f.spec['source_sha256'][p]for p in c.SOURCES})
    class Lock:
        owner=dict(pid=888,start_ticks='8888');fd=44
        def acquire(self):events.append('lock-acquire')
        def check(self):events.append('lock-check')
        def release(self,ok):assert ok;events.append('lock-release')
    class History:
        def __init__(self,*a):
            self.guard=a[-1]
            self.meta=SimpleNamespace(read=lambda ref:dict(source_extension=extension)if ref==f.hs['history_overlay']else f.read(ref))
        def resolve(self):
            events.append('complete-history-resolve');events.append('resolve-loop')
            count=10000 if mode=='resolve_dense'else 4
            before=events.count('storage-check')
            for i in range(count):
                if mode!='slow_full_check':clock[0]=5*i/count
                if mode=='signal_during_resolve'and i==1:signals[caller.signal.SIGTERM](None,None)
                if mode=='deadline_during_resolve'and i==1:clock[0]=7200
                self.guard(mode=='forced_resolve_callback')
            resolve_checks[0]=events.count('storage-check')-before
            if mode=='source_at_exit'or mode=='source_on_failure':source_bad[0]=True
            if mode=='storage_at_exit':storage_bad[0]=True
            if mode=='metadata_at_exit':identities[request_ref['path']]['inode']+=1
            if mode=='tip_at_exit':latest[0]=dict(path='/synthetic/unknown.json',identity={})
            if mode in ('history_failure','source_on_failure'):raise ValueError('Synthetic History failure')
            events.append('resolve-return');return self
        def physical_pair(self,*a):
            events.append('complete-history-physical')
            for _ in range(4):self.guard()
            return {'complete':True}
        def unchanged(self):events.append('history-unchanged')
    def once(path,value):
        ref=c.base.publication_ref(path,value);f.values[str(path)]=copy.deepcopy(value);written.append(ref)
        events.append('write-'+Path(path).name)
        if mode=='tip_after_actual_exit'and str(path)==f.spec['paths']['actual_exit']:
            latest[0]=dict(path='/synthetic/unknown.json',identity={})
        return ref
    def inspect(ref,scopes):
        holder_count[0]+=1;events.append('all-uid-holder-scan')
        if mode=='holder_timeout':raise subprocess.TimeoutExpired(['synthetic-sudo'],90)
        if mode=='tip_during_scan'and holder_count[0]==4:latest[0]=dict(path='/synthetic/new.json',identity={})
        return dict(complete=True)
    def audit(*args):events.append('full-keeper-postcondition');return dict(f.ref('synthetic-independent-postcondition'),bytes=100)
    io=SimpleNamespace(identity=ident,read=f.read,once=once,RetainedLock=Lock,
        process_api=lambda:SimpleNamespace(gone=lambda owner:events.append('actual-remover-gone')))
    original_drain=caller.ForegroundScopes.drain
    def drain(manager):events.append('helper-scope-drain');return original_drain(manager)
    error=None;result=None
    with ExitStack()as stack:
        stack.enter_context(patch.object(caller.ForegroundScopes,'drain',autospec=True,side_effect=drain))
        for obj,name,value in ((caller,'api',lambda:io),(caller,'source_check',source_check),(caller,'storage',storage),(caller,'holders',inspect),
            (caller,'latest',lambda:copy.deepcopy(latest[0])),(history_module,'History',History),(history_module,'Authority',lambda *a:None),
            (serial,'save_history',lambda *a:[f.ref('synthetic-full-final-history')]),(transport,'validate_plan',lambda plan:plan),
            (transport,'audit_retired_state',audit),(transport,'_verification',lambda *a:dict(records=[dict(original_allocated_bytes=4325376)]*2352))):
            stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.object(caller.signal,'signal',side_effect=set_signal))
        stack.enter_context(patch.object(caller.time,'monotonic',side_effect=lambda:clock[0]))
        stack.enter_context(patch.object(Path,'exists',lambda self:str(self)==f.spec['paths']['control']));stack.enter_context(patch.object(Path,'mkdir'))
        stack.enter_context(patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':mini.consumer}))
        try:result=caller.finalize(request_ref)
        except(ValueError,subprocess.TimeoutExpired)as e:error=str(e)
    return dict(events=events,written=written,error=error,result=result,fixture=f,resolve_full_checks=resolve_checks[0])


class ActualFinalizeSequence(unittest.TestCase):
    def test_independent_full_history_and_keeper_audit_precede_actual_catalog_idle(self):
        p=final_probe();self.assertIsNone(p['error']);f=p['fixture']
        self.assertEqual([r['path']for r in p['written']][-4:-1],[f.spec['paths'][k]for k in ('actual_exit','catalog','idle')])
        self.assertEqual(p['written'][-1]['path'],str(Path(f.spec['paths']['control'])/'finalizer-helper-drain.json'))
        pivot=p['events'].index('write-actual-exit.json')
        for name in ('complete-history-resolve','complete-history-physical','actual-remover-gone','full-keeper-postcondition','all-uid-holder-scan'):
            self.assertLess(p['events'].index(name),pivot)
    def test_new_tip_during_final_blocking_holder_scan_prevents_idle(self):
        p=final_probe('tip_during_scan');self.assertTrue(p['error'])
        self.assertNotIn(p['fixture'].spec['paths']['idle'],[r['path']for r in p['written']])
    def test_new_external_writer_after_ack_prevents_transaction(self):
        result=run_probe('late_other_writer');self.assertIn('writer became visible',result['error'])
        self.assertNotIn('frozen-public-transaction-retire',result['events'])


class ResolveCycle(unittest.TestCase):
    def test_dense_readonly_callbacks_keep_checks_bounded_and_restore_full_guard(self):
        p=final_probe('resolve_dense');self.assertIsNone(p['error'])
        self.assertEqual(p['resolve_full_checks'],10)
        events=p['events'];end=events.index('resolve-return');physical=events.index('complete-history-physical')
        self.assertIn('storage-check',events[end:physical])
        # The retained History callback dispatches to the original full guard
        # after resolve, even though all four physical calls share one clock.
        tail=events[physical:events.index('write-finalizer-permit.json')]
        self.assertGreaterEqual(tail.count('storage-check'),4)
        start=events.index('resolve-loop')
        self.assertGreaterEqual(events[start:end].count('lock-check'),10000)
        self.assertGreaterEqual(events[start:end].count('actual-remover-gone'),10000)

    def test_resolve_exit_catches_source_metadata_storage_and_tip_changes_before_physical(self):
        for mode in ('source_at_exit','metadata_at_exit','storage_at_exit','tip_at_exit'):
            with self.subTest(mode=mode):
                p=final_probe(mode);self.assertTrue(p['error'])
                self.assertNotIn('complete-history-physical',p['events'])
                self.assertNotIn(p['fixture'].spec['paths']['actual_exit'],[x['path']for x in p['written']])
                self.assertEqual(p['events'][-1],'lock-release')

    def test_failure_path_runs_the_full_source_check_before_scope_cleanup(self):
        p=final_probe('history_failure');self.assertIn('Synthetic History failure',p['error'])
        self.assertNotIn('complete-history-physical',p['events'])
        self.assertIn('source-check',p['events'][p['events'].index('resolve-loop'):])
        p=final_probe('source_on_failure');self.assertIn('Synthetic source changed',p['error'])
        self.assertEqual(p['events'][-1],'lock-release')

    def test_signal_and_original7200_deadline_fail_on_the_current_callback(self):
        for mode in ('signal_during_resolve','deadline_during_resolve'):
            with self.subTest(mode=mode):
                p=final_probe(mode);self.assertIn('Finite independent retirement finalizer',p['error'])
                self.assertNotIn('resolve-return',p['events']);self.assertNotIn('complete-history-physical',p['events'])
                self.assertEqual(p['events'][-1],'lock-release')

    def test_forced_callback_and_slow_scan_do_not_extend_detection_cadence(self):
        for mode in ('forced_resolve_callback','slow_full_check'):
            with self.subTest(mode=mode):
                p=final_probe(mode);self.assertIsNone(p['error']);self.assertEqual(p['resolve_full_checks'],4)

    def test_original_publication_boundary_still_rejects_new_tip(self):
        p=final_probe('tip_after_actual_exit');self.assertTrue(p['error'])
        written=[x['path']for x in p['written']]
        self.assertIn(p['fixture'].spec['paths']['actual_exit'],written)
        self.assertNotIn(p['fixture'].spec['paths']['catalog'],written)
        self.assertNotIn(p['fixture'].spec['paths']['idle'],written)

    def test_four_namespace_successors_preserve_every_other_function(self):
        changes={'minicpm5_retained_retirement_contract_v3':'minicpm5_retained_retirement_contract_v4',
            'minicpm5_retained_retirement_caller_v3':'minicpm5_retained_retirement_caller_v4',
            'minicpm5_retained_retirement_holders_v3':'minicpm5_retained_retirement_holders_v4',
            'minicpm5_retained_retirement_scope_v2':'minicpm5_retained_retirement_scope_v3',
            'minicpm5_retained_retirement_spec_v3':'minicpm5_retained_retirement_spec_v4'}
        for old,new in list(changes.items())[:4]:
            before=(Q/(old+'.py')).read_text()
            for a,b in changes.items():before=before.replace(a,b)
            after=(Q/(new+'.py')).read_text()
            if 'caller_'not in new:self.assertEqual(before,after);continue
            a=ast.parse(before);b=ast.parse(after)
            fa=next(n for n in a.body if isinstance(n,ast.FunctionDef)and n.name=='finalize')
            fb=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name=='finalize')
            # Original guard body is exactly unchanged; only an adapter for
            # resolve and a try/finally entry/exit boundary surround it.
            ga=next(n for n in fa.body if isinstance(n,ast.FunctionDef)and n.name=='guard')
            gb=next(n for n in fb.body if isinstance(n,ast.FunctionDef)and n.name=='guard')
            self.assertEqual(ast.dump(ga,include_attributes=False),ast.dump(gb,include_attributes=False))
            a.body.remove(fa);b.body.remove(fb)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

    def test_helper_ancestry_and_exact_owner_argv_use_the_four_new_namespaces(self):
        import minicpm5_retained_retirement_scope_v3 as scope
        self.assertIs(scope.c,c);self.assertIs(holders.c,c);self.assertIs(caller.c,c)
        self.assertIs(caller.ForegroundScopes,scope.ForegroundScopes)
        self.assertIn('caller_v4.py',c.SOURCES[1]);self.assertIn('holders_v4.py',c.SOURCES[2])
        permit=dict(path='/synthetic/permit.json',sha256='a'*64);nonce='b'*64
        self.assertEqual(scope.command(permit,nonce)[4],c.SOURCES[2])
        self.assertEqual(c.SOURCES[-2:],(str(Q/'minicpm5_retained_raw_transaction_v2.py'),str(Q/'minicpm5_retained_raw_source_io_v1.py')))


if __name__=='__main__':unittest.main(verbosity=2)
