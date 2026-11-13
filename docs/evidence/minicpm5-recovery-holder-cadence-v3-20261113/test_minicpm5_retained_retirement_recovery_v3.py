"""Audited SOURCE ONLY recovery tests; all lifecycle/raw/process IO is synthetic."""
import ast
import builtins
from contextlib import contextmanager,ExitStack
import copy
import hashlib
import io as stringio
import json
from pathlib import Path,PurePosixPath
import stat
import sys
from types import SimpleNamespace as NS,ModuleType
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import minicpm5_retained_retirement_recovery_v3 as r
import minicpm5_retained_retirement_recovery_scope_v3 as scope
import minicpm5_retained_retirement_recovery_holders_v3 as helper
import minicpm5_retained_retirement_caller_v2 as old
import minicpm5_retained_retirement_contract_v2 as c
import test_minicpm5_retained_retirement_caller_v2 as previous
import minicpm5_retained_replay_history_v1 as history_module
import qwen35_package_preflight_serial_v2 as serial
import minicpm5_retained_raw_transport_v1 as transport


def ident(path,payload=b'{}'):
    return dict(realpath=str(path),device=1,inode=2,size=len(payload),mtime_ns=3,ctime_ns=4)


class Fixture:
    def __init__(self):
        f=self.f=previous.Fixture();f.tool['exit_code']=1;f.terminal.update(failed=True,transaction_result=None,
            spec=f.spec_ref,review=f.review_ref,running=f.running_ref,original_lock_identity=c.LOCK)
        self.request_ref=f.ref('synthetic-recovery-request')
        self.request=dict(original_request=f.ref('synthetic-original-finalizer-request'),interruption=f.ref('synthetic-interruption'),
            recovery_review=f.ref('synthetic-recovery-review'),updated_at='synthetic-new-time')
        self.original=dict(spec=f.spec_ref,review=f.review_ref,terminal=f.ref('synthetic-original-terminal'),
            actual_tool=copy.deepcopy(f.tool),updated_at='synthetic-old-time')
        self.original['terminal']['path']=f.spec['paths']['control']+'/terminal.json'
        self.supplement={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in r.EXTRA_SOURCES}
        self.review=dict(schema='minicpm5_retirement_failed_finalizer_recovery_review_v1',reviewer='/root',passed=True,
            blockers=[],original_request=self.request['original_request'],interruption=self.request['interruption'],
            original_spec=f.spec_ref,original_review=f.review_ref,source_sha256=self.supplement,recovery_name='synthetic',
            paths=r.paths('synthetic'),launch_authorized=True,deletion_authorized=False,model_or_raw_replay_authorized=False,
            original_writer_exclusion_retained=True)
        self.failed_owner=dict(pid=2222,ppid=1,start_ticks='22222',cgroup='/synthetic',argv=[c.base.PYTHON,'-B','-u',c.SOURCES[1],
            '--mode','finalize','--request',self.request['original_request']['path'],
            '--request-sha256',self.request['original_request']['sha256']])
        root=f.spec['paths']['control'];self.payloads={root+'/terminal.json':b'{}',root+'/finalizer-helper-drain.json':b'{}'}
        self.tree={root:dict(kind='directory',identity=dict(device=1,inode=10,mode=stat.S_IFDIR|0o700,mtime_ns=3,ctime_ns=4))}
        for p,data in self.payloads.items():self.tree[p]=dict(kind='file',identity=ident(p,data),sha256=hashlib.sha256(data).hexdigest())
        logfile=str(Q/'synthetic-finalizer.log');self.payloads[logfile]=b'failed before publication'
        self.observation=dict(schema='minicpm5_retirement_finalizer_failed_observation_v1',reviewer='/root',
            original_request=self.request['original_request'],failed_finalizer_tool=dict(kind='tool_session',session_id=8888,exit_code=1,terminal_chunk='abc456'),
            failed_finalizer_owner=self.failed_owner,original_finalizer_source=dict(path=c.SOURCES[1],sha256=f.spec['source_sha256'][c.SOURCES[1]]),
            original_lock_identity=c.LOCK,running=f.running_ref,running_identity=ident(f.running_ref['path']),
            control_tree=self.tree,journal_absent=f.spec['paths']['journal'],
            publication_paths_absent=[f.spec['paths'][k]for k in ('actual_exit','catalog','idle')],
            finalizer_helper_drain=f.ref('unused'),finalizer_log=dict(path=logfile,sha256=hashlib.sha256(self.payloads[logfile]).hexdigest(),identity=ident(logfile,self.payloads[logfile])),
            actual_exit_observed=True,termination_cause='synthetic original deadline failure',retirement_deletion_not_resumed=True)
        self.observation['finalizer_helper_drain']['path']=root+'/finalizer-helper-drain.json'
        self.values=f.values
        self.values[self.request_ref['path']]=self.request
        for ref,value in ((self.request['original_request'],self.original),(self.request['interruption'],self.observation),
            (self.request['recovery_review'],self.review),(self.original['terminal'],f.terminal),(f.running_ref,f.running)):
            self.values[ref['path']]=value
        self.values[self.observation['finalizer_helper_drain']['path']]=scope.ForegroundScopes(None,NS(owner=self.failed_owner)).summary()
    def validate(self):return r.validate_observation(self.observation,self.request,self.original,self.f.spec,self.f.terminal,self.f.running)


class PureContracts(unittest.TestCase):
    def test_old_spec_and_source_map_unchanged(self):
        f=Fixture();before=copy.deepcopy(f.f.spec);full=r.validate_review(f.review,f.request,f.original,f.f.spec)
        self.assertEqual(f.f.spec,before);self.assertEqual(set(full)-set(before['source_sha256']),set(r.EXTRA_SOURCES))
        f.validate()
    def test_source_replacement_and_deletion_permission_rejected(self):
        f=Fixture();f.review['deletion_authorized']=True
        with self.assertRaises(ValueError):r.validate_review(f.review,f.request,f.original,f.f.spec)
        f=Fixture();f.f.spec['source_sha256'][r.SELF]='a'*64
        with self.assertRaises(ValueError):r.validate_review(f.review,f.request,f.original,f.f.spec)
    def test_real_failed_tools_must_be_distinct_and_failure_retained(self):
        for change in ('publisher-success','same-tool','remover-success','transaction-result','no-real-exit'):
            f=Fixture()
            if change=='publisher-success':f.observation['failed_finalizer_tool']['exit_code']=0
            if change=='same-tool':f.observation['failed_finalizer_tool']['session_id']=f.original['actual_tool']['session_id']
            if change=='remover-success':f.original['actual_tool']['exit_code']=0
            if change=='transaction-result':f.f.terminal['transaction_result']=f.f.ref('forged-result')
            if change=='no-real-exit':f.observation['actual_exit_observed']=False
            with self.subTest(change=change),self.assertRaises(ValueError):f.validate()
    def test_old_publisher_exact_argv_required(self):
        f=Fixture();f.observation['failed_finalizer_owner']['argv'][3]=r.SELF
        with self.assertRaises(ValueError):f.validate()
    def test_old_finalizer_drain_path_and_all_absent_roles_required(self):
        f=Fixture();f.observation['publication_paths_absent'].reverse()
        with self.assertRaises(ValueError):f.validate()
        f=Fixture();f.observation['finalizer_helper_drain']['path']=f.f.ref('other')['path']
        with self.assertRaises(ValueError):f.validate()
    def test_original_composer_and_rebind_preserve_failed_execution_sources(self):
        f=Fixture();composed=f.f.outputs();old_rows=copy.deepcopy(composed[1][1]['runs'])
        provenance=dict(post_exit_audit_source_sha256=f.supplement,audit_sources_attributed_to_original_remover=False)
        with patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':previous.mini.consumer}):
            objects=r.rebind(composed,f.f.spec,f.review,provenance,f.f.read,lambda:None)
        self.assertEqual(objects[1][1]['runs'][:-1],old_rows[:-1])
        self.assertEqual(objects[1][1]['runs'][-1]['source_sha256'],old_rows[-1]['source_sha256'])
        self.assertFalse(set(r.EXTRA_SOURCES)&set(objects[1][1]['runs'][-1]['source_sha256']))
        self.assertFalse(objects[0][1]['passed']);self.assertEqual(objects[0][1]['actual_tool'],f.original['actual_tool'])
        self.assertEqual(objects[0][1]['reclaimed_bytes'],0)
        for ref,value in objects:f.values[ref['path']]=value
        checked=previous.mini.consumer.serial_catalog(dict(serial_predecessors=objects[1][0],
            source_sha256=dict(f.f.spec['source_sha256'],**f.supplement)),f.f.read,lambda:None)
        self.assertEqual(checked['current_handoff'],objects[2][0])
    def test_only_metadata_no_unlink_observation_covers_all2352(self):
        entries=[];identities={};records=[];calls=[]
        for i in range(2352):
            row=dict(index=i,original=ident('/synthetic/original-'+str(i)),keeper=ident('/synthetic/keeper-'+str(i)))
            entries.append(row);identities.update({row[k]['realpath']:row[k]for k in ('original','keeper')})
            records.append(dict(original_allocated_bytes=2048))
        result=r.no_unlink_state(dict(entries=entries),dict(records=records),lambda:calls.append(1),identities.__getitem__,lambda p:NS(st_blocks=4))
        self.assertEqual(len(calls),2353);self.assertEqual(result['original_allocated_bytes_present'],2352*2048)
        self.assertEqual(result['raw_payload_bytes_read'],result['reclaimed_bytes'],0)
        with self.assertRaisesRegex(ValueError,'allocation changed'):
            r.no_unlink_state(dict(entries=entries),dict(records=records),lambda:None,identities.__getitem__,lambda p:NS(st_blocks=3))
        identities[entries[3]['keeper']['realpath']]=dict(entries[3]['keeper'],inode=999)
        with self.assertRaisesRegex(ValueError,'identity changed'):
            r.no_unlink_state(dict(entries=entries),dict(records=records),lambda:None,identities.__getitem__,lambda p:NS(st_blocks=4))


class PreservedSnapshot(unittest.TestCase):
    def check(self,f,change=None,full=True):
        tree=copy.deepcopy(f.tree);payloads=copy.deepcopy(f.payloads);absent=set()
        if change=='missing':tree.pop(next(p for p,row in tree.items()if row['kind']=='file'))
        if change=='extra':tree[f.f.spec['paths']['control']+'/extra.json']=dict(kind='file',identity=ident('extra'),sha256='a'*64)
        if change=='bytes':payloads[next(p for p in tree if p.endswith('/terminal.json'))]=b'xx'
        if change=='identity':tree[next(p for p in tree if p.endswith('/terminal.json'))]['identity']['inode']=123
        if change=='directory':tree[f.f.spec['paths']['control']]['identity']['mtime_ns']=123
        if change=='journal':absent.add(f.f.spec['paths']['journal'])
        if change=='publication':absent.add(f.f.spec['paths']['catalog'])
        if change=='temporary':absent.add(f.f.spec['paths']['idle']+'.publishing')
        class P(PurePosixPath):
            def resolve(self):return self
            def is_symlink(self):return False
            def lstat(self):
                row=tree[str(self)]
                if row['kind']=='directory':return NS(**{dict(device='st_dev',inode='st_ino',mode='st_mode',mtime_ns='st_mtime_ns',ctime_ns='st_ctime_ns')[k]:v for k,v in row['identity'].items()})
                return NS(st_mode=stat.S_IFREG|0o600)
        @contextmanager
        def scandir(path):
            yield iter(NS(path=p,name=PurePosixPath(p).name)for p in tree if PurePosixPath(p).parent==PurePosixPath(path))
        def identity(path):
            if path in tree:return tree[path]['identity']
            return f.observation['finalizer_log']['identity']
        fake=NS(identity=identity,raw=lambda p,cap:payloads[p])
        with patch.object(r,'Path',P),patch.object(r.os,'scandir',scandir),patch.object(r.os.path,'lexists',side_effect=lambda p:p in absent):
            r.preserved(fake,f.f.spec,f.observation,full)
    def test_entire_saved_namespace_preserved(self):self.check(Fixture())
    def test_every_missing_extra_replaced_changed_old_node_is_rejected(self):
        for change in ('missing','extra','bytes','identity','directory','journal','publication','temporary'):
            with self.subTest(change=change),self.assertRaises((ValueError,KeyError)):self.check(Fixture(),change)


def recovery_probe(mode='ok'):
    """Execute the real recovery entry; every external observation is injected."""
    f=Fixture();b=f.f;p=f.review['paths'];events=[];made=set();written=[];identities={};clock=[100.0]
    latest=[dict(path=b.running_ref['path'],identity=copy.deepcopy(f.observation['running_identity']))]
    old_values=copy.deepcopy(f.values);holder_count=[0];selected_counts=[0];signals={};cadence={};changed=[False];slow=[False]
    selected_active=[None];callback_context=[None];periodic_callbacks=[0];holder_records=[];snapshot_records=[];global_records=[];failed_due=[False]
    targeted=mode.startswith('holders_')
    owner=dict(f.failed_owner,pid=3333,start_ticks='33333',argv=[c.base.PYTHON,'-B','-u',r.SELF,
        '--request',f.request_ref['path'],'--request-sha256',f.request_ref['sha256']])
    for i,row in enumerate(b.plan['entries']):
        row['index']=i
        for side in ('original','keeper'):row[side]=ident(row[side]['realpath'])
    verification=dict(records=[dict(original_allocated_bytes=2048)for _ in b.plan['entries']])
    f.values[c.PLAN['path']]=b.plan;f.values[b.verification['path']]=verification
    selected={row[side]['realpath']:row[side]for row in b.plan['entries']for side in ('original','keeper')}
    extension={path:sha for row in f.values[b.hs['current_serial_catalog']['path']]['runs']for path,sha in row['source_sha256'].items()}
    extension.update({path:b.spec['source_sha256'][path]for path in c.SOURCES})
    def identity(path):
        if path==b.running_ref['path']:return copy.deepcopy(f.observation['running_identity'])
        if path not in identities:identities[path]=dict(realpath=path,device=1,inode=len(identities)+100,size=100,mtime_ns=2,ctime_ns=3)
        return copy.deepcopy(identities[path])
    def read(ref,*args):return copy.deepcopy(f.values[ref['path']])
    def meta_read(ref):return dict(source_extension=extension)if ref==b.hs['history_overlay']else read(ref)
    def source_check(spec):
        events.append('source-check')
        if slow[0]or(mode=='holders_slow_global'and selected_active[0]is not None):clock[0]+=.6
        if changed[0]:raise ValueError('synthetic global source mutation')
    def exercise(callback,force,label):
        if mode not in ('cycles','slow_cycles','physical_due_mutation','physical_exit_mutation','physical_immediate_stop'):return
        if mode=='cycles':
            force();before=events.count('all-storage-budgets')
            for _ in range(100):callback()
            same=events.count('all-storage-budgets')-before
            clock[0]+=.49;callback();early=events.count('all-storage-budgets')-before
            clock[0]+=.02;callback();due=events.count('all-storage-budgets')-before
            cadence[label]=dict(same_time_full=same,before_due_full=early,due_full=due)
        elif mode=='slow_cycles':
            slow[0]=True;force();before=events.count('all-storage-budgets');callback();slow[0]=False
            cadence[label]=dict(after_slow_check_full=events.count('all-storage-budgets')-before)
        elif label=='physical-authority':
            force()
            if mode=='physical_immediate_stop':signals[r.signal.SIGTERM](None,None);callback()
            changed[0]=True
            if mode=='physical_due_mutation':clock[0]+=.51;callback()
            elif mode=='physical_exit_mutation':callback()
    def once(path,value):
        path=str(path);c.require(path not in made,'Synthetic once must be fresh')
        ref=c.base.publication_ref(path,value);made.add(path);f.values[path]=copy.deepcopy(value);written.append(ref)
        events.append('write:'+path)
        if path==p['idle']:latest[0]=dict(path=path,identity=identity(path))
        if mode=='mutate_after_receipt'and path==p['actual_exit']:identities[f.original['terminal']['path']]['inode']+=1
        if mode=='write_catalog_failure'and path==p['catalog']:raise ValueError('synthetic catalog write failure')
        return ref
    class Lock:
        fd=41
        def __init__(self):self.owner=owner
        def acquire(self):events.append('lock-acquire')
        def check(self):events.append('lock-check')
        def release(self,drained):c.require(drained is True,'Only drained release');events.append('lock-release')
    class History:
        def __init__(self,hs,*args):
            self.guard=args[-1]
            def guarded_read(ref):
                self.meta.guard();value=meta_read(ref);self.meta.guard();return value
            self.meta=NS(read=guarded_read,guard=self.guard)
            self.resolved=dict(plan=dict(source_sha256=copy.deepcopy(b.spec['source_sha256'])))
            c.require(hs['source_sha256']==dict(b.spec['source_sha256'],**f.supplement),'Only in-memory audit source union')
        def resolve(self):
            events.append('history-resolve');self.guard()
            exercise(self.guard,lambda:self.guard(True),'resolve')
            if mode=='resolve_failure':raise ValueError('synthetic resolve failure')
            if mode=='deadline':clock[0]+=7201;self.guard()
            if mode=='signal':signals[r.signal.SIGTERM](None,None);self.guard()
            if mode=='resolved_map':self.resolved['plan']['source_sha256'][r.SELF]='b'*64
            return self
        def physical_pair(self,authority):
            events.append('history-physical');authority.check();self.meta.guard();self.guard();authority.check()
            exercise(authority.check,lambda:authority.guard(True),'physical-authority')
            exercise(self.meta.guard,lambda:self.meta.guard(True),'physical-metadata')
            if mode=='physical_failure':raise ValueError('synthetic physical failure')
            return dict(complete=True,full_physical_passes=2)
        def unchanged(self):events.append('history-unchanged')
    class Scopes(scope.ForegroundScopes):
        def drain(self):
            events.append('helper-drain')
            if mode=='drain_failure':raise ValueError('synthetic unresolved helper scope')
            return self.summary()
    def holders(ref,scopes):
        holder_count[0]+=1;events.append('all-uid-holders')
        row=dict(phase=selected_active[0],context=callback_context[0],start=clock[0],end=None,failed=False)
        holder_records.append(row)
        if mode=='holders_due_failure'and selected_active[0]is not None and not failed_due[0]:
            failed_due[0]=True;row.update(end=clock[0],failed=True);raise ValueError('synthetic due holder failure')
        if targeted and selected_active[0]is not None:clock[0]+=.7
        row['end']=clock[0]
        if mode=='holder_failure':raise ValueError('synthetic all UID holder failure')
        if mode=='tip_during_holder':latest[0]=dict(path='/synthetic/concurrent-idle.json',identity={})
        return dict(complete=True,zero_other_holders=True)
    def check_preserved(*args):
        events.append('old-snapshot')
        snapshot_records.append(dict(phase=selected_active[0],full=args[-1]))
        if mode=='old_snapshot_changed'and p['root']in made:raise ValueError('synthetic old snapshot changed')
    def selected_identity(path):
        selected_counts[0]+=1
        if mode=='missing_original'and path==b.plan['entries'][4]['original']['realpath']:raise ValueError('synthetic missing original')
        return copy.deepcopy(selected[path])
    actual_no_unlink=r.no_unlink_state;selected_pass=[0]
    def no_unlink(*args):
        selected_pass[0]+=1;check=args[2];label='selected-pass-'+str(selected_pass[0])
        exercise(check,lambda:check(True),label)
        before=events.count('all-storage-budgets');per_pass=[0]
        def periodic():
            periodic_callbacks[0]+=1;per_pass[0]+=1
            clock[0]+=.01
            explicit=mode=='holders_explicit_force'and per_pass[0]==20
            callback_context[0]='explicit_force'if explicit else 'periodic'
            try:check(True)if explicit else check()
            finally:callback_context[0]=None
        events.append('selected-enter:'+label);selected_active[0]=label
        try:result=actual_no_unlink(*(args[:2]+((periodic if targeted else check),)+args[3:]))
        finally:selected_active[0]=None;events.append('selected-leave:'+label)
        if mode=='cycles':cadence[label]['during_all2352_full']=events.count('all-storage-budgets')-before
        return result
    actual_rebind=r.rebind
    def rebind(*args):
        check=args[-1];exercise(check,lambda:check(True),'composition-consumer')
        before=events.count('all-storage-budgets')
        def consumer_guard(*a):events.append('consumer-callback');return check(*a)
        result=actual_rebind(*args[:-1],consumer_guard)
        if mode=='cycles':cadence['composition-consumer']['during_original_consumer_full']=events.count('all-storage-budgets')-before
        if mode=='consumer_mutation':changed[0]=True
        return result
    def save_history(path,pair):
        events.append('save-history');return [once(Path(path)/'pair.json',pair)]
    def budget(*args):
        events.append('all-storage-budgets');global_records.append(dict(phase=selected_active[0],time=clock[0]))
    io=NS(identity=identity,read=read,once=once,RetainedLock=Lock,loaded_local_sources=lambda pins:events.append('loaded-source-union'),
        process_api=lambda:NS(gone=lambda owner:events.append('owner-gone')))
    class Capture(stringio.StringIO):
        def write(self,value):
            if mode=='telemetry_stdout_failure'and 'mini_retirement_recovery_stage'in value:
                raise BrokenPipeError('synthetic observational stdout failure')
            return super().write(value)
    captured=Capture();result=error=None
    with ExitStack()as stack:
        patches=((old,'api',lambda:io),(old,'source_check',source_check),
            (old,'latest',lambda:copy.deepcopy(latest[0])),(old,'holders',holders),
            (r,'verify_sources',lambda *args:events.append('audit-source-check')),
            (r,'preserved',check_preserved),(r,'old_owners_gone',lambda *args:events.append('all-old-owners-gone')),
            (r,'recovery_storage',budget),(r,'ForegroundScopes',Scopes),
            (history_module,'History',History),(serial,'save_history',save_history),
            (transport,'validate_plan',lambda plan:plan),(transport,'_verification',lambda *args:copy.deepcopy(verification)),
            (transport,'identity',selected_identity),(r,'no_unlink_state',no_unlink),(r,'rebind',rebind))
        for obj,name,value in patches:stack.enter_context(patch.object(obj,name,value))
        stack.enter_context(patch.object(Path,'mkdir',lambda self,*args,**kwargs:made.add(str(self))))
        stack.enter_context(patch.object(Path,'exists',lambda self:str(self)in made))
        stack.enter_context(patch.object(Path,'stat',lambda self:NS(st_blocks=4)))
        stack.enter_context(patch.object(r.os.path,'lexists',lambda path:str(path)in made))
        stack.enter_context(patch.object(r.signal,'signal',lambda sig,fn:signals.__setitem__(sig,fn)))
        stack.enter_context(patch.object(r.time,'monotonic',lambda:clock[0]))
        stack.enter_context(patch.dict(sys.modules,{'research.minicpm5_layer1_operator_sdk_runtime_completion_v4':previous.mini.consumer}))
        stack.enter_context(patch('sys.stdout',new=captured))
        try:result=r.recover(f.request_ref)
        except(ValueError,AssertionError,KeyError)as e:error=str(e)
    return dict(fixture=f,events=events,written=written,result=result,error=error,old_values=old_values,
        selected_identity_calls=selected_counts[0],cadence=cadence,holder_records=holder_records,snapshot_records=snapshot_records,global_records=global_records,periodic_callbacks=periodic_callbacks[0],stdout=[json.loads(line)for line in captured.getvalue().splitlines()])


class FullSyntheticEntry(unittest.TestCase):
    def test_stage_telemetry_has_paired_entries_elapsed_and_full_guard_totals(self):
        probe=recovery_probe('slow_cycles');self.assertIsNone(probe['error'])
        rows=[r for r in probe['stdout']if r['event']=='mini_retirement_recovery_stage']
        self.assertEqual(len(rows),22)
        for left,right in zip(rows[::2],rows[1::2]):
            self.assertEqual(left['stage'],right['stage']);self.assertEqual((left['status'],right['status']),('enter','exit'))
            self.assertIsNone(left['success']);self.assertIs(right['success'],True);self.assertIsNone(right['error_type'])
            self.assertGreaterEqual(right['elapsed_seconds'],left['elapsed_seconds'])
            self.assertGreaterEqual(right['global_guard_count'],left['global_guard_count'])
            self.assertGreaterEqual(right['global_guard_seconds'],left['global_guard_seconds'])
            self.assertEqual(right['telemetry_write_failures'],0)
        self.assertGreater(rows[-1]['global_guard_seconds'],0)
        self.assertEqual(rows[-1]['global_guard_count'],probe['events'].count('all-storage-budgets'))
        self.assertEqual(rows[-1]['stage'],'final_drain');self.assertEqual(probe['events'][-1],'lock-release')
    def test_failure_stage_exit_is_reported_and_finally_still_drains(self):
        probe=recovery_probe('physical_failure');self.assertIn('physical failure',probe['error'])
        rows=[r for r in probe['stdout']if r['event']=='mini_retirement_recovery_stage']
        failures=[r for r in rows if r['success']is False]
        self.assertEqual(len(failures),1);self.assertEqual(failures[0]['stage'],'physical')
        self.assertEqual(failures[0]['error_type'],'ValueError');self.assertEqual(rows[-1]['stage'],'final_drain')
        self.assertIs(rows[-1]['success'],True);self.assertNotIn('write:'+probe['fixture'].review['paths']['idle'],probe['events'])
    def test_observational_stdout_failure_does_not_bypass_publication_checks_or_drain(self):
        probe=recovery_probe('telemetry_stdout_failure');self.assertIsNone(probe['error']);p=probe['fixture'].review['paths']
        self.assertEqual([ref['path']for ref in probe['written']][-4:],[p[k]for k in ('actual_exit','catalog','idle','drain')])
        self.assertEqual(probe['events'][-1],'lock-release')
        self.assertEqual(len(probe['stdout']),1);self.assertEqual(probe['stdout'][0]['event'],'mini_retirement_recovery_live')
    def test_real_recovery_entry_preserves_failed_sources_idle_last_and_zero_reclaim(self):
        probe=recovery_probe();self.assertIsNone(probe['error']);f=probe['fixture'];p=f.review['paths'];events=probe['events']
        self.assertFalse(probe['result']['retirement_passed']);self.assertEqual(probe['result']['reclaimed_bytes'],0)
        self.assertEqual(probe['selected_identity_calls'],2352*3*2)
        output=[ref['path']for ref in probe['written']]
        self.assertEqual(output[-4:],[p[k]for k in ('actual_exit','catalog','idle','drain')])
        for name in ('history-resolve','history-physical','save-history','all-uid-holders'):
            self.assertLess(events.index(name),events.index('write:'+p['actual_exit']))
        self.assertEqual(events[-1],'lock-release')
        receipt=f.values[p['actual_exit']];self.assertEqual(receipt['actual_tool'],f.original['actual_tool'])
        self.assertFalse(receipt['passed']);self.assertIsNone(receipt['transaction_result'])
        catalog=f.values[p['catalog']];old=f.f.values[f.f.hs['current_serial_catalog']['path']]
        self.assertEqual(catalog['runs'][:-1],old['runs'])
        self.assertFalse(set(r.EXTRA_SOURCES)&set(catalog['runs'][-1]['source_sha256']))
        for path,value in probe['old_values'].items():self.assertEqual(f.values[path],value)
    def test_failures_never_publish_idle_and_helpers_drain_before_release(self):
        for mode in ('resolve_failure','physical_failure','holder_failure','tip_during_holder','old_snapshot_changed',
            'missing_original','resolved_map','deadline','signal','mutate_after_receipt','write_catalog_failure','consumer_mutation'):
            with self.subTest(mode=mode):
                probe=recovery_probe(mode);self.assertIsNotNone(probe['error']);p=probe['fixture'].review['paths']
                self.assertNotIn(p['idle'],[ref['path']for ref in probe['written']]);events=probe['events']
                self.assertEqual(events[-1],'lock-release');self.assertIn('helper-drain',events)
                self.assertIn(p['drain'],[ref['path']for ref in probe['written']])
    def test_unresolved_helper_scope_does_not_release_original_lock(self):
        probe=recovery_probe('drain_failure');self.assertIn('unresolved helper scope',probe['error'])
        self.assertNotIn('lock-release',probe['events'])
        self.assertNotIn(probe['fixture'].review['paths']['idle'],[ref['path']for ref in probe['written']])
    def test_explicit_read_only_cycles_cover_real_authority_metadata_path(self):
        probe=recovery_probe('cycles');self.assertIsNone(probe['error']);cadence=probe['cadence']
        self.assertEqual(set(cadence),{'resolve','physical-authority','physical-metadata','selected-pass-1','selected-pass-2','composition-consumer'})
        for label,row in cadence.items():
            with self.subTest(label=label):
                self.assertEqual(row['same_time_full'],0);self.assertEqual(row['before_due_full'],0);self.assertEqual(row['due_full'],1)
                if label.startswith('selected-'):self.assertEqual(row['during_all2352_full'],0)
                if label=='composition-consumer':self.assertEqual(row['during_original_consumer_full'],0)
        self.assertEqual(probe['selected_identity_calls'],2352*3*2)
        events=probe['events'];catalog=probe['fixture'].values[probe['fixture'].review['paths']['catalog']]
        self.assertGreaterEqual(events.count('consumer-callback'),len(catalog['runs']))
        cursor=max(i for i,event in enumerate(events)if event=='consumer-callback')
        for key in ('actual_exit','catalog','idle'):
            end=events.index('write:'+probe['fixture'].review['paths'][key])
            self.assertIn('all-storage-budgets',events[cursor+1:end]);cursor=end
    def test_slow_checks_count_cadence_from_start_and_do_not_defer_next_due(self):
        probe=recovery_probe('slow_cycles');self.assertIsNone(probe['error']);self.assertEqual(len(probe['cadence']),6)
        for row in probe['cadence'].values():self.assertEqual(row['after_slow_check_full'],1)
    def test_authority_mutation_detected_at_due_or_forced_exit_and_stop_immediate(self):
        for mode in ('physical_due_mutation','physical_exit_mutation','physical_immediate_stop'):
            with self.subTest(mode=mode):
                probe=recovery_probe(mode);self.assertIsNotNone(probe['error'])
                self.assertNotIn('save-history',probe['events']);self.assertEqual(probe['events'][-1],'lock-release')


class NamespaceParity(unittest.TestCase):
    def test_recovery_budget_counts_old_reservations_and_new128_mib(self):
        p=r.paths('synthetic-budget');old_totals=dict(control=100,journal=200)
        needed=c.LIMITS['root_floor_bytes']+sum(c.LIMITS[k+'_bytes']-old_totals[k]for k in old_totals)+r.RECOVERY_BYTES
        with patch.object(old,'storage',return_value=old_totals),patch.object(Path,'exists',return_value=False):
            for free in (needed,needed-1):
                with self.subTest(free=free),patch.object(r.os,'statvfs',return_value=NS(f_bavail=free,f_frsize=1)):
                    if free==needed:r.recovery_storage({},p,[])
                    else:
                        with self.assertRaisesRegex(ValueError,'ALL remaining reservations'):r.recovery_storage({},p,[])
    def test_recovery_directory_enumeration_is_bounded_before_statting_unbounded_entries(self):
        p=r.paths('synthetic-budget');visited=[]
        class P(PurePosixPath):
            def exists(self):return str(self)==p['root']
            def lstat(self):
                visited.append(str(self));return NS(st_mode=stat.S_IFDIR|0o700,st_size=4096,st_blocks=8)
            def iterdir(self):
                for i in range(c.LIMITS['control_entries']+3):yield self/('synthetic-'+str(i)+'.json')
        with patch.object(old,'storage',return_value=dict(control=0,journal=0)),patch.object(r,'Path',P):
            with self.assertRaisesRegex(ValueError,'Bounded recovery directory enumeration'):r.recovery_storage({},p,[])
        self.assertEqual(visited,[p['root']])
    def test_original_r12_reader_path_reaches_original_authority_external_guard(self):
        reader_scope=dict(require=c.require,Path=Path,hashlib=hashlib,strict_json=json.loads,
            merge=lambda pins,new:pins.update(new),copy=copy,captured_processes=lambda value:[])
        payload=b'{}';ref=dict(path='/synthetic/terminal.json',sha256=hashlib.sha256(payload).hexdigest())
        reader_scope['terminal']=NS(ref=lambda path:ref)
        tree=ast.parse((Q/'minicpm5_after_nemotron_package_v2.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Reader')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<original Reader source>','exec'),reader_scope)
        class MetadataObserved(Exception):pass
        def metadata(read,guard):reader_scope['Reader'](guard).path(ref['path']);raise MetadataObserved
        physical_scope=dict(require=c.require,validate_spec=lambda *a:None,STORAGE_EXIT=ref,metadata=metadata)
        previous.mini.original_functions(Q/'minicpm5_native_r12_completion_physical_v1.py',{'physical'},physical_scope)
        calls=[];authority=history_module.Authority(NS(names={}),lambda:calls.append('external-global'),lambda:[])
        stable=ModuleType('nemotron9b_package_metadata_v24');stable.stable_bytes=lambda *args:payload
        with patch.dict(sys.modules,{'nemotron9b_package_metadata_v24':stable}),self.assertRaises(MetadataObserved):
            physical_scope['physical'](dict(units={'original':'synthetic-original-unit'}),lambda ref:dict(lock_identity=c.LOCK),None,authority)
        self.assertEqual(len(calls),6)
    def test_scope_is_only_exact_command_target_change(self):
        original=(Q/'minicpm5_retained_retirement_scope_v1.py').read_text()
        actual=(Q/'minicpm5_retained_retirement_recovery_scope_v3.py').read_text()
        expected=original.replace("c.SOURCES[2],'--permit'","str(c.Q/'minicpm5_retained_retirement_recovery_holders_v3.py'),'--permit'")
        self.assertEqual(actual,expected)
        nonce='b'*64;permit=dict(path=str(Q/'synthetic-permit.json'),sha256='a'*64)
        self.assertEqual(scope.command(permit,nonce)[4],r.HELPER)
    def test_holder_scan_body_exactly_preserved(self):
        trees=[ast.parse((Q/p).read_text())for p in ('minicpm5_retained_retirement_holders_v2.py','minicpm5_retained_retirement_recovery_holders_v3.py')]
        bodies=[next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inspect')for tree in trees]
        self.assertEqual(ast.dump(bodies[0]),ast.dump(bodies[1]))
    def test_no_delete_model_or_raw_replay_calls_and_original_predicates_called(self):
        tree=ast.parse(Path(r.SELF).read_text());calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        for name in ('old.load','old.prerequisites','c.completed_outputs'):
            self.assertEqual(calls.count(name),1)
        self.assertFalse(any(name.endswith(('.unlink','.rmtree','.retire','.Replay','.replay'))for name in calls))
    def test_new_helper_permit_binds_exact_new_owner(self):
        f=Fixture();p=f.review['paths'];owner=dict(f.failed_owner,argv=[c.base.PYTHON,'-B','-u',r.SELF,
            '--request',f.request_ref['path'],'--request-sha256',f.request_ref['sha256']])
        permit=dict(schema='minicpm5_retirement_recovery_holder_permit_v1',mode='recovery_finalize',
            spec=f.f.spec_ref,review=f.f.review_ref,recovery_request=f.request_ref,recovery_review=f.request['recovery_review'],
            owner=owner,retained_fd=9,lock_identity=c.LOCK)
        fake=NS(read=lambda ref,*a:copy.deepcopy(f.values[ref['path']]),loaded_local_sources=lambda pins:None)
        with patch.object(r,'verify_sources'),patch.object(old,'source_check'):
            r.validate_helper_permit(fake,dict(path=p['permit'],sha256='a'*64),permit,f.f.spec)
            permit['owner']['argv'][3]=c.SOURCES[1]
            with self.assertRaises(ValueError):r.validate_helper_permit(fake,dict(path=p['permit'],sha256='a'*64),permit,f.f.spec)


class HolderCadenceRegression(unittest.TestCase):
    """Only these new related regressions run by default; frozen v2 suite is not repeated."""
    def successful(self,mode):
        probe=recovery_probe(mode);self.assertIsNone(probe['error'])
        self.assertEqual(probe['periodic_callbacks'],2353*2)
        self.assertEqual(probe['selected_identity_calls'],2352*3*2)
        self.assertEqual(probe['events'][-1],'lock-release')
        for row in probe['snapshot_records']:
            if row['phase']is not None:self.assertIs(row['full'],True)
        return probe
    def check_due(self,probe):
        previous=None;periodic=0;forced=0
        for row in probe['holder_records']:
            if row['phase']is not None:
                self.assertIsNotNone(previous)
                if row['context']=='periodic':
                    periodic+=1;self.assertGreaterEqual(row['start']-previous['end'],10-1e-8)
                elif row['context']=='explicit_force':
                    forced+=1;self.assertLess(row['start']-previous['end'],10)
                else:self.fail('Unclassified holder call inside selected loop')
            previous=row
        return periodic,forced
    def test_many_row_callbacks_keep_periodic_holders_due_and_global_full_hashes(self):
        probe=self.successful('holders_many_rows');periodic,forced=self.check_due(probe)
        self.assertGreater(periodic,0);self.assertEqual(forced,0);self.assertLess(periodic,10)
        global_count=sum(row['phase']is not None for row in probe['global_records'])
        self.assertGreater(global_count,periodic*10)
    def test_slow_global_checks_do_not_force_holders_on_every_row(self):
        probe=self.successful('holders_slow_global');periodic,forced=self.check_due(probe)
        global_count=sum(row['phase']is not None for row in probe['global_records'])
        self.assertGreater(global_count,2353*2-100)
        self.assertLessEqual(global_count,2353*2)
        self.assertGreater(periodic,0);self.assertLess(periodic,global_count//5);self.assertEqual(forced,0)
    def test_explicit_force_inside_both_readonly_passes_still_scans_immediately(self):
        probe=self.successful('holders_explicit_force');periodic,forced=self.check_due(probe)
        self.assertGreater(periodic,0);self.assertEqual(forced,2)
    def test_every_stage_boundary_and_actual_publication_keeps_forced_scan(self):
        probe=self.successful('holders_many_rows');events=probe['events'];p=probe['fixture'].review['paths']
        last=events.index('write:'+p['permit'])
        for label in ('selected-pass-1','selected-pass-2'):
            begin=events.index('selected-enter:'+label);end=events.index('selected-leave:'+label)
            self.assertIn('all-uid-holders',events[last+1:begin])
            next_boundary=events.index('selected-enter:selected-pass-2')if label=='selected-pass-1'else events.index('consumer-callback')
            self.assertIn('all-uid-holders',events[end+1:next_boundary]);last=end
        cursor=max(i for i,event in enumerate(events)if event=='consumer-callback')
        for key in ('actual_exit','catalog','idle'):
            end=events.index('write:'+p[key]);self.assertIn('all-uid-holders',events[cursor+1:end]);cursor=end
        self.assertEqual([ref['path']for ref in probe['written']][-4:],[p[k]for k in ('actual_exit','catalog','idle','drain')])
    def test_due_holder_failure_forces_exceptional_exit_then_drains_without_idle(self):
        probe=recovery_probe('holders_due_failure');self.assertIn('due holder failure',probe['error'])
        events=probe['events'];p=probe['fixture'].review['paths'];end=events.index('selected-leave:selected-pass-1')
        self.assertIn('all-uid-holders',events[end+1:]);self.assertEqual(events[-1],'lock-release')
        self.assertNotIn('write:'+p['idle'],events)
        failures=[row for row in probe['holder_records']if row['failed']]
        self.assertEqual(len(failures),1);self.assertEqual(failures[0]['context'],'periodic')
        stages=[row for row in probe['stdout']if row['event']=='mini_retirement_recovery_stage']
        self.assertTrue(any(row['stage']=='selected_metadata_first'and row['success']is False for row in stages))
        self.assertEqual(stages[-1]['stage'],'final_drain');self.assertIs(stages[-1]['success'],True)
    def test_minimal_source_delta_preserves_every_other_predicate_and_callback_rule(self):
        old=(Q/'minicpm5_retained_retirement_recovery_v2.py').read_text()
        for name in ('minicpm5_retained_retirement_recovery','minicpm5_retained_retirement_recovery_holders','minicpm5_retained_retirement_recovery_scope'):
            old=old.replace(name+'_v2',name+'_v3')
        new=Path(r.SELF).read_text()
        new=new.replace('def full_guard(force=False,*,force_holders=None):\n        if force_holders is None:force_holders=force', 'def full_guard(force=False):')
        new=new.replace('and(force_holders or time.monotonic()-last_scan[0]>=10)', 'and(force or time.monotonic()-last_scan[0]>=10)')
        new=new.replace('def guard(force=False,*,force_holders=None):','def guard(force=False):')
        new=new.replace('full_guard(force,force_holders=force_holders)','full_guard(force)')
        new=new.replace('guard(True,force_holders=force)','guard(True)')
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse(new)))
        original_scope=(Q/'minicpm5_retained_retirement_recovery_scope_v2.py').read_text()
        self.assertEqual(original_scope.replace('recovery_holders_v2','recovery_holders_v3'),(Q/'minicpm5_retained_retirement_recovery_scope_v3.py').read_text())
        original_helper=(Q/'minicpm5_retained_retirement_recovery_holders_v2.py').read_text()
        self.assertEqual(original_helper.replace('recovery_scope_v2','recovery_scope_v3').replace('recovery_v2','recovery_v3'),(Q/'minicpm5_retained_retirement_recovery_holders_v3.py').read_text())


if __name__=='__main__':
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0]in ('torch','torch_neuronx','torch_xla','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('No native imports in source-only tests')
        return original_import(name,*args,**kwargs)
    builtins.__import__=guarded_import
    def audit(event,args):
        if event in ('subprocess.Popen','os.system','socket.connect','socket.bind','os.listdir','os.scandir','fcntl.flock'):
            raise AssertionError('No actual process/namespace/network/lock operation')
        if event=='open':
            name,mode,flags=args
            if not isinstance(name,str)or not name.endswith(('.py','.pyc')):
                raise AssertionError('No actual metadata/model/raw file read')
            if flags&(1|2|64|512):raise AssertionError('No actual file mutation')
    sys.addaudithook(audit)
    unittest.main(defaultTest='HolderCadenceRegression',verbosity=2)
