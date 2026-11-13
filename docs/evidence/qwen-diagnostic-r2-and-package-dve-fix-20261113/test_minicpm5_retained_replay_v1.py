"""Source/AST and /tmp fixtures only; never exercise real History or services."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(Q),str(ROOT)]
import minicpm5_retained_replay_contract_v1 as c
import minicpm5_retained_replay_caller_v1 as caller
import minicpm5_retained_replay_history_v1 as history
import minicpm5_retained_replay_observer_v1 as observer
import qwen35_sampler_runtime_io_v12 as io
import minicpm5_history_recovery_lifecycle_v3 as life
from qwen35_package_preflight_history_v3 import History as OriginalHistory


def ref(name):return dict(path=str(Q/(name+'.json')),sha256='a'*64)

def binding():
    # Immutable historical JSON/source inputs only; no current refs are chosen.
    data=Path(c.ORIGINAL_PLAN['path']).read_bytes();original=json.loads(data)
    pins=dict(original['source_sha256'],**c.FROZEN);pins[c.ACCEPTED['path']]=c.ACCEPTED['sha256']
    ctx=dict(run_name='synthetic-tests-only',history_overlay=ref('synthetic-overlay'),
        current_serial_catalog=ref('synthetic-catalog'),lfm_r2_scope=ref('synthetic-lfm'),
        inherited_source_sha256=pins,input_extension={},exact_envelopes={})
    sources={p:Path(p).read_bytes()for p in c.FILES}
    return ctx,sources,data


def execution():
    a_ref=ref('synthetic-admission');w_ref=ref('synthetic-worker')
    admission=dict(schema='minicpm5_retained_replay_outer_admission_v1',run_name='synthetic-tests-only',
        lock_identity=c.LOCK,attempts=dict(pair='1'*32,replay='2'*32),outer_owner=dict(pid=123,start_ticks='456'),
        current_serial_catalog=ref('synthetic-catalog'))
    def worker(phase):
        return dict(unit=c.names(admission['run_name'])[phase],nonce=admission['attempts'][phase],
            process=dict(pid=789,start_ticks='1011',argv=c.phase_argv(phase,a_ref)),owner=dict(pid=789,start_ticks='1011'))
    previous=dict(serial_execution=True,live_tool_sessions=[],active_heavy_units=[],
        original_failed_model_exit_code=1,all_owners=[dict(pid=99,start_ticks='8')],arbitrary_nested=dict(retained=[1,2,3]))
    tool=dict(kind='tool_session',session_id=111)
    running=c.running_value(previous,'pair',tool,a_ref,admission,worker('pair'),w_ref,'synthetic-time')
    replay=c.running_value(running,'replay',tool,a_ref,admission,worker('replay'),w_ref,'synthetic-time')
    return admission,a_ref,w_ref,previous,running,replay,worker


class PureBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=binding()
    def render(self):return c.render_binding(*copy.deepcopy(self.fixture))
    def test_pure_render_performs_no_io(self):
        with mock.patch('builtins.open',side_effect=AssertionError('No IO')),mock.patch.object(Path,'open',side_effect=AssertionError('No IO')):
            value=self.render()
        self.assertFalse(value['runtime_enrolled']);self.assertFalse(value['current_references_selected_here'])
        self.assertEqual(c.sha(value['spec_bytes']),value['spec_sha256'])
    def test_full_original_sources_and_pins_retained(self):
        value=self.render()['spec'];original=json.loads(self.fixture[2]);self.assertEqual(len(original['source_sha256']),1869)
        for p,h in original['source_sha256'].items():self.assertEqual(value['source_sha256'][p],h)
        self.assertEqual(value['lock_identity'],dict(device=66306,inode=8294421))
        self.assertFalse(value['deletion_authorized']);self.assertEqual(value['storage_plan'],c.PLAN)
    def test_reject_missing_original_source(self):
        ctx,sources,data=copy.deepcopy(self.fixture);ctx['inherited_source_sha256'].pop(next(iter(json.loads(data)['source_sha256'])))
        with self.assertRaises(ValueError):c.render_binding(ctx,sources,data)
    def test_reject_replaced_accepted_source(self):
        ctx,sources,data=copy.deepcopy(self.fixture);ctx['inherited_source_sha256'][next(iter(c.FROZEN))]='0'*64
        with self.assertRaises(ValueError):c.render_binding(ctx,sources,data)
    def test_reject_original_plan_change(self):
        ctx,sources,data=self.fixture
        with self.assertRaises(ValueError):c.render_binding(ctx,sources,data+b' ')
    def test_reject_new_source_replacement(self):
        ctx,sources,data=copy.deepcopy(self.fixture);ctx['inherited_source_sha256'][c.FILES[0]]='0'*64
        with self.assertRaises(ValueError):c.render_binding(ctx,sources,data)
    def test_reject_unknown_phase_or_schema_field(self):
        with self.assertRaises(ValueError):c.phase_argv('retire',ref('synthetic-admission'))
        spec=self.render()['spec'];spec['cleanup']=True
        with self.assertRaises(ValueError):c.validate_spec(spec)
    def test_namespaces_resources_and_no_delete(self):
        spec=self.render()['spec'];self.assertNotEqual(spec['output_root'],spec['replay_root'])
        self.assertEqual(c.PROPERTIES['MemoryMax'],str(6<<30));self.assertEqual(c.PROPERTIES['TasksMax'],'32')
        self.assertEqual(c.PROPERTIES['AllowedCPUs'],'7');self.assertEqual(c.PROPERTIES['Nice'],'19')
        self.assertEqual(c.PROPERTIES['MemorySwapMax'],'0');self.assertFalse(spec['automatic_successor'])
    def test_exact_large_envelope_only(self):
        ctx,sources,data=copy.deepcopy(self.fixture);p=str(Q/'synthetic-envelope.json')
        ctx['exact_envelopes'][p]=dict(sha256='b'*64,bytes=(8<<20)+1,role='complete_json_metadata_envelope')
        ctx['inherited_source_sha256'][p]='b'*64
        value=c.render_binding(ctx,sources,data);self.assertEqual(value['spec']['exact_envelopes'],ctx['exact_envelopes'])
        ctx['exact_envelopes'][p]['bytes']=65<<20
        with self.assertRaises(ValueError):c.render_binding(ctx,sources,data)
    def test_envelope_must_be_exact_pinned_json(self):
        ctx,sources,data=copy.deepcopy(self.fixture);p=str(Q/'synthetic-envelope.raw')
        ctx['exact_envelopes'][p]=dict(sha256='b'*64,bytes=(8<<20)+1,role='complete_json_metadata_envelope')
        ctx['inherited_source_sha256'][p]='b'*64
        with self.assertRaises(ValueError):c.render_binding(ctx,sources,data)


class HistorySourceTests(unittest.TestCase):
    def test_original_resolve_ast_only_two_reader_substitutions(self):
        def method(path):
            tree=ast.parse(Path(path).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='History')
            return next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='resolve')
        before=method(Q/'qwen35_package_preflight_history_v3.py');after=method(Q/'minicpm5_retained_replay_history_v1.py')
        class Substitute(ast.NodeTransformer):
            def visit_Call(self,node):
                node=self.generic_visit(node)
                if ast.unparse(node.func)=='sources.text':
                    node.func=ast.parse('self.text',mode='eval').body;node.args.insert(0,ast.Name(id='sources',ctx=ast.Load()))
                if ast.unparse(node.func)=='source_text.verify_text_sources':
                    node.func=ast.parse('self.verify_text',mode='eval').body
                    node.args=[ast.Name(id='sources',ctx=ast.Load()),node.args[0]]
                return node
        self.assertEqual(ast.dump(Substitute().visit(before),include_attributes=False),ast.dump(after,include_attributes=False))
    def test_original_physical_and_freshness_are_inherited(self):
        for name in ('physical_pair','freshness','unchanged'):
            self.assertIs(getattr(history.History,name),getattr(OriginalHistory,name))
    def test_only_explicit_envelope_dispatches(self):
        h=history.History.__new__(history.History);h.meta=SimpleNamespace(envelopes={'large':{}},envelope=mock.Mock(return_value=b'large'))
        with mock.patch.object(history.source_text,'text_bytes',return_value=b'ordinary')as ordinary:
            self.assertEqual(h.text('sources','normal','sha'),b'ordinary');ordinary.assert_called_once_with('sources','normal','sha')
            self.assertEqual(h.text('sources','large','sha'),b'large');h.meta.envelope.assert_called_once_with('large','sha')
    def test_complete_envelope_hash_and_identity_required(self):
        data=b'{"preserve": [1,2,3]}';h=hashlib.sha256(data).hexdigest();p='/tmp/synthetic-envelope.json'
        meta=history.Metadata(lambda:None,{p:dict(sha256=h,bytes=len(data))});identity=dict(size=len(data))
        with mock.patch.object(history,'identity',return_value=identity),mock.patch.object(history,'raw',return_value=data):
            self.assertEqual(meta.read(dict(path=p,sha256=h)),json.loads(data))
            with self.assertRaises(ValueError):meta.read(dict(path=p,sha256='b'*64))
        with mock.patch.object(history,'identity',side_effect=[identity,dict(size=len(data),changed=True)]),mock.patch.object(history,'raw',return_value=data):
            with self.assertRaises(ValueError):meta.read(dict(path=p,sha256=h))
    def test_no_retirement_or_native_entry(self):
        for filename in c.FILES:
            tree=ast.parse(Path(filename).read_text());names=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
            self.assertFalse(any('unlink' in n or 'retire' in n or n.endswith('.run_recovery')for n in names))
            imports=[n.name for node in ast.walk(tree)if isinstance(node,ast.Import)for n in node.names]
            self.assertFalse({'torch','numpy','transformers','torch_neuronx','vllm'}&set(imports))
    def test_worker_release_precedes_payload_and_json_writer_is_compatible(self):
        tree=ast.parse(Path(c.FILES[2]).read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
        calls=[(n.lineno,ast.unparse(n.func))for n in ast.walk(worker)if isinstance(n,ast.Call)]
        release=next(line for line,name in calls if name=='read_running')
        for name in ('transport.verify_pairs','legacy.Payloads','legacy.run_replay'):
            self.assertGreater(next(line for line,n in calls if n==name),release)
        body=ast.unparse(worker)
        self.assertIn("Path(spec['replay_root']) / 'caller-result.json'",body)
        self.assertIn("transport._write_once if phase == 'replay' else io.once",body)
        self.assertIn('legacy.run_replay(inputs, spec[\'replay_root\'], guard)',body)


class IsolatedAuditTests(unittest.TestCase):
    def test_imports_and_help_cannot_perform_runtime_operations(self):
        code=r'''
import os,sys,runpy
from pathlib import Path
q=Path(sys.argv[1]);filename=sys.argv[2];sys.path[:0]=[str(q),str(q.parent)]
def audit(event,args):
    assert event not in {'subprocess.Popen','os.system','os.fork','os.posix_spawn','fcntl.flock','socket.connect','socket.bind','os.mkdir','os.remove'},event
    if event=='open'and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0]);mode=args[1];flags=args[2]
        assert not path.startswith(('/proc/','/sys/','/mnt/nvme/')),path
        assert Path(path).suffix not in ('.pt','.raw','.safetensors','.neff','.bin'),path
        assert not(isinstance(mode,str)and any(k in mode for k in 'wax+')or isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)),path
sys.addaudithook(audit)
for suffix in ('contract','history','caller','observer'):__import__('minicpm5_retained_replay_'+suffix+'_v1')
assert not {'torch','numpy','transformers','torch_neuronx','vllm'}&set(sys.modules)
sys.argv=[str(q/filename),'--help']
try:runpy.run_path(str(q/filename),run_name='__main__')
except SystemExit as e:assert e.code==0
print('SAFE_SOURCE_IMPORT_AND_HELP')
'''
        for name in (c.FILES[2],c.FILES[3]):
            result=subprocess.run(['/usr/bin/python3','-B','-c',code,str(Q),Path(name).name],text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('SAFE_SOURCE_IMPORT_AND_HELP',result.stdout)
    def test_accepted_json_writer_survives_exact_frozen_runtime_audit(self):
        code=r'''
import ast,os,sys,json
from pathlib import Path
from types import SimpleNamespace
q=Path(sys.argv[1]);root=Path(sys.argv[2]);sys.path[:0]=[str(q),str(q.parent)]
import minicpm5_retained_raw_transport_v1 as transport
import qwen35_sampler_runtime_io_v12 as io
tree=ast.parse((q/'minicpm5_retained_raw_legacy_v1.py').read_text())
node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='runtime_audit')
scope=dict(Path=Path,os=os,self=SimpleNamespace(DEST=root,FORBIDDEN_IMPORTS={'transformers','torch_neuronx'}),allowed=set(),original={})
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<exact frozen runtime_audit>','exec'),scope)
sys.addaudithook(scope['runtime_audit'])
result=transport._write_once(root/'caller-result.json',dict(complete=True))
assert json.loads((root/'caller-result.json').read_bytes())==dict(complete=True)
try:io.once(root/'forbidden-publisher.json',{})
except AssertionError:pass
else:raise AssertionError('The original audit must reject the .publishing temporary')
print('EXACT_FROZEN_AUDIT_COMPATIBLE')
'''
        with tempfile.TemporaryDirectory(prefix='mini-retained-audit-',dir='/tmp')as root:
            result=subprocess.run(['/usr/bin/python3','-B','-c',code,str(Q),root],text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('EXACT_FROZEN_AUDIT_COMPATIBLE',result.stdout)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='mini-retained-caller-',dir='/tmp');self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.control=root/'control';self.replay=root/'replay';self.control.mkdir();self.replay.mkdir()
        self.spec=dict(output_root=str(self.control),replay_root=str(self.replay))
        self.patch=mock.patch.object(caller,'api',return_value=(io,None));self.patch.start();self.addCleanup(self.patch.stop)
    def test_scan_exact_json_and_log(self):
        (self.control/'receipt.json').write_text('{}');(self.replay/'trace.log').write_text('ok')
        rows,total=caller.scan_storage(self.spec);self.assertGreater(total,0);self.assertEqual(rows['output_root']['entries'],2)
    def test_known_exclusive_publication_retries_then_settles(self):
        temporary=self.control/'receipt.json.publishing';final=self.control/'receipt.json';temporary.write_text('{}');os.link(temporary,final)
        with self.assertRaises(io.PublicationRace):caller.scan_storage(self.spec)
        temporary.unlink();self.assertEqual(caller.scan_storage(self.spec)[0]['output_root']['entries'],2)
    def test_unknown_hardlinks_are_rejected(self):
        p=self.control/'a.json';p.write_text('{}');os.link(p,self.control/'b.json')
        with self.assertRaises(ValueError):caller.scan_storage(self.spec)
    def test_unrelated_disappearance_is_not_ignored(self):
        with mock.patch.object(io,'observed_stat',side_effect=FileNotFoundError('ordinary file disappeared')):
            with self.assertRaises(FileNotFoundError):caller.scan_storage(self.spec)
    def test_only_known_temporary_race_gets_bounded_retries(self):
        fake=SimpleNamespace(f_bavail=1000<<30,f_frsize=1)
        with mock.patch.object(caller,'scan_storage',side_effect=[io.PublicationRace(),({},0)]),mock.patch.object(caller.os,'statvfs',return_value=fake),mock.patch.object(caller.time,'sleep')as sleep:
            self.assertEqual(caller.storage(self.spec),{});sleep.assert_called_once_with(.01)
        with mock.patch.object(caller,'scan_storage',side_effect=io.PublicationRace()),mock.patch.object(caller.time,'sleep')as sleep:
            with self.assertRaises(ValueError):caller.storage(self.spec)
            self.assertEqual(sleep.call_count,7)
    def test_log_cap_is_lower_than_json_cap(self):
        with (self.control/'large.log').open('wb')as f:f.truncate(c.LIMITS['log_bytes']+1)
        with self.assertRaises(ValueError):caller.scan_storage(self.spec)
    def test_file_and_root_budget_are_separate(self):
        limits=dict(c.LIMITS,control_bytes=100000,file_bytes=200000)
        for name in ('a.json','b.json'):
            with (self.control/name).open('wb')as f:f.truncate(70000)
        with mock.patch.object(c,'LIMITS',limits):
            with self.assertRaises(ValueError):caller.scan_storage(self.spec)
    def test_symlink_and_arbitrary_publishing_suffix_rejected(self):
        p=self.control/'foreign.publishing';p.write_text('x')
        with self.assertRaises(ValueError):caller.scan_storage(self.spec)
        p.unlink();p.symlink_to(self.replay)
        with self.assertRaises(ValueError):caller.scan_storage(self.spec)


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.spec=c.render_binding(*binding())['spec'];self.stage=caller.Phase(self.spec,'pair',ref('synthetic-admission'),'1'*32)
        self.proc=dict(pid=123,start_ticks='55',argv=self.stage.argv,cgroup='0::'+life.group_name(self.stage.unit))
        self.owner=dict(pid=123,start_ticks='55',unit=self.stage.unit,script=self.stage.argv[3])
        self.props=dict(self.stage.props,Id=life.unit_name(self.stage.unit),LoadState='loaded',MainPID='123',ControlPID='0',
            ActiveState='active',SubState='running',ControlGroup=life.group_name(self.stage.unit),Job='',InvocationID='2'*32,
            Environment=' '.join(k+'='+__import__('shlex').quote(v)for k,v in dict(self.stage.env,MINI_RECOVERY_ATTEMPT='1'*32).items()))
        self.props['typed_exec']=dict(unit=life.unit_name(self.stage.unit),object_path=life.object_name(self.stage.unit),
            properties={k:[[self.stage.argv[0],self.stage.argv,False,0,0,0,0,123,0,0]]if k=='ExecStart'else []for k in life.EXEC_KEYS})
        self.fake=SimpleNamespace(sample=lambda unit:copy.deepcopy(self.props),properties=lambda unit:copy.deepcopy(self.props),
            contract=life.contract,parse_properties=life.parse_properties,group_name=life.group_name,
            authenticate=self.authenticate,start=lambda pid:None,empty_group=lambda path:dict(empty=True),
            drained=mock.Mock(return_value=[dict(empty=True)]),with_attempt=life.with_attempt)
        self.io=SimpleNamespace(process_api=lambda:SimpleNamespace(proc=lambda pid:copy.deepcopy(self.proc)))
        self.patches=[mock.patch.object(caller,'api',return_value=(self.io,self.fake)),mock.patch.object(caller.subprocess,'check_output',side_effect=self.show)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
    def authenticate(self,state,value,unit,argv,cleanup,nonce):
        life.contract(value,unit,argv,cleanup,nonce)
        if state.get('owner'):assert state['owner']==self.owner
        state['owner']=copy.deepcopy(self.owner)
    def show(self,command,**kwargs):
        if 'RemainAfterExit' in command[-1]:return 'RemainAfterExit=yes\nLimitFSIZE=33554432\nRuntimeMaxUSec=2h\nTimeoutStopUSec=1min\n'
        p=dict(Id=life.unit_name(self.stage.unit),InvocationID=self.props['InvocationID'],ExecMainPID='123',ExecMainCode='1',ExecMainStatus='7',
            Result='exit-code',MainPID='0',ControlPID='0',ActiveState='failed',SubState='failed',Job='')
        return ''.join(k+'='+v+'\n'for k,v in p.items())
    def test_owner_nonce_and_invocation_are_retained(self):
        self.stage.sample();self.assertEqual(self.stage.owner['process'],self.proc)
        self.props['InvocationID']='3'*32
        with self.assertRaises(ValueError):self.stage.sample()
    def test_nonce_replacement_rejected(self):
        self.props['Environment']=self.props['Environment'].replace('MINI_RECOVERY_ATTEMPT='+'1'*32,'MINI_RECOVERY_ATTEMPT='+'3'*32)
        with self.assertRaises(AssertionError):self.stage.sample()
    def test_resource_and_environment_change_rejected(self):
        self.props['MemoryMax']='1'
        with self.assertRaises(ValueError):self.stage.sample()
        self.props['MemoryMax']=self.stage.props['MemoryMax'];self.props['Environment']=self.props['Environment'].replace('OMP_NUM_THREADS=1','OMP_NUM_THREADS=2')
        with self.assertRaises(ValueError):self.stage.sample()
    def test_actual_nonzero_terminal_is_retained(self):
        self.stage.sample();self.props.update(MainPID='0',ActiveState='failed',SubState='failed')
        value=self.stage.terminal();self.assertEqual(value['properties']['ExecMainStatus'],'7');self.assertTrue(value['actual_service_exit_observed'])
    def test_unobserved_worker_never_claims_success(self):
        self.props.update(MainPID='0',ActiveState='active',SubState='exited')
        with self.assertRaises(ValueError):self.stage.terminal()
        self.stage.started=True
        with mock.patch.object(caller.subprocess,'run',return_value=None):
            value=self.stage.cleanup();self.assertTrue(value['complete']);self.assertIsNone(value['recorded_exit'])
    def test_unknown_invocation_is_never_stopped(self):
        self.stage.sample();self.stage.started=True;self.props['InvocationID']='3'*32
        with mock.patch.object(caller.subprocess,'run')as stop:
            with self.assertRaises(ValueError):self.stage.cleanup()
            stop.assert_not_called()
    def test_successful_cleanup_can_be_reobserved_after_unit_unloaded(self):
        self.stage.sample();self.stage.started=True;self.props.update(MainPID='0',ActiveState='failed',SubState='failed')
        with mock.patch.object(caller.subprocess,'run',return_value=None):saved=self.stage.cleanup()
        self.props['LoadState']='not-found';self.assertEqual(self.stage.cleanup(),saved)
    def test_observed_disappearance_before_terminal_not_accepted(self):
        self.stage.sample();self.stage.started=True;self.props['LoadState']='not-found'
        with self.assertRaises(ValueError):self.stage.cleanup()
    def test_never_loaded_install_failure_can_drain(self):
        self.stage.started=True;self.props['LoadState']='not-found';self.assertTrue(self.stage.cleanup()['never_loaded'])


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.admission,self.a_ref,self.w_ref,self.previous,self.pair,self.running,self.worker=execution()
        self.actual=dict(kind='tool_session',session_id=111,exit_code=0,terminal_chunk='synthetic-terminal')
        self.terminal=dict(schema='minicpm5_retained_replay_outer_terminal_v1',self_exit_claimed=False,
            actual_external_exit_pending=True,full_owned_scope_drained=True,admission=self.a_ref,outer_owner=self.admission['outer_owner'],
            failed=False,phase_states=[dict(phase=p,owner=self.worker(p),exit=dict(actual_service_exit_observed=True,
                owner=self.worker(p),properties=dict(ExecMainCode='1',ExecMainStatus='0',Result='success')))for p in ('pair','replay')])
    def test_complete_handoff_preserved_across_both_phase_acks(self):
        for k in ('all_owners','original_failed_model_exit_code','arbitrary_nested'):self.assertEqual(self.previous[k],self.running[k])
        self.assertEqual(self.previous['live_tool_sessions'],[])
        with self.assertRaises(ValueError):c.running_value(self.pair,'replay',dict(kind='tool_session',session_id=112),self.a_ref,self.admission,self.worker('replay'),self.w_ref,'time')
    def test_idle_and_real_worker_required(self):
        previous=copy.deepcopy(self.previous);previous['live_tool_sessions']=[999]
        with self.assertRaises(ValueError):c.running_value(previous,'pair',dict(kind='tool_session',session_id=111),self.a_ref,self.admission,self.worker('pair'),self.w_ref,'time')
        worker=self.worker('pair');worker['nonce']='9'*32
        with self.assertRaises(ValueError):c.running_value(self.previous,'pair',dict(kind='tool_session',session_id=111),self.a_ref,self.admission,worker,self.w_ref,'time')
    def test_actual_outer_session_must_match(self):
        self.assertTrue(observer.reconcile(self.terminal,self.running,self.actual)['passed'])
        actual=dict(self.actual,session_id=112)
        with self.assertRaises(ValueError):observer.reconcile(self.terminal,self.running,actual)
    def test_nonzero_outer_failure_is_preserved(self):
        terminal=copy.deepcopy(self.terminal);terminal['failed']=True
        actual=dict(self.actual,exit_code=1);value=observer.reconcile(terminal,self.running,actual)
        self.assertFalse(value['passed']);self.assertTrue(value['original_failure_preserved'])
        with self.assertRaises(ValueError):observer.reconcile(terminal,self.running,self.actual)
    def test_both_real_successful_phase_exits_required(self):
        terminal=copy.deepcopy(self.terminal);terminal['phase_states'][1]['exit']['properties']['ExecMainStatus']='7'
        with self.assertRaises(ValueError):observer.reconcile(terminal,self.running,self.actual)
        terminal=copy.deepcopy(self.terminal);terminal['phase_states'].pop()
        with self.assertRaises(ValueError):observer.reconcile(terminal,self.running,self.actual)
    def final_inputs(self):
        owner=dict(pid=456,start_ticks='77');request=ref('synthetic-observer-request');actual=dict(self.actual,session_id=222)
        observation=dict(admission=self.a_ref,observer_owner=owner,request=request,actual_observer_exit_pending=True,self_exit_claimed=False,
            final_owned=dict(physically_drained=True),full_replay_equality=ref('synthetic-equality'),running=ref('synthetic-running'),
            running_identity=dict(inode=222),actual_outer_exit=observer.reconcile(self.terminal,self.running,self.actual))
        live=dict(schema='minicpm5_retained_replay_observer_live_v1',independent_actual_live_observation=True,
            owner=owner,request=request,actual_tool=dict(kind='tool_session',session_id=222))
        final=dict(schema='minicpm5_retained_replay_fresh_final_publication_v1',running=observation['running'],running_identity=observation['running_identity'],
            observer_owner=owner,actual_observer_tool=actual,observer_owner_gone=True,final_owned=dict(physically_drained=True),
            lock_identity=c.LOCK,complete_history=[ref('synthetic-complete-history')])
        return [self.running,ref('synthetic-observation'),observation,ref('synthetic-live'),live,actual,'time',final]
    def test_final_render_needs_actual_observer_exit_and_fresh_full_proof(self):
        args=self.final_inputs();value=observer.render_completed(*args)
        self.assertEqual(value['live_tool_sessions'],[]);self.assertEqual(value['all_owners'],self.previous['all_owners'])
        self.assertFalse(value['minicpm5_retained_replay_completed']['deletion_authorized'])
        args[-1]['observer_owner_gone']=False
        with self.assertRaises(ValueError):observer.render_completed(*args)
    def test_wrong_observer_session_and_running_identity_rejected(self):
        args=self.final_inputs();args[5]=dict(args[5],session_id=333)
        with self.assertRaises(ValueError):observer.render_completed(*args)
        args=self.final_inputs();args[-1]['running_identity']=dict(inode=333)
        with self.assertRaises(ValueError):observer.render_completed(*args)
    def test_observer_nonzero_exit_cannot_publish_idle(self):
        args=self.final_inputs();args[5]=dict(args[5],exit_code=1)
        with self.assertRaises(ValueError):observer.render_completed(*args)
    def test_finalizer_contains_fresh_history_and_physical_gate(self):
        tree=ast.parse(Path(c.FILES[3]).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='publish_completed')
        names=[ast.unparse(n.func)for n in ast.walk(fn)if isinstance(n,ast.Call)]
        for expected in ('lock.acquire','history.physical_pair','history.unchanged','drained','caller.latest_observation','render_completed','io.once'):
            self.assertIn(expected,names)
        observe=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='observe')
        self.assertNotIn('terminal_handoff_path',ast.unparse(observe))


if __name__=='__main__':unittest.main(verbosity=2)
