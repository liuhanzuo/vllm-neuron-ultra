"""Pure AST/mock checks; no imported experiment modules or native execution.

Only text source is read. Runtime definitions are AST-extracted into a mocked
namespace. All process, proc, resource, framework and Store IO is replaced.
"""
import ast
import builtins
import copy
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
SOURCE = Q/'minicpm5_pv_sdk_import_diagnostic_v1.py'
ORIGINAL = Q/'minicpm5_layer1_fixed_input_pv_adapter_v3.py'
ORIGINAL_SHA = '50f7bedf915c3fbf83ce5e623f84fd391b8d270dc29135bd4b6c357ecbdbbba0'
LIMITS = dict(address_space_bytes=2147483648, cpu=6, cpu_seconds=600,
              hard_cpu_seconds=605, nice=19, output_file_bytes=2097152, wall_seconds=900)
CAPS = dict(raw_files=4392, raw_bytes=12288000, metadata_bytes=8*2**20,
            allocated_bytes=32*2**20, one_file_bytes=2*2**20)
OWNER = dict(pid=101, start_ticks='200', ppid=99, argv=['sdk-python', '-B', 'worker'], cgroup='original-group')


def require(value, message):
    if not value: raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def forbidden(*args, **kwargs):
    raise AssertionError('Real process/native/raw IO is forbidden in pure checks')


def tree(path=SOURCE):
    raw = path.read_bytes()
    if path == ORIGINAL: require(hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, 'Immutable original adapter')
    return ast.parse(raw, filename=str(path))


def namespace():
    ns = dict(copy=copy, json=json, Path=Path, re=re, require=require, ROOT=Q.parent,
        serial=NS(encoded=encoded, exact=lambda a,b: encoded(a)==encoded(b), CAPS=CAPS,
                  design=NS(ENVIRONMENTS={'sdk': {'torch_threads': 1}})),
        prior=NS(LIMITS=LIMITS), core=NS(), r=NS(), original=NS(Store=object),
        os=NS(open=forbidden, read=forbidden), resource=NS(), signal=NS(), subprocess=NS(Popen=forbidden),
        sys=NS(stderr=io.StringIO()), time=NS(monotonic=lambda: 1, sleep=lambda x: None),
        authenticate_framework=forbidden)
    module = tree()
    for node in module.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
            'SUMMARY_CAP','STATUS_CAP','ERROR_CAP','DIAGNOSTIC_FILES','SPEC_SCHEMA','REVIEW_SCHEMA',
            'ADMISSION_SCHEMA','SUMMARY_SCHEMA','RELEASE_SCHEMA','RUNTIME_SOURCES'}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class StatusTests(unittest.TestCase):
    def setUp(self): self.ns = namespace()
    def test_real_units(self):
        got = self.ns['parse_status'](b'Name:\tpython\nVmSize:\t1234 kB\nVmPeak: 1500 kB\nVmRSS: 12 kB\nThreads:\t1\n')
        self.assertEqual(got['VmSize']['value'],1234); self.assertEqual(got['Threads']['unit'],'count')
    def test_missing_is_missing(self):
        self.assertEqual(self.ns['parse_status'](b'Name: python\n')['VmRSS'], {'present':False})
    def test_bad_fields(self):
        for raw in (b'VmRSS: 1 MB', b'VmRSS: -1 kB', b'Threads: 1 kB', b'VmRSS: 1 kB\nVmRSS: 1 kB', b'VmRSS: unknown'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.ns['parse_status'](raw)
    def test_oversize(self):
        with self.assertRaises(ValueError): self.ns['parse_status'](b'x'*65537)
    def test_eof_required_and_fd_closed(self):
        reads=iter([b'VmSize: 2 kB\n',b'']); log=[]
        self.ns['os']=NS(open=lambda *a:3,read=lambda *a:next(reads),close=lambda f:log.append(f),O_RDONLY=0,O_CLOEXEC=0)
        self.assertEqual(self.ns['read_status']()['VmSize']['value'],2); self.assertEqual(log,[3])
    def test_exact_cap_then_extra_rejected(self):
        reads=iter([b'x'*65536,b'x']); log=[]
        self.ns['os']=NS(open=lambda *a:3,read=lambda *a:next(reads),close=lambda f:log.append(f),O_RDONLY=0,O_CLOEXEC=0)
        with self.assertRaises(ValueError): self.ns['read_status']()
        self.assertEqual(log,[3])
    def test_observe_exact_mock_values(self):
        self.ns['os']=NS(environ={'A':'original'},sched_getaffinity=lambda p:{6},getpriority=lambda *a:19,PRIO_PROCESS=0)
        self.ns['resource']=NS(RLIMIT_AS=1,RLIMIT_CPU=2,RLIMIT_FSIZE=3,getrlimit=lambda k:{1:(2**31,2**31),2:(600,605),3:(2**21,2**21)}[k])
        self.ns['sys']=NS(executable='/mock/sdk/python',dont_write_bytecode=True)
        self.ns['Path']=lambda p:NS(resolve=lambda:p)
        self.ns['time']=NS(monotonic_ns=lambda:123,time_ns=lambda:456)
        self.ns['read_status']=lambda:{'Threads':{'present':True,'value':1}}
        got=self.ns['observe']('before')
        self.assertEqual(got['environment'],{'A':'original'});self.assertEqual(got['limits']['AS'],[2**31]*2)
        self.assertEqual((got['nice'],got['affinity'],got['monotonic_ns']),(19,[6],123))


class ExceptionTests(unittest.TestCase):
    def setUp(self): self.record=namespace()['exception_record']
    def test_no_errno_inferred(self):
        got=self.record(ImportError('libcusparseLt.so.0: failed to map segment from shared object'))
        self.assertFalse(got['nodes'][0]['errno_obtained']);self.assertIsNone(got['nodes'][0]['errno'])
        self.assertFalse(got['errno_inferred']);self.assertFalse(got['root_cause_determined'])
    def test_cause_context_actual_errno(self):
        error=ImportError('wrapper');error.__cause__=OSError(12,'actual errno');error.__context__=ValueError('context')
        got=self.record(error); row=got['nodes'][0]
        self.assertEqual(got['nodes'][row['cause']]['errno'],12)
        self.assertEqual(got['nodes'][row['context']]['message'],'context')
    def test_shared_cause_context_not_cycle(self):
        error=ImportError('wrapper');error.__cause__=error.__context__=OSError(5,'shared')
        got=self.record(error);self.assertEqual(len(got['nodes']),2)
    def test_cycle_rejected(self):
        error=ImportError('cycle');error.__cause__=error
        with self.assertRaisesRegex(ValueError,'cycle'): self.record(error)
    def test_message_overflow_rejected(self):
        with self.assertRaises(ValueError):self.record(ValueError('x'*8193))
    def test_chain_overflow_rejected(self):
        root=error=ValueError('0')
        for i in range(16):error.__cause__=ValueError(str(i));error=error.__cause__
        with self.assertRaisesRegex(ValueError,'graph'):self.record(root)
    def test_full_frames_not_last_line(self):
        def inner(): raise ImportError('mapping failed')
        try:inner()
        except ImportError as error: got=self.record(error)
        frames=got['nodes'][0]['traceback_frames']
        self.assertEqual([f['function'] for f in frames],['test_full_frames_not_last_line','inner'])
        self.assertIn('Traceback (most recent call last)',got['nodes'][0]['traceback'])
        self.assertTrue(got['complete']); self.assertFalse(got['traceback_source_lines_included'])


def import_case(failure=None, *, huge=False):
    ns=namespace(); log=[]
    def mark(stage):
        log.append(stage)
        if stage==failure: raise ImportError('mock failure in '+stage)
    torch=NS(set_num_threads=lambda n:mark('torch_set_num_threads'))
    repeat=object()
    def imported(name,*args,**kwargs):
        if name=='torch':mark('torch_import');return torch
        if name=='neuronx_distributed_inference.modules.attention.utils':mark('sdk_repeat_kv_import');return NS(repeat_kv=repeat)
        raise AssertionError('Unmocked import '+name)
    ns['authenticate_framework']=lambda *args:mark('framework_authentication')
    ns['r']=NS(verify_loaded=lambda *args:mark('loaded_source_authentication'))
    def observer(label):
        mark(label)
        return {'label':label,'mock':True,'environment':{'x':'x'*17000 if huge else '1'}}
    fake_builtins=dict(vars(builtins),__import__=imported)
    ns['execute_import'].__globals__['__builtins__']=fake_builtins
    # Functions retain their builtins at definition: recompile only this actual
    # function under mocked __import__, never import any native package.
    node=next(n for n in tree().body if isinstance(n,ast.FunctionDef) and n.name=='execute_import')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns['execute_import']({'source_sha256':{}},lambda:None,observer=observer),log


class ImportTests(unittest.TestCase):
    def test_original_sequence(self):
        (summary,code),log=import_case()
        self.assertEqual(code,0)
        self.assertEqual(log,['before_import','torch_import','sdk_repeat_kv_import','framework_authentication',
            'torch_set_num_threads','loaded_source_authentication','after_import'])
        self.assertTrue(summary['diagnostic_passed']);self.assertFalse(summary['self_exit_claimed'])
    def test_every_import_failure_stage(self):
        for stage in ('torch_import','sdk_repeat_kv_import','framework_authentication','torch_set_num_threads','loaded_source_authentication'):
            with self.subTest(stage=stage):
                (summary,code),log=import_case(stage)
                self.assertEqual(code,1);self.assertEqual(summary['stage'],stage)
                self.assertTrue(summary['complete_error_record_saved']);self.assertEqual(log[-1],'after_failure')
    def test_observer_failure_stops_import(self):
        (summary,code),log=import_case('before_import')
        self.assertEqual(code,1);self.assertNotIn('torch_import',log)
    def test_after_observer_failure_preserves_import_fact(self):
        (summary,code),_=import_case('after_import')
        self.assertEqual(code,1);self.assertTrue(summary['import_sequence_completed'])
    def test_summary_overflow_explicitly_incomplete(self):
        (summary,code),_=import_case(huge=True)
        self.assertEqual(code,1);self.assertFalse(summary['observation_and_error_payload_complete'])
        self.assertFalse(summary['complete_error_record_saved']);self.assertLessEqual(len(encoded(summary)),16384)


def terminal_fixture(polls, observed, exit_code=0, *, guard_error=False, gone_error=False, members=False, write_error=False):
    ns=namespace();log=[];writes=[];iterator=iter(polls)
    def guard(**kw):
        log.append(('guard',));require(not guard_error,'Guard failed')
    def proc(pid):log.append(('proc',));return copy.deepcopy(observed)
    def gone(owner):log.append(('gone',));require(not gone_error,'Owner still present')
    def group(pid):log.append(('group_members',));return ['descendant'] if members else []
    def write(name,value):
        log.append(('write',name));require(not write_error,'Store guard/write failed');writes.append((name,copy.deepcopy(value)))
    def poll():value=next(iterator);log.append(('poll',value));return value
    def wait(timeout):log.append(('wait',timeout));return exit_code
    child=NS(pid=OWNER['pid'],poll=poll,wait=wait,stdin=NS(closed=False,close=lambda:log.append(('close',))))
    ns['r']=NS(p=NS(proc=proc,gone=gone),drain_group=lambda *a:log.append(('drain',)),group_members=group)
    ports=ns['SingleSDKAdapter']({}, {}, {}, NS(), guard, NS(), NS(json=write))
    ports.state='released';ports.child_started=True;ports.child=child;ports.owner=copy.deepcopy(OWNER);ports.argv=copy.deepcopy(OWNER['argv'])
    return ports,log,writes,ns


class TerminalTests(unittest.TestCase):
    # Seventeen checks cover the actual new adapter finish, plus cleanup paths.
    def test_01_exact_success(self):
        p,_,w,_=terminal_fixture([None,0],OWNER);self.assertEqual(p.finish(OWNER,OWNER['argv'])['returncode'],0);self.assertEqual(len(w),1)
    def test_02_original_argv_race_now_observed(self):
        p,_,_,_=terminal_fixture([None,1],dict(OWNER,argv=[]),1);self.assertEqual(p.finish(OWNER,OWNER['argv'])['returncode'],1)
    def test_03_original_disappearance_race_now_observed(self):
        p,_,_,_=terminal_fixture([None,1],None,1);self.assertEqual(p.finish(OWNER,OWNER['argv'])['returncode'],1)
    def test_04_state_drained_not_phase0(self):
        p,_,_,_=terminal_fixture([0],None);p.finish(OWNER,OWNER['argv']);self.assertEqual(p.state,'drained');self.assertFalse(hasattr(p,'last_exit'))
    def test_05_nonzero_never_promoted(self):
        p,_,w,_=terminal_fixture([1],None,1);v=p.finish(OWNER,OWNER['argv']);self.assertEqual(w[0][1]['returncode'],1);self.assertFalse(v['self_exit_claimed'])
    def test_06_same_owner_preserved(self):
        p,_,_,_=terminal_fixture([None,1],dict(OWNER,argv=[]),1);self.assertEqual(p.finish(OWNER,OWNER['argv'])['owner'],OWNER)
    def test_07_signal_returncode_preserved(self):
        p,_,_,_=terminal_fixture([-9],None,-9);self.assertEqual(p.finish(OWNER,OWNER['argv'])['returncode'],-9)
    def check_changed(self,**change):
        p,_,w,_=terminal_fixture([None,1],dict(OWNER,**change),1)
        with self.assertRaisesRegex(ValueError,'Unexpected child identity'):p.finish(OWNER,OWNER['argv'])
        self.assertEqual(w,[])
    def test_08_changed_argv(self):self.check_changed(argv=['different'])
    def test_09_changed_pid(self):self.check_changed(pid=102,argv=[])
    def test_10_changed_start(self):self.check_changed(start_ticks='201',argv=[])
    def test_11_changed_parent(self):self.check_changed(ppid=98,argv=[])
    def test_12_changed_cgroup(self):self.check_changed(cgroup='different',argv=[])
    def test_13_zombie_shape_live_rejected(self):
        p,_,w,_=terminal_fixture([None,None],dict(OWNER,argv=[]))
        with self.assertRaisesRegex(ValueError,'Exact child identity while live'):p.finish(OWNER,OWNER['argv'])
        self.assertEqual(w,[])
    def test_14_owner_gone_before_receipt(self):
        p,_,w,_=terminal_fixture([1],None,1,gone_error=True)
        with self.assertRaises(ValueError):p.finish(OWNER,OWNER['argv'])
        self.assertEqual(w,[]);self.assertIsNotNone(p.child)
    def test_15_group_empty_before_receipt(self):
        p,_,w,_=terminal_fixture([1],None,1,members=True)
        with self.assertRaises(ValueError):p.finish(OWNER,OWNER['argv'])
        self.assertEqual(w,[]);self.assertIsNotNone(p.child)
    def test_16_guard_failure_preserved(self):
        p,_,w,_=terminal_fixture([None],None,1,guard_error=True)
        with self.assertRaisesRegex(ValueError,'Guard failed'):p.finish(OWNER,OWNER['argv'])
        self.assertEqual(w,[])
    def test_17_wait_drain_gone_group_write(self):
        p,log,_,_=terminal_fixture([None,1],dict(OWNER,argv=[]),1);p.finish(OWNER,OWNER['argv'])
        self.assertEqual([e[0] for e in log if e[0] in ('wait','drain','gone','group_members','write')],['wait','drain','gone','group_members','write'])
    def test_cleanup_wait_records_actual(self):
        p,log,w,_=terminal_fixture([],None,7);p.drain_owned()
        self.assertEqual(w[0][1]['returncode'],7);self.assertTrue(w[0][1]['cleanup_after_failure'])
        self.assertEqual([e[0] for e in log],['close','drain','wait','gone','group_members','write'])
    def test_sticky_guard_write_failure_no_weak_writer(self):
        p,_,w,ns=terminal_fixture([],None,1,write_error=True)
        with self.assertRaisesRegex(ValueError,'Store guard/write failed'):p.drain_owned()
        self.assertEqual(w,[]);self.assertEqual(p.state,'drained');self.assertIsNone(p.child)
        event=json.loads(ns['sys'].stderr.getvalue());self.assertFalse(event['exit_receipt_saved'])
        self.assertEqual(event['actual_parent_observed_terminal']['returncode'],1)
        self.assertFalse(p.drain_owned()['exit_receipt_saved'])
    def test_unacknowledged_owner_never_invented(self):
        p,_,w,_=terminal_fixture([],None,1);p.owner=None;p.drain_owned()
        self.assertIsNone(w[0][1]['owner']);self.assertIsNone(w[0][1]['exact_owner_gone']);self.assertFalse(w[0][1]['owner_acknowledged'])
    def test_second_child_rejected_after_success(self):
        p,_,_,_=terminal_fixture([0],None);p.finish(OWNER,OWNER['argv'])
        with self.assertRaisesRegex(ValueError,'Only one'):p.start()
    def test_finish_before_release_rejected(self):
        p,_,_,_=terminal_fixture([],None);p.state='new'
        with self.assertRaises(ValueError):p.finish(OWNER,OWNER['argv'])


class SourceContractTests(unittest.TestCase):
    def test_full_preimport_ast_equivalence(self):
        old=next(n for n in tree(ORIGINAL).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        new=next(n for n in tree().body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        class Normalize(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='ADMISSION_SCHEMA':return ast.copy_location(ast.Constant('minicpm5_fixed_input_pv_serial_admission_v1'),node)
                if node.id=='RELEASE_SCHEMA':return ast.copy_location(ast.Constant('minicpm5_fixed_input_pv_phase_release_v1'),node)
                if node.id=='DiagnosticStore':return ast.copy_location(ast.Name(id='Store',ctx=node.ctx),node)
                return node
            def visit_Constant(self,node):
                if node.value=='Exact independent diagnostic admission':node.value='Exact PV admission'
                return node
        for before,after in zip(old.body[1:32],new.body[2:33]):
            self.assertEqual(ast.dump(before),ast.dump(Normalize().visit(copy.deepcopy(after))))
        self.assertEqual(ast.dump(old.body[32].body[0]),ast.dump(new.body[33]))
        self.assertEqual(ast.dump(old.body[33]),ast.dump(new.body[34]))
    def test_no_scientific_callbacks_or_predecessor_state(self):
        source=SOURCE.read_text()
        for bad in ('last_exit','FivePhases(','.PVLeaf(','.read_bytes(','.raw_sink(','.event_sink(','.run_exact_reference('):
            self.assertNotIn(bad,source)
    def test_original_limits_alias_and_outer_order(self):
        source=SOURCE.read_text();self.assertIn('impose_original_limits = original.impose_original_limits',source)
        run=next(n for n in tree().body if isinstance(n,ast.FunctionDef) and n.name=='run')
        code=ast.unparse(run)
        self.assertLess(code.index('impose_original_limits()'),code.index('auth.authenticate('))
        self.assertLess(code.index('auth.authenticate('),code.index('ports.start()'))
    def test_only_four_output_names(self):
        ns=namespace();klass=ns['DiagnosticStore']
        self.assertTrue(all(klass.legal(n) for n in ns['DIAGNOSTIC_FILES']))
        for name in ('final.json','own-gate.json','phase-0-exit.json','phase-1-audit.json','1-r-'+64*'a'+'.bin'):
            self.assertFalse(klass.legal(name))
    def test_summary_budget_rejects_before_write(self):
        ns=namespace();store=ns['DiagnosticStore']();store.write=forbidden
        with self.assertRaisesRegex(ValueError,'summary'):store.json('phase-1-summary.json',{'x':'x'*16384})
    def test_original_store_allocation_guard_retained(self):
        cls=next(n for n in tree(ORIGINAL).body if isinstance(n,ast.ClassDef) and n.name=='Store')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='write')
        ns=dict(require=require,serial=NS(CAPS=CAPS),RUNTIME=set(),RUNTIME_METADATA_CAP=79008,
            time=NS(monotonic=lambda:0),os=NS(open=forbidden))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(ORIGINAL),'exec'),ns)
        store=NS(legal=lambda n:True,guard=lambda:None,last=0,_directory=lambda:None,
            root=NS(__truediv__=lambda s,n:None),rows={},logical_raw=0,logical_metadata=0,allocated=32*2**20)
        class Root:
            def __truediv__(self,n):return NS(exists=lambda:False)
        store.root=Root()
        with self.assertRaisesRegex(ValueError,'allocation'):ns['write'](store,'phase-1-exit.json',b'{}')
    def test_cli_only_phase1(self):
        source=(Q/'minicpm5_pv_sdk_import_diagnostic_worker_v1.py').read_text()
        self.assertIn('choices=(1,)',source);self.assertIn('sys.exit(worker(',source)

    def test_original_limit_values_and_order(self):
        log=[]
        ns=dict(prior=NS(LIMITS=LIMITS),os=NS(sched_setaffinity=lambda *a:log.append(('affinity',a)),
            getpriority=lambda *a:0,nice=lambda n:log.append(('nice',n)),PRIO_PROCESS=0),
            resource=NS(RLIMIT_AS=1,RLIMIT_CPU=2,RLIMIT_FSIZE=3,setrlimit=lambda *a:log.append(('rlimit',a))),
            signal=NS(SIGALRM=14,signal=lambda *a:log.append(('handler',a[0])),alarm=lambda n:log.append(('alarm',n))))
        node=next(n for n in tree(ORIGINAL).body if isinstance(n,ast.FunctionDef) and n.name=='impose_original_limits')
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(ORIGINAL),'exec'),ns);ns['impose_original_limits']()
        self.assertEqual(log,[('affinity',(0,{6})),('nice',19),('rlimit',(1,(2**31,2**31))),
            ('rlimit',(2,(600,605))),('rlimit',(3,(2**21,2**21))),('handler',14),('alarm',900)])


def start_fixture():
    ns=namespace();log=[];payload=[];outer=dict(pid=99,cgroup='original-group');worker='/mock/diagnostic-worker.py'
    admission_ref=dict(path='/mock/admission.json',sha256='a'*64)
    argv=['/mock/sdk/python','-B',worker,'--admission',admission_ref['path'],'--admission-sha256','a'*64,'--phase','1']
    owner=dict(OWNER,argv=argv);expected_env={'PATH':'original-sdk','ATTEMPT':'mock-attempt'}
    pins={worker:'b'*64,'/mock/full-source.py':'c'*64}
    ns['serial'].phase_argv=lambda w,a,i: (require((w,a,i)==(worker,admission_ref,1),'Exact original SDK argv request') or argv)
    ns['serial'].python_for=lambda i:(require(i==1,'SDK index') or '/mock/sdk/python')
    ns['PIPE_CAP']=16*2**20
    ns['Path']=lambda p:NS(resolve=lambda:'/mock/sdk/python')
    def env(base,kind,attempt):
        log.append(('environment',base,kind,attempt));require((base,kind,attempt)==({},'cpu_control','mock-attempt'),'Original environment args');return expected_env
    def text(path,sha):log.append(('source',path,sha));require(pins[path]==sha,'Exact worker pin')
    def process(actual,actual_argv,parent,cgroup):
        log.append(('identity',));require((actual,actual_argv,parent,cgroup)==(owner,argv,99,'original-group'),'Exact real owner')
    ns['r']=NS(p=NS(proc=lambda pid:copy.deepcopy(owner),environment=env),life=NS(process=process),sources=NS(text=text))
    class Pipe:
        closed=False
        def write(self,raw):payload.append(raw);log.append(('write_pipe',));return len(raw)
        def flush(self):log.append(('flush_pipe',))
        def close(self):self.closed=True;log.append(('close_pipe',))
    child=NS(pid=101,stdin=Pipe(),poll=lambda:None)
    def popen(got_argv,**kw):
        log.append(('Popen',copy.deepcopy(got_argv),kw));return child
    ns['subprocess']=NS(Popen=popen,PIPE='mock-PIPE')
    a=dict(spec={'worker':worker},context={'pins_plan':{'source_contract':{'all':'original'},'source_sha256':pins}})
    lease=NS(admission={'outer_owner':outer,'attempt_id':'mock-attempt'})
    p=ns['SingleSDKAdapter'](a,admission_ref,{},lease,lambda **kw:log.append(('guard',)),NS(),
        NS(json=lambda name,row:log.append(('owner_record',name,copy.deepcopy(row)))))
    return p,ns,log,payload,owner,argv,expected_env


class StartTests(unittest.TestCase):
    def test_exact_env_argv_and_fd_policy(self):
        p,ns,log,_,owner,argv,env=start_fixture();self.assertEqual(p.start(),(owner,argv))
        launch=next(row for row in log if row[0]=='Popen')
        self.assertEqual(launch[1],argv)
        self.assertEqual(launch[2],dict(env=env,cwd=ns['ROOT'],stdin='mock-PIPE',close_fds=True,start_new_session=True))
    def test_whole_source_map_eof_gate(self):
        p,ns,log,payload,_,_,_=start_fixture();p.start();gate=json.loads(payload[0])
        self.assertEqual(gate['source_plan'],p.authenticated['context']['pins_plan'])
        self.assertEqual(gate['phase'],1);self.assertIsNone(gate['own_gate'])
        self.assertEqual([row[0] for row in log][-3:],['write_pipe','flush_pipe','close_pipe'])
        self.assertEqual(p.state,'released')
    def test_duplicate_start_rejected(self):
        p,_,log,_,_,_,_=start_fixture();p.start()
        with self.assertRaisesRegex(ValueError,'Only one'):p.start()
        self.assertEqual(sum(row[0]=='Popen' for row in log),1)
    def test_start_failure_never_retry(self):
        p,ns,log,_,_,_,_=start_fixture();ns['subprocess'].Popen=forbidden
        with self.assertRaises(AssertionError):p.start()
        self.assertEqual(p.state,'starting')
        self.assertFalse(p.drain_owned()['child_started'])
        with self.assertRaisesRegex(ValueError,'Only one'):p.start()
    def test_owner_mismatch_prevents_release(self):
        p,ns,_,payload,owner,_,_=start_fixture();ns['r'].p.proc=lambda pid:dict(owner,ppid=123)
        with self.assertRaisesRegex(ValueError,'Exact real owner'):p.start()
        self.assertEqual(payload,[])
    def test_pipe_budget_prevents_release(self):
        p,ns,_,payload,_,_,_=start_fixture();ns['PIPE_CAP']=1
        with self.assertRaisesRegex(ValueError,'16MiB'):p.start()
        self.assertEqual(payload,[]);self.assertIsNotNone(p.owner)


class ReviewBindingTests(unittest.TestCase):
    def fixture(self):
        ns=namespace();ns['r']=NS(life=NS(ref=lambda ref:require(set(ref)=={'path','sha256'},'Reference')))
        ns['original'].__file__=str(ORIGINAL)
        ref={'path':'/mock/spec.json','sha256':'a'*64};review_ref={'path':'/mock/review.json','sha256':'b'*64}
        pins={p:'c'*64 for p in (*ns['RUNTIME_SOURCES'],str(ORIGINAL))}
        spec=dict(schema=ns['SPEC_SCHEMA'],pv_spec=ref,pv_review=review_ref,worker=ns['RUNTIME_SOURCES'][2],
            limits=copy.deepcopy(LIMITS),artifact_limits=copy.deepcopy(CAPS),source_extension=pins)
        review=dict(schema=ns['REVIEW_SCHEMA'],reviewer='/root',passed=True,blockers=[],spec=ref,source_extension=pins)
        return ns,ref,review_ref,spec,review
    def test_valid_concrete_review(self):
        ns,ref,review_ref,spec,review=self.fixture();ns['validate_spec'](ref,review_ref,spec,review)
    def test_limits_cannot_expand(self):
        ns,ref,review_ref,spec,review=self.fixture();spec['limits']['address_space_bytes']=4*2**30
        with self.assertRaisesRegex(ValueError,'limits'):ns['validate_spec'](ref,review_ref,spec,review)
    def test_unreviewed_child_source_rejected(self):
        ns,ref,review_ref,spec,review=self.fixture();del spec['source_extension'][ns['RUNTIME_SOURCES'][0]]
        with self.assertRaisesRegex(ValueError,'pinned'):ns['validate_spec'](ref,review_ref,spec,review)
    def test_child_review_cannot_authorize(self):
        ns,ref,review_ref,spec,review=self.fixture();review['reviewer']='/root/child'
        with self.assertRaisesRegex(ValueError,'root'):ns['validate_spec'](ref,review_ref,spec,review)
    def test_complete_original_extension_and_namespace(self):
        ns,ref,review_ref,spec,review=self.fixture()
        out=str(ns['ROOT']/'research'/'minicpm5-fixed-input-pv-sdk-import-mock')
        a=dict(spec={'worker':spec['worker'],'source_extension':spec['source_extension'],'output_root':out},
            witness={'spec':ref,'review':review_ref},output_root=out,context={'pins_plan':{'source_sha256':dict(spec['source_extension'])}})
        ns['bind_authenticated'](spec,a)
        del a['context']['pins_plan']['source_sha256'][ns['RUNTIME_SOURCES'][0]]
        with self.assertRaisesRegex(ValueError,'Whole authenticated'):ns['bind_authenticated'](spec,a)


if __name__=='__main__':unittest.main(verbosity=2)
