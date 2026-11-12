"""Original-child/observer coordination tests; all process/native/IO ports mocked."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'research'), str(ROOT)]
class BlockNativeRelatedImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('ctypes','_ctypes','torch','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('Native-related import forbidden in source test: '+fullname)

assert all(name not in sys.modules for name in ('ctypes','_ctypes','torch','numpy'))
sys.meta_path.insert(0,BlockNativeRelatedImports())
from research import minicpm5_sdk_mapping_observed_adapter_v3 as module
from research import check_minicpm5_sdk_mapping_observer_logic_v1 as f
from research import check_minicpm5_sdk_mapping_observer_backend_v2 as bf


def authenticate_example(authority, guard):
    raise AssertionError('Source identification fixture only, never native authority')


def verify_example(ctypes, libc, receipt, guard):
    raise AssertionError('Source identification fixture only, never native authority')


class Guard:
    def __init__(self, lease): self.lease=lease; self.calls=[]
    def __call__(self, **kwargs): self.calls.append(kwargs)


class Store(f.FakeStore):
    def __init__(self, guard):
        super().__init__(); self.guard=guard; self.files={'admission.json':b'{}','phase-1-owner.json':b'{}'}
    def json(self, name, value): return self.write(name,m.serial.encoded(value))
    def load(self, name): return json.loads(self.files[name]), self.reference(name)
    def reference(self, name):
        raw=self.files[name];return dict(path='/mock/'+name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    def inventory(self): return {'names':sorted(self.files)}


class Child(f.FakeChild):
    def __init__(self):
        self.returncode=None; self.stdin=NS(closed=False,write=Mock(side_effect=AssertionError('No early gate write')))
        self._waitpid_lock=bf.Lock()


class Ports(f.FakePorts):
    def __init__(self, child, events=None):
        super().__init__(events); self.child=child; self.wait_lock_owned=False; self.handoffs=0
        self.wait_handoff_started=False; self.wait_transition_uncertain=False
        self.wait_lock=child._waitpid_lock; self.signal=bf.FakeSignal()
    _transition_wait_lock=module.backend.NativePorts._transition_wait_lock
    begin_wait_handoff=module.backend.NativePorts.begin_wait_handoff
    def finish_wait_handoff(self, result):
        assert self.wait_lock_owned and self.closed
        assert result['physically_drained'] and result['sdk_terminal']['returncode']==self.child.returncode
        assert all(self.identity(int(tid)) is None for tid in result['task_terminals'])
        assert self.identity(self.child.pid) is None and self.owned_group_members()==[]
        self._transition_wait_lock(acquire=False);self.handoffs+=1


def source_class(path, name, namespace):
    # Execute only the original class definition with explicit mocked dependencies.
    # Whole original modules would transitively import historical ctypes.
    tree=ast.parse(path.read_bytes(),filename=str(path))
    nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name]
    assert len(nodes)==1
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def fixture_original():
    logic=module.logic
    serial=NS(encoded=logic.encoded,CAPS=logic.CAPS,exact=lambda a,b:a==b,
              phase_argv=lambda *a:None,python_for=lambda i:'/mock/python')
    def text(path, sha):
        raw=Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==sha
        return raw
    r=NS(p=NS(gone=Mock(),proc=Mock(),environment=Mock()),group_members=Mock(return_value=[]),
         sources=NS(text=text),life=NS(process=Mock()),verify_loaded=Mock(),drain_group=Mock())
    original=NS(__file__=str(ROOT/'research/minicpm5_pv_sdk_import_diagnostic_v1.py'),
                r=r,serial=serial,require=logic.require,prior=NS(LIMITS=logic.LIMITS),
                ROOT=ROOT,PIPE_CAP=logic.GATE_CAP,SUMMARY_SCHEMA='synthetic-summary',
                RELEASE_SCHEMA='synthetic-release',subprocess=NS(Popen=Mock(),PIPE=-1))
    # Reuse actual original parent terminal/cleanup methods with every side effect
    # explicitly mocked; do not substitute their implementations in the adapter.
    original.SingleSDKAdapter=source_class(Path(original.__file__),'SingleSDKAdapter',
        dict(copy=copy,r=r,require=logic.require,serial=serial,sys=sys,
             SUMMARY_CAP=logic.SUMMARY_CAP))
    class OriginalStore(Store):
        def __init__(self,root,guard):super().__init__(guard)
    original.original=NS(Store=OriginalStore,RUNTIME_METADATA_CAP=logic.RUNTIME_METADATA_CAP,
        serial=serial,RUNTIME={'admission.json','phase-1-owner.json','phase-1-summary.json','phase-1-exit.json'},
        short=lambda ref:{k:ref[k] for k in ('path','sha256')})
    return original


m=module.make_adapter_class(fixture_original())


def setup(case, code=0, events=None):
    lease=NS(admission={'outer_owner':{'pid':99,'cgroup':'original'},'attempt_id':'a'*32})
    guard=Guard(lease);store=m.ObserverStore('/mock',guard);metadata=NS(unchanged=Mock())
    # In-memory fixture only. Backend's actual allocation/inode accounting is
    # separately checked by its accepted Store tests; no real output IO here.
    store.write=Store.write.__get__(store)
    store.inventory=Store.inventory.__get__(store)
    store.validate_observer_budget=f.FakeStore.validate_observer_budget.__get__(store)
    authenticated={'spec':{'worker':'/mock/worker.py'},'context':{'pins_plan':{
        'source_contract':{'path':'/mock/source-contract.json','sha256':'b'*64},
        'source_sha256':{'/mock/worker.py':'a'*64}}}}
    a=m.ObservedSingleSDKAdapter(authenticated,{'path':'/mock/admission.json','sha256':'c'*64},
        {'synthetic':'admission'},lease,guard,metadata,store,authority={'synthetic':'authority'},
        outer_deadline=m.time.monotonic()+850,
        authenticate=authenticate_example,verify_loaded_native=verify_example)
    child=Child();ports=Ports(child,f.normal(code) if events is None else events)
    a.child=child;a.child_started=True;a.owner=copy.deepcopy(f.OWNER);a.argv=f.OWNER['argv'][:]
    a.gate=b'whole-source-gate';a.state='gated';a.prerequisites_checked=True
    case.enterContext(patch.object(m.backend,'make_ports',return_value=ports))
    case.enterContext(patch.object(m.r.p,'gone',return_value=None))
    case.enterContext(patch.object(m.r,'group_members',return_value=[]))
    case.enterContext(patch.object(m.r,'verify_loaded',return_value=None))
    store.files['phase-1-summary.json']=m.serial.encoded(dict(schema=m.original.SUMMARY_SCHEMA,
        diagnostic_passed=code==0,stage='complete' if code==0 else 'torch_import',
        observation_and_error_payload_complete=True))
    return a,child,ports,store,guard


class CoordinationTests(unittest.TestCase):
    def enterContext(self, context):
        value=context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)
        return value

    def test_actual_zero_exit_and_complete_observation_pass(self):
        a,c,p,s,g=setup(self);terminal=a.finish(a.owner,a.argv);result=a.result()
        self.assertEqual(terminal['returncode'],0);self.assertTrue(result['diagnostic_passed'])
        self.assertEqual(result['exit_code'],0);self.assertEqual(set(s.files),m.FILES)
        self.assertEqual(bytes(p.written),b'whole-source-gate');self.assertEqual(p.handoffs,1)
        self.assertIsNone(a.child);self.assertIs(a.native_ports.child,c)
        self.assertFalse(p.wait_lock_owned)

    def test_nonzero_sdk_exit_remains_failure_after_complete_trace(self):
        a,c,p,s,g=setup(self,code=1);terminal=a.finish(a.owner,a.argv);result=a.result()
        self.assertTrue(result['observer']['observation_complete'])
        self.assertEqual(terminal['returncode'],1);self.assertFalse(result['diagnostic_passed'])
        self.assertEqual(result['exit_code'],1)
        trace=[json.loads(line) for line in s.files[m.logic.TRACE_NAME].splitlines()]
        self.assertEqual(next(x for x in trace if x['kind']=='memory_exit')['errno'],12)

    def test_zero_child_cannot_hide_trace_persistence_failure(self):
        a,c,p,s,g=setup(self);s.fail_name=m.logic.TRACE_NAME
        a.finish(a.owner,a.argv);result=a.result()
        self.assertEqual(result['terminal']['returncode'],0);self.assertFalse(result['diagnostic_passed'])
        self.assertEqual(result['exit_code'],1);self.assertFalse(result['observer']['trace_saved'])

    def test_zero_child_cannot_hide_observer_receipt_failure(self):
        a,c,p,s,g=setup(self);s.fail_name=m.logic.RECEIPT_NAME
        a.finish(a.owner,a.argv);result=a.result()
        self.assertEqual(result['exit_code'],1);self.assertFalse(result['observer']['exit_receipt_saved'])

    def test_actual_terminal_persistence_failure_does_not_rewait(self):
        a,c,p,s,g=setup(self);s.fail_name='phase-1-exit.json'
        with patch('sys.stderr.write'):
            with self.assertRaises(ValueError):a.finish(a.owner,a.argv)
        self.assertEqual(a.state,'drained');self.assertFalse(a.receipt_saved)
        self.assertTrue(a.drain_owned()['physically_drained']);self.assertEqual(p.handoffs,1)
        self.assertEqual(a.result()['exit_code'],1)

    def test_factory_failure_remains_original_pre_handoff_cleanup(self):
        a,c,p,s,g=setup(self)
        with patch.object(m.backend,'make_ports',side_effect=ValueError('authority rejected')):
            with self.assertRaises(ValueError):a.finish(a.owner,a.argv)
        self.assertIsNone(a.native_ports)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',return_value={'plain':True}) as old:
            self.assertEqual(a.drain_owned(),{'plain':True});old.assert_called_once()

    def test_logic_constructor_failure_precedes_wait_handoff(self):
        a,c,p,s,g=setup(self);a.gate=b''
        with patch.object(m.backend,'make_ports') as factory:
            with self.assertRaises(m.logic.ObservationError):a.finish(a.owner,a.argv)
            factory.assert_not_called()
        self.assertIsNone(a.native_ports)

    def test_foreign_group_keeps_handle_and_wait_owner(self):
        a,c,p,s,g=setup(self);p.extra_group=[999]
        with self.assertRaises(m.logic.ObservationError):a.finish(a.owner,a.argv)
        self.assertIs(a.child,c);self.assertTrue(p.wait_lock_owned)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',side_effect=AssertionError('No Popen cleanup')):
            with self.assertRaises(Exception):a.drain_owned()
        self.assertIs(a.native_ports,p);self.assertTrue(p.wait_lock_owned)

    def test_later_group_drain_uses_same_observer_without_counter_reset(self):
        a,c,p,s,g=setup(self);p.extra_group=[999]
        with self.assertRaises(m.logic.ObservationError):a.finish(a.owner,a.argv)
        observer=a.observer;counts=copy.deepcopy(observer.counts);steps=observer.steps
        p.extra_group=[]
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',side_effect=AssertionError('No old wait')):
            proof=a.drain_owned()
        self.assertTrue(proof['physically_drained']);self.assertIs(a.observer,observer)
        self.assertEqual(observer.counts,counts);self.assertEqual(observer.steps,steps)
        self.assertFalse(a.result()['diagnostic_passed'])

    def test_finish_rejects_wrong_owner_before_factory(self):
        a,c,p,s,g=setup(self)
        with patch.object(m.backend,'make_ports') as factory:
            with self.assertRaises(Exception):a.finish(dict(a.owner,start_ticks='201'),a.argv)
            factory.assert_not_called()

    def test_result_requires_actual_drain(self):
        a,c,p,s,g=setup(self)
        with self.assertRaises(Exception):a.result()

    def test_success_requires_exact_six_files(self):
        a,c,p,s,g=setup(self);a.finish(a.owner,a.argv);s.files['extra.json']=b'{}'
        with self.assertRaises(Exception):a.result()

    def test_success_requires_original_import_summary(self):
        a,c,p,s,g=setup(self);a.finish(a.owner,a.argv)
        summary=json.loads(s.files['phase-1-summary.json']);summary['diagnostic_passed']=False
        s.files['phase-1-summary.json']=m.serial.encoded(summary)
        with self.assertRaises(Exception):a.result()

    def test_signalled_child_exit_is_preserved_and_cli_fails(self):
        a,c,p,s,g=setup(self,events=[f.INITIAL,dict(status=9)])
        a.finish(a.owner,a.argv);result=a.result()
        self.assertEqual(result['terminal']['returncode'],-9);self.assertEqual(result['exit_code'],1)

    def test_passing_diagnosis_cannot_use_failed_child(self):
        with self.assertRaises(Exception):m.result_exit_code({'diagnostic_passed':True,'terminal':{'returncode':1}})
        self.assertEqual(m.result_exit_code({'diagnostic_passed':False,'terminal':{'returncode':120}}),120)

    def test_start_retains_unwritten_whole_gate(self):
        a,c,p,s,g=setup(self);a.child=None;a.state='new';s.files.pop('phase-1-owner.json')
        argv=['/sdk/python','-B','/mock/worker.py'];owner=dict(f.OWNER,argv=argv)
        with patch.object(m.original.subprocess,'Popen',return_value=c) as popen, \
             patch.object(m.serial,'phase_argv',return_value=argv), \
             patch.object(m.r.sources,'text',return_value=b'#mock'), \
             patch.object(m.r.p,'environment',return_value={'ORIGINAL':'environment'}), \
             patch.object(m.r.p,'proc',return_value=owner), \
             patch.object(m.r.life,'process'), \
             patch.object(module,'Path',side_effect=lambda _:NS(resolve=lambda:'/mock/exact/python')):
            got,aargv=a.start()
        self.assertEqual(got,owner);self.assertEqual(aargv,argv);self.assertEqual(a.state,'gated')
        c.stdin.write.assert_not_called();self.assertFalse(c.stdin.closed)
        self.assertEqual(json.loads(a.gate)['source_plan'],a.authenticated['context']['pins_plan'])
        self.assertEqual(popen.call_args.kwargs,dict(env={'ORIGINAL':'environment'},cwd=m.original.ROOT,
            stdin=m.original.subprocess.PIPE,close_fds=True,start_new_session=True))
        with self.assertRaises(Exception):a.start()

    def test_prerequisites_keep_original_outer_alarm_and_full_auth(self):
        a,c,p,s,g=setup(self);a.prerequisites_checked=False
        modules=[m.original,m.backend,m.logic,module]
        pins={str(Path(x.__file__).resolve()):hashlib.sha256(Path(x.__file__).read_bytes()).hexdigest() for x in modules}
        path=str(Path(__file__).resolve());pins[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        a.authenticated['context']['pins_plan']['source_sha256']=pins
        with patch.object(m.original.SingleSDKAdapter,'authenticate_complete_prerequisites') as full, \
             patch.object(m.signal,'getitimer',return_value=(14.0,0.0)) as timer, \
             patch.object(m.signal,'alarm',side_effect=AssertionError('Never restart alarm')):
            a.authenticate_complete_prerequisites()
        full.assert_called_once();timer.assert_called_once_with(m.signal.ITIMER_REAL)
        self.assertTrue(a.prerequisites_checked)
        with self.assertRaises(Exception):a.authenticate_complete_prerequisites()

    def test_expired_or_periodic_alarm_rejected_without_spawn(self):
        for value in ((0,0),(901,0),(20,1)):
            with self.subTest(value=value):
                a,c,p,s,g=setup(self);a.prerequisites_checked=False
                with patch.object(m.original.SingleSDKAdapter,'authenticate_complete_prerequisites'), \
                     patch.object(m.signal,'getitimer',return_value=value), \
                     patch.object(m.original.subprocess,'Popen') as popen:
                    with self.assertRaises(Exception):a.authenticate_complete_prerequisites()
                    popen.assert_not_called()

    def test_unpinned_authority_callback_rejected(self):
        with self.assertRaises(Exception):m.callback_source(authenticate_example,{})
        with self.assertRaises(Exception):m.callback_source(print,{})

    def test_absolute_deadline_is_not_reset_by_observer(self):
        a,c,p,s,g=setup(self);deadline=a.outer_deadline
        with patch.object(m.time,'monotonic',return_value=deadline+1):
            a.finish(a.owner,a.argv)
        self.assertEqual(a.outer_deadline,deadline)
        self.assertFalse(a.result()['diagnostic_passed'])
        self.assertFalse(any(item[0]=='seize' for item in p.log))

    def test_alarm_between_factory_assignment_and_binding_keeps_owner(self):
        a,c,p,s,g=setup(self)
        a.observer=m.logic.ObservedSDK(c,a.owner,a.gate,a.authority,a.observer_guard,s,None)
        a.native_ports=p
        # Factory preparation has no wait ownership, so this is still original cleanup.
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',return_value={'pre_handoff':True}) as old:
            result=a.drain_owned();old.assert_called_once()
        self.assertEqual(result,{'pre_handoff':True});self.assertFalse(p.wait_lock_owned)
        self.assertFalse(c._waitpid_lock.held);self.assertFalse(p.wait_handoff_started)


    def test_alarm_before_callsite_ports_assignment_keeps_original_cleanup(self):
        a,c,p,s,g=setup(self)
        with bf.interrupt_before_opcode(a.finish,'STORE_ATTR','native_ports') as hit:
            with self.assertRaises(bf.AlarmRaised):a.finish(a.owner,a.argv)
        self.assertEqual(len(hit),1);self.assertIsNone(a.native_ports)
        self.assertFalse(p.wait_handoff_started);self.assertFalse(c._waitpid_lock.held)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',return_value={'pre_handoff':True}) as old:
            self.assertEqual(a.drain_owned(),{'pre_handoff':True});old.assert_called_once()

    def test_alarm_before_begin_call_has_both_refs_but_no_wait_owner(self):
        a,c,p,s,g=setup(self)
        with patch.object(p,'begin_wait_handoff',side_effect=bf.AlarmRaised('before acquire')):
            with self.assertRaises(bf.AlarmRaised):a.finish(a.owner,a.argv)
        self.assertIs(a.native_ports,p);self.assertIs(a.observer.ports,p)
        self.assertFalse(c._waitpid_lock.held)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',return_value={'pre_handoff':True}) as old:
            self.assertEqual(a.drain_owned(),{'pre_handoff':True});old.assert_called_once()

    def test_alarm_during_acquire_retains_both_refs_and_observer_cleanup(self):
        a,c,p,s,g=setup(self);original=c._waitpid_lock.acquire
        def acquire(blocking):
            result=original(blocking);p.signal.fire();return result
        c._waitpid_lock.acquire=acquire
        with self.assertRaises(bf.AlarmRaised):a.finish(a.owner,a.argv)
        self.assertIs(a.native_ports,p);self.assertIs(a.observer.ports,p)
        self.assertTrue(p.wait_lock_owned);self.assertTrue(c._waitpid_lock.held)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',side_effect=AssertionError('No old Popen wait')):
            result=a.drain_owned()
        self.assertTrue(result['physically_drained']);self.assertFalse(p.wait_lock_owned)
        self.assertFalse(c._waitpid_lock.held);self.assertEqual(c._waitpid_lock.releases,1)

    def test_non_alarm_ownership_uncertainty_never_uses_original_cleanup(self):
        a,c,p,s,g=setup(self);original=c._waitpid_lock.acquire
        def broken(blocking):original(blocking);raise RuntimeError('unexpected lock failure')
        c._waitpid_lock.acquire=broken
        with self.assertRaises(RuntimeError):a.finish(a.owner,a.argv)
        self.assertIs(a.native_ports,p);self.assertIs(a.observer.ports,p)
        self.assertTrue(p.wait_transition_uncertain)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',side_effect=AssertionError('No uncertain Popen cleanup')):
            with self.assertRaises(m.logic.ObservationError):a.drain_owned()
        self.assertTrue(c._waitpid_lock.held);self.assertIs(a.child,c)

    def test_alarm_during_return_preserves_actual_drain_without_double_release(self):
        a,c,p,s,g=setup(self);original=c._waitpid_lock.release
        def release():original();p.signal.fire()
        c._waitpid_lock.release=release
        with self.assertRaises(bf.AlarmRaised):a.finish(a.owner,a.argv)
        self.assertFalse(p.wait_lock_owned);self.assertFalse(c._waitpid_lock.held)
        with patch.object(m.original.SingleSDKAdapter,'drain_owned',side_effect=AssertionError('No old wait')):
            result=a.drain_owned()
        self.assertTrue(result['physically_drained']);self.assertEqual(c._waitpid_lock.releases,1)
        self.assertEqual(a.terminal['returncode'],0)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoordinationTests))
    assert all(name not in sys.modules for name in ('ctypes','_ctypes','torch','numpy'))
    print('Native-related imports blocked throughout; selected original AST + fake IO only.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
