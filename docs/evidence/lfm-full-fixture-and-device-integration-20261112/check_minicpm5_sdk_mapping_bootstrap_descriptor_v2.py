"""Bootstrap descriptor/guard source tests; every process/lock operation fake."""
import copy
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
class RejectNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('ctypes','_ctypes','torch','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('Forbidden import in bootstrap source test: '+name)
sys.meta_path.insert(0,RejectNative())
from research import minicpm5_sdk_mapping_bootstrap_descriptor_v2 as b
from research import check_descriptor_serial_lock_proof_v1 as df

class Ports:
    def __init__(self):
        self.fixture=df.Fixture();f=self.fixture
        f.lock=copy.deepcopy(b.ORIGINAL_LOCK);f.info.st_dev=f.lock['device'];f.info.st_ino=f.lock['inode']
        f.fields='FLOCK ADVISORY WRITE 123 %02x:%02x:%d 0 EOF' % (os.major(f.lock['device']),os.minor(f.lock['device']),f.lock['inode'])
        f.fdinfo='pos:\t0\nflags:\t%o\nmnt_id:\t31\nino:\t%d\nlock:\t1: %s\n' % (os.O_CLOEXEC|os.O_RDWR,f.lock['inode'],f.fields)
        f.global_text='28: '+f.fields+'\n30: '+f.fields+'\n'
        self.calls=[];self.now_value=100.;self.alarm=(0.0,0.0);self.handler=None
        self.affinity_value={0,1};self.nice_value=0;self.limits={};self.inheritable_value=False
        self.exe=(b.ORIGINAL_PYTHON,b.ORIGINAL_EXECUTABLE,b.ORIGINAL_EXECUTABLE)
        self.flag_value=(True,True,0);self.open_error=None;self.flock_error=None
        self.closed=[];self.locked=False;self.timer_calls=[];self.alarm_set_at=100.
    def lstat(self,path):
        if path==b.descriptor.LOCK_PATH:return copy.deepcopy(self.fixture.info)
        return NS(st_mode=stat.S_IFDIR|0o755)
    def stat(self,path):return copy.deepcopy(self.fixture.info)
    def readlink(self,path):return b.descriptor.LOCK_PATH
    def open_binary(self,path):return self.fixture.open(path)
    def open_lock(self):
        self.calls.append('open')
        if self.open_error:raise self.open_error
        return 9
    def fstat(self,fd):return copy.deepcopy(self.fixture.info)
    def inheritable(self,fd):return self.inheritable_value
    def flock(self,fd):
        self.calls.append('flock')
        if self.flock_error:raise self.flock_error
        self.locked=True
    def close(self,fd):self.closed.append(fd);self.locked=False
    def proc(self,pid):return copy.deepcopy(self.fixture.owner)
    def pid(self):return 123
    def uid(self):return 1000
    def executable(self):return self.exe
    def flags(self):return self.flag_value
    def now(self):return self.now_value
    def affinity(self):return self.affinity_value
    def set_affinity(self,value):self.affinity_value=set(value);self.calls.append('affinity')
    def nice(self):return self.nice_value
    def lower_priority(self,value):self.nice_value+=value
    def get_limit(self,name):return self.limits[name]
    def set_limit(self,name,values):self.limits[name]=values;self.calls.append(name)
    def get_alarm(self):return (max(0.,self.alarm[0]-(self.now_value-self.alarm_set_at)) if self.alarm[0] else 0.,self.alarm[1])
    def set_alarm(self,seconds):
        self.alarm=(seconds,0.0);self.alarm_set_at=self.now_value;self.timer_calls.append(seconds)
    def install_alarm_handler(self,handler):self.handler=handler
    def alarm_handler(self):return self.handler

def fixture(start=True):
    ports=Ports();owner=b.BootstrapDescriptor('a'*32,'trn1-mini-test-bootstrap',ports=ports)
    if start:owner.acquire();owner.impose_original_limits()
    return owner,ports

class Checks(unittest.TestCase):
    def test_original_flock_full_descriptor_proof_and_shared_deadline(self):
        owner,ports=fixture();binding=owner.binding
        self.assertEqual(binding['lock_identity'],b.ORIGINAL_LOCK);self.assertEqual(binding['retained_fd'],9)
        self.assertTrue(binding['initial_descriptor_proof']['passed'])
        self.assertEqual(binding['initial_descriptor_proof']['global']['target_row_count'],2)
        self.assertEqual(ports.calls[:2],['open','flock']);self.assertEqual(owner.deadline,1000.)
        self.assertEqual(ports.timer_calls,[900.]);self.assertEqual(ports.affinity_value,{6})
        self.assertEqual(ports.nice_value,19);self.assertEqual(ports.limits,b.BootstrapDescriptor._limit_pairs())
        for _ in range(3):owner()
        self.assertEqual(ports.timer_calls,[900.])

    def test_import_and_construction_have_no_runtime_operations(self):
        owner,ports=fixture(False);self.assertEqual(ports.calls,[]);self.assertIsNone(owner.fd)
        self.assertNotIn('minicpm5_python_trace_lifecycle_v4',sys.modules)

    def test_previous_timer_is_not_replaced(self):
        owner,ports=fixture(False);ports.alarm=(25.,0.)
        with self.assertRaises(Exception):owner.acquire()
        self.assertEqual(ports.calls,[]);self.assertEqual(ports.timer_calls,[])

    def test_wrong_inode_before_open_rejected(self):
        owner,ports=fixture(False);ports.fixture.info.st_ino+=1
        with self.assertRaises(Exception):owner.acquire()
        self.assertEqual(ports.calls,[])

    def test_nonblocking_acquire_failure_keeps_open_fd_for_explicit_cleanup(self):
        owner,ports=fixture(False);ports.flock_error=BlockingIOError('held elsewhere')
        with self.assertRaises(BlockingIOError):owner.acquire()
        self.assertEqual(owner.fd,9);self.assertEqual(ports.closed,[])
        owner.release_before_adoption(dict(physically_drained=True,child_started=False))
        self.assertEqual(ports.closed,[9])

    def test_no_child_drain_required_to_release(self):
        owner,ports=fixture()
        for proof in ({},{'physically_drained':True},{'physically_drained':True,'child_started':True}):
            with self.assertRaises(Exception):owner.release_before_adoption(proof)
        self.assertEqual(ports.closed,[])
        owner.release_before_adoption(dict(physically_drained=True,child_started=False))
        self.assertEqual(ports.timer_calls,[900.,0]);self.assertEqual(ports.closed,[9])
        with self.assertRaises(Exception):owner.release_before_adoption(dict(physically_drained=True,child_started=False))

    def test_no_second_acquire_or_limit_reset(self):
        owner,ports=fixture()
        with self.assertRaises(Exception):owner.acquire()
        with self.assertRaises(Exception):owner.impose_original_limits()
        self.assertEqual(ports.calls.count('flock'),1);self.assertEqual(ports.timer_calls,[900.])

    def test_elapsed_bootstrap_time_not_reset(self):
        owner,ports=fixture();ports.now_value=1001.
        with self.assertRaisesRegex(Exception,'deadline'):owner()
        ports.now_value=200.
        with self.assertRaisesRegex(Exception,'deadline'):owner()
        self.assertEqual(ports.timer_calls,[900.])

    def test_guard_rejects_changed_original_limits_affinity_priority_timer(self):
        changes=[lambda p:p.limits.update(RLIMIT_AS=(3*2**30,)*2),lambda p:setattr(p,'affinity_value',{5}),
            lambda p:setattr(p,'nice_value',0),lambda p:setattr(p,'alarm',(0,0)),
            lambda p:setattr(p,'alarm',(900,1)),lambda p:setattr(p,'handler',lambda *a:None)]
        for change in changes:
            with self.subTest(change=change):
                owner,ports=fixture();change(ports)
                with self.assertRaises(Exception):owner()

    def test_guard_rejects_wrong_interpreter_or_assertion_flags(self):
        for kind in ('exe','flags'):
            owner,ports=fixture()
            if kind=='exe':ports.exe=('python3','/wrong/python','/wrong/python')
            else:ports.flag_value=(False,True,0)
            with self.assertRaises(Exception):owner()

    def test_guard_rejects_descriptor_or_owner_changes(self):
        for change in (lambda p:setattr(p,'inheritable_value',True),
                       lambda p:p.fixture.owner.update(start_ticks='different'),
                       lambda p:setattr(p.fixture,'global_text','')):
            owner,ports=fixture();change(ports)
            with self.assertRaises(Exception):owner()

    def test_bootstrap_inputs_keep_exact_caps_owner_deadline(self):
        owner,ports=fixture();pins={'/mock/source.py':'a'*64};inputs=owner.bootstrap_inputs(pins)
        self.assertEqual(inputs['binding'],owner.binding);self.assertEqual(inputs['limits'],b.logic.LIMITS)
        self.assertEqual(inputs['artifact_limits'],b.logic.CAPS);self.assertEqual(inputs['outer_deadline'],owner.deadline)
        inputs['binding']['retained_fd']=11;self.assertEqual(owner.fd,9)

    def test_adoption_constructs_original_lease_on_same_descriptor_only(self):
        owner,ports=fixture();calls=[]
        class Lease:
            def __init__(self,binding):self.admission=copy.deepcopy(binding);calls.append('lease')
        class Guard:
            def __init__(self,lease):self.lease=lease;calls.append('guard')
            def __call__(self,*,force=False):calls.append(('original_guard',force))
        class Metadata:
            def __init__(self,guard):self.guard=guard;calls.append('metadata')
        original=NS(r=NS(ORIGINAL_LOCK=b.ORIGINAL_LOCK,LOCK_POLICY=b.LOCK_POLICY,
            validate_initial_descriptor=Mock(),Lease=Lease,Metadata=Metadata),original=NS(PVGuard=Guard))
        lease,guard,metadata=owner.adopt(original)
        self.assertEqual(lease.admission,owner.binding);self.assertIs(lease.metadata,metadata)
        self.assertIs(guard.lease,lease);self.assertEqual(calls,['lease','guard','metadata'])
        self.assertEqual(ports.calls.count('flock'),1);self.assertEqual(ports.timer_calls,[900.])
        with self.assertRaises(Exception):owner.release_before_adoption(dict(physically_drained=True,child_started=False))
        with self.assertRaises(Exception):owner.adopt(original)
        self.assertEqual(ports.closed,[])
        guard(force=True);self.assertIn(('original_guard',True),calls)
        ports.now_value=1001.
        with self.assertRaises(Exception):guard(force=True)

    def test_changed_original_policy_rejects_adoption(self):
        owner,ports=fixture();original=NS(r=NS(ORIGINAL_LOCK=b.ORIGINAL_LOCK,LOCK_POLICY={}))
        with self.assertRaises(Exception):owner.adopt(original)
        self.assertFalse(owner.adopted);self.assertEqual(ports.closed,[])

    def test_new_alarm_raises_original_timeout(self):
        owner,ports=fixture()
        with self.assertRaises(TimeoutError):ports.handler(14,None)
        self.assertEqual(ports.timer_calls,[900.])


    def test_same_handler_restarted_timer_is_rejected_against_absolute_deadline(self):
        owner,ports=fixture();ports.now_value=400.;ports.set_alarm(900.)
        self.assertIs(ports.handler,owner.alarm_callback)
        with self.assertRaises(Exception):owner()
        self.assertIsNotNone(owner.failed)

    def test_alarm_handler_failure_is_sticky_even_when_caught_before_deadline(self):
        owner,ports=fixture()
        with self.assertRaises(TimeoutError) as first:ports.handler(14,None)
        self.assertIs(owner.failed,first.exception)
        with self.assertRaises(TimeoutError) as second:owner()
        self.assertIs(second.exception,first.exception)
        with self.assertRaises(TimeoutError) as third:ports.handler(14,None)
        self.assertIs(third.exception,first.exception)

    def test_elapsed_timer_remaining_matches_original_absolute_deadline(self):
        owner,ports=fixture();ports.now_value=400.;owner()
        self.assertEqual(ports.get_alarm(),(600.,0.));self.assertEqual(owner.deadline,1000.)
        self.assertEqual(ports.timer_calls,[900.])

    def test_original_guard_constructors_do_not_restart_alarm_or_deadline(self):
        import ast
        locations=[('minicpm5_layer1_fixed_input_pv_adapter_v3.py','PVGuard'),
                   ('compare_minicpm5_layer1_saved_operators_v1.py','ComparisonGuard'),
                   ('audit_minicpm5_layer1_operator_native_result_v4.py','AuditGuard')]
        for filename,name in locations:
            tree=ast.parse((ROOT/'research'/filename).read_bytes())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
            for method in cls.body:
                if isinstance(method,ast.FunctionDef) and method.name=='__init__':
                    code=ast.unparse(method)
                    self.assertNotIn('deadline',code);self.assertNotIn('alarm',code)
                    self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                        and n.func.attr in ('setitimer','alarm','setrlimit') for n in ast.walk(method)))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    assert not any(name.split('.')[0] in ('ctypes','_ctypes','torch','numpy') for name in sys.modules)
    assert 'minicpm5_python_trace_lifecycle_v4' not in sys.modules
    print('Native/history imports rejected; all descriptor/process/resource/signal operations fake.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
