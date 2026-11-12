"""Pure fake-ctypes/OS backend checks; no native import or real process IO."""
import ast
import copy
from collections import deque
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research import minicpm5_sdk_mapping_observer_backend_v1 as b
from research import minicpm5_sdk_mapping_observer_logic_v1 as logic

OWNER = dict(pid=101, ppid=99, start_ticks='200', cgroup='original', argv=['sdk','-B','worker'])
AUTH = {'path':'/future-authority.json','sha256':'a'*64}


class Scalar:
    def __init__(self,value=0): self.value=value


class Int(Scalar): pass
class Long(Scalar): pass
class ULong(Scalar): pass
class VoidP(Scalar): pass


class Buffer:
    def __init__(self,size): self.raw=b'\0'*size


class FakePtrace:
    def __init__(self): self.calls=[]; self.responses=deque(); self.restype=None; self.argtypes=None
    def __call__(self,request,pid,address,data):
        self.calls.append((request.value,pid.value,address.value,data))
        response=self.responses.popleft() if self.responses else 0
        if callable(response): return response(data)
        return response


class FakeCtypes:
    __file__=b.CTYPES_ROOT+'/__init__.py'
    c_int,c_long,c_ulong,c_void_p=Int,Long,ULong,VoidP
    def __init__(self):
        self.log=[]; self.errno=12; self.fn=FakePtrace(); self.lib=NS(_name=b.LIBC,ptrace=self.fn)
    def sizeof(self,kind):return 4 if kind is Int else 8
    def CDLL(self,name,**kw):self.log.append(('CDLL',name,kw)); return self.lib
    def set_errno(self,value):self.log.append(('set_errno',value))
    def get_errno(self):self.log.append(('get_errno',)); return self.errno
    def byref(self,value):return value
    def create_string_buffer(self,size):self.log.append(('buffer',size)); return Buffer(size)


class Lock:
    def __init__(self):self.held=False; self.releases=0
    def acquire(self,blocking):
        assert blocking is False
        if self.held:return False
        self.held=True;return True
    def release(self):assert self.held;self.held=False;self.releases+=1


class Stdin:
    def __init__(self):self.closed=False;self.fd=7
    def fileno(self):return self.fd
    def close(self):self.closed=True


class Child:
    def __init__(self):self.pid=101;self.returncode=None;self.stdin=Stdin();self._waitpid_lock=Lock()
    def poll(self):raise AssertionError('Popen poll forbidden')
    def wait(self,*a):raise AssertionError('Popen wait forbidden')


class FakeOS:
    def __init__(self):self.log=[];self.waits=deque();self.writes=deque();self.set_failure=None
    def set_blocking(self,fd,mode):
        self.log.append(('blocking',fd,mode))
        if self.set_failure:raise self.set_failure
    def waitpid(self,tid,flags):
        self.log.append(('waitpid',tid,flags))
        value=self.waits.popleft() if self.waits else (0,0)
        if isinstance(value,BaseException):raise value
        return value
    def write(self,fd,view):
        self.log.append(('write',fd,bytes(view)))
        value=self.writes.popleft() if self.writes else len(view)
        if isinstance(value,BaseException):raise value
        return value
    def kill(self,tid,sig):self.log.append(('kill',tid,sig))


def receipt():
    return dict(schema='mini_mapping_observer_native_authority_v1',authority=copy.deepcopy(AUTH),
        current_bytes_authenticated=True,current_identities_authenticated=True,pure_sources_authenticated=True,
        dependency_closure_verified=True,original_nine_native_checks_preserved=True,unresolved=[],
        outer_executable=b.OUTER_EXE,ctypes_root=b.CTYPES_ROOT,libc_path=b.LIBC,abi='linux-x86_64-lp64',
        evidence={'mock_only':'never a real native authority'})


def fixture():
    c=Child();ct=FakeCtypes();os=FakeOS();log=[];owners={101:copy.deepcopy(OWNER)}
    def auth(value,guard):log.append('authenticate');return receipt()
    def loader():log.append('load_ctypes');return ct
    def verify(*args):log.append('verify_loaded');return True
    kw=dict(authenticate=auth,verify_loaded_native=verify,inspect_owner=lambda tid:copy.deepcopy(owners.get(tid)),
        owned_group_members=lambda pid:[],ctypes_loader=loader,os_loader=lambda:os,
        time_loader=lambda:NS(monotonic=lambda:1.,sleep=lambda seconds:log.append(('idle',seconds))))
    def make():return b.make_ports(c,OWNER,AUTH,lambda:log.append('guard'),**kw)
    return make,kw,c,ct,os,log,owners


class NativePortTests(unittest.TestCase):
    def test_authentication_precedes_import_and_provider_check(self):
        make,kw,c,ct,os,log,owners=fixture();p=make()
        self.assertLess(log.index('authenticate'),log.index('load_ctypes'))
        self.assertLess(log.index('load_ctypes'),log.index('verify_loaded'))
        self.assertEqual(ct.log,[('CDLL',b.LIBC,{'use_errno':True})])
        self.assertEqual(ct.fn.calls,[]);self.assertTrue(c._waitpid_lock.held)
        self.assertEqual(os.log,[('blocking',7,False)])
        self.assertIs(ct.fn.restype,Long);self.assertEqual(ct.fn.argtypes,[Int,Int,VoidP,VoidP])
    def test_incomplete_authority_never_imports(self):
        for flag in ('current_bytes_authenticated','current_identities_authenticated','pure_sources_authenticated',
                     'dependency_closure_verified','original_nine_native_checks_preserved'):
            with self.subTest(flag=flag):
                make,kw,c,ct,os,log,owners=fixture()
                value=receipt();value[flag]=False;kw['authenticate']=lambda *a:value
                with self.assertRaises(logic.ObservationError):make()
                self.assertNotIn('load_ctypes',log);self.assertEqual(ct.fn.calls,[])
    def test_candidate_metadata_not_accepted_as_authority(self):
        make,kw,c,ct,os,log,owners=fixture();kw['authenticate']=lambda *a:{'actual_interpreter_authenticated':False}
        with self.assertRaises(logic.ObservationError):make()
        self.assertNotIn('load_ctypes',log)
    def test_failed_loaded_provider_blocks_seize(self):
        make,kw,c,ct,os,log,owners=fixture();kw['verify_loaded_native']=lambda *a:False
        with self.assertRaises(logic.ObservationError):make()
        self.assertFalse(c._waitpid_lock.held);self.assertEqual(ct.fn.calls,[])
    def test_nonblocking_failure_restores_preseize_wait_owner(self):
        make,kw,c,ct,os,log,owners=fixture();os.set_failure=ValueError('mock set_blocking')
        with self.assertRaises(ValueError):make()
        self.assertFalse(c._waitpid_lock.held);self.assertEqual(c._waitpid_lock.releases,1)
        self.assertEqual(ct.fn.calls,[])
    def test_competing_wait_owner_rejected(self):
        make,kw,c,ct,os,log,owners=fixture();c._waitpid_lock.held=True
        with self.assertRaises(logic.ObservationError):make()
        self.assertTrue(c._waitpid_lock.held);self.assertEqual(os.log,[])
    def test_seize_exact_options_and_interrupt(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();p.seize(101,logic.OPTIONS);p.interrupt(101)
        self.assertEqual(ct.fn.calls[0][:3],(0x4206,101,0))
        self.assertEqual(ct.fn.calls[0][3].value,b.OPTIONS_VALUE)
        self.assertEqual(ct.fn.calls[1][:3],(0x4207,101,0))
        with self.assertRaises(logic.ObservationError):p.seize(101,logic.OPTIONS)
    def test_ptrace_success_does_not_read_stale_errno(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();p.seize(101,logic.OPTIONS)
        self.assertNotIn(('get_errno',),ct.log)
    def test_ptrace_failure_reports_actual_errno(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();ct.fn.responses.append(-1);ct.errno=1
        with self.assertRaises(OSError) as caught:p.seize(101,logic.OPTIONS)
        self.assertEqual(caught.exception.errno,1);self.assertEqual(ct.log[-1],('get_errno',))
    def test_syscall_info_exact_available_length(self):
        make,kw,c,ct,os,log,owners=fixture();p=make()
        def result(buffer):buffer.raw=bytes(range(88));return 80
        ct.fn.responses.append(result);packet=p.syscall_info(101)
        self.assertEqual(packet,dict(data=bytes(range(80)),reported_length=80))
        self.assertEqual(ct.fn.calls[-1][:3],(0x420e,101,88))
    def test_larger_kernel_required_size_is_rejected(self):
        for size in (0,23,89,128):
            with self.subTest(size=size):
                make,kw,c,ct,os,log,owners=fixture();p=make();ct.fn.responses.append(size)
                with self.assertRaises(logic.ObservationError):p.syscall_info(101)
    def test_wait_only_exact_known_tid_and_preserves_status(self):
        make,kw,c,ct,os,log,owners=fixture();p=make()
        os.waits.extend([(101,0x857f),(101,256)])
        self.assertEqual(p.wait_known(101),(101,0x857f));self.assertEqual(p.waited_terminals,{})
        self.assertEqual(p.wait_known(101),(101,256));self.assertEqual(p.waited_terminals,{101:256})
        self.assertEqual(os.log[-1],('waitpid',101,0x40000001))
        for tid in (-1,102):
            with self.assertRaises(logic.ObservationError):p.wait_known(tid)
    def test_echild_and_eintr_preserved(self):
        make,kw,c,ct,os,log,owners=fixture();p=make()
        for error in (ChildProcessError(10,'ECHILD'),InterruptedError(4,'EINTR')):
            os.waits.append(error)
            with self.assertRaises(type(error)):p.wait_known(101)
        self.assertEqual(p.waited_terminals,{})
    def test_kernel_event_is_only_source_of_new_tid(self):
        make,kw,c,ct,os,log,owners=fixture();p=make()
        with self.assertRaises(logic.ObservationError):p.identity(102)
        def born(pointer):pointer.value=102;return 0
        ct.fn.responses.append(born);self.assertEqual(p.event_message(101),102)
        owners[102]=dict(OWNER,pid=102,start_ticks='201');self.assertEqual(p.identity(102),owners[102])
        self.assertEqual(set(p.known),{101,102})
    def test_gate_short_write_and_eagain_are_not_hidden(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();os.writes.extend([2,BlockingIOError()])
        self.assertEqual(p.write_gate(memoryview(b'abc')),2)
        with self.assertRaises(BlockingIOError):p.write_gate(memoryview(b'c'))
        p.close_gate();p.close_gate();self.assertTrue(c.stdin.closed)
        with self.assertRaises(logic.ObservationError):p.write_gate(memoryview(b'x'))
    def test_cleanup_signal_checks_exact_owner(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();p.signal_owned(OWNER,15)
        self.assertEqual(os.log[-1],('kill',101,15));owners[101]['start_ticks']='999'
        with self.assertRaises(logic.ObservationError):p.signal_owned(OWNER,9)
    def test_plain_sigtrap_forwarded_by_resume(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();p.resume(101,'syscall',5)
        self.assertEqual(ct.fn.calls[-1][:3],(24,101,0));self.assertEqual(ct.fn.calls[-1][3].value,5)
    def test_real_terminal_and_drain_release_only_wait_lock(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();os.waits.append((101,256));p.wait_known(101)
        c.returncode=1;p.close_gate();owners.pop(101)
        result=dict(physically_drained=True,sdk_terminal=dict(pid=101,wait_status=256,returncode=1,parent_observed_wait=True))
        p.finish_wait_handoff(result);self.assertFalse(c._waitpid_lock.held)
        self.assertFalse(p.wait_lock_owned);self.assertEqual(c._waitpid_lock.releases,1)
    def test_forged_or_unfinished_terminal_never_releases(self):
        make,kw,c,ct,os,log,owners=fixture();p=make();c.returncode=0;p.close_gate();owners.clear()
        result=dict(physically_drained=True,sdk_terminal=dict(pid=101,wait_status=0,returncode=0,parent_observed_wait=True))
        with self.assertRaises(logic.ObservationError):p.finish_wait_handoff(result)
        self.assertTrue(c._waitpid_lock.held)
        with self.assertRaises(logic.ObservationError):p.finish_wait_handoff({'physically_drained':False})
        self.assertTrue(c._waitpid_lock.held)
    def test_source_has_no_implicit_native_or_current_binding(self):
        tree=ast.parse(Path(b.__file__).read_text())
        top_imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertEqual([getattr(x,'module',None) or x.names[0].name for x in top_imports],['copy','research'])
        text=Path(b.__file__).read_text();self.assertNotIn('CDLL(None)',text.replace('No CDLL(None)',''))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr in ('poll','wait','Popen','killpg','find_library') for n in ast.walk(tree)))


class OriginalStoreMock:
    """Original aggregate/guard behavior is explicit; no filesystem access."""
    def __init__(self):
        self.rows={};self.logical_metadata=0;self.allocated=0;self.files={};self.guard_failed=False
    def inventory(self):
        if self.guard_failed:raise ValueError('sticky original guard')
        return dict(names=sorted(self.rows))
    def write(self,name,raw):
        self.inventory()
        if name in self.rows:raise ValueError('original immutable write')
        if self.logical_metadata+len(raw)>logic.CAPS['metadata_bytes']:raise ValueError('original metadata cap')
        if len(raw)>logic.CAPS['one_file_bytes']:raise ValueError('original one-file cap')
        self.rows[name]=(0,0,len(raw));self.files[name]=raw;self.logical_metadata+=len(raw)
        self.allocated+=((len(raw)+4095)//4096)*4096
        return {'mock_original_write':name}


def store_fixture():
    original=NS(Store=OriginalStoreMock,serial=NS(CAPS=copy.deepcopy(logic.CAPS)),
        RUNTIME_METADATA_CAP=79008,RUNTIME={'admission.json','phase-1-owner.json','phase-1-exit.json','phase-1-summary.json'})
    return b.make_store_class(original)()


class StoreTests(unittest.TestCase):
    def test_exact_six_names_and_original_writes(self):
        store=store_fixture();self.assertTrue(store.legal(logic.TRACE_NAME));self.assertTrue(store.legal(logic.RECEIPT_NAME))
        self.assertFalse(store.legal('extra.json'));store.write(logic.TRACE_NAME,b'raw\n')
        self.assertEqual(store.files[logic.TRACE_NAME],b'raw\n');self.assertEqual(store.logical_metadata,4)
    def test_new_receipt_counts_with_original_runtime(self):
        store=store_fixture();store.rows['admission.json']=(0,0,79000)
        with self.assertRaises(logic.ObservationError):store.write(logic.RECEIPT_NAME,b'x'*9)
        self.assertEqual(store.files,{})
    def test_original_runtime_write_counts_existing_new_receipt(self):
        store=store_fixture();store.rows[logic.RECEIPT_NAME]=(0,0,16000)
        store.rows['admission.json']=(0,0,63000)
        with self.assertRaises(logic.ObservationError):store.write('phase-1-exit.json',b'x'*9)
    def test_inventory_rejects_combined_runtime_excess(self):
        store=store_fixture();store.rows['admission.json']=(0,0,79000);store.rows[logic.RECEIPT_NAME]=(0,0,9)
        with self.assertRaises(logic.ObservationError):store.inventory()
    def test_summary_caps_and_metadata_totals(self):
        store=store_fixture()
        with self.assertRaises(logic.ObservationError):store.write(logic.RECEIPT_NAME,b'x'*16385)
        store.logical_metadata=logic.CAPS['metadata_bytes']-2
        with self.assertRaises(ValueError):store.write(logic.TRACE_NAME,b'abc')
    def test_sticky_original_guard_has_no_bypass(self):
        store=store_fixture();store.guard_failed=True
        with self.assertRaises(ValueError):store.write(logic.TRACE_NAME,b'abc')
        self.assertEqual(store.files,{})
    def test_full_reservation_against_original_metadata_and_allocation(self):
        store=store_fixture();extra=dict(trace_bytes=logic.TRACE_CAP,summary_bytes=logic.SUMMARY_CAP,
            runtime_metadata_bytes=79008,trace_name=logic.TRACE_NAME,receipt_name=logic.RECEIPT_NAME)
        store.validate_observer_budget(logic.LIMITS,logic.CAPS,extra)
        store.logical_metadata=logic.CAPS['metadata_bytes']-logic.TRACE_CAP
        with self.assertRaises(logic.ObservationError):store.validate_observer_budget(logic.LIMITS,logic.CAPS,extra)
        store.logical_metadata=0;store.allocated=logic.CAPS['allocated_bytes']-logic.TRACE_CAP
        with self.assertRaises(logic.ObservationError):store.validate_observer_budget(logic.LIMITS,logic.CAPS,extra)


if __name__=='__main__':unittest.main(verbosity=2)
