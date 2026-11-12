"""Source-only integration checks. Fake process, descriptor, timer and native ports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class RejectNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('ctypes','_ctypes','torch','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('Source-only integration forbids native import: '+name)


sys.meta_path.insert(0,RejectNative())
from research import check_minicpm5_sdk_mapping_bootstrap_descriptor_v2 as old
from research import check_minicpm5_sdk_mapping_nocache_entry_v5 as fixture
from research import minicpm5_sdk_mapping_bootstrap_descriptor_v4 as descriptor
from research import minicpm5_sdk_mapping_nocache_launch_v2 as launch
from research import minicpm5_sdk_mapping_nocache_entry_v5 as entry_source
from research import minicpm5_sdk_mapping_outer_source_entry_v3 as outer
from research import minicpm5_sdk_mapping_worker_guard_v2 as worker_guard
from research import minicpm5_sdk_mapping_worker_native_v2 as worker_native
from research import minicpm5_sdk_mapping_worker_source_entry_v3 as worker


class FakeEntry:
    def __init__(self,ports):
        self.started=40.;self.deadline=940.;self.role='outer312';self.calls=0;self.error=None
        self.plan=dict(python=descriptor.previous.ORIGINAL_PYTHON,executable=descriptor.previous.ORIGINAL_EXECUTABLE)
        self.ports=ports
    def check(self):
        self.calls+=1
        if self.error:raise self.error
        if self.ports.now_value>=self.deadline:raise TimeoutError('fake original deadline')


def descriptor_fixture():
    ports=old.Ports();entry=FakeEntry(ports)
    old_set=ports.set_alarm
    def checked_set(seconds):
        prior=ports.get_alarm();old_set(seconds);return prior
    ports.set_alarm=checked_set
    owner=descriptor.EntryBootstrapDescriptor(entry,'a'*32,'trn1-mini-source-fixture',ports=ports)
    return owner,ports,entry


def handoff_fixture(role='outer312'):
    e,p,s,x,plan,ref=fixture.fixture(role)
    class Auth:
        def load_ctypes(self):pass
        def load_history(self):pass
        def verify_loaded_native(self):pass
        def verify_loaded_history(self):pass
    class Guard:pass
    class Gate:pass
    class Fence:pass
    class Finder:
        def __init__(self,auth):self.auth=auth
    guard=Guard();guard.entry=e;guard.started=e.started;guard.deadline=e.deadline;guard.binding={'actual':'fake'}
    auth=Auth();auth.guard=guard;auth.failure=None;auth.native_receipt=None
    auth.bootstrap=dict(binding=guard.binding,outer_deadline=e.deadline,source_sha256=dict(plan['bootstrap_sources']))
    authority={k:plan['native_authority'][k] for k in ('path','sha256')}
    auth.authority=authority
    gate=Gate();gate.guard=guard;gate.state='new';gate.bootstrap=copy.deepcopy(auth.bootstrap)
    gate.authority=authority;gate.receipt=None
    gate.ctypes_loader=auth.load_ctypes;gate.history_loader=auth.load_history
    gate.verify_loaded_native=auth.verify_loaded_native;gate.verify_loaded_history=auth.verify_loaded_history
    gate.entry=e;gate.auth=auth
    gp=str(ROOT/'research/fake_gate.py');ap=str(ROOT/'research/fake_auth.py');bp=str(ROOT/'research/fake_guard.py')
    gm=NS(HistoricalImportGate=Gate,PreImportFence=Fence,__file__=gp)
    am=NS(NativeAuthenticator=Auth,SourceFinder=Finder,__file__=ap)
    bm=NS(EntryBootstrapDescriptor=Guard,__file__=bp)
    if role=='outer312':
        s.modules.update({'research.minicpm5_sdk_mapping_history_gate_v1':gm,
            'research.minicpm5_sdk_mapping_native_authenticator_v2':am,
            'research.minicpm5_sdk_mapping_bootstrap_descriptor_v4':bm})
    else:
        gm=NS(SDKHistoryGate=Gate,SDKNativeAuthenticator=Auth,NativeAuthenticator=Auth,
            PreImportFence=Fence,SourceFinder=Finder,__file__=ap)
        am=gm;bm=NS(SDKEntryGuard=Guard,__file__=bp)
        s.modules.update({'research.minicpm5_sdk_mapping_worker_native_v2':gm,
            'research.minicpm5_sdk_mapping_worker_guard_v2':bm})
    for path in (gp,ap,bp):
        row=dict(sha256='c'*64,bytes=12,identity=tuple(range(len(entry_source.FIELDS))))
        e.sources[path]=row;plan['bootstrap_sources'][path]='c'*64
        e.plan['bootstrap_sources'][path]='c'*64
        auth.bootstrap['source_sha256'][path]='c'*64
        gate.bootstrap['source_sha256'][path]='c'*64
    auth.reader=NS(verified={path:dict(sha256=row['sha256'],bytes=row['bytes'],
        identity=dict(zip(entry_source.FIELDS,row['identity']))) for path,row in e.sources.items()})
    return e,p,s,gate,auth,guard,gm,am


class Checks(unittest.TestCase):
    def test_descriptor_arms_only_remaining_original_entry_deadline(self):
        b,p,e=descriptor_fixture();b.acquire();b.impose_original_limits()
        self.assertEqual(b.started,40.);self.assertEqual(b.deadline,940.)
        self.assertEqual(p.timer_calls,[839.95])
        self.assertEqual(p.limits,b._limit_pairs())
        for _ in range(3):b()
        self.assertEqual(p.timer_calls,[839.95])

    def test_descriptor_deadline_drift_latches(self):
        b,p,e=descriptor_fixture();b.acquire();b.impose_original_limits()
        e.deadline+=1
        with self.assertRaises(ValueError) as first:b()
        e.deadline-=1
        with self.assertRaises(ValueError) as second:b()
        self.assertIs(first.exception,second.exception)

    def test_descriptor_entry_expired_before_limits_never_arms(self):
        b,p,e=descriptor_fixture();b.acquire();p.now_value=950.
        with self.assertRaises(TimeoutError):b.impose_original_limits()
        self.assertEqual(p.timer_calls,[])
        self.assertEqual(p.closed,[])

    def test_descriptor_original_timer_cannot_be_replaced_after_acquire(self):
        b,p,e=descriptor_fixture();b.acquire();p.alarm=(50.,0.)
        with self.assertRaises(ValueError):b.impose_original_limits()
        self.assertEqual(p.timer_calls,[])

    def test_descriptor_alarm_is_sticky_and_no_double_limits(self):
        b,p,e=descriptor_fixture();b.acquire();b.impose_original_limits()
        with self.assertRaises(Exception):b.impose_original_limits()
        with self.assertRaises(TimeoutError) as first:p.handler()
        with self.assertRaises(TimeoutError) as second:b()
        self.assertIs(first.exception,second.exception)

    def test_descriptor_sdk_role_and_fresh900_envelope_rejected(self):
        for role,deadline in [('sdk310',940.),('outer312',1000.)]:
            p=old.Ports();e=FakeEntry(p);e.role=role;e.deadline=deadline
            with self.assertRaises(ValueError):descriptor.EntryBootstrapDescriptor(e,'a'*32,'trn1-mini-source',ports=p)

    def test_outer_exact_history_handoff_and_original_fence(self):
        e,p,s,g,a,b,gm,am=handoff_fixture()
        e.bind_history_owner(g,a,b)
        self.assertIs(e.history_owner[0],g)
        g.state='authenticating';s.meta_path.insert(0,gm.PreImportFence())
        e.check();a.native_receipt={'authentic':'synthetic'}
        e.confirm_bootstrap_sources(a)
        self.assertTrue(e.native_authority_verified)
        g.receipt=dict(native_authority=copy.deepcopy(a.native_receipt))
        s.meta_path.pop(0);s.meta_path.insert(0,am.SourceFinder(a))
        e.check();self.assertTrue(e.native_import_allowed('_ctypes'))
        self.assertFalse(e.native_import_allowed('torch'))

    def test_source_handoff_rejects_wrong_authenticator_instance(self):
        e,p,s,g,a,b,gm,am=handoff_fixture();g.ctypes_loader=lambda:None
        with self.assertRaises(ValueError):e.bind_history_owner(g,a,b)

    def test_source_handoff_rejects_unexecuted_owner_source(self):
        e,p,s,g,a,b,gm,am=handoff_fixture();del e.sources[am.__file__]
        with self.assertRaises(ValueError):e.bind_history_owner(g,a,b)

    def test_auth_source_identity_drift_blocks_ctypes_transition(self):
        e,p,s,g,a,b,gm,am=handoff_fixture();e.bind_history_owner(g,a,b)
        g.state='authenticating';a.native_receipt={'authentic':'synthetic'}
        a.reader.verified[am.__file__]['identity']['st_ino']=999
        with self.assertRaises(ValueError):e.confirm_bootstrap_sources(a)
        self.assertFalse(e.native_authority_verified)

    def test_handoff_rejects_foreign_sourcefinder_or_duplicate_fence(self):
        e,p,s,g,a,b,gm,am=handoff_fixture();e.bind_history_owner(g,a,b)
        s.meta_path.insert(0,object())
        with self.assertRaises(ValueError):e.check()
        e,p,s,g,a,b,gm,am=handoff_fixture();e.bind_history_owner(g,a,b);g.state='authenticating'
        s.meta_path[:0]=[gm.PreImportFence(),gm.PreImportFence()]
        with self.assertRaises(ValueError):e.check()

    def test_sdk_history_uses_distinct_concrete_role_and_full_auth_before_torch(self):
        e,p,s,g,a,b,gm,am=handoff_fixture('sdk310')
        e.bind_sdk_history_owner(g,a,b)
        g.state='authenticating';a.native_receipt={'sdk':'distinct'}
        e.confirm_bootstrap_sources(a);g.receipt=dict(native_authority=a.native_receipt)
        s.meta_path.insert(0,am.SourceFinder(a))
        self.assertTrue(e.native_import_allowed('_ctypes'))
        self.assertFalse(e.native_import_allowed('torch'))
        g.state='complete_original_authentication'
        self.assertTrue(e.native_import_allowed('torch'))

    def test_sdk_plan_embedded_in_existing_admission(self):
        e,p,s,x,plan,ref=fixture.fixture('sdk310',load=False)
        plan['successor_input']='admission:self'
        admission=dict(schema='minicpm5_fixed_input_pv_sdk_import_diagnostic_admission_v1',
            source_entry_plan=plan,source_entry_binding=dict(outer_deadline=e.deadline))
        raw=launch.encoded(admission);p.files[fixture.PLAN_PATH]=raw
        ref=dict(path=fixture.PLAN_PATH,sha256=hashlib.sha256(raw).hexdigest())
        e.load_plan(ref,json)
        self.assertEqual(e.admission_envelope,admission)
        argv=launch.admitted_sdk(fixture.ENTRY,fixture.ENTRY_REF,admission,ref,e.deadline)
        self.assertEqual(argv[11],ref['sha256'])
        self.assertEqual(argv[-3],'admission:self')
        self.assertEqual(argv[-1],repr(e.deadline))

    def test_original_admission_hash_cannot_be_changed(self):
        plan=fixture.make_plan('sdk310');plan['successor_input']='admission:self'
        admission=dict(schema='minicpm5_fixed_input_pv_sdk_import_diagnostic_admission_v1',
            source_entry_plan=plan,source_entry_binding=dict(outer_deadline=750.))
        ref=dict(path=fixture.PLAN_PATH,sha256='0'*64)
        with self.assertRaises(ValueError):launch.admitted_sdk(fixture.ENTRY,fixture.ENTRY_REF,admission,ref,750.)

    def test_outer_concrete_load_history_order_uses_actual_bound_authenticator_methods(self):
        calls=[]
        class Auth:
            def __init__(self,guard):self.guard=guard;calls.append('auth-created')
            def authenticate_bootstrap(self,*args):calls.append('native-auth');return {'real':'receipt'}
            def verify_loaded_native(self,*args):pass
            def verify_loaded_history(self,*args):pass
            def load_ctypes(self):pass
            def load_history(self):pass
        class Gate:
            def __init__(self,inputs,authority,**kwargs):self.kw=kwargs;calls.append('gate-created')
            def load(self):self.kw['authenticate_bootstrap']({}, {}, b);calls.append('history-load');return 'original'
        e=NS(plan={'native_authority':{'path':'/a','sha256':'a'*64}},
            bind_history_owner=lambda *args:calls.append('bind'),
            confirm_bootstrap_sources=lambda auth:calls.append('executed-bytes-confirmed'))
        b=NS(bootstrap_inputs=lambda pins:{'pins':pins})
        with patch.object(outer.native,'NativeAuthenticator',Auth),patch.object(outer.history,'HistoricalImportGate',Gate):
            result=outer.load_history(e,b,{})
        self.assertEqual(calls,['auth-created','gate-created','bind','native-auth','executed-bytes-confirmed','history-load'])
        self.assertEqual(result[0],'original')

    def test_outer_pre_adoption_failure_releases_only_explicit_no_child(self):
        calls=[]
        class Descriptor:
            fd=9;adopted=False
            def __init__(self,*args):pass
            def acquire(self):calls.append('acquire')
            def impose_original_limits(self):raise RuntimeError('fake bounds failure')
            def release_before_adoption(self,proof):calls.append(proof)
        with patch.object(outer,'validate_inputs',return_value=({},dict(attempt_id='a'*32,unit='trn1-mini-test'),{})),\
             patch.object(outer.descriptor,'EntryBootstrapDescriptor',Descriptor):
            with self.assertRaisesRegex(RuntimeError,'bounds'):outer.source_entry(object(),'fake')
        self.assertEqual(calls,['acquire',dict(physically_drained=True,child_started=False)])

    def test_sdk_origin_cannot_reuse_outer_archive_receipt_schema(self):
        guard=NS(binding={},entry=NS(deadline=900))
        guard_type=type('Guard',(),{'__call__':lambda self:None})
        g=guard_type();g.binding={};g.entry=NS(deadline=900)
        reader=NS(json=lambda ref:dict(schema='minicpm5_sdk_mapping_native_source_review_v1'))
        auth=worker_native.SDKNativeAuthenticator(g,{},dict(path='/a',sha256='a'*64),reader=reader)
        with self.assertRaises((KeyError,ValueError)):auth.authenticate_bootstrap()
        self.assertIsNotNone(auth.failure)

    def test_integration_bodies_keep_original_gates_and_actual_lifecycle(self):
        ot=ast.parse(Path(outer.__file__).read_text());wt=ast.parse(Path(worker.__file__).read_text())
        attrs=lambda tree:{node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertTrue({'authenticate_original','SourceEntrySDKAdapter','finish','result','drain_owned',
            'release_after_physical_drain'}<=attrs(ot))
        self.assertTrue({'check_native_identities','restore_descriptor_plan','validate_plan','validate_spec',
            'execute_import','verify_loaded'}<=attrs(wt))
        self.assertNotIn('impose_original_limits',attrs(wt))
        self.assertNotIn('setrlimit',attrs(wt))
        self.assertNotIn('ctypes',sys.modules)
        self.assertNotIn('_ctypes',sys.modules)


if __name__=='__main__':unittest.main()
