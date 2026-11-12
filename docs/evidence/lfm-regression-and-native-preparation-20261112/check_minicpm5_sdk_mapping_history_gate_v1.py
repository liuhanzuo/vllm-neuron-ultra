"""Source/bootstrap-order checks only; no original history or native loaded."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORBIDDEN = ('ctypes', '_ctypes', 'torch', 'numpy', 'neuronx_distributed',
             'neuronx_distributed_inference')
class RejectNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('No native imports in this source test: '+fullname)
assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
sys.meta_path.insert(0, RejectNative())
from research import minicpm5_sdk_mapping_history_gate_v1 as g
from research import check_minicpm5_sdk_mapping_observer_backend_v2 as bf


def fixture():
    events=[]; authority=copy.deepcopy(bf.AUTH)
    pins={g.HISTORY_PATH:'a'*64, str(Path(g.__file__).resolve()):'b'*64,
        str(Path(g.backend.__file__).resolve()):'c'*64,
        str(Path(g.backend.logic.__file__).resolve()):'d'*64,
        str(ROOT/'minicpm5_python_trace_lifecycle_v4.py'):'e'*64,
        str(ROOT/'research/original-inherited-history.py'):'f'*64}
    binding=dict(outer_owner={'pid':101},lock_identity=g.ORIGINAL_LOCK,retained_fd=9,
                 attempt_id='f'*32,other_original_fields='preserved')
    bootstrap=dict(schema='minicpm5_sdk_mapping_history_bootstrap_v1',binding=binding,
        limits=copy.deepcopy(g.backend.logic.LIMITS),artifact_limits=copy.deepcopy(g.backend.logic.CAPS),
        outer_deadline=900.0,source_sha256=pins)
    native=bf.receipt()
    receipt=dict(schema='minicpm5_sdk_mapping_history_bootstrap_receipt_v1',binding=copy.deepcopy(binding),
        outer_deadline=900.0,source_map_sha256=g.digest(pins),importable_sources_authenticated=True,
        original_descriptor_verified=True,original_limits_verified=True,native_authority=native)
    state={'receipt':receipt,'loaded':(),'native_ok':True,'history_ok':True}
    ct=bf.FakeCtypes()
    original=NS(__name__=g.HISTORY_NAME,__file__=g.HISTORY_PATH)
    full=dict(context={'pins_plan':{'source_sha256':copy.deepcopy(pins)}},
              witness={'complete_source_map_sha256':g.digest(pins)})
    original.r=NS(digest=g.digest)
    original.validate_spec=Mock(side_effect=lambda *a:events.append('original_validate_spec'))
    original.bind_authenticated=Mock(side_effect=lambda *a:events.append('original_bind_authenticated'))
    original.auth=NS(authenticate=Mock(side_effect=lambda *a,**k:events.append('original_full_auth') or full))
    original.run=Mock(side_effect=AssertionError('Never reacquire original lock through old run'))
    original.impose_original_limits=Mock(side_effect=AssertionError('Never reset original timer'))
    def guard():events.append('bootstrap_guard')
    def authenticate(boot,auth,passed_guard):
        events.append('authenticate_bootstrap');assert boot==bootstrap and auth==authority
        assert passed_guard is guard
        return copy.deepcopy(state['receipt'])
    def native_verify(actual_ct,libc,receipted,passed_guard):
        events.append('verify_loaded_native');assert actual_ct is ct and receipted==native
        return state['native_ok']
    def history_verify(actual,pinned,actual_ct,passed_guard):
        events.append('verify_loaded_history');assert actual is original and pinned==pins and actual_ct is ct
        return state['history_ok']
    def load_ct():events.append('load_ctypes');return ct
    def load_history():events.append('load_original_history');return original
    gate=g.HistoricalImportGate(bootstrap,authority,guard=guard,authenticate_bootstrap=authenticate,
        verify_loaded_native=native_verify,verify_loaded_history=history_verify,
        ctypes_loader=load_ct,history_loader=load_history,loaded_names=lambda:state['loaded'])
    spec_ref={'path':'/mock/spec','sha256':'1'*64};review_ref={'path':'/mock/review','sha256':'2'*64}
    spec={'pv_spec':{'path':'/mock/pv-spec','sha256':'3'*64},
          'pv_review':{'path':'/mock/pv-review','sha256':'4'*64}}
    review={'mock':'review'}
    metadata=NS(read=Mock(side_effect=lambda ref:spec if ref==spec_ref else review))
    lease=NS(admission=copy.deepcopy(binding),metadata=metadata)
    class Guard:
        def __init__(self):self.lease=lease
        def __call__(self,**kwargs):events.append('original_guard')
    full_guard=Guard()
    return NS(gate=gate,events=events,bootstrap=bootstrap,authority=authority,state=state,ct=ct,
        original=original,full=full,lease=lease,metadata=metadata,guard=full_guard,
        spec_ref=spec_ref,review_ref=review_ref,spec=spec,review=review)


def full_auth(f):
    return f.gate.authenticate_original(f.spec_ref,f.review_ref,lease=f.lease,
                                       metadata=f.metadata,guard=f.guard)


class HistoryGateTests(unittest.TestCase):
    def test_native_and_history_order_then_full_original_authentication(self):
        f=fixture();self.assertEqual(f.events,[]);self.assertIs(f.gate.load(),f.original)
        ordered=[x for x in f.events if x in ('authenticate_bootstrap','load_ctypes','verify_loaded_native',
                                              'load_original_history','verify_loaded_history')]
        self.assertEqual(ordered,['authenticate_bootstrap','load_ctypes','verify_loaded_native',
                                 'load_original_history','verify_loaded_history'])
        self.assertIs(full_auth(f),f.full)
        f.original.auth.authenticate.assert_called_once_with(f.spec['pv_spec'],f.spec['pv_review'],
            lease=f.lease,metadata=f.metadata,guard=f.guard)
        f.original.bind_authenticated.assert_called_once_with(f.spec,f.full)
        inputs=f.gate.adapter_inputs();self.assertIs(inputs['original'],f.original)
        self.assertIs(inputs['authenticated'],f.full);self.assertEqual(inputs['outer_deadline'],900)
        f.original.run.assert_not_called();f.original.impose_original_limits.assert_not_called()

    def test_default_import_is_inert_and_original_graph_remains_absent(self):
        self.assertNotIn(g.HISTORY_NAME,sys.modules);self.assertNotIn(g.LIFECYCLE_NAME,sys.modules)
        self.assertFalse(any(name.split('.')[0] in FORBIDDEN for name in sys.modules))

    def test_authentication_failure_never_loads_native_or_history(self):
        f=fixture();f.gate.authenticate_bootstrap=Mock(side_effect=ValueError('reject authority'))
        with self.assertRaises(ValueError):f.gate.load()
        self.assertNotIn('load_ctypes',f.events);self.assertNotIn('load_original_history',f.events)
        self.assertEqual(f.gate.state,'failed')
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()

    def test_invalid_bootstrap_receipt_rejected_before_native(self):
        for field,value in [('source_map_sha256','0'*64),('outer_deadline',1800),
                            ('importable_sources_authenticated',False),('original_descriptor_verified',False),
                            ('original_limits_verified',False)]:
            with self.subTest(field=field):
                f=fixture();f.state['receipt'][field]=value
                with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
                self.assertNotIn('load_ctypes',f.events)

    def test_nine_native_preservation_requirement_cannot_be_removed(self):
        f=fixture();f.state['receipt']['native_authority']['original_nine_native_checks_preserved']=False
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
        self.assertNotIn('load_ctypes',f.events)

    def test_already_imported_native_or_history_is_rejected(self):
        for name in ('ctypes','_ctypes','torch.nn',g.HISTORY_NAME,g.LIFECYCLE_NAME):
            with self.subTest(name=name):
                f=fixture();f.state['loaded']=(name,)
                with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
                self.assertEqual(f.events,[])

    def test_authenticator_cannot_import_ctypes_under_gate_fence(self):
        f=fixture();before=tuple(sys.meta_path)
        def wrong(*args):__import__('ctypes');raise AssertionError('unreachable')
        f.gate.authenticate_bootstrap=wrong
        with self.assertRaisesRegex(g.backend.logic.ObservationError,'Pre-authority'):f.gate.load()
        self.assertEqual(tuple(sys.meta_path),before);self.assertNotIn('ctypes',sys.modules)

    def test_authenticator_cannot_import_original_history(self):
        f=fixture();before=tuple(sys.meta_path)
        def wrong(*args):__import__(g.HISTORY_NAME);raise AssertionError('unreachable')
        f.gate.authenticate_bootstrap=wrong
        with self.assertRaisesRegex(g.backend.logic.ObservationError,'Pre-authority'):f.gate.load()
        self.assertEqual(tuple(sys.meta_path),before);self.assertNotIn(g.HISTORY_NAME,sys.modules)

    def test_guard_cannot_import_native_before_authentication(self):
        f=fixture()
        def wrong():__import__('_ctypes');raise AssertionError('unreachable')
        f.gate.guard=wrong
        with self.assertRaisesRegex(g.backend.logic.ObservationError,'Pre-authority'):f.gate.load()
        self.assertNotIn('_ctypes',sys.modules);self.assertNotIn('authenticate_bootstrap',f.events)

    def test_post_auth_preimport_observation_catches_unreported_load(self):
        f=fixture();original=f.gate.authenticate_bootstrap
        def changed(*args):
            value=original(*args);f.state['loaded']=('ctypes',);return value
        f.gate.authenticate_bootstrap=changed
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
        self.assertNotIn('load_ctypes',f.events)

    def test_native_provider_failure_never_loads_history(self):
        f=fixture();f.state['native_ok']=False
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
        self.assertNotIn('load_original_history',f.events);self.assertIs(f.gate.ctypes,f.ct)
        self.assertIs(f.gate.libc,f.ct.lib)

    def test_wrong_ctypes_path_never_opens_libc(self):
        f=fixture();f.ct.__file__='/wrong/ctypes/__init__.py'
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
        self.assertEqual(f.ct.log,[]);self.assertNotIn('load_original_history',f.events)

    def test_wrong_history_module_or_path_is_rejected(self):
        for field,value in [('__name__','alternate'),('__file__','/alternate.py')]:
            with self.subTest(field=field):
                f=fixture();setattr(f.original,field,value)
                with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
                self.assertNotIn('verify_loaded_history',f.events)

    def test_loaded_history_failure_keeps_original_auth_blocked(self):
        f=fixture();f.state['history_ok']=False
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
        with self.assertRaises(g.backend.logic.ObservationError):full_auth(f)
        f.original.auth.authenticate.assert_not_called()

    def test_no_adapter_inputs_before_complete_original_authentication(self):
        f=fixture()
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.adapter_inputs()
        f.gate.load()
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.adapter_inputs()

    def test_original_auth_failure_is_not_replaced_by_bootstrap_receipt(self):
        f=fixture();f.gate.load();f.original.auth.authenticate.side_effect=ValueError('original nine/history failure')
        with self.assertRaises(ValueError):full_auth(f)
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.adapter_inputs()
        self.assertEqual(f.gate.state,'failed')

    def test_no_original_source_pin_can_be_omitted_or_replaced(self):
        for kind in ('omit','replace','add'):
            with self.subTest(kind=kind):
                f=fixture();f.gate.load();pins=f.full['context']['pins_plan']['source_sha256']
                key=str(ROOT/'research/original-inherited-history.py')
                if kind=='omit':pins.pop(key)
                elif kind=='replace':pins[key]='0'*64
                else:pins['/unknown.py']='0'*64
                f.full['witness']['complete_source_map_sha256']=g.digest(pins)
                with self.assertRaises(g.backend.logic.ObservationError):full_auth(f)
                with self.assertRaises(g.backend.logic.ObservationError):f.gate.adapter_inputs()

    def test_old_witness_cannot_bind_a_different_full_map(self):
        f=fixture();f.gate.load();f.full['witness']['complete_source_map_sha256']='0'*64
        with self.assertRaises(g.backend.logic.ObservationError):full_auth(f)

    def test_descriptor_lease_cannot_be_reacquired_or_rebound(self):
        f=fixture();f.gate.load();f.lease.admission['retained_fd']=10
        with self.assertRaises(g.backend.logic.ObservationError):full_auth(f)
        f.original.auth.authenticate.assert_not_called()

    def test_original_shared_guard_and_metadata_required(self):
        for target in ('guard','metadata'):
            with self.subTest(target=target):
                f=fixture();f.gate.load()
                if target=='guard':f.guard.lease=NS(admission=f.lease.admission)
                else:f.lease.metadata=NS()
                with self.assertRaises(g.backend.logic.ObservationError):full_auth(f)
                f.original.auth.authenticate.assert_not_called()

    def test_exact_original_limits_must_be_preserved(self):
        f=fixture();f.bootstrap['limits']['wall_seconds']=1800
        with self.assertRaises(g.backend.logic.ObservationError):g.validate_bootstrap(f.bootstrap)

    def test_no_successful_retry_or_second_full_auth(self):
        f=fixture();f.gate.load()
        with self.assertRaises(g.backend.logic.ObservationError):f.gate.load()
        full_auth(f)
        with self.assertRaises(g.backend.logic.ObservationError):full_auth(f)

    def test_gate_source_has_no_history_rewrite_or_runtime_actions(self):
        source=Path(g.__file__).read_text();tree=ast.parse(source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden={'alarm','setitimer','flock','acquire','Popen','run','exec','compile','setattr'}
        self.assertFalse(any((isinstance(n.func,ast.Attribute) and n.func.attr in forbidden)
            or (isinstance(n.func,ast.Name) and n.func.id in forbidden) for n in calls))
        self.assertFalse(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript)
            and ast.unparse(t.value)=='sys.modules' for t in n.targets) for n in ast.walk(tree)))
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('ctypes' in ast.unparse(n) or 'pv_sdk_import_diagnostic' in ast.unparse(n) for n in imports))

    def test_saved_chain_matches_exact_sources_without_importing_them(self):
        import json
        p=ROOT/'research/minicpm5-sdk-mapping-historical-ctypes-import-chain-20261112-v1.json'
        record=json.loads(p.read_text());self.assertEqual(record['local_module_count'],233)
        self.assertEqual(record['local_edges_count'],697);self.assertEqual(len(record['native_top_level_imports']),1)
        row=record['native_top_level_imports'][0];self.assertEqual(row['line'],7)
        self.assertEqual(len(row['shortest_local_chain']),7)
        for item in record['nodes']:
            raw=Path(item['path']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])
        for edge in row['shortest_local_chain']:
            tree=ast.parse(Path(edge['source']).read_bytes())
            imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
            self.assertTrue(any(n.lineno==edge['line'] and ast.unparse(n)==edge['statement'] for n in imports))
        self.assertNotIn(g.HISTORY_NAME,sys.modules);self.assertNotIn(g.LIFECYCLE_NAME,sys.modules)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(HistoryGateTests))
    assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
    assert g.HISTORY_NAME not in sys.modules and g.LIFECYCLE_NAME not in sys.modules
    print('Native imports rejected; original history inspected only as source; all runtime ports fake.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
