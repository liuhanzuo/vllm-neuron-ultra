"""Real boundary producer/consumer methods; OS/native actions are never run.

Original Store methods use a disposable test directory and an explicit mock
guard. Original parent lifecycle methods are loaded as AST with mock process
ports, following the existing source-review tests. Saved r2 inputs are sizing
fixtures, not reconstructed unsaved r2 descriptor observations or new admission.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / 'research'
NATIVE = {'ctypes', '_ctypes', 'torch', 'numpy', 'neuronxcc', 'torch_neuronx',
          'neuronx_distributed', 'neuronx_distributed_inference'}

class BlockNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in NATIVE:
            raise AssertionError('No native import in source checks: ' + fullname)

assert not NATIVE.intersection(sys.modules)
sys.meta_path.insert(0, BlockNative())
from research import minicpm5_formal_literal_codec_v1 as codec
from research import minicpm5_formal_literal_result_v1 as results
from research import minicpm5_sdk_mapping_source_adapter_v2 as adapter
from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
from research import minicpm5_sdk_mapping_worker_source_entry_v6 as worker
from research import minicpm5_sdk_mapping_nocache_entry_v6 as frozen_entry
from research import minicpm5_sdk_mapping_outer_source_entry_v6 as outer

LITERAL = Path(codec.LITERAL['path']).read_text()
SOURCE_REF = {key: codec.LITERAL[key] for key in ('path', 'sha256')}
SIZING = []


def mock_native_authenticate(*args):
    """Explicit inert callback used only in source-method tests."""
    return None


def saved(name):
    return json.loads((Q / name).read_text())


def source_class(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def admission_fixture(output_root):
    """Actual saved owner/plan/spec references, synthetic unsaved dynamic proof.

    The witness is reconstructed from original ordered report references. The
    fd/proof/types/time values are mock fields; this is not an r2 runtime result.
    """
    prefix = 'minicpm5-formal-import-r2-20261112-'
    live = saved(prefix + 'root-live.json')
    spec = saved(prefix + 'outer-spec-final.json')
    plan = saved(prefix + 'sdk-entry-template-final.json')
    prep = saved(prefix + 'original-preparation.json')
    pv = saved(prefix + 'pv-spec.json')
    read = lambda ref: json.loads(Path(ref['path']).read_text())
    ended = read(pv['independent_audit_exit'])
    audit = read(ended['audit_spec'])
    comparison = read(audit['comparison_exit'])
    summary = read(comparison['result'])
    ref = lambda name: dict(path=str(Q / name), sha256=hashlib.sha256((Q / name).read_bytes()).hexdigest())
    witness = dict(schema='minicpm5_fixed_input_pv_prerequisite_witness_v1',
        spec=ref(prefix + 'pv-spec.json'), review=ref(prefix + 'pv-review-accepted.json'),
        independent_audit_exit=pv['independent_audit_exit'], independent_audit_result=ended['result'],
        comparison_exit=audit['comparison_exit'], comparison_summary=comparison['result'],
        comparison_reports=[row['report'] for row in summary['reports']],
        descriptor_plan_sha256=prep['descriptor_plan_sha256'],
        complete_source_map_sha256=prep['complete_source_map_sha256'],
        complete_source_files=prep['complete_source_files'], current_serial_catalog=pv['current_serial_catalog'],
        original_sdk_model_exit_code=1, source_history_and_catalog_reconstructed=True,
        raw_capture_bytes_read=0, full_model_qualified=False, new_speedup_claim=False)
    owner = live['owner']
    # No current descriptor proof is obtained or invented as actual evidence.
    proof = dict(owner=copy.deepcopy(owner), test_fixture_only=True, full_unknown_r2_observation_not_reconstructed=True)
    return dict(schema='minicpm5_fixed_input_pv_sdk_import_diagnostic_admission_v1', witness=witness,
        lease=dict(outer_owner=owner, initial_descriptor_proof=proof, retained_fd=4,
                   lock_identity=dict(device=66306, inode=8294421), lock_policy={'test_fixture_only': True},
                   unit='source-mock', attempt_id='a' * 32), output_root=str(output_root),
        original_native_source_receipt=dict(path='/mock/original-nine.json', sha256='a' * 64),
        diagnostic_spec=spec['diagnostic_spec'], diagnostic_review=spec['diagnostic_review'],
        source_entry_plan=plan, source_entry_binding=dict(outer_spec=ref(prefix + 'outer-spec-final.json'),
            outer_review=ref(prefix + 'outer-review-accepted.json'),
            outer_entry_plan=ref(prefix + 'outer-entry-plan-final.json'), outer_started=100.0, outer_deadline=1000.0,
            outer_types_source_observation={'test_fixture_only': True},
            original_worker='/mock/original-worker.py', sdk_entry_source=SOURCE_REF))


def fixture_original():
    logic = adapter.observed.logic
    require = logic.require
    serial = NS(encoded=codec.encoded, CAPS=logic.CAPS, exact=lambda a, b: a == b,
                python_for=lambda i: '/home/ubuntu/venvs/vllm_neuron/bin/python')
    core = NS(sha=lambda raw: hashlib.sha256(raw).hexdigest())
    base_path = Q / 'minicpm5_layer1_fixed_input_pv_adapter_v3.py'
    namespace = dict(copy=copy, Path=Path, stat=stat, os=os, time=time, json=json, re=__import__('re'),
        serial=serial, core=core, require=require, RUNTIME_METADATA_CAP=79008,
        RUNTIME={'admission.json', 'phase-1-owner.json', 'phase-1-exit.json', 'phase-1-summary.json'},
        ref=lambda raw, path: dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    Store = source_class(base_path, 'Store', namespace)
    base = NS(Store=Store, serial=serial, RUNTIME_METADATA_CAP=79008, RUNTIME=namespace['RUNTIME'],
              short=lambda x: {key: x[key] for key in ('path', 'sha256')})
    r = NS(p=NS(gone=Mock(), proc=Mock(), environment=Mock(return_value={})),
           group_members=Mock(return_value=[]), life=NS(process=Mock()), verify_loaded=Mock())
    original = NS(r=r, serial=serial, require=require, original=base, SUMMARY_CAP=16384,
        SUMMARY_SCHEMA='minicpm5_fixed_input_pv_sdk_import_diagnostic_summary_v1',
        RELEASE_SCHEMA='minicpm5_fixed_input_pv_sdk_import_diagnostic_release_v1',
        ROOT=ROOT, PIPE_CAP=16 * 2**20, subprocess=NS(Popen=Mock(), PIPE=-1))
    source = Q / 'minicpm5_pv_sdk_import_diagnostic_v1.py'
    original.SingleSDKAdapter = source_class(source, 'SingleSDKAdapter',
        dict(copy=copy, r=r, require=require, serial=serial, sys=sys, os=os, SUMMARY_CAP=16384))
    return original


def terminal(owner, code=0):
    return dict(schema='minicpm5_pv_sdk_import_diagnostic_actual_child_exit_v1', phase=1,
        owner=owner, argv=owner['argv'], child_pid=owner['pid'], returncode=code,
        parent_observed_wait=True, process_group_empty=True, exact_owner_gone=True,
        owner_acknowledged=True, self_exit_claimed=False, cleanup_after_failure=False)


def actual_metadata_with_mock_sources(guard, values):
    """Original Metadata methods with explicit in-memory input/read-only temp IO."""
    def raw(path):
        return codec.encoded(values[path]) if path in values else Path(path).read_bytes()
    def identity(path):
        return hashlib.sha256(raw(path)).hexdigest()
    def load(ref):
        data = raw(ref['path'])
        codec.require(hashlib.sha256(data).hexdigest() == ref['sha256'], 'Mock source full hash')
        return codec.parsed(data)
    sources = NS(reference=codec.reference, load=Mock(side_effect=load))
    cls = source_class(Q / 'minicpm5_layer1_operator_native_process_adapter_v7.py', 'Metadata',
                       dict(sources=sources, identity=identity, require=codec.require))
    return cls(guard), sources


class Fixture(unittest.TestCase):
    def setUp(self):
        self.contexts = contextlib.ExitStack()
        self.addCleanup(self.contexts.close)
        self.temp = self.contexts.enter_context(tempfile.TemporaryDirectory(prefix='mini-wire-source-mock-'))
        self.original = fixture_original()
        self.classes = adapter.make_adapter_class(self.original)
        self.guard = Mock()
        self.store = self.classes.ObserverStore(Path(self.temp) / 'output', self.guard, literal=LITERAL)
        self.store.create()
        self.value = admission_fixture(self.store.root)
        self.raw, self.ref = self.store.prospective_admission(self.value)
        self.argv = launch.admitted_sdk(LITERAL.encode(), SOURCE_REF, self.value, self.ref, 1000.0)
        self.owner = dict(pid=23456, ppid=self.value['lease']['outer_owner']['pid'], start_ticks='123456',
                          cgroup=self.value['lease']['outer_owner']['cgroup'], argv=self.argv)

    def admitted(self):
        budget = self.store.reserve_runtime(self.value, self.ref, self.argv)
        actual = self.store.json('admission.json', self.value)
        self.assertEqual({k: actual[k] for k in ('path', 'sha256')}, self.ref)
        self.store.before_spawn(self.value, self.ref, self.argv)
        return budget

    def sdk_adapter(self, *, complete_prerequisites=False):
        lease = NS(admission=self.value['lease'])
        guard = Mock(); guard.lease = lease
        self.store.guard = guard
        authenticated = {'spec': {'worker': '/mock/original-worker.py'}, 'context': {'pins_plan': {
            'source_contract': {'test': True}, 'source_sha256': {
                self.value['source_entry_plan']['entry_path']: self.value['source_entry_plan']['bootstrap_sources'][self.value['source_entry_plan']['entry_path']]}}}}
        metadata = NS(read=Mock(side_effect=lambda ref: codec.parsed(self.store.read(self.store.reference('admission.json')))),
                      unchanged=Mock(), identities={})
        lease.metadata = metadata
        wrapped = codec.AdmissionMetadata(metadata, self.ref, LITERAL, guard)
        item = self.classes.SourceEntrySDKAdapter(authenticated, self.ref, self.value, lease, guard,
            wrapped, self.store, outer_deadline=1000.0, authority={'test_fixture_only': True},
            authenticate=mock_native_authenticate, verify_loaded_native=mock_native_authenticate,
            sdk_entry_bytes=LITERAL.encode(), sdk_entry_ref=SOURCE_REF)
        item.observer_guard = Mock()
        item.prerequisites_checked = not complete_prerequisites
        if complete_prerequisites:
            self.original.__file__ = str(Q / 'minicpm5_pv_sdk_import_diagnostic_v1.py')
            pins = authenticated['context']['pins_plan']['source_sha256']
            for path in (self.original.__file__, adapter.observed.__file__, self.classes.backend.__file__,
                         self.classes.logic.__file__, __file__):
                pins[str(Path(path).resolve())] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            self.original.r.sources = NS(text=lambda path, sha: Path(path).read_text())
            self.original.r.digest = lambda value: hashlib.sha256(codec.encoded(value)).hexdigest()
            self.original.prior = NS(LIMITS=self.classes.logic.LIMITS)
            authenticated['context']['plan'] = {'synthetic_descriptor_plan': True}
            authenticated['witness'] = self.value['witness']
            authenticated['witness']['complete_source_map_sha256'] = self.original.r.digest(pins)
            authenticated['witness']['descriptor_plan_sha256'] = self.original.r.digest(authenticated['context']['plan'])
            self.original.r.p.proc.return_value = self.value['lease']['outer_owner']
        return item

    def test_real_outer_inherited_prerequisites_decodes_after_original_read(self):
        item = self.sdk_adapter(complete_prerequisites=True)
        # Fixture witness changed to bind the mock complete map; reserve only now.
        self.raw, self.ref = self.store.prospective_admission(self.value)
        self.argv = launch.admitted_sdk(LITERAL.encode(), SOURCE_REF, self.value, self.ref, 1000.0)
        item.admission_ref = self.ref
        item.metadata.admission_ref = self.ref
        self.admitted()
        with patch.object(adapter.observed.signal, 'getitimer', return_value=(800, 0)), patch.object(
                adapter.observed.time, 'monotonic', return_value=100.0):
            item.authenticate_complete_prerequisites()
        self.assertTrue(item.prerequisites_checked)
        item.lease.metadata.read.assert_called_once_with(self.ref)
        self.assertIs(item.metadata.original, item.lease.metadata)
        self.assertEqual(item.admission['lease']['outer_owner']['argv'][7], LITERAL)
        # Exact inherited check must fail when deliberately passed raw wire.
        item.metadata = item.lease.metadata
        with self.assertRaises(Exception):
            self.original.SingleSDKAdapter.authenticate_complete_prerequisites(item)

    def test_frozen_stage0_load_plan_then_real_worker_decode(self):
        self.admitted()
        plan = self.value['source_entry_plan']
        def fail(error):
            raise error
        entry = NS(check=Mock(), state='fenced', plan=None, role='sdk310', admission_envelope=None,
            runtime=NS(executable=plan['python'], orig_argv=self.argv), entry_sha256=codec.LITERAL['sha256'],
            ports=NS(readlink=Mock(return_value=plan['executable'])), fail=fail,
            read_source=Mock(return_value=(self.store.read(self.store.reference('admission.json')), {})))
        loaded = frozen_entry.Entry.load_plan(entry, self.ref, json)
        self.assertEqual(loaded, plan)
        self.assertEqual(entry.state, 'plan_verified')
        self.assertIn(codec.WIRE, entry.admission_envelope)
        entry.read_source.assert_called_once_with(self.ref['path'], self.ref['sha256'], frozen_entry.PLAN_CAP)
        class ReachedGuard(Exception): pass
        def guard(actual_entry, admission):
            self.assertEqual(admission, self.value)
            self.assertEqual(actual_entry.runtime.orig_argv, self.argv)
            raise ReachedGuard()
        with patch.object(worker.startup, 'SDKEntryGuard', side_effect=guard), patch.object(
                worker.native, 'SDKNativeAuthenticator', side_effect=AssertionError('native constructed')):
            with self.assertRaises(ReachedGuard): worker.source_entry(entry, 'admission:self')

    def test_actual_outer_producer_original_prerequisites_and_frozen_sdk_consumer(self):
        configured = self.sdk_adapter(complete_prerequisites=True)
        a, guard, lease = configured.authenticated, configured.guard, configured.lease
        pins = a['context']['pins_plan']['source_sha256']
        worker_path = str(Q / 'minicpm5_sdk_mapping_worker_source_entry_v6.py')
        pins[worker_path] = hashlib.sha256(Path(worker_path).read_bytes()).hexdigest()
        pins[SOURCE_REF['path']] = SOURCE_REF['sha256']
        plan = dict(self.value['source_entry_plan'], entry_path=worker_path,
                    entry_module='research.minicpm5_sdk_mapping_worker_source_entry_v6', bootstrap_sources=dict(pins))
        a['witness']['complete_source_map_sha256'] = self.original.r.digest(pins)
        a['output_root'] = str(Path(self.temp) / 'outer-output')
        values = {}
        def input_ref(name, value):
            path = str(Path(self.temp) / name)
            values[path] = value
            return dict(path=path, sha256=hashlib.sha256(codec.encoded(value)).hexdigest())
        a['context']['original_spec'] = dict(model_exit=input_ref('model.json',
            {'outputs': {'source_authentication': self.value['original_native_source_receipt']}}))
        spec = dict(diagnostic_spec=self.value['diagnostic_spec'], diagnostic_review=self.value['diagnostic_review'],
            attempt_id=lease.admission['attempt_id'], unit=lease.admission['unit'],
            complete_source_map=input_ref('map.json', pins), source_extension=dict(pins),
            sdk_entry_template=input_ref('sdk-plan.json', plan), sdk_entry_source=SOURCE_REF)
        review = {'test_fixture_only': True}
        refs = dict(spec=input_ref('outer-spec.json', spec), review=input_ref('outer-review.json', review))
        metadata, source_ports = actual_metadata_with_mock_sources(guard, values)
        lease.metadata = metadata
        original_outer_plan = saved('minicpm5-formal-import-r2-20261112-outer-entry-plan-final.json')
        entry = NS(plan=dict(original_outer_plan, bootstrap_sources=dict(pins)),
            plan_reference=self.value['source_entry_binding']['outer_entry_plan'], started=100.0, deadline=1000.0,
            types_observation={'test_fixture_only': True})
        bootstrap = NS(fd=4, adopted=True, binding=lease.admission, acquire=Mock(), impose_original_limits=Mock(),
                       adopt=Mock(return_value=(lease, guard, metadata)))
        reader = NS(json=Mock(return_value=pins), unchanged=Mock())
        authenticator = NS(reader=NS(read=Mock(return_value=(LITERAL.encode(), {}))),
            authenticate=mock_native_authenticate, verify_loaded_native=mock_native_authenticate)
        gate = NS(authenticate_original=Mock(return_value=a), authority={'test_fixture_only': True})
        self.original.ADMISSION_SCHEMA = self.value['schema']
        self.original.r.p.user_bus = Mock()
        released = Mock()
        self.original.prior.audited = NS(support=NS(release_after_physical_drain=released))
        captured = {}
        real_class = self.classes.SourceEntrySDKAdapter
        class Captured(real_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                captured['ports'] = self
        classes = NS(ObserverStore=self.classes.ObserverStore, SourceEntrySDKAdapter=Captured)
        class StopAtPopen(Exception): pass
        self.original.subprocess.Popen.side_effect = StopAtPopen()
        with patch.object(outer, 'validate_inputs', return_value=(refs, spec, review)), patch.object(
                outer.descriptor, 'EntryBootstrapDescriptor', return_value=bootstrap), patch.object(
                outer.native, 'StableReader', return_value=reader), patch.object(
                outer, 'load_history', return_value=(self.original, gate, authenticator)), patch.object(
                outer.native, 'SourceFinder', side_effect=lambda unused: contextlib.nullcontext()), patch.object(
                outer, '_complete_source_map', return_value=pins), patch.object(
                outer.adapter, 'make_adapter_class', return_value=classes), patch.object(
                adapter.observed.signal, 'getitimer', return_value=(800, 0)), patch.object(
                adapter.observed.signal, 'alarm'), patch.object(adapter.observed.time, 'monotonic', return_value=100.0):
            with self.assertRaises(StopAtPopen): outer.source_entry(entry, 'mock input')
        ports = captured['ports']
        self.assertTrue(ports.prerequisites_checked)
        self.assertIs(lease.metadata, metadata)
        self.assertIs(ports.metadata.original, metadata)
        self.assertEqual(ports.argv, self.original.subprocess.Popen.call_args.args[0])
        self.assertEqual(ports.argv[7], LITERAL)
        self.assertEqual(ports.store.inventory()['names'], ['admission.json'])
        self.assertIn(ports.admission_ref['path'], metadata.identities)
        released.assert_called_once()
        def fail(error): raise error
        stage0 = NS(check=Mock(), state='fenced', plan=None, role='sdk310', admission_envelope=None,
            runtime=NS(executable=plan['python'], orig_argv=ports.argv), entry_sha256=codec.LITERAL['sha256'],
            ports=NS(readlink=Mock(return_value=plan['executable'])), fail=fail,
            read_source=Mock(return_value=(ports.store.read(ports.store.reference('admission.json')), {})))
        self.assertEqual(frozen_entry.Entry.load_plan(stage0, ports.admission_ref, json), plan)
        class StopBeforeGuard(Exception): pass
        def sdk_guard(e, admission):
            self.assertEqual(admission, ports.admission)
            raise StopBeforeGuard()
        with patch.object(worker.startup, 'SDKEntryGuard', side_effect=sdk_guard), patch.object(
                worker.native, 'SDKNativeAuthenticator', side_effect=AssertionError('native constructed')):
            with self.assertRaises(StopBeforeGuard): worker.source_entry(stage0, 'admission:self')

    def test_actual_metadata_hash_identity_and_guard_precede_decode(self):
        self.admitted()
        original, sources = actual_metadata_with_mock_sources(self.guard, {})
        view = codec.AdmissionMetadata(original, self.ref, LITERAL, self.guard)
        self.assertEqual(view.read(self.ref), self.value)
        self.assertIs(view.identities, original.identities)
        original.unchanged()
        with patch.object(codec, 'decode', side_effect=AssertionError('decode reached')):
            with self.assertRaises(ValueError): view.read(dict(self.ref, sha256='0' * 64))
        with patch.object(sources, 'load', side_effect=OSError('original stable IO failed')):
            with self.assertRaises(OSError): view.read(self.ref)

    def test_original_store_roundtrip_full_owner_and_actual_wire_hash(self):
        budget = self.admitted()
        self.assertGreater(len(codec.encoded(self.value)), 79008)
        self.assertLess(len(self.raw), 79008)
        decoded, ref = self.store.load('admission.json')
        self.assertEqual(decoded, self.value)
        self.assertNotIn(codec.WIRE, self.value)
        self.assertEqual(ref['sha256'], hashlib.sha256(self.raw).hexdigest())
        self.store.json('phase-1-owner.json', self.owner)
        ended = terminal(self.owner)
        self.store.json('phase-1-exit.json', ended)
        for name, expected in [('phase-1-owner.json', self.owner), ('phase-1-exit.json', ended)]:
            self.assertEqual(self.store.load(name)[0], expected)
        sizes = {name: self.store.reference(name)['bytes'] for name in self.store.inventory()['names']}
        SIZING.append(dict(scope='saved r2 owner/plan/witness plus explicitly synthetic dynamic proof/time/SDK owner fixture',
                           expanded_admission_bytes=len(codec.encoded(self.value)), actual_wire_file_bytes=sizes,
                           complete_pre_spawn_reservation=budget))

    def test_mismatched_literal_rejected(self):
        with self.assertRaises(ValueError):
            codec.encode('admission', self.value, LITERAL + ' ')
        changed = copy.deepcopy(self.value)
        changed['lease']['outer_owner']['argv'][7] += ' '
        with self.assertRaises(ValueError):
            codec.encode('admission', changed, LITERAL)

    def test_expanded_hash_and_wire_hash_independent(self):
        wire = codec.parsed(self.raw)
        with self.assertRaises(ValueError):
            codec.decode('admission', wire, LITERAL, wire_sha256='0' * 64)
        wire[codec.WIRE]['expanded_sha256'] = '0' * 64
        raw = codec.encoded(wire)
        with self.assertRaises(ValueError):
            codec.decode('admission', raw, LITERAL, wire_sha256=codec.digest(raw))

    def test_unknown_fields_pointer_and_partial_recovery_rejected(self):
        for location in ('metadata', 'node', 'admission', 'outside', 'missing'):
            wire = codec.parsed(self.raw)
            if location == 'metadata': wire[codec.WIRE]['extra'] = True
            elif location == 'node': wire['lease']['outer_owner']['argv'][7][codec.NODE]['path'] = '/elsewhere.py'
            elif location == 'admission': wire['unknown'] = True
            elif location == 'outside': wire['witness']['unexpected'] = {codec.NODE: dict(codec.LITERAL)}
            else: wire['lease']['outer_owner']['argv'][7] = LITERAL
            raw = codec.encoded(wire)
            with self.subTest(location=location), self.assertRaises(ValueError):
                codec.decode('admission', raw, LITERAL, wire_sha256=codec.digest(raw))

    def test_duplicate_and_noncanonical_json_rejected(self):
        for raw in (b'{"a":1,"a":1}\n', b'{ "a":1 }\n', b'{"a":NaN}\n'):
            with self.assertRaises(ValueError): codec.parsed(raw)

    def test_real_worker_decodes_before_sdk_guard_and_native_owner(self):
        entry = NS(role='sdk310', admission_envelope=codec.parsed(self.raw), plan_reference=self.ref,
                   plan=self.value['source_entry_plan'], runtime=NS(orig_argv=self.argv), check=Mock())
        class StopBeforeGuard(Exception): pass
        def guard(e, admission):
            self.assertEqual(admission, self.value)
            self.assertEqual(e.admission_envelope, self.value)
            self.assertEqual(e.runtime.orig_argv, self.argv)
            self.assertIn(codec.WIRE, e.literal_wire_admission)
            raise StopBeforeGuard()
        with patch.object(worker.startup, 'SDKEntryGuard', side_effect=guard), patch.object(
                worker.native, 'SDKNativeAuthenticator', side_effect=AssertionError('native constructed')):
            with self.assertRaises(StopBeforeGuard): worker.source_entry(entry, 'admission:self')

    def test_real_worker_rejects_wire_hash_before_guard(self):
        entry = NS(role='sdk310', admission_envelope=codec.parsed(self.raw),
            plan_reference=dict(self.ref, sha256='0' * 64), plan=self.value['source_entry_plan'],
            runtime=NS(orig_argv=self.argv), check=Mock())
        with patch.object(worker.startup, 'SDKEntryGuard', side_effect=AssertionError('guard constructed')):
            with self.assertRaises(ValueError): worker.source_entry(entry, 'admission:self')

    def test_original_metadata_read_then_exact_one_reference_decode(self):
        original = NS(read=Mock(return_value=codec.parsed(self.raw)), unchanged=Mock(), identities={'retained': 'original'})
        view = codec.AdmissionMetadata(original, self.ref, LITERAL, self.guard)
        self.assertEqual(view.read(self.ref), self.value)
        original.read.assert_called_once_with(self.ref)
        self.assertIs(view.identities, original.identities)
        original.read.return_value = {'ordinary': True}
        self.assertEqual(view.read({'path': '/other', 'sha256': 'b' * 64}), {'ordinary': True})
        original.read.side_effect = IOError('original stable read failed')
        with self.assertRaises(IOError): view.read(self.ref)

    def test_whole_reservation_rejects_before_real_start_popen(self):
        item = self.sdk_adapter()
        with self.assertRaises(Exception): item.start()
        self.original.subprocess.Popen.assert_not_called()
        changed = copy.deepcopy(self.value)
        changed['lease']['initial_descriptor_proof']['complete_test_padding'] = 'x' * 79008
        raw, ref = self.store.prospective_admission(changed)
        argv = launch.admitted_sdk(LITERAL.encode(), SOURCE_REF, changed, ref, 1000.0)
        with self.assertRaises(ValueError): self.store.reserve_runtime(changed, ref, argv)
        self.assertEqual(self.store.inventory()['names'], [])
        self.original.subprocess.Popen.assert_not_called()

    def test_successful_real_start_reaches_popen_with_unmodified_complete_argv(self):
        self.admitted()
        item = self.sdk_adapter()
        class StopAtPopen(Exception): pass
        self.original.subprocess.Popen.side_effect = StopAtPopen()
        with self.assertRaises(StopAtPopen): item.start()
        self.assertEqual(self.original.subprocess.Popen.call_args.args[0], self.argv)
        self.assertEqual(item.argv[7], LITERAL)

    def test_real_failed_exit_persistence_fallback_is_complete_and_small(self):
        self.admitted()
        item = self.sdk_adapter()
        item.terminal = terminal(self.owner, 1); item.state = 'drained'
        captured = io.StringIO()
        with patch.object(self.store, 'json', side_effect=OSError('simulated durable write failure')):
            with contextlib.redirect_stderr(captured), self.assertRaises(OSError): item._persist_terminal()
        raw = captured.getvalue().encode()
        self.assertLessEqual(len(raw), 16384)
        decoded = codec.decode('exit_fallback', raw, LITERAL, wire_sha256=codec.digest(raw))
        self.assertEqual(decoded['actual_parent_observed_terminal'], item.terminal)
        self.assertFalse(decoded['exit_receipt_saved'])
        self.assertFalse(item.receipt_saved)
        self.assertTrue(item.receipt_attempted)
        self.assertNotIn('phase-1-exit.json', self.store.inventory()['names'])
        self.assertEqual(item.terminal['argv'][7], LITERAL)
        SIZING.append({'scope': 'actual source-adapter persistence-failure method with mocked failed Store write',
                       'fallback_wire_stdout_bytes': len(raw), 'original_cap': 16384})
        with patch.object(self.store, 'json', side_effect=AssertionError('second persistence forbidden')):
            item._persist_terminal()

    def test_real_successful_exit_persistence_once_and_unacknowledged_fallback(self):
        self.admitted()
        item = self.sdk_adapter()
        item.terminal = terminal(self.owner); item.state = 'drained'
        item._persist_terminal()
        self.assertEqual(self.store.load('phase-1-exit.json')[0], item.terminal)
        self.assertTrue(item.receipt_saved)
        with patch.object(self.store, 'json', side_effect=AssertionError('second persistence forbidden')):
            item._persist_terminal()
        ended = dict(item.terminal, owner=None, owner_acknowledged=False, exact_owner_gone=None)
        wire = codec.encode('exit', ended, LITERAL)
        self.assertEqual(codec.decode('exit', wire, LITERAL, wire_sha256=codec.digest(wire)), ended)

    def test_signal_and_unacknowledged_failure_fallback_restore_actual_terminal(self):
        self.admitted()
        for acknowledged in (True, False):
            item = self.sdk_adapter()
            ended = dict(terminal(self.owner, -15), cleanup_after_failure=True)
            if not acknowledged:
                ended.update(owner=None, owner_acknowledged=False, exact_owner_gone=None)
            item.terminal, item.state = ended, 'drained'
            captured = io.StringIO(); actual_error = OSError('actual failed write fixture')
            with patch.object(self.store, 'json', side_effect=actual_error), contextlib.redirect_stderr(captured):
                with self.assertRaises(OSError) as caught: item._persist_terminal()
            self.assertIs(caught.exception, actual_error)
            raw = captured.getvalue().encode()
            restored = codec.decode('exit_fallback', raw, LITERAL, wire_sha256=codec.digest(raw))
            self.assertEqual(restored['actual_parent_observed_terminal'], ended)
            self.assertEqual(restored['reporting_error_type'], 'OSError')
            self.assertFalse(item.receipt_saved)
            self.assertNotIn('phase-1-exit.json', self.store.inventory()['names'])

    def test_secondary_fallback_encode_or_print_failure_preserves_original_error(self):
        self.admitted()
        for failure in ('long_name', 'print'):
            item = self.sdk_adapter(); item.terminal = terminal(self.owner, -15); item.state = 'drained'
            error = type('X' * 17000, (OSError,), {})('mock write failure') if failure == 'long_name' else OSError('mock write failure')
            captured = io.StringIO()
            with contextlib.ExitStack() as contexts:
                contexts.enter_context(patch.object(self.store, 'json', side_effect=error))
                contexts.enter_context(contextlib.redirect_stderr(captured))
                if failure == 'print': contexts.enter_context(patch('builtins.print', side_effect=OSError('mock stream failure')))
                with self.assertRaises(OSError) as caught: item._persist_terminal()
            self.assertIs(caught.exception, error)
            self.assertFalse(item.receipt_saved)
            self.assertTrue(item.receipt_attempted)
            self.assertEqual(captured.getvalue(), '')
            self.assertNotIn('phase-1-exit.json', self.store.inventory()['names'])

    def successful_result(self):
        self.admitted()
        self.store.json('phase-1-owner.json', self.owner)
        ended = terminal(self.owner)
        self.store.json('phase-1-exit.json', ended)
        summary = dict(schema=self.original.SUMMARY_SCHEMA, diagnostic_passed=True, stage='complete',
                       observation_and_error_payload_complete=True, raw_capture_bytes_read=0, model_executed=False)
        summary_ref = self.store.json('phase-1-summary.json', summary)
        self.store.write('phase-1-memory-syscalls.jsonl', b'{"test_fixture_only":true}\n')
        observer = dict(schema='minicpm5_sdk_mapping_observer_logic_result_v1', observation_complete=True,
            trace_complete=True, trace_saved=True, exit_receipt_saved=True, failure=None, physically_drained=True,
            sdk_terminal=dict(pid=self.owner['pid'], wait_status=0, returncode=0, parent_observed_wait=True,
                              self_exit_claimed=False), model_executed=False, full_model_qualified=False, new_speedup_claim=False)
        saved_observer = dict(observer); saved_observer.pop('exit_receipt_saved'); saved_observer['self_persistence_claimed'] = False
        self.store.json('phase-1-observer-exit.json', saved_observer)
        item = self.sdk_adapter()
        item.state, item.terminal, item.observed, item.receipt_saved = 'drained', ended, observer, True
        result = item.result()
        self.assertEqual(result['summary'], {k: summary_ref[k] for k in ('path', 'sha256')})
        return result

    def test_complete_six_files_stdout_reconstructs_exact_original_result(self):
        result = self.successful_result()
        self.assertGreater(len(codec.encoded(result)), 16384)
        raw = results.render(result, self.store, LITERAL)
        self.assertLessEqual(len(raw), 16384)
        self.assertLessEqual(len(raw), self.store.runtime_reservation['stdout_reference_bound'])
        report = codec.parsed(raw)
        self.assertEqual(results.reconstruct(report, self.store.read, LITERAL), result)
        self.assertEqual(set(report['files']), results.FILES)
        actual = {name: self.store.reference(name)['bytes'] for name in self.store.inventory()['names']}
        runtime = sum(actual[name] for name in ('admission.json', 'phase-1-owner.json', 'phase-1-exit.json', 'phase-1-observer-exit.json'))
        self.assertLessEqual(runtime, 79008)
        SIZING.append(dict(scope='complete actual Store producer/consumer with saved-input and explicitly mock dynamic fields',
            all_six_wire_file_bytes=actual, actual_runtime_wire_total=runtime, original_runtime_cap=79008,
            stdout_reference_report_bytes=len(raw), original_stdout_cap=16384,
            original_expanded_result_bytes=len(codec.encoded(result))))

    def test_stdout_consumer_rejects_missing_changed_or_arbitrary_artifact(self):
        report = codec.parsed(results.render(self.successful_result(), self.store, LITERAL))
        for kind in ('missing', 'hash', 'pointer', 'expanded'):
            changed = copy.deepcopy(report)
            if kind == 'missing': changed['files'].pop('phase-1-observer-exit.json')
            elif kind == 'hash': changed['files']['phase-1-exit.json']['sha256'] = '0' * 64
            elif kind == 'pointer': changed['files']['phase-1-exit.json']['path'] = '/other/exit.json'
            else: changed['expanded_result_sha256'] = '0' * 64
            with self.subTest(kind=kind), self.assertRaises(Exception):
                results.reconstruct(changed, self.store.read, LITERAL)

    def test_missing_observer_receipt_never_becomes_complete_reference(self):
        self.admitted()
        self.store.json('phase-1-owner.json', self.owner)
        ended = terminal(self.owner)
        self.store.json('phase-1-exit.json', ended)
        result = dict(terminal=ended, observer={'failure': 'persistence_failed:OSError', 'exit_receipt_saved': False},
            diagnostic_passed=False, full_model_qualified=False, new_speedup_claim=False, exit_code=1)
        raw = results.render(result, self.store, LITERAL); report = codec.parsed(raw)
        self.assertEqual(report['schema'], results.FAILED)
        self.assertFalse(report['observer_complete_payload_available'])
        self.assertNotIn('phase-1-observer-exit.json', report['saved_files'])
        self.assertEqual(codec.decode('exit', report['terminal_wire'], LITERAL,
            wire_sha256=report['terminal_wire_sha256']), ended)
        self.assertEqual(report['observer_payload_sha256'], codec.digest(codec.encoded(result['observer'])))
        self.assertLessEqual(len(raw), self.store.runtime_reservation['stdout_unavailable_bound'])
        SIZING.append(dict(scope='missing-observer stdout failure preserves full actual terminal, explicitly unavailable observer',
                           stdout_unavailable_bytes=len(raw), original_stdout_cap=16384))
        with self.assertRaises(ValueError): results.reconstruct(report, self.store.read, LITERAL)

    def test_result_inventory_reference_or_read_failure_never_claims_saved_evidence(self):
        result = self.successful_result()
        for method in ('inventory', 'reference', 'read'):
            with self.subTest(method=method), patch.object(self.store, method, side_effect=OSError('mock original guarded IO failed')):
                with self.assertRaises(OSError): results.render(result, self.store, LITERAL)

    def test_stdout_whole_preflight_rejects_oversize_actual_root(self):
        with self.assertRaises(ValueError):
            results.future_stdout_budget(self.raw, self.argv, LITERAL, '/' + 'x' * 3000)
        self.original.subprocess.Popen.assert_not_called()

    def test_owner_scalar_bounds_reject_noncanonical_ticks(self):
        for ticks in ('0' * 100 + '1', str(2**64), '\u0661'):
            owner = dict(self.owner, start_ticks=ticks)
            with self.assertRaises(ValueError): codec.encode('owner', owner, LITERAL)

    def test_frozen_v6_and_original_body_methods_unchanged(self):
        self.assertEqual(hashlib.sha256(Path(codec.LITERAL['path']).read_bytes()).hexdigest(), codec.LITERAL['sha256'])
        old = ast.parse((Q / 'minicpm5_sdk_mapping_observed_adapter_v3.py').read_text())
        new = ast.parse((Q / 'minicpm5_sdk_mapping_observed_adapter_v4.py').read_text())
        methods = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(methods(old), methods(new))
        old_worker = ast.parse((Q / 'minicpm5_sdk_mapping_worker_source_entry_v5.py').read_text())
        new_worker = ast.parse((Q / 'minicpm5_sdk_mapping_worker_source_entry_v6.py').read_text())
        old_body = next(n for n in old_worker.body if isinstance(n, ast.FunctionDef) and n.name == 'source_entry')
        new_body = next(n for n in new_worker.body if isinstance(n, ast.FunctionDef) and n.name == 'source_entry')
        # Remove only the two reviewed boundary operations before comparing
        # the complete original authentication/native/scientific body.
        for node in ast.walk(new_body):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == 'codec' and fn.attr == 'adopt_entry_admission':
                    node.value = ast.Attribute(value=ast.Name(id='entry', ctx=ast.Load()), attr='admission_envelope', ctx=ast.Load())
        new_body.body = [n for n in new_body.body if not (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
            and n.value.func.value.id == 'codec' and n.value.func.attr == 'AdmissionMetadata')]
        self.assertEqual(ast.dump(old_body, include_attributes=False), ast.dump(new_body, include_attributes=False))
        old_outer = ast.parse((Q / 'minicpm5_sdk_mapping_outer_source_entry_v5.py').read_text())
        new_outer = ast.parse((Q / 'minicpm5_sdk_mapping_outer_source_entry_v6.py').read_text())
        final = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody).finalbody
        self.assertEqual([ast.dump(n, include_attributes=False) for n in final(old_outer)],
                         [ast.dump(n, include_attributes=False) for n in final(new_outer)])
        self.assertFalse(NATIVE.intersection(sys.modules))


if __name__ == '__main__':
    unittest.main()
