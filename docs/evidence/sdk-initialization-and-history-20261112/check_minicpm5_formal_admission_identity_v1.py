"""Saved r4 regression and pure successor lifecycle checks; no native or locks.

Saved admission is evidence of the pre-spawn r4 failure only. Hypothetical SDK
owners below are clearly test fixtures, never saved as actual process records.
"""
import ast
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Q))
BLOCKED = {'ctypes', '_ctypes', 'torch', 'numpy', 'neuronxcc', 'torch_neuronx',
           'neuronx_distributed', 'neuronx_distributed_inference'}
class NoNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('No native import: ' + fullname)
assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0, NoNative())

from research import minicpm5_formal_literal_codec_v1 as codec
from research import minicpm5_sdk_mapping_outer_source_entry_v8 as outer
from research import minicpm5_sdk_mapping_observer_logic_v2 as logic
from research import minicpm5_sdk_mapping_observer_backend_v4 as backend
from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
from research import check_minicpm5_formal_literal_boundary_v1 as boundary
from research import check_minicpm5_sdk_mapping_observer_logic_v1 as schedules
from research import check_minicpm5_sdk_mapping_observer_backend_v2 as backend_fakes
from research import check_minicpm5_completed_qwen_catalog_v1 as catalog_checks
from research import prepare_minicpm5_catalog_formal_binding_v2 as wiring

ADMISSION_PATH = Q / 'minicpm5-fixed-input-pv-sdk-import-formal-r4-20261112/admission.json'
RAW = ADMISSION_PATH.read_bytes()
LITERAL = Path(codec.LITERAL['path']).read_text()
REF = dict(path=str(ADMISSION_PATH), sha256=hashlib.sha256(RAW).hexdigest())
SAVED = codec.decode('admission', RAW, LITERAL, wire_sha256=REF['sha256'])
MEASUREMENTS = []


def memory_admission():
    value = copy.deepcopy(SAVED)
    observation = value['source_entry_binding']['outer_types_source_observation']
    observation['current_source_identity'] = tuple(observation['current_source_identity'])
    observation['execution_receipt']['identity'] = tuple(observation['execution_receipt']['identity'])
    return value


def sdk_owner():
    """Finite prospective sizing fixture; no SDK process is run or inspected."""
    binding = SAVED['source_entry_binding']
    argv = launch.admitted_sdk(LITERAL.encode(), binding['sdk_entry_source'], SAVED,
                              REF, binding['outer_deadline'])
    return codec.prospective_terminal(RAW, argv, LITERAL)['owner']


def function(path, name, class_name=None):
    tree = ast.parse((Q / path).read_text())
    body = tree.body if class_name is None else next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


class AdmissionTests(unittest.TestCase):
    def test_exact_saved_failure_and_complete_wire_reproduction(self):
        value = memory_admission()
        self.assertEqual(len(RAW), 43273)
        self.assertEqual(len(LITERAL.encode()), 37178)
        self.assertEqual(codec.encode('admission', value, LITERAL), RAW)
        self.assertNotEqual(value, SAVED)
        self.assertEqual(value['lease'], SAVED['lease'])
        self.assertEqual(value['witness'], SAVED['witness'])
        self.assertEqual(SAVED['lease']['outer_owner'], json.loads(
            (Q / 'minicpm5-formal-import-r4-20261112-root-live.json').read_text())['owner'])

    def test_boundary_changes_only_two_types_and_preserves_frozen_observation(self):
        value = memory_admission()
        original = value['source_entry_binding']['outer_types_source_observation']
        before = copy.deepcopy(original)
        result = outer.admission_types_observation(original)
        self.assertEqual(original, before)
        self.assertIs(type(original['current_source_identity']), tuple)
        self.assertIs(type(original['execution_receipt']['identity']), tuple)
        self.assertEqual(result, SAVED['source_entry_binding']['outer_types_source_observation'])
        self.assertEqual(codec.encoded(result), codec.encoded(original))
        result['execution_receipt']['identity'][0] += 1
        self.assertEqual(original, before)

    def test_exact_stat_schema_rejects_missing_wrong_or_pre_normalized_receipts(self):
        original = memory_admission()['source_entry_binding']['outer_types_source_observation']
        changes = [lambda o: o.update(schema='other'),
                   lambda o: o.update(executed_through_pinned_loader=False),
                   lambda o: o.update(execution_receipt=None),
                   lambda o: o['execution_receipt'].update(extra=1),
                   lambda o: o.update(current_source_identity=list(o['current_source_identity'])),
                   lambda o: o.update(current_source_identity=o['current_source_identity'][:-1]),
                   lambda o: o['execution_receipt'].update(identity=(True,) + o['execution_receipt']['identity'][1:]),
                   lambda o: o['execution_receipt'].update(identity=(-1,) + o['execution_receipt']['identity'][1:])]
        for change in changes:
            value = copy.deepcopy(original); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                outer.admission_types_observation(value)

    def test_original_entire_predicate_fails_before_proc_then_passes_after_exact_boundary(self):
        value = memory_admission()
        method = function('minicpm5_pv_sdk_import_diagnostic_v1.py',
                          'authenticate_complete_prerequisites', 'SingleSDKAdapter')
        pins = json.loads((Q / 'minicpm5-formal-import-r4-20261112-complete-source-map.json').read_text())
        plan = {'explicit_mock_descriptor_plan': True}
        witness = copy.deepcopy(value['witness'])
        digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        witness['descriptor_plan_sha256'] = digest(plan)
        value['witness'] = witness
        r = NS(digest=digest, p=NS(proc=Mock(return_value=value['lease']['outer_owner'])))
        namespace = dict(r=r, os=NS(getpid=lambda: value['lease']['outer_owner']['pid']), require=codec.require)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(Q / 'minicpm5_pv_sdk_import_diagnostic_v1.py'), 'exec'), namespace)
        guard = Mock()
        wire = codec.encode('admission', value, LITERAL)
        ref = dict(path=REF['path'], sha256=hashlib.sha256(wire).hexdigest())
        metadata = codec.AdmissionMetadata(NS(read=lambda unused: codec.parsed(wire)), ref, LITERAL, guard)
        item = NS(authenticated=dict(witness=witness, context=dict(pins_plan=dict(source_sha256=pins), plan=plan)),
                  admission_ref=ref, admission=value, lease=NS(admission=value['lease']), guard=guard, metadata=metadata)
        with self.assertRaisesRegex(ValueError, 'Actual original authenticated outer'):
            namespace['authenticate_complete_prerequisites'](item)
        r.p.proc.assert_not_called()
        value['source_entry_binding']['outer_types_source_observation'] = outer.admission_types_observation(
            value['source_entry_binding']['outer_types_source_observation'])
        self.assertEqual(codec.encode('admission', value, LITERAL), wire)
        namespace['authenticate_complete_prerequisites'](item)
        r.p.proc.assert_called_once()
        self.assertIn(unittest.mock.call(force=True), guard.call_args_list)

    def test_actual_outer_producer_with_full_types_observation_reaches_mock_popen(self):
        # Reuse the real producer/consumer regression with precisely one fixture
        # correction: its synthetic types dict becomes the actual frozen shape.
        # AST compilation is confined to tests; runtime modules never use it.
        node = function('check_minicpm5_formal_literal_boundary_v1.py',
                        'test_actual_outer_producer_original_prerequisites_and_frozen_sdk_consumer', 'Fixture')
        changed = 0
        for child in ast.walk(node):
            if isinstance(child, ast.keyword) and child.arg == 'types_observation':
                child.value = ast.Call(func=ast.Name(id='real_types_fixture', ctx=ast.Load()), args=[], keywords=[])
                changed += 1
        self.assertEqual(changed, 1)
        namespace = dict(vars(boundary), outer=outer, adapter=outer.adapter,
            real_types_fixture=lambda: memory_admission()['source_entry_binding']['outer_types_source_observation'])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(Q / 'check_minicpm5_formal_literal_boundary_v1.py'), 'exec'), namespace)
        case = boundary.Fixture(methodName='runTest')
        with patch.object(boundary, 'adapter', outer.adapter), patch.object(
                outer.catalog_auth, 'authenticate_original',
                side_effect=lambda gate, *args, **kw: gate.authenticate_original(*args, **kw)):
            try:
                case.setUp()
                namespace[node.name](case)
            finally:
                case.doCleanups()


class OwnerTests(unittest.TestCase):
    def test_old_budget_block_and_complete_new_owner(self):
        owner = sdk_owner()
        with self.assertRaisesRegex(logic.ObservationError, 'task_identity_budget'):
            logic.previous.valid_owner(owner, owner['pid'])
        result = logic.valid_owner(owner, owner['pid'])
        self.assertEqual(result, owner)
        self.assertIsNot(result, owner)
        self.assertEqual(result['argv'][7], LITERAL)
        wire = codec.encode('owner', owner, LITERAL)
        self.assertLessEqual(len(wire), 2048)
        self.assertEqual(codec.decode('owner', wire, LITERAL, wire_sha256=hashlib.sha256(wire).hexdigest()), owner)
        MEASUREMENTS.append(dict(scope='prospective SDK owner budget only; no actual child',
                                 expanded_owner_bytes=len(codec.encoded(owner)), wire_owner_bytes=len(wire), original_budget=2048))

    def test_small_original_owner_behavior_unchanged(self):
        owner = copy.deepcopy(schedules.OWNER)
        self.assertEqual(logic.valid_owner(owner, owner['pid']), logic.previous.valid_owner(owner, owner['pid']))

    def test_unknown_literal_layout_hash_fields_and_pid_rejected(self):
        changes = [lambda o: o['argv'].__setitem__(7, LITERAL + ' '),
                   lambda o: o['argv'].__setitem__(7, LITERAL[:-1]),
                   lambda o: o['argv'].__setitem__(0, '/other/python'),
                   lambda o: o['argv'].__setitem__(9, 'outer312'),
                   lambda o: o['argv'].__setitem__(13, '0' * 64),
                   lambda o: o['argv'].append('extra'),
                   lambda o: o.update(extra='field'),
                   lambda o: o.update(start_ticks='01'),
                   lambda o: o.update(cgroup='x' * 3000)]
        for change in changes:
            owner = sdk_owner(); change(owner)
            with self.subTest(change=change), self.assertRaises(ValueError):
                logic.valid_owner(owner, owner['pid'])
        owner = sdk_owner()
        with self.assertRaises(ValueError): logic.valid_owner(owner, owner['pid'] - 1)

    def schedule_context(self):
        ctx = contextlib.ExitStack()
        self.addCleanup(ctx.close)
        root = dict(sdk_owner(), pid=101, ppid=99, start_ticks='200', cgroup='original')
        child = dict(copy.deepcopy(root), pid=102, ppid=101, start_ticks='201')
        ctx.enter_context(patch.object(schedules, 'OWNER', root))
        ctx.enter_context(patch.object(schedules, 'CHILD_OWNER', child))
        ctx.enter_context(patch.object(schedules, 'm', NS(**dict(vars(logic.previous), ObservedSDK=logic.ObservedSDK))))
        return root, child

    def test_complete_literal_owner_retained_through_actual_mock_wait_status(self):
        owner, _ = self.schedule_context()
        engine, ports, store, child = schedules.fixture()
        result = engine.run()
        self.assertTrue(result['observation_complete'])
        self.assertTrue(result['physically_drained'])
        self.assertEqual(child.returncode, 1)
        self.assertEqual(result['sdk_terminal']['wait_status'], 256)
        self.assertEqual(engine.tasks[101]['owner'], owner)
        self.assertEqual(len(store.files), 2)

    def test_kernel_child_full_owner_and_trace_survive_new_validator(self):
        owner, child_owner = self.schedule_context()
        case = schedules.ForkTests()
        engine, ports, store, child = case.setup_fork([schedules.INITIAL, schedules.entry(231), dict(status=0)])
        result = engine.run()
        self.assertTrue(result['observation_complete'])
        self.assertEqual(engine.tasks[101]['owner'], owner)
        self.assertEqual(engine.tasks[102]['owner'], child_owner)
        record = next(row for row in schedules.records(store) if row['kind'] == 'kernel_task')
        self.assertEqual(record['owner'], child_owner)
        self.assertLessEqual(len(store.files[logic.previous.TRACE_NAME]), logic.previous.TRACE_CAP)

    def test_exact_identity_mismatch_is_still_rejected(self):
        self.schedule_context()
        engine, ports, _, _ = schedules.fixture()
        for key, changed in [('start_ticks', '201'), ('cgroup', 'different'), ('argv', ['short'])]:
            saved = ports.owners[101][key]
            ports.owners[101][key] = changed
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, 'known_task_identity_changed'):
                engine._identity(101)
            ports.owners[101][key] = saved

    def test_backend_factory_selects_successor_and_discovers_only_kernel_known_owner(self):
        owner = dict(sdk_owner(), pid=101, ppid=99, start_ticks='200', cgroup='original')
        with patch.object(backend_fakes, 'OWNER', owner):
            _, kw, child, ct, _, log, owners = backend_fakes.fixture(acquire=False)
            ports = backend.make_ports(child, owner, backend_fakes.AUTH, lambda: log.append('guard'), **kw)
        self.assertIs(type(ports), backend.NativePorts)
        self.assertFalse(ports.wait_lock_owned)
        self.assertEqual(ports.owner, owner)
        ports.begin_wait_handoff()  # Explicit fake Lock, never an OS lock.
        with self.assertRaises(ValueError): ports.identity(102)
        ct.fn.responses.append(lambda value: setattr(value, 'value', 102) or 0)
        self.assertEqual(ports.event_message(101), 102)
        owners[102] = dict(copy.deepcopy(owner), pid=102, ppid=101, start_ticks='201')
        self.assertEqual(ports.identity(102), owners[102])
        self.assertEqual(ports.known[102], owners[102])
        changed = dict(owner, start_ticks='201')
        with self.assertRaises(ValueError): ports.signal_owned(changed, 15)


class FrozenBehaviorTests(unittest.TestCase):
    def test_complete_binding_source_closure_uses_all_five_successors(self):
        node = function('check_minicpm5_completed_qwen_catalog_v1.py',
                        'test_current_v98_real_binding_orchestration_and_every_new_bootstrap_import', 'Checks')
        changes = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and child.value == 'research.minicpm5_sdk_mapping_outer_source_entry_v7':
                child.value = 'research.minicpm5_sdk_mapping_outer_source_entry_v8'; changes += 1
        self.assertEqual(changes, 1)
        namespace = dict(vars(catalog_checks), wiring=wiring)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(Q / 'check_minicpm5_completed_qwen_catalog_v1.py'), 'exec'), namespace)
        namespace[node.name](self)

    def test_copied_observer_and_native_methods_are_exact(self):
        for name in ('__init__', '_new_task'):
            self.assertEqual(ast.dump(function('minicpm5_sdk_mapping_observer_logic_v1.py', name, 'ObservedSDK')),
                             ast.dump(function('minicpm5_sdk_mapping_observer_logic_v2.py', name, 'ObservedSDK')))
        for name, cls in [('make_ports', None), ('identity', 'NativePorts')]:
            self.assertEqual(ast.dump(function('minicpm5_sdk_mapping_observer_backend_v2.py', name, cls)),
                             ast.dump(function('minicpm5_sdk_mapping_observer_backend_v4.py', name, cls)))
        for name in ('_identity', '_consume', '_cleanup', '_drained', '_result', 'run'):
            self.assertIs(getattr(logic.ObservedSDK, name), getattr(logic.previous.ObservedSDK, name))
        for name in ('_transition_wait_lock', 'wait_known', 'signal_owned', 'finish_wait_handoff'):
            self.assertIs(getattr(backend.NativePorts, name), getattr(backend.original.NativePorts, name))

    def test_complete_outer_body_only_two_field_projection_and_dependency_alias(self):
        old = ast.parse((Q / 'minicpm5_sdk_mapping_outer_source_entry_v7.py').read_text())
        new = ast.parse((Q / 'minicpm5_sdk_mapping_outer_source_entry_v8.py').read_text())
        new.body = [n for n in new.body if not (isinstance(n, ast.FunctionDef) and n.name == 'admission_types_observation')]
        changes = 0
        for node in ast.walk(new):
            if isinstance(node, ast.ImportFrom):
                for item in node.names:
                    if item.name == 'minicpm5_sdk_mapping_source_adapter_v3': item.name = 'minicpm5_sdk_mapping_source_adapter_v2'
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'admission_types_observation':
                node.func = ast.Attribute(value=ast.Name(id='copy', ctx=ast.Load()), attr='deepcopy', ctx=ast.Load()); changes += 1
        self.assertEqual(changes, 1)
        self.assertEqual(ast.dump(old), ast.dump(new))
        for old_name, new_name, substitutions in [
            ('minicpm5_sdk_mapping_observed_adapter_v4.py', 'minicpm5_sdk_mapping_observed_adapter_v5.py',
             [('minicpm5_sdk_mapping_observer_backend_v4', 'minicpm5_sdk_mapping_observer_backend_v3'),
              ('minicpm5_sdk_mapping_observer_logic_v2', 'minicpm5_sdk_mapping_observer_logic_v1')]),
            ('minicpm5_sdk_mapping_source_adapter_v2.py', 'minicpm5_sdk_mapping_source_adapter_v3.py',
             [('minicpm5_sdk_mapping_observed_adapter_v5', 'minicpm5_sdk_mapping_observed_adapter_v4')])]:
            text = (Q / new_name).read_text()
            for a, b in substitutions: text = text.replace(a, b)
            self.assertEqual(text, (Q / old_name).read_text())
        self.assertEqual(hashlib.sha256(LITERAL.encode()).hexdigest(), codec.LITERAL['sha256'])
        self.assertFalse(BLOCKED.intersection(sys.modules))


if __name__ == '__main__':
    unittest.main()
