"""Small source/memory prototype controls; no original plans or runtime IO."""
import ast
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'research')]


def no_runtime(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'fcntl.flock', 'fcntl.lockf'):
        raise AssertionError('No runtime operation: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)) and Path(args[0]).suffix not in ('.py', '.pyc', '.so'):
        raise AssertionError('Only source/module reads: ' + str(args[0]))


sys.addaudithook(no_runtime)
from research import minicpm5_native_source_phase_seal_v1 as seal
from research import minicpm5_layer1_operator_native_manifest_v1 as manifest
import minicpm5_layer0_runtime_auth_v1 as original
import nemotron9b_package_metadata_alias_v1 as aliases


def definitions(path, names, scope):
    tree = ast.parse(path.read_bytes())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names): raise AssertionError('Exact source functions')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)


# The historical source package performs original JSON reads at import time.
# Execute only its unchanged pure function definitions against memory fixtures;
# no import hook, original data file, or production-global patch is used.
pure = dict(require=seal.require, NATIVE_BYTES=3920408)
definitions(ROOT / 'research/dory_saved_retry_source_partition_v1.py', {'native_pins'}, pure)
metadata = SimpleNamespace(query_sources=SimpleNamespace(native_pins=pure['native_pins'],
    OBSERVATION=dict(path='/synthetic/query-observation.json', sha256='f' * 64)),
    python_source_bytes=lambda *args: None)
scope = dict(require=seal.require, digest=original.digest, aliases=aliases, hashlib=hashlib,
    metadata=metadata, PARTITION_SCHEMA='minicpm5_layer1_sdk_batch_layout_source_partition_v2',
    R12_SPEC=dict(path='/synthetic/r12-spec.json', sha256='e' * 64),
    CATEGORIES=('text_sha256', 'historical_payload_sha256', 'python_bytes_sha256', 'query_native_binary_sha256'))
definitions(ROOT / 'research/minicpm5_layer1_operator_sdk_sources_v1.py', {'check_partition', 'verify_text_sources'}, scope)
scope.update(text=lambda *args: None, classify=lambda *args: None)
sources = SimpleNamespace(**scope)


def fixture(count=4):
    path = ROOT / 'research/check_minicpm5_layer1_operator_native_driver_v2.py'
    tree = ast.parse(path.read_bytes())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fixture')
    scope = dict(hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    plan, native = scope['fixture']()
    raw = {f'/synthetic/source-{n}.py': f'value = {n}\n'.encode() for n in range(count)}
    runner = seal.life.ROOT + '/synthetic_native_runner.py'
    raw[runner] = b'runner = 1\n'
    text = {path: hashlib.sha256(data).hexdigest() for path, data in raw.items()}
    native_pins = {f'/synthetic/library-{n}.so': 'd' * 64 for n in range(9)}
    observation = dict(schema='dory_service_query_dependency_observation_v1', complete_observation=True,
        source_and_binary_bytes_hashed=4326816, source_and_binary_pins={}, new_native_file_mappings={})
    for n, path in enumerate(native_pins):
        observation['source_and_binary_pins'][path] = dict(device=1, inode=n + 1,
            bytes=400000 if n < 8 else 720408, sha256='d' * 64)
        observation['new_native_file_mappings'][path] = dict(device=1, inode=n + 1)
    for n in range(32):
        observation['source_and_binary_pins'][f'/synthetic/query-{n}.py'] = dict(sha256='e' * 64)
    categories = dict(text_sha256=text,
        historical_payload_sha256={f'/synthetic/old-{n}.raw': 'a' * 64 for n in range(20)},
        python_bytes_sha256={f'/synthetic/byte-{n}.py': 'b' * 64 for n in range(3)},
        query_native_binary_sha256=native_pins)
    pins = {path: sha for part in categories.values() for path, sha in part.items()}
    source_ref = dict(path='/synthetic/source-contract.json', sha256='c' * 64)
    partition = dict(schema=sources.PARTITION_SCHEMA, source_contract=source_ref,
        r12_spec=sources.R12_SPEC, source_map_sha256=original.digest(pins),
        partition_sha256=original.digest(categories), categories=categories, metadata_aliases={},
        query_observation_ref=sources.metadata.query_sources.OBSERVATION, query_observation=observation)
    plan.update(schema='minicpm5_layer0_runtime_inputs_v1', source_sha256=pins, source_contract=source_ref,
        inputs={f'/synthetic/input-{n}.json': dict(sha256='f' * 64, identity=dict(size=n)) for n in range(count)},
        output_root='/synthetic/native-output', interpreters=dict(native=seal.life.PYTHON))
    plan['native_operator_manifest'] = manifest.build(plan, native)
    c = seal.catalog.catalog(plan['native_operator_manifest'])
    plan['runtime_output_manifest'] = c
    resolved = dict(plan=plan, plan_sha256=original.digest(plan), runtime_admitted=False,
        full_runtime_catalog_pending=False, runtime_integration_pending=True, physical_checks_performed=False,
        native_runtime_output_catalog=c, native_runtime_output_catalog_sha256=original.digest(c))
    contract = seal.life.build_contract(resolved, dict(path='/synthetic/overlay.json', sha256='a' * 64),
        dict(path=runner, sha256=text[runner]), 'trn1-mini-layer1-operator-native-synthetic-seal')
    return plan, resolved, contract, partition, raw


class Harness:
    def __init__(self, count=4):
        self.plan, self.resolved, self.contract, self.partition, self.raw = fixture(count)
        self.events = []
        self.lease = SimpleNamespace(check=lambda stage: self.events.append(('lease', stage)))
        self.ports = SimpleNamespace(resolved=self.resolved, contract=self.contract,
            admission={'outer_owner': {'pid': 10}}, coordinator={'pid': 11}, read_owner={'pid': 12},
            storage_available=lambda: (self.events.append(('storage',)) or dict(root=40 * 2**30, output=40 * 2**30)))
        self.entries = {'runtime.log': dict(kind='file', uid=1000, nlink=1, symlink=False, size=1, blocks=8)}
        self.root = dict(kind='directory', uid=1000, symlink=False, size=4096, blocks=8)
        self.writers = dict(outer=10, coordinator=11, native=12)
        self.permit = dict(schema='minicpm5_layer0_runtime_permit_v1', plan_sha256=self.resolved['plan_sha256'],
            role='native', rank=-1, arm='HF_native', attempt_id='a' * 32, serial_exclusive=True,
            source_review_accepted=True, all_predecessors_drained=True, pid=os.getpid(), uid=1000,
            start_ticks='123', cgroup='0::/synthetic', argv=['synthetic'], python=seal.life.PYTHON,
            environment={}, output_root=self.plan['output_root'])

    def old_guard(self):
        self.lease.check('native.runtime')
        seal.life.storage(self.contract, self.ports)
        c = seal.catalog.catalog(self.plan['native_operator_manifest'])
        entries, root, writers = self.walk(self.contract['output_root'], c, self.writers)
        seal.life.usage(self.contract, self.resolved, entries, root, writers=writers)
        return self.permit

    def walk(self, root, catalog, writers):
        self.events.append(('walk',))
        return copy.deepcopy(self.entries), copy.deepcopy(self.root), dict(writers)

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(original, 'clear_window'))
        stack.enter_context(patch.object(original.os, 'getuid', return_value=1000))
        stack.enter_context(patch.object(original.sys, 'executable', seal.life.PYTHON))
        stack.enter_context(patch.object(Path, 'read_text', lambda path, *a, **k:
            'x) ' + ' '.join(['0'] * 19 + ['123']) if str(path) == '/proc/self/stat' else '0::/synthetic'))
        stack.enter_context(patch.object(Path, 'read_bytes', lambda path: b'synthetic\0'))
        stack.enter_context(patch.object(seal, 'walk_sealed', side_effect=self.walk))
        text = stack.enter_context(patch.object(sources, 'text', side_effect=lambda path, wanted=None:
            self.events.append(('source', path)) or self.raw[path]))
        stack.enter_context(patch.dict(scope, text=text))
        stack.enter_context(patch.object(sources.metadata, 'python_source_bytes', side_effect=lambda path, wanted:
            self.events.append(('python-bytes', path))))
        return stack

    def context(self):
        return original.RuntimeContext(self.plan, self.resolved['plan_sha256'], self.permit, self.old_guard)

    def phase(self, context):
        return seal.SourcePhase(self.contract, self.resolved, context, self.partition, self.lease, self.ports, sources)


class Tests(unittest.TestCase):
    def test_recursive_real_readonly_and_unknown_types(self):
        value = {'rows': [{'nested': [1, True, None, 2.5]}]}
        frozen = seal.seal_json(value)
        self.assertIs(type(frozen), MappingProxyType)
        self.assertIs(type(frozen['rows']), tuple)
        self.assertIs(type(frozen['rows'][0]), MappingProxyType)
        with self.assertRaises(TypeError):
            frozen['rows'][0]['nested'] = []
        with self.assertRaises(TypeError):
            frozen['rows'][0]['nested'][0] = 9
        value['rows'][0]['nested'][0] = 999
        self.assertEqual(frozen['rows'][0]['nested'][0], 1)
        class Custom(dict): pass
        cyclic = []; cyclic.append(cyclic)
        for unknown in (Custom(), set(), object(), b'bytes', (1,), {1: 'bad'}, float('nan'), cyclic):
            with self.subTest(kind=type(unknown).__name__), self.assertRaises(RuntimeError):
                seal.seal_json(unknown)

    def test_dynamic_bodies_preserved_with_only_static_catalog_removed(self):
        def function(path, name):
            return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ROOT / 'research/minicpm5_native_source_phase_seal_v1.py'
        old = function(ROOT / 'research/minicpm5_layer1_operator_native_process_adapter_v6.py', 'output_inventory')
        new = function(module, 'walk_sealed')
        old.name = new.name; old.args = new.args
        old.body = [n for n in old.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'c' for t in n.targets))]
        self.assertEqual(ast.dump(old), ast.dump(new))
        old = function(ROOT / 'research/minicpm5_layer1_operator_native_output_catalog_v2.py', 'inventory')
        new = function(module, 'inventory_sealed')
        old.name = new.name; old.args = new.args
        old.body = [n for n in old.body if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'c' for t in n.targets))]
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_complete_sources_original_context_copy_and_dynamic_guards(self):
        h = Harness()
        with h.patches():
            context = h.context()
            self.assertIsNot(context.plan, h.plan)
            original_copy = context.plan
            with h.phase(context) as phase:
                self.assertIs(context.plan, phase.plan)
                self.assertIs(type(context.plan), MappingProxyType)
                report = phase.verify_text_sources()
            self.assertIs(context.plan, original_copy)
            self.assertFalse(context._sources_checked)
        self.assertTrue(report['complete_text_verification'])
        self.assertEqual({e[1] for e in h.events if e[0] == 'source'}, set(h.raw))
        self.assertEqual(len([e for e in h.events if e[0] == 'python-bytes']), 3)
        self.assertEqual(len([e for e in h.events if e[0] == 'lease']), len([e for e in h.events if e[0] == 'walk']))
        self.assertEqual(len([e for e in h.events if e[0] == 'storage']), len([e for e in h.events if e[0] == 'walk']))

    def test_old_mutable_aliases_all_rejected_at_exit_before_model(self):
        for alias in ('resolved', 'context_plan', 'contract', 'partition', 'permit'):
            h = Harness()
            with h.patches(), self.subTest(alias=alias):
                context = h.context(); model_calls = []
                with self.assertRaisesRegex(RuntimeError, 'Mutable alias changed'):
                    with h.phase(context) as phase:
                        phase.verify_text_sources()
                        target = {'resolved': h.resolved, 'context_plan': phase.original_plan,
                            'contract': h.contract, 'partition': h.partition, 'permit': context.permit}[alias]
                        target['mutation'] = ['changed']
                    model_calls.append('forbidden')
                self.assertEqual(model_calls, [])
                self.assertFalse(context._sources_checked)

    def test_deep_plan_mutation_and_rebinding_reject(self):
        for target in ('source', 'input', 'catalog', 'context_rebind', 'callback_rebind', 'permission'):
            h = Harness()
            with h.patches(), self.subTest(target=target):
                context = h.context()
                with self.assertRaises(RuntimeError):
                    with h.phase(context) as phase:
                        phase.verify_text_sources()
                        if target == 'source': phase.original_plan['source_sha256'][next(iter(h.raw))] = '0' * 64
                        elif target == 'input': h.plan['inputs'][next(iter(h.plan['inputs']))]['identity']['size'] += 1
                        elif target == 'catalog': h.plan['runtime_output_manifest']['runtime_files'].pop()
                        elif target == 'context_rebind': context.plan = copy.deepcopy(phase.original_plan)
                        elif target == 'callback_rebind': h.lease.check = lambda stage: None
                        else: context._sources_checked = True
                        if target.endswith('rebind') or target == 'permission': context.guard()
                self.assertFalse(context._sources_checked)

    def test_runtime_guards_hashes_and_allocation_fail_closed(self):
        for fault in ('lease', 'storage', 'extra', 'symlink', 'directory', 'root_cap', 'hash'):
            h = Harness()
            with h.patches(), self.subTest(fault=fault):
                context = h.context()
                with self.assertRaises(RuntimeError):
                    with h.phase(context) as phase:
                        if fault == 'lease': h.lease.check = Mock(side_effect=RuntimeError('lost lease'))
                        elif fault == 'storage': h.ports.storage_available = lambda: dict(root=1, output=1)
                        elif fault == 'extra': h.entries['evil'] = dict(h.entries['runtime.log'])
                        elif fault == 'symlink': h.entries['runtime.log']['symlink'] = True
                        elif fault == 'directory': h.entries['native-operator'] = dict(kind='directory', uid=1000, symlink=False, size=1, blocks=256 * 2**20 // 512)
                        elif fault == 'root_cap': h.root['blocks'] = 256 * 2**20 // 512
                        else: h.raw[next(iter(h.raw))] = b'changed\n'
                        phase.verify_text_sources()
                self.assertFalse(context._sources_checked)

    def test_partition_and_context_source_map_must_match(self):
        h = Harness()
        with h.patches():
            context = h.context()
            context.plan['inputs']['unexpected'] = {'identity': {}}
            with self.assertRaises(RuntimeError):
                with h.phase(context): pass

    def test_every_failure_is_permanently_latched_even_if_caught_inside(self):
        for fault in ('source_hash', 'dynamic_guard', 'original_guard', 'body_exception', 'exit_alias'):
            h = Harness()
            with h.patches(), self.subTest(fault=fault):
                context = h.context(); first = None
                with self.assertRaises((RuntimeError, ValueError)) as outer:
                    with h.phase(context) as phase:
                        try:
                            if fault == 'source_hash':
                                h.raw[next(iter(h.raw))] = b'corrupt'
                                phase.verify_text_sources()
                            elif fault == 'dynamic_guard':
                                h.entries['runtime.log']['symlink'] = True
                                context.guard()
                            elif fault == 'original_guard':
                                context.permit['rank'] = 1
                                context.guard()
                            elif fault == 'body_exception':
                                raise ValueError('body failed')
                            else:
                                phase.verify_text_sources()
                                phase.original_plan['inputs']['changed'] = {}
                        except RuntimeError as error:
                            first = error
                            # Catching and restoring the original cause must not
                            # permit another source pass, a guard, or phase exit.
                            h.entries['runtime.log']['symlink'] = False
                            context.permit['rank'] = -1
                            with self.assertRaises(RuntimeError) as again: context.guard()
                            self.assertIs(again.exception, first)
                            with self.assertRaises(RuntimeError): phase.verify_text_sources()
                failure = outer.exception
                if first is not None: self.assertIs(failure, first)
                for operation in (lambda: context.guard(), lambda: context.verify_input('/synthetic/no-model'),
                                  lambda: context.external_guard(), lambda: h.phase(context).__enter__()):
                    with self.assertRaises(type(failure)) as rejected: operation()
                    self.assertIs(rejected.exception, failure)
                self.assertFalse(context._sources_checked)
        h = Harness()
        with h.patches():
            context = h.context()
            h.partition['categories']['text_sha256'][next(iter(h.raw))] = '0' * 64
            with self.assertRaises(RuntimeError):
                with h.phase(context): pass

    def test_classifier_uses_detached_plain_plan_and_keeps_original_transport(self):
        h = Harness()
        with h.patches():
            context = h.context()
            def classify(pins, contract, read):
                self.assertIs(type(pins), dict)
                self.assertIsNot(pins, context.plan['source_sha256'])
                self.assertEqual(pins, context.plan['source_sha256'])
                return h.partition
            with patch.object(sources, 'classify', side_effect=classify) as called:
                self.assertIs(seal.phase_partition(context, sources), h.partition)
            called.assert_called_once()


def comparison(count):
    results = {}
    for mode in ('original', 'sealed'):
        h = Harness(count)
        with h.patches():
            context = h.context(); h.events.clear()
            plan_hashes = 0
            boundary_plan_hashes = boundary_resolved_hashes = 0
            original_digest = seal.life.digest
            def counted(value):
                nonlocal plan_hashes
                if type(value) is dict and value.get('schema') == 'minicpm5_layer0_runtime_inputs_v1': plan_hashes += 1
                return original_digest(value)
            def counted_boundary(value):
                nonlocal boundary_plan_hashes, boundary_resolved_hashes
                if type(value) is dict and value.get('schema') == 'minicpm5_layer0_runtime_inputs_v1': boundary_plan_hashes += 1
                if type(value) is dict and type(value.get('plan')) is dict: boundary_resolved_hashes += 1
                return original_digest(value)
            started = time.perf_counter()
            with patch.object(seal.life, 'digest', side_effect=counted), \
                 patch.object(seal, 'digest', side_effect=counted_boundary), \
                 patch.object(seal.catalog, 'catalog', wraps=seal.catalog.catalog) as catalogs:
                if mode == 'original':
                    sources.verify_text_sources(h.partition, context.guard)
                else:
                    with h.phase(context) as phase: phase.verify_text_sources()
            results[mode] = dict(seconds=round(time.perf_counter() - started, 6),
                full_plan_digests_inside_original_lifecycle=plan_hashes, catalog_builds=catalogs.call_count,
                live_walks=sum(e[0] == 'walk' for e in h.events), lease_checks=sum(e[0] == 'lease' for e in h.events),
                storage_checks=sum(e[0] == 'storage' for e in h.events),
                text_hashes=sum(e[0] == 'source' for e in h.events), python_byte_hashes=sum(e[0] == 'python-bytes' for e in h.events),
                full_plan_boundary_digests_additional=boundary_plan_hashes,
                resolved_with_full_plan_boundary_digests=boundary_resolved_hashes)
    return dict(extra_synthetic_text_sources=count, original_source_count=len(h.plan['source_sha256']),
        synthetic_plan_bytes=len(original.encoded(h.plan)), measurements=results,
        real_runtime_speedup_claim=False)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if result.wasSuccessful():
        print(json.dumps(dict(comparisons=[comparison(8), comparison(16)], actual_runtime=False,
            original_data_reads=False, model_permission_granted=False), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
