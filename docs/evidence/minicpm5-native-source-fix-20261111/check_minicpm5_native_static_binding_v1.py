"""Pure source/memory controls; no model, raw payload, service or storage IO."""
import ast
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'research')]
from research import check_minicpm5_native_source_phase_seal_v1 as memory
from research import minicpm5_native_static_binding_v1 as static
from research import minicpm5_layer1_operator_native_lifecycle_core_v3 as current


def isolated_ports():
    """Unchanged base class definitions, never historical import-time IO."""
    path = ROOT / 'research/minicpm5_layer1_operator_native_process_adapter_v7.py'
    tree = ast.parse(path.read_bytes())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Ports')
    scope = {}
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(path), 'exec'), scope)
    return type('PurePorts', (static.StaticPortsMixin, scope['Ports']), {})


class Harness(memory.Harness):
    def __init__(self):
        super().__init__()
        self.ports = isolated_ports()(self.contract, self.resolved, {}, {'outer_owner': {'pid': 10}},
                                      {'pid': 11}, self.lease, None)
        self.ports.read_owner = {'pid': 12}
        self.ports.storage_available = lambda: (self.events.append(('storage',)) or
                                               dict(root=40 * 2**30, output=40 * 2**30))

    def new_guard(self):
        self.lease.check('native.runtime')
        current.storage(self.contract, self.ports)
        entries, root, writers = self.ports.inventory()
        current.usage(self.contract, self.resolved, entries, root, writers=writers,
                      static_binding=self.ports.static_binding)
        return self.permit


def function_nodes(path):
    return {node.name: node for node in ast.parse(path.read_bytes()).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


class RemoveStaticCalls(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        node.keywords = [value for value in node.keywords if value.arg != 'static_binding']
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            if node.value.func.attr == 'verify_full':
                return None
        return self.generic_visit(node)


class Tests(unittest.TestCase):
    def test_live_callback_order_counts_and_zero_inner_rebuilds(self):
        h = Harness()
        with patch.object(static.sealed, 'walk_sealed', side_effect=h.walk), \
             patch.object(memory.seal.catalog, 'catalog', wraps=memory.seal.catalog.catalog) as catalogs, \
             patch.object(memory.seal.life, 'digest', wraps=memory.seal.life.digest) as digests:
            for _ in range(4):
                h.old_guard()
            previous = list(h.events)
            self.assertEqual(catalogs.call_count, 12)
            self.assertGreaterEqual(digests.call_count, 4)
            catalogs.reset_mock(); digests.reset_mock(); h.events.clear()
            for _ in range(4):
                self.assertEqual(h.new_guard(), h.permit)
            self.assertEqual(h.events, previous)
            self.assertEqual(catalogs.call_count, 0)
            self.assertEqual(digests.call_count, 0)

    def test_nested_mutations_fail_at_next_guard(self):
        changes = (
            lambda h: h.plan['source_sha256'].__setitem__(next(iter(h.plan['source_sha256'])), '0' * 64),
            lambda h: h.plan['inputs'][next(iter(h.plan['inputs']))]['identity'].__setitem__('size', 999),
            lambda h: h.contract['resources'].__setitem__('TasksMax', '129'),
            lambda h: h.contract['topology'].reverse(),
            lambda h: h.resolved.__setitem__('runtime_admitted', True),
            lambda h: h.resolved.__setitem__('plan_sha256', '0' * 64),
            lambda h: h.plan['native_operator_manifest'].__setitem__('new', True),
            lambda h: h.resolved['native_runtime_output_catalog']['payload_files'].__setitem__('extra.raw', 1),
        )
        for change in changes:
            with self.subTest(change=change):
                h = Harness(); change(h)
                with patch.object(static.sealed, 'walk_sealed', side_effect=h.walk):
                    with self.assertRaises(RuntimeError):
                        h.new_guard()

    def test_equal_copy_rebinds_rejected(self):
        changes = (
            lambda h: h.resolved.__setitem__('plan', copy.deepcopy(h.plan)),
            lambda h: h.plan.__setitem__('source_sha256', dict(h.plan['source_sha256'])),
            lambda h: h.resolved.__setitem__('native_runtime_output_catalog', copy.deepcopy(h.resolved['native_runtime_output_catalog'])),
            lambda h: setattr(h.ports, 'contract', copy.deepcopy(h.contract)),
            lambda h: setattr(h.ports, 'resolved', dict(h.resolved)),
        )
        for change in changes:
            with self.subTest(change=change):
                h = Harness(); change(h)
                with patch.object(static.sealed, 'walk_sealed', side_effect=h.walk):
                    with self.assertRaises(RuntimeError):
                        h.new_guard()

    def test_mutation_during_walk_rejected_before_guard_return(self):
        h = Harness()
        def changed(root, catalog, writers):
            rows = h.walk(root, catalog, writers)
            h.plan['source_sha256'][next(iter(h.plan['source_sha256']))] = '0' * 64
            return rows
        with patch.object(static.sealed, 'walk_sealed', side_effect=changed):
            with self.assertRaises(RuntimeError): h.new_guard()

    def test_mutable_output_root_never_controls_walk(self):
        h = Harness(); expected = h.contract['output_root']; h.contract['output_root'] = '/unbound/output'
        with patch.object(static.sealed, 'walk_sealed', side_effect=h.walk) as walk:
            with self.assertRaises(RuntimeError): h.new_guard()
        self.assertEqual(walk.call_args.args[0], expected)

    def test_exact_types_zero_cycles_and_shared_containers(self):
        class Text(str):
            pass
        for initial, replacement in ((0, False), (1, True), (1, 1.0), (0.0, -0.0), ('x', Text('x'))):
            value = {'x': initial}; facts = static.JsonFacts((value,)); value['x'] = replacement
            with self.assertRaises(RuntimeError):
                facts.check()
        key = {'x': 1}; facts = static.JsonFacts((key,)); del key['x']; key[Text('x')] = 1
        with self.assertRaises(RuntimeError): facts.check()
        for value in (float('nan'), float('inf'), (), object()):
            with self.assertRaises(RuntimeError): static.JsonFacts(({'x': value},))
        cycle = {}; cycle['self'] = cycle
        with self.assertRaises(RuntimeError): static.JsonFacts((cycle,))
        child = {'x': [1]}; value = {'a': child, 'b': child}; facts = static.JsonFacts((value,))
        self.assertEqual(len(facts._records), 3); facts.check()
        child['x'].append(2)
        with self.assertRaises(RuntimeError): facts.check()

    def test_private_catalog_and_binding_cannot_be_changed(self):
        h = Harness(); binding = h.ports.static_binding
        with self.assertRaises(TypeError): binding._catalog['payload_files']['other.raw'] = 1
        for obj, name, value in ((binding, '_catalog', {}), (binding._facts, '_records', ()),
                                 (h.ports, 'static_binding', binding), (h.ports, '_static_binding', binding)):
            with self.assertRaises(AttributeError): setattr(obj, name, value)
            with self.assertRaises(AttributeError): delattr(obj, name)

    def test_unrelated_history_not_snapshotted_or_serialized(self):
        h = Harness()
        h.resolved['history_only'] = {'opaque': object()}
        h.ports.static_binding.check(h.contract, h.resolved)
        h.ports.static_binding.verify_full(h.contract, h.resolved)

    def test_complete_and_partial_inventory_match_original(self):
        h = Harness(); c = h.resolved['native_runtime_output_catalog']
        self.assertEqual((len(c['payload_files']), len(c['driver_metadata_files']), len(c['driver_directories'])), (936, 144, 97))
        row = lambda kind, size: dict(kind=kind, uid=1000, nlink=1, symlink=False, size=size, blocks=(size+511)//512)
        complete = {name: row('file', size) for name, size in c['payload_files'].items()}
        complete.update({name: row('file', 2) for name in c['driver_metadata_files']})
        complete.update({name: row('directory', 4096) for name in c['driver_directories']})
        for entries, success in ((h.entries, False), (complete, True), ({}, False)):
            old = memory.seal.life.usage(h.contract, h.resolved, entries, h.root, successful_driver=success, writers=h.writers)
            new = current.usage(h.contract, h.resolved, entries, h.root, successful_driver=success,
                                writers=h.writers, static_binding=h.ports.static_binding)
            self.assertEqual(old, new)
        for mutate in (lambda rows: rows.pop(next(iter(c['payload_files']))),
                       lambda rows: rows[next(iter(c['payload_files']))].__setitem__('size', 0),
                       lambda rows: rows.__setitem__('unlisted.raw', row('file', 1))):
            bad = copy.deepcopy(complete); mutate(bad)
            for impl, kwargs in ((memory.seal.life.usage, {}), (current.usage, {'static_binding': h.ports.static_binding})):
                with self.assertRaises(RuntimeError):
                    impl(h.contract, h.resolved, bad, h.root, successful_driver=True, writers=h.writers, **kwargs)

    def test_original_caps_and_live_storage_failure(self):
        h = Harness()
        h.ports.storage_available = lambda: dict(root=27 * 2**30, output=40 * 2**30)
        with self.assertRaises(RuntimeError): h.new_guard()
        bad = dict(h.root, size=256 * 2**20, blocks=0)
        with self.assertRaises(RuntimeError):
            current.usage(h.contract, h.resolved, h.entries, bad, writers=h.writers, static_binding=h.ports.static_binding)
        self.assertEqual(current.PROPS, memory.seal.life.PROPS)

    def test_acknowledged_pending_writers_remain_live(self):
        h = Harness()
        entries = {'.pending-native-result-12.json': dict(kind='file', uid=1000, nlink=1,
                                                        symlink=False, size=2, blocks=8)}
        for writers, passes in ((h.writers, True), (dict(h.writers, native=13), False), ({}, False)):
            for impl, kwargs in ((memory.seal.life.usage, {}), (current.usage, {'static_binding': h.ports.static_binding})):
                if passes:
                    result = impl(h.contract, h.resolved, entries, h.root, writers=writers, **kwargs)
                    self.assertEqual(result['pending_destinations'], ['native-result.json'])
                else:
                    with self.assertRaises(RuntimeError):
                        impl(h.contract, h.resolved, entries, h.root, writers=writers, **kwargs)
        h.ports.read_owner = {'pid': 13}
        with patch.object(static.sealed, 'walk_sealed', side_effect=h.walk) as walk:
            h.new_guard()
        self.assertEqual(walk.call_args.args[2]['native'], 13)

    def test_existing_source_phase_restores_original_static_guard(self):
        from research.minicpm5_native_source_phase_seal_v3 import SourcePhase
        h = Harness()
        with h.patches():
            context = h.context(); context.external_guard = h.new_guard
            original_plan = context.plan
            with SourcePhase(h.contract, h.resolved, context, h.partition, h.lease, h.ports, memory.sources) as phase:
                phase.verify_text_sources()
            self.assertIs(context.plan, original_plan)
            self.assertEqual(context.external_guard, h.new_guard)
            self.assertFalse(context._sources_checked)
            context.guard('native')
        h.ports.static_binding.verify_full(h.contract, h.resolved)

    def test_full_binding_at_entry_and_final_boundary(self):
        h = memory.Harness()
        with patch.object(static.original, 'binding', wraps=static.original.binding) as binding:
            result = static.StaticBinding(h.contract, h.resolved)
            self.assertEqual(binding.call_count, 1)
            result.check(h.contract, h.resolved); result.check(h.contract, h.resolved)
            self.assertEqual(binding.call_count, 1)
            result.verify_full(h.contract, h.resolved)
            self.assertEqual(binding.call_count, 2)

    def test_lifecycle_dynamic_paths_unchanged(self):
        q = ROOT / 'research'
        old = function_nodes(q / 'minicpm5_layer1_operator_native_lifecycle_core_v2.py')
        new = function_nodes(q / 'minicpm5_layer1_operator_native_lifecycle_core_v3.py')
        self.assertEqual(set(old), set(new))
        for name in old:
            if name == 'usage':
                self.assertEqual(ast.dump(ast.Module(body=old[name].body[2:], type_ignores=[])),
                                 ast.dump(ast.Module(body=new[name].body[1:], type_ignores=[])))
                continue
            self.assertEqual(ast.dump(old[name]), ast.dump(RemoveStaticCalls().visit(new[name])), name)

    def test_runner_outer_preparer_only_intended_changes(self):
        pairs = (
            (ROOT/'minicpm5_layer1_operator_native_runner_v11.py', ROOT/'minicpm5_layer1_operator_native_runner_v12.py',
             {'minicpm5_layer1_operator_native_process_adapter_v9': 'minicpm5_layer1_operator_native_process_adapter_v8',
              'minicpm5_layer1_operator_native_lifecycle_core_v3': 'minicpm5_layer1_operator_native_lifecycle_core_v2'}),
            (ROOT/'minicpm5_layer1_operator_native_outer_runner_v4.py', ROOT/'minicpm5_layer1_operator_native_outer_runner_v5.py',
             {'minicpm5_layer1_operator_native_process_adapter_v9': 'minicpm5_layer1_operator_native_process_adapter_v7'}),
            (ROOT/'research/prepare_minicpm5_layer1_operator_native_v5.py', ROOT/'research/prepare_minicpm5_layer1_operator_native_v6.py',
             {'minicpm5_layer1_operator_native_process_adapter_v9': 'minicpm5_layer1_operator_native_process_adapter_v8',
              'minicpm5_layer1_operator_native_outer_runner_v5': 'minicpm5_layer1_operator_native_outer_runner_v4',
              'minicpm5_layer1_operator_native_runner_v12': 'minicpm5_layer1_operator_native_runner_v11'}),
        )
        for old, new, replacements in pairs:
            text = new.read_text()
            for before, after in replacements.items(): text = text.replace(before, after)
            self.assertEqual(ast.dump(ast.parse(old.read_text())), ast.dump(RemoveStaticCalls().visit(ast.parse(text))))

    def test_walker_and_inventory_keep_original_definitions(self):
        # Production calls exactly these original reviewed functions; no copy or
        # altered gate exists in this increment. Their own AST equivalence is
        # established by the original source-phase controls.
        self.assertIs(static.sealed.walk_sealed, memory.seal.walk_sealed)
        self.assertIs(static.sealed.inventory_sealed, memory.seal.inventory_sealed)
        tree = ast.parse((ROOT/'research/minicpm5_layer1_operator_native_process_adapter_v9.py').read_bytes())
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        self.assertEqual(ast.unparse(klass.bases[0]), 'StaticPortsMixin')
        self.assertEqual(ast.unparse(klass.bases[1]), 'previous.Ports')


def benchmark():
    h = Harness()
    pins = h.plan['source_sha256']
    for i in range(44503-len(pins)): pins[f'/synthetic/lifecycle-source-{i}.py'] = 'a' * 64
    inputs = h.plan['inputs']
    for i in range(8791-len(inputs)): inputs[f'/synthetic/lifecycle-input-{i}.json'] = dict(sha256='b'*64, identity=dict(size=i))
    h.resolved['plan_sha256'] = memory.original.digest(h.plan)
    h.contract = memory.seal.life.build_contract(h.resolved, h.contract['overlay'], h.contract['runner'], h.contract['unit'])
    h.ports = isolated_ports()(h.contract, h.resolved, {}, {'outer_owner': {'pid': 10}}, {'pid': 11}, h.lease, None)
    h.ports.read_owner = {'pid': 12}
    h.ports.storage_available = lambda: dict(root=40*2**30, output=40*2**30)
    times = {'old': [], 'new': []}
    with patch.object(static.sealed, 'walk_sealed', side_effect=h.walk):
        for _ in range(9):
            for name, callback in (('old', h.old_guard), ('new', h.new_guard)):
                start = time.perf_counter(); callback(); times[name].append(time.perf_counter()-start)
    values = {name: statistics.median(rows) for name, rows in times.items()}
    print(json.dumps(dict(source_only=True, model_imports=0, raw_payload_reads=0, native_bytes_read=0,
                         source_files=len(pins), inputs=len(inputs), iterations=9,
                         median_seconds=values, old_over_new=values['old']/values['new'],
                         scope='synthetic binding and catalog work; mocked live IO; not inference speedup')))


if __name__ == '__main__':
    if sys.argv[1:] == ['--benchmark']:
        benchmark()
    else:
        unittest.main()
