"""Pure regression tests for explicit integer control APIs; no SDK execution.

Opaque scalar tiles below are a 1x1 control-only test double, not a simulator.
Scientific suite/three-arm evidence is synthetic metadata from the existing
complete envelope. No model/native bytes, actual history or serial operations.
"""
import ast
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'ml_dtypes', 'torch', 'nki', 'neuronxcc', 'torch_neuronx'}:
            raise AssertionError('No real framework/native import in pure successor tests')


sys.meta_path.insert(0, DenyNative())
import lfm3_nki05_transport_diagnostic_v3 as diagnostic
from research import check_lfm3_nki05_characterization_delivery_v1 as envelope

KERNEL = ROOT / 'lfm3_nki05_transport_kernel_v2.py'
DIAGNOSTIC = ROOT / 'lfm3_nki05_transport_diagnostic_v3.py'
RUNNER = ROOT / 'run_lfm3_nki05_transport_serial_v4.py'


def functions(path):
    tree = ast.parse(path.read_bytes(), filename=str(path)); compile(tree, str(path), 'exec')
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def control_block(path):
    make = functions(path)['make_kernel']
    kernel = next(node for node in make.body if isinstance(node, ast.FunctionDef))
    destination = next(node for node in kernel.body if isinstance(node, ast.For))
    selector = destination.body[0]
    assert ast.unparse(selector.test) == "operation == 'selector'"
    return copy.deepcopy(selector.body)


class Tile:
    # Deliberately no overloaded __eq__, __sub__, conversions or payload math.
    def __init__(self, value, dtype='int32', shape=(1, 1)):
        self.value, self.dtype, self.shape = value, dtype, shape

    def reshape(self, shape):
        assert shape == (1, 1)
        return Tile(self.value, self.dtype, shape)


class Slots:
    def __init__(self, values): self.values = values
    def __getitem__(self, selection):
        values = self.values[selection]; assert len(values) == 1
        return Tile(values[0], 'int32', (1,))


class Language:
    int32, uint8, sbuf = 'int32', 'uint8', 'sbuf'
    static_range = staticmethod(range)
    def __init__(self): self.events = []
    def full(self, shape, value, *, dtype, buffer):
        assert shape == (1, 1) and dtype == 'int32' and buffer == 'sbuf'
        return Tile(value, dtype, shape)
    def load(self, tile): return tile
    def equal(self, left, right, *, dtype):
        assert type(left) is Tile and left.dtype == 'int32' and type(right) is int and dtype == 'uint8'
        self.events.append(('equal', left.value, right, dtype))
        return Tile(int(left.value == right), dtype)
    def where(self, condition, yes, no, *, dtype):
        assert condition.shape == (1, 1)  # Same required property as actual nl.where.
        assert condition.dtype == 'uint8' and condition.value in (0, 1) and dtype == 'int32'
        return Tile(yes.value if condition.value else no.value, dtype)
    def subtract(self, left, right, *, dtype):
        assert type(left) is Tile and left.dtype == dtype == 'int32' and type(right) is int
        self.events.append(('subtract', left.value, right, dtype))
        return Tile(left.value - right, dtype)
    def maximum(self, left, right): return Tile(max(left.value, right), left.dtype)
    def zeros(self, shape, *, dtype): return Tile(0, dtype, shape)
    def add(self, left, right, *, dtype):
        assert left.shape == (1, 128) and dtype == 'uint8'
        return Tile(left.value + right.value, dtype, left.shape)


def run_control(path, mapping, destination, *, rewrite_equal_only=False):
    body = control_block(path)
    if rewrite_equal_only:
        class EqualOnly(ast.NodeTransformer):
            def visit_Compare(self, node):
                if ast.unparse(node) == 'mapping == destination':
                    return ast.copy_location(ast.parse('nl.equal(mapping, destination, dtype=nl.uint8)', mode='eval').body, node)
                return self.generic_visit(node)
        body = [EqualOnly().visit(node) for node in body]
    scope = dict(nl=Language(), rows=len(mapping), slots=Slots(mapping), destination=destination)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), '<control-only synthetic AST>', 'exec'), scope)
    return scope


def runner_scope():
    scope = dict(p=NS(require=envelope.require), diagnostic=diagnostic, Path=Path, json=json)
    envelope.definitions(RUNNER, {'require_diagnostic_success', 'require_characterization', 'execute_diagnostic', 'encode'}, scope)
    return scope


class Tests(unittest.TestCase):
    def test_01_original_bool_shape_and_next_subtraction_failure_reproduced_purely(self):
        legacy = ROOT / 'lfm3_nki05_transport_diagnostic_v1.py'
        with self.assertRaisesRegex(AttributeError, "bool.*shape"):
            run_control(legacy, [0], 0)
        with self.assertRaisesRegex(TypeError, 'unsupported operand'):
            run_control(legacy, [0], 0, rewrite_equal_only=True)

    def test_02_explicit_controls_retain_last_writer_inactive_and_integer_dtype(self):
        cases = 0
        for capacity in (4, 8, 16):
            mappings = [[0], [-1, capacity, -(2**31), 2**31-1], [0, 0, -1, 0],
                        [capacity-1, capacity-1, -1, capacity], list(reversed(range(capacity)))]
            for mapping in mappings:
                for destination in range(capacity):
                    result = run_control(KERNEL, mapping, destination)
                    winners = [row for row, slot in enumerate(mapping) if slot == destination]
                    selected = winners[-1] + 1 if winners else 0
                    self.assertEqual(result['selected'].value, selected)
                    self.assertEqual(result['source_row'].value, max(selected-1, 0))
                    self.assertEqual(result['source_row'].dtype, 'int32')
                    self.assertEqual(result['predicate'].value, selected)
                    self.assertEqual(result['predicate'].dtype, 'uint8')
                    self.assertEqual(result['predicate'].shape, (1, 128))
                    self.assertEqual(len(result['nl'].events), len(mapping)+1)
                    cases += 1
        self.assertEqual(cases, 140)

    def test_03_exact_two_kernel_changes_and_unchanged_byte_observer(self):
        old = functions(ROOT / 'lfm3_nki05_transport_diagnostic_v1.py')
        new = functions(KERNEL)
        self.assertEqual(set(new), {'make_kernel', 'run_case'})
        class ExplicitControls(ast.NodeTransformer):
            def visit_Compare(self, node):
                if ast.unparse(node) == 'mapping == destination':
                    return ast.parse('nl.equal(mapping, destination, dtype=nl.uint8)', mode='eval').body
                return self.generic_visit(node)
            def visit_BinOp(self, node):
                if ast.unparse(node) == 'selected - 1':
                    return ast.parse('nl.subtract(selected, 1, dtype=nl.int32)', mode='eval').body
                return self.generic_visit(node)
        expected = ExplicitControls().visit(copy.deepcopy(old['make_kernel']))
        self.assertEqual(ast.dump(expected), ast.dump(new['make_kernel']))
        self.assertEqual(ast.dump(old['run_case']), ast.dump(new['run_case']))
        self.assertIs(diagnostic.original.run_case.__globals__['make_kernel'], diagnostic.original.make_kernel)

    def test_04_full_suite_and_three_arm_gates_unchanged_except_trace_and_source_metadata(self):
        old = functions(ROOT / 'lfm3_nki05_transport_diagnostic_v2.py'); new = functions(DIAGNOSTIC)
        self.assertEqual(set(new)-set(old), {'bounded_error_trace'})
        class RemoveOnlyNewTrace(ast.NodeTransformer):
            def visit_Assign(self, node):
                if any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                       and target.slice.value == 'error_trace' for target in node.targets): return None
                return self.generic_visit(node)
            def visit_Call(self, node):
                node.keywords = [keyword for keyword in node.keywords if keyword.arg != 'error_trace']
                return self.generic_visit(node)
        for name in old:
            after = RemoveOnlyNewTrace().visit(copy.deepcopy(new[name]))
            if name == 'diagnose':
                source_loop = next(node for node in ast.walk(after) if isinstance(node, ast.For)
                                   and ast.unparse(node.target) == 'path')
                self.assertEqual(ast.unparse(source_loop.iter.elts[-1]), "ROOT / 'lfm3_nki05_transport_diagnostic_v1.py'")
                source_loop.iter.elts.pop()
            self.assertEqual(ast.dump(old[name]), ast.dump(after), name)

    def test_05_entire_runner_lifecycle_verifier_and_limits_unchanged(self):
        old = functions(ROOT / 'run_lfm3_nki05_transport_serial_v3.py'); new = functions(RUNNER)
        self.assertEqual(set(old), set(new))
        for name in old: self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        def normalized(path):
            tree = ast.parse(path.read_bytes())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname == 'diagnostic': alias.name = 'SAME_DIAGNOSTIC_INTERFACE'
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'AUTHOR' for t in node.targets):
                    node.value = ast.Constant('SOURCE_AUTHOR')
            return ast.dump(tree)
        self.assertEqual(normalized(ROOT / 'run_lfm3_nki05_transport_serial_v3.py'), normalized(RUNNER))

    def test_06_real_python_exception_trace_retains_inner_location_without_locals(self):
        def deepest():
            private_local = b'not exported'
            raise AttributeError("'bool' object has no attribute 'shape'")
        def recurse(depth):
            if depth: return recurse(depth-1)
            return deepest()
        try: recurse(8)
        except AttributeError as error: trace = diagnostic.bounded_error_trace(error)
        self.assertEqual(trace['error_type'], 'AttributeError')
        self.assertEqual(len(trace['innermost_frames']), 4)
        self.assertGreater(trace['omitted_outer_frames'], 0)
        self.assertEqual(trace['innermost_frames'][-1]['function'], 'deepest')
        self.assertEqual(trace['innermost_frames'][-1]['filename'], str(Path(__file__)))
        self.assertFalse(trace['frame_locals_captured']); self.assertFalse(trace['source_text_read'])
        self.assertNotIn('private_local', json.dumps(trace)); self.assertNotIn('not exported', json.dumps(trace))

    def test_07_trace_string_bounds_and_no_traceback_case(self):
        cursor = None
        for index in reversed(range(10)):
            cursor = NS(tb_frame=NS(f_code=NS(co_filename='X'*1000+str(index), co_name='Y'*1000)),
                        tb_lineno=index+1, tb_next=cursor)
        trace = diagnostic.bounded_error_trace(NS(__traceback__=cursor, filename='Z'*1000))
        self.assertEqual(trace['total_frames'], 10); self.assertEqual(trace['omitted_outer_frames'], 6)
        self.assertEqual([frame['line'] for frame in trace['innermost_frames']], [7, 8, 9, 10])
        self.assertEqual(len(trace['exception_filename']), 192)
        self.assertTrue(all(len(frame['filename']) == 192 and len(frame['function']) == 64 for frame in trace['innermost_frames']))
        self.assertEqual(diagnostic.bounded_error_trace(RuntimeError('no stack'))['innermost_frames'], [])

    def test_08_run_case_records_inner_error_and_resets_label(self):
        class BytesOnly:
            dtype = NS(name='bfloat16'); shape = (1, 2)
            def tobytes(self): return b'\0\0\x80\x3f'
        probe = NS(installed=True, label=None, case_errors=[])
        def fail(*args): raise AttributeError("'bool' object has no attribute 'shape'")
        with patch.object(diagnostic, '_ACTIVE_PROBE', probe), patch.object(diagnostic.original, 'run_case', fail):
            with self.assertRaises(AttributeError): diagnostic.run_case(None, 'synthetic', 'selector', *[BytesOnly() for _ in range(3)])
        self.assertIsNone(probe.label); self.assertEqual(len(probe.case_errors), 1)
        entry = probe.case_errors[0]
        self.assertEqual(entry['error_trace']['innermost_frames'][-1]['function'], 'fail')
        self.assertEqual(len(entry['host_input_differences']), 3)
        self.assertTrue(all(row['difference']['changed_words'] == 0 for row in entry['host_input_differences']))

    def test_09_complete_three_arm_gate_still_requires_all_counts_and_failure(self):
        scope = runner_scope(); report = envelope.representative()
        status = scope['require_characterization'](report)
        self.assertEqual((status['movement_cases'], status['bf16_encoding_count'], status['extraction_calls'],
                          status['precise_bf16_writeback_substitutions']), (32, 65536, 132, 40))
        self.assertTrue(status['nonzero_worker_exit_required']); self.assertFalse(status['original_passed'])
        for change in ('scope', 'encodings', 'events', 'substitutions', 'restoration', 'negative', 'arithmetic', 'original', 'promoted'):
            value = copy.deepcopy(report)
            if change == 'scope': value['candidate']['suite']['checks'].pop()
            elif change == 'encodings': value['candidate']['suite']['bf16_encoding_count'] = 65535
            elif change == 'events': value['candidate']['probe']['extraction_calls'].pop()
            elif change == 'substitutions': value['candidate']['probe']['substitutions'] = 39
            elif change == 'restoration': value['restored_original']['probe']['restored'] = False
            elif change == 'negative': value['candidate']['suite']['negative_controls']['wrong_dtype'] = False
            elif change == 'arithmetic': value['candidate']['suite']['arithmetic_control'] = False
            elif change == 'original': value['original_suite_passed'] = True
            else: value['full_model_qualified'] = True
            with self.subTest(change=change), self.assertRaises(RuntimeError): scope['require_characterization'](value)

    def test_10_complete_evidence_written_then_nonzero_worker_failure_preserved(self):
        scope = runner_scope(); report = envelope.representative(); writes = {}
        scope.update(diagnostic=NS(diagnose=lambda admit: report, negative_controls=diagnostic.negative_controls, OLD_HASHES=diagnostic.OLD_HASHES),
            once=lambda path, value: writes.setdefault(path.name, copy.deepcopy(value)),
            reference=lambda path: dict(path=str(path), sha256='a'*64),
            report_failure=lambda path, error, **context: writes.setdefault(path.name, dict(error=str(error), **context)))
        with self.assertRaisesRegex(RuntimeError, 'Characterization complete; original implementation failed'):
            scope['execute_diagnostic']({}, {}, Path('/synthetic/no-output-created'), lambda: None)
        self.assertEqual(set(writes), {'diagnostic-result.json', 'characterization-status.json', 'worker-failure.json'})
        self.assertFalse(writes['characterization-status.json']['original_passed'])
        self.assertTrue(writes['characterization-status.json']['nonzero_worker_exit_required'])

    def test_11_complete_report_and_bounded_failure_trace_still_fit_original_cap(self):
        report = envelope.representative()
        trace = dict(error_type='X'*64, exception_filename='Y'*192,
            innermost_frames=[dict(filename='F'*192, function='N'*64, line=2147483647) for _ in range(4)],
            total_frames=1000, omitted_outer_frames=996, frame_limit=4, frame_locals_captured=False, source_text_read=False)
        # Conservative metadata envelope: retain all32 complete candidate rows
        # plus a failure and its case trace, although a failed case normally is
        # absent from rows. This is serialization only, not real replay.
        for name in ('original', 'candidate', 'restored_original'):
            report[name]['suite']['error_trace'] = copy.deepcopy(trace)
        report['candidate']['suite']['error'] = 'E'*4096
        report['candidate']['probe']['case_errors'] = [dict(name='N'*64, operation='selector', error='E'*2048,
            error_trace=copy.deepcopy(trace), host_input_differences=copy.deepcopy(report['candidate']['suite']['checks'][-1]['host_input_differences']))]
        report['source_sha256'][str(KERNEL)] = 'f'*64
        compact = len(runner_scope()['encode'](report)); default = len(json.dumps(report, allow_nan=False).encode())
        self.assertLessEqual(compact, 256*2**10); self.assertLessEqual(default, 256*2**10)
        Tests.envelope_sizes = dict(compact_bytes=compact, diagnostic_default_json_bytes=default, cap_bytes=256*2**10)

    def test_12_frozen_executed_files_and_installed_api_sources(self):
        frozen = {
            'lfm3_nki05_transport_diagnostic_v1.py': '62fcdb4faba6c26420f2a5d29123b704b297216f7f4ae6d3c916cca4d547fb55',
            'lfm3_nki05_transport_diagnostic_v2.py': 'cf83c634b0f71f5d8a483704b7fed2bdad2588736866d81f28d19416ac11263a',
            'run_lfm3_nki05_transport_serial_v3.py': 'c8a4dfbc29b7b8aab8f2634c4ec245ce38d7cdeb5ebc84ad1217cc7e1ee2bcd8',
            'lfm3_nki05_writeback_probe_v1.py': 'abdb58d5acc086ac11f177fa98d38b6424d5fa02d9612e5343c534331534e073',
            'lfm3_nki05_torch_namespace_sources_v1.py': 'f220e415895c2a265939cbf7fc7bf30cce09f44ec92eadf68179aac41a7b191c'}
        for name, sha in frozen.items(): self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), sha, name)
        sdk = Path(diagnostic.SDK)
        for name, sha in {'language/_operators.py': '2a10f359a099f2b0630746eca236b1a4278d0e0e584c0215084b9bb848e36789',
                          'language/_math_funcs.py': 'd3d985e6eab4d5144d42c49f46b1faa5c995b61d83fcddc4cd18e4f4067bb559',
                          'language/tensor.py': '66e6c75be3a340475b1affa1b69b820e500fa2953159c13884ce0d4199926691',
                          '_simulator.py': '3c28f10492ef8c6c5413acbd811c059586f7ff16d3ef795430e794c57ed8f119'}.items():
            self.assertEqual(hashlib.sha256((sdk/name).read_bytes()).hexdigest(), sha, name)
        for filename, class_name in [('language/tensor.py', 'NkiTensor'), ('language/_nki_base.py', 'NKIObject')]:
            node = functions(sdk/filename)[class_name]
            self.assertTrue({'__eq__', '__sub__'}.isdisjoint({row.name for row in node.body if isinstance(row, ast.FunctionDef)}))
        operators = functions(sdk/'language/_operators.py')
        self.assertIn('equal', operators); self.assertIn('subtract', operators)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(source_only=True, synthetic_control_cases=140, actual_native_imports=False,
        actual_model_or_capture_bytes_read=0, actual_history_resolved=False, actual_serial_operations=False,
        envelope_sizes=getattr(Tests, 'envelope_sizes', None)), sort_keys=True))
    raise SystemExit(not result.wasSuccessful())
