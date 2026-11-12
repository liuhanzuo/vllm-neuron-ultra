"""Standard-library source/mocks only; no native imports or runtime admission."""
import ast
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
R = Q.parent
sys.path[:0] = [str(Q), str(R)]


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_xla', 'torch_neuronx', 'numpy', 'ctypes', '_ctypes', 'neuronxcc', 'nki', 'transformers', 'vllm', 'libneuronxla'}:
            raise AssertionError('Native import forbidden in source/mock check: ' + fullname)


sys.meta_path.insert(0, RejectNative())
import qwen35_predicated_slot_protocol_v1 as original
import qwen35_integer_reference_protocol_v1 as protocol
import qwen35_integer_reference_device_leaf_v1 as leaf
import qwen35_integer_slot_reference_v1 as reference


def dump(source):
    return ast.dump(ast.parse(source), include_attributes=False)


class ProtocolTests(unittest.TestCase):
    def test_every_original_science_field_and_function_retained(self):
        old, new = original.workload(), protocol.workload()
        for key in old:
            if key != 'schema':
                self.assertEqual(new[key], old[key], key)
        for name in ('layout', 'counts', 'paired_order', 'sequences', 'compare_storage', 'measured_call', 'digest_file'):
            self.assertIs(getattr(protocol, name), getattr(original, name))
        self.assertEqual(list(protocol.sequences()), list(original.sequences()))
        self.assertEqual(protocol.counts()['total'], dict(graphs=32, correctness_calls=40000, warmup_calls=1024, measured_calls=5120, reset_calls=224, total_runtime_calls=46368))

    def test_reference_identity_cannot_claim_old_baseline_success(self):
        self.assertEqual(protocol.REFERENCE_IMPLEMENTATION['integer_select_types'], ['s16', 's32'])
        self.assertTrue(protocol.REFERENCE_IMPLEMENTATION['original_floating_reference_failed_status_preserved'])
        self.assertTrue(protocol.REFERENCE_IMPLEMENTATION['original_old_value_gather_unchanged'])
        self.assertFalse(protocol.REFERENCE_IMPLEMENTATION['original_model_baseline_speedup_claim'])
        rows = [dict(round=i, order=list(original.paired_order(i)), elapsed_ns=dict(reference=[20]*16, candidate=[10]*16)) for i in range(10)]
        new = protocol.summarize_pairs(rows)
        self.assertEqual({k:v for k,v in new.items() if k != 'reference_implementation'}, original.summarize_pairs(rows))
        self.assertFalse(new['model_speedup_claim'])

    def test_actual_failure_and_success_receipts_remain_hash_bound(self):
        protocol.verify_sources()
        old, diagnostic, final = [json.loads(Path(row['path']).read_text()) for row in protocol.REFERENCE_EVIDENCE]
        self.assertNotEqual(old['actual_tool']['exit_code'], 0)
        self.assertEqual(diagnostic['actual_tool'], dict(kind='tool_session', session_id=22831, terminal_chunk='a07595', exit_code=0))
        self.assertEqual(diagnostic['runtime_calls'], 12)
        self.assertTrue(final['passed'])
        self.assertFalse(diagnostic['full_model_qualified'])
        pins = dict(protocol.FROZEN_SOURCES)
        with patch.object(protocol, 'FROZEN_SOURCES', dict(pins, **{'qwen35_integer_slot_reference_v1.py':'0'*64})), self.assertRaises(RuntimeError):
            protocol.verify_sources()
        evidence = copy.deepcopy(protocol.REFERENCE_EVIDENCE)
        evidence[0]['sha256'] = '0'*64
        with patch.object(protocol, 'REFERENCE_EVIDENCE', evidence), self.assertRaises(RuntimeError):
            protocol.verify_sources()

    def test_leaf_exact_delta_preserves_oracle_old_read_and_full_loop(self):
        old = (R/'qwen35_predicated_slot_device_leaf_v1.py').read_text()
        new = (R/'qwen35_integer_reference_device_leaf_v1.py').read_text()
        new = new.replace('import qwen35_integer_reference_protocol_v1 as protocol', 'import qwen35_predicated_slot_protocol_v1 as protocol')
        new = new.replace('"schema": "qwen35_integer_reference_primitive_result_v1", "case": case,\n              "reference_implementation": dict(protocol.REFERENCE_IMPLEMENTATION),', '"schema": "qwen35_predicated_slot_primitive_result_v1", "case": case,')
        new = new.replace('        from qwen35_integer_slot_reference_v1 import make_integer_writer\n        integer_writer = make_integer_writer(case["dtype"], case["shape"])\n\n', '')
        new = new.replace('writer = integer_writer if variant == "reference" else store_slots_nki_predicated', 'writer = store_slots if variant == "reference" else store_slots_nki_predicated')
        self.assertEqual(dump(new), dump(old))

    def test_serial_entire_body_only_namespace_and_schema_changes(self):
        old = (Q/'qwen35_predicated_slot_serial_v3.py').read_text()
        new = (Q/'qwen35_integer_reference_serial_v1.py').read_text()
        new = new.replace('qwen35_integer_reference_serial_contract_v1', 'qwen35_predicated_slot_serial_contract_v3')
        new = new.replace('qwen35_integer_reference_device_leaf_v1', 'qwen35_predicated_slot_device_leaf_v1')
        for part in ('layout_binding', 'serial_admission', 'outer_terminal', 'reconciliation'):
            new = new.replace('qwen35_integer_reference_'+part+'_v1', 'qwen35_predicated_slot_'+part+'_v3')
        new = new.replace('qwen35_integer_reference_running_handoff_required', 'qwen35_predicated_slot_running_handoff_required')
        self.assertEqual(dump(new), dump(old))

    def test_contract_resource_environment_scheduler_functions_identical(self):
        old = ast.parse((Q/'qwen35_predicated_slot_serial_contract_v3.py').read_text())
        new = ast.parse((Q/'qwen35_integer_reference_serial_contract_v1.py').read_text())
        functions = lambda tree:{n.name:ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        a, b = functions(old), functions(new)
        changed = {'names', 'validate', 'verify_delivery', 'validate_report'}
        self.assertEqual(set(a), set(b))
        self.assertEqual({k:v for k,v in a.items() if k not in changed}, {k:v for k,v in b.items() if k not in changed})

    def test_refusal_at_native_boundary_retains_failed_report(self):
        reports = []
        def guard(event):
            if event == 'before_native_imports':
                raise PermissionError('mock closed native boundary')
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(PermissionError, 'closed native'):
            leaf.run_case('conv', Path(tmp)/'absent', guard=guard, save_report=lambda r:reports.append(copy.deepcopy(r)), record_artifact=lambda _:self.fail('compiler cannot run'))
        self.assertFalse(reports[-1]['complete'])
        self.assertFalse(reports[-1]['passed'])
        self.assertEqual(reports[-1]['graphs'], [])
        self.assertEqual(reports[-1]['reference_implementation'], protocol.REFERENCE_IMPLEMENTATION)

    def test_native_imports_are_only_within_original_boundary(self):
        for module in (protocol, leaf, reference):
            tree = ast.parse(Path(module.__file__).read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse({a.name.split('.')[0] for a in node.names} & {'torch', 'numpy', 'torch_neuronx'})
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module.split('.')[0], {'torch', 'numpy', 'torch_neuronx'})
        for name in ('_as_s16', '_as_s32', '_as_bf16', '_as_f32'):
            tree = ast.parse(Path(reference.__file__).read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].func.attr, 'bitcast')


class BitcastWiringTests(unittest.TestCase):
    def fixture(self):
        events = []
        class Tensor:
            def __init__(self, rows, dtype, shape):
                self.rows, self.dtype, self.shape = copy.deepcopy(rows), dtype, tuple(shape)
            def bitcast(self, dtype):
                events.append((self.dtype, dtype))
                self_dtype = {'bf16':'bfloat16', 'f32':'float32'}.get(dtype, dtype)
                return Tensor(self.rows, self_dtype, self.shape)
        torch = types.ModuleType('torch')
        torch.bfloat16, torch.float32, torch.int64 = 'bfloat16', 'float32', 'int64'
        base = types.ModuleType('torch_neuronx.xla_impl.base')
        base.xla_call = lambda fn, name: fn
        old = types.ModuleType('qwen35_text_port')
        def store(cache, updates, slots):
            self.assertEqual(cache.dtype, updates.dtype)
            self.assertIn(cache.dtype, ('s16', 's32'))
            output = copy.deepcopy(cache.rows)
            for index, dest in enumerate(slots.values):
                if dest >= 0:
                    output[dest] = copy.deepcopy(updates.rows[index])
            return Tensor(output, cache.dtype, cache.shape)
        old.store_slots = store
        modules = {'torch':torch, 'torch_neuronx.xla_impl.base':base, 'qwen35_text_port':old}
        return Tensor, events, modules

    def test_both_integer_widths_all_layouts_all625_last_writer_mappings(self):
        Tensor, events, modules = self.fixture()
        for _, dtype, shape in original.LAYOUTS:
            bits = [0, 0x8000, 0x7fc1, 0xffc1, 0xffff] if dtype == 'bfloat16' else [0,0x80000000,0x7fc00001,0xffc00001,0xffffffff]
            cache_rows = [bits[:] for _ in range(4)]
            update_rows = [[row + 1] + bits for row in range(4)]
            with patch.dict(sys.modules, modules):
                writer = reference.make_integer_writer(dtype, shape)
                for mapping in original.CORRECTNESS_MAPPINGS:
                    cache, updates = Tensor(cache_rows,dtype,shape), Tensor(update_rows,dtype,shape)
                    slots = types.SimpleNamespace(shape=(4,), dtype='int64', values=mapping)
                    before = len(events)
                    actual = writer(cache, updates, slots)
                    expected = [update_rows[max(i for i,d in enumerate(mapping) if d == dest)] if dest in mapping else cache_rows[dest] for dest in range(4)]
                    self.assertEqual(actual.rows, expected)
                    self.assertEqual((actual.dtype, actual.shape), (dtype, tuple(shape)))
                    target, back = ('s16','bf16') if dtype == 'bfloat16' else ('s32','f32')
                    self.assertEqual(events[before:], [(dtype,target),(dtype,target),(target,back)])
                    self.assertEqual(cache.rows, cache_rows)
                    self.assertEqual(updates.rows, update_rows)

    def test_bad_shape_dtype_and_slot_type_fail_before_bitcast(self):
        Tensor, events, modules = self.fixture()
        with self.assertRaises(ValueError):
            reference.make_integer_writer('float32', (4,3072,4))
        with patch.dict(sys.modules, modules):
            writer = reference.make_integer_writer('bfloat16', (4,3072,4))
            a = Tensor([[0]]*4,'bfloat16',(4,3072,4))
            slots = types.SimpleNamespace(shape=(4,),dtype='int64',values=(0,1,2,3))
            invalid = [Tensor([[0]]*4,'float32',(4,3072,4)), Tensor([[0]]*4,'bfloat16',(4,3072,3))]
            for bad in invalid:
                with self.assertRaises(ValueError): writer(bad,a,slots)
                with self.assertRaises(ValueError): writer(a,bad,slots)
            for bad in [types.SimpleNamespace(shape=(3,),dtype='int64'), types.SimpleNamespace(shape=(4,),dtype='int32')]:
                with self.assertRaises(ValueError): writer(a,a,bad)
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
