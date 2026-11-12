"""Source/mock checks only; does not claim native bitcast/select support."""
import ast
import hashlib
from pathlib import Path
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
BLOCKED = {'torch', 'torch_xla', 'torch_neuronx', 'numpy', 'ctypes', '_ctypes',
           'neuronxcc', 'libneuronxla', 'ml_dtypes', 'transformers', 'vllm'}


class BlockNative:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise AssertionError('Native import forbidden: ' + name)


assert not BLOCKED.intersection(sys.modules)
sys.meta_path.insert(0, BlockNative())
import qwen35_slot_select_bitcast_proposal_v1 as proposal
import qwen35_slot_reference_canonicalization_evidence_v1 as inputs


class Tensor:
    def __init__(self, rows, dtype):
        self.rows = [list(row) for row in rows]
        self.dtype = dtype
        self.shape = (len(rows), 3072, 4)
        self.ndim = 3
        self.device = 'source-mock'

    def __getitem__(self, item):
        return Tensor(self.rows[item], self.dtype)

    def bitcast(self, dtype):
        return Tensor(self.rows, {'s16': 'int16', 'bf16': 'bfloat16'}[dtype])

    def payload(self):
        return struct.pack('<%dH' % sum(map(len, self.rows)),
                           *(word for row in self.rows for word in row))


class Mask:
    def __init__(self, values):
        self.values = values

    def reshape(self, shape):
        assert shape in ((4, 1, 1), (-1, 1, 1))
        return self


class Indices:
    def __eq__(self, slot):
        return Mask([index == slot for index in range(4)])


class Module:
    def eval(self):
        return self


class SourceTests(unittest.TestCase):
    def test_inert_import_and_fixed_twelve_call_schedule(self):
        self.assertFalse(BLOCKED.intersection(sys.modules))
        self.assertEqual(len(proposal.schedule()), 12)
        self.assertEqual(len({tuple(row['slots']) for row in proposal.schedule()}), 2)
        self.assertEqual(proposal.SHAPE, (4, 3072, 4))

    def test_pinned_original_writer_is_used_and_not_edited(self):
        source = (Q.parent / 'qwen35_text_port.py').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '3b560a3dcbb925a4c010a679ebe8c5532178672d77edbe768d490e307cdd2e34')
        tree = ast.parse((Q / 'qwen35_slot_select_bitcast_proposal_v1.py').read_text())
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == 'qwen35_text_port'
                            and [item.name for item in node.names] == ['store_slots'] for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in ('to', 'float')
                             for node in ast.walk(tree)))

    def test_actual_sdk_bitcast_dispatch_is_distinct_from_numeric_convert(self):
        site = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
        tree = ast.parse((site / 'torch_xla/core/xla_builder.py').read_text())
        op = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Op')
        methods = {node.name: node for node in op.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(ast.unparse(methods['bitcast'].body[0]),
                         "return mkop('BitcastConvert', (self.op,), to_type=to_type)")
        self.assertEqual(ast.unparse(methods['cast'].body[0]),
                         "return mkop('Convert', (self.op,), to_type=to_type)")
        base = ast.parse((site / 'torch_neuronx/xla_impl/base.py').read_text())
        function = next(node for node in base.body if isinstance(node, ast.FunctionDef) and node.name == 'xla_call')
        self.assertEqual(ast.unparse(function.body[-1]), 'return xla_op_registry.register(name, opfn)')

    def test_all_65536_bf16_encodings_survive_integer_storage_reinterpretation(self):
        # Exhaustive host storage arithmetic only, not a device test.
        raw = struct.pack('<65536H', *range(65536))
        signed = struct.unpack('<65536h', raw)
        self.assertEqual(struct.pack('<65536h', *signed), raw)
        self.assertEqual(signed[0x8000], -32768)
        self.assertEqual(signed[0xffa1], -95)

    def test_real_proposal_mock_for_all_twelve_calls_uses_complete_raw_oracle(self):
        def where(mask, updates, cache):
            return Tensor([updates.rows[0] if active else row
                           for active, row in zip(mask.values, cache.rows)], cache.dtype)

        torch = NS(nn=NS(Module=Module), bfloat16='bfloat16', int16='int16', int64='int64',
                   arange=lambda n, device: Indices(), where=where)
        source = ast.parse((Q.parent / 'qwen35_text_port.py').read_text())
        writer = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'store_slots')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[writer], type_ignores=[]), '<frozen-store-slots-source-only>', 'exec'), namespace)
        registered = []

        def xla_call(function, name):
            registered.append(name)
            return function

        mocks = {'torch': torch, 'torch_neuronx.xla_impl.base': NS(xla_call=xla_call),
                 'qwen35_text_port': NS(store_slots=namespace['store_slots'])}
        with patch.dict(sys.modules, mocks):
            modules = proposal.make_modules()
            for call in proposal.schedule():
                negative = call['profile'] == 'negative_nan_raw'
                raw_cache, raw_updates = inputs.input_bytes(0, negative_nan=negative), inputs.input_bytes(17, negative_nan=negative)
                dtype = 'int16' if call['arm'] == 'external_s16_select' else 'bfloat16'
                def tensor(raw):
                    words = struct.unpack('<49152H', raw)
                    return Tensor([words[index:index+12288] for index in range(0, 49152, 12288)], dtype)
                cache, updates = tensor(raw_cache), tensor(raw_updates)
                class Slots(list):
                    shape = (4,)
                    dtype = 'int64'
                slots = Slots(call['slots'])
                actual = modules[call['arm']].forward(cache, updates, slots)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.payload(), proposal.byte_oracle(raw_cache, raw_updates, slots))
                self.assertEqual(cache.payload(), raw_cache)
                self.assertEqual(updates.payload(), raw_updates)
        self.assertEqual(registered, ['qwen35_select_probe_as_s16_v1', 'qwen35_select_probe_as_bf16_v1'])
        self.assertFalse(BLOCKED.intersection(sys.modules))


if __name__ == '__main__':
    result = unittest.main(exit=False, verbosity=2)
    assert not BLOCKED.intersection(sys.modules)
    raise SystemExit(not result.result.wasSuccessful())
