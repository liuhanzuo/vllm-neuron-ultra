"""Stdlib-only mocks plus AST checks against fixed installed SDK source text.

No SDK modules or native libraries are imported. No model/artifact/raw input,
History, lock, compiler or device is opened. Synthetic tensors contain no data.
"""
import ast
import copy
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parent
SDK = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
BLOCKED = {'torch', 'torch_xla', 'torch_neuronx', 'neuronxcc', 'numpy',
           'nki', 'ctypes', 'vllm', 'vllm_neuron'}


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('Native import forbidden in source test: ' + fullname)


assert not BLOCKED.intersection(sys.modules), 'Run tests in fresh stdlib process'
sys.meta_path.insert(0, NoNative())
MODULE_PATH = ROOT / 'qwen35_kv256_s16_concat_control_v1.py'
spec = importlib.util.spec_from_file_location('kv256_s16_control_source_test', MODULE_PATH)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def node(path, *names):
    tree = ast.parse(path.read_text())
    for name in names:
        tree = next(n for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    return tree


def sdk_function(path, names, globals_):
    """Execute only a selected pure function definition, with explicit fake globals."""
    definition = copy.deepcopy(node(path, *names))
    definition.decorator_list = []
    module = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(globals_)
    exec(compile(module, str(path) + ':' + '.'.join(names), 'exec'), namespace)
    return namespace[definition.name]


BASE = SDK / 'torch_neuronx/xla_impl/base.py'
BUILDER = SDK / 'torch_xla/core/xla_builder.py'
REGISTRY = SDK / 'torch_xla/core/xla_op_registry.py'


class XOp:
    def __init__(self, sizes, dtype='bf16', expression=None):
        self.sizes, self.dtype, self.expression = tuple(sizes), dtype, expression
        self.op = self

    def shape(self):
        return types.SimpleNamespace(sizes=self.sizes, dtype=self.dtype)


class Tensor:
    def __init__(self, shape, dtype='torch.bfloat16', device='xla:0', label='tensor'):
        self.shape, self.dtype, self.device, self.label = tuple(shape), dtype, Device(device), label

    def clone(self):
        return Tensor(self.shape, self.dtype, str(self.device), ('clone', self.label))

    def clamp_min(self, value):
        return Tensor(self.shape, self.dtype, str(self.device), ('clamp_min', self.label, value))

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            return Tensor(self.shape, self.dtype, str(self.device), ('gather', self.label, key.label))
        sizes = list(self.shape)
        for axis, part in enumerate(key):
            self_start, self_end, step = part.indices(sizes[axis])
            assert step == 1
            sizes[axis] = self_end - self_start
        return Tensor(sizes, self.dtype, str(self.device), ('slice', self.label, key))


class Device:
    def __init__(self, text):
        self.text, self.type = text, text.split(':')[0]

    def __str__(self):
        return self.text


class Fixture:
    def __init__(self):
        self.events, self.registrations = [], []

        def mkop(operation, operands, **kwargs):
            self.events.append((operation, operands, kwargs))
            if operation == 'BitcastConvert':
                value, = operands
                return XOp(value.sizes, kwargs['to_type'], (operation, value, kwargs['to_type']))
            if operation == 'ConcatInDim':
                assert kwargs['dimension'] == 2
                assert len(operands) == 2 and all(x.dtype == 's16' for x in operands)
                sizes = list(operands[0].sizes)
                sizes[2] += operands[1].sizes[2]
                return XOp(sizes, 's16', (operation, operands, kwargs['dimension']))
            raise AssertionError('Unexpected lowering operation: ' + operation)

        XOp.bitcast = sdk_function(BUILDER, ('Op', 'bitcast'), {'mkop': mkop})
        XOp._extract_xla_ops = classmethod(sdk_function(BUILDER, ('Op', '_extract_xla_ops'), {}))
        XOp.concat_in_dim = classmethod(sdk_function(BUILDER, ('Op', 'concat_in_dim'),
                                                    {'mkop': mkop, 'Op': XOp}))
        self.xb = types.SimpleNamespace(Type=types.SimpleNamespace(S16='s16', BF16='bf16'), Op=XOp)
        owner = self

        class RegisteredOp:
            def __init__(self, name, opfn):
                owner.registrations.append((name, opfn))
                self.opfn = opfn

            def __call__(self, prefix, tail):
                result = self.opfn(XOp(prefix.shape, expression=prefix.label),
                                   XOp(tail.shape, expression=tail.label))
                return Tensor(result.sizes, 'torch.' + {'bf16': 'bfloat16'}[result.dtype],
                              str(prefix.device), result.expression)

        register = sdk_function(REGISTRY, ('register',), {'Op': RegisteredOp})
        self.xla_call = sdk_function(BASE, ('xla_call',),
                                    {'xla_op_registry': types.SimpleNamespace(register=register)})
        self.assemble = candidate._build_assembler(self.xla_call, self.xb)
        self.lower = self.registrations[0][1]


class LoweringTests(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()

    def test_actual_sdk_bitcast_concat_bodies_emit_only_expected_ops(self):
        output = self.f.assemble(Tensor(candidate.PREFIX_SHAPE, label='prefix'),
                                 Tensor(candidate.TAIL_SHAPE, label='tail'))
        self.assertEqual(output.shape, candidate.STATE_SHAPE)
        self.assertEqual(output.dtype, 'torch.bfloat16')
        self.assertEqual([e[0] for e in self.f.events],
                         ['BitcastConvert', 'BitcastConvert', 'ConcatInDim', 'BitcastConvert'])
        self.assertEqual([self.f.events[i][2]['to_type'] for i in (0, 1, 3)], ['s16', 's16', 'bf16'])
        self.assertEqual(self.f.events[2][2]['dimension'], 2)
        self.assertEqual(self.f.events[0][1][0].expression, 'prefix')
        self.assertEqual(self.f.events[1][1][0].expression, 'tail')
        self.assertEqual(self.f.registrations[0][0], candidate.OP_NAME)

    def test_reject_numeric_tensor_dtype(self):
        for dtype in ('torch.float32', 'torch.float16', 'torch.int16'):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                self.f.assemble(Tensor(candidate.PREFIX_SHAPE, dtype), Tensor(candidate.TAIL_SHAPE))

    def test_reject_other_bucket_or_row_geometry(self):
        for shape in ((4,1,128,256), (1,1,256,256), (4,1,256,128)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.f.assemble(Tensor(shape), Tensor(candidate.TAIL_SHAPE))

    def test_reject_bad_tail(self):
        with self.assertRaises(ValueError):
            self.f.assemble(Tensor(candidate.PREFIX_SHAPE), Tensor((4,1,4096,256)))

    def test_reject_cpu_and_mismatched_devices(self):
        for device in ('cpu', 'xla:1', 'privateuseone:0'):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.f.assemble(Tensor(candidate.PREFIX_SHAPE), Tensor(candidate.TAIL_SHAPE, device=device))

    def test_lowering_rejects_incorrect_operand_dtype(self):
        with self.assertRaises(ValueError):
            self.f.lower(XOp(candidate.PREFIX_SHAPE, 's16'), XOp(candidate.TAIL_SHAPE))
        self.assertEqual(self.f.events, [])

    def test_lowering_rejects_incorrect_operand_geometry(self):
        with self.assertRaises(ValueError):
            self.f.lower(XOp(candidate.PREFIX_SHAPE), XOp((4,1,3841,256)))

    def test_reject_changed_sdk_type_spellings(self):
        self.f.xb.Type.S16 = 's32'
        with self.assertRaises(ValueError):
            candidate._build_assembler(self.f.xla_call, self.f.xb)

    def test_reject_malformed_lowered_output(self):
        original = XOp.bitcast
        def bad(value, dtype):
            result = original(value, dtype)
            if dtype == 'bf16': result.sizes = (4,1,4095,256)
            return result
        XOp.bitcast = bad
        with self.assertRaises(ValueError):
            self.f.lower(XOp(candidate.PREFIX_SHAPE), XOp(candidate.TAIL_SHAPE))

    def test_native_factory_cannot_run_in_source_tests(self):
        with self.assertRaisesRegex(AssertionError, 'Native import forbidden'):
            candidate.make_assembler()


class ControlTests(unittest.TestCase):
    def fixture(self):
        class Module:
            def eval(self): return self
        def parameter(value, requires_grad):
            self.assertFalse(requires_grad)
            return value
        torch_mock = types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module, Parameter=parameter))
        self.writer_calls, self.assembly_calls = [], []
        def writer(prefix, updates, slots):
            self.writer_calls.append((prefix, updates, slots))
            return Tensor(prefix.shape, label='written-prefix')
        def assembler(prefix, tail):
            self.assembly_calls.append((prefix, tail))
            return Tensor(candidate.STATE_SHAPE, label='assembled-state')
        return candidate._build_control(torch_mock, assembler, Tensor(candidate.STATE_SHAPE, label='initial'), writer)

    def test_original_slice_writer_old_new_order_and_alias1(self):
        control = self.fixture()
        updates, slots = Tensor(candidate.PREFIX_SHAPE), Tensor((4,), 'torch.int32', label='slots')
        old, new = control.forward(updates, slots)
        self.assertEqual(old.label, ('gather', control.cache.label, ('clamp_min','slots',0)))
        self.assertEqual(new.label, 'assembled-state')
        self.assertEqual(candidate.alias_map(control), {control.cache: 1})
        prefix, same_updates, same_slots = self.writer_calls[0]
        self.assertEqual(prefix.shape, candidate.PREFIX_SHAPE)
        self.assertIs(same_updates, updates)
        self.assertIs(same_slots, slots)
        assembled_prefix, tail = self.assembly_calls[0]
        self.assertEqual(assembled_prefix.label, 'written-prefix')
        self.assertEqual(tail.shape, candidate.TAIL_SHAPE)
        self.assertEqual(tail.label[1], control.cache.label)

    def test_control_rejects_widened_slots(self):
        control = self.fixture()
        with self.assertRaises(ValueError):
            control.forward(Tensor(candidate.PREFIX_SHAPE), Tensor((4,), 'torch.int64'))
        self.assertEqual(self.writer_calls, [])

    def test_control_rejects_wrong_slot_count(self):
        control = self.fixture()
        with self.assertRaises(ValueError):
            control.forward(Tensor(candidate.PREFIX_SHAPE), Tensor((3,), 'torch.int32'))

    def test_alias_map_rejects_wrong_state_precision(self):
        control = self.fixture()
        control.cache.dtype = 'torch.int16'
        with self.assertRaises(ValueError): candidate.alias_map(control)


class SourceAPITests(unittest.TestCase):
    def test_fixed_sdk_signatures(self):
        checks = ((BASE, ('xla_call',), ['opfn','name'], 1),
                  (REGISTRY, ('register',), ['name','opfn'], 0),
                  (BUILDER, ('Op','bitcast'), ['self','to_type'], 0),
                  (BUILDER, ('Op','concat_in_dim'), ['cls','ops','dimension','builder'], 1))
        for path, names, args, defaults in checks:
            with self.subTest(names=names):
                function = node(path, *names)
                self.assertEqual([x.arg for x in function.args.args], args)
                self.assertEqual(len(function.args.defaults), defaults)
                self.assertFalse(function.args.posonlyargs or function.args.kwonlyargs)
                self.assertIsNone(function.args.vararg)
                self.assertIsNone(function.args.kwarg)

    def test_shape_api_and_literal_dtypes(self):
        shape = node(BUILDER, 'Shape')
        for name in ('sizes','dtype'):
            prop = next(x for x in shape.body if isinstance(x, ast.FunctionDef) and x.name == name)
            self.assertTrue(any(isinstance(d,ast.Name) and d.id=='property' for d in prop.decorator_list))
        type_node = node(BUILDER, 'Type')
        constants = {n.targets[0].id:ast.literal_eval(n.value) for n in type_node.body if isinstance(n,ast.Assign)}
        self.assertEqual((constants['S16'],constants['BF16']), ('s16','bf16'))

    def test_candidate_no_numeric_convert_or_implicit_dtype_view(self):
        tree = ast.parse(MODULE_PATH.read_text())
        banned = {'cast','to','view','float','half','int','short','trace','load','save'}
        for call in (n for n in ast.walk(tree) if isinstance(n,ast.Call)):
            if isinstance(call.func,ast.Attribute): self.assertNotIn(call.func.attr,banned)
        self.assertFalse(any(isinstance(n, (ast.Import,ast.ImportFrom)) for n in tree.body))

    def test_tests_imported_no_native_modules(self):
        self.assertFalse(BLOCKED.intersection(sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)
