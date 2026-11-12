"""Pure regression using isolated pinned class ASTs; never imports Torch."""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(Q), str(ROOT)]


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'nki', 'transformers', 'torch_neuronx', 'vllm'}:
            raise AssertionError('No native imports in source-only checks')


sys.meta_path.insert(0, DenyNative())
import qwen35_sampler_runtime_io_v2 as old
import qwen35_sampler_runtime_io_v8 as new
import qwen35_sampler_torch_namespace_sources_v1 as helper


def namespace_fixture():
    modules = {}; parent = types.ModuleType('torch')
    parent.__file__ = helper.PREFIX + '/torch/__init__.py'
    parent.__spec__ = importlib.machinery.ModuleSpec('torch', None, origin=parent.__file__)
    modules['torch'] = parent
    for name, (attr, backend_name, class_name, label) in helper.NAMES.items():
        path = helper.PREFIX + '/torch/' + label
        backend = types.ModuleType(backend_name); backend.__file__ = path
        backend.__spec__ = importlib.machinery.ModuleSpec(backend_name, None, origin=path)
        backend.__dict__.update(types=types, Iterator=Iterator, Any=Any,
            _HigherOrderNamespace=lambda: types.ModuleType('synthetic_higher_order'),
            _OpNamespace=types.ModuleType, _ClassNamespace=types.ModuleType)
        node = next(n for n in ast.parse(Path(path).read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == class_name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), backend.__dict__)
        value = backend.__dict__[class_name]()
        setattr(backend, attr, value); setattr(parent, attr, value)
        modules[backend_name] = backend; modules[name] = value
    return modules


def check(io, modules, pins):
    # Replace only this module's sys reference, never actual sys.modules.
    with patch.object(io, 'sys', types.SimpleNamespace(modules=modules)):
        return io.loaded_local_sources(pins)


class VirtualNamespaces(unittest.TestCase):
    def setUp(self):
        self.cwd = Path.cwd(); os.chdir(ROOT)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_original_failure_and_actual_new_verifier(self):
        modules = namespace_fixture()
        with self.assertRaisesRegex(ValueError, 'unreviewed loaded local source.*_ops.py'):
            check(old, modules, helper.SOURCE_SHA256)
        before = {name: dict(vars(module)) for name, module in modules.items()}
        self.assertEqual(check(new, modules, helper.SOURCE_SHA256), {})
        self.assertEqual(before, {name: dict(vars(module)) for name, module in modules.items()})
        self.assertNotIn('torch', sys.modules)

    def test_spoofed_or_changed_namespace_is_rejected(self):
        faults = ('wrong_type', 'parent_identity', 'backend_identity', 'foreign_source',
                  'foreign_origin', 'instance_file', 'changed_label', 'changed_init', 'changed_getattr')
        for name in helper.NAMES:
            for fault in faults:
                modules = namespace_fixture(); attr, backend_name, class_name, label = helper.NAMES[name]
                target = modules[name]; cls = type(target)
                if fault == 'wrong_type': modules[name] = types.ModuleType(name)
                elif fault == 'parent_identity': setattr(modules['torch'], attr, object())
                elif fault == 'backend_identity': setattr(modules[backend_name], attr, object())
                elif fault == 'foreign_source': modules[backend_name].__file__ = '/foreign/' + label
                elif fault == 'foreign_origin': modules[backend_name].__spec__.origin = '/foreign/' + label
                elif fault == 'instance_file': target.__file__ = label
                elif fault == 'changed_label': cls.__file__ = 'changed.py'
                elif fault == 'changed_init': cls.__init__ = lambda self: None
                else: cls.__getattr__ = lambda self, key: None
                with self.subTest(name=name, fault=fault), self.assertRaises(RuntimeError):
                    check(new, modules, helper.SOURCE_SHA256)

    def test_missing_pin_or_changed_source_is_rejected(self):
        for path in helper.SOURCE_SHA256:
            pins = dict(helper.SOURCE_SHA256); pins.pop(path)
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                check(new, namespace_fixture(), pins)
        with patch.object(new, 'raw', return_value=b'changed'), self.assertRaises(RuntimeError):
            check(new, namespace_fixture(), helper.SOURCE_SHA256)

    def test_unknown_relative_source_keeps_original_hash_gate(self):
        module = types.ModuleType('ordinary'); module.__file__ = 'research/test_qwen35_sampler_virtual_namespace_v3.py'
        modules = {'ordinary': module}; path = str(Path(module.__file__).resolve())
        with self.assertRaises(ValueError): check(new, modules, {})
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.assertEqual(check(new, modules, {path: sha}), {path: sha})
        with self.assertRaises(ValueError): check(new, modules, {path: '0'*64})

    def test_unknown_missing_or_class_relative_source_is_not_exempt(self):
        class Unknown(types.ModuleType):
            __file__ = '_ops.py'
        module = Unknown('ordinary')
        with self.assertRaises(ValueError): check(new, {'ordinary': module}, {})
        path = str(ROOT/'_ops.py'); self.assertFalse(Path(path).exists())
        with self.assertRaises(FileNotFoundError): check(new, {'ordinary': module}, {path: '0'*64})

    def test_ordinary_module_dynamic_getattr_is_never_called(self):
        module = types.ModuleType('ordinary'); module.__getattr__ = lambda name: (_ for _ in ()).throw(AssertionError(name))
        self.assertEqual(check(new, {'ordinary': module}, {}), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
