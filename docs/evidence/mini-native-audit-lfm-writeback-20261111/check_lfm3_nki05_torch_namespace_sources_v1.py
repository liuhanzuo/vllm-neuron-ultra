"""Pure stdlib namespace/source checks; never import Torch, NKI or NumPy.

Installed source is read/compiled as text. Fake ModuleType namespaces are built
from isolated class ASTs, with no package import or actual native execution.
"""
import ast
import copy
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import types
from typing import Any, Iterator
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'nki', 'ml_dtypes', 'transformers',
                                      'torch_neuronx', 'neuronxcc', 'neuronx_distributed'}:
            raise AssertionError('Native import forbidden in source checks: ' + fullname)


sys.meta_path.insert(0, DenyNative())
import lfm3_nki05_torch_namespace_sources_v1 as helper
from research import check_lfm3_nki05_characterization_delivery_v1 as fixture

OLD = ROOT / 'run_lfm3_nki05_transport_serial_v2.py'
NEW = ROOT / 'run_lfm3_nki05_transport_serial_v3.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def definitions(path, names, scope):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def namespace_fixture():
    modules = {}
    package = types.ModuleType('torch')
    path = helper.PREFIX + '/torch/__init__.py'
    package.__file__ = path
    package.__spec__ = importlib.machinery.ModuleSpec('torch', loader=None, origin=path)
    modules['torch'] = package
    for name, (attr, backend_name, class_name, label) in helper.NAMES.items():
        path = helper.PREFIX + '/torch/' + label
        backend = types.ModuleType(backend_name)
        backend.__file__ = path
        backend.__spec__ = importlib.machinery.ModuleSpec(backend_name, loader=None, origin=path)
        backend.__dict__.update(types=types, Iterator=Iterator, Any=Any,
            _HigherOrderNamespace=lambda: types.ModuleType('synthetic_higher_order'),
            _OpNamespace=types.ModuleType, _ClassNamespace=types.ModuleType)
        definitions(Path(path), {class_name}, backend.__dict__)
        value = backend.__dict__[class_name]()
        setattr(backend, attr, value)
        setattr(package, attr, value)
        modules[backend_name] = backend
        modules[name] = value
    return modules


def verifier(path, modules):
    ns = dict(Path=Path, ROOT=ROOT, SELF=str(path), ORIGINAL_HELPERS={},
        sys=types.SimpleNamespace(modules=modules), namespace_sources=helper,
        diagnostic=types.SimpleNamespace(__file__=str(ROOT/'lfm3_nki05_transport_diagnostic_v2.py')),
        history=types.SimpleNamespace(__file__=str(ROOT/'lfm3_nki05_serial_history_v1.py')),
        p=types.SimpleNamespace(require=helper.require, sha=sha))
    definitions(path, {'verify_sources'}, ns)
    pins = dict(helper.SOURCE_SHA256)
    for p in (path, Path(ns['diagnostic'].__file__), Path(ns['history'].__file__)):
        pins[str(p)] = sha(p)
    return ns['verify_sources'], {'source_sha256': pins}


class NamespaceChecks(unittest.TestCase):
    def test_01_original_verifier_reproduces_cwd_missing_file(self):
        modules = namespace_fixture()
        verify, spec = verifier(OLD, modules)
        before = Path.cwd()
        try:
            os.chdir(ROOT)
            for name, label in (('torch.ops', '_ops.py'), ('torch.classes', '_classes.py')):
                self.assertFalse((ROOT/label).exists())
                isolated = {name: modules[name]}
                verify, spec = verifier(OLD, isolated)
                with self.subTest(name=name), self.assertRaises(FileNotFoundError) as caught:
                    verify(spec)
                self.assertEqual(caught.exception.filename, str(ROOT/label))
        finally:
            os.chdir(before)

    def test_02_exact_supported_namespace_types_and_provenance(self):
        modules = namespace_fixture()
        for name in helper.NAMES:
            self.assertTrue(helper.known_namespace(name, modules[name], modules, helper.SOURCE_SHA256,
                                                  lambda path: Path(path).read_bytes()))
        verify, spec = verifier(NEW, modules)
        verify(spec)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('nki', sys.modules)

    def test_03_changed_singleton_class_and_source_fail_closed(self):
        faults = ('wrong_type', 'parent_identity', 'backend_identity', 'foreign_source',
                  'foreign_origin', 'instance_file', 'changed_label', 'changed_init',
                  'changed_getattr', 'missing_pin', 'changed_source_text')
        for name in helper.NAMES:
            for fault in faults:
                modules = namespace_fixture()
                attr, backend_name, class_name, label = helper.NAMES[name]
                target = modules[name]
                pins = dict(helper.SOURCE_SHA256)
                read = lambda path: Path(path).read_bytes()
                if fault == 'wrong_type':
                    target = types.ModuleType(name)
                elif fault == 'parent_identity':
                    setattr(modules['torch'], attr, object())
                elif fault == 'backend_identity':
                    setattr(modules[backend_name], attr, object())
                elif fault == 'foreign_source':
                    modules[backend_name].__file__ = '/foreign/' + label
                elif fault == 'foreign_origin':
                    modules[backend_name].__spec__.origin = '/foreign/' + label
                elif fault == 'instance_file':
                    target.__file__ = label
                elif fault == 'changed_label':
                    type(target).__file__ = 'other.py'
                elif fault == 'changed_init':
                    type(target).__init__ = lambda self: None
                elif fault == 'changed_getattr':
                    type(target).__getattr__ = lambda self, name: None
                elif fault == 'missing_pin':
                    del pins[helper.PREFIX + '/nki/framework/_torch_convert.py']
                elif fault == 'changed_source_text':
                    read = lambda path: Path(path).read_bytes() + b'\n'
                with self.subTest(name=name, fault=fault), self.assertRaises(RuntimeError):
                    helper.known_namespace(name, target, modules, pins, read)

    def test_04_unknown_module_is_not_a_namespace_exception(self):
        modules = namespace_fixture()
        self.assertFalse(helper.known_namespace('alien.ops', modules['torch.ops'], modules,
                                               helper.SOURCE_SHA256, lambda path: b''))
        fake = types.ModuleType('ordinary')
        fake.__file__ = 'lfm3_nki05_torch_namespace_sources_v1.py'
        verify, spec = verifier(NEW, {'ordinary': fake})
        before = Path.cwd()
        try:
            os.chdir(ROOT)
            with self.assertRaises(RuntimeError): verify(spec)
            spec['source_sha256'][str(ROOT/fake.__file__)] = sha(ROOT/fake.__file__)
            verify(spec)
            spec['source_sha256'][str(ROOT/fake.__file__)] = '0'*64
            with self.assertRaises(RuntimeError): verify(spec)
        finally:
            os.chdir(before)

    def test_05_ordinary_missing_relative_file_remains_failure(self):
        value = types.ModuleType('ordinary')
        value.__file__ = 'lfm3-nonexistent-source-test.py'
        verify, spec = verifier(NEW, {'ordinary': value})
        before = Path.cwd()
        try:
            os.chdir(ROOT)
            self.assertFalse((ROOT/value.__file__).exists())
            with self.assertRaises(FileNotFoundError): verify(spec)
        finally:
            os.chdir(before)

    def test_06_original_science_lifecycle_and_frozen_sources_unchanged(self):
        old = {node.name: node for node in ast.parse(OLD.read_text()).body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        new = {node.name: node for node in ast.parse(NEW.read_text()).body
               if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(set(old), set(new))
        for name in old:
            if name != 'verify_sources':
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        for name, expected in (
                ('run_lfm3_nki05_transport_serial_v2.py', '7029adf0cb320c9857db0de5ca92b1d3dea52ec4003324b45b7d120def3910c7'),
                ('lfm3_nki05_transport_diagnostic_v2.py', 'cf83c634b0f71f5d8a483704b7fed2bdad2588736866d81f28d19416ac11263a'),
                ('lfm3_nki05_writeback_probe_v1.py', 'abdb58d5acc086ac11f177fa98d38b6424d5fa02d9612e5343c534331534e073')):
            self.assertEqual(sha(ROOT/name), expected)

    def test_07_full_characterization_remains_nonzero_and_gates_unchanged(self):
        scope = dict(p=types.SimpleNamespace(require=helper.require), diagnostic=fixture.original)
        definitions(NEW, {'require_diagnostic_success', 'require_characterization'}, scope)
        report = fixture.representative()
        status = scope['require_characterization'](report)
        self.assertTrue(status['nonzero_worker_exit_required'])
        self.assertFalse(status['original_passed'])
        self.assertEqual(status['movement_cases'], 32)
        self.assertEqual(status['bf16_encoding_count'], 65536)
        for fault in ('fewer_cases', 'partial_encoding', 'no_restore', 'fake_original_success'):
            bad = copy.deepcopy(report)
            if fault == 'fewer_cases': bad['candidate']['suite']['checks'].pop()
            elif fault == 'partial_encoding': bad['candidate']['suite']['bf16_encoding_count'] = 65535
            elif fault == 'no_restore': bad['restored_original_reproduced'] = False
            else: bad['original_suite_passed'] = True
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                scope['require_characterization'](bad)

    def test_08_repeated_admission_after_lazy_namespace_creation(self):
        modules = {}
        verify, spec = verifier(NEW, modules)
        report = fixture.representative()
        names = ['original', 'candidate', 'restored_original']
        seen = []
        for name in names:
            verify(spec)
            seen.append(name)
            if name == 'original': modules.update(namespace_fixture())
        # Simulate the final post-diagnostic admit() too. No simulator work or
        # synthetic data is promoted; this checks the original failure trigger.
        verify(spec)
        self.assertEqual(seen, names)
        self.assertFalse(report['original_suite_passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
