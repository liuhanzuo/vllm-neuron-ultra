"""Stdlib-only adversarial closure checks; no native imports or model data."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import minicpm5_native_python_sources_v1 as sources


class ClosureChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='mini-source-closure-test-')
        self.root = Path(self.temporary.name)
        self.package = self.root / 'transformers'
        self.package.mkdir()
        (self.package / '__init__.py').write_text('from . import dependency_versions_check\n')
        (self.package / 'dependency_versions_check.py').write_text('VERSION = 1\n')
        self.calls = []
        self.guard = lambda: self.calls.append('guard')
        self.read = lambda path, expected: Path(path).read_bytes()
        self.pins = sources.collect_package_sources(self.read, self.guard, package_root=self.package)

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_selected_package_includes_initialization_dependency(self):
        self.assertEqual(len(self.pins), 2)
        value = sources.require_package_coverage(dict(source_sha256=self.pins), self.guard, package_root=self.package)
        self.assertTrue(value['complete_filename_coverage'])
        self.assertFalse(value['content_authentication_performed'])
        self.assertFalse(value['native_import_performed'])

    def test_missing_initialization_dependency_is_identified(self):
        path = str(self.package / 'dependency_versions_check.py')
        pins = {key: value for key, value in self.pins.items() if key != path}
        with self.assertRaises(sources.SourceClosureError) as caught:
            sources.require_package_coverage(dict(source_sha256=pins), self.guard, package_root=self.package)
        report = caught.exception.source_closure_error
        self.assertEqual(report['reason'], 'package_python_sources_unpinned')
        self.assertEqual(report['first_paths'], [path])

    def test_package_tree_growth_and_shrinkage_fail(self):
        extra = self.package / 'late_dependency.py'
        extra.write_text('LATE = True\n')
        with self.assertRaises(sources.SourceClosureError):
            sources.require_package_coverage(dict(source_sha256=self.pins), self.guard, package_root=self.package)
        extra.unlink()
        (self.package / 'dependency_versions_check.py').unlink()
        with self.assertRaises(sources.SourceClosureError) as caught:
            sources.require_package_coverage(dict(source_sha256=self.pins), self.guard, package_root=self.package)
        self.assertEqual(caught.exception.source_closure_error['reason'], 'pinned_package_python_sources_absent')

    def test_package_source_symlink_fails(self):
        (self.package / 'alias.py').symlink_to(self.package / '__init__.py')
        with self.assertRaises(sources.SourceClosureError):
            sources.python_paths(self.package)

    def test_loaded_missing_source_diagnostic_contains_actual_module_and_path(self):
        path = str(self.package / 'dependency_versions_check.py')
        modules = {'transformers.dependency_versions_check': SimpleNamespace(__file__=path)}
        with self.assertRaises(sources.SourceClosureError) as caught:
            sources.verify_loaded(dict(source_sha256={}), self.guard, self.read, modules=modules, root=self.root)
        report = caught.exception.source_closure_error
        self.assertEqual(report['module'], 'transformers.dependency_versions_check')
        self.assertEqual(report['path'], path)
        self.assertEqual(report['reason'], 'loaded_local_source_unpinned')

    def test_loaded_source_changed_after_preparation_fails(self):
        path = self.package / 'dependency_versions_check.py'
        path.write_text('VERSION = 2\n')
        modules = {'transformers.dependency_versions_check': SimpleNamespace(__file__=str(path))}
        with self.assertRaises(sources.SourceClosureError) as caught:
            sources.verify_loaded(dict(source_sha256=self.pins), self.guard, self.read, modules=modules, root=self.root)
        self.assertEqual(caught.exception.source_closure_error['reason'], 'loaded_local_source_digest_changed')

    def test_loaded_source_hashes_all_in_scope_and_keeps_guard(self):
        modules = {Path(path).stem: SimpleNamespace(__file__=path) for path in self.pins}
        modules['outside'] = SimpleNamespace(__file__='/not-opened/outside.py')
        reads = []
        def read(path, expected):
            reads.append(path)
            raw = Path(path).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
            return raw
        self.calls.clear()
        proof = sources.verify_loaded(dict(source_sha256=self.pins), self.guard, read, modules=modules, root=self.root)
        self.assertEqual(proof['loaded_local_modules_verified'], 2)
        self.assertEqual(set(reads), set(self.pins))
        self.assertEqual(len(self.calls), 2)

    def test_successor_sources_compile_without_executing_imports(self):
        names = ['minicpm5_native_python_sources_v1.py', 'minicpm5_layer1_operator_native_process_adapter_v8.py',
                 'minicpm5_layer1_operator_native_driver_v5.py', 'prepare_minicpm5_layer1_operator_native_v5.py',
                 '../minicpm5_layer1_operator_native_runner_v11.py']
        for name in names:
            path = Path(__file__).parent / name
            compile(ast.parse(path.read_text(), filename=str(path)), str(path), 'exec')
        self.assertFalse(any(name == 'torch' or name == 'transformers' or name.startswith(('torch.', 'transformers.'))
                             for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
