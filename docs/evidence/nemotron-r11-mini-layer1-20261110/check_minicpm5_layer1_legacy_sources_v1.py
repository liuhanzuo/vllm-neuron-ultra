"""Regression tests for historical/current source scope; no native/model reads."""
import ast
import copy
from pathlib import Path
import sys
import unittest

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / 'research')]
from research import minicpm5_layer1_legacy_sources_v1 as legacy


class SourcePins(unittest.TestCase):
    def check(self, old=None, current=None, loaded=None, archives=None, actual=None):
        old = {'old.py': 'a'} if old is None else old
        current = {'old.py': 'a', 'new.py': 'b'} if current is None else current
        loaded = ['old.py', 'new.py'] if loaded is None else loaded
        archives = [{'old.py': 'a', 'new.py': 'b'}] if archives is None else archives
        actual = {'old.py': 'a', 'new.py': 'b'} if actual is None else actual
        legacy.validate_source_maps(old, current, loaded, archives, actual.__getitem__)

    def test_newly_loaded_archived_helper_need_not_belong_to_historical_run(self):
        old = {'old.py': 'a'}
        with self.assertRaises(KeyError):
            _ = old['new.py']
        self.check(old=old)
        self.assertEqual(old, {'old.py': 'a'})

    def test_missing_current_helper_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'missing from current'):
            self.check(current={'old.py': 'a'})

    def test_tampered_current_helper_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'differs from current'):
            self.check(actual={'old.py': 'a', 'new.py': 'changed'})

    def test_archive_conflict_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'archive pin'):
            self.check(archives=[{'new.py': 'changed'}])

    def test_one_of_multiple_archives_conflicts(self):
        with self.assertRaisesRegex(RuntimeError, 'archive pin'):
            self.check(archives=[{'new.py': 'b'}, {'new.py': 'changed'}])

    def test_historical_source_cannot_be_removed_even_when_not_loaded(self):
        with self.assertRaisesRegex(RuntimeError, 'historical source'):
            self.check(current={'new.py': 'b'}, loaded=['new.py'])

    def test_historical_source_cannot_be_replaced(self):
        with self.assertRaisesRegex(RuntimeError, 'historical source'):
            self.check(current={'old.py': 'changed', 'new.py': 'b'})

    def test_unrelated_archive_entries_do_not_become_current_imports(self):
        self.check(archives=[{'unloaded.py': 'c'}])

    def test_no_native_modules_loaded(self):
        self.assertFalse({'torch', 'numpy', 'neuronxcc', 'torch_neuronx',
                          'transformers', 'vllm'} & set(sys.modules))


def function(path, name):
    return copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))


class PreservedHistoricalChecks(unittest.TestCase):
    def test_oproj_checks_only_change_current_import_lookup(self):
        old = function(R / 'minicpm5_original32_oproj_cpu_runner_v1.py', 'check_prerequisites')
        new = function(Path(legacy.__file__), 'check_oproj')

        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'oproj':
                    return ast.Name(id=node.attr, ctx=node.ctx)
                return self.generic_visit(node)

            def visit_Subscript(self, node):
                if ast.unparse(node) == 'current_sources[path]':
                    return ast.parse("spec['source_sha256'][path]", mode='eval').body
                return self.generic_visit(node)

        new = Normalize().visit(new)
        new.name = old.name
        new.args.args.pop()
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_layer0_checks_only_add_current_closure_and_route_checked_predecessor(self):
        old = function(R / 'minicpm5_original32_layer0_full_bf16_cpu_runner_v1.py', 'check_prerequisites')
        new = function(Path(legacy.__file__), 'check_layer0')
        self.assertEqual(ast.unparse(new.body.pop(0)), 'validate_current_sources(spec, current_sources)')

        class Normalize(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'layer0':
                    return ast.Name(id=node.attr, ctx=node.ctx)
                return self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'check_oproj':
                    node.func = ast.parse('predecessor.check_prerequisites', mode='eval').body
                    node.args.pop()
                return self.generic_visit(node)

        new = Normalize().visit(new)
        new.name = old.name
        new.args.args.pop()
        self.assertEqual(ast.dump(old), ast.dump(new))

    def test_collector_scientific_and_lifecycle_functions_unchanged(self):
        old_path = R / 'minicpm5_original32_layer1_observation_cpu_runner_v1.py'
        new_path = R / 'minicpm5_original32_layer1_observation_cpu_runner_v2.py'
        for node in ast.parse(old_path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name != 'check_prerequisites':
                with self.subTest(function=node.name):
                    self.assertEqual(ast.dump(node), ast.dump(function(new_path, node.name)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
