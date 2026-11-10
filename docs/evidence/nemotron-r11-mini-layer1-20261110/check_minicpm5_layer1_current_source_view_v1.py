import ast
import copy
from pathlib import Path
import sys
import unittest
Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q.parent), str(Q)]
from minicpm5_layer1_current_source_view_v1 import current_source_view


class CurrentContextTests(unittest.TestCase):
    def test_new_audit_import_preserves_original_spec_and_all_scientific_fields(self):
        original = {'source_sha256': {'old.py': 'a'}, 'plan': {'sha256': 'b'},
                    'resources': {'memory': 16}, 'passed': False}
        saved = copy.deepcopy(original)
        current = {'old.py': 'a', 'new-audit.py': 'c'}
        result = current_source_view(original, current)
        self.assertEqual(original, saved)
        self.assertEqual({k: v for k, v in result.items() if k != 'source_sha256'},
                         {k: v for k, v in original.items() if k != 'source_sha256'})
        self.assertEqual(result['source_sha256'], current)
        result['plan']['sha256'] = 'changed'
        result['source_sha256']['old.py'] = 'changed'
        self.assertEqual(original, saved)
        self.assertEqual(current['old.py'], 'a')

    def test_missing_historical_source_rejected(self):
        with self.assertRaises(RuntimeError):
            current_source_view({'source_sha256': {'old.py': 'a'}}, {'new.py': 'b'})

    def test_changed_historical_source_rejected(self):
        with self.assertRaises(RuntimeError):
            current_source_view({'source_sha256': {'old.py': 'a'}}, {'old.py': 'wrong'})

    def test_saved_audit_keeps_complete_validation_and_numeric_flow(self):
        old = ast.parse((Q / 'minicpm5_layer1_saved_audit_runner_v1.py').read_text())
        new = ast.parse((Q / 'minicpm5_layer1_saved_audit_runner_v2.py').read_text())

        class Normalize(ast.NodeTransformer):
            def visit_Call(self, node):
                if ast.unparse(node.func) == 'collector.check_prerequisites':
                    self_view = node.args[0]
                    if ast.unparse(self_view.func) != 'current_source_view':
                        raise AssertionError('Current caller context missing')
                    node.args[0] = self_view.args[0]
                return self.generic_visit(node)

        originals = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        for n in new.body:
            if isinstance(n, ast.FunctionDef):
                with self.subTest(function=n.name):
                    self.assertEqual(ast.dump(originals[n.name]),
                                     ast.dump(Normalize().visit(copy.deepcopy(n))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
