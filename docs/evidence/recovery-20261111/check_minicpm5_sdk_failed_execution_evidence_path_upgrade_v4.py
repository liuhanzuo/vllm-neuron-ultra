"""Source-only preservation checks; never import actual model/runtime modules."""
import ast
import hashlib
from pathlib import Path
import unittest

Q=Path(__file__).resolve().parent
REPLACEMENTS=(
    ('minicpm5_layer1_sdk_batch_layout_cpu_runner_v3.py','minicpm5_layer1_sdk_batch_layout_cpu_runner_v4.py'),
    ('minicpm5_layer1_sdk_batch_layout_cpu_worker_v3.py','minicpm5_layer1_sdk_batch_layout_cpu_worker_v4.py'),
    ('audit_minicpm5_layer1_sdk_batch_layout_result_v3.py','audit_minicpm5_layer1_sdk_batch_layout_result_v4.py'),
    ('pinned sdk cpu runner_v3 module','pinned sdk cpu runner_v4 module'))


class Preservation(unittest.TestCase):
    def test_only_three_paths_and_runner_doc_version_changed(self):
        old=(Q/'minicpm5_sdk_failed_execution_evidence_v3.py').read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),
            'c91b747bc61aba68cfb54f2ccbd74bc874d952a1b3937bd01212456057f5ffc0')
        expected=old.decode()
        for source,target in REPLACEMENTS:
            self.assertEqual(expected.count(source),1)
            expected=expected.replace(source,target)
        actual=(Q/'minicpm5_sdk_failed_execution_evidence_v4.py').read_text()
        self.assertEqual(actual,expected)
        ast.parse(actual)

    def test_all_39_tests_have_only_import_target_changed(self):
        old=(Q/'check_minicpm5_sdk_failed_execution_evidence_v3.py').read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),
            '0e22905cc7beebec61051f3576362cf4306cc5276276def5c4d2821ad9be6583')
        source='from research import minicpm5_sdk_failed_execution_evidence_v3 as m'
        target='from research import minicpm5_sdk_failed_execution_evidence_v4 as m'
        self.assertEqual(old.decode().count(source),1)
        actual=(Q/'check_minicpm5_sdk_failed_execution_evidence_v4.py').read_text()
        self.assertEqual(actual,old.decode().replace(source,target))
        self.assertEqual(sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_')
            for n in ast.walk(ast.parse(actual))),39)

    def test_exact_successor_entrypoints_have_expected_ast_paths(self):
        required=[Q.parent/REPLACEMENTS[0][1],Q.parent/REPLACEMENTS[1][1],
            Q/REPLACEMENTS[2][1],Q/'minicpm5_layer1_sdk_batch_layout_post_exit_core_v3.py']
        trees={path:ast.parse(path.read_text()) for path in required}
        runner,worker,auditor,core=required
        def imported(tree,name):
            return any((isinstance(n,ast.Import) and any(x.name==name for x in n.names))
                or (isinstance(n,ast.ImportFrom) and any(x.name==name for x in n.names))
                for n in ast.walk(tree))
        def strings(tree):
            return {n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertTrue(imported(trees[runner],worker.stem))
        self.assertIn(worker.name,strings(trees[runner]))
        self.assertTrue(imported(trees[auditor],runner.stem))
        self.assertTrue(imported(trees[auditor],core.stem))
        for path in (runner,worker,core):
            self.assertTrue(imported(trees[path],'minicpm5_layer1_sdk_batch_layout_completion_v3'))
        self.assertIn(runner.name,strings(trees[core]))
        self.assertIn(worker.name,strings(trees[core]))


if __name__=='__main__':unittest.main(verbosity=2)
