"""Static dependency and receipt-admission regressions; no model execution."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import prepare_dory_residual_spec_v1 as p


class LocalSources(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def put(self, name, content):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def graph(self, entry='main.py'):
        return p.local_imports([self.root/entry], [self.root])

    def test_never_execute_parsed_module(self):
        self.put('main.py', "raise RuntimeError('must not execute')\nimport sibling\n")
        self.put('sibling.py', "raise RuntimeError('must not execute either')\n")
        self.assertEqual(self.graph()['files'], 2)

    def test_circular_local_imports(self):
        self.put('main.py', 'import sibling\n')
        self.put('sibling.py', 'import main\n')
        self.assertEqual(self.graph()['files'], 2)

    def test_namespace_package_import(self):
        self.put('main.py', 'from package import worker\n')
        self.put('package/worker.py', 'value = 1\n')
        graph = self.graph()
        self.assertEqual(graph['files'], 2)
        self.assertNotIn('package', graph['external_import_roots'])

    def test_relative_package_imports(self):
        self.put('package/main.py', 'from . import worker\nfrom .inner import nested\n')
        self.put('package/worker.py', 'value = 1\n')
        self.put('package/inner/__init__.py', '')
        self.put('package/inner/nested.py', 'value = 2\n')
        self.assertEqual(self.graph('package/main.py')['files'], 4)

    def test_function_attribute_not_misclassified_as_external(self):
        self.put('main.py', 'from sibling import helper\n')
        self.put('sibling.py', 'def helper(): pass\n')
        self.assertEqual(self.graph()['external_import_roots'], [])

    def test_external_and_dynamic_limits_are_explicit(self):
        self.put('main.py', "import importlib\nimport torch\nimportlib.import_module('dynamic')\n")
        graph = self.graph()
        self.assertEqual(graph['external_import_roots'], ['importlib', 'torch'])
        self.assertFalse(graph['dynamic_import_resolution_verified'])
        self.assertFalse(graph['runtime_import_identity_verified'])

    def test_no_binary_entry_read(self):
        self.put('weights.bin', 'binary fixture')
        with self.assertRaises(RuntimeError):
            p.python_bytes(self.root/'weights.bin')

    def test_escape_symlink_rejected(self):
        self.put('main.py', 'import outside\n')
        with tempfile.TemporaryDirectory() as elsewhere:
            target = Path(elsewhere)/'outside.py'
            target.write_text('value = 1\n')
            (self.root/'outside.py').symlink_to(target)
            with self.assertRaises(RuntimeError):
                self.graph()

    def test_original_hash_never_rebound(self):
        pins = {'/original.py': 'a'*64}
        with self.assertRaises(RuntimeError):
            p.merge(pins, {'/original.py': 'b'*64})
        self.assertEqual(pins, {'/original.py': 'a'*64})

    def test_exact_pins_and_source_bytes(self):
        path = self.put('main.py', 'value = 1\n')
        graph = self.graph()
        self.assertEqual(graph['source_sha256'][str(path)], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(graph['bytes_read'], path.stat().st_size)


class FinalizationPrerequisites(unittest.TestCase):
    def test_no_metadata_without_actual_r9_exit(self):
        with patch.object(p.contract, 'load') as read:
            with self.assertRaises(RuntimeError):
                p.assemble('metadata')
            read.assert_not_called()

    def test_no_native_without_actual_metadata_exit(self):
        with patch.object(p.contract, 'load') as read:
            with self.assertRaises(RuntimeError):
                p.assemble('native')
            read.assert_not_called()

    def test_no_review_without_actual_native_exit(self):
        with patch.object(p.contract, 'load') as read:
            with self.assertRaises(RuntimeError):
                p.assemble('review', metadata_exit={}, producer_spec={})
            read.assert_not_called()

    def test_missing_r9_exit_file_cannot_reconstruct_or_finalize(self):
        with patch.object(p.history, 'reconstruct') as reconstruct, \
             patch.object(p, 'local_imports') as parse:
            with self.assertRaises(FileNotFoundError):
                p.assemble('metadata', r9_actual_exit={'path': '/does-not-exist-dory-actual-exit', 'sha256': 'a'*64})
            reconstruct.assert_not_called()
            parse.assert_not_called()


class RecoveryHistory(unittest.TestCase):
    def ledger(self):
        names = ('inspect_nemotron_r9_failed_v1.py', 'inspect_nemotron_r9_recovery_sources_v1.py',
                 'check_nemotron_r9_failed_release_v1.py', 'finalize_nemotron_r9_failed_release_review_v1.py')
        records, values = [], {}
        for i, name in enumerate(names):
            owner = dict(pid=100+i, start_ticks=str(200+i), argv=['/usr/bin/python3','-B',str(Q/name)])
            ref = dict(path='/synthetic/'+name, sha256='a'*64)
            values[ref['path']] = dict(owner=owner, actual_self_exit_claimed=False)
            records.append(dict(actual_external_exit_observed=True, synchronous_tool_exit=True,
                session_id=None, exit_code=0, tool_chunk='abc', owner=owner, intent=ref))
        return dict(schema='nemotron_r9_recovery_operations_actual_exits_v1',reviewer='/root',operations=records), values

    def test_all_real_operation_identities_retained(self):
        ledger, values = self.ledger()
        owners = p.history.operation_owners(ledger, load=lambda ref: values[ref['path']])
        self.assertEqual([x['pid'] for x in owners], [100,101,102,103])

    def test_missing_recovery_operation_rejected(self):
        ledger, values = self.ledger()
        ledger['operations'].pop()
        with self.assertRaises(RuntimeError):
            p.history.operation_owners(ledger, load=lambda ref: values[ref['path']])

    def test_invented_async_session_rejected(self):
        ledger, values = self.ledger()
        ledger['operations'][0]['session_id'] = 123
        with self.assertRaises(RuntimeError):
            p.history.operation_owners(ledger, load=lambda ref: values[ref['path']])

    def test_wrong_recovery_owner_rejected(self):
        ledger, values = self.ledger()
        values[ledger['operations'][0]['intent']['path']]['owner'] = {'pid': 999}
        with self.assertRaises(RuntimeError):
            p.history.operation_owners(ledger, load=lambda ref: values[ref['path']])

    def test_release_wrapper_must_be_exact_failed_only_entry(self):
        with self.assertRaises(RuntimeError):
            p.history.release_extension(dict(owner=dict(argv=['python','-B','/unrelated','--audit'])),
                {}, load=lambda ref: self.fail('must reject before metadata read'), read=lambda path: None)


if __name__ == '__main__':
    unittest.main(verbosity=2)
