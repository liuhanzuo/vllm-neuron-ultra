"""Source equivalence and real admitted-predecessor scope; never terminal proof."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import minicpm5_after_nemotron_package_v2 as history
import minicpm5_saved_tap_runtime_contract_v2 as contract
import prepare_minicpm5_saved_tap_runtime_v2 as builder
import nemotron9b_package_protocol_v24 as p
from research import nemotron9b_after_completed_storage_retry_v2 as storage

Q=history.Q
BASES=('minicpm5_after_nemotron_package','check_minicpm5_after_nemotron_package',
 'minicpm5_saved_tap_runtime_contract','run_minicpm5_saved_tap_serial','prepare_minicpm5_saved_tap_runtime',
 'check_minicpm5_saved_tap_runtime_contract','check_run_minicpm5_saved_tap_serial',
 'check_prepare_minicpm5_saved_tap_runtime','check_minicpm5_saved_tap_runtime_metadata')
SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v24-r10-20261110-spec.json'),
 sha256='dfb80b098d9f25d8925b1855701b5824e7a8466e75f02b792cb0d7535d2a6a74')
REVIEW=dict(path=str(Q/'nemotron9b-package-runtime-v24-r10-20261110-source-review.json'),
 sha256='ce51d85c45bc39917826ee4e628d4bdec076293e71f0137aec4fdc23342575ac')

def normalize(s):
    for name in BASES:s=s.replace(name+'_v1',name+'_v2')
    return s.replace('minicpm5_saved_tap_runtime_source_review_v1','minicpm5_saved_tap_runtime_source_review_v2').replace(
      'minicpm5_after_nemotron_package_history_v1','minicpm5_after_nemotron_package_history_v2').replace(
      'minicpm5_saved_tap_runtime_preparation_v1','minicpm5_saved_tap_runtime_preparation_v2')

def functions(s,exclude=()):
    return {x.name:ast.dump(x)for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name not in exclude}

class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec=p.metadata.load(SPEC);cls.review=p.metadata.load(REVIEW)
        cls.prior=storage.completed(cls.spec['required_completed_storage'])['history']

    def test_actual_r10_predecessor_scope_supported_without_claiming_completion(self):
        history.validate_scope('retry-v24',self.spec,self.review,SPEC,self.prior,p)
        self.assertEqual((len(self.prior['rows']),len(self.prior['processes']),len(self.prior['known_admitted_directories'])),(1149,350,38))
        self.assertFalse(self.prior['current_physical_release_verified'])
        self.assertEqual(history.PROFILES['retry-v24'],(24,'nemotron9b_completed_storage_completion_after_retry_v2',350,38))

    def test_old_profile_cannot_accept_actual_new_execution(self):
        for profile in ('direct-v22','retry-v23'):
            with self.subTest(profile=profile),self.assertRaises(RuntimeError):
                history.validate_scope(profile,self.spec,self.review,SPEC,self.prior,p)

    def test_live_admission_is_not_a_completed_exit(self):
        # Actual admission JSON lacks terminal/actual-exit/release evidence.
        admission=Path(self.spec['control_root'])/'admission.json'
        ref=dict(path=str(admission),sha256=hashlib.sha256(admission.read_bytes()).hexdigest())
        with self.assertRaises((KeyError,RuntimeError,FileNotFoundError)):
            history.completed('retry-v24',ref)

    def test_original_executor_functions_only_update_dependencies(self):
        old=(Q/'run_minicpm5_saved_tap_serial_v1.py').read_text()
        old=normalize(old).replace('nemotron9b_package_protocol_v23','nemotron9b_package_protocol_v24')
        self.assertEqual(functions(old),functions((Q/'run_minicpm5_saved_tap_serial_v2.py').read_text()))

    def test_original_contract_functions_only_update_namespace(self):
        old=normalize((Q/'minicpm5_saved_tap_runtime_contract_v1.py').read_text())
        self.assertEqual(functions(old),functions((Q/'minicpm5_saved_tap_runtime_contract_v2.py').read_text()))
        import minicpm5_saved_tap_runtime_contract_v1 as prior
        self.assertEqual(contract.BUDGET,prior.BUDGET)
        for key in ('MANIFEST','IO_REVIEW','QUERY_OBSERVATION','OUTPUT_NAMES'):
            self.assertEqual(getattr(contract,key),getattr(prior,key))

    def test_original_terminal_extension_and_builder_science_retained(self):
        old=normalize((Q/'minicpm5_after_nemotron_package_v1.py').read_text())
        self.assertEqual(functions(old,('validate_scope',)),functions((Q/'minicpm5_after_nemotron_package_v2.py').read_text(),('validate_scope',)))
        old=normalize((Q/'prepare_minicpm5_saved_tap_runtime_v1.py').read_text())
        self.assertEqual(functions(old),functions((Q/'prepare_minicpm5_saved_tap_runtime_v2.py').read_text()))

    def test_dynamic_new_profile_and_all_original_entries_in_source_graph(self):
        import prepare_minicpm5_saved_tap_runtime_v1 as prior
        self.assertTrue(set(prior.ENTRIES)<=set(builder.ENTRIES))
        self.assertIn(history.ROOT/'nemotron9b_package_protocol_v24.py',builder.ENTRIES)
        self.assertIn(history.ROOT/'nemotron9b_completed_storage_completion_after_retry_v2.py',builder.ENTRIES)
        self.assertTrue(all((Q/(name+'_v2.py'))in builder.ENTRIES for name in BASES))
        self.assertIn(Path(__file__).resolve(),builder.ENTRIES)

    def test_missing_real_terminal_prevents_any_assembly(self):
        absent=dict(path='/tmp/no-actual-nemotron-completion-for-mini.json',sha256='0'*64)
        with patch.object(builder,'local_imports')as graph:
            with self.assertRaises(FileNotFoundError):builder.assemble('retry-v24',absent)
        graph.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
