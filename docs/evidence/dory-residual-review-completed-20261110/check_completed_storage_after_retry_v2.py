"""No maintenance; exact target/budget tests and unchanged retirement checks."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import completed_storage_contract_after_retry_v2 as c
import completed_storage_contract_v1 as original
import completed_storage_privileged_after_retry_v2 as privileged
import completed_storage_files_v1 as files


def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text()).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}


class Contract(unittest.TestCase):
    def setUp(self):
        self.candidate,self.weights,self.journals=c.fixed_targets()
        self.plan=dict(schema=c.SCHEMA,output=str(c.OUT),executor=str(c.EXECUTOR),
            privileged_helper=str(c.PRIVILEGED),candidate=c.CANDIDATE,audit=c.AUDIT,
            protected_model_spec=c.PARENT,budget=copy.deepcopy(c.BUDGET),
            weights=copy.deepcopy(self.weights),journals=copy.deepcopy(self.journals),
            scientific_constraints_changed=False,runtime_admitted=False,automatic_retry=False,
            original_qwen_bf16_restore_required=True,
            source_sha256={name:'0'*64 for name in [*c.p.required_sources(),*[str(c.Q/n)for n in c.NAMES]]})

    def test_same_exact_completed_weights_closed_journals_and_budget(self):
        c.validation(self.plan)
        self.assertEqual(c.fixed_targets(),original.fixed_targets())
        self.assertEqual(c.BUDGET,original.BUDGET)
        self.assertEqual(c.OUT,original.OUT)
        self.assertEqual(sum(row['stat']['st_size'] for row in self.weights),16381516776)
        self.assertEqual(sum(row['stat']['st_size'] for row in self.journals),1073741824)

    def test_failed_retry_completion_precedes_any_target_access(self):
        with (patch.object(c.residual,'completed',side_effect=RuntimeError('retry incomplete')),
              patch.object(c,'fixed_targets')as targets,patch.object(c,'once')as save):
            with self.assertRaisesRegex(RuntimeError,'retry incomplete'):c.build({'synthetic':'no actual exit'})
            targets.assert_not_called();save.assert_not_called()

    def test_original348_owner_history_is_not_sufficient_for_post_retry_storage(self):
        incomplete=dict(history=dict(rows=[{}]*1149,processes=[{}]*348,known_admitted_directories=['x']*37))
        with patch.object(c.residual,'completed',return_value=incomplete),patch.object(c,'fixed_targets')as targets:
            with self.assertRaisesRegex(RuntimeError,'completed predecessor'):c.build({})
            targets.assert_not_called()

    def test_target_cannot_expand_to_current_model_or_int8(self):
        self.plan['weights'][0]['path']='/mnt/nvme/models/Qwen3-8B-int8/model.safetensors'
        with self.assertRaises(RuntimeError):c.validation(self.plan)

    def test_active_journal_is_still_forbidden(self):
        self.plan['journals'][0]['path']=str(c.JOURNAL_ROOT/'user-1000.journal')
        with self.assertRaises(RuntimeError):c.validation(self.plan)

    def test_original_budget_cannot_be_reduced(self):
        self.plan['budget']['gzip_cap']*=2
        with self.assertRaises(RuntimeError):c.validation(self.plan)

    def test_original_executor_or_privileged_helper_cannot_be_substituted(self):
        for field,value in [('executor',str(original.EXECUTOR)),('privileged_helper',str(original.PRIVILEGED))]:
            with self.subTest(field=field):
                self.setUp();self.plan[field]=value
                with self.assertRaises(RuntimeError):c.validation(self.plan)

    def test_missing_new_source_is_rejected(self):
        self.plan['source_sha256'].pop(str(c.Q/'dory_saved_retry_source_partition_v1.py'))
        with self.assertRaises(RuntimeError):c.validation(self.plan)

    def test_fixed_targets_evidence_and_protected_trees_have_identical_ast(self):
        old,new=functions(original.__file__),functions(c.__file__)
        for name in ['fixed_targets','evidence','fixed_evidence','protected_trees']:
            self.assertEqual(old[name],new[name],name)

    def test_all_hashing_and_archive_verification_before_retirement_is_unchanged(self):
        old=functions(c.Q/'run_completed_storage_serial_v1.py')
        new=functions(c.Q/'run_completed_storage_after_retry_v2.py')
        for name in ['available','prepare_all','privileged','free']:
            self.assertEqual(old[name],new[name],name)

    def test_privileged_scan_unused_and_journal_retirement_are_unchanged(self):
        self.assertEqual(functions(c.Q/'completed_storage_privileged_v1.py'),functions(privileged.__file__))

    def test_unadmitted_journal_request_still_prevents_unlink(self):
        original_load=c.load
        def load(ref,**kw):
            if ref['path']=='/synthetic/plan':return self.plan
            if ref['path']==str(c.OUT/'admission.json'):return dict(plan={},fully_drained=False)
            return original_load(ref,**kw)
        with patch.object(c,'load',side_effect=load),patch.object(files,'unlink_exact')as unlink:
            with self.assertRaises(RuntimeError):
                privileged.retire(dict(plan=dict(path='/synthetic/plan',sha256='0'*64),
                    admission=dict(path=str(c.OUT/'admission.json'),sha256='0'*64)))
            unlink.assert_not_called()

    def test_complete_worker_functions_identical_to_prior_retry_worker(self):
        self.assertEqual(functions(c.Q/'run_completed_storage_after_retry_v1.py'),
                         functions(c.Q/'run_completed_storage_after_retry_v2.py'))

    def test_maintenance_contract_functions_preserve_existing_targets_and_checks(self):
        self.assertEqual(functions(c.Q/'completed_storage_contract_after_retry_v1.py'),functions(c.__file__))

    def test_completion_dependency_is_the_reviewed_delta_semantics_successor(self):
        self.assertEqual(Path(c.residual.__file__).name,'nemotron9b_after_dory_saved_retry_v2.py')
        for name in('nemotron9b_after_dory_saved_retry_v1.py','nemotron9b_after_dory_saved_retry_v2.py',
                    'check_nemotron9b_after_dory_saved_retry_v2.py'):
            self.assertIn(name,c.NAMES)


if __name__=='__main__':
    unittest.main(verbosity=2)
