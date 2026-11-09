"""Retry builder boundary tests; never supplies a real failed review receipt."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import prepare_dory_saved_review_retry_v1 as b


class Builder(unittest.TestCase):
    def test_failed_history_must_validate_before_source_graph_or_candidate(self):
        with patch.object(b.contract,'failed_history',side_effect=RuntimeError('not terminal')), \
             patch.object(b,'local_imports') as graph:
            with self.assertRaisesRegex(RuntimeError,'not terminal'):
                b.assemble(dict(path='/synthetic/not-terminal.json',sha256='0'*64))
            graph.assert_not_called()

    def test_fixed_entries_include_executor_protocol_and_all_new_tests(self):
        names={p.name for p in b.ENTRIES}
        self.assertEqual(len(names),10)
        self.assertTrue({'run_dory_saved_review_retry_v1.py','dory_saved_review_retry_contract_v1.py',
            'check_dory_saved_review_retry_contract_v1.py','check_dory_saved_review_retry_executor_v1.py',
            'check_dory_saved_review_retry_imports_v1.py','check_dory_saved_review_retry_builder_v1.py'}<=names)
        for path in b.ENTRIES:
            self.assertTrue(path.is_file())
            ast.parse(path.read_text(),filename=str(path))

    def test_source_pin_rebinding_remains_forbidden(self):
        pins={'/synthetic/existing.py':'1'*64}
        with self.assertRaisesRegex(RuntimeError,'rebind'):
            b.merge(pins,{'/synthetic/existing.py':'2'*64})

    def test_source_graph_contains_new_query_and_original_full_raw_review(self):
        graph=b.local_imports(entries=b.ENTRIES)
        names={Path(p).name for p in graph['source_sha256']}
        self.assertTrue({'dory_residual_saved_complete_v2.py','dory_residual_saved_payloads_v2.py',
            'dory_residual_saved_archives_v2.py','dory_residual_saved_observations_v1.py',
            'dory_service_query_auth_v1.py','dory_saved_review_service_query_v1.py',
            'dory_saved_review_guard_v1.py','run_dory_saved_review_retry_v1.py'}<=names)
        self.assertFalse(graph['native_executed'])
        self.assertFalse(graph['runtime_import_identity_verified'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
