"""Candidate builder uses mocked completion, actual source/metadata only.

No real completion is claimed, no specification is written, no native/tensor
payload is opened. Actual fixed input manifest is validated as metadata only.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_minicpm5_saved_tap_runtime_v2 as b


class Builder(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='mini-builder-fixture-');self.addCleanup(self.tmp.cleanup)
        self.receipt=Path(self.tmp.name)/'synthetic-exit.json';self.receipt.write_text('{}\n')
        self.ref=dict(path=str(self.receipt),sha256=hashlib.sha256(self.receipt.read_bytes()).hexdigest())
        self.files=[b.Q/n for n in ('minicpm5_after_nemotron_package_v2.py',
            'minicpm5_saved_tap_runtime_contract_v2.py','run_minicpm5_saved_tap_serial_v2.py',
            'minicpm5_saved_tap_inputs_v1.py','minicpm5_saved_tap_io_v1.py','minicpm5_saved_tap_report_v1.py',
            'minicpm5_saved_tap_analysis_v1.py','dory_saved_review_guard_v1.py','dory_service_query_auth_v1.py')]
        self.graph=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in self.files},
                        synthetic_graph_for_builder_test=True)
        self.done=dict(history={'processes':[{}]*404},source_sha256={self.ref['path']:self.ref['sha256']})

    def invoke(self,profile='retry-v24',release=None):
        with patch.object(b.c.history,'completed',return_value=self.done),patch.object(b,'local_imports',return_value=self.graph):
            return b.assemble(profile,self.ref,release)

    def test_actual_completion_required_before_any_other_preparation(self):
        with (patch.object(b.c.history,'completed',side_effect=ValueError('no actual exit')),
              patch.object(b,'local_imports')as graph):
            with self.assertRaises(ValueError):b.assemble('retry-v24',self.ref)
            graph.assert_not_called()

    def test_all_three_profiles_candidate_only_and_all_inputs_unchanged(self):
        for profile in b.c.history.PROFILES:
            spec,record=self.invoke(profile)
            self.assertEqual(spec['budget'],b.c.BUDGET);self.assertEqual(spec['input_manifest'],b.c.MANIFEST)
            self.assertFalse(spec['runtime_admitted']);self.assertFalse(record['runtime_specification_written'])
            self.assertFalse(record['serial_lock_acquired']);self.assertEqual(record['raw_tensor_bytes_read'],0)
            self.assertEqual(record['predecessor_services'],1202)
        self.assertFalse(b.c.OUT.exists())

    def test_historical_source_rebinding_is_rejected(self):
        self.done['source_sha256'][str(self.files[3])]='f'*64
        with self.assertRaises(RuntimeError):self.invoke()

    def test_missing_required_executable_is_rejected(self):
        self.graph['source_sha256'].pop(str(b.c.EXECUTOR))
        with self.assertRaises(ValueError):self.invoke()

    def test_existing_output_blocks_candidate_reuse(self):
        with patch.object(b.c,'OUT',Path(self.tmp.name)):
            with self.assertRaises(ValueError):self.invoke()

    def test_real_failed_release_reference_is_retained_in_candidate(self):
        # Synthetic metadata only: this test does not certify a real release.
        path=Path(self.tmp.name)/'synthetic-failed-release.json';path.write_text('{}\n')
        ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.done['source_sha256'][ref['path']]=ref['sha256']
        spec,record=self.invoke(release=ref)
        self.assertEqual(spec['nemotron_failed_release_exit'],ref)
        self.assertEqual(record['actual_failed_release_exit'],ref)


if __name__=='__main__':unittest.main(verbosity=2)
