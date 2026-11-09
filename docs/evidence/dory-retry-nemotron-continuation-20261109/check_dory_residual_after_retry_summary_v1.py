"""Completion binding tests; original numerical summary tests remain intact."""
import copy
import unittest
from unittest.mock import Mock, patch

import summarize_dory_residual_propagation_after_retry_v1 as s


class SummaryBinding(unittest.TestCase):
    def setUp(self):
        self.exit=dict(path='/synthetic/retry-exit.json',sha256='1'*64)
        self.failed=dict(path='/synthetic/failed-review.json',sha256='2'*64)
        self.saved_ref=dict(path='/synthetic/saved-review.json',sha256='3'*64)
        self.saved,self.plan={'saved':'unchanged'},{'plan':'unchanged'}

    def test_incomplete_retry_prevents_all_analysis_and_output_reads(self):
        with (patch.object(s.completion,'completed',side_effect=RuntimeError('retry incomplete')),
             patch.object(s.completion,'Reader') as reader,
             patch.object(s.original,'analyze') as analyze):
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                s.from_actual_review(self.exit)
            reader.assert_not_called()
            analyze.assert_not_called()

    def test_original_saved_analysis_receives_unchanged_inputs(self):
        done=dict(history=dict(processes=[{}]*349,failed_residual_saved_review=self.failed))
        def read(ref):
            if ref==self.exit:return dict(saved_review=self.saved_ref)
            if ref==self.saved_ref:return self.saved
            if ref==s.original.PLAN:return self.plan
            raise AssertionError(ref)
        result=dict(original_pointwise_failures=7,full_model_qualified=False,new_speedup_claim=False,
                    existing_profile={'unchanged':True})
        with (patch.object(s.completion,'completed',return_value=done) as completed,
             patch.object(s.completion,'Reader',return_value=read),
             patch.object(s.original,'analyze',return_value=copy.deepcopy(result)) as analyze):
            actual=s.from_actual_review(self.exit)
        completed.assert_called_once_with(self.exit)
        analyze.assert_called_once_with(self.saved,self.plan)
        self.assertEqual({k:actual[k] for k in result},result)
        self.assertEqual(actual['failed_saved_review_exit'],self.failed)
        self.assertEqual(actual['completed_history_owners'],349)
        self.assertEqual((actual['raw_tensor_bytes_read'],actual['weight_bytes_read']),(0,0))

    def test_output_namespace_cannot_overwrite_original_summary(self):
        self.assertNotEqual(s.OUTPUT,s.Q/'dory-saved-residual-propagation-summary-20261109-v1.json')


if __name__=='__main__':
    unittest.main(verbosity=2)
