"""Pure metadata fixtures. No producer output or tensor payload is read."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import summarize_dory_residual_propagation_v1 as s
from check_nemotron9b_after_dory_residual_review_v1 import fixture


def equal_stats():
    return dict(elements=2560,bitwise_equal_words=2560,numerically_equal_values=2560,
        max_abs=0.0,rms=0.0,reference_rms=1.0,relative_rms=0.0,cosine=1.0,mean_delta=0.0,
        reference_row_sha256='1'*64,incremental_row_sha256='1'*64,
        arithmetic='Exact BF16-to-float values with Python float64 accumulation; explanatory only',
        acceptance_metric=False)


class Summary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reader=s.completion.Reader();anchor=s.completion.anchor(reader)
        metadata=fixture(anchor)
        cls.template=copy.deepcopy(metadata['store'][metadata['saved']['path']])
        cls.plan=s.completion.Reader()(s.PLAN)
        original=s.original_frames(cls.plan)
        frames=[]
        for event,row in sorted(original.items()):
            frames.append(dict(event=event,fixture=row['fixture'],position=row['end']-1,
                comparisons={key:equal_stats()for key in s.BOUNDARIES},first_nonidentical_boundary=None,
                original_FP32_metrics=copy.deepcopy(row['original_FP32_metrics']),
                original_pointwise_failed=row['original_pointwise_failed'],
                original_metrics_recomputed=False,comparison_is_acceptance_metric=False,
                root_cause_established=False,full_model_qualified=False))
        cls.template['observations'].update(frames=frames,failure_review=dict(
            event=6194,original_failure_tensor_bytes_matched=False,saved_bytes_verified=8345600))

    def setUp(self):self.saved=copy.deepcopy(self.template)

    def run_summary(self):return s.analyze(self.saved,self.plan)

    def differ(self,frame_index,key,rms=0.25,numeric=True):
        frame=self.saved['observations']['frames'][frame_index]
        stats=frame['comparisons'][key]
        stats.update(bitwise_equal_words=2559,numerically_equal_values=2559 if numeric else 2560,
            max_abs=rms,rms=rms,relative_rms=rms,cosine=0.99 if numeric else 1.0,
            incremental_row_sha256='2'*64)
        frame['first_nonidentical_boundary']=next(k for k in s.BOUNDARIES
            if frame['comparisons'][k]['bitwise_equal_words']<2560)

    def test_complete_scope_retained_even_when_all_synthetic_boundaries_equal(self):
        result=self.run_summary()
        self.assertEqual(len(result['frames']),87)
        self.assertEqual(len(result['boundary_summary']),224)
        self.assertEqual(result['first_bit_difference_counts'],{'all_equal':87})
        self.assertEqual(len(result['failed_events']),7)
        self.assertFalse(result['original_FP32_acceptance_passed'])
        self.assertFalse(result['full_model_qualified'])
        self.assertEqual(sum(len(row['visits'])for row in result['frames']),87*112)

    def test_zero_input_to_nonzero_output_is_described_without_division(self):
        self.differ(0,'layer-0-loop-0-residual_output')
        result=self.run_summary();frame=result['frames'][0]
        self.assertEqual(frame['first_numeric_difference'],'layer-0-loop-0-residual_output')
        jump=frame['largest_rms_increases'][0]
        self.assertEqual((jump['physical_layer'],jump['recurrent_loop']),(0,0))
        self.assertIsNone(jump['rms_ratio'])
        self.assertTrue(jump['numeric_difference_first_appears_here'])
        self.assertFalse(jump['root_cause_established'])

    def test_recurrent_order_uses_execution_order_not_lexicographic_layer_names(self):
        self.differ(0,'layer-9-loop-0-residual_output')
        self.differ(0,'layer-16-loop-1-residual_output',rms=1.0)
        result=self.run_summary();frame=result['frames'][0]
        self.assertEqual(frame['first_bit_difference'],'layer-9-loop-0-residual_output')
        largest=frame['largest_rms_increases'][0]
        self.assertEqual((largest['physical_layer'],largest['recurrent_loop']),(16,1))
        self.assertEqual(largest['visit'],56)

    def test_signed_zero_bit_difference_is_not_numeric_difference(self):
        self.differ(0,s.BOUNDARIES[0],rms=0.0,numeric=False)
        result=self.run_summary();frame=result['frames'][0]
        self.assertEqual(frame['first_bit_difference'],s.BOUNDARIES[0])
        self.assertIsNone(frame['first_numeric_difference'])
        self.assertEqual(result['boundary_summary'][0]['numeric_different_frames'],0)

    def test_repeated_fixture_positions_in_different_requests_are_not_merged(self):
        result=self.run_summary()
        rows=result['frames']
        pairs=[(row['fixture'],row['position'])for row in rows]
        self.assertLess(len(set(pairs)),len(rows))
        self.assertEqual(len({row['event']for row in rows}),87)
        self.assertEqual(len(rows),len(self.saved['observations']['frames']))

    def test_missing_frame_boundary_or_original_failure_rejected(self):
        self.saved['observations']['frames'].pop()
        with self.assertRaises(RuntimeError):self.run_summary()
        self.setUp();self.saved['observations']['frames'][0]['comparisons'].popitem()
        with self.assertRaises(RuntimeError):self.run_summary()
        self.setUp()
        frame=next(row for row in self.saved['observations']['frames']if row['original_pointwise_failed'])
        frame['original_pointwise_failed']=False
        with self.assertRaises(RuntimeError):self.run_summary()

    def test_original_fp32_metrics_cannot_be_recomputed_or_changed(self):
        self.saved['observations']['frames'][0]['original_FP32_metrics']['max_abs_error']+=1
        with self.assertRaises(RuntimeError):self.run_summary()
        self.setUp();self.saved['observations']['frames'][0]['original_metrics_recomputed']=True
        with self.assertRaises(RuntimeError):self.run_summary()

    def test_nonfinite_or_wrong_width_metrics_rejected(self):
        for key,value in [('elements',2559),('rms',float('nan')),('relative_rms',float('inf')),('acceptance_metric',True)]:
            self.setUp();self.saved['observations']['frames'][0]['comparisons'][s.BOUNDARIES[0]][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.run_summary()

    def test_first_boundary_record_is_independently_rechecked(self):
        self.saved['observations']['frames'][0]['first_nonidentical_boundary']=s.BOUNDARIES[0]
        with self.assertRaises(RuntimeError):self.run_summary()

    def test_complete_input_metadata_is_unchanged(self):
        before=copy.deepcopy(self.saved)
        self.run_summary()
        self.assertEqual(self.saved,before)

    def test_actual_exit_gate_precedes_all_saved_metadata_reads(self):
        with patch.object(s.completion,'completed',side_effect=RuntimeError('review still live')), \
             patch.object(s.completion,'Reader')as read:
            with self.assertRaisesRegex(RuntimeError,'still live'):
                s.from_actual_review({'path':'/synthetic/unwritten-exit.json','sha256':'0'*64})
        read.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
