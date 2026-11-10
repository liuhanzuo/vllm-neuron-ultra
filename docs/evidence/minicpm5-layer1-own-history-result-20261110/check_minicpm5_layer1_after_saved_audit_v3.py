"""Real authenticated metadata plus negative edits; no tensors or native imports."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
Q = Path(__file__).resolve().parent
sys.path.insert(0, str(Q.parent))
from research import minicpm5_layer1_after_saved_audit_v3 as h


class ReviewedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def ref(name):
            p = Q / name
            return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        cls.binding = dict(schema='minicpm5_layer1_completed_collection_reviewed_metrics_v1',
            collector_exit=ref('minicpm5-layer1-collector-r2-actual-exit-20261110-v1.json'),
            diagnostic_exit=ref('minicpm5-layer1-metric-diagnostic-r1-actual-exit-20261110-v1.json'),
            metric_equivalence=ref('minicpm5-layer1-metric-history-equivalence-20261110-v1.json'))
        cls.docs = {}
        def read(ref):
            value = h.p.read_ref(ref)
            cls.docs[ref['path']] = value
            return value
        cls.bundle = h.completed(cls.binding, read)
        cls.exit = cls.docs[cls.binding['diagnostic_exit']['path']]
        cls.eq = cls.docs[cls.binding['metric_equivalence']['path']]

    def reject(self, path, edit):
        value = copy.deepcopy(self.docs[path])
        edit(value)
        with self.assertRaises((ValueError, AssertionError, KeyError)):
            h.completed(self.binding, lambda ref: value if ref['path'] == path else self.docs[ref['path']])

    def change_terminal(self, edit):
        self.reject(self.exit['terminal']['path'], edit)

    def test_real_metadata_keeps_failure_and_no_native_execution(self):
        self.assertTrue(self.bundle['metadata_complete'])
        self.assertFalse(self.bundle['audit_result']['strict_float64_scalar_tolerance_passed'])
        self.assertEqual(len(self.bundle['audit_result']['layer1_cache']['raw_records']), 84)
        self.assertFalse({'torch', 'numpy', 'neuronxcc'} & set(sys.modules))

    def test_diagnostic_exit_failure_rejected(self):
        self.reject(self.binding['diagnostic_exit']['path'], lambda d: d.update(actual_exit_code=1))

    def test_replaced_tool_session_rejected(self):
        self.reject(self.binding['diagnostic_exit']['path'],
                    lambda d: d['actual_execution_handle'].update(session_id=1))

    def test_wrong_live_process_rejected(self):
        self.reject(self.exit['live']['path'], lambda d: d['owner'].update(pid=1))

    def test_false_scalar_pass_rejected(self):
        self.change_terminal(lambda d: d.update(audit_passed=True))

    def test_relabeling_numeric_result_as_old_audit_rejected(self):
        self.change_terminal(lambda d: d['result']['numeric'].update(schema='minicpm5_layer1_full_logits_numeric_audit_v1'))

    def test_missing_metric_frame_rejected(self):
        self.change_terminal(lambda d: d['result']['numeric']['rows'][0]['rows'].pop())

    def test_hidden_scalar_violation_rejected(self):
        def edit(d):
            for arm in d['result']['numeric']['rows']:
                for row in arm['rows']:
                    if row['metric_tolerance_violations']:
                        row['metric_tolerance_violations'] = []
                        return
        self.change_terminal(edit)

    def test_relaxed_tolerance_rejected(self):
        self.change_terminal(lambda d: d['result']['numeric']['independent_float64_tolerances'].update(cosine=.1))

    def test_wrong_numeric_result_rejected(self):
        self.change_terminal(lambda d: d['result']['numeric']['rows'][0]['rows'][0]['metrics'].update(cosine=.1))

    def test_changed_model_failure_classification_rejected(self):
        self.change_terminal(lambda d: d['result']['numeric']['rows'][0]['rows'][1].update(failed_criteria=[]))

    def test_rank_replication_not_verified_rejected(self):
        self.change_terminal(lambda d: d['result']['numeric'].update(replicated_ranks_verified_and_counted_once=False))

    def test_missing_complete_cache_rejected(self):
        self.change_terminal(lambda d: d['result']['layer1_cache']['raw_records'].pop())

    def test_broken_cache_history_rejected(self):
        self.change_terminal(lambda d: d['result']['layer1_cache'].update(all_own_history_links_exact=False))

    def test_wrong_history_reference_rejected(self):
        self.reject(self.binding['metric_equivalence']['path'], lambda d: d['diagnostic_exit'].update(sha256='0'*64))

    def test_missing_history_call_rejected(self):
        self.reject(self.binding['metric_equivalence']['path'], lambda d: d['calls'].pop())

    def test_changed_producer_layout_rejected(self):
        self.reject(self.eq['calls'][0]['current']['path'], lambda d: d['output_metadata']['stride'].__setitem__(0, 999))

    def test_changed_producer_precision_rejected(self):
        self.reject(self.eq['calls'][0]['current']['path'], lambda d: d['output_metadata'].update(dtype='torch.bfloat16'))

    def test_historical_native_replay_not_exact_rejected(self):
        e = self.docs[self.eq['previous_metric_replay']['path']]
        self.reject(e['result']['path'], lambda d: d.update(all96_original_metrics_exact=False))

    def test_model_promotion_rejected(self):
        self.change_terminal(lambda d: d.update(full_model_qualified=True))


if __name__ == '__main__':
    unittest.main()
