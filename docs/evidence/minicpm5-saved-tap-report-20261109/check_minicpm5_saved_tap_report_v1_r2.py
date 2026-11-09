"""Generated complete95-JSON/84-bundle trajectory; no original tensor IO."""
import copy
import hashlib
import json
import struct
import unittest

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
import minicpm5_saved_tap_report_v1 as report
from check_minicpm5_saved_tap_analysis_v1 import fixture
from minicpm5_saved_tap_analysis_v1 import ARMS, TAPS, WIDTH


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def trajectory():
    docs, payloads = {}, {}

    def put(path, value):
        raw = json.dumps(value, sort_keys=True).encode()
        ref = dict(path=str(path), sha256=sha(raw))
        docs[str(path)] = (ref, value)
        return ref

    groups = []
    for group in (0, 1):
        start = group*16
        groups.extend([(start+i,) for i in range(4)])
        groups.extend([(start+i+2, start+i+1, start+i+3, start+i) for i in (4, 8, 12)])
    all_frames = [dict(index=i) for i in range(32)]
    rank_refs = {}
    for rank in (0, 1):
        arms = []
        for ai, arm in enumerate(ARMS):
            entries = []
            for ci, order in enumerate(groups):
                _, call = fixture(order)
                phase = 'prefill' if len(order) == 1 else 'decode'
                call.update(rank=rank, arm=arm)
                call['call'].update(index=ci, phase=phase)
                for endpoint, row in zip(call['HF_metrics'], call['physical_rows']):
                    endpoint.update(physical_row=row['physical_row'],
                        metrics=dict(cosine=.998+ai*.0001, kl_reference_to_neuron=.1,
                                     argmax_match=ai == 0), failed_criteria=['KL'])
                parts = []
                for ti, tap in enumerate(TAPS):
                    delta = (0 if ai == 0 else 1) if phase == 'prefill' else (
                        (0 if ti < 2 else 4) if ai == 1 else (2 if ti < 2 else 6) if ai == 2 else 0)
                    raw = b''.join(struct.pack('<H', struct.unpack('<I', struct.pack(
                        '<f', float(frame+ti+1+delta)))[0] >> 16)*WIDTH for frame in order)
                    parts.append(raw)
                    call['selected_tap_bundle']['entries'][ti]['sha256'] = sha(raw)
                    call['stage_sha256'][tap] = sha(raw)
                raw = b''.join(parts)
                folder = metadata.BASE/f'cpu-rank{rank}'/arm/f'call-{ci:02d}'
                path = str(folder/'selected-taps.bf16')
                call['selected_tap_bundle'].update(path=path, sha256=sha(raw))
                payloads[path] = raw
                ref = put(folder/'result.json', call)
                entries.append(dict(call=ci, report=ref['path'], sha256=ref['sha256']))
            ref = put(metadata.BASE/f'cpu-rank{rank}'/arm/'result.json',
                      dict(complete=True, arm=arm, rank=rank, model_forwards=14, reports=entries))
            arms.append(dict(arm=arm, report=ref['path'], sha256=ref['sha256']))
        rank_refs[f'rank{rank}_result'] = put(metadata.BASE/f'cpu-rank{rank}'/'result.json',
            dict(complete=True, rank=rank, arms=list(ARMS), model_forwards=42,
                 selected_tap_names=list(TAPS), arm_reports=arms,
                 summaries={'fixture_only': True, 'not_actual_measurements': True}))
    plan = put(metadata.Q/'synthetic-plan.json', dict(output_root=str(metadata.BASE),
               frames=all_frames, remaining_gates=['synthetic_original_gates']))
    spec = put(metadata.Q/'synthetic-spec.json', dict(output_root=str(metadata.BASE), plan=plan))
    ended = dict(actual_exit_observed=True, actual_exit_code=0,
                 all_four_processes_gone=True, cgroup_empty=True, spec=spec, outputs=rank_refs)
    # This original exit reference is mocked in memory, never read from disk.
    docs[metadata.EXIT['path']] = (metadata.EXIT, ended)
    return docs, payloads


class FullReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_docs, cls.original_payloads = trajectory()

    def setUp(self):
        self.docs = copy.deepcopy(self.original_docs)
        self.payloads = dict(self.original_payloads)
        self.reads, self.guards = [], 0

    def read_metadata(self, ref):
        expected, value = self.docs[ref['path']]
        self.assertEqual(ref, expected)
        return copy.deepcopy(value)

    def read_raw(self, ref, size):
        self.reads.append((ref, size))
        return self.payloads[ref['path']]

    def guard(self):
        self.guards += 1

    def run_report(self):
        return report.analyze(self.read_metadata, self.read_raw, self.guard)

    def call(self, rank=0, arm=0, ci=4):
        path = str(metadata.BASE/f'cpu-rank{rank}'/ARMS[arm]/f'call-{ci:02d}'/'result.json')
        return self.docs[path][1]

    def test_complete_trajectory_counts_replicas_once_and_preserves_original_endpoints(self):
        result = self.run_report()
        self.assertEqual(len(self.reads), 84)
        self.assertEqual(sum(size for _, size in self.reads), 3145728)
        self.assertEqual(len(result['metadata_inputs']), 95)
        self.assertEqual(result['pairwise_vector_comparisons'], 384)
        self.assertTrue(result['replicated_ranks_byte_equal'])
        self.assertFalse(result['full_model_qualified'])
        self.assertFalse(result['new_speedup_claim'])
        self.assertFalse(result['runtime_admission_proven_by_this_module'])
        self.assertEqual(self.guards, 95*2+84*2+14*2+1)
        for pair in result['comparisons']:
            self.assertEqual(sorted(r['frame_index'] for r in pair['frames']), list(range(32)))
        frame = next(x for x in result['comparisons'][0]['frames'] if x['frame_index'] == 4)
        self.assertEqual(frame['physical_frame_order'], [6, 5, 7, 4])
        self.assertEqual(frame['physical_row'], 3)
        self.assertEqual(frame['left_original_endpoint'], self.call()['HF_metrics'][0])
        self.assertEqual(frame['taps'][TAPS[2]]['anchor_rms'], 7)
        self.assertEqual(frame['taps'][TAPS[2]]['candidate_rms'], 11)
        self.assertEqual(frame['taps'][TAPS[2]]['delta_rms'], 4)
        self.assertIsNone(frame['propagation']['attention']['late_to_early_rms_ratio'])
        self.assertTrue(frame['propagation']['attention']['zero_early_nonzero_late'])
        both = next(x for x in result['comparisons'][1]['frames'] if x['frame_index'] == 4)
        self.assertEqual(both['propagation']['attention']['late_to_early_rms_ratio'], 3)
        candidate_pair = result['comparisons'][2]['frames']
        for row in candidate_pair:
            for tap in row['taps'].values():
                self.assertFalse(tap['reference_is_failing_cpu_anchor'])
                self.assertFalse(tap['absolute_HF_error_measured'])
                self.assertEqual(tap['delta_rms'], 0 if row['phase'] == 'prefill' else 2)

    def test_rank0_or_rank1_payload_corruption_rejected(self):
        for rank in (0, 1):
            path = self.call(rank)['selected_tap_bundle']['path']
            original = self.payloads[path]
            self.payloads[path] = original[:-1]+bytes([original[-1] ^ 1])
            with self.subTest(rank=rank), self.assertRaisesRegex(ValueError, 'bundle bytes'):
                self.run_report()
            self.payloads[path] = original

    def test_missing_replica_cannot_be_skipped(self):
        del self.payloads[self.call(1)['selected_tap_bundle']['path']]
        with self.assertRaises(KeyError): self.run_report()

    def test_truncated_or_overlong_raw_bundle_rejected(self):
        path = self.call()['selected_tap_bundle']['path']
        original = self.payloads[path]
        for value in (original[:-2], original+b'\0\0'):
            self.payloads[path] = value
            with self.subTest(size=len(value)), self.assertRaisesRegex(ValueError, 'bundle bytes'):
                self.run_report()

    def test_replica_mapping_difference_rejected_before_raw_reads(self):
        self.call(1)['call']['extra'] = 'different trajectory'
        with self.assertRaisesRegex(ValueError, 'Replicated ranks'): self.run_report()
        self.assertEqual(self.reads, [])

    def test_wrong_endpoint_physical_row_rejected_before_raw_reads(self):
        for rank in (0, 1): self.call(rank)['HF_metrics'][0]['physical_row'] = 0
        with self.assertRaisesRegex(ValueError, 'endpoint physical'): self.run_report()
        self.assertEqual(self.reads, [])

    def test_rank_endpoint_difference_rejected(self):
        self.call(1)['HF_metrics'][0]['metrics']['cosine'] = 1
        with self.assertRaisesRegex(ValueError, 'endpoints'): self.run_report()
        self.assertEqual(self.reads, [])

    def test_missing_call_rejected_before_raw_reads(self):
        path = str(metadata.BASE/f'cpu-rank0/{ARMS[0]}/result.json')
        self.docs[path][1]['reports'].pop()
        with self.assertRaisesRegex(ValueError, '14 original'): self.run_report()
        self.assertEqual(self.reads, [])

    def test_unknown_phase_rejected_before_raw_reads(self):
        for rank in (0, 1):
            for arm in range(3): self.call(rank, arm)['call']['phase'] = 'new_phase'
        with self.assertRaisesRegex(ValueError, 'prefills'): self.run_report()
        self.assertEqual(self.reads, [])

    def test_metadata_reader_error_propagates_without_raw_reads(self):
        def broken(ref): raise RuntimeError('metadata authentication failure')
        with self.assertRaisesRegex(RuntimeError, 'authentication'):
            report.analyze(broken, self.read_raw, self.guard)
        self.assertEqual(self.reads, [])

    def test_guard_failure_before_first_read_prevents_all_callbacks(self):
        def stopped(): raise RuntimeError('lost serial ownership')
        def unread(_): self.fail('Metadata read after failed guard')
        with self.assertRaisesRegex(RuntimeError, 'ownership'):
            report.analyze(unread, self.read_raw, stopped)
        self.assertEqual(self.reads, [])

    def test_guard_failure_after_raw_callback_prevents_partial_report(self):
        def stopped_after_raw():
            if self.reads: raise RuntimeError('budget expired')
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            report.analyze(self.read_metadata, self.read_raw, stopped_after_raw)
        self.assertEqual(len(self.reads), 1)

    def test_unsigned_zero_ratio_never_infinite(self):
        result = report.propagation({'delta_rms': 0}, {'delta_rms': 2})
        self.assertIsNone(result['late_to_early_rms_ratio'])
        self.assertTrue(result['zero_early_nonzero_late'])
        self.assertFalse(result['causal_amplification_established'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
