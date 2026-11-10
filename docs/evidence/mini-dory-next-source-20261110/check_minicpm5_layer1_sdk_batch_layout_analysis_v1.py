"""Synthetic complete-byte analysis controls; no file/model payload inputs."""
import ast
import copy
import struct
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import minicpm5_layer1_sdk_batch_layout_analysis_v1 as core


def tensor(value):
    word = struct.unpack('<I', struct.pack('<f', value))[0] >> 16
    return struct.pack('<H', word)*8192


def fixture():
    cases, rows, controls = [], [], {}
    for rank in (0, 1):
        for group in range(2):
            for step in (1, 2, 3):
                ci = core.CALLS[group*3+step-1]
                frames = [dict(index=group*12+(step-1)*4+slot, fixture=group*4+slot, slot=slot, step=step) for slot in range(4)]
                call = dict(index=ci, group=group, phase='decode', frames=list(reversed(frames)))
                original = dict(path=f'/synthetic/rank{rank}/call{ci}.json', sha256='a'*64)
                case = dict(rank=rank, arm='strict_anchor_s0', call=call, original_call=original,
                            physical_frame_order=frames, expected_attention=dict(logical_sha256=core.digest(tensor(1))))
                cases.append(case)
                controls[rank, ci] = dict(call=copy.deepcopy(call), original_call=original,
                    original=tensor(1), previous_own=tensor(1), previous_restore=tensor(1),
                    previous_both=tensor(2), official_target=tensor(4))
                for mode, value in zip(core.VARIANTS, (1, 2, 3, 4, 6, 1)):
                    rows.append(dict(rank=rank, call=copy.deepcopy(call), variant=mode, raw=tensor(value)))
    return cases, rows, controls


class AnalysisChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=fixture();cls.result=core.analyze(*cls.data)

    def test_complete_frames_and_replication_not_double_counted(self):
        r=self.result
        self.assertEqual(r['complete_output_tensors'],72);self.assertEqual(r['complete_output_bytes'],1179648)
        self.assertEqual(len(r['records']),144);self.assertEqual(len(r['pair_records']),96)
        self.assertEqual(len(r['interactions']),24);self.assertEqual(r['unique_frames'],24)
        self.assertTrue(r['both_ranks_identical'])

    def test_known_exact_full_distance_and_direction(self):
        summary={r['variant']:r for r in self.result['summary']}
        self.assertEqual(summary['b4_strided']['aggregate_relative_l2'],0.5)
        self.assertEqual(summary['b4_contiguous']['aggregate_relative_l2'],0.25)
        self.assertEqual(summary['b1_strided']['raw_exact_frames'],24)
        self.assertEqual(summary['b1_strided']['aggregate_relative_l2'],0)
        self.assertEqual(summary['b1_strided']['improved_frames'],24)
        self.assertEqual(summary['own']['worsened_frames'],24)

    def test_pair_effects_and_interaction_have_distinct_denominators(self):
        rows={r['comparison']:r for r in self.result['pair_summary']}
        self.assertEqual(rows['batch_at_strided']['aggregate_relative_l2'],1.0)
        self.assertEqual(rows['layout_at_b1']['aggregate_relative_l2'],0.5)
        self.assertEqual(self.result['interaction_summary']['aggregate_relative_l2'],0.25)
        self.assertEqual(self.result['interaction_summary']['nonzero_frames'],24)

    def test_small_interaction_survives_large_term_cancellation(self):
        data=copy.deepcopy(self.data)
        for control in data[2].values():control['previous_both']=tensor(1)
        magnitudes=dict(own=1,b4_strided=1,b4_contiguous=1,b1_strided=2**-60,b1_contiguous=0,restore=1)
        for row in data[1]:row['raw']=tensor(magnitudes[row['variant']])
        result=core.analyze(*data)
        self.assertEqual(result['interaction_summary']['nonzero_frames'],24)
        self.assertEqual(result['interaction_summary']['aggregate_relative_l2'],2**-62)
        self.assertTrue(all(row['max_abs']==2**-60 for row in result['interactions']))

    def test_signed_zero_raw_difference_is_not_numeric_change(self):
        d=core.distance(b'\x00\x80',b'\x00\x00')
        self.assertFalse(d['raw_exact']);self.assertEqual(d['changed_words'],1)
        self.assertEqual(d['numeric_changed_elements'],0);self.assertEqual(d['l2_squared'],0)
        self.assertIsNone(d['relative_l2'])

    def test_zero_target_norm_returns_none_not_divide_by_zero(self):
        data=copy.deepcopy(self.data)
        for c in data[2].values():c['official_target']=bytes(16384)
        r=core.analyze(*data)
        self.assertTrue(all(x['aggregate_relative_l2']is None for x in r['summary']))
        self.assertIsNone(r['interaction_summary']['aggregate_relative_l2'])

    def reject(self, change):
        data=copy.deepcopy(self.data);change(*data)
        with self.assertRaises(ValueError):core.analyze(*data)

    def test_missing_duplicate_and_extra_rows_rejected(self):
        for change in [lambda c,r,k:r.pop(),lambda c,r,k:r.append(r[0]),lambda c,r,k:r.__setitem__(-1,r[0])]:
            with self.subTest(change=change):self.reject(change)

    def test_wrong_call_rank_variant_and_physical_order_rejected(self):
        for change in [lambda c,r,k:r[0].update(rank=True),lambda c,r,k:r[0]['call'].update(group=100),
                       lambda c,r,k:r[0].update(variant='native_b1'),lambda c,r,k:c[0]['physical_frame_order'].reverse()]:
            with self.subTest(change=change):self.reject(change)

    def test_full_unselected_channel_corruption_detected(self):
        def change(c,r,k):
            row=next(x for x in r if x['rank']==1 and x['variant']=='b1_contiguous')
            row['raw']=row['raw'][:-2]+b'\x80\x40'
        self.reject(change)

    def test_every_old_control_and_original_hash_required(self):
        for name in ['original','previous_own','previous_restore','previous_both']:
            with self.subTest(name=name):self.reject(lambda c,r,k:k[0,4].update({name:tensor(8)}))
        self.reject(lambda c,r,k:c[0]['expected_attention'].update(logical_sha256='b'*64))

    def test_restore_failure_in_both_ranks_rejected(self):
        def change(c,r,k):
            for row in r:
                if row['variant']=='restore':row['raw']=tensor(8)
        self.reject(change)

    def test_control_provenance_and_rank_target_mismatch_rejected(self):
        self.reject(lambda c,r,k:k[0,4].update(original_call=dict(path='/wrong',sha256='c'*64)))
        self.reject(lambda c,r,k:k[1,4].update(official_target=tensor(3)))

    def test_nonfinite_signed_inf_and_nan_all_rejected(self):
        for word in (0x7f80,0xff80,0x7fc0,0xffc1):
            raw=struct.pack('<H',word)+bytes(16382)
            with self.subTest(word=word):
                self.reject(lambda c,r,k:r[0].update(raw=raw))
                self.reject(lambda c,r,k:k[0,4].update(official_target=raw))

    def test_short_payload_missing_case_extra_control_rejected(self):
        for change in [lambda c,r,k:r[0].update(raw=bytes(16382)),lambda c,r,k:c.pop(),lambda c,r,k:k.update({(2,4):k[0,4]})]:
            self.reject(change)

    def test_all_original_fixture_step_pairs_required(self):
        def change(c,r,k):
            for case in c:
                for frame in case['physical_frame_order']:frame['fixture']=0
        self.reject(change)

    def test_no_runtime_or_qualification_claim(self):
        for flag in ('runtime_admitted','root_cause_proven','full_model_qualified','new_speedup_claim','authenticated_file_provenance_checked_here'):
            self.assertFalse(self.result[flag])
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'} & set(sys.modules))

    def test_python310_syntax_and_stdlib_only_imports(self):
        source=Path(core.__file__).read_text();tree=ast.parse(source,feature_version=(3,10))
        self.assertTrue(all(n.names[0].name in {'hashlib','math','struct'} for n in ast.walk(tree)if isinstance(n,ast.Import)))
        self.assertFalse(any(isinstance(n,ast.ImportFrom)for n in ast.walk(tree)))


if __name__=='__main__':unittest.main(verbosity=2)
