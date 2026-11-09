"""Small generated fixtures for saved MiniCPM tap interpretation; no model IO."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import minicpm5_saved_tap_analysis_v1 as a
import analyze_minicpm5_saved_tap_metadata_v1 as m


def fixture(order=(6,5,7,4)):
    frames = sorted(order)
    rows = [dict(frame=dict(index=i),physical_row=order.index(i)) for i in frames]
    source = dict(shape=[len(order),1,2048],stride=[1,4,4],offset=0,dtype='torch.bfloat16')
    call = dict(complete=True,arm=a.ARMS[0],rank=0,call=dict(index=4,frames=[r['frame'] for r in rows]),
        physical_rows=rows,HF_metrics=[dict(frame=r['frame'],metrics={}) for r in rows],
        stage_names=list(a.STAGES),stage_sha256={name:'a'*64 for name in a.STAGES},
        stage_metadata={name:dict(source) for name in a.STAGES},
        before_metadata=[dict(name=str(i),sha256='a'*64) for i in range(84)],
        post_metadata=[dict(name=str(i),sha256='a'*64) for i in range(84)],
        output_metadata=dict(sha256='a'*64),full_model_qualified=False,new_speedup_claim=False)
    parts,entries,offset = [],[],0
    for si,name in enumerate(a.TAPS):
        data = b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',float(frame+si+1)))[0]>>16)
                        *a.WIDTH for frame in order)
        entries.append(dict(name=name,offset=offset,raw_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),shape=[len(order),1,2048],
            dtype='torch.bfloat16',source_metadata=source,frame_indices=list(order)))
        parts.append(data);offset += len(data)
    raw=b''.join(parts)
    call['selected_tap_bundle']=dict(encoding='raw_bf16_le',raw_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),entries=entries)
    return raw,call


class Interpretation(unittest.TestCase):
    def test_decode_preserves_original_physical_permutation(self):
        raw,call=fixture()
        decoded=a.decode_bundle(raw,call)
        for si,name in enumerate(a.TAPS):
            for frame in (4,5,6,7):
                self.assertEqual(decoded[name][frame],[float(frame+si+1)]*2048)

    def test_singleton_source_stride_need_not_be_canonical(self):
        raw,call=fixture((1,))
        self.assertEqual(a.decode_bundle(raw,call)[a.TAPS[0]][1],[2.0]*2048)

    def test_sorted_logical_order_cannot_replace_physical_order(self):
        _,call=fixture()
        call['selected_tap_bundle']['entries'][0]['frame_indices'].sort()
        with self.assertRaisesRegex(ValueError,'physical'):a.validate_call(call)

    def test_duplicate_physical_row_rejected(self):
        _,call=fixture()
        call['physical_rows'][0]['physical_row']=call['physical_rows'][1]['physical_row']
        with self.assertRaises(ValueError):a.validate_call(call)

    def test_missing_stage_cannot_look_like_no_difference(self):
        _,left=fixture();right=copy.deepcopy(left)
        right['stage_sha256'].pop(a.STAGES[3])
        with self.assertRaisesRegex(ValueError,'213'):a.compare_call(left,right)

    def test_first_difference_and_later_equal_stage_are_reported(self):
        _,left=fixture();right=copy.deepcopy(left)
        right['stage_sha256'][a.STAGES[2]]='b'*64
        result=a.compare_call(left,right)
        self.assertEqual(result['first_recorded_different_stage'],'layers.0.self_attn')
        self.assertIn(a.STAGES[3],result['equal_stages_after_first_difference'])
        self.assertTrue(result['logits_hash_equal'])
        self.assertFalse(result['per_frame_internal_magnitude_known'])

    def test_different_frame_or_cache_order_rejected(self):
        _,left=fixture()
        for field in ('physical_rows','before_metadata'):
            right=copy.deepcopy(left);right[field].reverse()
            with self.subTest(field=field),self.assertRaises(ValueError):a.compare_call(left,right)

    def test_bundle_byte_change_or_truncation_rejected(self):
        raw,call=fixture()
        for changed in (raw[:-1],b'\x00'+raw[1:-1]+b'\x01'):
            with self.assertRaisesRegex(ValueError,'bundle'):a.decode_bundle(changed,call)

    def test_per_entry_hash_checked_even_if_bundle_hash_valid(self):
        raw,call=fixture()
        call['selected_tap_bundle']['entries'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'tap bytes'):a.decode_bundle(raw,call)

    def test_nonfinite_bf16_rejected_after_all_hashes_match(self):
        raw,call=fixture()
        raw=struct.pack('<H',0x7f80)+raw[2:]
        call['selected_tap_bundle']['sha256']=hashlib.sha256(raw).hexdigest()
        first=call['selected_tap_bundle']['entries'][0]
        first['sha256']=hashlib.sha256(raw[:first['raw_bytes']]).hexdigest()
        with self.assertRaisesRegex(ValueError,'Finite BF16'):a.decode_bundle(raw,call)

    def test_gapped_overlapping_or_wrong_width_bundle_rejected(self):
        raw,base=fixture()
        for key,value in (('offset',1),('raw_bytes',1),('shape',[4,1,1024])):
            call=copy.deepcopy(base);call['selected_tap_bundle']['entries'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):a.decode_bundle(raw,call)

    def test_descriptive_statistics_analytic_case_and_zero_anchor(self):
        result=a.perturbation([2.0]*2048,[3.0]*2048)
        self.assertEqual(result['delta_rms'],1)
        self.assertEqual(result['relative_l2'],.5)
        self.assertEqual(result['changed_values'],2048)
        self.assertEqual(result['cosine'],1)
        self.assertFalse(result['absolute_HF_error_measured'])
        zero=a.perturbation([0.0]*2048,[1.0]*2048)
        self.assertIsNone(zero['relative_l2']);self.assertIsNone(zero['cosine'])

    def test_opposite_direction_keeps_signed_delta(self):
        result=a.perturbation([1.0]*2048,[-1.0]*2048)
        self.assertEqual(result['cosine'],-1)
        self.assertEqual(result['signed_mean_delta'],-2)
        self.assertEqual(result['delta_rms'],2)

    def test_nonfinite_or_wrong_size_vector_rejected(self):
        for vector in ([1.0]*2047,[math.nan]*2048):
            with self.assertRaises(ValueError):a.perturbation([1.0]*2048,vector)

    def test_metadata_reader_never_opens_tensor_or_unapproved_roots(self):
        read=m.Reader()
        for path in (str(m.BASE/'x.bf16'),str(m.BASE/'x.pt'),'/tmp/unapproved.json'):
            with patch.object(m.os,'open') as opened,self.assertRaises(ValueError):
                read(dict(path=path,sha256='0'*64))
            opened.assert_not_called()

    def test_incomplete_original_exit_stops_before_spec_read(self):
        def read(ref):
            self.assertEqual(ref,m.EXIT)
            return dict(actual_exit_observed=True,actual_exit_code=1,
                        all_four_processes_gone=True,cgroup_empty=True)
        with self.assertRaisesRegex(ValueError,'actually completed'):m.analyze(read)


if __name__ == '__main__':
    unittest.main(verbosity=2)
