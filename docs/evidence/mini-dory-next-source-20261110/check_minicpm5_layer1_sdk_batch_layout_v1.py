"""Synthetic buffers only; exercise physical row, stride, masking and controls."""
import copy
from pathlib import Path
import struct
import sys
import unittest
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q.parent))
from research import minicpm5_layer1_sdk_batch_layout_v1 as core
from research import check_minicpm5_layer1_current_history_factorial_v1 as fixture


class LayoutChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CurrentHistoryChecks.setUpClass();f=fixture.CurrentHistoryChecks
        cls.case=copy.deepcopy(f.case);cls.case['call']['frames']=sorted(cls.case['physical_frame_order'],key=lambda v:v['index'])
        cls.prepared=core.prepare(cls.case,f.normalized,f.own,f.prior,f.current)
        cls.tasks={variant:core.tasks(cls.prepared,variant)for variant in core.VARIANTS}

    def output_rows(self,variant):
        return [dict(rank=t['rank'],call=t['call'],variant=variant,slots=t['slots'],physical_frames=t['physical_frames'],
            raw=b''.join(struct.pack('<H',0x4000+s)*2048 for s in t['slots']))for t in self.tasks[variant]]

    def test_four_variants_select_identical_full_current_words(self):
        for variant,rows in self.tasks.items():
            raw=b''.join(core.logical_bytes(self.prepared['normalized_raw']if variant.endswith('contiguous')else self.prepared['packed_raw'],
                t['normalized'])for t in rows)
            self.assertEqual(raw,self.prepared['normalized_raw'])
            self.assertEqual(len(rows),4 if variant.startswith('b1_')else 1)

    def test_b1_strided_retains_real_offset_and_all_storage_holes(self):
        for slot,t in enumerate(self.tasks['b1_strided']):
            self.assertEqual(t['normalized']['shape'],[1,1,2048]);self.assertEqual(t['normalized']['stride'],[1,4,4])
            self.assertEqual(t['normalized']['offset'],slot);self.assertEqual(t['normalized']['storage_bytes'],16384)
            self.assertEqual(t['normalized']['logical_bytes'],4096);self.assertEqual(t['normalized_unselected_backing_bytes'],12288)
            self.assertEqual(core.logical_bytes(self.prepared['packed_raw'],t['normalized']),struct.pack('<H',0x4000+slot)*2048)

    def test_contiguous_b1_has_original_slot_offset_not_zero_relabel(self):
        for slot,t in enumerate(self.tasks['b1_contiguous']):
            self.assertEqual(t['normalized']['stride'],[2048,2048,1]);self.assertEqual(t['normalized']['offset'],slot*2048)
            self.assertEqual(t['seq_ids'],[slot]);self.assertEqual(t['physical_frames'],[self.case['physical_frame_order'][slot]])

    def test_cache_view_preserves_anchor_padding_and_full_physical_stride(self):
        for slot,t in enumerate(self.tasks['b1_strided']):
            for kind,name in enumerate(core.boundary.CACHE_NAMES):
                v=t['caches'][name];self.assertEqual(v['shape'],[1,1,128,128])
                self.assertEqual(v['stride'],[540672,540672,128,1]);self.assertEqual(v['offset'],slot*540672)
                self.assertEqual(v['storage_bytes'],4325376)
                raw=core.logical_bytes(self.prepared['cache_raw'][name],v);position=self.case['positions'][slot]
                self.assertEqual(raw[:position*256],struct.pack('<H',0x4000+32*kind+8+slot)*(position*128))
                self.assertEqual(raw[position*256:],struct.pack('<H',0x3f80+16*kind+slot)*((128-position)*128))

    def test_mask_remains_prior_only_bucket_sized(self):
        for rows in self.tasks.values():
            for t in rows:
                for index,position in enumerate(t['positions']):
                    self.assertEqual(t['mask_raw'][index*128:(index+1)*128],bytes([1])*position+bytes(128-position))
                self.assertEqual(t['sdk_prior_width'],128);self.assertEqual(t['native_total_lengths'],[p+1 for p in t['positions']])
                self.assertEqual(t['sdk_active_width'],1)

    def test_long_bucket_metadata_uses_2048_not_native_length(self):
        f=fixture.CurrentHistoryChecks;case=copy.deepcopy(self.case);case['bucket']=2048
        for desc in case['caches']:desc['sdk_view']['shape']=[4,1,2048,128]
        prepared=core.prepare(case,f.normalized,f.own,f.prior,f.current)
        for t in core.tasks(prepared,'b1_contiguous'):
            self.assertEqual(t['sdk_prior_width'],2048);self.assertEqual(t['mask']['logical_bytes'],2048)
            self.assertEqual(t['caches'][core.boundary.CACHE_NAMES[0]]['shape'],[1,1,2048,128])

    def test_rope_is_original_b4_selected_and_never_recomputed_for_b1(self):
        for rows in self.tasks.values():
            for t in rows:
                self.assertFalse(t['rope']['recompute_for_B1']);self.assertTrue(t['rope']['require_same_selected_cos_sin_bytes'])
                self.assertEqual(t['rope']['slots'],t['slots'])

    def test_no_precision_reduction_or_native_path_change(self):
        for rows in self.tasks.values():
            for t in rows:
                self.assertEqual(t['qkv_projection']['reduction_width'],2048)
                self.assertEqual(t['attention_arithmetic']['qk_reduction_width'],128)
                self.assertEqual(t['output_projection']['rank_reduction_width'],1024)
                self.assertEqual(t['output_projection']['collective_dtype'],'torch.bfloat16')
                self.assertFalse(t['model_config_changed']);self.assertFalse(t['native_reference_reexecuted'])

    def test_physical_slot_order_survives_fixture_order_scrambling(self):
        self.assertNotEqual(self.case['call']['frames'],self.case['physical_frame_order'])
        expected=self.prepared['normalized_raw']
        for variant in core.VARIANTS:self.assertEqual(core.combine(self.case,variant,self.output_rows(variant)),expected)

    def test_missing_duplicate_reordered_wrong_rank_and_call_outputs_rejected(self):
        for change in [lambda r:r.pop(),lambda r:r.reverse(),lambda r:r[1].update(slots=[0]),lambda r:r[0].update(rank=0),lambda r:r[0].update(call={})]:
            rows=self.output_rows('b1_strided');change(rows)
            with self.assertRaises(ValueError):core.combine(self.case,'b1_strided',rows)

    def test_short_or_nonfinite_output_rejected(self):
        for raw in (bytes(4094),b'\x80\x7f'+bytes(4094)):
            rows=self.output_rows('b1_contiguous');rows[0]['raw']=raw
            with self.assertRaises(ValueError):core.combine(self.case,'b1_contiguous',rows)

    def test_original_and_previous_both_full_bytes_controls_required(self):
        own=bytes(16384);official=b'\x80\x3f'*8192
        self.assertTrue(core.verify_controls(own,own,own,official,official,own)['previous_both_official_full_bytes_exact'])
        for index in range(6):
            values=[own,own,own,official,official,own];values[index]=b'\x00\x3f'*8192
            with self.subTest(index=index),self.assertRaises(ValueError):core.verify_controls(*values)

    def test_mutated_case_packed_source_or_padding_rejected(self):
        for change in [lambda p:p['case']['positions'].__setitem__(0,2),lambda p:p.update(packed_raw=bytes(16384)),
            lambda p:p['cache_raw'].__setitem__(core.boundary.CACHE_NAMES[0],p['cache_raw'][core.boundary.CACHE_NAMES[0]][:-2]+b'\x80\x40')]:
            prepared=copy.deepcopy(self.prepared);change(prepared)
            with self.assertRaises(ValueError):core.tasks(prepared,'b1_strided')

    def test_other_arm_or_invented_native_variant_rejected(self):
        f=fixture.CurrentHistoryChecks;case=copy.deepcopy(self.case);case['arm']=core.boundary.ARMS[1]
        with self.assertRaises(ValueError):core.prepare(case,f.normalized,f.own,f.prior,f.current)
        with self.assertRaises(ValueError):core.tasks(self.prepared,'native_b1')

    def test_no_native_imports(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
