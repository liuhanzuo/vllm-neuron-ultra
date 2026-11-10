"""Real original JSON and synthetic BF16 words; no real tensor reads."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_decode_boundary_v1 as b


def read(path,digest):
    raw=Path(path).read_bytes();assert len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()==digest
    return json.loads(raw)


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=read(str(Q/'minicpm5-layer1-runner-source-plan-20261110-v1.json'),
            '122cf50cac06022356a0681712ff762f343b80c74ac53520964e02b9451537de')
        cls.rows=b.all_boundaries(cls.plan,read)
        cls.ref={k:cls.plan['layer1_observation']['original_calls'][4][k]for k in ('path','sha256')}
        cls.record=read(cls.ref['path'],cls.ref['sha256'])
    def test_all36_actual_decode_boundaries_masks_positions_and_full_outputs(self):
        self.assertEqual(len(self.rows),36)
        self.assertEqual({row['bucket']for row in self.rows},{128,2048})
        self.assertEqual({row['call']['index']for row in self.rows},set(b.DECODE_CALLS))
        self.assertEqual(sum(len(row['physical_frame_order'])for row in self.rows),144)
        for row in self.rows:
            self.assertTrue(row['full_attention_output_required'])
            self.assertTrue(row['own_history_reproduction_required'])
            self.assertFalse(row['paired_history_intervention_allowed'])
            self.assertFalse(row['runtime_admitted'])
    def test_physical_slot_order_is_not_logical_prompt_order(self):
        row=self.rows[0]
        self.assertEqual([x['slot']for x in row['call']['frames']],[3,1,0,2])
        self.assertEqual([x['slot']for x in row['physical_frame_order']],[0,1,2,3])
        self.assertEqual(row['positions'],[28,1231,27,24])
    def test_all65536_bf16_words_restore_original_source_layout_without_conversion(self):
        for start in range(0,65536,8192):
            logical=struct.pack('<8192H',*range(start,start+8192))
            storage=b.normalized_storage(logical,b.NORM_META,b.sha(logical))
            self.assertEqual(len(storage),16384)
            # Independent channel-major expectation, covering NaNs and -0 bits too.
            self.assertEqual(struct.unpack('<8192H',storage),tuple(start+batch*2048+channel
                for channel in range(2048)for batch in range(4)))
    def test_naive_contiguous_storage_is_a_negative_layout_control(self):
        logical=struct.pack('<8192H',*range(8192))
        storage=b.normalized_storage(logical,b.NORM_META,b.sha(logical))
        self.assertNotEqual(storage,logical)
        self.assertEqual(struct.unpack_from('<H',storage,2)[0],2048)
        self.assertEqual(struct.unpack_from('<H',logical,2)[0],1)
    def test_truncated_altered_or_unapproved_normalized_layout_rejected(self):
        raw=bytes(16384)
        for meta,data,digest in ((b.NORM_META,raw[:-2],b.sha(raw)),(b.NORM_META,raw,'a'*64),
            (dict(b.NORM_META,stride=[2048,2048,1]),raw,b.sha(raw)),
            (dict(b.NORM_META,offset=1),raw,b.sha(raw)),
            (dict(b.NORM_META,stride=[0,4,4]),raw,b.sha(raw))):
            with self.subTest(meta=meta),self.assertRaises(ValueError):b.normalized_storage(data,meta,digest)
    def reject(self,change,message):
        plan=copy.deepcopy(self.plan);record=copy.deepcopy(self.record);ref=copy.deepcopy(self.ref)
        change(plan,record,ref)
        with self.assertRaisesRegex(ValueError,message):b.boundary(plan,record,ref)
    def test_prefill_prediction_taps_cannot_be_used_as_full_current_inputs(self):
        ref={k:self.plan['layer1_observation']['original_calls'][0][k]for k in ('path','sha256')}
        record=read(ref['path'],ref['sha256'])
        with self.assertRaisesRegex(ValueError,'prefill'):b.boundary(self.plan,record,ref)
    def test_changed_slots_positions_or_token_identity_rejected(self):
        self.reject(lambda p,r,f:r['physical_rows'][0].update(physical_row=0),'physical rows')
        self.reject(lambda p,r,f:r['physical_rows'][0].update(current_position=0),'current position')
        self.reject(lambda p,r,f:p['fixtures'][0]['continuation_ids'].__setitem__(0,0),'reproduced exactly')
    def test_inclusive_mask_or_changed_current_cell_cannot_replace_strict_mask(self):
        self.reject(lambda p,r,f:r['changed_mask_cells'][0].update(after=1),'strict current-mask')
        self.reject(lambda p,r,f:r['execution_input_metadata'][1].update(sha256='0'*64),'reproduced exactly')
    def test_wrong_layer_route_or_original_input_layout_rejected(self):
        self.reject(lambda p,r,f:r['route_records'][1].update(active_mask={}), 'entry mask')
        self.reject(lambda p,r,f:r['stage_metadata'][b.NORMALIZED].update(stride=[2048,2048,1]),'source/output layouts')
    def test_all_cache_slots_padding_and_unchanged_before_state_required(self):
        self.reject(lambda p,r,f:r['before_metadata'][2].update(shape=[4,1,2048,128]),'4224cache')
        self.reject(lambda p,r,f:r['input_after_metadata'][2].update(sha256='0'*64),'unchanged')
    def test_sdk_cache_view_preserves_full_physical_batch_stride(self):
        for row in self.rows:
            for cache in row['caches']:
                view=cache['sdk_view'];self.assertEqual(view['shape'],[4,1,row['bucket'],128])
                self.assertEqual(view['stride'],[540672,540672,128,1])
                self.assertGreater(view['stride'][0],row['bucket']*128)
                self.assertEqual(cache['complete_original_bytes'],4325376)
                self.assertTrue(cache['view_must_retain_complete_storage'])
    def test_original_reference_and_order_cannot_be_replaced(self):
        self.reject(lambda p,r,f:f.update(sha256='0'*64),'frozen original call')
        plan=copy.deepcopy(self.plan);plan['layer1_observation']['original_calls'].pop()
        with self.assertRaisesRegex(ValueError,'catalog'):b.all_boundaries(plan,read)
    def test_two_ranks_retained_for_cache_while_current_and_output_are_replicas(self):
        for i in range(18):
            a,c=self.rows[i],self.rows[i+18]
            self.assertEqual(a['normalized_input'],c['normalized_input'])
            self.assertEqual(a['expected_attention'],c['expected_attention'])
            self.assertEqual([a['rank'],c['rank']],[0,1])
        self.assertTrue(any(a['caches']!=c['caches']for a,c in zip(self.rows[:18],self.rows[18:])))
    def test_full_original_layer1_weight_header_and_both_rank_slices(self):
        reference=self.plan['safetensors_header'];header=read(reference['path'],reference['sha256'])
        rows=b.weight_slices(self.plan,header);self.assertEqual(len(rows),8)
        for rank in (0,1):
            selected=[r for r in rows if r['rank']==rank]
            self.assertEqual([r['shard_axis']for r in selected],[0,0,0,1])
            self.assertTrue(all(r['logical_kv_heads']==[rank]for r in selected))
            self.assertTrue(all(r['raw_weight_comparison_required_at_runtime']for r in selected))
        bad=copy.deepcopy(header);bad['model.layers.1.self_attn.q_proj.weight']['dtype']='F32'
        with self.assertRaisesRegex(ValueError,'official layer1'):b.weight_slices(self.plan,bad)
    def test_no_tensor_or_native_runtime_import(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx','transformers','vllm'}&set(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
