"""Standard-library controls for physical-slot official cache interventions."""
import ast
import copy
from pathlib import Path
import struct
import sys
import unittest

Q = Path(__file__).resolve().parent
sys.path.insert(0, str(Q.parent))
from research import minicpm5_layer1_official_history_v1 as core
from research import minicpm5_layer1_official_history_callable_v1 as native


class OfficialHistoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [dict(index=i, fixture=f, slot=s, step=1)
                      for s, (i, f) in enumerate(((6, 2), (5, 7), (7, 5), (4, 0)))]
        cls.positions = [3, 5, 4, 2]
        cls.own = {name: b''.join(struct.pack('<H', 0x3f80 + 16*kind + slot) * (4224*128)
                    for slot in range(4)) for kind, name in enumerate(core.boundary.CACHE_NAMES)}
        cls.case = dict(schema='minicpm5_layer1_decode_boundary_v1', rank=1,
            arm=core.boundary.ARMS[0], call=dict(index=4, phase='decode', frames=cls.frames),
            bucket=128, original_layer_index=1, seq_ids=[0, 1, 2, 3], strict_mask_kind='prior_positions_only',
            full_model_qualified=False, new_speedup_claim=False, physical_frame_order=cls.frames,
            positions=cls.positions, original_call=dict(path='/original/call-04/result.json', sha256='a'*64), caches=[])
        for name in core.boundary.CACHE_NAMES:
            cls.case['caches'].append(dict(name=name, complete_source_metadata=dict(name=name,
                shape=[4, 1, 4224, 128], stride=[540672, 540672, 128, 1], offset=0,
                dtype='torch.bfloat16', sha256=core.boundary.sha(cls.own[name])),
                complete_original_bytes=4*4224*128*2, view_must_retain_complete_storage=True,
                sdk_view=dict(shape=[4, 1, 128, 128], stride=[540672, 540672, 128, 1], offset=0, dtype='torch.bfloat16')))
        cls.donors = []
        for frame, length in zip(cls.frames, cls.positions):
            slot = frame['slot']
            row = dict(schema='minicpm5_layer1_native_cache_frame_v1', complete=True,
                frame=dict(frame, index=frame['index']-4, step=0), native_fixture_order=frame['fixture'],
                step=0, cache_length=length, cache_metadata={}, outputs={},
                full_logits_raw_exact=True, all213_selected_taps_raw_exact=True, official_bf16=True,
                history_unmodified=True, full_model_qualified=False, new_speedup_claim=False,
                original_frame=dict(path='/original/frame-%d.json' % (frame['index']-4), sha256='b'*64),
                original_reference=dict(path='/original/hf-reference.pt', sha256='c'*64))
            donor = dict(record=row)
            for kind_index, kind in enumerate(('key', 'value')):
                raw = b''.join(struct.pack('<H', 0x4000 + 32*kind_index + 8*head + slot) * (length*128)
                               for head in (0, 1))
                donor[kind+'_raw'] = raw
                row['cache_metadata'][kind] = dict(shape=[1, 2, length, 128], stride=[length*256, length*128, 128, 1],
                    offset=0, dtype='torch.bfloat16')
                row['outputs'][kind] = dict(path='/native/frame-%d/%s.bf16' % (frame['index']-4, kind),
                    raw_bytes=len(raw), sha256=core.boundary.sha(raw), identity=dict(size=len(raw)))
            cls.donors.append(donor)

    def run_overlay(self, mode, case=None, own=None, donors=None):
        return core.overlay(self.case if case is None else case, self.own if own is None else own,
                            self.donors if donors is None else donors, mode)

    def test_own_and_restore_keep_every_original_byte(self):
        for mode in ('own', 'restore'):
            out, proof = self.run_overlay(mode)
            self.assertEqual(out, self.own)
            self.assertEqual(proof['input_cache_sha256'], proof['output_cache_sha256'])
            self.assertTrue(proof['own_control']); self.assertEqual(proof['replaced_kinds'], [])

    def test_all_three_interventions_select_correct_slot_head_and_kind(self):
        for mode, kinds in [('official_key', {'key'}), ('official_value', {'value'}), ('official_both', {'key', 'value'})]:
            out, proof = self.run_overlay(mode)
            for kind_index, (kind, name) in enumerate(zip(('key', 'value'), core.boundary.CACHE_NAMES)):
                for slot, length in enumerate(self.positions):
                    begin = slot*4224*128*2; end = begin+length*128*2; stop=(slot+1)*4224*128*2
                    expected = struct.pack('<H', 0x4000+32*kind_index+8+slot)*(length*128)
                    self.assertEqual(out[name][begin:end], expected if kind in kinds else self.own[name][begin:end])
                    self.assertEqual(out[name][end:stop], self.own[name][end:stop])
                    self.assertEqual(proof['donors'][slot]['caches'][kind]['replaced'], kind in kinds)

    def test_rank_zero_uses_first_native_head(self):
        case=copy.deepcopy(self.case); case['rank']=0
        out, _=self.run_overlay('official_key',case=case)
        for slot,length in enumerate(self.positions):
            start=slot*4224*128*2
            self.assertEqual(out[core.boundary.CACHE_NAMES[0]][start:start+length*256],
                             struct.pack('<H',0x4000+slot)*(length*128))

    def test_future_padding_changes_remain_untouched(self):
        case=copy.deepcopy(self.case); own=dict(self.own);name=core.boundary.CACHE_NAMES[0]
        storage=bytearray(own[name]); storage[128*256:128*256+2]=b'\x80\x40';own[name]=bytes(storage)
        case['caches'][0]['complete_source_metadata']['sha256']=core.boundary.sha(own[name])
        out,_=self.run_overlay('official_both',case=case,own=own)
        self.assertEqual(out[name][128*256:128*256+2],b'\x80\x40')

    def test_wrong_fixture_step_slot_or_index_rejected(self):
        for field,value in [('fixture',6),('step',1),('slot',1),('index',31)]:
            donors=copy.deepcopy(self.donors);donors[0]['record']['frame'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_overlay('official_both',donors=donors)

    def test_misordered_donor_or_physical_rows_rejected(self):
        donors=list(self.donors);donors[0],donors[1]=donors[1],donors[0]
        with self.assertRaises(ValueError):self.run_overlay('official_key',donors=donors)
        case=copy.deepcopy(self.case);case['physical_frame_order'].reverse()
        with self.assertRaises(ValueError):self.run_overlay('official_key',case=case)

    def test_wrong_rank_mode_or_decode_route_rejected(self):
        for field,value in [('rank',True),('rank',2),('bucket',4096),('seq_ids',[3,1,0,2]),('strict_mask_kind','all_positions')]:
            case=copy.deepcopy(self.case);case[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.run_overlay('own',case=case)
        with self.assertRaises(ValueError):self.run_overlay('unknown')

    def test_missing_donor_or_cache_rejected(self):
        with self.assertRaises(ValueError):self.run_overlay('official_both',donors=self.donors[:3])
        with self.assertRaises(ValueError):self.run_overlay('official_both',own={core.boundary.CACHE_NAMES[0]:self.own[core.boundary.CACHE_NAMES[0]]})

    def test_wrong_native_shape_length_precision_and_claims_rejected(self):
        modifications=[('cache_length',2),('official_bf16',False),('full_logits_raw_exact',False),
                       ('all213_selected_taps_raw_exact',False),('history_unmodified',False),('full_model_qualified',True)]
        for field,value in modifications:
            donors=copy.deepcopy(self.donors);donors[0]['record'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_overlay('official_key',donors=donors)
        for field,value in [('shape',[1,1,3,128]),('dtype','torch.float32')]:
            donors=copy.deepcopy(self.donors);donors[0]['record']['cache_metadata']['key'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_overlay('official_key',donors=donors)

    def test_incomplete_native_or_wrong_digest_rejected(self):
        for update in (False, True):
            donors=copy.deepcopy(self.donors);donors[0]['key_raw']=donors[0]['key_raw'][:-2]
            if update:
                donors[0]['record']['outputs']['key']['sha256']=core.boundary.sha(donors[0]['key_raw'])
                donors[0]['record']['outputs']['key']['raw_bytes']=len(donors[0]['key_raw'])
            with self.assertRaises(ValueError):self.run_overlay('official_key',donors=donors)

    def test_nonfinite_unselected_native_head_rejected_even_with_fresh_hash(self):
        donors=copy.deepcopy(self.donors);donors[0]['key_raw']=b'\x80\x7f'+donors[0]['key_raw'][2:]
        donors[0]['record']['outputs']['key']['sha256']=core.boundary.sha(donors[0]['key_raw'])
        with self.assertRaises(ValueError):self.run_overlay('official_value',donors=donors)

    def test_incomplete_layout_or_nonfinite_original_store_rejected(self):
        case=copy.deepcopy(self.case);own=dict(self.own);name=core.boundary.CACHE_NAMES[0]
        own[name]=b'\xc0\x7f'+own[name][2:];case['caches'][0]['complete_source_metadata']['sha256']=core.boundary.sha(own[name])
        with self.assertRaises(ValueError):self.run_overlay('own',case=case,own=own)
        case=copy.deepcopy(self.case);case['caches'][0]['sdk_view']['stride']=[16384,16384,128,1]
        with self.assertRaises(ValueError):self.run_overlay('official_key',case=case)
        own=dict(self.own);own[name]=own[name][:-2]
        with self.assertRaises(ValueError):self.run_overlay('restore',own=own)

    def test_inputs_remain_immutable_and_proof_keeps_real_sources(self):
        case=copy.deepcopy(self.case);donors=copy.deepcopy(self.donors);oldcase=copy.deepcopy(case);olddonors=copy.deepcopy(donors)
        _,proof=self.run_overlay('official_both',case=case,donors=donors)
        self.assertEqual(case,oldcase);self.assertEqual(donors,olddonors)
        for slot,d in enumerate(donors):
            self.assertEqual(proof['donors'][slot]['previous_native_frame'],d['record']['frame'])
            self.assertEqual(proof['donors'][slot]['caches']['key']['native_output'],d['record']['outputs']['key'])
        proof['donors'][0]['previous_native_frame']['index']=99
        self.assertEqual(donors,olddonors)

    def test_guard_denial_precedes_any_torch_or_model_access(self):
        class Denied(RuntimeError):pass
        def deny():raise Denied('No runtime admission')
        with self.assertRaises(Denied):native.replay(None,None,None,None,None,None,None,None,None,deny)

    def test_original_attention_and_cache_getter_invocation_ast_unchanged(self):
        def expressions(path,name):
            tree=ast.parse(path.read_text())
            return [ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.Call)
                    and ((isinstance(n.func,ast.Name)and n.func.id==name)or
                         (isinstance(n.func,ast.Attribute)and n.func.attr==name))]
        original=Q/'minicpm5_layer1_kv_factorial_callable_v1.py';current=Q/'minicpm5_layer1_official_history_callable_v1.py'
        for name in ('attention','get_kv_by_layer_id','rope'):
            self.assertEqual(len(expressions(original,name)),1)
            self.assertEqual(expressions(original,name),expressions(current,name))

    def test_no_runtime_or_native_dependency_imports(self):
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'} & set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
