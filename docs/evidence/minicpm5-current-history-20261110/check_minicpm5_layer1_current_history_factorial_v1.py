"""Small synthetic current/history factorial controls, with no native imports."""
import ast
import copy
from pathlib import Path
import struct
import sys
import unittest
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q.parent))
from research import minicpm5_layer1_current_history_factorial_v1 as core
from research import minicpm5_layer1_current_history_factorial_callable_v1 as native
from research import check_minicpm5_layer1_official_history_core_v1 as fixture


class CurrentHistoryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.OfficialHistoryChecks.setUpClass();base=fixture.OfficialHistoryChecks
        cls.case=copy.deepcopy(base.case);cls.own=base.own;cls.prior=base.donors
        cls.normalized=b''.join(struct.pack('<H',0x3f80+slot)*2048 for slot in range(4))
        cls.case['normalized_input']=dict(name=core.boundary.NORMALIZED,metadata=copy.deepcopy(core.boundary.NORM_META),
            logical_bytes=16384,restore_source_stride=True,logical_sha256=core.boundary.sha(cls.normalized))
        cls.current=[]
        for slot,(frame,prior)in enumerate(zip(cls.case['physical_frame_order'],cls.prior)):
            raw=struct.pack('<H',0x4000+slot)*2048;row=copy.deepcopy(prior['record'])
            row.update(frame=copy.deepcopy(frame),step=frame['step'],cache_length=cls.case['positions'][slot]+1,
                original_frame=dict(path='/native/current-%d.json'%frame['index'],sha256='d'*64))
            row['outputs']['taps']=dict(path='/native/frame-%02d/selected-taps.bf16'%frame['index'],raw_bytes=8192,sha256='e'*64)
            row['tap_entries']=[dict(name=core.boundary.NORMALIZED,offset=0,bytes=4096,shape=[1,1,2048],sha256=core.boundary.sha(raw))]
            row['tap_metadata']={core.boundary.NORMALIZED:dict(source_shape=[1,1,2048],source_stride=[2048,2048,1],
                local_prediction_position=0,logical_prediction_position=cls.case['positions'][slot],dtype='torch.bfloat16')}
            cls.current.append(dict(record=row,normalized_raw=raw))

    def select(self,mode,case=None,normalized=None,prior=None,current=None):
        return core.select(self.case if case is None else case,self.normalized if normalized is None else normalized,
            self.own,self.prior if prior is None else prior,self.current if current is None else current,mode)

    def test_exact_2by2_sources_plus_restore(self):
        for mode,wantcurrent,wanthistory in [('own',False,False),('official_current',True,False),
                ('official_history',False,True),('official_both',True,True),('restore',False,False)]:
            selected,packed,cache,proof=self.select(mode)
            self.assertEqual(selected,b''.join(v['normalized_raw']for v in self.current)if wantcurrent else self.normalized)
            expected,_=core.history.overlay(self.case,self.own,self.prior,'official_both'if wanthistory else 'own')
            self.assertEqual(cache,expected);self.assertEqual(proof['current_qkv_change_together'],wantcurrent)
            self.assertEqual(proof['current_source'],'official'if wantcurrent else 'own')
            self.assertEqual(proof['history_source'],'official'if wanthistory else 'own')
            self.assertEqual(proof['own_control'],mode in ('own','restore'))

    def test_official_rows_preserve_original_physical_stride(self):
        selected,packed,_,proof=self.select('official_current')
        for channel in (0,1,1023,2047):
            for slot in range(4):
                self.assertEqual(packed[2*(slot+4*channel):2*(slot+4*channel)+2],struct.pack('<H',0x4000+slot))
        self.assertEqual(proof['selected_layout']['stride'],[1,4,4])
        self.assertEqual(core.boundary.sha(packed),proof['selected_storage_sha256'])

    def test_original_input_and_source_case_cannot_be_relabelled(self):
        case=copy.deepcopy(self.case);case['normalized_input']['logical_sha256']='0'*64
        with self.assertRaises(ValueError):self.select('official_both',case=case)
        case=copy.deepcopy(self.case);case['normalized_input']['metadata']['stride']=[2048,2048,1]
        with self.assertRaises(ValueError):self.select('official_current',case=case)
        with self.assertRaises(ValueError):self.select('official_history',normalized=self.normalized[:-2])

    def test_wrong_current_fixture_step_slot_and_index_rejected(self):
        for field,value in [('fixture',1),('step',0),('slot',1),('index',2)]:
            current=copy.deepcopy(self.current);current[0]['record']['frame'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.select('official_current',current=current)

    def test_previous_and_current_are_not_interchangeable(self):
        current=copy.deepcopy(self.current);current[0]['record']['cache_length']-=1
        with self.assertRaises(ValueError):self.select('official_both',current=current)
        prior=copy.deepcopy(self.prior);prior[0]['record']['frame']['step']=1
        with self.assertRaises(ValueError):self.select('official_both',prior=prior)

    def test_wrong_layer_stage_or_position_precision_rejected(self):
        for field,value in [('logical_prediction_position',0),('dtype','torch.float32'),('source_shape',[1,2,2048])]:
            current=copy.deepcopy(self.current);current[0]['record']['tap_metadata'][core.boundary.NORMALIZED][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.select('official_current',current=current)
        current=copy.deepcopy(self.current);current[0]['record']['tap_entries'][0]['name']='layers.0.input_layernorm'
        with self.assertRaises(ValueError):self.select('official_current',current=current)

    def test_wrong_tap_extent_duplicate_or_digest_rejected(self):
        for field,value in [('offset',8192),('bytes',4094),('shape',[1,1,1024]),('sha256','0'*64)]:
            current=copy.deepcopy(self.current);current[0]['record']['tap_entries'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.select('own',current=current)
        current=copy.deepcopy(self.current);current[0]['record']['tap_entries']*=2
        with self.assertRaises(ValueError):self.select('restore',current=current)

    def test_nonfinite_current_rejected_even_after_rehash(self):
        current=copy.deepcopy(self.current);current[0]['normalized_raw']=b'\x80\x7f'+current[0]['normalized_raw'][2:]
        current[0]['record']['tap_entries'][0]['sha256']=core.boundary.sha(current[0]['normalized_raw'])
        with self.assertRaises(ValueError):self.select('official_current',current=current)
        case=copy.deepcopy(self.case);own=b'\xc0\x7f'+self.normalized[2:]
        case['normalized_input']['logical_sha256']=core.boundary.sha(own)
        with self.assertRaises(ValueError):self.select('official_both',case=case,normalized=own)

    def test_reference_mismatch_and_qualification_claim_rejected(self):
        current=copy.deepcopy(self.current);current[0]['record']['original_reference']['sha256']='f'*64
        with self.assertRaises(ValueError):self.select('official_both',current=current)
        for key in ('full_model_qualified','new_speedup_claim'):
            current=copy.deepcopy(self.current);current[0]['record'][key]=True
            with self.assertRaises(ValueError):self.select('own',current=current)

    def test_missing_or_reordered_current_rows_rejected(self):
        with self.assertRaises(ValueError):self.select('official_current',current=self.current[:-1])
        current=list(reversed(self.current))
        with self.assertRaises(ValueError):self.select('official_both',current=current)
        with self.assertRaises(ValueError):self.select('query_only')

    def test_for_call_selects_current_normalized_and_previous_history(self):
        plan=dict(frames=self.case['physical_frame_order'],calls=[{}for _ in range(14)])
        plan['calls'][4]=self.case['call'];data={}
        for frame,previous,current in zip(plan['frames'],self.prior,self.current):
            data[frame['fixture'],0]=previous
            data[frame['fixture'],1]=dict(record=current['record'],targets={core.boundary.NORMALIZED:current['normalized_raw'],
                core.boundary.ATTENTION:b'\x00\x3f'*2048})
        prior,current,target,refs=core.for_call(plan,data,self.case)
        self.assertEqual(prior,self.prior);self.assertEqual(current,self.current);self.assertEqual(target,b'\x00\x3f'*8192)
        self.assertEqual([x['frame']for x in refs],plan['frames'])

    def test_select_does_not_mutate_source_case_or_donors(self):
        case=copy.deepcopy(self.case);current=copy.deepcopy(self.current);before=copy.deepcopy((case,current))
        _,_,_,proof=self.select('official_both',case=case,current=current)
        proof['original_normalized_input']['logical_sha256']='mutated'
        proof['official_current_sources'][0]['frame']['index']=99
        self.assertEqual((case,current),before)

    def test_guard_denies_before_native_access(self):
        def deny():raise RuntimeError('No admission')
        with self.assertRaises(RuntimeError):native.replay(None,None,None,None,None,None,None,None,None,None,deny)

    def test_attention_cache_and_rope_call_asts_unchanged(self):
        def nodes(path,name):
            return [ast.dump(v,include_attributes=False)for v in ast.walk(ast.parse(path.read_text()))
                if isinstance(v,ast.Call)and ((isinstance(v.func,ast.Name)and v.func.id==name)or
                (isinstance(v.func,ast.Attribute)and v.func.attr==name))]
        old=Q/'minicpm5_layer1_official_history_callable_v1.py';new=Q/'minicpm5_layer1_current_history_factorial_callable_v1.py'
        for name in ('attention','get_kv_by_layer_id','rope'):
            self.assertEqual(len(nodes(old,name)),1);self.assertEqual(nodes(old,name),nodes(new,name))

    def test_imports_remain_stdlib_only(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
