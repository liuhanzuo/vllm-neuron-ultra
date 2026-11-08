"""Source and pinned metadata controls only; deliberately no torch import."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
sys.path.insert(0,str(ROOT))
import minicpm5_frame1_mlp_worker_v1 as w
import minicpm5_frame1_mlp_replay_v1 as replay


def read(ref):
    data=Path(ref['path']).read_bytes()
    assert len(data)<2*w.MIB and hashlib.sha256(data).hexdigest()==ref['sha256']
    return json.loads(data)


def core():
    class Module:
        def forward(self,*args): return args
    layer=Module();layer.self_attn=Module();layer.self_attn.o_proj=Module();layer.mlp=Module()
    for name in ('gate','up','down'):setattr(layer.mlp,name+'_proj',Module())
    return types.SimpleNamespace(layers=[layer])


class SourceControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((ROOT/'research/minicpm5-layer0-phase-cpu-input-plan-20261107-v1.json').read_text())
        cls.plan['frame1_mlp']=w.science()
        for ref in (*w.CPU_ANCHORS,*w.FULL_CPU_ANCHORS):
            read(ref);cls.plan['inputs'][ref['path']]=dict(sha256=ref['sha256'])
        cls.pred=[read(ref) for ref in w.CPU_ANCHORS]
        cls.full=[read(ref) for ref in w.FULL_CPU_ANCHORS]

    def test_no_model_imports(self):
        self.assertFalse({'torch','torch_neuronx','transformers','neuronx_distributed'} & set(sys.modules))

    def test_real_science_metadata(self):
        self.assertEqual(w.validate_science(self.plan),w.science())

    def test_reject_wrong_science(self):
        for path,value in [(('frame1_mlp','arm'),'strict_layer0_oproj_fp32_prefill_only'),
                           (('frame1_mlp','native_length'),1230),(('frame1_mlp','cpu_bucket'),128),
                           (('frame1_mlp','precision_interventions'),['down']),
                           (('frame1_mlp','actual_capture_bytes'),32*w.MIB)]:
            with self.subTest(path=path):
                p=copy.deepcopy(self.plan);p[path[0]][path[1]]=value
                with self.assertRaises(RuntimeError):w.validate_science(p)

    def test_reject_changed_prompt_or_call(self):
        for key in ('prompt','position','frame','bucket','anchor'):
            with self.subTest(key=key):
                p=copy.deepcopy(self.plan)
                if key=='prompt':p['fixtures'][7]['input_ids'].pop()
                if key=='position':p['graph_calls'][1]['call']['logical_positions']=[1231]
                if key=='frame':p['graph_calls'][1]['call']['frames'][0]['index']=17
                if key=='bucket':p['graph_calls'][1]['bucket']=128
                if key=='anchor':p['inputs'][w.CPU_ANCHORS[0]['path']]['sha256']='0'*64
                with self.assertRaises(RuntimeError):w.validate_science(p)

    def test_science_does_not_alias_pins(self):
        a=w.science();a['cpu_anchors'][0]['sha256']='0'*64
        self.assertNotEqual(w.science()['cpu_anchors'][0]['sha256'],'0'*64)

    def test_all_four_real_baselines(self):
        for full,records in ((False,self.pred),(True,self.full)):
            for rank,record in enumerate(records):
                with self.subTest(full=full,rank=rank):self.assertTrue(w.validate_baseline(record,rank,full))

    def test_baseline_rejects_candidates_and_incomplete_records(self):
        for field,value in [('arm','strict_layer0_oproj_fp32_prefill_only'),('candidate_targets',['down']),
                            ('candidate_layer0_phase_enabled',True),('changed_mask_cells',[1]),
                            ('rank',1),('complete',False),('bucket',128)]:
            with self.subTest(field=field):
                record=copy.deepcopy(self.pred[0]);record[field]=value
                with self.assertRaises(RuntimeError):w.validate_baseline(record,0,False)
        for flag in self.pred[0]['strict_anchor_exact']:
            record=copy.deepcopy(self.pred[0]);record['strict_anchor_exact'][flag]=False
            with self.assertRaises(RuntimeError):w.validate_baseline(record,0,False)
        record=copy.deepcopy(self.full[0]);record['candidate_full_down_enabled']=True
        with self.assertRaises(RuntimeError):w.validate_baseline(record,0,True)

    def test_whole_cpu_boundary_anchors(self):
        for rank in (0,1):
            context=types.SimpleNamespace(read_json=lambda path,wanted:read(dict(path=path,sha256=wanted)))
            values={key:dict(metadata=self.full[rank]['layer0_full_observations'][name]) for key,name in w.FULL_FIELDS.items()}
            result=w.check_full_anchor(context,values,self.pred[rank],rank)
            self.assertTrue(result['all_four_fullshape_exact'])

    def test_changed_full_bytes_and_layout_rejected(self):
        context=types.SimpleNamespace(read_json=lambda path,wanted:read(dict(path=path,sha256=wanted)))
        for field in w.FULL_FIELDS:
            for changed in ('sha256','stride'):
                with self.subTest(field=field,changed=changed):
                    values={key:dict(metadata=copy.deepcopy(self.full[0]['layer0_full_observations'][name])) for key,name in w.FULL_FIELDS.items()}
                    values[field]['metadata'][changed]='0'*64 if changed=='sha256' else [4194304,2048,1]
                    with self.assertRaises(RuntimeError):w.check_full_anchor(context,values,self.pred[0],0)

    def test_cross_rank_anchor_substitution_rejected(self):
        context=types.SimpleNamespace(read_json=lambda *args:copy.deepcopy(self.full[1]))
        values={key:dict(metadata=self.full[0]['layer0_full_observations'][name]) for key,name in w.FULL_FIELDS.items()}
        with self.assertRaises(RuntimeError):w.check_full_anchor(context,values,self.pred[0],0)

    def test_real_dense_layouts_and_budget(self):
        for rank in (0,1):
            for m in self.full[rank]['layer0_full_observations'].values():
                self.assertEqual(w.checked_layout(m['shape'],m['stride'],m['offset']),m['shape'][1]*m['shape'][2])
        self.assertEqual(w.capture_bytes(True),55463936)
        self.assertEqual(w.capture_bytes(False),54525952)
        self.assertEqual(w.science()['actual_capture_bytes'],164515840)
        self.assertLess(164515840,w.OUTPUT_CAP)
        for native in (True,False):
            for shape in w.expected_shapes(native).values():
                self.assertLessEqual(shape[1]*shape[2]*2,w.CAPTURE_FILE_CAP)
                self.assertEqual(w.checked_layout(shape,[shape[1]*shape[2],shape[2],1],0),shape[1]*shape[2])

    def test_layout_rejects_row_only_overlap_padding_and_offset(self):
        for shape,stride,offset in [([1,1,2048],[2048,2048,1],0),([1,28,2048],[57344,2048,1],0),
                                   ([1,2048,2048],[1,1,1],0),([1,2048,2048],[1,1,2049],0),
                                   ([1,2048,2048],[1,1,2048],1),([1,2048,2048],[1,0,2048],0)]:
            with self.subTest(shape=shape,stride=stride,offset=offset):
                with self.assertRaises(RuntimeError):w.checked_layout(shape,stride,offset)

    def test_original_dispatch_unchanged(self):
        c=core();before=w.original_dispatch(c);w.check_dispatch(c,before)
        self.assertEqual(len(before),7)

    def test_precision_patch_rejected(self):
        for index in range(7):
            c=core();before=w.original_dispatch(c);before[index][0].forward=lambda *a:a
            with self.assertRaises(RuntimeError):w.check_dispatch(c,before)

    def test_replaced_projection_rejected(self):
        c=core();before=w.original_dispatch(c);c.layers[0].mlp.down_proj=core().layers[0].mlp.down_proj
        with self.assertRaises(RuntimeError):w.check_dispatch(c,before)

    def test_causal_classification_limits(self):
        self.assertEqual(replay.classify(True,False,True),'down_path_sufficient_for_this_fixed_input_not_final_KL_causality')
        self.assertEqual(replay.classify(False,False,False),'front_half_differs_down_is_not_established_as_unique_cause')
        self.assertEqual(replay.classify(True,True,True),'no_output_difference_for_this_fixed_input')
        with self.assertRaises(RuntimeError):replay.classify(True,False,False)

    def test_no_precision_patch_constructor_in_capture(self):
        tree=ast.parse(Path(w.__file__).read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('OProjPatch',calls)
        self.assertNotIn('float',calls)

    def test_original_activation_sources(self):
        for path in w.ACTIVATIONS.values():self.assertTrue(w.activation_source(Path(path).read_text()))

    def test_cpu_history_routes_rejected(self):
        valid=dict(past_key_value=None,get_kv_per_layer=False,update_kv_per_layer=False,is_for_context_encoding=True,idx=0)
        w.cpu_route(valid)
        for name,value in [('past_key_value',object()),('get_kv_per_layer',True),('update_kv_per_layer',True),
                           ('is_for_context_encoding',False),('idx',1),('windowed_context_encoding_window_idx',1)]:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):w.cpu_route(dict(valid,**{name:value}))


if __name__=='__main__':unittest.main()
