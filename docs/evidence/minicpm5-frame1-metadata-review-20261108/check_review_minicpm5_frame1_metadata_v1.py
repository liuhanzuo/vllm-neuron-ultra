"""Actual historical metadata and explicitly synthetic future-result controls."""
import copy
import hashlib
import io
import types
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q))
import review_minicpm5_frame1_metadata_v1 as a
R=a.R

def reference(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def metric(count,changed=False):
    return dict(raw_exact=not changed,different_elements=1 if changed else 0,compared_elements=count,nonfinite_elements=0,max_abs=.01 if changed else 0.,mean_abs=.001 if changed else 0.,cosine=.99 if changed else 1.)

def scopes(width,changed=False,full=True):
    lengths={'full':2048,'real_tokens':1231,'prediction':1}if full else{'real_tokens':1231,'prediction':1}
    return {name:metric(length*width,changed)for name,length in lengths.items()}

def synthetic_fixture():
    reader=a.Reader();plan=reader.load(a.BASE)
    # Reuse actual historical schema and constructor/weight metadata. All future
    # capture tensor hashes and comparison values below are synthetic controls.
    ended=reader.load(dict(path=str(Q/'minicpm5-frame17-mlp-root-exit-r2-20261107-v1.json'),sha256='7fb95b7a8aea24f65bfe8bd499da4b3ec6c4af43e2f68c845c907d90595f89b5'))
    oldreplay=reader.load(ended['report']);oldcaps={role:reader.load(ref)for role,ref in oldreplay['captures'].items()}
    native=reader.load(plan['hf_frame_records'][1]);root=R/'minicpm5-frame1-mlp-r999-20261108-v99'
    prep=json.loads((Q/'minicpm5-frame1-v2-actual-source-history-probe-20261108-v1.json').read_text())
    spec=dict(science=prep['science'],output_root=str(root),new_sources=prep['new_source_sha256'])
    captures={};controls={};refs={}
    for role in a.ROLES:
        cap=copy.deepcopy(oldcaps[role]);isnative=role=='native';rank=-1 if isnative else int(role[-1]);length=1231 if isnative else 2048
        old=native if isnative else reader.load(a.CPU[rank]);full=None if isnative else reader.load(a.FULL_CPU[rank])
        cap.update(schema='minicpm5_frame1_mlp_capture_v2',frame=copy.deepcopy(a.FRAME),precision_interventions=[],oproj_witnesses=[],cpu_full_anchor=None if isnative else dict(reference=a.FULL_CPU[rank],all_four_fullshape_exact=True,fields=a.FULL))
        cap['anchor']['hashes']={name:old['tap_raw_sha256'if isnative else'stage_sha256'][name]for name in a.TAPS}
        if isnative:
            for name in a.TAPS:
                m=old['metadata'][name];cap['anchor']['source_metadata'][name]=dict(shape=m['source_shape'],stride=m['source_stride'],dtype=m['dtype'],prediction_positions=[m['local_prediction_position']])
        else:cap['anchor']['source_metadata']={name:copy.deepcopy(old['stage_metadata'][name])for name in a.TAPS}
        cc={}
        for row in cap['captures']:
            name=row['name'];width=2048 if name in ('input','output')else(6144 if isnative else 3072)
            meta=dict(shape=[1,length,width],stride=[length*width,width,1],offset=0,dtype='torch.bfloat16',sha256=hashlib.sha256((role+name).encode()).hexdigest())
            if not isnative and name in a.FULL:meta=copy.deepcopy(full['layer0_full_observations'][a.FULL[name]])
            if not isnative and name=='output':meta['sha256']='b'*64
            row.update(path=str(root/role/(name+'.bf16')),metadata=meta,sha256=meta['sha256'],bytes=2*length*width)
            row['identity'].update(realpath=row['path'],size=row['bytes']);cc[name]=dict(raw_exact=True,source_layout_exact=True,sha256=row['sha256'])
        captures[role]=cap;controls[role]=cc;refs[role]=dict(path=str(root/role/'capture.json'),sha256='a'*64)
    replay=dict(schema='minicpm5_frame1_mlp_replay_v2',complete=True,frame=a.FRAME,captures=refs,
        controls=dict(native=controls['native'],cpu=[controls['cpu-rank0'],controls['cpu-rank1']],all15_fullshape_exact=True),
        weight_proof=oldreplay['weight_proof'],fixed_input=captures['cpu-rank0']['captures'][0]['metadata'],
        front_half={name:scopes(6144)for name in ('gate','up','down_input')},
        contrasts=dict(S_vs_F=scopes(2048,True),D_vs_F=scopes(2048),D_vs_S=scopes(2048,True)),
        observed_native_vs_cpu_boundaries={name:scopes(2048 if name in ('input','output')else 6144,full=False)for name in a.FIELDS},
        classification='down_path_sufficient_for_this_fixed_input_not_final_KL_causality',original_remaining_gates=plan['remaining_gates'],
        native_own1231_control_threads=2,fixed_input2048_contrast_threads=1,
        **{key:False for key in ('actual_gloo_reexecuted_in_replay','native_interpreter_reused_in_replay','whole_native_input_equality_previously_known','unique_final_KL_cause_proven','full_model_qualified','new_speedup_claim')})
    replay['observed_native_vs_cpu_input']=replay['observed_native_vs_cpu_boundaries']['input']['real_tokens']
    return plan,spec,replay,captures,reader.load

class MetadataReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=synthetic_fixture()
    def fresh(self):return copy.deepcopy(self.fixture[:4])+ (self.fixture[4],)
    def run_case(self,change=None):
        values=list(self.fresh())
        if change:change(*values[:4])
        return a.audit(*values)
    def reject(self,change):
        with self.assertRaises((ValueError,KeyError)):self.run_case(change)
    def test_synthetic_complete_future_is_only_metadata_consistency(self):
        result=self.run_case();self.assertEqual(result['capture_bytes'],164515840);self.assertEqual(result['capture_count'],15)
        for key in ('raw_payload_recomputed','actual_exit_verified','physical_drain_verified','full_model_qualified','new_speedup_claim'):self.assertIs(result[key],False)
    def test_actual_four_strict_baselines(self):
        reader=a.Reader()
        for full,refs in ((False,a.CPU),(True,a.FULL_CPU)):
            for rank,ref in enumerate(refs):a.baseline(reader.load(ref),rank,full)
    def test_actual_prior_frame17_metric_arithmetic(self):
        reader=a.Reader();end=reader.load(dict(path=str(Q/'minicpm5-frame17-mlp-root-exit-r2-20261107-v1.json'),sha256='7fb95b7a8aea24f65bfe8bd499da4b3ec6c4af43e2f68c845c907d90595f89b5'));replay=reader.load(end['report'])
        for rows in replay['contrasts'].values():
            for name,row in rows.items():a.metric(row,{'full':128,'real_tokens':28,'prediction':1}[name]*2048)
    def test_frame17_is_not_accepted_as_frame1(self):self.reject(lambda p,s,r,c:r.update(frame=dict(index=17,fixture=1,slot=1,step=0)))
    def test_reject_other_prompt_length(self):self.reject(lambda p,s,r,c:p['fixtures'][7]['input_ids'].pop())
    def test_reject_smaller_capture(self):self.reject(lambda p,s,r,c:c['cpu-rank0']['captures'][0].update(bytes=2048))
    def test_reject_layout_change(self):self.reject(lambda p,s,r,c:c['cpu-rank0']['captures'][0]['metadata'].update(stride=[4194304,2048,1]))
    def test_reject_cpu_full_anchor_hash_change(self):self.reject(lambda p,s,r,c:c['cpu-rank0']['captures'][1]['metadata'].update(sha256='c'*64))
    def test_reject_missing_role(self):self.reject(lambda p,s,r,c:c.pop('cpu-rank1'))
    def test_reject_missing_capture_field(self):self.reject(lambda p,s,r,c:c['native']['captures'].pop())
    def test_reject_missing_control(self):self.reject(lambda p,s,r,c:r['controls']['cpu'][0].pop('gate'))
    def test_reject_prediction_anchor_change(self):self.reject(lambda p,s,r,c:c['native']['anchor']['hashes'].update({'layers.0':'d'*64}))
    def test_reject_precision_intervention(self):self.reject(lambda p,s,r,c:c['cpu-rank0'].update(precision_interventions=['full_down']))
    def test_reject_recomputed_capture(self):self.reject(lambda p,s,r,c:c['native']['captures'][0].update(observed_not_recomputed=False))
    def test_reject_capture_path_escape(self):self.reject(lambda p,s,r,c:c['native']['captures'][0].update(path='/tmp/input.bf16'))
    def test_reject_claim_of_later_layers(self):self.reject(lambda p,s,r,c:c['native'].update(downstream_layers_executed=1))
    def test_reject_claim_of_history_consumption(self):self.reject(lambda p,s,r,c:c['cpu-rank0'].update(history_consumed=True))
    def test_reject_replica_output_change(self):
        def change(p,s,r,c):
            row=c['cpu-rank1']['captures'][-1];row['metadata']['sha256']=row['sha256']='f'*64;r['controls']['cpu'][1]['output']['sha256']='f'*64
        self.reject(change)
    def test_reject_missing_scope(self):self.reject(lambda p,s,r,c:r['contrasts']['S_vs_F'].pop('real_tokens'))
    def test_reject_scope_shrink(self):self.reject(lambda p,s,r,c:r['contrasts']['S_vs_F']['full'].update(compared_elements=1231*2048))
    def test_reject_bitwise_flag_inconsistency(self):self.reject(lambda p,s,r,c:r['contrasts']['S_vs_F']['full'].update(raw_exact=True))
    def test_reject_nonfinite_metric(self):self.reject(lambda p,s,r,c:r['contrasts']['S_vs_F']['full'].update(max_abs=float('nan')))
    def test_reject_missing_observed_boundary(self):self.reject(lambda p,s,r,c:r['observed_native_vs_cpu_boundaries'].pop('gate'))
    def test_reject_remaining_gate_omission(self):self.reject(lambda p,s,r,c:r.update(original_remaining_gates={}))
    def test_reject_unearned_full_model_claim(self):self.reject(lambda p,s,r,c:r.update(full_model_qualified=True))
    def test_reject_final_kl_causality_claim(self):self.reject(lambda p,s,r,c:r.update(unique_final_KL_cause_proven=True))
    def test_classification_front_half_different(self):
        def change(p,s,r,c):r['front_half']['down_input']=scopes(6144,True);r['contrasts']['D_vs_F']=scopes(2048,True);r['classification']='front_half_differs_down_is_not_established_as_unique_cause'
        self.assertEqual(self.run_case(change)['classification'],'front_half_differs_down_is_not_established_as_unique_cause')
    def test_classification_no_output_difference(self):
        def change(p,s,r,c):r['contrasts']['S_vs_F']=scopes(2048);r['contrasts']['D_vs_S']=scopes(2048);r['classification']='no_output_difference_for_this_fixed_input'
        self.assertEqual(self.run_case(change)['classification'],'no_output_difference_for_this_fixed_input')
    def test_reject_same_product_contradictory_down(self):self.reject(lambda p,s,r,c:r['contrasts'].update(D_vs_F=scopes(2048,True)))
    def test_reject_inconsistent_symmetric_distance(self):self.reject(lambda p,s,r,c:r['contrasts'].update(D_vs_S=scopes(2048)))
    def test_signed_zero_is_not_raw_equal(self):
        row=metric(2,True);row.update(max_abs=0.,mean_abs=0.,cosine=None);a.metric(row,2)
    def test_payload_rejected_before_open(self):
        with patch.object(a.os,'open')as opened:
            with self.assertRaisesRegex(ValueError,'Only experiment JSON'):a.Reader().load(dict(path=str(R/'synthetic.bf16'),sha256='a'*64))
            opened.assert_not_called()
    def test_real_metadata_wrong_hash_rejected(self):
        with self.assertRaisesRegex(ValueError,'Metadata digest'):a.Reader().load(dict(a.BASE,sha256='a'*64))
    def test_no_native_imported(self):self.assertFalse({'torch','transformers','numpy','neuronx_distributed'}&set(sys.modules))


class CommandBindings(unittest.TestCase):
    def execute_synthetic(self,mutation=None):
        plan,spec,replay,captures,original_load=synthetic_fixture();root=Path(spec['output_root'])
        spec.update(schema='minicpm5_frame1_mlp_spec_v1',base_plan=a.BASE,current_catalog=dict(path='synthetic-complete-catalog',sha256='c'*64),required_sio14_catalog=dict(path='synthetic-smol-catalog',sha256='d'*64),added_json={})
        for reference in (*a.CPU,*a.FULL_CPU):spec['added_json'][reference['path']]=dict(sha256=reference['sha256'],identity={'synthetic':True})
        expanded=copy.deepcopy(plan);expanded['output_root']=str(root);expanded['source_sha256'].update(spec['new_sources']);expanded['inputs'].update(spec['added_json']);expanded['frame1_mlp']=spec['science']
        spec_ref=dict(path=str(Q/'synthetic-never-written-spec.json'),sha256='1'*64)
        review_ref=dict(path=str(Q/'synthetic-never-written-review.json'),sha256='2'*64)
        admission_ref=dict(path=str(root/'admission.json'),sha256='3'*64)
        replay_ref=dict(path=str(root/'replay.json'),sha256='4'*64)
        result_ref=dict(path=str(root/'result.json'),sha256='5'*64)
        owner=dict(pid=123,start_ticks='456',argv=['synthetic'])
        review=dict(spec=spec_ref,reviewer='/root',passed=True,independent_review=True,source_only=True,blockers=[])
        admission=dict(spec=spec_ref,review=review_ref,outer_owner=owner,current_catalog=spec['current_catalog'],required_sio14_catalog=spec['required_sio14_catalog'],current_history_drained=True,plan_digest=a.hash_value(expanded))
        result=dict(complete=True,self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False,owner=owner,admission=admission_ref,report=replay_ref)
        values={spec_ref['path']:spec,review_ref['path']:review,admission_ref['path']:admission,replay_ref['path']:replay,result_ref['path']:result}
        values.update({reference['path']:captures[role]for role,reference in replay['captures'].items()})
        args=types.SimpleNamespace(spec=spec_ref['path'],spec_sha256=spec_ref['sha256'],result=result_ref['path'],result_sha256=result_ref['sha256'],output=str(Q/'synthetic-never-written-output.json'))
        if mutation:mutation(spec,admission,result,review,replay,args)
        class FakeReader:
            def __init__(self):self.references={}
            def load(self,reference):return values[reference['path']]if reference['path']in values else original_load(reference)
        class Sink(io.StringIO):
            def close(self):pass
        sink=Sink();real_open=Path.open
        def opened(path,*args,**kwargs):
            if str(path)==str(Q/'synthetic-never-written-output.json'):
                self.assertEqual(args,('x',));return sink
            return real_open(path,*args,**kwargs)
        with patch.object(a,'Reader',FakeReader),patch.object(Path,'open',opened),redirect_stdout(io.StringIO()):a.main(args)
        return json.loads(sink.getvalue())
    def reject(self,mutation):
        with self.assertRaises((ValueError,KeyError)):self.execute_synthetic(mutation)
    def test_main_synthetic_chain_reports_only_metadata(self):
        result=self.execute_synthetic();self.assertTrue(result['checks']['metadata_consistent']);self.assertFalse(result['checks']['actual_exit_verified']);self.assertFalse(result['checks']['full_model_qualified'])
    def test_reject_admission_spec_mismatch(self):self.reject(lambda s,a,r,v,p,args:a.update(spec=dict(path='wrong',sha256='a'*64)))
    def test_reject_result_owner_mismatch(self):self.reject(lambda s,a,r,v,p,args:r.update(owner=dict(pid=999)))
    def test_reject_plan_digest_mismatch(self):self.reject(lambda s,a,r,v,p,args:a.update(plan_digest='a'*64))
    def test_reject_self_exit_claim(self):self.reject(lambda s,a,r,v,p,args:r.update(self_exit_claimed=True))
    def test_reject_prior_catalog_mismatch(self):self.reject(lambda s,a,r,v,p,args:a.update(current_catalog=dict(path='other',sha256='b'*64)))
    def test_reject_missing_original_anchor_addition(self):self.reject(lambda s,a,r,v,p,args:s['added_json'].pop(next(iter(s['added_json']))))
    def test_reject_source_review_blocker(self):self.reject(lambda s,a,r,v,p,args:v.update(blockers=['unresolved']))
    def test_reject_unbound_replay_path(self):self.reject(lambda s,a,r,v,p,args:r.update(report=dict(path='/tmp/replay.json',sha256='a'*64)))
    def test_reject_capture_path_reference(self):self.reject(lambda s,a,r,v,p,args:p['captures']['native'].update(path='/tmp/capture.json'))

if __name__=='__main__':unittest.main(verbosity=2)
