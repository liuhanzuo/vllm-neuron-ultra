"""Metadata, mock-stream and lifecycle controls; no model/raw-payload reads."""
import ast
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_after_native_cache_v1 as completed
from research import minicpm5_layer1_official_history_inputs_v1 as inputs
from research import minicpm5_layer1_official_history_results_v1 as results
from research import minicpm5_layer1_official_history_v1 as contract
import minicpm5_layer1_official_history_cpu_runner_v1 as runner
import minicpm5_layer1_official_history_cpu_worker_v1 as worker


def sha(data):return hashlib.sha256(data).hexdigest()


class OfficialIntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h=json.loads((Q/'continuation-handoff-20261110-v416-mini-native-cache-complete.json').read_text())['native_cache']
        cls.binding=dict(schema='minicpm5_layer1_completed_native_cache_v1',model_exit=h['actual_exit'],
            audit_exit=h['independent_audit_exit'],comparison_exit=h['comparison_exit'])
        cls.metadata={}
        def reader(ref):
            value=completed.p.read_ref(ref);cls.metadata[ref['path']]=value;return value
        cls.bundle=completed.completed(cls.binding,reader)
        cls.plan=dict(output_root='/synthetic/official-history',inputs={})
        pairs=(('completed_layer1_data','layer1_own_history_replay'),('completed_own_history','layer1_cross_history'),
            ('completed_cross_history','layer1_kv_factorial'),('completed_factorial','layer1_native_cache'),
            ('completed_native_cache','layer1_official_history'))
        for _,key in pairs:cls.plan[key]={'synthetic':key}
        cls.result=[];cls.items={};cls.data={};cls.old={}
        for rank in (0,1):
            items=[];rows=[]
            for arm in results.stream.raw.ARMS:
                for ci in results.stream.boundary.DECODE_CALLS:
                    frames=[dict(index=4+s,fixture=s,slot=s,step=1)for s in range(4)]
                    case=dict(rank=rank,arm=arm,call=dict(index=ci,phase='decode'),bucket=128,physical_frame_order=frames)
                    own=bytes(16384);items.append(dict(case=case,cache_raw={},expected_attention_raw=own,audited_call=dict(payload_sha256='a'*64)))
                    target=b'\x80\x3f'*8192;refs=[dict(frame=f,sha='b'*64)for f in frames]
                    for mode in contract.MODES:
                        raw=own if mode in ('own','restore')else b'\x00\x3f'*8192
                        path=str(Path(cls.plan['output_root'])/('cpu-rank%d'%rank)/arm/('call-%02d'%ci)/mode/'attention.bf16')
                        cls.data[path]=raw
                        actual=dict(schema='minicpm5_layer1_official_history_callable_result_v1',complete=True,rank=rank,arm=arm,
                            call=case['call'],mode=mode,intervention=dict(rank=rank,arm=arm,call=ci,mode=mode),
                            normalized_original_layout=True,full_input_cache_unchanged=True,original_official_weights_unchanged=True,
                            attention_reexecuted=True,model_reexecuted=False,attention_causality_proven=False,full_model_qualified=False,
                            new_speedup_claim=False,official_history_intervention_performed=mode not in ('own','restore'),
                            attention_output_sha256=sha(raw),all_four_rows_exact=mode in ('own','restore'))
                        rows.append(dict(result=actual,output=dict(path=path,raw_bytes=16384,sha256=sha(raw)),
                            bucket=dict(bucket=128,bucket_index=0,phase='decode',n_positions=128,n_active_tokens=1,cache_aliases=84,configuration_unchanged=True),
                            independent_audit_payload_sha256='a'*64,official_target_sha256=sha(target),official_target_refs=refs))
            load=dict(rank=rank,original=True);weight=dict(rank=rank,original=True);cls.old['rank%d_result'%rank]=dict(load_proof=load,official_layer1_shards=weight)
            result=dict(schema='minicpm5_layer1_official_history_cpu_driver_v1',complete=True,rank=rank,
                full_model_qualified=False,new_speedup_claim=False,attention_causality_proven=False,full_model_forward_reexecuted=False,self_exit_claimed=False,
                complete_original_model_constructed=True,layer1_attention_reexecuted=True,all_own_history_outputs_exact=True,
                official_history_intervention_performed=True,replay_calls=90,physical_rows=360,raw_output_bytes=1474560,
                load_proof=load,official_layer1_shards=weight,records=rows,source_stream=dict(rank=rank,complete=True))
            for result_key,plan_key in pairs:result[result_key]=cls.plan[plan_key]
            cls.result.append(result);cls.items[rank]=items

    def completed_with(self,mutation=None,binding=None):
        def read(ref):
            value=copy.deepcopy(self.metadata[ref['path']])
            if mutation:mutation(ref['path'],value)
            return value
        return completed.completed(self.binding if binding is None else binding,read)

    def validate(self,values=None,raw_changes=None):
        raw=dict(self.data);raw.update(raw_changes or {})
        parent=self
        class Stream:
            def __init__(self,plan,result,audit,read,guard):self.rank=result['rank']
            def __iter__(self):return iter(parent.items[self.rank])
            def finish(self):return dict(rank=self.rank,complete=True)
        def targets(plan,data,case):
            return [],b'\x80\x3f'*8192,[dict(frame=f,sha='b'*64)for f in case['physical_frame_order']]
        def overlay(case,own,donors,mode):return {},dict(rank=case['rank'],arm=case['arm'],call=case['call']['index'],mode=mode)
        bundle=dict(collector_plan={},collector_chain=dict(rank_results=[dict(rank=0),dict(rank=1)]),audit_result={})
        with contextlib.ExitStack()as stack:
            for owner,name,value in [(results.history,'completed',lambda *a:bundle),(results.official,'completed',lambda *a:{}),
                (results.factored,'completed',lambda *a:dict(records=self.old)),(results.constructor,'check_load_metadata',lambda *a:True),
                (results.inputs,'load',lambda *a:{}),(results.inputs,'for_call',targets),(results.stream,'RankStream',Stream),
                (results.contract,'overlay',overlay),(results.payload,'read_owned_payload',lambda desc,path,guard:raw[str(path)])]:
                stack.enter_context(patch.object(owner,name,side_effect=value))
            return results.validate_rank_pair(self.result if values is None else values,self.plan,lambda:None)

    def rejected(self,change,raw_changes=None):
        data=copy.deepcopy(self.result);change(data)
        with self.assertRaises((RuntimeError,ValueError,AssertionError,KeyError,IndexError)):self.validate(data,raw_changes)

    def test_actual_metadata_links_include_all_five_ended_processes(self):
        bundle=self.completed_with();self.assertEqual(len(bundle['owners']),5)
        self.assertEqual(bundle['owners'][-1],bundle['comparison_exit']['owner'])

    def test_missing_comparator_or_failed_auditor_cannot_admit(self):
        b=dict(self.binding);b.pop('comparison_exit')
        with self.assertRaises(RuntimeError):self.completed_with(binding=b)
        for key in ('audit_exit','comparison_exit','model_exit'):
            path=self.binding[key]['path']
            def mutate(p,v):
                if p==path:v['actual_exit_code']=1
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.completed_with(mutate)

    def test_forged_comparison_lineage_or_qualification_rejected(self):
        path=self.bundle['comparison_exit']['result']['path']
        for field,value in [('native_audit_exit',{}),('comparisons',287),('full_model_qualified',True),('new_speedup_claim',True)]:
            def mutate(p,v):
                if p==path:v[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.completed_with(mutate)

    def test_native_drain_checks_comparator_and_auditor_before_release(self):
        checked=[]
        with patch.object(completed.p,'gone',side_effect=lambda o:checked.append(o)),patch.object(completed.original,'verify_owned_drain',return_value={'ok':True}):
            proof=completed.physical(self.bundle,lambda:None)
        self.assertEqual(checked,self.bundle['owners']);self.assertTrue(proof['independent_auditor_and_comparator_gone'])
        def gone(owner):
            if owner==self.bundle['comparison_exit']['owner']:raise RuntimeError('Comparator live')
        with patch.object(completed.p,'gone',side_effect=gone),patch.object(completed.original,'verify_owned_drain')as verify:
            with self.assertRaises(RuntimeError):completed.physical(self.bundle,lambda:None)
            verify.assert_not_called()

    def test_all_five_modes_and_180_full_outputs(self):
        proof=self.validate()
        self.assertEqual([proof[k]for k in ('decode_calls','original_controls','restored_controls','official_interventions','raw_bytes')],[180,36,36,108,2949120])
        self.assertEqual(len(proof['rank_proofs'][0]['outputs']),90)

    def test_wrong_rank_mode_or_reordered_rows_rejected(self):
        self.rejected(lambda v:v[0].update(rank=1))
        self.rejected(lambda v:v[0]['records'][1]['result'].update(mode='official_value'))
        self.rejected(lambda v:v[0]['records'].reverse())

    def test_missing_restore_or_wrong_count_rejected(self):
        self.rejected(lambda v:v[1]['records'].pop())
        self.rejected(lambda v:v[0].update(replay_calls=89))

    def test_changed_restore_raw_rejected_after_rehash(self):
        path=self.result[0]['records'][4]['output']['path'];raw=b'\x80\x3f'*8192
        def change(v):
            row=v[0]['records'][4];row['output']['sha256']=sha(raw);row['result'].update(attention_output_sha256=sha(raw),all_four_rows_exact=False)
        self.rejected(change,{path:raw})

    def test_wrong_current_official_target_digest_and_refs_rejected(self):
        self.rejected(lambda v:v[0]['records'][1].update(official_target_sha256='0'*64))
        self.rejected(lambda v:v[0]['records'][1]['official_target_refs'][0]['frame'].update(step=0))

    def test_source_overlay_or_model_claim_rejected(self):
        self.rejected(lambda v:v[0]['records'][2]['result']['intervention'].update(mode='own'))
        self.rejected(lambda v:v[0].update(full_model_qualified=True))
        self.rejected(lambda v:v[1]['records'][3]['result'].update(new_speedup_claim=True))

    def test_nonfinite_output_and_cross_rank_divergence_rejected(self):
        path=self.result[1]['records'][1]['output']['path']
        for raw in (b'\x80\x7f'+bytes(16382),b'\x80\x3f'*8192):
            def change(v):
                row=v[1]['records'][1];row['output']['sha256']=sha(raw);row['result']['attention_output_sha256']=sha(raw)
            self.rejected(change,{path:raw})

    def test_native_previous_donors_and_current_targets_are_distinct(self):
        frames=self.items[0][0]['case']['physical_frame_order'];case=dict(self.items[0][0]['case'])
        plan=dict(calls=[{}for _ in range(14)],frames=frames);plan['calls'][4]=case['call'];data={}
        for f in frames:
            for step in (0,1):
                frame=dict(f,step=step,index=f['index']-(4 if step==0 else 0))
                data[f['fixture'],step]=dict(record=dict(frame=frame,original_frame={'step':step},outputs={'taps':{'step':step}}),
                    key_raw=b'k'+bytes([step]),value_raw=b'v'+bytes([step]),targets={inputs.boundary.ATTENTION:bytes([step])*4096})
        donors,target,refs=inputs.for_call(plan,data,case)
        self.assertTrue(all(d['record']['frame']['step']==0 for d in donors));self.assertEqual(target,b'\x01'*16384)
        self.assertTrue(all(r['frame']['step']==1 for r in refs))
        data[0,1]['record']['frame']['step']=0
        with self.assertRaises(ValueError):inputs.for_call(plan,data,case)

    def test_guard_denies_input_read_before_payload_access(self):
        with patch.object(inputs.payload,'read_owned_payload')as read:
            def deny():raise RuntimeError('No serial admission')
            with self.assertRaises(RuntimeError):inputs.load({}, {}, deny)
            read.assert_not_called()

    def test_worker_guard_precedes_native_imports(self):
        class Context:
            def guard(self,role):raise RuntimeError('Not admitted')
        with self.assertRaises(RuntimeError):worker.run(Context(),'/unused')
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))

    def test_runner_exact_interpreter_resources_and_rank_environment(self):
        self.assertEqual(runner.PYTHON,'/home/ubuntu/venvs/vllm_neuron/bin/python')
        self.assertEqual(runner.PROPS['MemoryMax'],str(16*2**30));self.assertEqual(runner.PROPS['MemorySwapMax'],'0')
        self.assertEqual(runner.PROPS['AllowedCPUs'],'6-7');self.assertEqual(runner.PROPS['KillMode'],'control-group')
        env=runner.env('synthetic',29627,1)
        self.assertEqual([env[k]for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE')],['1','1','2','2'])
        self.assertEqual(env['TORCHELASTIC_RUN_ID'],'mini-layer1-official-history-synthetic')
        self.assertEqual(runner.argv('rank',dict(path='/admission',sha256='a'*64))[:4],[runner.PYTHON,'-B',runner.SELF,'--mode'])

    def test_runner_drain_cannot_skip_either_rank(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);(root/'rank0-owner.json').write_text('{}');(root/'rank1-owner.json').write_text('{}')
            values=dict(LoadState='not-found',InvocationID='',MainPID='0',ControlPID='0',ExecMainPID='0')
            checked=[]
            with patch.object(runner.p,'properties',return_value=values),patch.object(runner.p,'read',side_effect=lambda path:{'rank':path.name}),\
                patch.object(runner.p,'gone',side_effect=lambda owner:checked.append(owner)),patch.object(runner.p,'empty_group',return_value={}):
                runner.verify_owned_drain(dict(output_root=str(root),unit='synthetic'),None,None,None,False)
            self.assertEqual(checked,[{'rank':'rank0-owner.json'},{'rank':'rank1-owner.json'}])

    def test_lease_release_follows_physical_drain_and_entrypoints_are_frozen(self):
        tree=ast.parse(Path(runner.__file__).read_text());outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='outer')
        calls=[n for n in ast.walk(outer)if isinstance(n,ast.Call)]
        drain=next(n for n in calls if isinstance(n.func,ast.Name)and n.func.id=='retain_until_drained')
        release=next(n for n in calls if isinstance(n.func,ast.Attribute)and n.func.attr=='release_lock')
        self.assertLess(drain.lineno,release.lineno)
        source=Path(runner.__file__).read_text();self.assertIn("completed_native_cache=proof['completed_native_cache']",source)
        self.assertIn("official.completed(spec['completed_native_cache']",source)
        self.assertIn("official.physical(priors['completed_native_cache']",source)

    def test_prepare_preserves_sources_and_pins_all_native_outputs(self):
        tree=ast.parse((Q/'prepare_minicpm5_layer1_official_history_v1.py').read_text())
        text=ast.unparse(tree)
        self.assertIn("copy.deepcopy(completed['spec'])",text);self.assertIn("copy.deepcopy(completed['plan'])",text)
        self.assertIn("row['outputs'].values()",text);self.assertIn("plan['inputs'][desc['path']] = value",text)
        self.assertIn("reader(previous.ref(folder / 'result.json')) == row",text)
        self.assertIn("plan['source_sha256'][path] == sha",text);self.assertIn('completed_native_cache=binding',text)


if __name__=='__main__':unittest.main(verbosity=2)
