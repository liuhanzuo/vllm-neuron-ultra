"""Stdlib synthetic streams and real metadata lineage; no model/payload imports."""
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
from research import minicpm5_layer1_after_official_history_v1 as completed
from research import minicpm5_layer1_current_history_factorial_results_v1 as results
from research import minicpm5_layer1_current_history_factorial_controls_v1 as controls
from research import minicpm5_layer1_current_history_factorial_v1 as contract
import minicpm5_layer1_current_history_factorial_cpu_runner_v1 as runner
import minicpm5_layer1_current_history_factorial_cpu_worker_v1 as worker


def sha(raw):return hashlib.sha256(raw).hexdigest()


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata={}
        def read(ref):
            v=completed.p.read_ref(ref);cls.metadata[ref['path']]=v;return v
        h=read(completed.HANDOFF)
        cls.binding=dict(schema='minicpm5_layer1_completed_official_history_v1',model_exit=h['official_history']['actual_exit'],
            audit_exit=h['official_history']['independent_audit_exit'])
        cls.bundle=completed.completed(cls.binding,read)
        cls.storage_binding=dict(schema='minicpm5_completed_storage_maintenance_v1',handoff=completed.HANDOFF,
            maintenance_exit=h['storage_maintenance']['actual_exit'],audit_exit=h['storage_maintenance']['independent_audit_exit'])
        cls.storage=completed.storage_completed(cls.storage_binding,cls.bundle,read)
        cls.plan=dict(output_root='/synthetic/current-history',inputs={})
        cls.pairs=(('completed_layer1_data','layer1_own_history_replay'),('completed_own_history','layer1_cross_history'),
            ('completed_cross_history','layer1_kv_factorial'),('completed_factorial','layer1_native_cache'),
            ('completed_native_cache','layer1_official_history'),('completed_official_history','layer1_current_history_factorial'),
            ('completed_storage','completed_storage_maintenance'))
        for _,key in cls.pairs:cls.plan[key]={'synthetic':key}
        cls.data={};cls.items={};cls.values=[];cls.old={};cls.controls={}
        for rank in (0,1):
            items=[];rows=[];oldrows=[]
            for arm in results.stream.raw.ARMS:
                for ci in results.stream.boundary.DECODE_CALLS:
                    frames=[dict(index=4+s,fixture=s,slot=s,step=1)for s in range(4)]
                    case=dict(rank=rank,arm=arm,call=dict(index=ci,phase='decode'),bucket=128,physical_frame_order=frames)
                    own=bytes(16384);items.append(dict(case=case,normalized_raw=own,cache_raw={},expected_attention_raw=own,audited_call=dict(payload_sha256='a'*64)))
                    oldcontrols={}
                    for old_mode in controls.OLD_MODES:
                        raw=own if old_mode in ('own','restore') else b'\x00\x3f'*8192
                        path=str(Path('/synthetic/previous')/('cpu-rank%d'%rank)/arm/('call-%02d'%ci)/old_mode/'attention.bf16')
                        desc=dict(path=path,raw_bytes=16384,sha256=sha(raw),identity={'synthetic':path})
                        cls.plan['inputs'][path]=dict(sha256=sha(raw),identity=desc['identity']);cls.data[path]=raw
                        oldcontrols[old_mode]=dict(raw=raw,reference=desc)
                        oldrows.append(dict(result=dict(schema='minicpm5_layer1_official_history_callable_result_v1',
                            complete=True,rank=rank,arm=arm,call=case['call'],mode=old_mode,attention_output_sha256=sha(raw)),output=desc))
                    cls.controls[rank,arm,ci]=oldcontrols
                    target=b'\x80\x3f'*8192;refs=[dict(frame=f,sha='b'*64)for f in frames]
                    for mode in contract.MODES:
                        raw=own if mode in ('own','restore') else b'\x00\x3f'*8192
                        path=str(Path(cls.plan['output_root'])/('cpu-rank%d'%rank)/arm/('call-%02d'%ci)/mode/'attention.bf16');cls.data[path]=raw
                        actual=dict(schema='minicpm5_layer1_current_history_factorial_callable_result_v1',complete=True,rank=rank,arm=arm,
                            call=case['call'],mode=mode,intervention=dict(rank=rank,arm=arm,call=ci,mode=mode),
                            normalized_original_layout=True,full_input_cache_unchanged=True,original_official_weights_unchanged=True,
                            attention_reexecuted=True,model_reexecuted=False,attention_causality_proven=False,full_model_qualified=False,
                            new_speedup_claim=False,official_history_intervention_performed=mode in ('official_history','official_both'),
                            official_current_intervention_performed=mode in ('official_current','official_both'),
                            current_qkv_change_together=mode in ('official_current','official_both'),
                            attention_output_sha256=sha(raw),all_four_rows_exact=mode in ('own','restore'))
                        rows.append(dict(result=actual,output=dict(path=path,raw_bytes=16384,sha256=sha(raw)),
                            previous_control=controls.verify(oldcontrols,mode,raw),
                            bucket=dict(bucket=128,bucket_index=0,phase='decode',n_positions=128,n_active_tokens=1,cache_aliases=84,configuration_unchanged=True),
                            independent_audit_payload_sha256='a'*64,official_target_sha256=sha(target),official_target_refs=refs))
            load=dict(rank=rank,original=True);weights=dict(rank=rank,original=True)
            cls.old['rank%d_result'%rank]=dict(rank=rank,complete=True,records=oldrows,load_proof=load,official_layer1_shards=weights)
            value=dict(schema='minicpm5_layer1_current_history_factorial_cpu_driver_v1',complete=True,rank=rank,
                full_model_qualified=False,new_speedup_claim=False,attention_causality_proven=False,full_model_forward_reexecuted=False,self_exit_claimed=False,
                complete_original_model_constructed=True,layer1_attention_reexecuted=True,all_own_history_outputs_exact=True,
                official_history_intervention_performed=True,official_current_intervention_performed=True,current_qkv_change_together=True,
                previous_real_controls_full_bytes_exact=True,replay_calls=90,physical_rows=360,raw_output_bytes=1474560,
                load_proof=load,official_layer1_shards=weights,records=rows,source_stream=dict(rank=rank,complete=True))
            for result_key,plan_key in cls.pairs:value[result_key]=cls.plan[plan_key]
            cls.values.append(value);cls.items[rank]=items

    def metadata_read(self,mutate=None):
        def read(ref):
            v=copy.deepcopy(self.metadata[ref['path']])
            if mutate:mutate(ref['path'],v)
            return v
        return read

    def validate(self,values=None,raw_changes=None):
        raw=dict(self.data);raw.update(raw_changes or {});parent=self
        class Stream:
            def __init__(self,plan,result,audit,read,guard):self.rank=result['rank']
            def __iter__(self):return iter(parent.items[self.rank])
            def finish(self):return dict(rank=self.rank,complete=True)
        def targets(plan,data,case):return [],[],b'\x80\x3f'*8192,[dict(frame=f,sha='b'*64)for f in case['physical_frame_order']]
        def select(case,normal,cache,donors,current,mode):return b'',b'',{},dict(rank=case['rank'],arm=case['arm'],call=case['call']['index'],mode=mode)
        bundle=dict(collector_plan={},collector_chain=dict(rank_results=[dict(rank=0),dict(rank=1)]),audit_result={})
        previous=dict(records=self.old,spec=dict(output_root='/synthetic/previous'))
        with contextlib.ExitStack()as stack:
            for owner,name,fun in [(results.history,'completed',lambda *a:bundle),(results.official,'completed',lambda *a:{}),
                (results.completed,'completed',lambda *a:previous),(results.completed,'storage_completed',lambda *a:{}),
                (results.constructor,'check_load_metadata',lambda *a:True),(results.inputs,'load',lambda *a:{}),
                (contract,'for_call',targets),(contract,'select',select),(results.stream,'RankStream',Stream),
                (results.payload,'read_owned_payload',lambda desc,path,guard:raw[str(path)])]:
                stack.enter_context(patch.object(owner,name,side_effect=fun))
            return results.validate_rank_pair(self.values if values is None else values,self.plan,lambda:None)

    def rejected(self,change,raw_changes=None):
        v=copy.deepcopy(self.values);change(v)
        with self.assertRaises((RuntimeError,ValueError,KeyError,IndexError)):self.validate(v,raw_changes)

    def test_all_180_outputs_and_108_prior_full_byte_controls(self):
        v=self.validate();self.assertEqual([v[k]for k in ('decode_calls','original_controls','restored_controls','official_interventions','raw_bytes','previous_control_calls')],[180,36,36,108,2949120,108])
        self.assertTrue(v['previous_real_controls_full_bytes_exact'])

    def test_actual_official_history_and_storage_metadata(self):
        b=completed.completed(self.binding,self.metadata_read());s=completed.storage_completed(self.storage_binding,b,self.metadata_read())
        self.assertEqual(len(b['owners']),5);self.assertEqual(len(s['owners']),2)

    def test_each_actual_exit_must_be_successful(self):
        for ref in [self.binding['model_exit'],self.binding['audit_exit'],self.storage_binding['maintenance_exit'],self.storage_binding['audit_exit']]:
            def mutate(p,v):
                if p==ref['path']:v['actual_exit_code']=1
            with self.subTest(ref=ref),self.assertRaises(RuntimeError):
                read=self.metadata_read(mutate);b=completed.completed(self.binding,read);completed.storage_completed(self.storage_binding,b,read)

    def test_storage_requires_exact_handoff_both_exits_and_full_audit(self):
        for key in ('audit_exit','maintenance_exit','handoff'):
            binding=dict(self.storage_binding);binding.pop(key)
            with self.subTest(key=key),self.assertRaises(RuntimeError):completed.storage_completed(binding,self.bundle,self.metadata_read())
        path=self.storage['audit_exit']['result']['path']
        for key,val in [('all_original_bytes_reread_after_maintenance_exit',False),('actual_exit',{}),('canonical_unchanged',False)]:
            def mutate(p,v):
                if p==path:v[key]=val
            with self.subTest(key=key),self.assertRaises(RuntimeError):completed.storage_completed(self.storage_binding,self.bundle,self.metadata_read(mutate))

    def test_actual_model_audit_scope_and_lineage(self):
        path=self.bundle['audit_exit']['result']['path']
        for key,val in [('model_exit',{}),('full_model_qualified',True),('every_raw_byte_reread_after_all_model_processes_exited',False)]:
            def mutate(p,v):
                if p==path:v[key]=val
            with self.subTest(key=key),self.assertRaises(RuntimeError):completed.completed(self.binding,self.metadata_read(mutate))

    def test_physical_drains_include_all_seven_recent_owners(self):
        checked=[]
        with patch.object(completed.p,'gone',side_effect=lambda o:checked.append(o)),patch.object(completed.original,'verify_owned_drain',return_value={}):
            completed.physical(self.bundle,lambda:None);completed.storage_physical(self.storage,lambda:None)
        self.assertEqual(checked,self.bundle['owners']+self.storage['owners'])

    def test_forged_history_control_rejected_even_with_rehashed_output(self):
        row=self.values[0]['records'][2];path=row['output']['path'];raw=b'\x40\x3f'*8192
        def change(v):v[0]['records'][2]['output']['sha256']=sha(raw);v[0]['records'][2]['result']['attention_output_sha256']=sha(raw)
        self.rejected(change,{path:raw})

    def test_forged_previous_raw_rejected_even_when_current_claim_unchanged(self):
        desc=self.values[0]['records'][2]['previous_control']['output']
        with self.assertRaises(ValueError):self.validate(raw_changes={desc['path']:b'\x40\x3f'*8192})

    def test_prior_restore_bytes_must_equal_prior_own(self):
        desc=self.values[0]['records'][4]['previous_control']['output']
        with self.assertRaises(ValueError):self.validate(raw_changes={desc['path']:b'\x40\x3f'*8192})

    def test_wrong_prior_control_descriptor_or_claim_rejected(self):
        self.rejected(lambda v:v[0]['records'][2]['previous_control'].update(previous_mode='own'))
        self.rejected(lambda v:v[0]['records'][1].update(previous_control={'invented':True}))
        self.rejected(lambda v:v[0].update(previous_real_controls_full_bytes_exact=False))

    def test_current_qkv_and_history_flags_independent(self):
        self.rejected(lambda v:v[0]['records'][1]['result'].update(official_history_intervention_performed=True))
        self.rejected(lambda v:v[0]['records'][2]['result'].update(official_current_intervention_performed=True))
        self.rejected(lambda v:v[0]['records'][3]['result'].update(current_qkv_change_together=False))

    def test_counts_modes_missing_restore_and_reordered_rows_rejected(self):
        self.rejected(lambda v:v[1]['records'].pop());self.rejected(lambda v:v[0].update(replay_calls=89))
        self.rejected(lambda v:v[0]['records'][1]['result'].update(mode='official_key'));self.rejected(lambda v:v[0]['records'].reverse())

    def test_nonfinite_output_and_rank_divergence_rejected(self):
        path=self.values[1]['records'][1]['output']['path']
        for raw in (b'\x80\x7f'+bytes(16382),b'\x40\x3f'*8192):
            def change(v):v[1]['records'][1]['output']['sha256']=sha(raw);v[1]['records'][1]['result']['attention_output_sha256']=sha(raw)
            self.rejected(change,{path:raw})

    def test_wrong_target_source_intervention_and_whole_model_claim_rejected(self):
        self.rejected(lambda v:v[0]['records'][1].update(official_target_sha256='0'*64))
        self.rejected(lambda v:v[0]['records'][1]['result']['intervention'].update(mode='own'))
        self.rejected(lambda v:v[0].update(full_model_qualified=True))
        self.rejected(lambda v:v[0]['records'][1]['result'].update(new_speedup_claim=True))

    def test_control_guard_denies_payload_read(self):
        def deny():raise RuntimeError('No lease')
        with patch.object(controls.payload,'read_owned_payload')as raw:
            with self.assertRaises(RuntimeError):controls.for_call({}, {}, {}, deny)
            raw.assert_not_called()

    def test_worker_guard_denies_native_imports(self):
        class Context:
            def guard(self,role):raise RuntimeError('No admission')
        with self.assertRaises(RuntimeError):worker.run(Context(),'/unused')
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))

    def test_original_resources_and_all_predecessor_fields(self):
        self.assertEqual(runner.PYTHON,'/home/ubuntu/venvs/vllm_neuron/bin/python')
        self.assertEqual(runner.PROPS['MemoryMax'],str(16*2**30));self.assertEqual(runner.PROPS['AllowedCPUs'],'6-7')
        self.assertEqual(runner.PROPS['MemorySwapMax'],'0');self.assertEqual(runner.CAP,256*2**20)
        env=runner.env('synthetic',29629,1)
        self.assertEqual([env[k]for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE')],['1','1','2','2'])
        text=Path(runner.__file__).read_text()
        for fragment in ("latest_completed.physical(priors['completed_official_history']", "latest_completed.storage_physical(priors['completed_storage']", "completed_storage=spec['completed_storage']", "completed_official_history=proof['completed_official_history']"):
            self.assertIn(fragment,text)

    def test_serial_lock_release_after_physical_drain(self):
        tree=ast.parse(Path(runner.__file__).read_text());outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='outer')
        calls=[n for n in ast.walk(outer)if isinstance(n,ast.Call)]
        drain=next(n for n in calls if isinstance(n.func,ast.Name)and n.func.id=='retain_until_drained')
        release=next(n for n in calls if isinstance(n.func,ast.Attribute)and n.func.attr=='release_lock')
        self.assertLess(drain.lineno,release.lineno)

    def test_builder_requires_v420_preserves_sources_and_pins_old_raw_without_reading(self):
        text=(Q/'prepare_minicpm5_layer1_current_history_factorial_v1.py').read_text()
        for fragment in ("handoff_ref==official.HANDOFF", "copy.deepcopy(completed['spec'])", "copy.deepcopy(completed['plan'])", "plan['inputs'][desc['path']]=value", "plan['source_sha256'][path] == sha", "completed_storage=storage_binding", "audit_minicpm5_layer1_current_history_factorial_result_v1.py"):
            self.assertIn(fragment,text)
        self.assertNotIn('read_owned_payload',text)


if __name__=='__main__':unittest.main(verbosity=2)
