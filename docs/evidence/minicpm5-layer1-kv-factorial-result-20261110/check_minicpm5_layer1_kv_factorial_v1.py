"""Adversarial factorial provenance/output checks without native imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_kv_factorial_callable_v1 as native
from research import minicpm5_layer1_kv_factorial_results_v1 as verify
from research import minicpm5_layer1_after_cross_history_v1 as crossed
from research.check_minicpm5_layer1_cross_history_v1 import CrossChecks
from research.check_minicpm5_layer1_own_history_results_v2 import read_ref

class FactorialChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CrossChecks.setUpClass();cls.cross=CrossChecks();cls.plan=CrossChecks.plan
        cls.paired_proof=cls.cross.validate();cls.plan['layer1_kv_factorial']={'synthetic_cross_exit':True}
        cls.results=[];cls.cases=CrossChecks.cases
        for rank,old in enumerate(CrossChecks.results):
            value=copy.deepcopy(old);value.update(schema='minicpm5_layer1_kv_factorial_cpu_driver_v1',
                replay_calls=180,physical_rows=720,raw_output_bytes=2949120,independent_key_value_intervention_performed=True,
                completed_cross_history=cls.plan['layer1_kv_factorial'],records=[])
            index={(r['result']['arm'],r['result']['call']['index'],r['result']['history_arm']):r for r in old['records']if r['phase']=='matrix'}
            cases={(r['arm'],r['call']['index']):r for r in cls.cases if r['rank']==rank}
            for ci in native.boundary.DECODE_CALLS:
                for arm in native.boundary.ARMS:
                    pairs=[(arm,arm,'matrix')]+[(k,v,'matrix')for k in native.boundary.ARMS for v in native.boundary.ARMS if (k,v)!=(arm,arm)]+[(arm,arm,'restore')]
                    for k,v,phase in pairs:
                        row=copy.deepcopy(index[arm,ci,k]);case=cases[arm,ci];kc=cases[k,ci];vc=cases[v,ci]
                        data=Path(row['output']['path']).read_bytes()
                        if k!=v:data=b'\x00\x3f'+data[2:]
                        path=Path(cls.plan['output_root'])/('cpu-rank'+str(rank))/arm/('call-%02d'%ci)/k/v/phase/'attention.bf16'
                        path.parent.mkdir(parents=True);path.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
                        row.pop('history_original_call');row.pop('history_audit_payload_sha256')
                        row.update(phase=phase,key_original_call=kc['original_call'],value_original_call=vc['original_call'],
                            key_audit_payload_sha256='b'*64,value_audit_payload_sha256='b'*64,
                            output=dict(path=str(path),raw_bytes=16384,sha256=sha,identity=verify.payload.file_identity(path,path.lstat())))
                        row['result'].pop('history_arm')
                        row['result'].update(schema='minicpm5_layer1_kv_factorial_callable_result_v1',key_arm=k,value_arm=v,
                            intervention=native.contract(case,kc,vc),paired_history_intervention_performed=k==v and k!=arm,
                            independent_key_value_intervention_performed=k!=v,
                            all_four_rows_exact=sha==case['expected_attention']['logical_sha256'],attention_output_sha256=sha)
                        value['records'].append(row)
            cls.results.append(value)
    @classmethod
    def tearDownClass(cls):CrossChecks.tearDownClass()
    def validate(self,results=None):
        with patch.object(verify.p,'read_ref',side_effect=self.cross.f.read),patch.object(verify.history,'completed',
            return_value=dict(audit_result=dict(layer1_cache=dict(raw_records=self.cross.f.audited)))),patch.object(verify.crossed,'completed',
            return_value=dict(audit=dict(result=self.paired_proof))):
            return verify.validate_rank_pair(results or self.results,self.plan,lambda:None)
    def reject(self,mutation):
        r=copy.deepcopy(self.results);mutation(r)
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):self.validate(r)
    def test_all324_combinations_and360_full_outputs(self):
        p=self.validate();self.assertEqual([p[k]for k in ('decode_calls','matrix_cells','restored_controls','raw_bytes')],[360,324,36,5898240])
        self.assertTrue(p['all_previous_paired_outputs_exact'])
    def test_wrong_rank_call_slot_mask_or_layout_rejected(self):
        q=self.cases[0]
        for key in ('rank','positions','seq_ids','attention_route','physical_frame_order'):
            d=copy.deepcopy(q);d[key]=None
            with self.assertRaises(ValueError):native.contract(q,q,d)
        d=copy.deepcopy(q);d['caches'][0]['sdk_view']['stride'][0]=1
        with self.assertRaises(ValueError):native.contract(q,d,q)
    def test_original_paired_contract_not_relabelled(self):
        q=copy.deepcopy(self.cases[0]);q['paired_history_intervention_allowed']=True
        with self.assertRaises(ValueError):native.contract(q,self.cases[0],self.cases[0])
    def test_missing_restore_and_reordered_cells_rejected(self):
        self.reject(lambda r:r[0]['records'].pop());self.reject(lambda r:r[1]['records'].reverse())
    def test_wrong_key_or_value_source_rejected(self):
        self.reject(lambda r:r[0]['records'][1].update(value_original_call=r[0]['records'][0]['value_original_call']))
        self.reject(lambda r:r[0]['records'][1].update(key_audit_payload_sha256='0'*64))
        self.reject(lambda r:r[0]['records'][1].update(value_audit_payload_sha256='0'*64))
    def test_failed_diagonal_and_restored_controls_rejected(self):
        self.reject(lambda r:r[0]['records'][0]['result'].update(all_four_rows_exact=False))
        self.reject(lambda r:r[0]['records'][9]['result'].update(all_four_rows_exact=False))
    def test_incorrect_independent_history_flag_rejected(self):
        self.reject(lambda r:r[0]['records'][1]['result'].update(independent_key_value_intervention_performed=False))
    def test_short_outputs_and_other_rank_paths_rejected(self):
        self.reject(lambda r:r[0]['records'][1]['output'].update(raw_bytes=4096))
        self.reject(lambda r:r[0]['records'][1].update(output=r[1]['records'][1]['output']))
    def test_changed_paired_output_rejected_even_with_consistent_digest(self):
        r=copy.deepcopy(self.results);row=r[0]['records'][4];assert row['result']['key_arm']==row['result']['value_arm']
        path=Path(row['output']['path']);original=path.read_bytes()
        try:
            data=b'\x00\x3e'+original[2:];path.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
            row['output'].update(sha256=sha,identity=verify.payload.file_identity(path,path.lstat()));row['result']['attention_output_sha256']=sha
            with self.assertRaisesRegex(ValueError,'previous real experiment'):self.validate(r)
        finally:
            path.write_bytes(original);self.results[0]['records'][4]['output']['identity']=verify.payload.file_identity(path,path.lstat())
    def test_nonfinite_mixed_output_rejected(self):
        r=copy.deepcopy(self.results);row=r[0]['records'][1];path=Path(row['output']['path']);original=path.read_bytes()
        try:
            data=b'\x80\x7f'+original[2:];path.write_bytes(data);sha=hashlib.sha256(data).hexdigest()
            row['output'].update(sha256=sha,identity=verify.payload.file_identity(path,path.lstat()));row['result']['attention_output_sha256']=sha
            with self.assertRaisesRegex(ValueError,'finiteness'):self.validate(r)
        finally:
            path.write_bytes(original);self.results[0]['records'][1]['output']['identity']=verify.payload.file_identity(path,path.lstat())
    def test_unmodified_attention_invocation_and_resources(self):
        def call(path):
            return next(ast.dump(n)for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='attention')
        self.assertEqual(call(Q/'minicpm5_layer1_cross_history_callable_v1.py'),call(Path(native.__file__)))
        import minicpm5_layer1_kv_factorial_cpu_runner_v1 as current
        import minicpm5_layer1_cross_history_cpu_runner_v1 as old
        self.assertEqual(current.PROPS,old.PROPS)
        for name in ('usage','verify_owned_drain','cleanup_owned_service','cleanup_children'):
            def node(m):return next(ast.dump(n)for n in ast.parse(Path(m.__file__).read_text()).body if getattr(n,'name',None)==name)
            self.assertEqual(node(current),node(old))
    def test_actual_cross_execution_and_postexit_audit_required(self):
        h=json.loads((Q/'continuation-handoff-20261110-v412-mini-cross-history-complete.json').read_text())['cross_history']
        b=dict(schema='minicpm5_layer1_completed_cross_history_v1',model_exit=h['actual_exit'],audit_exit=h['independent_audit_actual_exit'])
        got=crossed.completed(b,read_ref);self.assertEqual(len(got['owners']),5)
        for target,field,value in ((b['model_exit'],'actual_exit_code',1),(b['audit_exit'],'actual_exit_code',1),
            (h['independent_audit'],'every_raw_byte_reread_after_all_model_processes_exited',False)):
            def altered(ref):
                v=read_ref(ref)
                if ref==target:v[field]=value
                return v
            with self.assertRaises(RuntimeError):crossed.completed(b,altered)
    def test_actual_model_and_auditor_drained(self):
        fake=dict(owners=[{'pid':i}for i in range(5)],ended=dict(outputs=dict(admission={}),coordinator={}),records=dict(admission=dict(attempt_id='test')),spec={})
        with patch.object(crossed.p,'gone')as gone,patch.object(crossed.original,'verify_owned_drain',return_value={})as drain:
            crossed.physical(fake,lambda:None);self.assertEqual(gone.call_count,5);drain.assert_called_once()
    def test_guard_denial_before_output_access(self):
        def denied():raise RuntimeError('busy')
        with patch.object(verify.payload,'read_owned_payload')as read,self.assertRaisesRegex(RuntimeError,'busy'):verify.validate_rank_pair(None,None,denied)
        read.assert_not_called()
    def test_precision_change_or_qualification_claim_rejected(self):
        self.reject(lambda r:r[0]['official_layer1_shards'].update(precision_changed=True))
        self.reject(lambda r:r[0].update(new_speedup_claim=True))
        self.reject(lambda r:r[0]['records'][1]['result'].update(full_model_qualified=True))

if __name__=='__main__':
    assert not {'torch','numpy','transformers'}&set(sys.modules)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(FactorialChecks)
    sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
