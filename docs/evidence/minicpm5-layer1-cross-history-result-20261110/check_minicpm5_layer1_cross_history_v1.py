"""Negative scientific-boundary and independent-output checks, stdlib only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_cross_history_callable_v1 as native
from research import minicpm5_layer1_cross_history_results_v1 as verify
from research import minicpm5_layer1_after_own_history_v1 as own
from research.check_minicpm5_layer1_own_history_results_v2 import ResultTests as Fixture,read_ref

class CrossChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass();cls.f=Fixture();cls.plan=Fixture.plan;cls.plan['layer1_cross_history']={'synthetic':True}
        cls.cases=native.boundary.all_boundaries(cls.plan,lambda path,sha:cls.f.read(dict(path=path,sha256=sha)))
        cls.results=[]
        for rank,original in enumerate(Fixture.results):
            value=copy.deepcopy(original);value.update(schema='minicpm5_layer1_cross_history_cpu_driver_v1',
                replay_calls=72,physical_rows=288,raw_output_bytes=1179648,paired_history_intervention_performed=True,
                completed_own_history=cls.plan['layer1_cross_history'],records=[])
            index={(r['result']['arm'],r['result']['call']['index']):r for r in original['records']}
            cases={(r['arm'],r['call']['index']):r for r in cls.cases if r['rank']==rank}
            for ci in native.boundary.DECODE_CALLS:
                for arm in native.boundary.ARMS:
                    for donor,phase in [(arm,'matrix')]+[(d,'matrix')for d in native.boundary.ARMS if d!=arm]+[(arm,'restore')]:
                        row=copy.deepcopy(index[arm,ci]);case=cases[arm,ci];dc=cases[donor,ci]
                        data=Path(row['output']['path']).read_bytes()
                        if donor!=arm:data=b'\x00\x00'+data[2:]
                        path=Path(row['output']['path']).parent/donor/phase/'attention.bf16';path.parent.mkdir(parents=True);path.write_bytes(data)
                        sha=hashlib.sha256(data).hexdigest()
                        row.update(phase=phase,history_original_call=dc['original_call'],history_audit_payload_sha256='b'*64,
                            output=dict(path=str(path),raw_bytes=16384,sha256=sha,identity=verify.payload.file_identity(path,path.lstat())))
                        row['result'].update(schema='minicpm5_layer1_cross_history_callable_result_v1',history_arm=donor,
                            intervention=native.contract(case,dc),paired_history_intervention_performed=donor!=arm,
                            all_four_rows_exact=donor==arm,attention_output_sha256=sha)
                        value['records'].append(row)
            cls.results.append(value)
    @classmethod
    def tearDownClass(cls):Fixture.tearDownClass()
    def validate(self,result=None):
        with patch.object(verify.p,'read_ref',side_effect=self.f.read),patch.object(verify.history,'completed',
            return_value=dict(audit_result=dict(layer1_cache=dict(raw_records=Fixture.audited)))):
            return verify.validate_rank_pair(result or self.results,self.plan,lambda:None)
    def reject(self,mutation):
        r=copy.deepcopy(self.results);mutation(r)
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):self.validate(r)
    def test_all108_metadata_combinations_and144_outputs(self):
        for q in self.cases:
            for d in self.cases:
                if q['rank']==d['rank'] and q['call']==d['call']:native.contract(q,d)
        p=self.validate();self.assertEqual([p[k]for k in ('decode_calls','matrix_cells','restored_controls','raw_bytes')],[144,108,36,2359296])
    def test_different_slots_positions_masks_and_cache_layout_rejected(self):
        q=self.cases[0]
        for key in ('rank','positions','seq_ids','attention_route','physical_frame_order'):
            d=copy.deepcopy(q);d[key]=None
            with self.assertRaises(ValueError):native.contract(q,d)
        d=copy.deepcopy(q);d['caches'][1]['sdk_view']['stride'][0]=1
        with self.assertRaises(ValueError):native.contract(q,d)
    def test_original_contract_cannot_be_relabelled_as_cross(self):
        q=copy.deepcopy(self.cases[0]);q['paired_history_intervention_allowed']=True
        with self.assertRaises(ValueError):native.contract(q,self.cases[0])
    def test_missing_restore_or_reordered_intervention_rejected(self):
        self.reject(lambda r:r[0]['records'].pop());self.reject(lambda r:r[0]['records'].reverse())
    def test_mixed_or_wrong_donor_provenance_rejected(self):
        self.reject(lambda r:r[0]['records'][1].update(history_original_call=r[0]['records'][0]['history_original_call']))
        self.reject(lambda r:r[0]['records'][1].update(history_audit_payload_sha256='0'*64))
    def test_failed_diagonal_or_restored_control_rejected(self):
        self.reject(lambda r:r[0]['records'][0]['result'].update(all_four_rows_exact=False))
        self.reject(lambda r:r[0]['records'][3]['result'].update(all_four_rows_exact=False))
    def test_short_output_wrong_rank_path_and_weight_change_rejected(self):
        self.reject(lambda r:r[0]['records'][1]['output'].update(raw_bytes=4096))
        self.reject(lambda r:r[0]['records'][1].update(output=r[1]['records'][1]['output']))
        self.reject(lambda r:r[0]['official_layer1_shards'].update(precision_changed=True))
    def test_nonfinite_output_rejected_even_with_consistent_digest(self):
        result=copy.deepcopy(self.results);row=result[0]['records'][1];path=Path(row['output']['path']);data=path.read_bytes()
        try:
            changed=b'\x80\x7f'+data[2:];path.write_bytes(changed);sha=hashlib.sha256(changed).hexdigest()
            row['output'].update(sha256=sha,identity=verify.payload.file_identity(path,path.lstat()));row['result']['attention_output_sha256']=sha
            with self.assertRaisesRegex(ValueError,'finiteness'):self.validate(result)
        finally:
            path.write_bytes(data);self.results[0]['records'][1]['output']['identity']=verify.payload.file_identity(path,path.lstat())
    def test_attention_invocation_identical_to_successful_own_history(self):
        def call(path):
            tree=ast.parse(path.read_text())
            return next(ast.dump(n)for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='attention')
        self.assertEqual(call(Q/'minicpm5_layer1_own_history_callable_v1.py'),call(Path(native.__file__)))
    def test_guard_denial_precedes_native_or_saved_output_access(self):
        def denied():raise RuntimeError('busy')
        with patch.object(verify.payload,'read_owned_payload')as read,self.assertRaisesRegex(RuntimeError,'busy'):
            verify.validate_rank_pair(None,None,denied)
        read.assert_not_called()
    def test_actual_completed_replay_and_audit_required(self):
        h=json.loads((Q/'continuation-handoff-20261110-v410-mini-own-history-reproduced.json').read_text())['own_history']
        b=dict(schema='minicpm5_layer1_completed_own_history_v1',model_exit=h['actual_exit'],audit_exit=h['independent_audit_actual_exit'])
        got=own.completed(b,read_ref);self.assertEqual(len(got['owners']),5)
        for target,field,value in ((b['model_exit'],'actual_exit_code',1),(b['audit_exit'],'actual_exit_code',1),
            (h['independent_audit'],'every_raw_byte_reread_after_all_model_processes_exited',False)):
            def altered(ref):
                v=read_ref(ref)
                if ref==target:v[field]=value
                return v
            with self.assertRaises(RuntimeError):own.completed(b,altered)
    def test_model_and_auditor_processes_both_drained(self):
        fake=dict(owners=[{'pid':i}for i in range(5)],ended=dict(outputs=dict(admission={}),coordinator={}),
            records=dict(admission=dict(attempt_id='test')),spec={})
        with patch.object(own.p,'gone')as gone,patch.object(own.original,'verify_owned_drain',return_value={})as drain:
            own.physical(fake,lambda:None)
            self.assertEqual(gone.call_count,5);drain.assert_called_once()
    def test_model_qualification_and_speed_claims_rejected(self):
        self.reject(lambda r:r[0].update(new_speedup_claim=True))
        self.reject(lambda r:r[0]['records'][1]['result'].update(full_model_qualified=True))

if __name__=='__main__':
    # Do not discover imported fixture tests: they have their own old contract.
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CrossChecks)
    sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
