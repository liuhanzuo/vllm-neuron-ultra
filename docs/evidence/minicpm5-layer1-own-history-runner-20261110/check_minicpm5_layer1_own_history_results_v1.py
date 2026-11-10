"""Synthetic replay output bytes with original JSON templates; no model reads."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_own_history_results_v1 as verify


def read_ref(ref):
    path=Path(ref['path']);assert path.suffix=='.json'and path.stat().st_size<8*2**20
    data=path.read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256'];return json.loads(data)


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='mini-own-history-result-synthetic-');cls.root=Path(cls.tmp.name)
        cls.plan=read_ref(dict(path=str(Q/'minicpm5-layer1-own-history-driver-source-plan-20261110-v1.json'),sha256='bf3a581007ee578566f171335c1ae60dc3d799a9e951c8bc2336f69a6f6d0705'))
        cls.plan['output_root']=str(cls.root);cls.plan['layer1_own_history_replay']=dict(synthetic_exit_lineage=True)
        cls.originals={};cls.results=[];cls.audited=[dict(payload_sha256='b'*64)for _ in range(84)]
        source_exit=read_ref(cls.plan['layer1_observation']['original_exit'])
        header=read_ref(cls.plan['safetensors_header'])
        slices=verify.boundary.weight_slices(cls.plan,header)
        for rank in (0,1):
            original_rank=read_ref(source_exit['outputs']['rank'+str(rank)+'_result'])
            arm=original_rank['arm_reports'][0];arm=read_ref(dict(path=arm['report'],sha256=arm['sha256']))
            load=copy.deepcopy(arm['constructors'][1]['proof'])
            weights=[];hashes={}
            for row in slices[rank*4:(rank+1)*4]:
                key=verify.shards.local_key(row['official_key'].split('.')[-2]);shape=list(row['header_entry']['shape']);shape[row['shard_axis']]//=2
                digest=hashlib.sha256((str(rank)+key).encode()).hexdigest();hashes[key]=digest
                weights.append(dict(official_key=row['official_key'],local_key=key,rank=rank,shard_axis=row['shard_axis'],
                    start=row['start'],stop=row['stop'],shape=shape,shard_bytes=2*shape[0]*shape[1],all_raw_bytes_equal=True,
                    shard_sha256=digest,official_matrix_sha256=hashlib.sha256(key.encode()).hexdigest()))
            path=cls.plan['official_layer0_weights']['path']
            weight_proof=dict(schema='minicpm5_layer1_official_shards_v1',complete=True,rank=rank,
                all_four_projection_shards_raw_exact=True,official_file=dict(path=path,**cls.plan['inputs'][path]),
                header=cls.plan['safetensors_header'],precision_changed=False,weights=weights,hashes=hashes)
            stream=dict(schema='minicpm5_layer1_rank_decode_stream_v1',complete=True,rank=rank,all_history_calls=42,
                decode_calls=18,physical_decode_rows=72,full_cache_records=84,restored_full_cache_bytes=84*4325376,
                all_read_bytes_match_saved_audit=True,prefill_replayed=False)
            result=dict(schema='minicpm5_layer1_own_history_cpu_driver_v1',complete=True,rank=rank,
                load_proof=load,official_layer1_shards=weight_proof,source_stream=stream,completed_layer1_data=cls.plan['layer1_own_history_replay'],
                replay_calls=18,physical_rows=72,raw_output_bytes=294912,records=[],complete_original_model_constructed=True,
                full_model_forward_reexecuted=False,layer1_attention_reexecuted=True,all_own_history_outputs_exact=True,
                paired_history_intervention_performed=False,attention_causality_proven=False,
                full_model_qualified=False,new_speedup_claim=False,self_exit_claimed=False)
            for ai,armname in enumerate(verify.payload.raw.ARMS):
                for ci in verify.boundary.DECODE_CALLS:
                    ref=cls.plan['layer1_observation']['original_calls'][rank*42+ai*14+ci]
                    original=read_ref(ref);data=bytes([ci+ai])*16384;sha=hashlib.sha256(data).hexdigest()
                    original['stage_sha256'][verify.boundary.ATTENTION]=sha;cls.originals[ref['path']]=original
                    output=cls.root/('cpu-rank'+str(rank))/armname/('call-%02d'%ci)/'attention.bf16';output.parent.mkdir(parents=True);output.write_bytes(data)
                    descriptor=dict(path=str(output),raw_bytes=len(data),sha256=sha,
                        identity=verify.payload.file_identity(output,output.lstat()))
                    native=dict(schema='minicpm5_layer1_own_history_callable_result_v1',complete=True,rank=rank,
                        arm=armname,call=original['call'],all_four_rows_exact=True,normalized_original_layout=True,
                        full_input_cache_unchanged=True,original_official_weights_unchanged=True,attention_reexecuted=True,
                        model_reexecuted=False,paired_history_intervention_performed=False,attention_causality_proven=False,
                        full_model_qualified=False,new_speedup_claim=False,attention_output_sha256=sha)
                    bucket=original['bucket'];result['records'].append(dict(result=native,output=descriptor,
                        bucket=dict(bucket=bucket,bucket_index=verify.constructor.BUCKETS.index(bucket),phase='decode',
                            n_positions=bucket,n_active_tokens=1,cache_aliases=84,configuration_unchanged=True),
                        collected_original_call={k:ref[k]for k in ('path','sha256')},independent_audit_payload_sha256='b'*64))
            cls.results.append(result)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def read(self,ref):return copy.deepcopy(self.originals[ref['path']])if ref['path']in self.originals else read_ref(ref)
    def validate(self,results=None):
        # Actual lineage is tested independently. This synthetic output fixture
        # authenticates original JSON then substitutes generated attention hashes.
        with patch.object(verify.p,'read_ref',side_effect=self.read),patch.object(verify.history,'completed',
                return_value=dict(audit_result=dict(layer1_cache=dict(raw_records=self.audited)))):
            return verify.validate_rank_pair(results or self.results,self.plan,lambda:None)
    def test_all36_full_outputs_independently_reread_and_rank_replicas_counted_once(self):
        proof=self.validate()
        self.assertEqual((proof['decode_calls'],proof['physical_rows'],proof['unique_replicated_output_rows'],proof['raw_bytes']),(36,144,72,589824))
        self.assertTrue(proof['every_output_independently_reread']);self.assertFalse(proof['attention_causality_proven'])
        self.assertFalse(proof['full_model_qualified']);self.assertFalse(proof['new_speedup_claim'])
    def reject(self,mutate):
        results=copy.deepcopy(self.results);mutate(results)
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):self.validate(results)
    def test_missing_or_reordered_rank_and_calls_rejected(self):
        self.reject(lambda r:r.pop());self.reject(lambda r:r.reverse())
        self.reject(lambda r:r[1]['records'].pop());self.reject(lambda r:r[0]['records'].reverse())
    def test_wrong_native_geometry_or_failed_contract_rejected(self):
        self.reject(lambda r:r[0]['records'][0]['result'].update(all_four_rows_exact=False))
        self.reject(lambda r:r[0]['records'][0]['result'].update(full_input_cache_unchanged=False))
        self.reject(lambda r:r[0]['records'][0]['output'].update(raw_bytes=4096))
    def test_wrong_original_call_or_audit_payload_provenance_rejected(self):
        self.reject(lambda r:r[0]['records'][0].update(collected_original_call=dict(path='/fake',sha256='a'*64)))
        self.reject(lambda r:r[0]['records'][0].update(independent_audit_payload_sha256='0'*64))
    def test_wrong_bucket_or_changed_full_model_constructor_rejected(self):
        self.reject(lambda r:r[0]['records'][0]['bucket'].update(n_active_tokens=128))
        self.reject(lambda r:r[0]['load_proof'].update(sdk_math_modified=True))
        self.reject(lambda r:r[0]['load_proof'].update(rank=1))
    def test_missing_whole_cache_or_original_history_rejected(self):
        self.reject(lambda r:r[0]['source_stream'].update(all_history_calls=41))
        self.reject(lambda r:r[1]['source_stream'].update(restored_full_cache_bytes=84*4096*128*2))
    def test_wrong_official_shard_axis_rank_or_matrix_rejected(self):
        self.reject(lambda r:r[0]['official_layer1_shards']['weights'][3].update(shard_axis=0))
        self.reject(lambda r:r[1]['official_layer1_shards']['weights'][0].update(rank=0))
        self.reject(lambda r:r[1]['official_layer1_shards']['weights'][0].update(official_matrix_sha256='0'*64))
    def test_missing_official_projection_or_precision_change_rejected(self):
        self.reject(lambda r:r[0]['official_layer1_shards']['weights'].pop())
        self.reject(lambda r:r[1]['official_layer1_shards'].update(precision_changed=True))
    def test_cross_rank_output_path_cannot_replace_original(self):
        self.reject(lambda r:r[0]['records'][0].update(output=copy.deepcopy(r[1]['records'][0]['output'])))
    def test_model_qualification_or_causality_promotion_rejected(self):
        self.reject(lambda r:r[0].update(full_model_qualified=True))
        self.reject(lambda r:r[1]['records'][0]['result'].update(attention_causality_proven=True))
    def test_recomputed_payload_digest_cannot_hide_changed_output(self):
        results=copy.deepcopy(self.results);row=results[0]['records'][0];path=Path(row['output']['path']);original=path.read_bytes()
        try:
            data=b'!'+original[1:];path.write_bytes(data)
            row['output'].update(sha256=hashlib.sha256(data).hexdigest(),identity=verify.payload.file_identity(path,path.lstat()))
            with self.assertRaisesRegex(ValueError,'owned attention output'):self.validate(results)
        finally:
            path.write_bytes(original);self.results[0]['records'][0]['output']['identity']=verify.payload.file_identity(path,path.lstat())
    def test_guard_denial_precedes_real_output_reads(self):
        def denied():raise RuntimeError('busy')
        with patch.object(verify.payload,'read_owned_payload')as read,self.assertRaisesRegex(RuntimeError,'busy'):
            verify.validate_rank_pair(None,None,denied)
        read.assert_not_called()


if __name__=='__main__':
    assert not {'torch','numpy','torch_neuronx','transformers'}&set(sys.modules)
    unittest.main(verbosity=2)
