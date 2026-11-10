"""Synthetic full driver/validator streams; no model imports or real tensor reads."""
import contextlib
import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_sdk_batch_layout_cpu_driver_v1 as driver
from research import minicpm5_layer1_sdk_batch_layout_results_v1 as results
from research import check_minicpm5_layer1_sdk_batch_layout_callable_v1 as fake
inputs=driver.inputs;core=driver.core;native=driver.native


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake.CallableChecks.setUpClass();f=fake.CallableChecks.fixture
        cls.resources={};cls.data={};cls.plan=dict(output_root='/synthetic/new',inputs={},
            layer1_sdk_batch_layout={'current_history':'synthetic'},layer1_official_history={'native':'synthetic'})
        cls.native_bundle=dict(binding=cls.plan['layer1_official_history'])
        cls.previous=dict(binding=cls.plan['layer1_sdk_batch_layout'],spec=dict(output_root='/synthetic/previous'),records={})
        cls.collector=dict(collector_plan={},collector_chain=dict(rank_results=[dict(rank=0),dict(rank=1)]),audit_result={})
        cls.events=[];cls.items={};cls.values=[]
        for rank in (0,1):
            for ci in core.boundary.DECODE_CALLS:
                group=int(ci>=11);step=ci-(10 if group else 3)
                fixtures=[6,1,3,4]if group else [2,7,5,0];indices=[22,21,23,20]if group else [6,5,7,4]
                case=copy.deepcopy(fake.CallableChecks.case)
                frames=[dict(index=indices[s]+4*(step-1),fixture=fixtures[s],slot=s,step=step)for s in range(4)]
                case.update(rank=rank,physical_frame_order=frames,call=dict(index=ci,group=group,phase='decode',frames=sorted(frames,key=lambda x:x['index'])),
                    original_call=dict(path='/synthetic/original/r%d/c%d.json'%(rank,ci),sha256='a'*64))
                prior=copy.deepcopy(f.prior);current=copy.deepcopy(f.current)
                for slot,frame in enumerate(frames):
                    prior[slot]['record'].update(frame=dict(frame,index=frame['index']-4,step=step-1),native_fixture_order=frame['fixture'],step=step-1)
                    current[slot]['record'].update(frame=frame,native_fixture_order=frame['fixture'],step=step)
                item=dict(case=case,normalized_raw=f.normalized,cache_raw=f.own,expected_attention_raw=fake.CallableChecks.expected,
                    audited_call=dict(payload_sha256='b'*64))
                prepared=core.prepare(case,f.normalized,f.own,prior,current)
                resource=dict(item=item,prior=prior,current=current,target=b'\x80\x3f'*8192,
                    refs=[dict(frame=frame,target='synthetic')for frame in frames],prepared=prepared,control_results={})
                for mode in ('own','restore'):
                    _,_,_,proof=core.factorial.select(case,f.normalized,f.own,prior,current,mode)
                    resource['control_results'][mode]=dict(schema='minicpm5_layer1_current_history_factorial_callable_result_v1',complete=True,
                        rank=rank,arm=core.ANCHOR,call=case['call'],mode=mode,intervention=proof,
                        attention_output_sha256=core.boundary.sha(item['expected_attention_raw']),all_four_rows_exact=True,
                        normalized_original_layout=True,full_input_cache_unchanged=True,original_official_weights_unchanged=True,
                        attention_reexecuted=True,model_reexecuted=False,attention_causality_proven=False,full_model_qualified=False,
                        new_speedup_claim=False,official_current_intervention_performed=False,official_history_intervention_performed=False,current_qkv_change_together=False)
                cls.resources[rank,ci]=resource
            stream_items=[];oldrows=[]
            for arm in inputs.stream.raw.ARMS:
                for ci in core.boundary.DECODE_CALLS:
                    resource=cls.resources[rank,ci];item=copy.deepcopy(resource['item']);item['case']['arm']=arm
                    stream_items.append(item)
                    for mode in core.factorial.MODES:
                        raw=resource['prepared']['normalized_raw']if mode=='official_both'else resource['item']['expected_attention_raw']
                        actual=copy.deepcopy(resource['control_results']['restore'if mode=='restore'else 'own'])
                        actual.update(arm=arm,mode=mode,attention_output_sha256=core.boundary.sha(raw))
                        if mode=='official_both':actual['intervention']=resource['prepared']['intervention']
                        path=str(Path('/synthetic/previous')/('cpu-rank%d'%rank)/arm/('call-%02d'%ci)/mode/'attention.bf16')
                        desc=dict(path=path,raw_bytes=16384,sha256=core.boundary.sha(raw),identity=dict(synthetic=path))
                        cls.plan['inputs'][path]=dict(sha256=desc['sha256'],identity=desc['identity']);cls.data[path]=raw
                        oldrows.append(dict(result=actual,output=desc))
            cls.items[rank]=stream_items
            hashes={key:core.boundary.sha(b'\x80\x3f')for key in native.WEIGHTS}
            cls.previous['records']['rank%d_result'%rank]=dict(schema='minicpm5_layer1_current_history_factorial_cpu_driver_v1',
                complete=True,rank=rank,records=oldrows,load_proof=dict(rank=rank,synthetic=True),
                official_layer1_shards=dict(rank=rank,hashes=hashes,all_four_projection_shards_raw_exact=True))
        with cls.patch_common():
            for rank in (0,1):
                ftest=fake.CallableChecks(methodName='test_ten_official_calls_one_b4_rope_four_complete_outputs');ftest.setUp()
                ftest.torch=fake.FakeTorch(rank)
                context=SimpleNamespace(_sources_checked=True,plan=cls.plan,permit=dict(rank=rank),guard=lambda role:None)
                def build(rank,phase):return ftest.model,cls.previous['records']['rank%d_result'%rank]['load_proof']
                def weights(context,torch,model,rank):return cls.previous['records']['rank%d_result'%rank]['official_layer1_shards']
                def control(torch,model,case,mode,*args):
                    resource=cls.resources[case['rank'],case['call']['index']]
                    cls.events.append(('control',case['rank'],case['call']['index'],mode))
                    return copy.deepcopy(resource['control_results'][mode]),resource['item']['expected_attention_raw']
                def bucket(model,value):
                    model.n_positions=value
                    return dict(bucket=value,bucket_index=0,phase='decode',n_positions=value,n_active_tokens=1,cache_aliases=84,configuration_unchanged=True)
                def save(context,case,variant,raw,guard):
                    cls.events.append(('save',case['rank'],case['call']['index'],variant))
                    path=str(Path(cls.plan['output_root'])/('cpu-rank%d'%case['rank'])/core.ANCHOR/('call-%02d'%case['call']['index'])/variant/'attention.bf16')
                    cls.data[path]=raw;return dict(path=path,raw_bytes=16384,sha256=core.boundary.sha(raw),identity={'synthetic':path})
                with ftest.patches(),patch.object(driver.constructor,'build',side_effect=build),patch.object(driver.shards,'verify_model',side_effect=weights),                    patch.object(native.original,'replay',side_effect=control),patch.object(driver.constructor,'select_bucket',side_effect=bucket),                    patch.object(driver,'save_output',side_effect=save):
                    result=driver.run_rank(context,ftest.torch,{},dict(rank=rank),{},lambda *a:{},lambda:None,cls.native_bundle,cls.previous)
                    cls.values.append(result)
        cls.baseline=cls.validate_values(cls.values)

    @classmethod
    def patch_common(cls,raw_changes=None):
        parent=cls;stack=contextlib.ExitStack();raw=dict(cls.data);raw.update(raw_changes or {})
        class Stream:
            def __init__(self,plan,result,audit,read_json,guard):self.rank=result['rank'];self.complete=False
            def __iter__(self):
                yield from parent.items[self.rank];self.complete=True
            def finish(self):
                if not self.complete:raise ValueError('Incomplete synthetic source')
                return dict(rank=self.rank,all_history_calls=42,decode_calls=18,attention_reexecuted=False,prefill_replayed=False)
        def sources(plan,data,case):
            r=parent.resources[case['rank'],case['call']['index']]
            return r['prior'],r['current'],r['target'],r['refs']
        def reader(desc,path,guard):
            guard();return raw[str(path)]
        for owner,name,value in [(inputs.stream,'RankStream',Stream),(driver.official,'load',lambda *a:{}),
            (core.factorial,'for_call',sources),(inputs.payload,'read_owned_payload',reader),
            (driver.constructor,'check_load_metadata',lambda *a:True)]:stack.enter_context(patch.object(owner,name,side_effect=value))
        return stack

    @classmethod
    def validate_values(cls,values,raw_changes=None):
        with cls.patch_common(raw_changes):
            return results.validate_rank_pair(values,cls.plan,cls.collector,cls.native_bundle,cls.previous,lambda *a:{},lambda:None)

    def rejected(self,change,raw_changes=None):
        values=copy.deepcopy(self.values);change(values)
        with self.assertRaises((ValueError,RuntimeError,KeyError,IndexError)):self.validate_values(values,raw_changes)

    def test_driver_replays_only_six_anchor_calls_per_rank_and_saves72full_outputs(self):
        saved=[x for x in self.events if x[0]=='save'];controls=[x for x in self.events if x[0]=='control']
        self.assertEqual(len(saved),72);self.assertEqual(len(controls),24)
        self.assertEqual({x[2]for x in saved},set(core.boundary.DECODE_CALLS))
        for result in self.values:
            self.assertEqual(len(result['records']),6);self.assertEqual(result['raw_output_bytes'],589824)
            self.assertFalse(result['other_arms_reexecuted'])
            self.assertEqual(result['source_stream']['skipped_decode_calls'],12)

    def test_complete_independent_pair_and_formula_checks(self):
        p=self.baseline
        self.assertEqual([p[k]for k in ('physical_attention_invocations','assembled_four_row_outputs','unique_decode_frames','raw_output_bytes')],[144,72,24,1179648])
        self.assertTrue(p['every_output_independently_reread']);self.assertTrue(p['every_view_formula_reconstructed'])
        self.assertFalse(p['rope_payload_independently_reread'])
        self.assertEqual(len(p['descriptive']['summary']),6)

    def test_skip_claim_count_and_stream_proof_cannot_be_promoted(self):
        self.rejected(lambda v:v[0].update(other_arms_reexecuted=True))
        self.rejected(lambda v:v[0].update(anchor_decode_calls=18))
        self.rejected(lambda v:v[0]['source_stream'].update(skipped_decode_calls=0))

    def test_missing_duplicate_reordered_outputs_rejected(self):
        self.rejected(lambda v:v[0]['records'][0]['outputs'].pop())
        self.rejected(lambda v:v[0]['records'][0]['outputs'].reverse())
        self.rejected(lambda v:v[0]['records'].pop())

    def test_wrong_history_control_raw_rejected_even_with_old_report_hash_unchanged(self):
        path=self.values[0]['records'][0]['result']['previous_controls']['official_both']['path']
        with self.assertRaises(ValueError):self.validate_values(self.values,{path:b'\x00\x3f'*8192})

    def test_changed_current_output_bytes_rejected_without_rehash(self):
        path=self.values[0]['records'][0]['outputs'][2]['output']['path']
        with self.assertRaises(ValueError):self.validate_values(self.values,{path:b'\x00\x3f'*8192})

    def test_wrong_physical_slot_or_mask_offset_formula_rejected(self):
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][2]['tasks'][1]['task']['seq_ids'].__setitem__(0,0))
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][2]['tasks'][1]['actual_inputs'][1].update(offset=0))

    def test_wrong_cache_stride_padding_source_or_normalized_holes_rejected(self):
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][2]['tasks'][0]['actual_inputs'][6]['stride'].__setitem__(0,16384))
        self.rejected(lambda v:v[0]['records'][0]['result']['intervention']['history_intervention']['output_cache_sha256'].update({core.boundary.CACHE_NAMES[0]:'0'*64}))
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][2]['tasks'][0]['task'].update(normalized_unselected_backing_bytes=0))

    def test_wrong_rope_offset_digest_or_recomputation_claim_rejected(self):
        self.rejected(lambda v:v[0]['records'][0]['result']['variants'].update(official_rope_evaluations=4))
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][3]['tasks'][1]['actual_inputs'][4].update(offset=0))
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][1]['tasks'][0]['selected_rope'][0].update(sha256='0'*64))

    def test_part_output_digest_and_b1_assembly_are_checked(self):
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][2]['tasks'][1]['output'].update(sha256='0'*64))
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['variants'][2]['tasks'].reverse())

    def test_old_control_metadata_and_own_proof_cannot_change(self):
        self.rejected(lambda v:v[0]['records'][0]['result']['previous_controls']['own'].update(path='/another/output'))
        self.rejected(lambda v:v[0]['records'][0]['result']['own_result'].update(all_four_rows_exact=False))
        self.rejected(lambda v:v[0]['records'][0]['result']['restore_result']['intervention'].update(mode='own'))

    def test_scope_original_weights_target_and_qualification_rejected(self):
        self.rejected(lambda v:v[0]['records'][0]['result']['variants']['scope'].update(collective_dtype='torch.float32'))
        self.rejected(lambda v:v[0]['official_layer1_shards'].update(all_four_projection_shards_raw_exact=False))
        self.rejected(lambda v:v[0]['records'][0].update(official_target_sha256='0'*64))
        self.rejected(lambda v:v[0].update(full_model_qualified=True))

    def changed_variant(self,ranks):
        values=copy.deepcopy(self.values);changed={};raw=b'\x80\x3f'*8192
        for rank in ranks:
            entry=values[rank]['records'][0];desc=entry['outputs'][4]['output'];changed[desc['path']]=raw
            desc['sha256']=core.boundary.sha(raw);variant=entry['result']['variants']['variants'][3]
            variant['sha256']=core.boundary.sha(raw)
            for task in variant['tasks']:task['output']['sha256']=core.boundary.sha(raw[:4096])
        return values,changed

    def test_rehashed_valid_variant_must_still_replicate_across_ranks(self):
        values,changed=self.changed_variant([1])
        with self.assertRaisesRegex(ValueError,'Replicated outputs identical'):self.validate_values(values,changed)

    def test_matched_variant_difference_is_reported_without_forcing_native_equality(self):
        values,changed=self.changed_variant([0,1]);proof=self.validate_values(values,changed)
        summary=next(v for v in proof['descriptive']['summary']if v['variant']=='b1_contiguous')
        self.assertEqual(summary['raw_exact_frames'],4)
        self.assertEqual(summary['improved_frames'],4)
        self.assertFalse(proof['full_model_qualified']);self.assertFalse(proof['new_speedup_claim'])

    def test_guard_before_stream_oldpayload_driver_or_validator_access(self):
        def deny():raise RuntimeError('No admission')
        with self.assertRaises(RuntimeError):inputs.AnchorStream(None,None,None,None,deny)
        with self.assertRaises(RuntimeError):inputs.previous_controls(None,None,None,deny)
        with self.assertRaises(RuntimeError):driver.run_rank(None,None,None,None,None,None,deny,None,None)
        with self.assertRaises(RuntimeError):results.validate_rank_pair(None,None,None,None,None,None,deny)

    def test_anchor_stream_requires_exhausting_skipped_arms(self):
        with self.patch_common():
            source=inputs.AnchorStream({},dict(rank=0),{},lambda *a:{},lambda:None)
            it=iter(source)
            for _ in range(6):next(it)
            with self.assertRaises(ValueError):source.finish()
            self.assertEqual(list(it),[]);self.assertEqual(source.finish()['skipped_decode_calls'],12)

    def test_real_save_function_on_synthetic_bytes_is_exclusive_and_reread(self):
        with tempfile.TemporaryDirectory()as tmp:
            context=SimpleNamespace(plan=dict(output_root=tmp),permit=dict(rank=0),guard=lambda role:None)
            case=self.resources[0,4]['item']['case'];raw=bytes(16384)
            desc=driver.save_output(context,case,'own',raw,lambda:None)
            self.assertEqual(Path(desc['path']).read_bytes(),raw);self.assertEqual(desc['raw_bytes'],16384)
            with self.assertRaises(FileExistsError):driver.save_output(context,case,'own',raw,lambda:None)

    def test_no_native_dependencies_imported(self):
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
