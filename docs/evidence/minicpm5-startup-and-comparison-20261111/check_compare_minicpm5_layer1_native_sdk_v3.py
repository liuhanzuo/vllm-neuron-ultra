"""Full24-frame/2-rank/624+936 synthetic saved-byte comparison, no actual IO."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import compare_minicpm5_layer1_native_sdk_v3 as c
from research import check_minicpm5_layer1_operator_native_results_v2 as previous


def pack(value,dtype):
    if dtype=='torch.bfloat16':return struct.pack('<I',struct.unpack('<I',struct.pack('<f',value))[0])[2:]
    return struct.pack({'torch.float32':'<f','torch.int64':'<q','torch.bool':'<B'}[dtype],value)


def contiguous(shape):return [math.prod(shape[i+1:])for i in range(len(shape))]


def fixture():
    # Reverse logical fixture identities within each physical batch, without changing slots.
    original=previous.old.fixture
    def unordered():
        plan,native=original()
        for frame in plan['frames']:frame['fixture']=frame['fixture']//4*4+3-frame['fixture']%4
        # Original rows and call frames share these exact frame objects. Rebuild lengths and metadata consistently.
        for row in native['records']:
            frame=row['frame'];length=frame['fixture']+2+frame['step'];row['cache_length']=length
            for name in ('key','value'):
                row['cache_metadata'][name]=dict(shape=[1,2,length,128],stride=[length*256,length*128,128,1],offset=0,dtype='torch.bfloat16')
                row['outputs'][name]['raw_bytes']=length*512
        for call in plan['calls']:call['logical_positions']=[f['fixture']+1+f['step']for f in call['frames']]
        return plan,native
    with patch.object(previous.old,'fixture',side_effect=unordered):driver,plan,native,shards,mem=previous.fixture();cases=[];results=[];sdk_plan=dict(output_root='/synthetic/sdk-output',sdk_operator_manifest=c.sdk_manifest.manifest())
    # Nonuniform head values catch wrong rank slices; normalized rows catch fixture/slot confusion.
    def native_bytes(label,case,wanted):
        if label=='normalized_input':return pack(case['frame']['fixture']+1,wanted['dtype'])*2048
        if label in ('q_projection','q_rope_input','q_post_rope'):
            return b''.join(pack(h+1,wanted['dtype'])*128 for h in range(16))
        return None
    for report,case in zip(driver['records'],plan['native_operator_manifest']['frames']):
        for arm in report['arms']:
            ref=arm['result'];row=json.loads(mem.data[ref['path']])
            if arm['arm']=='full_trace':
                for event in row['capture']['snapshots']:
                    wanted=next(x['value']for x in case['events']if x['label']==event['label'])
                    if type(wanted)is not dict or'shape'not in wanted:continue
                    raw=native_bytes(event['label'],case,wanted)
                    if raw is not None:
                        old=event['value']['reference'];mem.data[old['path']]=raw;old['sha256']=c.sha(raw)
            raw=previous.encoded(row);mem.data[ref['path']]=raw;ref['bytes']=len(raw);ref['sha256']=c.sha(raw)
    full=plan['native_operator_manifest']['frames']
    for rank in (0,1):
        records=[];rank_cases=[]
        for call,bucket in zip(c.sdk_manifest.CALLS,c.sdk_manifest.BUCKETS):
            chosen=sorted([x for x in full if x['call']==call],key=lambda x:x['frame']['slot']);frames=[x['frame']for x in chosen]
            original_call=next(x for x in plan['calls']if x['index']==call)
            case=dict(rank=rank,call=original_call,bucket=bucket,arm='strict_anchor_s0',physical_frame_order=frames,
                positions=[x['logical_position']for x in chosen]);rank_cases.append(case)
            manifest=c.sdk_manifest.call_manifest(rank,call,bucket);events=[]
            folder=sdk_plan['output_root']+'/cpu-rank%d/strict_anchor_s0/call-%02d'%(rank,call)
            for index,event in enumerate(manifest['tensor_events']):
                context=dict(arm='C_Hd_control',rank=rank,call=original_call,frames=frames,phase='decode',thread=100+rank,route='sdk_decode',event_index=index)
                value={}if event['tensors']and event['tensors'][0]['path']and isinstance(event['tensors'][0]['path'][0],str)else([]if event['tensors']and event['tensors'][0]['path']else{})
                for j,wanted in enumerate(event['tensors']):
                    shape=wanted['shape'];dtype=wanted['dtype'];label=event['label'];raw=bytes(wanted['bytes'])
                    if label in ('position_ids','rope_position_ids')or(label=='route_inputs_sdk_decode'and wanted['path']==['position_ids']):
                        raw=struct.pack('<4q',*case['positions'])
                    if label in ('attention_mask','prior_actual_mask')or(label=='route_inputs_sdk_decode'and wanted['path']==['attention_mask']):
                        raw=b''.join(bytes([1]*p+[0]*(bucket-p))for p in case['positions'])
                    if label=='normalized_input':raw=b''.join(pack(f['fixture']+1,dtype)*2048 for f in frames)
                    if label in ('q_projection','q_rope_input','q_post_rope','q_after_prep')or(label=='route_inputs_sdk_decode'and wanted['path']==['Q']):
                        raw=b''.join(pack(h+1,dtype)*128 for h in range(rank*8,(rank+1)*8))*4
                    if label in ('qk0_unscaled','qk0_scaled','prior_mask_input','prior_mask_output','softmax_input'):
                        # A distinctive nonzero SDK-only padded suffix must be read but excluded from valid comparison.
                        raw=b''.join((bytes(p*c.WIDTH[dtype])+pack(-2,dtype)*(bucket-p))*8 for p in case['positions'])
                    path=folder+'/official_full/capture/%02d-%s-%02d.bin'%(index,label,j);ref=mem.save(path,raw)
                    ctx=dict(context,event_label=label,tensor_index=j,tensor_path=wanted['path'])
                    meta=dict(context=ctx,source_shape=shape,selected_shape=shape,source_stride=contiguous(shape),source_offset=0,
                        source_dtype=dtype,source_numel=math.prod(shape),selected_bytes=len(raw),selection=wanted['selection'],cpu=True,owned_clone=True)
                    leaf=dict(metadata=meta,reference=ref)
                    if not wanted['path']:value=leaf
                    else:
                        node=value
                        for n,part in enumerate(wanted['path']):
                            last=n==len(wanted['path'])-1;new=leaf if last else ([]if isinstance(wanted['path'][n+1],int)else{})
                            if isinstance(node,list):
                                while len(node)<=part:node.append(None)
                                if node[part]is None:node[part]=new
                            else:
                                if part not in node:node[part]=new
                            node=node[part]
                if event['label'].endswith('_scale_operation'):value=dict(operator='python.divide',scalar=128**.5,scaled_value_recomputed=False)
                events.append(dict(label=event['label'],context=context,value=value))
            capture=dict(summary=dict(complete=True,frames=frames,call=original_call),snapshots=dict(complete=True,manifest=manifest,records=events))
            controls=[dict(variant=mode,output=mem.save(folder+'/'+mode+'/attention.bf16',bytes(16384)))for mode in c.sdk_manifest.MODES]
            observations=[dict(mode=mode,capture=capture if mode=='official_full'else None)for mode in c.sdk_manifest.MODES[1:-1]]
            value=dict(schema='minicpm5_layer1_operator_sdk_callable_result_v2',complete=True,rank=rank,arm='strict_anchor_s0',call=original_call,
                source_case_sha256=c.native_manifest.digest(case),observations=observations)
            records.append(dict(result=value,outputs=controls))
        results.append(dict(schema='minicpm5_layer1_operator_sdk_driver_v2',complete=True,rank=rank,manifest=c.sdk_manifest.manifest(),records=records,
            official_layer1_shards=shards[rank]['official_layer1_shards']));cases.append(rank_cases)
    proof=dict(schema='minicpm5_layer1_operator_native_independent_verification_v2',complete=True,frames=24,direct_attention_calls=72,raw_files=936,
        journals=72,arm_result_files=72,full_model_qualified=False,new_speedup_claim=False,
        frames_verified=[dict(frame=x['frame'],all_three_arms_exact=True,official_attention_sha256=c.sha(bytes(4096)))for x in full])
    audit=dict(schema='minicpm5_native_post_exit_independent_audit_v1',complete=True,actual_native_model_exit_verified=True,actual_native_recorder_exit_verified=True,
        every_raw_byte_reread_after_all_model_processes_exited=True,result=proof,full_model_qualified=False,new_speedup_claim=False)
    sp=dict(schema='minicpm5_layer1_operator_sdk_pair_verification_v1',complete=True,ranks=2,unique_checkpoints=24,physical_attention_invocations=60,
        raw_capture_files=564,raw_control_files=60,raw_bytes=c.sdk_manifest.manifest()['total_raw_bytes'],every_raw_payload_reread=True,
        all_endpoint_controls_full_bytes_exact=True,both_rank_reduced_outputs_raw_exact=True,full_model_qualified=False,new_speedup_claim=False)
    recovery=dict(schema='minicpm5_operator_sdk_failed_run_recovery_audit_v1',audit_complete=True,original_exit_code=1,original_runtime_successful=False,
        every_raw_byte_reread_after_all_model_processes_exited=True,result=sp,full_model_qualified=False,new_speedup_claim=False)
    return driver,plan,audit,results,cases,sdk_plan,recovery,mem


class Comparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=fixture()
    def test_full24frames48rankframes_936_and624_raw_all_read_once_and_padding_retained(self):
        *args,mem=self.f;mem.reads.clear();result=c.compare(*args,mem.read,lambda:None)
        self.assertEqual(result['raw_files'],dict(native=936,sdk=624));self.assertEqual(len(mem.reads),1632)
        self.assertEqual(len(set(mem.reads)),1632);self.assertEqual(len(result['comparisons']),48)
        self.assertIsNone(result['first_observed_difference']);self.assertEqual(result['original_sdk_model_exit_code'],1)
        for row in result['comparisons']:
            self.assertEqual(set(row['unknown_boundaries']),{'pv_prior','pv_active','o_proj_local_before_reduce'})
            self.assertTrue(any(p['nonzero_elements']>0 for p in row['sdk_padding']))
            self.assertEqual(row['alignment']['q_head_ids'],list(range(row['rank']*8,(row['rank']+1)*8)))
        self.assertNotEqual(result['comparisons'][0]['frame']['fixture'],result['comparisons'][0]['slot'])
        self.assertEqual(result['comparisons'][0]['observed_scale_operations']['sdk']['qk0_scale_operation']['operator'],'python.divide')
        self.assertFalse(result['arithmetic_reexecuted']);self.assertFalse(result['full_model_qualified'])
        self.assertFalse({'torch','numpy','transformers','torch_neuronx','neuronxcc'}&set(sys.modules))
    def test_guard_or_incomplete_audit_rejects_before_read(self):
        *args,mem=self.f;mem.reads.clear()
        def deny():raise RuntimeError('no actual completed admission')
        with self.assertRaisesRegex(RuntimeError,'admission'):c.compare(*args,mem.read,deny)
        self.assertEqual(mem.reads,[])
        bad=copy.deepcopy(args[2]);bad['complete']=False;args[2]=bad
        with self.assertRaises(RuntimeError):c.compare(*args,mem.read,lambda:None)
        self.assertEqual(mem.reads,[])
    def test_unmatched_frame_slot_or_head_not_guessed(self):
        driver,plan,audit,sdk,cases,*_=self.f;case=plan['native_operator_manifest']['frames'][0]
        for fault in ('slot','position','heads'):
            sc=copy.deepcopy(cases[0][0]);sh=copy.deepcopy(sdk[0]['official_layer1_shards'])
            if fault=='slot':sc['physical_frame_order'].reverse()
            elif fault=='position':sc['positions'][0]+=1
            else:sh['q_head_ids']=list(range(8,16))
            with self.subTest(fault=fault),self.assertRaises(RuntimeError):c.alignment.frame_alignment(sc,case,sh)
    def test_statistics_compare_all_values_dtype_change_and_zero_reference(self):
        def t(values,dtype):return dict(raw=b''.join(pack(x,dtype)for x in values),shape=[len(values)],dtype=dtype,evidence=[])
        result=c.compare_values(t([1,2,3],'torch.bfloat16'),t([1,2,5],'torch.bfloat16'),lambda:None)
        self.assertEqual(result['first_numeric_difference_flat_index'],2);self.assertEqual(result['maximum_absolute_error'],2)
        self.assertAlmostEqual(result['relative_l2'],2/math.sqrt(14))
        result=c.compare_values(t([1,2],'torch.bfloat16'),t([1,2],'torch.float32'),lambda:None)
        self.assertIsNone(result['bytes_equal']);self.assertTrue(result['values_equal'])
        result=c.compare_values(t([0,0],'torch.bfloat16'),t([0,1],'torch.bfloat16'),lambda:None)
        self.assertIsNone(result['relative_l2']);self.assertTrue(result['relative_l2_undefined_zero_reference'])
    def test_logical_slice_ignores_original_stride_and_join_only_concatenates(self):
        value=dict(raw=b''.join(pack(x,'torch.bfloat16')for x in range(24)),shape=[2,3,4],dtype='torch.bfloat16',
            evidence=[{'metadata':{'source_stride':[1,2,6],'source_offset':999}}])
        cut=c.select(value,[[1,2],[1,3],[1,3]])
        self.assertEqual(list(c.numbers(cut['raw'],cut['dtype'],lambda:None)),[17,18,21,22])
        a=c.select(value,[[1,2],[0,3],[0,2]]);b=c.select(value,[[1,2],[0,3],[2,4]])
        self.assertEqual(c.join(a,b,2)['raw'],value['raw'][24:])
        again=c.select(cut,[[0,1],[0,1],[0,2]])
        self.assertEqual(len(again['evidence'][0]['logical_selection_chain']),2)
        self.assertEqual(again['evidence'][0]['logical_selection_chain'][0]['ranges'][0],[1,2])
    def test_actual_frame_first_difference_is_full_qk_values_with_adjacent_stages_retained(self):
        driver,plan,audit,sdk,cases,sdk_plan,recovery,mem=self.f
        reader=c.Reader(mem.read,lambda:None);native=c.load_native(driver,plan,audit['result'],reader)
        decoded=c.load_sdk(sdk[1],cases[1],sdk_plan,reader);case=plan['native_operator_manifest']['frames'][0]
        events=decoded[case['call']]['events'];value=events['qk0_scaled',()]
        changed=bytearray(value['raw']);changed[0:2]=pack(1,'torch.bfloat16');events['qk0_scaled',()]=dict(value,raw=bytes(changed))
        result=c.compare_frame(native[case['frame']['index']],decoded[case['call']],sdk[1]['official_layer1_shards'],lambda:None)
        self.assertEqual(result['first_observed_difference'],'qk0_scaled')
        by={r['boundary']:r for r in result['boundaries']}
        self.assertEqual(by['qk0_unscaled']['status'],'equal');self.assertEqual(by['qk0_scaled']['numeric_mismatch_elements'],1)
        self.assertEqual(by['qk0_scaled']['maximum_absolute_error'],1)
        self.assertEqual(by['softmax_input']['status'],'equal')
        self.assertFalse(result['first_difference_is_causal_localization'])
    def test_reader_rejects_corrupt_short_duplicate_and_nonfinite_data(self):
        raw=b'12';ref=dict(path='/synthetic/x',raw_bytes=2,sha256=c.sha(raw))
        for data in (b'1',b'13'):
            reader=c.Reader(lambda *a:data,lambda:None)
            with self.assertRaises(RuntimeError):reader.get(ref,ref['path'],2,'native')
        reader=c.Reader(lambda *a:raw,lambda:None);reader.get(ref,ref['path'],2,'native')
        with self.assertRaises(RuntimeError):reader.get(ref,ref['path'],2,'native')
        with self.assertRaises(RuntimeError):list(c.numbers(b'\x80\x7f','torch.bfloat16',lambda:None))


if __name__=='__main__':unittest.main(verbosity=2)
