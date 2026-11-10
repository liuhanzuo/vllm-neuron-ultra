"""Independent complete raw-output, view-formula and anchor-only scope verifier.

No native imports or admission. Caller authenticates completed collector/native/
current-history bundles and retains its guard. All output and old control bytes
are reread. RoPE raw was not saved by the frozen callable: only its runtime byte
checks, repeated digests and view formulas can be checked here independently.
"""
import copy
from pathlib import Path
import re
import struct
from research import minicpm5_layer1_sdk_batch_layout_inputs_v1 as inputs
from research import minicpm5_layer1_sdk_batch_layout_callable_v1 as native
from research import minicpm5_layer1_sdk_batch_layout_analysis_v1 as analysis
from research import minicpm5_layer1_official_history_inputs_v1 as official
import minicpm5_layer0_cpu_constructor_v1 as constructor
core=inputs.core;require=core.require;boundary=core.boundary


def meta(raw,shape,stride,offset=0,dtype='torch.bfloat16'):
    return dict(shape=shape,stride=stride,offset=offset,dtype=dtype,sha256=boundary.sha(raw))


def diagnostic(value,false,true):
    require(all(value[k]is False for k in false),'Every diagnostic qualification boundary retained')
    require(all(value[k]is True for k in true),'Every required complete execution guarantee retained')


def scope_expected(case):
    config=dict(batch_size=4,max_batch_size=4,tkg_batch_size=4,kv_cache_batch_size=4,
        n_active_tokens=1,is_prefill_stage=False,token_generation_batches=None,rpl_reduce_dtype=None,
        sequence_parallel_enabled=False,is_prefix_caching=False,fused_qkv=False,
        qkv_kernel_enabled=False,qkv_nki_kernel_enabled=False,out_proj_kernel_enabled=False)
    return dict(configuration=config,model_batch_size=4,torch_dtype='torch.bfloat16',
        attention_dtype='torch.bfloat16',collective_dtype='torch.bfloat16',bucket=case['bucket'],rank=case['rank'],layers=42,tp=2)


def check_original_control(value,case,mode,proof,raw):
    require(value['schema']=='minicpm5_layer1_current_history_factorial_callable_result_v1'
        and value['complete']is True and [value[k]for k in ('rank','arm','call','mode')]
        ==[case['rank'],core.ANCHOR,case['call'],mode] and value['intervention']==proof
        and value['attention_output_sha256']==boundary.sha(raw),'Exact original own/restore callable result')
    diagnostic(value,('model_reexecuted','attention_causality_proven','full_model_qualified','new_speedup_claim',
        'official_current_intervention_performed','official_history_intervention_performed','current_qkv_change_together'),
        ('all_four_rows_exact','normalized_original_layout','full_input_cache_unchanged','original_official_weights_unchanged','attention_reexecuted'))


def check_call(result,item,donors,current,previous,outputs,guard):
    guard();case=item['case'];prepared=core.prepare(case,item['normalized_raw'],item['cache_raw'],donors,current)
    require(result['schema']=='minicpm5_layer1_sdk_batch_layout_callable_result_v1' and result['complete']is True
        and [result[k]for k in ('rank','arm','call','original_call')]==[case['rank'],core.ANCHOR,case['call'],case['original_call']]
        and result['source_case_sha256']==prepared['source_case_sha256'] and result['intervention']==prepared['intervention'],
        'Exact original anchor case and reconstructed official normalized/history state')
    require([result[k]for k in ('physical_attention_invocations','full_B4_invocations','B1_invocations',
        'assembled_four_row_outputs','raw_output_bytes')]==[12,4,8,6,98304], 'Exact twelve invocations and six complete outputs')
    diagnostic(result,('full_model_reexecuted','native_reference_reexecuted','attention_causality_proven','full_model_qualified','new_speedup_claim'),
        ('all_original_precision_and_model_configuration_preserved','same_B4_RoPE_used_for_all_official_variants'))
    refs=native._previous(case,previous,prepared)
    require(result['previous_controls']==refs,'Exact independently reread predecessor descriptors')
    require(set(outputs)==set(native.ORDER),'All six complete output tensors')
    expected_controls=core.verify_controls(item['expected_attention_raw'],previous['own']['raw'],outputs['own'],
        previous['official_both']['raw'],outputs['b4_strided'],outputs['restore'])
    require(result['controls']==expected_controls and previous['restore']['raw']==outputs['restore'],
        'All original own/restore and previous both-official complete bytes exact')
    for mode,key in (('own','own_result'),('restore','restore_result')):
        guard();_,_,_,proof=core.factorial.select(case,item['normalized_raw'],item['cache_raw'],donors,current,mode)
        check_original_control(result[key],case,mode,proof,outputs[mode])
    variants=result['variants'];require(variants['scope']==scope_expected(case)
        and variants['official_rope_evaluations']==1 and variants['physical_attention_invocations']==10
        and variants['all_backing_and_cache_bytes_unchanged']is True,'Original full B4 scope and single official RoPE pair')
    rope=variants['official_rope']
    require(type(rope)is list and len(rope)==2,'Both original full B4 positional caches')
    for m in rope:
        require(set(m)=={'shape','stride','offset','dtype','sha256'} and m['shape']==[4,1,128]
            and m['dtype']=='torch.bfloat16' and len(m['stride'])==3
            and all(type(s)is int and s>0 for s in m['stride']) and type(m['offset'])is int and m['offset']>=0
            and type(m['sha256'])is str and re.fullmatch('[a-f0-9]{64}',m['sha256']),
            'Actual full B4 RoPE metadata without inventing raw storage')
    require([v['variant']for v in variants['variants']]==list(core.VARIANTS),'All four independent shape/layout variants in order')
    selected_rope={}
    for variant,entry in zip(core.VARIANTS,variants['variants']):
        guard();recipes=core.tasks(prepared,variant);raw=outputs[variant]
        require(entry['physical_invocations']==len(recipes) and entry['raw_bytes']==16384
            and entry['sha256']==boundary.sha(raw) and len(entry['tasks'])==len(recipes),
            'Every full output and every actual B1/B4 task retained')
        for task,observed in zip(recipes,entry['tasks']):
            require(observed['task']=={k:v for k,v in task.items()if k!='mask_raw'},'All independently reconstructed view and arithmetic formulas')
            slots=task['slots'];batch=len(slots);slot=slots[0];actual=observed['actual_inputs']
            require(len(actual)==8,'All actual normalized/mask/position/slot/RoPE/K/V input metadata')
            row_input=b''.join(prepared['normalized_raw'][s*4096:(s+1)*4096]for s in slots)
            require(actual[0]==meta(row_input,task['normalized']['shape'],task['normalized']['stride'],task['normalized']['offset']),
                'Exact actual normalized view and selected BF16 words')
            require(actual[1]==meta(task['mask_raw'],task['mask']['shape'],task['mask']['stride'],slot*case['bucket'],'torch.bool')
                and actual[2]==meta(struct.pack('<'+'q'*batch,*task['positions']),[batch,1],[1,1],slot,'torch.int64')
                and actual[3]==meta(struct.pack('<'+'q'*batch,*slots),[batch],[1],slot,'torch.int64'),
                'Exact original physical mask, position and slot view formulas')
            require(observed['selected_rope']==actual[4:6],'Recorded positional inputs are the actual selected RoPE pair')
            for kind,(selected,source)in enumerate(zip(actual[4:6],rope)):
                expected=dict(source,shape=[batch,1,128],offset=source['offset']+slot*source['stride'][0])
                require(set(selected)==set(expected) and all(selected[k]==expected[k]for k in expected if k!='sha256')
                    and type(selected['sha256'])is str and re.fullmatch('[a-f0-9]{64}',selected['sha256']),
                    'Actual selected RoPE shape, original stride and absolute physical offset')
                if batch==4:require(selected==source,'Both B4 variants use exact same complete RoPE metadata')
                key=(kind,tuple(slots))
                require(key not in selected_rope or selected_rope[key]==selected,'Both layouts use same selected RoPE bytes as checked at runtime')
                selected_rope[key]=selected
            for index,name in enumerate(boundary.CACHE_NAMES):
                view=task['caches'][name];data=prepared['cache_raw'][name]
                selected=b''.join(data[s*1081344:s*1081344+case['bucket']*256]for s in slots)
                require(actual[6+index]==meta(selected,view['shape'],view['stride'],view['offset']),
                    'Actual cache selected bytes and retained full-store physical stride formula')
            piece=raw[slot*4096:(slot+batch)*4096]
            require(observed['output']==meta(piece,[batch,1,2048],[2048,2048,1]),
                'Every individual output matches its complete assembled physical-slot bytes')
    guard()
    return dict(complete=True,all_previous_controls_full_bytes_exact=True,every_view_formula_reconstructed=True,
        all_output_parts_full_bytes_verified=True,rope_selected_bytes_checked_by_callable=True,
        rope_payload_independently_reread=False,full_model_qualified=False,new_speedup_claim=False)


def _validate_rank(result,plan,rank,collector_bundle,native_bundle,previous_bundle,read_json,guard):
    guard();require(type(rank)is int and rank in (0,1) and result['schema']=='minicpm5_layer1_sdk_batch_layout_cpu_driver_v1'
        and result['complete']is True and result['rank']==rank,'Complete actual SDK batch/layout rank')
    require(result['completed_current_history']==previous_bundle['binding']==plan['layer1_sdk_batch_layout']
        and result['completed_native_cache']==native_bundle['binding']==plan['layer1_official_history'],
        'Exact externally authenticated completed prerequisite bindings')
    diagnostic(result,('other_arms_reexecuted','full_model_forward_reexecuted','native_reference_reexecuted',
        'attention_causality_proven','full_model_qualified','new_speedup_claim','self_exit_claimed'),
        ('all_other_arms_verified_not_reexecuted','complete_original_model_constructed','all_previous_controls_full_bytes_exact'))
    require([result[k]for k in ('anchor_decode_calls','physical_attention_invocations','full_B4_invocations','B1_invocations',
        'assembled_four_row_outputs','raw_output_bytes')]==[6,72,24,48,36,589824] and len(result['records'])==6,
        'Only six anchor cases and36 complete output tensors per rank')
    constructor.check_load_metadata(result['load_proof'],rank)
    previous_rank=previous_bundle['records']['rank%d_result'%rank]
    require(result['load_proof']==previous_rank['load_proof'] and result['official_layer1_shards']==previous_rank['official_layer1_shards']
        and result['official_layer1_shards']['all_four_projection_shards_raw_exact']is True,
        'Actual same original full model and verified official weight shards')
    data=official.load(plan,native_bundle,guard)
    source=inputs.AnchorStream(collector_bundle['collector_plan'],collector_bundle['collector_chain']['rank_results'][rank],
        collector_bundle['audit_result'],read_json,guard)
    proofs=[];cases=[];rows=[];controls={};index=0
    for item in source:
        case=item['case'];entry=result['records'][index];index+=1;guard()
        donors,current,target,refs=core.factorial.for_call(plan,data,case)
        old=inputs.previous_controls(plan,previous_bundle,case,guard)
        require(entry['bucket']==dict(bucket=case['bucket'],bucket_index=constructor.BUCKETS.index(case['bucket']),
            phase='decode',n_positions=case['bucket'],n_active_tokens=1,cache_aliases=84,configuration_unchanged=True)
            and entry['independent_audit_payload_sha256']==item['audited_call']['payload_sha256']
            and entry['official_target_refs']==refs and entry['official_target_sha256']==boundary.sha(target),
            'Actual original bucket, independent source audit and current official target')
        require([r['variant']for r in entry['outputs']]==list(native.ORDER),'Exactly six outputs in original callable order')
        outputs={}
        for row in entry['outputs']:
            variant=row['variant'];desc=row['output']
            path=Path(plan['output_root'])/('cpu-rank%d'%rank)/core.ANCHOR/('call-%02d'%case['call']['index'])/variant/'attention.bf16'
            require(desc['path']==str(path) and desc['raw_bytes']==16384,'Owned complete four-row output path')
            raw=inputs.payload.read_owned_payload(dict(desc,payload_bytes=16384,payload_sha256=desc['sha256']),path,guard)
            require(type(raw)is bytes and len(raw)==16384 and boundary.sha(raw)==desc['sha256']
                and core.factorial.history.finite_bf16(raw),'Independent entire finite output reread and digest')
            outputs[variant]=raw;rows.append(dict(rank=rank,call=case['call'],variant=variant,raw=raw))
        proof=check_call(entry['result'],item,donors,current,old,outputs,guard)
        proofs.append(dict(call=case['call'],proof=proof,outputs=copy.deepcopy(entry['outputs'])))
        cases.append(case);controls[rank,case['call']['index']]=dict(call=case['call'],original_call=case['original_call'],
            original=item['expected_attention_raw'],previous_own=old['own']['raw'],previous_both=old['official_both']['raw'],
            previous_restore=old['restore']['raw'],official_target=target)
    require(index==6 and source.finish()==result['source_stream'],'Every original history verified; only six anchor cases reexecuted')
    guard()
    proof=dict(complete=True,rank=rank,anchor_decode_calls=6,physical_attention_invocations=72,assembled_four_row_outputs=36,
        raw_output_bytes=589824,records=proofs,every_output_independently_reread=True,all_previous_controls_full_bytes_exact=True,
        every_view_formula_reconstructed=True,other_arms_reexecuted=False,all_other_arms_verified_not_reexecuted=True,
        rope_payload_independently_reread=False,full_model_qualified=False,new_speedup_claim=False)
    return proof,cases,rows,controls


def validate_rank_result(result,plan,rank,collector_bundle,native_bundle,previous_bundle,read_json,guard):
    return _validate_rank(result,plan,rank,collector_bundle,native_bundle,previous_bundle,read_json,guard)[0]


def validate_rank_pair(results,plan,collector_bundle,native_bundle,previous_bundle,read_json,guard):
    guard();require(type(results)is list and len(results)==2,'Both actual original TP ranks')
    proofs=[];cases=[];rows=[];controls={}
    for rank,result in enumerate(results):
        proof,new_cases,new_rows,new_controls=_validate_rank(result,plan,rank,collector_bundle,native_bundle,previous_bundle,read_json,guard)
        proofs.append(proof);cases+=new_cases;rows+=new_rows;controls.update(new_controls)
    guard();descriptive=analysis.analyze(cases,rows,controls);guard()
    return dict(schema='minicpm5_layer1_sdk_batch_layout_pair_verification_v1',complete=True,ranks=2,
        anchor_decode_calls=12,physical_attention_invocations=144,full_B4_invocations=48,B1_invocations=96,
        assembled_four_row_outputs=72,unique_replicated_four_row_outputs=36,unique_decode_frames=24,raw_output_bytes=1179648,
        rank_proofs=proofs,descriptive=descriptive,every_output_independently_reread=True,all_previous_controls_full_bytes_exact=True,
        every_view_formula_reconstructed=True,other_arms_reexecuted=False,all_other_arms_verified_not_reexecuted=True,
        rope_payload_independently_reread=False,full_model_qualified=False,new_speedup_claim=False)
