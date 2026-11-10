"""Full original trajectory result validation plus layer1 observation metadata.

No tensor payload reads here. Raw reconstruction is a separately admitted step.
"""
import math
from pathlib import Path
import re
import stat
import minicpm5_original32_layer0_full_bf16_cpu_runner_v1 as old
from research import minicpm5_layer1_observation_v1 as obs

p,same,identity=old.p,old.same,old.identity
ARMS,SELECTED_TAPS,REPEAT_FIELDS=old.ARMS,old.SELECTED_TAPS,old.REPEAT_FIELDS
TARGET,FULL_WEIGHT_SHA=old.TARGET,old.FULL_WEIGHT_SHA
checked_output_json=old.checked_output_json
state_values=old.state_values
patch_enabled=old.patch_enabled
validate_constructors=old.validate_constructors
validate_selected_bundle=old.validate_selected_bundle

def observation_metadata(call,original,original_ref,folder,previous):
    descriptor=call['layer1_observation']
    ci=call['call']['index']
    p.require(type(ci)is int and 0<=ci<14,'Finite original call')
    if ci==0:same(previous,None,'First observation cannot inherit history')
    else:
        p.require(type(previous)is dict,'Later observation requires history')
        same([previous['rank'],previous['arm'],previous['call']['index']],
             [call['rank'],call['arm'],ci-1],'Previous observation belongs to same arm/rank and preceding call')
        same(state_values(call['before_metadata']),state_values(previous['post_metadata']),
             'Current full input is actual previous returned state')
    same(descriptor['schema'],'minicpm5_layer1_observation_v1','Exact observation schema')
    same([descriptor[k]for k in ('rank','arm','call')],
         [call['rank'],call['arm'],call['call']['index']],'Actual observation identity')
    same(descriptor['original_call'],original_ref,'Exact original completed call')
    order=obs.same_call(call,original)
    same(descriptor['physical_frame_order'],order,'All original physical frame rows')
    for field in ('all_original_fields_exact','observation_occurs_after_forward',
                  'full_cache_reconstruction_required','caches_are_original_input_not_post_state'):
        p.require(descriptor[field]is True,'Exact observation boundary')
    p.require(descriptor['full_model_qualified'] is descriptor['new_speedup_claim'] is False,
              'Diagnostic observation cannot claim model qualification')
    path=folder/'layer1-observation.bin'
    same(descriptor['path'],str(path),'Only this original call payload')
    file_identity=identity(path)
    same(descriptor['identity'],file_identity,'Actual observation file identity')
    info=path.lstat()
    p.require(stat.S_ISREG(info.st_mode)and info.st_uid==1000 and info.st_nlink==1
        and 0<info.st_size<=obs.MAX_PAYLOAD<16*2**20,'Owned bounded payload file')
    same(info.st_size,descriptor['payload_bytes'],'Exact payload length')
    p.require(re.fullmatch('[0-9a-f]{64}',descriptor['payload_sha256'])is not None,'Whole payload digest')
    entries=descriptor['entries']
    same([x['name']for x in entries],list(obs.TAPS+obs.CACHES),'Complete ordered layer1 payload')
    offset=0
    for i,e in enumerate(entries):
        size=e['bytes']
        p.require(type(size)is int and size>=0 and type(e['offset'])is int
            and e['offset']==offset and offset+size<=info.st_size,'Complete contiguous payload spans')
        offset+=size
        if i<3:
            same(e['kind'],'tap','Existing prediction tap')
            same(e['shape'],[len(order),1,2048],'Exact selected prediction geometry')
            same(e['dtype'],'torch.bfloat16','Original prediction precision')
            same(e['frame_indices'],order,'Original physical frame permutation')
            same(e['source_metadata'],call['stage_metadata'][e['name']],'Original unsliced source geometry')
            same(e['sha256'],call['stage_sha256'][e['name']],'Original prediction byte digest')
            same(size,len(order)*4096,'Every prediction element stored')
        else:
            same(e['kind'],'cache','Full input cache record')
            source=call['before_metadata'][i-1]
            same(e['source_metadata'],source,'Exact original complete input cache')
            h=e['codec']
            same(set(h),{'schema','shape','dtype','encoding','page_bytes','raw_bytes','sha256',
                'base_sha256','packed_bytes','packed_sha256','pages'},'Exact full-cache codec fields')
            same([h[k]for k in ('schema','shape','dtype','encoding','page_bytes','raw_bytes')],
                [obs.codec.SCHEMA,list(obs.codec.SHAPE),'torch.bfloat16','absolute_changed_pages_zlib',
                 obs.codec.PAGE_BYTES,obs.codec.CACHE_BYTES],'Full4224-position cache geometry')
            same(h['sha256'],source['sha256'],'Reconstruction targets original cache hash')
            base=None if previous is None else previous['before_metadata'][i-1]['sha256']
            same(h['base_sha256'],base,'Delta is based on previous captured input cache, not poststate')
            same(h['packed_bytes'],size,'Complete compressed bytes span')
            p.require(size<=obs.codec.MAX_PACKED_BYTES
                and re.fullmatch('[0-9a-f]{64}',h['packed_sha256'])is not None,'Bounded encoded cache')
            pages=h['pages'];p.require(type(pages)is list and len(pages)<=obs.codec.PAGES,'Bounded page table')
            last=-1;at=0
            for page in pages:
                same(set(page),{'index','offset','packed_bytes','raw_bytes','sha256'},'Exact page fields')
                index=page['index'];n=page['packed_bytes']
                p.require(type(index)is int and last<index<obs.codec.PAGES,'Ordered distinct full-cache pages')
                expected=min(obs.codec.PAGE_BYTES,obs.codec.CACHE_BYTES-index*obs.codec.PAGE_BYTES)
                p.require(type(n)is int and 0<n<=expected+128 and page['offset']==at
                    and page['raw_bytes']==expected and at+n<=size
                    and re.fullmatch('[0-9a-f]{64}',page['sha256'])is not None,'Bounded complete page')
                at+=n;last=index
            same(at,size,'All compressed cache bytes accounted for')
            if previous is None:same([x['index']for x in pages],list(range(obs.codec.PAGES)),'Every first-snapshot page')
    same(offset,info.st_size,'No unreferenced payload bytes')
    same(identity(path),file_identity,'Observation file stable across metadata checks')
    return dict(metadata_complete=True,bytes=info.st_size,raw_payload_verified=False,
        complete_cache_words_verified=False,full_model_qualified=False,new_speedup_claim=False)

def validate_rank_result(result,plan,rank_id):
    p.require(result['schema']=='minicpm5_layer1_observation_cpu_rank_v1' and result['complete'] is True and type(result['rank'])is int and result['rank']==rank_id,'Complete targeted rank')
    same(result['arms'],ARMS,'Three finite arms in order')
    same(result['selected_tap_names'],SELECTED_TAPS,'Rank four selected taps')
    p.require(result['selected_taps_are_prediction_queries']is True and result['first_group_prefill_repeat_all4_exact']is True and result['call4_before_equal_prefill_only']is True,'Complete phase contrast controls')
    same([result[k]for k in ('model_forwards','seed_import_count','anchor_all14_exact')],[42,3,True],'All42 forwards and three own seeds with exact anchor')
    same([result[k]for k in ('full_cache_payloads_written','tap_payloads_written','selected_tap_payloads_written','selected_tap_raw_bytes')],[84,42,42,1572864],'Only three-arm bounded selected taps')
    same([result[k]for k in ('complete84_state_snapshots_written','layer1_full_cache_records_written','layer1_observation_calls')],[0,84,42],'Full layer1 records only; all original forwards')
    same(result['original_remaining_gates'],plan['remaining_gates'],'Original remaining gates')
    same(result['original_HF_reference'],plan['hf_reference'],'Original HF reference')
    p.require(result['full_model_qualified']is result['new_speedup_claim']is False,'Diagnostic only')
    same([row['arm']for row in result['arm_reports']],ARMS,'All arm reports ordered')
    anchors={}
    for ref in plan['layer0_full_bf16_trajectory']['strict_anchor_reports']:
        if ref['rank']==rank_id:
            old=p.read_ref(ref)
            same([old['rank'],old['arm'],old['call']['index']],[rank_id,'strict_cpu_s0',ref['call']],'Frozen strict anchor identity')
            anchors[ref['call']]=(ref,old)
    same(sorted(anchors),list(range(14)),'All14 immutable strict anchors')
    output=[];prefill_only=None
    for arm_index,entry in enumerate(result['arm_reports']):
        arm=ARMS[arm_index];folder=Path(plan['output_root'])/('cpu-rank'+str(rank_id))/arm
        report=checked_output_json(dict(path=entry['report'],sha256=entry['sha256']),folder/'result.json')
        p.require(report['schema']=='minicpm5_layer1_observation_cpu_arm_v1'and report['complete']is True,'Complete targeted arm')
        same([report['arm'],report['rank'],report['model_forwards'],report['seed_import_count']],[arm,rank_id,14,1],'Original arm identity and seed count')
        p.require(report['continuous_actual_cache_chain']is True,'Actual own-history cache chain')
        same([row['call']for row in report['reports']],list(range(14)),'All14 original calls ordered')
        same([report[k]for k in ('full_cache_payloads_written','tap_payloads_written','selected_tap_payloads_written','selected_tap_raw_bytes')],[28,14,14,524288],'Only bounded selected tap bundles')
        same([report[k]for k in ('complete84_state_snapshots_written','layer1_full_cache_records_written','layer1_observation_calls')],[0,28,14],'Every original call has full layer1 K/V')
        same(report['anchor_all14_exact'],arm_index==0,'Anchor gate')
        same(report['first_group_prefill_repeat_all4_exact'],arm_index==2,'Candidate first-group repeat gate')
        same(report['call4_before_equal_prefill_only'],arm_index==2,'Candidate first decode incoming gate')
        same(report['selected_tap_names'],SELECTED_TAPS,'Arm four selected taps')
        p.require(report['selected_taps_are_prediction_queries']is True,'Only gathered prediction taps')
        validate_constructors(report,rank_id,arm)
        calls=[];scores=[];previous=None;tap_bytes=0;previous_observation=None
        for ci,call_entry in enumerate(report['reports']):
            callfolder=folder/('call-%02d'%ci)
            call=checked_output_json(dict(path=call_entry['report'],sha256=call_entry['sha256']),callfolder/'result.json')
            oldref,old=anchors[ci]
            p.require(call['schema']=='minicpm5_layer1_observation_cpu_call_v1'and call['complete']is True,'Complete original targeted call')
            same([call['arm'],call['rank'],call['call']],[arm,rank_id,plan['calls'][ci]],'Original call identity')
            same(call['bucket'],plan['graph_calls'][ci]['bucket'],'Original bucket')
            for key in ('input_metadata','execution_input_metadata','integer_widening','changed_mask_cells','physical_rows','route_records'):
                same(call[key],old[key],'Original strict inputs mapping and routes '+key)
            p.require(len(call['execution_input_metadata'])==7,'All seven inputs')
            same(call['stage_names'],old['stage_names'],'All original ordered taps')
            same(list(call['stage_sha256']),call['stage_names'],'Every tap digest present')
            same(list(call['stage_metadata']),call['stage_names'],'Every tap layout present')
            p.require(len(call['stage_names'])==213 and all(re.fullmatch('[0-9a-f]{64}',v)is not None for v in call['stage_sha256'].values()),'All213 valid tap digests')
            before=state_values(call['before_metadata']);state_values(call['input_after_metadata']);post=state_values(call['post_metadata'])
            same(call['input_after_metadata'],call['before_metadata'],'All84 input states unchanged')
            same(before,previous if ci else state_values(old['before_metadata']),'Actual own-history chain including call7; original call0 seed')
            previous=post
            same(call['seed_imported_this_call'],ci==0,'Only call0 seed import')
            same(call['history_origin'],'original_call0_seed'if ci==0 else'previous_actual_post_same_arm','Actual own-cache lineage')
            same(call['group_reset_preserves_cache'],ci==7,'Call7 preserves actual cache')
            same([row['layer']for row in call['route_records']],list(range(42)),'All42 mask routes')
            preserve=call['preservation']
            same(preserve['inactive_slots'],sorted(set(range(4))-{f['slot']for f in call['call']['frames']}),'Inactive slot mapping')
            p.require(preserve['all84_inactive_slots_raw_exact']is preserve['all84_physical_padding_raw_exact']is True,'Inactive slots and padding preserved')
            enabled=patch_enabled(arm,call['call']['phase'])
            same(call['candidate_layer0_full_bf16_enabled'],enabled,'Finite phase-specific intervention')
            same(call['candidate_targets'],[TARGET]if enabled else[],'Only exact layer0 target')
            witnesses=call['oproj_dtype_records'];same([row['target']for row in witnesses],[TARGET]if enabled else[],'One or zero candidate witnesses')
            input_shape=[1,call['bucket'],1024]if call['call']['phase']=='prefill'else[4,1,1024]
            for row in witnesses:
                same(row['input_shape'],input_shape,'Original full projection input')
                same(row['output_shape'],input_shape[:2]+[2048],'Original full projection output')
                same([row[k]for k in ('input_dtype','gathered_dtype','weight_dtype','output_dtype')],['torch.bfloat16']*4,'All BF16 gather and full projection')
                same(row['gathered_shape'],input_shape[:2]+[2048],'Actual complete gathered input')
                same(row['full_weight_shape'],[2048,2048],'Complete original weight geometry');same(row['full_weight_sha256'],FULL_WEIGHT_SHA,'Complete original weight bytes')
                same(row['actual_group_ranks'],[0,1],'Actual NxD TP rank order')
                p.require(row['actual_tp_size']==2 and row['full_linear_calls']==1 and row['output_reduction']is False and row['explicit_fp32_casts']==0 and row['original_parameter_object_unchanged']is True,'One BF16 full projection and original parameter identity')
                same(row['actual_gather_function'],'torch.distributed.all_gather','Actual BF16 input gather')
            if arm_index==0:
                same(call['strict_anchor_ref'],oldref,'Frozen strict anchor reference')
                same(call['strict_anchor_exact'],dict(all7_inputs_exact=True,all213_taps_exact=True,all84_before_input_post_exact=True,full_logits_exact=True),'Anchor exact witness')
                for key in REPEAT_FIELDS+('HF_metrics','physical_rows','preservation'):same(call[key],old[key],'Frozen strict anchor '+key)
            for name in ('embed_tokens','layers.0.input_layernorm'):
                same(call['stage_metadata'][name],old['stage_metadata'][name],'Before selected projection original metadata')
                same(call['stage_sha256'][name],old['stage_sha256'][name],'Before selected projection original digest')
            same(call['upstream_matches_anchor'],dict(names=['embed_tokens','layers.0.input_layernorm'],all_exact=True),'Pre-projection anchor proof')
            if arm_index==2 and ci<4:
                same(call['first_group_prefill_repeat_exact'],dict(all7_inputs_exact=True,all213_taps_exact=True,all84_before_input_post_exact=True,full_logits_exact=True),'First-group candidate repeat proof')
                for key in REPEAT_FIELDS+('HF_metrics','physical_rows','preservation'):same(call[key],prefill_only[ci][key],'First group prefill candidates exact '+key)
            if arm_index==2 and ci==4:
                p.require(call['call4_before_equal_prefill_only']is True,'First decode incoming proof')
                same(call['before_metadata'],prefill_only[ci]['before_metadata'],'First decode equal candidate incoming caches')
            if arm_index!=0:
                same(call['strict_anchor_ref'],None,'No false exact-anchor claim on candidate')
                same(call['strict_anchor_exact'],None,'No false exact-anchor witness on candidate')
            if not(arm_index==2 and ci<4):same(call['first_group_prefill_repeat_exact'],None,'No unrelated first-group repeat witness')
            if not(arm_index==2 and ci==4):same(call['call4_before_equal_prefill_only'],None,'No unrelated incoming decode witness')
            same(call['payload'],call_entry['payload'],'Full logits payload bound')
            same(call['payload']['path'],str(callfolder/'output.pt'),'Exact logits payload path')
            p.require(type(call['payload']['bytes'])is int and 0<call['payload']['bytes']<=3*2**20 and re.fullmatch('[0-9a-f]{64}',call['payload']['sha256'])is not None,'Bounded logits payload metadata')
            same(call['output_metadata']['shape'],[len(call['physical_rows']),1,130560],'Complete vocabulary logits')
            same(call['output_metadata']['dtype'],'torch.float32','FP32 logits')
            p.require(re.fullmatch('[0-9a-f]{64}',call['output_metadata']['sha256'])is not None,'Complete logits digest')
            original_ref=plan['layer1_observation']['original_calls'][rank_id*42+arm_index*14+ci]
            original_ref={k:original_ref[k]for k in ('path','sha256')}
            original_call=p.read_ref(original_ref)
            observation_metadata(call,original_call,original_ref,callfolder,previous_observation)
            previous_observation=call
            tap_bytes+=validate_selected_bundle(call,callfolder)
            same([row['frame']for row in call['HF_metrics']],plan['calls'][ci]['frames'],'Every original HF frame')
            for row in call['HF_metrics']:
                values=row['metrics'];p.require(all(math.isfinite(values[k])for k in ('cosine','kl_reference_to_neuron','max_abs_error','reference_margin')),'Finite HF metrics')
                checks=dict(cosine=values['cosine']>=.995,KL=values['kl_reference_to_neuron']<=.02,max_abs=values['max_abs_error']<=2,high_margin_argmax=values['reference_margin']<=.5 or values['argmax_match'])
                same(row['failed_criteria'],[k for k,v in checks.items()if not v],'Original thresholds')
            calls.append(call);scores.extend(call['HF_metrics'])
        same(tap_bytes,524288,'Exact bounded per-arm tap bytes')
        same([row['frame']for row in scores],plan['frames'],'All32 original frame order')
        same(report['rows'],scores,'Arm scores derive from calls')
        fraction=sum(row['metrics']['argmax_match']for row in scores)/32;summary=report['summary']
        same([summary[k]for k in ('complete','frames','prefill_frames','decode_frames')],[True,32,8,24],'Original scoring coverage')
        same(summary['failed_frames'],[row['frame']['index']for row in scores if row['failed_criteria']],'Exact failure frames')
        same(summary['prefill_failed'],sum(bool(row['failed_criteria'])and row['frame']['step']==0 for row in scores),'Exact prefill failures')
        same(summary['decode_failed'],sum(bool(row['failed_criteria'])and row['frame']['step']>0 for row in scores),'Exact decode failures')
        same(summary['argmax_match_fraction'],fraction,'Exact top1 fraction')
        same(summary['passed'],all(not row['failed_criteria']for row in scores)and fraction>=.9,'Original aggregate gate')
        same(summary,result['summaries'][arm],'Rank summary bound')
        if arm_index==1:prefill_only=calls
        output.append(dict(arm=arm,calls=calls,summary=summary))
    return output

def validate_rank_pair(results,plan):
    p.require(len(results)==2,'Exactly two ranks')
    evidence=[validate_rank_result(result,plan,rank)for rank,result in enumerate(results)]
    for left,right in zip(*evidence):
        same(left['arm'],right['arm'],'Same paired arm')
        same(left['summary'],right['summary'],'Cross-rank summary')
        for a,b in zip(left['calls'],right['calls']):
            same(a['output_metadata'],b['output_metadata'],'Cross-rank complete logits tensor bytes/layout')
            same(a['HF_metrics'],b['HF_metrics'],'Cross-rank HF metrics')
            aa,bb=a['selected_tap_bundle'],b['selected_tap_bundle']
            for key in ('sha256','encoding','raw_bytes','entries'):same(aa[key],bb[key],'Cross-rank selected tap bundle '+key)
    return dict(complete=True,rank_count=2,arm_count=3,calls_per_rank=42,HF_frames_per_arm=32,
                selected_tap_raw_bytes_total=3145728,cross_rank_full_logits_tensor_hashes_equal=True,
                cross_rank_HF_metrics_equal=True,cross_rank_selected_tap_bundles_equal=True,raw_payloads_read=False,layer1_metadata_calls=84,layer1_full_cache_records=168,layer1_raw_reconstruction_pending=True)
