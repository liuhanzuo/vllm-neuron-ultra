"""Pure bounded decoded-metadata analysis for all five native trace arms.

Caller authenticates JSON references and fresh owners, independently hashes raw
witnesses, and proves runtime drain. This module performs no file/process IO.
"""
from itertools import combinations
import re
from minicpm5_python_trace_observer_v1 import ARMS,SCHEMA,require,validate_capture_summary,validate_operation
from minicpm5_native_ablation_witness_v1 import VOCAB,VECTOR_BYTES

WITNESSES=('actual-bf16','actual-fp32','expected-fp32','expected-bf16-derived')


def hash_value(value):
    require(type(value)is str and re.fullmatch('[0-9a-f]{64}',value),'Exact digest')


def validate_witness(value,arm,frame):
    identity=value['identity']
    require(identity['arm']==arm and identity['frame']==frame and
            identity['own_arm_cache_history'] is True and identity['frozen_reference_teacher_forcing'] is True,
            'Own-history original teacher-forced identity')
    require(value['schema']=='minicpm5_native_ablation_frame_witness_v1' and value['original_full_vocab']==VOCAB,
            'Original complete-vocabulary witness')
    rows=value['tensors'];require([r['name']for r in rows]==list(WITNESSES),'All four original witnesses')
    require(sum(r['metadata']['bytes']for r in rows)==VECTOR_BYTES,'Original witness byte budget')
    for row,dtype,size in zip(rows,('torch.bfloat16','torch.float32','torch.float32','torch.bfloat16'),(2,4,4,2)):
        m=row['metadata'];ref=row['reference']
        require(m['shape']==[VOCAB] and m['dtype']==dtype and m['bytes']==VOCAB*size and
                m['identity']==identity and m['derived']is(row['name']=='expected-bf16-derived'),
                'Unchanged witness representation')
        hash_value(m['sha256']);require(ref['sha256']==m['sha256'] and ref['bytes']==m['bytes'],'Raw witness binding')
    actual=value['actual_fp32_sha256'];expected=value['original_expected_fp32_sha256']
    hash_value(actual);hash_value(expected)
    require(actual==rows[1]['metadata']['sha256'] and expected==rows[2]['metadata']['sha256'],'Endpoint digest binding')
    require(value['passed']is value['raw_exact']is(actual==expected),'Original exact gate, no tolerance change')
    require(value['full_model_qualified']is value['new_speedup_claim']is False,'No qualification promotion')
    require(value['metrics']['actual_nonfinite']==value['metrics']['reference_nonfinite']==0,
            'Complete diagnostic requires original finite gate')
    return actual,expected


def validate_observation(value,arm,frame,witness,expected):
    require(value['complete']is True and value['arm']==arm and value['frame']==frame and
            value['passed']is witness['passed'] and value['raw_exact']is witness['raw_exact'] and
            value['full_model_qualified']is False,'Original observation receipt')
    capture=value['observer']
    if arm=='full_trace':
        require(capture['frames']==[frame] and capture['arm']=='HF_native' and capture['rank']==-1,
                'Exact full observer frame')
        validate_capture_summary(capture,expected)
        snapshots=capture['snapshots']
        require([r['label']for r in snapshots]==capture['events'] and len(snapshots)<=64,
                'All original snapshots retained in event order')
        tensors={'normalized_input','attention_mask','position_ids','q_projection','k_projection','v_projection',
                 'q_rope_input','k_rope_input','v_heads','rope_position_ids','q_post_rope','k_post_rope',
                 'rope_cos','rope_sin','context_before_o','attention_output','qk0_unscaled','qk0_scaled',
                 'softmax_input','probabilities_bf16','native_effective_key','native_effective_value',
                 'probabilities_fp32','actual_eager_mask','cache_update_active_key'}
        for snapshot in snapshots:
            context=snapshot['context']
            require(context['arm']=='HF_native' and context['frames']==[frame],'Snapshot actual frame binding')
            item=snapshot['value'];label=snapshot['label']
            if label in tensors:validate_snapshot_tensor(item,context)
            elif label=='cache_after' or label=='cache_before' and frame['step']>0:
                require(isinstance(item,list) and len(item)==2,'Complete actual key/value cache pair')
                for part in item:validate_snapshot_tensor(part,context)
            elif label=='cache_before':require(item is None,'Native prefill starts empty')
            elif label=='rotary_route':require(item['position_source']=='actual_original_model_call','Original rotary provenance')
        operations=[r['value']for r in snapshots if r['label']=='qk0_scale_operation']
        require(operations==capture['scale_provenance'],'Observed schema equals saved operation')
        validate_operation(operations[0],expected)
        scaled=next(r for r in snapshots if r['label']=='qk0_scaled')
        require(scaled['context']['frames']==[frame] and scaled['context']['arm']=='HF_native',
                'Actual scaled snapshot frame binding')
        m=scaled['value']['metadata']
        require(m['owned_clone']is True and m['cpu']is True and m['selection']['kind']=='prediction_query_rows' and
                m['selection']['axis']==2 and m['selected_shape'][2]==1 and m['selected_bytes']>0,
                'Actual selected scaled tensor must be retained, not metadata only')
        ref=scaled['value']['reference'];hash_value(ref['sha256'])
        require(ref['encoding']=='raw_little_endian' and ref['bytes']==m['selected_bytes'] and
                ref['identity']['size']==ref['bytes'] and type(ref['path'])is str and ref['path'].endswith('.raw'),
                'Durable complete scaled tensor reference')
    else:
        require(capture is None,'No full snapshots in control arm')
    if arm=='previous213_hooks':require(value['previous213']['count']==213,'Original previous213 controls')
    else:require(value['previous213']is None,'No invented hooks outside hook arm')


def validate_snapshot_tensor(value,context):
    require(isinstance(value,dict) and set(value)=={'metadata','reference'},'Actual snapshot tensor required')
    m=value['metadata'];ref=value['reference'];hash_value(ref['sha256'])
    require(m['context']==context and m['owned_clone']is True and m['cpu']is True and
            type(m['selected_bytes'])is int and 0<m['selected_bytes']<=16*2**20 and
            ref['bytes']==m['selected_bytes'] and ref['identity']['size']==ref['bytes'] and
            ref['encoding']=='raw_little_endian','Original bounded owned snapshot bytes')


def summarize(frames,runs,read_ref):
    """All160 metadata, including finite mismatches; never short-circuit false.

    runs[arm] supplies the decoded driver result plus a caller-authenticated
    owner identity. The caller must never run all five models in one process.
    """
    require(len(frames)==32 and {f['index']for f in frames}==set(range(32)) and
            set(runs)==set(ARMS) and callable(read_ref),'Full five-arm32 matrix')
    owners=set();originals={};indexed={};summaries=[];shared_contract=None
    for arm in ARMS:
        run=runs[arm];owner=run['owner'];key=(owner['pid'],str(owner['start_ticks']))
        require(type(key[0])is int and key[0]>0 and key[1].isdigit() and key not in owners,'Fresh process per arm')
        owners.add(key);result=run['result']
        require(result['schema']=='minicpm5_python_trace_arm_result_v1' and result['arm']==arm and
                result['diagnostic_complete']is True and result['observed_frames']==32 and
                result['original_hard_gate_preserved']is result['original88_scope_preserved']is True and
                result['aten_dispatch_used']is result['full_model_qualified']is result['new_speedup_claim']is False and
                result['scale_observation_schema']==SCHEMA,'Complete successor result with original gates')
        counts=result['trace_call_counts'];expected=None
        if arm in ('noop_trace','full_trace'):
            require(counts['mode']==arm and counts['calls']==32 and counts['aten_dispatch_used']is False and
                    counts['observed_scale_events']==(32 if arm=='full_trace' else 0) and
                    counts['noop_scale_events']==(32 if arm=='noop_trace' else 0),'Independent32 trace scopes')
            expected=counts['contract']
            if shared_contract is None:shared_contract=expected
            else:require(expected==shared_contract,'Same authenticated original trace boundary in both arms')
        else:require(counts is None,'Trace absent in other controls')
        if arm=='noop_wrappers':require(result['noop_call_counts']['attention.forward']==32,'Original noop wrapper calls')
        rows=result['frames'];require(len(rows)==32 and {r['frame']['index']for r in rows}==set(range(32)),
                                     'All32 frames per arm')
        by_index={r['frame']['index']:r for r in rows};values={};different=[]
        for frame in frames:
            row=by_index[frame['index']];require(row['frame']==frame,'Original frame identity')
            witness=read_ref(row['witness']);actual,reference=validate_witness(witness,arm,frame)
            require(row['passed']is witness['passed'],'Result retains failed endpoint')
            observation=read_ref(row['observation']);require(observation['witness']==row['witness'],'Durable original witness reference')
            validate_observation(observation,arm,frame,witness,expected)
            originals.setdefault(frame['index'],reference)
            require(originals[frame['index']]==reference,'Same frozen reference across all five arms')
            values[frame['index']]=dict(actual=actual,passed=witness['passed'])
            if not witness['passed']:different.append(frame)
        equal=32-len(different)
        require(result['raw_exact_frames']==equal and result['passed']is(equal==32),'Truthful unchanged arm gate')
        summaries.append(dict(arm=arm,frames=32,exact_original_frames=equal,differing_original_frames=different))
        indexed[arm]=values
    pairs=[]
    for left,right in combinations(ARMS,2):
        different=[f for f in frames if indexed[left][f['index']]['actual']!=indexed[right][f['index']]['actual']]
        pairs.append(dict(left=left,right=right,frames=32,equal_frames=32-len(different),differing_frames=different))
    return dict(schema='minicpm5_python_trace_metadata_analysis_v1',diagnostic_metadata_complete=True,frames=160,
                arm_summaries=summaries,paired_comparisons=pairs,all_arms_match_original=all(x['exact_original_frames']==32 for x in summaries),
                all_arm_pairs_equal=all(x['equal_frames']==32 for x in pairs),original88_scope_preserved=True,
                raw_audit_and_runtime_drain_still_required=True,full_model_qualified=False,new_speedup_claim=False,
                causal_operator_proven=False,limitations=['Decoded authenticated metadata only; caller must verify all raw payloads and owners/drain.',
                'Original32 per arm and fresh process histories retained; finite mismatches are results, not early exit.',
                'Noop/full trace separately require exact original endpoints; pairwise agreement is insufficient.',
                'Original88 then full-model/cold-cache/long-decode/slot-reuse/quality/paired-performance remain.'])
