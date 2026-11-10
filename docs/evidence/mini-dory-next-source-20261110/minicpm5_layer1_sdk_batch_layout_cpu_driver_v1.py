"""Admitted-caller driver: full original model, anchor-only SDK batch/layout calls.

Import is stdlib only. This module launches no process and supplies no admission.
"""
import os
from pathlib import Path
from research import minicpm5_layer1_sdk_batch_layout_inputs_v1 as inputs
from research import minicpm5_layer1_sdk_batch_layout_callable_v1 as native
from research import minicpm5_layer1_official_history_inputs_v1 as official
from research import minicpm5_layer1_official_shards_v2 as shards
import minicpm5_layer0_cpu_constructor_v1 as constructor
core=inputs.core;require=core.require


def save_output(context,case,variant,raw,guard):
    guard();context.guard('cpu')
    require(case['rank']==context.permit['rank'] and case['arm']==core.ANCHOR
        and case['call']['index'] in core.boundary.DECODE_CALLS and variant in native.ORDER
        and type(raw)is bytes and len(raw)==16384 and core.factorial.history.finite_bf16(raw),
        'Only complete finite original four-row anchor output')
    root=inputs.payload.safe_path(context.plan['output_root'])
    require(root.is_dir() and root.stat().st_uid==os.getuid()==1000,'Owned admitted output root')
    folder=root/('cpu-rank%d'%case['rank'])/core.ANCHOR/('call-%02d'%case['call']['index'])/variant
    folder.mkdir(parents=True,exist_ok=False);path=inputs.payload.safe_path(folder/'attention.bf16')
    with path.open('xb')as f:
        require(f.write(raw)==16384,'Complete raw output write');f.flush();os.fsync(f.fileno())
    desc=dict(path=str(path),raw_bytes=16384,sha256=core.boundary.sha(raw),identity=inputs.payload.file_identity(path,path.lstat()))
    require(inputs.payload.read_owned_payload(dict(desc,payload_bytes=16384,payload_sha256=desc['sha256']),path,guard)==raw,
        'Every saved output independently reread in full')
    return desc


def run_rank(context,torch,collector_plan,rank_result,audit,read_json,guard,native_bundle,previous_bundle):
    guard();context.guard('cpu');require(context._sources_checked,'All local sources frozen before model work')
    rank=context.permit['rank']
    require(type(rank)is int and rank in (0,1) and rank_result['rank']==rank
        and torch.is_inference_mode_enabled() and torch.get_num_threads()==1
        and torch.distributed.is_initialized() and torch.distributed.get_world_size()==2
        and torch.distributed.get_rank()==rank and torch.distributed.get_backend()=='gloo',
        'Original admitted TP2 CPU model execution scope')
    require(previous_bundle['binding']==context.plan['layer1_sdk_batch_layout'],'Exact current/history prerequisite')
    data=official.load(context.plan,native_bundle,guard)
    source=inputs.AnchorStream(collector_plan,rank_result,audit,read_json,guard)
    model,loading=constructor.build(rank,'decode');constructor.check_load_metadata(loading,rank)
    weights=shards.verify_model(context,torch,model,rank)
    previous=previous_bundle['records']['rank%d_result'%rank]
    require(weights['all_four_projection_shards_raw_exact']is True
        and loading==previous['load_proof'] and weights==previous['official_layer1_shards'],
        'Same complete original model and freshly verified official shards')
    records=[]
    for item in source:
        guard();case=item['case'];donors,current,target,refs=core.factorial.for_call(context.plan,data,case)
        old=inputs.previous_controls(context.plan,previous_bundle,case,guard)
        bucket=constructor.select_bucket(model,case['bucket'])
        result,rows=native.replay(torch,model,case,item['normalized_raw'],item['cache_raw'],donors,current,
            item['expected_attention_raw'],old,weights['hashes'],guard)
        require(result['complete']is True and result['rank']==rank and result['arm']==core.ANCHOR
            and result['call']==case['call'] and [r['variant']for r in rows]==list(native.ORDER),
            'Complete actual anchor call and all six outputs')
        outputs=[]
        for row in rows:
            require(row['rank']==rank and row['call']==case['call'],'Every output retains original full call')
            outputs.append(dict(variant=row['variant'],output=save_output(context,case,row['variant'],row['raw'],guard)))
        records.append(dict(result=result,outputs=outputs,bucket=bucket,
            independent_audit_payload_sha256=item['audited_call']['payload_sha256'],
            official_target_sha256=core.boundary.sha(target),official_target_refs=refs))
    proof=source.finish();guard();require(len(records)==6,'Exactly six complete anchor decode cases')
    return dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_driver_v1',complete=True,rank=rank,
        completed_current_history=previous_bundle['binding'],completed_native_cache=native_bundle['binding'],
        load_proof=loading,official_layer1_shards=weights,source_stream=proof,records=records,
        anchor_decode_calls=6,physical_attention_invocations=72,full_B4_invocations=24,B1_invocations=48,
        assembled_four_row_outputs=36,raw_output_bytes=589824,other_arms_reexecuted=False,
        all_other_arms_verified_not_reexecuted=True,complete_original_model_constructed=True,
        all_previous_controls_full_bytes_exact=True,full_model_forward_reexecuted=False,
        native_reference_reexecuted=False,attention_causality_proven=False,full_model_qualified=False,
        new_speedup_claim=False,self_exit_claimed=False)
