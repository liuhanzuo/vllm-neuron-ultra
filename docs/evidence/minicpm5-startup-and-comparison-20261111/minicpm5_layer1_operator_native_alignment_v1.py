"""Pure SDK B4/TP2 to native B1 metadata alignment; no IO or admission.

Input dictionaries must be independently authenticated by the caller. This
helper does not validate actual exits, read tensors or establish equivalence.
All full source tensors remain preserved; selectors only describe comparison.
"""
import copy


def require(ok,message):
    if not ok:raise RuntimeError(message)


# native, SDK prior/main, SDK active (if split), interpretation
EVENTS=(
    ('normalized_input','normalized_input',None,'full_hidden'),
    ('position_ids','position_ids',None,'exact_position'),
    ('q_projection','q_projection',None,'query_head_slice'),
    ('k_projection','k_projection',None,'kv_head_slice'),
    ('v_projection','v_projection',None,'kv_head_slice'),
    ('q_rope_input','q_rope_input',None,'query_head_slice'),
    ('k_rope_input','k_rope_input',None,'kv_head_slice'),
    ('rope_cos','rope_cos',None,'full_rope'),
    ('rope_sin','rope_sin',None,'full_rope'),
    ('q_post_rope','q_post_rope',None,'query_head_slice'),
    ('k_post_rope','k_post_rope',None,'kv_head_slice'),
    ('native_effective_key','effective_prior_kv','k_after_prep','prior_prefix_plus_active_key'),
    ('native_effective_value','effective_prior_kv','v_after_prep','prior_prefix_plus_active_value'),
    ('qk0_unscaled','qk0_unscaled','qk1_unscaled','valid_prior_plus_active_scores'),
    ('qk0_scaled','qk0_scaled','qk1_scaled','valid_prior_plus_active_scores'),
    ('softmax_input','softmax_input','softmax_active_input','mask_and_dtype_aware_scores'),
    ('probabilities_fp32','probabilities_fp32','probabilities_active_fp32','valid_prior_plus_active_probabilities'),
    ('probabilities_bf16','probabilities_bf16','probabilities_active_bf16','valid_prior_plus_active_probabilities'),
    ('context_before_o','context_before_o',None,'query_head_slice_after_original_pv_merge'),
    ('attention_output','o_proj_after_reduce',None,'full_hidden_after_original_collective'),
)


def frame_alignment(sdk_case,native_case,shards):
    """Return one exact comparison mapping, without touching saved raw tensors.

    sdk_case is the original authenticated AnchorStream case, not a reordered
    list inferred from fixture numbers. native_case is one manifest frame row.
    shards is the original all-four-projection shard proof for this SDK rank.
    """
    rank=sdk_case['rank'];call=sdk_case['call']['index'];frame=native_case['frame']
    require(type(rank)is int and rank in (0,1) and sdk_case['call']['phase']=='decode'
        and call in (4,5,6,11,12,13),'Only the original two ranks and six decode calls')
    frames=sdk_case['physical_frame_order'];positions=sdk_case['positions'];length=native_case['cache_length']
    require(type(frames)is list and len(frames)==4 and type(positions)is list and len(positions)==4
        and len({f['index']for f in frames})==4 and all(f['slot']==i for i,f in enumerate(frames)),
        'Complete original physical row order, not logical frame sort order')
    require(frames.count(frame)==1 and native_case['call']==call,'Native checkpoint belongs to this exact original call')
    row=frames.index(frame)
    require(type(length)is int and 2<=length<=4096 and frame['step']in (1,2,3)
        and native_case['logical_position']==length-1==positions[row],
        'Exact native checkpoint history length and original absolute decode position')
    bucket=sdk_case['bucket']
    require(type(bucket)is int and bucket==((2048,2048,2048,128,128,128)[(4,5,6,11,12,13).index(call)])
        and length-1<=bucket,'Original SDK full prior bucket retained')
    require(shards['schema']=='minicpm5_layer1_official_shards_v1' and shards['complete']is True
        and type(shards['rank'])is int and shards['rank']==rank
        and shards['all_four_projection_shards_raw_exact']is True and shards['precision_changed']is False
        and shards['q_head_ids']==list(range(rank*8,(rank+1)*8)) and shards['kv_head_ids']==[rank],
        'Authenticated original projection shard metadata determines global head identity')
    return dict(frame=copy.deepcopy(frame),call=call,rank=rank,sdk_physical_row=row,native_batch_row=0,
        q_head_ids=copy.deepcopy(shards['q_head_ids']),kv_head_ids=copy.deepcopy(shards['kv_head_ids']),head_dim=128,
        native_full_key_length=length,sdk_full_prior_bucket=bucket,
        native_prior_key_range=[0,length-1],sdk_valid_prior_key_range=[0,length-1],
        native_active_key_index=length-1,sdk_active_key_index=0,
        sdk_masked_prior_key_range=[length-1,bucket],
        native_q_projection_columns=[rank*1024,(rank+1)*1024],
        native_kv_projection_columns=[rank*128,(rank+1)*128],
        event_pairs=[dict(native=n,sdk=s,sdk_active=a,interpretation=kind)for n,s,a,kind in EVENTS],
        sdk_full_padding_still_required=True,source_tensors_not_truncated=True,
        native_probabilities_are_joint_prior_and_active=True,
        sdk_partial_oproj_has_no_native_local_capture=True,
        native_pv_has_no_observed_prior_active_split=True,
        caller_must_authenticate_inputs=True,metadata_alignment_only=True,
        actual_exits_validated=False,raw_bytes_read=False,numerical_comparison_performed=False,
        attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)
