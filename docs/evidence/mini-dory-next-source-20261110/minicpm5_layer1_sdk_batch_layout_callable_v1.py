"""Unadmitted same-SDK B4/B1 and input-layout diagnostic callable.

No I/O or native imports. A separately admitted caller supplies the original
full TP2 model, exact official weight hashes, completed metadata and full old
output bytes. Original model configuration and BF16 arithmetic remain fixed.
B1 narrows original B4 cache/RoPE views; it never invokes a B1 native model.
"""
import copy
from research import minicpm5_layer1_sdk_batch_layout_v1 as core
from research import minicpm5_layer1_current_history_factorial_callable_v1 as original
from research.minicpm5_layer1_own_history_callable_v1 import tensor_bytes, metadata
boundary=core.boundary
require=core.require
ORDER=('own',*core.VARIANTS,'restore')
WEIGHTS=('qkv_proj.q_proj.weight','qkv_proj.k_proj.weight','qkv_proj.v_proj.weight','o_proj.o_proj.weight')


def scope(torch,model,case):
    """Assert the original route, including the actual BF16 reduction dtype."""
    config=model.neuron_config;attention=model.layers[1].self_attn
    require(torch.is_inference_mode_enabled() and torch.get_num_threads()==1
        and torch.distributed.is_initialized() and torch.distributed.get_backend()=='gloo'
        and torch.distributed.get_world_size()==2 and torch.distributed.get_rank()==case['rank'],
        'Admitted original TP2 CPU inference scope')
    require(len(model.layers)==42 and model.n_positions==case['bucket']
        and type(attention).__name__=='NeuronLlamaAttention'
        and type(attention).__module__=='neuronx_distributed_inference.models.llama.modeling_llama',
        'Original full model and original installed attention')
    for module in attention.modules():
        require('forward'not in module.__dict__ and not module._forward_hooks and not module._forward_pre_hooks,
            'No instance forward overrides or attention hooks')
    expected=dict(batch_size=4,max_batch_size=4,tkg_batch_size=4,kv_cache_batch_size=4,
        n_active_tokens=1,is_prefill_stage=False,token_generation_batches=None,rpl_reduce_dtype=None,
        sequence_parallel_enabled=False,is_prefix_caching=False,fused_qkv=False,
        qkv_kernel_enabled=False,qkv_nki_kernel_enabled=False,out_proj_kernel_enabled=False)
    require(all(getattr(config,k)==v and type(getattr(config,k))is type(v)for k,v in expected.items()),
        'Original full B4 compiler configuration; no B1 rebuild or precision override')
    require(config.torch_dtype==torch.bfloat16 and attention.torch_dtype==torch.bfloat16
        and attention.o_proj.o_proj.reduce_dtype==torch.bfloat16,
        'Original BF16 projection and BF16 collective reduction')
    require(attention.tp_degree==2 and attention.cp_degree==attention.dp_degree==1
        and attention.num_heads==8 and attention.num_key_value_heads==1 and attention.head_dim==128,
        'Original rank-local query and KV head geometry')
    flags=('sequence_parallel_enabled','k_cache_transposed','attn_block_tkg_nki_kernel_enabled',
        'attn_block_cte_nki_kernel_enabled','attn_tkg_builtin_kernel_enabled','attn_tkg_nki_kernel_enabled',
        'flash_decoding_enabled','use_qk_norm')
    require(all(getattr(attention,k)is False for k in flags)
        and attention.attention_chunk_size is None and attention.sliding_window is None,
        'Original unfused global attention and manual prior/active path')
    return dict(configuration=expected,model_batch_size=4,torch_dtype=str(config.torch_dtype),
        attention_dtype=str(attention.torch_dtype),collective_dtype=str(attention.o_proj.o_proj.reduce_dtype),
        bucket=case['bucket'],rank=case['rank'],layers=42,tp=2)


def check_weights(torch,attention,hashes):
    require(set(hashes)==set(WEIGHTS),'All four original official shard hashes')
    params=dict(attention.named_parameters())
    for name in WEIGHTS:
        value=params[name]
        require(value.dtype==torch.bfloat16 and value.device.type=='cpu'
            and boundary.sha(tensor_bytes(torch,value))==hashes[name],'Unchanged original official shard '+name)


def check_view(torch,value,recipe,backing):
    require(list(value.shape)==recipe['shape'] and list(value.stride())==recipe['stride']
        and value.storage_offset()==recipe['offset'] and value.dtype==torch.bfloat16
        and value.device.type=='cpu'
        and value.untyped_storage().data_ptr()==backing.untyped_storage().data_ptr()
        and value.untyped_storage().nbytes()==recipe['storage_bytes'],
        'Exact stride, absolute storage offset, alias and full physical backing')
    return metadata(torch,value)


def _previous(case,previous,prepared):
    require(type(previous)is dict and set(previous)=={'own','official_both','restore'},
        'All actual completed predecessor controls supplied separately')
    refs={}
    for mode in ('own','official_both','restore'):
        row=previous[mode];result=row['result'];desc=row['output'];raw=row['raw']
        require(set(row)=={'result','output','raw'} and result['schema']=='minicpm5_layer1_current_history_factorial_callable_result_v1'
            and result['complete']is True and [result[k]for k in ('rank','arm','call','mode')]
            ==[case['rank'],case['arm'],case['call'],mode]
            and result['full_model_qualified']is result['new_speedup_claim']is False,
            'Exact original predecessor model row and diagnostic boundary')
        require(type(raw)is bytes and len(raw)==desc['raw_bytes']==16384
            and boundary.sha(raw)==desc['sha256']==result['attention_output_sha256']
            and type(desc['path'])is str and type(desc['identity'])is dict
            and core.factorial.history.finite_bf16(raw),
            'Complete separately authenticated previous raw control bytes and descriptor')
        if mode=='official_both':require(result['intervention']==prepared['intervention'],
            'Previous both-official output used the exact same anchor physical state')
        refs[mode]=copy.deepcopy(desc)
    require(previous['own']['raw']==previous['restore']['raw'],'Previous own and restore full bytes exact')
    return refs


def official_variants(torch,model,prepared,official_shard_hashes,guard):
    """Ten actual attention invocations, four complete assembled B4 outputs.

    This helper launches no process and grants no admission. The guard must
    reject before the first model/tensor access when the caller lacks a lease.
    """
    require(callable(guard),'Independent runtime guard required');guard()
    case=prepared['case'];before_scope=scope(torch,model,case);attention=model.layers[1].self_attn
    config_identity=id(model.neuron_config);attention_identity=id(attention)
    check_weights(torch,attention,official_shard_hashes)
    recipes={variant:core.tasks(prepared,variant)for variant in core.VARIANTS}
    guard()
    buffers={name:torch.frombuffer(bytearray(raw),dtype=torch.bfloat16) for name,raw in (
        ('normalized_packed',prepared['packed_raw']),('normalized_logical',prepared['normalized_raw']))}
    normalized=buffers['normalized_packed'].as_strided((4,1,2048),(1,4,4),0)
    require(tensor_bytes(torch,normalized)==prepared['normalized_raw'],'Original selected B4 normalized words')
    positions=torch.tensor(case['positions'],dtype=torch.int64).reshape(4,1)
    seq_ids=torch.tensor(case['seq_ids'],dtype=torch.int64)
    mask=torch.tensor([[column<position for column in range(case['bucket'])]for position in case['positions']],
        dtype=torch.bool).reshape(4,1,1,case['bucket'])
    require(metadata(torch,mask)==case['attention_route']['mask']
        and metadata(torch,positions)==case['attention_route']['positions'],'Original full B4 route and physical positions')
    for index,name in zip((2,3),boundary.CACHE_NAMES):
        guard();raw=prepared['cache_raw'][name]
        value=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(boundary.CACHE_SHAPE)
        require(bool(torch.isfinite(value).all()),'Finite complete anchor cache including padding')
        model.kv_mgr.past_key_values[index].copy_(value)
        actual=model.kv_mgr.past_key_values[index]
        require(metadata(torch,actual)==dict(shape=boundary.CACHE_SHAPE,stride=boundary.CACHE_STRIDE,offset=0,
            dtype='torch.bfloat16',sha256=boundary.sha(raw)),'Installed entire original-stride anchor cache')
    guard()
    past=model.kv_mgr.get_kv_by_layer_id(idx=1,seq_len=case['bucket'],skip_slice=False,
        seq_ids=seq_ids,is_for_speculation=False,windowed_context_encoding_window_idx=-1)
    require(len(past)==2,'Original B4 SDK getter returned both cache views')
    for index,name in enumerate(boundary.CACHE_NAMES):
        check_view(torch,past[index],recipes['b4_strided'][0]['caches'][name],model.kv_mgr.past_key_values[index+2])
    rope=model.layers[0].self_attn.rotary_emb
    require(type(rope).__name__=='RotaryEmbedding'
        and type(rope).__module__=='neuronx_distributed_inference.modules.attention.utils'
        and rope.base==5000000 and rope.dim==128 and rope.factor is None,'Original carried B4 RoPE module')
    guard();cos,sin=rope(normalized,positions)
    require(all(list(v.shape)==[4,1,128] and v.dtype==torch.bfloat16 and v.device.type=='cpu'
        and bool(torch.isfinite(v).all())for v in (cos,sin)),'One complete finite original BF16 B4 RoPE pair')
    rope_raw=[tensor_bytes(torch,v)for v in (cos,sin)];rope_metadata=[metadata(torch,v)for v in (cos,sin)]
    unchanged=[*buffers.values(),mask,positions,seq_ids,cos,sin,*model.kv_mgr.past_key_values[2:4]]
    before=[metadata(torch,v)for v in unchanged]
    results=[];outputs=[]
    for variant in core.VARIANTS:
        parts=[];observed=[]
        for task in recipes[variant]:
            guard();batch=len(task['slots']);slot=task['slots'][0];backing=buffers[task['normalized']['source']]
            selected=backing.as_strided(tuple(task['normalized']['shape']),tuple(task['normalized']['stride']),task['normalized']['offset'])
            check_view(torch,selected,task['normalized'],backing)
            require(boundary.sha(tensor_bytes(torch,selected))==task['normalized_logical_sha256'],
                'Exact complete current row words under selected layout')
            pick=lambda value:value if batch==4 else value.narrow(0,slot,1)
            selected_past=[pick(v)for v in past];selected_positions=pick(positions)
            selected_seq=seq_ids if batch==4 else seq_ids.narrow(0,slot,1)
            selected_mask=pick(mask);selected_cos=pick(cos);selected_sin=pick(sin)
            for index,name in enumerate(boundary.CACHE_NAMES):
                check_view(torch,selected_past[index],task['caches'][name],model.kv_mgr.past_key_values[index+2])
            require(tensor_bytes(torch,selected_mask)==task['mask_raw']
                and list(selected_mask.shape)==task['mask']['shape']
                and list(selected_mask.stride())==task['mask']['stride']
                and selected_mask.storage_offset()==slot*case['bucket'],
                'Original B4 mask view, exact slot offset and prior-only bucket mask')
            selected_rope=[]
            for value,source,source_raw in zip((selected_cos,selected_sin),(cos,sin),rope_raw):
                require(list(value.shape)==[batch,1,128] and list(value.stride())==list(source.stride())
                    and value.storage_offset()==source.storage_offset()+slot*source.stride()[0]
                    and value.untyped_storage().data_ptr()==source.untyped_storage().data_ptr()
                    and tensor_bytes(torch,value)==source_raw[slot*256:(slot+batch)*256],
                    'Same B4-generated RoPE bytes, source stride and absolute physical slot offset')
                selected_rope.append(metadata(torch,value))
            inputs=[selected,selected_mask,selected_positions,selected_seq,selected_cos,selected_sin,*selected_past]
            input_metadata=[metadata(torch,v)for v in inputs]
            guard()
            output=attention(hidden_states=selected,attention_mask=selected_mask,position_ids=selected_positions,past_key_value=selected_past,
                active_mask=None,adapter_ids=torch.zeros(batch,dtype=torch.int32),cos_cache=selected_cos,sin_cache=selected_sin,
                rmsnorm=None,rotary_position_ids=None,kv_mgr=model.kv_mgr,get_kv_per_layer=False,update_kv_per_layer=False,
                residual=None,windowed_context_encoding_window_idx=-1,seq_ids=selected_seq,idx=1,
                is_for_context_encoding=False,seq_len=case['bucket'],local_mask=None,padding_mask=None)
            guard()
            require([metadata(torch,v)for v in unchanged]==before and [metadata(torch,v)for v in inputs]==input_metadata,
                'Every full backing, padding word, cache and selected route unchanged')
            require(id(model.neuron_config)==config_identity and id(model.layers[1].self_attn)==attention_identity
                and scope(torch,model,case)==before_scope,'Original full model and route remain unchanged')
            check_weights(torch,attention,official_shard_hashes)
            actual=output.hidden_states;output_metadata=metadata(torch,actual)
            require(list(actual.shape)==[batch,1,2048] and list(actual.stride())==[2048,2048,1]
                and actual.storage_offset()==0 and actual.dtype==torch.bfloat16 and actual.device.type=='cpu'
                and bool(torch.isfinite(actual).all()),'Complete finite original output geometry')
            require(output.residual is None and output.cos_cache is selected_cos and output.sin_cache is selected_sin,
                'Original carried positional caches and non-fused residual')
            raw=tensor_bytes(torch,actual)
            parts.append(dict(rank=case['rank'],call=case['call'],variant=variant,slots=task['slots'],physical_frames=task['physical_frames'],raw=raw))
            observed.append(dict(task={k:copy.deepcopy(v)for k,v in task.items()if k!='mask_raw'},
                actual_inputs=input_metadata,selected_rope=selected_rope,output=output_metadata))
        raw=core.combine(case,variant,parts)
        outputs.append(dict(rank=case['rank'],call=copy.deepcopy(case['call']),variant=variant,raw=raw))
        results.append(dict(variant=variant,physical_invocations=len(parts),tasks=observed,raw_bytes=len(raw),sha256=boundary.sha(raw)))
    guard()
    return dict(variants=results,scope=before_scope,official_rope=rope_metadata,official_rope_evaluations=1,
        physical_attention_invocations=10,all_backing_and_cache_bytes_unchanged=True),outputs


def replay(torch,model,case,own_normalized_raw,own_cache_raw,prior_donors,current_donors,
        expected_attention_raw,previous_controls,official_shard_hashes,guard):
    """Return (JSON metadata, six complete {rank,call,variant,raw} rows)."""
    require(callable(guard),'Independent runtime guard required');guard()
    prepared=core.prepare(case,own_normalized_raw,own_cache_raw,prior_donors,current_donors)
    refs=_previous(case,previous_controls,prepared)
    require(expected_attention_raw==previous_controls['own']['raw']==previous_controls['restore']['raw']
        and boundary.sha(expected_attention_raw)==case['expected_attention']['logical_sha256'],
        'Actual complete original own and predecessor controls agree')
    before_scope=scope(torch,model,case)
    own_result,own_raw=original.replay(torch,model,case,'own',own_normalized_raw,own_cache_raw,
        prior_donors,current_donors,expected_attention_raw,official_shard_hashes,guard)
    require(own_raw==expected_attention_raw,'Untouched full B4 own control before interventions')
    variants,rows=official_variants(torch,model,prepared,official_shard_hashes,guard)
    require(rows[0]['raw']==previous_controls['official_both']['raw'],
        'Actual previous both-official complete B4 output reproduced')
    restore_result,restore_raw=original.replay(torch,model,case,'restore',own_normalized_raw,own_cache_raw,
        prior_donors,current_donors,expected_attention_raw,official_shard_hashes,guard)
    controls=core.verify_controls(expected_attention_raw,previous_controls['own']['raw'],own_raw,
        previous_controls['official_both']['raw'],rows[0]['raw'],restore_raw)
    require(scope(torch,model,case)==before_scope,'Original full model unchanged after restore')
    guard()
    rows=[dict(rank=case['rank'],call=copy.deepcopy(case['call']),variant='own',raw=own_raw),*rows,
        dict(rank=case['rank'],call=copy.deepcopy(case['call']),variant='restore',raw=restore_raw)]
    require(tuple(r['variant']for r in rows)==ORDER and all(len(r['raw'])==16384 for r in rows),
        'All six full four-row results in reviewed order')
    return dict(schema='minicpm5_layer1_sdk_batch_layout_callable_result_v1',complete=True,
        rank=case['rank'],arm=case['arm'],call=copy.deepcopy(case['call']),original_call=copy.deepcopy(case['original_call']),
        intervention=prepared['intervention'],source_case_sha256=prepared['source_case_sha256'],
        own_result=own_result,restore_result=restore_result,previous_controls=refs,controls=controls,
        variants=variants,physical_attention_invocations=12,full_B4_invocations=4,B1_invocations=8,
        assembled_four_row_outputs=6,raw_output_bytes=98304,all_original_precision_and_model_configuration_preserved=True,
        same_B4_RoPE_used_for_all_official_variants=True,full_model_reexecuted=False,
        native_reference_reexecuted=False,attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False),rows
