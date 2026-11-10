"""Original SDK attention with explicit current-input/history factorial sources.

No source case is relabelled. The current normalized input controls this step's
Q/K/V jointly. Original mask, positions, RoPE, cache layout, BF16 weights and the
complete four-row attention invocation are preserved.
"""
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research.minicpm5_layer1_own_history_callable_v1 import tensor_bytes, metadata
from research import minicpm5_layer1_current_history_factorial_v1 as factorial
require=boundary.require


def replay(torch,model,case,mode,own_normalized_raw,own_cache_raw,prior_donors,current_donors,
           expected_attention_raw,official_shard_hashes,guard):
    """One complete original four-row call with two explicitly selected sources."""
    require(callable(guard),'Independent native replay admission guard required')
    guard()
    normalized_raw,packed,cache_raw,intervention=factorial.select(
        case,own_normalized_raw,own_cache_raw,prior_donors,current_donors,mode)
    require(torch.is_inference_mode_enabled()and torch.get_num_threads()==1,'Original CPU inference mode')
    require(len(model.layers)==42 and model.neuron_config.is_prefill_stage is False
        and model.n_positions==case['bucket'],
        'Complete unchanged original decode model required')
    require(set(cache_raw)==set(boundary.CACHE_NAMES),'Both complete rank-local own-history caches')
    require(type(expected_attention_raw)is bytes and len(expected_attention_raw)==16384
        and boundary.sha(expected_attention_raw)==case['expected_attention']['logical_sha256'],
        'Authenticated complete original4row attention output')
    attention=model.layers[1].self_attn
    require(type(attention).__name__=='NeuronLlamaAttention'
        and type(attention).__module__=='neuronx_distributed_inference.models.llama.modeling_llama',
        'Original installed attention class')
    for module in attention.modules():
        require('forward'not in module.__dict__ and not module._forward_hooks and not module._forward_pre_hooks,
            'No instance overrides or new hooks in direct own-history replay')
    require(attention.tp_degree==2 and attention.cp_degree==attention.dp_degree==1
        and attention.num_heads==8 and attention.num_key_value_heads==1 and attention.head_dim==128,
        'Original rank-local head geometry')
    for name in ('sequence_parallel_enabled','k_cache_transposed','attn_block_tkg_nki_kernel_enabled',
                 'attn_block_cte_nki_kernel_enabled','attn_tkg_builtin_kernel_enabled'):
        require(getattr(attention,name)is False,'Original SDK route '+name)
    require(attention.attention_chunk_size is None and attention.sliding_window is None,
        'Original global attention path')
    params=dict(attention.named_parameters());keys=('qkv_proj.q_proj.weight','qkv_proj.k_proj.weight',
        'qkv_proj.v_proj.weight','o_proj.o_proj.weight')
    require(set(official_shard_hashes)==set(keys),'All4 actual official shard digests required from admitted caller')
    def weights():
        for key in keys:
            value=params[key]
            require(value.dtype==torch.bfloat16 and value.device.type=='cpu'
                and boundary.sha(tensor_bytes(torch,value))==official_shard_hashes[key],
                'Exact original official attention shard '+key)
    weights();guard()
    normalized=torch.frombuffer(bytearray(packed),dtype=torch.bfloat16).as_strided((4,1,2048),(1,4,4),0)
    require(list(normalized.stride())==[1,4,4]and tensor_bytes(torch,normalized)==normalized_raw,
        'Exact original normalized words and physical source layout')
    require(bool(torch.isfinite(normalized).all()),'Original normalized input is finite')
    positions=torch.tensor(case['positions'],dtype=torch.int64).reshape(4,1)
    seq_ids=torch.tensor(case['seq_ids'],dtype=torch.int64)
    mask=torch.tensor([[column<position for column in range(case['bucket'])]for position in case['positions']],
        dtype=torch.bool).reshape(4,1,1,case['bucket'])
    require(metadata(torch,mask)==case['attention_route']['mask']
        and metadata(torch,positions)==case['attention_route']['positions'],'Exact original layer entry routes')
    for i,name in zip((2,3),boundary.CACHE_NAMES):
        raw=cache_raw[name];descriptor=case['caches'][i-2]
        prepared_metadata=dict(descriptor['complete_source_metadata'],sha256=intervention['history_intervention']['output_cache_sha256'][name])
        require(type(raw)is bytes and len(raw)==descriptor['complete_original_bytes']
            and boundary.sha(raw)==prepared_metadata['sha256'],
            'Exact full own-history cache bytes including physical padding')
        value=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(boundary.CACHE_SHAPE)
        require(bool(torch.isfinite(value).all()),'Original finite full cache')
        model.kv_mgr.past_key_values[i].copy_(value)
        require(metadata(torch,model.kv_mgr.past_key_values[i])==
            {k:v for k,v in prepared_metadata.items()if k!='name'},
            'Restored original whole cache storage')
    guard()
    past=model.kv_mgr.get_kv_by_layer_id(idx=1,seq_len=case['bucket'],skip_slice=False,
        seq_ids=seq_ids,is_for_speculation=False,windowed_context_encoding_window_idx=-1)
    require(len(past)==2,'Original SDK K/V pair')
    for i,value in enumerate(past):
        view=case['caches'][i]['sdk_view'];source=model.kv_mgr.past_key_values[i+2]
        require(list(value.shape)==view['shape']and list(value.stride())==view['stride']
            and value.storage_offset()==view['offset']and value.dtype==torch.bfloat16
            and value.untyped_storage().data_ptr()==source.untyped_storage().data_ptr(),
            'Original SDK cache view retains original full storage and physical stride')
    # Original layer0 produces these caches and layer1 consumes them. The pinned
    # SDK RotaryEmbedding uses x only for dtype/device, so use the same module,
    # same BF16 CPU dtype and original positions, retaining non-None caches.
    rope=model.layers[0].self_attn.rotary_emb
    require(type(rope).__name__=='RotaryEmbedding'
        and type(rope).__module__=='neuronx_distributed_inference.modules.attention.utils'
        and rope.base==5000000 and rope.dim==128 and rope.factor is None,'Exact original carried RoPE path')
    cos,sin=rope(normalized,positions)
    require(all(list(v.shape)==[4,1,128]and v.dtype==torch.bfloat16 and v.device.type=='cpu'
        and bool(torch.isfinite(v).all())for v in (cos,sin)),'Original BF16 positional caches')
    unchanged=[normalized,mask,positions,seq_ids,cos,sin,*model.kv_mgr.past_key_values[2:4]]
    before=[metadata(torch,v)for v in unchanged]
    guard()
    output=attention(hidden_states=normalized,attention_mask=mask,position_ids=positions,past_key_value=past,
        active_mask=None,adapter_ids=torch.zeros(4,dtype=torch.int32),cos_cache=cos,sin_cache=sin,
        rmsnorm=None,rotary_position_ids=None,kv_mgr=model.kv_mgr,get_kv_per_layer=False,update_kv_per_layer=False,
        residual=None,windowed_context_encoding_window_idx=-1,seq_ids=seq_ids,idx=1,
        is_for_context_encoding=False,seq_len=case['bucket'],local_mask=None,padding_mask=None)
    guard()
    require([metadata(torch,v)for v in unchanged]==before,'Every original current input and complete history unchanged')
    weights()
    actual=output.hidden_states
    expected=case['expected_attention']['metadata']
    require(list(actual.shape)==expected['shape']and list(actual.stride())==expected['stride']
        and actual.storage_offset()==expected['offset']and actual.dtype==torch.bfloat16
        and actual.device.type=='cpu'and bool(torch.isfinite(actual).all()),'Complete original attention output geometry')
    raw=tensor_bytes(torch,actual)
    if intervention['own_control']:
        require(raw==expected_attention_raw,'Diagonal/restored output must reproduce original bytes exactly')
    require(output.residual is None and output.cos_cache is cos and output.sin_cache is sin,
        'Original non-fused residual and carried positional cache identity')
    return dict(schema='minicpm5_layer1_current_history_factorial_callable_result_v1',mode=mode,
        intervention=intervention,complete=True,rank=case['rank'],arm=case['arm'],
        call=case['call'],attention_output_sha256=boundary.sha(raw),all_four_rows_exact=(raw==expected_attention_raw),
        normalized_original_layout=True,full_input_cache_unchanged=True,original_official_weights_unchanged=True,
        model_reexecuted=False,attention_reexecuted=True,
        official_current_intervention_performed=(mode in ('official_current','official_both')),
        official_history_intervention_performed=(mode in ('official_history','official_both')),
        current_qkv_change_together=(mode in ('official_current','official_both')),
        attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False),raw
