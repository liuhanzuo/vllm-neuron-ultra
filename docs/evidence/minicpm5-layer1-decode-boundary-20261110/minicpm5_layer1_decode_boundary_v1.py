"""Exact decode replay boundaries, derived from original authenticated metadata.

No model/SDK import, tensor read, launch, or runtime admission. Reconstructing
request masks below is metadata-derived; real tap/cache bytes must come from a
separately completed and independently audited observation collection.
"""
import hashlib
import math
import struct
from research.minicpm5_layer1_raw_verifier_v1 import ARMS
from research.minicpm5_layer1_payload_io_v1 import require
DECODE_CALLS=(4,5,6,11,12,13)
NORMALIZED='layers.1.input_layernorm'
ATTENTION='layers.1.self_attn'
CACHE_NAMES=('kv_mgr.past_key_values.2','kv_mgr.past_key_values.3')
CACHE_SHAPE=[4,1,4224,128]
CACHE_STRIDE=[540672,540672,128,1]
NORM_META=dict(shape=[4,1,2048],stride=[1,4,4],offset=0,dtype='torch.bfloat16',prediction_positions=[0]*4)
OUTPUT_META=dict(shape=[4,1,2048],stride=[2048,2048,1],offset=0,dtype='torch.bfloat16',prediction_positions=[0]*4)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def tensor_meta(raw,shape,stride,dtype):
    widths={'torch.bfloat16':2,'torch.bool':1,'torch.int64':8,'torch.int32':4}
    require(dtype in widths and len(raw)==math.prod(shape)*widths[dtype],'Complete logical bytes')
    return dict(shape=list(shape),stride=list(stride),offset=0,dtype=dtype,sha256=sha(raw))


def normalized_storage(logical,source_metadata,wanted_sha256):
    """Preserve BF16 words while restoring the actual [1,4,4] source stride."""
    require(source_metadata==NORM_META,'Exact observed normalized decode source layout')
    require(type(logical)is bytes and len(logical)==16384 and sha(logical)==wanted_sha256,
        'All original BF16 normalized words')
    storage=bytearray(16384)
    for batch in range(4):
        for channel in range(2048):
            src=2*(batch*2048+channel);dst=2*(batch+4*channel)
            storage[dst:dst+2]=logical[src:src+2]
    return bytes(storage)


def boundary(plan,record,original_ref):
    call=record['call'];index=call['index'];rank=record['rank'];arm=record['arm']
    require(record['complete']is True and type(rank)is int and rank in (0,1)and arm in ARMS,
        'Actual complete original rank and arm')
    require(index in DECODE_CALLS and call==plan['calls'][index]and call['phase']=='decode',
        'Only complete original six decode calls; prefill taps are not full inputs')
    expected_ref=plan['layer1_observation']['original_calls'][rank*42+ARMS.index(arm)*14+index]
    require(original_ref=={k:expected_ref[k]for k in ('path','sha256')},'Exact frozen original call reference')
    bucket=record['bucket'];require(bucket==plan['graph_calls'][index]['bucket']and bucket in (128,2048),
        'Exact original decode bucket')
    mapping=record['physical_rows'];frames=call['frames']
    require(len(mapping)==4 and [r['frame']for r in mapping]==frames
        and sorted(r['physical_row']for r in mapping)==list(range(4)),
        'All four original physical rows, in recorded logical order')
    physical={};cells=[]
    for row in mapping:
        frame=row['frame'];slot=row['physical_row']
        require(frame in plan['frames']and frame['step']in (1,2,3)and slot==frame['slot'],
            'Original slot/frame identity')
        fixture=plan['fixtures'][frame['fixture']]
        require(fixture['id']==frame['fixture'],'Original indexed fixture')
        position=len(fixture['input_ids'])+frame['step']-1
        require(row['current_position']==position and 0<position<bucket,'Original logical current position')
        physical[slot]=dict(frame=frame,position=position,token=fixture['continuation_ids'][frame['step']-1])
        cells.append(dict(row=slot,column=position,before=1,after=0))
    require(record['changed_mask_cells']==cells,'Only the original strict current-mask cells changed')
    tokens=[physical[i]['token']for i in range(4)]
    positions=[physical[i]['position']for i in range(4)]
    pack=lambda values:struct.pack('<'+'q'*len(values),*values)
    mask=[int(column<position)for position in positions for column in range(bucket)]
    raw=[pack(tokens),pack(mask),pack(positions),pack(range(4))]
    metas=[tensor_meta(raw[0],[4,1],[1,1],'torch.int64'),
        tensor_meta(raw[1],[4,bucket],[bucket,1],'torch.int64'),
        tensor_meta(raw[2],[4,1],[1,1],'torch.int64'),tensor_meta(raw[3],[4],[1],'torch.int64')]
    require(record['execution_input_metadata'][:4]==metas,
        'Original widened tokens/strict mask/positions/slots reproduced exactly')
    for i in (5,6):
        require(record['execution_input_metadata'][i]==tensor_meta(bytes(16),[4],[1],'torch.int32'),
            'Original zero placeholder/adapter IDs')
    bool_mask=bytes(mask);mask_meta=tensor_meta(bool_mask,[4,1,1,bucket],[bucket,bucket,bucket,1],'torch.bool')
    route=dict(layer=1,mask=mask_meta,positions=metas[2],active_mask=None,physical_key_width=bucket,phase='decode')
    require(len(record['route_records'])==42 and record['route_records'][1]==route,
        'Exactly reconstructed actual layer1 entry mask and positions')
    require(record['stage_metadata'][NORMALIZED]==NORM_META
        and record['stage_metadata'][ATTENTION]==OUTPUT_META,'Original complete decode source/output layouts')
    caches=[]
    for i,name in zip((2,3),CACHE_NAMES):
        meta=record['before_metadata'][i]
        require({k:v for k,v in meta.items()if k!='sha256'}==dict(name=name,shape=CACHE_SHAPE,
            stride=CACHE_STRIDE,offset=0,dtype='torch.bfloat16'),'All4slots/all4224cache positions retained')
        require(record['input_after_metadata'][i]==meta,'Full input cache unchanged by original forward')
        caches.append(dict(name=name,complete_source_metadata=meta,
            sdk_view=dict(shape=[4,1,bucket,128],stride=CACHE_STRIDE,offset=0,
                dtype='torch.bfloat16',operation='original_KVCacheManager_get_kv_by_layer_id',
                idx=1,seq_len=bucket,skip_slice=False,seq_ids=[0,1,2,3],padding_side='right'),
            view_must_retain_complete_storage=True,complete_original_bytes=4325376))
    return dict(schema='minicpm5_layer1_decode_boundary_v1',rank=rank,arm=arm,call=call,bucket=bucket,
        original_call=original_ref,physical_frame_order=[physical[i]['frame']for i in range(4)],
        tokens=tokens,positions=positions,seq_ids=[0,1,2,3],strict_mask_kind='prior_positions_only',
        original_execution_metadata=metas,attention_route=route,
        normalized_input=dict(name=NORMALIZED,metadata=NORM_META,logical_sha256=record['stage_sha256'][NORMALIZED],
            logical_bytes=16384,restore_source_stride=True),
        expected_attention=dict(name=ATTENTION,metadata=OUTPUT_META,logical_sha256=record['stage_sha256'][ATTENTION],logical_bytes=16384),
        caches=caches,original_layer_index=1,full_attention_output_required=True,
        paired_history_intervention_allowed=False,own_history_reproduction_required=True,
        source_only=True,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)


def all_boundaries(plan,read_json):
    require(callable(read_json),'Authenticated bounded metadata reader required')
    refs=plan['layer1_observation']['original_calls'];require(len(refs)==84,'Complete original catalog')
    output=[]
    for rank in (0,1):
        for ai,arm in enumerate(ARMS):
            for ci in DECODE_CALLS:
                item=refs[rank*42+ai*14+ci];ref={k:item[k]for k in ('path','sha256')}
                require([item['rank'],item['arm'],item['call']]==[rank,arm,ci],'Original complete decode ordering')
                record=read_json(ref['path'],ref['sha256']);output.append(boundary(plan,record,ref))
    require(len(output)==36,'All36 original rank/arm/decode boundaries')
    for ai,arm in enumerate(ARMS):
        for ci in range(6):
            a=output[ai*6+ci];b=output[18+ai*6+ci]
            require(a['normalized_input']==b['normalized_input']and a['expected_attention']==b['expected_attention']
                and a['attention_route']==b['attention_route'],'Both rank replicated current/output boundaries')
            # Caches belong to their own rank and must not be deduplicated.
    return output


def weight_slices(plan,header):
    source=plan['official_layer0_weights']['path'];extent=plan['inputs'][source]['identity']['size']
    rows=[]
    for rank in (0,1):
        for name,shape,width,axis in (('q_proj',[2048,2048],1024,0),('k_proj',[256,2048],128,0),
            ('v_proj',[256,2048],128,0),('o_proj',[2048,2048],1024,1)):
            key='model.layers.1.self_attn.'+name+'.weight';entry=header[key];start,end=entry['data_offsets']
            require(entry['dtype']=='BF16'and entry['shape']==shape and 0<=start<end<extent
                and end-start==math.prod(shape)*2<=8*2**20,'Complete original official layer1 matrix')
            rows.append(dict(rank=rank,official_key=key,source_file=source,header_entry=entry,
                shard_axis=axis,start=rank*width,stop=(rank+1)*width,
                logical_query_heads=list(range(rank*8,(rank+1)*8)),logical_kv_heads=[rank],
                raw_weight_comparison_required_at_runtime=True))
    return rows
