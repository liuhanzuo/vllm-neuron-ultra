"""Layer1 observation payload construction; no model execution or admission.

The caller authenticates the frozen original call, runs the unchanged model,
checks it, then passes existing prediction taps and unchanged input caches.
All14 calls/all32 frames and both ranks belong to the outer runtime contract.
"""
import hashlib
from research.minicpm5_saved_tap_analysis_v1 import ARMS, frame_order
from research import minicpm5_layer1_cache_codec_v1 as codec

TAPS=('layers.1.input_layernorm','layers.1.self_attn','layers.1')
CACHES=('kv_mgr.past_key_values.2','kv_mgr.past_key_values.3')
REPEAT_FIELDS=('call','bucket','input_metadata','execution_input_metadata',
 'integer_widening','changed_mask_cells','output_metadata','stage_names',
 'stage_metadata','stage_sha256','before_metadata','input_after_metadata',
 'post_metadata','route_records','physical_rows','HF_metrics',
 'preservation','candidate_layer0_full_bf16_enabled','candidate_targets',
 'oproj_dtype_records')
MAX_PAYLOAD=3*4*2048*2+2*codec.MAX_PACKED_BYTES
require=codec.require
sha=codec.sha

def same_call(record,original):
    require(record['arm']==original['arm'] and record['arm']in ARMS
        and type(record['rank'])is int and record['rank']==original['rank']
        and record['rank']in(0,1),'Exact original arm and rank')
    for key in REPEAT_FIELDS:require(record[key]==original[key],'Observer changed original '+key)
    require(record['full_model_qualified'] is record['new_speedup_claim'] is False,
            'Original diagnostic remains unqualified')
    require(record['before_metadata']==record['input_after_metadata'],'Forward left input caches unchanged')
    order=frame_order(record)
    require(len(record['stage_names'])==213 and len(record['before_metadata'])==84,
            'Complete original prediction and state observation')
    for i,name in zip((2,3),CACHES):
        row=record['before_metadata'][i]
        require(row['name']==name and row['shape']==list(codec.SHAPE)
            and row['dtype']=='torch.bfloat16','Original entire layer1 cache')
    return order

def build(record,original,original_ref,taps_raw,caches_raw,previous=None):
    order=same_call(record,original);call=record['call']['index']
    require(type(call)is int and 0<=call<14,'Original finite call index')
    require(type(original_ref)is dict and set(original_ref)=={'path','sha256'},
            'Authenticated original call reference supplied by caller')
    require(set(taps_raw)==set(TAPS) and set(caches_raw)==set(CACHES),'All three taps and both full caches')
    if call==0:
        require(previous is None,'First observation has no inherited cache record')
    else:
        require(type(previous)is dict and previous['call']==call-1
            and previous['rank']==record['rank'] and previous['arm']==record['arm']
            and set(previous['caches'])==set(CACHES),'Unbroken same-arm/rank full-cache observation chain')
    parts=[];entries=[];offset=0
    for name in TAPS:
        raw=taps_raw[name]
        require(type(raw)is bytes and len(raw)==len(order)*2048*2
            and sha(raw)==record['stage_sha256'][name],'Exact existing prediction tap '+name)
        entries.append(dict(kind='tap',name=name,offset=offset,bytes=len(raw),sha256=sha(raw),
            shape=[len(order),1,2048],dtype='torch.bfloat16',
            frame_indices=order,source_metadata=record['stage_metadata'][name]))
        parts.append(raw);offset+=len(raw)
    for i,name in zip((2,3),CACHES):
        raw=caches_raw[name];codec.validate_bytes(raw)
        require(sha(raw)==record['before_metadata'][i]['sha256'],'Export is the unchanged input cache')
        base=None if previous is None else previous['caches'][name]
        header,packed=codec.encode(raw,base)
        entries.append(dict(kind='cache',name=name,offset=offset,bytes=len(packed),
            source_metadata=record['before_metadata'][i],codec=header))
        parts.append(packed);offset+=len(packed)
    payload=b''.join(parts)
    require(len(payload)<=MAX_PAYLOAD<16*2**20,'Original single-file bound retained')
    descriptor=dict(schema='minicpm5_layer1_observation_v1',rank=record['rank'],arm=record['arm'],
        call=call,original_call=original_ref,physical_frame_order=order,entries=entries,
        payload_bytes=len(payload),payload_sha256=sha(payload),all_original_fields_exact=True,
        observation_occurs_after_forward=True,full_cache_reconstruction_required=True,
        caches_are_original_input_not_post_state=True,full_model_qualified=False,new_speedup_claim=False)
    next_state=dict(call=call,rank=record['rank'],arm=record['arm'],caches=dict(caches_raw))
    return descriptor,payload,next_state

def reconstruct(descriptor,payload,previous=None):
    require(descriptor['schema']=='minicpm5_layer1_observation_v1'
        and type(payload)is bytes and len(payload)<=MAX_PAYLOAD
        and len(payload)==descriptor['payload_bytes'] and sha(payload)==descriptor['payload_sha256'],
        'Exact bounded observation payload')
    require(descriptor['all_original_fields_exact'] is descriptor['observation_occurs_after_forward']
        is descriptor['full_cache_reconstruction_required'] is descriptor['caches_are_original_input_not_post_state'] is True
        and descriptor['full_model_qualified'] is descriptor['new_speedup_claim'] is False,
        'Observation remains an exact unqualified diagnostic')
    call=descriptor['call']
    require(type(call)is int and 0<=call<14 and type(descriptor['rank'])is int
        and descriptor['rank']in(0,1) and descriptor['arm']in ARMS,'Original call identity')
    if call==0:require(previous is None,'First reconstructed call has no history')
    else:require(type(previous)is dict and previous['call']==call-1
        and previous['rank']==descriptor['rank'] and previous['arm']==descriptor['arm'],
        'Reconstruct complete same-arm/rank sequence')
    entries=descriptor['entries']
    require([x['name']for x in entries]==list(TAPS+CACHES),'Complete ordered observation entries')
    offset=0;taps={};caches={}
    order=descriptor['physical_frame_order']
    require(type(order)is list and len(order)in(1,4) and len(set(order))==len(order)
        and all(type(x)is int and 0<=x<32 for x in order),'Original physical prediction frames')
    for i,e in enumerate(entries):
        size=e['bytes']
        require(type(size)is int and size>=0 and type(e['offset'])is int
            and e['offset']==offset and offset+size<=len(payload),'Complete contiguous payload')
        data=payload[offset:offset+size];offset+=size
        if i<3:
            require(e['kind']=='tap' and e['frame_indices']==order and e['shape']==[len(order),1,2048]
                and e['dtype']=='torch.bfloat16' and size==len(order)*4096
                and sha(data)==e['sha256'],'Exact existing selected tap')
            taps[e['name']]=data
        else:
            require(e['kind']=='cache' and e['source_metadata']['name']==e['name']
                and e['source_metadata']['shape']==list(codec.SHAPE)
                and e['source_metadata']['dtype']=='torch.bfloat16','Complete cache identity')
            base=None if previous is None else previous['caches'][e['name']]
            raw=codec.decode(e['codec'],data,base)
            require(sha(raw)==e['source_metadata']['sha256'],'Reconstructed complete input cache')
            caches[e['name']]=raw
    require(offset==len(payload),'No unreferenced observation bytes')
    return taps,dict(call=call,rank=descriptor['rank'],arm=descriptor['arm'],caches=caches)
