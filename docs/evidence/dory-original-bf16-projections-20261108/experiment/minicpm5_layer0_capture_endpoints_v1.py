"""Full original endpoint checks; model/tensor operations only when called.

Caller authenticates frozen source/JSON/payload identities before using these
helpers. No model import, tensor read, service launch or baseline rewrite here.
"""
import hashlib
import math


def require(value,message):
    if not value:raise RuntimeError(message)


def tensor_digest(torch,value):
    require(value.device.type=='cpu' and value.numel()*value.element_size()<=8*2**20,'Bounded CPU endpoint only')
    return hashlib.sha256(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy())).hexdigest()


def tensor_metadata(torch,value):
    return dict(shape=list(value.shape),stride=list(value.stride()),offset=value.storage_offset(),dtype=str(value.dtype),sha256=tensor_digest(torch,value))


def native_endpoint(torch,logits,expected,frame,expected_record,expected_payload_sha):
    require(list(logits.shape)==list(expected.shape)==[130560] and str(logits.dtype)==str(expected.dtype)=='torch.float32','Native full-vocabulary endpoint shape/dtype')
    actual=tensor_digest(torch,logits);wanted=tensor_digest(torch,expected);require(actual==wanted,'Original native32 logits changed')
    require(expected_record['frame']==frame and expected_record['original_logits_raw_exact'] is True,'Wrong retained nativeframe')
    # The retained v2 lm_head hash is BF16; original full-float logits are checked
    # directly against the original frozen baseline payload, never re-created.
    return dict(schema='minicpm5_layer0_native_endpoint_v1',passed=True,raw_exact=True,original_scope_preserved=True,
        frame=frame,full_vocab=130560,logits_sha256=actual,original_reference_payload_sha256=expected_payload_sha,
        original_v2_frame_payload_sha256=expected_record['payload_sha256'],full_model_qualified=False)


def verify_states(torch,values,records,rank,label):
    require(len(values)==len(records)==84,'All84 '+label+' states required')
    require([(r['rank'],r['name']) for r in records]==[(rank,'kv_mgr.past_key_values.%d'%i) for i in range(84)],'State record order/rank')
    hashes=[]
    for value,row in zip(values,records):
        require(list(value.shape)==row['shape']==[4,1,4224,128] and str(value.dtype)==row['dtype']=='torch.bfloat16','State shape/dtype')
        digest=tensor_digest(torch,value);require(digest==row['raw_sha256'],'Original '+label+' differs: '+row['name']);hashes.append(digest)
    return hashes


def cpu_endpoint(torch,output,taps,tap_metadata,input_states,post_states,expected,before_records,after_records,rank):
    require(expected['complete'] is True and expected['rank']==rank and expected['arm'] in ('C_Hd_control','C_Hc_history_generation'),'Wrong original CPU endpoint')
    require(len(expected['stage_names'])==213 and list(taps)==expected['stage_names']==list(tap_metadata),'All213 original taps required')
    require(tensor_metadata(torch,output)==expected['output_metadata'],'Original CPU output metadata/bytes changed')
    require(list(output.shape)==[len(expected['call']['frames']),1,130560] and str(output.dtype)=='torch.float32','CPU full-vocabulary output')
    tap_hashes={name:tensor_digest(torch,taps[name]) for name in expected['stage_names']}
    require(tap_hashes==expected['stage_raw_sha256'] and tap_metadata==expected['stage_metadata'],'Original CPU taps/layout changed')
    before=verify_states(torch,input_states,before_records,rank,'unchanged input')
    after=verify_states(torch,post_states,after_records,rank,'post')
    return dict(schema='minicpm5_layer0_cpu_endpoint_v1',passed=True,raw_exact=True,original_scope_preserved=True,arm=expected['arm'],rank=rank,call=expected['call'],
        output_sha256=expected['output_metadata']['sha256'],all213_tap_sha256=tap_hashes,
        all84_input_hashes=before,all84_post_hashes=after,input_state_unchanged=True,original_poststate_reproduced=True,full_model_qualified=False)


def physical_rows(call,seq_ids):
    wanted=[call['frames'][0]['slot']] if call['phase']=='prefill' else [0,1,2,3]
    require(seq_ids==wanted,'Actual physical seq_ids changed')
    rows=[dict(frame=f,physical_row=seq_ids.index(f['slot'])) for f in call['frames']]
    require(len({r['physical_row'] for r in rows})==len(rows),'Duplicate physical row')
    return rows


def check_head_shards(torch,native_layer,shard,rank):
    """Prove contiguous-head mapping by exact official weight bytes, no assumption."""
    require(rank in (0,1),'TP rank');rows={}
    for name,width,axis in [('q_proj',1024,0),('k_proj',128,0),('v_proj',128,0),('o_proj',1024,1)]:
        official=getattr(native_layer.self_attn,name).weight
        key='layers.0.self_attn.'+('o_proj.o_proj.weight' if name=='o_proj' else 'qkv_proj.'+name+'.weight')
        local=shard[key];sl=official[rank*width:(rank+1)*width,:] if axis==0 else official[:,rank*width:(rank+1)*width]
        require(list(sl.shape)==list(local.shape) and sl.dtype==local.dtype,'Head shard shape/dtype')
        require(tensor_digest(torch,sl)==tensor_digest(torch,local),'Official-to-TP2 head weight map differs')
        rows[name]=dict(axis=axis,start=rank*width,stop=(rank+1)*width,sha256=tensor_digest(torch,local))
    return dict(passed=True,rank=rank,q_head_ids=list(range(rank*8,(rank+1)*8)),kv_head_ids=[rank],weights=rows,logical_head_mapping_raw_proven=True)
