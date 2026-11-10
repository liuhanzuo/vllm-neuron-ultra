"""Read-only layer1 cache collection on the exact original official32 trajectory."""
import hashlib
import os
from pathlib import Path
import minicpm5_layer0_native_adapter_v2 as loader
import minicpm5_hf32_boundary_observer_v2 as observer
from research import minicpm5_layer1_payload_io_v1 as payload
from minicpm5_layer0_runtime_auth_v1 import require


def raw(torch,value):return bytes(memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()))
def sha(data):return hashlib.sha256(data).hexdigest()


def persist(context,directory,name,data,guard):
    guard();context.guard('native')
    require(name in ('key.bf16','value.bf16','logits.fp32','selected-taps.bf16')
        and type(data)is bytes and 0<len(data)<=16*2**20,'Bounded complete original tensor output')
    path=payload.safe_path(directory/name)
    with path.open('xb')as f:require(f.write(data)==len(data),'Complete byte write');f.flush();os.fsync(f.fileno())
    desc=dict(path=str(path),raw_bytes=len(data),sha256=sha(data),identity=payload.file_identity(path,path.lstat()))
    got=payload.read_owned_payload(dict(desc,payload_bytes=len(data),payload_sha256=desc['sha256']),path,guard)
    require(got==data,'Complete saved-byte reread');return desc


def run(context,guard):
    guard();context.guard('native');require(context._sources_checked,'Source authentication before native load')
    with context.activate():
        torch,model,reference,original_records=loader.load(context)
        guard();plan=context.plan
        require(torch.get_num_threads()==2 and not model.training and len(model.model.layers)==42,'Original native geometry and threads')
        original_records={r['frame']['index']:r for r in original_records}
        mapping={(f['fixture'],f['step']):f for f in plan['frames']}
        out=Path(plan['output_root'])/'native';out.mkdir(mode=0o700)
        hooks=observer.Hooks(model,torch);rows=[];cache_bytes=logits_bytes=tap_bytes=0
        try:
            with torch.inference_mode():
                for fixture,expected in zip(plan['fixtures'],reference):
                    require(fixture['id']==expected['id'] and fixture['input_ids']==expected['input_ids']
                        and fixture['continuation_ids']==expected['continuation_ids'],'Original fixture and teacher-forced sequence')
                    length=len(fixture['input_ids']);output=None;previous_cache=None
                    for step in range(4):
                        guard();frame=mapping[fixture['id'],step];original=original_records[frame['index']]
                        logical=length+step-1;hooks.begin(length if step==0 else 1,logical)
                        if step==0:
                            output=model(input_ids=torch.tensor([fixture['input_ids']]),attention_mask=torch.ones((1,length),dtype=torch.long),
                                position_ids=torch.arange(length)[None],use_cache=True,logits_to_keep=1)
                        else:
                            token=fixture['continuation_ids'][step-1]
                            require(token==int(expected['logits'][step-1].argmax()),'Original greedy teacher-forced token')
                            output=model(input_ids=torch.tensor([[token]]),attention_mask=torch.ones((1,length+step),dtype=torch.long),
                                position_ids=torch.tensor([[logical]]),past_key_values=output.past_key_values,use_cache=True,logits_to_keep=1)
                        loader.original._cache(output.past_key_values,length+step)
                        taps,metadata=hooks.finish();logits=output.logits[0,-1].detach().float().cpu().clone()
                        logits_raw=raw(torch,logits)
                        require(logits_raw==raw(torch,expected['logits'][step]),'All original complete vocabulary logits raw exact')
                        require(metadata==original['metadata'],'Every213 original tap source metadata exact')
                        pieces=[];entries=[];offset=0
                        for name,value in taps.items():
                            data=raw(torch,value)
                            require(sha(data)==original['tap_raw_sha256'][name],'Original selected tap bytes '+name)
                            pieces.append(data);entries.append(dict(name=name,offset=offset,bytes=len(data),shape=list(value.shape),sha256=sha(data)))
                            offset+=len(data)
                        require(len(entries)==213 and raw(torch,taps['lm_head'].float().reshape(-1))==logits_raw,
                            'All213 selected taps and exact full-vocabulary BF16-to-FP32 endpoint')
                        layer=output.past_key_values.layers[1];cache={};cache_meta={}
                        for name,value in (('key',layer.keys),('value',layer.values)):
                            require(list(value.shape)==[1,2,length+step,128]and value.dtype==torch.bfloat16
                                and value.device.type=='cpu'and bool(torch.isfinite(value).all()),'Complete finite official layer1 cache')
                            cache[name]=raw(torch,value)
                            cache_meta[name]=dict(shape=list(value.shape),stride=list(value.stride()),offset=value.storage_offset(),dtype=str(value.dtype))
                        if previous_cache is not None:
                            old_length=length+step-1
                            for name in ('key','value'):
                                for head in (0,1):
                                    old=previous_cache[name][head*old_length*256:(head+1)*old_length*256]
                                    new=cache[name][head*(old_length+1)*256:head*(old_length+1)*256+old_length*256]
                                    require(old==new,'Native original history prefix retained for every head')
                        folder=out/('frame-%02d'%frame['index']);folder.mkdir()
                        saved={}
                        for name,filename,data in (('key','key.bf16',cache['key']),('value','value.bf16',cache['value']),
                            ('logits','logits.fp32',logits_raw),('taps','selected-taps.bf16',b''.join(pieces))):
                            saved[name]=persist(context,folder,filename,data,guard)
                        row=dict(schema='minicpm5_layer1_native_cache_frame_v1',complete=True,frame=frame,native_fixture_order=fixture['id'],
                            step=step,cache_length=length+step,original_frame=plan['hf_frame_records'][frame['index']],
                            original_reference=plan['hf_reference'],cache_metadata=cache_meta,tap_metadata=metadata,tap_entries=entries,outputs=saved,
                            full_logits_raw_exact=True,all213_selected_taps_raw_exact=True,official_bf16=True,history_unmodified=True,
                            full_model_qualified=False,new_speedup_claim=False)
                        # No extra forward, cache update, cast or precision change is used for capture.
                        from research.minicpm5_layer1_native_cache_results_v1 import p
                        p.once(folder/'result.json',row);rows.append(row)
                        cache_bytes+=len(cache['key'])+len(cache['value']);logits_bytes+=len(logits_raw);tap_bytes+=offset
                        previous_cache=cache;guard();print('NATIVE_LAYER1_FRAME',frame['index'],flush=True)
                    del output,previous_cache
        finally:hooks.close()
        require(len(rows)==32 and cache_bytes==5894144 and logits_bytes==16711680,'Complete original32 native cache and logits capture')
        guard()
        return dict(schema='minicpm5_layer1_native_cache_result_v1',complete=True,rank=-1,records=rows,
            frames=32,full_cache_bytes=cache_bytes,full_logits_bytes=logits_bytes,selected_tap_bytes=tap_bytes,
            native_provenance=plan['native_provenance'],original_32_model_forwards=True,full_logits_raw_exact=True,
            all213_selected_taps_raw_exact=True,original_native_history_preserved=True,
            completed_factorial=plan['layer1_native_cache'],full_model_qualified=False,new_speedup_claim=False,self_exit_claimed=False)
