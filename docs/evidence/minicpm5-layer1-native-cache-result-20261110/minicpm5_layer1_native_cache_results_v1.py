"""Independent stdlib validation of full native caches, logits and all selected taps."""
import hashlib
import struct
from pathlib import Path
from research import minicpm5_layer1_payload_io_v1 as payload
import minicpm5_layer1_kv_factorial_cpu_runner_v1 as predecessor
p=predecessor.p;require=p.require


def sha(data):return hashlib.sha256(data).hexdigest()
def finite_bf16(data):return len(data)%2==0 and all((word&0x7f80)!=0x7f80 for (word,)in struct.iter_unpack('<H',data))

def validate(result,plan,guard):
    guard();require(result['schema']=='minicpm5_layer1_native_cache_result_v1'and result['complete']is True
        and result['rank']==-1 and result['frames']==32,'Complete official native32 result')
    for name in ('full_model_qualified','new_speedup_claim','self_exit_claimed'):require(result[name]is False,'Diagnostic only '+name)
    for name in ('original_32_model_forwards','full_logits_raw_exact','all213_selected_taps_raw_exact','original_native_history_preserved'):
        require(result[name]is True,'Original native reproduction '+name)
    require(result['native_provenance']==plan['native_provenance']and result['completed_factorial']==plan['layer1_native_cache'],
        'Exact native model and actual prerequisite')
    mapping={(f['fixture'],f['step']):f for f in plan['frames']}
    expected=[mapping[f['id'],step]for f in plan['fixtures']for step in range(4)]
    require(len(result['records'])==32,'All32 native frames')
    cache_count=logit_count=tap_count=0;outputs=[];previous={}
    for row,frame in zip(result['records'],expected):
        guard();fixture=plan['fixtures'][frame['fixture']];length=len(fixture['input_ids'])+frame['step']
        require(row['schema']=='minicpm5_layer1_native_cache_frame_v1'and row['complete']is True
            and row['frame']==frame and row['native_fixture_order']==fixture['id']and row['step']==frame['step']
            and row['cache_length']==length,'Exact original fixture/step/slot order and complete cache length')
        require(row['original_reference']==plan['hf_reference']and row['original_frame']==plan['hf_frame_records'][frame['index']],
            'Exact original native reference provenance')
        for name in ('full_logits_raw_exact','all213_selected_taps_raw_exact','official_bf16','history_unmodified'):require(row[name]is True,'Original native frame '+name)
        require(row['full_model_qualified']is row['new_speedup_claim']is False,'No qualification promotion')
        ref=row['original_frame'];require(plan['inputs'][ref['path']]['sha256']==ref['sha256'],'Bound original metadata input')
        original=p.read_ref(ref)
        require(row['tap_metadata']==original['metadata']and len(row['tap_entries'])==213,'All original tap layouts retained')
        folder=Path(plan['output_root'])/'native'/('frame-%02d'%frame['index'])
        require(p.read(folder/'result.json')==row,'Native frame persisted exactly')
        require(set(row['outputs'])=={'key','value','logits','taps'},'All complete native raw payloads')
        data={}
        for name,filename in (('key','key.bf16'),('value','value.bf16'),('logits','logits.fp32'),('taps','selected-taps.bf16')):
            descriptor=row['outputs'][name];path=folder/filename
            require(descriptor['path']==str(path)and 0<descriptor['raw_bytes']<=16*2**20,'Bounded owned full output path')
            data[name]=payload.read_owned_payload(dict(descriptor,payload_bytes=descriptor['raw_bytes'],payload_sha256=descriptor['sha256']),path,guard)
            if name!='logits':require(finite_bf16(data[name]),'Every original BF16 word finite')
        require(set(row['cache_metadata'])=={'key','value'},'Full K/V metadata')
        for name in ('key','value'):
            meta=row['cache_metadata'][name]
            require(meta['shape']==[1,2,length,128]and meta['dtype']=='torch.bfloat16'
                and len(meta['stride'])==4 and all(type(v)is int and v>=0 for v in meta['stride'])
                and type(meta['offset'])is int and meta['offset']>=0 and len(data[name])==length*512,
                'Both KV heads and every native valid prefix position')
            if frame['step']:
                old=previous[fixture['id']][name];old_length=length-1
                for head in (0,1):
                    require(old[head*old_length*256:(head+1)*old_length*256]==data[name][head*length*256:head*length*256+old_length*256],
                        'All previous native cache bytes retained by each original cached step')
        previous[fixture['id']]={k:data[k]for k in ('key','value')}
        entries=row['tap_entries'];offset=0;lmhead=None
        require([e['name']for e in entries]==list(original['tap_raw_sha256']),'All213 original selected tap names in order')
        for e in entries:
            width=original['metadata'][e['name']]['source_shape'][-1];nbytes=width*2
            require(e['shape']==[1,1,width]and e['offset']==offset and e['bytes']==nbytes,'Complete selected tap extent')
            piece=data['taps'][offset:offset+nbytes]
            require(len(piece)==nbytes and sha(piece)==e['sha256']==original['tap_raw_sha256'][e['name']],
                'Every full selected tap independently equal to original bytes')
            if e['name']=='lm_head':lmhead=piece
            offset+=nbytes
        require(offset==len(data['taps'])and lmhead is not None and len(lmhead)==130560*2,'Complete packed selected taps and vocabulary')
        expected_logits=b''.join(b'\x00\x00'+lmhead[i:i+2]for i in range(0,len(lmhead),2))
        require(data['logits']==expected_logits and len(data['logits'])==130560*4,
            'Complete original BF16 lm_head promoted exactly to full FP32 logits')
        cache_count+=len(data['key'])+len(data['value']);logit_count+=len(data['logits']);tap_count+=len(data['taps'])
        outputs.append(dict(frame=frame,cache_length=length,outputs=row['outputs'],original_frame=ref))
    require([cache_count,logit_count,tap_count]==[result['full_cache_bytes'],result['full_logits_bytes'],result['selected_tap_bytes']]
        and cache_count==5894144 and logit_count==16711680,'All32 full native payload totals')
    guard();return dict(schema='minicpm5_layer1_native_cache_independent_verification_v1',complete=True,frames=32,
        cache_tensors=64,full_cache_bytes=cache_count,full_logits_bytes=logit_count,selected_tap_bytes=tap_count,
        raw_bytes=cache_count+logit_count+tap_count,selected_taps=32*213,outputs=outputs,
        every_raw_byte_independently_reread=True,original_full_logits_and_all_selected_taps_exact=True,
        all_native_cache_prefix_links_exact=True,full_model_qualified=False,new_speedup_claim=False)
