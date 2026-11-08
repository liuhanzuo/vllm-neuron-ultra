"""Actual complete target-request graph inputs for write observer repeat."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=ROOT/'research'
sys.path.insert(0,str(Q));sys.path.insert(0,str(ROOT))
import analyze_smollm3_sio14_write_first_divergence_v1 as first

def run():
    prior=Q/'smollm3-sio14-first-write-divergence-graph-input-analysis-20261108-v1.json'
    repeated=Q/'smollm3-sio14-write-on-repeat-root-audit-20261108-v1.json'
    assert first.reference(prior)['sha256']=='68a48ce8b146feae55c62455268c5aa8c50d52e24464be632e796be0f1e49c19'
    assert first.reference(repeated)['sha256']=='6f126cdd89ef979cd2b393a76a159c168f70bbe70bd2d9ac44e6dad2ce395684'
    original=json.loads(prior.read_text());repeat=json.loads(repeated.read_text())
    refs={name:original['provenance'][name]['export']for name in ('r01','r05')};refs['r06']=repeat['export']
    requests={};frames={};decoded={};provenance={}
    for name,ref in refs.items():
        path=ROOT/'sio14'/name/'replay-export.json';assert first.reference(path)==ref
        export=first.audit.l.read(path);request=export['requests'][-1]
        assert request['group']=='short-round0-single'and request['index']==0 and len(request['samples'])==512
        frames[name],chunks=first.load_frames(name,export,request)
        decoded[name],raw=first.raw_selected(name,export,frames[name]);requests[name]=request
        provenance[name]=dict(export=ref,chunks=chunks,observer=raw,graph_frames=export['frames_count'],engine_steps=export['engine_steps_count'])
    pairs={}
    for left_name,right_name in [('r05','r06'),('r01','r06')]:
        a,b=requests[left_name],requests[right_name];assert a['prompt_token_ids']==b['prompt_token_ids']
        different_tokens=[];different_inputs=[];rows=[]
        for index,(x,y)in enumerate(zip(a['samples'],b['samples'])):
            assert len(x['frames'])==len(y['frames'])==1
            lf=frames[left_name][x['frames'][0]];rf=frames[right_name][y['frames'][0]]
            differences=[i for i,(v,w)in enumerate(zip(decoded[left_name][lf['index']],decoded[right_name][rf['index']]))if v!=w]
            equal=x['token']==y['token']
            if not equal:different_tokens.append(index)
            if differences:different_inputs.append(index)
            rows.append(dict(generated_index=index,token_equal=equal,left_token=x['token'],right_token=y['token'],
                physical_slots=[x['physical_slot'],y['physical_slot']],physical_slot_equal=x['physical_slot']==y['physical_slot'],
                graph_tag_equal=lf['tag']==rf['tag'],graph_route_equal=lf['route']==rf['route'],different_external_input_arguments=differences))
        limit=different_tokens[0]+1 if different_tokens else 512
        pairs[left_name+'_versus_'+right_name]=dict(different_generated_indices=different_tokens,different_external_input_indices=different_inputs,
            external_inputs_slot_tag_route_equal_through_first_difference=all(not row['different_external_input_arguments']and row['physical_slot_equal']and row['graph_tag_equal']and row['graph_route_equal']for row in rows[:limit]),
            invocations_through_first_difference=limit,per_generated_token=rows)
    report=dict(schema='smollm3_sio14_write_repeat_graph_inputs_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        first_divergence_analysis=first.reference(prior),write_repeat_audit=first.reference(repeated),provenance=provenance,pairs=pairs,
        actual_raw_graph_arguments_decoded=True,full_request_steps=512,external_arguments_per_step=7,
        internal_cache_values_equal_proven=False,internal_storage_aliases_equal_proven=False,global_history_equal_proven=False,
        model_execution_performed=False,teacher_forced_replay_executed=False,causal_attribution_proven=False,full_model_qualified=False,new_speedup_claim=False)
    p=Q/'smollm3-sio14-write-repeat-graph-input-analysis-20261108-v1.json';ref=first.once(p,report)
    print(json.dumps(dict(report=ref,pairs={k:{f:v[f]for f in ['different_generated_indices','different_external_input_indices','external_inputs_slot_tag_route_equal_through_first_difference','invocations_through_first_difference']}for k,v in pairs.items()})))
if __name__=='__main__':run()
