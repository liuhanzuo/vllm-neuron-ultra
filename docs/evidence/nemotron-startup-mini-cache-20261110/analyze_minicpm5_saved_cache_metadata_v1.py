"""Cross-check all saved Mini cache/input/route metadata; no tensor IO."""
from pathlib import Path
import hashlib,json
import analyze_minicpm5_saved_tap_metadata_v1 as old
from minicpm5_saved_tap_analysis_v1 import ARMS,STAGES,require

Q=old.Q
DIAGNOSIS=dict(path=str(Q/'minicpm5-saved-tap-metadata-diagnosis-20261109-v1.json'),
              sha256='d14e24cbc371039b709ea9acfe5bda7e6b71e1208c050f8e87d9bc26b384e4be')
SDK=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/modules/kvcache/kv_cache_manager.py')

def values(rows):
    return [{k:r[k]for k in ('name','shape','dtype','sha256')}for r in rows]

def differences(a,b):
    require(len(a)==len(b)==84,'All 42 K/V pairs')
    indices=[]
    for i,(left,right)in enumerate(zip(a,b)):
        require(left['name']==right['name']==f'kv_mgr.past_key_values.{i}','Exact ordered state names')
        require({k:v for k,v in left.items()if k!='sha256'}==
                {k:v for k,v in right.items()if k!='sha256'},'Same recorded cache geometry/layout')
        if left['sha256']!=right['sha256']:indices.append(i)
    return dict(different_tensor_indices=indices,different_layers=sorted({i//2 for i in indices}),
                layer0_KV_equal=not any(i in indices for i in (0,1)),
                layer1_K_equal=2 not in indices,layer1_V_equal=3 not in indices)

def main():
    raw=Path(DIAGNOSIS['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==DIAGNOSIS['sha256'],'Frozen diagnosis fingerprint')
    diagnosis=json.loads(raw);reader=old.Reader();calls={}
    def read(ref):
        value=reader(ref)
        if 'selected_tap_bundle'in value:calls[value['rank'],value['arm'],value['call']['index']]=(ref,value)
        return value
    current=old.analyze(read)
    require(all(diagnosis.get(k)==v for k,v in current.items()),'Complete original diagnosis unchanged')
    spec=reader(current['original_spec'])
    sdk_sha=hashlib.sha256(SDK.read_bytes()).hexdigest()
    require(spec['source_sha256'][str(SDK)]==sdk_sha,'Original SDK cache mapping source')
    chained=0
    for rank in (0,1):
        for arm in ARMS:
            previous=None
            for i in range(14):
                _,row=calls[rank,arm,i]
                require(row['before_metadata']==row['input_after_metadata'],'Forward did not change input caches')
                if previous is not None:
                    require(values(row['before_metadata'])==values(previous),'All84 own-history cache values preserved')
                    chained+=1
                previous=row['post_metadata']
    comparisons=[]
    for left,right in ((ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])):
        ranks=[]
        for rank in (0,1):
            rows=[]
            for i in range(14):
                ar,a=calls[rank,left,i];br,b=calls[rank,right,i]
                require(a['call']==b['call'],'Same complete trajectory')
                require(a['execution_input_metadata']==b['execution_input_metadata'],'All seven current inputs exactly match')
                require(a['route_records']==b['route_records'],'All42 masks and positions exactly match')
                changed=[n for n in STAGES if a['stage_sha256'][n]!=b['stage_sha256'][n]]
                rows.append(dict(call=i,phase=a['call']['phase'],physical_frame_indices=[x['frame']['index']for x in sorted(a['physical_rows'],key=lambda v:v['physical_row'])],
                    left=ar,right=br,current_inputs_equal=True,all42_routes_equal=True,
                    layer1_input_layernorm_hash_equal=a['stage_sha256']['layers.1.input_layernorm']==b['stage_sha256']['layers.1.input_layernorm'],
                    first_recorded_prediction_difference=changed[0]if changed else None,
                    before=differences(a['before_metadata'],b['before_metadata']),
                    post=differences(a['post_metadata'],b['post_metadata'])))
            ranks.append(dict(rank=rank,rows=rows))
        comparisons.append(dict(left_arm=left,right_arm=right,ranks=ranks))
    result=dict(schema='minicpm5_saved_cache_metadata_crosscheck_v1',complete=True,diagnosis=DIAGNOSIS,
        sdk_mapping_source=dict(path=str(SDK),sha256=sdk_sha),sdk_mapping='K=2*layer, V=2*layer+1',
        metadata_source_sha256=reader.pins,metadata_bytes_read=reader.bytes,unique_metadata_files=len(reader.pins),
        calls=84,all_own_history_links_checked=chained,comparisons=comparisons,
        cache_payloads_opened=False,cache_difference_magnitude_measured=False,cache_cells_or_slots_localized=False,
        model_reexecuted=False,causal_root_cause_established=False,full_model_qualified=False,new_speedup_claim=False)
    path=Q/'minicpm5-saved-cache-metadata-crosscheck-20261110-v1.json'
    with path.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        original_trajectory_retained=True,metadata_files=len(reader.pins),history_links=chained,
        selected=[dict(rank=r['rank'],rows=[dict(call=x['call'],phase=x['phase'],
            layer1_input_equal=x['layer1_input_layernorm_hash_equal'],first_difference=x['first_recorded_prediction_difference'],
            before_different_layers=x['before']['different_layers'],post_different_layers=x['post']['different_layers'])
            for x in r['rows']])for r in comparisons[0]['ranks']])))
if __name__=='__main__':main()
