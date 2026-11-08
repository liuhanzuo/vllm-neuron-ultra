"""Independent frame1 metadata consistency audit; no raw/native/model work.

This validates recorded claims against pinned original anchors. It never recomputes
raw tensor equality, certifies an exit, drains a process, or qualifies a model.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
BASE=dict(path=str(R/'research/minicpm5-layer0-phase-cpu-input-plan-20261107-v1.json'),sha256='ce599a2beae22df0a05c13591854cfab8ba57519df11ee93a0d25ad4392306c8')
FRAME=dict(index=1,fixture=7,slot=1,step=0)
FIELDS=('input','gate','up','down_input','output')
TAPS=('embed_tokens','layers.0.input_layernorm','layers.0.self_attn','layers.0.post_attention_layernorm','layers.0.mlp','layers.0')
FULL=dict(input='postnorm',gate='gate',up='up',down_input='product')
ROLES=('native','cpu-rank0','cpu-rank1')
CPU=tuple(dict(path=str(R/f'minicpm5-original32-layer0-phase-cpu-r1-20261107-v1/cpu-rank{i}/strict_anchor_s0/call-01/result.json'),sha256=h) for i,h in enumerate(('04a6334a2950b16f9fb4c8dc3870603584b5c6e0da010bc8fab7583c98c7ffc0','d1cac9b1a0f1c6095a4dd6a885a782251da9cab7bed3d0f254e14201aa357d02')))
FULL_CPU=tuple(dict(path=str(R/f'minicpm5-original32-full-down-cpu-r1-20261107/cpu-rank{i}/strict_anchor_s0/call-01/result.json'),sha256=h) for i,h in enumerate(('9141f970060fb3ba0857b482c4f79505e72da48734cfae763ceb473e8684c74a','6e315aa0a88a7d52fb87f362a882e50b06d8089d92e5d6d6113b621f54b18523')))

def require(value,message):
    if not value:raise ValueError(message)

def same(left,right,message):require(left==right,message)

def hash_value(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(value):require(type(value)is str and re.fullmatch('[0-9a-f]{64}',value),'SHA256 format');return value

class Reader:
    def __init__(self):self.references={}
    def load(self,reference):
        require(set(reference)=={'path','sha256'},'Exact metadata reference');sha(reference['sha256'])
        path=Path(reference['path'])
        # Payload paths are rejected before any stat/open operation.
        require(path.is_absolute() and path.suffix=='.json' and path.is_relative_to(R),'Only experiment JSON metadata')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_size<=5*2**20,'Bounded ordinary metadata')
            with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(5*2**20+1)
            after=os.fstat(fd)
            same(tuple(getattr(before,k)for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),tuple(getattr(after,k)for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Metadata changed during read')
        finally:os.close(fd)
        require(len(raw)<=5*2**20 and hashlib.sha256(raw).hexdigest()==reference['sha256'],'Metadata digest')
        self.references[str(path)]=dict(reference,bytes=len(raw))
        return json.loads(raw)

def layout(meta,shape):
    same(meta['shape'],shape,'Full observed shape');same(meta['dtype'],'torch.bfloat16','Original BF16')
    require(type(meta['offset'])is int and meta['offset']==0,'Original zero offset')
    require(len(meta['stride'])==3 and all(type(x)is int and x>0 for x in meta['stride']),'Positive original strides')
    span=1
    for step,size in sorted((s,n)for s,n in zip(meta['stride'],shape)if n>1):
        same(step,span,'Dense original layout');span*=size
    sha(meta['sha256']);return 2*math.prod(shape)

def metric(row,count):
    same(set(row),{'raw_exact','different_elements','compared_elements','nonfinite_elements','max_abs','mean_abs','cosine'},'Complete original metric fields')
    require(type(row['raw_exact'])is bool and type(row['different_elements'])is int and type(row['compared_elements'])is int,'Metric types')
    same(row['compared_elements'],count,'Full comparison element count')
    require(type(row['nonfinite_elements'])is int and row['nonfinite_elements']==0,'Finite captures only')
    n=row['different_elements'];require(0<=n<=count and row['raw_exact']==(n==0),'Bitwise exactness agrees with difference count')
    for key in ('max_abs','mean_abs'):require(type(row[key])in (int,float) and math.isfinite(row[key]),'Finite distance')
    require(0<=row['mean_abs']<=row['max_abs'],'Bounded nonnegative distance')
    if n==0:require(row['max_abs']==row['mean_abs']==0,'Exact tensors have zero distance')
    # Different BF16 bit patterns may be +0/-0, so distance zero is not a raw-equality proof.
    cosine=row['cosine'];require(cosine is None or (type(cosine)in (int,float) and math.isfinite(cosine)and -1.000000000001<=cosine<=1.000000000001),'Valid cosine or undefined zero-norm result')

def scopes(rows,width,full=True):
    expected={'real_tokens':1231,'prediction':1}
    if full:expected={'full':2048,**expected}
    same(set(rows),set(expected),'Complete comparison scopes')
    for name,length in expected.items():metric(rows[name],length*width)
    chain=list(expected)
    for larger,smaller in zip(chain,chain[1:]):
        require(rows[smaller]['different_elements']<=rows[larger]['different_elements'],'Subset difference count')
        require(rows[smaller]['max_abs']<=rows[larger]['max_abs'],'Subset maximum error')

def baseline(row,rank,full):
    require(row['complete']is True and row['rank']==rank,'Actual complete baseline rank')
    same(row['arm'],'strict_anchor_s0','Strict original baseline')
    same(row['call'],dict(index=1,group=0,phase='prefill',frames=[FRAME],logical_positions=[1230]),'Original long prefill')
    same(row['bucket'],2048,'Original CPU bucket');same(row['changed_mask_cells'],[],'No changed mask');same(row['candidate_targets'],[],'No precision candidate')
    require(row['candidate_full_down_enabled'if full else'candidate_layer0_phase_enabled']is False,'No candidate baseline')
    for key in ('all7_inputs_exact','all213_taps_exact','all84_before_input_post_exact','full_logits_exact'):require(row['strict_anchor_exact'][key]is True,'Original full baseline equality')

def audit(plan,spec,replay,captures,load):
    science=spec['science'];root=Path(spec['output_root'])
    expected=dict(frame=FRAME,capture_fields=list(FIELDS),cpu_anchors=list(CPU),cpu_full_anchors=list(FULL_CPU),arm='strict_anchor_s0',precision_interventions=[],native_length=1231,cpu_bucket=2048,actual_capture_bytes=164515840,capture_file_cap_bytes=16*2**20,output_cap_bytes=192*2**20)
    same(science,expected,'Exact original frame1 science')
    same(plan['graph_calls'][1]['call'],dict(index=1,group=0,phase='prefill',frames=[FRAME],logical_positions=[1230]),'Original graph call')
    same(plan['graph_calls'][1]['bucket'],2048,'Original physical bucket')
    require(plan['fixtures'][7]['id']==7 and len(plan['fixtures'][7]['input_ids'])==1231,'Original prompt length')
    same(replay['schema'],'minicpm5_frame1_mlp_replay_v2','Actual frame1 schema');same(replay['frame'],FRAME,'Actual frame1')
    require(replay['complete']is True and replay['controls']['all15_fullshape_exact']is True,'All15 full controls required')
    same(set(captures),set(ROLES),'All actual capture roles');same(set(replay['captures']),set(ROLES),'All capture references')
    require(set(replay['controls'])=={'native','cpu','all15_fullshape_exact'} and len(replay['controls']['cpu'])==2,'Three control sets')
    native_old=load(plan['hf_frame_records'][1]);same(native_old['frame'],FRAME,'Original native frame')
    header=load(plan['safetensors_header']);proof=replay['weight_proof']
    same(proof['file'],next(x for x in plan['model_files']if x['path'].endswith('.safetensors')),'Original checkpoint identity')
    same(proof['header'],plan['safetensors_header'],'Original weight header')
    same(set(proof['matrices']),{'gate','up','down'},'All official MLP matrices');same(proof['bytes_read'],72*2**20,'Exact original matrix byte budget')
    require(proof['full_file_rehashed']is False,'No full weight hash claim')
    for name,row in proof['matrices'].items():
        same(row['name'],'model.layers.0.mlp.'+name+'_proj.weight','Official matrix name');same(row['tensor'],header[row['name']],'Official tensor header')
        same(row['bytes_read'],24*2**20,'Full original matrix interval');same(row['tensor']['data_offsets'][1]-row['tensor']['data_offsets'][0],row['bytes_read'],'Original interval length');sha(row['sha256'])
    count=0;total=0;metadata={}
    for role in ROLES:
        value=captures[role];native=role=='native';rank=-1 if native else int(role[-1]);length=1231 if native else 2048
        same(value['schema'],'minicpm5_frame1_mlp_capture_v2','Strict original capture schema');same(value['frame'],FRAME,'Capture frame')
        require(value['complete']is True and value['native']is native and value['rank']==rank,'Exact completed role')
        require(value['actual_prefix_calls']==1 and value['history_consumed']is False and value['original_full_history_replayed']is False and value['downstream_layers_executed']==0,'Only first layer prefix')
        same(value['precision_interventions'],[],'No precision intervention');same(value['oproj_witnesses'],[],'No output projection intervention')
        for key in ('full_model_qualified','new_speedup_claim'):require(value[key]is False,'No capture-level qualification')
        old=native_old if native else load(CPU[rank])
        if not native:baseline(old,rank,False)
        anchor=value['anchor'];same(list(anchor['hashes']),list(TAPS),'Six prediction anchors')
        same(anchor['hashes'],{k:old['tap_raw_sha256'if native else'stage_sha256'][k]for k in TAPS},'Original prediction raw hashes')
        require(anchor['all_six_exact']is True and anchor['whole_model_endpoint_claimed']is False,'Local anchor scope')
        same(set(anchor['source_metadata']),set(TAPS),'Six original layout records')
        for tap in TAPS:
            actual=anchor['source_metadata'][tap]
            if native:
                prior=old['metadata'][tap]
                for key,original in (('shape','source_shape'),('stride','source_stride'),('dtype','dtype')):same(actual[key],prior[original],'Native original source geometry')
                same(actual['prediction_positions'],[prior['local_prediction_position']],'Native prediction position')
            else:same(actual,old['stage_metadata'][tap],'CPU original source geometry')
        same([x['name']for x in value['captures']],list(FIELDS),'All five ordered full captures')
        controls=replay['controls']['native']if native else replay['controls']['cpu'][rank];same(set(controls),set(FIELDS),'All five controls')
        metadata[role]={}
        for row in value['captures']:
            name=row['name'];width=2048 if name in ('input','output')else(6144 if native else 3072);n=layout(row['metadata'],[1,length,width])
            same(row['path'],str(root/role/(name+'.bf16')),'Exact capture path');same(row['identity']['realpath'],row['path'],'Identity path')
            require(row['bytes']==row['identity']['size']==n and n<=16*2**20,'Complete bounded tensor bytes')
            same(row['encoding'],'packed_bf16_le','Original packed encoding');same(row['sha256'],row['metadata']['sha256'],'Capture hash links')
            require(row['observed_not_recomputed']is True,'Actual observation required')
            same(controls[name],dict(raw_exact=True,source_layout_exact=True,sha256=row['sha256']),'Exact whole-tensor replay control')
            metadata[role][name]=row['metadata'];count+=1;total+=n
        same(value['weights'],proof,'Shared official weights')
        same(set(value['actual_weight_shards']),{'gate','up','down'},'Actual matrix inventory')
        for name,meta in value['actual_weight_shards'].items():
            shape=([2048,6144]if name=='down'else[6144,2048])if native else([2048,3072]if name=='down'else[3072,2048])
            same(meta['shape'],shape,'Actual original shard shape');same(meta['dtype'],'torch.bfloat16','Actual original shard dtype');sha(meta['sha256'])
            if native:same(meta['sha256'],proof['matrices'][name]['sha256'],'Native official interval hash')
            else:
                for key in ('shard_metadata','model_metadata'):same(value['constructor'][key]['layers.0.mlp.'+name+'_proj.weight'],dict(shape=shape,dtype='torch.bfloat16'),'Actual CPU model shard')
        if native:
            same(value['constructor'],plan['native_provenance'],'Original native constructor');require(value['cpu_full_anchor']is None,'No native CPU anchor')
        else:
            constructor=value['constructor'];require(constructor['rank']==rank and constructor['learned_tensors']==381 and constructor['all_shard_values_raw_exact']is True and constructor['compiled']is False and constructor['resharded']is False and constructor['sdk_math_modified']is False,'Original CPU weights and math')
            full=load(FULL_CPU[rank]);baseline(full,rank,True);same(full['stage_sha256'],old['stage_sha256'],'Original full/prediction anchors');same(full['stage_metadata'],old['stage_metadata'],'Original full/prediction layout')
            same(value['cpu_full_anchor'],dict(reference=FULL_CPU[rank],all_four_fullshape_exact=True,fields=FULL),'Four full original CPU anchors')
            same(set(full['layer0_full_observations']),set(FULL.values()),'Full original CPU boundary coverage')
            for field,name in FULL.items():same(metadata[role][field],full['layer0_full_observations'][name],'Whole original CPU boundary exact')
            same(value['environment_transition']['before']['RANK'],str(rank),'Actual rank');same(value['environment_transition']['before']['WORLD_SIZE'],'2','Actual world size')
        activation=value['activation_dispatch'];require(activation['runtime_code_exact']is True and activation['dispatch']=='original_functional_silu_fallback','Original activation dispatch')
        require(activation['source']in spec['new_sources'] or activation['source']in plan['source_sha256'],'Activation source pinned')
    same(total,164515840,'All15 full capture byte total');same(count,15,'All15 captures')
    for field in ('input','output'):same(metadata['cpu-rank0'][field],metadata['cpu-rank1'][field],'Actual replicated CPU boundary')
    same(replay['fixed_input'],metadata['cpu-rank0']['input'],'Contrast uses actual CPU input')
    same(set(replay['front_half']),{'gate','up','down_input'},'Complete front half comparisons')
    for row in replay['front_half'].values():scopes(row,6144)
    same(set(replay['contrasts']),{'S_vs_F','D_vs_F','D_vs_S'},'All three paired contrasts')
    for row in replay['contrasts'].values():scopes(row,2048)
    same(set(replay['observed_native_vs_cpu_boundaries']),set(FIELDS),'All five observed boundary comparisons')
    for field,row in replay['observed_native_vs_cpu_boundaries'].items():scopes(row,2048 if field in ('input','output')else 6144,False)
    same(replay['observed_native_vs_cpu_input'],replay['observed_native_vs_cpu_boundaries']['input']['real_tokens'],'Original input comparison reference')
    front=replay['front_half']['down_input']['full']['raw_exact'];same_output=replay['contrasts']['S_vs_F']['full']['raw_exact'];down=replay['contrasts']['D_vs_F']['full']['raw_exact']
    if front:
        require(down,'Identical complete down inputs contradict complete full-down output')
        same(replay['contrasts']['S_vs_F'],replay['contrasts']['D_vs_S'],'Same full output gives symmetric distances')
        classification='no_output_difference_for_this_fixed_input'if same_output else'down_path_sufficient_for_this_fixed_input_not_final_KL_causality'
    else:classification='front_half_differs_down_is_not_established_as_unique_cause'
    same(replay['classification'],classification,'Causal classification follows complete controls')
    same(replay['original_remaining_gates'],plan['remaining_gates'],'All original full-model gates retained')
    same(replay['native_own1231_control_threads'],2,'Native original threads');same(replay['fixed_input2048_contrast_threads'],1,'CPU original threads')
    for key in ('actual_gloo_reexecuted_in_replay','native_interpreter_reused_in_replay','whole_native_input_equality_previously_known','unique_final_KL_cause_proven','full_model_qualified','new_speedup_claim'):require(replay[key]is False,'Explicit result limitation '+key)
    return dict(metadata_consistent=True,capture_count=count,capture_bytes=total,classification=classification,raw_payload_recomputed=False,actual_exit_verified=False,physical_drain_verified=False,full_model_qualified=False,new_speedup_claim=False)

def main(args):
    read=Reader();spec_ref=dict(path=args.spec,sha256=args.spec_sha256);spec=read.load(spec_ref)
    same(spec['schema'],'minicpm5_frame1_mlp_spec_v1','Actual frame1 execution spec');same(spec['base_plan'],BASE,'Frozen complete scientific base')
    root=Path(spec['output_root']);require(root.parent==R and re.fullmatch('minicpm5-frame1-mlp-r[0-9]+-20261108-v[0-9]+',root.name),'Actual finite output namespace')
    result_ref=dict(path=args.result,sha256=args.result_sha256);same(result_ref['path'],str(root/'result.json'),'Original outer result path');result=read.load(result_ref)
    require(result['complete']is True and result['self_exit_claimed']is False and result['full_model_qualified']is False and result['new_speedup_claim']is False,'Complete diagnostic only, no exit or model qualification')
    same(result['admission']['path'],str(root/'admission.json'),'Actual admission path');admission=read.load(result['admission']);same(admission['spec'],spec_ref,'Actual bound specification')
    review=read.load(admission['review']);same(review['spec'],spec_ref,'Exact reviewed specification');require(review['reviewer']=='/root' and review['passed']is True and review['independent_review']is True and review['source_only']is True and review['blockers']==[],'Actual independent source review')
    same(result['owner'],admission['outer_owner'],'Actual outer owner chain')
    same(admission['current_catalog'],spec['current_catalog'],'Actual completed predecessor catalog')
    same(admission['required_sio14_catalog'],spec['required_sio14_catalog'],'Actual Smol anchor')
    require(admission['current_history_drained']is True,'Runtime admission recorded completed history drain')
    same(set(spec['added_json']),{ref['path']for ref in (*CPU,*FULL_CPU)},'Exactly four original CPU anchor additions')
    for reference in (*CPU,*FULL_CPU):same(spec['added_json'][reference['path']]['sha256'],reference['sha256'],'Original CPU metadata additions')
    plan=read.load(BASE);expanded=copy.deepcopy(plan);expanded['output_root']=str(root)
    for name,digest in spec['new_sources'].items():require(name not in expanded['source_sha256'],'Original source not overwritten');sha(digest);expanded['source_sha256'][name]=digest
    for name,row in spec['added_json'].items():read.load(dict(path=name,sha256=row['sha256']));require(name not in expanded['inputs'] or expanded['inputs'][name]==row,'Original input not overwritten');expanded['inputs'][name]=row
    expanded['frame1_mlp']=spec['science'];same(hash_value(expanded),admission['plan_digest'],'Exact actual expanded plan')
    same(result['report']['path'],str(root/'replay.json'),'Actual replay path');replay=read.load(result['report']);captures={}
    for role,reference in replay['captures'].items():require(role in ROLES,'Finite capture role');same(reference['path'],str(root/role/'capture.json'),'Actual role capture path');captures[role]=read.load(reference)
    checked=audit(plan,spec,replay,captures,read.load)
    output=dict(schema='minicpm5_frame1_independent_metadata_review_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=spec_ref,result=result_ref,report=result['report'],checks=checked,metadata_references=read.references,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Metadata consistency against original anchors only. Actual raw replay, real external exit, full lifecycle and fresh physical drain remain separate required evidence.')
    dest=Path(args.output);require(dest.parent==R/'research' and dest.suffix=='.json','New research JSON only')
    with dest.open('x')as stream:stream.write(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(dest),metadata_consistent=True,raw_payload_recomputed=False,model_qualified=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','spec-sha256','result','result-sha256','output'):parser.add_argument('--'+name,required=True)
    main(parser.parse_args())
