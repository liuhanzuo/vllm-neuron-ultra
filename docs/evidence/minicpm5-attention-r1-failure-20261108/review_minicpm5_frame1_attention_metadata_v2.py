"""Exact original-anchor and new attention metadata review; stdlib only.

Metadata is evidence of the claimed relationships. Actual complete bytes and
process lifecycle must be checked separately before interpreting the result.
"""
import hashlib,json,math,re,struct,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
if str(R)not in sys.path:sys.path.insert(0,str(R))
import minicpm5_frame1_attention_observer_v2 as source
import minicpm5_attention_k_consumer_layout_v1 as k_layout
import review_minicpm5_frame1_attention_bytes_v1 as raw
require=source.require
ROLES=('native','cpu-rank0','cpu-rank1')
RETAINED=('anchor','constructor','weights','actual_weight_shards','cpu_full_anchor',
          'environment_transition','activation_dispatch','norm_dispatch')


def same(left,right,message):
    require(json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False),message)


def sha(value):
    require(type(value)is str and re.fullmatch('[0-9a-f]{64}',value)is not None,'Complete SHA256 digest')


def load(reference):
    path=Path(reference['path']);require(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in (path,*path.parents)),'Canonical metadata reference')
    sha(reference['sha256'])
    with path.open('rb')as stream:data=stream.read(2*2**20+1)
    require(len(data)<=2*2**20 and hashlib.sha256(data).hexdigest()==reference['sha256'],'Exact bounded metadata bytes')
    return json.loads(data)


def tensor(row,directory,name,shape,dtype,observed,extension,encoding):
    same(row['name'],name,'Unique original tensor name');same(row['path'],str(directory/(name+extension)),'Exact tensor output path')
    same(row['encoding'],encoding,'Actual saved encoding');require(row['observed_not_recomputed']is observed,'Actual observation/control provenance')
    if not observed:require(row['own_input_recomputed']is True,'Computed own-input output only')
    meta=row['metadata'];same(meta['shape'],shape,'Complete tensor shape');same(meta['dtype'],dtype,'Original tensor dtype');raw.original_layout(meta)
    size=math.prod(shape)*{'torch.bfloat16':2,'torch.bool':1,'torch.int64':8}[dtype]
    require(type(row['bytes'])is int and row['bytes']==size<=16*2**20,'Complete bounded byte count')
    ident=row['identity'];same(ident['realpath'],row['path'],'Recorded canonical output path');same(ident['size'],size,'Actual file length')
    for key in ('device','inode','size','mtime_ns','ctime_ns'):require(type(ident[key])is int and ident[key]>=0,'Actual integer file identity')
    sha(row['sha256']);same(row['sha256'],meta['sha256'],'Captured bytes and tensor metadata tied')
    return size


def projection_proof(proof,native,rank,plan):
    require(proof['passed']is True,'Actual official projection proof')
    rows=proof['weights']
    if native:
        same(list(rows),['q','k','v','o'],'All four original full matrices')
        require(proof['all_original_full_matrices']is True and proof['maximum_interval_bytes']==8*2**20,'Bounded whole native projection intervals')
        for name,row in rows.items():
            key='model.layers.0.self_attn.'+name+'_proj.weight';original=plan['official_layer0_weights']['tensors'][key]
            same(row['official_key'],key,'Original official tensor key');meta=row['metadata'];same(meta['shape'],original['shape'],'Actual full projection matrix')
            same(meta['dtype'],'torch.bfloat16','Original BF16 projection');raw.original_layout(meta);sha(meta['sha256'])
            same(row['bytes'],math.prod(original['shape'])*2,'Whole original projection bytes')
    else:
        same(proof['rank'],rank,'Actual projection shard rank');same(proof['q_head_ids'],list(range(rank*8,(rank+1)*8)),'Original Q head assignment');same(proof['kv_head_ids'],[rank],'Original KV head assignment')
        require(proof['logical_head_mapping_raw_proven']is True and proof['native_model_constructed']is False,'Original sharded model only')
        same(proof['additional_tensor_upper_bytes'],12*2**20,'Original bounded shard comparison');same(proof['additional_header_upper_bytes'],2**20,'Original bounded header')
        same(list(rows),['q_proj','k_proj','v_proj','o_proj'],'All four original TP2 head shards')
        for name,row in rows.items():
            width=128 if name in ('k_proj','v_proj')else 1024
            same({k:row[k]for k in ('axis','start','stop')},dict(axis=1 if name=='o_proj'else 0,start=rank*width,stop=(rank+1)*width),'Original shard geometry');sha(row['sha256'])


def validate(*,root,reports,plan,read=load):
    root=Path(root);require(root.is_absolute()and root.resolve()==root,'Canonical intended output root')
    raw.validate_reports(reports);total=files=0;prior_refs=[];controls={};k_boundaries={}
    for role in ROLES:
        record=reports[role];native=role=='native';rank=-1 if native else int(role[-1]);ref=source.UPSTREAM[0 if native else rank+1]
        prior=read(ref);prior_refs.append(ref)
        require(prior['schema']=='minicpm5_frame1_upstream_capture_v1'and prior['complete']is True and prior['native']is native,'Actual completed upstream capture')
        same(prior['rank'],rank,'Actual prior rank');same(prior['frame'],source.old.FRAME,'Actual original frame')
        for name in RETAINED:same(record[name],prior[name],'Original upstream '+name+' unchanged')
        for name in ('history_consumed','original_full_history_replayed','full_model_qualified','new_speedup_claim'):require(record[name]is False,'Bounded diagnostic claim '+name)
        same(record['oproj_witnesses'],[],'No output projection substitution')
        same([x['name']for x in record['captures']],list(source.old.FIELDS),'All seven upstream captures exactly once')
        same([x['name']for x in prior['captures']],list(source.old.FIELDS),'All seven original reference captures')
        upstream={row['name']:row for row in record['captures']}
        for current,original in zip(record['captures'],prior['captures']):
            same(current['metadata'],original['metadata'],'All original upstream bytes and layouts unchanged')
            total+=tensor(current,root/role,current['name'],[1,source.LENGTH if native else source.BUCKET,2048],'torch.bfloat16',True,'.bf16','packed_bf16_le');files+=1
        same(list(record['norm_weights']),list(source.old.NORM_NAMES),'Both full norm weights')
        for name,row in record['norm_weights'].items():
            old=prior['norm_weights'][name]
            same({k:v for k,v in row.items()if k!='path'},{k:v for k,v in old.items()if k!='path'},'Original complete norm weight metadata')
            same(row['path'],str(root/role/(name+'.weight.bf16')),'Exact norm output path');same(row['bytes'],4096,'Complete norm bytes');total+=4096;files+=1
        obs=record['attention'];same(obs['frame'],source.old.FRAME,'Internal observations are same frame')
        path=plan['native_provenance']['source']if native else plan['sdk_attention_source']
        same(obs['source'],dict(path=path,sha256=plan['source_sha256'][path]),'Pinned active original attention source')
        same(obs['instrumentation'],'original_python_call_return_profile_and_module_hooks','Reviewed observer mechanism')
        require(obs['full_model_qualified']is False and obs['new_speedup_claim']is False,'Internal observation cannot promote model')
        for counts in obs['calls'].values():
            require(all(type(n)is int for n in counts.values()),'Actual integer call/return counts')
        values={row['name']:row for row in obs['captures']}
        for name,row in values.items():
            expected=source.geometry(native)[name]
            total+=tensor(row,root/role/'attention',name,expected['shape'],expected['dtype'],True,'.raw','packed_logical_little_endian');files+=1
        same(values['normalized_input']['metadata'],upstream['attention_input']['metadata'],'Same full original normalized attention input')
        same(values['projection_output']['metadata'],upstream['attention_output']['metadata'],'Same full original attention output')
        if native:
            expected=hashlib.sha256(struct.pack('<'+str(source.LENGTH)+'q',*range(source.LENGTH))).hexdigest()
            same(values['position_ids']['sha256'],expected,'All native positions unchanged')
        else:
            baseline=read(source.old.CPU_ANCHORS[rank]);source.old.validate_baseline(baseline,rank,False)
            same(values['position_ids']['metadata'],baseline['execution_input_metadata'][2],'All original CPU positions/layout including padding')
        projection_proof(record['official_head_weights'],native,rank,plan)
        boundary=obs['k_consumer_boundary'];reference=boundary['reference']
        same(reference['path'],str(root/role/'attention/k-consumer-boundary.json'),'Exact actual K consumer metadata path')
        same(read(reference),boundary['record'],'Complete persisted K consumer record');k_layout.validate(boundary['record'],native)
        same(boundary['record']['producer'],values['k_post_rope']['metadata'],'Actual original RoPE producer preserved')
        own=record['own_input_controls'];same(own['k_consumer_boundary'],boundary['record'],'Original operators replayed with actual consumer layout')
        k_boundaries[role]=boundary
        require(own['original_interpreter_and_thread_settings']is True,'Original runtime and threads retained')
        require(own['cpu_output_reduction_uses_original_gloo']is(not native),'Original actual CPU reduction')
        for name in ('independently_recomputed_trigonometry','whole_model_endpoint_claimed','new_speedup_claim'):require(own[name]is False,'No unsupported control claim '+name)
        same(own['precision_interventions'],[],'No intervention during own-input controls');controls[role]={}
        for name,control in own['controls'].items():
            row=control['saved'];expected=source.geometry(native)[name]
            total+=tensor(row,root/role/'attention/controls',name,expected['shape'],expected['dtype'],False,'.raw','packed_logical_little_endian');files+=1
            original=values[name]['metadata'];actual=row['metadata']
            outcome=dict(raw_exact=original['sha256']==actual['sha256'],layout_exact=all(original[k]==actual[k]for k in ('shape','stride','offset','dtype')))
            require(type(control['raw_exact'])is bool and type(control['layout_exact'])is bool,'Actual boolean control conclusions')
            same({k:control[k]for k in outcome},outcome,'No false own-input equality from metadata');controls[role][name]=outcome
        passed=source.controls_passed(controls[role]);require(type(own['passed'])is bool and type(record['own_input_controls_passed'])is bool,'Actual own-input verdict booleans')
        same(own['passed'],passed,'Complete per-role own-input outcome');same(record['own_input_controls_passed'],passed,'Top-level control outcome')
    same(files,96,'Exactly21upstream+6norm+48internal+21control payloads')
    same(total,source.science()['total_raw_bytes'],'Exact whole raw payload budget')
    return dict(complete=True,prior_upstream_references=prior_refs,all21_prior_upstream_metadata_exact=True,
                all_six_prediction_anchors_per_role_exact=True,original_constructor_norm_mlp_environment_retained=True,
                files=files,raw_bytes=total,controls=controls,k_consumer_boundaries=k_boundaries,all21_own_input_controls_claim_exact=all(source.controls_passed(v)for v in controls.values()),
                raw_payloads_verified=False,official_projection_bytes_independently_read=False,
                model_math_reexecuted=False,process_lifecycle_verified=False,full_model_qualified=False,new_speedup_claim=False)
