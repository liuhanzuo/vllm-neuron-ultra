"""Five-boundary layer0 comparison. NumPy and raw IO are runtime-only.

The CLI requires finalized successful original88 evidence and root-observed
actual finalizer exit, then acquires serial ownership. No model is imported.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import minicpm5_layer0_execution_protocol_v7 as p
import minicpm5_original88_python_trace_result_parser_v3 as parser
import minicpm5_original88_python_trace_validation_v2 as science
from minicpm5_layer0_runtime_auth_v1 import identity
from research import audit_minicpm5_original88_python_trace_root_20261107_v3 as post
from research import finalize_minicpm5_original88_root_20261107_v1 as finalizer
ROOT=p.ROOT
STAGES=('positions','normalized_input','q_projection','k_projection','v_projection')
PAIRS=(('HF_native','C_Hd_control'),('HF_native','C_Hc_history_generation'),('C_Hd_control','C_Hc_history_generation'))
PENDING=('qk_rope_inputs','rope_parameters','qk_post_rope','active_values','cache_prior','cache_after','mask_semantics','score_unscaled','score_scaled','score_masked','probabilities','context_before_o','attention_output')
DTYPE_BYTES={'torch.bfloat16':2,'torch.float32':4,'torch.int64':8,'torch.int32':4}
same=finalizer.same

def checked_finalizer_exit(audit_ref,catalog_ref,exit_ref,audit):
    ended=p.read_ref(exit_ref);live_ref=post.reference(ended['root_live']);live=p.read_ref(live_ref)
    p.require(live['schema']=='minicpm5_original88_finalizer_root_live_v1'and live['reviewer']=='/root'and live['actual_identity_verified']is True,'Actual finalizer root live receipt')
    p.require(ended['schema']=='minicpm5_original88_finalizer_root_exit_v1'and ended['reviewer']=='/root'and
              ended['actual_exit_observed']is True and ended['finalizer_self_exit_claimed']is False and
              type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,'Actual finalizer tool exit0')
    owner=post.process(live['owner']);same(owner,audit['finalizer_owner'],'Root finalizer same audited owner');same(ended['owner'],owner,'Actual exit same live owner')
    handle=live['actual_execution_handle'];p.require(set(handle)=={'kind','session_id'}and handle['kind']=='tool_session'and type(handle['session_id'])is int and handle['session_id']>0,'Actual finalizer tool session')
    same(ended['actual_execution_handle'],handle,'Actual finalizer same completed tool session')
    same(ended['finalization_outputs'],dict(audit=audit_ref,catalog=catalog_ref),'Actual finalizer outputs bound')
    same(live['input_refs'],audit['references'],'Exact9 finalizer input refs')
    prefix=Path(audit_ref['path'].removesuffix('-audit.json'))
    p.require(catalog_ref['path']==str(prefix)+'-catalog.json'and live['output_prefix']==str(prefix),'Exact finalizer output prefix')
    command=['/usr/bin/python3','-B',str(ROOT/'research/finalize_minicpm5_original88_root_20261107_v1.py')]
    for key in('execution_spec','source_review','external_invocations','post_audit','post_append','proposed_catalog','post_root_live','post_root_exit','helper_source_review'):
        ref=audit['references'][key];command+=['--'+key.replace('_','-'),ref['path'],'--'+key.replace('_','-')+'-sha256',ref['sha256']]
    command+=['--output-prefix',str(prefix)];same(owner['argv'],command,'Exact actual finalizer argv')
    return owner,live_ref

def load_finalized(audit_ref,catalog_ref,exit_ref):
    for ref in(audit_ref,catalog_ref,exit_ref):post.reference(ref)
    audit=p.read_ref(audit_ref);catalog=p.read_ref(catalog_ref)
    p.require(audit['schema']=='minicpm5_original88_root_finalization_v1'and audit['reviewer']=='/root'and
              audit['passed']is audit['runtime_successful']is audit['diagnostic_complete']is audit['all_groups_drained']is
              audit['all_planned_launches_accounted_for']is audit['canonical_unchanged']is True,'Successful finalized original88 only')
    p.require(audit['full_model_qualified']is audit['new_speedup_claim']is audit['causal_operator_proven']is False,'Diagnostic-only finalized result')
    owner,live_ref=checked_finalizer_exit(audit_ref,catalog_ref,exit_ref,audit)
    p.require(catalog['root_finalized']is catalog['original88_post_auditor_exit_resolved']is True and catalog['complete_count']==len(catalog['rows'])==444,'Actual final444 catalog')
    same(catalog['root_original88_finalization'],audit_ref,'Catalog same finalization audit')
    proposal=p.read_ref(audit['references']['proposed_catalog'])
    for key,value in proposal.items():
        if key not in('root_finalized','source_sha256','root_finalization_observed_at'):same(catalog[key],value,'Final catalog preserves proposal '+key)
    p.require(all(catalog['source_sha256'].get(path)==wanted for path,wanted in proposal['source_sha256'].items()),'All proposal pins preserved')
    same(catalog['source_sha256'][audit_ref['path']],audit_ref['sha256'],'Finalization pin in catalog')
    spec,plan,prior=p.bundle(audit['references']['execution_spec'],audit['references']['source_review'])
    post_audit=p.read_ref(audit['references']['post_audit']);post.completed_science(post_audit['completion'])
    same(audit['completion'],post_audit['completion'],'Same successful original88 full raw completion')
    control=Path(spec['control_root']);admission=p.read(control/'admission.json');admission_ref=dict(path=str(control/'admission.json'),sha256=p.sha(control/'admission.json'))
    reports,owners,contracts=parser.ordered_reports(spec,plan,admission_ref,admission)
    audited={str(Path(row['directory'])/'capture.json'):row['receipt_sha256']for row in audit['completion']['contexts']}
    same({row['path']:row['sha256']for row in reports},audited,'Every88 receipt hash bound to finalized raw audit')
    checked=science.validate_references(reports,plan,contracts,p.read_ref)
    same(checked,audit['completion']['science'],'Original88 science summary retained')
    return dict(spec=spec,plan=plan,catalog=catalog,audit=audit,reports=reports,owners=owners,finalizer_owner=owner,
                binding=dict(finalization=audit_ref,catalog=catalog_ref,finalizer_exit=exit_ref,finalizer_live=live_ref))

def prepare_index(plan,reports,read_receipt):
    """Metadata only, reuse audited typed snapshot/frame context contracts."""
    contexts={science.owner_key(row):row for row in plan['writer_contexts']};index={}
    for reference in reports:
        record=read_receipt(reference);key=science.owner_key(record['permit']);p.require(key in contexts,'Original writer context')
        context=contexts[key];observer=record['observer'];parser.check_observer_receipt(observer,context,plan,record['payloads'])
        call=plan['calls'][context['call']]
        if context['arm']=='HF_native':frames=[next(f for f in plan['frames']if f['index']==context['frame_indices'][0])]
        else:frames,_=science.expected_cpu_geometry(plan,call)
        science.exact_frames(observer['frames'],frames)
        snapshots={row['label']:row for row in observer['snapshots']}
        for row,frame in enumerate(frames):
            frame_key=(context['arm'],context['rank'],frame['index']);p.require(frame_key not in index,'No duplicated frame arm/rank')
            index[frame_key]=dict(frame=copy.deepcopy(frame),physical_row=row,call=call['index'],phase=call['phase'],
                context=context,receipt=reference,stages={label:snapshots.get('position_ids'if label=='positions'else label)for label in STAGES})
    p.require(len(index)==160,'All native32 plusCPU2arms2ranks32 frames')
    return index

def selection(plan,entry,label,rank):
    """Plan semantic slicing only; never reads a tensor or assumes raw stride."""
    snapshot=entry['stages'][label]
    if snapshot is None:return dict(blocked='missing applicable label '+label)
    payload=snapshot['value']
    if type(payload)is not dict or set(payload)!={'metadata','reference'}:return dict(blocked='label is not one complete tensor')
    meta=payload['metadata'];shape=meta['selected_shape'];source=meta['source_shape'];dtype=meta['source_dtype'];native=entry['context']['arm']=='HF_native'
    if meta['selection']is not None or not parser.typed_equal(shape,source):return dict(blocked='full positions not captured')
    ndim=2 if label=='positions'else 3
    # positions is the public analysis name for the captured position_ids label.
    p.require(type(shape)is list and len(shape)==ndim and all(type(x)is int and x>0 for x in shape),'Positive exact full tensor shape')
    p.require(type(meta['source_stride'])is list and len(meta['source_stride'])==ndim and all(type(x)is int and x>=0 for x in meta['source_stride'])and type(meta['source_offset'])is int and meta['source_offset']>=0,'Original source layout metadata')
    frame=entry['frame'];fixture=next(x for x in plan['fixtures']if x['id']==frame['fixture']);length=len(fixture['input_ids'])if entry['phase']=='prefill'else 1
    batch=1 if native or entry['phase']=='prefill'else 4
    p.require(shape[0]==batch and 0<=entry['physical_row']<batch and shape[1]>=length,'Actual physical row and real position domain')
    expected_sequence=length if native else 1 if entry['phase']=='decode'else plan['graph_calls'][entry['call']]['bucket']
    p.require(shape[1]==expected_sequence,'Exact native real sequence / CPU decode1 / original prefill bucket')
    if dtype not in DTYPE_BYTES:return dict(blocked='unsupported captured dtype '+str(dtype))
    if label=='positions':p.require(dtype in('torch.int64','torch.int32'),'Position integer dtype');columns=None
    else:
        p.require(dtype in('torch.bfloat16','torch.float32'),'Floating captured dtype')
        width={'normalized_input':2048,'q_projection':1024,'k_projection':128,'v_projection':128}[label]
        total=width*(2 if native and label!='normalized_input'else 1);p.require(shape[2]==total,'Official native/TP2 projection width')
        start=rank*width if native and label!='normalized_input'else 0;columns=[start,start+width]
    p.require(math.prod(shape)*DTYPE_BYTES[dtype]==meta['selected_bytes']==payload['reference']['bytes'],'Exact contiguous clone byte size')
    return dict(payload=payload,row=entry['physical_row'],positions=[0,length],columns=columns,shape=shape,dtype=dtype,
        layout=dict(shape=source,stride=meta['source_stride'],offset=meta['source_offset'],selection=meta['selection']),
        global_q_heads=list(range(rank*8,(rank+1)*8)),global_kv_heads=[rank])

def schedule(plan,index):
    """All192 frame/rank/pairs × five boundaries, including explicit blockers."""
    tasks=[]
    for frame in plan['frames']:
        for rank in(0,1):
            for left,right in PAIRS:
                a=index[left,-1 if left=='HF_native'else rank,frame['index']];b=index[right,rank,frame['index']]
                same(a['frame'],frame,'Full left frame identity');same(b['frame'],frame,'Full right frame identity')
                for label in STAGES:
                    tasks.append(dict(frame=copy.deepcopy(frame),call=a['call'],phase=a['phase'],rank=rank,left_arm=left,right_arm=right,stage=label,
                        left=selection(plan,a,label,rank),right=selection(plan,b,label,rank),left_receipt=a['receipt'],right_receipt=b['receipt']))
    p.require(len(tasks)==960,'All192 comparisons at five stages')
    return tasks

def read_raw(payload,guard):
    """One existing <=16MiB file, verify raw identity/bytes/hash; no writes."""
    reference=payload['reference'];meta=payload['metadata'];path=Path(reference['path']);guard();before=identity(path);info=path.lstat()
    same(before,reference['identity'],'Captured raw identity')
    p.require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_uid==1000 and stat.S_IMODE(info.st_mode)==0o600,'Original exclusive owned raw')
    p.require(reference['encoding']=='raw_little_endian'and path.suffix=='.raw'and before['size']==reference['bytes']==meta['selected_bytes']<=16*p.MIB,'Bounded actual raw bytes')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);data=bytearray();digest=hashlib.sha256()
    try:
        opened=os.fstat(fd);same([opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns],[before[k]for k in('device','inode','size','mtime_ns','ctime_ns')],'Opened exact raw descriptor')
        while True:
            guard();chunk=os.read(fd,min(p.MIB,before['size']+1-len(data)))
            if not chunk:break
            data.extend(chunk);p.require(len(data)<=before['size'],'Raw grew');digest.update(chunk)
        after=os.fstat(fd);same([after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns],[opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns],'Raw descriptor unchanged')
    finally:os.close(fd)
    same(identity(path),before,'Raw path unchanged');p.require(len(data)==reference['bytes']and digest.hexdigest()==reference['sha256'],'Exact full raw byte hash')
    return data

def view(np,data,chosen):
    dtype={'torch.bfloat16':'<u2','torch.float32':'<f4','torch.int64':'<i8','torch.int32':'<i4'}[chosen['dtype']]
    raw=np.frombuffer(data,dtype=dtype).reshape(chosen['shape']);slices=(chosen['row'],slice(*chosen['positions']))
    if chosen['columns']is not None:slices+= (slice(*chosen['columns']),)
    selected=raw[slices]
    numeric=((selected.astype(np.uint32)<<16).view(np.float32))if chosen['dtype']=='torch.bfloat16'else selected
    exact=selected.view(np.uint32)if chosen['dtype']=='torch.float32'else selected
    return exact,numeric

def compare(np,task,reader,guard):
    left,right=task['left'],task['right'];row={key:copy.deepcopy(task[key])for key in('frame','call','phase','rank','left_arm','right_arm','stage','left_receipt','right_receipt')}
    row.update(left_layout=left.get('layout'),right_layout=right.get('layout'),left_dtype=left.get('dtype'),right_dtype=right.get('dtype'),
        physical_rows=[left.get('row'),right.get('row')],positions=[left.get('positions'),right.get('positions')],feature_slices=[left.get('columns'),right.get('columns')])
    if 'blocked'in left or 'blocked'in right:row.update(status='blocked_missing_or_unmapped',reason=left.get('blocked',right.get('blocked')));return row
    if left['dtype']!=right['dtype']:row.update(status='dtype_mismatch',reason='Actual captured dtypes differ; no implicit cast equality');return row
    a_bytes=reader(left['payload'],guard);b_bytes=reader(right['payload'],guard)
    ar,a=view(np,a_bytes,left);br,b=view(np,b_bytes,right);p.require(a.shape==b.shape,'Same complete logical domain')
    bit_equal=bool(np.array_equal(ar,br));a=a.astype(np.float64);b=b.astype(np.float64)
    finite=np.isfinite(a)&np.isfinite(b);nonfinite=int(a.size-np.count_nonzero(finite));delta=np.abs(a[finite]-b[finite])
    dot=float(np.sum(a[finite]*b[finite],dtype=np.float64));norm=float(np.sqrt(np.sum(a[finite]*a[finite],dtype=np.float64)*np.sum(b[finite]*b[finite],dtype=np.float64)))
    row.update(status='comparable_exact'if bit_equal else'comparable_different',reason=None,compared_elements=int(a.size),raw_exact=bit_equal,
        different_elements=int(np.count_nonzero(ar!=br)),max_abs=float(delta.max())if delta.size else None,mean_abs=float(delta.mean())if delta.size else None,
        cosine=dot/norm if norm>0 and nonfinite==0 else None,nonfinite_elements=nonfinite)
    return row

def summary32(plan,rows):
    result=[]
    for frame in plan['frames']:
        selected=[row for row in rows if row['frame']['index']==frame['index']];p.require(len(selected)==30,'All3pairs2ranks5stages per frame')
        result.append(dict(frame=frame,rows=30,exact=sum(x['status']=='comparable_exact'for x in selected),different=sum(x['status']=='comparable_different'for x in selected),
            blocked=sum(x['status']in('blocked_missing_or_unmapped','dtype_mismatch')for x in selected),full_attention_analyzed=False))
    return result

def run(audit_ref,catalog_ref,exit_ref,output_prefix):
    bound=load_finalized(audit_ref,catalog_ref,exit_ref);spec=bound['spec'];plan=bound['plan'];prefix=Path(output_prefix)
    p.require(prefix.parent==ROOT/'research'and re.fullmatch('minicpm5-original88-first-five-[a-z0-9-]+',prefix.name),'Fresh report prefix')
    paths={name:Path(str(prefix)+suffix)for name,suffix in(('report','.json'),('csv','.csv'),('summary','-frames.json'))}
    p.require(all(not path.exists()for path in paths.values()),'Never overwrite report or original artifacts')
    fd,owner,lock=post.external.acquire_lock()
    try:
        def guard():
            p.clear_window();p.lock_proof(owner,lock);p.control_usage(spec);p.active_heavy()
            same({name:p.sha(ROOT/name)for name in spec['canonical_sha256']},spec['canonical_sha256'],'Canonical unchanged during analysis')
        guard();p.gone(bound['finalizer_owner']);p.predecessor_drain(bound['catalog'])
        index=prepare_index(plan,bound['reports'],parser.checked_reference)
        # Public positions category selects the actual existing position_ids label.
        tasks=schedule(plan,index)
        import numpy as np
        rows=[compare(np,task,read_raw,guard)for task in tasks];frames=summary32(plan,rows);guard()
        report=dict(schema='minicpm5_original88_first_five_analysis_v1',binding=bound['binding'],frames=32,frame_rank_pairs=192,rows=rows,
            implemented_boundaries=list(STAGES),pending_boundaries=list(PENDING),full_attention_analyzed=False,
            complete_coverage=all(x['blocked']==0 for x in frames),original_remaining_gates=copy.deepcopy(plan['remaining_gates']),
            original_logits_thresholds=dict(cosine_min=0.995,kl_max=0.02,max_abs_max=2,aggregate_argmax_min=0.9,hf_margin_gt_0_5_must_match=True),
            new_thresholds=False,full_model_qualified=False,new_speedup_claim=False,causal_operator_proven=False)
        p.once(paths['report'],report);p.once(paths['summary'],dict(frames=frames,pending_boundaries=list(PENDING),full_attention_analyzed=False))
        fields=['frame','call','phase','rank','left_arm','right_arm','stage','status','reason','compared_elements','raw_exact','different_elements','max_abs','mean_abs','cosine','nonfinite_elements','left_dtype','right_dtype','physical_rows','positions','feature_slices','left_layout','right_layout']
        with paths['csv'].open('x',newline='')as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({key:json.dumps(row.get(key),separators=(',',':'))if isinstance(row.get(key),(dict,list))else row.get(key)for key in fields})
        guard();return {name:dict(path=str(path),sha256=p.sha(path))for name,path in paths.items()}
    finally:post.external.release_lock(fd)

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    for name in('finalization','catalog','finalizer-exit'):cli.add_argument('--'+name,required=True);cli.add_argument('--'+name+'-sha256',required=True)
    cli.add_argument('--output-prefix',required=True);args=cli.parse_args();p.require(sys.dont_write_bytecode and __debug__,'python -B required')
    refs=[dict(path=getattr(args,name),sha256=getattr(args,name+'_sha256'))for name in('finalization','catalog','finalizer_exit')]
    print(json.dumps(run(*refs,args.output_prefix),indent=2))
