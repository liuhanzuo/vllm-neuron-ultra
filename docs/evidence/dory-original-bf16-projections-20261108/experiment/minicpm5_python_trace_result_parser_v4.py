"""Native160 raw and decoded-result parser; no operations at import.

Explicit completion is runtime-only. Callers provide authenticated spec and
strict guard; source tests must mock every raw/process/filesystem operation.
"""
import hashlib
import ast
import os
from pathlib import Path
import re
import stat
import struct
import minicpm5_python_trace_lifecycle_v4 as lifecycle
from minicpm5_python_trace_observer_v2 import SCHEMA
import minicpm5_python_trace_analysis_v1 as analysis

require=lifecycle.require;read=lifecycle.read;low=lifecycle.low
MIB=2**20;ARMS=lifecycle.ARMS
RuntimeGuard=lifecycle.RuntimeGuard
directory_identity=lifecycle.directory_identity
environment=lifecycle.environment
canonical=lifecycle.canonical

def expected_native_trace(inputs):
    """Reconstruct source AST boundary; native bytecode stays native evidence.

    Controller uses Python3.10, native worker3.12. Their line-table byte hashes
    must never be compared as if emitted by one interpreter. The accepted
    worker helper verifies code/line bytes against source in the native process;
    this parser checks source AST boundary and equal native contracts across
    noop/full arms, retaining their exact two native line-table hashes.
    """
    provenance=inputs['native_provenance'];path=Path(provenance['source'])
    require(path.suffix=='.py' and path.stat().st_size<1024*1024,'Bound original native Python source')
    source=path.read_text();require(hashlib.sha256(source.encode()).hexdigest()==provenance['source_sha256'],'Original native source digest')
    nodes=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='eager_attention_forward']
    require(len(nodes)==1,'Unique original eager function')
    node=nodes[0];matches=[]
    wanted=ast.dump(ast.parse('torch.matmul(query, key_states.transpose(2, 3)) * scaling',mode='eval').body)
    for index,statement in enumerate(node.body[:-1]):
        if isinstance(statement,ast.Assign) and len(statement.targets)==1 and isinstance(statement.targets[0],ast.Name) and statement.targets[0].id=='attn_weights' and ast.dump(statement.value)==wanted:
            following=node.body[index+1];require(following.lineno>statement.end_lineno,'Distinct post-assignment boundary')
            matches.append(dict(variable='attn_weights',role='qk0',assignment_line=statement.lineno,event_line=following.lineno))
    require(len(matches)==1,'Exact original source scaling assignment')
    return dict(source=dict(path=str(path),sha256=provenance['source_sha256']),function=node.name,point=matches[0],first_line=node.lineno)

def validate_native_contract(value,expected):
    require(set(value)=={'source','function','point','line_metadata'} and
        all(value[k]==expected[k]for k in ('source','function','point')),'Native contract exact original source AST boundary')
    metadata=value['line_metadata']
    require(set(metadata)=={'first_line','co_linetable_sha256','co_lnotab_sha256'} and metadata['first_line']==expected['first_line'] and
        all(type(metadata[k])is str and re.fullmatch('[0-9a-f]{64}',metadata[k])for k in ('co_linetable_sha256','co_lnotab_sha256')),
        'Retain complete native-interpreter line-table digests')

def raw_verify(ref,directory,guard):
    path=Path(ref['path']);require(path.parent==directory and re.fullmatch(r'tensor-[0-9]{4}\.raw',path.name),'Exact partition raw path')
    s=path.lstat();require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==1000 and stat.S_IMODE(s.st_mode)==0o600
        and 0<s.st_size<=16*MIB and s.st_size==ref['bytes'],'Owned raw size/type')
    key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    expected=dict(realpath=str(path),device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
    require(ref['identity']==expected and ref['encoding']=='raw_little_endian','Exact raw identity/encoding')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);data=bytearray();digest=hashlib.sha256()
    try:
        while True:
            guard();block=os.read(fd,MIB)
            if not block:break
            digest.update(block);data.extend(block);require(len(data)<=ref['bytes'],'Raw grew during audit')
        end=os.fstat(fd)
        require(key==(end.st_dev,end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns),'Raw changed during audit')
    finally:os.close(fd)
    require(len(data)==ref['bytes'] and digest.hexdigest()==ref['sha256'],'Exact raw content SHA')
    require(key==RuntimeGuard.file_key(path),'Raw path replaced during audit')
    return bytes(data)


def bf16_to_finite_fp32(raw):
    """BF16 finite values embed exactly into IEEE binary32, preserving signed0.

    NaN/Inf are rejected as incomplete numeric diagnostics; no assumed NaN
    payload/canonicalization rule is used to promote equality.
    """
    require(len(raw)==130560*2,'Full-vocabulary BF16 bytes')
    output=bytearray(len(raw)*2)
    for index,(value,) in enumerate(struct.iter_unpack('<H',raw)):
        require(value&0x7f80!=0x7f80,'Nonfinite BF16 cannot complete native diagnostic')
        struct.pack_into('<I',output,index*4,value<<16)
    return bytes(output)


def validate_arm_result(result,arm,owner,science):
    require(result['schema']=='minicpm5_python_trace_worker_result_v1' and result['arm']==arm
        and result['pid']==owner['pid'] and result['start_ticks']==owner['start_ticks']
        and result['source_plan_sha256']==low.digest(science) and result['diagnostic_complete'] is True,'Actual complete worker result')
    inner=result['result'];require(inner['arm']==arm and inner['diagnostic_complete'] is True and inner['observed_frames']==32
        and len(inner['frames'])==32 and {x['frame']['index'] for x in inner['frames']}==set(range(32)),'Exactly32 reported frames')
    equal=sum(x['passed'] is True for x in inner['frames'])
    require(inner['raw_exact_frames']==equal and result['passed'] is inner['passed'] is (equal==32)
        and inner['original_hard_gate_preserved'] is True,'Exact gate count retained')
    require(result['full_model_qualified'] is result['new_speedup_claim'] is inner['full_model_qualified'] is inner['new_speedup_claim'] is False,'Diagnostic only')
    require(inner['schema']=='minicpm5_python_trace_arm_result_v1' and all(type(x['passed']) is bool for x in inner['frames']),'Typed original arm result')
    require(inner['scale_observation_schema']==SCHEMA and inner['aten_dispatch_used'] is False and inner['original88_scope_preserved'] is True,'Exact Python trace schema and original88 scope')
    counts=inner['noop_call_counts']
    require(isinstance(counts,dict) and counts.get('attention.forward')==32 if arm=='noop_wrappers' else counts is None,'Original noop wrapper scope')
    counts=inner['trace_call_counts']
    if arm in ('noop_trace','full_trace'):
        require(isinstance(counts,dict) and counts['mode']==arm and counts['calls']==32 and counts['aten_dispatch_used'] is False and
            counts['observed_scale_events']==(32 if arm=='full_trace'else 0) and counts['noop_scale_events']==(32 if arm=='noop_trace'else 0),'Exact32 independent trace calls')
    else:require(counts is None,'No trace invented in control arm')
    return inner


def completion(spec,science,inputs,admission,guard):
    from minicpm5_python_trace_factory_v3 import validate_partitions
    rows=validate_partitions(science,inputs);root=Path(spec['output_root']);control=Path(spec['control_root'])
    require(directory_identity(root)==admission['output_root_identity'],'Audited output identity')
    require({p.name for p in root.iterdir()}=={Path(row['directory']).name for row in rows.values()},'Exact160 partition inventory')
    hf={i:read(row['path'],row['sha256']) for i,row in enumerate(inputs['hf_frame_records'])}
    totals={};raw_files=0;logical_bytes=0;runs={};decoded={};expected_trace=expected_native_trace(inputs)
    for arm in ARMS:
        guard();owner=read(control/(arm+'-owner.json'));result=read(control/(arm+'-result.json'))
        require(result['attempt_id']==spec['attempt_id'],'Original common attempt result')
        inner=validate_arm_result(result,arm,owner,science);results={x['frame']['index']:x for x in inner['frames']}
        runs[arm]=dict(owner=owner,result=inner)
        if arm in ('noop_trace','full_trace'):validate_native_contract(inner['trace_call_counts']['contract'],expected_trace)
        before=read(control/(arm+'-backend-before.json'));after=read(control/(arm+'-backend-after.json'))
        require(before==after and before['threads']==2 and before['environment']==environment(spec)
            and before['python']==spec['python'] and before['settings_changed'] is False,'Stable original native backend/environment')
        require(before['schema']=='minicpm5_native_backend_observation_v1' and before['affinity']==[6,7] and before['nice']==19
            and before['module_files']['native_llama']==inputs['native_provenance']['source']
            and all(path in spec['source_sha256'] for path in before['module_files'].values()),'Actual pinned original native backend')
        equal=0
        for index in range(32):
            guard();row=rows[arm,index];directory=Path(row['directory']);frame=row['frame'];r=results[index]
            require(r['frame']==frame,'Result original frame')
            witness=read(r['witness']['path'],r['witness']['sha256']);observation=read(r['observation']['path'],r['observation']['sha256'])
            for reference,value in ((r['witness'],witness),(r['observation'],observation)):
                key=reference['path'],reference['sha256'];require(key not in decoded,'No shared receipt across original frames/arms');decoded[key]=value
            require(Path(r['witness']['path'])==directory/'witness.json' and Path(r['observation']['path'])==directory/'observation.json'
                and r['witness']['durable'] is r['observation']['durable'] is True,'Exact durable frame receipt paths')
            binding=witness['binding'];permit=binding['base_permit']
            expected_permit=dict(schema='minicpm5_layer0_writer_permit_v1',directory=str(directory),pid=owner['pid'],start_ticks=owner['start_ticks'],
                arm='HF_native',rank=-1,call=row['call'],frame_indices=[index],attempt_id=spec['attempt_id'],plan_sha256=low.digest(science))
            expected_binding=dict(schema='minicpm5_python_trace_writer_binding_v1',base_permit=expected_permit,frame=frame,arm=arm,
                raw_cap=row['raw_cap'],observer_raw_cap=row['observer_raw_cap'],metadata_cap=row['metadata_cap'],max_payloads=row['max_payloads'])
            require(binding==expected_binding,'Exact immutable writer permit and partition caps')
            require(binding['arm']==arm and binding['frame']==frame and observation['binding']==binding
                and permit['directory']==str(directory) and permit['pid']==owner['pid'] and permit['start_ticks']==owner['start_ticks']
                and permit['attempt_id']==spec['attempt_id'] and permit['plan_sha256']==low.digest(science)
                and permit['frame_indices']==[index] and permit['call']==row['call'],'Actual producer frame binding')
            identity=witness['identity'];require(identity['arm']==arm and identity['frame']==frame and identity['call']==row['call']
                and identity['reference_payload_sha256']==inputs['hf_reference']['sha256']
                and identity['saved_hf_frame_payload_sha256']==hf[index]['payload_sha256']
                and identity['own_arm_cache_history'] is identity['frozen_reference_teacher_forcing'] is True,'Original reference/own-history binding')
            fixture=next(x for x in inputs['fixtures'] if x['id']==frame['fixture']);step=frame['step'];length=len(fixture['input_ids'])
            require(identity['input_ids']==([fixture['input_ids']] if step==0 else [[fixture['continuation_ids'][step-1]]])
                and identity['attention_mask_shape']==[1,length+step]
                and identity['position_ids']==([list(range(length))] if step==0 else [[length+step-1]])
                and identity['cache_length']==length+step,'Original arguments/teacher tokens')
            require(witness['original_full_vocab']==130560 and witness['diagnostic_observed'] is True
                and witness['full_model_qualified'] is witness['new_speedup_claim'] is False,'Fullvocab diagnostic scope')
            require(observation['arm']==arm and observation['frame']==frame,'Observation original arm/frame')
            payloads=observation['payloads'];refs={x['reference']['path']:x['reference'] for x in payloads}
            require(len(refs)==len(payloads) and 4<=len(refs)<=row['max_payloads'],'Unique bounded complete payloads')
            require({p.name for p in directory.iterdir()}=={'witness.json','observation.json'}|{Path(p).name for p in refs},'No partial/failure/foreign frame files')
            vectors={x['name']:x for x in witness['tensors']}
            expected_names=['actual-bf16','actual-fp32','expected-fp32','expected-bf16-derived']
            require(len(witness['tensors'])==4 and list(vectors)==expected_names,'All four ordered unique witnesses')
            for payload in payloads:
                meta=payload['metadata'];context=meta['context']
                require(meta['selected_bytes']==payload['reference']['bytes'] and context['arm']=='HF_native' and context['rank']==-1
                    and context['call']==row['call'] and context['frames']==[frame],'Payload original call/frame metadata')
            data={}
            for path,refvalue in refs.items():
                content=raw_verify(refvalue,directory,guard);raw_files+=1;logical_bytes+=len(content)
                for name,value in vectors.items():
                    if value['reference']['path']==path:
                        require(value['reference']==refvalue,'Witness raw reference bound to observation inventory')
                        meta=value['metadata'];dtype='torch.float32' if name in ('actual-fp32','expected-fp32') else 'torch.bfloat16'
                        require(meta['shape']==[130560] and meta['dtype']==dtype and meta['bytes']==len(content)
                            and meta['sha256']==refvalue['sha256'] and meta['identity']==identity,'Witness tensor schema/hash')
                        data[name]=content
            require(set(data)==set(expected_names),'Every witness raw present')
            require(hashlib.sha256(data['expected-bf16-derived']).hexdigest()==hf[index]['tap_raw_sha256']['lm_head'],'Original successfully captured BF16 head binding')
            require(bf16_to_finite_fp32(data['expected-bf16-derived'])==data['expected-fp32']
                and bf16_to_finite_fp32(data['actual-bf16'])==data['actual-fp32'],'Exact BF16/FP32 conversion, finite only')
            exact=data['actual-fp32']==data['expected-fp32'];equal+=exact
            require(witness['actual_fp32_sha256']==hashlib.sha256(data['actual-fp32']).hexdigest()
                and witness['original_expected_fp32_sha256']==hashlib.sha256(data['expected-fp32']).hexdigest(),'Endpoint raw hashes independently verified')
            require(witness['passed'] is witness['raw_exact'] is observation['passed'] is observation['raw_exact'] is r['passed'] is exact,'No relaxed endpoint gate')
            require(witness['original_gate_failure']==(None if exact else 'Original native32 logits changed')
                and observation['complete'] is True and observation['full_model_qualified'] is False
                and observation['witness']==r['witness'],'Durable endpoint result binding')
            require(sum(x['reference']['bytes'] for x in payloads)==observation['raw_bytes']<=row['raw_cap'],'Actual frame bytes')
            if arm=='full_trace':
                observed=observation['observer'];require(observed['complete'] is True and observed['arm']=='HF_native'
                    and observed['frames']==[frame] and observed['call']==row['call'] and observed['original_layer_forward_calls']==1,'Original full observer scope')
                require(observed['rank']==-1 and observed['phase']==('prefill' if step==0 else 'decode') and observed['actual_route']=='hf_eager'
                    and observed['wrappers_return_original_objects'] is observed['full_model_endpoint_gate_still_required'] is True
                    and observed['production_runtime_tested'] is False,'Original observer route and diagnostic scope')
                required={'normalized_input','attention_mask','position_ids','q_projection','k_projection','v_projection','q_rope_input','k_rope_input',
                    'v_heads','rope_position_ids','rotary_route','q_post_rope','k_post_rope','rope_cos','rope_sin','context_before_o','attention_output',
                    'qk0_unscaled','qk0_scaled','qk0_scale_operation','softmax_input','probabilities_bf16','cache_before','cache_after',
                    'native_effective_key','native_effective_value','probabilities_fp32'}
                require(required<=set(observed['events']) and len(set(observed['events']))==len(observed['events'])
                    and [x['label'] for x in observed['snapshots']]==observed['events'],'Original complete unique observer labels')
                snapshot_refs=[];payload_index={x['reference']['path']:x for x in payloads}
                def collect(value):
                    if isinstance(value,dict):
                        if {'reference','metadata'}<=set(value) and isinstance(value['reference'],dict) and 'path' in value['reference']:
                            require(payload_index.get(value['reference']['path'])==value,'Exact observer snapshot metadata/inventory')
                            snapshot_refs.append(value['reference'])
                        for item in value.values():collect(item)
                    elif isinstance(value,list):
                        for item in value:collect(item)
                collect(observed['snapshots'])
                for event_index,snapshot_row in enumerate(observed['snapshots']):
                    context=snapshot_row['context']
                    require(context['arm']=='HF_native' and context['rank']==-1 and context['call']==row['call'] and context['frames']==[frame]
                        and context['phase']==observed['phase'] and context['event_index']==event_index,'Ordered original snapshot context')
                witnesses={x['reference']['path'] for x in vectors.values()}
                require({x['path'] for x in snapshot_refs}==set(refs)-witnesses and all(refs[x['path']]==x for x in snapshot_refs),'All full observer snapshot raw references preserved')
            else:require(observation['observer'] is None and len(refs)==4,'Non-full arm has no observer snapshots')
            if arm=='previous213_hooks':
                taps=observation['previous213'];require(taps['count']==213 and len(taps['raw_sha256'])==len(taps['metadata'])==213,'Exact original213 hook scope')
                require(taps['original_metadata_equal'] is (taps['metadata']==hf[index]['metadata'])
                    and taps['original_tap_hashes_equal'] is (taps['raw_sha256']==hf[index]['tap_raw_sha256']),'Original213 comparisons truthful')
            else:require(observation['previous213'] is None,'No extra prior hooks')
        require(equal==inner['raw_exact_frames'],'Independent per-arm endpoint count');totals[arm]=equal
    metadata=analysis.summarize(inputs['frames'],runs,lambda ref:decoded[(ref['path'],ref['sha256'])])
    require({x['arm']:x['exact_original_frames']for x in metadata['arm_summaries']}==totals,'Independent raw and decoded160 result counts agree')
    require(canonical()==admission['canonical_sha256'],'Final unchanged canonical')
    return dict(diagnostic_complete=True,frames=160,arms=totals,all_raw_exact=all(x==32 for x in totals.values()),
        raw_files=raw_files,logical_raw_bytes=logical_bytes,original_hard_gate_preserved=True,original88_scope_preserved=True,
        scale_observation_schema=SCHEMA,paired_comparisons=metadata['paired_comparisons'],all_arm_pairs_equal=metadata['all_arm_pairs_equal'],
        raw_audit_complete=True,metadata_analysis=metadata,
        full_model_qualified=False,new_speedup_claim=False,causal_operator_proven=False)


