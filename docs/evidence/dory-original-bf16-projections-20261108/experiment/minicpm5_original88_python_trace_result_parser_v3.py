"""Original88 independently authenticated metadata/raw completion.

Import is stdlib/source only. completion and native160_prerequisite are explicit
future serial-runtime operations; source controls must mock every raw/process
boundary. No model execution or native imports are performed by this parser.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import minicpm5_layer0_execution_protocol_v7 as p
import minicpm5_original88_python_trace_validation_v2 as science
from minicpm5_layer0_runtime_auth_v1 import identity


def checked_reference(reference,directory=None):
    p.require(type(reference)is dict and set(reference)>={'path','sha256'},'Completion reference fields')
    path=Path(reference['path'])
    if directory is not None:p.require(path.parent==directory and path.name=='capture.json','Exact writer receipt location')
    before=identity(path);value=p.read(path,reference['sha256'])
    p.require(identity(path)==before,'Completion metadata replaced during read')
    return value


def ordered_reports(spec,plan,admission_ref,admission):
    """Authenticate worker owners/results; preserve actual fixture report order."""
    p.require(admission['input_plan']==spec['input_plan']and admission['predecessor_catalog']==spec['predecessor_catalog'], 'Completion admission binding')
    contracts=p.contracts(spec,plan);control=Path(spec['control_root']);reports=[];owners={};results={}
    for stage in p.STAGES:
        service=p.read(control/(stage+'-owner.json'))
        if stage=='native':partitions=[(-1,'native')]
        else:partitions=[(rank,stage+'-rank'+str(rank))for rank in(0,1)]
        for rank,prefix in partitions:
            owner=service if rank==-1 else p.read(control/(prefix+'-owner.json'))
            row=p.read(control/(prefix+'-result.json'))
            p.require(row['passed']is row['diagnostic_complete']is True and row['stage']==stage and row['rank']==rank and
                      row['process']=={k:v for k,v in owner.items()if k not in('unit','invocation_id')}and
                      row['service_owner']==service and row['admission_sha256']==admission_ref['sha256']and
                      row['full_model_qualified']is row['new_speedup_claim']is False,'Worker result/actual owner/source admission')
            result=row['result']
            verified=science.validate_worker_result(result,plan,contracts,p.read_ref,p.ARMS[stage],rank)
            p.require(row['science']==verified,'Worker scientific summary recomputed')
            owners[p.ARMS[stage],rank]=owner;results[p.ARMS[stage],rank]=row
            reports.extend(result['reports'])
        if stage!='native':
            combined=p.read(control/(stage+'-result.json'))
            p.require(combined['passed']is True and combined['stage']==stage and combined['admission_sha256']==admission_ref['sha256']and
                      combined['calls']==28 and combined['frame_occurrences']==64 and combined['scale_events']==40 and
                      combined['ranks']==[results[p.ARMS[stage],rank]for rank in(0,1)]and
                      combined['full_model_qualified']is combined['new_speedup_claim']is False,'Complete exact coordinator rank results')
    p.require(len(reports)==88 and len({x['path']for x in reports})==88,'Exact88 unique ordered report paths')
    contexts={science.owner_key(row):row for row in plan['writer_contexts']}
    p.require(len(contexts)==88,'Unique88 writer context keys')
    keys=[]
    for reference in reports:
        row=checked_reference(reference);key=science.owner_key(row['permit']);keys.append(key)
        p.require(key in contexts,'Unknown report owner');context=contexts[key]
        p.require(reference['path']==str(Path(context['directory'])/'capture.json'),'Report belongs to exact writer directory')
    p.require(keys==science.expected_keys(plan),'Original native fixture order and CPU physical order retained')
    return reports,owners,contracts


def typed_equal(actual,expected):
    """JSON value equality preserving scalar/container types recursively."""
    if type(actual)is not type(expected):return False
    if type(expected)is dict:
        return set(actual)==set(expected)and all(typed_equal(actual[key],value)for key,value in expected.items())
    if type(expected)is list:
        return len(actual)==len(expected)and all(typed_equal(a,b)for a,b in zip(actual,expected))
    return actual==expected


def check_observer_receipt(observer,context,plan,payloads):
    phase=plan['calls'][context['call']]['phase'];route='hf_eager'if context['arm']=='HF_native'else'sdk_'+phase
    p.require(observer['actual_route']==route and observer['phase']==phase and observer['original_layer_forward_calls']==1 and
              observer['wrappers_return_original_objects']is True,'Original attention route')
    events=observer['events'];snapshots=observer['snapshots']
    p.require(len(events)==len(set(events))<=64 and [x['label']for x in snapshots]==events,'Complete observed label tree')
    references=[]
    def walk(value,expected_context,depth=0):
        p.require(depth<=12,'Bounded snapshot tree')
        if isinstance(value,dict):
            if set(value)=={'metadata','reference'}:
                p.require(type(value['metadata'])is dict and typed_equal(value['metadata']['context'],expected_context),
                          'Every exported payload binds complete snapshot context')
                references.append(value);return
            for child in value.values():walk(child,expected_context,depth+1)
        elif isinstance(value,list):
            for child in value:walk(child,expected_context,depth+1)
    for event_index,snapshot in enumerate(snapshots):
        ctx=snapshot['context'];p.require(typed_equal([ctx['arm'],ctx['rank'],ctx['call']],[context['arm'],context['rank'],context['call']]),'Typed snapshot owner')
        science.exact_frames(ctx['frames'],observer['frames'])
        p.require(type(ctx['phase'])is str and ctx['phase']==phase,'Complete original snapshot phase')
        p.require(ctx['route']is None or(type(ctx['route'])is str and ctx['route']==route),'Snapshot route None or actual arm/phase route')
        p.require(type(ctx['event_index'])is int and ctx['event_index']==event_index,'Exact original snapshot event order')
        walk(snapshot['value'],ctx)
    p.require(typed_equal(references,payloads),'Every exported tensor belongs to exact typed observer snapshot tree')


def raw_verify(reference,metadata,context,index,guard):
    """Stream exact original writer raw bytes, no whole-array native imports."""
    guard();path=Path(reference['path']);directory=Path(context['directory'])
    p.require(path==directory/('tensor-%04d.raw'%index),'Exact writer raw ordinal')
    before=identity(path);info=path.lstat()
    p.require(stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_uid==1000 and stat.S_IMODE(info.st_mode)==0o600,'Owned exclusive raw')
    p.require(before==reference['identity']and before['size']==reference['bytes']==metadata['selected_bytes']<=context['tensor_cap']and
              reference['encoding']=='raw_little_endian','Exact raw identity/bytes/encoding')
    p.require(metadata['context']['arm']==context['arm']and metadata['context']['rank']==context['rank']and metadata['context']['call']==context['call'],'Raw exact owner')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);h=hashlib.sha256();count=0
    try:
        opened=os.fstat(fd);p.require((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)==
            tuple(before[k]for k in('device','inode','size','mtime_ns','ctime_ns')),'Raw descriptor identity')
        while True:
            guard();block=os.read(fd,p.MIB)
            if not block:break
            count+=len(block);p.require(count<=before['size'],'Raw grew');h.update(block)
        after=os.fstat(fd)
        p.require((opened.st_dev,opened.st_ino,opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Raw descriptor changed')
    finally:os.close(fd)
    p.require(count==before['size']and h.hexdigest()==reference['sha256']and identity(path)==before,'All raw bytes/SHA independently verified')
    return count


def completion(spec,plan,admission_ref,admission,monitor,guard):
    guard();observed=monitor.poll()
    p.require(observed['observed_contexts']==observed['observed_receipt_files']==88,'All88 final contexts/receipts')
    reports,owners,contracts=ordered_reports(spec,plan,admission_ref,admission)
    scientific=science.validate_references(reports,plan,contracts,p.read_ref)
    contexts={science.owner_key(row):row for row in plan['writer_contexts']};rows=[];seen=set();raw_files=0;raw_bytes=0
    for reference in reports:
        guard();record=checked_reference(reference);permit=record['permit'];context=contexts[science.owner_key(permit)];folder=Path(context['directory'])
        owner=owners[context['arm'],context['rank']]
        p.require(record['complete']is True and permit['schema']=='minicpm5_layer0_writer_permit_v1'and permit['attempt_id']==admission['attempt_id']and permit['plan_sha256']==spec['input_plan']['canonical_sha256']and
                  all(permit[k]==context[k]for k in('directory','arm','rank','call','frame_indices'))and
                  permit['pid']==owner['pid']and permit['start_ticks']==owner['start_ticks'],'Complete actual producer permit')
        payloads=record['payloads'];p.require(0<len(payloads)<=64,'Original per-context64payload cap')
        check_observer_receipt(record['observer'],context,plan,payloads);names=set();total=0
        for index,row in enumerate(payloads):
            path=Path(row['reference']['path']);p.require(path not in seen,'No shared raw across contexts');seen.add(path);names.add(path.name)
            total+=raw_verify(row['reference'],row['metadata'],context,index,guard);raw_files+=1
        p.require(set(x.name for x in folder.iterdir())==names|{'capture.json'}and total==record['raw_bytes']<=context['raw_cap'],'Exact partition file inventory and raw total')
        raw_bytes+=total;rows.append(dict(directory=str(folder),receipt_sha256=reference['sha256'],raw_bytes=total,payloads=len(payloads)))
    guard();final=monitor.poll()
    p.require(final['observed_contexts']==final['observed_receipt_files']==88 and final['allocated_bytes']==observed['allocated_bytes'],'Quiescent88 tree')
    return dict(passed=True,diagnostic_complete=True,contexts=rows,context_count=88,scale_events=112,science=scientific,
                all88_original_endpoints=True,all_exported_payload_hashes_verified=True,raw_files=raw_files,logical_raw_bytes=raw_bytes,
                tree=final,full_model_qualified=False,new_speedup_claim=False,causal_operator_proven=False)


def native160_inventory(old_spec,old_science,old_inputs,guard):
    """Metadata/identity inventory only; actual full bytes are reread by old parser."""
    from minicpm5_python_trace_factory_v3 import validate_partitions
    partitions=validate_partitions(old_science,old_inputs);rows=[]
    for arm in p.identities.ARMS:
        result=p.read(Path(old_spec['control_root'])/(arm+'-result.json'))
        inner=result['result']
        for item in inner['frames']:
            guard();frame=item['frame'];directory=Path(partitions[arm,frame['index']]['directory'])
            observation=p.read_ref(item['observation'])
            for payload in observation['payloads']:
                reference=payload['reference'];path=Path(reference['path'])
                p.require(path.parent==directory and identity(path)==reference['identity'],'Original160 inventory identity')
                rows.append(dict(path=str(path),sha256=reference['sha256'],bytes=reference['bytes'],identity=reference['identity']))
    p.require(len(rows)==len({row['path']for row in rows})==1552 and sum(row['bytes']for row in rows)==314961320,'Complete original1552raw inventory')
    return rows


def native160_prerequisite(spec,spec_ref,review_ref,serial_owner,lock_identity,attempt_id,guard):
    """Future explicit serial pre-admission only; never invoked in source checks."""
    guard();p.lock_proof(serial_owner,lock_identity)
    old_spec,old_science,old_inputs,old_catalog=p.identities.bundle(p.NATIVE160['spec'],p.NATIVE160['review'])
    old_audit=p.read_ref(p.NATIVE160['audit']);finalized=p.read_ref(p.NATIVE160['finalization']);release=p.read_ref(p.NATIVE160['release'])
    p.require(old_audit['passed']is old_audit['runtime_successful']is old_audit['diagnostic_complete']is True and
              finalized['passed']is finalized['post_auditor_gone']is finalized['all_groups_drained']is True and
              finalized['actual_post_auditor_exit_code']==0 and finalized['post_audit']==p.NATIVE160['audit'], 'Actually successful finalized native160 audit')
    p.gone(finalized['post_auditor']);p.require(release['successful']is True and release['released']is True,'Original successful native160 release')
    admission=p.read(Path(old_spec['control_root'])/'admission.json')
    before=native160_inventory(old_spec,old_science,old_inputs,guard)
    from minicpm5_python_trace_result_parser_v4 import completion as original_completion
    result=original_completion(old_spec,old_science,old_inputs,admission,guard)
    after=native160_inventory(old_spec,old_science,old_inputs,guard)
    p.require(before==after and result==old_audit['completion'],'Fresh full raw audit matches original successful scientific outcome')
    receipt=dict(schema='minicpm5_original88_native160_prerequisite_v1',passed=True,execution_spec=spec_ref,source_review=review_ref,
        controller=serial_owner,lock_identity=lock_identity,attempt_id=attempt_id,predecessor_catalog=spec['predecessor_catalog'],native160=p.NATIVE160,
        completion=result,raw_inventory=before,raw_inventory_sha256=p.digest(before),historical_payloads_rehashed=False)
    binding=dict(execution_spec=spec_ref,source_review=review_ref,controller=serial_owner,lock_identity=lock_identity,attempt_id=attempt_id,predecessor_catalog=spec['predecessor_catalog'])
    p.validate_native160_receipt(receipt,binding);guard()
    return receipt
