"""Root post-exit audit of finite system package metadata, never a launcher.

Root supplies the actual external terminal and a separate source review. Saved
output authentication is not package, native ABI, or model qualification.
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time
import types
import zlib
Q=Path(__file__).resolve().parent;ROOT=Q.parent;sys.path[:0]=[str(Q),str(ROOT)]
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import run_dory_serial_stage_root_v5 as lifecycle
import dory_ubuntu_metadata_input_binding_v5 as binding
import dory_ubuntu_metadata_saved_audit_v5 as audit

OUT=None;PREFIX=None;DEST=None;PINS=None
TOP={'initial-lock-proof.json','owner.json','admission.json','child-owner.json','child-exit.json',
 'result.json','worker-result.json','metadata-result.json','failure.json','worker-failure.json','cleanup-pending.json','worker.log','events.jsonl.gz',
 'first-descriptor-duplicate-parent.json','first-descriptor-duplicate-worker.json',
 'descriptor-lock-failure-parent.json','descriptor-lock-failure-worker.json'}


def decode(raw):
    def finite(s):
        v=float(s)
        if not math.isfinite(v):raise ValueError('Finite JSON number')
        return v
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('Unique JSON keys')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=unique,parse_float=finite,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Finite JSON')))


def same_ref(a,b):
    return a['path']==b['path'] and a['sha256']==b['sha256'] and a.get('bytes',b.get('bytes'))==b.get('bytes',a.get('bytes'))


def stream_hash(stream,cap,guard):
    h=hashlib.sha256();count=0
    while True:
        guard();block=stream.read(min(65536,cap-count+1))
        if not block:break
        count+=len(block);c.require(count<=cap,'Complete bounded input');h.update(block)
    return dict(bytes=count,sha256=h.hexdigest())


def stable_open(path,cap):
    path=Path(path);c.require(path.is_absolute() and path.resolve()==path,'Canonical output path')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        c.require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_uid==os.getuid()
            and 0<=info.st_size<=cap and c.identity(info)==c.identity(path.lstat()),'Stable bounded ordinary output')
        return os.fdopen(fd,'rb'),info
    except BaseException:os.close(fd);raise


def unchanged(path,stream,info):
    c.require(Path(path).resolve()==Path(path) and c.identity(info)==c.identity(os.fstat(stream.fileno()))==c.identity(Path(path).lstat()),'Saved output unchanged')


def output_inventory(guard,*,producer_failed=False):
    refs={};values={};parse_errors={};total=0;count=0;pending=[OUT]
    c.require(OUT.resolve()==OUT,'Canonical original output tree')
    while pending:
        path=pending.pop();guard();info=path.lstat();count+=1
        c.require(count<=2000 and not stat.S_ISLNK(info.st_mode),'Bounded output names without links')
        total+=max(info.st_size,info.st_blocks*512);c.require(total<=c.INVENTORY['output_bytes'],'Original complete allocated output cap including directories')
        if stat.S_ISDIR(info.st_mode):
            c.require(path in (OUT,OUT/'signed-metadata'),'Only root and fixed signed-metadata directory');pending.extend(sorted(path.iterdir()));continue
        relative=str(path.relative_to(OUT));c.require(relative in TOP or relative in audit.NESTED,'Known finite signed metadata output only')
        cap=audit.NESTED[relative] if relative in audit.NESTED else 64*c.MIB if relative=='events.jsonl.gz' else 8*c.MIB if relative=='worker.log' or 'descriptor' in relative or relative in ('initial-lock-proof.json','metadata-result.json') else 32*c.MIB
        stream,before=stable_open(path,cap)
        with stream:
            ref=dict(path=str(path),**stream_hash(stream,cap,guard))
            if path.suffix=='.json':
                stream.seek(0);raw=stream.read(cap+1);c.require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'Exact bounded JSON reread')
                try:values[relative]=decode(raw)
                except (ValueError,RuntimeError) as error:
                    c.require(producer_failed,'Successful producer requires complete output JSON')
                    parse_errors[relative]=dict(error_type=type(error).__name__,error=str(error)[:1024],complete_json=False)
            unchanged(path,stream,before)
        refs[relative]=ref
    return refs,values,total,parse_errors


def saved_history(plan,owner,p):
    rows=owner['history'];expected=plan['predecessor_services']
    c.require(len(rows)==len(expected)>0 and len(plan['predecessor_owners'])>0,'Complete reviewed plan history without guessed counts')
    c.require(len({r['unit'] for r in rows})==len(rows),'No duplicate saved history rows')
    for row,want in zip(rows,expected):
        c.require(row['unit']==want['unit'] and len(row['samples'])==2 and row['samples'][0]==row['samples'][1],'Ordered stable original terminal samples')
        sample=row['samples'][0];properties=sample['properties'];group=sample['group']
        c.require(p.terminal(properties),'Saved original terminal/no Job')
        if want.get('invocation_id'):c.require(properties['InvocationID']==want['invocation_id'],'Exact known saved InvocationID')
        c.require(group['group']==p.GROUP_PREFIX+want['unit']+'.service' and type(group['absent'])is bool,'Exact saved owned service group')
        c.require(group['absent'] or type(group.get('cgroups'))is int and 1<=group['cgroups']<=256,'Saved empty group coverage')
    return dict(entries=len(rows),stable_exact_known_invocations=True,current_services_reobserved=False)


def saved_descriptor(observed,owner,fd):
    """Independently parse saved text only; no live IO or new lifetime claim."""
    lock=runtime.ORIGINAL_LOCK;core=runtime.core
    runtime.initial_record(observed,owner,lock,fd,PINS['plan'],PINS['review'])
    c.require(observed['lock_path']==core.LOCK_PATH and observed['before']==observed['after'],'Original stable descriptor snapshots')
    for key in ('continuous_lifetime_proven','same_inode_fd_ABA_excluded','fork_inheritance_excluded','original_guard_invoked','proof_retried'):
        c.require(observed[key] is False,'Original descriptor proof boundary')
    snap=observed['before'];c.require(set(snap)=={'path_stat','fd_stat','fd_target'} and snap['path_stat']==snap['fd_stat']
        and snap['fd_target']==core.LOCK_PATH,'Exact original descriptor/path stat equality')
    mapping=dict(device='st_dev',inode='st_ino',mode='st_mode',nlink='st_nlink',uid='st_uid',gid='st_gid',size='st_size',mtime_ns='st_mtime_ns',ctime_ns='st_ctime_ns')
    s=snap['path_stat'];c.require(set(s)==set(mapping) and dict(device=s['device'],inode=s['inode'])==lock,'Fixed original stat identity')
    c.require(core.stat_record(types.SimpleNamespace(**{value:s[key] for key,value in mapping.items()}))==s,'Complete saved stat semantics')
    for name,path,cap,check in (
        ('fdinfo','/proc/%d/fdinfo/%d'%(owner['pid'],fd),core.FDINFO_CAP,core.check_fdinfo),
        ('global','/proc/locks',core.GLOBAL_CAP,core.check_global)):
        row=observed[name+'_read'];raw=row['text'].encode('ascii')
        c.require(row['path']==path and row['byte_limit']==cap and row['eof_observed'] is True
            and type(row['read_calls'])is int and 2<=row['read_calls']<=core.MAX_READS
            and row['bytes_read']==len(raw)<=cap and raw.endswith(b'\n'),'Complete saved bounded EOF text')
        c.require(check(row['text'],owner,lock)==observed[name],'Recomputed saved descriptor parsing and target row counts')


def validate_bindings(plan,review,terminal,admitted,refs,v,p,authenticated):
    audit.terminal_bindings(PINS,dict(plan=plan,review=review,terminal=terminal,admitted=admitted))
    parent=terminal['owner'];child=terminal['child_owner']
    for reference in terminal['saved_owner_provenance']:
        name=Path(reference['path']).name;c.require(name in ('owner.json','child-owner.json') and same_ref(reference,refs[name]),'Exact post-terminal saved-owner provenance')
    c.require({Path(r['path']).name for r in terminal['saved_owner_provenance']}=={'owner.json','child-owner.json'},'Both original saved owner references')
    c.require(same_ref(admitted['admission'],refs['admission.json']) and same_ref(admitted['initial_lock_proof'],refs['initial-lock-proof.json']), 'Exact saved admission/initial proof observation')
    if terminal['actual_tool']['exit_code']==0:c.require(same_ref(admitted['result'],refs['result.json']) and same_ref(terminal['result'],refs['result.json']),'Both actual terminal result references')
    a=v['admission.json'];first=v['initial-lock-proof.json'];owner=v['owner.json'];co=v['child-owner.json'];fd=a['retained_fd']
    binding.validate_admission(a,authenticated)
    c.require(owner['work_kind']==binding.WORK_KIND and owner['metadata_input_binding']==authenticated,'Saved owner binds authenticated metadata task')
    duplicate=v.get('first-descriptor-duplicate-parent.json') if a['parent_first_duplicate'] not in (None,a['first_lock_proof']) else None
    runtime.validate_admission(a,plan,PINS['plan'],first,duplicate)
    saved_descriptor(first['first_observation'],parent,a['retained_fd'])
    c.require(a['review']==PINS['review'] and a['owner']==parent and type(fd)is int and fd>=0
        and a['first_lock_proof']==refs['initial-lock-proof.json'],'Original proof/owner/fd/review binding')
    c.require(owner['owner']==parent and owner['lock']==runtime.ORIGINAL_LOCK and owner['retained_fd']==fd
        and owner['lock_policy']==runtime.POLICY and owner['plan']==PINS['plan'] and owner['review']==PINS['review']
        and owner['first_lock_proof']==refs['initial-lock-proof.json'],'Saved owner complete bindings')
    c.require(co['owner']==child and co['command']==child['argv']==a['command']+['--admission-sha256',refs['admission.json']['sha256']]
        and co['admission']==refs['admission.json'],'Original child ACK command binding')
    for role in ('parent','worker'):
        name='first-descriptor-duplicate-'+role+'.json'
        if name not in v:continue
        d=v[name];c.require(d['schema']=='dory_inventory_first_duplicate_observation_v1' and d['role']==role
            and d['owner']==parent and d['lock']==runtime.ORIGINAL_LOCK and d['retained_fd']==fd
            and d['plan']==PINS['plan'] and d['review']==PINS['review'] and d['lock_policy']==runtime.POLICY,'Complete first duplicate metadata bindings')
        runtime.initial_record(d['observations'],parent,runtime.ORIGINAL_LOCK,fd,PINS['plan'],PINS['review'])
        saved_descriptor(d['observations'],parent,fd)
        c.require(d['observations']['global']['target_row_count']>1,'Saved accepted duplicate rows')
        if d['admission'] is None:c.require(role=='parent' and a['parent_first_duplicate']==refs[name],'Pre-admission reverse binding')
        else:c.require(same_ref(d['admission'],refs['admission.json']),'Actual duplicate admission binding')
    ce=v['child-exit.json'];c.require(ce['owner']==child and type(ce['returncode'])is int and ce['parent_observed'] is True and ce['entire_process_group_empty'] is True,'Saved child exit; live drain checked separately')
    return parent,child,saved_history(plan,owner,p)


def read_events(stream,guard,accept,*,raw_cap=256*c.MIB,line_cap=64*c.MIB,event_cap=200000):
    h=hashlib.sha256();size=0;counts=collections.Counter();last=None;last_hash=None
    def incomplete(error):
        return dict(complete=False,raw_scope='bounded_returned_prefix_only',error_type=type(error).__name__,error=str(error)[:1024],raw_bytes=size,raw_sha256=h.hexdigest(),events=sum(counts.values()),event_counts=dict(counts),last_event_sha256=last_hash),last
    with gzip.GzipFile(fileobj=stream,mode='rb') as zipped:
        while True:
            guard()
            try:line=zipped.readline(min(line_cap,raw_cap-size)+1)
            except (OSError,EOFError,zlib.error) as error:return incomplete(error)
            if not line:break
            size+=len(line);c.require(size<=raw_cap and len(line)<=line_cap,'Bounded JSONL event');h.update(line)
            if not line.endswith(b'\n'):return incomplete(ValueError('Incomplete final JSONL line'))
            try:value=decode(line)
            except ValueError as error:return incomplete(error)
            c.require(type(value)is dict and type(value.get('event'))is str and 1<=len(value['event'])<=128,'Named event')
            counts[value['event']]+=1;c.require(sum(counts.values())<=event_cap and len(counts)<=64,'Bounded event count/kinds')
            last=value;last_hash=hashlib.sha256(line).hexdigest();accept(value)
    return dict(complete=True,raw_bytes=size,raw_sha256=h.hexdigest(),events=sum(counts.values()),event_counts=dict(counts),last_event_sha256=last_hash),last


def event_audit(ref,guard,values,refs,success):
    checker=audit.Events(OUT,refs)
    stream,before=stable_open(ref['path'],64*c.MIB)
    with stream:
        c.require(dict(path=ref['path'],**stream_hash(stream,64*c.MIB,guard))==ref,'Complete compressed journal SHA')
        stream.seek(0);summary,last=read_events(stream,guard,checker.accept,event_cap=19);unchanged(ref['path'],stream,before)
    summary['compressed']=ref
    if success:
        expected=values['worker-result.json']['events']
        c.require(summary['complete'] and same_ref(ref,expected) and all(summary[k]==expected[k] for k in ('raw_bytes','raw_sha256','events')),'Complete journal receipt equality')
        c.require(summary['events']==19 and summary['event_counts']==dict(ubuntu_metadata_download=12,ubuntu_gpgv_start=3,ubuntu_gpgv_exit=3,ubuntu_metadata_discovery_complete=1),'Exactly12downloads/3verifiers/final completion')
        def read_saved(name,cap):
            c.require(name in refs and name in audit.NESTED and cap<=audit.NESTED[name], 'Fixed saved metadata read')
            expected=refs[name]
            stream,before=stable_open(expected['path'],cap)
            with stream:
                raw=stream.read(cap+1);c.require(len(raw)==expected['bytes']<=cap and hashlib.sha256(raw).hexdigest()==expected['sha256'],'Complete saved metadata reread')
                unchanged(expected['path'],stream,before)
            guard();return raw
        raw_inputs={key:c.reference(binding.INPUTS[key]['path'],c.MIB)[1] for key in audit.discovery.INPUT_PINS}
        summary['recomputed']=audit.replay_saved(checker,values,refs,raw_inputs,read_saved,guard)
        summary['last_event']=last
    else:
        summary.update(last_event_kind=last.get('event') if last else None,stage_success_claimed=False,
            complete_workflow_claimed=False,accepted_event_prefix=checker.index,negative_or_partial_verifier_retained=checker.failed_verifier)
    return summary


def main():
    global OUT,PREFIX,DEST,PINS
    parser=argparse.ArgumentParser()
    for name in ('plan','review','admitted','terminal'):
        parser.add_argument('--'+name,required=True);parser.add_argument('--'+name+'-sha256',required=True)
    for name in ('source-sha256','recorder-review','recorder-review-sha256','destination'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();c.require(sys.dont_write_bytecode and sys.flags.no_user_site and __debug__,'Original system Python -B -s assertions')
    PINS={name:dict(path=getattr(a,name),sha256=getattr(a,name+'_sha256')) for name in ('plan','review','admitted','terminal')}
    c.require(all(re.fullmatch('[0-9a-f]{64}',ref['sha256']) for ref in PINS.values()),'Exact supplied metadata SHA256 inputs')
    DEST=Path(a.destination);c.require(DEST.parent==Q and DEST.suffix=='.json' and DEST.name.startswith('dory-ubuntu-metadata-'),'Fresh fixed research receipt namespace')
    PREFIX=DEST.stem+'-recorder'
    review_ref=dict(path=a.recorder_review,sha256=a.recorder_review_sha256);review=c.load(review_ref)
    c.require(review['schema']=='dory_ubuntu_metadata_post_exit_source_review_v1' and review['reviewer']=='/root'
        and review['passed'] is True and review['blockers']==[] and review['producer_inputs']==PINS and review['destination']==str(DEST)
        and review['source_sha256'][str(Path(__file__).resolve())]==a.source_sha256,'Separate exact root source review')
    pins=review['source_sha256'];c.verify_sources(pins)
    inputs={k:c.load(r) for k,r in PINS.items()};plan=binding.validate(inputs['plan'],inputs['review'],PINS['plan'])
    OUT=Path(plan['output_root']);c.require(OUT.name.startswith('dory-serial-inventory-ubuntu-metadata-'),'Metadata-only output namespace')
    tool,parent,child=audit.terminal_bindings(PINS,inputs);terminal=inputs['terminal'];code=tool['exit_code']
    c.require(all(pins.get(k)==v for k,v in plan['source_sha256'].items()),'Original frozen source pins included unchanged')
    import minicpm5_layer0_execution_protocol_v4 as p
    c.verify_loaded(pins);p.user_bus()
    c.require(not DEST.exists(),'Fresh external record; preserve all prior attempts')
    for suffix in ('initial','duplicate','failure'):c.require(not (Q/(PREFIX+'-'+suffix+'.json')).exists(),'Fresh compact recorder evidence')
    start=time.monotonic();os.sched_setaffinity(0,{5});os.nice(19)
    budget=c.INVENTORY
    resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
    resource.setrlimit(resource.RLIMIT_FSIZE,(budget['single_file_bytes'],)*2)
    fd,recorder,lock,first=runtime.acquire(p,PINS['plan'],review_ref)
    failed=None;duplicate=None;pending_duplicate=None;observations=0;written=0;last_check=0.;last_proof=0.;handlers={}
    def save(path,value,cap=8*c.MIB):
        nonlocal written
        raw=runtime.encoded(value);c.require(written+len(raw)<=32*c.MIB and lifecycle.free(ROOT)>=28*c.GIB+cap,'Original root floor and bounded32MiB recorder receipts')
        ref=c.save(path,value,cap);written+=ref['bytes'];return ref
    def guard(force=False):
        # Explicit recorder-only cadence: every read checks the wall/latch;
        # the single process retains its one original fd without child/fork or
        # close while full descriptor snapshots run at most once per 0.5 s.
        # Admission and final receipt require forced snapshots. This does not
        # change the producer runtime_v2 guard or prove continuous lifetime.
        nonlocal failed,duplicate,pending_duplicate,observations,last_check,last_proof
        if failed is not None:raise failed
        try:
            now=time.monotonic();c.require(now-start<budget['wall_seconds'],'Original audit wall cap')
            if not force and now-last_proof<.5:return
            observed=runtime.observe(p,recorder,lock,fd);observations+=1
            last_proof=time.monotonic()
            if observed['global']['target_row_count']>1 and duplicate is None:
                pending_duplicate=observed
                duplicate=save(Q/(PREFIX+'-duplicate.json'),dict(schema='dory_inventory_recorder_first_duplicate_v1',owner=recorder,lock=lock,retained_fd=fd,
                    source_review=review_ref,producer_inputs=PINS,observations=observed))
                pending_duplicate=None
            now=time.monotonic();c.require(now-start<budget['wall_seconds'],'Original audit wall cap')
            if now-last_check>=1:
                p.clear_window();p.gone(parent);p.gone(child);c.require(not lifecycle.group_members(child['pid']),'Original owned child process group empty')
                c.require(lifecycle.free(ROOT)>=28*c.GIB and lifecycle.free('/mnt/nvme')>=30*c.GIB and lifecycle.memory_free()>=c.GIB,'Original storage/memory floors')
                last_check=now
        except BaseException as error:failed=error;raise
    def interrupted(signum,frame):raise InterruptedError('Post-exit audit interrupted')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        c.require(lifecycle.free(ROOT)>=28*c.GIB+32*c.MIB and lifecycle.free('/mnt/nvme')>=30*c.GIB,'Bounded32MiB compact recorder reserve and original disk floors')
        initial=save(Q/(PREFIX+'-initial.json'),dict(schema='dory_inventory_recorder_initial_lock_v1',owner=recorder,lock=lock,retained_fd=fd,producer_inputs=PINS,source_review=review_ref,first_observation=first))
        if first['global']['target_row_count']>1:duplicate=initial
        guard(True);p.active_heavy();refs,values,allocated,parse_errors=output_inventory(guard,producer_failed=code==1)
        authenticated=binding.authenticate(guard)
        parent,child,history=validate_bindings(plan,inputs['review'],terminal,inputs['admitted'],refs,values,p,authenticated)
        success=audit.successful_outputs(refs,values,code,PINS,authenticated)
        events=event_audit(refs['events.jsonl.gz'],guard,values,refs,success) if 'events.jsonl.gz' in refs else dict(complete=False,not_created=True)
        c.require(not success or events['complete'],'Successful metadata producer requires complete journal')
        c.verify_sources(pins,guard);c.verify_loaded(pins)
        guard(True);p.active_heavy();p.gone(parent);p.gone(child);c.require(not lifecycle.group_members(child['pid']),'Final actual owned group empty')
        report=dict(schema='dory_ubuntu_metadata_actual_exit_v1',work_kind=binding.WORK_KIND,live_owner_observed_by_root=False,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            actual_tool=tool,
            producer_inputs=PINS,output_root=str(OUT),parent_owner=parent,child_owner=child,parent_owner_gone=True,child_owner_gone=True,child_group_empty=True,
            evidence=refs,partial_json_outputs=parse_errors,stage_success_observed=success,original_failure_retained=not success,saved_history=history,allocated_output_bytes=allocated,
            complete_events_audit=events,original_lock_reacquired=True,lock_identity=lock,recorder=recorder,recorder_source_review=review_ref,
            recorder_initial_proof=initial,recorder_first_duplicate=duplicate,recorder_successful_lock_observations=observations,
            recorder_actual_exit_claimed=False,services_spawned_by_recorder=[],producer_services_spawned=[],service_claim_basis='Reviewed signed-metadata-only producer/recorder source; saved history authenticated, no fresh all-service census.',
            native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        result=save(DEST,report);print(json.dumps(result),flush=True)
    except BaseException as error:
        try:save(Q/(PREFIX+'-failure.json'),dict(schema='dory_ubuntu_metadata_post_exit_audit_failure_v1',owner=recorder,lock=lock,retained_fd=fd,
            producer_inputs=PINS,source_review=review_ref,error_type=type(error).__name__,error=str(error),observations=getattr(error,'observations',None),
            unpersisted_first_duplicate=pending_duplicate,
            original_producer_exit_code_supplied=code,original_producer_failure_reclassified=False,recorder_actual_exit_claimed=False))
        except BaseException as secondary:print(json.dumps(dict(error=str(error),evidence_error=str(secondary))),file=sys.stderr,flush=True)
        raise
    finally:
        # Recorder starts no children or services. Hold original fd through every
        # audit read and the final receipt/failure write; root records our exit.
        os.close(fd)
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':main()
