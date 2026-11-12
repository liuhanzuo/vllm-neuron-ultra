"""Explicit Mini snapshot archive successor: parent, worker and external recorder.

Lifecycle bodies retain the frozen v3 behavior; only explicit input binding,
worker executable/--mode and the metadata source-count contract are rewired.
No work, lock, service, native import or payload read occurs on import.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import select
import signal
import subprocess
import sys
import time
Q=Path(__file__).resolve().parent;ROOT=Q.parent;sys.path[:0]=[str(Q),str(ROOT)]
import minicpm5_sdk310_snapshot_contract_v1 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import minicpm5_sdk310_snapshot_read_binding_v1 as inputs
import dory_deb_transport_v3 as io
import minicpm5_sdk310_deb_events_v1 as event_protocol
import run_dory_serial_stage_root_v5 as life
from run_dory_serial_stage_root_v5 import free,memory_free,drain,release_after_drain
import minicpm5_sdk310_retired_lfm_r2_v2 as retirement
from minicpm5_sdk310_retired_lfm_r2_v2 import preflight_history
from minicpm5_sdk310_deb_input_binding_v1 import storage_usage,save_final_result


def parent_main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('plan','plan-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Original python -B assertions')
    plan_ref=dict(path=a.plan,sha256=a.plan_sha256);review_ref=dict(path=a.review,sha256=a.review_sha256)
    plan=inputs.validate(c.load(plan_ref),c.load(review_ref),plan_ref);c.verify_sources(plan['source_sha256'])
    c.require(str(Path(__file__).resolve())in plan['source_sha256'],'Launcher itself reviewed')
    import minicpm5_layer0_execution_protocol_v4 as p
    import minicpm5_layer0_external_supervisor_v4 as external
    c.verify_loaded(plan['source_sha256']);p.user_bus()
    out=Path(plan['output_root']);c.require(not out.exists(),'Fresh stage output; preserve all prior evidence')
    fd,owner,lock,first_proof=runtime.acquire(p,plan_ref,review_ref);started=time.monotonic();last=[0.];child=None;child_owner=None;readfd=writefd=None;handlers={}
    initial=None;budget=plan['budget'];created=False
    proof=runtime.Guard(p,owner,lock,fd,out,plan_ref,review_ref,budget,role='parent',created=lambda:created,first_observation=first_proof)
    def guard():
        proof();p.clear_window();c.require(time.monotonic()-started<budget['wall_seconds'],'Stage wall bound')
        now=time.monotonic()
        if now-last[0]>=1:
            p.active_heavy()
            for previous in plan['predecessor_owners']:p.gone(previous)
            c.require(free(ROOT)>=28*c.GIB and free('/mnt/nvme')>=max(30*c.GIB,initial-budget['output_bytes']),'Original reserved disk floors')
            c.require(memory_free()>=budget['memory_available_floor_bytes'],'Reviewed memory availability floor')
            if created:c.require(storage_usage(out)<=budget['output_bytes'],'Complete allocated output budget')
            last[0]=now
    def interrupted(signum,frame):raise InterruptedError('Dory serial stage interrupted')
    try:
        initial=free('/mnt/nvme')
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        guard();history=preflight_history(plan,p)
        metadata_binding=inputs.authenticate(plan,guard)
        c.require(initial>=30*c.GIB+budget['output_bytes'],'Full fresh output reservation above original floor')
        os.sched_setaffinity(0,{5});os.nice(19)
        # Parent never loads a model. Child has independently reviewed limits.
        resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(budget['single_file_bytes'],)*2)
        out.mkdir(mode=0o700);created=True
        directory_fd=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        first_ref=c.save(out/'initial-lock-proof.json',runtime.initial_record(first_proof,owner,lock,fd,plan_ref,review_ref),runtime.CAP)
        proof.initial_persisted(first_ref)
        c.save(out/'owner.json',dict(owner=owner,lock=lock,retained_fd=fd,lock_policy=runtime.POLICY,first_lock_proof=first_ref,plan=plan_ref,review=review_ref,history=history,work_kind=inputs.WORK_KIND,metadata_input_binding=metadata_binding))
        worker=str(Q/'run_minicpm5_sdk310_snapshot_archive_v1.py');c.require(worker in plan['source_sha256'],'Worker source reviewed')
        readfd,writefd=os.pipe();python='/usr/bin/python3'
        argv=[python,'-B','-s',worker,'--mode','worker','--plan',a.plan,'--plan-sha256',a.plan_sha256,'--admission',str(out/'admission.json'),'--ready-fd',str(readfd)]
        admission=runtime.admission(plan,plan_ref,review_ref,owner,lock,fd,argv,started,first_ref,proof.duplicate_reference)
        admission.update(work_kind=inputs.WORK_KIND,metadata_input_binding=metadata_binding)
        admission_ref=c.save(out/'admission.json',admission)
        proof.admission_ref=admission_ref
        argv.extend(['--admission-sha256',admission_ref['sha256']]);admission_command=argv[:]
        env=dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
        c.verify_loaded(plan['source_sha256'])
        with (out/'worker.log').open('xb')as log:
            child=subprocess.Popen(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(readfd,))
            os.close(readfd);readfd=None;deadline=time.monotonic()+20
            while True:
                guard();actual=p.proc(child.pid)
                if actual is not None and actual['argv']==admission_command:
                    c.require(actual['ppid']==owner['pid']and int(actual['start_ticks'])>=int(owner['start_ticks']),'Exact live child before ACK');child_owner=actual;break
                c.require(child.poll()is None and time.monotonic()<deadline,'Child startup deadline');time.sleep(.05)
            c.save(out/'child-owner.json',dict(owner=child_owner,command=argv,admission=admission_ref));os.write(writefd,b'R');os.close(writefd);writefd=None
            while child.poll()is None:
                guard();c.require(os.fstat(log.fileno()).st_size<=budget['log_bytes'],'Bounded native log');time.sleep(.5)
            c.require(os.fstat(log.fileno()).st_size<=budget['log_bytes'],'Final native log cap')
            code=child.returncode;drain(child,child_owner,p)
            c.save(out/'child-exit.json',dict(returncode=code,owner=child_owner,parent_observed=True,entire_process_group_empty=True))
            child=None;c.require(code==0,'Successful actual child exit')
        worker_result=c.reference(out/'worker-result.json')[0]
        result=dict(stage=plan['stage'],worker_result=worker_result,child_started=True,child_actual_exit_code=code,child_group_empty=True,lock_policy=runtime.POLICY,retained_fd=fd,first_lock_proof=first_ref)
        guard();c.require(storage_usage(out)<=budget['output_bytes'],'Final allocated stage output cap');c.verify_loaded(plan['source_sha256']);result.update(schema='dory_ubuntu_metadata_parent_result_v1',work_kind=inputs.WORK_KIND,metadata_input_binding=metadata_binding,native_ABI_qualified=False,owner=owner,lock=lock,plan=plan_ref,complete=True,actual_process_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        result['descriptor_guard']=proof.summary()
        result_ref=save_final_result(out,result,budget);print(json.dumps(result_ref),flush=True)
    except BaseException as error:
        if not created and proof.failed is not error:
            runtime.preoutput_failure(error,owner,lock,fd,plan_ref,review_ref,proof.pending_evidence())
        if created:
            try:c.save(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),owner=owner,child_owner=child_owner,command_cleanup_errors=getattr(error,'dory_command_cleanup_errors',[]),actual_process_exit_claimed=False))
            except BaseException as report_error:print(json.dumps(dict(original_error=str(error),failure_record_error=str(report_error))),file=sys.stderr,flush=True)
        raise
    finally:
        for descriptor in (readfd,writefd):
            if descriptor is not None:
                try:os.close(descriptor)
                except OSError:pass
        for sig in handlers:signal.signal(sig,lambda *_:None)
        release_after_drain(child,child_owner,p,external,fd,out if created else None)
        for sig,handler in handlers.items():signal.signal(sig,handler)


def worker_main(argv=None):
    parser=argparse.ArgumentParser()
    for name in ('plan','plan-sha256','admission','admission-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--ready-fd',required=True,type=int);a=parser.parse_args(argv)
    c.require(sys.dont_write_bytecode and sys.flags.no_user_site and __debug__,'Fresh child -B -s with assertions')
    plan_ref=dict(path=a.plan,sha256=a.plan_sha256);plan=c.load(plan_ref)
    admission=c.load(dict(path=a.admission,sha256=a.admission_sha256))
    review=c.load(admission['review']);inputs.validate(plan,review,plan_ref)
    c.require(admission['first_lock_proof']['path']==str(Path(plan['output_root'])/'initial-lock-proof.json'),'Owned first proof metadata')
    duplicate_ref=admission['parent_first_duplicate'];parent_duplicate=None
    if duplicate_ref is not None and duplicate_ref!=admission['first_lock_proof']:
        c.require(duplicate_ref['path']==str(Path(plan['output_root'])/'first-descriptor-duplicate-parent.json'),'Owned parent duplicate metadata')
        parent_duplicate=c.load(duplicate_ref,runtime.CAP)
    runtime.validate_admission(admission,plan,plan_ref,c.load(admission['first_lock_proof'],runtime.CAP),parent_duplicate)
    c.verify_sources(plan['source_sha256'])
    import minicpm5_layer0_execution_protocol_v4 as p
    c.verify_loaded(plan['source_sha256']);parent=admission['owner'];lock=admission['lock']
    c.require(p.proc(os.getpid())['argv']==admission['command']+['--admission-sha256',a.admission_sha256], 'Exact live child command including interpreter flags')
    c.require(os.getppid()==parent['pid'],'Original parent still owns child')
    ready,_,_=select.select([a.ready_fd],[],[],20)
    c.require(ready and os.read(a.ready_fd,1)==b'R','Actual parent live-owner ACK');os.close(a.ready_fd)
    out=Path(plan['output_root']);budget=plan['budget'];started=admission['started_monotonic'];last=[0.]
    c.verify_loaded(plan['source_sha256'])
    proof=runtime.Guard(p,parent,lock,admission['retained_fd'],out,plan_ref,admission['review'],budget,role='worker',admission_ref=dict(path=a.admission,sha256=a.admission_sha256))
    def guard():
        proof();p.clear_window();c.require(os.getppid()==parent['pid']and time.monotonic()-started<budget['wall_seconds'],'Original admitted parent/wall bound')
        if time.monotonic()-last[0]>=1:
            for path,floor in ((ROOT,28*c.GIB),(Path('/mnt/nvme'),30*c.GIB)):
                fs=os.statvfs(path);c.require(fs.f_bavail*fs.f_frsize>=floor,'Original child storage floor')
            c.require(inputs.storage_usage(out)<=budget['output_bytes'],'Complete allocated metadata output cap')
            last[0]=time.monotonic()
    os.sched_setaffinity(0,{5})
    resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
    resource.setrlimit(resource.RLIMIT_FSIZE,(budget['single_file_bytes'],)*2)
    journal=None
    try:
        guard()
        c.require(sys.executable=='/usr/bin/python3','Inventory system interpreter only')
        from minicpm5_sdk310_snapshot_read_binding_v1 import run
        from dory_serial_stage_journal_v2 import Journal
        c.verify_loaded(plan['source_sha256'])
        metadata_binding=inputs.authenticate(plan,guard);inputs.validate_admission(admission,metadata_binding)
        journal=Journal(out/'events.jsonl.gz',guard)
        value=run(plan,guard,journal.emit,inputs.TRUST,out/'packages',plan['archive_mode'],plan.get('replay_from'))
        result_ref=inputs.save_collected(out,value,metadata_binding,plan_ref,dict(path=a.admission,sha256=a.admission_sha256),guard,plan=plan)
        journal.close();events=journal.verify()
        summary=dict(stage='inventory',work_kind=inputs.WORK_KIND,metadata_result=result_ref,input_binding=metadata_binding,events=events)
        guard();c.verify_loaded(plan['source_sha256'])
        summary.update(schema='dory_ubuntu_metadata_worker_result_v1',native_ABI_qualified=False,complete=True,actual_process_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        summary.update(lock_policy=runtime.POLICY,retained_fd=admission['retained_fd'],descriptor_guard=proof.summary())
        c.save(out/'worker-result.json',summary)
        c.require(inputs.storage_usage(out)<=budget['output_bytes'],'Final worker allocated output cap')
    except BaseException as error:
        close_error=None
        if journal:
            try:journal.close()
            except BaseException as secondary:close_error=repr(secondary)
        try:c.save(out/'worker-failure.json',dict(error_type=type(error).__name__,error=str(error),journal_durable=journal.durable if journal else None,journal_close_error=close_error,command_cleanup_errors=getattr(error,'dory_command_cleanup_errors',[]),actual_process_exit_claimed=False))
        except BaseException as secondary:print(json.dumps(dict(original_error=str(error),report_error=str(secondary))),file=sys.stderr,flush=True)
        raise


def recorder_main(argv=None):
    cli=argparse.ArgumentParser()
    for k in ('plan','plan-sha256','review','review-sha256','chunk','destination'):cli.add_argument('--'+k,required=True)
    cli.add_argument('--session',required=True,type=int);cli.add_argument('--exit-code',required=True,type=int);a=cli.parse_args(argv)
    c.require(a.exit_code in (0,1)and a.session>0 and sys.dont_write_bytecode,'Root supplies actual zero/one terminal and system -B')
    pr=dict(path=a.plan,sha256=a.plan_sha256);rr=dict(path=a.review,sha256=a.review_sha256);plan=inputs.validate(c.load(pr),c.load(rr),pr);c.verify_sources(plan['source_sha256'])
    import minicpm5_layer0_execution_protocol_v4 as p
    p.user_bus();os.sched_setaffinity(0,{5});os.nice(19)
    for limit,key in ((resource.RLIMIT_AS,'address_space_bytes'),(resource.RLIMIT_CPU,'cpu_seconds'),(resource.RLIMIT_FSIZE,'single_file_bytes')):resource.setrlimit(limit,(c.INVENTORY[key],)*2)
    out=Path(plan['output_root']);owner=c.load(c.reference(out/'owner.json')[0]);child=c.load(c.reference(out/'child-owner.json')[0]);parent=owner['owner'];co=child['owner']
    p.gone(parent);p.gone(co);c.require(not life.group_members(co['pid']),'Actual complete group gone')
    fd,recorder,lock,first=runtime.acquire(p,pr,rr);started=time.monotonic();last=[0.]
    def guard():
        c.require(time.monotonic()-started<900,'Original recorder wall limit')
        if time.monotonic()-last[0]>=.5:
            runtime.observe(p,recorder,lock,fd);p.gone(parent);p.gone(co);p.active_heavy()
            c.require(life.free(Q.parent)>=28*c.GIB and life.free('/mnt/nvme')>=30*c.GIB,'Original disk floors');last[0]=time.monotonic()
    try:
        guard();authenticated=inputs.authenticate(plan,guard);evidence={};values={}
        names={'owner.json','initial-lock-proof.json','admission.json','child-owner.json','child-exit.json','worker.log','events.jsonl.gz','result.json','worker-result.json','metadata-result.json','failure.json','worker-failure.json','first-descriptor-duplicate-parent.json','first-descriptor-duplicate-worker.json','descriptor-lock-failure-parent.json','descriptor-lock-failure-worker.json'}
        n,_=inputs.authenticated(plan,guard);nested={'sources-result.json'}|{x['package']+'.deb'for x in n['packages']}
        for path in sorted(out.rglob('*')):
            guard();c.require(not path.is_symlink(),'No output links')
            if path.is_dir():c.require(path==out/'packages','Only owned archive directory');continue
            name=str(path.relative_to(out));c.require(name in names or path.parent==out/'packages'and path.name in nested,'Finite output inventory')
            if path.stat().st_size==0:
                evidence[name]=dict(path=str(path),bytes=0,sha256=hashlib.sha256(b'').hexdigest());continue
            ref,_=io.file_ref(path,guard,cap=64*c.MIB);evidence[name]=ref
            if path.suffix=='.json':values[name]=c.load(ref,8*c.MIB)
        c.require(inputs.storage_usage(out)<=c.INVENTORY['output_bytes'],'Complete output size cap')
        admission=values['admission.json'];inputs.validate_admission(admission,authenticated)
        c.require(owner['plan']==pr and owner['review']==rr and owner['lock']==lock and child['command']==co['argv']==admission['command']+['--admission-sha256',evidence['admission.json']['sha256']],'Original source/owner/command/admission binding')
        c.require(owner['owner']==admission['owner']==parent and values['child-exit.json']['owner']==co and values['child-exit.json']['entire_process_group_empty'],'Saved child actual exit/drain binding')
        retirement_verification=retirement.verify_recorded_history(plan,owner['history'],p)
        events=[];rawhash=hashlib.sha256();rawbytes=0
        with gzip.open(out/'events.jsonl.gz','rb')as f:
            while True:
                guard();line=f.readline(8*c.MIB+1)
                if not line:break
                rawbytes+=len(line);c.require(rawbytes<=64*c.MIB and len(line)<=8*c.MIB and line.endswith(b'\n'),'Complete bounded events')
                rawhash.update(line);events.append(json.loads(line));c.require(len(events)<=event_protocol.MAX_EVENTS,'Finite event count')
        if a.exit_code==0:
            c.require(not any(x in values for x in ('failure.json','worker-failure.json'))and values['child-exit.json']['returncode']==0,'Successful producer and child without failure')
            result=values['result.json'];worker=values['worker-result.json'];receipt=values['metadata-result.json'];pkg=values['packages/sources-result.json']
            c.require(result['complete']and result['plan']==pr and result['worker_result']==evidence['worker-result.json']and worker['metadata_result']==evidence['metadata-result.json']and receipt['sources_result']==evidence['packages/sources-result.json'],'Whole receipt chain')
            inputs.validate_result(pkg,n,plan['archive_mode'],plan.get('replay_from'))
            event_protocol.validate_success(events,n['packages'],plan['archive_mode'],pkg,evidence['packages/sources-result.json'])
            ev=worker['events'];c.require(ev['raw_bytes']==rawbytes and ev['raw_sha256']==rawhash.hexdigest()and ev['events']==len(events)and all(ev[k]==evidence['events.jsonl.gz'][k]for k in ('path','bytes','sha256')),'Full durable journal matches receipt')
            for row,saved in zip(n['packages'],pkg['packages']):
                c.require(row['package']==saved['package']and row['version']==saved['version'],'Ordered exact package versions')
                if plan['archive_mode']=='download':c.require(saved['inspection']['archive']==evidence['packages/'+row['package']+'.deb'],'Full stored archive hash tied to result')
        else:c.require('failure.json'in values,'Actual failure retained')
        c.verify_sources(plan['source_sha256'],guard);guard()
        value=dict(schema='minicpm5_sdk310_deb_stage_actual_exit_v1',retirement_verification=retirement_verification,actual_tool=dict(kind='tool_session',session_id=a.session,terminal_chunk=a.chunk,exit_code=a.exit_code),plan=pr,review=rr,output_root=str(out),owner=parent,child_owner=co,owners_gone=True,child_group_empty=True,stage_success_observed=a.exit_code==0,mode=plan['archive_mode'],evidence=evidence,events=len(events),raw_event_bytes=rawbytes,raw_event_sha256=rawhash.hexdigest(),result=evidence.get('packages/sources-result.json'),recorder=recorder,lock_identity=lock,recorder_self_exit_claimed=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        print(json.dumps(c.save(Path(a.destination),value,8*c.MIB)),flush=True)
    finally:os.close(fd)


def main(argv=None):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode',required=True,choices=('parent','worker','record'))
    args,remaining=parser.parse_known_args(argv)
    c.require(sys.dont_write_bytecode and __debug__, 'Explicit -B and unoptimized interpreter required')
    return {'parent':parent_main,'worker':worker_main,'record':recorder_main}[args.mode](remaining)


if __name__=='__main__':main()
