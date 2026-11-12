"""One root-reviewed Dory stage per invocation. No automatic successor.

Inventory runs only stdlib/source metadata work. ABI and Stage A use a fresh
original interpreter child after parent-side native authority validation. Parent
holds the original serial lock until the actual child and its group are gone.
The root must still record this launcher's real external tool exit separately.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(Q),str(ROOT)]
import dory_serial_stage_contract_v2 as c


def free(path):
    fs=os.statvfs(path);return fs.f_bavail*fs.f_frsize


def memory_free():
    with Path('/proc/meminfo').open()as stream:raw=stream.read(65537)
    c.require(len(raw)<=65536,'Bounded memory metadata')
    fields=dict(line.split(':',1)for line in raw.splitlines());return int(fields['MemAvailable'].split()[0])*1024


def group_members(pgid):
    # Same owned-process-group drain criterion as run_dory_layer0_serial_v2,
    # bounded by process count rather than scanning any filesystem payload.
    found=[];count=0
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        count+=1;c.require(count<=100000,'Bounded process-group membership enumeration')
        try:
            with (directory/'stat').open()as stream:raw=stream.read(65537)
        except FileNotFoundError:continue
        c.require(len(raw)<=65536,'Bounded process stat');fields=raw.rsplit(')',1)[1].split()
        if int(fields[2])==pgid and fields[0]not in ('Z','X'):
            found.append(dict(pid=int(directory.name),start_ticks=fields[19],state=fields[0]))
    return found


def drain(child,owner,p):
    for sig,seconds in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        if not group_members(child.pid):break
        current=p.proc(child.pid)
        c.require(current is None or owner is not None and current['start_ticks']==owner['start_ticks'],'Never signal reused process-group leader')
        try:os.killpg(child.pid,sig)
        except ProcessLookupError:pass
        deadline=time.monotonic()+seconds
        while group_members(child.pid)and time.monotonic()<deadline:time.sleep(.1)
    child.wait(timeout=10);c.require(not group_members(child.pid),'Actual complete owned child drain')


def release_after_drain(child,child_owner,p,external,fd,out):
    # Never release the original lock merely because cleanup reached a time
    # limit. Retain it while the root resolves a genuinely undrained group.
    if child is not None:
        warned=False
        while True:
            try:drain(child,child_owner,p);break
            except BaseException as error:
                if out is not None and not warned:
                    try:c.save(out/'cleanup-pending.json',dict(error=str(error),child_owner=child_owner,lock_retained=True))
                    except BaseException:pass
                    warned=True
                time.sleep(.5)
        if out is not None and not (out/'child-exit.json').exists():
            try:c.save(out/'child-exit.json',dict(returncode=child.returncode,owner=child_owner,parent_observed=True,entire_process_group_empty=True,terminated_after_parent_failure=True))
            except BaseException as reporting:print(json.dumps(dict(child_returncode=child.returncode,child_exit_record_error=str(reporting))),file=sys.stderr,flush=True)
    external.release_lock(fd)


def storage_usage(out):
    count=0;total=0;pending=[out]
    while pending:
        parent=pending.pop()
        for path in parent.iterdir():
            count+=1;c.require(count<=2000,'Bounded output names');info=path.lstat()
            c.require(not path.is_symlink(),'No output links')
            if path.is_dir():pending.append(path)
            else:total+=max(info.st_size,info.st_blocks*512)
    return total


def save_final_result(out,result,budget):
    reference=c.save(out/'result.json',result)
    c.require(storage_usage(out)<=budget['output_bytes'],'Final result included in allocated file budget')
    return reference


def preflight_history(plan,p):
    c.check_predecessor(c.load(plan['predecessor_exit']),plan['predecessor_expected_exit_code'])
    for owner in plan['predecessor_owners']:p.gone(owner)
    rows=[]
    for saved in plan['predecessor_services']:
        samples=[]
        for _ in range(2):
            value=p.properties(saved['unit']);c.require(p.terminal(value),'Original prior service terminal/no Job')
            if saved.get('invocation_id'):c.require(value['InvocationID']==saved['invocation_id'],'Known original invocation')
            group=p.empty_group(p.GROUP_PREFIX+saved['unit']+'.service');samples.append(dict(properties=value,group=group))
        c.require(samples[0]==samples[1],'Two stable prior terminal observations');rows.append(dict(unit=saved['unit'],samples=samples))
    return rows


def main():
    parser=argparse.ArgumentParser()
    for name in ('plan','plan-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();c.require(sys.dont_write_bytecode and __debug__,'Original python -B assertions')
    plan_ref=dict(path=a.plan,sha256=a.plan_sha256);review_ref=dict(path=a.review,sha256=a.review_sha256)
    plan=c.validate(c.load(plan_ref),c.load(review_ref),plan_ref);c.verify_sources(plan['source_sha256'])
    c.require(str(Path(__file__).resolve())in plan['source_sha256'],'Launcher itself reviewed')
    import minicpm5_layer0_execution_protocol_v4 as p
    import minicpm5_layer0_external_supervisor_v4 as external
    c.verify_loaded(plan['source_sha256']);p.user_bus()
    out=Path(plan['output_root']);c.require(not out.exists(),'Fresh stage output; preserve all prior evidence')
    fd,owner,lock=external.acquire_lock();started=time.monotonic();last=[0.];child=None;child_owner=None;readfd=writefd=None;handlers={}
    initial=None;budget=plan['budget'];created=False
    def guard():
        p.lock_proof(owner,lock);p.clear_window();c.require(time.monotonic()-started<budget['wall_seconds'],'Stage wall bound')
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
        c.require(initial>=30*c.GIB+budget['output_bytes'],'Full fresh output reservation above original floor')
        os.sched_setaffinity(0,{5});os.nice(19)
        # Parent never loads a model. Child has independently reviewed limits.
        resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(budget['single_file_bytes'],)*2)
        out.mkdir(mode=0o700);created=True;c.save(out/'owner.json',dict(owner=owner,lock=lock,plan=plan_ref,review=review_ref,history=history))
        decision=None
        if plan['stage']in ('abi','stage_a'):
            decision=c.native_authority(plan,guard)
        if plan['stage']=='stage_a':
            accepted=c.load(plan['abi_acceptance']);c.require(accepted['schema']=='dory_abi_root_acceptance_v1'and accepted['passed']is True and accepted['native_authority']==plan['native_authority'],'Actual ABI root acceptance')
            c.check_predecessor(c.load(accepted['actual_exit']),0)
        if plan['stage']=='review':
            producer=c.load(plan['producer_acceptance']);c.require(producer['schema']=='dory_stage_a_root_producer_acceptance_v1'and producer['passed']is True,'Actual producer root acceptance')
            c.check_predecessor(c.load(producer['actual_exit']),0)
        worker=str(Q/'dory_serial_stage_worker_v4.py');c.require(worker in plan['source_sha256'],'Worker source reviewed')
        readfd,writefd=os.pipe();python=str(c.ENV/'bin/python')if plan['stage']in ('abi','stage_a')else '/usr/bin/python3'
        argv=[python,'-B','-s',worker,'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--admission',str(out/'admission.json'),'--ready-fd',str(readfd)]
        admission=dict(schema='dory_serial_stage_admission_v1',owner=owner,lock=lock,plan=plan_ref,review=review_ref,stage=plan['stage'],command=argv,started_monotonic=started)
        admission_ref=c.save(out/'admission.json',admission)
        argv.extend(['--admission-sha256',admission_ref['sha256']]);admission_command=argv[:]
        env=dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
        if decision is not None:
            from dory_stage_a_native_resolver_v4 import ENVIRONMENT
            env.update(ENVIRONMENT)
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
        result=dict(stage=plan['stage'],worker_result=worker_result,child_started=True,child_actual_exit_code=code,child_group_empty=True)
        guard();c.require(storage_usage(out)<=budget['output_bytes'],'Final allocated stage output cap');c.verify_loaded(plan['source_sha256']);result.update(schema='dory_serial_stage_result_v1',owner=owner,lock=lock,plan=plan_ref,complete=True,actual_process_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
        result_ref=save_final_result(out,result,budget);print(json.dumps(result_ref),flush=True)
    except BaseException as error:
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


if __name__=='__main__':main()
