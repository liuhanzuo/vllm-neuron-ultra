"""System package metadata-only successor; unchanged descriptor/lifecycle helpers.

No native/ABI/stage-A branches or automatic successor. Fresh runtime admission
and actual external exit verification remain root responsibilities.
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
Q=Path(__file__).resolve().parent;ROOT=Q.parent;sys.path[:0]=[str(Q),str(ROOT)]
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_system_package_input_binding_v1 as inputs
from run_dory_serial_stage_root_v5 import free,memory_free,drain,release_after_drain,storage_usage,save_final_result,preflight_history


def main():
    parser=argparse.ArgumentParser()
    for name in ('plan','plan-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();c.require(sys.dont_write_bytecode and __debug__,'Original python -B assertions')
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
        metadata_binding=inputs.authenticate(guard)
        c.require(initial>=30*c.GIB+budget['output_bytes'],'Full fresh output reservation above original floor')
        os.sched_setaffinity(0,{5});os.nice(19)
        # Parent never loads a model. Child has independently reviewed limits.
        resource.setrlimit(resource.RLIMIT_AS,(budget['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(budget['cpu_seconds'],budget['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE,(budget['single_file_bytes'],)*2)
        out.mkdir(mode=0o700);created=True
        first_ref=c.save(out/'initial-lock-proof.json',runtime.initial_record(first_proof,owner,lock,fd,plan_ref,review_ref),runtime.CAP)
        proof.initial_persisted(first_ref)
        c.save(out/'owner.json',dict(owner=owner,lock=lock,retained_fd=fd,lock_policy=runtime.POLICY,first_lock_proof=first_ref,plan=plan_ref,review=review_ref,history=history,work_kind=inputs.WORK_KIND,metadata_input_binding=metadata_binding))
        worker=str(Q/'dory_system_package_metadata_worker_v1.py');c.require(worker in plan['source_sha256'],'Worker source reviewed')
        readfd,writefd=os.pipe();python='/usr/bin/python3'
        argv=[python,'-B','-s',worker,'--plan',a.plan,'--plan-sha256',a.plan_sha256,'--admission',str(out/'admission.json'),'--ready-fd',str(readfd)]
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
        guard();c.require(storage_usage(out)<=budget['output_bytes'],'Final allocated stage output cap');c.verify_loaded(plan['source_sha256']);result.update(schema='dory_system_package_metadata_parent_result_v1',work_kind=inputs.WORK_KIND,metadata_input_binding=metadata_binding,native_ABI_qualified=False,owner=owner,lock=lock,plan=plan_ref,complete=True,actual_process_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
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


if __name__=='__main__':main()
