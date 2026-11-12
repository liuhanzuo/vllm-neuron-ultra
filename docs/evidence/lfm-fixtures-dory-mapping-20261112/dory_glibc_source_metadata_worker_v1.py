"""Exact glibc source metadata-only worker; original parent fd is never inherited."""
import argparse
import json
import os
from pathlib import Path
import resource
import select
import sys
import time
Q=Path(__file__).resolve().parent;ROOT=Q.parent;sys.path[:0]=[str(Q),str(ROOT)]
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import dory_glibc_source_metadata_binding_v1 as inputs


def main():
    parser=argparse.ArgumentParser()
    for name in ('plan','plan-sha256','admission','admission-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--ready-fd',required=True,type=int);a=parser.parse_args()
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
        from dory_glibc_source_metadata_runtime_v1 import run
        from dory_serial_stage_journal_v2 import Journal
        c.verify_loaded(plan['source_sha256'])
        metadata_binding=inputs.authenticate(guard);inputs.validate_admission(admission,metadata_binding)
        journal=Journal(out/'events.jsonl.gz',guard)
        value=run(guard,journal.emit,out/'glibc-source-metadata')
        result_ref=inputs.save_collected(out,value,metadata_binding,plan_ref,dict(path=a.admission,sha256=a.admission_sha256),guard)
        journal.close();events=journal.verify()
        summary=dict(stage='inventory',work_kind=inputs.WORK_KIND,metadata_result=result_ref,input_binding=metadata_binding,events=events)
        guard();c.verify_loaded(plan['source_sha256'])
        summary.update(schema='dory_glibc_source_metadata_worker_result_v1',native_ABI_qualified=False,complete=True,actual_process_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False)
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


if __name__=='__main__':main()
