"""One original-lock discovery stage; two sequential original Python children."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(ROOT),str(Q)]
from research import minicpm5_bootstrap_observation_v1 as core
import dory_serial_stage_contract_v2 as c
import dory_descriptor_inventory_runtime_v2 as runtime
import minicpm5_sdk310_retired_lfm_r2_v2 as retirement
import run_dory_serial_stage_root_v5 as life
import prepare_minicpm5_sdk310_deb_plan_v4 as history_builder


def inventory(out):
    raw=metadata=count=allocated=0
    for path in out.rglob('*'):
        info=path.lstat();core.require(not path.is_symlink(),'Owned outputs have no links')
        if path.is_dir():continue
        count+=1;allocated+=max(info.st_size,info.st_blocks*512)
        core.require(info.st_size<=core.CAPS['one_file_bytes'],'Original per-file limit')
        if path.suffix in ('.maps','.log'):raw+=info.st_size
        else:metadata+=info.st_size
    core.require(count<=core.CAPS['raw_files'] and allocated<=core.CAPS['allocated_bytes']
      and raw<=core.CAPS['raw_bytes'] and metadata<=core.CAPS['metadata_bytes'],'All original aggregate output caps')
    return dict(files=count,allocated_bytes=allocated,metadata_bytes=metadata,raw_bytes=raw)


def history(plan,p,read=c.load):
    catalog=read(plan['complete_current_serial_catalog'])
    core.require(catalog['reviewer']=='/root' and catalog['passed'] is True and catalog['blockers']==[]
      and catalog['all_later_serial_runs_included'] is True and catalog['current_handoff']==plan['root_handoff']
      and core.ref2(catalog['runs'][-1]['terminal'])==core.ref2(plan['predecessor_exit']), 'Latest complete actual history')
    # Reconstruct the complete existing history rather than invent a new
    # normalization for old actual-tool shapes. The existing builder retains
    # its reviewed121-row prefix and validates all actual later terminals.
    rebuilt=history_builder.build(read,plan['complete_current_serial_catalog'],
       plan['r2_predecessor_scope'],plan['predecessor_exit'],'download',plan['output_root'],
       retirement_acceptance_ref=plan['source_only_acceptance'])
    for key in ('predecessor_owners','predecessor_services','predecessor_expected_exit_code',
       'root_handoff','physical_history_source','r2_predecessor_scope','r2_actual_exit','r2_full_audit',
       'retired_lfm_r2','retired_mini_history','lock_policy','mini_scientific_constraints'):
        core.require(plan[key]==rebuilt[key],'Complete inherited history unchanged: '+key)
    # Existing complete retirement proofs, all 874 extra r2 children and stable
    # live service observations stay in the original implementation.
    return retirement.preflight_history(plan,p,read)


def save_history(out,rows,save=c.save,guard=lambda:None):
    """Retain every original observation while keeping each file below2MiB."""
    parts=[];pending=[];size=0;first=0
    def flush():
        nonlocal pending,size,first
        if not pending:return
        guard();value=dict(schema='minicpm5_bootstrap_history_part_v1',first_row=first,rows=pending)
        ref=save(out/('history-%04d.json'%len(parts)),value,2**20)
        parts.append(dict(reference=ref,first_row=first,row_count=len(pending)))
        first+=len(pending);pending=[];size=0
    for row in rows:
        length=len(core.encoded(row))+1
        core.require(length+1024<=2**20,'One complete unchanged history row fits shard')
        if size+length+1024>2**20:flush()
        pending.append(row);size+=length
    flush();core.require(first==len(rows),'All original history observations persisted')
    guard()
    return save(out/'history-manifest.json',dict(schema='minicpm5_bootstrap_history_manifest_v1',
        row_count=first,parts=parts,complete_unchanged_rows=True),2**20)


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','plan-sha256','review','review-sha256'):cli.add_argument('--'+name,required=True)
    a=cli.parse_args(argv)
    core.require(sys.executable=='/usr/bin/python3' and sys.dont_write_bytecode
      and sys.flags.no_user_site and __debug__ and sys.pycache_prefix=='/dev/null',
      'System parent -B -s -X pycache_prefix=/dev/null with assertions')
    pr=dict(path=a.plan,sha256=a.plan_sha256);rr=dict(path=a.review,sha256=a.review_sha256)
    plan=core.validate(c.load(pr,8*2**20),c.load(rr,8*2**20),pr)
    core.require(plan['lock_policy']==runtime.POLICY,'Original exact descriptor policy')
    core.verify_sources(plan['source_sha256'],c.reference)
    import minicpm5_layer0_execution_protocol_v4 as p
    import minicpm5_layer0_external_supervisor_v4 as external
    from research import minicpm5_sdk_mapping_native_authenticator_v2 as native
    from research import minicpm5_sdk_mapping_native_closure_v1 as closure
    c.verify_loaded(plan['source_sha256']);p.user_bus()
    out=Path(plan['output_root']);core.require(not out.exists(),'Fresh output; no retry overwrite')
    fd,owner,lock,first=runtime.acquire(p,pr,rr)
    started=time.monotonic();deadline=started+900
    created=False;child=child_owner=None;readfd=writefd=None;handlers={};results=[]
    proof=runtime.Guard(p,owner,lock,fd,out,pr,rr,core.BUDGET,role='parent',created=lambda:created,first_observation=first)
    last=[0.0];content_reader=None
    def guard():
        proof();core.require(time.monotonic()<deadline,'Same absolute discovery deadline')
        if time.monotonic()-last[0]>=.5:
            p.clear_window();p.active_heavy()
            for previous in plan['predecessor_owners']:p.gone(previous)
            core.require(life.free(ROOT)>=core.BUDGET['root_floor_bytes']
              and life.free('/mnt/nvme')>=core.BUDGET['nvme_floor_bytes']
              and life.memory_free()>=core.BUDGET['memory_available_floor_bytes'],'Original live resource floors')
            if created:inventory(out)
            last[0]=time.monotonic()
    def stopped(*args):raise TimeoutError('Discovery interrupted or original900s expired')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):handlers[sig]=signal.signal(sig,stopped)
        core.require(signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0),'No inherited timer replaced')
        signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
        os.sched_setaffinity(0,{6});priority=os.getpriority(os.PRIO_PROCESS,0)
        if priority<19:os.nice(19-priority)
        resource.setrlimit(resource.RLIMIT_AS,(2147483648,)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(600,605))
        resource.setrlimit(resource.RLIMIT_FSIZE,(2097152,)*2)
        guard();saved_history=history(plan,p)
        core.require(life.free('/mnt/nvme')>=core.BUDGET['nvme_floor_bytes']+core.CAPS['allocated_bytes'],'Full original output reserve')
        out.mkdir(mode=0o700);created=True
        first_ref=c.save(out/'initial-lock-proof.json',runtime.initial_record(first,owner,lock,fd,pr,rr),2097152)
        proof.initial_persisted(first_ref)
        history_ref=save_history(out,saved_history,guard=guard)
        c.save(out/'owner.json',dict(owner=owner,lock_identity=lock,retained_fd=fd,plan=pr,review=rr,
          history=history_ref,started_monotonic=started,deadline=deadline),2097152)
        table=core.candidates(c.load)
        content_reader=native.StableReader(guard)
        content=core.collect_content(table,content_reader,closure,guard)
        content_ref=c.save(out/'candidate-content.json',content,2097152)
        literal_sources=[]
        for name in (core.HELPER,core.CHILD):
            reference,raw=c.reference(Q/name,2*2**20)
            core.require(reference['sha256']==plan['source_sha256'][str(Q/name)],'Exact executed literal component')
            literal_sources.append(raw.decode('utf8'))
        source=core.literal(*literal_sources)
        core.verify_sources(plan['source_sha256'],c.reference,guard)
        for role in core.ROLES:
            guard();core.require(child is None,'Strict SDK then outer, never overlapping children')
            role_out=out/role;role_out.mkdir(mode=0o700)
            readfd,writefd=os.pipe();admission_path=str(role_out/'admission.json')
            template=core.argv_for(role,table,source,pr,dict(path=admission_path,sha256='admission:self'),readfd,deadline)
            admission=dict(schema='minicpm5_bootstrap_observation_admission_v1',role=role,plan=pr,review=rr,
              parent_owner=owner,lock_identity=lock,retained_fd=fd,first_lock_proof=first_ref,content=content_ref,
              started_monotonic=started,deadline=deadline,command=template,output_root=str(role_out),
              literal_sha256=hashlib.sha256(source.encode()).hexdigest(),accepted_native_authority_created=False)
            ar=c.save(role_out/'admission.json',admission,2097152)
            command=core.argv_for(role,table,source,pr,ar,readfd,deadline)
            env=dict(PATH='/usr/bin:/bin',HOME='/home/ubuntu',LANG='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',
              PYTHONNOUSERSITE='1',PYTHONOPTIMIZE='0',XDG_RUNTIME_DIR='/run/user/1000',
              DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
            with (role_out/'worker.log').open('xb') as log:
                child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                  start_new_session=True,close_fds=True,pass_fds=(readfd,))
                os.close(readfd);readfd=None;ack_deadline=min(deadline,time.monotonic()+20)
                while True:
                    guard();actual=p.proc(child.pid)
                    if actual is not None and actual['argv']==command:
                        core.require(actual['ppid']==owner['pid'] and int(actual['start_ticks'])>=int(owner['start_ticks']), 'Exact current child owner')
                        child_owner=actual;break
                    core.require(child.poll() is None and time.monotonic()<ack_deadline,'Bounded child exec ACK');time.sleep(.02)
                c.save(role_out/'owner.json',dict(owner=child_owner,admission=ar,command=command),2097152)
                core.require(os.write(writefd,b'R')==1,'Whole actual ACK');os.close(writefd);writefd=None
                while child.poll() is None:
                    guard();core.require(os.fstat(log.fileno()).st_size<=core.BUDGET['log_bytes'],'Bounded child log');time.sleep(.2)
                code=child.returncode;life.drain(child,child_owner,p)
                c.save(role_out/'exit.json',dict(returncode=code,owner=child_owner,parent_observed=True,
                  entire_process_group_empty=True,actual_self_exit_claimed=False),2097152)
                child=child_owner=None
                core.require(code==0,'Actual discovery child completed; failures remain failures')
            content_reader.unchanged(content['files'])
            result_ref=c.reference(role_out/'observation.json',2097152)[0]
            result=c.load(result_ref,2097152)
            core.require(result['complete'] is True and result['role']==role and result['deadline']==deadline
              and result['admission']==core.ref2(ar) and result['accepted_native_authority_created'] is False,
              'Exact own child observation without authority promotion')
            results.append(dict(role=role,result=result_ref,missing_content_origins=result['missing_content_origins'],
              ffi_definitions_observed=result['ffi_definitions'] is not None))
        guard();c.verify_loaded(plan['source_sha256']);content_reader.unchanged(content['files'])
        result=dict(schema='minicpm5_bootstrap_observation_result_v1',complete=True,plan=pr,review=rr,owner=owner,
          deadline=deadline,candidate_content=content_ref,children=results,role_order=list(core.ROLES),
          output_inventory=inventory(out),accepted_native_authority_created=False,actual_process_exit_claimed=False,
          native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False)
        ref=c.save(out/'result.json',result,2097152);inventory(out);print(json.dumps(ref),flush=True)
    except BaseException as error:
        if created:
            try:c.save(out/'failure.json',dict(error_type=type(error).__name__,error=str(error),owner=owner,
                child_owner=child_owner,completed_children=results,accepted_native_authority_created=False),2097152)
            except BaseException as reporting:print(json.dumps(dict(error=str(error),failure_record_error=str(reporting))),file=sys.stderr,flush=True)
        else:runtime.preoutput_failure(error,owner,lock,fd,pr,rr,proof.pending_evidence())
        raise
    finally:
        for pipe in (readfd,writefd):
            if pipe is not None:
                try:os.close(pipe)
                except OSError:pass
        for sig in handlers:signal.signal(sig,lambda *_:None)
        signal.setitimer(signal.ITIMER_REAL,0)
        # Original release policy waits for full physical drain even on failure.
        life.release_after_drain(child,child_owner,p,external,fd,out if created else None)
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':main()
