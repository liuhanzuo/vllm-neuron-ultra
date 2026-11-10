"""Retained serial qualification controller. No runtime without finalized review."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import stat
import sys
import time
import traceback
import nemotron9b_package_protocol_v24 as p


def acquire():
    fd=os.open(p.low.LOCK,os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);info=os.fstat(fd)
        p.require(stat.S_ISREG(info.st_mode)and info.st_nlink==1,'Whole-file serial lock')
        owner=p.low.proc(os.getpid());lock=dict(device=info.st_dev,inode=info.st_ino)
        p.low.lock_proof(owner,lock);return fd,owner,lock
    except BaseException:os.close(fd);raise


def release(fd):fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def status(spec,stage,performance_window=False,**extra):
    p.save(Path(spec['control_root'])/'status.json',dict(stage=stage,performance_window=performance_window,**extra))


def wait_phase(spec,stage,spec_ref,review_ref,admission):
    holder=[None]
    try:return _wait_phase(spec,stage,spec_ref,review_ref,admission,holder)
    except BaseException:
        if holder[0]is not None:
            from nemotron9b_package_window_v24 import failed_hold
            failed_hold(spec,stage,holder[0]);holder[0]=None
        raise


def _wait_phase(spec,stage,spec_ref,review_ref,admission,holder):
    c=Path(spec['control_root']);p.healthy(spec,admission);p.sources(spec);p.accelerator_idle()
    highwater=dict(root_used=0,nvme_used=0,samples=0,measurement_windows_not_scanned=True)
    for old in p.STAGES:
        if old==stage:break
        p.retained_drain(spec,old,spec_ref,review_ref,p.read(c/(old+'-owner.json')))
    command=p.service_command(spec,stage,spec_ref,review_ref)
    p.once(c/(stage+'-launch.json'),dict(phase=stage,command=command,execution_spec=spec_ref,admission=p.ref(c/'admission.json')))
    p.command(command);deadline=time.monotonic()+30;owner=None
    while owner is None:
        owner,value=p.service_identity(spec,stage,spec_ref,review_ref)
        p.require(not p.low.terminal(value),'Worker exited before owner ACK')
        p.require(time.monotonic()<deadline,'Owner ACK timeout')
        if owner is None:time.sleep(.05)
    p.once(c/(stage+'-owner.json'),owner)
    lease=dict(phase=stage,execution_spec=spec_ref,source_review=review_ref,producer=p.producer(owner),
        controller=admission['controller'],admission=p.ref(c/'admission.json'),runtime_admitted=True)
    p.once(c/(stage+'-lease.json'),lease)
    deadline=time.monotonic()+p.low.duration_seconds(p.resources(stage)['RuntimeMaxUSec'])+180
    while True:
        if p.kind(stage)=='workloads'and(c/(stage+'-measurement-ready.json')).exists()and not(c/(stage+'-measurement-begin.json')).exists():
            p.healthy(spec,admission,stage);p.sources(spec)
            ready=p.read(c/(stage+'-measurement-ready.json'));p.require(ready['owner']==owner and ready['prepared']is True,'Actual premeasurement ready')
            from nemotron9b_package_window_v24 import begin
            holder[0]=begin(spec,stage,owner,admission)
            status(spec,stage,True)
            p.once(c/(stage+'-measurement-begin.json'),dict(owner=owner,allowed=True,observer_barrier=p.ref(c/(stage+'-observer-barrier.json'))))
        timed=p.kind(stage)=='workloads'and(c/(stage+'-measurement-begin.json')).exists()and not(c/(stage+'-measurement-end.json')).exists()
        if holder[0]is not None and(c/(stage+'-measurement-end.json')).exists():
            from nemotron9b_package_window_v24 import finish
            finish(spec,stage,holder[0],owner);holder[0]=None
        status(spec,stage,timed)
        p.healthy(spec,admission,stage,timed=timed)
        if not timed:
            if p.kind(stage)in ('compile','quality','workloads'):
                from nemotron9b_package_server_lifecycle_v24 import controller_observe
                controller_observe(spec,stage,owner)
            space=p.storage(spec,reserve=True);highwater['samples']+=1
            for key in ('root_used','nvme_used'):highwater[key]=max(highwater[key],space[key])
            p.save(c/(stage+'-storage-highwater.json'),highwater)
        seen,value=p.service_identity(spec,stage,spec_ref,review_ref,owner);p.require(seen==owner,'Owner changed')
        if p.low.terminal(value):
            p.require(not timed,'Worker died inside timed window before completion marker')
            break
        p.require(time.monotonic()<deadline,'Phase deadline');time.sleep(1)
    saved=p.snapshot(spec,stage,spec_ref,review_ref,owner);p.once(c/(stage+'-retained-exit.json'),saved)
    p.require(saved['successful'],'Original phase exit failure')
    p.once(c/(stage+'-retained-drain.json'),p.retained_drain(spec,stage,spec_ref,review_ref,owner))
    p.accelerator_idle()
    space=p.storage(spec,reserve=True)
    for key in ('root_used','nvme_used'):highwater[key]=max(highwater[key],space[key])
    highwater['samples']+=1;p.save(c/(stage+'-storage-highwater.json'),highwater)
    result=p.read(c/(stage+'-worker-result.json'))
    p.require(result['complete']is True and result['passed']is True and result['phase']==stage
        and result['producer']==p.producer(owner)and result['lease']==p.ref(c/(stage+'-lease.json')),'Exact worker result')
    p.validate_result(spec,stage,result['result']);p.sources(spec);status(spec,stage+'-complete')


def cleanup_phase(spec,stage,spec_ref,review_ref):
    c=Path(spec['control_root']);launch=c/(stage+'-launch.json');ownerpath=c/(stage+'-owner.json')
    value=p.low.properties(spec['units'][stage]);owner=p.read(ownerpath)if ownerpath.exists()else None
    if launch.exists():p.require(p.read(launch)['command']==p.service_command(spec,stage,spec_ref,review_ref),'Owned durable launch')
    else:p.require(value['LoadState']=='not-found'and owner is None,'Unowned unit')
    if value['LoadState']=='loaded':
        actual,value=p.service_identity(spec,stage,spec_ref,review_ref,owner,observation=value)
        saved=dict(stage=stage,owner=owner or actual,properties=value,successful=False,identity_verified=owner is not None)
    else:saved=dict(stage=stage,owner=owner,properties=value,successful=False,identity_verified=False)
    path=c/(stage+'-cleanup-before-stop.json')
    if path.exists():saved=p.read(path)
    else:p.once(path,saved)
    return p.stop_and_drain(spec,stage,spec_ref,review_ref,saved)


def execute(spec_ref,review_ref):
    p.require(sys.dont_write_bytecode and __debug__,'Python -B without optimization')
    spec,catalog=p.bundle(spec_ref,review_ref);p.low.user_bus();c=Path(spec['control_root'])
    fd,actual,lock=acquire();handlers={};started=[]
    def interrupt(signum,frame):raise RuntimeError('Interrupted; drain while holding serial lock')
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupt)
        owner,_=p.service_identity(spec,'controller',spec_ref,review_ref)
        p.require(owner and p.proc_owner(owner)==actual,'Actual lock-owning service')
        intent=p.read(c/'external-launch-intent.json');deadline=time.monotonic()+60
        while not(c/'external-controller-owner.json').exists():
            p.require(time.monotonic()<deadline,'Independent external owner ACK');time.sleep(.05)
        ack=p.read(c/'external-controller-owner.json')
        p.require(ack['launch_intent']==p.ref(c/'external-launch-intent.json')and ack['owner']==owner,'External actual owner')
        p.require(intent['execution_spec']==spec_ref and intent['source_review']==review_ref and intent['lock_identity']==lock
            and intent['canonical_sha256']==p.canonical(),'External intent')
        p.predecessor_drain(catalog,allowed_units=[spec['units']['controller']],spec=spec);p.low.active_heavy([spec['units']['controller']]);p.storage(spec,reserve=True)
        for phase in p.STAGES:
            value=p.low.properties(spec['units'][phase]);p.require(value['LoadState']=='not-found'and p.low.terminal(value),'Fresh unit')
        for field in ('work_root','evidence_root'):
            path=Path(spec[field]);p.require(not path.exists(),'Fresh output root');path.mkdir(mode=0o700)
        work=Path(spec['work_root'])
        for leaf in ('primitives','temporary/tmp','temporary/xdg','temporary/torch'):(work/leaf).mkdir(parents=True,exist_ok=False)
        admission=dict(execution_spec=spec_ref,source_review=review_ref,controller=owner,lock_identity=lock,
            canonical_sha256=p.canonical(),final_predecessor_catalog=spec['final_predecessor_catalog'],native_authorized=True)
        p.once(c/'admission.json',admission)
        for phase in p.STAGES:
            started.append(phase);wait_phase(spec,phase,spec_ref,review_ref,admission)
        result=p.completion(spec,admission,p.AuditGuard(lambda:p.healthy(spec,admission)))
        p.predecessor_drain(catalog,allowed_units=[spec['units']['controller']],spec=spec);p.sources(spec)
        p.once(c/'terminal.json',dict(passed=True,diagnostic_complete=True,admission=p.ref(c/'admission.json'),completion=result,
            all_groups_physically_drained=True,successful_units_retained_for_external=True,automatic_successor_launch=False,
            model_qualified=False,awaiting_external_raw_byte_audit=True,new_speedup_claim=False))
        status(spec,'complete')
    except BaseException as error:
        retry=[True];errors=[];cleaned=[]
        def request_retry(signum,frame):retry[0]=True
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
            if sig not in handlers:handlers[sig]=signal.getsignal(sig)
            signal.signal(sig,request_retry)
        for phase in started:
            try:cleaned.append(dict(phase=phase,drain=cleanup_phase(spec,phase,spec_ref,review_ref)))
            except BaseException as exc:errors.append(dict(phase=phase,error=repr(exc)))
        if errors:p.once(c/'cleanup-pending.json',dict(error=repr(error),errors=errors,automatic_successor_release=False))
        while errors:
            if retry[0]:
                retry[0]=False;pending=[]
                for item in errors:
                    try:cleaned.append(dict(phase=item['phase'],drain=cleanup_phase(spec,item['phase'],spec_ref,review_ref)))
                    except BaseException as exc:pending.append(dict(phase=item['phase'],error=repr(exc)))
                errors=pending
            if errors:time.sleep(.5)
        p.once(c/'failed-terminal.json',dict(passed=False,error=repr(error),traceback=traceback.format_exc(),
            all_started_groups_physically_drained=True,cleanup=cleaned,automatic_successor_release=False,model_qualified=False,new_speedup_claim=False))
        status(spec,'failed',error=repr(error));raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        release(fd)


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('check','execute','launch','audit','recover-observer-release'):mode.add_argument('--'+name,action='store_true')
    parser.add_argument('--release-failed',action='store_true')
    for name in ('spec','spec-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();s=dict(path=args.spec,sha256=args.spec_sha256);r=dict(path=args.review,sha256=args.review_sha256)
    if args.check:p.bundle(s,r)
    elif args.execute:execute(s,r)
    elif args.recover_observer_release:
        from nemotron9b_package_window_v24 import recover_released_registry
        recover_released_registry(s,r)
    else:
        from nemotron9b_package_external_v24 import launch,audit
        (launch(s,r)if args.launch else audit(s,r,args.release_failed))


if __name__=='__main__':main()
