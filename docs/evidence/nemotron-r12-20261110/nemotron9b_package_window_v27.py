"""Fixed discoverable observer gate; use --observe for every external read job.

No file is created on import. Provisioning and each registration are explicit
runtime actions, with source review and whole-run serial ownership elsewhere.
Shared observer locks cover the complete command, not just a boolean check.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import time
import nemotron9b_package_protocol_v27 as p

GATE=p.ROOT/'nemotron9b-package-observer-gate-v3.lock'
REGISTRY=p.ROOT/'nemotron9b-package-window-v3.json'


def gate_identity():
    info=GATE.lstat()
    p.require(not GATE.is_symlink()and stat.S_ISREG(info.st_mode)and info.st_nlink==1 and info.st_uid==os.getuid()
        and stat.S_IMODE(info.st_mode)==0o600 and info.st_size==0,'Fixed owned observer lock')
    return dict(path=str(GATE),device=info.st_dev,inode=info.st_ino)


def registry():
    # Deliberately independent of p.read()/external gate to avoid recursion.
    fd=os.open(REGISTRY,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd);p.require(stat.S_ISREG(before.st_mode)and before.st_uid==os.getuid()and before.st_size<16384,'Bounded window registry')
        raw=os.read(fd,16385);after=os.fstat(fd)
        p.require(len(raw)<=16384 and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Registry changed during read')
        row=json.loads(raw);p.require(row['schema']=='nemotron_external_window_v3'and row['gate']==gate_identity(),'Fixed registry/gate identity')
        return row
    finally:os.close(fd)


def write_registry(row):
    p.require(row['gate']==gate_identity(),'Registry lock identity')
    temporary=REGISTRY.with_name(REGISTRY.name+'.next')
    # A crash leaving .next requires root recovery; never overwrite its intent.
    p.once(temporary,row);os.replace(temporary,REGISTRY)
    fd=os.open(REGISTRY.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def open_gate(expected=None):
    found=gate_identity();p.require(expected is None or found==expected,'Admitted observer gate identity')
    fd=os.open(GATE,os.O_RDONLY|os.O_NOFOLLOW);info=os.fstat(fd)
    p.require((info.st_dev,info.st_ino)==(found['device'],found['inode']),'Observer lock changed on open')
    return fd


def lock(fd,exclusive,timeout=60):
    deadline=time.monotonic()+timeout
    while True:
        try:fcntl.flock(fd,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|fcntl.LOCK_NB);return
        except BlockingIOError:
            p.require(time.monotonic()<deadline,'Observer barrier timeout; no timed admission');time.sleep(.05)


def unlock(fd):fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def provision():
    """Root explicitly provisions once before final spec; never resets registry."""
    p.quiet()
    fd=os.open(GATE,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
    p.once(REGISTRY,dict(schema='nemotron_external_window_v3',gate=gate_identity(),state='idle',performance_window=False,
        execution_spec=None,source_review=None,owner=None,release=None))
    return gate_identity()


def contract(spec):
    policy=spec['external_observer_contract'];row=p.read(policy['path'],policy['sha256'])
    p.require(row['schema']=='nemotron_external_observer_contract_v3','Exact observer contract schema')
    p.require(row['root_reviewed']is True and row['all_external_operations_use_gate']is True
        and row['unregistered_observers_paused']is True and row['shared_gate_covers_entire_operation']is True and row['no_detached_observer_work']is True,
        'Root reviewed complete external observer isolation policy')
    p.require(row['gate']==spec['observer_gate']==gate_identity()and row['registry']==str(REGISTRY),'Fixed discoverable gate binding')
    p.require(row['source_sha256']==spec['source_sha256'][str(Path(__file__).resolve())],'Exact reviewed observer protocol source')
    p.require(type(row['participants'])is list and bool(row['participants'])and len(set(row['participants']))==len(row['participants']),
        'Named complete cooperating observer set')
    return row


def register(spec,spec_ref,review_ref,owner):
    contract(spec);fd=open_gate(spec['observer_gate'])
    try:lock(fd,True)
    except BaseException:os.close(fd);raise
    try:
        old=registry();p.require(old['state']in ('idle','released')and old['performance_window']is False,'No unresolved prior window/run')
        if old['state']=='released':
            ref=old['release'];release=p.read(ref['path'],ref['sha256'])
            p.require(release['released']and release['all_current_groups_recursively_drained'],'Prior physical release')
        row=dict(schema='nemotron_external_window_v3',gate=spec['observer_gate'],state='active',performance_window=False,
            execution_spec=spec_ref,source_review=review_ref,owner=owner,control_root=spec['control_root'],phase=None,
            observer_contract=spec['external_observer_contract'],release=None)
        write_registry(row)
    finally:unlock(fd)


def bind_controller(spec,spec_ref,review_ref,owner):
    fd=open_gate(spec['observer_gate'])
    try:lock(fd,True)
    except BaseException:os.close(fd);raise
    try:
        row=registry();p.require(row['state']=='active'and row['performance_window']is False
            and row['execution_spec']==spec_ref and row['source_review']==review_ref,'Registered exact new run')
        row['owner']=p.proc_owner(owner);write_registry(row)
    finally:unlock(fd)


def begin(spec,stage,owner,admission):
    contract(spec);fd=open_gate(spec['observer_gate'])
    try:
        lock(fd,True) # Wait for every pre-existing shared observer to finish.
        row=registry();p.require(row['state']=='active'and row['performance_window']is False
            and row['execution_spec']==admission['execution_spec']and row['owner']==p.proc_owner(admission['controller']),
            'Actual controller owns registered window')
        p.require(p.low.proc(row['owner']['pid'])==row['owner'],'Registered controller actually alive')
        row.update(state='measurement',performance_window=True,phase=stage)
        write_registry(row)
        proof=dict(registry=row,worker=owner,exclusive_observer_gate_held=True,prior_shared_observers_drained=True)
        p.once(Path(spec['control_root'])/(stage+'-observer-barrier.json'),proof)
        return fd
    except BaseException:
        unlock(fd);raise


def finish(spec,stage,fd,owner):
    row=registry();p.require(row['state']=='measurement'and row['phase']==stage and row['performance_window']is True,'Exact open window')
    c=Path(spec['control_root']);end=p.read(c/(stage+'-measurement-end.json'))
    p.require(end['owner']==owner,'Actual measurement-end owner')
    p.once(c/(stage+'-observer-window-end.json'),dict(barrier=p.ref(c/(stage+'-observer-barrier.json')),
        measurement_end=p.ref(c/(stage+'-measurement-end.json')),complete=end['complete'],exclusive_observer_gate_held=True))
    row.update(state='active',performance_window=False,phase=None);write_registry(row);unlock(fd)


def failed_hold(spec,stage,fd):
    # Keep fail-closed registry through controller cleanup and external retained
    # audit/release, even when the advisory lock itself is released.
    row=registry();p.require(row['control_root']==spec['control_root'],'Failed window exact run')
    row.update(state='failed-awaiting-drain-audit',performance_window=True,phase=stage);write_registry(row);unlock(fd)


def release_registry(spec,release_ref):
    fd=open_gate(spec['observer_gate'])
    try:lock(fd,True)
    except BaseException:os.close(fd);raise
    try:
        row=registry();p.require(row['control_root']==spec['control_root'],'Release exact registered run')
        release=p.read(release_ref['path'],release_ref['sha256'])
        p.require(release['released']and release['all_current_groups_recursively_drained'],'Independent physical release before reopening observers')
        row.update(state='released',performance_window=False,phase=None,release=release_ref);write_registry(row)
    finally:unlock(fd)


def recover_released_registry(spec_ref,review_ref):
    """Repair only an interrupted observer reopening, never rerun/promote audit."""
    from nemotron9b_package_controller_v27 import acquire,release
    spec,catalog=p.bundle(spec_ref,review_ref);p.low.user_bus();fd,auditor,lock=acquire()
    try:
        c=Path(spec['control_root']);release_ref=p.ref(c/'root-release.json');record=p.read(release_ref['path'],release_ref['sha256'])
        p.require(record['released']is True and record['all_current_groups_recursively_drained']is True,'Existing immutable physical release required')
        audit=p.read(record['audit']['path'],record['audit']['sha256'])
        p.require(audit['passed']is record['successful']and record['failed']is(not record['successful']),
            'Recovery preserves original success/failure')
        intent=p.read(c/'external-launch-intent.json')
        p.require(intent['execution_spec']==spec_ref and intent['source_review']==review_ref and intent['lock_identity']==lock,
            'Exact released execution identity')
        p.low.lock_proof(auditor,lock);p.predecessor_drain(catalog,spec=spec);p.low.active_heavy()
        for stage in ('controller',*p.STAGES):
            value=p.low.properties(spec['units'][stage]);p.require(p.low.terminal(value),'All released units still terminal')
            p.low.empty_group(p.group(spec,stage))
        p.require(p.canonical()==intent['canonical_sha256'],'Canonical unchanged during observer recovery')
        release_registry(spec,release_ref)
        return dict(observer_registry_reopened=True,existing_release=release_ref,original_success=record['successful'],no_reclassification=True)
    finally:release(fd)


@contextmanager
def external_observer(timeout=60):
    fd=open_gate()
    try:lock(fd,False,timeout)
    except BaseException:os.close(fd);raise
    try:
        row=registry()
        p.require(row['performance_window']is False and row['state']in ('idle','active','released'),'External operation refused during unresolved window')
        if row['state']=='active':p.require(p.low.proc(row['owner']['pid'])==row['owner'],'Active owner stale; fail closed')
        if row['state']=='released':
            ref=row['release'];release=p.read(ref['path'],ref['sha256'])
            p.require(release['released']and release['all_current_groups_recursively_drained'],'Bound physical release')
        yield dict(row,observer_gate_fd=fd)
    finally:unlock(fd)


def validate_window(spec,stage,owner):
    c=Path(spec['control_root']);barrier=p.read(c/(stage+'-observer-barrier.json'));begin=p.read(c/(stage+'-measurement-begin.json'))
    end=p.read(c/(stage+'-observer-window-end.json'));ready=p.read(c/(stage+'-measurement-ready.json'))
    admission=p.read(c/'admission.json');row=barrier['registry']
    p.require(row['gate']==spec['observer_gate']and row['state']=='measurement'and row['performance_window']is True
        and row['execution_spec']==admission['execution_spec']and row['source_review']==admission['source_review']
        and row['owner']==p.proc_owner(admission['controller'])and row['phase']==stage,'Window exact admitted controller/spec/gate')
    p.require(barrier['worker']==owner and barrier['exclusive_observer_gate_held']is True and barrier['prior_shared_observers_drained']is True,
        'Complete external observer barrier')
    p.require(ready['owner']==owner and ready['prepared']is True and begin==dict(owner=owner,allowed=True,
        observer_barrier=p.ref(c/(stage+'-observer-barrier.json'))),'Only begin after barrier and ready')
    p.require(end==dict(barrier=begin['observer_barrier'],measurement_end=p.ref(c/(stage+'-measurement-end.json')),
        complete=True,exclusive_observer_gate_held=True),'Successful window end before releasing exclusive gate')
    p.require(p.read(end['measurement_end']['path'],end['measurement_end']['sha256'])==dict(owner=owner,complete=True),'Exact complete end')
    return dict(barrier=begin['observer_barrier'],end=p.ref(c/(stage+'-observer-window-end.json')))


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--provision',action='store_true');mode.add_argument('--observe',action='store_true')
    parser.add_argument('--timeout',type=float,default=60);parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.provision:print(json.dumps(provision()));return
    command=args.command[1:]if args.command[:1]==['--']else args.command
    p.require(bool(command),'Explicit observer command required')
    with external_observer(args.timeout)as lease:
        # Inherit the same shared-lock open-file description: killing this
        # wrapper cannot release isolation while its command is still alive.
        fd=lease['observer_gate_fd']
        result=subprocess.run(command,check=False,pass_fds=(fd,),
            env=dict(os.environ,NEMOTRON_OBSERVER_GATE_FD=str(fd)))
        raise SystemExit(result.returncode)


if __name__=='__main__':main()
