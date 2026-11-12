"""External launch/exit/drain authority; stdlib only and inert on import.

The controller cannot release its own experiment. Its retained exit and every
phase's pre-stop state are persisted externally before any stop operation.
An explicit failed release records failure; it never qualifies a diagnostic.
"""
import fcntl
import os
from pathlib import Path
import stat
import signal
import time
import minicpm5_layer0_execution_protocol_v4 as p


def ref(path):return dict(path=str(path),sha256=p.sha(path))


def control_directory(spec):
    path=Path(spec['control_root']);info=path.lstat()
    p.require(path.resolve()==path and not path.is_symlink() and stat.S_ISDIR(info.st_mode) and info.st_uid==1000 and stat.S_IMODE(info.st_mode)==0o700,'Fixed owned control directory')
    return path


def intent(spec,spec_ref,review_ref):
    control=control_directory(spec);path=control/'external-launch-intent.json';record=p.read(path)
    p.require(record['schema']=='minicpm5_layer0_external_launch_v2' and record['execution_spec']==spec_ref and record['source_review']==review_ref,'Exact durable external launch binding')
    p.require(record['command']==p.controller_command(spec,spec_ref,review_ref) and record['namespace_absent']==[spec['controller_unit'],*spec['units'].values()],'Exact external command and fresh namespace')
    p.require(record['automatic_successor_launch'] is False,'No automatic enrollment')
    return record,ref(path)


def saved_owner(spec,launch_ref):
    path=Path(spec['control_root'])/'external-controller-owner.json'
    if not path.exists():return None
    row=p.read(path);p.require(row['launch_intent']==launch_ref and p.invocation(row['invocation_id']),'External owner launch binding')
    owner=row['owner'];p.require(owner['argv'] and owner['pid']>0 and int(owner['start_ticks'])>0,'Externally observed real owner')
    return row


def controller_ack(spec,spec_ref,review_ref,owner,invocation):
    """Controller may start no phase until an independent live owner is durable."""
    _,launch_ref=intent(spec,spec_ref,review_ref);deadline=time.monotonic()+60
    while True:
        row=saved_owner(spec,launch_ref)
        if row is not None:
            p.require(row['owner']==owner and row['invocation_id']==invocation,'Controller external owner ACK mismatch');return
        p.require(time.monotonic()<deadline,'External controller owner ACK timeout');time.sleep(.05)


def acquire_lock():
    fd=os.open(p.LOCK,os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);info=os.fstat(fd)
        p.require(stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_ino==p.LOCK.lstat().st_ino,'External serial lock identity')
        owner=p.proc(os.getpid());lock=dict(device=info.st_dev,inode=info.st_ino);p.lock_proof(owner,lock)
        return fd,owner,lock
    except BaseException:os.close(fd);raise


def release_lock(fd):
    fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)


def launch(spec_ref,review_ref,completion):
    p.clear_window();spec,plan,catalog=p.bundle(spec_ref,review_ref);p.user_bus()
    control=Path(spec['control_root']);p.require(not control.exists() and not Path(plan['output_root']).exists(),'Fresh external runtime namespace')
    fd,external_owner,lock=acquire_lock()
    try:
        p.predecessor_drain(catalog);p.active_heavy()
        units=[spec['controller_unit'],*spec['units'].values()]
        for unit in units:
            value=p.properties(unit);p.require(value['LoadState']=='not-found' and p.terminal(value) and not value['InvocationID'],'Never reuse controller or phase service')
            p.empty_group(p.GROUP_PREFIX+unit+'.service')
        fs=os.statvfs(p.ROOT);p.require(fs.f_bavail*fs.f_frsize>=30*p.GIB+spec['control_cap_bytes'],'Full capture storage reservation')
        command=p.controller_command(spec,spec_ref,review_ref);control.mkdir(mode=0o700)
        record=dict(schema='minicpm5_layer0_external_launch_v2',execution_spec=spec_ref,source_review=review_ref,command=command,
            namespace_absent=units,external_owner=external_owner,lock_identity=lock,
            canonical_sha256={n:p.sha(p.ROOT/n) for n in ('continuation-state.json','roadmap.json','best-known-configurations.json')},automatic_successor_launch=False)
        p.once(control/'external-launch-intent.json',record);launch_ref=ref(control/'external-launch-intent.json')
    finally:release_lock(fd)
    # The controller now acquires the same lock and waits for this owner's ACK.
    try:
        p.command(command,timeout=15);deadline=time.monotonic()+20;captured=None
        while captured is None:
            captured,value=p.controller_identity(spec,spec_ref,review_ref)
            p.require(not p.terminal(value),'Controller exited before independent owner observation')
            p.require(time.monotonic()<deadline,'Controller owner observation timeout')
            if captured is None:time.sleep(.05)
        observed=dict(launch_intent=launch_ref,owner=captured,invocation_id=value['InvocationID'])
        p.once(control/'external-controller-owner.json',observed)
        deadline=time.monotonic()+40*3600+180
        while True:
            _,value=p.controller_identity(spec,spec_ref,review_ref,captured,observed['invocation_id'])
            if p.terminal(value):break
            p.require(time.monotonic()<deadline,'Controller external wait deadline; explicit audit required')
            time.sleep(.5)
    except BaseException as error:
        p.once(control/'external-launch-failure.json',dict(error=repr(error),launch_intent=launch_ref,automatic_successor_release=False))
        # Preserve the original failure; a completed/abnormal terminal can still
        # be audited and physically cleaned. A live controller is never stopped
        # by an unauthenticated external exception handler.
        try:audit(spec_ref,review_ref,completion,release_failed=False)
        except BaseException:pass
        raise
    return audit(spec_ref,review_ref,completion,release_failed=False)


def controller_snapshot(spec,spec_ref,review_ref,observed):
    owner=observed['owner'] if observed else None;inv=observed['invocation_id'] if observed else None
    value=p.properties(spec['controller_unit'])
    if value['LoadState']=='not-found':
        p.require(not value['InvocationID'] and p.terminal(value),'Missing controller not terminal')
        if owner:p.gone(owner)
        return dict(properties=value,owner=owner,identity_verified=False,successful=False,uncaptured_owner=observed is None)
    owner,value=p.controller_identity(spec,spec_ref,review_ref,owner,inv,observation=value)
    p.require(p.terminal(value),'Controller still live; terminal audit only')
    if owner:p.gone(owner)
    return dict(properties=value,owner=owner,identity_verified=observed is not None,
        successful=observed is not None and value['Result']=='success' and value['ExecMainCode']=='1' and value['ExecMainStatus']=='0',uncaptured_owner=observed is None)


def phase_snapshot(spec,stage,admission_ref,admission):
    control=Path(spec['control_root']);unit=spec['units'][stage];value=p.properties(unit)
    launch_path=control/(stage+'-launch.json');owner_path=control/(stage+'-owner.json')
    launch=p.read(launch_path) if launch_path.exists() else None;owner=p.read(owner_path) if owner_path.exists() else None
    if launch is None:
        p.require(owner is None and value['LoadState']=='not-found' and not value['InvocationID'] and p.terminal(value),'Unenrolled phase appeared')
        return dict(stage=stage,properties=value,launch=None,owner=None,identity_verified=False,successful=False,never_launched=True)
    p.require(admission is not None and launch['command']==p.service_command(spec,stage,admission_ref,admission['attempt_id']) and launch['admission_sha256']==admission_ref['sha256'] and launch['attempt_id']==admission['attempt_id'],'Exact durable phase launch')
    if value['LoadState']=='loaded':
        p.service_contract(spec,stage,admission_ref,admission['attempt_id'],value)
        if owner:p.service_identity(spec,stage,admission_ref,admission['attempt_id'],owner,observation=value)
    else:p.require(value['LoadState']=='not-found' and not value['InvocationID'] and p.terminal(value),'Phase lost identity')
    ok=owner is not None and value['LoadState']=='loaded' and p.terminal(value) and value['Result']=='success' and value['ExecMainCode']=='1' and value['ExecMainStatus']=='0'
    return dict(stage=stage,properties=value,launch=launch,owner=owner,identity_verified=owner is not None and value['LoadState']=='loaded',successful=ok,never_launched=False)


def validate_observation(spec,spec_ref,review_ref,snapshot,admission_ref,admission,key,actual):
    """Bind every single observation to the saved original invocation and PID.

    Post-stop not-found is a legitimate GC state. It carries no new success
    claim; only previously persisted pre-stop results can establish success.
    """
    row=snapshot['controller'] if key=='controller' else snapshot['phases'][key];before=row['properties']
    if actual['LoadState']=='loaded':
        p.require(before['LoadState']=='loaded' and actual['InvocationID']==before['InvocationID'],'Pre-stop service invocation replaced')
        p.require(actual['ExecMainPID']==before['ExecMainPID'],'Pre-stop service PID replaced')
        if key=='controller':
            p.controller_identity(spec,spec_ref,review_ref,row['owner'],before['InvocationID'],observation=actual)
            p.require(actual['MainPID']=='0','Controller restarted')
        else:
            p.require(admission is not None,'Phase exists without admission')
            p.service_contract(spec,key,admission_ref,admission['attempt_id'],actual)
            if row['owner']:p.service_identity(spec,key,admission_ref,admission['attempt_id'],row['owner'],observation=actual)
    else:p.require(actual['LoadState']=='not-found' and not actual['InvocationID'] and p.terminal(actual),'Unknown post-stop service')
    return actual


def revalidate_snapshot(spec,spec_ref,review_ref,snapshot,admission_ref,admission):
    """No second properties read hidden inside one identity validation."""
    for key in ('controller',*p.STAGES):
        unit=spec['controller_unit'] if key=='controller' else spec['units'][key]
        validate_observation(spec,spec_ref,review_ref,snapshot,admission_ref,admission,key,p.properties(unit))


def drain(spec,spec_ref,review_ref,snapshot,admission_ref,admission):
    """Saved four-service status precedes every stop; always stop exited groups."""
    revalidate_snapshot(spec,spec_ref,review_ref,snapshot,admission_ref,admission)
    # Stopping the retained controller also propagates BindsTo. Explicit phase
    # stops cover races and guarantee already-exited parents' descendants die.
    for key in ('controller',*p.STAGES):
        unit=spec['controller_unit'] if key=='controller' else spec['units'][key]
        value=validate_observation(spec,spec_ref,review_ref,snapshot,admission_ref,admission,key,p.properties(unit))
        if value['LoadState']=='loaded':
            try:p.command(['systemctl','--user','stop','--no-block',unit])
            except RuntimeError:
                # GC can also win the query->stop interval. Only authenticated
                # not-found plus actual empty group permits a benign retry.
                after=validate_observation(spec,spec_ref,review_ref,snapshot,admission_ref,admission,key,p.properties(unit))
                if after['LoadState']!='not-found':raise
                saved=snapshot['controller'] if key=='controller' else snapshot['phases'][key]
                if saved['owner']:p.gone(saved['owner'])
                p.empty_group(p.GROUP_PREFIX+unit+'.service')
    deadline=time.monotonic()+150
    while time.monotonic()<deadline:
        revalidate_snapshot(spec,spec_ref,review_ref,snapshot,admission_ref,admission);rows=[];pending=False
        for key in ('controller',*p.STAGES):
            unit=spec['controller_unit'] if key=='controller' else spec['units'][key];saved=snapshot['controller'] if key=='controller' else snapshot['phases'][key]
            samples=[]
            for _ in range(2):
                # Identity failures are fatal, never swallowed as ordinary
                # stop progress. Only liveness/unstable samples cause retry.
                value=validate_observation(spec,spec_ref,review_ref,snapshot,admission_ref,admission,key,p.properties(unit))
                try:
                    p.require(p.terminal(value) and value['ActiveState'] in ('inactive','failed'),'Stopped service not terminal')
                    if saved['owner']:p.gone(saved['owner'])
                    group=p.empty_group(p.GROUP_PREFIX+unit+'.service')
                except RuntimeError:pending=True;break
                samples.append(dict(properties=value,group=group))
            if pending:break
            if samples[0]!=samples[1]:pending=True;break
            rows.append(dict(unit=unit,samples=samples))
        if not pending:return dict(physically_drained=True,rows=rows)
        time.sleep(.2)
    raise RuntimeError('External cleanup incomplete; no root release')


def _audit_locked(spec_ref,review_ref,completion,release_failed,spec,plan,catalog,control,launch,launch_ref,observed,controller,auditor,lock,recovery_errors):
    p.require(lock==launch['lock_identity'],'Serial lock changed after launch');p.active_heavy(list(spec['units'].values()))
    p.predecessor_drain(catalog)
    admission_path=control/'admission.json';admission=p.read(admission_path) if admission_path.exists() else None;admission_ref=ref(admission_path) if admission is not None else None
    if admission is not None:
        p.require(admission['execution_spec']==spec_ref and admission['source_review']==review_ref and admission['controller_unit']==spec['controller_unit'] and admission['lock_identity']==lock,'Exact admitted execution')
        p.require(observed is not None and admission['controller']==observed['owner'] and admission['controller_invocation']==observed['invocation_id'],'Admission external owner binding')
    snapshot_path=control/'external-pre-stop.json'
    if snapshot_path.exists():
        snapshot=p.read(snapshot_path);p.require(snapshot['launch_intent']==launch_ref and snapshot['admission']==admission_ref,'Saved stop evidence binding')
        revalidate_snapshot(spec,spec_ref,review_ref,snapshot,admission_ref,admission)
    else:
        phases={s:phase_snapshot(spec,s,admission_ref,admission) for s in p.STAGES}
        snapshot=dict(schema='minicpm5_layer0_external_pre_stop_v2',launch_intent=launch_ref,admission=admission_ref,controller=controller,phases=phases,auditor=auditor)
        p.once(snapshot_path,snapshot)
    # Retained results are now durable even if stop causes transient GC.
    physical=drain(spec,spec_ref,review_ref,snapshot,admission_ref,admission)
    qualified=False;failure=list(recovery_errors);result=None
    previous_audit=control/'external-audit.json';pending=control/'external-cleanup-pending.json'
    if pending.exists():failure.append('External cleanup required recovery; explicit failed release only')
    if previous_audit.exists() and not p.read(previous_audit)['passed']:failure.append('Prior external failure remains failed')
    terminal_path=control/'terminal.json';failed_path=control/'failed-terminal.json'
    if not snapshot['controller']['successful']:failure.append('Controller did not have independently observed successful exit')
    if not all(row['successful'] for row in snapshot['phases'].values()):failure.append('Not all three phase retained results succeeded')
    if admission is None or not terminal_path.exists() or failed_path.exists():failure.append('Missing complete admission/terminal or explicit controller failure')
    if not failure:
        try:
            terminal=p.read(terminal_path)
            p.require(terminal['passed'] and terminal['diagnostic_complete'] and terminal['admission_sha256']==admission_ref['sha256'] and terminal['all_groups_drained'],'Exact controller diagnostic completion')
            p.require([x['stage'] for x in terminal['phases']]==list(p.STAGES),'Exact phase terminal sequence')
            from minicpm5_layer0_tree_monitor_v1 import TreeMonitor
            monitor=TreeMonitor(plan,spec['input_plan']['canonical_sha256'],admission['attempt_id'],admission['output_root_identity'])
            result=completion(spec,plan,admission_ref,admission,monitor)
            p.require(result['passed'] and result['all88_original_endpoints'] and result['all_exported_payload_hashes_verified'],'Independent all88 receipt/payload completion')
            qualified=True
        except BaseException as error:failure.append(repr(error))
    p.predecessor_drain(catalog);p.active_heavy();p.lock_proof(auditor,lock)
    p.require({n:p.sha(p.ROOT/n) for n in launch['canonical_sha256']}==launch['canonical_sha256'],'Canonical changed during external audit')
    p.require(all(p.sha(path)==wanted for path,wanted in spec['source_sha256'].items()),'Frozen source changed during audit')
    p.control_usage(spec)
    audit_path=control/'external-audit.json'
    report=dict(schema='minicpm5_layer0_external_audit_v2',passed=qualified,diagnostic_complete=qualified,independent_controller_exit=snapshot['controller']['identity_verified'],
        launch_intent=launch_ref,pre_stop=ref(snapshot_path),admission=admission_ref,drain=physical,completion=result,failures=failure,
        controller_terminal=ref(terminal_path) if terminal_path.exists() else None,controller_failed_terminal=ref(failed_path) if failed_path.exists() else None,
        all257_predecessors_drained=True,canonical_unchanged=True,automatic_successor_launch=False,full_model_qualified=False,new_speedup_claim=False)
    if audit_path.exists():
        prior=p.read(audit_path);p.require(prior['launch_intent']==launch_ref and prior['pre_stop']==ref(snapshot_path) and prior['passed']==qualified,'Prior audit qualification changed')
    else:p.once(audit_path,report)
    if qualified or release_failed:
        release_path=control/'root-release.json';release=dict(schema='minicpm5_layer0_root_release_v2',released=True,successful=qualified,
            failed=not qualified,explicit_failed_release=bool(release_failed and not qualified),audit=ref(audit_path),launch_intent=launch_ref,
            all_current_groups_recursively_drained=True,all257_predecessors_drained=True,automatic_successor_launch=False,full_model_qualified=False,new_speedup_claim=False)
        p.lock_proof(auditor,lock)
        if release_path.exists():p.require(p.read(release_path)==release,'Prior root release changed')
        else:p.once(release_path,release)
        return release
    raise RuntimeError('Failed diagnostic physically drained and recorded; explicit --audit --release-failed required')



def audit(spec_ref,review_ref,completion,release_failed=False):
    p.clear_window();spec,plan,catalog=p.bundle(spec_ref,review_ref);p.user_bus();control=control_directory(spec)
    launch,launch_ref=intent(spec,spec_ref,review_ref);observed=saved_owner(spec,launch_ref)
    controller=controller_snapshot(spec,spec_ref,review_ref,observed)
    fd,auditor,lock=acquire_lock();handlers={};retry=[True];errors=[]
    def request_retry(signum,frame):retry[0]=True
    def physically_clear():
        for unit in [spec['controller_unit'],*spec['units'].values()]:
            for _ in range(2):
                p.require(p.terminal(p.properties(unit)),'Cleanup service remains live')
                p.empty_group(p.GROUP_PREFIX+unit+'.service')
        return True
    try:
        while True:
            try:
                return _audit_locked(spec_ref,review_ref,completion,release_failed,spec,plan,catalog,control,launch,launch_ref,observed,controller,auditor,lock,errors)
            except BaseException as error:
                # The lock may be released on a refusal only after every actual
                # group is empty. Otherwise retain it while cleanup is pending.
                try:physically_clear()
                except BaseException:
                    errors.append(repr(error))
                    if not handlers:
                        for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM,signal.SIGHUP):handlers[sig]=signal.signal(sig,request_retry)
                    pending=control/'external-cleanup-pending.json'
                    try:
                        if not pending.exists():p.once(pending,dict(passed=False,error=repr(error),launch_intent=launch_ref,automatic_successor_release=False))
                    except BaseException:pass
                    while not retry[0]:time.sleep(.5)
                    retry[0]=False
                    # Reenter the same authenticated snapshot/drain path. All
                    # original errors remain failures requiring explicit audit.
                    continue
                raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        release_lock(fd)
