"""Independent owner ACK, retained exit audit and physical release authority."""
import os
from pathlib import Path
import signal
import stat
import time
import nemotron9b_package_protocol_v23 as p
from nemotron9b_package_controller_v23 import acquire,release


def control_dir(spec):
    path=Path(spec['control_root']);info=path.lstat()
    p.require(path.resolve()==path and stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode)==0o700 and info.st_uid==1000,'Owned Nemotron package control directory')
    return path


def launch(spec_ref,review_ref):
    spec,catalog=p.bundle(spec_ref,review_ref);p.low.user_bus();fd,owner,lock=acquire()
    control=Path(spec['control_root'])
    try:
        p.predecessor_drain(catalog,spec=spec);p.low.active_heavy();p.storage(spec,reserve=True)
        p.require(all(not Path(spec[k]).exists() for k in ('control_root','work_root','evidence_root')),'Fresh external namespaces')
        for stage in ('controller',*p.STAGES):
            props=p.low.properties(spec['units'][stage])
            p.require(props['LoadState']=='not-found' and p.low.terminal(props) and not props['InvocationID'],'Never reuse Nemotron package services')
            p.low.empty_group(p.group(spec,stage))
        control.mkdir(mode=0o700)
        command=p.service_command(spec,'controller',spec_ref,review_ref)
        from nemotron9b_package_window_v23 import register
        register(spec,spec_ref,review_ref,owner)
        p.once(control/'external-launch-intent.json',dict(schema='nemotron9b_package_external_intent_v3',execution_spec=spec_ref,source_review=review_ref,
            command=command,external_owner=owner,lock_identity=lock,canonical_sha256=p.canonical(),automatic_successor_launch=False))
    finally:release(fd)
    try:
        p.command(command);deadline=time.monotonic()+20;captured=None
        while captured is None:
            captured,value=p.service_identity(spec,'controller',spec_ref,review_ref)
            p.require(not p.low.terminal(value),'Controller exited before independent owner ACK')
            p.require(time.monotonic()<deadline,'Independent live-owner timeout')
            if captured is None:time.sleep(.05)
        from nemotron9b_package_window_v23 import bind_controller
        bind_controller(spec,spec_ref,review_ref,captured)
        p.once(control/'external-controller-owner.json',dict(launch_intent=p.ref(control/'external-launch-intent.json'),owner=captured))
        deadline=time.monotonic()+48*3600+180
        while True:
            owner,value=p.service_identity(spec,'controller',spec_ref,review_ref,captured)
            p.require(owner==captured,'Controller owner changed')
            if p.low.terminal(value):break
            p.require(time.monotonic()<deadline,'Controller external wait deadline; explicit audit required');time.sleep(.5)
    except BaseException as error:
        p.once(control/'external-launch-failure.json',dict(error=repr(error),automatic_successor_release=False))
        try:audit(spec_ref,review_ref,False)
        except BaseException:pass
        raise
    return audit(spec_ref,review_ref,False)


def all_snapshots(spec,spec_ref,review_ref,control):
    ack=control/'external-controller-owner.json';observed=p.read(ack) if ack.exists() else None
    if observed:p.require(observed['launch_intent']==p.ref(control/'external-launch-intent.json'),'External ACK launch identity')
    rows={'controller':p.snapshot(spec,'controller',spec_ref,review_ref,observed['owner'] if observed else None)}
    for stage in p.STAGES:
        launch_path=control/(stage+'-launch.json');owner_path=control/(stage+'-owner.json')
        owner=p.read(owner_path) if owner_path.exists() else None
        if launch_path.exists():
            p.require(p.read(launch_path)['command']==p.service_command(spec,stage,spec_ref,review_ref),'Exact phase launch evidence')
        else:
            value=p.low.properties(spec['units'][stage]);p.require(value['LoadState']=='not-found' and owner is None,'Unenrolled phase appeared')
        rows[stage]=p.snapshot(spec,stage,spec_ref,review_ref,owner,allow_live=True)
    return rows


def audited_success(rows,admission,terminal,failed):
    return (admission is not None and terminal is not None and not failed
        and all(row['successful'] and row['identity_verified'] for row in rows.values())
        and terminal.get('passed') is True and terminal.get('diagnostic_complete') is True
        and terminal.get('all_groups_physically_drained') is True and terminal.get('successful_units_retained_for_external') is True)


def audit(spec_ref,review_ref,release_failed=False):
    spec,catalog=p.bundle(spec_ref,review_ref);p.low.user_bus();control=control_dir(spec)
    intent=p.read(control/'external-launch-intent.json')
    p.require(intent['execution_spec']==spec_ref and intent['source_review']==review_ref
        and intent['command']==p.service_command(spec,'controller',spec_ref,review_ref),'Original external launch intent')
    fd,auditor,lock=acquire();handlers={};retry=[True];cleaned={};cleanup_errors={};rows=None
    def request_retry(signum,frame):retry[0]=True
    try:
        p.require(lock==intent['lock_identity'],'Serial file unchanged')
        p.predecessor_drain(catalog,spec=spec)
        p.low.active_heavy([spec['units'][s] for s in ('controller',*p.STAGES)])
        admission=p.read(control/'admission.json') if (control/'admission.json').exists() else None
        if admission:p.require(admission['execution_spec']==spec_ref and admission['lock_identity']==lock,'Exact runtime admission')
        before_path=control/'external-pre-stop.json';recovering=before_path.exists()
        audit_path=control/'external-audit.json';release_path=control/'root-release.json'
        prior_audit=p.read(audit_path) if audit_path.exists() else None
        prior_release=p.read(release_path) if release_path.exists() else None
        # Never reclassify an already released diagnostic. A caller can inspect
        # that immutable release; a new audit requires a fresh namespace.
        p.require(prior_release is None,'Diagnostic already released; preserve its immutable audit/release')
        p.require(prior_audit is None or recovering,'Prior audit requires original pre-stop evidence')
        if recovering:
            before=p.read(before_path);p.require(before['launch_intent']==p.ref(control/'external-launch-intent.json'),'Original external retained snapshot')
            rows=before['rows']
            for stage in ('controller',*p.STAGES):p.validate_after(spec,stage,spec_ref,review_ref,rows[stage],p.low.properties(spec['units'][stage]))
        else:
            rows=all_snapshots(spec,spec_ref,review_ref,control)
            before=dict(launch_intent=p.ref(control/'external-launch-intent.json'),rows=rows,auditor=auditor)
            p.once(before_path,before)
        terminal=p.read(control/'terminal.json') if (control/'terminal.json').exists() else None
        failed=(control/'failed-terminal.json').exists() or (control/'external-cleanup-pending.json').exists()
        if prior_audit is not None:
            p.require(prior_audit['pre_stop']==p.ref(before_path),'Prior audit binds original retained snapshot')
            p.require(type(prior_audit['passed']) is bool,'Prior audit has explicit result')
        # The first call alone may qualify current retained exits. Once pre-stop
        # is durable, an interrupted/repeated call is conservatively a failed
        # recovery even if the saved snapshot once contained successful exits.
        # In particular, final-verification failure cannot become success, and
        # explicit failed release must not demand services already stopped/GC'd.
        candidate=not recovering and prior_audit is None and audited_success(rows,admission,terminal,failed)
        if candidate:
            p.require(admission['controller']==rows['controller']['owner'] and terminal['admission']==p.ref(control/'admission.json'),'Successful controller admission binds independent owner')
            for stage in p.STAGES:p.retained_drain(spec,stage,spec_ref,review_ref,rows[stage]['owner'])
        # Pre-stop evidence is now independently durable; success no longer
        # depends on systemd retaining these objects during ordinary stop GC.
        for stage in ('controller',*p.STAGES):
            try:cleaned[stage]=p.stop_and_drain(spec,stage,spec_ref,review_ref,rows[stage])
            except BaseException as error:cleanup_errors[stage]=repr(error)
        if cleanup_errors:
            candidate=False
            for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):handlers[signum]=signal.signal(signum,request_retry)
            pending=control/'external-cleanup-pending.json'
            if not pending.exists():p.once(pending,dict(errors=cleanup_errors,automatic_successor_release=False))
        while cleanup_errors:
            if retry[0]:
                retry[0]=False
                for stage in list(cleanup_errors):
                    try:
                        cleaned[stage]=p.stop_and_drain(spec,stage,spec_ref,review_ref,rows[stage]);del cleanup_errors[stage]
                    except BaseException as error:cleanup_errors[stage]=repr(error)
            if cleanup_errors:time.sleep(.5)
        result=None;failure=[]
        if candidate:
            try:
                def full_audit():
                    p.low.lock_proof(auditor,lock);p.storage(spec);p.low.active_heavy()
                    p.require(p.canonical()==intent['canonical_sha256'],'Canonical unchanged during external validation')
                guard=p.AuditGuard(full_audit)
                guard.force();result=p.completion(spec,admission,guard)
                from nemotron9b_package_evidence_v23 import final_audit
                raw=final_audit(spec,admission,guard);guard.force(terminal=True)
                result.update(full_model_qualified=True,awaiting_external_raw_byte_audit=False,external_raw_byte_audit=raw)
            except BaseException as error:candidate=False;failure.append(repr(error))
        else:failure.append('Recovery after durable pre-stop is conservatively failed; original failure cannot be promoted' if recovering else 'Missing independent retained successful exits or incomplete/failed controller evidence')
        p.predecessor_drain(catalog,spec=spec);p.low.active_heavy();p.low.lock_proof(auditor,lock);p.storage(spec)
        p.require(p.canonical()==intent['canonical_sha256'],'Final canonical unchanged')
        p.sources(spec)
        report=dict(schema='nemotron9b_package_external_audit_v3',passed=candidate,independent_retained_exits=all(x['identity_verified'] for x in rows.values()),
            pre_stop=p.ref(before_path),physical_drain=cleaned,completion=result,failures=failure,canonical_unchanged=True,
            model_qualified=candidate,new_speedup_claim=False,automatic_successor_launch=False)
        if prior_audit is not None:
            # Keep the first audit immutable, including an interrupted success
            # with no release. The recovery result binds it and the original
            # pre-stop identity, and records the newly checked physical drain.
            report.update(schema='nemotron9b_package_external_recovery_audit_v3',passed=False,
                original_audit=p.ref(audit_path),recovery_only=True,success_not_promoted=True)
            audit_path=control/'external-recovery-audit.json'
            if audit_path.exists():
                existing=p.read(audit_path)
                p.require(existing['passed'] is False and existing['recovery_only'] is True
                    and existing['original_audit']==report['original_audit'] and existing['pre_stop']==report['pre_stop'],
                    'Existing recovery audit retains exact original failure/provenance')
            else:p.once(audit_path,report)
        else:p.once(audit_path,report)
        if candidate or release_failed:
            record=dict(released=True,successful=candidate,failed=not candidate,explicit_failed_release=bool(release_failed and not candidate),
                audit=p.ref(audit_path),all_current_groups_recursively_drained=True,canonical_unchanged=True,automatic_successor_launch=False,
                model_qualified=candidate,new_speedup_claim=False)
            if release_path.exists():p.require(p.read(release_path)==record,'Release identity unchanged')
            else:p.once(release_path,record)
            from nemotron9b_package_window_v23 import release_registry
            release_registry(spec,p.ref(release_path))
            return record
        raise RuntimeError('Failed diagnostic drained; root must explicitly release failed audit')
    finally:
        # No exception (including source/metadata failure before pre-stop) may
        # release the experiment while any original Nemotron package group remains live.
        # Unknown or replaced identities are never stopped blindly.
        def physically_clear():
            for stage in ('controller',*p.STAGES):
                value=p.low.properties(spec['units'][stage])
                p.require(p.low.terminal(value),'Cannot unlock live Nemotron package unit')
                p.low.empty_group(p.group(spec,stage))
            return True
        while True:
            try:
                physically_clear();break
            except BaseException:
                for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
                    if signum not in handlers:handlers[signum]=signal.getsignal(signum)
                    signal.signal(signum,request_retry)
                if retry[0]:
                    retry[0]=False
                    if rows is not None and (control/'external-pre-stop.json').exists():
                        for stage in ('controller',*p.STAGES):
                            try:p.stop_and_drain(spec,stage,spec_ref,review_ref,rows[stage])
                            except BaseException:pass
                time.sleep(.5)
        for signum,handler in handlers.items():signal.signal(signum,handler)
        release(fd)
