"""Real retained-lock owning parent; unbound and currently blocked pre-cost."""
import json
from pathlib import Path
import signal
import time
import uuid
from .contract import LIMITS,LOCK_IDENTITY,PHASES,digest,phase_directory,require,units,validate_spec
from .io import RetainedLock,identity,loaded_local_sources,once,process_api,read,ref,wait_until
from .resources import load_policy,resources
from .bootstrap import Observer,coordinator_service
from .evidence import file_sha256,release_for,seal_outputs
from .phase import Phase
from .readiness import require_loaded_layout_producer
from .checkpoint import authenticate_checkpoint

def fresh_paths(spec):
    inputs=[Path(spec['original_root']),Path(spec['original_case']['model_path']),Path(spec['original_case']['artifact'])]
    for text in spec['paths'].values():
        path=Path(text)
        require(not path.exists()and not path.is_symlink()and path.parent.resolve()==path.parent
            and path.parent.is_dir()and not any(p.is_symlink()for p in path.parents),'Fresh ordinary explicit output root')
        require(all(path!=old and path not in old.parents and old not in path.parents for old in inputs),
            'Never place outputs inside or around original evidence/model/artifact')
    require(not Path(spec['running_handoff_path']).exists(),'Fresh exact external running-handoff output')

class Serial:
    def __init__(self,spec,admission_ref,admission,coordinator,observer,history):
        self.spec,self.admission_ref,self.admission=spec,admission_ref,admission
        self.coordinator,self.observer,self.history=coordinator,observer,history
        self.active=None;self.phases=[];self.completed=[];self.sealed=dict(spec['original_input_sha256'])

    def guard(self,stage=None,run_id=None):
        self.observer.check(full=True);self.history.unchanged()

    def execute(self):
        control=Path(self.spec['paths']['control'])
        for arm,name in PHASES:
            self.guard('release_'+name)
            release=release_for(self.spec,arm,name,self.admission_ref,self.admission['outer_owner'],self.completed,self.sealed)
            release_ref=once(control/(arm+'-'+name+'-input-release.json'),release)
            phase=Phase(self,arm,name,phase_directory(self.spec,arm,name),uuid.uuid4().hex,release_ref)
            self.phases.append(phase);result=phase.leaf()
            self.guard('seal_'+name)
            outputs=seal_outputs(self.spec,phase,result,self.observer.check)
            require(not set(outputs)&set(self.sealed),'No earlier evidence is overwritten or rebound')
            seal=once(control/(arm+'-'+name+'-sealed.json'),dict(schema='deepseek_cold_parent_output_seal_v1',
                admission=self.admission_ref,owner=self.admission['outer_owner'],phase=[arm,name],
                phase_spec=phase.phase_ref,actual_child_exit=result['actual_child_exit'],
                actual_service_terminal=result['actual_service_terminal'],sha256=outputs,
                independently_sealed_after_actual_exit_and_drain=True,self_exit_claimed=False,
                full_model_qualified=False,new_speedup_claim=False))
            if name=='build':once(control/'build-immutable-identities.json',dict(seal=seal,sha256=outputs))
            self.sealed.update(outputs);self.completed.append(dict(phase=[arm,name],seal=seal))
        return dict(completed=self.completed,full_model_qualified=False,new_speedup_claim=False)

def owned_terminal(service):
    if service is None or not service.started:return None
    if service.terminal is not None and service.terminal.get('never_loaded')is True:return service.cleanup()
    return service.drained()

def outer(spec_ref,review_ref):
    # No part of the expensive prefix may run while phase six is unprovable.
    require_loaded_layout_producer()
    spec=read(spec_ref,LIMITS['spec_envelope_bytes']);review=read(review_ref,LIMITS['spec_envelope_bytes'])
    names=validate_spec(spec,review,spec_ref);policy=load_policy(spec);fresh_paths(spec)
    resources(spec,policy)
    from qwen35_package_serial_runtime_v9 import compiler_assets
    compiler_assets.verify_all(dict(source_sha256=spec['source_sha256']))
    from minicpm5_retained_replay_history_v1 import History,Authority
    from .lifecycle import check_running
    from qwen35_package_preflight_serial_v2 import save_history
    from minicpm5_layer1_drain_hold_v1 import retain_until_drained
    lock=RetainedLock();lock.acquire();started=time.monotonic();interrupted=[False];handlers={}
    history=None;observer=None;coordinator=None;serial=None;admission_ref=None;result=None;failure=None
    control=Path(spec['paths']['control']);last_resource=[-float('inf')]
    for sig in(signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:interrupted.__setitem__(0,True))
    def guard():
        lock.check();require(not interrupted[0]and time.monotonic()-started<spec['outer_seconds'],'Finite uninterrupted owning attempt')
        if observer is not None:observer.check()
        elif time.monotonic()-last_resource[0]>=policy['monitor_seconds']:
            resources(spec,policy);last_resource[0]=time.monotonic()
    try:
        guard();process_api().active_heavy()
        history=History(spec,spec_ref,review_ref,names,guard).resolve()
        pair=history.physical_pair(Authority(history,guard,lambda:[]));freshness=history.freshness();history.unchanged();guard()
        checkpoint=authenticate_checkpoint(spec,spec_ref,lock.owner,guard)
        for path,expected in {**spec['original_input_sha256'],**spec['original_artifact_sha256']}.items():
            require(file_sha256(path,guard)==expected,'Every original full input/artifact externally admitted before native work')
        fresh_paths(spec)
        # Compiler paths must remain absent for graph.build's fresh namespace.
        for key,value in spec['paths'].items():
            if key not in ('capture_artifact','capture_cache','capture_workdir'):Path(value).mkdir(mode=0o700)
        (control/'tmp').mkdir(mode=0o700)
        checkpoint_ref=once(control/'original-checkpoint-provenance.json',checkpoint.proof)
        transported=dict(physical_pair=pair,current_serial_catalog=spec['current_serial_catalog'],history_overlay=spec['history_overlay'],
            resolved_plan_sha256=history.resolved['plan_sha256'],metadata_identities=history.meta.identities,freshness=freshness)
        history_refs=save_history(control,transported)
        admission=dict(schema='deepseek_cold_serial_admission_v2',spec=spec_ref,review=review_ref,outer_owner=lock.owner,
            retained_fd=lock.fd,initial_descriptor_proof=lock.initial,lock_identity=LOCK_IDENTITY,
            current_serial_catalog=spec['current_serial_catalog'],history_overlay=spec['history_overlay'],freshness=freshness,
            units=names,history=history_refs,started_monotonic=started,deadline_monotonic=started+spec['outer_seconds'],
            source_map_sha256=digest(spec['source_sha256']),checkpoint_provenance=checkpoint_ref,full_model_qualified=False,new_speedup_claim=False,
            self_exit_claimed=False,automatic_successor=False)
        admission_ref=once(control/'admission.json',admission)
        observer=Observer(admission,spec,policy);observer.history=transported;observer.cancelled=lambda:interrupted[0]
        observer.checkpoint=checkpoint
        coordinator=coordinator_service(spec,admission_ref);observer.current=coordinator
        coordinator.start(guard);observer.current=None;observer.coordinator=coordinator
        owner_ref=wait_until(lambda:ref(control/'coordinator-owner.json')if(control/'coordinator-owner.json').exists()else None,
            guard,LIMITS['startup_ack_seconds'],message='Actual coordinator owner receipt missing')
        require(read(owner_ref)==coordinator.owner,'Parent independently observed exact coordinator identity')
        print(json.dumps(dict(event='deepseek_cold_serial_running_handoff_required',admission=admission_ref,
            coordinator_owner=owner_ref,coordinator=coordinator.owner,running_handoff_path=spec['running_handoff_path'],
            actual_external_tool_session_required=True,native_work_released=False)),flush=True)
        wait_until(lambda:True if Path(spec['running_handoff_path']).exists()else None,guard,
            LIMITS['startup_ack_seconds'],message='Root-authenticated actual tool running handoff missing')
        running=check_running(history,admission_ref,admission,owner_ref,coordinator.owner);observer.running=running
        once(control/'running-ack.json',dict(admission=admission_ref,coordinator_owner=owner_ref,coordinator=coordinator.owner,running=running))
        serial=Serial(spec,admission_ref,admission,coordinator,observer,history);result=serial.execute()
        once(control/'serial-complete.json',dict(admission=admission_ref,coordinator=coordinator.owner,
            completed_phases=[list(p)for p in PHASES],full_model_qualified=False,new_speedup_claim=False))
        # Coordinator may exit during this wait: its separate live requirement
        # ends, but actual original descriptor/deadline/resources remain guarded.
        observer.coordinator=None
        wait_until(lambda:True if process_api().terminal(coordinator.observe())else None,guard,
            LIMITS['owned_cleanup_seconds'],message='Coordinator actual exit missing')
        terminal=coordinator.cleanup()
        require(terminal['systemd_exit_status']==dict(ExecMainCode=1,ExecMainStatus=0,Result='success'),'Actual coordinator failed')
        history.unchanged();guard()
    except BaseException as error:failure=error
    finally:
        def cleanup():
            lock.check()
            if serial is not None:
                for phase in reversed(serial.phases):phase.finish()
            if coordinator is not None and coordinator.started:coordinator.cleanup()
        def verify():
            lock.check();terminals=[]
            if serial is not None:
                for phase in serial.phases:
                    value=owned_terminal(phase.service)
                    if value is not None:terminals.append(value)
                    if phase.child is not None:process_api().gone(phase.child)
            terminal=owned_terminal(coordinator)
            if terminal is not None:terminals.append(terminal)
            return dict(physically_drained=True,terminals=terminals)
        proof=retain_until_drained(cleanup,verify,
            lambda value:print(json.dumps(dict(event='deepseek_cold_cleanup_pending',serial_lease_retained=True,details=value)),flush=True),
            lambda:time.sleep(1))
        try:
            if admission_ref is not None:
                once(control/'outer-terminal.json',dict(schema='deepseek_cold_outer_terminal_v1',admission=admission_ref,outer_owner=lock.owner,
                    complete_owned_unit_terminals=proof['terminals'],whole_owned_scope_drained=True,result_present=result is not None,
                    failed=failure is not None,failure=None if failure is None else dict(type=type(failure).__name__,message=str(failure)[:2048]),
                    completed_phases=[]if serial is None else serial.completed,full_model_qualified=False,new_speedup_claim=False,
                    self_exit_claimed=False,actual_external_tool_exit_pending=True,automatic_successor=False))
        finally:
            lock.release(proof['physically_drained'])
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if failure is not None:raise failure
    return 0
