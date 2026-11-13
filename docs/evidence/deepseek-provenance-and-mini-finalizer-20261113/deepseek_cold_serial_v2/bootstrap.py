"""Fresh coordinator/supervisor/leaf authentication, before any native import."""
import copy
import os
from pathlib import Path
import time
from .contract import LIMITS,LOCK_IDENTITY,PHASES,digest,expected_phase,require,units,validate_spec
from .io import descriptor,identity,loaded_local_sources,once,process_api,read,read_if_present,ref,wait_until
from .resources import load_policy,resources
from .service import Service,argv,properties
from .checkpoint import load_bound_proof

def coordinator_service(spec,admission_ref,owner=None):
    return Service(units(spec['run_name'])['coordinator'],argv('coordinator',admission_ref),
        spec['phase_environments']['coordinator'],spec['phase_resources']['coordinator'],
        Path(spec['paths']['control'])/'coordinator.log',owner=owner)

def phase_service(spec,phase,phase_ref,owner=None):
    return Service(phase['unit'],argv('supervisor',phase_ref),spec['phase_environments'][phase['arm']+':'+phase['phase']],
        phase['properties'],phase['log'],parent=units(spec['run_name'])['coordinator'],owner=owner)

def admit_shape(admission,spec,spec_ref,review_ref):
    require(type(admission)is dict and set(admission)=={'schema','spec','review','outer_owner','retained_fd','initial_descriptor_proof',
        'lock_identity','current_serial_catalog','history_overlay','freshness','units','history','started_monotonic','deadline_monotonic',
        'source_map_sha256','checkpoint_provenance','full_model_qualified','new_speedup_claim','self_exit_claimed','automatic_successor'},'Exact owning-parent admission')
    require(admission['schema']=='deepseek_cold_serial_admission_v2'and admission['spec']==spec_ref and admission['review']==review_ref
        and admission['lock_identity']==LOCK_IDENTITY and admission['current_serial_catalog']==spec['current_serial_catalog']
        and admission['history_overlay']==spec['history_overlay']and admission['units']==units(spec['run_name'])
        and admission['source_map_sha256']==digest(spec['source_sha256'])
        and all(admission[k]is False for k in ('full_model_qualified','new_speedup_claim','self_exit_claimed','automatic_successor')),
        'Exact independent full scope/source/lease admission')
    proof=admission['initial_descriptor_proof']
    require(type(admission['retained_fd'])is int and admission['retained_fd']>=0 and proof['passed']is True
        and proof['owner']==admission['outer_owner']and proof['retained_fd']==admission['retained_fd']
        and proof['lock']==LOCK_IDENTITY and proof['before']==proof['after'],'Retained original descriptor proof')
    require(type(admission['started_monotonic'])in(int,float)and type(admission['deadline_monotonic'])in(int,float)
        and admission['deadline_monotonic']-admission['started_monotonic']==spec['outer_seconds'],
        'Finite unchanged whole-attempt deadline')

def load_history(admission):
    from qwen35_package_preflight_serial_v2 import join_history
    references=admission['history'];require(type(references)is list and references,'Complete lossless parent history transport')
    manifest=read({k:v for k,v in references[0].items()if k!='bytes'})
    require(manifest['references']==references[1:],'All complete history chunk references retained')
    parts=[]
    for item in references[1:]:
        require(identity(item['path'])['size']==item['bytes'],'Exact history chunk byte length')
        parts.append(read({k:v for k,v in item.items()if k!='bytes'}))
    value=join_history(manifest,parts)
    require(value['current_serial_catalog']==admission['current_serial_catalog']and value['history_overlay']==admission['history_overlay']
        and value['freshness']==admission['freshness']and value['physical_pair']['full_physical_passes']==2
        and value['physical_pair']['lock_identity']==LOCK_IDENTITY
        and value['physical_pair']['original_failed_model_exit_code']==1,'Complete unchanged inherited history; original failure retained')
    return value

class Observer:
    """Cheap lease/deadline checks; resource and unit checks at bounded cadence.

    Full history identities and loaded-source authentication run at explicit
    boundaries. No .1-second loop rereads or rehashes the complete source table.
    """
    def __init__(self,admission,spec,policy):
        self.admission,self.spec,self.policy=admission,spec,policy
        self.coordinator=None;self.current=None;self.running=None;self.history=None;self.checkpoint=None
        self.last_slow=-float('inf');self.failure=None;self.discover=False;self.cancelled=None

    def check(self,*,full=False):
        if self.failure is not None:raise self.failure
        try:
            require(self.cancelled is None or not self.cancelled(),'Owning parent interruption revokes active scientific phase')
            descriptor(self.admission['outer_owner'],self.admission['retained_fd'])
            require(time.monotonic()<self.admission['deadline_monotonic'],'Finite complete cold attempt deadline')
            if self.coordinator is not None:self.coordinator.live()
            if full or time.monotonic()-self.last_slow>=self.policy['monitor_seconds']:
                resources(self.spec,self.policy);allowed=[]
                require(self.checkpoint is not None,'Actual complete checkpoint byte proof must precede monitoring')
                self.checkpoint.check()
                if self.coordinator is not None:allowed.append(self.coordinator.unit)
                if self.discover:self.discover_phase()
                if self.current is not None and(self.current.started or self.current.owner is not None):
                    self.current.observe();allowed.append(self.current.unit)
                process_api().active_heavy(allowed)
                if self.running is not None:
                    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
                    require(latest.latest_handoff()==self.running['observation'],'Unknown/replaced current running handoff')
                self.last_slow=time.monotonic()
            if full:
                if self.history is not None:
                    for path,wanted in self.history['metadata_identities'].items():
                        require(identity(path)==wanted,'Inherited complete metadata identity changed')
                    for owner in self.history['physical_pair']['owners']:process_api().gone(owner)
                loaded_local_sources(self.spec['source_sha256'])
            return dict(admitted=True,exclusive=True)
        except BaseException as error:self.failure=error;raise

    def discover_phase(self):
        candidates=[]
        for arm,name in PHASES:
            path=Path(self.spec['paths']['control'])/(arm+'-'+name)/'phase-spec.json'
            if not path.exists():continue
            binding=ref(path);phase=read(binding)
            require(phase==expected_phase(self.spec,phase['admission'],self.coordinator.owner,phase['run_id'],arm,name,
                str(path.parent),phase['host_identity'])and phase['admission']['path']==str(Path(self.spec['paths']['control'])/'admission.json'),
                'Coordinator authenticates every installed exact phase')
            owner=read_if_present(path.parent/'supervisor-owner.json')
            installed=properties(phase['unit'])
            if installed['LoadState']=='not-found':
                require(owner is None and installed['InvocationID']==''and not(path.parent/'root-terminal.json').exists(),
                    'Only declared not-yet-started phase may be absent')
                continue
            service=phase_service(self.spec,phase,binding,owner);service.started=True
            observed=service.observe()
            if not process_api().terminal(observed):candidates.append(service)
        require(len(candidates)<=1,'At most one independently authenticated current phase')
        self.current=candidates[0]if candidates else None

def load_context(admission_ref):
    admission=read(admission_ref)
    # These two exact types alone permit the complete 64 MiB envelope.
    spec=read(admission['spec'],LIMITS['spec_envelope_bytes'])
    review=read(admission['review'],LIMITS['spec_envelope_bytes'])
    validate_spec(spec,review,admission['spec']);admit_shape(admission,spec,admission['spec'],admission['review'])
    require(admission_ref['path']==str(Path(spec['paths']['control'])/'admission.json'),'Fresh exact parent admission path')
    descriptor(admission['outer_owner'],admission['retained_fd'])
    observer=Observer(admission,spec,load_policy(spec));observer.history=load_history(admission)
    observer.checkpoint=load_bound_proof(spec,admission['spec'],admission['outer_owner'],admission['checkpoint_provenance'],
        lambda:descriptor(admission['outer_owner'],admission['retained_fd']))
    return admission,spec,observer

def authenticate_running(admission_ref,admission,spec,coordinator):
    from .lifecycle import expected_running
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    ack=read(ref(Path(spec['paths']['control'])/'running-ack.json'))
    owner_ref=ref(Path(spec['paths']['control'])/'coordinator-owner.json')
    require(ack['admission']==admission_ref and ack['coordinator_owner']==owner_ref and ack['coordinator']==coordinator,
        'Root ACK binds actual parent and live coordinator')
    running=ack['running'];value=read(running['reference']);previous=read(admission['freshness']['latest_handoff'])
    wanted=expected_running(previous,value['updated_at'],value['deepseek_cold_serial_current_execution']['tool'],
        admission_ref,admission,owner_ref,coordinator)
    require(value==wanted and running['value']==value and latest.latest_handoff()==running['observation'],
        'Full inherited running handoff and externally observed actual tool identity')
    return running

def worker_bootstrap(phase_ref,*,role):
    phase=read(phase_ref);admission,spec,observer=load_context(phase['admission'])
    coordinator=coordinator_service(spec,phase['admission'],phase['coordinator']);coordinator.live();observer.coordinator=coordinator
    require(read(ref(Path(spec['paths']['control'])/'coordinator-owner.json'))==coordinator.owner,'Exact live coordinator owner receipt')
    expected=expected_phase(spec,phase['admission'],coordinator.owner,phase['run_id'],phase['arm'],phase['phase'],
        phase['directory'],phase['host_identity'])
    require(phase==expected and phase_ref['path']==str(Path(phase['control'])/'phase-spec.json'),'Exact complete scientific phase binding')
    service=phase_service(spec,phase,phase_ref,read_if_present(Path(phase['control'])/'supervisor-owner.json'))
    service.live();observer.current=service
    observer.running=authenticate_running(phase['admission'],admission,spec,coordinator.owner)
    current=process_api().proc(os.getpid())
    require(role in ('supervisor','leaf'),'Only acknowledged scientific worker roles')
    if role=='supervisor':require(current==service.owner['process']and current['argv']==argv(role,phase_ref),'Actual supervisor PID/start/argv')
    else:
        require(current['ppid']==service.owner['process']['pid']and current['argv']==argv(role,phase_ref)
            and current['cgroup']=='0::'+service.group,'Actual direct fresh native child')
        ack=wait_until(lambda:read_if_present(Path(phase['control'])/'child-release.json'),observer.check,
            LIMITS['startup_ack_seconds'],message='Independent child-owner ACK missing')
        require(ack==dict(phase_spec=phase_ref,supervisor=service.owner,child_bootstrap_owner=current,coordinator=coordinator.owner),
            'Independent actual child-owner release before native imports')
    release=read(phase['host_identity'])
    require(release['admission']==phase['admission']and release['owner']==admission['outer_owner']
        and release['phase']==[phase['arm'],phase['phase']]and release['producer_is_child']is False,
        'Parent-authenticated post-exit dependency release')
    observer.check(full=True)
    return phase,dict(guard=observer.check,service=service,observer=observer,admission=admission,spec=spec,release=release,
        child_environment=copy.deepcopy(spec['phase_environments'][phase['arm']+':'+phase['phase']]))

def native_leaf(phase_ref):
    phase,context=worker_bootstrap(phase_ref,role='leaf');guard=context['guard'];guard()
    from .leaf import dispatch
    result=dispatch(context['spec'],phase,context['release'],guard)
    guard(full=True)
    once(Path(phase['control'])/'leaf-result.json',dict(schema='deepseek_cold_actual_leaf_result_v1',phase_spec=phase_ref,
        actual_process=process_api().proc(os.getpid()),result=result,self_exit_claimed=False))
    return 0

def coordinator(admission_ref):
    admission,spec,observer=load_context(admission_ref);service=coordinator_service(spec,admission_ref)
    service.live();require(service.owner['process']==process_api().proc(os.getpid()),'Actual fresh coordinator')
    observer.coordinator=service;observer.discover=True
    stage=Path(spec['paths']['control']);once(stage/'coordinator-owner.json',service.owner)
    wait_until(lambda:read_if_present(stage/'running-ack.json'),observer.check,LIMITS['startup_ack_seconds'],message='Root actual-tool running ACK missing')
    observer.running=authenticate_running(admission_ref,admission,spec,service.owner);observer.check(full=True)
    complete=wait_until(lambda:read_if_present(stage/'serial-complete.json'),observer.check,spec['outer_seconds'],message='Finite serial completion')
    require(complete==dict(admission=admission_ref,coordinator=service.owner,completed_phases=[list(p)for p in PHASES],
        full_model_qualified=False,new_speedup_claim=False),'Every full phase independently completed before coordinator release')
    observer.check(full=True);return 0
