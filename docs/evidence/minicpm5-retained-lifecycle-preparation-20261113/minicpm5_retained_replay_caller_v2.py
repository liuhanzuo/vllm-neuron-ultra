"""Finite future pair->full replay caller. Import/--help never launch anything.

No cleanup/retirement phase exists. The external root supplies actual tool
sessions through the explicit ACK entry; workers wait before any payload IO.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import shlex
import stat
import subprocess
import sys
import time

Q=Path(__file__).resolve().parent;ROOT=Q.parent
for directory in (Q,ROOT):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_replay_contract_v2 as c
import minicpm5_retained_raw_transport_v1 as transport


def api():
    import qwen35_sampler_runtime_io_v12 as io
    import minicpm5_history_recovery_lifecycle_v3 as lifecycle
    return io,lifecycle


def refs(spec_ref,review_ref):
    io,_=api();spec=io.read(spec_ref,c.LIMITS['spec_bytes']);c.validate_spec(spec)
    review=io.read(review_ref,c.LIMITS['spec_bytes'])
    c.require(review==dict(schema='minicpm5_retained_replay_caller_source_review_v2',reviewer='/root',passed=True,
        blockers=[],spec=spec_ref,source_sha256=spec['source_sha256'],launch_authorized=True,deletion_authorized=False),
        'Exact separate root caller launch review; v1 source acceptance alone does not admit execution')
    for path in c.FILES:
        c.require(hashlib.sha256(io.raw(path,1<<20)).hexdigest()==spec['source_sha256'][path],'Actual caller source changed')
    accepted=io.read(c.ACCEPTED)
    c.require(accepted['passed']is True and accepted['source_ready_for_caller_integration']is True
        and accepted['production_deletion_admitted']is accepted['launch_authorized']is False,'Retain precise original source-only acceptance')
    io.loaded_local_sources(spec['source_sha256']);return spec


def bounded_ref(path):
    io,_=api();return dict(io.ref(path,c.LIMITS['file_bytes']),bytes=io.identity(path)['size'])


def phase_properties(phase):
    return dict(c.PROPERTIES,RuntimeMaxSec=str(c.LIMITS[phase+'_seconds']))


def source_unchanged(spec):
    io,_=api()
    for path in c.FILES:
        c.require(hashlib.sha256(io.raw(path,1<<20)).hexdigest()==spec['source_sha256'][path],'Executing source digest changed')


def scan_storage(spec):
    io,_=api()
    total=0;rows={}
    for name,cap in (('output_root',c.LIMITS['control_bytes']),('replay_root',c.LIMITS['replay_bytes'])):
        root=Path(spec[name]);pending=[root]if os.path.lexists(root)else[];size=count=0
        while pending:
            path=pending.pop();s=io.observed_stat(path);count+=1
            c.require(not stat.S_ISLNK(s.st_mode)and count<=c.LIMITS['max_entries'],'Finite nonsymlink output tree')
            if stat.S_ISDIR(s.st_mode):pending.extend(path.iterdir())
            else:
                file_cap=c.LIMITS['log_bytes']if path.suffix=='.log'else c.LIMITS['file_bytes']
                c.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_size<=file_cap,'Bounded singleton output/log')
                c.require(path.suffix in ('.json','.log')or io.transient(path),'Only explicit metadata/log outputs')
            size+=max(s.st_size,s.st_blocks*512)
        c.require(size<=cap,'Original output budget');rows[name]=dict(bytes=size,entries=count);total+=size
    return rows,total


def storage(spec,reserve=False):
    io,_=api()
    for attempt in range(8):
        try:rows,total=scan_storage(spec);break
        except io.PublicationRace:
            if attempt==7:raise ValueError('Authenticated atomic output publication did not settle')
            time.sleep(.01)
    for path,floor in ((ROOT,c.LIMITS['root_floor_bytes']),(Path('/mnt/nvme'),c.LIMITS['nvme_floor_bytes'])):
        fs=os.statvfs(path);free=fs.f_bavail*fs.f_frsize
        extra=max(0,c.LIMITS['control_bytes']+c.LIMITS['replay_bytes']-total)if reserve and path==ROOT else 0
        c.require(free>=floor+extra,'Original disk floor plus actual remaining reservation')
    return rows


class Phase:
    """Original Mini typed argv/attempt/identity contract plus retained exit facts."""
    def __init__(self,spec,phase,admission_ref,nonce):
        self.spec,self.phase,self.nonce=spec,phase,nonce;self.unit=spec['units'][phase]
        self.argv=c.phase_argv(phase,admission_ref);self.state={};self.client=None;self.exit=None
        self.started=False;self.owner=None;self.cleaned=None
        self.env=c.environment(spec['replay_root']if phase=='replay'else str(Path(spec['output_root'])/'pair-work'))
        self.props=phase_properties(phase)

    def command(self):
        _,life=api();log=Path(self.spec['output_root'])/(self.phase+'.log')
        props=dict(self.props,StandardOutput='append:'+str(log),StandardError='inherit',
            UnsetEnvironment='PYTHONPATH PYTHONHOME PYTHONSTARTUP LD_PRELOAD LD_LIBRARY_PATH VLLM_PLUGINS')
        return life.with_attempt(['systemd-run','--user','--quiet','--unit='+self.unit,'--service-type=exec',
            *[arg for key,value in props.items()for arg in ('-p',key+'='+value)],
            *['--setenv='+key+'='+value for key,value in self.env.items()],*self.argv],self.nonce)

    def sample(self,allow_unobserved_terminal=False):
        io,life=api();value=life.sample(self.unit)
        life.contract(value,self.unit,self.argv,None,self.nonce)
        invocation=value['InvocationID']
        if self.state.get('invocation_id'):c.require(invocation==self.state['invocation_id'],'Exact service invocation retained')
        self.state['invocation_id']=invocation
        for key in ('MemoryMax','MemoryHigh','MemorySwapMax','TasksMax','WorkingDirectory','AllowedCPUs','CPUAffinity','Nice'):
            c.require(value[key]==self.props[key],'Original worker resource changed: '+key)
        actual_env=dict(item.split('=',1)for item in shlex.split(value['Environment']))
        c.require(all(actual_env.get(k)==v for k,v in self.env.items()),'Original worker environment changed')
        from qwen35_sampler_runtime_service_v12 import duration
        raw=subprocess.check_output(['systemctl','--user','show',self.unit,'--property=RemainAfterExit,LimitFSIZE,RuntimeMaxUSec,TimeoutStopUSec'],text=True,timeout=15)
        limits=life.parse_properties(raw,('RemainAfterExit','LimitFSIZE','RuntimeMaxUSec','TimeoutStopUSec'))
        c.require(limits['RemainAfterExit']=='yes'and limits['LimitFSIZE']==self.props['LimitFSIZE']
            and duration(limits['RuntimeMaxUSec'])==int(self.props['RuntimeMaxSec'])
            and duration(limits['TimeoutStopUSec'])==int(self.props['TimeoutStopSec']),'Original finite service deadlines/file cap and retained exit')
        main=int(value['MainPID']or 0)
        if main:
            life.authenticate(self.state,value,self.unit,self.argv,None,self.nonce)
            observed=io.process_api().proc(main)
            c.require(observed is not None,'Exact actual worker process')
            if self.owner is not None:c.require(observed==self.owner['process'],'Actual worker identity rebound')
            self.owner=dict(unit=self.unit,nonce=self.nonce,process=observed,owner=copy.deepcopy(self.state['owner']))
        else:
            c.require(self.owner is not None or allow_unobserved_terminal,'No successful terminal from an unobserved worker')
            if self.owner is not None:c.require(life.start(self.state['owner']['pid'])!=self.state['owner']['start_ticks'],'Original worker still live')
            c.require(not int(value['ControlPID']or 0)and value['ControlGroup']in ('',life.group_name(self.unit)),
                'No unexpected control process or foreign cgroup')
        return value

    def live(self):
        value=self.sample();c.require(int(value['MainPID']or 0)>0,'Only actual live unit can be excluded from History');return self.owner

    def start(self,guard):
        io,life=api();empty=life.properties(self.unit)
        c.require(empty['LoadState']=='not-found','Fresh exact unit namespace')
        life.empty_group(Path('/sys/fs/cgroup')/life.group_name(self.unit).lstrip('/'))
        self.started=True;self.client=subprocess.Popen(self.command(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,close_fds=True)
        def installed():
            value=life.properties(self.unit)
            if value['LoadState']=='not-found':return None
            self.sample();return self.owner
        io.wait_until(installed,guard,60,message='Actual typed worker owner was not observed')
        out,_=self.client.communicate(timeout=30)
        c.require(len(out)<=65536 and self.client.returncode==0,'Actual systemd installation client failed')

    def terminal(self):
        io,life=api();value=self.sample()
        if int(value['MainPID']or 0)>0 or int(value['ControlPID']or 0)>0 or value['Job']:return None
        c.require(value['ActiveState']in ('inactive','failed')or(value['ActiveState'],value['SubState'])==('active','exited'),'Known retained terminal state')
        keys=('Id','InvocationID','ExecMainPID','ExecMainCode','ExecMainStatus','Result','MainPID','ControlPID','ActiveState','SubState','Job')
        raw=subprocess.check_output(['systemctl','--user','show','--all',self.unit,'--property='+','.join(keys)],text=True,timeout=15)
        props=life.parse_properties(raw,keys)
        c.require(props['InvocationID']==self.state['invocation_id']and props['ExecMainPID']==str(self.state['owner']['pid'])
            and props['MainPID']==props['ControlPID']=='0'and not props['Job'],'Actual retained process exit identity')
        group=life.empty_group(Path('/sys/fs/cgroup')/life.group_name(self.unit).lstrip('/'))
        self.exit=dict(actual_service_exit_observed=True,properties=props,owner=copy.deepcopy(self.owner),group=group,
            installation_client_exit=self.client.returncode if self.client is not None else None)
        return self.exit

    def cleanup(self):
        io,life=api()
        if not self.started:return dict(never_launched=True)
        value=life.properties(self.unit)
        if value['LoadState']=='not-found':
            if self.cleaned is not None:
                c.require(self.owner is None or life.start(self.owner['owner']['pid'])!=self.owner['owner']['start_ticks'],'Captured worker remains gone')
                life.empty_group(Path('/sys/fs/cgroup')/life.group_name(self.unit).lstrip('/'));return self.cleaned
            c.require(self.owner is None,'Observed unit disappeared without retained terminal')
            if self.client is not None and self.client.poll()is None:self.client.communicate(timeout=30)
            c.require(life.properties(self.unit)['LoadState']=='not-found','Installation state changed during cleanup')
            self.cleaned=dict(never_loaded=True,group=life.empty_group(Path('/sys/fs/cgroup')/life.group_name(self.unit).lstrip('/')))
            return self.cleaned
        self.sample(allow_unobserved_terminal=True)
        if self.exit is None and self.owner is not None:
            try:self.terminal()
            except (AssertionError,ValueError):raise
        # Authenticate exact nonce, typed argv and invocation before stop. A live
        # failed worker remains an error, but still must be drained under this lock.
        current=life.sample(self.unit);life.contract(current,self.unit,self.argv,None,self.nonce)
        c.require(current['InvocationID']==self.state['invocation_id'],'Never stop a foreign invocation')
        subprocess.run(['systemctl','--user','stop',self.unit],check=True,capture_output=True,timeout=75)
        drained=life.drained(self.unit,self.state)
        if self.client is not None and self.client.poll()is None:self.client.communicate(timeout=15)
        self.cleaned=dict(complete=True,recorded_exit=copy.deepcopy(self.exit),state=copy.deepcopy(self.state),drained=drained)
        return self.cleaned


def latest_observation():
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    return latest.latest_handoff()


def read_running(spec,phase,admission_ref,admission,owner):
    io,_=api();path=spec['running_paths'][phase];ref=io.ref(path);value=io.read(ref)
    execution=value['minicpm5_retained_replay_current_execution']
    previous_ref=admission['previous']if phase=='pair'else io.ref(spec['running_paths']['pair'])
    previous=io.read(previous_ref)
    expected=c.running_value(previous,phase,execution['tool'],admission_ref,admission,owner,execution['worker_ref'],value['updated_at'],previous_ref)
    c.require(value==expected and io.read(execution['worker_ref'])==owner,'Exact actual root running publication')
    observation=dict(path=path,identity=io.identity(path));c.require(latest_observation()==observation,'Unknown later running handoff')
    return dict(reference=ref,value=value,observation=observation)


def worker(phase,admission_ref):
    io,life=api();admission=io.read(admission_ref,c.LIMITS['spec_bytes']);spec=refs(admission['spec'],admission['review'])
    c.require(admission['lock_identity']==c.LOCK and admission['current_serial_catalog']==spec['current_serial_catalog'],'Same root lease and complete History binding')
    unit=spec['units'][phase];argv=c.phase_argv(phase,admission_ref);own=io.process_api().proc(os.getpid())
    c.require(own['argv']==argv and own['cgroup']=='0::'+life.group_name(unit),'Exact admitted worker argv/cgroup')
    owner=dict(unit=unit,nonce=admission['attempts'][phase],process=own,owner=life.process_owner(os.getpid(),unit,argv))
    current=life.sample(unit);life.contract(current,unit,argv,None,owner['nonce'])
    io.descriptor(admission['outer_owner'],admission['retained_fd'])
    destination=spec['replay_root']if phase=='replay'else str(Path(spec['output_root'])/'pair-work')
    c.require(all(os.environ.get(k)==v for k,v in c.environment(destination).items())
        and not any(os.environ.get(k)for k in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','LD_PRELOAD','LD_LIBRARY_PATH','VLLM_PLUGINS'))
        and sys.pycache_prefix==str(Path(destination)/'empty-pycache')and Path.cwd()==ROOT
        and Path(sys.executable).resolve()==Path(c.PYTHON).resolve()and os.sched_getaffinity(0)=={7}
        and os.getpriority(os.PRIO_PROCESS,0)==19,'Original admitted process environment/resources')
    root=Path(spec['output_root']);owner_ref=io.once(root/(phase+'-owner.json'),owner)
    started=time.monotonic();last=[-1.0];running=[None]
    def guard():
        c.require(time.monotonic()-started<c.LIMITS[phase+'_seconds'],'Original finite worker deadline')
        if time.monotonic()-last[0]>=.25:
            io.descriptor(admission['outer_owner'],admission['retained_fd'])
            c.require(io.process_api().proc(os.getpid())==own,'Actual worker identity unchanged')
            if running[0]is not None:c.require(io.identity(running[0]['reference']['path'])==running[0]['observation']['identity'],'Actual root running publication changed')
            source_unchanged(spec);last[0]=time.monotonic()
        return dict(schema='minicpm5_retained_raw_replay_worker_lease_v1',output_root=spec['replay_root'],
            original_plan_sha256=c.ORIGINAL_PLAN['sha256'],storage_plan_sha256=c.PLAN['sha256'],
            worker_unit=unit,worker_owner=dict(pid=own['pid'],start_ticks=str(own['start_ticks'])),worker_argv=argv,
            outer_owner=admission['outer_owner'],lock_identity=c.LOCK,complete_history=admission['complete_history'])
    release=root/(phase+'-release.json')
    io.wait_until(lambda:True if release.exists()else None,guard,c.LIMITS['ack_seconds'],message='External root actual-tool ACK missing')
    running[0]=read_running(spec,phase,admission_ref,admission,owner)
    released=io.read(io.ref(release));c.require(released==dict(admission=admission_ref,owner=owner_ref,running=running[0]),'Exact root release')
    guard();plan=io.read(c.PLAN)
    if phase=='pair':
        c.require(not {'torch','numpy','transformers','torch_neuronx'}&set(sys.modules),'Pair verifier must remain stdlib only')
        result=transport.verify_pairs(plan,root/'paired-verification.json',guard)
    else:
        import minicpm5_retained_raw_legacy_v1 as legacy
        verification=bounded_ref(root/'paired-verification.json')
        inputs=legacy.Payloads(io.raw(c.ORIGINAL_PLAN['path'],16<<20),plan,verification,guard)
        result=legacy.run_replay(inputs,spec['replay_root'],guard)
    guard();io.loaded_local_sources(spec['source_sha256'])
    result_path=Path(spec['replay_root'])/'caller-result.json'if phase=='replay'else root/'pair-result.json'
    # Frozen legacy audit permits exact .json/.log paths, not .json.publishing.
    # Reuse the accepted durable exclusive writer; partial files fail closed.
    write=transport._write_once if phase=='replay'else io.once
    write(result_path,dict(phase=phase,complete=True,owner=owner,admission=admission_ref,
        result=result,self_exit_claimed=False,full_model_qualified=False,deletion_authorized=False))
    return 0


def outer(spec_ref,review_ref):
    io,life=api();spec=refs(spec_ref,review_ref);root=Path(spec['output_root'])
    c.require(not root.exists()and not Path(spec['replay_root']).exists()
        and all(not Path(p).exists()for p in [*spec['running_paths'].values(),*spec['publication_paths'].values(),spec['terminal_handoff_path']]),'Fresh finite caller namespaces')
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
    lock=io.RetainedLock();lock.acquire();started=time.monotonic();stopped=[False];handlers={}
    phases=[];history=None;admission_ref=None;running=None;failure=None;terminals=[];last=[-1.0]
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def guard(force=False):
        lock.check();c.require(not stopped[0]and time.monotonic()-started<c.LIMITS['outer_seconds'],'Outer finite/interruption guard')
        if running is not None:c.require(io.identity(running['reference']['path'])==running['observation']['identity'],'Running handoff changed')
        if force or time.monotonic()-last[0]>=.5:
            storage(spec,reserve=True);source_unchanged(spec)
            if running is not None:c.require(latest_observation()==running['observation'],'Unknown later serial handoff')
            last[0]=time.monotonic()
    try:
        guard();io.process_api().active_heavy()
        history=History(spec,spec_ref,review_ref,spec['units'],guard).resolve()
        guard(True);pair=history.physical_pair(Authority(history,guard,lambda:[]));guard(True)
        freshness=history.freshness();history.unchanged();guard(True)
        root.mkdir(mode=0o700);(root/'pair-work').mkdir();(root/'pair-work/tmp').mkdir(mode=0o700);(root/'pair-work/empty-pycache').mkdir()
        history_refs=save_history(root,dict(physical_pair=pair,current_serial_catalog=spec['current_serial_catalog'],
            history_overlay=spec['history_overlay'],resolved_plan_sha256=history.resolved['plan_sha256'],
            metadata_identities=history.meta.identities,freshness=freshness))
        admission=dict(schema='minicpm5_retained_replay_outer_admission_v1',run_name=spec['run_name'],spec=spec_ref,review=review_ref,
            outer_owner=lock.owner,retained_fd=lock.fd,lock_identity=c.LOCK,initial_descriptor=lock.initial,
            current_serial_catalog=spec['current_serial_catalog'],history_overlay=spec['history_overlay'],
            previous=freshness['latest_handoff'],complete_history=history_refs,
            attempts={p:os.urandom(16).hex()for p in ('pair','replay')},automatic_successor=False,deletion_authorized=False)
        guard(True);admission_ref=io.once(root/'admission.json',admission);guard(True)
        for phase in ('pair','replay'):
            guard(True);history.unchanged();io.process_api().active_heavy()
            stage=Phase(spec,phase,admission_ref,admission['attempts'][phase]);phases.append(stage);stage.start(guard)
            owner_ref=io.wait_until(lambda:io.ref(root/(phase+'-owner.json'))if(root/(phase+'-owner.json')).exists()else None,
                guard,60,message='Actual worker identity ACK missing')
            c.require(io.read(owner_ref)==stage.owner,'Independent typed owner equals worker ACK')
            print(json.dumps(dict(event='mini_retained_actual_running_ack_required',phase=phase,admission=admission_ref,
                worker=stage.owner,worker_ref=owner_ref,running_handoff_path=spec['running_paths'][phase],
                actual_external_tool_session_required=True,payload_io_released=False)),flush=True)
            # Our old running observation is replaced only by the exact next
            # phase publication after its actual owner is independently captured.
            def wait_guard():
                lock.check();storage(spec,reserve=True)
                c.require(not stopped[0]and time.monotonic()-started<c.LIMITS['outer_seconds'],'Outer remains admitted while waiting ACK')
                if running is not None and not Path(spec['running_paths'][phase]).exists():
                    c.require(latest_observation()==running['observation'],'Unknown handoff before next ACK')
            io.wait_until(lambda:True if Path(spec['running_paths'][phase]).exists()else None,wait_guard,c.LIMITS['ack_seconds'],message='Actual external root running handoff missing')
            running=read_running(spec,phase,admission_ref,admission,stage.owner)
            guard(True);io.once(root/(phase+'-release.json'),dict(admission=admission_ref,owner=owner_ref,running=running));guard(True)
            terminal=io.wait_until(stage.terminal,guard,c.LIMITS[phase+'_seconds'],message='Actual worker terminal missing',interval=.5)
            drained=stage.cleanup();terminals.append(drained)
            io.once(root/(phase+'-terminal.json'),drained)
            props=terminal['properties'];c.require(props['ExecMainCode']=='1'and props['ExecMainStatus']=='0'and props['Result']=='success','Actual worker scientific/IO failure retained')
            result_path=Path(spec['replay_root'])/'caller-result.json'if phase=='replay'else root/'pair-result.json'
            result=io.read(io.ref(result_path))
            c.require(result['complete']is True and result['admission']==admission_ref and result['owner']==stage.owner,'Complete exact worker result')
            guard(True);history.unchanged()
            guard(True);after=history.physical_pair(Authority(history,guard,lambda:[]));guard(True)
            (root/(phase+'-post-history')).mkdir();save_history(root/(phase+'-post-history'),after)
        source_unchanged(spec);guard(True)
    except BaseException as error:failure=error
    finally:
        def cleanup():
            lock.check()
            for stage in reversed(phases):stage.cleanup()
        def drained():
            lock.check();rows=[life.drained(s.unit,s.state)for s in phases]
            return dict(physically_drained=True,all_phase_drains=rows)
        proof=retain_until_drained(cleanup,drained,
            lambda value:print(json.dumps(dict(event='mini_retained_cleanup_blocked',serial_lock_retained=True,details=value)),flush=True),
            lambda:time.sleep(1))
        try:
            if admission_ref is not None:
                io.once(root/'outer-terminal.json',dict(schema='minicpm5_retained_replay_outer_terminal_v1',admission=admission_ref,
                    outer_owner=lock.owner,phase_states=[dict(phase=s.phase,unit=s.unit,nonce=s.nonce,state=s.state,
                        owner=s.owner,exit=s.exit)for s in phases],all_phase_drains=proof['all_phase_drains'],
                    full_owned_scope_drained=True,failed=failure is not None,
                    failure=None if failure is None else dict(type=type(failure).__name__,message=str(failure)[:2048]),
                    last_running=running,self_exit_claimed=False,actual_external_exit_pending=True,
                    deletion_authorized=False,automatic_successor=False,full_model_qualified=False))
        finally:
            lock.release(proof['physically_drained'])
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if failure is not None:raise failure
    return 0


def acknowledge(request_ref):
    """Explicit external root call; supplied actual tool session is never inferred."""
    io,_=api();request=io.read(request_ref)
    c.require(set(request)=={'admission','phase','worker_ref','actual_tool','updated_at'},'Exact actual-tool ACK request')
    admission_ref=request['admission'];admission=io.read(admission_ref,c.LIMITS['spec_bytes'])
    spec=refs(admission['spec'],admission['review']);phase=request['phase'];c.require(phase in ('pair','replay'),'Exact phase')
    io.descriptor(admission['outer_owner'],admission['retained_fd'])
    worker=io.read(request['worker_ref']);c.require(worker['unit']==spec['units'][phase],'Actual phase owner')
    c.require(request['worker_ref']['path']==str(Path(spec['output_root'])/(phase+'-owner.json')),'Exact caller owner evidence')
    c.require(io.process_api().proc(worker['process']['pid'])==worker['process'],'External root must see actual live worker')
    observed=Phase(spec,phase,admission_ref,admission['attempts'][phase]).live()
    c.require(observed==worker,'Independent typed argv/nonce/invocation/environment owner capture')
    previous_ref=admission['previous']if phase=='pair'else io.ref(spec['running_paths']['pair'])
    previous=io.read(previous_ref)
    c.require(latest_observation()==dict(path=previous_ref['path'],identity=io.identity(previous_ref['path'])),'Exact latest predecessor before publication')
    value=c.running_value(previous,phase,request['actual_tool'],admission_ref,admission,worker,request['worker_ref'],request['updated_at'],previous_ref)
    return io.once(Path(spec['running_paths'][phase]),value)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('outer','pair','replay','ack'),required=True)
    for name in ('spec','spec-sha256','review','review-sha256','admission','admission-sha256','request','request-sha256'):parser.add_argument('--'+name)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B and assertions')
    if args.mode=='outer':
        os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
        return outer(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
    if args.mode=='ack':print(json.dumps(acknowledge(dict(path=args.request,sha256=args.request_sha256))),flush=True);return 0
    return worker(args.mode,dict(path=args.admission,sha256=args.admission_sha256))


if __name__=='__main__':raise SystemExit(main())
