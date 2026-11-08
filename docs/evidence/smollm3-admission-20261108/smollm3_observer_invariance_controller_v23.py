"""Future bounded24-service supervisor; execution requires external lock owner.

This v20 source successor retains the v19 scientific matrix and thresholds.
It needs a fresh root-owned namespace/spec migration before any runtime use.
This module is source only until root binds final catalog and independently
accepts the exact spec. Importing it does not create directories or services.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import socket
import sys
import time
import urllib.request

import smollm3_observer_invariance_lifecycle_v21 as l
import smollm3_observer_invariance_protocol_v17 as p
import smollm3_authorized_stop_transition_v1 as stop_transition


def stop_absence_observation(state, value, origin):
    intent=state['intent'];path=p.directory(intent['run_id'])/(intent['kind']+'-stop-intent.json')
    p.require(path.exists(), 'Owned service vanished without durable stop authorization')
    observation=stop_transition.authorized_absent_sample(l,state,value,l.read(path),origin)
    state.setdefault('stop_absence_observations',[]).append(observation)
    state['stop_authenticated']=True
    return observation


def authenticate_parent(state, initial=None, allow_stop_absence=False):
    intent=state['intent'];root=p.directory(intent['run_id'])
    def classified(value, origin):
        if value['LoadState']=='not-found' and allow_stop_absence:
            raise stop_transition.AuthorizedStopAbsent(stop_absence_observation(state,value,origin))
        return stop_transition.authenticate_loaded_sample(l,intent,value,state.get('claim'))
    # Classify and authenticate the SAME observation. Initial props from the
    # stop caller must never be discarded in favor of a second unclassified read.
    value=classified(l.props(intent['unit']) if initial is None else initial,'initial')
    path=root/(intent['kind']+'-service-binding.json')
    binding=l.bind_service(path,intent,value,state.get('owner'),state.get('claim'))
    state['claim']=binding['invocation_id'];state['binding']=binding
    def refresh():
        current=classified(l.props(intent['unit']),'exit-transition-refresh')
        return l.verify_binding(intent,current,binding)
    if state.get('owner')is not None:
        value=l.owned_live_or_retained_exit(state['owner'],value,refresh)
    return l.verify_binding(intent,value,binding)


def successful_exit(state, value):
    p.require(state.get('owner') is not None and l.terminal(value), 'Retained actual owner exit')
    p.require(value['ExecMainPID'] == str(state['owner']['pid']) and value['InvocationID'] == state['owner']['invocation_id'], 'Retained exit identity')
    return value['Result'] == 'success' and value['ExecMainCode'] == '1' and value['ExecMainStatus'] == '0'


def physical(state, stop_context=False):
    name = state['intent']['unit']; samples = []
    for index in range(2):
        value = l.props(name)
        if stop_context:
            if value['LoadState']=='not-found':
                stop_absence_observation(state,value,'physical-drain-sample-'+str(index))
            else:
                stop_transition.authenticate_loaded_sample(l,state['intent'],value,state.get('claim'))
                p.require(state.get('binding')is not None,'Retained service binding required for owned drain')
                l.verify_binding(state['intent'],value,state['binding'])
                if state.get('owner')is not None:
                    l.verify_retained_binding(state['intent'],state['owner'],state['binding'])
        p.require(l.terminal(value) and value['Id'] == name+'.service', 'Physical unit terminal')
        p.require(not value['ControlGroup'] or value['ControlGroup'] == l.GROUP_PREFIX+name+'.service', 'Physical group namespace')
        if state.get('owner'): l.gone(state['owner'])
        group = l.cgroup_inventory(l.GROUP_PREFIX+name+'.service', require_empty=True)
        samples.append(dict(properties=value, group=group))
        if index == 0: time.sleep(.05)
    p.require(samples[0] == samples[1], 'Stable recursive physical drain')
    return dict(drained=True, physical_only=True, success_proven=False, samples=samples)


def stopped_physical(state):
    result=physical(state,stop_context=True)
    return dict(result,authorized_stop=state.get('stop_authenticated',False),
                original_owner=state.get('owner'),claim=state.get('claim'),success_proven=False,
                stop_absence_observations=state.get('stop_absence_observations',[]))


def stop_and_drain(state):
    """Handle authorized disappearance in every sample; never infer exit success."""
    intent = state['intent']; root = p.directory(intent['run_id']); kind = intent['kind']
    value = l.props(intent['unit'])
    if value['LoadState']=='not-found' and not state.get('claim'):
        p.require(state.get('owner')is None and state.get('binding')is None,
                  'Unclaimed missing service cannot retain an owner/binding')
        return physical(state)  # Failed launch before any observed service claim only.
    try:
        value = authenticate_parent(state,initial=value,allow_stop_absence=True)
        group = l.cgroup_inventory(l.GROUP_PREFIX+intent['unit']+'.service')
        if not l.terminal(value) or group['pids']:
            path = root/(kind+'-stop-intent.json')
            record = dict(intent=intent, owner=state.get('owner'), claim=value['InvocationID'], binding=state['binding'], properties=value,
                          authorized_stop=True, server_normal_shutdown=kind=='server', successful_client_exit=False)
            if not path.exists(): l.record(path, record)
            else: stop_transition.retained_stop_authorization(l,state,l.read(path))
            state['stop_authenticated'] = True
            authenticate_parent(state,allow_stop_absence=True)
            l.command(['systemctl','--user','stop','--no-block',intent['unit']])
            deadline = time.monotonic()+75; forced = False
            while True:
                # This returned value is the actual classified/verified sample,
                # including any bounded exit refresh; no stale outer props value.
                value = authenticate_parent(state,allow_stop_absence=True)
                group = l.cgroup_inventory(l.GROUP_PREFIX+intent['unit']+'.service')
                if l.terminal(value) and not group['pids']: break
                if not forced and time.monotonic() > deadline-15:
                    value=authenticate_parent(state,allow_stop_absence=True)
                    group=l.cgroup_inventory(l.GROUP_PREFIX+intent['unit']+'.service')
                    if l.terminal(value) and not group['pids']:break
                    l.command(['systemctl','--user','kill','--kill-who=all','--signal=KILL',intent['unit']]); forced = True
                p.require(time.monotonic() < deadline, 'Bounded stop/drain deadline'); time.sleep(.2)
    except stop_transition.AuthorizedStopAbsent:
        # Only the typed, fully authenticated authorization transition is handled.
        # Identity, resource, filesystem and systemctl errors still propagate.
        pass
    return stopped_physical(state)


def state_from_disk(run_id, kind):
    root = p.directory(run_id); intent_path = root/(kind+'-launch.json')
    if not intent_path.exists(): return None
    state = dict(intent=l.read(intent_path), owner=None, claim=None)
    owner = root/(kind+'-owner.json')
    if owner.exists():
        state['owner'] = l.read(owner); state['claim'] = state['owner']['invocation_id']
    binding=root/(kind+'-service-binding.json')
    if binding.exists():
        state['binding']=l.read(binding)
        p.require(state['claim']is None or state['claim']==state['binding']['invocation_id'],'Saved owner/binding invocation')
        state['claim']=state['binding']['invocation_id']
    return state


def cleanup_all():
    proofs = []; failures = []
    for run in reversed(p.matrix()):
        for kind in ('client','server'):
            state = None
            try:
                state = state_from_disk(run['run_id'],kind)
                if state is None: continue
                proofs.append(dict(run_id=run['run_id'],kind=kind,proof=stop_and_drain(state)))
            except BaseException as error:
                if isinstance(error,l.DeadlineExceeded):raise
                failure=dict(run_id=run['run_id'],kind=kind,error_type=type(error).__name__,error=str(error)[:4096])
                failures.append(failure)
                try:
                    p.require(state is not None,'Cannot authenticate damaged launch/owner records')
                    proofs.append(dict(run_id=run['run_id'],kind=kind,proof=physical(state),identity_error=failure))
                except BaseException as physical_error:
                    if isinstance(physical_error,l.DeadlineExceeded):raise
                    failure['physical_error']=str(physical_error)[:4096]
    return proofs, failures


class Supervisor:
    def __init__(self, args, spec, plan, base, envelope):
        self.args=args; self.spec=spec; self.plan=plan; self.base=base; self.envelope=envelope
        self.active=[]; self.producers={}; self.current=None;self.seen={};self.pending_spawn={}
        self.matrix_started=envelope['matrix_started_monotonic']

    def poll(self, run_id):
        p.require(time.monotonic() < self.matrix_started+86400, 'Matrix24h deadline')
        l.lock_proof(self.envelope['external_owner'], self.envelope['lock'])
        l.same_process(self.envelope['external_owner'])
        l.verify_stat_closure(self.envelope['source_stats']); l.verify_stat_closure(self.envelope['payload_stats'])
        for name,wanted in self.spec['canonical_sha256'].items():
            p.require(l.digest_bytes(p.bounded_source(p.ROOT/name))==wanted,'Canonical changed')
        if self.current:
            p.require(time.monotonic() < self.current['run_started_monotonic']+7200,'One shared run7200s deadline')
        l.monitor(run_id)
        for state in self.active:
            if state['intent']['run_id'] == run_id and not state.get('drained'):
                self.acknowledge(state)

    def acknowledge(self,state):
        value=authenticate_parent(state)
        if state.get('owner') is None or l.terminal(value):return
        intent=state['intent'];run_id=intent['run_id'];kind=intent['kind'];root=p.directory(run_id)
        actual_group=l.cgroup_inventory(l.GROUP_PREFIX+intent['unit']+'.service')
        for pid in actual_group['pids']:
            try:actual=l.process(pid)
            except FileNotFoundError:continue
            identity=(run_id,kind,pid,actual['start_ticks'])
            if identity not in self.seen:
                ancestry=[] if pid==state['owner']['pid'] else l.parent_chain(actual,state['owner'],run_id)
                witness=dict(process=actual,unit=intent['unit'],invocation_id=value['InvocationID'],run=p.run_row(run_id),
                             mode=p.run_row(run_id)['mode'],attempt_id=intent['attempt_id'],parent_chain=ancestry)
                l.record(root/('process-seen-'+kind+'-'+str(pid)+'.json'),witness);self.seen[identity]=witness
            if actual['argv']==[l.title.TITLE.decode()]:
                p.require((run_id,kind,pid) in self.producers, 'Unknown title cannot evade initial spawn admission')
            if '--multiprocessing-fork'in actual['argv']:
                key=(run_id,kind,pid)
                self.pending_spawn.setdefault(key,time.monotonic())
                p.require(key in self.producers or time.monotonic()<self.pending_spawn[key]+60,'Every actual model spawn must authenticate mode')
        for path in sorted(root.glob('hello-'+kind+'-*.json')):
            hello=l.read(path);pid=hello['process']['pid'];key=(run_id,kind,pid)
            if key in self.producers:
                ack=self.producers[key]
                try:l.title_runtime.acknowledge(ack['permit'],self.current['supervisor'])
                except FileNotFoundError:pass
                continue
            try:producer=l.authenticate_hello(hello,intent,state['owner'],value,self.base,self.producers)
            except l.PendingSpawnWitness:
                self.pending_spawn.setdefault(key,time.monotonic())
                p.require(time.monotonic()<self.pending_spawn[key]+60,'Spawn witness original ACK deadline')
                continue
            p.require(hello['spec_sha256']==self.args.spec_sha,'Hello exact execution spec')
            count=sum(key[0]==run_id and key[1]=='server' for key in self.producers)
            p.require(kind!='server' or count<8,'Parent/engine process cap')
            admission=dict(self.current,service_parent=state['owner'])
            permit=l.permit_for(self.plan,self.base,run_id,kind,producer,admission,self.spec['runtime_source_sha256'])
            ack=dict(hello_sha256=p.digest(hello),producer=producer,permit=permit,admission=admission,intent=intent)
            l.record(root/('permit-'+kind+'-'+str(pid)+'.json'),ack)
            self.producers[key]=ack

    def launch(self,run_id,kind):
        root=p.directory(run_id);nonce=self.spec['attempt_ids'][run_id]
        intent=l.worker_command(self.args.spec,self.args.spec_sha,self.args.review,self.args.review_sha,self.base,run_id,kind,nonce)
        p.require(l.props(intent['unit'])['LoadState']=='not-found','Fresh named service only')
        l.record(root/(kind+'-launch.json'),intent)
        state=dict(intent=intent,owner=None,claim=None);self.active.append(state)
        l.command(intent['command'],timeout=30)
        deadline=min(self.current['run_started_monotonic']+7200,time.monotonic()+45)
        while True:
            value=authenticate_parent(state)
            if value['MainPID']!='0':
                try:actual=l.process(int(value['MainPID']))
                except FileNotFoundError:actual=None
                if actual and actual['argv']==intent['argv']:
                    p.require(actual['cgroup']=='0::'+l.GROUP_PREFIX+intent['unit']+'.service','Live owner cgroup')
                    state['owner']=dict(actual,unit=intent['unit'],invocation_id=value['InvocationID'],attempt_id=nonce)
                    l.record(root/(kind+'-owner.json'),state['owner']);self.acknowledge(state);return state
            p.require(not l.terminal(value) and time.monotonic()<deadline,'Service failed before owner authentication')
            l.monitor(run_id);time.sleep(.1)

    def ready(self,server):
        run_id=server['intent']['run_id']
        while True:
            self.poll(run_id);p.require(not l.terminal(authenticate_parent(server)),'Server exited before readiness')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8010/v1/models',timeout=2)as response:
                    raw=response.read(65537);p.require(len(raw)<=65536,'Ready response cap')
                    value=p.json.loads(raw)
                p.require([x['id']for x in value['data']]==['smollm3-3b'],'Exact served model')
                l.record(p.directory(run_id)/'server-ready.json',dict(owner=server['owner'],response=value,monotonic=time.monotonic()))
                return
            except (OSError,ValueError):time.sleep(.2)

    def wait_client(self,client,server):
        run_id=client['intent']['run_id'];root=p.directory(run_id)
        while True:
            self.poll(run_id)
            p.require(not l.terminal(authenticate_parent(server)),'Server must survive client')
            value=authenticate_parent(client)
            if l.terminal(value):
                success=successful_exit(client,value)
                proof=physical(client);client['drained']=True
                l.record(root/'client-terminal.json',dict(passed=success,original_properties=value,owner=client['owner'],drain=proof))
                p.require(success,'Client original exit failure; preserve partial evidence')
                return
            time.sleep(.2)

    def execute(self):
        import smollm3_observer_invariance_audit_v22 as audit
        captures={};evidence={};exports={}
        for row in p.matrix():
            run_id=row['run_id'];root=p.directory(run_id)
            p.require(not root.exists() and not p.temporary(run_id).exists(),'Fresh run and IPC dirs')
            root.mkdir(mode=0o700);p.temporary(run_id).mkdir(mode=0o700)
            for name in ('compile-forbidden','cache-forbidden'):(root/name).mkdir(mode=0o700)
            self.current=dict(self.envelope,run_started_monotonic=time.monotonic(),run=row,
                              proofs=dict(self.envelope['proofs'],previous_run_fully_drained=True,fresh_run_namespace=True))
            self.current['directory_identities']={str(path):l.directory_identity(path)for path in(p.DEST,l.CONTROL,l.CONTROL/'source-maps',l.CONTROL/'catalog-maps',root,p.temporary(run_id))}
            l.record(root/'run-admission.json',self.current)
            with socket.socket()as s:p.require(s.connect_ex(('127.0.0.1',8010))!=0,'Port8010 occupied')
            server=self.launch(run_id,'server');self.ready(server)
            self.current['live_server']=dict(intent=server['intent'],owner=server['owner'])
            client=self.launch(run_id,'client');self.wait_client(client,server)
            drain=stop_and_drain(server);server['drained']=True
            l.record(root/'server-terminal.json',drain)
            capture,proof,export=audit.audit_run(self.base,run_id,write_export=True)
            captures[run_id]=capture;evidence[run_id]=proof
            if export is not None:exports[run_id]=export
            self.poll(run_id)
        from smollm3_observer_invariance_analysis_v17 import compare_matrix
        comparison=compare_matrix(self.base,captures,evidence,exports)
        l.record(l.CONTROL/'matrix-result.json',comparison)
        return comparison


def main():
    parser=argparse.ArgumentParser()
    for name in ('spec','spec-sha','review','review-sha'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();p.require(sys.dont_write_bytecode and __debug__,'Use system Python -B')
    spec,plan,base,_=l.load_spec(args.spec,args.spec_sha)
    accepted=l.reference(dict(path=args.review,sha256=args.review_sha))
    p.require(accepted['reviewer']=='/root'and accepted['passed']is True and accepted['independent_review']is True and
              accepted['spec_sha256']==args.spec_sha and accepted['runtime_source_sha256']==spec['runtime_source_sha256'],'Independent root acceptance')
    deadline=time.monotonic()+60
    while not(l.CONTROL/'supervisor-owner-bundle.json').exists():
        p.require(time.monotonic()<deadline,'External supervisor-owner ACK timeout');time.sleep(.1)
    envelope=l.read(l.CONTROL/'supervisor-owner-bundle.json')
    p.require(l.process(os.getpid())==envelope['supervisor'],'Actual supervisor process binding')
    l.lock_proof(envelope['external_owner'],envelope['lock'])
    supervisor=Supervisor(args,spec,plan,base,envelope)
    def interrupted(signum,frame):raise RuntimeError('Supervisor signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
    original=None;cleanup_errors=[];result=None
    try:result=supervisor.execute()
    except BaseException as error:original=dict(type=type(error).__name__,message=str(error)[:4096])
    finally:
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
        proofs,cleanup_errors=cleanup_all()
        terminal=dict(protocol_completed=result is not None,original_error=original,cleanup_errors=cleanup_errors,
                      drain_proofs=proofs,api_invariance_passed=None if result is None else result['observer_api_invariance_passed'],
                      external_audit_required=True,full_model_qualified=False,new_speedup_claim=False)
        l.record(l.CONTROL/'supervisor-terminal.json',terminal)
    if original or cleanup_errors:raise SystemExit(1)


if __name__=='__main__':main()
