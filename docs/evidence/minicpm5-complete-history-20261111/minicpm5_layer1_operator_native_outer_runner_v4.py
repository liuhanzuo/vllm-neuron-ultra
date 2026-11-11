"""Actual outer main path; requires reviewed concrete inputs and root FIFO release.

Source-only delivery. No operation executes on import. External model/recorder
exit and independent post-exit raw audit remain outside this process.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import sys
import time
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
from research import minicpm5_layer1_operator_native_process_adapter_v7 as r
from research import minicpm5_layer1_operator_native_outer_support_v3 as support
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
life=r.life;p=r.p;require=r.require;SELF=str(Path(__file__).resolve())


def reference(path):
    path=str(path);return dict(path=path,sha256=hashlib.sha256(r.sources.text(path)).hexdigest())


def receive_root_release(spec_ref,review_ref,*,fd=0):
    """Actual external root observes owner/session and answers this fresh challenge."""
    require(stat.S_ISFIFO(os.fstat(fd).st_mode),'Root release requires inherited stdin FIFO')
    owner=p.proc(os.getpid());require(owner is not None,'Actual pre-acquire outer identity')
    nonce=os.urandom(32).hex()
    challenge=dict(schema='minicpm5_native_outer_release_challenge_v1',owner=owner,spec=spec_ref,review=review_ref,nonce=nonce)
    print(json.dumps(challenge),flush=True)
    end=time.monotonic()+120;raw=bytearray()
    while True:
        require(p.proc(owner['pid'])==owner and time.monotonic()<end,'Original outer remains live during bounded root release')
        ready,_,_=select.select([fd],[],[],.1)
        if not ready:continue
        block=os.read(fd,4097-len(raw));raw.extend(block);require(len(raw)<=4096,'Bounded root release packet')
        if not block:break
    require(raw.endswith(b'\n')and raw.count(b'\n')==1,'One complete root release followed by EOF')
    value=r.result_core.json_object(bytes(raw))
    require(type(value)is dict and set(value)=={'schema','reviewer','challenge','tool','handoff_path'}
        and value['schema']=='minicpm5_native_outer_release_v1'and value['reviewer']=='/root'
        and value['challenge']==challenge,'Root response binds fresh nonce and actual outer/spec/review')
    r.execution.tool(value['tool']);r.execution.handoff_path(value['handoff_path'])
    require(not Path(value['handoff_path']).exists(),'Fresh externally selected running handoff')
    return value


def service_command(contract,admission_ref,admission):
    return ['systemd-run','--user','--quiet','--unit='+contract['unit'],'--service-type=exec',
        *[arg for key,value in life.PROPS.items()for arg in ('-p',key+'='+value)],
        '-p','StandardOutput=append:'+contract['output_root']+'/runtime.log','-p','StandardError=inherit',
        '/usr/bin/env','-i',*[key+'='+value for key,value in r.environment(admission).items()],
        *life.command(contract,'coordinator',admission_ref)]


def service_identity(contract,admission_ref,admission,value,owner=None):
    require(value['LoadState']=='loaded'and value['Id']==contract['unit']+'.service'and p.invocation(value['InvocationID']),
        'Actual exact loaded native service invocation')
    cmd=service_command(contract,admission_ref,admission)
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[' '.join(cmd[cmd.index('/usr/bin/env'):])]
        and value['ExecStopPost']=='','Exact owned original native command and no extra cleanup')
    for key,wanted in life.PROPS.items():
        if key in ('RuntimeMaxSec','TimeoutStopSec'):
            require(p.duration_seconds(value[key.replace('Sec','USec')])==int(wanted),'Original native duration '+key)
        else:require(value[key]==wanted,'Original native resource '+key)
    require(value['ControlGroup']in('',p.GROUP_PREFIX+contract['unit']+'.service'),'Exact owned native cgroup')
    if owner is not None:
        require(value['InvocationID']==owner['invocation_id']and value['ExecMainPID']==str(owner['pid']),
            'Never operate on replaced service invocation or PID')
    return value


def resolve(spec_ref,review_ref,lease,metadata):
    """Actual complete inherited metadata/source authentication and physical pair."""
    guard=lambda:lease.check('outer.resolve');spec=metadata.read(spec_ref);review=metadata.read(review_ref)
    require(set(spec)=={'schema','overlay','runner','unit','lock_policy'}and spec['schema']=='minicpm5_native_process_spec_v1'
        and spec['lock_policy']==r.LOCK_POLICY,'Exact original small native spec')
    require(review['schema']=='minicpm5_native_process_source_review_v1'and review['reviewer']=='/root'
        and review['passed']is True and review['blockers']==[]and review['spec']==spec_ref
        and review['overlay']==spec['overlay']and review['lock_policy']==r.LOCK_POLICY,
        'Independent exact concrete source review before any native source IO')
    recovered=r.completion.completed(r.completion.BINDING,metadata.read,guard,lambda path,sha:r.sources.text(path,sha))
    resolved=r.overlays.resolve(metadata.read(spec['overlay']),recovered,metadata.read,guard)
    require(resolved['plan_sha256']==review['resolved_plan_sha256'],'Exact reviewed full resolved plan')
    contract=life.build_contract(resolved,spec['overlay'],spec['runner'],spec['unit'])
    for path in (SELF,str(Path(support.__file__).resolve())):
        require(resolved['plan']['source_sha256'].get(path)==hashlib.sha256(r.sources.text(path)).hexdigest(),
            'Actual outer and native-auth compatibility source pinned in complete plan')
    for pin in (r.LOCK_POLICY['descriptor_source'],r.LOCK_POLICY['proc_source']):
        require(resolved['plan']['source_sha256'].get(pin['path'])==pin['sha256'],'Original descriptor and proc source pins')
    partition=r.sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],metadata.read)
    r.sources.verify_text_sources(partition,guard);r.verify_loaded(resolved['plan'],guard)
    lease.scope=resolved['current_scope'];lease.metadata=metadata
    require(lease.scope['lock_identity']==r.ORIGINAL_LOCK,'Original lock across complete recovered lineage')
    authority=r.current_authority.Authority(r,lease)
    pair=r.current_physical.collect_pair(lease.scope,guard,authority)
    required={row['unit']for row in lease.scope['nemotron']['history']['rows']};known=set(lease.scope['complete_known_units'])
    require(r.completion.physical.validate_pair(pair,required,lease.scope['owners'],r.ORIGINAL_LOCK,known_units=known)==known,
        'Complete original paired physical drain before native authentication')
    return spec,review,resolved,contract,partition,pair


class OwnedService:
    """Explicit lifetime state, including a failed/ambiguous systemd-run call."""
    def __init__(self,contract,admission_ref,admission,metadata):
        self.contract,self.admission_ref,self.admission,self.metadata=contract,admission_ref,admission,metadata
        self.attempted=False;self.owner=None;self.owner_ref=None;self.running_ref=None
        self.launch_process=None;self.launch_returncode=None;self.drain=None
    def launch(self,guard):
        guard();require(not self.attempted and p.properties(self.contract['unit'])['LoadState']=='not-found',
            'One launch of fresh exact native service')
        require(self.metadata.read(self.admission_ref)==self.admission,'Unchanged original admitted launch')
        intent=self.metadata.read(self.admission['current_execution']['intent']);r.execution.validate_intent(intent,self.admission)
        self.attempted=True
        # Original command helper reaps its systemd-run helper before returning,
        # including error/timeout. This object retains attempted ownership on failure.
        p.command(service_command(self.contract,self.admission_ref,self.admission),timeout=30)
        self.launch_returncode=0
    def acknowledge(self,guard):
        require(self.attempted and self.launch_returncode==0,'Service launch helper actually exited zero')
        deadline=time.monotonic()+60
        while True:
            guard();value=service_identity(self.contract,self.admission_ref,self.admission,r.native_service_properties(self.contract['unit']))
            pid=int(value['MainPID']);actual=p.proc(pid)if pid else None
            if actual is not None:
                require(actual['argv']==life.command(self.contract,'coordinator',self.admission_ref)
                    and actual['cgroup']=='0::'+p.GROUP_PREFIX+self.contract['unit']+'.service',
                    'Actual coordinator exec and original owned service cgroup')
                self.owner=dict(actual,unit=self.contract['unit'],invocation_id=value['InvocationID'])
                life.coordinator_process(self.contract,self.admission_ref,self.owner)
                service_identity(self.contract,self.admission_ref,self.admission,value,self.owner)
                r.atomic_once(Path(self.contract['output_root'])/'coordinator-owner.json',self.owner)
                self.owner_ref=reference(Path(self.contract['output_root'])/'coordinator-owner.json');return self.owner
            require(not p.terminal(value)and time.monotonic()<deadline,'Actual coordinator acknowledgement before deadline');time.sleep(.05)
    def publish_running(self,intent,guard):
        require(self.owner is not None and self.running_ref is None,'One running handoff after actual coordinator ACK')
        guard();fresh=r.completion.latest.require_latest_terminal(
            dict(latest_completion=self.admission['latest_completion'],serial_predecessors=self.admission['serial_predecessors'],
                source_sha256=self.source_pins),self.metadata.read,guard)
        require(fresh==intent['pre_admission_freshness'],'No intervening handoff after original pre-admission freshness')
        previous=self.metadata.read(fresh['latest_handoff'])
        value=r.execution.expected_running(previous,datetime.now(timezone.utc).isoformat(),self.admission_ref,self.admission,
            intent,self.owner_ref,self.owner)
        r.execution.validate_running(value,previous,self.admission_ref,self.admission,intent,self.owner_ref,self.owner,
            p.proc(self.admission['outer_owner']['pid']),p.proc(self.owner['pid']))
        path=Path(self.admission['current_execution']['handoff_path']);r.atomic_once(path,value);self.running_ref=reference(path)
        return self.running_ref
    def monitor(self,lease,ports):
        deadline=time.monotonic()+7230
        while True:
            lease.check('outer.monitor');life.storage(self.contract,ports)
            entries,root,writers=ports.inventory();life.usage(self.contract,ports.resolved,entries,root,writers=writers)
            value=service_identity(self.contract,self.admission_ref,self.admission,r.native_service_properties(self.contract['unit']),self.owner)
            if p.terminal(value):return value
            require(p.proc(self.owner['pid'])=={key:self.owner[key]for key in life.PROCESS_KEYS},'Live original coordinator identity')
            require(time.monotonic()<deadline,'Original native outer monitor deadline');time.sleep(.5)
    def cleanup(self):
        if not self.attempted:return
        value=r.native_service_properties(self.contract['unit'])
        if value['LoadState']=='not-found':return
        service_identity(self.contract,self.admission_ref,self.admission,value,self.owner)
        p.command(['systemctl','--user','stop',self.contract['unit']],timeout=30)
    def verify_drained(self):
        value=r.native_service_properties(self.contract['unit'])
        if value['LoadState']=='loaded':
            require(self.attempted,'Never claim preexisting service ownership')
            service_identity(self.contract,self.admission_ref,self.admission,value,self.owner)
            require(p.terminal(value),'Actual owned service terminal without Job')
        else:
            require(value['LoadState']=='not-found'and not value['InvocationID']
                and value['MainPID']==value['ControlPID']==value['ExecMainPID']=='0'and not value['Job'],
                'Fully absent native service with no residual PID or Job')
        owners=[]
        if self.owner is not None:owners.append(self.owner)
        path=Path(self.contract['output_root'])/'native-owner.json'
        if path.exists():
            require(self.owner is not None,'Native owner only after coordinator ACK')
            native=r.sources.load(reference(path))
            life.process(native,life.command(self.contract,'native',self.admission_ref),self.owner['pid'],self.owner['cgroup']);owners.append(native)
        for owner in owners:p.gone(owner)
        group=p.empty_group(p.GROUP_PREFIX+self.contract['unit']+'.service')
        return dict(physically_drained=True,properties=value,group=group,owners_checked=owners,self_exit_claimed=False)


class OuterPorts(r.Ports):
    """One physical drain receipt owner; finally reuses it without duplicate writes."""
    def __init__(self,*args,owned,**kwargs):
        super().__init__(*args,**kwargs);self.owned=owned;self.drain_ref=None
    def inventory(self):
        path=Path(self.contract['output_root'])/'native-owner.json'
        if path.exists():
            value=self.metadata.read(reference(path))
            self.read_owner=life.process(value,life.command(self.contract,'native',self.admission_ref),self.coordinator['pid'],self.coordinator['cgroup'])
        return super().inventory()
    def assert_service_drained(self,contract,owner):
        self.lease.check('outer.owned_service_drain');proof=self.owned.verify_drained();self.owned.drain=proof
        if self.drain_ref is None:self.drain_ref=self.publish('cleanup-drained.json',proof)
        return self.drain_ref


def outer(spec_ref,review_ref):
    require(sys.dont_write_bytecode and __debug__,'Original -B/assertions required')
    p.user_bus();release=receive_root_release(spec_ref,review_ref)
    fd,owner,first=support.acquire();bound=dict(outer_owner=owner,lock_identity=copy.deepcopy(r.ORIGINAL_LOCK),retained_fd=fd,
        lock_policy=copy.deepcopy(r.LOCK_POLICY),initial_descriptor_proof=first,unit='trn1-mini-layer1-operator-native-unbound')
    lease=r.Lease(bound);metadata=r.Metadata(lambda:lease.check('outer.metadata'));lease.metadata=metadata
    root=None;owned=None;ports=None;primary=None;handlers={};released=False
    def interrupted(signum,frame):raise InterruptedError('Native outer interrupted '+str(signum))
    try:
        require(owner==release['challenge']['owner'],'Root-released outer identity unchanged at acquisition')
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
        p.active_heavy();spec,review,resolved,contract,partition,pair=resolve(spec_ref,review_ref,lease,metadata)
        bound['unit']=contract['unit'];lease.admission=copy.deepcopy(bound)
        require(p.properties(contract['unit'])['LoadState']=='not-found','Fresh actual native unit before output creation')
        root=Path(contract['output_root'])
        require(not root.exists()and root.parent==Path('/mnt/nvme')and root.parent.resolve()==root.parent,'Fresh original NVMe output root')
        for path,floor,reserve in [(ROOT,28*2**30,8*2**20),(root.parent,30*2**30,256*2**20)]:
            fs=os.statvfs(path);require(fs.f_bavail*fs.f_frsize>=floor+reserve,'Original storage floor plus complete output reserve')
        root.mkdir(mode=0o700)
        attempt=os.urandom(16).hex();guard=lambda:lease.check('outer.admission')
        auth=support.authenticate_native_sources(partition,owner,r.ORIGINAL_LOCK,attempt,fd,guard)
        r.atomic_once(root/'source-authentication.json',auth)
        r.verify_loaded(resolved['plan'],guard)
        drain=dict(schema='minicpm5_native_original_predecessor_pair_v1',resolved_plan_sha256=resolved['plan_sha256'],
            outer_owner=owner,lock_identity=r.ORIGINAL_LOCK,physical_pair_schema=r.completion.physical.PAIR_SCHEMA,
            complete_observation_sha256=r.digest(pair),paired_physical=r.completion.audit.packed_physical(pair))
        r.atomic_once(root/'predecessor-drain.json',drain)
        intent=r.prepare_launch_intent(contract,owner,r.ORIGINAL_LOCK,resolved['current_serial_view'],release['tool'],release['handoff_path'],metadata.read,guard)
        r.atomic_once(root/'launch-intent.json',intent)
        a=dict(bound,schema='minicpm5_layer1_operator_native_admission_v1',spec=spec_ref,review=review_ref,overlay=spec['overlay'],
            contract=contract,resolved_plan_sha256=resolved['plan_sha256'],attempt_id=attempt,
            source_contract=resolved['plan']['source_contract'],sdk_operator_recovery=r.completion.BINDING,
            latest_completion=resolved['plan']['latest_completion'],completed_sdk_layout=resolved['plan']['layer1_operator_boundary'],
            serial_predecessors=resolved['plan']['serial_predecessors'],observed_drains=reference(root/'predecessor-drain.json'),
            source_authentication=reference(root/'source-authentication.json'),runtime_admitted=True,predecessors_checked=True,
            current_execution=dict(intent=reference(root/'launch-intent.json'),handoff_path=release['handoff_path'],tool=release['tool']),
            outer_release=release,full_model_qualified=False,new_speedup_claim=False)
        r.atomic_once(root/'admission.json',a);admission_ref=reference(root/'admission.json');lease.admission=copy.deepcopy(a)
        owned=OwnedService(contract,admission_ref,a,metadata);owned.source_pins=resolved['plan']['source_sha256']
        owned.launch(guard);co=owned.acknowledge(guard);owned.publish_running(intent,guard)
        r.check_current_execution(admission_ref,a,co,lease,metadata)
        ports=OuterPorts(contract,resolved,admission_ref,a,co,lease,metadata,owned=owned)
        value=owned.monitor(lease,ports);proof=owned.verify_drained()
        r.atomic_once(root/'service-exit.json',dict(owner=co,properties=value,group=proof['group'],observed_by_outer=owner,
            successful=value['Result']=='success'and value['ExecMainCode']=='1'and value['ExecMainStatus']=='0',self_exit_claimed=False))
        result=life.outer_terminal_step(contract,resolved,admission_ref,co,lease,ports)
        r.atomic_once(root/'outer-result.json',dict(result,outer_owner=owner,service_exit=reference(root/'service-exit.json')))
        return reference(root/'outer-result.json')
    except BaseException as error:
        primary=error;lease.retain_after_failure(error)
        if root is not None and root.exists():
            try:r.atomic_once(root/'outer-failure.json',dict(error_type=type(error).__name__,error=str(error)[:2048],outer_owner=owner,
                descriptor_observations=getattr(error,'observations',None),self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False))
            except BaseException as secondary:print(json.dumps(dict(stage='outer_failure_receipt',secondary=repr(secondary)[:1024])),file=sys.stderr,flush=True)
        raise
    finally:
        # Cleanup intentionally does not reuse a latched work guard: exact owned
        # identity checks still apply, but failed work cannot prevent draining.
        retry=[True]
        def retry_requested(signum,frame):retry[0]=True
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1):
            if sig not in handlers:handlers[sig]=signal.getsignal(sig)
            signal.signal(sig,retry_requested)
        def wait_retry():
            while not retry[0]:time.sleep(.5)
            retry[0]=False
        def pending(value):
            if root is not None and root.exists():r.atomic_once(root/'cleanup-pending.json',dict(value,outer_owner=owner))
        if owned is None:
            proof=dict(physically_drained=True,no_service_launch_attempted=True,owners_checked=[],self_exit_claimed=False)
        else:
            proof=retain_until_drained(owned.cleanup,owned.verify_drained,pending,wait_retry)
        if root is not None and root.exists()and (ports is None or ports.drain_ref is None):
            try:r.atomic_once(root/'cleanup-drained.json',proof)
            except BaseException as secondary:print(json.dumps(dict(stage='cleanup_drained_receipt',secondary=repr(secondary)[:1024])),file=sys.stderr,flush=True)
        support.release_after_physical_drain(fd,proof);released=True
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser()
    for key in ('spec','review'):
        parser.add_argument('--'+key,required=True);parser.add_argument('--'+key+'-sha256',required=True)
    args=parser.parse_args()
    result=outer(dict(path=args.spec,sha256=args.spec_sha256),dict(path=args.review,sha256=args.review_sha256))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
