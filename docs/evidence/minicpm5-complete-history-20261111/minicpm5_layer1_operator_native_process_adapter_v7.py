"""Actual native child/process/descriptor adapters; no IO on import/construction.

Only new runner child modes call these methods after a root-produced admission.
No lock acquisition/release or service launch is implemented here.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import sys
import time
from research import minicpm5_layer1_operator_native_lifecycle_core_v2 as life
from research import minicpm5_layer1_operator_native_output_catalog_v2 as catalog
from research import minicpm5_layer1_operator_native_results_v2 as result_core
from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
from research import minicpm5_layer1_operator_recovered_completion_v2 as completion
from research import minicpm5_layer1_operator_native_overlay_v3 as overlays
from research import descriptor_serial_lock_proof_v1 as descriptor
from research import minicpm5_layer1_operator_native_current_execution_v1 as execution
from research import minicpm5_native_physical_authority_v1 as current_authority
from research import minicpm5_native_current_physical_v1 as current_physical
from research.run_dory_serial_stage_root_v5 import group_members,drain as drain_group
import minicpm5_layer0_execution_protocol_v4 as p
from minicpm5_python_trace_lifecycle_v4 import once as atomic_once
from minicpm5_layer0_runtime_auth_v1 import identity,digest
require=life.require
ORIGINAL_LOCK=dict(device=66306,inode=8294421)
LOCK_POLICY=dict(schema='minicpm5_native_descriptor_lease_policy_v1',identity=ORIGINAL_LOCK,
    parent_only=True,close_on_exec=True,child_inherits_lock_fd=False,no_release_before_owned_service_drained=True,
    descriptor_source=dict(path=str(Path(descriptor.__file__).resolve()),sha256='f3faa10356e16dce6b689298ceebf11441f3821ce391f54896cbffed1c82b003'),
    proc_source=dict(path=str(Path(p.__file__).resolve()),sha256='81e7d7dd913c6b85f0add7b946692d3270f80cc8f2fe980ecad543251657f185'))
GATE_CAP=4096


class Metadata:
    def __init__(self,guard):self.guard=guard;self.identities={}
    def read(self,reference):
        self.guard();sources.reference(reference);before=identity(reference['path']);value=sources.load(reference)
        require(identity(reference['path'])==before,'Stable complete authenticated metadata')
        require(reference['path']not in self.identities or self.identities[reference['path']]==before,'No metadata identity rebound')
        self.identities[reference['path']]=before;self.guard();return value
    def unchanged(self):
        require(all(identity(path)==wanted for path,wanted in self.identities.items()),'Every admitted metadata identity unchanged')


def validate_initial_descriptor(admission):
    proof=admission['initial_descriptor_proof']
    require(proof['schema']=='descriptor_serial_lock_observation_v1'and proof['stage']=='complete'
        and proof['passed']is True and proof['runtime_admitted']is False
        and proof['owner']==admission['outer_owner']and proof['lock']==ORIGINAL_LOCK
        and proof['retained_fd']==admission['retained_fd']and proof['lock_path']==descriptor.LOCK_PATH
        and proof['before']==proof['after'],'Actual initial original descriptor proof bound in admission')
    for side in ('before','after'):
        row=proof[side]
        require(row['path_stat']==row['fd_stat']and row['fd_target']==descriptor.LOCK_PATH
            and row['path_stat']['device']==ORIGINAL_LOCK['device']and row['path_stat']['inode']==ORIGINAL_LOCK['inode']
            and stat.S_ISREG(row['path_stat']['mode'])and row['path_stat']['nlink']==1,'Initial canonical fd/path singleton identity')
    for key,path in [('fdinfo_read','/proc/%d/fdinfo/%d'%(admission['outer_owner']['pid'],admission['retained_fd'])),('global_read','/proc/locks')]:
        row=proof[key];require(row['path']==path and row['eof_observed']is True and row['bytes_read']==len(row['text'].encode('ascii')),
            'Complete original initial fd/global proc evidence retained')
    require(descriptor.check_fdinfo(proof['fdinfo_read']['text'],admission['outer_owner'],ORIGINAL_LOCK)==proof['fdinfo']
        and descriptor.check_global(proof['global_read']['text'],admission['outer_owner'],ORIGINAL_LOCK)==proof['global'],
        'Initial evidence revalidated by accepted descriptor semantics')
    return proof


def descriptor_observation(admission):
    require(admission['lock_policy']==LOCK_POLICY and admission['lock_identity']==ORIGINAL_LOCK
        and type(admission['retained_fd'])is int and admission['retained_fd']>=0,'Reviewed original parent-only descriptor binding')
    lock=Path(descriptor.LOCK_PATH)
    for parent in lock.parents:
        info=parent.lstat();require(stat.S_ISDIR(info.st_mode),'Canonical no-symlink original lock ancestry')
    return descriptor.prove(admission['outer_owner'],ORIGINAL_LOCK,admission['retained_fd'],proc=p.proc,
        open_binary=lambda name:Path(name).open('rb'),lstat=lambda name:Path(name).lstat(),
        stat=lambda name:Path(name).stat(),readlink=os.readlink)


class Lease:
    """Actual observer of outer-owned fd; this object never owns/unlocks it."""
    def __init__(self,admission):
        self.admission=copy.deepcopy(admission);self.failed=None;self.metadata=None;self.scope=None
        self.processes=[];self.last_full=None;self.first_duplicate=None;self.execution_observation=None
    def check(self,stage):
        if self.failed is not None:raise self.failed
        try:
            observed=descriptor_observation(self.admission);p.clear_window()
            if observed['global']['target_row_count']>1 and self.first_duplicate is None:
                self.first_duplicate=copy.deepcopy(observed)
                raw=json.dumps(dict(stage=stage,event='native_first_descriptor_duplicate',observation=observed),allow_nan=False)
                require(len(raw.encode())<8*2**20,'Bounded first descriptor duplicate evidence')
                print(raw,file=sys.stderr,flush=True)
            for owner in self.processes:require(p.proc(owner['pid'])==owner,'Exact admitted current process identity')
            if self.execution_observation is not None:
                obs=self.execution_observation
                require(identity(obs['path'])==obs['identity'],'Admitted running handoff original byte identity unchanged')
            now=time.monotonic()
            if self.last_full is None or now-self.last_full>=1:
                if self.metadata:self.metadata.unchanged()
                if self.execution_observation is not None:
                    require(completion.latest.latest_handoff()==self.execution_observation,'No later unknown serial handoff')
                if self.scope:
                    for owner in self.scope['owners']:p.gone(owner)
                p.active_heavy((self.admission['unit'],));self.last_full=now
            return observed
        except BaseException as error:
            self.failed=error;raise
    def retain_after_failure(self,error):
        if self.failed is None:self.failed=error
        try:print(json.dumps(dict(event='native_owned_service_requires_outer_drain',error_type=type(error).__name__,
            error=str(error)[:2048],descriptor_observations=getattr(error,'observations',None),
            serial_ownership_must_remain_held=True,actual_external_tool_exit_verified=False),allow_nan=False),file=sys.stderr,flush=True)
        except BaseException:pass


def environment(admission):return p.environment({},'native',admission['attempt_id'])


def verify_loaded(plan,guard):
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and str(path).startswith(life.ROOT+'/'):
            guard();path=str(Path(path).resolve());wanted=plan['source_sha256'].get(path)
            require(wanted is not None and hashlib.sha256(sources.text(path,wanted)).hexdigest()==wanted,'Every actually loaded local source pinned')


def resolve_admitted(admission_ref,lease,metadata):
    """Actual guarded recovery/overlay metadata; no native/raw payload reads."""
    a=metadata.read(admission_ref)
    validate_initial_descriptor(a)
    require(a==lease.admission and a['schema']=='minicpm5_layer1_operator_native_admission_v1'
        and a['runtime_admitted']is a['predecessors_checked']is True and a['lock_policy']==LOCK_POLICY,
        'Exact root-produced new native admission')
    spec=metadata.read(a['spec']);review=metadata.read(a['review'])
    require(set(spec)=={'schema','overlay','runner','unit','lock_policy'}
        and spec['schema']=='minicpm5_native_process_spec_v1'and spec['lock_policy']==LOCK_POLICY
        and review['schema']=='minicpm5_native_process_source_review_v1'and review['reviewer']=='/root'
        and review['passed']is True and review['blockers']==[]and review['spec']==a['spec']
        and review['overlay']==spec['overlay']==a['overlay']and review['lock_policy']==LOCK_POLICY,
        'Independent concrete small spec/review and explicit Mini descriptor policy')
    guard=lambda:lease.check('metadata.resolve')
    recovered=completion.completed(completion.BINDING,metadata.read,guard,lambda path,sha:sources.text(path,sha))
    resolved=overlays.resolve(metadata.read(spec['overlay']),recovered,metadata.read,guard)
    contract=life.build_contract(resolved,spec['overlay'],spec['runner'],spec['unit'])
    require(a['contract']==contract and a['resolved_plan_sha256']==review['resolved_plan_sha256']==resolved['plan_sha256']
        and a['unit']==spec['unit']and a['source_contract']==resolved['plan']['source_contract']
        and a['sdk_operator_recovery']==completion.BINDING
        and a['latest_completion']==resolved['plan']['latest_completion']
        and a['completed_sdk_layout']==resolved['plan']['layer1_operator_boundary']
        and a['serial_predecessors']==resolved['plan']['serial_predecessors'],'Complete current admission/plan/contract bindings')
    for pin in (LOCK_POLICY['descriptor_source'],LOCK_POLICY['proc_source'],spec['runner']):
        require(resolved['plan']['source_sha256'].get(pin['path'])==pin['sha256'],'Exact real lease/proc/runner source pin')
    partition=sources.classify(resolved['plan']['source_sha256'],resolved['plan']['source_contract'],metadata.read)
    sources.verify_text_sources(partition,guard)
    lease.scope=resolved['current_scope'];lease.metadata=metadata
    # Authenticate the original full paired drain using its existing exact validator.
    saved=metadata.read(a['observed_drains'])
    require(saved['schema']=='minicpm5_native_original_predecessor_pair_v1'and saved['resolved_plan_sha256']==resolved['plan_sha256']
        and saved['outer_owner']==a['outer_owner']and saved['lock_identity']==ORIGINAL_LOCK,'Root saved exact current physical pair binding')
    known=set(resolved['current_scope']['complete_known_units'])
    required={row['unit']for row in resolved['current_scope']['nemotron']['history']['rows']}
    require(saved['physical_pair_schema']==completion.physical.PAIR_SCHEMA,'Exact existing complete physical-pair schema')
    saved_pair=completion.audit.unpack(saved['paired_physical'])
    require(digest(saved_pair)==saved['complete_observation_sha256'],'Losslessly retained full pair digest')
    completion.physical.validate_pair(saved_pair,required,resolved['current_scope']['owners'],ORIGINAL_LOCK,known_units=known)
    co_path=a['contract']['output_root']+'/coordinator-owner.json'
    co_ref=dict(path=co_path,sha256=hashlib.sha256(sources.text(co_path)).hexdigest());co=metadata.read(co_ref)
    check_current_execution(admission_ref,a,co,lease,metadata)
    authority=current_authority.Authority(sys.modules[__name__],lease,admission_ref=admission_ref,admission=a,coordinator=co,metadata=metadata)
    fresh=current_physical.collect_pair(resolved['current_scope'],guard,authority)
    require(completion.physical.validate_pair(fresh,required,resolved['current_scope']['owners'],ORIGINAL_LOCK,known_units=known)==known,
        'Fresh complete current original paired drain before child/model work')
    co_ref=dict(path=a['contract']['output_root']+'/coordinator-owner.json',
        sha256=hashlib.sha256(sources.text(a['contract']['output_root']+'/coordinator-owner.json')).hexdigest())
    check_current_execution(admission_ref,a,metadata.read(co_ref),lease,metadata)
    verify_loaded(resolved['plan'],guard)
    # These refs are the original failed operator run's recovered successful ranks.
    rank_refs=[recovered['recovery']['original_model_outputs']['rank%d_result'%rank]for rank in (0,1)]
    require([row['path']for row in rank_refs]==[
        '/mnt/nvme/minicpm5-layer1-operator-sdk-r2-20261111/rank0-result.json',
        '/mnt/nvme/minicpm5-layer1-operator-sdk-r2-20261111/rank1-result.json'],'Only original recovered SDK operator rank outputs')
    return resolved,recovered,rank_refs



def prepare_launch_intent(contract,outer_owner,lock_identity,current_serial_view,execution_tool,running_path,read,guard):
    """Outer-only source adapter: original freshness under held lease, no write."""
    guard();require(p.proc(os.getpid())==outer_owner,'Only actual original outer prepares intent')
    fresh=completion.latest.require_latest_terminal(current_serial_view,read,guard)
    guard();return execution.make_intent(contract,outer_owner,lock_identity,
        current_serial_view['serial_predecessors'],execution_tool,running_path,fresh)


def check_current_execution(admission_ref,a,co,lease,metadata):
    """Authenticate exact current running handoff, then latch its raw identity."""
    guard=lambda:lease.check('entry.current_execution')
    guard();current=execution.declaration(a['current_execution'],a['contract'])
    intent=metadata.read(current['intent']);fresh=execution.validate_intent(intent,a)
    catalog_value=metadata.read(a['serial_predecessors'])
    require(catalog_value['current_handoff']==fresh['bound_terminal'],'Intent binds actual complete predecessor catalog terminal')
    refs=[*fresh['publication_only_successors'],fresh['bound_terminal']]
    chain=[(ref,metadata.read(ref))for ref in refs]
    previous=execution.validate_predecessor_chain(fresh,a['serial_predecessors'],chain[-1][1],chain)
    first=completion.latest.latest_handoff()
    require(first['path']==current['handoff_path'],'Latest handoff is exactly this admitted running execution')
    raw=sources.text(first['path']);require(len(raw)<=2*2**20,'Original bounded complete running handoff')
    handoff_ref=dict(path=first['path'],sha256=hashlib.sha256(raw).hexdigest());value=metadata.read(handoff_ref)
    co_path=a['contract']['output_root']+'/coordinator-owner.json'
    co_ref=dict(path=co_path,sha256=hashlib.sha256(sources.text(co_path)).hexdigest())
    require(metadata.read(admission_ref)==a and metadata.read(co_ref)==co,'Stable original admission/coordinator bytes')
    result=execution.validate_running(value,previous,admission_ref,a,intent,co_ref,co,
        p.proc(a['outer_owner']['pid']),p.proc(co['pid']))
    service_properties(a['contract'],admission_ref,a,co)
    guard();metadata.unchanged()
    require(identity(first['path'])==first['identity']and completion.latest.latest_handoff()==first,
        'Exact same raw handoff identity and no newer serial update during current-run validation')
    require(lease.execution_observation is None or lease.execution_observation==first,'Never rebind an admitted execution handoff')
    lease.execution_observation=copy.deepcopy(first)
    return result


def output_inventory(root,contract,writers):
    """Descriptor-relative live finite walk; only exact atomic rename accepted."""
    root=Path(root);c=catalog.catalog(contract);allowed=set(c['payload_files'])|set(c['driver_metadata_files'])|set(c['runtime_files'])
    dirs=set(c['driver_directories']);entries={};seen={}
    info=root.lstat();require(stat.S_ISDIR(info.st_mode)and info.st_uid==1000 and root.resolve()==root,'Canonical owned output root')
    stable=lambda info:(info.st_dev,info.st_ino,info.st_mode,info.st_uid)
    row=lambda info:dict(kind='directory'if stat.S_ISDIR(info.st_mode)else'file',uid=info.st_uid,nlink=info.st_nlink,
        symlink=False,size=info.st_size,blocks=info.st_blocks)
    root_entry=row(info);root_before=stable(info)
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    def walk(parent,prefix):
        before=stable(os.fstat(parent))
        with os.scandir(parent)as scan:
            for entry in scan:
                require(entry.name not in ('.','..')and '/'not in entry.name,'Exact direct output entry')
                leaf=entry.name;name=prefix+'/'+leaf if prefix else leaf;destination=None
                if name.startswith('.pending-'):destination=catalog.staging_destination(name,writers)
                require(name in allowed or name in dirs or destination is not None,'Exact finite native live output path')
                try:info=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
                except FileNotFoundError:
                    require(destination is not None and not prefix,'Only known root atomic staging rename may disappear')
                    leaf=name=destination;info=os.stat(leaf,dir_fd=parent,follow_symlinks=False)
                require(info.st_uid==1000 and not stat.S_ISLNK(info.st_mode),'No live output symlink or foreign owner')
                node=(info.st_dev,info.st_ino)
                if node in seen:
                    prior=seen[node]
                    require(prior.startswith('.pending-')!=name.startswith('.pending-'),'Only one pending/final name pair may share an inode')
                    temporary=prior if prior.startswith('.pending-')else name;final=name if prior.startswith('.pending-')else prior
                    require(catalog.staging_destination(temporary,writers)==final,'Exact acknowledged pending/final inode')
                    if name==temporary:continue
                    entries.pop(prior,None)
                seen[node]=name;directory=stat.S_ISDIR(info.st_mode)
                require(directory and name in dirs or stat.S_ISREG(info.st_mode)and info.st_nlink==1,'Exact directory or singleton live output')
                entries[name]=row(info)
                require(len(entries)<=len(allowed)+len(dirs)+len(catalog.STAGING_WRITERS),'Finite live walk extent')
                if directory:
                    child=os.open(leaf,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
                    try:
                        require(stable(os.fstat(child))==stable(info),'No directory replacement during live open');walk(child,name)
                        require(stable(os.stat(leaf,dir_fd=parent,follow_symlinks=False))==stable(info),'Live directory path remains bound')
                    finally:os.close(child)
        require(stable(os.fstat(parent))==before,'Held live directory identity unchanged')
    try:
        require(stable(os.fstat(fd))==root_before,'Root descriptor identity matches named root');walk(fd,'')
        require(stable(root.lstat())==root_before and root.resolve()==root,'Live root path binding unchanged')
    finally:os.close(fd)
    return entries,root_entry,writers


class Ports:
    def __init__(self,contract,resolved,admission_ref,admission,coordinator,lease,metadata):
        self.contract,self.resolved,self.admission_ref,self.admission=contract,resolved,admission_ref,admission
        self.coordinator,self.lease,self.metadata=coordinator,lease,metadata;self.child=None
        self.read_owner=None;self.released=False
    def current_process(self):return p.proc(os.getpid())
    def storage_available(self):
        return {name:(lambda fs:fs.f_bavail*fs.f_frsize)(os.statvfs(path))for name,path in [('root',life.ROOT),('output','/mnt/nvme')]}
    def inventory(self):
        writers=dict(outer=self.admission['outer_owner']['pid'],coordinator=self.coordinator['pid'])
        if self.read_owner is not None:writers['native']=self.read_owner['pid']
        return output_inventory(self.contract['output_root'],self.resolved['plan']['native_operator_manifest'],writers)
    def spawn_native(self,argv):
        require(self.child is None and argv==life.command(self.contract,'native',self.admission_ref),'One exact native spawn')
        # Only stdin pipe is inherited; original outer lock descriptor is CLOEXEC.
        self.child=subprocess.Popen(argv,env=environment(self.admission),cwd=life.ROOT,stdin=subprocess.PIPE,
            close_fds=True,start_new_session=True)
        return self.child
    def acknowledge(self,child,timeout):
        require(child is self.child and timeout==10,'Exact original child acknowledgement deadline')
        end=time.monotonic()+timeout
        while True:
            self.lease.check('process.await_native_exec');actual=p.proc(child.pid)
            if actual is not None and actual['argv']==life.command(self.contract,'native',self.admission_ref):
                life.process(actual,actual['argv'],self.coordinator['pid'],self.coordinator['cgroup']);self.read_owner=actual;return actual
            require(child.poll()is None and time.monotonic()<end,'Native exec/ack deadline');time.sleep(.01)
    def release_native(self,child,binding):
        require(child is self.child and not self.released and self.read_owner is not None,'Single exact native release')
        expected=life.release_binding(self.contract,self.admission_ref,{k:self.coordinator[k]for k in life.PROCESS_KEYS},self.reference('native-owner.json'))
        require(binding==expected and p.proc(child.pid)==self.read_owner and child.poll()is None,'Exact current native gate binding')
        self.lease.check('process.release_native');raw=(json.dumps(binding,separators=(',',':'),allow_nan=False)+'\n').encode()
        require(len(raw)<=GATE_CAP,'Bounded complete native gate packet')
        try:require(child.stdin.write(raw)==len(raw),'Complete native release packet');child.stdin.flush()
        finally:child.stdin.close()
        self.released=True
    def monotonic(self):return time.monotonic()
    def idle(self,seconds):require(seconds==.5,'Original native monitor interval');time.sleep(seconds)
    def publish(self,name,value):
        self.lease.check('process.publish.'+name);require(name in catalog.RUNTIME_FILES,'Finite original receipt name')
        path=Path(self.contract['output_root'])/name;atomic_once(path,value);return self.reference(name)
    def reference(self,name):
        require(name in catalog.RUNTIME_FILES,'Finite receipt lookup')
        path=str(Path(self.contract['output_root'])/name);return dict(path=path,sha256=hashlib.sha256(sources.text(path)).hexdigest())
    def read_json(self,reference):return self.metadata.read(reference)
    def assert_child_drained(self,child,owner):
        require(child is self.child and type(child.returncode)is int,'Exact child handle already reaped')
        drain_group(child,owner,p)
        if owner is not None:p.gone(owner)
        require(not group_members(child.pid),'Entire owned native process group drained')
        if child.stdin is not None and not child.stdin.closed:child.stdin.close()
    def observe_service_terminal(self,contract,owner):
        value=service_properties(contract,self.admission_ref,self.admission,owner)
        require(p.terminal(value),'Actual service terminal with no pending Job');return value
    def assert_service_drained(self,contract,owner):
        self.lease.check('process.owned_service_drain');value=self.observe_service_terminal(contract,owner)
        p.gone(owner)
        native_ref=self.reference('native-owner.json');native=self.read_json(native_ref);p.gone(native)
        group=p.empty_group(p.GROUP_PREFIX+contract['unit']+'.service')
        self.lease.check('process.owned_service_drained')
        return self.publish('cleanup-drained.json',dict(physically_drained=True,properties=value,group=group,
            owners_checked=[owner,native],self_exit_claimed=False))
    def secondary(self,primary,error,stage):
        print(json.dumps(dict(stage=stage,primary=repr(primary)[:2048],secondary=repr(error)[:2048],lock_must_remain_held=True)),file=sys.stderr,flush=True)


def receive_release(contract,admission_ref,coordinator,owner_ref,lease,*,fd=0):
    """Native child gate: bounded pipe bytes and EOF before any model import."""
    require(stat.S_ISFIFO(os.fstat(fd).st_mode),'Release only over inherited stdin pipe')
    deadline=time.monotonic()+60;raw=bytearray()
    while True:
        lease.check('native.await_release')
        remaining=deadline-time.monotonic();require(remaining>0,'Native explicit release deadline')
        ready,_,_=select.select([fd],[],[],min(.1,remaining))
        if not ready:continue
        block=os.read(fd,GATE_CAP+1-len(raw));raw.extend(block)
        require(len(raw)<=GATE_CAP,'Bounded native release pipe packet')
        if not block:break
    require(raw.endswith(b'\n')and raw.count(b'\n')==1,'One complete release packet followed by pipe EOF')
    value=result_core.json_object(bytes(raw))
    require(value==life.release_binding(contract,admission_ref,{k:coordinator[k]for k in life.PROCESS_KEYS},owner_ref),
        'Release binds actual owner receipt/admission/coordinator/resolved plan')
    lease.check('native.released');return value


def native_service_properties(unit):
    require(re.fullmatch(r'trn1-mini-layer1-operator-native-[a-z0-9-]{1,80}',unit),'Exact native service property target')
    keys=set(p.KEYS.split(','))|{key.replace('Sec','USec')if key in ('RuntimeMaxSec','TimeoutStopSec')else key for key in life.PROPS}
    raw=p.command(['systemctl','--user','show','--all',unit,'--property='+','.join(sorted(keys))])
    return p.parse_properties(raw)


def service_properties(contract,admission_ref,admission,owner):
    """Actual original systemd resource/command/invocation validation."""
    value=native_service_properties(contract['unit'])
    require(value['LoadState']=='loaded'and value['Id']==contract['unit']+'.service'
        and p.invocation(value['InvocationID'])and value['InvocationID']==owner['invocation_id']
        and value['ExecMainPID']==str(owner['pid']),'Exact owned coordinator invocation')
    for key,wanted in life.PROPS.items():
        if key in ('RuntimeMaxSec','TimeoutStopSec'):
            require(p.duration_seconds(value[key.replace('Sec','USec')])==int(wanted),'Exact native service duration '+key)
        else:require(value[key]==wanted,'Exact native service resource '+key)
    env=environment(admission)
    argv=['/usr/bin/env','-i',*[key+'='+item for key,item in env.items()],*life.command(contract,'coordinator',admission_ref)]
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[' '.join(argv)]and value['ExecStopPost']=='',
        'Exact original native coordinator command/environment and no extra cleanup')
    require(value['ControlGroup']in('',p.GROUP_PREFIX+contract['unit']+'.service'),'Exact owned native service group')
    return value
