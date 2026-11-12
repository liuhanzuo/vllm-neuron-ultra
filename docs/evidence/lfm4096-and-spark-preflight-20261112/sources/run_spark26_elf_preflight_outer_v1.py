"""Future metadata-only Spark retained-lock controller; import performs no runtime actions.

This binds a distinct metadata collector, never the scientific diagnostic. Real
history is imported only under the retained lock. Its transitive ctypes import
is an explicit system-Python host dependency, not application/native coverage.
Source tests block that import and use mocked process/service observations.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import sysconfig
import time

import spark26_elf_preflight_worker_v1 as worker

ROOT, PYTHON = worker.ROOT, worker.PYTHON
SELF = str(ROOT/'run_spark26_elf_preflight_outer_v1.py')
LOCK = worker.contract.LOCK_IDENTITY
LOCK_PATH = ROOT/'serial-enrollment.lock'
OUTER_LIMITS = dict(address_space_bytes=2*2**30, work_wall_seconds=10800,
                    single_file_bytes=worker.BUDGET['single_file_bytes'], cpu_affinity=[6,7], nice=19)
HOST_BOUNDARY = dict(system_python_stdlib_and_system_ELF=True,
    history_entrypoint='lfm3_nki05_serial_history_v1',
    history_ctypes_import='Deferred until original retained lock is proven; standard-library host dependency',
    application_native_imports_in_outer=False, third_party_payload_authentication='Owned worker only')
BLOCKED = frozenset(('torch','torch_xla','torch_neuronx','numpy','safetensors','transformers',
    'neuronx_distributed','neuronx_distributed_inference','neuronxcc','libneuronxla','vllm','vllm_neuron'))
FIXED = {**worker.HELPERS,
    str(ROOT/'spark26_native_serial_worker_v2.py'): 'b4392c2c10410e96498e53f2cca5f7c5c256eb11e80fc655fc4a8025d7234715',
    str(ROOT/'spark26_native_provider_v2.py'): '12f9d6e63dbe112359d3acd32984d69d993b7fddfee80ef670d4fd901aed93e6',
    str(ROOT/'run_spark26_native_serial_outer_v1.py'): '7ac3e3fe4529ea2eb7e480d9252292d857b74c9031ae8df0d1b755f33902b9ef',
}
ACK_SECONDS, CLEANUP_SECONDS, TERM_GRACE_SECONDS = 60, 60, 30
require, encode, read = worker.require, worker.encode, worker.read


def file_identity(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and not any(x.is_symlink() for x in (path,*path.parents)), 'Canonical source/metadata identity')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Singleton source/metadata input')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def immutable_publish(path, value):
    """Complete file before atomic rename in this outer's private output tree.

    The unique 0700 run directory and single declared writer own these names.
    Check-and-rename does not claim a defense against another writer of the
    same UID violating that ownership contract. No existing receipt is reused.
    """
    path=Path(path); payload=encode(value)
    require(len(payload)<=worker.BUDGET['single_file_bytes'] and not os.path.lexists(path),
            'Fresh bounded immutable outer receipt')
    temporary=path.parent/'tmp'/('outer-'+path.name+'.publishing')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    require(not os.path.lexists(path),'No final receipt may be overwritten')
    os.rename(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest())


def observed_ref(path):
    path=Path(path);before=file_identity(path)
    require(before[2]<=worker.BUDGET['single_file_bytes'],'Bounded actual new receipt')
    with path.open('rb') as stream:raw=stream.read(worker.BUDGET['single_file_bytes']+1)
    require(len(raw)==before[2] and file_identity(path)==before,'Stable actual receipt bytes')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def outer_argv(binding_ref,review_ref):
    return [PYTHON,'-S','-B',SELF,'--binding',binding_ref['path'],'--binding-sha256',binding_ref['sha256'],
            '--review',review_ref['path'],'--review-sha256',review_ref['sha256']]


def validate_binding(binding,review,binding_ref,spec,worker_review):
    require(set(binding)=={'schema','worker_spec','worker_review','source_sha256','outer_limits','host_boundary'}
            and binding['schema']=='spark26_elf_preflight_outer_binding_v1'
            and binding['outer_limits']==OUTER_LIMITS and binding['host_boundary']==HOST_BOUNDARY,
            'Exact root-finalized outer scope and explicit host boundary')
    worker.validate_spec(spec)
    require(binding['source_sha256']==spec['source_sha256']
            and all(spec['source_sha256'].get(path)==sha for path,sha in FIXED.items())
            and SELF in spec['source_sha256'],'Same complete additive worker/outer source closure')
    require(review['schema']=='spark26_elf_preflight_outer_root_review_v1' and review['reviewer']=='/root'
            and review['binding']==binding_ref and review['source_sha256']==spec['source_sha256']
            and review['passed'] is review['source_only'] is True and review['blockers']==[]
            and review['runtime_admitted'] is False,'Independent root review of this exact concrete outer binding')
    require(worker_review['schema']=='spark26_elf_preflight_worker_root_review_v1'
            and worker_review['reviewer']=='/root' and worker_review['spec']==binding['worker_spec']
            and worker_review['source_sha256']==spec['source_sha256']
            and worker_review['passed'] is worker_review['source_only'] is True
            and worker_review['runtime_admitted'] is False and worker_review['blockers']==[],
            'Same independently reviewed concrete worker manifest/budget binding')


class PinnedHostFinder:
    """No application packages; local history sources pinned before execution."""
    def __init__(self,sources,finders):
        self.sources,self.finders=sources,tuple(finders)
        self.stdlib=Path(sysconfig.get_path('stdlib')).resolve()
        self.history_gate_open=False;self.ctypes_origins={}

    def check_name(self,name):
        root=name.split('.')[0]
        require(root not in BLOCKED,'Outer cannot import application native packages: '+name)
        require(root not in ('ctypes','_ctypes') or self.history_gate_open,
                'Historical ctypes entry requires an actual retained-lock gate')

    def check_spec(self,name,spec):
        origin=spec.origin
        if origin in ('built-in','frozen'):return
        if origin is None:
            require(spec.submodule_search_locations is not None
                    and all(Path(p).resolve()==ROOT/'research' for p in spec.submodule_search_locations),
                    'Exact reviewed research namespace only')
            return
        path=Path(origin).resolve()
        if path.is_relative_to(self.stdlib) and 'site-packages' not in path.parts and 'dist-packages' not in path.parts:
            require(path.suffix!='.pyc','Absent bytecode namespace remains in effect')
            if name.split('.')[0] in ('ctypes','_ctypes'):self.ctypes_origins[name]=str(path)
            return
        require(str(path) in self.sources and path.suffix=='.py','Only reviewed local host sources')
        worker.metadata_bytes(dict(path=str(path),sha256=self.sources[str(path)]))

    def find_spec(self,name,path=None,target=None):
        self.check_name(name)
        for finder in self.finders:
            spec=finder.find_spec(name,path,target)
            if spec is not None:
                self.check_spec(name,spec);return spec
        return None


class RealHost:
    """Concrete source/process/service IO; methods are mocked only in source tests."""
    def __init__(self,sources):
        self.sources=sources
        for path,sha in sources.items():worker.metadata_bytes(dict(path=path,sha256=sha))
        self.source_identities={path:file_identity(path) for path in sources}
        require(not BLOCKED.intersection(sys.modules) and 'ctypes' not in sys.modules and '_ctypes' not in sys.modules,
                'Fresh host starts before historical ctypes/application imports')
        self.finder=PinnedHostFinder(sources,sys.meta_path)
        sys.meta_path[:]=[self.finder]
        for name,module in tuple(sys.modules.items()):
            self.finder.check_name(name)
            spec=getattr(module,'__spec__',None)
            if spec is not None:self.finder.check_spec(name,spec)
        self.p=importlib.import_module('minicpm5_layer0_execution_protocol_v4')
        self.descriptor=importlib.import_module('research.descriptor_serial_lock_proof_v1')
        self.hold=importlib.import_module('research.minicpm5_layer1_drain_hold_v1').retain_until_drained
        self.history=None

    def unchanged(self):
        require(sys.meta_path==[self.finder],'Host import guard cannot be replaced')
        for path,before in self.source_identities.items():require(file_identity(path)==before,'Reviewed source changed')

    def load_history(self,lease):
        lease.check();self.unchanged();require(self.history is None,'One explicit history import only')
        self.finder.history_gate_open=True
        self.history=importlib.import_module(HOST_BOUNDARY['history_entrypoint'])
        require(not BLOCKED.intersection(sys.modules),'No application native imports through historical modules')
        lease.check();self.unchanged()
        return self.history

    def properties(self,unit):
        require(re.fullmatch('trn1-spark26-elf-preflight-[a-z0-9-]{1,80}',unit),'Exact Spark service namespace')
        return self.p.parse_properties(self.p.command(['systemctl','--user','show','--all',unit,
            '--property='+self.p.KEYS+',LimitAS,LimitFSIZE,StandardOutput,StandardError']))

    def command(self,argv,timeout=30):return self.p.command(argv,timeout=timeout)
    def proc(self,pid):return self.p.proc(pid)
    def gone(self,owner):return self.p.gone(owner)
    def empty_group(self,group):return self.p.empty_group(group)
    def terminal(self,value):return self.p.terminal(value)
    def clock(self):return time.monotonic()
    def wait(self,seconds):time.sleep(min(seconds,1.))
    def publish(self,path,value):return immutable_publish(path,value)


class RetainedLease:
    """Original FD acquired once; only complete independent drain permits release."""
    def __init__(self,host):self.host=host;self.fd=None;self.owner=None;self.initial=None

    def acquire(self):
        import fcntl
        require(self.fd is None,'Single parent-only acquisition')
        for parent in LOCK_PATH.parents:require(stat.S_ISDIR(parent.lstat().st_mode),'Nonsymlink lock ancestry')
        before=LOCK_PATH.lstat()
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1
                and dict(device=before.st_dev,inode=before.st_ino)==LOCK,'Original existing serial inode')
        fd=os.open(LOCK_PATH,os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            actual=os.fstat(fd)
            require((actual.st_dev,actual.st_ino)==(before.st_dev,before.st_ino)
                    and not os.get_inheritable(fd),'Exact parent-only opened descriptor')
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.fd=fd;self.owner=self.host.proc(os.getpid());self.initial=self.check()
        except BaseException:
            os.close(fd);self.fd=None;raise
        return self.initial

    def check(self):
        require(self.fd is not None and not os.get_inheritable(self.fd),'Original retained CLOEXEC fd remains owned')
        return self.host.descriptor.prove(self.owner,LOCK,self.fd,proc=self.host.proc,
            open_binary=lambda path:open(path,'rb'),lstat=os.lstat,stat=os.stat,readlink=os.readlink)

    def release(self,proof):
        import fcntl
        require(type(proof)is dict and proof.get('physically_drained') is True,'No unlock without actual complete drain')
        self.check();fcntl.flock(self.fd,fcntl.LOCK_UN);os.close(self.fd);self.fd=None


def installed_properties(spec,admission_ref,value,invocation=None,*,resources=True):
    """Allow startup/terminal states while retaining exact command and invocation."""
    require(value['LoadState']=='loaded' and value['Id']==spec['unit']+'.service'
            and re.fullmatch('[a-f0-9]{32}',value['InvocationID']) and value['InvocationID']!='0'*32,
            'Loaded exact nonempty service invocation')
    require(invocation is None or value['InvocationID']==invocation,'Service invocation replacement rejected')
    command=worker.service_command(spec,admission_ref)
    require(re.findall(r'argv\[\]=(.*?)\s*;',value['ExecStart'])==[' '.join(command[command.index('/usr/bin/env'):])]
            and value['Environment']==value['ExecStopPost']=='','Exact installed clean command')
    require(value['StandardOutput']=='append' and value['StandardError']=='inherit','Original bounded file logging mode')
    group='/user.slice/user-1000.slice/user@1000.service/app.slice/'+spec['unit']+'.service'
    require(value['ControlGroup'] in ('',group),'No foreign service cgroup')
    if resources:
        # Pure duration parser is safe to import; it does not import historical ctypes.
        from minicpm5_layer0_execution_protocol_v4 import duration_seconds
        for key,expected in worker.PROPS.items():
            actual=value[key.replace('Sec','USec')] if key.endswith('Sec') else value[key]
            require((duration_seconds(actual)==int(expected)) if key.endswith('Sec') else actual==expected,
                    'Original worker resource retained: '+key)
    return value


def exit_status(value,*,no_process_observed=False):
    require(value['Result'] in ('success','exit-code','signal','core-dump','timeout','watchdog','oom-kill',
                               'resources','protocol','start-limit-hit'),'Known actual systemd result')
    code,status=int(value['ExecMainCode']),int(value['ExecMainStatus'])
    if code==status==0:
        require(no_process_observed and value['ExecMainPID']=='0' and value['Result']!='success',
                'No-process startup failure preserves status without inventing an exit')
        return dict(ExecMainCode=code,ExecMainStatus=status,Result=value['Result'])
    require(code in (1,2,3) and 0<=status<=255,'Actual exited/killed/dumped process status required')
    if value['Result']=='exit-code':require(code==1 and status!=0,'Preserve real failed exit')
    if value['Result'] in ('signal','core-dump'):
        require(code==(2 if value['Result']=='signal' else 3) and status>0,'Actual terminating signal')
    return dict(ExecMainCode=code,ExecMainStatus=status,Result=value['Result'])


class OwnedService:
    """Durable attempt, env→Python identity continuity, real terminal and drain."""
    def __init__(self,host,spec,admission_ref,admission):
        self.host,self.spec,self.ref,self.admission=host,spec,admission_ref,admission
        self.root=Path(spec['output_root']);self.attempted=False;self.invocation=None
        self.owner=None;self.pending=None;self.unacknowledged=None;self.saved_terminal=None
        self.stop_requested=False

    @property
    def group(self):return '/user.slice/user-1000.slice/user@1000.service/app.slice/'+self.spec['unit']+'.service'

    def launch(self):
        require(not self.attempted,'One actual service installation attempt only')
        command=worker.service_command(self.spec,self.ref)
        self.intent=dict(command=command,admission=self.ref,outer_owner=self.admission['outer_owner'])
        self.intent_ref=self.host.publish(self.root/'launch-intent.json',self.intent)
        self.attempted=True
        self.host.command(command,timeout=30)

    def observe(self,*,resources=True):
        require(self.attempted,'No service observation adopted without durable launch intent')
        value=self.host.properties(self.spec['unit'])
        if value['LoadState']=='not-found':
            require(self.invocation is None and self.owner is None and self.pending is None
                    and self.unacknowledged is None,'Observed service cannot disappear before real terminal audit')
            require(value['Id']==self.spec['unit']+'.service' and value['ActiveState']=='inactive'
                    and value['SubState']=='dead' and all(value[k]=='0' for k in ('MainPID','ControlPID','ExecMainPID'))
                    and all(value[k]=='' for k in ('Job','InvocationID','ControlGroup')),'Complete never-loaded absence')
            return value
        installed_properties(self.spec,self.ref,value,self.invocation,resources=resources)
        self.invocation=value['InvocationID']
        pid=int(value['MainPID'])
        if self.owner is not None:
            require(value['ExecMainPID']==str(self.owner['process']['pid']),'Original actual main PID retained')
        elif pid:
            current=self.host.proc(pid)
            require(current is not None and current['cgroup']=='0::'+self.group,'Actual startup process/cgroup')
            command=worker.service_command(self.spec,self.ref)
            wrapper=command[command.index('/usr/bin/env'):]
            require(current['argv'] in (wrapper,worker.argv(self.ref)),'Only exact env wrapper or actual Python argv')
            if self.pending is not None:
                require(all(current[k]==self.pending[k] for k in ('pid','start_ticks','ppid','cgroup')),
                        'env exec must preserve actual PID/start/parent/cgroup')
            if current['argv']==wrapper:self.pending=copy.deepcopy(current)
            elif (value['ActiveState'],value['SubState'])==('active','running'):
                require(value['ExecMainPID']==str(pid),'Acknowledged main PID agrees with systemd')
                self.owner=dict(process=copy.deepcopy(current),invocation_id=self.invocation,
                                admission=self.ref,outer_owner=self.admission['outer_owner'])
        elif self.host.terminal(value) and self.owner is None:
            observation=dict(invocation_id=self.invocation,exec_main_pid=int(value['ExecMainPID']),
                             actual_start_ticks_observed=False)
            require(self.unacknowledged in (None,observation),'Unknown startup failure cannot be rebound')
            self.unacknowledged=observation
        return value

    def acknowledge(self,guard):
        deadline=self.host.clock()+ACK_SECONDS
        while True:
            guard();value=self.observe()
            if self.owner is not None:
                require(self.host.proc(self.owner['process']['pid'])==self.owner['process'],
                        'Same independent actual Python worker at ACK publication')
                worker.service_identity(self.spec,self.ref,value,self.owner)
                self.host.publish(self.root/'worker-owner.json',self.owner)
                return copy.deepcopy(self.owner)
            require(value['LoadState']=='not-found' or not self.host.terminal(value),
                    'Actual worker exited before independent ACK; never fabricate start identity')
            require(self.host.clock()<deadline,'Bounded actual worker ACK wait')
            self.host.wait(.05)

    def live_or_terminal(self):
        value=self.observe()
        require(self.owner is not None,'Only acknowledged worker can execute a successful phase')
        if not self.host.terminal(value):
            worker.service_identity(self.spec,self.ref,value,self.owner)
            require(self.host.proc(self.owner['process']['pid'])==self.owner['process'],'Actual acknowledged process remains')
        return value

    def drained(self):
        if not self.attempted:return dict(physically_drained=True,no_service_launch_attempted=True,exit_observed=False)
        samples=[]
        for _ in range(2):
            value=self.observe(resources=False)
            if value['LoadState']=='not-found':
                status=None
            else:
                require(self.host.terminal(value),'Terminal actual service before unlock')
                status=exit_status(value,no_process_observed=self.owner is None and self.pending is None)
                if self.owner is not None:self.host.gone(self.owner['process'])
                elif self.pending is not None:self.host.gone(self.pending)
                elif int(value['ExecMainPID']):
                    require(self.host.proc(int(value['ExecMainPID'])) is None,
                            'Uncaptured terminal PID entirely absent; cannot infer its start ticks')
            samples.append(dict(properties=copy.deepcopy(value),group=self.host.empty_group(self.group),status=status))
        require(samples[0]==samples[1],'Two complete stable terminal/recursive-empty observations')
        proof=dict(physically_drained=True,owner=copy.deepcopy(self.owner),pending=copy.deepcopy(self.pending),
                   unacknowledged=copy.deepcopy(self.unacknowledged),samples=samples,
                   exit_observed=samples[0]['status'] is not None and samples[0]['status']['ExecMainCode'] in (1,2,3),
                   stop_requested=self.stop_requested,
                   outer_self_exit_proven=False)
        self.saved_terminal=copy.deepcopy(proof)
        return proof

    def cleanup(self):
        if not self.attempted:return
        require(read(self.intent_ref)==self.intent,'Cleanup binds exact durable launch intent')
        value=self.observe(resources=False)
        if value['LoadState']=='not-found':return
        try:return self.drained()
        except (ValueError,RuntimeError):pass
        self.stop_requested=True
        self.observe(resources=False)
        self.host.command(['systemctl','--user','kill','--kill-who=all','--signal=SIGTERM',self.spec['unit']],timeout=10)
        deadline=self.host.clock()+CLEANUP_SECONDS;killed=False
        while True:
            self.observe(resources=False)
            try:return self.drained()
            except (ValueError,RuntimeError):pass
            if not killed and self.host.clock()>=deadline-(CLEANUP_SECONDS-TERM_GRACE_SECONDS):
                self.observe(resources=False)
                self.host.command(['systemctl','--user','kill','--kill-who=all','--signal=SIGKILL',self.spec['unit']],timeout=10)
                killed=True
            require(self.host.clock()<deadline,'Owned cleanup incomplete: retained lock must not unwind')
            self.host.wait(.1)


class HistoricalReader:
    """Original full metadata transports, with retained identities for every read."""
    def __init__(self,history,guard):self.history,self.guard,self.seen=history,guard,{}
    def read(self,ref):
        self.guard();before=self.history.serial.identity(ref['path'])
        value=self.history.serial.sources.load(ref)
        require(self.history.serial.identity(ref['path'])==before,'Historical metadata changed during read')
        record=dict(reference=copy.deepcopy(ref),identity=before)
        require(ref['path'] not in self.seen or self.seen[ref['path']]==record,'No substituted historical reference')
        self.seen[ref['path']]=record;return value
    def unchanged(self):
        for path,row in self.seen.items():require(self.history.serial.identity(path)==row['identity'],'Historical input changed')


def validate_worker_result(envelope,spec,admission_ref,event_receipt):
    require(envelope['admission']==admission_ref and envelope['worker_exit_or_outer_drain_proven']is False,
            'Actual metadata collector provenance; no self-exit claim')
    result=envelope['result'];plan=spec['preflight_plan']
    require(result['schema']=='spark26_elf_preflight_result_v1'and result['metadata_collection_complete']is True
        and result['plan_sha256']==worker.contract.digest(plan)and result['counts']==worker.COUNTS,
        'Complete exact metadata preflight scope')
    require(all(result[k]is False for k in ('conflicts_resolved','dependency_closure_proven',
        'source_candidate_runtime_admitted','native_imports_executed','model_checkpoint_reads','package_mutations',
        'full_model_qualified','new_speedup_claim','worker_exit_or_outer_drain_proven')),
        'Metadata completion cannot admit native runtime or qualify model')
    require(result['payload_count']==plan['payload_count']and result['payload_bytes']==plan['payload_bytes']
        and result['payload_sha256']==plan['payload_sha256']and result['conflicting_owner_records_preserved']==15,
        'Entire original source candidate unchanged')
    require(result['all_manifest_payloads_authenticated']==(result['authentication_failures']==0)
        and result['all_candidate_ELF_metadata_collected']==(result['ELF_metadata_failures']==0),
        'Actual partial failures cannot be hidden by summary flags')
    h,e=result['actual_hash_read_requests_bytes'],result['actual_ELF_read_requests_bytes']
    require(type(h)is int and type(e)is int and 0<=h<=worker.core.CAPS['payload_hash_bytes']
        and 0<=e<=worker.core.CAPS['ELF_metadata_bytes']and h+e<=worker.core.CAPS['total_payload_and_ELF_bytes'],
        'Original finite actual read-request ceilings')
    require(event_receipt==envelope['event_receipt']and event_receipt['complete']is True
        and event_receipt['counts']==worker.COUNTS and event_receipt['bytes']<=worker.BUDGET['event_total_bytes'],
        'Exact complete finite event receipt')


def verify_event_files(root,event_receipt,guard,spec,result):
    from spark26_native_provider_v2 import merge_manifest
    payloads=merge_manifest(read(spec['native_manifest']),spec['source_sha256'],lambda ref:worker.metadata_bytes(ref).decode())
    ordered=sorted(payloads);indexes={path:index for index,path in enumerate(ordered)}
    wanted=dict(payload_authentication=list(range(len(ordered))),
        ELF_metadata=[indexes[path]for path in spec['preflight_plan']['ELF_paths']],
        conflicting_owner=sorted(indexes[row['path']]for row in spec['preflight_plan']['conflicting_owners']))
    seen={kind:[]for kind in worker.COUNTS};references={};total=0;failures=0;ELF_failures=0
    expected_names={kind+'-'+str(i//4096).zfill(2)+'.jsonl'for kind,count in worker.COUNTS.items()for i in range(count)}
    require(set(event_receipt['files'])==expected_names,'Exact finite segmented metadata event files')
    for name in sorted(expected_names):
        guard();path=Path(root)/name;before=file_identity(path)
        require(before[2]<=worker.BUDGET['single_file_bytes'],'Original finite event file')
        with path.open('rb')as stream:raw=stream.read(worker.BUDGET['single_file_bytes']+1)
        require(len(raw)==before[2]and file_identity(path)==before and raw.endswith(b'\n'),'Stable complete event bytes')
        ref=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
        require(event_receipt['files'][name]==ref,'Actual event hash equals worker receipt')
        kind,segment=name[:-6].rsplit('-',1)
        rows=raw.splitlines(keepends=True)
        require(len(rows)==min(4096,worker.COUNTS[kind]-int(segment)*4096),'Every segment exact row count')
        for line in rows:
            require(len(line)<=worker.BUDGET['event_row_bytes'],'Original metadata row cap')
            value=json.loads(line);require(set(value)=={'event','row'}and value['event']==kind,'Exact event envelope')
            row=value['row'];index=row['index'];require(type(index)is int and 0<=index<len(ordered),'Valid original payload index')
            seen[kind].append(index);expected=payloads[ordered[index]]
            if kind=='payload_authentication':
                require(row['status']in ('observed','failed'),'No unknown authentication state')
                if row['status']=='failed':failures+=1
                else:
                    require(len(row['identity'])==5 and row['identity'][2]==expected['bytes']
                        and re.fullmatch('[a-f0-9]{64}',row['actual_sha256'])
                        and row['selected_RECORD_match']==(row['actual_sha256']==expected['sha256']),
                        'Observed complete original file identity and honest hash comparison')
                    failures+=not row['selected_RECORD_match']
            elif kind=='ELF_metadata':
                require(row['status']in ('observed','unresolved'),'No unknown ELF collection state')
                ELF_failures+=row['status']=='unresolved'
                if row['status']=='observed':
                    require(row['ELF']['actual_loader_selection_proven']is False
                        and row['ELF']['dependency_closure_proven']is False,'Static metadata cannot claim actual loader selection')
            else:
                conflict=next(c for c in spec['preflight_plan']['conflicting_owners']if c['path']==ordered[index])
                require(row['selected']==conflict['selected_record']and row['ambient']==conflict['ambient_record']
                    and row['unselected_distribution']==conflict['unselected_distribution']and row['resolved_ownership']is False,
                    'All original conflicting claims survive metadata collection')
        total+=len(raw);references[name]=ref
    require(seen==wanted and total==event_receipt['bytes']and total<=worker.BUDGET['event_total_bytes'],
            'Every original payload, ELF candidate and conflicting owner appears exactly once in order')
    require(failures==result['authentication_failures']and ELF_failures==result['ELF_metadata_failures'],
            'Summary failure counts match complete individual events')
    return references


def release_after_drain(host,lease,service,root):
    """This function cannot unwind ownership merely because cleanup/write failed."""
    def record_pending(value):
        if root is not None:host.publish(root/'cleanup-pending.json',value)
    cleanup=(lambda:None) if service is None else service.cleanup
    verify=(lambda:dict(physically_drained=True,no_service_launch_attempted=True,exit_observed=False)) if service is None else service.drained
    proof=host.hold(cleanup,verify,record_pending,lambda:host.wait(1.))
    try:
        if root is not None:host.publish(root/'cleanup-drained.json',proof)
    except BaseException:
        # Already-proven physical drain permits release even if disk reporting
        # fails. It cannot turn the run into a successful published result.
        pass
    lease.release(proof)
    return proof


def outer(binding_ref,review_ref):
    require(__debug__ and sys.version_info[:2]==(3,10) and sys.executable==PYTHON
            and sys.flags.no_site and sys.dont_write_bytecode,'Original Python 3.10 -S -B outer')
    binding,review=read(binding_ref),read(review_ref)
    spec=read(binding['worker_spec']);worker_review=read(binding['worker_review'])
    validate_binding(binding,review,binding_ref,spec,worker_review)
    root=Path(spec['output_root']);prefix=str(root/'tmp'/'outer-absent-pycache')
    require(sys.pycache_prefix==prefix and os.environ.get('PYTHONPYCACHEPREFIX')==prefix
            and os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and not os.path.lexists(prefix),
            'Outer absent startup bytecode namespace')
    os.sched_setaffinity(0,set(OUTER_LIMITS['cpu_affinity']));os.setpriority(os.PRIO_PROCESS,0,OUTER_LIMITS['nice'])
    for key,limit in ((resource.RLIMIT_AS,OUTER_LIMITS['address_space_bytes']),
                      (resource.RLIMIT_FSIZE,OUTER_LIMITS['single_file_bytes'])):
        resource.setrlimit(key,(limit,limit))
    host=RealHost(spec['source_sha256']);host.p.user_bus()
    require(host.proc(os.getpid())['argv']==outer_argv(binding_ref,review_ref),'Actual exact outer invocation')
    require(not os.path.lexists(root),'Fresh private Spark output namespace')
    # Parse every real frozen runtime RECORD now, but never read native payloads
    # or original weights in the controller. Missing manifest fails before lock.
    from spark26_native_provider_v2 import merge_manifest
    manifest=read(spec['native_manifest'])
    merge_manifest(manifest,spec['source_sha256'],lambda ref:worker.metadata_bytes(ref).decode('utf8'))
    lease=RetainedLease(host);service=None;created=False;history=None;reader=None;latest=None
    started=host.clock();handlers={};primary=None

    def guard():
        lease.check();host.unchanged();host.p.clear_window()
        require(host.clock()-started<worker.BUDGET['total_wall_seconds'],'Finite complete outer work window')
        host.p.active_heavy((spec['unit'],) if service is not None and service.attempted else ())
        if latest is not None:require(history.serial.latest_handoff()==latest,'No newer/replaced root handoff')
        if reader is not None:reader.unchanged()
        if created:worker.WorkerContext.output_usage(type('OutputView',(),{'root':root})())

    def interrupted(signum,frame):raise InterruptedError('Spark outer interrupted: '+str(signum))

    lease.acquire()
    try:
        for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            handlers[signum]=signal.signal(signum,interrupted)
        guard();history=host.load_history(lease);reader=HistoricalReader(history,lease.check)
        scope=history.resolve(spec['serial_catalog'],spec['terminal_handoff'],reader.read,guard)
        latest=history.serial.latest_handoff()
        require(latest['path']==scope['freshness']['latest_handoff']['path']
                and latest['identity']==reader.seen[latest['path']]['identity'],'Complete freshly bound terminal metadata')
        drains=history.collect_drain(scope,guard);guard()
        require(host.properties(spec['unit'])['LoadState']=='not-found','Unused exact service namespace')
        for path,floor in ((ROOT,worker.BUDGET['root_floor_bytes']),
                           (Path('/mnt/nvme'),worker.BUDGET['nvme_floor_bytes']+worker.BUDGET['qwen_reserved_bytes']
                            +worker.BUDGET['output_total_bytes'])):
            fs=os.statvfs(path);require(fs.f_bavail*fs.f_frsize>=floor,'Fresh original floors plus full output reservation')
        require(Path('/mnt/nvme').is_mount() and Path('/mnt/nvme').stat().st_dev!=ROOT.stat().st_dev,'Actual separate NVMe mount')
        # No checkpoint path or weight existence/read is part of metadata preflight.
        proof=dict(outer_owner=lease.owner,scope=scope,observed=drains)
        proof_bytes=len(encode(proof))
        require(proof_bytes<=worker.BUDGET['single_file_bytes'],'Complete unprojected history must fit original single receipt cap')
        require(proof_bytes+4*2**20<=worker.BUDGET['receipt_total_bytes'],'Reserve remaining control/result receipts without reducing history')
        root.mkdir(mode=0o700);created=True;(root/'tmp').mkdir(mode=0o700)
        proof_ref=host.publish(root/'predecessor-drain.json',proof)
        host.publish(root/'outer-binding.json',dict(binding=binding_ref,review=review_ref,outer_owner=lease.owner,
            host_boundary=HOST_BOUNDARY,outer_limits=OUTER_LIMITS,history_ctypes_origins=host.finder.ctypes_origins,
            initial_descriptor=lease.initial,historical_input_identities=reader.seen))
        admission=dict(schema='spark26_elf_preflight_worker_admission_v1',spec=binding['worker_spec'],
            review=binding['worker_review'],outer_owner=lease.owner,retained_fd=lease.fd,lock_identity=LOCK,
            metadata_preflight_admitted=True,runtime_admitted=False,predecessor_drain=proof_ref,latest_handoff_identity=latest)
        admission_ref=host.publish(root/'admission.json',admission)
        service=OwnedService(host,spec,admission_ref,admission)
        guard();service.launch();service.acknowledge(guard)
        while True:
            guard();value=service.live_or_terminal()
            if host.terminal(value):break
            host.wait(.5)
        drained=service.drained()
        exit_ref=host.publish(root/'service-exit.json',dict(outer_owner=lease.owner,admission=admission_ref,drain=drained))
        require(drained['exit_observed'] is True and drained['samples'][0]['status']==
                dict(ExecMainCode=1,ExecMainStatus=0,Result='success') and service.stop_requested is False,
                'Only actual clean acknowledged worker exit can validate complete results')
        result_ref=observed_ref(root/'result.json');events_ref=observed_ref(root/'event-receipt.json')
        result,events=read(result_ref),read(events_ref)
        validate_worker_result(result,spec,admission_ref,events)
        event_files=verify_event_files(root,events,guard,spec,result['result'])
        guard()
        return host.publish(root/'outer-result.json',dict(complete=True,passed=True,outer_owner=lease.owner,
            service_exit=exit_ref,worker_result=result_ref,event_receipt=events_ref,event_files=event_files,admission=admission_ref,
            metadata_collection_complete=True,source_candidate_runtime_admitted=False,dependency_closure_proven=False,
            full_model_qualified=False,new_speedup_claim=False,outer_self_exit_proven=False))
    except BaseException as error:
        primary=error
        if created:
            try:host.publish(root/'outer-failure.json',dict(error_type=type(error).__name__,error=str(error)[:2048],
                                                         outer_owner=lease.owner,success=False))
            except BaseException:pass
        raise
    finally:
        # During pending cleanup every catchable signal requests another check;
        # no exception may unwind the retained descriptor before physical drain.
        def retain(signum,frame):return None
        for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGUSR1):
            if signum not in handlers:handlers[signum]=signal.getsignal(signum)
            signal.signal(signum,retain)
        release_after_drain(host,lease,service,root if created else None)
        for signum,handler in handlers.items():signal.signal(signum,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('binding','binding-sha256','review','review-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    outer(dict(path=args.binding,sha256=args.binding_sha256),dict(path=args.review,sha256=args.review_sha256))
