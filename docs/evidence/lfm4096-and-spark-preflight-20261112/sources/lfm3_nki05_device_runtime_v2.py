"""Concrete admitted device context; actual PID/descriptor/service authority.

No native import or runtime action on module import. Constructed only by the
fixed new serial child after an exact future root-reviewed admission.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import sys
import time

from lfm3_nki05_device_context_v1 import DeviceContext,ARTIFACT_SCHEMA,VARIANT
from slot_cache_nki05_typed_jit_v1 import DEVICE_SOFTWARE
import lfm3_nki05_device_protocol_v2 as protocol
import lfm3_nki05_device_budget_v1 as budget
from minicpm5_layer0_runtime_auth_v1 import identity


def require(value,message):
    if not value:raise RuntimeError(message)


def stable_hash(path,cap):
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path and not any(p.is_symlink() for p in (path,*path.parents)),'Exact no-symlink owned artifact path')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=cap,'Bounded regular artifact')
        digest=hashlib.sha256();size=0
        while True:
            block=os.read(fd,2**20)
            if not block:break
            size+=len(block);require(size<=cap,'Complete bounded artifact hash');digest.update(block)
        after=os.fstat(fd)
        require(size==before.st_size and all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Stable complete artifact bytes')
        return digest.hexdigest()
    finally:os.close(fd)


class Context(DeviceContext):
    def __init__(self,host,admission_ref,task,cpu_proof):
        require(Path(host.__file__).resolve()==Path(__file__).resolve().parent/'run_lfm3_nki05_device_serial_v2.py','Only fixed concrete serial owner')
        require(task in protocol.tasks(),'Complete fixed device task')
        self.host,self.admission_ref,self.task=host,admission_ref,dict(task)
        self.phase=task['phase'];self.cpu_proof=cpu_proof;self.sdk_verified=False
        self.admission=host.p.read(admission_ref['path'],admission_ref['sha256'])
        self.spec=host.read_bundle(self.admission['spec'],self.admission['review']);self.run_root=Path(self.spec['output_root'])
        require(admission_ref['path']==str(self.run_root/'admission.json') and self.admission['schema']=='lfm3_nki05_device_admission_v1'
                and self.admission['lock_identity']==host.LOCK and self.admission['runtime_admitted'] is self.admission['predecessors_checked'] is True,'Exact outer-produced admission')
        require(cpu_proof['binding']==self.spec['full_cpu_prerequisite'] and cpu_proof['actual_outer_exit_verified'] is True,'Complete actual r2 predecessor audit')
        saved=host.p.read(self.admission['full_cpu_prerequisite']['path'],self.admission['full_cpu_prerequisite']['sha256'])
        require(saved==cpu_proof,'Exact outer-observed complete r2 audit')
        self.parent=host.p.read(self.run_root/'worker-owner.json')
        self.process=host.p.proc(os.getpid())
        self.ack=host.p.read(self.run_root/'tasks'/(protocol.key(task)+'-owner.json'))
        from lfm3_nki05_device_pipeline_v2 import child_argv,child_environment
        require(sys.dont_write_bytecode and __debug__ and sys.executable==host.PYTHON and sys.version_info[:2]==(3,12),'Pinned native child interpreter')
        require(self.process['pid']==self.ack['pid'] and self.process['start_ticks']==self.ack['start_ticks']
                and self.process['ppid']==self.parent['pid'] and self.process['argv']==self.ack['argv']==child_argv(host,admission_ref,task)
                and self.process['cgroup']=='0::'+self.ack['cgroup']
                and self.ack['parent']==dict(self.parent,cgroup=self.parent['cgroup'].removeprefix('0::')),'Exact actual child ACK and parent identity')
        require(all(os.environ.get(k)==v for k,v in child_environment(host,self.run_root,self.admission['attempt_id'],task).items()),'Exact clean isolated compiler child environment')
        require(os.sched_getaffinity(0)=={6,7} and os.getpriority(os.PRIO_PROCESS,0)==19,'Actual bounded child CPU policy')
        for kind,cap in ((resource.RLIMIT_AS,24*2**30),(resource.RLIMIT_CPU,budget.WORKER_SECONDS),(resource.RLIMIT_FSIZE,budget.COMPILER_FILE_BYTES)):
            require(resource.getrlimit(kind)==(cap,cap),'Inherited finite hard/soft device budgets')
        self.deadline=time.monotonic()+(budget.PHASE_SECONDS if task['index']==-1 else budget.CHILD_SECONDS)
        self.saved={};self.artifacts=[];self.metadata={};self.previous={}
        self.authorize(self.phase)

    def fast_authorize(self):
        require(time.monotonic()<self.deadline and self.host.p.proc(os.getpid())==self.process,'Original child PID/start/argv/cgroup and finite deadline')
        require(self.host.p.proc(self.parent['pid'])=={k:self.parent[k] for k in ('pid','start_ticks','ppid','argv','cgroup')},'Actual acknowledged parent remains alive')
        self.host.descriptor_proof(self.admission['outer_owner'],self.admission['retained_fd'])
        return True

    def authorize(self,phase):
        require(phase==self.phase and phase in protocol.PHASES,'Only current concrete scientific phase')
        self.fast_authorize();h=self.host
        h.bytecode_guard(self.run_root)
        require(not any(str(getattr(module,'__file__','')).endswith('.pyc') for module in tuple(sys.modules.values())),'No loaded sourceless bytecode is admitted')
        h.p.clear_window();h.check_latest(self.admission['latest_handoff_identity'])
        h.service_identity(self.spec,self.admission_ref,self.admission['attempt_id'],h.properties(self.spec['unit']),self.parent)
        h.p.active_heavy((self.spec['unit'],));h.usage(self.run_root);h.verify_sources(self.spec)
        require({path:h.p.sha(path) for path in self.spec['canonical_sha256']}==self.spec['canonical_sha256'],'Bound canonical metadata unchanged')
        require(all(identity(path)==stamp for path,stamp in self.metadata.items()),'Prior complete phase metadata has not rebound')
        require('lfm3_nki05_compact_writeback_v1' not in sys.modules and 'lfm3_nki05_writeback_probe_v1' not in sys.modules,'No simulator writeback workaround in device worker')
        return True

    def prerequisite(self,name):
        if name=='full_cpu':return self.cpu_proof['full_cpu_result']
        expected=protocol.PHASES[protocol.PHASES.index(self.phase)-1] if self.phase!='raw_device' else None
        require(name==expected,'Exact complete immediately preceding phase')
        if name not in self.previous:
            from lfm3_nki05_device_pipeline_v2 import read_phase
            self.previous[name]=read_phase(self.host,self.run_root,name,self.spec['source_sha256'],self.admission_ref)
            path=self.run_root/name/'phase-complete.json';self.metadata[str(path)]=identity(path)
        return self.previous[name]

    def paths(self,phase):
        require(phase==self.phase,'Owned phase output only')
        return self.host.ROOT,self.run_root/phase

    def sources(self):return self.spec['source_sha256']

    def writer(self,phase):
        require(self.sdk_verified,'Complete actual SDK RECORD authentication before real JIT')
        return super().writer(phase)

    def save(self,path,value,phase):
        self.authorize(phase);path=Path(path)
        require(path==protocol.scientific_path(self.run_root,self.task),'Only this task original scientific progress report')
        require(path.resolve()==path and not path.is_symlink(),'Exact scientific JSON target')
        if str(path) in self.saved:require(self.host.p.sha(path)==self.saved[str(path)],'Never overwrite another writer or changed evidence')
        else:require(not path.exists(),'Fresh scientific report')
        raw=self.host.encode(value);require(len(raw)<=budget.METADATA_BYTES,'Complete bounded scientific report')
        temp=path.with_name('.'+path.name+'.progress.tmp');require(not temp.exists(),'Exclusive progress temporary file')
        with temp.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        temp.replace(path);self.saved[str(path)]=hashlib.sha256(raw).hexdigest()
        self.fast_authorize()

    def hash_artifact(self,path,phase):
        self.authorize(phase);path=Path(path)
        require(path.name=='model.pt' and path.is_relative_to(self.run_root/phase),'Actual newly compiled phase artifact only')
        value=stable_hash(path,budget.ARTIFACT_BYTES);self.fast_authorize();return value

    def record_artifact(self,path,phase,case):
        self.authorize(phase);path=Path(path)
        value=dict(schema=ARTIFACT_SCHEMA,variant=VARIANT,software=DEVICE_SOFTWARE,source_sha256=self.sources(),
            phase=phase,path=str(path),case=case,sha256=self.hash_artifact(path,phase),
            admission=self.admission_ref,child=self.process,task=self.task)
        receipt=path.parent/'artifact-identity.json';self.host.once(receipt,value)
        self.artifacts.append(self.host.reference(receipt));return value
