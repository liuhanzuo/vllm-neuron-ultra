"""Runtime authentication only; no launcher or autonomous admission."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import copy
import json
import os
from pathlib import Path
import re
import stat
import sys
ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
_ACTIVE=ContextVar('mini_layer0_runtime_context',default=None)
MIB=2**20

def require(value,message):
    if not value:raise RuntimeError(message)
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def clear_window():
    for name in ('smollm3-cache-ablation-full-v8','lfm3-package-paired-full-v2'):
        p=ROOT/name/'status.json';require(p.stat().st_size<16384,'Status size')
        require(json.loads(p.read_text())['performance_window'] is False,'Timed workload active')
def identity(path):
    p=Path(path);require(p.is_absolute() and p.resolve()==p,'Canonical absolute input')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'Input symlink')
    s=p.lstat();require(stat.S_ISREG(s.st_mode),'Regular input')
    return dict(realpath=str(p),device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def current_context():
    value=_ACTIVE.get();require(value is not None,'No independently admitted layer0 runtime');return value

class RuntimeContext:
    def __init__(self,plan,plan_sha256,permit,external_guard):
        require(digest(plan)==plan_sha256,'Exact runtime plan digest')
        require(plan['schema']=='minicpm5_layer0_runtime_inputs_v1','Runtime input schema')
        self.plan=copy.deepcopy(plan);self.plan_sha256=plan_sha256;self.permit=copy.deepcopy(permit);self.external_guard=external_guard
        self._verified={};self._sources_checked=False;self._cpu_before=None;self._cpu_after=None;self.guard(permit['role'])
    def guard(self,role=None):
        clear_window();require(self.external_guard()==self.permit,'External admission changed');p=self.permit
        require(p['schema']=='minicpm5_layer0_runtime_permit_v1' and p['plan_sha256']==self.plan_sha256,'Runtime permit binding')
        require(p['role'] in ('cpu','native') and role in (None,p['role']),'Wrong runtime role')
        require(re.fullmatch('[0-9a-f]{32}',p['attempt_id']) is not None,'Runtime attempt')
        require(p['serial_exclusive'] is True and p['source_review_accepted'] is True and p['all_predecessors_drained'] is True,'External serial/source prerequisite')
        require(p['pid']==os.getpid() and p['uid']==os.getuid()==1000,'Runtime process identity')
        require(p['start_ticks']==Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'Runtime process start')
        require(Path('/proc/self/cgroup').read_text().strip()==p['cgroup'],'Runtime cgroup')
        require([x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]==p['argv'],'Runtime argv')
        require(sys.executable==p['python'] and sys.dont_write_bytecode and __debug__,'Runtime interpreter')
        require(p['python']==self.plan['interpreters'][p['role']],'Fixed interpreter')
        require(all(os.environ.get(k)==v for k,v in p['environment'].items()),'Admitted environment changed')
        require(p['output_root']==self.plan['output_root'],'Fixed output tree')
        if p['role']=='cpu':
            require(p['rank'] in (0,1) and p['arm'] in ('C_Hd_control','C_Hc_history_generation'),'CPU rank/arm')
            require(os.environ.get('NXD_CPU_MODE')=='1' and os.environ.get('RANK')==str(p['rank']) and os.environ.get('LOCAL_RANK')==str(p['rank']),'CPU rank environment')
            from minicpm5_history_environment_v1 import check_before,check_after,PJRT_KEYS,FORBIDDEN
            require(not set(p['environment'])&set((*PJRT_KEYS,*FORBIDDEN,'NEURON_RT_VISIBLE_CORES')),'SDK transition keys must not be hardcoded in stable environment')
            if self._cpu_before is None:
                require(not {'torch','torch_neuronx'}&set(sys.modules),'Authenticate before native imports');self._cpu_before=check_before(os.environ)
            elif self._cpu_after is None:require(check_before(os.environ)==self._cpu_before,'Preimport environment changed')
            else:require(check_after(self._cpu_before,os.environ)==self._cpu_after,'SDK environment changed')
        else:require(p['rank']==-1 and p['arm']=='HF_native','Native role')
        return p
    def record_cpu_import(self):
        require(self._cpu_before is not None and self._cpu_after is None,'Single authenticated CPU import')
        from minicpm5_history_environment_v1 import check_after,SDK
        for name in ('torch_neuronx','torch_neuronx.xla','torch_neuronx.initialization','torch_neuronx.initialization.initializer'):
            module=sys.modules[name];relative=name.replace('.','/')+('/__init__.py' if hasattr(module,'__path__') else '.py')
            require(Path(module.__file__).resolve()==SDK/relative,'Pinned SDK module path')
        initializer=sys.modules['torch_neuronx.initialization']._initializer
        require(initializer is not None and initializer.configured_pjrt_env is True and initializer.previous_pjrt_env_vars['NEURON_RT_VISIBLE_CORES']=='0-1','Original SDK initializer')
        self._cpu_after=check_after(self._cpu_before,os.environ);self.guard('cpu')
        return dict(before=self._cpu_before,after=self._cpu_after,environment_modified_by_diagnostic=False)
    @contextmanager
    def activate(self):
        self.guard();require(_ACTIVE.get() is None,'Nested runtime context');token=_ACTIVE.set(self)
        try:yield self
        finally:_ACTIVE.reset(token)
    def verify_sources(self):
        self.guard()
        for path,wanted in self.plan['source_sha256'].items():
            self.guard();p=Path(path);before=identity(p)
            require(p.suffix not in ('.pt','.safetensors','.bin','.neff','.so','.whl','.raw','.zlib','.pb') and before['size']<8*MIB,'Source file size/type')
            require(hashlib.sha256(p.read_bytes()).hexdigest()==wanted and identity(p)==before,'Frozen source changed')
        self._sources_checked=True
    def verify_input(self,path):
        self.guard();require(self._sources_checked,'Authenticate complete source closure first')
        path=str(Path(path));require(path in self.plan['inputs'],'Input not admitted')
        row=self.plan['inputs'][path];before=identity(path);require(before==row['identity'],'Frozen input identity changed')
        if self._verified.get(path)!=before:
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                h=hashlib.sha256();count=0
                while True:
                    block=os.read(fd,MIB)
                    if not block:break
                    count+=len(block);require(count<=before['size'],'Input grew');h.update(block)
                require(count==before['size'] and h.hexdigest()==row['sha256'],'Frozen payload digest')
            finally:os.close(fd)
            require(identity(path)==before,'Input changed while hashing');self._verified[path]=before
        return row
    def read_json(self,path,wanted=None):
        row=self.verify_input(path);require(row['identity']['size']<8*MIB and Path(path).suffix=='.json','Small JSON input')
        require(wanted is None or row['sha256']==wanted,'JSON reference digest')
        value=json.loads(Path(path).read_text());require(identity(path)==row['identity'],'JSON input changed');return value
    def read_bytes(self,path,cap):
        row=self.verify_input(path);require(row['identity']['size']<=cap,'Payload allocation cap')
        value=Path(path).read_bytes();require(len(value)<=cap and identity(path)==row['identity'] and hashlib.sha256(value).hexdigest()==row['sha256'],'Payload changed during read');return value
    def load_pt(self,path,cap):
        self.guard();row=self.verify_input(path);require(row['identity']['size']<=cap,'PT input cap')
        import torch
        value=torch.load(path,map_location='cpu',weights_only=True)
        require(identity(path)==row['identity'],'PT changed while loading');return value
