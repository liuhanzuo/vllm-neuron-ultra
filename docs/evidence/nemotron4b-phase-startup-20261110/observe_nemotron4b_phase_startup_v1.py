"""Observe original -S startup and bounded source metadata, without running site.

This is not worker admission or an interpreter/native byte measurement. Actual
site hooks, installed model/SDK modules, weights and compiled artifacts are not
executed or explicitly opened. The interpreter loads its usual stdlib support.
"""
import sys

INITIAL_PATH=list(sys.path)
INITIAL_PREFIXES={k:getattr(sys,k)for k in('prefix','exec_prefix','base_prefix','base_exec_prefix')}
INITIAL_MODULES=sorted(sys.modules)

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
PLAN=ROOT/'research/nemotron4b-native-prerequisite-plan-20261106-v2.json'
PLAN_SHA='7e5f16c07ad07c56b5bd3e6f5122ac6dd13b5e09d387da18bf116e035d81a2fd'
FIELDS=('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')


def require(value,message):
    if not value:raise ValueError(message)


def identity(s):return {k:getattr(s,k)for k in FIELDS}


def source(path):
    p=Path(path)
    require(p.suffix in('.py','.pth','.json')or p.name=='pyvenv.cfg','Source metadata only')
    require(p.is_absolute()and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents)),
            'Canonical source; aliases recorded separately')
    before=identity(p.lstat())
    require(stat.S_ISREG(before['st_mode'])and 0<=before['st_size']<=8*2**20,'Bounded ordinary source')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        require(identity(os.fstat(fd))==before,'Exact source descriptor')
        pieces=[];size=0
        while True:
            raw=os.read(fd,min(65536,8*2**20+1-size))
            if not raw:break
            pieces.append(raw);size+=len(raw);require(size<=8*2**20,'Source cap')
        raw=b''.join(pieces)
        require(identity(os.fstat(fd))==before==identity(p.lstat())and len(raw)==before['st_size'],
                'Stable complete original source')
        raw.decode('utf-8')
        return raw,dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                        fullstat=before,original_nlink=before['st_nlink'])
    finally:os.close(fd)


def observe(phase):
    require(phase in('native-meta','nxd-meta'),'Original phase')
    raw,plan_ref=source(PLAN);require(plan_ref['sha256']==PLAN_SHA,'Original plan pin')
    plan=json.loads(raw);expected=plan['environments'][phase]
    require(sys.executable==expected['python']and list(sys.version_info[:2])==expected['python_version'],
            'Actual original interpreter path and version')
    require(sys.flags.no_site==sys.flags.no_user_site==1 and sys.dont_write_bytecode
            and sys.flags.optimize==0 and 'site'not in sys.modules,'No automatic site execution')
    venv=Path(sys.executable).parent.parent;version='%d.%d'%sys.version_info[:2]
    package=venv/'lib'/('python'+version)/'site-packages'
    spec=importlib.util.find_spec('site');site_file=Path(sys.base_prefix)/'lib'/('python'+version)/'site.py'
    files={};texts={};aliases={};candidates=[]
    def record_aliases(path):
        p=Path(path)
        for item in(p,*p.parents):
            if item.is_symlink():
                aliases[str(item)]=dict(link_text=os.readlink(item),target=str(item.resolve()),
                                       fullstat=identity(item.lstat()),target_is_directory=item.resolve().is_dir())
    def add(path,text=False):
        record_aliases(path);resolved=Path(path).resolve();raw,row=source(resolved)
        files[str(resolved)]=row
        if text:texts[str(path)]=raw.decode('utf-8')
    pth=sorted(str(p)for p in package.glob('*.pth'))
    for path in[venv/'pyvenv.cfg',site_file,site_file.parent/'_sitebuiltins.py',*map(Path,pth)]:
        add(path,text=path.suffix=='.pth'or path.name=='pyvenv.cfg')
    for directory in(site_file.parent,package):
        p=directory/'sitecustomize.py';candidates.append(dict(path=str(p),exists=p.exists()))
        if p.exists():add(p)
    for path in[sys.executable,*INITIAL_PATH,*INITIAL_PREFIXES.values()]:record_aliases(path)
    require(sys.path==INITIAL_PATH and 'site'not in sys.modules
            and not {'torch','transformers','vllm','neuronx_distributed','libneuronxla'}&set(sys.modules),
            'Only stdlib observation, no site/SDK/model import')
    return dict(schema='nemotron4b_original_phase_startup_observation_v1',phase=phase,
        original_plan=plan_ref,executable=sys.executable,resolved_executable=str(Path(sys.executable).resolve()),
        version=sys.version,version_info=list(sys.version_info[:3]),initial_path=INITIAL_PATH,
        initial_prefixes=INITIAL_PREFIXES,initial_modules=INITIAL_MODULES,
        builtin_module_names=list(sys.builtin_module_names),no_site=sys.flags.no_site,
        no_user_site=sys.flags.no_user_site,site_spec=dict(name=spec.name,origin=spec.origin,
        loader_module=spec.loader.__module__,loader_name=getattr(spec.loader,'__name__',type(spec.loader).__name__)),
        site_file=str(site_file),site_packages=str(package),pth_files=pth,source_files=files,
        original_config_and_pth_text=texts,sitecustomize_candidates=candidates,aliases=aliases,
        installed_site_executed=False,actual_model_or_SDK_executed=False,
        original_native_payloads_explicitly_read_or_hashed=0,
        actual_frozen_payload_authenticated=False,complete_runtime_dependency_closure_proven=False,
        runtime_admitted=False,new_speedup_claim=False)


if __name__=='__main__':
    require(len(sys.argv)==2,'One original phase argument')
    print(json.dumps(observe(sys.argv[1]),sort_keys=True))
