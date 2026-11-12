"""Small shared contracts for separately root-reviewed Dory serial stages."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=ROOT/'research'
ENV=Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1')
MIB=2**20;GIB=2**30
STAGES=('inventory','abi','stage_a','review')
IDENTITY=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
# Inventory limits reproduce the accepted finish_v2 envelope. Native resource
# values are supplied by a separately reviewed concrete plan, not fabricated here.
INVENTORY=dict(address_space_bytes=2*GIB,cpu_seconds=600,wall_seconds=900,
               single_file_bytes=64*MIB,output_bytes=128*MIB,log_bytes=8*MIB,
               cpus=[5],nice=19,root_floor_bytes=28*GIB,nvme_floor_bytes=30*GIB,
               memory_available_floor_bytes=GIB)


def require(ok,message):
    if not ok:raise RuntimeError(message)


def identity(info):return {k:getattr(info,k)for k in IDENTITY}


def reference(path,cap=32*MIB):
    path=Path(path);require(path.is_absolute()and path.resolve()==path,'Canonical metadata/source path')
    require(path.suffix in ('.json','.py'), 'Only pinned metadata or source')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=cap,'Bounded ordinary metadata/source')
        parts=[];count=0
        while count<before.st_size:
            part=os.read(fd,min(65536,before.st_size-count));require(part,'Complete metadata/source');parts.append(part);count+=len(part)
        require(os.read(fd,1)==b''and identity(before)==identity(os.fstat(fd))==identity(path.lstat())and path.resolve()==path,'Stable metadata/source bytes')
        raw=b''.join(parts);return dict(path=str(path),bytes=count,sha256=hashlib.sha256(raw).hexdigest()),raw
    finally:os.close(fd)


def unique(pairs):
    result={}
    for key,value in pairs:require(key not in result,'Unique JSON keys');result[key]=value
    return result


def load(ref,cap=32*MIB):
    require(type(ref)is dict and set(ref)<={'path','sha256','bytes'}and {'path','sha256'}<=set(ref),'Exact metadata reference')
    observed,raw=reference(ref['path'],cap)
    require(observed['sha256']==ref['sha256']and ref.get('bytes',observed['bytes'])==observed['bytes'],'Pinned complete metadata SHA')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Finite JSON')))


def save(path,value,cap=8*MIB):
    raw=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode();require(len(raw)<=cap,'Bounded evidence write')
    path=Path(path);require(path.parent.resolve()==path.parent,'Canonical output parent')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as stream:
        require(stream.write(raw)==len(raw),'Complete evidence write');stream.flush();os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(directory)
    finally:os.close(directory)
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def verify_sources(pins,guard=lambda:None):
    require(type(pins)is dict and 1<=len(pins)<=100,'Bounded full local source closure')
    total=0
    for name,digest in pins.items():
        guard();require(Path(name).is_relative_to(ROOT)and Path(name).suffix=='.py','Local source closure only')
        ref,_=reference(name,2*MIB);total+=ref['bytes']
        require(total<=8*MIB and ref['sha256']==digest,'Complete exact local source SHA')
    return total


def verify_loaded(pins):
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_absolute()and Path(name).is_relative_to(ROOT):
            path=str(Path(name).resolve());require(path in pins,'Every loaded local helper reviewed')
            require(reference(path,2*MIB)[0]['sha256']==pins[path],'Loaded helper source unchanged')


def validate(plan,review,plan_ref):
    require(plan['schema']=='dory_serial_stage_plan_v1'and plan['stage']in STAGES,'Exact serial stage plan')
    require(plan['runtime_bound']is True and plan['automatic_successor_launch']is False,'Root actual binding; one stage only')
    require(review['schema']=='dory_serial_stage_source_review_v1'and review['reviewer']=='/root'
            and review['passed']is True and review['blockers']==[]and review['plan']==plan_ref
            and review['source_sha256']==plan['source_sha256'],'Exact root reviewed plan/closure')
    require(review['predecessor_exit']==plan['predecessor_exit']and review['current_physical_history_accepted']is True,'Latest predecessor/physical history reviewed')
    stage=plan['stage'];budget=plan['budget'];out=Path(plan['output_root'])
    require(out.parent==Path('/mnt/nvme')and out.name.startswith('dory-serial-'+stage.replace('_','-')+'-'),'Fixed fresh stage namespace')
    require(budget['root_floor_bytes']==28*GIB and budget['nvme_floor_bytes']==30*GIB,'Original disk floors')
    require(budget['cpus']==[5]and budget['nice']==19,'Original bounded diagnostic CPU placement')
    for key in ('address_space_bytes','cpu_seconds','wall_seconds','single_file_bytes','output_bytes','log_bytes','memory_available_floor_bytes'):
        require(type(budget[key])is int and budget[key]>0,'Explicit reviewed resource '+key)
    require(budget['log_bytes']<=8*MIB and budget['single_file_bytes']<=2*GIB,'Fixed evidence/file ceilings')
    if stage=='inventory':require(budget==INVENTORY,'Exact accepted inventory limits')
    if stage in ('abi','stage_a'):
        require(review['native_authority']==plan['native_authority']and review['native_preexec_authority_accepted']is True,'Native interpreter/preloads reviewed before exec')
    if stage=='stage_a':
        require(budget['output_bytes']==3*GIB and plan['checkpoint_identity']['size']==7950515472,'Original Stage A reservation/checkpoint size')
        require(review['stage_a_resource_estimate_accepted']is True,'Full model/extra traversal budget explicitly reviewed')
    require(len(plan['predecessor_owners'])>0 and type(plan['predecessor_expected_exit_code'])is int,'Real prior owners/expected actual status')
    return plan


def check_predecessor(record,expected):
    # Root binds one actual external exit record, including a legitimate failed
    # predecessor once independently drained. This never invents its tool exit.
    codes=[]
    if type(record.get('actual_exit_code'))is int:codes.append(record['actual_exit_code'])
    for key in ('actual_tool','tool'):
        value=record.get(key)
        if isinstance(value,dict)and type(value.get('exit_code'))is int:codes.append(value['exit_code'])
    require(codes and all(code==expected for code in codes),'Bound actual external predecessor exit')


def native_authority(plan,guard):
    """Authenticate the root's explicit closure decision, never infer from count."""
    decision=load(plan['native_authority'])
    require(decision['schema']=='dory_native_authority_root_acceptance_v1'and decision['reviewer']=='/root'
            and decision['accepted']is True and decision['unresolved']==[], 'Root-resolved native source/external/cache gaps')
    require(decision['native_files']==plan['native_files']and decision['inventory_result']==plan['inventory_result'],'Exact inventory decision binding')
    inventory=load(plan['inventory_result']);require(inventory['schema']=='dory_native_inventory_finishing_v2','Accepted finishing successor result')
    require(inventory['outputs']['files']==plan['native_files'],'Exact exported file reference')
    # Decision contains only current, independently authenticated runtime paths:
    # original interpreter, actual preloads/system refs and explicit cache/source
    # identities. Full SITE payload will be verified by the existing resolver.
    require(type(decision['preexec_files'])is list and 1<=len(decision['preexec_files'])<=256,'Bounded exact pre-exec files')
    known={};from dory_native_inventory_finish_v2 import file_bytes,Budget
    budget=Budget(guard)
    for row in decision['preexec_files']:
        require(set(row)=={'reference','identity'},'Exact pre-exec reference and identity')
        ref=row['reference'];require(set(ref)=={'path','bytes','sha256'},'Complete external expected content')
        result,_=file_bytes(ref,budget,64*MIB,budget_key='system_hash_bytes')
        require(result['identity']==row['identity'],'Current admitted external file identity')
        known[ref['path']]=ref
    require(str(ENV/'bin/python')in known and str(ENV/'lib/libiomp5.so')in known,'Original interpreter and OpenMP preload verified')
    require(decision['ld_preload']=='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:'+str(ENV/'lib/libiomp5.so'),'Exact original preloads')
    require(type(decision['preexec_links'])is list and len(decision['preexec_links'])<=32,'Bounded known loader symlinks')
    for row in decision['preexec_links']:
        guard();path=Path(row['path']);require(path.is_symlink()and os.readlink(path)==row['target']and str(path.resolve())==row['resolved']and row['resolved']in known,'Exact admitted loader link')
    require(any(row['path']=='/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4'for row in decision['preexec_links']),'Original tcmalloc symlink chain')
    return decision


def check_maps(decision,guard):
    """Observe only this admitted child; no all-machine scan or native probe."""
    guard();path=Path('/proc/self/maps')
    with path.open()as stream:text=stream.read(4*MIB+1)
    require(len(text)<=4*MIB,'Bounded actual mapped files')
    allowed=set(decision['mapped_file_paths']);require(len(allowed)<=100000,'Bounded admitted maps closure')
    actual=set()
    for line in text.splitlines():
        fields=line.split(None,5)
        if len(fields)<6 or not fields[5].startswith('/'):continue
        name=fields[5];require(not name.endswith(' (deleted)'),'No deleted loaded object')
        actual.add(name)
    missing=sorted(actual-allowed)
    return dict(actual_files=sorted(actual),unadmitted_files=missing,complete=not missing)
