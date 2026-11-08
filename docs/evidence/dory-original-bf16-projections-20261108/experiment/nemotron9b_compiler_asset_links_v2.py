"""Exact installed PWP/DVE resources authenticated against compiler RECORD.

Only pinned absolute targets within original model/bucket/subgraph scope.
"""
import hashlib,json,os,re,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ASSET_REF=dict(path=str(ROOT/'research/nemotron9b-installed-compiler-resources-20261108-v1.json'),sha256='8583bad9de38953c3497c4488bd89cb4e127700759da011b6b487286bb34c265')
CANONICAL_SHA='c8d36f7108771b98b4420f333a06c337fd0222a859cc0a3f0061824f3bcdc72c'
KEYS=('st_dev','st_ino','st_mode','st_uid','st_gid','st_nlink','st_size','st_mtime_ns','st_ctime_ns')
RELATIVE=re.compile(r'(context_encoding_model|token_generation_model)/_tp[01]_bk[0-5]/neuronxcc-[A-Za-z0-9_-]{1,64}/sg[0-9]{2}/[^/]+')

def require(value,message):
    if not value:raise RuntimeError(message)

def attributes(info):return {key:getattr(info,key)for key in KEYS}

def policy_shape(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    require(hashlib.sha256(raw).hexdigest()==CANONICAL_SHA,'Exact 83 RECORD-verified compiler assets and pins')
    return value

def load_policy():
    path=Path(ASSET_REF['path']);require(path.resolve()==path and not path.is_symlink(),'Canonical compiler asset policy')
    with path.open('rb')as f:raw=f.read(256*1024+1)
    require(len(raw)<=256*1024 and hashlib.sha256(raw).hexdigest()==ASSET_REF['sha256'],'Pinned small compiler asset policy')
    return policy_shape(json.loads(raw))

def verify_assets(policy,hash_content=False):
    policy_shape(policy)
    for target,row in policy['assets'].items():
        path=Path(row['path']);require(str(path)==target and path.parent==Path(row['root'])and path.name==row['name'] and path.resolve()==path,
                                     'Exact canonical installed compiler target')
        before=path.lstat();require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and attributes(before)==row['identity'],'Compiler target identity changed')
        if hash_content:
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                require(attributes(os.fstat(fd))==row['identity'],'Compiler target changed at open')
                digest=hashlib.sha256();size=0
                while True:
                    data=os.read(fd,65536)
                    if not data:break
                    size+=len(data);require(size<=row['identity']['st_size'],'Compiler target grew');digest.update(data)
                require(size==row['identity']['st_size']and digest.hexdigest()==row['sha256'],'Compiler target content changed')
                require(attributes(os.fstat(fd))==row['identity']and attributes(path.lstat())==row['identity'],'Compiler target changed during read')
            finally:os.close(fd)
    return True

def scope(root,compiler_root,policy):
    root=Path(root);compiler_root=Path(compiler_root);policy_shape(policy)
    require(compiler_root.is_absolute()and compiler_root.resolve()==compiler_root and compiler_root.name=='compile-work',
            'Canonical fixed compile-work scope')
    require(root in (compiler_root,compiler_root.parent),'Compiler exception only in whole work or compile-work inventory')
    require(root.resolve()==root and not root.is_symlink(),'Canonical inventory root')
    return compiler_root

def link_record(path,compiler_root,policy):
    path=Path(path);compiler_root=Path(compiler_root);policy_shape(policy)
    require(path.is_relative_to(compiler_root),'Compiler asset link outside compile-work')
    relative=path.relative_to(compiler_root).as_posix()
    require(RELATIVE.fullmatch(relative),'Only reviewed model/bucket/subgraph compiler assets')
    require(path.parent.resolve()==path.parent,'No compiler directory symlink')
    before=path.lstat();require(stat.S_ISLNK(before.st_mode)and before.st_uid==os.getuid()and before.st_nlink==1,'Owned single compiler symlink')
    target=os.readlink(path)
    require(target in policy['assets'],'Only exact RECORD-verified installed targets')
    row=policy['assets'][target]
    require(path.name==row['name'],'Original compiler resource basename')
    require(target==row['path'],'Exact absolute installed compiler target')
    info=Path(target).lstat();require(attributes(info)==row['identity']and Path(target).resolve()==Path(target),'Original compiler target identity')
    require(attributes(path.lstat())==attributes(before)and os.readlink(path)==target,'Compiler link changed during observation')
    return dict(path=str(path),compiler_asset=True,symlink=target,stat=attributes(before),target=dict(row))
