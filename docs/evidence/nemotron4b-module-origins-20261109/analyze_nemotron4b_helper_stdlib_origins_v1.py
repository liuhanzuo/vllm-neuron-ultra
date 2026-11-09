"""Snapshot helper standard-library source origins without importing those files.

Preserves source aliases and installed hardlink counts. Native extension files
are stat-only. This inventory does not authorize native loading or prove the
actual runtime dependency closure, and the current router still needs a
reviewed successor to read explicitly recorded hardlinked source files.
"""
import datetime
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import stat
import sys
import _imp

import nemotron4b_helper_dependency_catalog_v1 as c

Q = Path(__file__).resolve().parent
ROOT = Path('/usr/lib/python3.10')
OPTIONAL = Q/'nemotron4b-helper-optional-dependency-inventory-20261109-v1.json'
OPTIONAL_SHA = 'f3399ac3b8898e960bdcef3811809a4c2f0be94c3346ae10d97edf9a30fcfc1b'
STARTUP = Q/'nemotron4b-helper-startup-import-dispatch-inputs-20261109-v1.json'
STARTUP_SHA = '6952a2436b5d749569334a34643480cdf9354baf40d3c9f18adf60241353693b'
OUT = Q/'nemotron4b-helper-stdlib-origin-inventory-20261109-v1.json'


def identity(path):
    p = Path(path)
    s = p.lstat()
    return {k:getattr(s,k) for k in c.FIELDS}


def source_snapshot(path):
    """Bounded original source bytes; never open shared-library/model payloads."""
    path = Path(path)
    c.require(path.suffix == '.py' and path.is_absolute()
              and path.parent.resolve() == path.parent
              and not any(p.is_symlink() for p in path.parents), 'Canonical Python source parent')
    alias = None
    target = path
    if path.is_symlink():
        alias = dict(link_text=os.readlink(path),fullstat=identity(path),target=str(path.resolve()))
        target = Path(alias['target'])
        c.require(os.path.abspath(path.parent/alias['link_text']) == str(target),
                  'Only exact observed one-hop source aliases')
    c.require(target.suffix == '.py' and target.resolve() == target
              and not any(p.is_symlink() for p in (target,*target.parents)), 'Canonical original source target')
    before = identity(target)
    c.require(stat.S_ISREG(before['st_mode']) and before['st_nlink'] >= 1
              and 0 <= before['st_size'] <= 8*2**20, 'Bounded original regular Python file')
    fd = os.open(target,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        c.require({k:getattr(os.fstat(fd),k)for k in c.FIELDS} == before, 'Exact opened source identity')
        chunks=[];size=0
        while True:
            raw=os.read(fd,min(65536,8*2**20+1-size))
            if not raw:break
            chunks.append(raw);size+=len(raw)
            c.require(size <= 8*2**20,'Source read cap')
        raw=b''.join(chunks)
        c.require({k:getattr(os.fstat(fd),k)for k in c.FIELDS} == before == identity(target)
                  and len(raw)==before['st_size'],'Complete stable original source')
    finally:
        os.close(fd)
    if alias is not None:
        c.require(identity(path)==alias['fullstat'] and os.readlink(path)==alias['link_text']
                  and str(path.resolve())==alias['target'],'Original source alias unchanged through read')
    row=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
             fullstat=before,original_nlink=before['st_nlink'],
             current_router_reader_compatible=before['st_nlink']==1)
    if alias is not None:row['alias']=alias
    return row


def module_name(path):
    relative=path.relative_to(ROOT)
    parts=relative.parts[:-1]if path.name=='__init__.py'else(*relative.parts[:-1],path.stem)
    # sysconfig legitimately imports names containing hyphens using importlib.
    if not parts or not all(re.fullmatch('[A-Za-z_][A-Za-z0-9_-]*',part)for part in parts):
        return None
    return '.'.join(parts)


def main():
    c.require(sys.dont_write_bytecode and __debug__ and sys.version_info[:2]==(3,10),
              'Source-only Python3.10 -B without optimization')
    optional_raw, optional_ref=c.source(OPTIONAL,OPTIONAL_SHA);optional=json.loads(optional_raw)
    startup_raw, startup_ref=c.source(STARTUP,STARTUP_SHA);startup=json.loads(startup_raw)
    modules={};auxiliary={}
    paths=sorted(ROOT.rglob('*.py'))
    for path in paths:
        row=source_snapshot(path);name=module_name(path)
        if name is None:
            auxiliary[str(path)]=row
        else:
            c.require(name not in modules,'Unique stdlib module origin')
            modules[name]=dict(row,kind='source',package=path.name=='__init__.py')
    c.require(paths==sorted(ROOT.rglob('*.py')),'Complete stable stdlib source path set')
    revalidated=0
    for path,expected in startup['files'].items():
        # The prior startup inventory uses the canonical sitecustomize target.
        _,actual=c.source(path,expected['sha256'])
        c.require(actual['fullstat']==expected['fullstat'] and actual['bytes']==expected['bytes'],
                  'Pinned startup input unchanged')
        revalidated+=1
    merged=dict(modules);collisions=[]
    for name,row in optional['python_sources'].items():
        _,actual=c.source(row['path'],row['sha256'])
        c.require(actual['fullstat']==row['fullstat'] and actual['bytes']==row['bytes'],
                  'Pinned installed Python dependency unchanged')
        candidate=dict(actual,kind='source',package=Path(row['path']).name=='__init__.py')
        if name in merged and merged[name]['path']!=candidate['path']:
            collisions.append(dict(name=name,stdlib=merged[name],installed=candidate))
        else:
            merged[name]=candidate
    site=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
    for path,row in startup['files'].items():
        p=Path(path)
        if p.suffix!='.py' or not p.is_relative_to(site):continue
        name,package=c.module_name(p.relative_to(site).as_posix())
        candidate=dict(row,kind='source',package=package)
        if name in merged:
            c.require(merged[name]['path']==path and merged[name]['sha256']==row['sha256'],
                      'Repeated startup module has same source')
        else:
            merged[name]=candidate
    native={}
    suffixes=sorted(importlib.machinery.EXTENSION_SUFFIXES,key=len,reverse=True)
    for path in sorted((ROOT/'lib-dynload').glob('*.so')):
        c.require(path.resolve()==path and not any(p.is_symlink()for p in(path,*path.parents)),
                  'Original stdlib native file path')
        fields=identity(path);c.require(stat.S_ISREG(fields['st_mode']),'Native stat-only regular file')
        suffix=next((x for x in suffixes if path.name.endswith(x)),None)
        c.require(suffix is not None,'Known observer interpreter extension suffix')
        name=path.name[:-len(suffix)]
        c.require(name not in native,'Unique stdlib native metadata')
        native[name]=dict(path=str(path),fullstat=fields,payload_digest_recomputed=False)
    frozen_candidates=sorted(set(modules)|{'_frozen_importlib','_frozen_importlib_external','zipimport'})
    frozen=[name for name in frozen_candidates if _imp.is_frozen(name)]
    rows=list(modules.values())+list(auxiliary.values())
    hardlinked={name:row for name,row in modules.items()if row['original_nlink']>1}
    aliases={name:row['alias']for name,row in modules.items()if 'alias'in row}
    output=dict(schema='nemotron4b_helper_stdlib_origin_inventory_v1',reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        prior_optional_inventory=optional_ref,prior_startup_inventory=startup_ref,
        source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        stdlib_root=str(ROOT),stdlib_python_files=len(paths),stdlib_module_count=len(modules),
        stdlib_source_bytes=sum(row['bytes']for row in rows),stdlib_sources=modules,auxiliary_sources=auxiliary,
        source_aliases=aliases,hardlinked_sources=hardlinked,hardlinked_source_count=len(hardlinked),
        installed371source_identities_revalidated=len(optional['python_sources']),
        prior12startup_identities_revalidated=revalidated,
        merged_source_modules=merged,merged_source_module_count=len(merged),source_name_collisions=collisions,
        namespace_packages=optional['namespace_packages'],
        stdlib_native_stat_only=native,stdlib_native_stat_only_count=len(native),
        installed_native_stat_only=optional['native_payload_stat_only'],
        interpreter_builtin_names=list(sys.builtin_module_names),
        observed_frozen_candidate_names=frozen,frozen_name_scan_is_not_complete_interpreter_authentication=True,
        hardlinked_source_runtime_reader_prepared=False,
        original_sources_not_relinked_or_copied=True,
        source_inventory_is_not_runtime_permission=True,
        actual_installed_package_code_executed=False,actual_SDK_helper_executed=False,
        native_payload_bytes_read=0,runtime_admitted=False,complete_runtime_dependency_closure_proven=False,
        full_model_qualified=False,new_speedup_claim=False)
    with OUT.open('x')as f:json.dump(output,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({k:output[k]for k in ['stdlib_python_files','stdlib_module_count','stdlib_source_bytes',
        'hardlinked_source_count','merged_source_module_count','stdlib_native_stat_only_count',
        'installed371source_identities_revalidated','prior12startup_identities_revalidated']},
        ensure_ascii=False))


if __name__=='__main__':
    main()
