"""Finite metadata-only API for an independently admitted serial worker.

Caller retains the original descriptor/history/resource envelope and durable
event sink. No lock/worker/admission is created here. No library bytes are read.
"""
import os
import json
from pathlib import PurePosixPath
import stat
import dory_native_inventory_finish_v4 as existing

EXTERNAL=dict(path='/mnt/nvme/dory-serial-inventory-r5-20261111/inventory/external-closure.json',bytes=137884,
    sha256='8e97c912e0cbcd75ee4eead45565a05d727c5bc043ef5bf0ea2eeb51002ffd6e')
DIRECTORY='/usr/lib/x86_64-linux-gnu/'
# These are saved observed names, not inferred loader selections.
NAMES={
 'ld-linux-x86-64.so.2':('/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2','/lib64/ld-linux-x86-64.so.2'),
 'libc.so.6':('/lib/x86_64-linux-gnu/libc.so.6',),
 'libdl.so.2':('/lib/x86_64-linux-gnu/libdl.so.2',),
 'libgcc_s.so.1':('/lib/x86_64-linux-gnu/libgcc_s.so.1',),
 'libgomp.so.1.0.0':('/lib/x86_64-linux-gnu/libgomp.so.1',),
 'libm.so.6':('/lib/x86_64-linux-gnu/libm.so.6',),
 'libnuma.so.1.0.0':('/lib/x86_64-linux-gnu/libnuma.so.1',),
 'libpthread.so.0':('/lib/x86_64-linux-gnu/libpthread.so.0',),
 'librt.so.1':('/lib/x86_64-linux-gnu/librt.so.1',),
 'libstdc++.so.6.0.30':('/lib/x86_64-linux-gnu/libstdc++.so.6',),
 'libtcmalloc_minimal.so.4.5.9':('/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4',),
 'libutil.so.1':('/lib/x86_64-linux-gnu/libutil.so.1',)}
TARGETS={DIRECTORY+name:tuple(sorted((DIRECTORY+name,)+aliases)) for name,aliases in NAMES.items()}
PATHS=tuple(sorted({p for values in TARGETS.values() for p in values}))
PACKAGES=('libc6','libgcc-s1','libgomp1','libnuma1','libstdc++6','libtcmalloc-minimal4')
KNOWN_VERSIONS={'libc6':'2.35-0ubuntu3.15','libgcc-s1':'12.3.0-1ubuntu1~22.04.3',
    'libstdc++6':'12.3.0-1ubuntu1~22.04.3','libtcmalloc-minimal4':'2.9.1-0ubuntu3'}
VERSION_FORMAT='${Package}\t${Version}\t${Architecture}\t${Status}\n'
VERSION_COMMAND=('/usr/bin/dpkg-query','-W','-f='+VERSION_FORMAT,*PACKAGES)
COMMANDS=tuple(('/usr/bin/dpkg-query','-S',p) for p in PATHS)+(VERSION_COMMAND,)+tuple(('/usr/bin/dpkg-query','-L',p+':amd64') for p in PACKAGES)
# Only named targets, their ancestors, and canonicalized versions of observed
# aliases can be inspected. Unknown link targets are recorded, never followed.
KNOWN_NODES=frozenset({str(parent) for p in PATHS for parent in (PurePosixPath(p),*PurePosixPath(p).parents)}
    | {DIRECTORY+PurePosixPath(p).name for p in PATHS}
    | {'/usr/lib64','/usr/lib64/ld-linux-x86-64.so.2'})


def require(ok,message):
    if not ok:raise ValueError(message)


def identity(info):
    return {name:getattr(info,name) for name in ('st_dev','st_ino','st_mode','st_nlink','st_uid','st_gid','st_size','st_mtime_ns','st_ctime_ns')}


def normalize(path):
    parts=[]
    for item in path.split('/'):
        if item in ('','.'):continue
        if item=='..':
            require(parts,'Link cannot escape filesystem root');parts.pop()
        else:parts.append(item)
    return '/'+('/'.join(parts))


def observe_path(name,budget,*,lstat,readlink):
    """At most 32 lstat/readlink steps; no follow-stat, open or payload read."""
    require(name in PATHS,'One fixed observed path');pending=list(PurePosixPath(name).parts[1:]);current='/';steps=[];visited=set()
    while pending:
        require(len(steps)<32,'Bounded fixed link metadata traversal');part=pending.pop(0);candidate=normalize(current+'/'+part)
        require(candidate in KNOWN_NODES,'Only explicitly bounded path nodes');budget.check()
        try:before=lstat(candidate)
        except FileNotFoundError:return dict(requested=name,status='missing',missing=candidate,steps=steps)
        row=dict(path=candidate,identity=identity(before));steps.append(row)
        if stat.S_ISLNK(before.st_mode):
            target=readlink(candidate);require(type(target)is str and len(target.encode())<=4096 and '\0' not in target,'Bounded link text')
            budget.check();require(identity(before)==identity(lstat(candidate)),'Stable link metadata')
            resolved=normalize(target if target.startswith('/') else str(PurePosixPath(candidate).parent)+'/'+target);row.update(link_target=target,resolved_target=resolved)
            if resolved not in KNOWN_NODES:return dict(requested=name,status='unreviewed_link_target',target=resolved,steps=steps)
            require(candidate not in visited,'No link cycles');visited.add(candidate)
            pending=list(PurePosixPath(resolved).parts[1:])+pending;current='/';continue
        require(stat.S_ISDIR(before.st_mode) if pending else stat.S_ISREG(before.st_mode),'Directory ancestors and ordinary final library metadata')
        current=candidate
    require(current in KNOWN_NODES,'Known finite destination')
    return dict(requested=name,status='observed',resolved=current,identity=steps[-1]['identity'],steps=steps)


def parse_versions(value):
    require(value['returncode']==0,'Complete fixed package-version query');found={}
    for line in value['stdout'].splitlines():
        package,version,arch,status=line.split('\t')
        require(package in PACKAGES and package not in found and arch=='amd64' and status=='install ok installed'
            and 0<len(version)<=128,'Exact installed fixed package/architecture/version')
        found[package]=dict(version=version,architecture=arch,status=status)
    require(set(found)==set(PACKAGES),'All six fixed packages returned')
    for package,version in KNOWN_VERSIONS.items():require(found[package]['version']==version,'Previously observed package version unchanged')
    return found


def ownership(value,path):
    require(value['returncode'] in (0,1),'Expected package ownership status');owners=[];unresolved=[]
    if value['returncode']==1:return dict(owners=[],unresolved=['not_found_by_fixed_name'])
    for line in value['stdout'].splitlines():
        head,sep,tail=line.rpartition(': ')
        if not sep or tail!=path:
            unresolved.append('unparsed_or_nonexact_owner_row');continue
        for name in head.split(', '):
            package,sep,arch=name.partition(':')
            if package not in PACKAGES or sep and arch!='amd64':unresolved.append('owner_outside_reviewed_six_packages')
            else:owners.append(package)
    require(len(owners)<=6 and len(unresolved)<=32,'Bounded ownership rows')
    return dict(owners=sorted(set(owners)),unresolved=unresolved)


def collect(guard,emit,*,command,lstat,readlink):
    """Supplied callbacks make tests memory-only; run() fixes real providers."""
    require(all(callable(x) for x in (guard,emit,command,lstat,readlink)),'Explicit original guard and finite providers')
    budget=existing.Budget(guard);require(len(PATHS)==25 and len(COMMANDS)==32 and len(PACKAGES)==6,'Frozen finite operation counts')
    before={path:observe_path(path,budget,lstat=lstat,readlink=readlink) for path in PATHS}
    owners={};lists={};versions=None
    for index,argv in enumerate(COMMANDS):
        budget.check();value=command(list(argv),budget)
        require(value['argv']==list(argv),'Exact fixed executed command')
        emit(dict(event='system_package_metadata_command',index=index,**value))
        if argv[1]=='-S':owners[argv[2]]=ownership(value,argv[2])
        elif argv[1]=='-W':versions=parse_versions(value)
        else:
            require(value['returncode']==0,'Complete fixed package file list')
            # Retain the complete text in the durable journal above; only fixed
            # names are cross-referenced. Returned names never cause further IO.
            lines=value['stdout'].splitlines();require(len(lines)<=200000,'Bounded package file list')
            package=argv[2].removesuffix(':amd64');lists[package]=dict(entries=len(lines),fixed_names=sorted(set(lines)&set(PATHS)))
    after={path:observe_path(path,budget,lstat=lstat,readlink=readlink) for path in PATHS}
    require(before==after,'All fixed library/link metadata stable across package queries')
    rows=[];unresolved=[]
    for target,names in TARGETS.items():
        matched=set();reasons=[]
        for name in names:
            row=before[name]
            if row['status']!='observed' or row.get('resolved')!=target:
                reasons.append(dict(path=name,reason='fixed_alias_not_observed_at_expected_target',observation=row['status']));continue
            reasons.extend(dict(path=name,reason=reason) for reason in owners[name]['unresolved'])
            for package in owners[name]['owners']:
                if name in lists[package]['fixed_names']:matched.add(package)
                else:reasons.append(dict(path=name,reason='owner_name_absent_from_same_package_file_list',package=package))
        if len(matched)!=1:reasons.append(dict(path=target,reason='not_one_cross_checked_package',packages=sorted(matched)))
        rows.append(dict(target=target,observed_names=list(names),cross_checked_packages=sorted(matched),issues=reasons))
        unresolved.extend(reasons)
    budget.check()
    result=dict(schema='dory_system_package_metadata_v1',saved_external_evidence=EXTERNAL,versions=versions,
        paths=before,package_file_lists=lists,targets=rows,unresolved=unresolved,budget_used=dict(budget.used),
        fixed_paths=25,system_targets=12,fixed_commands=32,fixed_packages=6,library_payload_bytes_read=0,
        package_archive_authenticated=False,loader_selection_established=False,native_ABI_qualified=False,
        full_model_qualified=False,new_speedup_claim=False,actual_process_exit_claimed=False)
    require(len(json.dumps(result,separators=(',',':'),allow_nan=False).encode())<=existing.LIMITS['metadata_file_bytes'],'Small result within original metadata cap')
    emit(dict(event='system_package_metadata_complete',system_targets=12,fixed_commands=32,unresolved=len(unresolved)))
    return result


def run(guard,emit):
    return collect(guard,emit,command=existing.command,lstat=os.lstat,readlink=os.readlink)


if __name__=='__main__':raise RuntimeError('Root-admitted metadata API only; no automatic execution or admission')
