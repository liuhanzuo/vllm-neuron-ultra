"""Finite metadata-only inventory of the one evidenced completed Qwen3.5 tree.

No CLI, payload reads, lock acquisition, process inspection, or import-time IO.
Root schedules this under its existing serial guard and persists the return.
The row/inode accounting follows dory_build_artifact_inventory_v1.inventory;
directory descriptors replace path walking to avoid following swapped links.
Shared historical /tmp/nxd_model and /mnt/nvme/neuron-cache are NOT eligible.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import PurePosixPath
import stat
import time

TARGET='/mnt/nvme/neuron-artifacts/latest-20261031-qwen35-2b-text-bf16-cpu-tp2-b4-ctx4096-split-v4'
MODEL_ANCHOR=dict(device=66304,inode=13369694,size=294473003,
                  mtime_ns=1793638086016430575,ctime_ns=1793638086016430575)
LIMITS=dict(entries=65536,depth=32,wall_seconds=120,metadata_bytes=16*2**20)
DIRECTORY_FLAGS=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC


def require(value,message):
    if not value:raise ValueError(message)


def metadata(info):
    """Pure conversion, shared by root's admitted initial lstat and scan rows."""
    return dict(mode=info.st_mode,device=info.st_dev,inode=info.st_ino,nlink=info.st_nlink,
        size=info.st_size,blocks=info.st_blocks,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _same_object(a,b):
    return all(a[k]==b[k] for k in ('device','inode','mode'))


def _open_root(target,expected,guard):
    """Open every absolute component relative to its held parent, no links."""
    fd=os.open('/',DIRECTORY_FLAGS)
    try:
        for name in PurePosixPath(target).parts[1:]:
            guard()
            before=metadata(os.stat(name,dir_fd=fd,follow_symlinks=False))
            require(stat.S_ISDIR(before['mode']),'Ordinary directory ancestor required')
            child=os.open(name,DIRECTORY_FLAGS,dir_fd=fd)
            try:
                after=metadata(os.fstat(child))
                require(_same_object(before,after)
                    and _same_object(before,metadata(os.stat(name,dir_fd=fd,follow_symlinks=False))),
                    'Ancestor directory changed during descriptor open')
            except BaseException:
                os.close(child);raise
            os.close(fd);fd=child
        require(metadata(os.fstat(fd))==expected,'Completed tree root identity changed')
        return fd
    except BaseException:
        os.close(fd);raise


def _scan_pass(root_fd,expected,guard):
    rows={};directories=set();serialized_bytes=0
    def append(relative,info):
        nonlocal serialized_bytes
        row=metadata(info)
        require(row['device']==expected['device'],'Cross-filesystem child forbidden')
        require(all(type(row[k]) is int and row[k]>=0 for k in ('size','blocks','nlink')),
                'Nonnegative exact stat allocation fields')
        mode=row['mode']
        kind='directory' if stat.S_ISDIR(mode) else 'file' if stat.S_ISREG(mode) else 'symlink' if stat.S_ISLNK(mode) else None
        require(kind is not None,'Special file forbidden in completed build inventory')
        require(relative not in rows and len(rows)<LIMITS['entries'],'Unique bounded inventory entries')
        row.update(path=relative,kind=kind)
        serialized_bytes+=len(_json(row))+1
        require(serialized_bytes<=LIMITS['metadata_bytes'],'Complete metadata inventory exceeds cap')
        rows[relative]=row
        return row
    def walk(fd,relative,depth):
        guard();require(depth<=LIMITS['depth'],'Bounded directory depth')
        before=metadata(os.fstat(fd));key=(before['device'],before['inode'])
        require(key not in directories,'Repeated directory inode forbidden')
        directories.add(key)
        if relative=='.':append(relative,os.fstat(fd))
        names=[]
        with os.scandir(fd) as iterator:
            for entry in iterator:
                guard()
                name=entry.name
                require(type(name) is str and name not in ('','.','..') and '/' not in name and '\0' not in name,
                        'One ordinary relative entry name')
                require(len(names)<LIMITS['entries'],'Bounded directory entry list')
                names.append(name)
        require(len(names)==len(set(names)),'Unique directory listing')
        for name in sorted(names):
            guard();relative_child=name if relative=='.' else relative+'/'+name
            row=append(relative_child,os.stat(name,dir_fd=fd,follow_symlinks=False))
            if row['kind']=='directory':
                child=os.open(name,DIRECTORY_FLAGS,dir_fd=fd)
                try:
                    require(metadata(os.fstat(child))=={k:v for k,v in row.items() if k not in ('path','kind')},
                            'Child directory changed during descriptor open')
                    walk(child,relative_child,depth+1)
                finally:os.close(child)
            # Regular files and symlinks are never opened, read, or followed.
            require(metadata(os.stat(name,dir_fd=fd,follow_symlinks=False))
                    =={k:v for k,v in row.items() if k not in ('path','kind')},'Entry changed during scan')
        require(metadata(os.fstat(fd))==before,'Directory mutated during completed-tree scan')
    walk(root_fd,'.',0)
    model=rows.get('model.pt')
    require(model is not None and model['kind']=='file'
            and all(model[k]==value for k,value in MODEL_ANCHOR.items()),
            'Original completed full12 model.pt metadata anchor changed')
    return rows


def summarize(rows):
    """Pure inode accounting: count links, charge each physical inode once."""
    inodes={};counts=Counter();logical=0
    for path in sorted(rows):
        row=rows[path];counts[row['kind']]+=1
        if row['kind']=='file':logical+=row['size']
        key=(row['device'],row['inode'])
        if key in inodes:
            previous=inodes[key]['row']
            require(row['kind']!='directory' and all(row[k]==previous[k] for k in row if k!='path'),
                    'Hardlink metadata must agree for the same inode')
            inodes[key]['entries']+=1
        else:inodes[key]=dict(row=row,entries=1)
    require(all(v['entries']<=v['row']['nlink'] for v in inodes.values() if v['row']['kind']!='directory'),
            'Observed links exceed inode link count')
    unique_allocated=sum(v['row']['blocks']*512 for v in inodes.values())
    files=[row for row in rows.values() if row['kind']=='file']
    def largest(key):
        return sorted(files,key=lambda row:(-row[key],row['path']))[:8]
    metadata_digest=hashlib.sha256(b''.join(_json(rows[p])+b'\n' for p in sorted(rows))).hexdigest()
    return dict(entries=len(rows),entry_kinds=dict(sorted(counts.items())),unique_inodes=len(inodes),
        allocated_bytes=unique_allocated,path_counted_allocated_bytes=sum(r['blocks']*512 for r in rows.values()),
        regular_file_path_logical_bytes=logical,
        max_file_logical_bytes=max((r['size'] for r in files),default=0),
        max_file_allocated_bytes=max((r['blocks']*512 for r in files),default=0),
        largest_logical_files=largest('size'),largest_allocated_files=largest('blocks'),
        external_hardlinked_inodes=sum(v['row']['nlink']>v['entries'] for v in inodes.values() if v['row']['kind']!='directory'),
        symlinks_followed=False,symlink_targets_read=False,metadata_inventory_sha256=metadata_digest,
        reclaimable_bytes_claimed=False)


def inventory(target,expected_root_identity,execution_guard):
    """Called only by an independently admitted root-owned metadata child.

    Root supplies a fresh lstat identity and proof that this exact completed
    tree is outside all active scopes. Metadata stability is not content
    integrity or reclaimability. Shared workdir/cache are refused before IO.
    """
    require(target==TARGET,'Only evidenced completed split-v4 artifact is eligible; shared workdir/cache refused')
    require(type(expected_root_identity) is dict and set(expected_root_identity)==
        {'mode','device','inode','nlink','size','blocks','mtime_ns','ctime_ns'},'Exact root lstat identity required')
    require(all(type(v) is int and v>=0 for v in expected_root_identity.values())
            and stat.S_ISDIR(expected_root_identity['mode']),'Ordinary finite root identity')
    require(callable(execution_guard),'Existing serial execution guard required')
    started=time.monotonic()
    def guard():
        require(time.monotonic()-started<LIMITS['wall_seconds'],'Finite metadata scan wall deadline')
        proof=execution_guard()
        require(type(proof) is dict and proof.get('admitted') is True and proof.get('exclusive') is True
                and proof.get('completed_tree')==TARGET and proof.get('target_not_active') is True,
                'Actual root admission, exclusivity and inactive completed-tree proof required')
    guard()
    passes=[]
    for _ in range(2):
        fd=_open_root(target,expected_root_identity,guard)
        try:passes.append(_scan_pass(fd,expected_root_identity,guard))
        finally:os.close(fd)
    guard();require(passes[0]==passes[1],'Two complete metadata passes must agree exactly')
    result=summarize(passes[0])
    return dict(schema='qwen35_completed_allocation_inventory_v1',target=target,root_identity=expected_root_identity,
        model_anchor=MODEL_ANCHOR,summary=result,complete=True,passes=2,regular_payloads_opened=False,
        historical_compiler_workdir_attributed=False,historical_cache_attributed=False,
        elapsed_seconds=time.monotonic()-started,limits=LIMITS,
        native_executed=False,actual_process_exit_claimed=False,full_model_qualified=False)
