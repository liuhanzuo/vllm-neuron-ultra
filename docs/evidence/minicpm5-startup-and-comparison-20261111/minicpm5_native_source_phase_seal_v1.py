"""Source-only prototype: seal static data during native text authentication.

No runner uses this module. It cannot launch, admit, lock, or load a model.
The phase keeps every source hash and every live lease/resource/output check.
Old mutable aliases are checked at exit, before model work, not each guard.
"""
import hashlib
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType

from research import minicpm5_layer1_operator_native_lifecycle_core_v2 as life
from research import minicpm5_layer1_operator_native_output_catalog_v2 as catalog
from minicpm5_layer0_runtime_auth_v1 import digest

require = life.require
staging_destination = catalog.staging_destination
meta_cap = catalog.meta_cap


def seal_json(value, active=None):
    """Copy exact JSON containers into private mapping proxies and tuples."""
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        require(math.isfinite(value), 'Finite sealed JSON number')
        return value
    require(type(value) in (dict, list), 'Only exact JSON types; no custom containers')
    active = set() if active is None else active
    key = id(value)
    require(key not in active, 'No cyclic static JSON')
    active.add(key)
    try:
        if type(value) is list:
            return tuple(seal_json(item, active) for item in value)
        require(all(type(key) is str for key in value), 'Exact string JSON keys')
        return MappingProxyType({key: seal_json(item, active) for key, item in value.items()})
    finally:
        active.remove(key)


def plain_json(value):
    """A detached plain copy only at phase boundaries; never inside the loop."""
    if type(value) is MappingProxyType:
        return {key: plain_json(item) for key, item in value.items()}
    if type(value) is tuple:
        return [plain_json(item) for item in value]
    require(type(value) in (str, int, bool, float, type(None)), 'Only sealed JSON values')
    return value


def phase_partition(context, sources):
    """Original classifier on a detached plain plan, before snapshot activation.

    Its callbacks remain the original guarded, authenticated metadata transport.
    The caller passes the returned plain partition to SourcePhase immediately.
    """
    require(context._sources_checked is False, 'Source classification precedes model permission')
    frozen = seal_json(context.plan)
    plain = plain_json(frozen)
    def read(ref):
        context.guard()
        require(frozen['source_sha256'].get(ref['path']) == ref['sha256'],
                'Every classifier metadata reference source-pinned')
        return sources.load(ref)
    partition = sources.classify(plain['source_sha256'], plain['source_contract'], read)
    require(digest(context.plan) == digest(plain_json(frozen)), 'Plan unchanged across complete classification')
    return partition


class SourcePhase:
    """One synchronous source-only interval; no general cache or model permit.

    Both outer resolved.plan and the context's independently deep-copied plan
    are authenticated and retained for exit verification. During the interval,
    context.plan is a private recursive read-only snapshot. The original plan,
    resolved object, contract, permit and classifier partition must match their
    entry digests before the original plain context.plan is restored.
    """
    def __init__(self, contract, resolved, context, partition, lease, ports, sources):
        self.contract_alias, self.resolved_alias = contract, resolved
        self.context, self.partition_alias = context, partition
        self.lease, self.ports = lease, ports
        self.sources = sources
        self.started = self.closed = self.authenticated = False
        self.failure = None

    def __enter__(self):
        try:
            return self._enter()
        except BaseException as error:
            self._latch(error)
            raise

    def _latch(self, error):
        if self.failure is None:
            self.failure = error
        self.context._sources_checked = False
        self.context.external_guard = self._reject
        self.context.guard = self._reject

    def _reject(self, *args, **kwargs):
        raise self.failure

    def _enter(self):
        require(not self.started, 'One source phase entry')
        self.started = True
        context = self.context
        sources = self.sources
        require(context._sources_checked is False, 'No model permission inside source phase')
        context.guard('native')
        self.original_plan = context.plan
        self.original_external_guard = context.external_guard
        self.original_context_guard = context.guard
        self.original_permit = context.permit
        self.lease_callback = self.lease.check
        self.storage_callback = self.ports.storage_available
        require(self.ports.resolved is self.resolved_alias and self.ports.contract is self.contract_alias,
                'Exact original ports bindings')
        values = dict(contract=self.contract_alias, resolved=self.resolved_alias,
                      context_plan=self.original_plan, partition=self.partition_alias, permit=self.original_permit)
        # Reject unsupported values before invoking ordinary serialization.
        self.snapshots = MappingProxyType({key: seal_json(value) for key, value in values.items()})
        self.initial_digests = MappingProxyType({key: digest(value) for key, value in values.items()})
        life.binding(self.contract_alias, self.resolved_alias)
        require(digest(self.original_plan) == self.resolved_alias['plan_sha256']
                and self.original_plan == self.resolved_alias['plan'], 'Context copy and resolved plan agree completely')
        combined = sources.check_partition(self.partition_alias)
        require(combined == self.original_plan['source_sha256'], 'Lossless complete phase source map')
        self.plan = self.snapshots['context_plan']
        self.static_contract = self.snapshots['contract']
        self.static_catalog = self.snapshots['resolved']['native_runtime_output_catalog']
        self.partition = self.snapshots['partition']
        self.writers = seal_json(self._writers())
        self.phase_guard = self.guard
        def context_guard(role=None):
            if self.failure is not None:
                raise self.failure
            try:
                return self.original_context_guard(role)
            except BaseException as error:
                self._latch(error)
                raise
        self.phase_context_guard = context_guard
        context.plan = self.plan
        context.external_guard = self.phase_guard
        context.guard = self.phase_context_guard
        return self

    def _writers(self):
        value = dict(outer=self.ports.admission['outer_owner']['pid'], coordinator=self.ports.coordinator['pid'])
        if self.ports.read_owner is not None:
            value['native'] = self.ports.read_owner['pid']
        return value

    def guard(self):
        if self.failure is not None:
            raise self.failure
        try:
            require(self.started and not self.closed and self.context.plan is self.plan
                    and self.context.external_guard is self.phase_guard
                    and self.context.guard is self.phase_context_guard
                    and self.context.permit is self.original_permit
                    and self.context._sources_checked is False, 'Active sealed context; no rebound or model permission')
            require(self.lease.check == self.lease_callback and self.ports.storage_available == self.storage_callback
                    and self.ports.resolved is self.resolved_alias and self.ports.contract is self.contract_alias,
                    'Original live callbacks and ports remain bound')
            require(self._writers() == self.writers, 'Original acknowledged writer identities unchanged')
            self.lease.check('native.runtime')
            life.storage(self.static_contract, self.ports)
            entries, root, writers = walk_sealed(self.static_contract['output_root'], self.static_catalog, self._writers())
            usage_sealed(self.static_catalog, entries, root, writers=writers)
            return self.original_permit
        except BaseException as error:
            self._latch(error)
            raise

    def verify_text_sources(self):
        try:
            return self._verify_text_sources()
        except BaseException as error:
            self._latch(error)
            raise

    def _verify_text_sources(self):
        sources = self.sources
        require(not self.authenticated, 'One full text verification within this phase')
        self.context.guard()
        # Identical original per-item hashing/transport and guard cadence. The
        # complete original check_partition ran once before snapshot creation.
        for path, wanted in self.partition['categories']['text_sha256'].items():
            self.context.guard()
            actual = (sources.aliases.source_sha(path, self.partition['metadata_aliases'], sources.text)
                      if path in self.partition['metadata_aliases'] else hashlib.sha256(sources.text(path, wanted)).hexdigest())
            require(actual == wanted, 'Unchanged exact text source')
        for path, wanted in self.partition['categories']['python_bytes_sha256'].items():
            self.context.guard()
            sources.metadata.python_source_bytes(path, wanted)
        self.context.guard()
        self.authenticated = True
        return dict(complete_text_verification=True, source_map_sha256=self.partition['source_map_sha256'],
                    partition_sha256=self.partition['partition_sha256'], historical_payload_bytes_read=0, native_bytes_read=0)

    def __exit__(self, kind, error, traceback):
        try:
            if kind is not None:
                self._latch(error)
                return False
            if self.failure is not None:
                raise self.failure
            require(self.authenticated, 'No successful phase exit before all original source hashes')
            self.context.guard()
            values = dict(contract=self.contract_alias, resolved=self.resolved_alias,
                          context_plan=self.original_plan, partition=self.partition_alias, permit=self.original_permit)
            for name, value in values.items():
                seal_json(value)  # Exact supported JSON types at exit as well.
                require(digest(value) == self.initial_digests[name], 'Mutable alias changed before source phase exit: ' + name)
            life.binding(self.contract_alias, self.resolved_alias)
            self.sources.check_partition(self.partition_alias)
        except BaseException as failure:
            self._latch(failure)
            raise
        finally:
            self.closed = True
            self.context._sources_checked = False
            if self.failure is None:
                self.context.plan = self.original_plan
                self.context.external_guard = self.original_external_guard
                self.context.guard = self.original_context_guard
        # Native authentication receipt validation remains the original caller's
        # next step. This prototype never enables model/weight operations.
        return False


def usage_sealed(c, entries, root_entry, *, writers):
    proof = inventory_sealed(c, entries, successful_driver=False, writers=writers)
    require(root_entry['kind'] == 'directory' and root_entry['uid'] == 1000 and root_entry['symlink'] is False
            and type(root_entry['size']) is int and root_entry['size'] >= 0
            and type(root_entry['blocks']) is int and root_entry['blocks'] >= 0, 'Exact owned output-root directory extent')
    allocated = max(root_entry['size'], root_entry['blocks'] * 512)
    require(proof['total_allocated_bytes'] + allocated <= 256 * 2**20, 'Original total cap includes output-root allocation')
    return dict(proof, output_root_allocated_bytes=allocated, total_with_root_allocated_bytes=proof['total_allocated_bytes'] + allocated)


def walk_sealed(root,c,writers):
    """Descriptor-relative live finite walk; only exact atomic rename accepted."""
    root=Path(root);allowed=set(c['payload_files'])|set(c['driver_metadata_files'])|set(c['runtime_files'])
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


def inventory_sealed(c,entries,*,successful_driver,writers=None):
    """Validate caller's complete guarded lstat walk; this function does no IO.

    entries maps relative canonical names to {kind,uid,nlink,symlink,size,blocks}.
    Require no symlinks, extra files/directories or unrecognized pending files.
    Driver success cannot contain short/absent raws or trailing staging evidence.
    Runtime files may be partial only when successful_driver is false.
    """
    require(type(entries)is dict and type(successful_driver)is bool,'Complete explicit inventory and stage mode')
    raw=set(c['payload_files']);meta=set(c['driver_metadata_files']);dirs=set(c['driver_directories'])
    total=driver=metadata=raw_bytes=files=directory_allocation=0;seen=set();pending=[]
    for name,row in entries.items():
        require(type(name)is str and str(Path(name))==name and not Path(name).is_absolute() and '..'not in Path(name).parts,
            'Canonical exact relative output inventory path')
        require(row['symlink']is False and row['uid']==1000,'Every owned output tree entry, never symlinks')
        require(type(row['size'])is int and row['size']>=0 and type(row['blocks'])is int and row['blocks']>=0,
            'Every file and directory has a complete nonnegative size/block extent')
        size=row['size'];allocated=max(size,row['blocks']*512);total+=allocated
        if row['kind']=='directory':
            require(name in dirs,'No unbound output directory');driver+=allocated;directory_allocation+=allocated;continue
        require(row['kind']=='file' and row['nlink']==1,'Owned regular singleton bounded file')
        files+=1
        if name in raw:
            require(size==c['payload_files'][name]if successful_driver else size<=c['payload_files'][name],
                'Every original raw has its full exact extent, or explicit failed partial extent')
            raw_bytes+=size;driver+=allocated
        elif name in meta:
            require(0<size<=meta_cap(c,name)if successful_driver else size<=meta_cap(c,name),'Exact bounded driver JSON/journal')
            metadata+=size;driver+=allocated
        elif name in c['runtime_files']:
            require(size<=(c['json_runtime_file_cap_bytes']if name.endswith('.json')else c['runtime_file_cap_bytes']),
                'Original runtime evidence/log bound')
        else:
            require(not successful_driver,'Successful tree cannot hide a pending metadata write')
            destination=staging_destination(name,writers or {});pending.append(destination)
            require(destination not in entries and size<=c['json_runtime_file_cap_bytes'],
                'At most one pending or final destination, original cap')
        seen.add(name)
    require(len(pending)==len(set(pending)) and files<=c['maximum_file_count'],'Finite original writer staging and file slots')
    require(metadata<=c['driver_metadata_total_bytes'] and driver<=c['driver_allocation_cap_bytes']
        and total<=c['total_allocation_cap_bytes'],'Original metadata/native/total capacity, actual allocated blocks counted')
    if successful_driver:require(raw|meta<=seen and dirs<={name for name,r in entries.items()if r['kind']=='directory'},
        'All936 raw,144 metadata and97 directories independently inventoried')
    return dict(files=files,raw_bytes=raw_bytes,driver_metadata_bytes=metadata,driver_allocated_bytes=driver,
        directory_allocated_bytes=directory_allocation,total_allocated_bytes=total,
        pending_destinations=pending,complete_driver_inventory=successful_driver,
        physical_checks_performed=False,actual_process_exit_verified=False)
