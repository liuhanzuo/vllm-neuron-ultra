"""Pinned native-file and current mapping checks for a future gated SDK helper.

No automatic runtime admission or native loading. Actual native payload reads
require an explicit caller resource/serial guard and cannot run in source-only
preparation. The observer tests use synthetic byte files, never SDK libraries.

These are point-in-time disk/hash and mapping-metadata checks, not memory-byte
authentication or an exclusion mechanism for concurrent writers. In-place
writes with unchanged observed stat fields can evade metadata rechecks. A maps
snapshot after loading cannot prevent a dependency initializer from running,
or detect a transient library already unloaded. Real integration still needs
the serial owner, complete linked-library closure and interpreter policy.
"""
import copy
import hashlib
import importlib.machinery as machinery
import os
from pathlib import Path
import re
import stat

from nemotron4b_helper_dependency_catalog_v1 import FIELDS
from nemotron4b_import_router_v4 import CheckedDelegatingLoader
from nemotron4b_process_handshake_v1 import require

MAP_CAP = 8*2**20
FILE_CAP = 2**30


def fullstat(info):
    return {key:getattr(info,key)for key in FIELDS}


def parse_maps(text):
    require(type(text)is str and len(text.encode('utf-8'))<=MAP_CAP, 'Bounded complete process maps text')
    rows=[];previous_end=0
    lines=text.split('\n')
    if lines[-1]=='':lines.pop()
    for line in lines:
        fields=line.split(None,5)
        require(len(fields)>=5,'Complete process map fields')
        address,permissions,offset,device,inode=fields[:5]
        require(re.fullmatch('[0-9a-f]+-[0-9a-f]+',address)and
                re.fullmatch('[r-][w-][x-][ps]',permissions)and
                re.fullmatch('[0-9a-f]+',offset)and re.fullmatch('[0-9a-f]+:[0-9a-f]+',device)
                and inode.isdecimal(),'Original Linux process map syntax')
        begin,end=(int(value,16)for value in address.split('-'))
        require(0<=previous_end<=begin<end,'Ordered nonoverlapping mappings')
        previous_end=end
        path=fields[5]if len(fields)==6 else ''
        # Linux renders both a newline and a literal backslash-012 filename
        # identically. Never guess which disk path a mapping came from.
        require('\\012'not in path and '\x00'not in path,
                'Unambiguous process map pathname')
        row=dict(begin=begin,end=end,permissions=permissions,offset=int(offset,16),
                 device=[int(value,16)for value in device.split(':')],inode=int(inode),path=path)
        require(len(rows)<32768,'Bounded mapping row count')
        rows.append(row)
    require(rows,'Nonempty actual process maps')
    return rows


def canonical_file(path):
    p=Path(path)
    require(p.is_absolute()and p.resolve()==p
            and not any(part.is_symlink()for part in(p,*p.parents)),
            'Original canonical native/interpreter file')
    return p


class NativeFiles:
    def __init__(self,expected,*,native_read_guard):
        require(type(expected)is dict and 0<len(expected)<=512
                and callable(native_read_guard),'Explicit bounded native catalog and serial guard')
        self.expected=copy.deepcopy(expected)
        self.native_read_guard=native_read_guard
        self.verified={}
        for path,row in self.expected.items():
            require(row['path']==path and type(row['fullstat'])is dict
                    and set(row['fullstat'])==set(FIELDS),'Complete native source identity')
            require(type(row['bytes'])is int and 0<row['bytes']==row['fullstat']['st_size']<=FILE_CAP
                    and stat.S_ISREG(row['fullstat']['st_mode'])
                    and row['fullstat']['st_nlink']>=1,'Bounded recorded regular native file')
            require(type(row['sha256'])is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),
                    'Explicit native payload hash before authentication')

    def identity(self,path):
        require(path in self.expected,'Native path outside original catalog')
        p=canonical_file(path);actual=fullstat(p.lstat())
        require(actual==self.expected[path]['fullstat'],'Original native file identity unchanged')
        return actual

    def authenticate(self,path):
        require(path in self.expected,'Exact original native path')
        self.verified.pop(path,None)
        self.native_read_guard()  # caller must own the actual scientific slot
        expected=self.expected[path];before=self.identity(path)
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        digest=hashlib.sha256();count=0
        try:
            require(fullstat(os.fstat(fd))==before,'Exact opened native file')
            while True:
                self.native_read_guard()
                raw=os.read(fd,min(2**20,FILE_CAP+1-count))
                if not raw:break
                count+=len(raw)
                require(count<=expected['bytes'],'Native file grew beyond original byte count')
                digest.update(raw)
            require(fullstat(os.fstat(fd))==before==self.identity(path)
                    and count==expected['bytes']and digest.hexdigest()==expected['sha256'],
                    'Complete stable original native payload hash')
            self.native_read_guard()
        finally:os.close(fd)
        record=dict(path=path,sha256=digest.hexdigest(),bytes=count,fullstat=before,
                    actual_payload_read=True)
        self.verified[path]=record
        return copy.deepcopy(record)

    def authenticate_all(self):
        for path in self.expected:self.authenticate(path)
        return copy.deepcopy(self.verified)

    def require_complete(self):
        require(set(self.verified)==set(self.expected),'Complete original native catalog authenticated before load')
        for path,row in self.expected.items():
            require(self.verified[path]['sha256']==row['sha256']
                    and self.verified[path]['fullstat']==self.identity(path),
                    'Authenticated original native identity retained')


class NativeMappingAudit:
    def __init__(self,files,*,read_maps,approved_kernel_exec):
        require(type(files)is NativeFiles and callable(read_maps),'Explicit native authentication and maps provider')
        self.files,self.read_maps=files,read_maps
        self.kernel=copy.deepcopy(approved_kernel_exec)
        require(type(self.kernel)is list and len(self.kernel)<=4,'Exact bounded original kernel code mappings')
        for row in self.kernel:
            require(row['path']in('[vdso]','[vsyscall]')and row['inode']==0
                    and row['device']==[0,0]and 'x'in row['permissions'],'Original kernel-supplied code only')
        self.snapshots=[]
        self.calls=[]
        self.audit_count=0

    def audit(self):
        require(self.audit_count<4096,'Bounded native mapping audits')
        self.audit_count+=1
        self.files.require_complete()
        rows=parse_maps(self.read_maps())
        by_path={};kernel=[]
        for row in rows:
            path=row['path']
            if path.startswith('/'):
                by_path.setdefault(path,[]).append(row)
            elif 'x'in row['permissions']:
                kernel.append(row)
        require(kernel==self.kernel,'No unreviewed anonymous executable mappings')
        selected={}
        for path,segments in by_path.items():
            # Inspect every file with executable pages plus all visible shared
            # libraries, including their read/write-only partial load stages.
            if not(any('x'in r['permissions']for r in segments)
                   or re.search(r'\.so(?:\.|$)',Path(path.removesuffix(' (deleted)')).name)):
                continue
            require(not path.endswith(' (deleted)'),'No deleted executable or shared-library origin')
            identity=self.files.identity(path)
            require(path in self.files.verified,'Actually authenticated mapped native payload')
            device=[os.major(identity['st_dev']),os.minor(identity['st_dev'])]
            require(all(row['device']==device and row['inode']==identity['st_ino']for row in segments),
                    'Mapped device/inode agrees with original authenticated file')
            selected[path]=segments
        # Keep compact identities, not addresses from thousands of repeated guards.
        result=dict(paths=sorted(selected),map_rows=len(rows),
                    authenticated_native_files=len(self.files.verified),
                    anonymous_executable_mappings_unchanged=True)
        if not self.snapshots or self.snapshots[-1]!=result:self.snapshots.append(result)
        return result

    def authorize_native(self,fullname,spec,row):
        path=row['path']
        require(type(spec)is machinery.ModuleSpec and spec.name==fullname
                and spec.origin==path and path in self.files.expected
                and type(spec.loader)is machinery.ExtensionFileLoader
                and spec.loader.name==fullname and spec.loader.path==path,
                'Exact native import origin in authenticated catalog')
        self.audit()
        original=spec.loader
        fields=lambda:(spec.name,spec.origin,spec.has_location,spec.cached,
                       None if spec.submodule_search_locations is None
                       else tuple(spec.submodule_search_locations))
        original_fields=fields();state=spec.loader_state;checked=[]
        def validate_spec():
            require(fields()==original_fields and spec.loader_state is state
                    and original.name==fullname and original.path==path,
                    'Original native spec and loader fields unchanged')
            wrapper=spec.loader
            require(type(wrapper)is CheckedDelegatingLoader
                    and wrapper.original is original and wrapper.fullname==fullname
                    and wrapper.kind=='native'and wrapper.before is before
                    and wrapper.after is after,'Exact native checking wrapper')
            if not checked:checked.append(wrapper)
            require(wrapper is checked[0],'Selected native checking wrapper unchanged')
        def before(name,loader,effective_spec):
            require(name==fullname and loader is original and effective_spec is spec,
                    'Exact native loader lifecycle identity')
            validate_spec()
            self.audit()
        def after(name,loader,module,phase):
            require(name==fullname and loader is original
                    and phase in('create','exec','create_failed','exec_failed'),
                    'Exact native loader result lifecycle')
            validate_spec()
            if phase in('exec','exec_failed'):
                require(getattr(module,'__spec__',None)is spec,
                        'Exact executed native module spec')
            snapshot=self.audit()  # failed initialization may still map libraries
            require(len(self.calls)<4096,'Bounded native lifecycle record')
            self.calls.append(dict(name=name,phase=phase,mapped_paths=snapshot['paths']))
        return before,after
