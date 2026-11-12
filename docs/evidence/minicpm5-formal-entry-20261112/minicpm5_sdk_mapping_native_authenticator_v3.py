"""Concrete future bootstrap/native/source authenticator; inert on import.

All current file, map, loader and interpreter operations are explicit methods.
Source tests use synthetic data/ports only. No current authority/spec binding,
archive download, command, service, native import or model launch is performed
at import. The original nine-library authentication remains in frozen full auth.
"""
import copy
import gzip
import hashlib
import importlib
import importlib.machinery as machinery
import io
import json
import os
from pathlib import Path
import stat
import sys
from research import minicpm5_sdk_mapping_history_gate_v1 as gate
from research import minicpm5_sdk_mapping_native_closure_v2 as closure

require=gate.require
ROOT=gate.ROOT
EXE=gate.backend.OUTER_EXE
LIBC=gate.backend.LIBC
LOADER='/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
ORIGIN_REF=dict(path=str(ROOT/'research/minicpm5-sdk-mapping-observer-origin-saved-evidence-20261112-v1.json'),
    bytes=17855,sha256='2a47f38ea6ac4417c7feca3dc94e041a57389516b0f0066bafbef8b9dd5d5591')
FIELDS=('st_dev','st_ino','st_mode','st_nlink','st_size','st_mtime_ns','st_ctime_ns')


def identity(info):return {name:getattr(info,name) for name in FIELDS}
def unique(pairs):
    result={}
    for key,value in pairs:require(key not in result,'Unique JSON metadata keys');result[key]=value
    return result
def decode(raw):return json.loads(raw,object_pairs_hook=unique,
    parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Finite JSON metadata')))


class FilePorts:
    def canonical(self,path):
        p=Path(path)
        return p.is_absolute() and p.resolve()==p and not any(item.is_symlink() for item in (p,*p.parents))
    def lstat(self,path):return Path(path).lstat()
    def open(self,path):return os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    def fstat(self,fd):return os.fstat(fd)
    def read(self,fd,size):return os.read(fd,size)
    def close(self,fd):return os.close(fd)
    def maps(self):
        with Path('/proc/self/maps').open('rb') as stream:raw=stream.read(closure.MAP_CAP+1)
        require(len(raw)<=closure.MAP_CAP,'Bounded complete own process maps')
        return raw.decode('ascii')
    def resolve(self,path):return str(Path(path).resolve())
    def environment(self):return dict(os.environ)
    def running_executable(self):return os.readlink('/proc/self/exe')
    def module(self,name):return sys.modules.get(name)
    def modules(self):return tuple(sys.modules.values())
    def builtin_names(self):return tuple(sys.builtin_module_names)


class StableReader:
    """Complete hash/EOF and before/open/after identity checks, no writes."""
    def __init__(self,guard,ports=None):
        require(callable(guard),'Actual original descriptor/resource guard')
        self.guard=guard;self.ports=FilePorts() if ports is None else ports
        self.verified={};self.bytes_read=0
    def read(self,reference,cap=64*2**20,*,retain=True):
        self.guard();path=reference['path'];require(self.ports.canonical(path),'Canonical no-symlink authenticated file')
        before=identity(self.ports.lstat(path))
        require(stat.S_ISREG(before['st_mode']) and before['st_nlink']>=1
            and 0<=before['st_size']<=cap and reference.get('bytes',before['st_size'])==before['st_size'],
            'Bounded original regular source/native with observed link count')
        fd=self.ports.open(path);parts=[];count=0;h=hashlib.sha256()
        try:
            require(identity(self.ports.fstat(fd))==before,'Same exact file across open')
            while True:
                self.guard();raw=self.ports.read(fd,min(65536,before['st_size']+1-count))
                require(type(raw) is bytes,'Binary complete source/native reader')
                if not raw:break
                count+=len(raw);require(count<=before['st_size'],'No growth beyond exact original bytes')
                h.update(raw)
                if retain:parts.append(raw)
            require(count==before['st_size'] and h.hexdigest()==reference['sha256']
                and identity(self.ports.fstat(fd))==before==identity(self.ports.lstat(path))
                and self.ports.canonical(path),'Exact full hash, EOF and stable file identity')
        finally:self.ports.close(fd)
        self.guard();row=dict(path=path,bytes=count,sha256=h.hexdigest(),identity=before)
        self.verified[path]=row;self.bytes_read+=count
        return (b''.join(parts) if retain else None),copy.deepcopy(row)
    def json(self,reference,cap=8*2**20):return decode(self.read(reference,cap)[0])
    def unchanged(self,paths):
        self.guard()
        for path in paths:
            require(path in self.verified and self.ports.canonical(path)
                and identity(self.ports.lstat(path))==self.verified[path]['identity'],
                'Every authenticated source/native identity remains unchanged')
        self.guard()


def saved_origins(reader):
    """Replay saved JSON provenance, not archives/native payloads or booleans alone."""
    evidence=reader.json(ORIGIN_REF)
    review=reader.json(evidence['archive_root_review'])
    compare_ref=evidence['full_saved_comparison'];compressed,_=reader.read(compare_ref,2*2**20)
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw=stream.read(compare_ref['uncompressed_bytes']+1)
        require(stream.read(1)==b'','Complete bounded saved comparison EOF')
    require(len(raw)==compare_ref['uncompressed_bytes']
        and hashlib.sha256(raw).hexdigest()==compare_ref['uncompressed_sha256'],
        'Exact complete saved archive comparison payload')
    compared=decode(raw)
    require(review['complete_saved_comparison_independently_reread'] is True
        and review['full_comparison']==compare_ref and compared['archive_hashes_match_upstream'] is True
        and compared['archives']['cpython']==evidence['cpython_archive'],
        'Original independent archive review and exact official CPython archive')
    rows={row['path']:row for row in compared['findings']['cpython']['files']}
    for path,row in evidence['exact_original_executable_and_pure_source_rows'].items():
        require(rows.get(path)==row and row['status']=='exact'
            and row['sha256']==row['expected_sha256']==row['archive_sha256']
            and row['bytes']==row['archive_bytes'],'Every selected CPython executable/source is upstream exact')
    package=reader.json(evidence['package_result'])
    require(package['complete'] is True and package['package_content_authenticated'] is True,
            'Completed original full package-member replay')
    next_inputs=reader.json(package['inputs']['next_inputs'])
    signed=reader.json(next_inputs['signed_metadata'])
    require(signed['status']=='exact_addresses_ready' and signed['missing']==[],
            'Exact signed upstream package addresses')
    candidates=[row for row in package['packages'] if row['package']=='libc6' and row['version']=='2.35-0ubuntu3.15']
    signed_rows=[row for row in signed['packages'] if row['package']=='libc6' and row['version']=='2.35-0ubuntu3.15']
    require(len(candidates)==len(signed_rows)==1,'Exact original libc6 version')
    libc=candidates[0];official=signed_rows[0];inspection=libc['inspection']
    require(inspection['tar']['complete'] is True
        and inspection['archive']['sha256']==official['sha256']==evidence['libc_package']['archive']['sha256']
        and inspection['archive']['bytes']==official['bytes'],'Package replay archive matches signed package index')
    releases={row['release_url']:row for row in signed['releases']}
    for source in official['sources']:
        release=releases[source['release_url']];proof=release['signature_proof']
        require(release['release_sha256']==source['release_sha256'] and proof['verified'] is True
            and proof['process_returncode']==0 and proof['primary_fingerprint'] in signed['trust_anchor']['primary_fingerprints'],
            'Saved signed-release proof binds actual libc6 package source')
    native=evidence['saved_native_candidates'];require(len(native)==8 and EXE in native and LIBC in native and LOADER in native,
            'Original interpreter and seven libc6 members only')
    for path,row in native.items():
        origin=rows[path] if path==EXE else inspection['tar']['targets'][path]
        size=origin['bytes'] if path==EXE else origin['size']
        require(row['sha256']==origin['sha256'] and row['bytes']==size,'Every native candidate has exact independent member origin')
    return evidence,native


class AuthenticatedSourceLoader(machinery.SourceFileLoader):
    def __init__(self,name,path,auth):super().__init__(name,path);self.auth=auth
    def get_code(self,name):
        raw,_=self.auth.reader.read(dict(path=self.path,sha256=self.auth.source_pins[self.path]),8*2**20)
        self.auth.source_executed.add(self.path)
        return compile(raw,self.path,'exec',dont_inherit=True,optimize=0)


class SourceFinder:
    """Original PathFinder selection with exact source pins; never read .pyc."""
    def __init__(self,auth):self.auth=auth
    def find_spec(self,name,path=None,target=None):
        spec=machinery.PathFinder.find_spec(name,path,target)
        if spec is None:return None
        origin=spec.origin
        if origin in self.auth.source_pins:
            require(type(spec.loader) is machinery.SourceFileLoader and spec.loader.path==origin,
                    'Original source-file provider for authenticated code')
            spec.loader=AuthenticatedSourceLoader(name,origin,self.auth);return spec
        if origin and (origin.startswith(str(ROOT)+'/') or name=='ctypes' or name.startswith('ctypes.')):
            raise ValueError('Unpinned source provider before historical/native import: '+str(origin))
        return None
    def __enter__(self):sys.meta_path.insert(0,self);return self
    def __exit__(self,*args):
        require(sum(item is self for item in sys.meta_path)==1,'Exact original authenticated source finder retained')
        sys.meta_path.remove(self);return False


class NativeAuthenticator:
    """Implements all four history-gate callbacks plus backend revalidation."""
    def __init__(self,bootstrap_guard,*,reader=None):
        self.guard=bootstrap_guard;self.reader=StableReader(bootstrap_guard) if reader is None else reader
        self.bootstrap=self.authority=self.native_receipt=None
        self.native_pins={};self.source_pins={};self.elf={};self.ffi=None;self.mapping=None
        self.source_executed=set();self.loaded_ctypes=None;self.loaded_libc=None;self.failure=None
    def _check(self,guard=None):
        if self.failure is not None:raise self.failure
        self.guard()
        if guard is not None and guard is not self.guard:guard()
    def _maps(self):
        self.reader.unchanged(self.native_pins)
        identities={path:dict(device=[os.major(row['identity']['st_dev']),os.minor(row['identity']['st_dev'])],
            inode=row['identity']['st_ino']) for path,row in self.reader.verified.items() if path in self.native_pins}
        return closure.mapped_closure(self.elf,identities,self.reader.ports.maps(),EXE,LOADER,self.reader.ports.resolve)
    def authenticate_bootstrap(self,bootstrap,authority,guard):
        require(self.bootstrap is None,'Once-only concrete bootstrap source/native authentication')
        self.bootstrap=copy.deepcopy(bootstrap);self.authority=copy.deepcopy(authority)
        try:
            self._check(guard);gate.validate_bootstrap(bootstrap)
            require(bootstrap['binding']==self.guard.binding and bootstrap['outer_deadline']==self.guard.deadline,
                    'Actual bootstrap descriptor and deadline, no caller-supplied substitute')
            decision=self.reader.json(authority)
            require(decision['schema']=='minicpm5_sdk_mapping_native_source_review_v1' and decision['reviewer']=='/root'
                and decision['accepted'] is True and decision['origin_evidence']==ORIGIN_REF
                and decision['source_map_sha256']==gate.digest(bootstrap['source_sha256']),
                'Concrete reviewed source/FFI policy, with independent current verification below')
            environment=self.reader.ports.environment()
            require(not any(environment.get(k) for k in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH')),
                    'No additional dynamic-loader interposition/search environment')
            require(self.reader.ports.running_executable()==EXE,'Actual original executable')
            origin,self.native_pins=saved_origins(self.reader)
            require(decision['native_files']==self.native_pins,'Exact independently sourced native set')
            for path,row in self.native_pins.items():
                raw,_=self.reader.read(row);self.elf[path]=closure.parse_elf64(raw)
                if path==EXE:self.ffi=closure.ffi_definitions(raw)
            self.mapping=self._maps()
            require({'_ctypes','_struct'}<=set(self.reader.ports.builtin_names()),
                    'Original executable positively exposes the actual built-in ctypes/struct modules')
            self.source_pins={path:sha for path,sha in bootstrap['source_sha256'].items() if Path(path).suffix=='.py'}
            for path,row in origin['exact_original_executable_and_pure_source_rows'].items():
                if path!=EXE:
                    require(path not in self.source_pins or self.source_pins[path]==row['sha256'],'No upstream source pin replacement')
                    self.source_pins[path]=row['sha256']
            for path,sha in self.source_pins.items():self.reader.read(dict(path=path,sha256=sha),8*2**20,retain=False)
            self._check(guard)
            self.native_receipt=dict(schema='mini_mapping_observer_native_authority_v1',authority=copy.deepcopy(authority),
                current_bytes_authenticated=True,current_identities_authenticated=True,pure_sources_authenticated=True,
                dependency_closure_verified=True,original_nine_native_checks_preserved=True,unresolved=[],
                outer_executable=EXE,ctypes_root=gate.backend.CTYPES_ROOT,libc_path=LIBC,abi='linux-x86_64-lp64',
                evidence=dict(saved_origin=ORIGIN_REF,native_files=copy.deepcopy(self.native_pins),
                    ffi_definitions=copy.deepcopy(self.ffi),actual_mapped_paths=self.mapping['paths'],
                    original_nine_native_authenticated_here=False,original_full_auth_still_required=True,
                    source_map_sha256=gate.digest(bootstrap['source_sha256'])))
            return dict(schema='minicpm5_sdk_mapping_history_bootstrap_receipt_v1',binding=copy.deepcopy(bootstrap['binding']),
                outer_deadline=bootstrap['outer_deadline'],source_map_sha256=gate.digest(bootstrap['source_sha256']),
                importable_sources_authenticated=True,original_descriptor_verified=True,original_limits_verified=True,
                native_authority=copy.deepcopy(self.native_receipt))
        except BaseException as error:self.failure=error;raise
    def _load(self,name):
        try:
            self._check();require(self.native_receipt is not None,'Complete actual native authentication before source execution')
            with SourceFinder(self):return importlib.import_module(name)
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise
    def load_ctypes(self):return self._load('ctypes')
    def load_history(self):return self._load(gate.HISTORY_NAME)
    def verify_loaded_native(self,ctypes,libc,receipt,guard):
        try:
            self._check(guard);require(receipt==self.native_receipt,'Same concrete current native authority')
            for name in ('_ctypes','_struct'):
                module=self.reader.ports.module(name)
                require(module is not None and module.__spec__.origin=='built-in'
                    and module.__spec__.loader is machinery.BuiltinImporter and getattr(module,'__file__',None) is None,
                    'Actual built-in provider belongs to authenticated original executable')
            require(ctypes is self.reader.ports.module('ctypes') and ctypes.__file__==gate.backend.CTYPES_ROOT+'/__init__.py'
                and isinstance(ctypes.__loader__,AuthenticatedSourceLoader)
                and ctypes.__loader__.auth is self and libc._name==LIBC,
                'Actual exact source-executed ctypes and requested original libc')
            mapping=self._maps();address=ctypes.cast(libc.ptrace,ctypes.c_void_p).value
            require(type(address) is int and any(row['begin']<=address<row['end'] and 'x' in row['permissions']
                for row in mapping['selected_mappings'][LIBC]),'Actual ptrace function pointer maps to authenticated libc provider')
            self.reader.unchanged(self.source_pins);self.loaded_ctypes=ctypes;self.loaded_libc=libc
            self._check(guard);return True
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise
    def verify_loaded_history(self,original,pins,ctypes,guard):
        try:
            self._check(guard);require(pins==self.bootstrap['source_sha256'] and ctypes is self.loaded_ctypes,
                    'Whole original source map and exact already verified ctypes')
            for module in self.reader.ports.modules():
                path=getattr(module,'__file__',None)
                if path and path.startswith(str(ROOT)+'/'):
                    require(path in pins and path in self.reader.verified
                        and self.reader.verified[path]['sha256']==pins[path],'Every actual local history source pinned/authenticated')
            require(original.__file__ in self.source_executed,'Canonical original diagnostic executed from authenticated source, never pyc')
            self.reader.unchanged(self.source_pins);self._maps();self._check(guard);return True
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise
    def authenticate(self,authority,guard):
        try:
            self._check(guard);require(authority==self.authority and self.loaded_ctypes is not None,
                    'Revalidate same completed bootstrap native authority for actual backend')
            self.reader.unchanged(set(self.source_pins)|set(self.native_pins));self._maps()
            return copy.deepcopy(self.native_receipt)
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise
