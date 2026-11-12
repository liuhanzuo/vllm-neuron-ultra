"""Separate Python3.10 authority with explicit composite package origins; inert on import.

Concrete future checks replay signed package-member records, hash current full
files, parse ELF, and check actual pre/post ctypes maps. No Python3.12 archive or
outer embedded-FFI receipt can authenticate this separate dynamic _ctypes path.
No presently accepted SDK authority or native origin result is created here.
"""
import copy
import importlib
from importlib import machinery
import os
from pathlib import Path
import sys
from research import minicpm5_sdk_mapping_native_authenticator_v3 as shared
from research import minicpm5_sdk_mapping_history_gate_v1 as history
from research import minicpm5_sdk310_composite_origins_v2 as origins

require=shared.require
ROOT=shared.ROOT
PreImportFence=history.PreImportFence
CTYPES_ROOT='/usr/lib/python3.10/ctypes'
SDK='/home/ubuntu/venvs/vllm_neuron/bin/python'
EXE='/usr/bin/python3.10'
CTYPES_EXTENSION='/usr/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so'
LIBC='/usr/lib/x86_64-linux-gnu/libc.so.6'
LOADER='/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'


class SourceFinder(shared.SourceFinder):
    """SDK needs a dynamic _ctypes provider; outer312 embeds it instead.

    Pin the actual PathFinder result before ExtensionFileLoader may execute it.
    Merely hashing an available extension is not provider selection proof.
    """
    def find_spec(self,name,path=None,target=None):
        if name=='_ctypes':
            self.auth._check()
            spec=machinery.PathFinder.find_spec(name,path,target)
            require(spec is not None and spec.origin==CTYPES_EXTENSION
                and type(spec.loader) is machinery.ExtensionFileLoader,
                'Exact independently authenticated SDK310 _ctypes before extension execution')
            self.auth.reader.unchanged({CTYPES_EXTENSION})
            return spec
        return super().find_spec(name,path,target)


def replay_package_origins(reader,decision,pins):
    return origins.replay_package_origins(reader,decision,pins)


class SDKNativeAuthenticator:
    def __init__(self,guard,pins,authority,*,reader=None):
        self.guard=guard
        self.reader=shared.StableReader(guard) if reader is None else reader
        self.source_pins={p:sha for p,sha in pins.items() if p.endswith('.py')}
        self.bootstrap=dict(source_sha256=copy.deepcopy(pins),binding=copy.deepcopy(guard.binding),
            outer_deadline=guard.entry.deadline)
        self.authority=copy.deepcopy(authority)
        self.native_receipt=self.decision=None
        self.native_pins={};self.elf={};self.source_executed=set()
        self.failure=None;self.loaded_ctypes=self.loaded_libc=None

    def _check(self):
        if self.failure is not None:raise self.failure
        self.guard()

    def maps(self,paths):
        self.reader.unchanged(paths)
        metadata={p:self.elf[p] for p in paths}
        identities={p:dict(device=[os.major(self.reader.verified[p]['identity']['st_dev']),
            os.minor(self.reader.verified[p]['identity']['st_dev'])],
            inode=self.reader.verified[p]['identity']['st_ino']) for p in paths}
        return shared.closure.mapped_closure(metadata,identities,self.reader.ports.maps(),EXE,LOADER,
            self.reader.ports.resolve)

    def authenticate_bootstrap(self):
        require(self.native_receipt is None and self.decision is None,'Once-only independent SDK310 authority')
        try:
            self._check()
            d=self.reader.json(self.authority);self.decision=d
            require(d['schema']=='minicpm5_sdk310_native_source_review_v3' and d['reviewer']=='/root'
                and d['accepted'] is True and d['role']=='sdk310' and d['python']==SDK
                and d['executable']==EXE and d['ctypes_root']==CTYPES_ROOT
                and d['ctypes_extension']==CTYPES_EXTENSION
                and d['source_map_sha256']==history.digest(self.bootstrap['source_sha256']),
                'Concrete separate accepted SDK310 authority and exact complete map')
            require(self.reader.ports.running_executable()==EXE,
                    'Actual original SDK310 executable, never outer312')
            environment=self.reader.ports.environment()
            require(not any(environment.get(k) for k in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH')),
                    'No unauthenticated native loader interposition/search override')
            replay_package_origins(self.reader,d,self.bootstrap['source_sha256'])
            self.native_pins=d['native_files']
            before,after=set(d['mapped_before_ctypes']),set(d['mapped_after_ctypes'])
            require(before<after==set(self.native_pins) and {EXE,LOADER,LIBC}<=before
                and CTYPES_EXTENSION in after-before and d['ffi_provider'] in after
                and d['ffi_provider']!=EXE,'Separate complete pre/post SDK native providers')
            for path,row in self.native_pins.items():
                raw,_=self.reader.read(row)
                self.elf[path]=shared.closure.parse_elf64(raw)
                if path==d['ffi_provider']:ffi=shared.closure.ffi_definitions(raw)
            # Before loading any new _ctypes library, every declared needed
            # provider must already be a signed, hashed candidate. Actual mapped
            # provider selection is checked again after the import below.
            sonames={row['soname']:path for path,row in self.elf.items() if row['soname'] is not None}
            require(len(sonames)==sum(row['soname'] is not None for row in self.elf.values()),
                    'Unique authenticated SDK dependency SONAMEs')
            require(all(not row['extra_loader_directives'] and all(name in sonames for name in row['needed'])
                for row in self.elf.values()),'Complete prospective SDK dependency closure before dlopen')
            require(any(sonames[name]==d['ffi_provider'] for name in self.elf[CTYPES_EXTENSION]['needed']),
                    'Actual _ctypes DT_NEEDED positively selects signed external FFI provider')
            before_maps=self.maps(before)
            for path,row in d['pure_sources'].items():
                require(path.startswith('/usr/lib/python3.10/') and path.endswith('.py'),
                        'Separate original Python3.10 package pure source')
                require(path not in self.source_pins or self.source_pins[path]==row['sha256'],
                        'No replacement of inherited SDK source pin')
                self.source_pins[path]=row['sha256']
            require(CTYPES_ROOT+'/__init__.py' in self.source_pins
                and CTYPES_ROOT+'/_endian.py' in self.source_pins,'Complete ordinary ctypes source entry/endian closure')
            for path,sha in self.source_pins.items():
                self.reader.read(dict(path=path,sha256=sha),8*2**20,retain=False)
            self.native_receipt=dict(schema='minicpm5_sdk310_early_native_receipt_v1',role='sdk310',
                authority=copy.deepcopy(self.authority),executable=EXE,ctypes_extension=CTYPES_EXTENSION,
                ffi_provider=d['ffi_provider'],positive_external_ffi_definitions=ffi,
                native_files=copy.deepcopy(self.native_pins),before_mapping=before_maps,
                original_nine_still_required=True,current_after_mapping_verified=False)
            self._check()
            return copy.deepcopy(self.native_receipt)
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise self.failure

    def load_ctypes(self):
        try:
            self._check();require(self.native_receipt is not None,'Separate actual SDK native authority first')
            spec=machinery.PathFinder.find_spec('_ctypes')
            require(spec is not None and spec.origin==CTYPES_EXTENSION
                and type(spec.loader) is machinery.ExtensionFileLoader,
                'Actual SDK310 _ctypes selection matches hashed extension before execution')
            with SourceFinder(self):
                ctypes=importlib.import_module('ctypes')
            require(ctypes.__file__==CTYPES_ROOT+'/__init__.py'
                and isinstance(ctypes.__loader__,shared.AuthenticatedSourceLoader)
                and ctypes.__loader__.auth is self
                and sys.modules['_ctypes'].__spec__.origin==CTYPES_EXTENSION
                and type(sys.modules['_ctypes'].__spec__.loader) is machinery.ExtensionFileLoader,
                'Actual separately authenticated SDK source/native providers')
            require(ctypes.sizeof(ctypes.c_void_p)==ctypes.sizeof(ctypes.c_long)==8
                and ctypes.sizeof(ctypes.c_int)==4,'Original Linux x86_64 LP64 SDK ABI')
            libc=ctypes.CDLL(LIBC,use_errno=True)
            mapped=self.maps(set(self.decision['mapped_after_ctypes']))
            address=ctypes.cast(libc.ptrace,ctypes.c_void_p).value
            require(type(address) is int and any(row['begin']<=address<row['end'] and 'x' in row['permissions']
                for row in mapped['selected_mappings'][LIBC]),'Actual SDK libc function belongs to authenticated mapped provider')
            self.native_receipt['after_mapping']=mapped
            self.native_receipt['current_after_mapping_verified']=True
            self.loaded_ctypes,self.loaded_libc=ctypes,libc
            self._check()
            return ctypes
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise self.failure

    def load_history(self):
        try:
            self._check();require(self.loaded_ctypes is not None,'Verified actual SDK ctypes before historical imports')
            with SourceFinder(self):original=importlib.import_module(history.HISTORY_NAME)
            require(original.__file__==history.HISTORY_PATH and original.__file__ in self.source_executed,
                    'Full canonical history executed from authenticated source in SDK310')
            self.reader.unchanged(self.source_pins)
            self.maps(set(self.decision['mapped_after_ctypes']))
            self._check()
            return original
        except BaseException as error:
            if self.failure is None:self.failure=error
            raise self.failure


NativeAuthenticator=SDKNativeAuthenticator


class SDKHistoryGate:
    def __init__(self,entry,auth,guard):
        self.entry,self.auth,self.guard=entry,auth,guard
        self.bootstrap=copy.deepcopy(auth.bootstrap)
        self.authority=copy.deepcopy(auth.authority)
        self.state='new';self.receipt=self.original=None

    def load(self):
        require(self.state=='new','Once-only separate SDK history gate')
        self.state='authenticating'
        try:
            with PreImportFence():
                require(not any(name==prefix or name.startswith(prefix+'.')
                    for name in sys.modules for prefix in history.BLOCKED_BEFORE_AUTH),
                    'SDK starts without historical/ctypes/model imports')
                receipt=self.auth.authenticate_bootstrap()
                self.entry.confirm_bootstrap_sources(self.auth)
                self.receipt=dict(native_authority=self.auth.native_receipt)
            self.auth.load_ctypes()
            self.receipt['native_authority']=copy.deepcopy(self.auth.native_receipt)
            self.original=self.auth.load_history()
            self.guard();self.state='history_loaded'
            return self.original
        except BaseException:
            self.state='failed'
            raise
