"""Reviewed -c source entry; inert when imported for source-only tests.

Future callers must pass these exact source bytes as the -c argument, bind the
whole argv externally, and provide an independently reviewed role-specific
plan. No on-disk entry .py/.pyc is executed. This component does not authenticate
native libraries, acquire/release the serial descriptor, or start a child.
"""
import sys
import posix
import time

ROOT = '/home/ubuntu/experiments/trn1-latest-models-20261031'
OUTER_PYTHON = '/home/ubuntu/venvs/model_reference_2026/bin/python'
OUTER_EXE = '/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12'
SDK_PYTHON = '/home/ubuntu/venvs/vllm_neuron/bin/python'
PREFIX = '/dev/null'
PLAN_CAP = 2 ** 20
SOURCE_CAP = 2 ** 20
SOURCE_TOTAL_CAP = 16 * 2 ** 20
FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
BLOCKED = ('ctypes', '_ctypes', 'torch', 'numpy', 'neuronx_distributed',
    'neuronx_distributed_inference', 'minicpm5_python_trace_lifecycle_v4',
    'research.minicpm5_pv_sdk_import_diagnostic_v1')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(path):
    return (type(path) is str and path.startswith('/') and path != '/'
        and all(part and part not in ('.', '..') for part in path.split('/')[1:]))


def digest(raw, check=lambda: None):
    """Small pure-Python SHA256: no hashlib/SSL/native import before bootstrap.

    Only bounded source and JSON bytes use this implementation. Native payload
    hashing remains the original authenticator's responsibility.
    """
    require(type(raw) is bytes and len(raw) <= 2 * SOURCE_TOTAL_CAP, 'Bounded digest bytes')
    k = (0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
         0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
         0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
         0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
         0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
         0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
         0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
         0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2)
    h = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]
    message = raw + b'\x80' + b'\0' * ((55 - len(raw)) % 64) + (len(raw) * 8).to_bytes(8, 'big')
    mask = (1 << 32) - 1
    def rotate(x, n):
        return (x >> n) | ((x << (32 - n)) & mask)
    for offset in range(0, len(message), 64):
        if offset % 4096 == 0:
            check()
        w = [int.from_bytes(message[offset+i:offset+i+4], 'big') for i in range(0, 64, 4)]
        for i in range(16, 64):
            x,y = w[i-15],w[i-2]
            s0 = rotate(x,7) ^ rotate(x,18) ^ (x >> 3)
            s1 = rotate(y,17) ^ rotate(y,19) ^ (y >> 10)
            w.append((w[i-16] + s0 + w[i-7] + s1) & mask)
        a,b,c,d,e,f,g,hh = h
        for i in range(64):
            s1 = rotate(e,6) ^ rotate(e,11) ^ rotate(e,25)
            t1 = (hh + s1 + ((e & f) ^ (~e & g)) + k[i] + w[i]) & mask
            s0 = rotate(a,2) ^ rotate(a,13) ^ rotate(a,22)
            t2 = (s0 + ((a & b) ^ (a & c) ^ (b & c))) & mask
            hh,g,f,e,d,c,b,a = g,f,e,(d+t1)&mask,c,b,a,(t1+t2)&mask
        h = [(x+y)&mask for x,y in zip(h,(a,b,c,d,e,f,g,hh))]
    check()
    return ''.join('%08x' % value for value in h)


class SystemPorts:
    """Concrete operations, never called by the source-only test suite."""
    def now(self): return time.monotonic()
    def lstat(self, path): return posix.lstat(path)
    def open(self, path): return posix.open(path, posix.O_RDONLY|posix.O_NOFOLLOW|posix.O_NONBLOCK|posix.O_CLOEXEC)
    def fstat(self, fd): return posix.fstat(fd)
    def read(self, fd, size): return posix.read(fd, size)
    def close(self, fd): return posix.close(fd)
    def readlink(self, path): return posix.readlink(path)


class Entry:
    """Own the cache fence and source receipts for the whole future process.

    The original bootstrap successor must adopt deadline exactly: entry work
    is already charged to it. This object never initializes/restarts a timer.
    """
    def __init__(self, *, runtime=sys, ports=None):
        self.runtime = runtime
        self.ports = SystemPorts() if ports is None else ports
        self.started = self.ports.now()
        self.deadline = None
        self.failure = None
        self.role = self.plan = self.entry_sha256 = None
        self.audit = self.finder = None
        self.sources = {}
        self.source_bytes = 0
        self.prefix_identity = None
        self.state = 'new'
        self.native_authority_verified = False
        self.history_owner = None
        self.plan_reference = None
        self.admission_envelope = None
        self.types_tracker = self.types_observation = None

    def fail(self, error):
        if self.failure is None:
            self.failure = error
        raise self.failure

    def check(self):
        if self.failure is not None:
            raise self.failure
        try:
            require(self.deadline is not None and self.started <= self.ports.now() < self.deadline,
                    'Original absolute entry deadline, including pre-bootstrap imports')
            require(self.runtime.pycache_prefix == PREFIX and self.runtime.dont_write_bytecode,
                    'Original no-cache prefix/-B retained')
            require(self.prefix() == self.prefix_identity, 'Same non-directory cache prefix retained')
            if self.audit is not None:
                self.check_finder_ownership()
        except BaseException as error:
            self.fail(error)

    def check_finder_ownership(self):
        stack=self.runtime.meta_path
        require(sum(item is self.finder for item in stack)==1,'One retained no-cache source finder')
        index=stack.index(self.finder)
        if self.history_owner is None:
            require(index==0,'Entry source finder remains first before concrete history handoff')
            return
        gate,auth,guard,gate_module,auth_module=self.history_owner
        require(auth.guard is guard and gate.guard is guard and auth.failure is None
            and gate.state!='failed','Same concrete successful history/native owner')
        require(index<=3,'Only original temporary native-source/fence layers before entry finder')
        fences=0;sources=0
        for item in stack[:index]:
            if type(item) is gate_module.PreImportFence:
                fences+=1
                require(gate.state=='authenticating' and sources==0,
                        'Original preimport fence precedes source owner during native authentication')
            else:
                require(type(item) is auth_module.SourceFinder and item.auth is auth
                    and auth.native_receipt is not None and self.native_authority_verified,
                    'Only concrete authenticated source finder may precede entry finder')
                sources+=1
        require(fences<=1 and sources<=2,'Original one fence and at most nested concrete source loaders')

    def bind_history_owner(self,gate,auth,guard):
        """Concrete outer312 gate/authenticator handoff, not a caller predicate.

        The exact classes are already executed from this entry's pinned source
        loader. Bootstrap source verification below is required before ctypes.
        The permanent audit hook and original entry finder remain installed.
        """
        self.check()
        require(self.role=='outer312' and self.history_owner is None and not self.native_authority_verified,
                'One outer-only source ownership handoff')
        gate_module=self.runtime.modules.get('research.minicpm5_sdk_mapping_history_gate_v1')
        auth_module=self.runtime.modules.get('research.minicpm5_sdk_mapping_native_authenticator_v3')
        guard_module=self.runtime.modules.get('research.minicpm5_sdk_mapping_bootstrap_descriptor_v4')
        require(gate_module is not None and auth_module is not None and guard_module is not None
            and type(gate) is gate_module.HistoricalImportGate
            and type(auth) is auth_module.NativeAuthenticator
            and type(guard) is guard_module.EntryBootstrapDescriptor,
            'Exact source-verified concrete gate/authenticator/descriptor implementations')
        for module in (gate_module,auth_module,guard_module):
            require(module.__file__ in self.sources,'Concrete owner really executed from entry-pinned source')
        require(guard.entry is self and guard.deadline==self.deadline and guard.started==self.started
            and gate.guard is guard and auth.guard is guard and gate.state=='new'
            and gate.bootstrap['binding']==guard.binding
            and gate.bootstrap['outer_deadline']==self.deadline,
            'Same actual retained descriptor and absolute entry deadline')
        authority={key:self.plan['native_authority'][key] for key in ('path','sha256')}
        require(gate.authority==authority and auth.native_receipt is None,
                'Original outer role authority, not already imported native state')
        for method,name in ((gate.ctypes_loader,'load_ctypes'),(gate.history_loader,'load_history'),
                            (gate.verify_loaded_native,'verify_loaded_native'),
                            (gate.verify_loaded_history,'verify_loaded_history')):
            require(getattr(method,'__self__',None) is auth
                and getattr(method,'__func__',None) is getattr(auth_module.NativeAuthenticator,name),
                'Gate native/history steps are exact concrete authenticator methods')
        self.history_owner=(gate,auth,guard,gate_module,auth_module)
        self.check()

    def confirm_bootstrap_sources(self,auth):
        """Run after concrete native authentication and before gate fence exits."""
        self.check()
        require(self.history_owner is not None and self.history_owner[1] is auth
            and auth.native_receipt is not None and auth.failure is None,
            'Actual bound native authenticator completed before source ownership change')
        gate,_,guard,_,_=self.history_owner
        require(gate.state=='authenticating' and gate.receipt is None
            and auth.bootstrap==gate.bootstrap and auth.authority==gate.authority,
            'Same original preimport gate still owns transition')
        if self.role == 'outer312':
            require(self.types_tracker is not None and self.types_observation is None,
                    'Actual early outer types observation retained once')
            observed = self.types_tracker.finish(PIN)
            require(observed['executed_through_pinned_loader'] is True
                    and observed['original_execution_provenance_gap'] is False,
                    'Formal source ownership requires actual pinned types execution; never reload preexisting module')
            self.types_observation = observed
        pins=auth.bootstrap['source_sha256']
        for path,sha in self.plan['bootstrap_sources'].items():
            require(pins.get(path)==sha,'Every bootstrap pin retained in full original source map')
        for path,row in self.sources.items():
            current=auth.reader.verified.get(path)
            source_sha=pins.get(path)
            if self.role=='outer312' and path==PIN['path']:
                require(auth.source_pins.get(path)==PIN['sha256']
                    and (source_sha is None or source_sha==PIN['sha256']),
                    'Exact nonlocal types source authenticated by original archive origins, no PV domain expansion')
                source_sha=auth.source_pins[path]
            require(current is not None and source_sha==row['sha256']==current['sha256']
                and row['bytes']==current['bytes']
                and row['identity']==tuple(current['identity'][name] for name in FIELDS),
                'Actual executed bootstrap bytes and current full-auth source identities match')
        self.native_authority_verified=True
        self.check()

    def bind_sdk_history_owner(self,gate,auth,guard):
        self.check()
        require(self.role=='sdk310' and self.history_owner is None and not self.native_authority_verified,
                'Separate once-only SDK310 concrete source ownership')
        module=self.runtime.modules.get('research.minicpm5_sdk_mapping_worker_native_v4')
        guard_module=self.runtime.modules.get('research.minicpm5_sdk_mapping_worker_guard_v2')
        require(module is not None and guard_module is not None
            and type(gate) is module.SDKHistoryGate and type(auth) is module.SDKNativeAuthenticator
            and type(guard) is guard_module.SDKEntryGuard,
            'Exact separate SDK gate/authenticator/parent-lease guard source types')
        for item in (module,guard_module):
            require(item.__file__ in self.sources,'SDK owner source really executed from pinned bytes')
        require(guard.entry is self and gate.guard is guard and auth.guard is guard
            and gate.entry is self and gate.auth is auth and gate.state=='new'
            and auth.native_receipt is None and gate.bootstrap==auth.bootstrap
            and gate.bootstrap['outer_deadline']==self.deadline,
            'Same acknowledged SDK child, source release and inherited absolute deadline')
        authority={key:self.plan['native_authority'][key] for key in ('path','sha256')}
        require(gate.authority==auth.authority==authority,
                'Only independent SDK310 authority, never outer312 receipt')
        self.history_owner=(gate,auth,guard,module,module)
        self.check()

    def native_import_allowed(self,name):
        if self.history_owner is None or not self.native_authority_verified:
            return False
        gate,auth,guard,gate_module,auth_module=self.history_owner
        if name.split('.')[0] in ('torch','numpy','neuronx_distributed','neuronx_distributed_inference'):
            return self.role=='sdk310' and gate.state=='complete_original_authentication'
        require(gate.receipt is not None and gate.receipt['native_authority']==auth.native_receipt,
                'Concrete original gate accepted native receipt before ctypes/history import')
        require(any(type(item) is auth_module.SourceFinder and item.auth is auth
            for item in self.runtime.meta_path[:self.runtime.meta_path.index(self.finder)]),
            'ctypes/history only through exact authenticated no-cache source loader')
        return True

    def prefix(self):
        dev = self.ports.lstat('/dev')
        null = self.ports.lstat(PREFIX)
        require(dev.st_mode & 0o170000 == 0o040000 and null.st_mode & 0o170000 == 0o020000
                and null.st_rdev == 259, '/dev/null is canonical Linux character device 1:3, never a cache directory')
        return (dev.st_dev, dev.st_ino, dev.st_mode, null.st_dev, null.st_ino, null.st_mode, null.st_rdev)

    def read_source(self, path, sha, cap=SOURCE_CAP):
        self.check()
        require(canonical(path) and type(sha) is str and len(sha)==64
                and all(c in '0123456789abcdef' for c in sha), 'Exact source/plan path and SHA256')
        parts = path.split('/')[1:]
        for i in range(len(parts)-1):
            require(self.ports.lstat('/'+'/'.join(parts[:i+1])).st_mode & 0o170000 == 0o040000,
                    'Canonical source ancestry, no symlinks')
        before = self.ports.lstat(path)
        fields = tuple(getattr(before, name) for name in FIELDS)
        require(before.st_mode & 0o170000 == 0o100000 and before.st_nlink >= 1
                and 0 < before.st_size <= cap, 'Bounded regular source/plan')
        fd = self.ports.open(path)
        data = bytearray()
        try:
            require(tuple(getattr(self.ports.fstat(fd), name) for name in FIELDS)==fields,
                    'Source before/open identities equal')
            while True:
                self.check()
                block = self.ports.read(fd, min(65536, before.st_size + 1 - len(data)))
                require(type(block) is bytes, 'Binary source read')
                if not block:
                    break
                data.extend(block)
                require(len(data) <= before.st_size, 'Source size cannot grow')
            require(len(data)==before.st_size
                and tuple(getattr(self.ports.fstat(fd), name) for name in FIELDS)==fields
                and tuple(getattr(self.ports.lstat(path), name) for name in FIELDS)==fields,
                'Complete source EOF and stable identities')
        finally:
            self.ports.close(fd)
        raw = bytes(data)
        require(digest(raw, self.check)==sha, 'Executed source bytes match reviewed SHA256')
        return raw, fields

    def startup(self, source_literal):
        """Validate actual -c argv before decoding any external JSON/module."""
        require(self.state=='new', 'Once-only no-cache entry startup')
        self.state='starting'
        try:
            s = self.runtime
            flags = s.flags
            require(flags.isolated==1 and flags.no_site==1 and flags.ignore_environment==1
                and flags.no_user_site==1 and flags.optimize==0 and s.dont_write_bytecode
                and s.pycache_prefix==PREFIX, 'Exact -I -S -B plus non-directory pycache prefix')
            require(type(source_literal) is str and 0 < len(source_literal.encode()) < 65536,
                    'Bounded literal reviewed source entry')
            args = s.argv
            require(len(args)==8 and args[0]=='-c' and args[1]=='--mini-source-entry'
                and args[2] in ('outer312','sdk310'), 'Exact role-specific source-entry CLI')
            self.role=args[2]
            python = OUTER_PYTHON if self.role=='outer312' else SDK_PYTHON
            require(s.executable==python and tuple(s.version_info[:2])==
                ((3,12) if self.role=='outer312' else (3,10)), 'Original per-role Python, no outer/SDK substitution')
            require(s.orig_argv==[python,'-I','-S','-B','-X','pycache_prefix='+PREFIX,'-c',source_literal,*args[1:]],
                    'Actual immutable -c literal and complete original argv')
            require(args[7]=='new' if self.role=='outer312' else args[7]!='new',
                    'Only outer establishes deadline; worker inherits same outer absolute deadline')
            self.deadline = self.started+900 if args[7]=='new' else float(args[7])
            require(self.started < self.deadline <= self.started+900,
                    'Original finite 900s deadline, never worker restart')
            self.prefix_identity=self.prefix()
            self.entry_sha256=digest(source_literal.encode(), self.check)
            require(self.entry_sha256==args[6], 'Actual executed -c source bytes authenticated')
            require(not any(n==p or n.startswith(p+'.') for n in s.modules for p in BLOCKED),
                    'No historical/ctypes/model preimport before entry')
            for name in ('sys','posix','time'):
                require(name in s.builtin_module_names and s.modules[name].__spec__.origin=='built-in',
                        'Entry uses original built-in primitives only')
            # Startup source modules must have selected .py providers; CPython
            # attempted cache paths are descendants of the validated character
            # device. This also rejects a preloaded direct sourceless provider.
            for module in tuple(s.modules.values()):
                path=getattr(module,'__file__',None)
                require(not path or not str(path).endswith(('.pyc','.pyo')),
                        'No startup sourceless bytecode provider')
            self.state='prefix_verified'
            return args[3],args[4],args[5]
        except BaseException as error:
            self.fail(error)

    def install(self, external):
        """Use existing frozen import machinery; never import importlib/ctypes."""
        self.check()
        require(self.state=='prefix_verified' and self.audit is None,
                'Cache fence installed once before any bootstrap source import')
        entry=self

        class SourceLoader(external.SourceFileLoader):
            def get_code(self, name):
                try:
                    entry.check()
                    raw, fields=entry.read_source(self.path,entry.plan['bootstrap_sources'][self.path])
                    entry.source_bytes+=len(raw)
                    require(entry.source_bytes<=SOURCE_TOTAL_CAP, 'Bounded total bootstrap source bytes')
                    code=compile(raw,self.path,'exec',dont_inherit=True,optimize=0)
                    entry.sources[self.path]=dict(sha256=digest(raw,entry.check),bytes=len(raw),identity=fields)
                    return code
                except BaseException as error:
                    entry.fail(error)

        class Finder:
            def find_spec(self, name, path=None, target=None):
                try:
                    entry.check()
                    if any(name==p or name.startswith(p+'.') for p in BLOCKED):
                        require(entry.native_import_allowed(name),
                                'Entry still precedes separate native/history authority')
                    spec=external.PathFinder.find_spec(name,path,target)
                    if spec is None:
                        return None
                    origin=spec.origin
                    if origin and str(origin).endswith(('.pyc','.pyo')):
                        raise ValueError('Sourceless bytecode is forbidden')
                    if origin and origin.startswith(ROOT+'/'):
                        require(entry.plan is not None and origin in entry.plan['bootstrap_sources']
                            and type(spec.loader) is external.SourceFileLoader,
                            'Every bootstrap local source is explicitly pinned')
                        spec.loader=SourceLoader(name,origin)
                        return spec
                    return None
                except BaseException as error:
                    entry.fail(error)

        def audit(event, args):
            if event=='minicpm5.nocache.install' and args==(entry,):
                entry.audit_seen=True
            # Importlib treats this as an unavailable cache and proceeds to
            # source. Direct sourceless imports cannot get bytes either.
            if event=='open' and args and isinstance(args[0],(str,bytes)):
                name=args[0].decode('utf-8','surrogateescape') if isinstance(args[0],bytes) else args[0]
                if name.endswith(('.pyc','.pyo')):
                    raise FileNotFoundError('Entry refuses all bytecode cache reads/writes')

        self.finder=Finder()
        self.runtime.meta_path.insert(0,self.finder)
        self.audit_seen=False
        self.runtime.addaudithook(audit)
        self.audit=audit
        self.runtime.audit('minicpm5.nocache.install',self)
        require(self.audit_seen,'The actual runtime installed the permanent audit hook')
        self.state='fenced'

    def load_plan(self, reference, json_module):
        self.check()
        try:
            require(self.state=='fenced' and self.plan is None, 'One exact entry plan')
            raw,_=self.read_source(reference['path'],reference['sha256'],PLAN_CAP)
            def unique(pairs):
                out={}
                for key,value in pairs:
                    require(key not in out,'Unique entry plan keys');out[key]=value
                return out
            def nonfinite(value):
                raise ValueError('Finite entry JSON')
            plan=json_module.loads(raw,object_pairs_hook=unique,parse_constant=nonfinite)
            self.plan_reference=dict(reference)
            if plan.get('schema')=='minicpm5_fixed_input_pv_sdk_import_diagnostic_admission_v1':
                require(self.role=='sdk310' and type(plan.get('source_entry_plan')) is dict,
                        'Only SDK uses parent-authenticated existing admission as its entry plan')
                self.admission_envelope=plan
                plan=plan['source_entry_plan']
            require(set(plan)=={'schema','role','python','executable','native_authority',
                'bootstrap_sources','entry_sha256','entry_module','entry_path','successor_input'},
                'Exact reviewed per-role entry plan fields')
            require(plan['schema']=='minicpm5_source_entry_plan_v1' and plan['role']==self.role
                and plan['entry_sha256']==self.entry_sha256 and plan['python']==self.runtime.executable,
                'Entry plan binds actual literal/role/interpreter')
            exe=self.ports.readlink('/proc/self/exe')
            require(canonical(exe) and plan['executable']==exe and
                (exe==OUTER_EXE if self.role=='outer312' else exe!=OUTER_EXE),
                'Per-role actual original executable; SDK cannot reuse outer executable')
            authority=plan['native_authority']
            require(type(authority) is dict and set(authority)=={'role','python','path','sha256'}
                and authority['role']==self.role and authority['python']==plan['python']
                and canonical(authority['path']) and len(authority['sha256'])==64,
                'Explicit separate outer312 versus SDK310 native authority; not validated here')
            pins=plan['bootstrap_sources']
            require(type(pins) is dict and 0<len(pins)<=2048 and all(canonical(p)
                and p.startswith(ROOT+'/') and p.endswith('.py') and type(sha) is str
                and len(sha)==64 and all(c in '0123456789abcdef' for c in sha) for p,sha in pins.items()),
                'Exact bounded complete pre-native local source closure')
            require(type(plan['entry_module']) is str and all(part.isidentifier() for part in plan['entry_module'].split('.'))
                and plan['entry_path']==ROOT+'/'+plan['entry_module'].replace('.','/')+'.py'
                and plan['entry_path'] in pins, 'Explicit future reviewed source successor, no old CLI fallback')
            self.plan=plan
            self.state='plan_verified'
            return plan
        except BaseException as error:
            self.fail(error)

    def receipt(self):
        self.check()
        require(self.plan is not None, 'No entry receipt before exact plan authentication')
        return dict(schema='minicpm5_source_entry_receipt_v1',role=self.role,
            python=self.plan['python'],executable=self.plan['executable'],
            entry_sha256=self.entry_sha256,started_monotonic=self.started,
            outer_deadline=self.deadline,pycache_prefix=PREFIX,
            prefix_identity=self.prefix_identity,executed_sources={p:dict(r) for p,r in self.sources.items()},
            native_authority=dict(self.plan['native_authority']),native_authority_verified=False)


"""One outer312 source pin and honest pre-JSON execution observation.

Inert, import-free source for a future exact frozen literal. No filesystem or
process operations of its own. Source reads use the existing bounded Entry.
Do not import this local module before Entry.plan is available; embed its exact
reviewed source alongside the existing helper, before child.main is called.
"""

PIN = {
    'path': '/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/types.py',
    'sha256': '345474ef027a1273f353da9bdc1f7c18f65335e72e681bcc0376774cc51f2405',
    'bytes': 10993,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def provider_snapshot(runtime):
    module = runtime.modules.get('types')
    spec = getattr(module, '__spec__', None)
    return dict(present=module is not None,
                module_file=getattr(module, '__file__', None),
                spec_name=getattr(spec, 'name', None),
                spec_origin=getattr(spec, 'origin', None),
                loader_type=type(getattr(module, '__loader__', None)).__name__,
                loader_is_spec_loader=getattr(module, '__loader__', None) is getattr(spec, 'loader', None))


class TypesSourceObservation:
    """Keep loaded-at-entry evidence separate from execution through our loader."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.initial_module = runtime.modules.get('types')
        self.initial_snapshot = provider_snapshot(runtime)
        self.entry = self.external = self.finder = self.loader_type = None
        self.executed_module = self.execution_receipt = None
        self.after_json_snapshot = None
        self.finished = False

    def _source(self):
        raw, identity = self.entry.read_source(PIN['path'], PIN['sha256'], PIN['bytes'])
        require(len(raw) == PIN['bytes'], 'Exact reviewed outer types source size')
        self.entry.source_bytes += len(raw)
        require(self.entry.source_bytes <= self.source_total_cap, 'Original whole-source byte cap')
        return raw, dict(sha256=PIN['sha256'], bytes=len(raw), identity=identity)

    def _provider(self, module, loader):
        spec = getattr(module, '__spec__', None)
        require(module is not None and getattr(module, '__name__', None) == 'types'
                and getattr(module, '__file__', None) == PIN['path']
                and getattr(spec, 'name', None) == 'types'
                and getattr(spec, 'origin', None) == PIN['path']
                and getattr(module, '__loader__', None) is loader
                and getattr(spec, 'loader', None) is loader,
                'Exact loaded types module/source/spec/loader correspondence')

    def install_before_json(self, entry, external, source_total_cap):
        require(self.entry is None and self.runtime.modules.get('types') is self.initial_module,
                'One installation; preserve earliest types module identity')
        require(source_total_cap == 16 * 2**20, 'Unchanged original whole-source byte cap')
        require(self.runtime.meta_path and self.runtime.meta_path[0] is entry.finder,
                'Wrap the original first source finder')
        self.entry, self.external = entry, external
        self.source_total_cap = source_total_cap
        tracker, previous = self, entry.finder

        class TypesLoader(external.SourceFileLoader):
            def get_code(self, name):
                require(name == 'types' and self.path == PIN['path'], 'Only the exact types source')
                raw, row = tracker._source()
                code = compile(raw, PIN['path'], 'exec', dont_inherit=True, optimize=0)
                self.pending_receipt = row
                return code

            def exec_module(self, module):
                require(tracker.initial_module is None and tracker.executed_module is None,
                        'Never reload or replace an already loaded types module')
                tracker._provider(module, self)
                entry.check()
                super().exec_module(module)
                require(tracker.runtime.modules.get('types') is module,
                        'Same module remains registered after complete execution')
                tracker._provider(module, self)
                entry.check()
                # get_code alone is not an execution receipt. Record only after
                # SourceFileLoader.exec_module returns successfully.
                entry.sources[PIN['path']] = dict(self.pending_receipt)
                tracker.execution_receipt = dict(self.pending_receipt)
                tracker.executed_module = module

        class TypesFinder:
            def find_spec(self, name, path=None, target=None):
                found = previous.find_spec(name, path, target)
                if name != 'types':
                    return found
                require(found is None, 'No different earlier types provider')
                spec = external.PathFinder.find_spec(name, path, target)
                require(spec is not None and spec.origin == PIN['path']
                        and type(spec.loader) is external.SourceFileLoader,
                        'Only the reviewed ordinary types source provider')
                spec.loader = TypesLoader(name, PIN['path'])
                return spec

        self.loader_type = TypesLoader
        self.finder = TypesFinder()
        self.runtime.meta_path[0] = self.finder
        entry.finder = self.finder
        entry.check()

    def after_json(self):
        require(self.entry is not None and self.after_json_snapshot is None,
                'One explicit checkpoint after successful JSON import')
        self.after_json_snapshot = provider_snapshot(self.runtime)

    def finish(self, candidate_row):
        require(not self.finished and self.after_json_snapshot is not None,
                'Complete observation only after JSON checkpoint')
        require({key: candidate_row[key] for key in PIN} == PIN,
                'Early literal pin equals the later authenticated candidate table')
        module = self.runtime.modules.get('types')
        if self.executed_module is not None:
            require(module is self.executed_module
                    and type(getattr(module, '__loader__', None)) is self.loader_type
                    and self.entry.sources.get(PIN['path']) == self.execution_receipt,
                    'Same completed pinned types execution and retained receipt')
        else:
            require(self.initial_module is not None and module is self.initial_module
                    and type(getattr(module, '__loader__', None)) is self.external.SourceFileLoader
                    and PIN['path'] not in self.entry.sources,
                    'Preexisting module stays preexisting; never fabricate its execution receipt')
        self._provider(module, module.__loader__)
        _, current_source = self._source()
        self.entry.check()
        self.finished = True
        executed = self.executed_module is not None
        return dict(schema='minicpm5_outer_types_source_observation_v1',
                    first_checkpoint=self.initial_snapshot,
                    after_json=self.after_json_snapshot,
                    final_provider=provider_snapshot(self.runtime),
                    initial_load_timing='before_first_checkpoint' if self.initial_module is not None else 'after_first_checkpoint',
                    current_source_reference=dict(PIN), current_source_identity=current_source['identity'],
                    current_source_bytes_verified=True, loaded_provider_metadata_matches=True,
                    executed_through_pinned_loader=executed,
                    execution_receipt=dict(self.execution_receipt) if executed else None,
                    original_execution_provenance_gap=not executed,
                    interpreter_startup_cause_established=False,
                    preexisting_module_reloaded=False,
                    accepted_native_authority_created=False,
                    full_model_qualified=False, new_speedup_claim=False)

def main():
    """Future literal invocation only. No CLI fallback to historical worker."""
    entry=Entry()
    source=sys.orig_argv[7] if len(sys.orig_argv)>7 else ''
    plan_path,plan_sha,successor_input=entry.startup(source)
    if entry.role == 'outer312':
        entry.types_tracker = TypesSourceObservation(sys)
    external=sys.modules['_frozen_importlib_external']
    require(external.__spec__.origin=='frozen','Interpreter frozen import machinery')
    entry.install(external)
    if entry.types_tracker is not None:
        entry.types_tracker.install_before_json(entry, external, SOURCE_TOTAL_CAP)
    import json  # After the non-directory prefix and bytecode audit fence.
    if entry.types_tracker is not None:
        entry.types_tracker.after_json()
    plan=entry.load_plan(dict(path=plan_path,sha256=plan_sha),json)
    require(plan['successor_input']==successor_input,'Exact future original input reference')
    # Both trusted source roots are explicit; the installed finder still
    # requires every selected local file in the complete reviewed source map.
    # The actual dual-bootstrap discovery used this same ROOT/Q order.
    sys.path[:0]=[ROOT,ROOT+'/research']
    try:
        module=__import__(plan['entry_module'],fromlist=['source_entry'])
        require(module.__file__==plan['entry_path'] and module.__file__ in entry.sources,
                'Pinned successor really executed from source')
        result=module.source_entry(entry,successor_input)
        entry.check()
        require(type(result) is int and 0<=result<=255,'Successor returns actual explicit process exit code')
        return result
    except BaseException as error:
        entry.fail(error)


if __name__=='__main__':
    sys.exit(main())
