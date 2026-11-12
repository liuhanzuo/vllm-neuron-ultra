"""Synthetic ELF/native/ports and existing JSON only. No actual native read/import."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research'),str(ROOT)]
FORBIDDEN=('ctypes','_ctypes','torch','numpy','neuronx_distributed','neuronx_distributed_inference')
class RejectNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in FORBIDDEN:raise AssertionError('Native import forbidden: '+name)
assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
sys.meta_path.insert(0,RejectNative())
from research import minicpm5_sdk_mapping_native_authenticator_v2 as a
from research import minicpm5_sdk_mapping_native_closure_v1 as c
from research import check_dory_loader_candidate_elf_v1 as old_fixture


def elf(soname=None,needed=(),*,interpreter=False,ffi=False):
    strings=[(1,name) for name in needed]+([] if soname is None else [(14,soname)])
    raw,_=old_fixture.fixture(strings)
    if interpreter:
        struct.pack_into('<H',raw,56,3);name=b'/lib64/ld-linux-x86-64.so.2\0'
        raw[0x300:0x300+len(name)]=name
        struct.pack_into('<IIQQQQQQ',raw,176,3,4,0x300,0x400300,0,len(name),len(name),1)
    if ffi:
        struct.pack_into('<I',raw,68,5)
        struct.pack_into('<Q',raw,40,0x1400);struct.pack_into('<HHH',raw,58,64,4,0)
        symbols=b'\0'+b''.join(name.encode()+b'\0' for name in sorted(c.FFI_FUNCTIONS))
        raw[0x1800:0x1800+len(symbols)]=symbols
        struct.pack_into('<IIQQQQIIQQ',raw,0x1440,0,11,0,0,0x1600,24*(len(c.FFI_FUNCTIONS)+1),2,0,8,24)
        struct.pack_into('<IIQQQQIIQQ',raw,0x1480,0,3,0,0,0x1800,len(symbols),0,0,1,0)
        struct.pack_into('<IIQQQQIIQQ',raw,0x14c0,0,1,6,0x400200,0x200,0x100,0,0,16,0)
        for index,name in enumerate(sorted(c.FFI_FUNCTIONS),1):
            struct.pack_into('<IBBHQQ',raw,0x1600+24*index,symbols.index(name.encode()+b'\0'),0x12,0,3,0x400200+16*index,8)
    return bytes(raw)

class MemoryPorts:
    def __init__(self,files):
        self.files=dict(files);self.streams={};self.next_fd=10;self.closed=[];self.read_paths=[]
        self.identities={p:NS(st_dev=os.makedev(1,2),st_ino=i+100,st_mode=stat.S_IFREG|0o644,
            st_nlink=1,st_size=len(raw),st_mtime_ns=1,st_ctime_ns=2) for i,(p,raw) in enumerate(files.items())}
        self.env={};self.map_text='';self.registry={};self.builtins=('_ctypes','_struct');self.current_exe=a.EXE
    def canonical(self,path):return path in self.files
    def lstat(self,path):return copy.deepcopy(self.identities[path])
    def open(self,path):
        self.next_fd+=1;fd=self.next_fd;self.streams[fd]=(path,io.BytesIO(self.files[path]));self.read_paths.append(path);return fd
    def fstat(self,fd):return self.lstat(self.streams[fd][0])
    def read(self,fd,size):return self.streams[fd][1].read(size)
    def close(self,fd):self.closed.append(fd);self.streams[fd][1].close()
    def maps(self):return self.map_text
    def resolve(self,path):return a.LOADER if path=='/lib64/ld-linux-x86-64.so.2' else path
    def environment(self):return self.env
    def running_executable(self):return self.current_exe
    def module(self,name):return self.registry.get(name)
    def modules(self):return tuple(self.registry.values())
    def builtin_names(self):return self.builtins

def reference(path,raw):return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def fixture():
    paths=[a.EXE,a.LIBC,a.LOADER]+['/usr/lib/x86_64-linux-gnu/'+name for name in
        ('libdl.so.2','libm.so.6','libpthread.so.0','librt.so.1','libutil.so.1')]
    files={}
    for path in paths:
        needed=([Path(p).name for p in paths[1:] if p!=a.LOADER] if path==a.EXE else
                [] if path==a.LOADER else [Path(a.LOADER).name] if path==a.LIBC else [Path(a.LIBC).name])
        files[path]=elf(None if path==a.EXE else Path(path).name,needed,
                        interpreter=path==a.EXE,ffi=path==a.EXE)
    native={p:reference(p,files[p]) for p in paths}
    source_paths=[a.gate.HISTORY_PATH,str(Path(a.gate.__file__).resolve()),
        str(Path(a.gate.backend.__file__).resolve()),str(Path(a.gate.backend.logic.__file__).resolve()),
        a.gate.backend.CTYPES_ROOT+'/__init__.py',a.gate.backend.CTYPES_ROOT+'/_endian.py']
    for p in source_paths:files[p]=b'VALUE = 7\n'
    pins={p:hashlib.sha256(files[p]).hexdigest() for p in source_paths}
    class Guard:
        def __init__(self):
            self.binding=dict(lock_identity=a.gate.ORIGINAL_LOCK,retained_fd=9,outer_owner={'pid':101})
            self.deadline=900.;self.calls=0
        def __call__(self):self.calls+=1
    guard=Guard();bootstrap=dict(schema='minicpm5_sdk_mapping_history_bootstrap_v1',binding=guard.binding,
        outer_deadline=guard.deadline,limits=a.gate.backend.logic.LIMITS,
        artifact_limits=a.gate.backend.logic.CAPS,source_sha256=pins)
    decision=dict(schema='minicpm5_sdk_mapping_native_source_review_v1',reviewer='/root',accepted=True,
        origin_evidence=a.ORIGIN_REF,source_map_sha256=a.gate.digest(pins),native_files=native)
    files['/mock/authority.json']=json.dumps(decision).encode();authority=reference('/mock/authority.json',files['/mock/authority.json'])
    ports=MemoryPorts(files)
    for i,path in enumerate(paths):
        start=0x100000+i*0x10000;inode=ports.identities[path].st_ino
        ports.map_text+='%x-%x r-xp 00000000 01:02 %d %s\n' % (start,start+8192,inode,path)
    reader=a.StableReader(guard,ports);auth=a.NativeAuthenticator(guard,reader=reader)
    origins={'exact_original_executable_and_pure_source_rows':{}}
    return NS(auth=auth,reader=reader,ports=ports,guard=guard,bootstrap=bootstrap,authority=authority,
        native=native,origins=origins,paths=paths,pins=pins)

def authenticate(f):
    with patch.object(a,'saved_origins',return_value=(f.origins,f.native)):
        return f.auth.authenticate_bootstrap(f.bootstrap,f.authority,f.guard)

def loaded_fixture(f):
    for name in ('_ctypes','_struct'):
        spec=a.machinery.ModuleSpec(name,a.machinery.BuiltinImporter,origin='built-in')
        f.ports.registry[name]=NS(__spec__=spec)
    loader=a.AuthenticatedSourceLoader('ctypes',a.gate.backend.CTYPES_ROOT+'/__init__.py',f.auth)
    ctypes=NS(__file__=a.gate.backend.CTYPES_ROOT+'/__init__.py',__loader__=loader,
              c_void_p=object(),cast=lambda fn,kind:NS(value=0x110100))
    libc=NS(_name=a.LIBC,ptrace=object());f.ports.registry['ctypes']=ctypes
    return ctypes,libc

class ClosureTests(unittest.TestCase):
    def test_positive_embedded_ffi_function_proof(self):
        proof=c.ffi_definitions(elf(ffi=True));self.assertTrue(proof['positive_embedded_ffi_definitions'])
        self.assertEqual(set(proof['defined_functions']),c.FFI_FUNCTIONS)
    def test_absent_libffi_without_positive_symbols_is_rejected(self):
        raw=elf();self.assertEqual(c.parse_elf64(raw)['needed'],[])
        with self.assertRaises(ValueError):c.ffi_definitions(raw)
    def test_undefined_ffi_or_zero_size_or_nonfunction_rejected(self):
        for offset,fmt,value in [(0x1600+24+6,'<H',0),(0x1600+24+16,'<Q',0),(0x1600+24+4,'<B',0x11)]:
            raw=bytearray(elf(ffi=True));struct.pack_into(fmt,raw,offset,value)
            with self.assertRaises(ValueError):c.ffi_definitions(bytes(raw))
    def test_exact_actual_map_selection_closes_all_named_dependencies(self):
        f=fixture();authenticate(f)
        self.assertEqual(set(f.auth.mapping['paths']),set(f.native));self.assertTrue(f.auth.mapping['actual_original_path_selection'])
    def test_missing_provider_foreign_mapping_or_wrong_inode_rejected(self):
        for kind in ('missing','foreign','inode'):
            f=fixture()
            if kind=='missing':f.ports.map_text='\n'.join(f.ports.map_text.splitlines()[:-1])+'\n'
            elif kind=='foreign':f.ports.map_text+='900000-901000 r-xp 00000000 01:02 999 /foreign/libffi.so.8\n'
            else:f.ports.map_text=f.ports.map_text.replace('01:02 100 ','01:02 999 ')
            with self.assertRaises(Exception):authenticate(f)
            self.assertIsNone(f.auth.native_receipt)
    def test_foreign_anonymous_execution_rejected(self):
        f=fixture();f.ports.map_text+='900000-901000 rwxp 00000000 00:00 0\n'
        with self.assertRaises(Exception):authenticate(f)
    def test_original_interpreter_alias_must_resolve_to_authenticated_loader(self):
        f=fixture();f.ports.resolve=lambda path:'/wrong/loader'
        with self.assertRaises(Exception):authenticate(f)

class ReaderTests(unittest.TestCase):
    def test_full_eof_hash_and_current_hardlink_identity(self):
        raw=b'complete synthetic payload';p='/synthetic/native';ports=MemoryPorts({p:raw});ports.identities[p].st_nlink=2
        reader=a.StableReader(lambda:None,ports);value,row=reader.read(reference(p,raw))
        self.assertEqual(value,raw);self.assertEqual(row['identity']['st_nlink'],2);self.assertEqual(len(ports.closed),1)
    def test_wrong_hash_short_read_growth_and_identity_change_rejected(self):
        for kind in ('hash','short','growth','identity'):
            p='/synthetic/native';raw=b'abcdef';ports=MemoryPorts({p:raw});ref=reference(p,raw)
            if kind=='hash':ref['sha256']='0'*64
            elif kind=='short':ports.files[p]=b'abc'
            elif kind=='growth':ports.files[p]=b'abcdefg'
            else:
                original=ports.fstat;count=[0]
                def changed(fd):
                    count[0]+=1;row=original(fd)
                    if count[0]>1:row.st_mtime_ns+=1
                    return row
                ports.fstat=changed
            with self.assertRaises(Exception):a.StableReader(lambda:None,ports).read(ref)
            self.assertEqual(len(ports.closed),1)
    def test_saved_json_provenance_replay_reads_no_native_or_archive(self):
        class JSONOnly(a.FilePorts):
            def __init__(self):self.paths=[]
            def open(self,path):
                assert path.endswith(('.json','.json.gz')),path
                self.paths.append(path);return super().open(path)
            def maps(self):raise AssertionError('No real process metadata')
        ports=JSONOnly();reader=a.StableReader(lambda:None,ports)
        evidence,native=a.saved_origins(reader)
        self.assertEqual(len(native),8);self.assertEqual(sum(x['bytes'] for x in native.values()),34495336)
        self.assertTrue(all(path.endswith(('.json','.json.gz')) for path in ports.paths))

class AuthenticatorTests(unittest.TestCase):
    def test_concrete_authenticator_reconstructs_receipt_from_bytes_maps_symbols(self):
        f=fixture();receipt=authenticate(f)
        a.gate.validate_bootstrap_receipt(receipt,f.bootstrap,f.authority)
        self.assertTrue(receipt['native_authority']['dependency_closure_verified'])
        self.assertTrue(f.auth.ffi['positive_embedded_ffi_definitions'])
        self.assertEqual(set(f.auth.native_pins),set(f.native))
        self.assertFalse(receipt['native_authority']['evidence']['original_nine_native_authenticated_here'])
    def test_loader_environment_or_interpreter_mismatch_never_authenticates(self):
        for kind in ('preload','exe','builtins'):
            f=fixture()
            if kind=='preload':f.ports.env['LD_PRELOAD']='/unreviewed.so'
            elif kind=='exe':f.ports.current_exe='/wrong/python'
            else:f.ports.builtins=('_struct',)
            with self.assertRaises(Exception):authenticate(f)
            self.assertIsNone(f.auth.native_receipt)
    def test_wrong_binding_or_deadline_rejected_before_native_reads(self):
        for kind in ('binding','deadline'):
            f=fixture();f.bootstrap=copy.deepcopy(f.bootstrap)
            if kind=='binding':f.bootstrap['binding']['retained_fd']=17
            else:f.bootstrap['outer_deadline']+=900
            with self.assertRaises(Exception):authenticate(f)
            self.assertEqual(f.ports.read_paths,[])
    def test_byte_auth_failure_does_not_manufacture_authority(self):
        f=fixture();f.ports.files[a.EXE]=bytes(len(f.ports.files[a.EXE]))
        with self.assertRaises(Exception):authenticate(f)
        self.assertIsNone(f.auth.native_receipt)
        with self.assertRaises(Exception):f.auth.load_ctypes()
    def test_no_reauthentication_after_failed_or_completed_bootstrap(self):
        f=fixture();authenticate(f)
        with self.assertRaises(Exception):authenticate(f)
    def test_explicit_verified_source_loader_never_reads_cached_bytecode(self):
        f=fixture();authenticate(f);p=a.gate.HISTORY_PATH
        loader=a.AuthenticatedSourceLoader('synthetic',p,f.auth)
        with patch.object(loader,'get_data',side_effect=AssertionError('No pyc or alternate reader')):
            code=loader.get_code('synthetic')
        namespace={};exec(code,namespace)
        self.assertEqual(namespace['VALUE'],7);self.assertIn(p,f.auth.source_executed)
    def test_unpinned_historical_source_is_rejected_by_finder(self):
        f=fixture();authenticate(f);path=str(ROOT/'unreviewed.py')
        spec=a.machinery.ModuleSpec('unreviewed',a.machinery.SourceFileLoader('unreviewed',path),origin=path)
        with patch.object(a.machinery.PathFinder,'find_spec',return_value=spec):
            with self.assertRaises(ValueError):a.SourceFinder(f.auth).find_spec('unreviewed')
    def test_actual_builtin_module_metadata_and_libc_pointer_are_required(self):
        f=fixture();authenticate(f);ctypes,libc=loaded_fixture(f)
        self.assertTrue(f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard))
        ctypes.cast=lambda *args:NS(value=0x100100)
        with self.assertRaises(Exception):f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
    def test_extension_ctypes_provider_is_rejected(self):
        f=fixture();authenticate(f);ctypes,libc=loaded_fixture(f)
        f.ports.registry['_ctypes'].__spec__.origin='/unreviewed/_ctypes.so'
        with self.assertRaises(Exception):f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
    def test_history_verifier_and_backend_revalidation_preserve_entire_map(self):
        f=fixture();authenticate(f);ctypes,libc=loaded_fixture(f)
        f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
        original=NS(__file__=a.gate.HISTORY_PATH);f.auth.source_executed.add(original.__file__)
        f.ports.registry[a.gate.HISTORY_NAME]=original
        self.assertTrue(f.auth.verify_loaded_history(original,f.pins,ctypes,f.guard))
        self.assertEqual(f.auth.authenticate(f.authority,f.guard),f.auth.native_receipt)
        f.ports.registry['unreviewed']=NS(__file__=str(ROOT/'unreviewed.py'))
        with self.assertRaises(Exception):f.auth.verify_loaded_history(original,f.pins,ctypes,f.guard)
    def test_default_native_import_remains_blocked_in_source_tests(self):
        f=fixture();authenticate(f)
        def refused(name):
            self.assertEqual(name,'ctypes');self.assertIsInstance(sys.meta_path[0],a.SourceFinder)
            raise AssertionError('No real import attempted in source test')
        before=tuple(sys.meta_path)
        with patch.object(a.importlib,'import_module',side_effect=refused):
            with self.assertRaises(AssertionError):f.auth.load_ctypes()
        self.assertEqual(tuple(sys.meta_path),before);self.assertNotIn('ctypes',sys.modules)
    def test_authenticator_source_never_acquires_locks_runs_commands_or_writes(self):
        tree=ast.parse(Path(a.__file__).read_text());attrs=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):attrs.append(node.func.attr)
        self.assertFalse(set(attrs)&{'Popen','flock','setrlimit','setitimer','alarm','write','save','system'})


    def test_backend_revalidation_failure_permanently_latches_original_error(self):
        f=fixture();authenticate(f);ctypes,libc=loaded_fixture(f)
        f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
        original=f.ports.identities[a.LIBC].st_mtime_ns;f.ports.identities[a.LIBC].st_mtime_ns+=1
        with self.assertRaises(Exception) as first:f.auth.authenticate(f.authority,f.guard)
        self.assertIs(f.auth.failure,first.exception)
        f.ports.identities[a.LIBC].st_mtime_ns=original
        with self.assertRaises(Exception) as second:f.auth.authenticate(f.authority,f.guard)
        self.assertIs(second.exception,first.exception)

    def test_loaded_provider_failure_cannot_recover_after_pointer_restored(self):
        f=fixture();authenticate(f);ctypes,libc=loaded_fixture(f);good=ctypes.cast
        ctypes.cast=lambda *args:NS(value=0)
        with self.assertRaises(Exception) as first:f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
        ctypes.cast=good
        with self.assertRaises(Exception) as second:f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
        self.assertIs(f.auth.failure,first.exception);self.assertIs(second.exception,first.exception)

    def test_history_failure_cannot_recover_after_module_removed(self):
        f=fixture();authenticate(f);ctypes,libc=loaded_fixture(f)
        f.auth.verify_loaded_native(ctypes,libc,f.auth.native_receipt,f.guard)
        original=NS(__file__=a.gate.HISTORY_PATH);f.auth.source_executed.add(original.__file__)
        f.ports.registry['wrong']=NS(__file__=str(ROOT/'wrong.py'))
        with self.assertRaises(Exception) as first:f.auth.verify_loaded_history(original,f.pins,ctypes,f.guard)
        f.ports.registry.pop('wrong')
        with self.assertRaises(Exception) as second:f.auth.verify_loaded_history(original,f.pins,ctypes,f.guard)
        self.assertIs(second.exception,first.exception)

    def test_source_import_failure_is_permanent(self):
        f=fixture();authenticate(f);error=ImportError('synthetic import failure')
        with patch.object(a.importlib,'import_module',side_effect=error):
            with self.assertRaises(ImportError):f.auth.load_ctypes()
        with patch.object(a.importlib,'import_module',side_effect=AssertionError('No retry')) as loader:
            with self.assertRaises(ImportError) as second:f.auth.load_ctypes()
            loader.assert_not_called()
        self.assertIs(f.auth.failure,error);self.assertIs(second.exception,error)


if __name__=='__main__':
    suites=[unittest.defaultTestLoader.loadTestsFromTestCase(x) for x in (ClosureTests,ReaderTests,AuthenticatorTests)]
    result=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(suites))
    assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
    assert 'minicpm5_python_trace_lifecycle_v4' not in sys.modules
    print('Synthetic ELF/native data and fake process ports; saved provenance reads restricted to JSON/JSON.gz.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
