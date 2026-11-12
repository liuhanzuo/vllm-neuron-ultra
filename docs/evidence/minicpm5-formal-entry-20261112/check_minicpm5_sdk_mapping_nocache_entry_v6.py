"""Stdlib-only entry checks: synthetic argv, files, hooks and loader ports.

No subprocess, real entry launch, proc/dev/ELF read, lock, signal/resource setter,
service, model, Torch, or ctypes operation occurs. The actual stage0 main is
reviewed through AST only; runtime methods use fake ports below.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class RejectNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('ctypes','_ctypes','torch','numpy','neuronx_distributed','neuronx_distributed_inference'):
            raise AssertionError('No native imports in source-only entry checks: '+name)


sys.meta_path.insert(0,RejectNative())
from research import minicpm5_sdk_mapping_nocache_entry_v6 as m
from research import minicpm5_sdk_mapping_nocache_launch_v1 as build

ENTRY=Path(m.__file__).read_bytes()
ENTRY_REF=dict(path=str(Path(m.__file__)),sha256=hashlib.sha256(ENTRY).hexdigest())
PLAN_PATH=str(ROOT/'research/minicpm5-future-source-entry-synthetic.json')
TARGET=str(ROOT/'research/minicpm5_nocache_future_source_fixture.py')
SOURCE=b'answer = 42\n'


def info(mode=0o100600,size=1,ino=17):
    return NS(st_dev=11,st_ino=ino,st_mode=mode,st_nlink=1,st_size=size,
        st_mtime_ns=23,st_ctime_ns=24,st_rdev=0)


class Ports:
    def __init__(self):
        self.files={TARGET:SOURCE}
        self.infos={}
        self.offset={}
        self.opened=[]
        self.closed=[]
        self.reads=[]
        self.time=100.0
        self.running=m.OUTER_EXE
        self.dev=info(0o040755,ino=31)
        self.null=info(0o020666,ino=32);self.null.st_rdev=259
        self.next_fd=20
    def now(self): return self.time
    def lstat(self,path):
        if path=='/dev': return self.dev
        if path=='/dev/null': return self.null
        if path in self.infos: return self.infos[path]
        if path in self.files: return info(size=len(self.files[path]))
        return info(0o040755)
    def open(self,path):
        self.next_fd+=1;self.opened.append(path);self.offset[self.next_fd]=[path,0]
        return self.next_fd
    def fstat(self,fd): return self.lstat(self.offset[fd][0])
    def read(self,fd,n):
        path,offset=self.offset[fd];self.reads.append(path)
        raw=self.files[path][offset:offset+n];self.offset[fd][1]+=len(raw)
        return raw
    def close(self,fd): self.closed.append(fd)
    def readlink(self,path):
        assert path=='/proc/self/exe'
        return self.running


class Runtime(NS):
    def addaudithook(self,hook): self.hooks.append(hook)
    def audit(self,event,*args):
        for hook in self.hooks:hook(event,args)


class Loader:
    def __init__(self,name,path): self.name,self.path=name,path


class External:
    SourceFileLoader=Loader
    def __init__(self):
        self.specs={}
        self.PathFinder=NS(find_spec=self.find_spec)
    def find_spec(self,name,path,target): return self.specs.get(name)


def make_plan(role='outer312'):
    python=m.OUTER_PYTHON if role=='outer312' else m.SDK_PYTHON
    return dict(schema='minicpm5_source_entry_plan_v1',role=role,python=python,
        executable=m.OUTER_EXE if role=='outer312' else '/synthetic/sdk310/python3.10',
        native_authority=dict(role=role,python=python,path='/synthetic/'+role+'-authority.json',
            sha256=('a' if role=='outer312' else 'b')*64),
        bootstrap_sources={TARGET:hashlib.sha256(SOURCE).hexdigest()},
        entry_sha256=ENTRY_REF['sha256'],entry_module='research.minicpm5_nocache_future_source_fixture',
        entry_path=TARGET,successor_input='/synthetic/original-input-ref.json')


def fixture(role='outer312',*,start=True,install=True,load=True,change=None):
    plan=make_plan(role)
    if change: change(plan)
    raw=build.encoded(plan)
    ref=dict(path=PLAN_PATH,sha256=hashlib.sha256(raw).hexdigest())
    argv=build.launch(ENTRY,ENTRY_REF,plan,ref,outer_deadline=None if role=='outer312' else 750.)
    ports=Ports();ports.files[PLAN_PATH]=raw;ports.running=plan['executable']
    runtime=Runtime(flags=NS(isolated=1,no_site=1,ignore_environment=1,no_user_site=1,optimize=0),
        dont_write_bytecode=True,pycache_prefix='/dev/null',executable=plan['python'],
        version_info=(3,12,16) if role=='outer312' else (3,10,12),
        argv=['-c',*argv[8:]],orig_argv=argv,
        builtin_module_names=('sys','posix','time'),
        modules={name:NS(__spec__=NS(origin='built-in')) for name in ('sys','posix','time')},
        hooks=[],meta_path=[object()])
    entry=m.Entry(runtime=runtime,ports=ports)
    external=External()
    if start: entry.startup(ENTRY.decode())
    if install: entry.install(external)
    if load: entry.load_plan(ref,json)
    return entry,ports,runtime,external,plan,ref


class Checks(unittest.TestCase):
    def test_sha256_standard_and_padding_boundaries(self):
        for n in (0,1,3,55,56,63,64,65,119,120,127,128,129,4095,4096,4097):
            raw=bytes(i%251 for i in range(n))
            self.assertEqual(m.digest(raw),hashlib.sha256(raw).hexdigest())
        self.assertEqual(m.digest(b'abc'),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_sha256_periodic_deadline_checks(self):
        calls=[]
        m.digest(b'x'*10000,lambda:calls.append(True))
        self.assertGreaterEqual(len(calls),4)
        def fail(): raise TimeoutError('fake deadline')
        with self.assertRaises(TimeoutError): m.digest(b'x',fail)

    def test_outer_literal_flags_and_starting_deadline(self):
        e,p,s,x,plan,ref=fixture()
        self.assertEqual(e.deadline,1000.)
        self.assertEqual(e.started,100.)
        self.assertEqual(s.orig_argv[7],ENTRY.decode())
        self.assertEqual(e.receipt()['native_authority_verified'],False)
        self.assertEqual(p.opened,[PLAN_PATH])

    def test_sdk_separate_python_and_existing_deadline(self):
        e,p,s,x,plan,ref=fixture('sdk310')
        self.assertEqual(e.deadline,750.)
        self.assertEqual(e.receipt()['role'],'sdk310')
        self.assertNotEqual(plan['executable'],m.OUTER_EXE)

    def test_no_construction_or_plan_operation_acquires_lock_or_changes_resources(self):
        e,p,s,x,plan,ref=fixture(start=False,install=False,load=False)
        self.assertEqual(p.opened,[])
        self.assertEqual(s.hooks,[])
        self.assertEqual(p.reads,[])
        self.assertEqual(e.state,'new')

    def test_each_required_isolation_flag_is_enforced(self):
        for flag in ('isolated','no_site','ignore_environment','no_user_site'):
            e,p,s,*_=fixture(start=False,install=False,load=False)
            setattr(s.flags,flag,0)
            with self.assertRaises(ValueError): e.startup(ENTRY.decode())
            with self.assertRaises(ValueError): e.check()

    def test_optimize_and_bytecode_flags_are_rejected(self):
        for field,value in [('optimize',1),('dont_write_bytecode',False),('pycache_prefix','/tmp/cache')]:
            e,p,s,*_=fixture(start=False,install=False,load=False)
            setattr(s.flags if field=='optimize' else s,field,value)
            with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_literal_drift_rejected_before_any_plan_file_read(self):
        e,p,s,*_=fixture(start=False,install=False,load=False)
        s.orig_argv[7]+='\n'
        with self.assertRaises(ValueError): e.startup(ENTRY.decode())
        self.assertEqual(p.opened,[])

    def test_wrong_literal_pin_rejected(self):
        e,p,s,*_=fixture(start=False,install=False,load=False)
        s.argv[6]='0'*64;s.orig_argv[-2]='0'*64
        with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_interpreter_role_swap_rejected(self):
        e,p,s,*_=fixture(start=False,install=False,load=False)
        s.executable=m.SDK_PYTHON;s.version_info=(3,10,12)
        with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_sdk_cannot_establish_new_900s_deadline(self):
        e,p,s,*_=fixture('sdk310',start=False,install=False,load=False)
        s.argv[-1]='new';s.orig_argv[-1]='new'
        with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_sdk_expired_and_infinite_deadlines_rejected(self):
        for value in ('99','100','1001','nan','inf'):
            e,p,s,*_=fixture('sdk310',start=False,install=False,load=False)
            s.argv[-1]=value;s.orig_argv[-1]=value
            with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_dev_null_cannot_be_a_directory_or_symlink_or_wrong_device(self):
        for fault in ('directory','symlink','device','dev_parent'):
            e,p,s,*_=fixture(start=False,install=False,load=False)
            if fault=='directory': p.null.st_mode=0o040755
            if fault=='symlink': p.null.st_mode=0o120777
            if fault=='device': p.null.st_rdev=0
            if fault=='dev_parent': p.dev.st_mode=0o120777
            with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_prefix_identity_drift_permanently_latches(self):
        e,p,s,*_=fixture()
        p.null.st_ino+=1
        with self.assertRaises(ValueError) as first:e.check()
        p.null.st_ino-=1
        with self.assertRaises(ValueError) as second:e.check()
        self.assertIs(first.exception,second.exception)

    def test_deadline_failure_cannot_resume_after_clock_recovery(self):
        e,p,*_=fixture()
        p.time=1001
        with self.assertRaises(ValueError) as first:e.check()
        p.time=101
        with self.assertRaises(ValueError) as second:e.check()
        self.assertIs(first.exception,second.exception)

    def test_no_premature_ctypes_history_or_model_import(self):
        for name in ('ctypes','_ctypes','torch','minicpm5_python_trace_lifecycle_v4'):
            e,p,s,*_=fixture(start=False,install=False,load=False)
            s.modules[name]=object()
            with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_initial_sourceless_module_rejected(self):
        e,p,s,*_=fixture(start=False,install=False,load=False)
        s.modules['untrusted']=NS(__file__='/synthetic/old.pyc')
        with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_entry_native_primitives_must_be_builtin(self):
        e,p,s,*_=fixture(start=False,install=False,load=False)
        s.modules['posix'].__spec__.origin='/synthetic/posix.so'
        with self.assertRaises(ValueError): e.startup(ENTRY.decode())

    def test_audit_rejects_both_bytecode_reads_and_writes(self):
        e,p,s,*_=fixture()
        for path in ('/synthetic/cache.pyc',b'/synthetic/cache.pyc','/synthetic/cache.pyo'):
            for mode in ('r','w',None):
                with self.assertRaises(FileNotFoundError): s.hooks[0]('open',(path,mode,0))
        s.hooks[0]('open',('/synthetic/current.py','r',0))
        s.hooks[0]('open',(44,'r',0))

    def test_silently_rejected_audit_hook_cannot_proceed(self):
        e,p,s,x,*_=fixture(install=False,load=False)
        s.addaudithook=lambda hook: None
        with self.assertRaises(ValueError):e.install(x)

    def test_finder_rejects_direct_sourceless_provider(self):
        e,p,s,x,*_=fixture()
        x.specs['bad']=NS(origin='/synthetic/direct.pyc',loader=Loader('bad','/synthetic/direct.pyc'))
        with self.assertRaises(ValueError):e.finder.find_spec('bad')
        self.assertIsNotNone(e.failure)

    def test_future_native_import_still_requires_separate_gate_integration(self):
        e,p,s,x,*_=fixture()
        with self.assertRaises(ValueError):e.finder.find_spec('ctypes')
        self.assertFalse(e.native_authority_verified)

    def test_source_loader_executes_pinned_bytes_and_never_pyc(self):
        e,p,s,x,plan,ref=fixture()
        name=plan['entry_module'];x.specs[name]=NS(origin=TARGET,loader=Loader(name,TARGET))
        spec=e.finder.find_spec(name);code=spec.loader.get_code(name)
        namespace={};exec(code,namespace)
        self.assertEqual(namespace['answer'],42)
        self.assertEqual(e.sources[TARGET]['sha256'],hashlib.sha256(SOURCE).hexdigest())
        self.assertEqual(p.opened,[PLAN_PATH,TARGET])
        self.assertTrue(all(not name.endswith('.pyc') for name in p.reads))
        self.assertEqual(len(p.closed),2)

    def test_source_change_fails_before_execution_and_latches(self):
        e,p,s,x,plan,ref=fixture()
        name=plan['entry_module'];x.specs[name]=NS(origin=TARGET,loader=Loader(name,TARGET))
        p.files[TARGET]=b'answer = 99\n'
        loader=e.finder.find_spec(name).loader
        with self.assertRaises(ValueError) as first:loader.get_code(name)
        p.files[TARGET]=SOURCE
        with self.assertRaises(ValueError) as second:loader.get_code(name)
        self.assertIs(first.exception,second.exception)
        self.assertNotIn(TARGET,e.sources)

    def test_unpinned_local_source_rejected(self):
        e,p,s,x,*_=fixture()
        path=str(ROOT/'unreviewed.py');x.specs['unreviewed']=NS(origin=path,loader=Loader('unreviewed',path))
        with self.assertRaises(ValueError):e.finder.find_spec('unreviewed')

    def test_namespace_spec_and_nonlocal_provider_preserved(self):
        e,p,s,x,*_=fixture()
        x.specs['namespace']=NS(origin=None,loader=None)
        x.specs['stdlib']=NS(origin='/synthetic/stdlib/current.py',loader=Loader('stdlib','/synthetic/stdlib/current.py'))
        self.assertIsNone(e.finder.find_spec('namespace'))
        self.assertIsNone(e.finder.find_spec('stdlib'))

    def test_entry_plan_hash_mismatch_and_duplicate_keys(self):
        e,p,s,x,plan,ref=fixture(load=False)
        bad=dict(ref);bad['sha256']='0'*64
        with self.assertRaises(ValueError):e.load_plan(bad,json)
        e,p,s,x,plan,ref=fixture(load=False)
        raw=b'{"role":"outer312","role":"sdk310"}'
        p.files[PLAN_PATH]=raw;ref['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(ValueError,'Unique'):e.load_plan(ref,json)

    def test_plan_cannot_substitute_actual_executable(self):
        e,p,s,x,plan,ref=fixture(load=False)
        p.running='/synthetic/substitute-python'
        with self.assertRaises(ValueError):e.load_plan(ref,json)

    def test_worker_cannot_use_outer_native_authority(self):
        e,p,s,x,plan,ref=fixture('sdk310',load=False)
        plan['native_authority']=make_plan()['native_authority']
        raw=build.encoded(plan);p.files[PLAN_PATH]=raw;ref['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):e.load_plan(ref,json)

    def test_stable_source_identity_and_symlinks(self):
        e,p,s,x,plan,ref=fixture(load=False)
        p.infos[PLAN_PATH]=info(0o120777,len(p.files[PLAN_PATH]))
        with self.assertRaises(ValueError):e.load_plan(ref,json)
        self.assertEqual(p.opened,[])

    def test_finder_ownership_drift_latches(self):
        e,p,s,*_=fixture()
        s.meta_path.insert(0,object())
        with self.assertRaises(ValueError):e.check()

    def test_launch_constructor_rejects_source_hash_drift(self):
        plan=make_plan();ref=dict(path=PLAN_PATH,sha256=hashlib.sha256(build.encoded(plan)).hexdigest())
        with self.assertRaises(ValueError):build.launch(ENTRY+b'\n',ENTRY_REF,plan,ref)

    def test_launch_constructor_requires_worker_existing_deadline(self):
        plan=make_plan('sdk310');ref=dict(path=PLAN_PATH,sha256=hashlib.sha256(build.encoded(plan)).hexdigest())
        for bad in (None,True,0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):build.launch(ENTRY,ENTRY_REF,plan,ref,outer_deadline=bad)

    def test_separate_authority_pair_and_reuse_rejection(self):
        a,b=make_plan(),make_plan('sdk310')
        self.assertTrue(build.separate_authorities(a,b))
        b['native_authority']['sha256']=a['native_authority']['sha256']
        with self.assertRaises(ValueError):build.separate_authorities(a,b)

    def test_ast_entry_top_level_imports_only_builtins_and_real_main_uses_literal(self):
        tree=ast.parse(ENTRY)
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertEqual([n.names[0].name for n in imports],['sys','posix','time'])
        calls={getattr(n.func,'attr','') for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'setrlimit','setitimer','Popen','flock','fork','execve','system'})
        self.assertNotIn('ctypes',sys.modules)
        self.assertNotIn('_ctypes',sys.modules)


if __name__=='__main__':
    unittest.main()
