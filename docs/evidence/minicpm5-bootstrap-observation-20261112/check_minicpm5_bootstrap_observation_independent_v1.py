"""Independent child-success and saved-history mocks; no native or proc reads."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from research import minicpm5_bootstrap_observation_v1 as core
from research import minicpm5_sdk_mapping_native_closure_v1 as closure
from research import run_minicpm5_bootstrap_observation_v1 as runner

Q = core.Q


def child_world(role, *, bad_provider=False, wrong_limits=False):
    events, files, descriptors = [], {}, {}
    next_fd = [100]
    python, executable = '/mock/'+role+'/python', '/mock/'+role+'/executable'
    pure_path, extension = '/mock/'+role+'/ctypes/__init__.py', '/mock/'+role+'/_ctypes.so'
    parent = dict(pid=123, start_ticks='100', ppid=12, argv=['/mock/parent'], cgroup='0::/mock')
    pr = dict(path='/mock/plan.json', sha256='a'*64)
    ar = dict(path='/mock/admission.json', sha256='b'*64)
    rr = dict(path='/mock/review.json', sha256='c'*64)
    command = core.argv_for(role, {role:dict(python=python)}, 'literal', pr, ar, 12, 1000.)
    template = list(command); template[13] = 'admission:self'
    pins = {str(Q/name):'d'*64 for name in (core.RUNNER,core.CHILD,core.HELPER,
        'minicpm5_bootstrap_observation_v1.py','minicpm5_sdk_mapping_outer_source_entry_v4.py',
        'minicpm5_sdk_mapping_worker_source_entry_v4.py')}
    plan = dict(schema='minicpm5_bootstrap_observation_plan_v1',stage='inventory',runtime_bound=True,
        automatic_successor_launch=False,role_order=list(core.ROLES),candidate_inputs=core.CANDIDATES,
        limits=core.LIMITS,artifact_limits=core.CAPS,budget=core.BUDGET,source_sha256=pins,
        mini_scientific_constraints=dict(native_checks=9,scientific_descriptor_rows=1284,absolute_wall_seconds=900,
            address_space_bytes=2147483648,single_file_bytes=2097152),
        output_root='/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-mock-independent',
        complete_current_serial_catalog={},root_handoff={},predecessor_exit={},lock_policy={},
        observation_only=True,accepted_native_authority_created=False)
    review = dict(schema='minicpm5_bootstrap_observation_review_v1',reviewer='/root',passed=True,
                  blockers=[],plan=pr,observation_only=True,accepted_native_authority_created=False)
    for key in ('source_sha256','candidate_inputs','limits','artifact_limits','budget','role_order',
                'complete_current_serial_catalog','root_handoff','predecessor_exit','lock_policy'):
        review[key] = copy.deepcopy(plan[key])
    content_ref = dict(path='/mock/content.json',sha256='e'*64)
    admission = dict(schema='minicpm5_bootstrap_observation_admission_v1',role=role,plan=pr,review=rr,
        deadline=1000.,started_monotonic=100.,literal_sha256=command[-1],command=template,
        parent_owner=parent,accepted_native_authority_created=False,content=content_ref,
        output_root=plan['output_root']+'/'+role,lock_identity=dict(device=66306,inode=8294421),retained_fd=42)
    table = {role:dict(python=python,executable=executable,ctypes_extension=extension,
        pure_source_candidates={pure_path:dict(sha256=hashlib.sha256(b'pass\n').hexdigest())},
        native_candidates={},ffi_candidate='/mock/ffi.so')}
    content = dict(current_candidate_bytes_authenticated=True,accepted_native_authority_created=False,
                   files={'/mock/ffi.so':dict(ffi=None,ffi_failure={'message':'mock unknown FFI'})})
    docs = {pr['path']:plan,ar['path']:admission,rr['path']:review,
            core.CANDIDATES['path']:table,content_ref['path']:content}
    class SourceLoader:
        def __init__(self,name,path): self.name,self.path=name,path
    class ExtensionLoader: pass
    def find_spec(name,*_):
        if name == 'ctypes': return NS(origin=pure_path,loader=SourceLoader(name,pure_path))
        if name == '_ctypes': return NS(origin='/wrong/provider.so' if bad_provider else extension,loader=ExtensionLoader())
        return None
    external = NS(SourceFileLoader=SourceLoader,ExtensionFileLoader=ExtensionLoader,PathFinder=NS(find_spec=find_spec))
    runtime = NS(executable=python,argv=['-c',*command[8:]],orig_argv=command,
        flags=NS(isolated=1,no_site=1,no_user_site=1,ignore_environment=1,optimize=0),
        dont_write_bytecode=True,pycache_prefix='/dev/null',path=[],meta_path=[],
        modules={'_frozen_importlib_external':external},version_info=(3,10) if role=='sdk310' else (3,12),
        builtin_module_names=('_ctypes',) if role=='outer312' and not bad_provider else ())
    entries=[]
    class Entry:
        def __init__(self): self.sources={};self.source_bytes=0;entries.append(self)
        def check(self): pass
        def prefix(self): return ()
        def install(self,_): self.finder=NS(find_spec=lambda *args:None);runtime.meta_path[:]=[self.finder]
        def read_source(self,path,sha,*args):
            if path==pure_path:return b'pass\n',('mock-source-identity',)
            return core.encoded(docs[path]),('mock-json-identity',)
    def maps_bytes():
        stage='post' if 'libc' in events else 'pre' if 'formal' in events else 'earliest'
        return ('1000-2000 r-xp 00000000 01:02 123 /unknown-'+stage+'.so\n').encode()
    def opened(path,*args):
        next_fd[0]+=1;fd=next_fd[0]
        if path=='/proc/self/maps':
            descriptors[fd]=[path,0,maps_bytes()];events.append('maps')
        else:
            descriptors[fd]=[path,0,b'']
            if path.endswith(('.maps','.json')):files[path]=bytearray()
        return fd
    def read(fd,count):
        if fd==12:events.append('ACK');return b'R'
        path,at,raw=descriptors[fd];descriptors[fd][1]+=count;return raw[at:at+count]
    def write(fd,raw):
        if fd==2:events.append(('stderr',raw));return len(raw)
        files[descriptors[fd][0]].extend(raw);return len(raw)
    ports=NS(open=opened,read=read,write=write,close=lambda _:None,fsync=lambda _:None,
        getppid=lambda:123,getpid=lambda:222,readlink=lambda _:executable,lstat=lambda _:None,stat=lambda _:None,
        sched_getaffinity=lambda _:{6},getpriority=lambda *args:19,PRIO_PROCESS=0,
        O_RDONLY=0,O_CLOEXEC=0,O_WRONLY=0,O_CREAT=0,O_EXCL=0,O_NOFOLLOW=0,O_DIRECTORY=0)
    def prove(owner,lock,fd,**callbacks):
        assert owner==parent and lock==admission['lock_identity'] and fd==42
        events.append('lock-proof')
    def formal(name):
        assert name==core.ENTRIES[role];events.append('formal');return NS()
    def getrlimit(kind):
        if kind==0:return (0,0) if wrong_limits else (2147483648,)*2
        return (600,605) if kind==1 else (2097152,)*2
    research=NS(minicpm5_bootstrap_observation_v1=core,minicpm5_sdk_mapping_native_closure_v1=closure,
                descriptor_serial_lock_proof_v1=NS(prove=prove))
    modules={'sys':runtime,'posix':ports,'time':NS(monotonic=lambda:100.),'json':json,'research':research,
             'importlib':NS(import_module=formal),'minicpm5_layer0_execution_protocol_v4':NS(proc=lambda _:copy.deepcopy(parent)),
             'resource':NS(getrlimit=getrlimit,RLIMIT_AS=0,RLIMIT_CPU=1,RLIMIT_FSIZE=2),'pathlib':NS(Path=Path)}
    def importing(name,*args):
        if name=='ctypes':
            assert entries[0].allow_ctypes
            assert not entries[0].native_import_allowed('torch')
            spec=runtime.meta_path[0].find_spec('ctypes')
            runtime.meta_path[0].find_spec('_ctypes')
            spec.loader.get_code('ctypes')
            def cdll(path,**kwargs):
                assert path=='/usr/lib/x86_64-linux-gnu/libc.so.6';events.append('libc');return NS(_name=path)
            result=NS(__file__=pure_path,__loader__=spec.loader,CDLL=cdll,__spec__=NS(origin=pure_path,loader=spec.loader))
            runtime.modules['ctypes']=result
            runtime.modules['_ctypes']=NS(__spec__=NS(origin='built-in' if role=='outer312' else extension,loader=ExtensionLoader()))
            return result
        if name not in modules:raise AssertionError('Unexpected import in complete mock: '+name)
        return modules[name]
    custom=dict(vars(builtins));custom['__import__']=importing
    namespace=dict(__name__='independent_mock_child',__builtins__=custom,
        helpers=dict(require=core.require,digest=lambda raw,*args:hashlib.sha256(raw).hexdigest(),Entry=Entry,
                     ROOT=str(core.ROOT),SOURCE_TOTAL_CAP=16*2**20))
    exec(compile((Q/core.CHILD).read_text(),str(Q/core.CHILD),'exec'),namespace)
    return namespace['main'],files,events,admission


class IndependentReview(unittest.TestCase):
    def test_both_success_paths_retain_all_three_maps_and_unknown_gaps(self):
        for role in core.ROLES:
            main,files,events,admission=child_world(role)
            main();base=admission['output_root']
            result=json.loads(files[base+'/observation.json'])
            self.assertEqual(result['role'],role)
            self.assertEqual(set(result['maps']),{'earliest','pre_ctypes','post_ctypes'})
            for key,marker in [('earliest','earliest'),('pre_ctypes','pre'),('post_ctypes','post')]:
                raw=bytes(files[result['maps'][key]['path']])
                self.assertIn(('/unknown-'+marker+'.so').encode(),raw)
                self.assertEqual(result['maps'][key]['sha256'],hashlib.sha256(raw).hexdigest())
                self.assertEqual(result['map_summaries'][key]['gaps'][0]['reason'],'missing_saved_content_origin')
                self.assertFalse(result['map_summaries'][key]['accepted_native_authority_created'])
            self.assertTrue(result['complete'] and result['missing_content_origins'])
            self.assertFalse(result['accepted_native_authority_created'] or result['native_ABI_qualified'])
            self.assertIsNone(result['ffi_definitions'])
            self.assertIsNone(result['selected_stdlib_providers']['zlib']['origin'])
            self.assertTrue(all(row['pinned_source_receipt_present'] for row in result['known_pure_source_observations'].values()))
            self.assertLess(events.index('maps'),events.index('ACK'))
            self.assertLess(events.index('ACK'),events.index('formal'))
            self.assertLess(events.index('lock-proof'),events.index('libc'))

    def test_bad_sdk_or_outer_provider_keeps_failed_maps_without_promotion(self):
        for role in core.ROLES:
            main,files,events,admission=child_world(role,bad_provider=True)
            with self.assertRaises(ValueError):main()
            base=admission['output_root'];failure=json.loads(files[base+'/failure.json'])
            self.assertIn('failed_bootstrap',failure['maps'])
            self.assertNotIn('post_ctypes',failure['maps'])
            self.assertNotIn('libc',events)
            self.assertNotIn(base+'/observation.json',files)

    def test_wrong_inherited_resource_limit_prevents_ctypes(self):
        main,files,events,admission=child_world('sdk310',wrong_limits=True)
        with self.assertRaisesRegex(ValueError,'inherited original'):main()
        failure=json.loads(files[admission['output_root']+'/failure.json'])
        self.assertNotIn('pre_ctypes',failure['maps'])
        self.assertNotIn('libc',events)

    def test_saved_1455_history_rows_roundtrip_in_four_bounded_shards(self):
        path=Path('/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-replay-r1-20261112/owner.json')
        before=path.stat();raw=path.read_bytes();after=path.stat()
        self.assertEqual((before.st_ino,before.st_size,before.st_mtime_ns),(after.st_ino,after.st_size,after.st_mtime_ns))
        rows=json.loads(raw)['history'];self.assertEqual(len(rows),1455)
        saved={}
        def save(path,value,cap):
            encoded=(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode()
            self.assertLessEqual(len(encoded),cap);saved[str(path)]=(copy.deepcopy(value),len(encoded))
            return dict(path=str(path),bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
        reference=runner.save_history(Path('/mock/independent-history'),rows,save=save)
        manifest=saved[reference['path']][0];restored=[]
        self.assertEqual(len(manifest['parts']),4)
        for part in manifest['parts']:
            self.assertEqual(part['first_row'],len(restored))
            value,size=saved[part['reference']['path']]
            self.assertEqual(value['first_row'],len(restored));restored.extend(value['rows'])
            self.assertLessEqual(size,2**20)
        self.assertEqual(restored,rows)


if __name__=='__main__':unittest.main()
