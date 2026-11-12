"""R2 source/saved-JSON and complete child mocks; no native/archive/process work."""
import sys
class NoNative:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in {'ctypes','_ctypes','torch','numpy','vllm','transformers','neuronxcc','nki'}:
            raise AssertionError('Forbidden native import: '+name)
sys.meta_path.insert(0,NoNative())
import ast
import builtins
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'research')]
from research import minicpm5_bootstrap_observation_v2 as core
from research import minicpm5_sdk_mapping_native_closure_v2 as closure
from research import run_minicpm5_bootstrap_observation_v2 as runner

from research import minicpm5_bootstrap_types_source_v1 as types_tracker
from research import prepare_minicpm5_bootstrap_observation_v2 as builder
from research import minicpm5_bootstrap_observation_v1 as oldcore
import dory_serial_stage_contract_v2 as c

Q = core.Q


def child_world(role, *, bad_provider=False, wrong_limits=False, preexisting=False, failed_types=False):
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
    pins = {str(Q/name):'d'*64 for name in (core.RUNNER,core.CHILD,core.HELPER,core.TYPES,
        'minicpm5_sdk_mapping_native_closure_v2.py','minicpm5_bootstrap_observation_v2.py','minicpm5_sdk_mapping_outer_source_entry_v4.py',
        'minicpm5_sdk_mapping_worker_source_entry_v4.py')}
    plan = dict(schema='minicpm5_bootstrap_observation_plan_v2',stage='inventory',runtime_bound=True,
        automatic_successor_launch=False,role_order=list(core.ROLES),candidate_inputs=core.CANDIDATES,
        limits=core.LIMITS,artifact_limits=core.CAPS,budget=core.BUDGET,source_sha256=pins,
        mini_scientific_constraints=dict(native_checks=9,scientific_descriptor_rows=1284,absolute_wall_seconds=900,
            address_space_bytes=2147483648,single_file_bytes=2097152),
        output_root='/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-mock-independent',
        complete_current_serial_catalog={},root_handoff={},predecessor_exit={},lock_policy={},
        observation_only=True,accepted_native_authority_created=False,
        source_component_reviews=core.COMPONENT_REVIEWS,supplemental_origin_receipts=core.SUPPLEMENTAL_RECEIPTS)
    review = dict(schema='minicpm5_bootstrap_observation_review_v2',reviewer='/root',passed=True,
                  blockers=[],plan=pr,observation_only=True,accepted_native_authority_created=False)
    for key in ('source_sha256','candidate_inputs','limits','artifact_limits','budget','role_order',
                'complete_current_serial_catalog','root_handoff','predecessor_exit','lock_policy',
                'source_component_reviews','supplemental_origin_receipts'):
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
    if role=='outer312':table[role]['pure_source_candidates'][types_tracker.PIN['path']]=dict(types_tracker.PIN)
    class SourceLoader:
        def __init__(self,name,path): self.name,self.path=name,path
        def exec_module(self,module):exec(self.get_code(module.__name__),module.__dict__)
    class ExtensionLoader: pass
    def find_spec(name,*_):
        if name=='types':return NS(name='types',origin=types_tracker.PIN['path'],loader=SourceLoader('types',types_tracker.PIN['path']))
        if name == 'ctypes': return NS(origin=pure_path,loader=SourceLoader(name,pure_path))
        if name == '_ctypes': return NS(origin='/wrong/provider.so' if bad_provider else extension,loader=ExtensionLoader())
        return None
    external = NS(SourceFileLoader=SourceLoader,ExtensionFileLoader=ExtensionLoader,PathFinder=NS(find_spec=find_spec))
    runtime = NS(executable=python,argv=['-c',*command[8:]],orig_argv=command,
        flags=NS(isolated=1,no_site=1,no_user_site=1,ignore_environment=1,optimize=0),
        dont_write_bytecode=True,pycache_prefix='/dev/null',path=[],meta_path=[],
        modules={'_frozen_importlib_external':external},version_info=(3,10) if role=='sdk310' else (3,12),
        builtin_module_names=('_ctypes',) if role=='outer312' and not bad_provider else ())
    def types_module(loader):return NS(__name__='types',__file__=types_tracker.PIN['path'],__loader__=loader,
        __spec__=NS(name='types',origin=types_tracker.PIN['path'],loader=loader))
    if role=='outer312' and preexisting:
        runtime.modules['types']=types_module(SourceLoader('types',types_tracker.PIN['path']))
    entries=[]
    class Entry:
        def __init__(self): self.sources={};self.source_bytes=0;entries.append(self)
        def check(self):
            assert runtime.meta_path[0] is self.finder
            events.append('entry-guard')
        def prefix(self): return ()
        def install(self,_): self.finder=NS(find_spec=lambda *args:None);runtime.meta_path[:]=[self.finder]
        def read_source(self,path,sha,*args):
            if path==types_tracker.PIN['path']:
                assert sha==types_tracker.PIN['sha256'] and args==(types_tracker.PIN['bytes'],)
                raw=("raise ValueError('synthetic types failed')\n" if failed_types else 'value = 7\n').encode()
                events.append('types-source-read');return raw+b'\n'*(types_tracker.PIN['bytes']-len(raw)),('mock-types-identity',)
            if path==pure_path:return b'pass\n',('mock-source-identity',)
            return core.encoded(docs[path]),('mock-json-identity',)
    def maps_bytes():
        stage='post' if 'libc' in events else 'pre' if 'formal' in events else 'earliest'
        return ('1000-2000 r-xp 00000000 01:02 123 /unknown-'+stage+'.so\n').encode()
    def opened(path,*args):
        next_fd[0]+=1;fd=next_fd[0]
        if path=='/proc/self/maps':
            descriptors[fd]=[path,0,maps_bytes()];events.append('maps')
            events.append(('types-present-at-maps','types' in runtime.modules))
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
    research=NS(minicpm5_bootstrap_observation_v2=core,minicpm5_sdk_mapping_native_closure_v2=closure,
                descriptor_serial_lock_proof_v1=NS(prove=prove))
    modules={'sys':runtime,'posix':ports,'time':NS(monotonic=lambda:100.),'json':json,'research':research,
             'importlib':NS(import_module=formal),'minicpm5_layer0_execution_protocol_v4':NS(proc=lambda _:copy.deepcopy(parent)),
             'resource':NS(getrlimit=getrlimit,RLIMIT_AS=0,RLIMIT_CPU=1,RLIMIT_FSIZE=2),'pathlib':NS(Path=Path)}
    def importing(name,*args):
        if name=='json':
            events.append('json-import')
            if role=='outer312' and 'types' not in runtime.modules:
                spec=runtime.meta_path[0].find_spec('types');module=types_module(spec.loader)
                runtime.modules['types']=module
                try:spec.loader.exec_module(module)
                except BaseException:
                    del runtime.modules['types'];raise
            return json
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
    namespace=dict(__name__='independent_mock_child',__builtins__=custom,types_helpers=vars(types_tracker),
        helpers=dict(require=core.require,digest=lambda raw,*args:hashlib.sha256(raw).hexdigest(),Entry=Entry,
                     ROOT=str(core.ROOT),SOURCE_TOTAL_CAP=16*2**20))
    exec(compile((Q/core.CHILD).read_text(),str(Q/core.CHILD),'exec'),namespace)
    return namespace['main'],files,events,admission


def ref(name):return core.ref2(c.reference(Q/name)[0])


def source_manifest():
    pins=copy.deepcopy(c.load(core.RETAINED_MANIFEST)['source_sha256'])
    ended=c.load(core.SUPPLEMENTAL_RECEIPTS['actual_exit'])
    pins.update(c.load(ended['plan'])['source_sha256'])
    correction=c.load(builder.RECORDER_SOURCE_REVIEW)
    pins[correction['successor']['path']]=correction['successor']['sha256']
    for name in builder.SUCCESSORS:pins[str(Q/name)]=hashlib.sha256((Q/name).read_bytes()).hexdigest()
    return dict(schema='minicpm5_bootstrap_observation_source_manifest_v2',source_sha256=pins,
        retained_manifest=core.RETAINED_MANIFEST,candidate_inputs=core.CANDIDATES,limits=core.LIMITS,
        artifact_limits=core.CAPS,source_component_reviews=core.COMPONENT_REVIEWS,
        supplemental_origin_receipts=core.SUPPLEMENTAL_RECEIPTS,recorder_source_review=builder.RECORDER_SOURCE_REVIEW)


class R2WiringTests(unittest.TestCase):
    def test_01_actual_saved_member_provenance_adds_exactly_two_rows(self):
        old=oldcore.candidates(c.load);new=core.candidates(c.load)
        self.assertEqual(new['outer312'],old['outer312'])
        for path,row in old['sdk310']['native_candidates'].items():self.assertEqual(new['sdk310']['native_candidates'][path],row)
        added=set(new['sdk310']['native_candidates'])-set(old['sdk310']['native_candidates'])
        self.assertEqual(added,{'/usr/lib/python3.10/lib-dynload/'+name+'.cpython-310-x86_64-linux-gnu.so'
                               for name in ('_contextvars','resource')})
        self.assertEqual(sum(new['sdk310']['native_candidates'][p]['bytes']for p in added),33960)
        self.assertEqual(new['distinct_native_candidates'],20)
        self.assertEqual(new['distinct_native_candidate_bytes'],46398112)
        self.assertFalse(new['complete_native_dependency_closure_established'])

    def test_02_wrong_result_hash_member_identity_or_actual_exit_not_promoted(self):
        ended=c.load(core.SUPPLEMENTAL_RECEIPTS['actual_exit'])
        result_ref=ended['result'];record_ref=core.SUPPLEMENTAL_RECEIPTS['recorder_actual_exit']
        mutations=[(result_ref,lambda x:x['comparisons'][0]['member'].update(sha256='0'*64)),
                   (result_ref,lambda x:x['comparisons'][0]['current_identity'].update(ino=1)),
                   (core.SUPPLEMENTAL_RECEIPTS['actual_exit'],lambda x:x['actual_tool'].update(exit_code=1)),
                   (record_ref,lambda x:x.update(owner_gone=False)),
                   (record_ref,lambda x:x.update(full_saved_supplemental_result_revalidated=False)),
                   (record_ref,lambda x:x['new_members'].pop())]
        for target,mutate in mutations:
            def read(reference):
                value=c.load(reference)
                if reference['path']==target['path']:mutate(value)
                return value
            with self.assertRaises((ValueError,RuntimeError)):core.candidates(read)

    def test_03_changed_old_origin_and_false_total_are_rejected(self):
        for change in ('old','count','newhash'):
            def read(reference):
                value=c.load(reference)
                if reference['path']==core.CANDIDATES['path']:
                    if change=='old':next(iter(value['sdk310']['native_candidates'].values()))['sha256']='0'*64
                    elif change=='count':value['distinct_native_candidates']=21
                    else:value['sdk310']['native_candidates']['/usr/lib/python3.10/lib-dynload/resource.cpython-310-x86_64-linux-gnu.so']['sha256']='0'*64
                return value
            with self.assertRaises(ValueError):core.candidates(read)

    def test_04_outer_fresh_types_import_and_all_three_snapshots(self):
        main,files,events,admission=child_world('outer312')
        main();result=json.loads(files[admission['output_root']+'/observation.json'])
        observation=result['types_source_observation']
        self.assertFalse(observation['first_checkpoint']['present'])
        self.assertTrue(observation['after_json']['present'])
        self.assertTrue(observation['executed_through_pinned_loader'])
        self.assertEqual(result['source_execution_provenance_gaps'],[])
        self.assertEqual(observation['execution_receipt'],result['executed_sources'][types_tracker.PIN['path']])
        self.assertEqual([x[1] for x in events if type(x)is tuple and x[0]=='types-present-at-maps'],[False,True,True])
        self.assertLess(events.index('maps'),events.index('ACK'))
        self.assertLess(events.index('ACK'),events.index('json-import'))
        self.assertLess(events.index('json-import'),events.index('types-source-read'))
        self.assertLess(events.index('types-source-read'),events.index('formal'))
        self.assertEqual(set(result['maps']),{'earliest','pre_ctypes','post_ctypes'})
        self.assertEqual(events.count('types-source-read'),2)

    def test_05_preexisting_types_keeps_an_explicit_separate_source_gap(self):
        main,files,events,admission=child_world('outer312',preexisting=True)
        initial=main.__globals__['sys'].modules['types'];main()
        result=json.loads(files[admission['output_root']+'/observation.json'])
        observation=result['types_source_observation']
        self.assertTrue(observation['first_checkpoint']['present'])
        self.assertFalse(observation['executed_through_pinned_loader'])
        self.assertIsNone(observation['execution_receipt'])
        self.assertNotIn(types_tracker.PIN['path'],result['executed_sources'])
        self.assertEqual(result['source_execution_provenance_gaps'],[dict(path=types_tracker.PIN['path'],reason='preexisting_module_execution_source_unproved')])
        self.assertIs(main.__globals__['sys'].modules['types'],initial)
        self.assertEqual(events.count('types-source-read'),1)
        self.assertFalse(observation['interpreter_startup_cause_established'])
        self.assertFalse(observation['preexisting_module_reloaded'])

    def test_06_sdk_keeps_original_no_tracker_behavior(self):
        main,files,events,admission=child_world('sdk310');main()
        result=json.loads(files[admission['output_root']+'/observation.json'])
        self.assertIsNone(result['types_source_observation'])
        self.assertEqual(result['source_execution_provenance_gaps'],[])
        self.assertNotIn('types-source-read',events)
        self.assertEqual(set(result['maps']),{'earliest','pre_ctypes','post_ctypes'})

    def test_07_failed_types_execution_does_not_call_formal_entry_or_create_result(self):
        main,files,events,admission=child_world('outer312',failed_types=True)
        with self.assertRaisesRegex(ValueError,'synthetic types failed'):main()
        self.assertNotIn('formal',events);self.assertNotIn('libc',events)
        self.assertNotIn('types',main.__globals__['sys'].modules)
        self.assertNotIn(admission['output_root']+'/observation.json',files)

    def test_08_fake_preloaded_receipt_or_gap_omission_is_rejected(self):
        main,files,events,admission=child_world('outer312',preexisting=True);main()
        result=json.loads(files[admission['output_root']+'/observation.json'])
        for kind in ('receipt','claim','gap','path'):
            observation=copy.deepcopy(result['types_source_observation']);sources=copy.deepcopy(result['executed_sources'])
            if kind=='receipt':sources[types_tracker.PIN['path']]={'synthetic':True}
            elif kind=='claim':observation['executed_through_pinned_loader']=True
            elif kind=='gap':observation['original_execution_provenance_gap']=False
            else:observation['current_source_reference']['path']='/wrong/types.py'
            with self.assertRaises(ValueError):core.source_execution_gaps('outer312',observation,sources)

    def test_09_bad_provider_or_resource_limits_keep_failure_evidence(self):
        for role,kwargs in [('sdk310',dict(bad_provider=True)),('outer312',dict(bad_provider=True)),
                            ('sdk310',dict(wrong_limits=True))]:
            main,files,events,admission=child_world(role,**kwargs)
            with self.assertRaises(ValueError):main()
            failure=json.loads(files[admission['output_root']+'/failure.json'])
            self.assertIn('failed_bootstrap',failure['maps'])
            self.assertNotIn('libc',events)
            self.assertNotIn(admission['output_root']+'/observation.json',files)

    def test_10_three_exact_literal_components_fit_original_bound(self):
        sources=[(Q/name).read_text()for name in (core.HELPER,core.CHILD,core.TYPES)]
        literal=core.literal(*sources)
        self.assertLess(len(literal.encode()),65536)
        constants=[node.value for node in ast.walk(ast.parse(literal))if isinstance(node,ast.Constant)and type(node.value)is str]
        for source in sources:self.assertIn(source,constants)
        self.assertLess(literal.index("types_helpers={'__name__'"),literal.index("child_scope={'__name__'"))
        with self.assertRaises(ValueError):core.literal(sources[0],sources[1],sources[2]+' '*65536)
        command=core.argv_for('sdk310',{'sdk310':{'python':'/mock/python'}},literal,dict(path='/p',sha256='a'*64),dict(path='/a',sha256='b'*64),42,1000.)
        self.assertEqual(command[-1],hashlib.sha256(literal.encode()).hexdigest())

    def test_11_original_guards_history_math_and_limits_are_unchanged(self):
        def funcs(name):return {n.name:n for n in ast.parse((Q/name).read_text()).body if isinstance(n,ast.FunctionDef)}
        oldrun=funcs('run_minicpm5_bootstrap_observation_v1.py');newrun=funcs('run_minicpm5_bootstrap_observation_v2.py')
        for name in ('history','save_history','inventory'):self.assertEqual(ast.dump(oldrun[name]),ast.dump(newrun[name]))
        for name in ('guard','stopped'):
            def get(tree):return next(n for n in ast.walk(tree['main'])if isinstance(n,ast.FunctionDef)and n.name==name)
            self.assertEqual(ast.dump(get(oldrun)),ast.dump(get(newrun)))
        oldchild=funcs('minicpm5_bootstrap_observation_child_v1.py');newchild=funcs('minicpm5_bootstrap_observation_child_v2.py')
        for name in ('validate_arguments','validate_admission'):self.assertEqual(ast.dump(oldchild[name]),ast.dump(newchild[name]))
        old=funcs('minicpm5_bootstrap_observation_v1.py');new=funcs('minicpm5_bootstrap_observation_v2.py')
        for name in ('collect_content','summarize_maps','verify_sources','argv_for'):self.assertEqual(ast.dump(old[name]),ast.dump(new[name]))
        self.assertEqual(core.BUDGET,oldcore.BUDGET);self.assertEqual(core.CAPS,oldcore.CAPS);self.assertEqual(core.LIMITS,oldcore.LIMITS)
        source=(Q/core.CHILD).read_text()
        self.assertLess(source.index('types_tracker='),source.index('earliest=maps()'))
        self.assertLess(source.index("'Exact live-owner parent ACK'"),source.index('types_tracker.install_before_json'))
        self.assertLess(source.index('types_tracker.install_before_json'),source.index('    import json'))
        self.assertLess(source.index('types_tracker.after_json()'),source.index('original_finder=entry.finder'))
        self.assertNotIn('.source_entry(',source)

    def test_12_accepted_components_and_all145_sources_retained(self):
        manifest=source_manifest();pins=manifest['source_sha256']
        self.assertEqual(len(pins),145)
        core.component_reviews(c.load,pins)
        self.assertGreater(core.verify_sources(pins,c.reference),0)
        for path,sha in c.load(core.RETAINED_MANIFEST)['source_sha256'].items():self.assertEqual(pins[path],sha)
        bad=copy.deepcopy(pins);bad[core.COMPONENT_SOURCES['elf']['path']]='0'*64
        with self.assertRaises(ValueError):core.component_reviews(c.load,bad)

    def build_fixture(self,mutate=lambda docs:None):
        catalog_ref=ref('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v91.json')
        catalog=c.load(catalog_ref);oldplan=c.load(ref('minicpm5-bootstrap-observation-r1-20261112-plan.json'))
        manifest=source_manifest();mr=dict(path=str(Q/'synthetic-r2-manifest.json'),sha256='1'*64)
        ar=dict(path=str(Q/'synthetic-r2-acceptance.json'),sha256='2'*64)
        accepted=dict(reviewer='/root',passed=True,blockers=[],manifest=mr,source_sha256=manifest['source_sha256'])
        docs={mr['path']:manifest,ar['path']:accepted,catalog_ref['path']:catalog};mutate(docs)
        def read(reference):return copy.deepcopy(docs[reference['path']]) if reference['path']in docs else c.load(reference)
        return builder.build(read,catalog_ref=catalog_ref,scope_ref=oldplan['r2_predecessor_scope'],
            predecessor_ref=catalog['runs'][-1]['terminal'],retirement_acceptance_ref=oldplan['source_only_acceptance'],
            source_manifest_ref=mr,source_acceptance_ref=ar,
            output_root='/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-r2-synthetic')

    def test_13_saved_history_builder_is_parameterized_and_unadmitted(self):
        plan=self.build_fixture()
        self.assertFalse(plan['runtime_admitted']);self.assertFalse(plan['automatic_successor_launch'])
        self.assertEqual(len(plan['source_sha256']),145)
        scope=c.load(plan['r2_predecessor_scope'])
        self.assertTrue(all(child in plan['predecessor_owners']for child in scope['children']))
        self.assertEqual(plan['candidate_inputs'],core.CANDIDATES)
        review=dict(schema='minicpm5_bootstrap_observation_review_v2',reviewer='/root',passed=True,blockers=[],
                    plan=dict(path='/mock/plan.json',sha256='a'*64),observation_only=True,accepted_native_authority_created=False)
        for key in ('source_sha256','candidate_inputs','limits','artifact_limits','budget','role_order',
                    'complete_current_serial_catalog','root_handoff','predecessor_exit','lock_policy',
                    'source_component_reviews','supplemental_origin_receipts'):review[key]=copy.deepcopy(plan[key])
        core.validate(plan,review,review['plan'])
        for key,value in [('limits',{**core.LIMITS,'wall_seconds':901}),('budget',{**core.BUDGET,'single_file_bytes':4*2**20})]:
            changed=copy.deepcopy(plan);changed[key]=value
            with self.assertRaises(ValueError):core.validate(changed,review,review['plan'])

    def test_14_dropped_retained_source_or_catalog_origin_is_rejected(self):
        def dropped_source(docs):
            manifest=next(v for k,v in docs.items()if k.endswith('synthetic-r2-manifest.json'))
            manifest['source_sha256'].pop(next(iter(manifest['source_sha256'])))
        with self.assertRaises(ValueError):self.build_fixture(dropped_source)
        def missing_origin(docs):
            catalog=next(v for k,v in docs.items()if 'root-catalog-'in k)
            catalog['runs']=[row for row in catalog['runs']if core.ref2(row['terminal'])!=core.SUPPLEMENTAL_RECEIPTS['actual_exit']]
        with self.assertRaises((ValueError,RuntimeError)):self.build_fixture(missing_origin)


if __name__=='__main__':unittest.main(verbosity=2)
