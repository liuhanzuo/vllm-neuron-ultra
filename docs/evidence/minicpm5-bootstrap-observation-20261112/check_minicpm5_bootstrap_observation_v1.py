"""Pure maps/source tests and mocked complete parent lifecycle; no child/native run."""
import ast
import builtins
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research'
sys.path[:0]=[str(ROOT),str(Q)]
from research import minicpm5_bootstrap_observation_v1 as core
from research import run_minicpm5_bootstrap_observation_v1 as run
from research import minicpm5_sdk_mapping_native_closure_v1 as closure
from research import minicpm5_sdk_mapping_native_authenticator_v2 as native
from research import prepare_minicpm5_bootstrap_observation_v1 as prepare
import minicpm5_layer0_execution_protocol_v4 as proc
import minicpm5_layer0_external_supervisor_v4 as external


def plan_fixture():
    pins={str(Q/name):'a'*64 for name in (core.RUNNER,core.CHILD,core.HELPER,'minicpm5_bootstrap_observation_v1.py',
        'minicpm5_sdk_mapping_outer_source_entry_v4.py','minicpm5_sdk_mapping_worker_source_entry_v4.py')}
    pr=dict(path='/synthetic/plan.json',sha256='b'*64)
    value=dict(schema='minicpm5_bootstrap_observation_plan_v1',stage='inventory',runtime_bound=True,
        automatic_successor_launch=False,role_order=list(core.ROLES),candidate_inputs=core.CANDIDATES,
        limits=core.LIMITS,artifact_limits=core.CAPS,budget=core.BUDGET,source_sha256=pins,
        mini_scientific_constraints=dict(native_checks=9,scientific_descriptor_rows=1284,absolute_wall_seconds=900,
            address_space_bytes=2147483648,single_file_bytes=2097152),
        output_root='/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-bootstrap-synthetic',
        complete_current_serial_catalog={},root_handoff={},predecessor_exit={},lock_policy=run.runtime.POLICY,
        predecessor_owners=[],observation_only=True,accepted_native_authority_created=False)
    review=dict(schema='minicpm5_bootstrap_observation_review_v1',reviewer='/root',passed=True,blockers=[],plan=pr,
        observation_only=True,accepted_native_authority_created=False)
    for k in ('source_sha256','candidate_inputs','limits','artifact_limits','budget','role_order',
              'complete_current_serial_catalog','root_handoff','predecessor_exit','lock_policy'):review[k]=copy.deepcopy(value[k])
    return value,review,pr


class Checks(unittest.TestCase):
    def test_exact_scope_and_limit_regressions_fail(self):
        p,r,ref=plan_fixture();core.validate(p,r,ref)
        changes=[('role_order',['outer312','sdk310']),('limits',{**core.LIMITS,'wall_seconds':901}),
                 ('artifact_limits',{**core.CAPS,'one_file_bytes':4*2**20}),('automatic_successor_launch',True),
                 ('accepted_native_authority_created',True)]
        for key,value in changes:
            p,r,ref=plan_fixture();p[key]=value
            with self.assertRaises(ValueError):core.validate(p,r,ref)
        p,r,ref=plan_fixture();r['accepted_native_authority_created']=True
        with self.assertRaises(ValueError):core.validate(p,r,ref)

    def test_unknown_extension_and_anonymous_exec_are_reported_without_reads(self):
        table={'sdk310':dict(native_candidates={},executable='/python')}
        text='1000-2000 r-xp 00000000 01:02 123 /usr/lib/_blake2.so\n2000-3000 rwxp 00000000 00:00 0 \n'
        value=core.summarize_maps(text,'sdk310',table,{'files':{}},closure,lambda _:self.fail('No lookup for unknown objects'))
        self.assertEqual([g['reason'] for g in value['gaps']],['missing_saved_content_origin','uncovered_executable_mapping'])
        self.assertFalse(value['accepted_native_authority_created'])
        self.assertEqual(value['gaps'][0]['mapping']['device'],[1,2])
        self.assertEqual(value['gaps'][0]['mapping']['inode'],123)

    def test_known_candidate_identity_mismatch_is_not_authenticated(self):
        table={'sdk310':dict(native_candidates={'/lib.so':{}},executable='/python')}
        content={'files':{'/lib.so':dict(identity=dict(st_dev=0,st_ino=456),elf={})}}
        value=core.summarize_maps('1000-2000 r-xp 0 00:00 123 /lib.so\n','sdk310',table,content,closure,lambda _:None)
        self.assertEqual(value['gaps'][0]['reason'],'known_candidate_mapping_identity_mismatch')

    def test_candidate_ffi_failure_is_evidence_not_success(self):
        table={r:dict(native_candidates={'/known':dict(path='/known',sha256='a'*64,bytes=3)},
            pure_source_candidates={},ffi_candidate='/known') for r in core.ROLES}
        calls=[]
        reader=NS(read=lambda ref,cap:(calls.append(ref['path']) or b'bad',dict(identity={'st_dev':0,'st_ino':1})))
        bad=NS(parse_elf64=lambda _:(_ for _ in ()).throw(ValueError('unsupported')),
               ffi_definitions=lambda _:(_ for _ in ()).throw(ValueError('symbols missing')))
        value=core.collect_content(table,reader,bad,lambda:None)
        self.assertEqual(calls,['/known']);self.assertIn('ffi_failure',value['files']['/known'])
        self.assertNotIn('ffi',value['files']['/known']);self.assertFalse(value['accepted_native_authority_created'])

    def test_literal_uses_complete_frozen_helpers_and_no_script_main(self):
        helper=(Q/core.HELPER).read_text();child=(Q/core.CHILD).read_text()
        literal=core.literal(helper,child);tree=ast.parse(literal)
        strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
        self.assertIn(helper,strings);self.assertIn(child,strings)
        self.assertIn('mini_discovery_helpers',strings)
        with self.assertRaises(ValueError):core.literal(helper,child+'x'*65536)
        # Earliest maps precedes json, real formal imports precede pre maps,
        # ctypes follows the pre snapshot and cannot invoke formal source_entry.
        self.assertLess(child.index('earliest=maps()'),child.index('    import json'))
        self.assertLess(child.index('importlib.import_module(core.ENTRIES[role])'),child.index('before=maps()'))
        self.assertLess(child.index('before=maps()'),child.index('        import ctypes'))
        self.assertNotIn('.source_entry(',child)
        body=ast.parse(child)
        main=next(n for n in body.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        protected=[n for n in ast.walk(main) if isinstance(n,ast.Try) and
            any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='import_module'
                for item in n.body for x in ast.walk(item))]
        self.assertEqual(len(protected),1)
        self.assertIn("refs['failed_bootstrap']",ast.get_source_segment(child,protected[0]))

    def test_child_admission_and_arguments_reject_tampering_without_running_child(self):
        namespace=dict(__name__='mock_child_source',helpers=dict(require=core.require,
            digest=lambda raw:hashlib.sha256(raw).hexdigest()))
        exec(compile((Q/core.CHILD).read_text(),str(Q/core.CHILD),'exec'),namespace)
        command=core.argv_for('sdk310',{'sdk310':dict(python='/python')},'literal',
            dict(path='/plan',sha256='a'*64),dict(path='/admission',sha256='b'*64),12,1000.)
        runtime=NS(executable='/python',argv=['-c',*command[8:]],orig_argv=command,
            flags=NS(isolated=1,no_site=1,no_user_site=1,ignore_environment=1,optimize=0),
            dont_write_bytecode=True,pycache_prefix='/dev/null')
        self.assertEqual(namespace['validate_arguments'](runtime,100.),('sdk310',1000.))
        for kind in ('literal','flags','expired','restart'):
            wrong=copy.deepcopy(runtime);now=100.
            if kind=='literal':wrong.orig_argv[7]='changed'
            elif kind=='flags':wrong.flags.optimize=1
            elif kind=='expired':now=1001.
            else:now=99.
            with self.assertRaises(ValueError):namespace['validate_arguments'](wrong,now)
        template=list(command);template[13]='admission:self'
        admission=dict(schema='minicpm5_bootstrap_observation_admission_v1',role='sdk310',
            plan=dict(path='/plan',sha256='a'*64),deadline=1000.,started_monotonic=100.,
            literal_sha256=command[-1],command=template,parent_owner=dict(pid=123),
            accepted_native_authority_created=False)
        namespace['validate_admission'](runtime.argv,command,admission,1000.,123)
        for key,value in [('role','outer312'),('started_monotonic',101.),('deadline',1001.),
            ('parent_owner',dict(pid=124)),('accepted_native_authority_created',True),
            ('command',command),('plan',dict(path='/other',sha256='a'*64))]:
            wrong=copy.deepcopy(admission);wrong[key]=value
            with self.assertRaises(ValueError):namespace['validate_admission'](runtime.argv,command,wrong,1000.,123)

    def test_child_formal_bootstrap_failure_retains_earliest_and_partial_maps(self):
        events=[];files={};fds={};nextfd=[100];ack=[b'R']
        command=core.argv_for('sdk310',{'sdk310':dict(python='/python')},'literal',
            dict(path='/plan',sha256='a'*64),dict(path='/admission',sha256='b'*64),12,1000.)
        external=NS(SourceFileLoader=type('MockSourceFileLoader',(),{}))
        runtime=NS(executable='/python',argv=['-c',*command[8:]],orig_argv=command,
            flags=NS(isolated=1,no_site=1,no_user_site=1,ignore_environment=1,optimize=0),
            dont_write_bytecode=True,pycache_prefix='/dev/null',path=[],meta_path=[],
            modules={'_frozen_importlib_external':external},version_info=(3,10),builtin_module_names=())
        template=list(command);template[13]='admission:self'
        admission=dict(schema='minicpm5_bootstrap_observation_admission_v1',role='sdk310',
            plan=dict(path='/plan',sha256='a'*64),deadline=1000.,started_monotonic=100.,
            literal_sha256=command[-1],command=template,parent_owner=dict(pid=123),
            accepted_native_authority_created=False,content=dict(path='/content',sha256='c'*64),
            output_root='/output/sdk310',review=dict(path='/review',sha256='d'*64))
        documents={'/plan':dict(source_sha256={},candidate_inputs=dict(path='/table',sha256='e'*64),output_root='/output'),
            '/admission':admission,'/table':{'sdk310':dict(python='/python',executable='/realpython',pure_source_candidates={})},
            '/content':dict(current_candidate_bytes_authenticated=True,accepted_native_authority_created=False),'/review':{}}
        class Entry:
            def __init__(self):self.sources={};self.source_bytes=0
            def check(self):pass
            def prefix(self):return ()
            def install(self,_):self.finder=NS(find_spec=lambda *a:None);runtime.meta_path[:]=[self.finder]
            def read_source(self,path,sha,*a):return core.encoded(documents[path]),()
        raw=b'1000-2000 r-xp 0 01:02 123 /unknown.so\n'
        def opened(path,*a):
            nextfd[0]+=1;fd=nextfd[0];fds[fd]=[path,0]
            if path.endswith('.maps') or path.endswith('.json'):files[path]=bytearray()
            return fd
        def read(fd,count):
            if fd==12:events.append('ACK');return ack[0]
            path,at=fds[fd];self.assertEqual(path,'/proc/self/maps');fds[fd][1]+=count
            events.append('maps');return raw[at:at+count]
        def write(fd,value):
            self.assertNotEqual(fd,2,'Failure evidence must persist in mock')
            files[fds[fd][0]].extend(value);return len(value)
        ports=NS(open=opened,read=read,write=write,close=lambda _:None,fsync=lambda _:None,
            getppid=lambda:123,readlink=lambda _:'/realpython',O_RDONLY=0,O_CLOEXEC=0,O_WRONLY=0,O_CREAT=0,
            O_EXCL=0,O_NOFOLLOW=0,O_DIRECTORY=0)
        def formal(name):events.append('formal');raise ValueError('synthetic formal bootstrap failure')
        research=NS(minicpm5_bootstrap_observation_v1=NS(validate=lambda *a:None,ENTRIES=core.ENTRIES))
        imports={'sys':runtime,'posix':ports,'time':NS(monotonic=lambda:100.),'json':json,
            'research':research,'importlib':NS(import_module=formal)}
        def importing(name,*args):
            events.append('import:'+name)
            if name not in imports:self.fail('Unexpected imported dependency: '+name)
            return imports[name]
        built=dict(vars(builtins));built['__import__']=importing
        namespace=dict(__name__='mock_child',__builtins__=built,
            helpers=dict(require=core.require,digest=lambda raw,*a:hashlib.sha256(raw).hexdigest(),Entry=Entry,ROOT=str(ROOT)))
        exec(compile((Q/core.CHILD).read_text(),str(Q/core.CHILD),'exec'),namespace)
        with self.assertRaisesRegex(ValueError,'synthetic formal bootstrap'):namespace['main']()
        self.assertEqual(bytes(files['/output/sdk310/earliest.maps']),raw)
        self.assertEqual(bytes(files['/output/sdk310/failed-bootstrap.maps']),raw)
        self.assertNotIn('/output/sdk310/pre-ctypes.maps',files)
        failure=json.loads(files['/output/sdk310/failure.json'])
        self.assertFalse(failure['accepted_native_authority_created']);self.assertNotIn('import:ctypes',events)
        self.assertLess(events.index('maps'),events.index('ACK'));self.assertLess(events.index('ACK'),events.index('import:json'))
        ack[0]=b'X';files.clear();events.clear()
        with self.assertRaisesRegex(ValueError,'parent ACK'):namespace['main']()
        self.assertEqual(files,{});self.assertNotIn('import:json',events)

    def test_prepare_keeps_latest106_and_complete_history(self):
        plan,_,_=plan_fixture();plan.update(predecessor_services=[dict(unit='historical',invocation_id='fixed')],
            predecessor_owners=[dict(pid=1,start_ticks='1')],retired_mini_history=['retained'],
            archive_inputs='old',archive_mode='download',replay_from=None)
        pins={str(Q/('synthetic_'+str(i)+'.py')):'a'*64 for i in range(106)}
        plan['source_sha256']=dict(list(pins.items())[:98])
        manifest=dict(schema='minicpm5_bootstrap_observation_source_manifest_v1',source_sha256=pins,
            limits=core.LIMITS,artifact_limits=core.CAPS,candidate_inputs=core.CANDIDATES,
            retained_manifest=core.RETAINED_MANIFEST)
        ref=dict(path='/manifest',sha256='a'*64);accept=dict(path='/accepted',sha256='b'*64)
        accepted=dict(reviewer='/root',passed=True,blockers=[],manifest=ref)
        def read(r):return {ref['path']:manifest,accept['path']:accepted,
            core.RETAINED_MANIFEST['path']:dict(source_sha256=pins)}[r['path']]
        kwargs=dict(catalog_ref={},scope_ref={},predecessor_ref={},retirement_acceptance_ref={},
            source_manifest_ref=ref,source_acceptance_ref=accept,output_root=plan['output_root'])
        with patch.object(prepare.previous,'build',lambda *a,**k:copy.deepcopy(plan)),patch.object(core,'candidates',lambda _:None):
            built=prepare.build(read,**kwargs)
            for key in ('predecessor_owners','predecessor_services','retired_mini_history'):
                self.assertEqual(built[key],plan[key])
            self.assertEqual(built['source_sha256'],pins);self.assertNotIn('archive_inputs',built)
            manifest['source_sha256']=dict(list(pins.items())[:-1])
            with self.assertRaises(ValueError):prepare.build(read,**kwargs)

    def test_history_uses_existing_reconstruction_and_rejects_omissions(self):
        plan,_,_=plan_fixture();plan.update(complete_current_serial_catalog=dict(path='/catalog',sha256='a'*64),
            root_handoff=dict(path='/handoff',sha256='b'*64),predecessor_exit=dict(path='/exit',sha256='c'*64),
            r2_predecessor_scope={},source_only_acceptance={},predecessor_services=[dict(unit='one')],
            predecessor_expected_exit_code=0,physical_history_source={},r2_actual_exit={},r2_full_audit={},
            retired_lfm_r2={},retired_mini_history=[],predecessor_owners=[dict(pid=1,start_ticks='1')])
        catalog=dict(reviewer='/root',passed=True,blockers=[],all_later_serial_runs_included=True,
            current_handoff=plan['root_handoff'],runs=[dict(terminal=plan['predecessor_exit'])])
        events=[];rebuilt=copy.deepcopy(plan)
        def physical(p,*a):events.append('physical');return p['predecessor_owners']
        with patch.object(run.history_builder,'build',lambda *a,**k:rebuilt),patch.object(run.retirement,'preflight_history',physical):
            self.assertEqual(run.history(plan,None,lambda _:catalog),plan['predecessor_owners'])
            for key,value in [('predecessor_owners',[]),('predecessor_services',[]),('retired_lfm_r2',{'changed':1})]:
                wrong=copy.deepcopy(plan);wrong[key]=value
                with self.assertRaises(ValueError):run.history(wrong,None,lambda _:catalog)
            self.assertEqual(events,['physical'])

    def test_complete_history_shards_respect_original_single_file_limit(self):
        rows=[dict(unit='unit-'+str(i),samples=['x'*20000]*2) for i in range(100)]
        saved={}
        def save(path,value,cap):
            raw=core.encoded(value);self.assertLessEqual(len(raw),cap)
            saved[str(path)]=copy.deepcopy(value)
            return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
        ref=run.save_history(Path('/synthetic'),rows,save)
        manifest=saved[ref['path']];restored=[]
        for part in manifest['parts']:
            self.assertEqual(part['first_row'],len(restored))
            restored.extend(saved[part['reference']['path']]['rows'])
        self.assertEqual(restored,rows);self.assertGreater(len(manifest['parts']),1)
        with self.assertRaises(ValueError):run.save_history(Path('/synthetic'),[dict(oversize='x'*2**20)],save)

    def parent_flow(self,fail_role=None,never_exit=False):
        events=[];clock=[100.];active=[None];lock=[False];saved_load=run.c.load
        plan,review,pr=plan_fixture();rr=dict(path='/synthetic/review.json',sha256='c'*64)
        for name in (core.HELPER,core.CHILD):plan['source_sha256'][str(Q/name)]=hashlib.sha256((Q/name).read_bytes()).hexdigest()
        table={r:dict(python='/synthetic/'+r+'/python') for r in core.ROLES}
        owner=dict(pid=100,start_ticks='1000');content=dict(files={},current_candidate_bytes_authenticated=True)
        class Proof:
            def __init__(self,*a,**kw):self.pending_evidence=lambda:None
            def __call__(self):assert lock[0]
            def initial_persisted(self,ref):pass
        def acquire(*_):lock[0]=True;return 90,owner,dict(device=1,inode=2),{}
        def drain(child,child_owner,p):events.append(('drain',child.role));child.drained=True
        def release(child,child_owner,p,e,fd,out):
            if child is not None:drain(child,child_owner,p)
            events.append(('release',None));lock[0]=False
        class Child:
            def __init__(self,command,**kw):
                if active[0] is not None:assert active[0].drained
                self.command=command;self.role=command[9];self.pid=101+len(events);self.drained=False;self.returncode=None
                self.owner=dict(pid=self.pid,ppid=100,start_ticks='1001',argv=command)
                active[0]=self;events.append(('spawn',self.role));assert lock[0]
            def poll(self):
                if never_exit:return None
                self.returncode=1 if self.role==fail_role else 0
                if self.returncode==0 and not hasattr(self,'written'):
                    self.written=True;ar=dict(path=self.command[12],sha256=self.command[13]);ad=saved_load(ar)
                    run.c.save(Path(ad['output_root'])/'observation.json',dict(complete=True,role=self.role,
                      deadline=float(self.command[15]),admission=ar,accepted_native_authority_created=False,
                      missing_content_origins=True,ffi_definitions=None),2097152)
                return self.returncode
        with tempfile.TemporaryDirectory() as temporary,ExitStack() as stack:
            plan['output_root']=str(Path(temporary)/'output')
            def load(ref,cap=8*2**20):
                if ref['path']==pr['path']:return plan
                if ref['path']==rr['path']:return review
                return saved_load(ref,cap)
            patches=[patch.object(run,'sys',NS(executable='/usr/bin/python3',dont_write_bytecode=True,
                        flags=NS(no_user_site=True),pycache_prefix='/dev/null',stderr=sys.stderr)),
                patch.object(run.core,'validate',lambda p,r,ref:p),patch.object(run.core,'verify_sources',lambda *a:None),
                patch.object(run.core,'candidates',lambda read:table),patch.object(run.core,'collect_content',lambda *a:content),
                patch.object(run.c,'load',load),patch.object(run.c,'verify_loaded',lambda *a:None),
                patch.object(run,'history',lambda *a:[]),patch.object(run.runtime,'acquire',acquire),
                patch.object(run.runtime,'Guard',Proof),patch.object(run.runtime,'initial_record',lambda *a:{}),
                patch.object(native,'StableReader',lambda *a:NS(unchanged=lambda paths:events.append(('unchanged',None)))),
                patch.object(run.life,'drain',drain),patch.object(run.life,'release_after_drain',release),
                patch.object(run.life,'free',lambda *a:100*2**30),patch.object(run.life,'memory_free',lambda:10*2**30),
                patch.object(run,'resource',NS(setrlimit=lambda *a:None,RLIMIT_AS=0,RLIMIT_CPU=1,RLIMIT_FSIZE=2)),
                patch.object(run,'signal',NS(SIGTERM=15,SIGINT=2,SIGALRM=14,ITIMER_REAL=0,
                     signal=lambda *a:None,getitimer=lambda *a:(0.,0.),setitimer=lambda *a:events.append(('timer',a[-1])))),
                patch.object(run,'time',NS(monotonic=lambda:clock[0],sleep=lambda _:clock.__setitem__(0,clock[0]+100))),
                patch.object(run,'os',NS(sched_setaffinity=lambda *a:None,getpriority=lambda *a:19,PRIO_PROCESS=0,
                     pipe=lambda:(200,201),close=lambda *a:None,write=lambda fd,raw:len(raw),fstat=__import__('os').fstat)),
                patch.object(run.subprocess,'Popen',Child),patch.object(proc,'user_bus',lambda:None),
                patch.object(proc,'clear_window',lambda:None),patch.object(proc,'active_heavy',lambda:None),
                patch.object(proc,'gone',lambda *a:None),patch.object(proc,'proc',lambda pid:active[0].owner)]
            for item in patches:stack.enter_context(item)
            argv=['--plan',pr['path'],'--plan-sha256',pr['sha256'],'--review',rr['path'],'--review-sha256',rr['sha256']]
            if fail_role or never_exit:
                with self.assertRaises((ValueError,TimeoutError)):run.main(argv)
            else:
                run.main(argv);result=json.loads((Path(plan['output_root'])/'result.json').read_text())
                self.assertEqual(result['deadline'],1000.)
                self.assertEqual([x['role'] for x in result['children']],list(core.ROLES))
                self.assertFalse(result['accepted_native_authority_created'])
            self.assertFalse(lock[0])
        return events

    def test_complete_parent_sdk_then_outer_same_deadline_and_drain(self):
        events=self.parent_flow();seq=[e for e in events if e[0] in ('spawn','drain','release')]
        self.assertEqual(seq,[('spawn','sdk310'),('drain','sdk310'),('spawn','outer312'),('drain','outer312'),('release',None)])
        self.assertEqual([e for e in events if e[0]=='timer'],[('timer',900.),('timer',0)])

    def test_failed_sdk_is_drained_and_outer_never_starts(self):
        events=self.parent_flow(fail_role='sdk310')
        self.assertNotIn(('spawn','outer312'),events)
        self.assertLess(events.index(('drain','sdk310')),events.index(('release',None)))

    def test_original_deadline_kills_owned_work_before_release(self):
        events=self.parent_flow(never_exit=True)
        self.assertNotIn(('spawn','outer312'),events)
        self.assertLess(events.index(('drain','sdk310')),events.index(('release',None)))


if __name__=='__main__':unittest.main()
