"""Synthetic v2 lifecycle/source integration; no actual native/raw/model work."""
import ast
import builtins
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
import minicpm5_layer1_sdk_batch_layout_cpu_runner_v2 as run
import minicpm5_layer1_sdk_batch_layout_cpu_worker_v2 as worker
import minicpm5_layer1_sdk_batch_layout_cpu_runner_v1 as old
from research import minicpm5_layer1_sdk_batch_layout_completion_v1 as latest


class Checks(unittest.TestCase):
    def setUp(self):
        self.guard=Mock()
        self.binding=dict(schema='minicpm5_layer1_sdk_batch_layout_latest_completion_v1',status='complete',
            current_history=copy.deepcopy(latest.CURRENT),
            terminal_handoff={'path':'/synthetic/terminal.json','sha256':'a'*64},
            nemotron_r12=dict(schema='minicpm5_after_nemotron_r12_binding_v1',
                actual_exit={'path':'/synthetic/r12-exit.json','sha256':'b'*64},failed_release_exit=None))
        self.contract={'path':str(Q/'minicpm5-layer1-sdk-batch-layout-source-contract-synthetic.json'),'sha256':'c'*64}
        self.native={'binding':{'native':'synthetic'}}
        self.previous={'binding':{'previous':'synthetic'},'spec':{},'records':{}}
        self.original={'science':'unchanged','inputs':{},'source_sha256':{}}
        cs={k:{'synthetic':k}for k in ('finalization','catalog','finalizer_exit','canonical_sha256','nemotron_completion')}
        cs['unit']='synthetic-collector'
        self.collector=dict(binding={'collector':'synthetic'},collector_plan=self.original,collector_spec=cs,
            collector_chain={'rank_results':[{'rank':0},{'rank':1}]},audit_result={'audit':'synthetic'},
            saved_audit_spec={'source_sha256':{}})
        self.done=dict(collector=self.collector,native=self.native,current_history=self.previous,
            owners=[{'pid':11,'start_ticks':'2'}],lock_identity={'inode':3},source_sha256={},binding=self.binding,
            bound={'spec':{'synthetic':True},'finalizer_owner':{'pid':12}})
        self.plan=dict(copy.deepcopy(self.original),output_root='/synthetic/new',latest_completion=self.binding,
            layer1_sdk_batch_layout=self.previous['binding'],layer1_official_history=self.native['binding'],source_contract=self.contract)
        self.previous['plan']={k:copy.deepcopy(v)for k,v in self.plan.items()if k not in ('latest_completion','layer1_sdk_batch_layout','source_contract')}
        self.spec=dict(cs,plan={'path':'/synthetic/plan.json','sha256':'d'*64},latest_completion=self.binding,
            completed_current_history=self.previous['binding'],completed_native_cache=self.native['binding'],
            completed_layer1_data=self.collector['binding'],source_contract=self.contract)
        self.context=types.SimpleNamespace(_sources_checked=True,plan=self.plan,permit={'rank':0},
            guard=Mock(),read_json=Mock(),activate=Mock(return_value=contextlib.nullcontext()),record_cpu_import=Mock(),
            sdk_guard=Mock(),verify_loaded_sources=Mock(),
            _inherited_native_receipt={'source_contract':self.contract})

    def assert_no_native(self):
        self.assertFalse(any(n.split('.')[0]in {'torch','numpy','neuronxcc','torch_neuronx','vllm'}for n in sys.modules))

    def fixture_bundle(self):
        sha=hashlib.sha256(b'synthetic source').hexdigest()
        sr={'path':'/synthetic/spec.json','sha256':'a'*64};rr={'path':'/synthetic/review.json','sha256':'b'*64}
        plan=copy.deepcopy(self.plan)
        pins={run.SELF:sha,str(R/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v2.py'):sha,
            '/unopened/historical.raw':'1'*64,'/unopened/lib.so.2':'2'*64}
        plan.update(source_sha256=pins,output_root='/mnt/nvme/minicpm5-layer1-sdk-batch-layout-synthetic')
        maps={'completed_native_cache':'layer1_official_history','completed_official_history':'layer1_current_history_factorial',
            'completed_storage':'completed_storage_maintenance','completed_factorial':'layer1_native_cache',
            'completed_cross_history':'layer1_kv_factorial','completed_own_history':'layer1_cross_history',
            'completed_layer1_data':'layer1_own_history_replay','completed_current_history':'layer1_sdk_batch_layout'}
        spec=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v2',latest_completion=self.binding,
            runner=run.SELF,worker=str(R/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v2.py'),source_sha256=pins,
            plan={'path':'/synthetic/plan.json','sha256':'c'*64},output_root=plan['output_root'],output_cap_bytes=run.CAP,
            root_floor_bytes=28*2**30,output_floor_bytes=30*2**30,resources=copy.deepcopy(run.PROPS),
            master_port=29531,unit='trn1-mini-layer1-sdk-batch-layout-synthetic',source_contract=self.contract,
            model_qualification_claim=False,automatic_full_gates_launch=False,
            original_completed_collection=run.OBSERVED_EXIT,prepared_source_plan=run.RESULTS_SOURCE_PLAN)
        for sk,pk in maps.items():plan.setdefault(pk,{'synthetic':pk});spec[sk]=plan[pk]
        review=dict(schema='minicpm5_layer1_sdk_batch_layout_runtime_source_review_v2',reviewer='/root',passed=True,
            blockers=[],spec=sr,source_sha256=pins)
        def load(ref):
            if ref==sr:return spec
            if ref==rr:return review
            if ref==spec['plan']:return plan
            raise AssertionError('Unplanned metadata read')
        return sr,rr,spec,plan,review,load

    def test_source_only_bundle_uses_full_partition_no_native_and_no_latest_terminal(self):
        sr,rr,spec,plan,review,load=self.fixture_bundle()
        with patch.object(run.source_auth,'load',side_effect=load),patch.object(run.source_auth,'classify',return_value={'partition':True})as classify,\
            patch.object(run.source_auth,'verify_text_sources')as verify,patch.object(run.source_auth,'text',return_value=b'synthetic source')as text,\
            patch.object(run,'verify_loaded_sources')as loaded,patch.object(run.source_auth,'authenticate_native_sources')as native,\
            patch.object(latest,'require_latest_terminal')as fresh:
            self.assertEqual(run.read_bundle(sr,rr,self.guard),(spec,plan))
            classify.assert_called_once_with(spec['source_sha256'],self.contract)
            verify.assert_called_once_with({'partition':True},self.guard)
            self.assertEqual({c.args[0]for c in text.call_args_list},{run.SELF,spec['worker']})
            native.assert_not_called();fresh.assert_not_called();loaded.assert_called_once_with(plan,self.guard)

    def test_source_contract_rebind_and_resource_change_reject(self):
        for mutation in ('contract','resources'):
            sr,rr,spec,plan,review,load=self.fixture_bundle()
            if mutation=='contract':plan['source_contract']={}
            else:spec['resources']=dict(run.PROPS,AllowedCPUs='0-31')
            with patch.object(run.source_auth,'load',side_effect=load),patch.object(run.source_auth,'classify'),\
                patch.object(run.source_auth,'verify_text_sources'),patch.object(run.source_auth,'text',return_value=b'synthetic source'),\
                patch.object(run,'verify_loaded_sources'):
                with self.assertRaises(RuntimeError):run.read_bundle(sr,rr,self.guard)

    def test_pending_outer_precedes_lock_or_native_auth(self):
        with patch.object(run.source_auth,'load',return_value={'latest_completion':{'status':'pending'}}),\
            patch.object(run.p,'clear_window'),patch.object(run.post.external,'acquire_lock')as lock,\
            patch.object(run.source_auth,'authenticate_native_sources')as native,patch.object(run.p,'command')as launch:
            with self.assertRaises(RuntimeError):run.outer({}, {})
            lock.assert_not_called();native.assert_not_called();launch.assert_not_called()

    def test_loaded_new_helper_requires_exact_pin_before_native(self):
        path=str(Q/'synthetic_late.py');module=types.SimpleNamespace(__file__=path);sha=hashlib.sha256(b'late').hexdigest()
        with patch.object(run.sys,'modules',{'late':module}),patch.object(run.source_auth,'text',return_value=b'late')as text:
            with self.assertRaises(RuntimeError):run.verify_loaded_sources({'source_sha256':{}},self.guard)
            text.assert_not_called()
            proof=run.verify_loaded_sources({'source_sha256':{path:sha}},self.guard)
            self.assertEqual(proof['source_sha256'],{path:sha})
            with self.assertRaises(RuntimeError):run.verify_loaded_sources({'source_sha256':{path:'0'*64}},self.guard)

    def test_complete_old_science_inputs_sources_plus_single_new_contract(self):
        run.validate_current_history_extension(self.plan,self.done)
        for where in ('science','input','source','supplement'):
            old=copy.deepcopy(self.previous['plan'])
            if where=='science':self.previous['plan']['science']='changed'
            if where=='input':self.previous['plan']['inputs']['old']={'keep':True}
            if where=='source':self.previous['plan']['source_sha256']['old.py']='a'*64
            if where=='supplement':self.previous['plan']['completed_storage_maintenance']={'must':'retain'}
            with self.assertRaises(RuntimeError):run.validate_current_history_extension(self.plan,self.done)
            self.previous['plan']=old
        self.plan['unreviewed_extra']=True
        with self.assertRaises(RuntimeError):run.validate_current_history_extension(self.plan,self.done)

    def test_prerequisites_and_result_reader_use_large_metadata_transport(self):
        ref={'path':'/synthetic/large.json','sha256':'a'*64};ident={'size':14*2**20}
        self.plan['inputs'][ref['path']]={'sha256':ref['sha256'],'identity':ident}
        def complete(binding,read,guard):self.assertEqual(read(ref),{'large':True});return self.done
        with patch.object(latest,'completed',side_effect=complete),patch.object(run.source_auth,'load',return_value={'large':True})as load,\
            patch.object(run,'identity',return_value=ident),patch.object(run,'verify_loaded_sources'):
            done,read=run.admitted_bundles(self.plan,self.guard)
            self.assertIs(done,self.done);self.assertEqual(read(ref['path'],ref['sha256']),{'large':True})
            self.assertEqual(load.call_count,2)

    def guard_fixture(self):
        a={'outer_owner':{'pid':2},'lock_identity':{'inode':3}}
        spec={'output_root':'/synthetic','root_floor_bytes':1}
        process=dict(pid=4,start_ticks='4',ppid=3,argv=['rank'],cgroup='group')
        co=dict(pid=3,start_ticks='3',ppid=2,argv=['coordinator'],cgroup='group')
        done={'owners':[{'pid':10+i,'start_ticks':str(i+1)}for i in range(1000)]}
        permit={'permit':'synthetic'}
        return run.RankGuard(a,spec,self.plan,process,co,done,permit),co

    def test_44000_fast_checks_do_not_rescan_all_owners_but_keep_every_lock_proc_window(self):
        guard,co=self.guard_fixture()
        counts=dict(lock=0,proc=0,window=0,gone=0,usage=0)
        def count(key,result=None):
            def call(*a,**kw):counts[key]+=1;return result
            return call
        with patch.object(run.p,'lock_proof',side_effect=count('lock')),patch.object(run.p,'proc',side_effect=count('proc',co)),\
            patch.object(run.p,'clear_window',side_effect=count('window')),patch.object(run.p,'gone',side_effect=count('gone')),\
            patch.object(run,'usage',side_effect=count('usage')),patch.object(run.time,'monotonic',return_value=0.5):
            for _ in range(44000):self.assertEqual(guard(),guard.permit)
            self.assertEqual(guard.full_checks,1);self.assertEqual(counts['gone'],1000);self.assertEqual(counts['usage'],1)
            self.assertEqual(counts['lock'],44001);self.assertEqual(counts['proc'],44001);self.assertEqual(counts['window'],44001)
            proof=guard.evidence();self.assertTrue(proof['full_source_checks_not_sampled'])
            self.assertFalse(proof['every_source_fully_authenticated'])

    def test_guard_one_second_period_and_forced_boundaries(self):
        guard,co=self.guard_fixture();clock=[0.]
        with patch.object(run.p,'lock_proof'),patch.object(run.p,'proc',return_value=co),patch.object(run.p,'clear_window'),\
            patch.object(run.p,'gone')as gone,patch.object(run,'usage')as usage,patch.object(run.time,'monotonic',side_effect=lambda:clock[0]):
            guard();clock[0]=0.99;guard();self.assertEqual(guard.full_checks,1)
            clock[0]=1.;guard();self.assertEqual(guard.full_checks,2)
            guard.full('before_native_import');self.assertEqual(guard.full_checks,3)
            self.assertEqual(gone.call_count,3000);self.assertEqual(usage.call_count,3)
            self.assertEqual([x['boundary']for x in guard.events],['periodic','periodic','before_native_import'])

    def test_fast_lock_proc_or_performance_failure_stops_without_waiting_period(self):
        for method in ('lock_proof','proc','clear_window'):
            guard,co=self.guard_fixture();guard.last_full=0.
            with patch.object(run.p,'lock_proof'),patch.object(run.p,'proc',return_value=co),patch.object(run.p,'clear_window'),\
                patch.object(run.p,method,side_effect=RuntimeError('fast failure')),patch.object(run.time,'monotonic',return_value=0.1),\
                patch.object(run.p,'gone')as gone:
                with self.assertRaisesRegex(RuntimeError,'fast failure'):guard()
                gone.assert_not_called()

    def test_metadata_watch_is_complete_and_cannot_be_rebound(self):
        guard,co=self.guard_fixture();guard.watch_metadata({'/synthetic/meta.json':{'inode':1}})
        with self.assertRaises(RuntimeError):guard.watch_metadata({'/synthetic/meta.json':{'inode':2}})
        with patch.object(run.p,'lock_proof'),patch.object(run.p,'proc',return_value=co),patch.object(run.p,'clear_window'),\
            patch.object(run.p,'gone'),patch.object(run,'usage'),patch.object(run,'identity',return_value={'inode':2}):
            with self.assertRaises(RuntimeError):guard.full('before_model_replay')

    def test_worker_pending_before_activation(self):
        self.plan['latest_completion']={'status':'pending'}
        with self.assertRaises(RuntimeError):worker.run(self.context,'/synthetic/new/cpu-rank0')
        self.context.activate.assert_not_called();self.assert_no_native()

    def test_worker_metadata_watch_and_v2_scientific_extension(self):
        with patch.object(latest,'completed',return_value=self.done),patch.object(worker,'authenticate_outer_drain',return_value={'/synthetic/meta':{'inode':1}}):
            result=worker.load_binding(self.context)
            self.assertEqual(result[:3],(self.collector,self.native,self.previous))
            self.context.sdk_guard.watch_metadata.assert_called_once_with({'/synthetic/meta':{'inode':1}})
            self.context.sdk_guard.full.assert_called_with('completed_metadata_authenticated')
            self.plan['arbitrary_extra']=1
            with self.assertRaises(ValueError):worker.load_binding(self.context)

    def test_worker_native_imports_are_mocked_and_boundary_checks_preserved(self):
        events=[]
        dist=types.SimpleNamespace(init_process_group=lambda k:events.append(('gloo',k)),get_rank=lambda:0,
            get_world_size=lambda:2,destroy_process_group=lambda:events.append('destroy_dist'))
        fake_torch=types.SimpleNamespace(distributed=dist,set_num_threads=lambda n:events.append(('threads',n)),
            inference_mode=lambda:contextlib.nullcontext())
        parallel=types.SimpleNamespace(initialize_model_parallel=lambda n,**kw:events.append(('tp',n,kw)),
            destroy_model_parallel=lambda:events.append('destroy_tp'))
        original_import=builtins.__import__
        def imp(name,*args,**kw):
            if name=='torch':events.append('fake_torch');return fake_torch
            if name=='neuronx_distributed.parallel_layers':return types.SimpleNamespace(parallel_state=parallel)
            if name.startswith(('torch','numpy','neuron')):raise AssertionError('Real native import forbidden')
            return original_import(name,*args,**kw)
        with patch.object(worker,'load_binding',return_value=(self.collector,self.native,self.previous,self.guard)),\
            patch.object(builtins,'__import__',side_effect=imp),patch.object(worker.importlib,'import_module'),\
            patch.object(worker.driver,'run_rank',return_value={'complete':True})as driver,\
            patch.object(worker.source_auth,'authenticate_native_sources',side_effect=AssertionError('worker native hash'))as native:
            result=worker.run(self.context,'/synthetic/new/cpu-rank0')
            self.assertEqual(result['latest_completion'],self.binding)
            self.assertEqual([x.args[0]for x in self.context.sdk_guard.full.call_args_list],
                ['before_native_import','before_model_replay','after_model_replay'])
            self.assertEqual(self.context.verify_loaded_sources.call_count,3);native.assert_not_called()
        self.assertEqual(events[-2:],['destroy_tp','destroy_dist']);self.assert_no_native()

    def test_worker_missing_late_source_pin_rejects_before_native_import(self):
        self.context.verify_loaded_sources.side_effect=RuntimeError('unpinned late helper')
        with patch.object(worker,'load_binding',return_value=(self.collector,self.native,self.previous,self.guard)):
            with self.assertRaisesRegex(RuntimeError,'unpinned late'):worker.run(self.context,'/synthetic/new/cpu-rank0')
        self.context.activate.assert_not_called();self.assert_no_native()

    def test_resources_and_critical_cleanup_logic_unchanged(self):
        self.assertEqual(run.PROPS,old.PROPS);self.assertEqual((run.CAP,run.RANK_CAP),(old.CAP,old.RANK_CAP))
        def funcs(module):
            return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
        new,before=funcs(run),funcs(old)
        for name in ('usage','argv','service_command','validate_service','validate_service_identity','secondary_error',
            'guarded_receipt','cleanup_children','compact_drain'):
            self.assertEqual(new[name],before[name],name)
        self.assertTrue(sys.dont_write_bytecode);self.assert_no_native()

    def outer_case(self,fail_native=False):
        events=[];receipts={};owner={'pid':os.getpid()};lock=self.done['lock_identity']
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)/'new';spec=dict(self.spec,output_root=str(root),unit='synthetic',master_port=29531,
                root_floor_bytes=1,output_floor_bytes=1,source_sha256={},canonical_sha256={})
            def once(path,value):
                events.append(('write',Path(path).name));receipts[Path(path).name]=copy.deepcopy(value)
            def reference(path):return {'path':str(path),'sha256':'f'*64}
            def native(partition,actual_owner,actual_lock,attempt,guard):
                guard();events.append('native');self.assertEqual(actual_owner,owner);self.assertEqual(actual_lock,lock)
                if fail_native:raise RuntimeError('native failure')
                return {'schema':'synthetic','attempt_id':attempt}
            def drain(*a):events.append('full_physical');return {'complete':True}
            def launch(*a,**kw):events.append('launch');raise RuntimeError('stop before actual launch')
            with contextlib.ExitStack()as stack:
                for patcher in (patch.object(run,'read_bundle',return_value=(spec,self.plan)),patch.object(latest,'require_latest_terminal'),
                patch.object(run.p,'user_bus'),patch.object(run.post.external,'acquire_lock',return_value=(99,owner,lock)),
                patch.object(run.post.external,'release_lock',side_effect=lambda fd:events.append('release')),
                patch.object(run.p,'lock_proof'),patch.object(run.p,'clear_window'),patch.object(run.p,'active_heavy'),
                patch.object(run.p,'gone'),patch.object(run,'check_prerequisites',return_value=self.done),
                patch.object(run,'predecessor_drains',side_effect=drain),patch.object(run.p,'control_usage'),
                patch.object(run.p,'properties',return_value={'LoadState':'not-found'}),
                patch.object(run.os,'statvfs',return_value=types.SimpleNamespace(f_bavail=2**40,f_frsize=1)),
                patch.object(run,'usage'),patch.object(run.source_auth,'classify',return_value={'full':'partition'}),
                patch.object(run.source_auth,'authenticate_native_sources',side_effect=native),
                patch.object(run,'verify_loaded_sources'),patch.object(run.p,'once',side_effect=once),patch.object(run,'reference',side_effect=reference),
                patch.object(run,'service_command',return_value=['synthetic']),patch.object(run.p,'command',side_effect=launch),
                patch.object(run.signal,'signal'),patch.object(run.signal,'getsignal'),
                patch.object(run,'cleanup_owned_service',side_effect=lambda *a:events.append('cleanup')),
                patch.object(run,'verify_owned_drain',side_effect=lambda *a:(events.append('verify_drain')or {'physically_drained':True}))):
                    stack.enter_context(patcher)
                auth=run.source_auth.authenticate_native_sources
                with self.assertRaisesRegex(RuntimeError,'native failure'if fail_native else 'stop before'):run.outer({}, {})
            self.assertEqual(auth.call_count,1)
            self.assertLess(events.index('full_physical'),events.index('native'))
            if fail_native:
                self.assertNotIn('launch',events);self.assertNotIn('admission.json',receipts)
                self.assertNotIn('source-authentication.json',receipts)
                self.assertLess(events.index('verify_drain'),events.index('release'));return
            self.assertLess(events.index('native'),events.index(('write','source-authentication.json')))
            self.assertLess(events.index(('write','source-authentication.json')),events.index(('write','admission.json')))
            self.assertLess(events.index(('write','admission.json')),events.index('launch'))
            self.assertLess(events.index('verify_drain'),events.index('release'))
            a=receipts['admission.json'];self.assertEqual(a['source_authentication'],reference(root/'source-authentication.json'))
            self.assertEqual(a['source_contract'],self.contract)
            self.assertEqual(a['attempt_id'],receipts['source-authentication.json']['attempt_id'])

    def test_rank_guard_and_native_receipt_permit_bound_before_worker(self):
        admission={'path':'/synthetic/new/admission.json','sha256':'a'*64}
        a=dict(outer_owner={'pid':2},lock_identity=self.done['lock_identity'],attempt_id='b'*32,
            plan={'path':'/synthetic/plan.json','sha256':'c'*64},
            observed_drains={'path':'/synthetic/new/predecessor-drain.json','sha256':'d'*64},
            source_authentication={'path':'/synthetic/new/source-authentication.json','sha256':'e'*64})
        spec=dict(output_root='/synthetic/new',unit='trn1-mini-layer1-sdk-batch-layout-synthetic',master_port=29531,root_floor_bytes=1)
        group='0::'+run.p.GROUP_PREFIX+spec['unit']+'.service'
        co=dict(pid=3,start_ticks='100',ppid=2,argv=run.argv('coordinator',admission),cgroup=group,unit=spec['unit'],invocation_id='e'*32)
        process=dict(pid=os.getpid(),start_ticks='101',ppid=3,argv=run.argv('rank',admission),cgroup=group)
        events=[]
        class Context:
            def __init__(self,plan,sha,permit,guard):self.plan=plan;self.permit=permit;self.guard=guard;self._sources_checked=False
            def verify_sources(self):events.append('sources');self._sources_checked=True
        def actual_worker(context,directory):
            self.assertTrue(context._sources_checked)
            self.assertEqual(context.permit['source_authentication'],a['source_authentication'])
            self.assertTrue(callable(context.verify_loaded_sources));events.append('worker')
            return {'complete':True,'rank':0,'latest_completion':self.binding}
        def proc(pid):return process if pid==os.getpid()else {k:co[k]for k in process}
        with patch.object(run,'child_admission',return_value=(a,spec,self.plan)),patch.dict(os.environ,{'RANK':'0'}),\
            patch.object(run,'wait_file',return_value=process),patch.object(run.p,'proc',side_effect=proc),\
            patch.object(run,'reference',return_value={}),patch.object(run.source_auth,'load',return_value=co),\
            patch.object(run.p,'lock_proof'),patch.object(run,'admitted_bundles',return_value=(self.done,Mock())),\
            patch.object(run.p,'gone'),patch.object(run.p,'clear_window'),patch.object(run,'usage'),\
            patch.object(run,'RuntimeContext',Context),patch.object(run,'verify_loaded_sources'),\
            patch.object(worker,'run',side_effect=actual_worker),patch.object(run,'validate_rank_result'),patch.object(run.p,'once')as save:
            run.rank(admission)
            self.assertEqual(events,['sources','worker'])
            proof=save.call_args.args[1]['guard_checks']
            self.assertTrue(proof['every_source_fully_authenticated'])
            self.assertEqual([e['boundary']for e in proof['events']],
                ['before_source_verification','after_source_verification','after_worker','before_rank_output'])

    def test_source_only_reference_supports_exact_named_large_envelopes(self):
        path=Q/'minicpm5-layer1-sdk-batch-layout-synthetic-spec.json';raw=b'{"synthetic":true}'
        with patch.object(run.source_auth.metadata,'stable_bytes',return_value=raw)as read,\
            patch.object(run.source_auth,'envelope_shape')as shape:
            self.assertEqual(run.reference(path),{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
            read.assert_called_once_with(str(path),run.source_auth.metadata.ENVELOPE_LIMIT)
            shape.assert_called_once_with(str(path),{'synthetic':True})

    def test_owned_drain_still_checks_service_coordinator_and_both_ranks(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);owners=[{'pid':10,'start_ticks':'1'},{'pid':11,'start_ticks':'2'},{'pid':12,'start_ticks':'3'}]
            for rank,owner in enumerate(owners[1:]):(root/('rank%d-owner.json'%rank)).write_text(json.dumps(owner))
            with patch.object(run.p,'properties',return_value={'LoadState':'loaded'}),patch.object(run,'validate_service_identity')as service,\
                patch.object(run.p,'terminal',return_value=True),patch.object(run.p,'gone')as gone,\
                patch.object(run.p,'empty_group',return_value={'empty':True}),patch.object(run.p,'clear_window'):
                proof=run.verify_owned_drain({'output_root':directory,'unit':'synthetic'}, {},'attempt',owners[0],True)
                service.assert_called_once();self.assertEqual(proof['owners_checked'],owners)
                self.assertEqual([c.args[0]for c in gone.call_args_list],owners)
                self.assertTrue(proof['physically_drained'])

    def test_outer_failure_keeps_lock_until_real_cleanup_guard_verifies(self):
        events=[]
        with tempfile.TemporaryDirectory()as directory:
            spec={'output_root':str(Path(directory)/'new'),'latest_completion':self.binding}
            def verify(*a):
                events.append('verify')
                if events.count('verify')==1:raise RuntimeError('still live')
                return {'physically_drained':True}
            with patch.object(run,'read_bundle',return_value=(spec,{})),patch.object(latest,'require_latest_terminal'),\
                patch.object(run.p,'user_bus'),patch.object(run.post.external,'acquire_lock',return_value=(99,{},{})),\
                patch.object(run.post.external,'release_lock',side_effect=lambda fd:events.append('release')),\
                patch.object(run.p,'lock_proof'),patch.object(run.p,'clear_window'),patch.object(run.p,'active_heavy'),\
                patch.object(run.signal,'signal'),patch.object(run.signal,'getsignal'),\
                patch.object(run,'check_prerequisites',side_effect=RuntimeError('original failure')),\
                patch.object(run,'cleanup_owned_service',side_effect=lambda *a:events.append('cleanup')),\
                patch.object(run,'verify_owned_drain',side_effect=verify),patch.object(run.p,'command')as launch,\
                patch.object(run.source_auth,'authenticate_native_sources')as native:
                with self.assertRaisesRegex(RuntimeError,'original failure'):run.outer({}, {})
                native.assert_not_called();launch.assert_not_called()
            self.assertEqual(events,['cleanup','verify','cleanup','verify','release'])

    def test_outer_native_auth_once_after_drain_before_admission_and_launch(self):
        self.outer_case(False)

    def test_native_failure_blocks_admission_and_launch_and_preserves_cleanup(self):
        self.outer_case(True)

    def test_source_complete_claim_requires_context_verified(self):
        guard,co=self.guard_fixture()
        context=types.SimpleNamespace(sdk_guard=guard,_sources_checked=False)
        with self.assertRaises(RuntimeError):guard.mark_sources_verified(context)
        self.assertFalse(guard.evidence()['every_source_fully_authenticated'])
        context._sources_checked=True;guard.mark_sources_verified(context)
        self.assertTrue(guard.evidence()['every_source_fully_authenticated'])

    def test_worker_actual_hashed_runtime_receipts_bind_v2_source_contract(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);self.plan['output_root']=directory
            owner={'pid':999,'start_ticks':'100'};lock=self.done['lock_identity']
            def save(name,value):
                raw=json.dumps(value).encode();path=root/name;path.write_bytes(raw)
                return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            proof=dict(complete=True,latest_completion=self.binding,outer_owner=owner,lock_identity=lock,
                latest_physical=dict(complete=True,physical_checks_performed=True,lock_identity=lock,
                    all_additional_owners_checked=self.done['owners'],all_current_and_prior_services_preserved=True,
                    nemotron={'fresh_physical_drain_verified':True,'services':1308}))
            drain=save('predecessor-drain.json',proof);pref={'path':'/synthetic/plan.json','sha256':'a'*64}
            nref={'path':str(root/'source-authentication.json'),'sha256':'b'*64}
            a=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2',
                outer_owner=owner,lock_identity=lock,plan=pref,observed_drains=drain,attempt_id='a'*32,
                latest_completion=self.binding,source_contract=self.contract,source_authentication=nref,
                runtime_admitted=True,predecessors_checked=True,full_model_qualified=False,new_speedup_claim=False)
            aref=save('admission.json',a)
            self.context.permit.update(runtime_admission=aref,runtime_plan=pref,outer_owner=owner,
                lock_identity=lock,observed_drains=drain,attempt_id='a'*32,source_authentication=nref)
            with patch.object(worker.p,'lock_proof')as lease,patch.object(worker.p,'gone')as gone,\
                patch.object(worker.p,'clear_window'),patch.object(latest,'physical',side_effect=AssertionError('worker must not idle-check self')),\
                patch.object(worker.source_auth,'authenticate_native_sources',side_effect=AssertionError('worker must not hash native')):
                watched=worker.authenticate_outer_drain(self.context,self.done)
                self.assertEqual(set(watched),{str(root/'admission.json'),str(root/'predecessor-drain.json')})
                lease.assert_called_once_with(owner,lock);gone.assert_called_once_with(self.done['owners'][0])
                self.context._inherited_native_receipt['source_contract']={}
                with self.assertRaises(ValueError):worker.authenticate_outer_drain(self.context,self.done)
                self.assert_no_native()

    def test_result_adapters_keep_complete_bundles_reader_guard_interface(self):
        result={'latest_completion':self.binding};reader=Mock()
        with patch.object(run,'admitted_bundles',return_value=(self.done,reader)),\
            patch.object(run.layer1_results,'validate_rank_result',return_value={'rank':'verified'})as rank,\
            patch.object(run.layer1_results,'validate_rank_pair',return_value={'pair':'verified'})as pair:
            self.assertEqual(run.validate_rank_result(result,self.plan,1,self.guard),{'rank':'verified'})
            rank.assert_called_once_with(result,self.plan,1,self.collector,self.native,self.previous,reader,self.guard)
            self.assertEqual(run.validate_rank_pair([result,result],self.plan,self.guard),{'pair':'verified'})
            pair.assert_called_once_with([result,result],self.plan,self.collector,self.native,self.previous,reader,self.guard)

    def test_child_admission_verifies_lock_before_source_only_bundle_and_preserves_contract(self):
        a=dict(outer_owner={'pid':1},lock_identity={'inode':2},spec={'path':'spec'},review={'path':'review'},
            plan={'path':'plan'},source_contract=self.contract,schema='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2')
        spec=dict(plan=a['plan'],source_contract=self.contract);events=[]
        def lock(*args):events.append('lock')
        def bundle(sr,rr,guard):
            self.assertEqual(events,['lock']);guard();events.append('bundle');return spec,self.plan
        with patch.object(run.source_auth,'load',return_value=a),patch.object(run.p,'lock_proof',side_effect=lock),\
            patch.object(run,'read_bundle',side_effect=bundle):
            self.assertEqual(run.child_admission({'path':'admission'}),(a,spec,self.plan))
            self.assertEqual(events,['lock','lock','bundle','lock'])

if __name__=='__main__':unittest.main(verbosity=2)
