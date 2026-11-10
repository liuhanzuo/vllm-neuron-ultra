"""Synthetic lifecycle integration only; no native dependencies or real runtime I/O.

The completion implementation is parent-owned. A clearly synthetic substitute
is installed only in this test process; no complete real binding is constructed.
"""
import ast
import builtins
import contextlib
import copy
import importlib
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
NAME='research.minicpm5_layer1_sdk_batch_layout_completion_v1'
stub=types.ModuleType(NAME)
def pending_guard(binding):
    if not isinstance(binding,dict) or binding.get('status')!='complete':
        raise RuntimeError('Synthetic completion interface rejects pending')
stub.require_complete_binding=pending_guard
stub.require_latest_terminal=Mock()
stub.completed=Mock(side_effect=RuntimeError('No real completion admitted'))
stub.physical=Mock(side_effect=RuntimeError('No real physical operations allowed'))
sys.modules[NAME]=stub
import minicpm5_layer1_sdk_batch_layout_cpu_runner_v1 as run
import minicpm5_layer1_sdk_batch_layout_cpu_worker_v1 as worker
import minicpm5_layer1_current_history_factorial_cpu_runner_v1 as old
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
AUTH_OUTER=worker.authenticate_outer_drain


class Checks(unittest.TestCase):
    def setUp(self):
        self.guard=Mock()
        self.binding={'status':'complete','synthetic':True}
        self.native={'binding':{'native':'synthetic'}}
        self.previous={'binding':{'previous':'synthetic'},'spec':{},'records':{}}
        self.original={'scientific':'unchanged','inputs':{},'source_sha256':{}}
        cs={k:{'synthetic':k}for k in ('finalization','catalog','finalizer_exit','canonical_sha256','nemotron_completion')}
        cs['unit']='synthetic-collector'
        self.collector=dict(binding={'collector':'synthetic'},collector_plan=self.original,collector_spec=cs,
            collector_chain={'rank_results':[{'rank':0},{'rank':1}]},audit_result={'audit':'synthetic'},
            saved_audit_spec={'source_sha256':{}})
        self.done=dict(collector=self.collector,native=self.native,current_history=self.previous,
            owners=[{'pid':11,'start_ticks':'2'}],lock_identity={'inode':3},source_sha256={},binding=self.binding,
            bound={'spec':{'synthetic':True},'finalizer_owner':{'pid':12}})
        self.plan=dict(copy.deepcopy(self.original),output_root='/synthetic/new',latest_completion=self.binding,
            layer1_sdk_batch_layout=self.previous['binding'],layer1_official_history=self.native['binding'])
        self.previous['plan']={k:copy.deepcopy(v)for k,v in self.plan.items()if k not in ('latest_completion','layer1_sdk_batch_layout')}
        self.spec=dict(cs,plan={'path':'synthetic-plan'},latest_completion=self.binding,
            completed_current_history=self.previous['binding'],completed_native_cache=self.native['binding'],
            completed_layer1_data=self.collector['binding'])
        stub.completed.reset_mock();stub.physical.reset_mock();stub.require_latest_terminal.reset_mock()
        stub.require_latest_terminal.side_effect=None
        stub.completed.side_effect=None;stub.completed.return_value=self.done
        stub.physical.side_effect=None;stub.physical.return_value={'synthetic_physical':True}
        self.context=types.SimpleNamespace(_sources_checked=True,plan=self.plan,permit={'rank':0},
            guard=Mock(),read_json=Mock(),activate=Mock(return_value=contextlib.nullcontext()),record_cpu_import=Mock())
        runtime_patch=patch.object(worker,'authenticate_outer_drain',return_value={})
        self.runtime_drain=runtime_patch.start();self.addCleanup(runtime_patch.stop)

    def assert_no_native(self):
        self.assertFalse(any(n=='torch' or n.startswith('torch.')or n=='numpy'or n.startswith('numpy.')for n in sys.modules))

    def test_no_native_imports(self):
        self.assert_no_native()

    def test_original_resources_and_caps(self):
        self.assertEqual(run.PROPS,old.PROPS)
        self.assertEqual((run.CAP,run.RANK_CAP,run.PYTHON),(old.CAP,old.RANK_CAP,old.PYTHON))
        environment=run.env('x'*32,29531,1)
        self.assertEqual({k:environment[k]for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT')},
            dict(RANK='1',LOCAL_RANK='1',WORLD_SIZE='2',LOCAL_WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT='29531'))
        self.assertEqual(environment['NXD_CPU_MODE'],'1')
        self.assertEqual(environment['NEURON_RT_VISIBLE_CORES'],old.env('x'*32,29531,1)['NEURON_RT_VISIBLE_CORES'])
        self.assertEqual(environment['OMP_NUM_THREADS'],'1')

    def test_original_owner_cleanup_code_unchanged(self):
        def functions(module):
            source=Path(module.__file__).read_text()
            return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
        new,previous=functions(run),functions(old)
        for name in ('usage','argv','service_command','validate_service','validate_service_identity',
            'secondary_error','guarded_receipt','cleanup_children','wait_file','child_admission',
            'coordinator','verify_owned_drain','cleanup_owned_service'):
            self.assertEqual(new[name],previous[name],name)
        for name in ('rank','outer'):
            self.assertIn(name,new)

    def test_pending_outer_fails_before_lock_output_launch(self):
        for value in ({'status':'pending'},None,{}):
            with self.subTest(value=value),patch.object(run.p,'read_ref',return_value={'latest_completion':value}),\
                patch.object(run.post.external,'acquire_lock')as lock,patch.object(run.p,'command')as launch,\
                patch.object(run.p,'user_bus')as bus,patch.object(run.Path,'mkdir')as mkdir:
                with self.assertRaises((RuntimeError,ValueError)):run.outer({}, {})
                lock.assert_not_called();launch.assert_not_called();bus.assert_not_called();mkdir.assert_not_called()

    def test_pending_worker_blocks_activation_and_import(self):
        self.plan['latest_completion']={'status':'pending'}
        with self.assertRaises((RuntimeError,ValueError)):worker.run(self.context,'/synthetic/new/cpu-rank0')
        self.context.activate.assert_not_called();stub.completed.assert_not_called();stub.physical.assert_not_called()
        self.assert_no_native()

    def test_worker_sources_checked_before_binding(self):
        self.context._sources_checked=False
        with self.assertRaises((RuntimeError,ValueError)):worker.run(self.context,'/synthetic/new/cpu-rank0')
        stub.completed.assert_not_called();self.context.activate.assert_not_called()

    def test_worker_exact_rank_directory(self):
        with self.assertRaises((RuntimeError,ValueError)):worker.run(self.context,'/synthetic/new/cpu-rank1')
        stub.completed.assert_not_called()

    def test_worker_latest_binding_guard_and_physical(self):
        result=worker.load_binding(self.context)
        self.assertEqual(result[:3],(self.collector,self.native,self.previous))
        stub.physical.assert_not_called();self.runtime_drain.assert_called_once_with(self.context,self.done)
        self.context.guard.assert_any_call('cpu')
        result[3]()
        self.assert_no_native()

    def test_worker_original_scientific_plan_and_sources_reject(self):
        for modify in ('science','inputs','sources','lineage','binding'):
            with self.subTest(modify=modify):
                saved=copy.deepcopy(self.plan);done=copy.deepcopy(self.done)
                if modify=='science':self.plan['scientific']='changed'
                if modify=='inputs':self.original['inputs']['old']={'keep':True}
                if modify=='sources':self.original['source_sha256']['old']='a'
                if modify=='lineage':self.done['source_sha256']['lineage']='b'
                if modify=='binding':self.plan['layer1_sdk_batch_layout']={}
                stub.physical.reset_mock()
                with self.assertRaises((RuntimeError,ValueError)):worker.load_binding(self.context)
                stub.physical.assert_not_called()
                self.plan.clear();self.plan.update(saved)
                self.original['inputs'].clear();self.original['source_sha256'].clear()
                self.done['source_sha256']=done['source_sha256']

    def test_worker_metadata_read_and_identity_recheck(self):
        ref={'path':'/synthetic/metadata.json','sha256':'a'*64}
        self.plan['inputs'][ref['path']]={'sha256':ref['sha256'],'identity':{'inode':1}}
        self.context.read_json.return_value={'synthetic':True}
        def completed(binding,read,guard):
            guard();self.assertEqual(read(ref),{'synthetic':True});return self.done
        with patch.object(stub,'completed',side_effect=completed),patch.object(worker,'identity',return_value={'inode':1})as ident:
            *_,guard=worker.load_binding(self.context);guard()
            self.context.read_json.assert_called_with(ref['path'],ref['sha256'])
            ident.return_value={'inode':2}
            with self.assertRaises((RuntimeError,ValueError)):guard()

    def test_worker_tp2_setup_driver_and_teardown_synthetic(self):
        events=[]
        dist=types.SimpleNamespace(init_process_group=lambda kind:events.append(('init',kind)),
            get_rank=lambda:0,get_world_size=lambda:2,destroy_process_group=lambda:events.append('destroy_dist'))
        fake_torch=types.SimpleNamespace(distributed=dist,set_num_threads=lambda n:events.append(('threads',n)),
            inference_mode=lambda:contextlib.nullcontext())
        parallel=types.SimpleNamespace(initialize_model_parallel=lambda n,**kw:events.append(('tp',n,kw)),
            destroy_model_parallel=lambda:events.append('destroy_tp'))
        original_import=builtins.__import__
        def fake_import(name,*args,**kwargs):
            if name=='torch':events.append('import_fake_torch');return fake_torch
            if name=='neuronx_distributed.parallel_layers':return types.SimpleNamespace(parallel_state=parallel)
            if name.startswith(('torch','numpy','neuron')):raise AssertionError('Real native import prohibited')
            return original_import(name,*args,**kwargs)
        result={'complete':True}
        with patch.object(builtins,'__import__',side_effect=fake_import),\
            patch.object(worker.importlib,'import_module',return_value=None),\
            patch.object(worker.driver,'run_rank',return_value=result)as driver:
            self.assertIs(worker.run(self.context,'/synthetic/new/cpu-rank0'),result)
            args=driver.call_args.args
            self.assertIs(args[1],fake_torch);self.assertIs(args[2],self.original)
            self.assertIs(args[-2],self.native);self.assertIs(args[-1],self.previous)
        self.assertEqual(events[-2:],['destroy_tp','destroy_dist'])
        self.assertIn(('tp',2,{'skip_collective_init':True}),events)
        self.assertEqual(result['latest_completion'],self.binding);self.assert_no_native()

    def test_worker_driver_failure_cleans_both_groups_synthetic(self):
        events=[]
        dist=types.SimpleNamespace(init_process_group=lambda kind:None,get_rank=lambda:0,get_world_size=lambda:2,
            destroy_process_group=lambda:events.append('dist'))
        parallel=types.SimpleNamespace(initialize_model_parallel=lambda *a,**kw:None,destroy_model_parallel=lambda:events.append('tp'))
        fake_torch=types.SimpleNamespace(distributed=dist,set_num_threads=lambda n:None,inference_mode=lambda:contextlib.nullcontext())
        original_import=builtins.__import__
        def imp(name,*args,**kwargs):
            if name=='torch':return fake_torch
            if name=='neuronx_distributed.parallel_layers':return types.SimpleNamespace(parallel_state=parallel)
            return original_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=imp),patch.object(worker.importlib,'import_module'),\
            patch.object(worker.driver,'run_rank',side_effect=RuntimeError('synthetic driver failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic driver'):worker.run(self.context,'/synthetic/new/cpu-rank0')
        self.assertEqual(events,['tp','dist']);self.assert_no_native()

    def test_runner_metadata_prerequisites_no_legacy_physical(self):
        def read(ref):return self.plan if ref==self.spec['plan']else {'source_sha256':{}}
        with patch.object(run.p,'read_ref',side_effect=read),patch.object(run.collector,'check_prerequisites')as legacy:
            self.assertIs(run.check_prerequisites(self.spec,self.guard),self.done)
            legacy.assert_not_called();stub.physical.assert_not_called()
            self.plan['scientific']='changed'
            with self.assertRaises((RuntimeError,ValueError)):run.check_prerequisites(self.spec,self.guard)

    def test_latest_physical_only(self):
        with patch.object(run.collector,'predecessor_drains')as legacy:
            self.assertEqual(run.predecessor_drains(self.done,self.guard),{'synthetic_physical':True})
            legacy.assert_not_called();stub.physical.assert_called_once_with(self.done,self.guard)

    def test_complete_proof_compaction_preserves_units_owners_samples(self):
        proof={'owners':[{'pid':i,'start_ticks':str(i)}for i in range(8)],'nested':{
            'services':[{'unit':'unit%d'%i,'samples':[{'properties':{'Id':'unit%d'%i,'MainPID':'0','Other':'large'},
                'group':{'exists':False,'pids':[]},'owner':{'pid':i}}for j in range(2)]}for i in range(53)]}}
        result=run.compact_drain(proof)
        self.assertEqual(result['complete_observation_sha256'],run.digest(proof))
        self.assertEqual(result['latest_physical']['owners'],proof['owners'])
        for original,row in zip(proof['nested']['services'],result['latest_physical']['nested']['services']):
            self.assertEqual(row['unit'],original['unit']);self.assertEqual(len(row['samples']),2)
            for sample in row['samples']:
                self.assertEqual(sample['group'],{'exists':False,'pids':[]})
                self.assertNotIn('Other',sample['properties'])
                self.assertIn('owner',sample)

    def test_result_adapters_forward_authenticated_bundles_and_guard(self):
        reader=Mock();result={'latest_completion':self.binding}
        with patch.object(run,'admitted_bundles',return_value=(self.done,reader)),\
            patch.object(run.layer1_results,'validate_rank_result',return_value={'rank':'checked'})as one,\
            patch.object(run.layer1_results,'validate_rank_pair',return_value={'pair':'checked'})as pair:
            self.assertEqual(run.validate_rank_result(result,self.plan,1,self.guard),{'rank':'checked'})
            one.assert_called_once_with(result,self.plan,1,self.collector,self.native,self.previous,reader,self.guard)
            self.assertEqual(run.validate_rank_pair([result,result],self.plan,self.guard),{'pair':'checked'})
            pair.assert_called_once_with([result,result],self.plan,self.collector,self.native,self.previous,reader,self.guard)
            with self.assertRaises((RuntimeError,ValueError)):run.validate_rank_result({'latest_completion':{}},self.plan,0,self.guard)

    def test_admitted_reader_digest_identity_and_source_pins(self):
        ref={'path':'/synthetic/data.json','sha256':'a'*64};ident={'inode':1}
        self.plan['inputs'][ref['path']]={'sha256':ref['sha256'],'identity':ident}
        def complete(binding,read,guard):
            self.assertEqual(read(ref),{'data':True});return self.done
        with patch.object(stub,'completed',side_effect=complete),patch.object(run,'identity',return_value=ident)as identity,\
            patch.object(run.p,'read_ref',return_value={'data':True}):
            done,reader=run.admitted_bundles(self.plan,self.guard)
            self.assertIs(done,self.done);self.assertEqual(reader(ref['path'],ref['sha256']),{'data':True})
            identity.return_value={'inode':2}
            with self.assertRaises((RuntimeError,ValueError)):run.admitted_bundles(self.plan,self.guard)
            identity.return_value=ident;self.done['source_sha256']['/missing']='sha'
            with self.assertRaises((RuntimeError,ValueError)):run.admitted_bundles(self.plan,self.guard)

    def test_coordinator_parent_ack_blocks_spawning(self):
        a={'outer_owner':{},'lock_identity':{}}
        with patch.object(run,'child_admission',return_value=(a,{'output_root':'/synthetic'},{})),\
            patch.object(run,'wait_file',return_value=dict(pid=3,start_ticks='1',ppid=2,argv=[],cgroup='g')),\
            patch.object(run.p,'proc',return_value=dict(pid=4,start_ticks='1',ppid=2,argv=[],cgroup='g')),\
            patch.object(run.subprocess,'Popen')as spawn:
            with self.assertRaises((RuntimeError,ValueError)):run.coordinator({})
            spawn.assert_not_called()

    def test_coordinator_acks_both_actual_children_and_waits(self):
        a={'outer_owner':{},'lock_identity':{},'attempt_id':'x'*32}
        spec={'output_root':'/synthetic','master_port':29531,'root_floor_bytes':1}
        admission={'path':'/synthetic/admission.json','sha256':'a'*64}
        owner=dict(pid=3,start_ticks='1',ppid=2,argv=[],cgroup='g')
        children=[types.SimpleNamespace(pid=10+r,returncode=0,poll=Mock(side_effect=[None,0,0,0,0]),wait=Mock(return_value=0))for r in (0,1)]
        records=[dict(pid=10+r,start_ticks='1',ppid=3,argv=run.argv('rank',admission),cgroup='g')for r in (0,1)]
        outputs=[dict(process=records[r],complete=True,rank=r)for r in (0,1)]
        def proc(pid):return owner if pid==os.getpid()else records[pid-10]
        def read(path):return outputs[0 if 'rank0' in str(path)else 1]
        with patch.object(run,'child_admission',return_value=(a,spec,self.plan)),patch.object(run,'wait_file',return_value=owner),\
            patch.object(run.p,'proc',side_effect=proc),patch.object(run.p,'lock_proof'),patch.object(run.signal,'signal'),\
            patch.object(run.subprocess,'Popen',side_effect=children)as spawn,patch.object(run.p,'once')as save,\
            patch.object(run.p,'read',side_effect=read),patch.object(run,'validate_rank_pair',return_value={'checked':True})as verify:
            run.coordinator(admission)
            self.assertEqual(spawn.call_count,2);verify.assert_called_once_with(outputs,self.plan,unittest.mock.ANY)
            names=[str(c.args[0])for c in save.call_args_list]
            self.assertIn('/synthetic/rank0-owner.json',names);self.assertIn('/synthetic/rank1-owner.json',names)
            self.assertIn('/synthetic/children-exits.json',names)
            for child in children:self.assertTrue(child.wait.called)

    def test_rank_parent_ack_blocks_runtime_context(self):
        a={'outer_owner':{},'lock_identity':{}}
        with patch.object(run,'child_admission',return_value=(a,{'output_root':'/synthetic'},{})),\
            patch.dict(os.environ,{'RANK':'0'}),patch.object(run,'wait_file',return_value={'pid':3}),\
            patch.object(run.p,'proc',return_value={'pid':4}),patch.object(run,'RuntimeContext')as native:
            with self.assertRaises((RuntimeError,ValueError)):run.rank({})
            native.assert_not_called()

    def test_owned_drain_checks_coordinator_and_both_ranks(self):
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory)
            for rank in (0,1):(root/('rank%d-owner.json'%rank)).touch()
            owners=[{'pid':5},{'pid':6},{'pid':7}]
            value=dict(LoadState='loaded')
            with patch.object(run.p,'properties',return_value=value),patch.object(run,'validate_service_identity'),\
                patch.object(run.p,'terminal',return_value=True),patch.object(run.p,'gone')as gone,\
                patch.object(run.p,'read',side_effect=owners[1:]),patch.object(run.p,'empty_group',return_value={'empty':True}):
                proof=run.verify_owned_drain({'output_root':directory,'unit':'synthetic'},{},'attempt',owners[0],True)
                self.assertEqual(proof['owners_checked'],owners)
                self.assertEqual([c.args[0]for c in gone.call_args_list],owners)
                self.assertTrue(proof['physically_drained'])

    def test_outer_keeps_lock_until_cleanup_drain_proven_on_failure(self):
        events=[]
        with tempfile.TemporaryDirectory()as directory:
            spec={'output_root':str(Path(directory)/'new'),'latest_completion':self.binding}
            def check(*a):events.append('prerequisite_failure');raise RuntimeError('original failure')
            def verify(*a):
                events.append('verify')
                if events.count('verify')==1:raise RuntimeError('still live')
                return {'physically_drained':True}
            with patch.object(run,'read_bundle',return_value=(spec,{})),patch.object(run.p,'user_bus'),\
                patch.object(run.post.external,'acquire_lock',return_value=(99,{},{})),\
                patch.object(run.post.external,'release_lock',side_effect=lambda fd:events.append('release')),\
                patch.object(run.p,'lock_proof'),patch.object(run.p,'clear_window'),patch.object(run.p,'active_heavy'),\
                patch.object(run.signal,'signal'),patch.object(run.signal,'getsignal'),\
                patch.object(run,'check_prerequisites',side_effect=check),\
                patch.object(run,'cleanup_owned_service',side_effect=lambda *a:events.append('cleanup')),\
                patch.object(run,'verify_owned_drain',side_effect=verify),patch.object(run.p,'command')as launch:
                with self.assertRaisesRegex(RuntimeError,'original failure'):run.outer({}, {})
                launch.assert_not_called()
            self.assertEqual(events,['prerequisite_failure','cleanup','verify','cleanup','verify','release'])

    def test_cleanup_cannot_stop_different_owned_intent(self):
        with patch.object(run.p,'read',return_value={'command':['wrong'],'admission':{}}),patch.object(run.p,'command')as command:
            with self.assertRaises((RuntimeError,ValueError)):run.cleanup_owned_service({'output_root':'/synthetic','unit':'x','master_port':29531},{'path':'/synthetic/admission','sha256':'a'*64},'x',None,True)
            command.assert_not_called()

    def runtime_fixture(self):
        sha='a'*64
        spec_ref={'path':'synthetic-spec','sha256':sha};review_ref={'path':'synthetic-review','sha256':sha}
        pins={}
        for module in tuple(sys.modules.values()):
            path=getattr(module,'__file__',None)
            if path and str(path).startswith(str(R)+'/'):pins[str(Path(path).resolve())]=sha
        pins[str(R/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v1.py')]=sha
        plan=copy.deepcopy(self.plan)
        plan.update(source_sha256=pins,output_root='/mnt/nvme/minicpm5-layer1-sdk-batch-layout-synthetic')
        mappings={'completed_native_cache':'layer1_official_history','completed_official_history':'layer1_current_history_factorial',
            'completed_storage':'completed_storage_maintenance','completed_factorial':'layer1_native_cache',
            'completed_cross_history':'layer1_kv_factorial','completed_own_history':'layer1_cross_history',
            'completed_layer1_data':'layer1_own_history_replay','completed_current_history':'layer1_sdk_batch_layout'}
        spec=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_execution_spec_v1',latest_completion=self.binding,
            runner=run.SELF,worker=str(R/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v1.py'),source_sha256=pins,
            plan={'path':'synthetic-plan','sha256':sha},output_root=plan['output_root'],output_cap_bytes=run.CAP,
            root_floor_bytes=28*2**30,output_floor_bytes=30*2**30,resources=copy.deepcopy(run.PROPS),
            master_port=29531,unit='trn1-mini-layer1-sdk-batch-layout-synthetic',
            model_qualification_claim=False,automatic_full_gates_launch=False,
            original_completed_collection=run.OBSERVED_EXIT,prepared_source_plan=run.RESULTS_SOURCE_PLAN)
        for spec_key,plan_key in mappings.items():
            plan.setdefault(plan_key,{'synthetic':plan_key});spec[spec_key]=plan[plan_key]
        review=dict(schema='minicpm5_layer1_sdk_batch_layout_runtime_source_review_v1',reviewer='/root',
            passed=True,blockers=[],spec=spec_ref,source_sha256=pins)
        def read(ref):
            if ref==spec_ref:return spec
            if ref==review_ref:return review
            if ref==spec['plan']:return plan
            raise AssertionError('Unexpected real metadata read')
        return spec_ref,review_ref,spec,plan,review,read

    def test_reviewed_bundle_complete_sources_and_resources(self):
        sr,rr,spec,plan,review,read=self.runtime_fixture()
        with patch.object(run.p,'read_ref',side_effect=read),patch.object(run.p,'sha',return_value='a'*64),\
            patch.object(run,'identity',return_value={'size':123}):
            self.assertEqual(run.read_bundle(sr,rr),(spec,plan))
            spec['resources']=dict(run.PROPS,AllowedCPUs='0-31')
            with self.assertRaises(RuntimeError):run.read_bundle(sr,rr)

    def test_reviewed_bundle_loaded_source_and_entrypoint_pins(self):
        sr,rr,spec,plan,review,read=self.runtime_fixture()
        with patch.object(run.p,'read_ref',side_effect=read),patch.object(run.p,'sha',return_value='a'*64),\
            patch.object(run,'identity',return_value={'size':123}):
            omitted=str(Path(worker.driver.__file__).resolve());pin=spec['source_sha256'].pop(omitted)
            with self.assertRaises(RuntimeError):run.read_bundle(sr,rr)
            spec['source_sha256'][omitted]=pin
            spec['worker']=str(R/'minicpm5_layer1_current_history_factorial_cpu_worker_v1.py')
            with self.assertRaises(RuntimeError):run.read_bundle(sr,rr)

    def test_reviewed_bundle_latest_binding_and_review_required(self):
        sr,rr,spec,plan,review,read=self.runtime_fixture()
        with patch.object(run.p,'read_ref',side_effect=read),patch.object(run.p,'sha',return_value='a'*64),\
            patch.object(run,'identity',return_value={'size':123}):
            plan['latest_completion']={'different':True}
            with self.assertRaises(RuntimeError):run.read_bundle(sr,rr)
            plan['latest_completion']=spec['latest_completion'];review['passed']=False
            with self.assertRaises(RuntimeError):run.read_bundle(sr,rr)

    def test_completed_current_history_plan_all_inputs_sources_and_supplements_retained(self):
        for kind in ('input','source','supplement'):
            with self.subTest(kind=kind):
                previous=copy.deepcopy(self.previous['plan'])
                if kind=='input':self.previous['plan']['inputs']['past.json']={'sha256':'a'*64,'identity':{'inode':1}}
                if kind=='source':self.previous['plan']['source_sha256']['past.py']='a'*64
                if kind=='supplement':self.previous['plan']['completed_storage_maintenance']={'must':'retain'}
                with self.assertRaises((RuntimeError,ValueError)):worker.load_binding(self.context)
                with self.assertRaises((RuntimeError,ValueError)):run.admitted_bundles(self.plan,self.guard)
                self.previous['plan']=previous

    def test_stale_terminal_refuses_before_serial_lock(self):
        with patch.object(run,'read_bundle',return_value=({'latest_completion':self.binding},{})),\
            patch.object(stub,'require_latest_terminal',side_effect=RuntimeError('stale terminal')),\
            patch.object(run.post.external,'acquire_lock')as lock,patch.object(run.p,'command')as launch:
            with self.assertRaisesRegex(RuntimeError,'stale'):run.outer({}, {})
            lock.assert_not_called();launch.assert_not_called()

    @contextlib.contextmanager
    def actual_synthetic_runtime_receipts(self):
        # Actual tiny JSON files and hashes; every owner/service operation remains mocked.
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);self.plan['output_root']=str(root)
            owner={'pid':98765,'start_ticks':'123','argv':['synthetic-outer']}
            lock=self.done['lock_identity']
            proof=dict(complete=True,latest_completion=self.binding,outer_owner=owner,lock_identity=lock,
                latest_physical=dict(complete=True,physical_checks_performed=True,lock_identity=lock,
                    all_additional_owners_checked=self.done['owners'],all_current_and_prior_services_preserved=True,
                    nemotron={'fresh_physical_drain_verified':True,'services':1308}))
            def save(path,value):
                raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
                return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
            drain=save(root/'predecessor-drain.json',proof);planref={'path':'/synthetic/plan.json','sha256':'c'*64}
            admission=dict(schema='minicpm5_layer1_sdk_batch_layout_cpu_admission_v1',
                outer_owner=owner,lock_identity=lock,plan=planref,observed_drains=drain,
                attempt_id='a'*32,latest_completion=self.binding,runtime_admitted=True,predecessors_checked=True,
                full_model_qualified=False,new_speedup_claim=False)
            aref=save(root/'admission.json',admission)
            self.context.permit.update(runtime_admission=aref,runtime_plan=planref,
                observed_drains=drain,outer_owner=owner,lock_identity=lock,attempt_id='a'*32)
            with patch.object(worker.p,'lock_proof')as lease,patch.object(worker.p,'gone')as gone:
                yield root,proof,admission,save,lease,gone

    def test_worker_uses_actual_hashed_outer_drain_while_own_service_active(self):
        with self.actual_synthetic_runtime_receipts()as (root,proof,admission,save,lease,gone),\
            patch.object(worker,'authenticate_outer_drain',side_effect=AUTH_OUTER),\
            patch.object(worker.p,'active_heavy',side_effect=RuntimeError('own service active'))as idle:
            result=worker.load_binding(self.context)
            lease.assert_called_with(self.context.permit['outer_owner'],self.done['lock_identity'])
            self.assertEqual([c.args[0]for c in gone.call_args_list],self.done['owners'])
            idle.assert_not_called();stub.physical.assert_not_called()
            (root/'predecessor-drain.json').write_text('{}')
            with self.assertRaises((RuntimeError,ValueError)):result[3]()
            self.assert_no_native()

    def test_worker_missing_or_replaced_outer_drain_rejects(self):
        for mutation in ('missing','replaced'):
            with self.subTest(mutation=mutation),self.actual_synthetic_runtime_receipts()as (root,*rest):
                path=root/'predecessor-drain.json'
                if mutation=='missing':path.unlink()
                else:path.write_text('{}')
                with self.assertRaises((RuntimeError,ValueError,FileNotFoundError)):AUTH_OUTER(self.context,self.done)
                self.context.activate.assert_not_called();self.assert_no_native()

    def test_worker_changed_outer_owner_or_false_physical_evidence_rejects(self):
        for mutation in ('owner','physical','owners','r12'):
            with self.subTest(mutation=mutation),self.actual_synthetic_runtime_receipts()as (root,proof,admission,save,*rest):
                if mutation=='owner':admission['outer_owner']={'pid':999,'argv':['different'],'start_ticks':'456'}
                else:
                    if mutation=='physical':proof['latest_physical']['physical_checks_performed']=False
                    if mutation=='owners':proof['latest_physical']['all_additional_owners_checked']=[]
                    if mutation=='r12':proof['latest_completion']={'wrong':'r12'}
                    admission['observed_drains']=save(root/'predecessor-drain.json',proof)
                    self.context.permit['observed_drains']=admission['observed_drains']
                self.context.permit['runtime_admission']=save(root/'admission.json',admission)
                with self.assertRaises((RuntimeError,ValueError)):AUTH_OUTER(self.context,self.done)
                self.context.activate.assert_not_called()

    def test_worker_live_predecessor_owner_rejects_before_import(self):
        with self.actual_synthetic_runtime_receipts()as (root,proof,admission,save,lease,gone),\
            patch.object(worker,'authenticate_outer_drain',side_effect=AUTH_OUTER):
            gone.side_effect=RuntimeError('recorded owner still live')
            with self.assertRaisesRegex(RuntimeError,'still live'):worker.run(self.context,root/'cpu-rank0')
            self.context.activate.assert_not_called();self.assert_no_native()

    def test_rank_permit_binds_exact_outer_receipts_and_verifies_sources_before_worker(self):
        admission={'path':'/synthetic/new/admission.json','sha256':'a'*64}
        a=dict(outer_owner={'pid':2},lock_identity=self.done['lock_identity'],attempt_id='b'*32,
            plan={'path':'/synthetic/plan.json','sha256':'c'*64},
            observed_drains={'path':'/synthetic/new/predecessor-drain.json','sha256':'d'*64})
        spec=dict(output_root='/synthetic/new',unit='trn1-mini-layer1-sdk-batch-layout-synthetic',
            master_port=29531,root_floor_bytes=28*2**30)
        group='0::'+run.p.GROUP_PREFIX+spec['unit']+'.service'
        coordinator=dict(pid=3,start_ticks='100',ppid=2,argv=run.argv('coordinator',admission),cgroup=group,
            unit=spec['unit'],invocation_id='e'*32)
        process=dict(pid=os.getpid(),start_ticks='101',ppid=3,argv=run.argv('rank',admission),cgroup=group)
        events=[];contexts=[]
        class Context:
            def __init__(self,plan,sha,permit,guard):
                self.plan=plan;self.permit=permit;self.external_guard=guard;contexts.append(self)
                self._sources_checked=False
                assert guard()==permit
            def verify_sources(self):events.append('verify_sources');self._sources_checked=True
        def actual_worker(context,directory):
            self.assertTrue(context._sources_checked);events.append('worker')
            self.assertEqual(context.permit['runtime_admission'],admission)
            self.assertEqual(context.permit['runtime_plan'],a['plan'])
            self.assertEqual(context.permit['observed_drains'],a['observed_drains'])
            self.assertEqual(context.permit['outer_owner'],a['outer_owner'])
            self.assertEqual(context.permit['lock_identity'],a['lock_identity'])
            self.assertEqual(context.external_guard(),context.permit)
            return {'complete':True,'rank':0,'latest_completion':self.binding}
        def proc(pid):
            return process if pid==os.getpid()else {k:coordinator[k]for k in process}
        with patch.object(run,'child_admission',return_value=(a,spec,self.plan)),patch.dict(os.environ,{'RANK':'0'}),\
            patch.object(run,'wait_file',return_value=process),patch.object(run.p,'proc',side_effect=proc),\
            patch.object(run.p,'read',return_value=coordinator),patch.object(run.p,'lock_proof'),\
            patch.object(run,'admitted_bundles',return_value=(self.done,Mock())),patch.object(run.p,'gone'),\
            patch.object(run.p,'clear_window'),patch.object(run,'usage'),patch.object(run,'RuntimeContext',Context),\
            patch.object(worker,'run',side_effect=actual_worker),patch.object(run,'validate_rank_result')as validate,\
            patch.object(run.p,'once')as save:
            run.rank(admission)
            validate.assert_called_once()
            self.assertEqual(events,['verify_sources','worker'])
            saved=save.call_args.args[1]
            self.assertEqual(saved['process'],process);self.assertEqual(saved['admission'],admission)
        self.assert_no_native()


if __name__=='__main__':
    unittest.main(verbosity=2)
