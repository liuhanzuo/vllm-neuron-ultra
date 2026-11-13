"""In-memory full-protocol off DEVICE leaf tests; no actual runtime input IO."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace as N
import unittest
from unittest.mock import Mock,patch
from contextlib import ExitStack

R=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(R),str(R/'research')]
import deepseek_cold_control_device_entry_v1 as entry
import deepseek_capture_graph_entry_v1 as graph
import deepseek_cold_kv_protocol_v2 as protocol
import deepseek_cold_kv_contract_v2 as contract
import deepseek_cpu_capture_bridge_v1 as bridge
from test_deepseek_cold_boundary_device_entry_v1 import complete_report


class Wrapper:
    def __init__(self,owner):
        self.owner=owner;self.neuron_config=N(buckets=graph.BUCKETS);self.fail=False
    def __setattr__(self,key,value):
        if key=='_forward'and getattr(self,'fail',False):
            object.__setattr__(self,'fail',False);raise RuntimeError('partial route installation')
        object.__setattr__(self,key,value)
    def _forward(self,*args):
        self.owner.events.append('model-forward')
        if self.owner.revoke_at=='forward':self.owner.revoked=True
        return 'original-result'


class Fixture:
    def __init__(self):
        self.artifact=graph.ORIGINAL_ARTIFACT;self.dest=Path('/synthetic/deepseek-control/meta')
        self.reference=self.dest/'cold-reference';self.data=Path('/synthetic/deepseek-control/raw')
        self.case=dict(artifact=str(self.artifact),model_path=str(graph.MODEL_PATH),original='BF16 TP2 B4 context4096')
        self.artifact_hashes={str(self.artifact/'model.pt'):'a'*64,str(self.artifact/'neuron_config.json'):graph.ORIGINAL_CONFIG_SHA256}
        self.reference_hashes={str(self.reference/name):'b'*64 for name in ('cold-reference.pt','cold-reference.json')}
        self.config=dict(_name_or_path=self.case['model_path'],neuron_config=dict(save_sharded_checkpoint=False,
            skip_sharding=False,skip_warmup=False,tensor_capture_config=None,on_device_sampling_config=None,
            batch_size=4,tp_degree=2,async_mode=False,cast_type='config'))
        self.original=complete_report(self.case);self.current=copy.deepcopy(self.original)
        self.manifest=dict(complete=True,backend='device',scope=contract.scope(),frames=[
            dict(frame=frame,key=contract.key(frame,'device'),path='/synthetic/frame-'+str(i)+'.json',sha256='c'*64)
            for i,frame in enumerate(contract.frames('device'))],full_frame_count=84,cache_records=6048,
            compiled_model_unchanged=True,full_model_qualified=False,new_speedup_claim=False)
        self.lifecycle=dict(state_tensors_changed_by_warmup=144,same144_bindings_preserved=True)
        self.copy={'all144':'copy-proof'};self.events=[];self.revoked=False;self.revoke_at=None
        self.guard_error=RuntimeError('guard revoked');self.load_error=None;self.loop_error=None
        self.loop_returns=False;self.partial_driver=False;self.forward_count=0;self.captured_frames=[]
        self.load_calls=[];self.config_calls=[];self.native_count=0;self.existing=set();self.symlinks=set()
        owner=self
        class Config:
            @staticmethod
            def load(path):owner.config_calls.append(path);return owner.cfg
        self.cfg=N(neuron_config=N(**self.config['neuron_config']),to_json_string=lambda:json.dumps(self.config))
        class Model:
            @staticmethod
            def get_config_cls():return Config
            def __init__(self,path,config):
                owner.events.append('construct');owner.model=self
                self.path=path;self.config=config;self.neuron_config=config.neuron_config;self.is_loaded_to_neuron=False
                self.context_encoding_model=Wrapper(owner);self.token_generation_model=Wrapper(owner)
                self.token_generation_model._forward=self.token_generation_model._forward
                owner.original_own=self.token_generation_model._forward
                if owner.revoke_at=='partial_install':self.token_generation_model.fail=True
            def load(self,path,**kwargs):
                owner.events.append('load');owner.load_calls.append((path,kwargs))
                if owner.load_error:raise owner.load_error
                self.is_loaded_to_neuron=True
                if owner.revoke_at=='load':owner.revoked=True
        self.cls=Model;self.model=None;self.driver=ModuleType('validate_deepseek_cold_control_v1')
        self.driver.main=Mock(side_effect=self.loop)
        self.route_module=ModuleType('qwen35_split_state_checks_v4')
        self.routes=dict(complete=True,graphs=[{}]*12);self.route_module.graph_routes=lambda model:self.routes
        self.state_module=ModuleType('deepseek_prior_mask_state_v1');self.states=[{}]*144
        self.state_module.bindings=lambda model:self.states
        self.modules={m.__name__:m for m in (self.driver,self.route_module,self.state_module)}
        self.read=Mock(side_effect=lambda p:{'manifest.json':self.manifest,'cold-buckets.json':self.current,
            'state-copy.json':self.copy,'state-lifecycle.json':self.lifecycle}[Path(p).name])
        self.copy_check=Mock();self.lifecycle_check=Mock();self.hash_calls=[]
    def record(self,path):
        self.hash_calls.append(str(path));self.events.append('hash-'+Path(path).name)
        return dict(sha256=(self.artifact_hashes|self.reference_hashes)[str(path)],bytes=3)
    def native(self):
        self.events.append('native');self.native_count+=1;return self.cls
    def guard(self):
        self.events.append('guard')
        if self.revoked:raise self.guard_error
    def loop(self,model,case,destination,reference,data_root,guard):
        self.events.append('driver')
        for wrapper in (model.context_encoding_model,model.token_generation_model):
            old=wrapper._forward;wrapper._forward=lambda *a,_old=old,**kw:_old(*a,**kw)
            if self.partial_driver:raise RuntimeError('partial original driver installation')
        if self.loop_error:raise self.loop_error
        for prefix in bridge.PREFIXES:
            for slot in range(4):
                model.context_encoding_model._forward(prefix,slot);self.forward_count+=1
                self.captured_frames.append(dict(prefix=prefix,slot=slot,fixture=slot%2,step=0))
            for step in (1,2):
                model.token_generation_model._forward(prefix,step);self.forward_count+=1
                self.captured_frames.extend(dict(prefix=prefix,slot=slot,fixture=slot%2,step=step)for slot in range(4))
        if self.revoke_at=='driver_end':self.revoked=True
        if self.loop_returns:return
        raise SystemExit(entry.STRICT_FAILURE)
    def run(self):
        with ExitStack()as stack:
            for obj,name,value in ((Path,'resolve',lambda p:p),(Path,'is_dir',lambda p:True),
                (Path,'is_symlink',lambda p:str(p)in self.symlinks),(Path,'exists',lambda p:str(p)in self.existing),
                (graph,'file_record',self.record),(entry,'_config',lambda *args:copy.deepcopy(self.config)),
                (entry,'_native_model_class',self.native),(protocol,'expected_case',lambda:self.case),
                (protocol,'read',self.read),(protocol,'check_copy',self.copy_check),(protocol,'check_lifecycle',self.lifecycle_check)):
                stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.dict(sys.modules,self.modules))
            return entry.run(self.artifact,self.artifact_hashes,self.case,self.dest,self.reference,self.data,
                self.original,self.reference_hashes,self.guard)
    def restored(self,test):
        if self.model is not None:
            test.assertNotIn('_forward',vars(self.model.context_encoding_model))
            test.assertIs(self.model.token_generation_model._forward,self.original_own)


class Entry(unittest.TestCase):
    def test_exact_original_load_skip_warmup_then_entire42call84frame_driver(self):
        f=Fixture();out=f.run();self.assertEqual(f.native_count,1)
        self.assertEqual(f.load_calls,[(str(f.artifact),dict(skip_warmup=True))])
        self.assertEqual(f.config_calls,[str(f.artifact)])
        args=f.driver.main.call_args.args;self.assertIs(args[0],f.model);self.assertIs(args[1],f.case)
        self.assertEqual(args[2:5],(f.dest,f.reference,f.data));self.assertTrue(callable(args[5]))
        self.assertEqual(f.forward_count,42);self.assertEqual(f.captured_frames,contract.frames('device'))
        self.assertEqual(out['full_frame_classification']['original14_failed_indices'],list(bridge.FAILED_INDICES))
        self.assertEqual(out['state_objects'],144);self.assertEqual(out['full_device_KV_tensors'],6048)
        self.assertFalse(out['neutrality_proven']);self.assertFalse(out['full_model_qualified']);self.assertFalse(out['new_speedup_claim'])
        f.copy_check.assert_called_once_with(f.copy);f.lifecycle_check.assert_called_once_with(f.lifecycle);f.restored(self)
        before=f.events[:f.events.index('native')]
        self.assertEqual([e for e in before if e.startswith('hash-')],
            ['hash-model.pt','hash-neuron_config.json','hash-cold-reference.pt','hash-cold-reference.json'])

    def test_missing_extra_or_malformed_external_hashes_fail_before_native(self):
        for kind in ('artifact_missing','artifact_extra','reference_missing','reference_extra','invalid','original_config'):
            f=Fixture()
            if kind=='artifact_missing':f.artifact_hashes.pop(str(f.artifact/'model.pt'))
            elif kind=='artifact_extra':f.artifact_hashes['/synthetic/other.pt']='a'*64
            elif kind=='reference_missing':f.reference_hashes.pop(str(f.reference/'cold-reference.pt'))
            elif kind=='reference_extra':f.reference_hashes['/synthetic/other.pt']='b'*64
            elif kind=='invalid':f.artifact_hashes[str(f.artifact/'model.pt')]='invalid'
            else:f.artifact_hashes[str(f.artifact/'neuron_config.json')]='f'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.run()
            self.assertEqual(f.native_count,0);self.assertFalse(f.load_calls)

    def test_artifact_and_reference_mismatch_rejected_before_native(self):
        for name in ('model.pt','neuron_config.json','cold-reference.pt','cold-reference.json'):
            f=Fixture();record=f.record
            f.record=lambda path:dict(sha256='f'*64)if Path(path).name==name else record(path)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'input changed'):f.run()
            self.assertEqual(f.native_count,0)

    def test_artifact_change_after_model_construction_prevents_load(self):
        f=Fixture();record=f.record;count=[0]
        def changed(path):
            if Path(path).name=='model.pt':count[0]+=1
            return dict(sha256='f'*64)if Path(path).name=='model.pt'and count[0]>1 else record(path)
        f.record=changed
        with self.assertRaisesRegex(ValueError,'input changed'):f.run()
        self.assertFalse(f.load_calls);f.driver.main.assert_not_called();f.restored(self)

    def test_reference_change_after_load_prevents_deserializing_driver(self):
        f=Fixture();record=f.record
        f.record=lambda path:dict(sha256='f'*64)if f.load_calls and Path(path).name=='cold-reference.pt'else record(path)
        with self.assertRaisesRegex(ValueError,'input changed'):f.run()
        self.assertEqual(len(f.load_calls),1);f.driver.main.assert_not_called();f.restored(self)

    def test_original_config_capture_sampling_sharding_changes_rejected(self):
        for name,value in (('tensor_capture_config',{}),('on_device_sampling_config',{}),
                ('save_sharded_checkpoint',True),('skip_sharding',True),('skip_warmup',True)):
            f=Fixture();f.config['neuron_config'][name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):f.run()
            self.assertEqual(f.native_count,0)

    def test_loaded_config_case_or_state_layout_changes_rejected(self):
        for kind in ('batch','tp','async','cast','buckets','routes','states'):
            f=Fixture()
            if kind in ('batch','tp','async','cast'):
                key,value={'batch':('batch_size',1),'tp':('tp_degree',1),'async':('async_mode',True),'cast':('cast_type','fp16')}[kind]
                setattr(f.cfg.neuron_config,key,value)
            elif kind=='routes':f.routes['graphs'].pop()
            elif kind=='states':f.states.pop()
            else:
                native=f.native
                def changed():
                    cls=native();original=cls.load
                    def load(model,*args,**kwargs):
                        original(model,*args,**kwargs);model.token_generation_model.neuron_config.buckets=(128,4096)
                    cls.load=load;return cls
                f.native=changed
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.run()
            f.driver.main.assert_not_called()

    def test_load_failure_or_guard_revocation_never_enters_driver(self):
        for kind in ('load_error','before','load'):
            f=Fixture()
            if kind=='load_error':f.load_error=RuntimeError('load failed')
            elif kind=='before':f.revoked=True
            else:f.revoke_at='load'
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):f.run()
            f.driver.main.assert_not_called();f.restored(self)

    def test_guard_exit_with_expected_failure_text_is_never_classified(self):
        f=Fixture();f.revoke_at='forward';f.guard_error=SystemExit(entry.STRICT_FAILURE)
        with self.assertRaises(SystemExit)as caught:f.run()
        self.assertIs(caught.exception,f.guard_error);self.assertEqual(f.read.call_count,0);f.restored(self)

    def test_guard_revoked_at_completed_driver_cannot_publish_classification(self):
        f=Fixture();f.revoke_at='driver_end'
        with self.assertRaisesRegex(RuntimeError,'guard revoked'):f.run()
        self.assertEqual(f.forward_count,42);self.assertEqual(f.read.call_count,0);f.restored(self)

    def test_early_matching_exit_missing_manifest_and_lifecycle_errors_remain_failures(self):
        for mode in ('early','copy','lifecycle'):
            f=Fixture()
            if mode=='early':
                f.loop_error=SystemExit(entry.STRICT_FAILURE);f.read.side_effect=FileNotFoundError('no completed manifest')
            elif mode=='copy':f.copy_check.side_effect=ValueError('copy control failed')
            else:f.lifecycle_check.side_effect=ValueError('lifecycle failed')
            with self.subTest(mode=mode),self.assertRaises((FileNotFoundError,ValueError)):f.run()
            f.restored(self)

    def test_partial_entry_or_driver_hook_install_restores_both_attributes(self):
        for mode in ('entry','driver'):
            f=Fixture();f.revoke_at='partial_install'if mode=='entry'else None;f.partial_driver=mode=='driver'
            with self.subTest(mode=mode),self.assertRaisesRegex(RuntimeError,'partial'):f.run()
            f.restored(self)

    def test_arbitrary_error_or_exit_or_new_success_not_accepted(self):
        class DerivedExit(SystemExit):pass
        for error in (RuntimeError('driver failed'),KeyboardInterrupt(),SystemExit(0),SystemExit('other'),DerivedExit(entry.STRICT_FAILURE)):
            f=Fixture();f.loop_error=error
            expected=ValueError if isinstance(error,SystemExit)else type(error)
            with self.subTest(error=repr(error)),self.assertRaises(expected):f.run()
            f.restored(self);self.assertEqual(f.read.call_count,0)
        f=Fixture();f.loop_returns=True
        with self.assertRaisesRegex(ValueError,'unexpectedly changed'):f.run()
        f.restored(self)

    def test_incomplete_or_reordered_cache_manifest_rejected(self):
        for mode in ('missing','reorder','count','kv','claim','early'):
            f=Fixture()
            if mode=='missing':f.manifest['frames'].pop()
            elif mode=='reorder':f.manifest['frames'].reverse()
            elif mode=='count':f.manifest['full_frame_count']=14
            elif mode=='kv':f.manifest['cache_records']=72
            elif mode=='claim':f.manifest['full_model_qualified']=True
            else:f.loop_error=SystemExit(entry.STRICT_FAILURE);f.manifest['complete']=False
            with self.subTest(mode=mode),self.assertRaises(AssertionError):f.run()
            f.restored(self)

    def test_full84_metadata_and_all144_checks_cannot_be_shortened(self):
        for mode in ('original14','row','metric','zero','finite','state','bindings','error'):
            f=Fixture();item=f.current['experiments'][-1]
            if mode=='original14':
                for row in f.original['experiments']:row['rows']=row['rows'][:2]
            elif mode=='row':item['rows'].pop()
            elif mode=='metric':item['rows'][-1]['metrics']['max_abs_error']=1
            elif mode=='zero':item['initial_zero_state_checks'].pop()
            elif mode=='finite':item['state_after_two_decode_steps'][-1]['nan_elements']=1
            elif mode=='state':f.lifecycle['state_tensors_changed_by_warmup']=143
            elif mode=='bindings':f.lifecycle['same144_bindings_preserved']=False
            else:f.current['error']='partial failure'
            with self.subTest(mode=mode),self.assertRaises((ValueError,AssertionError)):f.run()
            f.restored(self)

    def test_fresh_paths_wrong_artifact_weight_shards_and_trace_rejected(self):
        for mode in ('output','tmp','overlap','symlink','weights','wrong_artifact','trace','native'):
            f=Fixture();context=ExitStack()
            if mode=='output':f.existing.add(str(f.dest/'cache-captures/device'))
            elif mode=='tmp':f.existing.add(str(f.dest/'cold-buckets.json.tmp'))
            elif mode=='overlap':f.data=f.dest
            elif mode=='symlink':f.symlinks.add(str(f.dest/'cold-buckets.pt'))
            elif mode=='weights':f.existing.add(str(f.artifact/'weights'))
            elif mode=='wrong_artifact':f.artifact=Path('/synthetic/other-artifact')
            elif mode=='trace':context.enter_context(patch.object(entry.sys,'gettrace',return_value=object()))
            else:context.enter_context(patch.dict(sys.modules,{'torch':ModuleType('torch')}))
            with context,self.subTest(mode=mode),self.assertRaises(ValueError):f.run()
            self.assertEqual(f.native_count,0)

    def test_config_raw_authentication_precedes_json_decode(self):
        with patch.object(Path,'stat',return_value=N(st_size=8378)),patch.object(Path,'read_bytes',return_value=b'{"synthetic":true}'):
            with self.assertRaisesRegex(ValueError,'config changed'):entry._config(Path('/synthetic/artifact'),'f'*64)
            raw=b'{"synthetic":true}'
            self.assertEqual(entry._config(Path('/synthetic/artifact'),hashlib.sha256(raw).hexdigest()),dict(synthetic=True))


class Source(unittest.TestCase):
    def test_no_scientific_loop_copied_or_replaced_and_frozen_driver_unchanged(self):
        text=(R/'deepseek_cold_control_device_entry_v1.py').read_text();tree=ast.parse(text)
        calls=[ast.unparse(n.func)for n in ast.walk(tree)if isinstance(n,ast.Call)]
        self.assertEqual(calls.count('original_loop'),1)
        for name in ('compile','from_pretrained','unlink','replace','rmtree','kill','flock','load_state_dict'):
            self.assertFalse(any(x.split('.')[-1]==name for x in calls))
        self.assertNotIn('deepseek_cold_boundary_capture_v1',text)
        old=ast.parse((R/'validate_deepseek_cold_kv_v2.py').read_text())
        control=ast.parse((R/'validate_deepseek_cold_control_v1.py').read_text())
        get=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
        loops=lambda n:[ast.dump(v,include_attributes=False)for v in ast.walk(n)if isinstance(v,ast.For)]
        self.assertEqual(loops(get(old)),loops(get(control)))
        self.assertEqual(hashlib.sha256((R/'deepseek_cold_boundary_device_entry_v1.py').read_bytes()).hexdigest(),
            'edba9a24ebe4f1485642c0b86c9745eee6e18db91be72fad442ad60878718008')
        pins={'validate_deepseek_cold_control_v1.py':'dae4e897d87d003144f31841231a4cfa57e1917db31ccf53251bf155ff3140f4',
            'deepseek_cold_control_capture_v1.py':'1a7170ca3993b711573f8bd8357cbfa0eeba8ed82edd82ad6f753ae7dc282a92',
            'deepseek_cold_control_lifecycle_v1.py':'a3e081a8877766d3eb31b88015757fdd277b5b8e40f4ee0174a8956f2c628359',
            'deepseek_qwen3_neuron_adapter.py':'6a220d8128a8a1ace024a224372ebf3811d3547c30705c4e4671e8002b8cfdfe'}
        for name,digest in pins.items():self.assertEqual(hashlib.sha256((R/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main(verbosity=2)
