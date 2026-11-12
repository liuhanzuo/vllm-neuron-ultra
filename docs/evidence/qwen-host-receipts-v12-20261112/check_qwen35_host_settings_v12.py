"""Source/mock routes with saved r9 JSON; never launch or qualify a model."""
import ast
import builtins
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
ORIGINAL_IMPORT = builtins.__import__


def source_only_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_neuronx', 'numpy', 'ctypes', 'transformers',
                            'vllm', 'vllm_neuron', 'neuronx_distributed', 'neuronx_distributed_inference'}:
        raise AssertionError('Native import forbidden: ' + name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


builtins.__import__ = source_only_import
import qwen35_sampler_pair_protocol_v1 as protocol
import qwen35_sampler_pair_driver_v2 as usage_driver
import qwen35_sampler_pair_driver_v3 as driver
import qwen35_sampler_host_settings_v1 as settings
import qwen35_sampler_runtime_contract_v11 as old_contract
import qwen35_sampler_runtime_contract_v12 as contract
import qwen35_sampler_runtime_adapter_v12 as adapter
import prepare_qwen35_sampler_v12_candidate_v1 as prep

SAVED = Path('/mnt/nvme/neuron-work/qwen35-sampler-pair-r9-v11-20261112')
OUTPUT = Path('/mnt/nvme/neuron-work/qwen35-sampler-pair-source-mock-v12')
CASE = protocol.expected_case()
PINS = {str(Q / 'qwen35_sampler_runtime_adapter_v12.py'):
        hashlib.sha256((Q/'qwen35_sampler_runtime_adapter_v12.py').read_bytes()).hexdigest()}


def fixture(arm='baseline', phase='http'):
    identity = protocol.host_identity(arm, 'correctness_capture' if phase == 'capture' else 'measured', PINS)
    directory = OUTPUT / arm / phase
    spec = dict(run_name='source-mock-v12', output_root=str(OUTPUT))
    control = contract.control_root(spec) / (arm + '-' + phase)
    environment = contract.native_environment(CASE, host_identity=identity,
        host_receipt=directory/'worker-attachment.json',
        capture_root=directory/'packets' if phase == 'capture' else None)
    environment = contract.phase_environment(spec, dict(arm=arm, phase=phase, control=str(control)), environment)
    argv = [contract.PYTHON, *driver.server_arguments(CASE, arm, identity['phase'],
        {'on_device_sampling_config': None, **CASE['extra_overrides']})]
    observed = settings.observed_settings(environment, argv, CASE, identity, directory, argv)
    live = json.loads((SAVED/arm/phase/'server-live-root-audit.json').read_text())
    live['api_process_owner']['argv'] = copy.deepcopy(argv)
    live['startup_config_proof'].update(actual_api_initial_environment=copy.deepcopy(environment),
        actual_api_initial_environment_sha256=contract.digest(environment), actual_api_argv=copy.deepcopy(argv),
        effective_host_settings=observed)
    return dict(identity=identity, directory=directory, environment=environment,
                argv=argv, live=live, control=control, spec=spec)


class HostSettings(unittest.TestCase):
    def replay(self, f):
        return settings.from_live_observation(f['live'], CASE, f['identity'], f['directory'], f['argv'])

    def test_both_arms_match_after_exact_bindings_are_checked(self):
        for phase in ('capture', 'quality', 'http'):
            with self.subTest(phase=phase):
                a, b = fixture('baseline', phase), fixture('candidate', phase)
                self.assertNotEqual(a['environment'], b['environment'])
                self.assertNotEqual(a['argv'], b['argv'])
                self.assertEqual(self.replay(a), self.replay(b))

    def test_observation_missing_field_or_not_live_rejected(self):
        for field in ('actual_api_initial_environment', 'effective_host_settings', 'actual_api_argv'):
            f=fixture(); del f['live']['startup_config_proof'][field]
            with self.subTest(field=field), self.assertRaises(KeyError): self.replay(f)
        for key,value in [('exit_observed', True), ('ready', False), ('server_active', False)]:
            f=fixture();f['live'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):self.replay(f)

    def test_all_environment_fields_remain_bound_even_if_digest_recomputed(self):
        f=fixture()
        for key in list(f['environment']) + ['VLLM_LOGGING_LEVEL', 'UNREVIEWED_OPTION']:
            bad=copy.deepcopy(f)
            environment=bad['live']['startup_config_proof']['actual_api_initial_environment']
            environment[key]='unexpected-value'
            bad['live']['startup_config_proof']['actual_api_initial_environment_sha256']=contract.digest(environment)
            with self.subTest(key=key),self.assertRaises(ValueError):self.replay(bad)

    def test_bound_capture_and_temp_paths_cannot_be_crossed(self):
        for phase in ('capture','quality','http'):
            f=fixture('candidate',phase)
            for key in settings.BOUND_ENVIRONMENT_KEYS:
                bad=copy.deepcopy(f)
                environment=bad['live']['startup_config_proof']['actual_api_initial_environment']
                environment[key]='/mnt/nvme/neuron-work/other-arm-or-phase'
                bad['live']['startup_config_proof']['actual_api_initial_environment_sha256']=contract.digest(environment)
                with self.subTest(phase=phase,key=key),self.assertRaises(ValueError):self.replay(bad)

    def test_changed_argv_rejected_even_if_process_and_proof_agree(self):
        original=fixture()
        for index in range(len(original['argv'])):
            f=copy.deepcopy(original);bad=f['live']['startup_config_proof']['actual_api_argv'];bad[index]='changed'
            f['live']['api_process_owner']['argv']=copy.deepcopy(bad)
            with self.subTest(index=index),self.assertRaises(ValueError):self.replay(f)

    def test_changed_recorded_projection_or_hash_cannot_pass(self):
        f=fixture();f['live']['startup_config_proof']['effective_host_settings']['environment']['OMP_NUM_THREADS']='8'
        with self.assertRaises(ValueError):self.replay(f)
        f=fixture();f['live']['startup_config_proof']['actual_api_initial_environment_sha256']='0'*64
        with self.assertRaises(ValueError):self.replay(f)

    def test_recorded_source_dictionaries_not_modified(self):
        f=fixture();before=copy.deepcopy(f);self.replay(f);self.assertEqual(f,before)

    def test_original_host_gate_sees_actual_projection(self):
        hosts=[]
        for arm in ('baseline','candidate'):
            f=fixture(arm);hosts.append({**f['identity'],'effective_host_settings':self.replay(f),
                'graph_reuse_receipt_sha256':'a'*64,'frozen_model_case':CASE})
        protocol.check_host_pair(*hosts)
        hosts[1]['effective_host_settings']['environment']['OMP_NUM_THREADS']='2'
        with self.assertRaisesRegex(ValueError,'unmatched setting'):protocol.check_host_pair(*hosts)

    def test_actual_api_ready_method_records_supplied_proc_values(self):
        f=fixture();obj=adapter.Phase.__new__(adapter.Phase)
        obj.arm='baseline';obj.name='http';obj.directory=f['directory'];obj.stage=f['control']
        obj.phase={'host_identity':f['identity']};obj.phase_ref={'path':'/mock-phase.json','sha256':'a'*64}
        obj.startup_proof=None;obj.reject_api_failure=lambda:None
        api_owner=copy.deepcopy(f['live']['api_process_owner']);api_owner['start_ticks']=str(api_owner['start_ticks'])
        child=copy.deepcopy(api_owner);child['argv']=['/mock-bootstrap'];obj.child=child
        attached=json.loads((SAVED/'baseline/http/worker-attachment.json').read_text())
        attached['host_identity_sha256']=f['identity']['sha256']
        group='/mock-cgroup';api_owner['cgroup']=child['cgroup']='0::'+group
        worker=dict(pid=attached['pid'],cgroup='0::'+group)
        obj.service=SimpleNamespace(group=group,live=lambda:None)
        obj.adapter=SimpleNamespace(observer=SimpleNamespace(check=lambda:None),
            spec={'source_sha256':{p:'b'*64 for p in contract.STARTUP_FILES}})
        intent=dict(phase_spec=obj.phase_ref,bootstrap_owner=child,argv=f['argv'],
                    environment_sha256=contract.digest(f['environment']))
        env_raw=b'\0'.join((k+'='+v).encode() for k,v in f['environment'].items())+b'\0'
        store={}
        def read(path):
            if Path(path).name=='api-exec-intent.json':return intent
            if Path(path).name=='worker-attachment.json':return attached
            raise AssertionError(path)
        def proc(pid):return api_owner if pid==child['pid'] else worker
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def read(self,n):return b''
        original_open=Path.open
        def open_fixture(path,*args,**kwargs):
            if str(path)==f'/proc/{api_owner["pid"]}/environ':return io.BytesIO(env_raw)
            return original_open(path,*args,**kwargs)
        with patch.object(adapter,'read_if_present',read),patch.object(adapter,'process_api',return_value=SimpleNamespace(proc=proc)), \
             patch.object(Path,'open',open_fixture), \
             patch.object(adapter.urllib.request,'urlopen',return_value=Response()), \
             patch.object(adapter,'once',lambda p,v:store.update({str(p):copy.deepcopy(v)})):
            actual=obj.api_ready()
            self.assertEqual(actual['api_owner'],api_owner)
            self.assertEqual(obj.startup_proof['actual_api_initial_environment'],f['environment'])
            self.assertEqual(obj.startup_proof['effective_host_settings'],self.replay(f))
            self.assertEqual(len(store),1)
            obj.api_ready();self.assertEqual(len(store),1)
            obj.startup_proof['effective_host_settings']['environment']['OMP_NUM_THREADS']='8'
            with self.assertRaises(ValueError):obj.api_ready()


class FullDriver(unittest.TestCase):
    def run_mock(self, change=None):
        events=[];saved={};active=[]
        root_owner=dict(unit='synthetic-root',pid=10,start_ticks=20,invocation_id='a'*32)
        original_read=driver.read_small
        def read(path):
            key=str(path)
            return copy.deepcopy(saved[key]) if key in saved else original_read(path)
        def write(path,value):
            self.assertNotIn(str(path),saved);saved[str(path)]=copy.deepcopy(value)
        def guard(stage,run_id):
            events.append(('guard',stage));return dict(admitted=True,exclusive=True,run_id=run_id,root_owner=root_owner)
        def reuse(case):
            events.append(('reuse',));return dict(validated=True,compile_permitted=False,provenance_unchanged=True,sha256='b'*64)
        def receipt(run_id,**kw):return dict(run_id=run_id,returncode=0,fresh_execution=True,source_test_fake_port=True,**kw)
        def model(arm,phase,directory,run_id):
            events.append(('model',arm,phase));write(directory/driver.MODEL_FILES[phase],original_read(protocol.BASE/driver.MODEL_FILES[phase]))
            return receipt(run_id,arm=arm,phase=phase)
        def vectors(arm,before,after,run_id):return receipt(run_id,arm=arm,phase='vectors203',complete=True,frames=203,identical_vectors=203)
        owner=self
        class Server:
            def __init__(self,arm,phase,directory,worker,run_id,identity):
                self.arm=arm;self.directory=directory;self.run_id=run_id;self.identity=identity
                self.f=fixture(arm,directory.name);owner.assertEqual(identity,self.f['identity'])
                self.attachment=json.loads((SAVED/arm/directory.name/'worker-attachment.json').read_text())
                self.attachment['host_identity_sha256']=identity['sha256']
            def __enter__(self):
                events.append(('enter',self.arm,self.directory.name));active.append(self)
                if self.directory.name=='capture':write(self.directory/'packets/capture-receipt.json',original_read(SAVED/self.arm/'capture/packets/capture-receipt.json'))
                return self.attachment
            def __exit__(self,*args):
                events.append(('exit',self.arm,self.directory.name));active.remove(self)
        def audit(manager,kind,run_id):
            suffix='server-live-root-audit.json' if kind=='live' else 'server-terminal-root-audit.json'
            value=copy.deepcopy(manager.f['live']) if kind=='live' else original_read(SAVED/manager.arm/manager.directory.name/suffix)
            value.update(run_id=run_id,root_owner=root_owner,observer_source=next(iter(PINS)),observer_source_sha256=next(iter(PINS.values())))
            if change:change(value,manager,kind,events)
            events.append(('audit',manager.arm,manager.directory.name,kind));return value
        def sampler(arm,directory,run_id):
            return receipt(run_id,arm=arm,phase='sampler_replay',passed=True,packets=32,rows=80,
                all_own_tokens_and_rng_reproduced=True,all_distribution_gates_passed=True,fast_path_rows=78)
        def controls(run_id):return receipt(run_id,phase='sampler_controls',passed=True,controls=[dict(control=n,passed=True) for n in protocol.SAMPLER_GATES['controls']])
        class Clients:
            def diagnostic(self,case,path):events.append(('diagnostic',path.parent.name))
            def smoke(self,case,path):return original_read(protocol.BASE/'smoke.json')
            def quality(self,case,path):
                events.append(('quality',path.parent.parent.name));return original_read(SAVED/path.parent.parent.name/'quality/quality96.json')
            def initial(self,case,path):
                events.append(('initial',path.parent.parent.name));return original_read(SAVED/path.parent.parent.name/'http/initial120.json')
            def workloads(self,case,path):
                events.append(('workloads',path.parent.parent.name));return original_read(SAVED/path.parent.parent.name/'http/workloads.json')
        ports=driver.GuardedPorts(guard,reuse,model,vectors,Server,sampler,controls,audit)
        original_compare=driver.check_host_pair
        def compare(a,b):events.append(('host_pair',));return original_compare(a,b)
        result=None;failure=None
        with patch.object(Path,'mkdir',lambda *a,**k:None),patch.object(Path,'exists',lambda p:False), \
             patch.object(driver,'write_new',write),patch.object(driver,'read_small',read),patch.object(driver,'check_host_pair',compare):
            try:result=driver.run(ports,OUTPUT,PINS,Clients())
            except (ValueError,KeyError,AssertionError) as exc:failure=exc
        self.assertEqual(active,[])
        return result,failure,events,saved

    def test_full_driver_saves_settings_runs_original_compare_and_final_reuse(self):
        result,error,events,saved=self.run_mock();self.assertIsNone(error,error)
        self.assertTrue(result['execution_finished']);self.assertFalse(result['full_model_qualified'])
        self.assertEqual(result['paired_http']['request_pairs'],124)
        self.assertEqual(sum(e[0]=='model' for e in events),8)
        self.assertEqual(sum(e[0]=='enter' for e in events),6)
        self.assertEqual(sum(e[0]=='reuse' for e in events),2)
        self.assertLess(events.index(('host_pair',)),events.index(('initial','candidate')))
        self.assertLess(events.index(('host_pair',)),events.index(('workloads','candidate')))
        hosts=[saved[str(OUTPUT/arm/'http/host.json')] for arm in ('baseline','candidate')]
        protocol.check_host_pair(*hosts)
        self.assertEqual(hosts[0]['effective_host_settings'],self.expected())

    def expected(self):
        f=fixture();return settings.from_live_observation(f['live'],CASE,f['identity'],f['directory'],f['argv'])

    def test_missing_settings_fails_before_first_client_and_cleans_server(self):
        def change(value,manager,kind,events):
            if kind=='live':del value['startup_config_proof']['effective_host_settings']
        result,error,events,_=self.run_mock(change);self.assertIsNone(result);self.assertIsInstance(error,KeyError)
        self.assertFalse(any(e[0] in ('diagnostic','quality','initial','workloads') for e in events))
        self.assertEqual(sum(e[0]=='exit' for e in events),1)

    def test_tampered_candidate_http_fails_before_candidate_timing(self):
        def change(value,manager,kind,events):
            if kind=='live' and manager.arm=='candidate' and manager.directory.name=='http':
                proof=value['startup_config_proof'];proof['actual_api_initial_environment']['OMP_NUM_THREADS']='8'
                proof['actual_api_initial_environment_sha256']=contract.digest(proof['actual_api_initial_environment'])
        result,error,events,_=self.run_mock(change);self.assertIsNone(result);self.assertIsInstance(error,ValueError)
        self.assertIn(('workloads','baseline'),events)
        self.assertNotIn(('initial','candidate'),events);self.assertNotIn(('workloads','candidate'),events)
        self.assertEqual(sum(e[0]=='enter' for e in events),sum(e[0]=='exit' for e in events))


class SourceContracts(unittest.TestCase):
    def test_original_usage_correction_is_byte_preserved_ast(self):
        def node(name,fn):
            return next(x for x in ast.parse((Q/name).read_text()).body if isinstance(x,ast.FunctionDef) and x.name==fn)
        self.assertEqual(ast.dump(node('qwen35_sampler_pair_driver_v2.py','check_initial_pair')),
                         ast.dump(node('qwen35_sampler_pair_driver_v3.py','check_initial_pair')))
        for n in ('server_arguments','NativeClients','authenticated','owner_identity','root_observation','attached_worker','cleaned_server'):
            trees=[ast.parse((Q/f'qwen35_sampler_pair_driver_v{v}.py').read_text()) for v in (1,3)]
            defs=[next(x for x in t.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==n) for t in trees]
            self.assertEqual(ast.dump(defs[0]),ast.dump(defs[1]),n)

    def test_unchanged_runtime_science_limits_ipc_and_storage(self):
        self.assertEqual(old_contract.LIMITS,contract.LIMITS)
        self.assertEqual(old_contract.PHASES,contract.PHASES)
        self.assertEqual(old_contract.LOCK_IDENTITY,contract.LOCK_IDENTITY)
        for part in ('io','history','service'):
            self.assertEqual((Q/f'qwen35_sampler_runtime_{part}_v11.py').read_text(),
                (Q/f'qwen35_sampler_runtime_{part}_v12.py').read_text().replace('_v12','_v11'))
        spec=dict(run_name='source-mock-v12',output_root=str(OUTPUT))
        self.assertEqual(old_contract.ipc_paths(spec),contract.ipc_paths(spec))
        manifest=json.loads((Q/'qwen35-sampler-pair-source-delivery-20261111-v1.json').read_text())
        for path,sha in manifest['new_source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)


class Preparation(unittest.TestCase):
    def fixture(self):
        future=dict(run_name='source-mock-v12',output_root=str(OUTPUT),
            running_handoff_path=str(Q/'mock-running.json'),overlay_path=str(Q/'mock-overlay.json'),
            spec_path=str(Q/'mock-spec.json'),review_candidate_path=str(Q/'mock-review.json'),
            successor_sources=dict(path=str(Q/'mock-v9.json'),sha256='9'*64))
        short=dict(path=str(Q/'mock-v10.json'),sha256='a'*64)
        storage=dict(path=str(Q/'mock-v11.json'),sha256='b'*64)
        ref=dict(path=str(Q/'mock-v12.json'),sha256='c'*64)
        pins={str(Q/'retained.py'):'d'*64}
        manifest=dict(schema='qwen35_sampler_observed_host_v12_sources_v1',previous_source_manifest=storage,
            initial_usage_source=prep.USAGE_FIX,limits=copy.deepcopy(contract.LIMITS),
            source_files={name:dict(path=str(Q/name),sha256='e'*64) for name in prep.NAMES})
        saved=dict(value=manifest,identity=dict(realpath=ref['path'],device=1,inode=1,size=100,mtime_ns=1,ctime_ns=1))
        result=dict(artifacts={future['spec_path']:dict(run_name=future['run_name'],output_root=str(OUTPUT),source_sha256=copy.deepcopy(pins),input_extension={}),
            future['overlay_path']:dict(source_extension=copy.deepcopy(pins),input_extension={}),
            future['review_candidate_path']:dict(passed=False,launch_authorized=False)},evidence={})
        return future,short,storage,ref,saved,result

    def test_additive_wrapper_retains_sources_inputs_and_unapproved_review(self):
        future,short,storage,ref,saved,result=self.fixture()
        with patch.object(prep.previous,'prepare',return_value=result) as old:
            value=prep.prepare(future,short,storage,ref,lambda r:saved,lambda:None)
        self.assertEqual(old.call_args.args[:3],(future,short,storage))
        spec=value['artifacts'][future['spec_path']];review=value['artifacts'][future['review_candidate_path']]
        self.assertEqual(len(spec['source_sha256']),9);self.assertEqual(spec['source_sha256'][str(Q/'retained.py')],'d'*64)
        self.assertEqual(spec['input_extension'],value['artifacts'][future['overlay_path']]['input_extension'])
        self.assertFalse(review['passed']);self.assertFalse(review['launch_authorized'])

    def test_bad_manifest_limits_source_shape_or_identity_rejected(self):
        for mode in ('limit','prior','missing','identity','usage'):
            f,short,storage,ref,saved,result=self.fixture()
            if mode=='limit':saved['value']['limits']['root_floor_bytes']-=1
            if mode=='prior':saved['value']['previous_source_manifest']={'path':storage['path'],'sha256':'0'*64}
            if mode=='missing':saved['value']['source_files'].pop(prep.NAMES[-1])
            if mode=='identity':saved['identity']['realpath']='/wrong.json'
            if mode=='usage':saved['value']['initial_usage_source']={'path':prep.USAGE_FIX['path'],'sha256':'0'*64}
            with self.subTest(mode=mode),patch.object(prep.previous,'prepare',side_effect=AssertionError('Must reject first')):
                with self.assertRaises(ValueError):prep.prepare(f,short,storage,ref,lambda r:saved,lambda:None)

    def test_no_frozen_pin_replacement(self):
        f,short,storage,ref,saved,result=self.fixture()
        result['artifacts'][f['spec_path']]['source_sha256'][str(Q/prep.NAMES[0])]='0'*64
        with patch.object(prep.previous,'prepare',return_value=result),self.assertRaisesRegex(ValueError,'conflict'):
            prep.prepare(f,short,storage,ref,lambda r:saved,lambda:None)


if __name__=='__main__':
    unittest.main(verbosity=2)
