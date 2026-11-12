"""In-memory auditor controls using saved JSON and synthetic packet bytes.

No main(), lock, process/service probe, native import or real science receipt.
Historical r9 failures are never rewritten. All success fields below are mocks.
"""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
original_import=builtins.__import__


def no_native(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','torch_neuronx','numpy','ctypes','vllm','vllm_neuron',
            'transformers','neuronx_distributed','neuronx_distributed_inference'}:
        raise AssertionError('Native import forbidden: '+name)
    return original_import(name,*args,**kwargs)


builtins.__import__=no_native
import audit_qwen35_sampler_r10_terminal_root_20261112_v2 as audit
import qwen35_sampler_pair_driver_v3 as driver
import qwen35_sampler_pair_protocol_v1 as protocol
import qwen35_sampler_runtime_contract_v12 as contract
import qwen35_sampler_host_settings_v1 as settings

C=Path('/mnt/nvme/neuron-work/qw-r9-v11-20261112')
O=Path('/mnt/nvme/neuron-work/qwen35-sampler-pair-r9-v11-20261112')
COMPLETED={arm+'-'+phase for arm,phase in contract.PHASES if phase not in ('capture','quality','http')}


class SavedJSONFixture:
    """Never opens binary payloads; synthetic replacement packets stay in RAM."""
    def __init__(self):
        self.overrides={};self.hidden=set();self.packet_sizes={};self.json_cache={}
        self.case=protocol.expected_case()
        saved_contract=self.real(O/'contract.json')
        sources=saved_contract['host_source_sha256']
        self.settings_checks=[]
        for arm in ('baseline','candidate'):
            for phase in ('capture','quality','http'):
                identity=protocol.host_identity(arm,'correctness_capture' if phase=='capture' else 'measured',sources)
                directory=O/arm/phase
                env=contract.native_environment(self.case,host_identity=identity,host_receipt=directory/'worker-attachment.json',
                    capture_root=directory/'packets' if phase=='capture' else None)
                spec=dict(run_name='r9-v11-20261112',output_root=str(O))
                env=contract.phase_environment(spec,dict(arm=arm,phase=phase,control=str(C/(arm+'-'+phase))),env)
                argv=[contract.PYTHON,*driver.server_arguments(self.case,arm,identity['phase'],
                    {'on_device_sampling_config':None,**self.case['extra_overrides']})]
                live=self.real(directory/'server-live-root-audit.json')
                live['api_process_owner']['argv']=copy.deepcopy(argv)
                projection=settings.observed_settings(env,argv,self.case,identity,directory,argv)
                live['startup_config_proof'].update(actual_api_initial_environment=env,
                    actual_api_initial_environment_sha256=contract.digest(env),actual_api_argv=argv,
                    effective_host_settings=projection)
                self.overrides[str(directory/'server-live-root-audit.json')]=live
                if phase=='http':
                    host=self.real(directory/'host.json');host['effective_host_settings']=projection
                    self.overrides[str(directory/'host.json')]=host
            capture=self.real(O/arm/'capture/packets/capture-receipt.json')
            for row in capture['files']:
                path=O/arm/'capture/packets'/row['file'];self.packet_sizes[str(path)]=row['bytes']
                row['sha256']=hashlib.sha256(b'\0'*row['bytes']).hexdigest()
            self.overrides[str(O/arm/'capture/packets/capture-receipt.json')]=capture
        left,right=[self.real(O/arm/'http/workloads.json') for arm in ('baseline','candidate')]
        hosts=[self.overrides[str(O/arm/'http/host.json')] for arm in ('baseline','candidate')]
        paired=protocol.compare_full_http(left,right,self.case,left['inputs'],*hosts)
        result=dict(run_id=saved_contract['run_id'],execution_finished=True,gates_checked_by_driver=True,
            paired_http=paired,full_model_qualified=False,independent_final_audit_required=True)
        self.overrides[str(O/'driver-result.json')]=result
        self.overrides[str(C/'coordinator-result.json')]=dict(driver_result=result,all_model_and_serving_phases_completed=True)

    def real(self,path):
        path=Path(path)
        assert path.suffix=='.json','JSON only'
        if str(path) not in self.json_cache:self.json_cache[str(path)]=json.loads(path.read_text())
        return copy.deepcopy(self.json_cache[str(path)])

    def read(self,path):
        assert str(path) not in self.hidden
        if str(path) in self.overrides:return copy.deepcopy(self.overrides[str(path)])
        return self.real(path)

    def raw(self,path):
        if str(path) in self.packet_sizes:return b'\0'*self.packet_sizes[str(path)]
        assert Path(path).suffix=='.json','No binary reads'
        return Path(path).read_bytes()

    def exists(self,path):
        return str(path) not in self.hidden and (str(path) in self.overrides or Path(path).is_file())


class AuditorScience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=SavedJSONFixture()

    def invoke(self,fixture=None,successful=True,completed=None):
        f=copy.deepcopy(self.base) if fixture is None else fixture
        owners=[];checked=[]
        actual=settings.from_live_observation
        def observe(*args):checked.append(str(args[3]));return actual(*args)
        with patch.object(audit,'C',C),patch.object(audit,'O',O),patch.object(audit,'read',f.read), \
             patch.object(audit,'raw',f.raw),patch.object(Path,'exists',lambda path:f.exists(path)), \
             patch.object(settings,'from_live_observation',observe):
            result=audit.science(COMPLETED if completed is None else completed,successful,owners)
        return result,owners,checked

    def test_complete_success_simulation_checks_all_six_observations_and_full_originals(self):
        result,owners,checked=self.invoke()
        self.assertEqual(len(owners),12)
        self.assertEqual(checked,[str(O/arm/phase) for arm in ('baseline','candidate') for phase in ('capture','quality','http')])
        self.assertTrue(result['native_preflight42_passed'])
        self.assertEqual(result['vector_reproductions'],dict(baseline=203,candidate=203))
        self.assertEqual(result['quality']['quality_questions'],96)
        self.assertEqual(result['paired_http']['request_pairs'],124)
        self.assertEqual(result['paired_http']['identical_greedy_texts'],57)

    def test_missing_settings_in_each_of_six_observations_rejected(self):
        for arm in ('baseline','candidate'):
            for phase in ('capture','quality','http'):
                f=copy.deepcopy(self.base)
                del f.overrides[str(O/arm/phase/'server-live-root-audit.json')]['startup_config_proof']['effective_host_settings']
                with self.subTest(arm=arm,phase=phase),self.assertRaises(KeyError):self.invoke(f)

    def test_host_receipt_must_equal_actual_identity_attachment_and_observed_settings(self):
        for key in ('effective_host_settings','pid','sha256'):
            f=copy.deepcopy(self.base);f.overrides[str(O/'candidate/http/host.json')][key]='wrong'
            with self.subTest(key=key),self.assertRaises(AssertionError):self.invoke(f)

    def test_contract_source_map_and_coordinator_owner_and_root_spec_are_bound(self):
        for key in ('host_source_sha256','root_owner','spec','run_id'):
            f=copy.deepcopy(self.base);value=f.real(O/'contract.json')
            if key=='host_source_sha256':value[key]={'invented':'a'*64}
            else:value['root_admission'][key]='wrong'
            f.overrides[str(O/'contract.json')]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):self.invoke(f)

    def test_precontract_failure_retains_only_completed_preflight_science(self):
        f=copy.deepcopy(self.base);f.hidden.add(str(O/'contract.json'))
        result,owners,checked=self.invoke(f,False,{'shared-native_preflight'})
        self.assertTrue(result['native_preflight42_passed']);self.assertFalse(result['contract_present'])
        self.assertEqual(result['model_phases'],{'baseline':[],'candidate':[]})
        self.assertIsNone(result['paired_http']);self.assertEqual(owners,[]);self.assertEqual(checked,[])

    def test_preflight_failure_claims_no_science_and_success_cannot_skip_it(self):
        f=copy.deepcopy(self.base);f.hidden.update(str(x) for x in (O/'contract.json',C/'native-preflight.json'))
        result,_,_=self.invoke(f,False,set())
        self.assertFalse(result['native_preflight42_passed']);self.assertFalse(result['contract_present'])
        with self.assertRaises(AssertionError):self.invoke(f,True,set())

    def test_nonzero_vector_child_is_not_promoted_from_saved_result(self):
        f=copy.deepcopy(self.base);completed=COMPLETED-{'candidate-vectors203'}
        result,_,_=self.invoke(f,False,completed)
        self.assertEqual(result['vector_reproductions'],{'baseline':203})
        with self.assertRaises(AssertionError):self.invoke(f,True,completed)


class SessionBinding(unittest.TestCase):
    def fixture(self):
        admission=json.loads((C/'admission.json').read_text());co=json.loads((C/'coordinator-owner.json').read_text())
        spec=json.loads(Path(admission['spec']['path']).read_text())
        running_path=Path(spec['running_handoff_path']);running=json.loads(running_path.read_text())
        ack=json.loads((C/'running-ack.json').read_text())
        values={str(running_path):running,str(C/'running-ack.json'):ack}
        def read(p):return copy.deepcopy(values[str(p)])
        def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
        def bound(r):
            self.assertEqual(r,ref(r['path']));return read(r['path'])
        return admission,co,spec,values,read,ref,bound

    def test_exact_immutable_tool_and_ack_binding(self):
        a,co,spec,values,read,ref,bound=self.fixture()
        with patch.object(audit,'C',C),patch.object(audit,'read',read),patch.object(audit,'ref',ref),patch.object(audit,'bound',bound):
            self.assertTrue(audit.session_binding(78094,a,spec,co,True)['bound'])
            with self.assertRaises(AssertionError):audit.session_binding(78095,a,spec,co,True)
            values[str(C/'running-ack.json')]['coordinator_owner']['sha256']='0'*64
            with self.assertRaises(AssertionError):audit.session_binding(78094,a,spec,co,True)

    def test_no_handoff_only_allows_explicit_unbound_failure(self):
        a,co,spec,*_=self.fixture()
        with patch.object(Path,'exists',return_value=False):
            result=audit.session_binding(78094,a,spec,co,False)
            self.assertFalse(result['bound']);self.assertIsNone(result['running_handoff'])
            with self.assertRaises(AssertionError):audit.session_binding(78094,a,spec,co,True)


if __name__=='__main__':unittest.main(verbosity=2)
