"""Source and pure receipt fixtures; never native imports or actual admission."""
import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path.insert(0,str(Q))
import run_dory_weight_storage_root_v1 as s


def owner(pid=100):return dict(pid=pid,start_ticks='1234',ppid=99,argv=['synthetic-source-test-only'],cgroup='0::/synthetic')
def reference(name):return dict(path='/synthetic/'+name,sha256='1'*64)
def pair():
    t=dict(actual_exit=dict(session_id=7,exit_code=0,tool_chunk='abc123'),all_own_input_controls_exact=False)
    tr=reference('terminal');rr=reference('run');run=dict(schema='minicpm5_attention_independent_review_run_v1',complete=True,terminal=tr,actual_model_exit=t['actual_exit'],model_exit_verified=True,physical_drain_verified=True,complete_report_reproduced=True,all21_own_input_controls_exact=False,actual_self_exit_claimed=False,full_model_qualified=False,new_speedup_claim=False,owner=owner())
    ended=dict(schema='minicpm5_attention_independent_review_actual_external_exit_v1',reviewer='/root',run=rr,owner=owner(),terminal=tr,actual_exit=dict(session_id=8,exit_code=0,tool_chunk='abc124'),actual_tool_exit_observed=True,exact_owner_gone=True,new_speedup_claim=False)
    return t,tr,run,rr,ended


class Completion(unittest.TestCase):
    def setUp(self):self.t,self.tr,self.run,self.rr,self.ended=pair()
    def check(self):return s.bind_review_exit(self.run,self.rr,self.ended,self.tr,self.t)
    def test_actual_process_success_preserves_false_scientific_control(self):self.assertEqual(self.check(),owner());self.assertFalse(self.run['all21_own_input_controls_exact'])
    def test_missing_actual_external_exit_rejected(self):
        self.ended['actual_tool_exit_observed']=False
        with self.assertRaisesRegex(Exception,'Observed real exit'):self.check()
    def test_failed_tool_exit_rejected(self):
        self.ended['actual_exit']['exit_code']=1
        with self.assertRaisesRegex(Exception,'successful tool exit'):self.check()
    def test_boolean_exit_code_rejected(self):
        self.ended['actual_exit']['exit_code']=False
        with self.assertRaisesRegex(Exception,'successful tool exit'):self.check()
    def test_boolean_session_rejected(self):
        self.ended['actual_exit']['session_id']=True
        with self.assertRaisesRegex(Exception,'successful tool exit'):self.check()
    def test_missing_external_chunk_rejected(self):
        self.ended['actual_exit']['tool_chunk']=''
        with self.assertRaisesRegex(Exception,'successful tool exit'):self.check()
    def test_different_review_owner_rejected(self):
        self.ended['owner']=owner(101)
        with self.assertRaisesRegex(Exception,'Exact original review owner'):self.check()
    def test_different_terminal_rejected(self):
        self.run['terminal']=reference('different')
        with self.assertRaisesRegex(Exception,'actual Mini terminal'):self.check()
    def test_report_cannot_claim_own_exit(self):
        self.run['actual_self_exit_claimed']=True
        with self.assertRaisesRegex(Exception,'No self-exit'):self.check()
    def test_false_control_cannot_be_promoted(self):
        self.run['all21_own_input_controls_exact']=True
        with self.assertRaisesRegex(Exception,'failed scientific controls'):self.check()
    def test_review_must_reproduce_whole_report(self):
        self.run['complete_report_reproduced']=False
        with self.assertRaisesRegex(Exception,'complete Mini review'):self.check()
    def test_original_model_exit_must_match(self):
        self.run['actual_model_exit']=dict(self.t['actual_exit'],session_id=9)
        with self.assertRaisesRegex(Exception,'actual Mini terminal'):self.check()
    def test_no_full_model_promotion(self):
        self.run['full_model_qualified']=True
        with self.assertRaisesRegex(Exception,'No self-exit'):self.check()
    def test_gone_owner_must_be_recorded(self):
        self.ended['exact_owner_gone']=False
        with self.assertRaisesRegex(Exception,'Observed real exit'):self.check()


class Request(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.pins=s.sources()
    def setUp(self):
        self.request=dict(schema='dory_weight_storage_runtime_request_v1',run_id='r1-20261108-v1',output_root=str(s.R/'dory-weight-storage-r1-20261108-v1'),budgets=copy.deepcopy(s.BUDGET),science=s.capture.probe.plan(),python=str(s.runtime.PYTHON),dependency_records=s.runtime.RECORDS,runtime_identity={},source_sha256=self.pins,canonical_sha256=s.m.CANONICAL,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False,automatic_successor_launch=False,preparation_owner=owner())
    def check(self):
        with patch.object(s.runtime,'validate_runtime_identity',return_value=[]):return s.validate_request(self.request)
    def test_complete_source_request_does_not_admit_runtime(self):self.assertIs(self.check(),self.request);self.assertFalse(self.request['runtime_admitted'])
    def test_unreduced_fixture_required(self):
        self.request['science']['saved_tensors']=17
        with self.assertRaisesRegex(Exception,'Full unchanged fixture'):self.check()
    def test_resource_budget_cannot_change(self):
        self.request['budgets']['cpu_seconds']*=2
        with self.assertRaisesRegex(Exception,'resource plan'):self.check()
    def test_unpinned_python_rejected(self):
        self.request['python']='/usr/bin/python3'
        with self.assertRaisesRegex(Exception,'CPU Torch installation'):self.check()
    def test_namespace_escape_rejected(self):
        self.request['run_id']='../../elsewhere'
        with self.assertRaisesRegex(Exception,'Finite fresh storage request'):self.check()
    def test_output_namespace_must_match(self):
        self.request['output_root']=str(s.R/'another-output')
        with self.assertRaisesRegex(Exception,'Exact fresh storage output'):self.check()
    def test_missing_source_rejected(self):
        self.request['source_sha256']=dict(self.pins);self.request['source_sha256'].pop(str(s.SELF))
        with self.assertRaisesRegex(Exception,'complete approved source closure'):self.check()
    def test_premature_runtime_admission_rejected(self):
        self.request['runtime_admitted']=True
        with self.assertRaisesRegex(Exception,'Preparation grants no runtime'):self.check()
    def test_automatic_successor_rejected(self):
        self.request['automatic_successor_launch']=True
        with self.assertRaisesRegex(Exception,'Preparation grants no runtime'):self.check()
    def test_original_metadata_and_sources_in_closure(self):
        self.assertIn(str(s.capture.probe.bridge.CANDIDATE),self.pins);self.assertIn(str(s.capture.probe.bridge.loader.CONTRACT_PATH),self.pins)
        self.assertIn(str(s.capture.probe.bridge.loader.METADATA/'safetensors-header.json'),self.pins)
        self.assertIn(str(s.Q/'dory_weight_storage_capture_v1.py'),self.pins);self.assertLessEqual(len(self.pins),256)
    def test_missing_real_predecessor_prevents_lock_and_preparation(self):
        with patch.object(s,'predecessor',side_effect=RuntimeError('actual predecessor absent')),patch.object(s.m.old.post.external,'acquire_lock')as acquire:
            with self.assertRaisesRegex(RuntimeError,'actual predecessor absent'):s.prepare(reference('t'),reference('r'),reference('e'),'r1-20261108-v1')
            acquire.assert_not_called()
    def test_failed_request_prevents_native_lock(self):
        with patch.object(s,'load',return_value=self.request),patch.object(s,'validate_request',side_effect=RuntimeError('invalid request')),patch.object(s.m.old.post.external,'acquire_lock')as acquire:
            with self.assertRaisesRegex(RuntimeError,'invalid request'):s.run(reference('request'),reference('review'))
            acquire.assert_not_called()
    def test_correct_actual_history_drain_interface(self):
        s.m.load_history_helpers();import inspect
        self.assertEqual(list(inspect.signature(s.m.history_metadata.physical_drain).parameters),['value','spec'])
    def test_no_native_framework_loaded_or_future_request_created(self):
        self.assertFalse({'torch','numpy','torch_neuronx','transformers'}&set(sys.modules));self.assertFalse(list(s.Q.glob('dory-weight-storage-r*-request.json')))

if __name__=='__main__':unittest.main(verbosity=2)
