"""Source-only exact runtime-bundle checks; no runtime spec or launch is written."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
import minicpm5_layer1_own_history_cpu_runner_v1 as runner
from research import prepare_minicpm5_layer1_own_history_runner_v1 as prepare


class AdmissionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((Q/'minicpm5-layer1-own-history-runner-source-plan-20261110-v1.json').read_text())
  for module in tuple(sys.modules.values()):
   path=getattr(module,'__file__',None)
   if path and str(path).startswith(str(R)+'/'):
    path=str(Path(path).resolve());sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert path not in cls.plan['source_sha256']or cls.plan['source_sha256'][path]==sha
    cls.plan['source_sha256'][path]=sha
  ended=runner.p.read_ref(runner.OBSERVED_EXIT);cls.original=runner.p.read_ref(ended['spec'])
 def fixture(self):
  ref=lambda name:dict(path='/synthetic/'+name+'.json',sha256='a'*64)
  plan=copy.deepcopy(self.plan)
  binding=dict(schema='minicpm5_layer1_completed_collection_and_audit_v1',collector_exit=ref('collector-exit'),saved_audit_exit=ref('audit-exit'))
  plan['layer1_own_history_replay']=binding
  spec=copy.deepcopy(self.original)
  spec.update(schema='minicpm5_layer1_own_history_cpu_execution_spec_v1',runner=runner.SELF,
   worker=str(R/'minicpm5_layer1_own_history_cpu_worker_v1.py'),output_root=plan['output_root'],
   source_sha256=plan['source_sha256'],output_floor_bytes=30*2**30,unit='trn1-mini-layer1-own-history-synthetic',
   prepared_source_plan=runner.RESULTS_SOURCE_PLAN,original_completed_collection=runner.OBSERVED_EXIT,
   nemotron_completion=dict(actual_exit=ref('nemotron-exit'),failed_release_exit=None),
   completed_layer1_data=binding,plan=ref('plan'))
  sr,rr=ref('spec'),ref('review')
  review=dict(schema='minicpm5_layer1_own_history_runtime_source_review_v1',reviewer='/root',passed=True,
   blockers=[],spec=sr,source_sha256=spec['source_sha256'])
  return spec,plan,review,sr,rr
 def read(self,values):
  spec,plan,review,sr,rr=values;docs={sr['path']:spec,rr['path']:review,spec['plan']['path']:plan}
  with patch.object(runner.p,'read_ref',side_effect=lambda ref:docs[ref['path']]):return runner.read_bundle(sr,rr)
 def test_all_actual_source_pins_and_runtime_shape_checked_without_launch(self):
  values=self.fixture();spec,plan=self.read(values)
  self.assertEqual(spec,values[0]);self.assertEqual(plan,values[1])
  self.assertFalse(Path(plan['output_root']).exists())
 def test_changed_source_or_worker_rejected(self):
  for action in ('source','worker'):
   values=self.fixture();spec,plan,review,*_=values
   if action=='source':spec['source_sha256'][runner.SELF]='0'*64
   else:spec['worker']=runner.SELF
   with self.subTest(action=action),self.assertRaises(RuntimeError):self.read(values)
 def test_mismatched_actual_data_binding_rejected(self):
  values=self.fixture();values[0]['completed_layer1_data']=dict(values[0]['completed_layer1_data'],saved_audit_exit={})
  with self.assertRaises(RuntimeError):self.read(values)
 def test_changed_output_resources_or_precision_cannot_enter_bundle(self):
  for key,value in (('output_root','/mnt/nvme/unrelated'),('output_cap_bytes',1),('root_floor_bytes',1),
    ('output_floor_bytes',1),('resources',{}),('unit','unrelated'),('master_port',True)):
   values=self.fixture();values[0][key]=value
   with self.subTest(key=key),self.assertRaises(RuntimeError):self.read(values)
 def test_unreviewed_or_qualified_claim_rejected(self):
  for action in ('review','blocked','claim'):
   values=self.fixture()
   if action=='review':values[2]['passed']=False
   elif action=='blocked':values[2]['blockers']=['pending']
   else:values[0]['model_qualification_claim']=True
   with self.subTest(action=action),self.assertRaises(RuntimeError):self.read(values)
 def test_actual_lineage_failure_precedes_native_or_service_start(self):
  values=self.fixture()
  with patch.object(runner.after_audit,'completed',side_effect=ValueError('missing actual audit exit')),    patch.object(runner.collector,'check_prerequisites')as prior,patch.object(runner.p,'command')as command:
   with self.assertRaisesRegex(ValueError,'actual audit exit'):runner.check_prerequisites(values[0],lambda:None)
  prior.assert_not_called();command.assert_not_called()
 def test_original_full_scientific_inputs_and_gates_preserved(self):
  original=runner.p.read_ref(runner.RESULTS_SOURCE_PLAN)
  self.assertEqual({k:v for k,v in self.plan.items()if k not in ('source_sha256','output_root')},
   {k:v for k,v in original.items()if k not in ('source_sha256','output_root')})
  self.assertEqual(len(self.plan['layer1_observation']['original_calls']),84)
  self.assertFalse({'torch','numpy','torch_neuronx','transformers','vllm'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
