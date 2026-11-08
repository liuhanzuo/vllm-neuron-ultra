"""Dory finalizer source/synthetic tests; no model computation or actual completion."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'research')]
import dory_rotary_completed_history_v1 as h

def ref(name):return dict(path='/fixture/'+name,sha256=hashlib.sha256(name.encode()).hexdigest())
def owner(pid):return dict(pid=pid,start_ticks=str(pid+1000),ppid=1,argv=['fixture'],cgroup='fixture')
def fixture():
 request,review,finalreview,exitref,audit,catalog=[ref(x)for x in ('request','review','finalreview','exit','audit','catalog')]
 native,finalizer=owner(500001),owner(500002)
 native['argv']=[str(h.runner.PYTHON),'-I','-S','-B',str(Path(h.runner.__file__).resolve()),'--run','--request',request['path'],'--request-sha256',request['sha256'],'--review',review['path'],'--review-sha256',review['sha256']]
 exited=dict(schema='dory_rotary_actual_external_exit_v1',reviewer='/root',request=request,source_review=review,output_root=str(h.output_root('fixture')),actual_tool_exit_observed=True,exact_owner_gone=True,owner=native,actual_exit=dict(session_id=19,exit_code=0,tool_chunk='abcdef'))
 result=ref('result');post=dict(schema='dory_rotary_independent_saved_boundary_review_v1',complete=True,raw_tensors=53,raw_bytes=2230320,native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False,result=result,observer_invariant=True,arithmetic_parity=True,all_reference_outputs_reproduced=True,all_candidate_outputs_reproduced=True)
 terminal=dict(schema='dory_rotary_actual_terminal_v1',complete=True,owner=native,request=request,actual_self_exit_claimed=False,native_reference_qualified=False,neuron_baseline_qualified=False,new_speedup_claim=False,arithmetic_passed=True,reference_replay_passed=True,candidate_replay_passed=True,passed=True,result=result)
 prior=dict(schema='previous',reviewer='/root',rows=[dict(unit='fixture-'+str(i))for i in range(978)],complete_count=978,processes=[owner(1)],process_predecessors=[owner(1)],source_sha256={'/old.py':'e'*64},root_release_refs=[ref('old-release')],root_finalized=True,all_planned_launches_accounted_for=True,runtime_physical_drain_required=True,other_model_outcomes={'Phi':'incomplete','Mini':'rejected'},appended_from=ref('old'),appended_terminal_audit=ref('old-audit'),latest_producer='r6',latest_producer_qualified=False,latest_producer_exit=ref('old-exit'),latest_producer_terminal_audit=ref('old-audit'))
 outcome=h.classify(exited,terminal,None,post);pins={x['path']:x['sha256']for x in (request,review,finalreview,exitref,audit,catalog)}
 value=h.compose(prior,catalog,audit,exitref,request,review,finalreview,outcome,native,finalizer,pins)
 return dict(original=prior,value=value,exited=exited,terminal=terminal,post=post,request=request,review=review)

class ExitContract(unittest.TestCase):
 def setUp(self):self.f=fixture()
 def check(self):return h.validate_exit(self.f['exited'],self.f['request'],self.f['review'],h.output_root('fixture'))
 def test_real_shape_accepted(self):self.assertEqual(self.check(),0)
 def test_no_terminal_observation(self):
  self.f['exited']['actual_tool_exit_observed']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_boolean_exit_rejected(self):
  self.f['exited']['actual_exit']['exit_code']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_missing_real_session(self):
  self.f['exited']['actual_exit']['session_id']=None
  with self.assertRaises(RuntimeError):self.check()
 def test_wrong_native_flags(self):
  self.f['exited']['owner']['argv'].remove('-I')
  with self.assertRaises(RuntimeError):self.check()
 def test_wrong_source_review(self):
  self.f['exited']['source_review']=ref('other')
  with self.assertRaises(RuntimeError):self.check()
 def test_old_owner_alive_rejected(self):
  self.f['exited']['exact_owner_gone']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_wrong_output_namespace(self):
  self.f['exited']['output_root']=str(h.output_root('another'))
  with self.assertRaises(RuntimeError):self.check()

class Outcomes(unittest.TestCase):
 def setUp(self):self.f=fixture();self.failure=None
 def check(self):return h.classify(self.f['exited'],self.f['terminal'],self.failure,self.f['post'])
 def fail(self):
  self.f['exited']['actual_exit']['exit_code']=1
  self.failure=dict(owner=self.f['exited']['owner'],request=self.f['request'],error_type='RuntimeError',error='retained original failure',actual_self_exit_claimed=False,native_reference_qualified=False,new_speedup_claim=False)
 def test_local_success_is_not_model_qualification(self):
  result=self.check();self.assertTrue(result['arithmetic_qualified']);self.assertFalse(result['producer_qualified']);self.assertFalse(result['new_speedup_claim'])
 def test_zero_exit_needs_complete_result(self):
  self.f['terminal']=None
  with self.assertRaises(RuntimeError):self.check()
 def test_zero_exit_cannot_hide_failure(self):
  self.fail();self.f['exited']['actual_exit']['exit_code']=0
  with self.assertRaises(RuntimeError):self.check()
 def test_nonzero_preserved_even_if_arithmetic_passed(self):
  self.fail();self.assertFalse(self.check()['arithmetic_qualified'])
 def test_early_failure_with_no_complete_payloads_retained(self):
  self.fail();self.f['terminal']=None;self.f['post']=None;result=self.check();self.assertFalse(result['complete_saved_data_reviewed']);self.assertFalse(result['arithmetic_qualified'])
 def test_nonzero_missing_failure_preserves_unknown_cause(self):
  self.f['exited']['actual_exit']['exit_code']=137
  result=self.check();self.assertFalse(result['arithmetic_qualified']);self.assertFalse(result['failure_record_present']);self.assertIsNone(result['failure_cause'])
 def test_death_without_terminal_or_failure_preserved(self):
  self.f['exited']['actual_exit']['exit_code']=137;self.f['terminal']=None;self.f['post']=None
  result=self.check();self.assertFalse(result['complete_saved_data_reviewed']);self.assertFalse(result['producer_qualified']);self.assertIsNone(result['failure_cause'])
 def test_reduced_saved_scope_rejected(self):
  self.f['post']['raw_tensors']=52
  with self.assertRaises(RuntimeError):self.check()
 def test_post_review_model_promotion_rejected(self):
  self.f['post']['neuron_baseline_qualified']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_review_booleans_must_be_booleans(self):
  self.f['post']['arithmetic_parity']=1
  with self.assertRaises(RuntimeError):self.check()
 def test_arithmetic_mismatch_retained(self):
  self.fail();self.f['terminal'].update(passed=False,arithmetic_passed=False,candidate_replay_passed=False);self.f['post'].update(arithmetic_parity=False,all_candidate_outputs_reproduced=False)
  self.assertFalse(self.check()['arithmetic_qualified'])
 def test_saved_review_disagrees(self):
  self.f['post']['all_reference_outputs_reproduced']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_independent_review_missing(self):
  self.f['post']=None
  with self.assertRaises(RuntimeError):self.check()
 def test_wrong_original_request(self):
  self.f['terminal']['request']=ref('other')
  with self.assertRaises(RuntimeError):self.check()
 def test_same_pid_new_owner_start_rejected(self):
  self.f['terminal']['owner']=dict(self.f['terminal']['owner'],start_ticks='999999')
  with self.assertRaises(RuntimeError):self.check()
 def test_no_model_promotion(self):
  self.f['terminal']['native_reference_qualified']=True
  with self.assertRaises(RuntimeError):self.check()

class History(unittest.TestCase):
 def setUp(self):self.f=fixture()
 def check(self):return h.validate_extension(self.f['original'],self.f['value'])
 def test_complete_append(self):self.assertTrue(self.check())
 def test_drop_previous_service(self):
  self.f['value']['rows'].pop()
  with self.assertRaises(RuntimeError):self.check()
 def test_rewrite_previous_failure(self):
  self.f['value']['other_model_outcomes']['Mini']='passed'
  with self.assertRaises(RuntimeError):self.check()
 def test_drop_previous_process(self):
  self.f['value']['processes'].pop(0)
  with self.assertRaises(RuntimeError):self.check()
 def test_drop_native_owner(self):
  self.f['value']['processes'].pop(-2)
  with self.assertRaises(RuntimeError):self.check()
 def test_finalizer_must_need_external_exit(self):
  self.f['value']['dory_rotary_append']['processes'][-1]['next_admission_must_prove_gone']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_no_raw_release_invention(self):
  self.f['value']['root_release_refs'].append(ref('invented'))
  with self.assertRaises(RuntimeError):self.check()
 def test_no_source_rebinding(self):
  self.f['value']['source_sha256']['/old.py']='a'*64
  with self.assertRaises(RuntimeError):self.check()
 def test_missing_new_exit_pin(self):
  del self.f['value']['source_sha256'][self.f['value']['dory_rotary_append']['actual_exit']['path']]
  with self.assertRaises(RuntimeError):self.check()
 def test_no_whole_model_promotion(self):
  self.f['value']['latest_producer_qualified']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_no_unreviewed_extra_field(self):
  self.f['value']['unreviewed']=True
  with self.assertRaises(RuntimeError):self.check()

class SourceAndOutput(unittest.TestCase):
 def test_source_only_import(self):self.assertFalse(set(sys.modules)&{'torch','numpy','transformers','torch_neuronx'})
 def test_source_closure_includes_reviewer_and_self(self):
  paths=set(h.source_paths());self.assertIn(str(h.SELF),paths);self.assertIn(str(R/'research/review_dory_rotary_saved_boundaries_v1.py'),paths);self.assertIn(str(Path(__file__).resolve()),paths)
 def test_invalid_run_id(self):
  for name in ('../escape','', 'x'*33):
   with self.assertRaises(RuntimeError):h.catalog_path(name)
 def test_immutable_writer_and_exact_digest(self):
  with tempfile.TemporaryDirectory()as directory:
   path=Path(directory)/'result.json';reference=h.write_once(path,dict(complete=False));self.assertEqual(reference['sha256'],hashlib.sha256(path.read_bytes()).hexdigest());self.assertEqual(json.loads(path.read_text()),dict(complete=False))
   with self.assertRaises(FileExistsError):h.write_once(path,dict(complete=True))
 def test_finalizer_refuses_live_observer_before_reads(self):
  with patch.object(h.runner,'ensure_observer',side_effect=RuntimeError('active predecessor')),patch.object(h,'load_small')as load:
   with self.assertRaises(RuntimeError):h.finalize(ref('not-real'),ref('not-real-review'))
   load.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
