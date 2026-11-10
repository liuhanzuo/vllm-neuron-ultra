"""Source and mocked lifecycle tests. Never loads a model or real saved tensor."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_original32_layer1_observation_cpu_runner_v2 as runner
from research import minicpm5_after_nemotron_r11_v1 as history
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained

def reference(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)

def terminal_fixture(success=True):
 spec=json.loads(Path(history.SPEC['path']).read_text())
 refs={k:reference(k)for k in ('intent','pre','admission','release','audit','terminal','failed')}
 controller=dict(pid=387618,start_ticks='407024336',ppid=1583,argv=['synthetic-controller'],cgroup='synthetic')
 intent=dict(external_owner=copy.deepcopy(history.LIVE_OWNER),execution_spec=history.SPEC,source_review=history.REVIEW,
  canonical_sha256={'synthetic':'b'*64},lock_identity={'device':1,'inode':2})
 admission=dict(execution_spec=history.SPEC,source_review=history.REVIEW,controller=controller,
  canonical_sha256=intent['canonical_sha256'],lock_identity=intent['lock_identity'])
 pre=dict(launch_intent=refs['intent'],auditor=copy.deepcopy(history.LIVE_OWNER),
  rows={key:dict(owner=controller if key=='controller'else None)for key in spec['units']})
 release=dict(released=True,all_current_groups_recursively_drained=True,canonical_unchanged=True,
  successful=success,failed=not success,model_qualified=success,explicit_failed_release=not success,
  audit=refs['audit'],automatic_successor_launch=False,new_speedup_claim=False)
 audit=dict(passed=success,pre_stop=refs['pre'],canonical_unchanged=True,model_qualified=success,
  automatic_successor_launch=False,new_speedup_claim=False,independent_retained_exits=True,
  failures=[]if success else['synthetic failure'],completion=dict(full_model_qualified=success,
   awaiting_external_raw_byte_audit=False,external_raw_byte_audit={'synthetic':True}))
 terminal=dict(passed=True,diagnostic_complete=True,all_groups_physically_drained=True,
  successful_units_retained_for_external=True,admission=refs['admission'])if success else None
 failed=None if success else dict(passed=False,all_started_groups_physically_drained=True)
 receipt=dict(reviewer='/root',actual_tool_exit_observed=True,execution_spec=history.SPEC,
  source_review=history.REVIEW,owner=copy.deepcopy(history.LIVE_OWNER),exact_owner_gone=True,
  actual_exit=dict(session_id=74703,exit_code=0 if success else 1,tool_chunk='synthetic-only'))
 release_receipt=None
 if not success:
  release_receipt=copy.deepcopy(receipt)
  release_receipt['actual_exit']=dict(session_id=90001,exit_code=0,tool_chunk='synthetic-release-only')
  release_receipt['owner']=dict(pid=99001,start_ticks='999',ppid=1,
   argv=[history.p.PYTHON,'-B',spec['controller'],'--audit','--release-failed',
    '--spec',history.SPEC['path'],'--spec-sha256',history.SPEC['sha256'],
    '--review',history.REVIEW['path'],'--review-sha256',history.REVIEW['sha256']],
   cgroup='synthetic-release')
 return spec,receipt,dict(intent=intent,pre=pre,admission=admission,release=release,audit=audit,
  terminal=terminal,failed=failed),refs,release_receipt

class R11History(unittest.TestCase):
 def test_successful_terminal_binds_observed_launcher_and_all53_stages(self):
  spec,receipt,records,refs,release=terminal_fixture()
  got=history.terminal_chain(spec,receipt,records,refs,release)
  self.assertTrue(got['producer_qualified']);self.assertFalse(got['new_speedup_claim'])
 def test_failed_original_launch_and_separate_release_stay_failed(self):
  spec,receipt,records,refs,release=terminal_fixture(False)
  got=history.terminal_chain(spec,receipt,records,refs,release)
  self.assertFalse(got['producer_qualified']);self.assertTrue(got['failed_release'])
 def test_missing_failed_release_never_passes(self):
  spec,receipt,records,refs,_=terminal_fixture(False)
  with self.assertRaises(RuntimeError):history.terminal_chain(spec,receipt,records,refs,None)
 def test_wrong_session_owner_start_spec_or_missing_tool_exit_rejected(self):
  for action in ('session','owner','start','spec','not_observed','not_gone'):
   spec,receipt,records,refs,release=terminal_fixture()
   if action=='session':receipt['actual_exit']['session_id']+=1
   if action=='owner':receipt['owner']['pid']+=1
   if action=='start':receipt['owner']['start_ticks']='999'
   if action=='spec':receipt['execution_spec']=reference('other')
   if action=='not_observed':receipt['actual_tool_exit_observed']=False
   if action=='not_gone':receipt['exact_owner_gone']=False
   with self.subTest(action=action),self.assertRaises(RuntimeError):
    history.terminal_chain(spec,receipt,records,refs,release)
 def test_release_must_be_actual_distinct_supported_successful_tool(self):
  for action in ('same_session','same_owner','bad_code','custom_wrapper','not_observed'):
   spec,receipt,records,refs,release=terminal_fixture(False)
   if action=='same_session':release['actual_exit']['session_id']=74703
   if action=='same_owner':release['owner'].update(pid=history.LIVE_OWNER['pid'],start_ticks=history.LIVE_OWNER['start_ticks'])
   if action=='bad_code':release['actual_exit']['exit_code']=1
   if action=='custom_wrapper':release['owner']['argv'][2]='/synthetic/wrapper.py'
   if action=='not_observed':release['actual_tool_exit_observed']=False
   with self.subTest(action=action),self.assertRaises(RuntimeError):
    history.terminal_chain(spec,receipt,records,refs,release)
 def test_success_requires_independent_raw_audit_and_unfailed_terminal(self):
  for action in ('raw','failure','controller','missing_phase','canonical'):
   spec,receipt,records,refs,release=terminal_fixture()
   if action=='raw':records['audit']['completion']['external_raw_byte_audit']={}
   if action=='failure':records['failed']=dict(passed=False,all_started_groups_physically_drained=True)
   if action=='controller':records['terminal']['diagnostic_complete']=False
   if action=='missing_phase':records['pre']['rows'].pop('long')
   if action=='canonical':records['admission']['canonical_sha256']={}
   with self.subTest(action=action),self.assertRaises(RuntimeError):
    history.terminal_chain(spec,receipt,records,refs,release)
 def test_recovery_requires_original_audit_without_outcome_promotion(self):
  spec,receipt,records,refs,release=terminal_fixture(False)
  refs['original_audit']=reference('original_audit')
  records['original_audit']=dict(pre_stop=refs['pre'])
  records['audit'].update(recovery_only=True,original_audit=refs['original_audit'],success_not_promoted=True)
  self.assertTrue(history.terminal_chain(spec,receipt,records,refs,release)['failed_release'])
  records['audit']['success_not_promoted']=False
  with self.assertRaises(RuntimeError):history.terminal_chain(spec,receipt,records,refs,release)
 def test_optional_reference_bytes_are_checked_not_dropped(self):
  with patch.object(history.OriginalReader,'__call__',return_value={'synthetic':True}):
   reader=history.Reader(lambda:None);ref=reference('metadata');reader.sizes[ref['path']]=12
   self.assertEqual(reader(dict(ref,bytes=12)),{'synthetic':True})
   for n in (11,True):
    with self.assertRaises(RuntimeError):reader(dict(ref,bytes=n))
 def test_actual_source_scope_preserves52stages_resources_and_full_prior_counts(self):
  spec=json.loads(Path(history.SPEC['path']).read_text())
  review=json.loads(Path(history.REVIEW['path']).read_text())
  prior=dict(rows=[dict(unit='synthetic-old-'+str(i))for i in range(1202)],processes=[None]*399,
   completed_mini_saved_tap_r2=history.p.startup_history.MINI_EXIT,
   completed_nemotron_preexec_timing=history.p.startup_history.TIMING_EXIT,
   nemotron_outcome=dict(failed_release=True),previous_mini_failure_promoted=False)
  history.scope(spec,review,prior)
  for action in ('phase','resource','history','location','qualified'):
   a,b=copy.deepcopy(spec),copy.deepcopy(prior)
   if action=='phase':a['stages'].remove('long')
   if action=='resource':a['resources']['compile']['MemoryMax']='1'
   if action=='history':b['processes'].pop()
   if action=='location':a['control_root']=str(R/'wrong')
   if action=='qualified':b['previous_mini_failure_promoted']=True
   with self.subTest(action=action),self.assertRaises(RuntimeError):history.scope(a,review,b)
 def extension_fixture(self,success):
  spec,_,_,refs,_=terminal_fixture(success)
  prior=dict(rows=[dict(unit='old'+str(i))for i in range(1202)],
   processes=[dict(pid=i+1,start_ticks=str(i+2),ppid=1,argv=['old'],cgroup='old')for i in range(399)],
   original_failure=dict(passed=False),known_admitted_directories=['old-dir'])
  owners=[];rows=[];summaries=[]
  for index,stage in enumerate(['controller',*spec['stages']]):
   owner=dict(pid=10000+index,start_ticks=str(20000+index),ppid=1,argv=['synthetic-'+stage],cgroup='synthetic')
   launched=success or index<2
   summary=dict(stage=stage,never_launched=not launched)
   if launched:
    summary.update(owner=owner,original_exit_success=success,worker_reported_complete=success,
     worker_reported_passed=success,normal_retained_exit_success=success)
    owners.append((reference(stage),[owner]))
   rows.append(dict(unit=spec['units'][stage]));summaries.append(summary)
  outcome=dict(producer_qualified=success,failed_release=not success,new_speedup_claim=False)
  return prior,spec,rows,summaries,owners,refs,outcome,{}
 def test_complete_history_extends_all1255_services_and_preserves_old_failures(self):
  for success in (False,True):
   args=self.extension_fixture(success);prior=copy.deepcopy(args[0])
   result=history.extend(*args)
   self.assertEqual(result['rows'][:1202],prior['rows'])
   self.assertEqual(result['processes'][:399],prior['processes'])
   self.assertEqual(result['original_failure'],dict(passed=False))
   self.assertEqual(len(result['rows']),1255);self.assertEqual(result['complete_count'],1255)
   self.assertEqual(len(result['processes']),399+(53 if success else 2))
   self.assertFalse(result['current_physical_release_verified'])
   self.assertFalse(result['full_model_qualified']);self.assertFalse(result['new_speedup_claim'])
 def test_dropped_stages_missing_owner_reordering_or_failed_success_rejected(self):
  for action in ('stage','owner','order','success','old_history'):
   args=list(self.extension_fixture(True))
   if action=='stage':args[2].pop()
   if action=='owner':args[4].pop()
   if action=='order':args[3][1],args[3][2]=args[3][2],args[3][1]
   if action=='success':args[3][2]['worker_reported_passed']=False
   if action=='old_history':args[0]['processes'].pop()
   with self.subTest(action=action),self.assertRaises(RuntimeError):history.extend(*args)
 def test_no_dependency_authentication_before_actual_terminal(self):
  with patch.object(history.Reader,'__call__',side_effect=RuntimeError('no actual terminal')),patch.object(history.p,'bundle')as bundle:
   with self.assertRaisesRegex(RuntimeError,'no actual terminal'):history.completed(reference('missing'),None,lambda:None)
   bundle.assert_not_called()

class DrainHold(unittest.TestCase):
 def test_cleanup_failure_keeps_ownership_until_independent_drain(self):
  events=[];attempt=[0]
  def cleanup():
   attempt[0]+=1;events.append('cleanup')
   if attempt[0]==1:raise RuntimeError('still running')
  def verify():
   events.append('verify')
   if attempt[0]<2:raise RuntimeError('group populated')
   return dict(physically_drained=True,group='synthetic-empty')
  proof=retain_until_drained(cleanup,verify,lambda v:events.append('held'),lambda:events.append('retry'))
  self.assertTrue(proof['physically_drained']);self.assertEqual(events,['cleanup','verify','held','retry','cleanup','verify'])
 def test_receipt_failure_and_interrupted_wait_cannot_unwind_lock(self):
  calls=[0];errors=[]
  def cleanup():calls[0]+=1
  def verify():
   if calls[0]<3:raise RuntimeError('pending')
   return dict(physically_drained=True)
  def receipt(value):errors.append('receipt');raise OSError('full')
  def wait():errors.append('wait');raise KeyboardInterrupt()
  self.assertTrue(retain_until_drained(cleanup,verify,receipt,wait)['physically_drained'])
  self.assertEqual(calls[0],3);self.assertEqual(errors,['receipt','wait','receipt','wait'])
 def test_successful_cleanup_without_actual_drain_is_insufficient(self):
  calls=[0]
  def verify():calls[0]+=1;return dict(physically_drained=calls[0]>1)
  proof=retain_until_drained(lambda:None,verify,lambda v:None,lambda:None)
  self.assertTrue(proof['physically_drained']);self.assertEqual(calls[0],2)
 def test_independently_drained_service_can_release_despite_cleanup_receipt_error(self):
  def cleanup():raise OSError('receipt unavailable after stop')
  proof=retain_until_drained(cleanup,lambda:dict(physically_drained=True),lambda v:self.fail('already drained'),lambda:self.fail('already drained'))
  self.assertTrue(proof['physically_drained'])

class RunnerLifecycle(unittest.TestCase):
 def test_outer_early_failure_releases_only_after_actual_drain_and_preserves_primary_error(self):
  events=[];checks=[0]
  with tempfile.TemporaryDirectory(prefix='mini-layer1-lifecycle-')as d:
   spec=dict(output_root=str(Path(d)/'uncreated'),unit='synthetic')
   def acquire():events.append('acquire');return 999,dict(pid=123,start_ticks='456'),dict(inode=1)
   def prerequisites(*args):events.append('prerequisites');raise ValueError('original admission failure')
   def verify(*args):
    checks[0]+=1;events.append('verify'+str(checks[0]))
    if checks[0]==1:raise RuntimeError('group not drained')
    return dict(physically_drained=True)
   with patch.object(runner,'read_bundle',return_value=(spec,{})),patch.object(runner.p,'user_bus'),\
    patch.object(runner.post.external,'acquire_lock',side_effect=acquire),\
    patch.object(runner.post.external,'release_lock',side_effect=lambda fd:events.append('unlock')),\
    patch.object(runner.p,'lock_proof'),patch.object(runner.p,'clear_window'),patch.object(runner.p,'active_heavy'),\
    patch.object(runner,'check_prerequisites',side_effect=prerequisites),\
    patch.object(runner,'cleanup_owned_service'),patch.object(runner,'verify_owned_drain',side_effect=verify):
    with self.assertRaisesRegex(ValueError,'original admission failure'):runner.outer(reference('spec'),reference('review'))
   self.assertEqual(events,['acquire','prerequisites','verify1','verify2','unlock'])
 def test_unowned_loaded_service_cannot_be_stopped_or_reported_drained(self):
  spec=dict(output_root='/synthetic/uncreated',unit='synthetic')
  with patch.object(runner.p,'properties',return_value=dict(LoadState='loaded')),patch.object(runner.p,'command')as command:
   with self.assertRaises(RuntimeError):runner.verify_owned_drain(spec,None,None,None,False)
   command.assert_not_called()
 def test_live_recorded_owner_or_untracked_group_member_prevents_drain(self):
  props=dict(LoadState='not-found',InvocationID='',MainPID='0',ControlPID='0',ExecMainPID='0')
  spec=dict(output_root='/synthetic/uncreated',unit='synthetic')
  owner=dict(pid=123,start_ticks='456')
  with patch.object(runner.p,'properties',return_value=props),patch.object(runner.p,'gone',side_effect=RuntimeError('owner live')):
   with self.assertRaisesRegex(RuntimeError,'owner live'):runner.verify_owned_drain(spec,None,None,owner,True)
  with patch.object(runner.p,'properties',return_value=props),patch.object(runner.p,'gone'),\
    patch.object(runner.p,'empty_group',side_effect=RuntimeError('untracked child live')):
   with self.assertRaisesRegex(RuntimeError,'untracked child live'):runner.verify_owned_drain(spec,None,None,owner,True)
 def test_changed_service_identity_prevents_stop(self):
  spec=dict(output_root='/synthetic/owned',unit='synthetic',master_port=29555)
  a=reference('admission');attempt='a'*32;command=runner.service_command(spec,a,attempt)
  with patch.object(runner.p,'read',return_value=dict(command=command,admission=a)),\
    patch.object(runner.p,'properties',return_value=dict(LoadState='loaded')),\
    patch.object(runner,'validate_service_identity',side_effect=RuntimeError('replaced invocation')),\
    patch.object(runner.p,'command')as stop:
   with self.assertRaisesRegex(RuntimeError,'replaced invocation'):runner.cleanup_owned_service(spec,a,attempt,None,True)
   stop.assert_not_called()
 def test_collection_worker_and_result_validator_are_new_but_original_resources_retained(self):
  self.assertEqual(runner.PROPS,runner.original_runner.PROPS)
  self.assertIs(runner.validate_rank_result,runner.layer1_results.validate_rank_result)
  self.assertIs(runner.validate_rank_pair,runner.layer1_results.validate_rank_pair)
  tree=ast.parse(Path(runner.__file__).read_text())
  rank=next(x for x in tree.body if getattr(x,'name',None)=='rank')
  imports=[n.names[0].name for n in ast.walk(rank)if isinstance(n,ast.Import)]
  self.assertEqual(imports,['minicpm5_original32_layer1_observation_cpu_worker_v1'])
  self.assertEqual((runner.CAP,runner.RANK_CAP),(256*2**20,128*2**20))
 def test_coordinator_and_rank_scientific_logic_preserved(self):
  old=ast.parse(Path(runner.original_runner.__file__).read_text())
  new=ast.parse(Path(runner.__file__).read_text())
  for name in ('coordinator','rank'):
   a=next(x for x in old.body if getattr(x,'name',None)==name)
   b=next(x for x in new.body if getattr(x,'name',None)==name)
   class Normalize(ast.NodeTransformer):
    def visit_alias(self,n):
     if n.name=='minicpm5_original32_layer1_observation_cpu_worker_v1':
      n.name='minicpm5_original32_layer0_full_bf16_cpu_worker_v1'
     return n
   self.assertEqual(ast.dump(a),ast.dump(Normalize().visit(b)))
 def test_complete_drain_receipt_keeps1255services_and_two_samples(self):
  sample=dict(properties=dict(Id='synthetic',LoadState='not-found',MainPID='0',ControlPID='0',
   ExecMainPID='0',InvocationID='',Unrelated='not-needed'),group=dict(absent=True))
  proof=dict(legacy=dict(original=True),current=dict(processes=450,
   inherited=[dict(unit='old'+str(i),samples=[copy.deepcopy(sample),copy.deepcopy(sample)])for i in range(1202)],
   current=[dict(stage='new'+str(i),samples=[copy.deepcopy(sample),copy.deepcopy(sample)])for i in range(53)]))
  got=runner.compact_drain(proof)
  self.assertEqual(len(got['observations']),1255)
  self.assertTrue(all(len(x['samples'])==2 for x in got['observations']))
  self.assertTrue(all('Unrelated'not in x['samples'][0]['properties']for x in got['observations']))
  proof['current']['current'].pop()
  with self.assertRaises(RuntimeError):runner.compact_drain(proof)
 def test_source_import_does_not_load_native_runtime(self):
  self.assertFalse({'torch','numpy','torch_neuronx','transformers','vllm','neuronxcc'}&set(sys.modules))



class SourcePreparation(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  from research import prepare_minicpm5_layer1_runner_v2 as prepare
  cls.prepare=prepare
  cls.plan,cls.checks=prepare.source_candidate()
  ending=runner.p.read_ref(runner.OBSERVED_EXIT)
  cls.original=runner.p.read_ref(ending['spec'])
 def fixture(self):
  spec=copy.deepcopy(self.original)
  spec.update(schema='minicpm5_layer1_observation_cpu_execution_spec_v1',
   runner=runner.SELF,worker=str(R/'minicpm5_original32_layer1_observation_cpu_worker_v1.py'),
   output_root=self.plan['output_root'],source_sha256=self.plan['source_sha256'],
   output_floor_bytes=30*2**30,unit='trn1-mini-layer1-observation-synthetic',
   prepared_source_plan=runner.RESULTS_SOURCE_PLAN,original_completed_collection=runner.OBSERVED_EXIT,
   nemotron_completion=dict(actual_exit=reference('not-an-actual-terminal'),failed_release_exit=None),
   plan=reference('synthetic-plan'))
  spec_ref=reference('synthetic-spec')
  review=dict(schema='minicpm5_layer1_observation_runtime_source_review_v1',
   reviewer='/root',passed=True,blockers=[],spec=spec_ref,source_sha256=spec['source_sha256'])
  return spec,review,spec_ref,reference('synthetic-review')
 def read_bundle(self,spec,review,spec_ref,review_ref,plan=None):
  values={spec_ref['path']:spec,review_ref['path']:review,spec['plan']['path']:self.plan if plan is None else plan}
  with patch.object(runner.p,'read_ref',side_effect=lambda ref:values[ref['path']]):
   return runner.read_bundle(spec_ref,review_ref)
 def test_actual_source_closure_accepts_all_original_text_types_without_native_reads(self):
  spec,review,sref,rref=self.fixture()
  got,plan=self.read_bundle(spec,review,sref,rref)
  self.assertEqual(plan,self.plan)
  for suffix in ('.jinja','.pth','.metadata','.log','.md','','.txt','.patch'):
   self.assertIn(suffix,self.checks['source_suffix_counts'])
  self.assertFalse(self.checks['native_dependency_authentication_performed'])
  self.assertFalse(self.checks['runtime_admitted'])
 def test_worker_inputs_cold_long_slot_and_quality_plan_unchanged(self):
  previous=runner.p.read_ref(runner.RESULTS_SOURCE_PLAN)
  self.assertEqual({k:v for k,v in self.plan.items()if k not in ('source_sha256','output_root')},
   {k:v for k,v in previous.items()if k not in ('source_sha256','output_root')})
  self.assertTrue(all(self.plan['source_sha256'][p]==sha for p,sha in previous['source_sha256'].items()))
  self.assertEqual(len(self.plan['layer1_observation']['original_calls']),84)
  self.assertEqual((len(self.plan['calls']),len(self.plan['frames'])),(14,32))
 def test_changed_source_or_worker_or_output_or_resources_rejected(self):
  for action in ('source','worker','output','cap','memory','nvme_floor','review','prior_plan'):
   spec,review,sref,rref=self.fixture()
   if action=='source':
    spec['source_sha256']=dict(spec['source_sha256']);spec['source_sha256'][runner.SELF]='0'*64
    review['source_sha256']=spec['source_sha256']
   if action=='worker':spec['worker']=str(R/'minicpm5_original32_layer0_full_bf16_cpu_worker_v1.py')
   if action=='output':spec['output_root']='/mnt/nvme/unrelated'
   if action=='cap':spec['output_cap_bytes']+=1
   if action=='memory':spec['resources']['MemoryMax']='1'
   if action=='nvme_floor':spec['output_floor_bytes']=8*2**30
   if action=='review':review['passed']=False
   if action=='prior_plan':spec['prepared_source_plan']=reference('wrong')
   with self.subTest(action=action),self.assertRaises(RuntimeError):self.read_bundle(spec,review,sref,rref)
 def test_final_runtime_candidate_requires_actual_terminal_before_source_assembly(self):
  with patch.object(self.prepare.history.Reader,'__call__',side_effect=RuntimeError('missing actual terminal')),    patch.object(self.prepare,'source_candidate')as source:
   with self.assertRaisesRegex(RuntimeError,'missing actual terminal'):
    self.prepare.assemble_runtime_candidate(reference('absent'),None,'r1-synthetic')
   source.assert_not_called()
 def test_source_candidate_never_creates_output_or_runtime_spec(self):
  self.assertFalse(Path(self.plan['output_root']).exists())
  self.assertFalse(self.checks['runtime_spec_created'])
  self.assertFalse(self.checks['model_or_real_tensor_payloads_read'])
  self.assertFalse({'torch','numpy','torch_neuronx','transformers','vllm','neuronxcc'}&set(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
