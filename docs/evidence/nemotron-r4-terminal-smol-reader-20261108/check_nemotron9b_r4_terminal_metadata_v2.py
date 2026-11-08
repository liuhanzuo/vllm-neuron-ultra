"""Real retained histories and explicit negative fixtures; no terminal fabrication."""
from pathlib import Path
import copy,json,sys,types,unittest
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import nemotron9b_r4_terminal_metadata_v2 as m
import finalize_nemotron9b_package_r4_root_20261108_v3 as f
import nemotron9b_package_protocol_v10 as p
import nemotron9b_package_protocol_v7 as p7
spec=m.load(f.SPEC);live=m.load(f.LIVE);C=f.C
OLD=R/'nemotron9b-package-r3-20261108-control';OSPEC=m.ref(Q/'nemotron9b-package-runtime-v7-r3-20261108-spec.json');ospec=m.load(OSPEC);oreview=m.ref(Q/'nemotron9b-package-runtime-v7-r3-20261108-review.json');old_before=m.load(m.ref(OLD/'external-pre-stop.json'))
def outcome_fixture():
 before={'path':'/fixture/pre-stop.json','sha256':'1'*64};auditref={'path':'/fixture/audit.json','sha256':'2'*64};admission={'path':'/fixture/admission.json','sha256':'3'*64}
 release=dict(released=True,successful=True,failed=False,explicit_failed_release=False,audit=auditref,all_current_groups_recursively_drained=True,canonical_unchanged=True,automatic_successor_launch=False,model_qualified=True,new_speedup_claim=False)
 audit=dict(pre_stop=before,canonical_unchanged=True,passed=True,model_qualified=True,independent_retained_exits=True,automatic_successor_launch=False,new_speedup_claim=False,failures=[],completion=dict(full_model_qualified=True,awaiting_external_raw_byte_audit=False,external_raw_byte_audit={'fixture':True}))
 terminal=dict(passed=True,diagnostic_complete=True,all_groups_physically_drained=True,successful_units_retained_for_external=True,admission=admission)
 return release,audit,terminal,before,auditref,admission
class Outcomes(unittest.TestCase):
 def test_real_r3_failure_stays_failed(self):
  release=m.load(m.ref(OLD/'root-release.json'));audit=m.load(release['audit']);failed=m.load(m.ref(OLD/'failed-terminal.json'))
  v=m.classify(1,release,audit,None,failed,m.ref(OLD/'external-pre-stop.json'),release['audit'],m.ref(OLD/'admission.json'));self.assertFalse(v['producer_qualified']);self.assertTrue(v['failed_release'])
 def test_cleanup_zero_does_not_upgrade_failed_release(self):
  release=m.load(m.ref(OLD/'root-release.json'));audit=m.load(release['audit']);failed=m.load(m.ref(OLD/'failed-terminal.json'))
  v=m.classify(0,release,audit,None,failed,m.ref(OLD/'external-pre-stop.json'),release['audit'],m.ref(OLD/'admission.json'));self.assertFalse(v['producer_qualified'])
 def test_positive_outcome_is_explicitly_synthetic(self):
  r,a,t,b,ar,ad=outcome_fixture();self.assertTrue(m.classify(0,r,a,t,None,b,ar,ad)['producer_qualified'])
 def test_failed_outer_exit_does_not_become_successful_from_release(self):
  r,a,t,b,ar,ad=outcome_fixture();v=m.classify(1,r,a,t,None,b,ar,ad);self.assertFalse(v['producer_qualified']);self.assertTrue(v['external_scientific_audit_passed'])
 def test_changed_reference_and_contradictory_release_rejected(self):
  for field,value in [('audit',{'path':'/wrong','sha256':'0'*64}),('failed',True),('explicit_failed_release',True),('model_qualified',False),('released',False)]:
   r,a,t,b,ar,ad=outcome_fixture();r[field]=value
   with self.subTest(field=field),self.assertRaises(RuntimeError):m.classify(0,r,a,t,None,b,ar,ad)
 def test_missing_raw_or_independent_exit_audit_rejected(self):
  for mode in ['raw','independent','recovering','controller']:
   r,a,t,b,ar,ad=outcome_fixture()
   if mode=='raw':a['completion']['external_raw_byte_audit']=None
   elif mode=='independent':a['independent_retained_exits']=False
   elif mode=='recovering':a['recovery_only']=True
   else:t=None
   with self.subTest(mode=mode),self.assertRaises(RuntimeError):m.classify(0,r,a,t,None,b,ar,ad)
class ExitReceipt(unittest.TestCase):
 def fixture(self):return dict(reviewer='/root',actual_tool_exit_observed=True,execution_spec=f.SPEC,source_review=f.REVIEW,owner=copy.deepcopy(live['owner']),exact_owner_gone=True,actual_exit=dict(session_id=50501,exit_code=1,tool_chunk='explicit-test-fixture'))
 def test_shape_binding_only_not_physical_exit_claim(self):self.assertEqual(m.validate_actual_exit(self.fixture(),live,f.SPEC,f.REVIEW,50501),1)
 def test_wrong_session_owner_and_unobserved_exit_rejected(self):
  for mode in ['session','owner','unobserved','gone','code']:
   v=self.fixture()
   if mode=='session':v['actual_exit']['session_id']=1
   elif mode=='owner':v['owner']['start_ticks']='1'
   elif mode=='unobserved':v['actual_tool_exit_observed']=False
   elif mode=='gone':v['exact_owner_gone']=False
   else:v['actual_exit']['exit_code']=True
   with self.subTest(mode=mode),self.assertRaises(RuntimeError):m.validate_actual_exit(v,live,f.SPEC,f.REVIEW,50501)
 def test_finalizer_has_no_default_exit(self):
  with self.assertRaisesRegex(RuntimeError,'Real external exit'):f.main(types.SimpleNamespace(actual_exit=None,actual_exit_sha256=None))
  self.assertFalse(f.AUDIT.exists());self.assertFalse(f.CATALOG.exists())
class Stages(unittest.TestCase):
 def saved(self):return m.load(m.ref(C/'prepare-retained-exit.json'))
 def test_real_r4_retained_prepare_and_worker_result(self):
  row,summary=m.retained_stage(spec,'prepare',f.SPEC,f.REVIEW,p,C,self.saved());self.assertFalse(row['never_launched']);self.assertTrue(summary['normal_retained_exit_success']);self.assertTrue(summary['worker_reported_passed'])
 def test_real_r3_garbage_collected_stage_uses_retained_original_exit(self):
  saved=old_before['rows']['prepare'];self.assertEqual(saved['properties']['LoadState'],'not-found')
  row,summary=m.retained_stage(ospec,'prepare',OSPEC,oreview,p7,OLD,saved);self.assertFalse(row['never_launched']);self.assertTrue(summary['original_exit_success'])
 def test_real_r3_never_launched_phase_retained(self):
  row,summary=m.retained_stage(ospec,'verify-raw',OSPEC,oreview,p7,OLD,old_before['rows']['verify-raw']);self.assertTrue(row['never_launched']);self.assertIsNone(row['owner'])
 def test_changed_snapshot_owner_or_resource_rejected(self):
  for mode in ['owner','resource']:
   saved=self.saved()
   if mode=='owner':saved['owner']['start_ticks']='1'
   else:saved['properties']['MemoryMax']='1'
   with self.subTest(mode=mode),self.assertRaises(RuntimeError):m.retained_stage(spec,'prepare',f.SPEC,f.REVIEW,p,C,saved)
 def test_missing_captured_owner_not_invented(self):
  with self.assertRaisesRegex(RuntimeError,'actual retained owner'):m.retained_stage(spec,'prepare',f.SPEC,f.REVIEW,p,C,self.saved(),exists=lambda path:False if path.name=='prepare-owner.json'else path.exists())
 def test_live_snapshot_not_treated_as_original_exit(self):
  saved=self.saved();saved['properties']['MainPID']=str(saved['owner']['pid']);saved['properties']['SubState']='running'
  with self.assertRaisesRegex(RuntimeError,'original process exit'):m.retained_stage(spec,'prepare',f.SPEC,f.REVIEW,p,C,saved)
class Processes(unittest.TestCase):
 def test_duplicate_exact_owner_deduplicated_and_prior_unchanged(self):
  a=copy.deepcopy(live['owner']);prior=[dict(a,role='prior')];value=m.extend_processes(prior,[(a,'new'),(a,'another')]);self.assertEqual(value,prior);self.assertIsNot(value,prior)
 def test_conflicting_same_pid_start_rejected(self):
  a=copy.deepcopy(live['owner']);changed=dict(a,argv=['changed'])
  with self.assertRaises(RuntimeError):m.extend_processes([a],[(changed,'new')])
 def test_no_unknown_standalone_start(self):
  a=dict(live['owner'],start_ticks=None)
  with self.assertRaises(RuntimeError):m.extend_processes([],[(a,'new')])

class StoppedLive(unittest.TestCase):
 def saved(self):return m.load(m.ref(C/'external-pre-stop.json'))['rows']['compile']
 def test_actual_interrupted_compile_never_promoted(self):
  row,s=m.retained_stage(spec,'compile',f.SPEC,f.REVIEW,p,C,self.saved());self.assertFalse(row['never_launched']);self.assertTrue(s['interrupted_while_live']);self.assertFalse(s['original_exit_success']);self.assertIsNone(s['original_exit_properties'])
 def test_invalid_stop_or_identity_rejected(self):
  for mode in ['passed','physical','samples','group','exit','owner','resource']:
   def reader(path):
    v=m.load(m.ref(path))
    if path.name=='failed-terminal.json':
     if mode=='passed':v['passed']=True
     if mode=='physical':v['all_started_groups_physically_drained']=False
     d=next(x['drain']for x in v['cleanup']if x['phase']=='compile')
     if mode=='samples':d['samples']=d['samples'][:1]
     if mode=='group':d['samples'][0]['group']['group']='/wrong'
     if mode=='exit':d['successful_exit']=True
    if path.name=='compile-cleanup-before-stop.json':
     if mode=='owner':v['owner']['start_ticks']='1'
     if mode=='resource':v['properties']['MemoryMax']='1'
    return v
   with self.subTest(mode=mode),self.assertRaises(RuntimeError):m.retained_stage(spec,'compile',f.SPEC,f.REVIEW,p,C,self.saved(),read=reader)
 def test_live_independent_snapshot_rejected(self):
  saved=self.saved();saved['properties']['MainPID']=str(saved['owner']['pid'])
  with self.assertRaises(RuntimeError):m.retained_stage(spec,'compile',f.SPEC,f.REVIEW,p,C,saved)
 def test_actual_all53_plan_units(self):
  before=m.load(m.ref(C/'external-pre-stop.json'));summaries=[m.retained_stage(spec,stage,f.SPEC,f.REVIEW,p,C,before['rows'][stage])[1]for stage in ['controller',*spec['stages']]]
  self.assertEqual(len(summaries),53);self.assertEqual(sum(x['never_launched']for x in summaries),9);self.assertEqual(sum(x.get('normal_retained_exit_success',False)for x in summaries),42);self.assertEqual([x['stage']for x in summaries if x.get('interrupted_while_live')],['compile'])
if __name__=='__main__':unittest.main(verbosity=2)

