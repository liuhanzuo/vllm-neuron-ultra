"""Synthetic maintenance metadata controls; no claimed real Smol/cleanup exit.

The real cache selection is read, but no cache payload/archive or process is
opened. All future completion records below are in-memory test fixtures only.
"""
from pathlib import Path
import copy,hashlib,json,sys,unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_post_smol_maintenance_v1 as m
CANDIDATE=json.loads(Path(m.CANDIDATE['path']).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fixture():
 docs={}
 def doc(path,value):
  ref=dict(path=str(path),sha256=digest(value));docs[str(path)]=value;return ref
 oldpin='/source-only-fixture/unchanged.py';oldrelease=dict(path='/source-only-fixture/failed-release.json',sha256='a'*64)
 prior=dict(schema='root_post_smollm3_sio14_complete_predecessors_v1',reviewer='/root',root_finalized=True,all_planned_launches_accounted_for=True,
  complete_count=1,rows=[dict(unit='fixture-historical-service')],processes=[dict(pid=100,start_ticks='100',role='fixture-historical-owner')],source_sha256={oldpin:'b'*64},root_release_refs=[oldrelease],latest_producer_qualified=False,failed_experiment_retained=True,original_failure='fixture original failure retained')
 prior_ref=doc(m.Q/'root-post-smol-sio14-complete-predecessors-20261108-v1.json',prior)
 owner=dict(pid=200,start_ticks='200',ppid=199,argv=['python3','-B',m.SOURCE['path'],'--apply','--catalog',prior_ref['path'],'--catalog-sha256',prior_ref['sha256']],cgroup='0::/user.slice/user-1000.slice/session-fixture.scope')
 canonical={'fixture_canonical':'c'*64};lock=dict(device=1,inode=2)
 started=dict(owner=owner,lock=lock,candidate=m.CANDIDATE,source=m.SOURCE,canonical_sha256=canonical,actual_completed_catalog=prior_ref)
 start_ref=doc(m.ARCHIVE_ROOT/'start.json',started)
 manifest=dict(files={str(Path(row['path']).relative_to(CANDIDATE['cache_root'])):dict(row,sha256='d'*64)for entry in CANDIDATE['candidates']for row in entry['files']},directories={entry['path']:entry['directory_identity']for entry in CANDIDATE['candidates']})
 manifest_ref=doc(m.ARCHIVE_ROOT/'files.json',manifest)
 result=dict(owner=owner,lock=lock,candidate=m.CANDIDATE,source=m.SOURCE,canonical_sha256=canonical,actual_completed_catalog=prior_ref,
  deletion_performed=True,all_original_directories_absent=True,canonical_unchanged=True,external_self_exit_claimed=False,next_admission_must_verify_owner_exit=True,maintenance_owner_must_be_appended_to_complete_history=True,
  experiment_outputs_deleted=False,model_inputs_changed=False,model_or_compiler_started=False,files_verified=1039,directories_verified=253,original_allocated_bytes=4607754240,archive_allocated_bytes=1000000000,net_allocated_reclaimed_bytes=3607754240,required_net_recovery_bytes=2*2**30,nvme_free_after=49*2**30,
  archive=dict(path=str(m.ARCHIVE_ROOT/'cache.tar.gz'),sha256='e'*64),manifest=manifest_ref,first_process_scan=dict(hits=[],permission_errors=[],processes=10),final_process_scan=dict(hits=[],permission_errors=[],processes=10))
 result_ref=doc(m.Q/'storage-nvme-completed-cache-retirement-20261108-v2.json',result)
 exited=dict(schema='completed_cache_archival_root_exit_v1',reviewer='/root',owner=owner,actual_exit_observed_by_root=True,gone=True,terminal=True,exit_code=0,result=result_ref,
  execution_handle=dict(kind='tool_session',session_id=999,actual_completion_observed=True,exit_code=0,tool_chunk='abc123'))
 exit_ref=doc(m.Q/'fixture-cache-exit.json',exited)
 ext=dict(prior_catalog=prior_ref,started=start_ref,result=result_ref,actual_exit=exit_ref,source=m.SOURCE,candidate=m.CANDIDATE)
 audit=dict(schema='completed_cache_archival_root_terminal_v1',reviewer='/root',passed=True,references=copy.deepcopy(ext),owner=owner,all_current_groups_recursively_drained=True,owner_absence_observed=True,archive_bytes_rehashed=True,
  archive=result['archive'],manifest=manifest_ref,canonical_sha256=canonical,full_model_qualified=False,new_speedup_claim=False,complete_services=1,complete_processes=2)
 ext['root_audit']=doc(m.Q/'fixture-cache-terminal.json',audit)
 catalog=copy.deepcopy(prior);catalog.update(schema=m.SCHEMA,maintenance_extension=ext,processes=prior['processes']+[dict(owner,role='completed_cache_archival_maintenance')])
 for ref in [*ext.values(),manifest_ref]:catalog['source_sha256'][ref['path']]=ref['sha256']
 current=doc(m.Q/'nemotron9b-package-post-smol-maintenance-fixture-catalog-v1.json',catalog)
 spec=dict(final_predecessor_catalog=current,source_sha256=dict(catalog['source_sha256'],**{current['path']:current['sha256']}),required_predecessor_releases=prior['root_release_refs'],required_post_smol_maintenance=copy.deepcopy(ext))
 docs[m.CANDIDATE['path']]=copy.deepcopy(CANDIDATE)
 return spec,catalog,docs

class Checks(unittest.TestCase):
 def setUp(self):self.spec,self.catalog,self.docs=fixture()
 def run_bridge(self):return m.validate(self.spec,self.catalog,lambda ref:self.docs[ref['path']])
 def document(self,key):return self.docs[self.catalog['maintenance_extension'][key]['path']]
 def rejected(self):
  with self.assertRaises((RuntimeError,KeyError,TypeError,ValueError)):self.run_bridge()
 def test_fixture_retains_original_failure_and_never_claims_fresh_drain(self):
  value=self.run_bridge();self.assertFalse(value['catalog']['latest_producer_qualified']);self.assertTrue(value['catalog']['failed_experiment_retained']);self.assertFalse(value['fresh_physical_drain_proven']);self.assertFalse(value['new_speedup_claim']);self.assertEqual(len(value['maintenance']),1)
 def test_no_maintenance_direct_catalog_path_preserved(self):
  anchor=self.catalog['maintenance_extension']['prior_catalog'];prior=self.docs[anchor['path']];spec=dict(self.spec,final_predecessor_catalog=anchor);spec.pop('required_post_smol_maintenance');value=m.validate(spec,prior,lambda r:self.docs[r['path']]);self.assertEqual(value['catalog_reference'],anchor);self.assertEqual(value['maintenance'],[])
 def test_maintenance_cannot_be_omitted(self):self.spec.pop('required_post_smol_maintenance');self.rejected()
 def test_different_selected_extension_rejected(self):self.spec['required_post_smol_maintenance']['source']['sha256']='0'*64;self.rejected()
 def test_source_removed_rejected(self):self.spec['source_sha256'].pop(m.SOURCE['path']);self.rejected()
 def test_old_failure_promoted_rejected(self):self.catalog['latest_producer_qualified']=True;self.rejected()
 def test_old_service_removed_rejected(self):self.catalog['rows']=[];self.rejected()
 def test_extra_service_rejected(self):self.catalog['rows'].append(dict(unit='other'));self.catalog['complete_count']+=1;self.rejected()
 def test_old_process_dropped_rejected(self):self.catalog['processes']=self.catalog['processes'][1:];self.rejected()
 def test_maintenance_owner_omitted_rejected(self):self.catalog['processes'].pop();self.rejected()
 def test_duplicate_maintenance_owner_rejected(self):self.catalog['processes'].append(self.catalog['processes'][-1]);self.rejected()
 def test_owner_mismatch_rejected(self):self.document('actual_exit')['owner']=dict(self.document('actual_exit')['owner'],start_ticks='999');self.rejected()
 def test_unobserved_tool_completion_rejected(self):self.document('actual_exit')['execution_handle']['actual_completion_observed']=False;self.rejected()
 def test_nonzero_tool_exit_rejected(self):self.document('actual_exit')['exit_code']=1;self.rejected()
 def test_boolean_exit_code_rejected(self):self.document('actual_exit')['exit_code']=False;self.rejected()
 def test_live_owner_rejected(self):self.document('actual_exit')['gone']=False;self.rejected()
 def test_wrong_smol_anchor_argv_rejected(self):self.document('started')['owner']['argv'][-1]='0'*64;self.rejected()
 def test_failed_archive_not_promoted(self):self.document('result')['deletion_performed']=False;self.rejected()
 def test_insufficient_net_space_rejected(self):self.document('result')['net_allocated_reclaimed_bytes']=1;self.rejected()
 def test_live_reader_rejected(self):self.document('result')['final_process_scan']['hits']=[dict(pid=1)];self.rejected()
 def test_scan_permission_error_rejected(self):self.document('result')['first_process_scan']['permission_errors']=['denied'];self.rejected()
 def test_missing_independent_archive_hash_rejected(self):self.document('root_audit')['archive_bytes_rehashed']=False;self.rejected()
 def test_speedup_from_maintenance_rejected(self):self.document('root_audit')['new_speedup_claim']=True;self.rejected()
 def test_wrong_manifest_directory_rejected(self):
  ref=self.document('result')['manifest'];dirs=self.docs[ref['path']]['directories'];value=dirs.pop(next(iter(dirs)));dirs['/wrong/directory']=value;self.rejected()
 def test_archive_member_path_relabel_rejected(self):
  ref=self.document('result')['manifest'];files=self.docs[ref['path']]['files'];value=files.pop(next(iter(files)));files['../escape']=value;self.rejected()
 def test_manifest_member_identity_rejected(self):
  ref=self.document('result')['manifest'];next(iter(self.docs[ref['path']]['files'].values()))['inode']+=1;self.rejected()
 def test_large_catalog_reader_keeps_original_limit_and_hash(self):
  data=json.dumps(self.catalog).encode();wanted=hashlib.sha256(data).hexdigest();caps=[]
  def read(path,cap):caps.append(cap);return data
  self.assertEqual(m.source_bytes(self.spec['final_predecessor_catalog']['path'],wanted,read),data);self.assertEqual(caps,[16*2**20])
  with self.assertRaises(RuntimeError):m.source_bytes(self.spec['final_predecessor_catalog']['path'],'0'*64,read)
 def test_unrelated_large_namespace_rejected(self):
  with self.assertRaises(RuntimeError):m.source_bytes('/tmp/unrelated.json','0'*64,lambda p,c:b'{}')
 def test_real_archive_source_and_candidate_hashes_match_pins(self):
  for ref in [m.SOURCE,m.CANDIDATE]:self.assertEqual(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest(),ref['sha256'])
if __name__=='__main__':unittest.main(verbosity=2)
