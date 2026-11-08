"""Synthetic future outcomes for validation only; actual failed predecessor files retained.
Mocked document loads below do not establish future completion or admission.
"""
import copy,hashlib,json,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v13 as p
import nemotron9b_post_smol_maintenance_v2 as m
from research import finalize_completed_cache_xz_maintenance_20261108_v1 as final

def fixture():
 docs={};pins={}
 def real(ref):
  value=p.metadata.load(ref);docs[ref['path']]=value;pins[ref['path']]=ref['sha256'];return value
 def add(path,value):
  ref=final.value_ref(Path(path),value);docs[ref['path']]=value;pins[ref['path']]=ref['sha256'];return ref
 prior=real(m.PRIOR);failed=real(m.FAILED);candidate=real(m.CANDIDATE)
 for ref in [failed['started'],failed['failure'],failed['manifest']]:real(ref)
 ps_ref=final.ref(m.PROBE_ROOT/'start.json');ps=real(ps_ref);canon=ps['canonical_sha256']
 verification=dict(files_rehashed=1039,directories_verified=253,uncompressed_tar_bytes=4608000000,uncompressed_tar_sha256='e'*64,tar_stream_byte_identical=True,archive=dict(path=str(m.PROBE_ROOT/'cache.tar.xz'),sha256='f'*64),identity={})
 pr=dict(ps,schema='completed_cache_recompression_probe_result_v1',new_archive=verification['archive'],new_archive_stat={},new_archive_allocated_bytes=2*2**30,tar_stream_byte_identical=True,member_verification=dict(files_rehashed=1039,directories_verified=253),all253original_cache_entries_untouched=True,old_archive_untouched=True,uncompressed_tar_sha256=verification['uncompressed_tar_sha256'],uncompressed_tar_bytes=verification['uncompressed_tar_bytes'])
 pr_ref=add(m.PROBE_ROOT/'result.json',pr)
 pe=dict(schema='completed_cache_xz_probe_root_exit_v1',reviewer='/root',owner=ps['owner'],actual_exit_observed_by_root=True,gone=True,terminal=True,exit_code=0,started=ps_ref,result=pr_ref,execution_handle=dict(kind='tool_session',session_id=43750,actual_completion_observed=True,exit_code=0,tool_chunk='abcdef'))
 pe_ref=add(m.Q/'completed-cache-xz-probe-root-exit-20261108-v1.json',pe)
 owner=dict(pid=900000001,start_ticks='999999',ppid=900000000,argv=['python3','-B',m.RETIRE_SOURCE['path'],'--probe-exit-sha256',pe_ref['sha256']],cgroup='0::/user.slice/user-1000.slice/synthetic-only.scope')
 rs=dict(schema='completed_cache_xz_retirement_start_v1',owner=owner,lock=dict(device=1,inode=2),source=m.RETIRE_SOURCE,catalog=m.PRIOR,failed_maintenance=m.FAILED,probe_exit=pe_ref,probe_result=pr_ref,probe_start=ps_ref,candidate=m.CANDIDATE,canonical_sha256=canon,deletion_performed=False)
 rs_ref=add(m.RETIRE_ROOT/'start.json',rs)
 rr=dict(rs,schema='completed_cache_xz_retirement_result_v1',archive=pr['new_archive'],manifest=failed['manifest'],old_archive=pr['original_archive'],verification=verification,first_process_scan=dict(processes=1,hits=[],permission_errors=[]),final_process_scan=dict(processes=1,hits=[],permission_errors=[]),original_allocated_bytes=4607754240,old_archive_allocated_bytes=failed['archive_file']['allocated_bytes'],archive_allocated_bytes=pr['new_archive_allocated_bytes'],retained_metadata_allocated_bytes=2**20,deletion_performed=True,all_original_directories_absent=True,old_duplicate_archive_absent=True,canonical_unchanged=True,nvme_free_after=49*2**30,retired_directories=253,retired_files=1039,external_self_exit_claimed=False,next_admission_must_verify_owner_exit=True,maintenance_owners_must_be_appended_to_complete_history=True,experiment_outputs_deleted=False,model_inputs_changed=False,model_or_compiler_started=False,new_speedup_claim=False,full_model_qualified=False)
 rr['storage']=dict(metadata_reserve_bytes=16*2**20,net_reclaimed_lower_bound=4607754240-rr['archive_allocated_bytes']-rr['retained_metadata_allocated_bytes']-16*2**20,projected_free_lower_bound=49*2**30)
 rr_ref=add(m.RETIRE_ROOT/'result.json',rr)
 drains=[dict(unit=row['unit']) for row in prior['rows']]
 source=final.ref(Path(final.__file__));sources=[source,m.old.SOURCE,m.PROBE_SOURCE,m.RETIRE_SOURCE]
 exited,audit,catalog,spec=final.build(prior,failed,pe,pe_ref,rs,rs_ref,rr,rr_ref,12345,'abcdef',drains,verification,sources)
 docs[str(final.EXIT)]=exited;docs[str(final.AUDIT)]=audit
 return spec,catalog,docs

class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.base=fixture()
 def setUp(self):self.spec,self.catalog,self.docs=copy.deepcopy(self.base)
 def validate(self):return m.validate(self.spec,self.catalog,lambda ref:self.docs[ref['path']])
 def reject(self):
  with self.assertRaises((RuntimeError,KeyError,AssertionError)):self.validate()
 def test_valid_three_outcomes_preserve_original_failed_model(self):
  result=self.validate();self.assertEqual(len(result['maintenance']),3);self.assertEqual(result['catalog_reference'],m.PRIOR);self.assertFalse(result['fresh_physical_drain_proven']);self.assertFalse(result['new_speedup_claim'])
 def test_missing_extension_rejected(self):self.spec.pop('required_post_smol_maintenance');self.reject()
 def test_missing_failed_owner_rejected(self):self.catalog['processes'].pop(-3);self.reject()
 def test_reordered_owners_rejected(self):self.catalog['processes'][-2:]=reversed(self.catalog['processes'][-2:]);self.reject()
 def test_rewritten_service_outcome_rejected(self):self.catalog['rows'][0]['source_test_changed']=True;self.reject()
 def test_original_source_rebound_rejected(self):name=next(iter(self.docs[m.PRIOR['path']]['source_sha256']));self.catalog['source_sha256'][name]='0'*64;self.reject()
 def test_original_failure_cannot_be_success(self):self.docs[m.FAILED['path']]['exit_code']=0;self.reject()
 def test_compression_cannot_claim_retirement(self):self.docs[str(m.PROBE_ROOT/'result.json')]['deletion_performed']=True;self.reject()
 def test_unobserved_probe_completion_rejected(self):self.docs[self.spec['required_post_smol_maintenance']['probe_exit']['path']]['actual_exit_observed_by_root']=False;self.reject()
 def test_unobserved_tool_completion_rejected(self):self.docs[str(final.EXIT)]['execution_handle']['actual_completion_observed']=False;self.reject()
 def test_wrong_retirement_producer_rejected(self):self.docs[str(final.EXIT)]['owner']['argv'][-1]='0'*64;self.reject()
 def test_missing_retirement_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['deletion_performed']=False;self.reject()
 def test_reduced_storage_floor_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['nvme_free_after']=47*2**30;self.reject()
 def test_reduced_recovery_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['storage']['net_reclaimed_lower_bound']=1;self.reject()
 def test_missing_member_rejected(self):self.docs[self.docs[m.FAILED['path']]['manifest']['path']]['files'].popitem();self.reject()
 def test_changed_canonical_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['canonical_sha256']={};self.reject()
 def test_archive_mismatch_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['archive']=dict(path='/tmp/wrong',sha256='a'*64);self.reject()
 def test_full_tar_mismatch_rejected(self):self.docs[str(final.AUDIT)]['archive_verification']['uncompressed_tar_sha256']='a'*64;self.reject()
 def test_incomplete_scan_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['final_process_scan']['permission_errors']=['123'];self.reject()
 def test_model_success_claim_rejected(self):self.docs[str(m.RETIRE_ROOT/'result.json')]['full_model_qualified']=True;self.reject()
 def test_missing_source_pin_rejected(self):self.spec['source_sha256'].pop(m.RETIRE_SOURCE['path']);self.reject()
 def test_fingerprint_reader(self):
  data=json.dumps(self.catalog).encode();sha=hashlib.sha256(data).hexdigest()
  self.assertEqual(m.source_bytes(m.CURRENT,sha,lambda path,limit:data),data)
  with self.assertRaises(RuntimeError):m.source_bytes(m.CURRENT,'0'*64,lambda path,limit:data)

if __name__=='__main__':unittest.main(verbosity=2)
