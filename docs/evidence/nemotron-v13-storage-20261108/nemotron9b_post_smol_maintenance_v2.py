"""Authenticate failed gzip, successful XZ probe and verified retirement separately.
Pure metadata validation. Fresh physical drain is always required at admission.
"""
import copy,hashlib,json,re
from pathlib import Path
import nemotron9b_post_smol_maintenance_v1 as old
require=old.require;reference=old.reference;R=old.R;Q=old.Q;LIMIT=old.LIMIT
SCHEMA='nemotron9b_post_smol_maintenance_catalog_v2'
CURRENT=Q/'nemotron9b-package-post-smol-maintenance-r4-20261108-catalog-v2.json'
PRIOR=dict(path=str(Q/'root-post-smol-sio14-complete-predecessors-20261108-v1.json'),sha256='63a5c8bba41fb1e377d29029fa9fc212163774721e529c58f3d872952fc16980')
FAILED=dict(path=str(Q/'completed-cache-archival-r2-failure-root-exit-20261108-v1.json'),sha256='27b9eb59790fb35ac4203927546899c44f400727c96a0f7364e1fbe15b25a003')
CANDIDATE=old.CANDIDATE
PROBE_SOURCE=dict(path=str(Q/'probe_completed_cache_xz_recompression_20261108_v1.py'),sha256='b7246ef058f32bcd3e87ceba7dab6ea5a7cf0a74b0430e04cdd3d349593e63ba')
RETIRE_SOURCE=dict(path=str(Q/'retire_completed_cache_xz_20261108_v1.py'),sha256='4bef7ca011eb84bbde6bed248f3f1a275a2a1a56772dc3cc1430156f976cfccb')
PROBE_ROOT=Path('/mnt/nvme/retired-compile-cache-20261108-r3-probe')
RETIRE_ROOT=Path('/mnt/nvme/retired-compile-cache-20261108-r4-retirement')
EXT_KEYS={'prior_catalog','failed_exit','probe_exit','retirement_exit','root_audit','candidate'}
ROLES=('completed_cache_failed_gzip_maintenance','completed_cache_xz_probe_maintenance','completed_cache_xz_retirement_maintenance')

def accepts(path):return Path(path)==CURRENT or old.accepts(path)

def shape(value):
 if value.get('schema')==old.SCHEMA:return old.shape(value)
 require(value.get('schema')==SCHEMA and value.get('reviewer')=='/root' and value.get('root_finalized') is True and value.get('all_planned_launches_accounted_for') is True,'Actual finalized three-producer maintenance history')
 ext=value.get('maintenance_extension');require(type(ext)is dict and set(ext)==EXT_KEYS,'Exact three-producer extension fields')
 for ref in ext.values():reference(ref);require(value['source_sha256'].get(ref['path'])==ref['sha256'],'Complete maintenance reference pins')
 require(ext['prior_catalog']==PRIOR and ext['failed_exit']==FAILED and ext['candidate']==CANDIDATE,'Original actual failed gzip and completed Smol retained')
 return ext

def source_bytes(path,expected,stable_reader):
 if old.accepts(path):return old.source_bytes(path,expected,stable_reader)
 require(Path(path)==CURRENT,'Exact maintenance history namespace');reference(dict(path=str(path),sha256=expected))
 raw=stable_reader(Path(path),LIMIT);require(hashlib.sha256(raw).hexdigest()==expected,'Maintenance catalog fingerprint');shape(json.loads(raw.decode('utf8')));return raw

def actual_exit(value,code,schema,session=None):
 require(value.get('schema')==schema and value.get('reviewer')=='/root','Exact external exit schema')
 require(value.get('actual_exit_observed_by_root') is value.get('gone') is value.get('terminal') is True and type(value.get('exit_code'))is int and value['exit_code']==code,'Actual observed terminal owner exit')
 owner=value['owner'];require(set(owner)==old.OWNER_KEYS and type(owner['pid'])is int and owner['pid']>0 and type(owner['ppid'])is int and owner['ppid']>0 and type(owner['start_ticks'])is str and owner['start_ticks'].isdigit(),'Exact original owner identity')
 require(type(owner['argv'])is list and owner['argv'][0]in ('python3','/usr/bin/python3') and owner['cgroup'].startswith('0::/user.slice/user-1000.slice/'),'Exact original process context')
 handle=value['execution_handle'];require(handle['kind']=='tool_session' and type(handle['session_id'])is int and handle['session_id']>0 and handle['actual_completion_observed'] is True and type(handle['exit_code'])is int and handle['exit_code']==code and re.fullmatch('[0-9a-f]{6}',handle['tool_chunk']) is not None,'Actual tool completion authority')
 if session is not None:require(handle['session_id']==session,'Exact original tool session')
 return owner

def validate(spec,catalog,load):
 if catalog.get('schema')!=SCHEMA:
  require(Path(spec['final_predecessor_catalog']['path'])!=CURRENT,'Cannot omit new maintenance history');return old.validate(spec,catalog,load)
 ext=shape(catalog);require(spec.get('required_post_smol_maintenance')==ext and Path(spec['final_predecessor_catalog']['path'])==CURRENT,'Explicit complete maintenance selection')
 pins=spec['source_sha256'];require(pins.get(str(CURRENT))==spec['final_predecessor_catalog']['sha256'],'Current history pin')
 require(all(pins.get(k)==v for k,v in catalog['source_sha256'].items()),'No omitted history source')
 def doc(ref):
  reference(ref);require(pins.get(ref['path'])==ref['sha256'],'Explicit source pin for maintenance document');return load(ref)
 prior=doc(ext['prior_catalog']);require(prior['root_finalized'] and prior['all_planned_launches_accounted_for'] and prior['schema']=='root_post_smollm3_sio14_complete_predecessors_v1','Actual completed Smol anchor')
 require(set(catalog)==set(prior)|{'maintenance_extension'},'No unrelated history fields')
 changed={'schema','processes','process_predecessors','source_sha256','maintenance_extension'}
 require(all(catalog[k]==v for k,v in prior.items() if k not in changed),'Every original service/outcome retained')
 require(len(catalog['rows'])==catalog['complete_count']==863 and len(prior['processes'])==165,'Original full predecessor counts')
 require(all(catalog['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()),'No historical source rebound')
 require(spec['required_predecessor_releases']==prior['root_release_refs'],'Original release history')
 candidate=doc(CANDIDATE);require(candidate['cache_entries']==253 and candidate['files']==1039 and candidate['allocated_bytes']==4607754240,'Exact complete cache selection')
 failed=doc(FAILED);fowner=actual_exit(failed,1,'completed_cache_archival_failed_actual_exit_v1',49193)
 require(fowner['pid']==2814497 and fowner['start_ticks']=='390565825' and fowner['argv'][1:]==['-B',old.SOURCE['path'],'--apply','--catalog',PRIOR['path'],'--catalog-sha256',PRIOR['sha256']],'Exact failed gzip invocation')
 fs=doc(failed['started']);ff=doc(failed['failure']);manifest=doc(failed['manifest'])
 require(fs['owner']==ff['owner']==fowner and fs['source']==old.SOURCE and fs['candidate']==CANDIDATE and fs['actual_completed_catalog']==PRIOR,'Original failed attempt bindings')
 require(failed['deletion_performed'] is False and failed['all_original_cache_entries_still_present'] is True and ff['error']==failed['original_error']=="AssertionError('Archive must recover sufficient actual allocated space')",'Actual failed gzip outcome retained')
 probe_exit=doc(ext['probe_exit']);powner=actual_exit(probe_exit,0,'completed_cache_xz_probe_root_exit_v1',43750)
 require(ext['probe_exit']['path']==str(Q/'completed-cache-xz-probe-root-exit-20261108-v1.json') and powner['pid']==2818967 and powner['start_ticks']=='390606329' and powner['argv'][1:]==['-B',PROBE_SOURCE['path']],'Exact XZ proof producer')
 ps=doc(probe_exit['started']);pr=doc(probe_exit['result'])
 require(probe_exit['started']['path']==str(PROBE_ROOT/'start.json') and probe_exit['result']['path']==str(PROBE_ROOT/'result.json'),'Exact XZ proof namespace')
 require(ps['owner']==pr['owner']==powner and ps['lock']==pr['lock'] and ps['source']==pr['source']==PROBE_SOURCE and ps['catalog']==pr['catalog']==PRIOR and ps['failed_maintenance']==pr['failed_maintenance']==FAILED,'XZ producer binding to original failure')
 require(pr['member_verification']==dict(files_rehashed=1039,directories_verified=253) and pr['tar_stream_byte_identical'] is True and pr['deletion_performed'] is False and pr['all253original_cache_entries_untouched'] and pr['old_archive_untouched'],'Compression proof did not retire anything')
 require(pr['manifest']==ps['manifest']==failed['manifest'] and pr['original_archive']==ps['original_archive'] and pr['original_archive_stat']==ps['original_archive_stat']==failed['archive_file']['stat'],'Same original gzip and manifest')
 require(pr['new_archive']['path']==str(PROBE_ROOT/'cache.tar.xz'),'Retained verified XZ archive')
 retired_exit=doc(ext['retirement_exit']);rowner=actual_exit(retired_exit,0,'completed_cache_xz_retirement_root_exit_v1')
 require(ext['retirement_exit']['path']==str(Q/'completed-cache-xz-retirement-root-exit-20261108-v1.json') and rowner['argv'][1:]==['-B',RETIRE_SOURCE['path'],'--probe-exit-sha256',ext['probe_exit']['sha256']],'Exact retirement invocation')
 rs=doc(retired_exit['started']);rr=doc(retired_exit['result'])
 require(retired_exit['started']['path']==str(RETIRE_ROOT/'start.json') and retired_exit['result']['path']==str(RETIRE_ROOT/'result.json'),'Exact retirement namespace')
 for item in [rs,rr]:
  require(item['owner']==rowner and item['source']==RETIRE_SOURCE and item['catalog']==PRIOR and item['failed_maintenance']==FAILED and item['probe_exit']==ext['probe_exit'] and item['probe_result']==probe_exit['result'] and item['probe_start']==probe_exit['started'] and item['candidate']==CANDIDATE,'Retirement bound to all actual predecessors')
 require(rs['lock']==rr['lock'] and rr['manifest']==failed['manifest'] and rr['archive']==pr['new_archive'] and rr['old_archive']==pr['original_archive'],'Same verified archive objects')
 require(rr['deletion_performed'] is rr['all_original_directories_absent'] is rr['old_duplicate_archive_absent'] is rr['canonical_unchanged'] is True,'Actual completed retirement')
 require(rr['retired_directories']==253 and rr['retired_files']==1039 and rr['original_allocated_bytes']==4607754240 and rr['nvme_free_after']>=48*2**30,'Complete original selection and 48 GiB floor')
 require(rr['archive_allocated_bytes']==pr['new_archive_allocated_bytes'] and rr['old_archive_allocated_bytes']==failed['archive_file']['allocated_bytes'],'Actual allocation accounting')
 space=rr['storage'];require(space['metadata_reserve_bytes']==16*2**20 and space['net_reclaimed_lower_bound']==4607754240-rr['archive_allocated_bytes']-rr['retained_metadata_allocated_bytes']-space['metadata_reserve_bytes'] and space['net_reclaimed_lower_bound']>=2*2**30 and space['projected_free_lower_bound']>=48*2**30,'Unreduced recovery budget')
 for key in ['experiment_outputs_deleted','model_inputs_changed','model_or_compiler_started','new_speedup_claim','full_model_qualified','external_self_exit_claimed']:require(rr[key] is False,'Maintenance is not model success: '+key)
 require(rr['next_admission_must_verify_owner_exit'] is rr['maintenance_owners_must_be_appended_to_complete_history'] is True,'Actual external terminal proof required')
 canonical=fs['canonical_sha256'];require(all(item['canonical_sha256']==canonical for item in [failed,ps,pr,rs,rr]),'Canonical hashes preserved across all outcomes')
 for key in ['first_process_scan','final_process_scan']:require(rr[key]['processes']>0 and rr[key]['hits']==[] and rr[key]['permission_errors']==[],'Complete fresh process scans')
 wanted={row['path']:row for entry in candidate['candidates'] for row in entry['files']}
 require(len(manifest['files'])==1039 and manifest['directories']=={e['path']:e['directory_identity'] for e in candidate['candidates']} and {row['path'] for row in manifest['files'].values()}==set(wanted),'Complete selected member inventory')
 for name,row in manifest['files'].items():
  require(name==str(Path(row['path']).relative_to(candidate['cache_root'])) and {k:v for k,v in row.items() if k!='sha256'}==wanted[row['path']] and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Original member identity and hash')
 owners=[fowner,powner,rowner];expected=[dict(owner,role=role) for owner,role in zip(owners,ROLES)]
 require(len({(o['pid'],o['start_ticks']) for o in owners})==3 and catalog['processes']==prior['processes']+expected,'All three actual owners retained, failures included')
 if 'process_predecessors' in prior:require(catalog['process_predecessors']==catalog['processes'],'Complete process alias')
 for source in [old.SOURCE,PROBE_SOURCE,RETIRE_SOURCE]:require(pins.get(source['path'])==source['sha256'],'Actual maintenance source pins')
 audit=doc(ext['root_audit']);require(audit['schema']=='completed_cache_xz_maintenance_root_terminal_v1' and audit['reviewer']=='/root' and audit['passed'] is True and audit['references']=={k:v for k,v in ext.items() if k!='root_audit'},'Independent complete maintenance review')
 require(audit['owners']==owners and audit['all_current_groups_recursively_drained'] is audit['owner_absence_observed'] is audit['archive_bytes_rehashed'] is True and audit['archive']==rr['archive'] and audit['manifest']==rr['manifest'] and audit['canonical_sha256']==canonical,'Independent physical and archive proof')
 require(audit['complete_services']==863 and audit['complete_processes']==168 and audit['full_model_qualified'] is audit['new_speedup_claim'] is False,'Complete history, no model qualification')
 for v in [rr['verification'],audit['archive_verification']]:
  require(v['files_rehashed']==1039 and v['directories_verified']==253 and v['archive']==pr['new_archive'] and v['uncompressed_tar_sha256']==pr['uncompressed_tar_sha256'] and v['uncompressed_tar_bytes']==pr['uncompressed_tar_bytes'] and v['tar_stream_byte_identical'] is True,'Independent full tar/member verification')
 return dict(catalog_reference=copy.deepcopy(PRIOR),catalog=prior,maintenance=[dict(owner=copy.deepcopy(o),actual_exit=copy.deepcopy(ext[key]),root_audit=copy.deepcopy(ext['root_audit']),archive=copy.deepcopy(rr['archive'])) for o,key in zip(owners,['failed_exit','probe_exit','retirement_exit'])],fresh_physical_drain_proven=False,new_speedup_claim=False)
