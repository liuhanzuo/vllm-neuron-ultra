"""Pure source validation of one completed cache-maintenance extension.

The original completed Smol catalog remains immutable. This layer accounts for
one later archive owner; native admission still proves every process gone and
all original service groups physically empty. No source-level success is a
new model result. No process, archive-payload or framework I/O on import.
"""
from pathlib import Path
import copy,hashlib,json,re
from minicpm5_layer0_runtime_auth_v1 import require
R=Path(__file__).resolve().parent;Q=R/'research'
SCHEMA='nemotron9b_post_smol_maintenance_catalog_v1'
LIMIT=16*2**20
SOURCE=dict(path=str(Q/'archive_completed_neuron_cache_20261108_v2.py'),sha256='770e6f524ad89bbba06861b90463dce31eb0c21c985ebf15ce0e2614537b8fbc')
CANDIDATE=dict(path=str(Q/'storage-nvme-completed-cache-candidates-20261108-v2.json'),sha256='9354c71921613e1bdc74b2c66281e66b48074d46f87487d6d61ea0396a4dd0bd')
ARCHIVE_ROOT=Path('/mnt/nvme/retired-compile-cache-20261108-r2')
OWNER_KEYS={'pid','start_ticks','ppid','argv','cgroup'}
EXT_KEYS={'prior_catalog','root_audit','started','result','actual_exit','source','candidate'}

def accepts(path):
 path=Path(path)
 return path.parent==Q and re.fullmatch(r'nemotron9b-package-post-smol-maintenance-[a-z0-9-]{1,40}-catalog-v1\.json',path.name)is not None

def reference(value):
 require(type(value)is dict and set(value)=={'path','sha256'}and type(value['path'])is str and Path(value['path']).is_absolute()
  and type(value['sha256'])is str and re.fullmatch('[0-9a-f]{64}',value['sha256']),'Exact immutable maintenance reference')
 return value

def shape(value):
 require(type(value)is dict and value.get('schema')==SCHEMA and value.get('reviewer')=='/root'
  and value.get('root_finalized')is True and value.get('all_planned_launches_accounted_for')is True,'Finalized maintenance extension schema')
 require(type(value.get('rows'))is list and type(value.get('processes'))is list and type(value.get('source_sha256'))is dict,'Complete maintenance history')
 require(type(value.get('complete_count'))is int and value['complete_count']==len(value['rows'])and bool(value['rows']),'Unchanged service count')
 require(len({row['unit']for row in value['rows']})==len(value['rows']),'Unique historical services')
 ext=value.get('maintenance_extension');require(type(ext)is dict and set(ext)==EXT_KEYS,'Exact maintenance extension fields')
 for ref in ext.values():reference(ref)
 require(ext['source']==SOURCE and ext['candidate']==CANDIDATE,'Exact reviewed archive source and candidate')
 for ref in ext.values():require(value['source_sha256'].get(ref['path'])==ref['sha256'],'Maintenance references in source closure')
 return ext

def source_bytes(path,expected,stable_reader):
 require(accepts(path),'Named maintenance catalog namespace')
 reference(dict(path=str(path),sha256=expected))
 raw=stable_reader(Path(path),LIMIT);require(hashlib.sha256(raw).hexdigest()==expected,'Maintenance catalog fingerprint')
 shape(json.loads(raw.decode('utf8')));return raw

def validate(spec,catalog,load):
 """Return the authenticated original Smol anchor, never relabel its outcome."""
 claim=spec.get('required_post_smol_maintenance')
 if claim is None:
  require(catalog.get('schema')!=SCHEMA and not accepts(spec['final_predecessor_catalog']['path']),'Maintenance extension cannot be omitted')
  return dict(catalog_reference=copy.deepcopy(spec['final_predecessor_catalog']),catalog=catalog,maintenance=[],fresh_physical_drain_proven=False)
 reference(spec['final_predecessor_catalog']);require(accepts(spec['final_predecessor_catalog']['path']),'Explicit maintenance catalog selected')
 ext=shape(catalog);require(claim==ext,'Exact independently selected maintenance extension')
 pins=spec['source_sha256'];require(pins.get(spec['final_predecessor_catalog']['path'])==spec['final_predecessor_catalog']['sha256'],'Current maintenance catalog pin')
 require(all(pins.get(k)==v for k,v in catalog['source_sha256'].items()),'Complete maintenance source closure')
 def document(ref):
  reference(ref);require(pins.get(ref['path'])==ref['sha256'],'Explicit maintenance document source pin');return load(ref)
 prior=document(ext['prior_catalog']);require(prior['schema']=='root_post_smollm3_sio14_complete_predecessors_v1'
  and prior['root_finalized']is True and prior['all_planned_launches_accounted_for']is True,'Actual immutable completed sio14 anchor')
 require(ext['prior_catalog']['path']==str(Q/'root-post-smol-sio14-complete-predecessors-20261108-v1.json'),'Exact sio14 anchor path')
 require(set(catalog)==set(prior)|{'maintenance_extension'},'No unreviewed history fields')
 changed={'schema','processes','process_predecessors','source_sha256','maintenance_extension'}
 require(all(catalog[k]==v for k,v in prior.items()if k not in changed),'Every original service, outcome, release and metadata field retained')
 require(all(catalog['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()),'No historical source pin removed or rewritten')
 require(spec['required_predecessor_releases']==prior['root_release_refs']==catalog['root_release_refs'],'Original release history retained')
 start,result,exited,audit=(document(ext[key])for key in ['started','result','actual_exit','root_audit'])
 candidate=document(ext['candidate']);require(candidate['cache_entries']==253 and candidate['files']==1039 and candidate['allocated_bytes']==4607754240,'Original complete-cache selection')
 require(ext['started']['path']==str(ARCHIVE_ROOT/'start.json')and ext['result']['path']==str(Q/'storage-nvme-completed-cache-retirement-20261108-v2.json'),'Exact archive output namespace')
 owner=start['owner'];require(type(owner)is dict and set(owner)==OWNER_KEYS,'Original archive owner fields')
 require(type(owner['pid'])is int and owner['pid']>0 and type(owner['ppid'])is int and owner['ppid']>0
  and type(owner['start_ticks'])is str and owner['start_ticks'].isdigit()and type(owner['cgroup'])is str and owner['cgroup'].startswith('0::/user.slice/user-1000.slice/'),'Exact archive process identity')
 argv=owner['argv'];require(type(argv)is list and argv[0]in ('python3','/usr/bin/python3')and argv[1:]==['-B',SOURCE['path'],'--apply','--catalog',ext['prior_catalog']['path'],'--catalog-sha256',ext['prior_catalog']['sha256']],'Exact archive invocation bound to actual completed Smol')
 require(start['candidate']==result['candidate']==CANDIDATE and start['source']==result['source']==SOURCE
  and start['actual_completed_catalog']==result['actual_completed_catalog']==ext['prior_catalog'],'Archive source/candidate/completed-Smol binding')
 require(result['owner']==owner and result['lock']==start['lock'] and result['canonical_sha256']==start['canonical_sha256'],'Same original maintenance producer and canonical records')
 require(result['deletion_performed']is result['all_original_directories_absent']is result['canonical_unchanged']is True
  and result['external_self_exit_claimed']is False and result['next_admission_must_verify_owner_exit']is True
  and result['maintenance_owner_must_be_appended_to_complete_history']is True,'Completed archive keeps self-exit separate')
 require(result['experiment_outputs_deleted']is result['model_inputs_changed']is result['model_or_compiler_started']is False,'Maintenance did not become model work')
 require(result['files_verified']==1039 and result['directories_verified']==253 and result['original_allocated_bytes']==4607754240,'Complete verified archive counts')
 require(type(result['archive_allocated_bytes'])is int and result['archive_allocated_bytes']>=0
  and result['net_allocated_reclaimed_bytes']==4607754240-result['archive_allocated_bytes']
  and result['net_allocated_reclaimed_bytes']>=result['required_net_recovery_bytes']>=2*2**30
  and result['nvme_free_after']>=48*2**30,'Measured unreduced storage recovery')
 require(result['archive']['path']==str(ARCHIVE_ROOT/'cache.tar.gz')and result['manifest']['path']==str(ARCHIVE_ROOT/'files.json'),'Exact retained archive and member manifest')
 for key in ['archive','manifest']:reference(result[key])
 manifest=document(result['manifest']);require(len(manifest['files'])==1039 and len(manifest['directories'])==253,'Complete archived member inventory')
 wanted={row['path']:row for entry in candidate['candidates']for row in entry['files']}
 require({row['path']for row in manifest['files'].values()}==set(wanted),'Only selected cache file members')
 require(manifest['directories']=={entry['path']:entry['directory_identity']for entry in candidate['candidates']},'Original complete archive directory identities')
 for name,row in manifest['files'].items():
  require(name==str(Path(row['path']).relative_to(Path(candidate['cache_root']))),'Exact relative archive member name')
  require({k:v for k,v in row.items()if k!='sha256'}==wanted[row['path']]and type(row['sha256'])is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Original member identities and byte hashes')
 for scan in ['first_process_scan','final_process_scan']:
  require(result[scan]['hits']==[]and result[scan]['permission_errors']==[]and result[scan]['processes']>0,'Fresh complete privileged process checks')
 require(exited['schema']=='completed_cache_archival_root_exit_v1'and exited['reviewer']=='/root'and exited['owner']==owner
  and exited['actual_exit_observed_by_root']is exited['gone']is exited['terminal']is True and type(exited['exit_code'])is int and exited['exit_code']==0
  and exited['result']==ext['result'],'Independent original archive process exit')
 handle=exited['execution_handle'];require(handle['kind']=='tool_session'and type(handle['session_id'])is int and handle['session_id']>0
  and handle['actual_completion_observed']is True and type(handle['exit_code'])is int and handle['exit_code']==0 and type(handle['tool_chunk'])is str and re.fullmatch('[0-9a-f]{6}',handle['tool_chunk']),'Actual maintenance tool completion authority')
 require(audit['schema']=='completed_cache_archival_root_terminal_v1'and audit['reviewer']=='/root'and audit['passed']is True
  and audit['references']=={k:v for k,v in ext.items()if k!='root_audit'}and audit['owner']==owner,'Exact independent maintenance terminal review')
 require(audit['all_current_groups_recursively_drained']is audit['owner_absence_observed']is audit['archive_bytes_rehashed']is True
  and audit['archive']==result['archive']and audit['manifest']==result['manifest']and audit['canonical_sha256']==result['canonical_sha256']
  and audit['full_model_qualified']is audit['new_speedup_claim']is False,'Maintenance proof is not model qualification')
 expected=dict(owner,role='completed_cache_archival_maintenance')
 require(catalog['processes']==prior['processes']+[expected],'Exactly one retained new maintenance owner')
 if 'process_predecessors'in prior:require(catalog['process_predecessors']==catalog['processes'],'Complete process alias retained')
 require(audit['complete_services']==len(catalog['rows'])and audit['complete_processes']==len(catalog['processes']),'Full history count binding')
 return dict(catalog_reference=copy.deepcopy(ext['prior_catalog']),catalog=prior,
  maintenance=[dict(owner=copy.deepcopy(owner),actual_exit=copy.deepcopy(ext['actual_exit']),root_audit=copy.deepcopy(ext['root_audit']),archive=copy.deepcopy(result['archive']))],fresh_physical_drain_proven=False,new_speedup_claim=False)
