"""Actual complete Mini history plus exact completed space retirement."""
import hashlib,json,re,sys
from pathlib import Path
import minicpm5_frame1_completed_history_v3 as prior
require=prior.require;R=prior.R;Q=prior.Q
PRIOR=dict(path=str(prior.PATH),sha256='acc5534fb0ae86c73633ae159fd638fa3b6119c8782f551119c320ff877eebc1')
CURRENT=Q/'nemotron9b-package-post-mini-maintenance-r6-20261108-catalog-v1.json'
SCHEMA='root_post_mini_completed_cache_retirement_v1'
PLAN=dict(path=str(Q/'nemotron9b-r6-space-retirement-plan-20261108-v1.json'),sha256='2cca86bc626cb1167ed5d112b81f5f4fae28634e4306cc18ab94de3a9d171d93')
FAILED=dict(path=str(Q/'nemotron9b-r6-space-retirement-r1-external-exit-20261108-v1.json'),sha256='0549331466e49cdb56aee4964e4b59632c8443483f1cbb757a398388e0734e5f')
MINI_FINALIZER_EXIT=dict(path=str(Q/'minicpm5-frame1-r3-finalizer-external-exit-20261108-v1.json'),sha256='859d9262828a34a234a7edf0706c7b6e621222c81653c0bc232d0ae6534a7f96')

def validate(value):
    original=prior.load(PRIOR);require(value['schema']==SCHEMA and value['rows']==original['rows']and value['complete_count']==925,'All925original service outcomes retained')
    ext=value['post_mini_maintenance'];require(set(ext)=={'prior_catalog','plan','failed_exit','result','final_audit','mini_finalizer_exit'}and ext['prior_catalog']==PRIOR and ext['plan']==PLAN and ext['failed_exit']==FAILED and ext['mini_finalizer_exit']==MINI_FINALIZER_EXIT,'Exact completed maintenance context')
    docs={k:prior.load(v)for k,v in ext.items()if k!='prior_catalog'}
    failed,done,audit,plan=docs['failed_exit'],docs['result'],docs['final_audit'],docs['plan']
    require(failed['actual_exit']==dict(session_id=22922,exit_code=1,tool_chunk='cee415')and failed['no_files_deleted']and failed['gone'],'Original failed maintenance retained')
    require(done['complete']and done['plan']==PLAN and done['catalog']==PRIOR and done['previous_failed_attempt']==FAILED,'Actual successful exact selection')
    require(done['removed_files']==len(plan['files'])==739 and done['removed_directories']==len(plan['directories'])==2908,'All selected files and directories accounted')
    require(done['protected_identities_unchanged']and done['canonical_unchanged']and not any(done[k]for k in ('model_weights_deleted','serving_artifacts_deleted','scientific_results_deleted','full_model_qualified','new_speedup_claim')),'Only declared completed intermediates/caches retired')
    require(audit['actual_exit']==dict(session_id=83143,exit_code=0,tool_chunk='e52b22')and audit['result']==ext['result']and audit['failed_exit']==FAILED and audit['all_removed_paths_absent']and audit['known_maintenance_owners_gone'],'Actual terminal external review')
    mini=docs['mini_finalizer_exit'];require(mini['catalog']==PRIOR and mini['actual_finalizer_exit']==dict(session_id=64030,exit_code=1,tool_chunk='d4609c')and mini['catalog_independently_validated']and mini['all_925_service_groups_checked'],'Mini final-print failure retained with actual valid catalog')
    expected=[dict(failed['owner'],role='completed_cache_retirement_r6_failed_probe'),dict(done['owner'],role='completed_cache_retirement_r6_success'),dict(audit['finalizer_owner'],role='completed_cache_retirement_r6_finalizer')]
    require(value['processes']==original['processes']+expected and len(value['processes'])==182,'All179original processes and three actual maintenance owners')
    if 'process_predecessors'in value:require(value['process_predecessors']==value['processes'],'Complete process alias')
    changed={'schema','processes','process_predecessors','source_sha256'}
    require(set(value)==set(original)|{'post_mini_maintenance'}and all(value[k]==v for k,v in original.items()if k not in changed),'Model outcome and scientific history unchanged')
    require(all(value['source_sha256'].get(k)==v for k,v in original['source_sha256'].items()),'No source pin omitted/rebound')
    for ref in ext.values():require(value['source_sha256'].get(ref['path'])==ref['sha256'],'Complete maintenance evidence pins')
    return value

def accepts(path):return Path(path)==CURRENT

def text(path,expected=None):
    if not accepts(path):return prior.text(path,expected)
    require(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'Caller pins exact completed maintenance catalog')
    transport=prior.previous.previous.original;transport.low.clear_window();raw=transport.stable_bytes(Path(path),transport.ENVELOPE_LIMIT)
    require(hashlib.sha256(raw).hexdigest()==expected,'Exact complete history bytes');validate(json.loads(raw));return raw

def load(ref):
    if not accepts(ref['path']):return prior.load(ref)
    require(set(ref)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact reference');raw=text(ref['path'],ref['sha256'])
    require('bytes'not in ref or type(ref['bytes'])is int and ref['bytes']==len(raw),'Exact optional size');return json.loads(raw)

def physical_drain(value):
    validate(value);observed=prior.physical_drain(prior.load(PRIOR))
    if str(Q)not in sys.path:sys.path.insert(0,str(Q))
    from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain as drain
    drain(dict(rows=[],processes=value['processes'][179:]));return observed
