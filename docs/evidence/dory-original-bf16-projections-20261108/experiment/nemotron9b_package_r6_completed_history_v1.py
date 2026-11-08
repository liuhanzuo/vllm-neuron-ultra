"""Append-only r6 terminal history contract; no future completion is fabricated."""
import copy,hashlib,json,re,sys
from pathlib import Path
import nemotron9b_post_mini_history_v1 as prior
import nemotron9b_completed_history_drain_v1 as old_drain
R=prior.R;Q=prior.Q;require=prior.require
if str(Q)not in sys.path:sys.path.insert(0,str(Q))
import nemotron9b_r4_terminal_metadata_v2 as terminal
PRIOR=dict(path=str(prior.CURRENT),sha256='f58472178de26c52ecbf5cfdefdde05a866d981772e9d0dc109a9767007aeec4')
SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v14-r6-20261108-spec.json'),sha256='cd779a014effdca2e4fd9a8595b9764ba5e0b51d37df05279292ed266de2c4ff')
REVIEW=dict(path=str(Q/'nemotron9b-package-runtime-v14-r6-20261108-root-review.json'),sha256='8fa6c810f5c69d8831960a8c92e364dd6eea2cf0c2d17b3777a084c4cf6cfc79')
LAUNCH=dict(path=str(Q/'nemotron9b-r6-actual-launch-20261108-v1.json'),sha256='7fb4d80a3d6fea5a5836a7ddb5e823eab9069f82aa70587f69c0f1c938b1abaf')
CURRENT=Q/'nemotron9b-package-r6-20261108-complete-catalog-v1.json'
SCHEMA='root_post_nemotron_r6_complete_history_v1'
CHANGED={'schema','rows','processes','process_predecessors','source_sha256','root_release_refs','complete_count','appended_from','appended_terminal_audit','latest_producer','latest_producer_qualified','latest_producer_exit','latest_producer_terminal_audit','root_finalized','all_planned_launches_accounted_for'}

def validate_extension(original,value,spec,release_ref,audit_ref,exit_ref,outcome,finalizer):
 require(len(original['rows'])==925 and len(original['processes'])==182,'Actual925/182prior history')
 require(set(value)==set(original)|{'nemotron9b_package_r6_append'},'Only exact r6 extension field')
 require(all(value[k]==v for k,v in original.items()if k not in CHANGED),'Every earlier model outcome retained')
 require(value['schema']==SCHEMA and value['rows'][:925]==original['rows']and value['processes'][:182]==original['processes'],'Original services/processes unchanged')
 rows=value['rows'][925:];processes=value['processes'][182:]
 require(len(rows)==53 and len(value['rows'])==value['complete_count']==978,'All53plannedr6services')
 require([x['unit']for x in rows]==[spec['units'][stage]for stage in ['controller',*spec['stages']]],'Original service order')
 require(len({x['unit']for x in value['rows']})==978,'Never replace earlier service')
 require(bool(processes)and len({(x['pid'],x['start_ticks'])for x in value['processes']})==len(value['processes']),'Unique actual standalone owners')
 require(value['processes'][-1]==dict(finalizer,role='nemotron9b_package_r6_finalizer',next_admission_must_prove_gone=True),'Actual finalizer retained; self-exit not claimed')
 if 'process_predecessors'in original:require(value['process_predecessors']==value['processes'],'Complete process alias')
 require(all(value['source_sha256'].get(k)==v for k,v in original['source_sha256'].items()),'No original source rebound or omitted')
 require(value['root_release_refs']==original['root_release_refs']+[release_ref],'Only original current release appended')
 require(value['appended_from']==PRIOR and value['appended_terminal_audit']==audit_ref and value['latest_producer_exit']==exit_ref and value['latest_producer_terminal_audit']==audit_ref,'Exact actual terminal references')
 require(value['latest_producer']=='nemotron9b_package_r6'and value['latest_producer_qualified']is outcome['producer_qualified'],'Actual r6 outcome only')
 require(value['root_finalized']is value['all_planned_launches_accounted_for']is True,'Actual finalized current history')
 require(value['nemotron9b_package_r6_append']==dict(services=rows,processes=processes,actual_exit=exit_ref,terminal_review=audit_ref,release=release_ref,execution_spec=SPEC,source_review=REVIEW,launch=LAUNCH,all_planned_services=53,finalizer_owner=finalizer),'Exact extension contents')
 return value

def validate(value):
 original=prior.load(PRIOR);ext=value['nemotron9b_package_r6_append']
 spec=terminal.load(SPEC);audit=terminal.load(ext['terminal_review']);exited=terminal.load(ext['actual_exit']);release=terminal.load(ext['release']);launch=terminal.load(LAUNCH)
 terminal.validate_actual_exit(exited,dict(owner=launch['coordinator_owner']),SPEC,REVIEW,46838)
 require(exited['schema']=='nemotron9b_package_r6_actual_external_exit_v1'and re.fullmatch('[0-9a-f]{6}',exited['actual_exit']['tool_chunk']),'Actual r6 tool completion')
 require(audit['schema']=='nemotron9b_package_r6_root_terminal_v1'and audit['execution_spec']==SPEC and audit['source_review']==REVIEW and audit['previous_catalog']==PRIOR and audit['actual_external_exit']==ext['actual_exit'],'Exact terminal audit bindings')
 require(audit['release']==ext['release']and audit['finalizer_owner']==ext['finalizer_owner']and audit['physical_drain_passed']is True and audit['all_53_planned_services_retained']is True,'Actual finalizer and complete physical drain')
 require(audit['original_tool_exit']==exited['actual_exit']and audit['new_speedup_claim']is False,'Original exit/outcome kept')
 controls=audit['source_references'];before=terminal.load(controls['external-pre-stop.json']);external=terminal.load(release['audit']);a_ref=controls['admission.json']
 finished=terminal.load(controls['terminal.json'])if 'terminal.json'in controls else None
 failed=terminal.load(controls['failed-terminal.json'])if 'failed-terminal.json'in controls else None
 outcome=terminal.classify(exited['actual_exit']['exit_code'],release,external,finished,failed,controls['external-pre-stop.json'],release['audit'],a_ref)
 require(outcome==audit['outcome']and outcome['producer_qualified']is audit['qualified']is audit['model_qualified'],'No terminal outcome promoted')
 import nemotron9b_package_protocol_v14 as p
 require(set(before['rows'])=={'controller',*spec['stages']},'Complete independent stage snapshots')
 rows=[];summaries=[]
 for stage in ['controller',*spec['stages']]:
  row,summary=terminal.retained_stage(spec,stage,SPEC,REVIEW,p,Path(spec['control_root']),before['rows'][stage]);rows.append(row);summaries.append(summary)
 require(rows==ext['services']and summaries==audit['services'],'Every new service derived from actual frozen evidence')
 for name,ref in controls.items():require(terminal.ref(ref['path'])==ref and value['source_sha256'].get(ref['path'])==ref['sha256'],'Every original control evidence pin')
 for ref in [PRIOR,SPEC,REVIEW,LAUNCH,ext['actual_exit'],ext['terminal_review'],ext['release']]:require(value['source_sha256'].get(ref['path'])==ref['sha256'],'Exact completed history source references')
 require(audit['finalizer_self_exit_claimed']is False and audit['next_admission_must_verify_finalizer_exit']is True,'Finalizer self-exit still needs external proof')
 owners=[(launch['coordinator_owner'],'nemotron9b_package_r6_external_launcher'),(before['auditor'],'nemotron9b_package_r6_original_pre_stop_auditor')]
 for ref in audit['recovery_receipts']:
  receipt=terminal.load(ref)
  require(receipt['reviewer']=='/root'and receipt['execution_spec']==SPEC and receipt['source_review']==REVIEW and receipt['actual_tool_exit_observed']is True,'Actual recovery binding')
  owners.append((receipt['owner'],'nemotron9b_package_r6_supported_recovery_auditor'))
 require(audit['standalone_owners']==[dict(owner=owner,role=role)for owner,role in owners],'No invented standalone owners')
 processes=terminal.extend_processes(original['processes'],owners)+[dict(ext['finalizer_owner'],role='nemotron9b_package_r6_finalizer',next_admission_must_prove_gone=True)]
 require(value['processes']==processes,'Every standalone owner derives from actual evidence')
 require(audit['complete_rows']==978 and audit['complete_processes']==len(value['processes']),'Actual complete counts')
 return validate_extension(original,value,spec,ext['release'],ext['terminal_review'],ext['actual_exit'],outcome,ext['finalizer_owner'])

def text(path,expected=None):
 if Path(path)!=CURRENT:return prior.text(path,expected)
 require(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'Exact completed r6 hash required')
 from nemotron9b_package_metadata_v14 import stable_bytes
 raw=stable_bytes(CURRENT,32*2**20);require(hashlib.sha256(raw).hexdigest()==expected,'Exact completed r6 bytes');validate(json.loads(raw));return raw

def load(ref):
 if Path(ref['path'])!=CURRENT:return prior.load(ref)
 require(set(ref)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact current reference');raw=text(ref['path'],ref['sha256'])
 require('bytes'not in ref or type(ref['bytes'])is int and len(raw)==ref['bytes'],'Exact optional current size');return json.loads(raw)

def physical_drain(value):
 validate(value);older=old_drain.physical_drain(prior.load(PRIOR))
 from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain as added_drain
 newer=added_drain(dict(rows=value['rows'][925:],processes=value['processes'][182:]))
 return dict(older=older,newer=newer)
