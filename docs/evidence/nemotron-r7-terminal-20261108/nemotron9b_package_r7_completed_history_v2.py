"""Append-only r7 terminal history contract; no future completion is fabricated."""
import copy,hashlib,json,re,sys
from pathlib import Path
import minicpm5_upstream_completed_history_v1 as prior
import nemotron9b_post_upstream_history_v1 as maintenance
import nemotron9b_completed_history_drain_v2 as old_drain
R=prior.R;Q=prior.Q;require=prior.require
if str(Q)not in sys.path:sys.path.insert(0,str(Q))
import nemotron9b_r7_terminal_metadata_v3 as terminal
PRIOR=dict(maintenance.CATALOG)
SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v15-r7-20261108-spec.json'),sha256='005718e2c909d58682bb99ede64bbd8ed0b7afc609dfb164398166218afc9a20')
REVIEW=dict(path=str(Q/'nemotron9b-package-runtime-v15-r7-20261108-source-review.json'),sha256='69f832d6f7b7f822d52be4c715475c630a26f584056b2de86f23ac51758ded86')
LAUNCH=dict(path=str(Q/'nemotron9b-r7-actual-launch-20261108-v1.json'),sha256='8434efd990f972a98128391925f3cf9b7dcc456b22cb41fd63dda43f82bc61f8')
INTENT=dict(path=str(R/'nemotron9b-package-r7-20261108-control/external-launch-intent.json'),sha256='08f05d1ac650a3c7b92acb0e433f167b2acb6baeb2df6e74138f5859bd7d1957')
MAINTENANCE=maintenance.MAINTENANCE
CURRENT=Q/'nemotron9b-package-r7-20261108-complete-catalog-v1.json'
SCHEMA='root_post_nemotron_r7_complete_history_v1'
CHANGED={'schema','rows','processes','process_predecessors','source_sha256','root_release_refs','complete_count','appended_from','appended_terminal_audit','latest_producer','latest_producer_qualified','latest_producer_exit','latest_producer_terminal_audit','root_finalized','all_planned_launches_accounted_for'}

def checked_launch():
 value=terminal.load(LAUNCH);intent=terminal.load(INTENT)
 require(value['schema']=='nemotron9b_r7_actual_launch_requested_v1'and value['reviewer']=='/root','Actual r7 launch request')
 require(value['spec']==SPEC and value['source_review']==REVIEW and value['actual_launch_session']==12721 and value['actual_final_exit_observed']is False,'Original actual launch session and unchanged requested spec')
 require(intent['schema']=='nemotron9b_package_external_intent_v3'and intent['execution_spec']==SPEC and intent['source_review']==REVIEW and intent['automatic_successor_launch']is False,'Actual original external launch intent')
 return dict(value,coordinator_owner=intent['external_owner'])


def validate_extension(original,value,spec,release_ref,audit_ref,exit_ref,outcome,finalizer):
 require(len(original['rows'])==981 and len(original['processes'])==196,'Actual925/182prior history')
 require(set(value)==set(original)|{'nemotron9b_package_r7_append'},'Only exact r7 extension field')
 require(all(value[k]==v for k,v in original.items()if k not in CHANGED),'Every earlier model outcome retained')
 require(value['schema']==SCHEMA and value['rows'][:981]==original['rows']and value['processes'][:196]==original['processes'],'Original services/processes unchanged')
 rows=value['rows'][981:];processes=value['processes'][196:]
 require(len(rows)==53 and len(value['rows'])==value['complete_count']==1034,'All53plannedr7services')
 require([x['unit']for x in rows]==[spec['units'][stage]for stage in ['controller',*spec['stages']]],'Original service order')
 require(len({x['unit']for x in value['rows']})==1034,'Never replace earlier service')
 require(dict(terminal.load(MAINTENANCE)['owner'],role='nemotron9b_r7_prelaunch_maintenance')in processes,'Actual completed maintenance owner retained')
 require(bool(processes)and len({(x['pid'],x['start_ticks'])for x in value['processes']})==len(value['processes']),'Unique actual standalone owners')
 require(value['processes'][-1]==dict(finalizer,role='nemotron9b_package_r7_finalizer',next_admission_must_prove_gone=True),'Actual finalizer retained; self-exit not claimed')
 if 'process_predecessors'in original:require(value['process_predecessors']==value['processes'],'Complete process alias')
 require(all(value['source_sha256'].get(k)==v for k,v in original['source_sha256'].items()),'No original source rebound or omitted')
 require(value['root_release_refs']==original['root_release_refs']+[release_ref],'Only original current release appended')
 require(value['appended_from']==PRIOR and value['appended_terminal_audit']==audit_ref and value['latest_producer_exit']==exit_ref and value['latest_producer_terminal_audit']==audit_ref,'Exact actual terminal references')
 require(value['latest_producer']=='nemotron9b_package_r7'and value['latest_producer_qualified']is outcome['producer_qualified'],'Actual r7 outcome only')
 require(value['root_finalized']is value['all_planned_launches_accounted_for']is True,'Actual finalized current history')
 require(value['nemotron9b_package_r7_append']==dict(services=rows,processes=processes,actual_exit=exit_ref,terminal_review=audit_ref,release=release_ref,execution_spec=SPEC,source_review=REVIEW,launch=LAUNCH,all_planned_services=53,finalizer_owner=finalizer,prelaunch_maintenance=MAINTENANCE,launch_intent=INTENT),'Exact extension contents')
 return value

def validate(value):
 original=prior.load(PRIOR);ext=value['nemotron9b_package_r7_append']
 spec=terminal.load(SPEC);audit=terminal.load(ext['terminal_review']);exited=terminal.load(ext['actual_exit']);release=terminal.load(ext['release']);launch=checked_launch()
 terminal.validate_actual_exit(exited,dict(owner=launch['coordinator_owner']),SPEC,REVIEW,12721)
 require(exited['schema']=='nemotron9b_package_r7_actual_external_exit_v1'and re.fullmatch('[0-9a-f]{6}',exited['actual_exit']['tool_chunk']),'Actual r7 tool completion')
 require(audit['schema']=='nemotron9b_package_r7_root_terminal_v1'and audit['execution_spec']==SPEC and audit['source_review']==REVIEW and audit['previous_catalog']==PRIOR and audit['actual_external_exit']==ext['actual_exit'],'Exact terminal audit bindings')
 require(audit['release']==ext['release']and audit['finalizer_owner']==ext['finalizer_owner']and audit['physical_drain_passed']is True and audit['all_53_planned_services_retained']is True,'Actual finalizer and complete physical drain')
 require(audit['original_tool_exit']==exited['actual_exit']and audit['new_speedup_claim']is False,'Original exit/outcome kept')
 controls=audit['source_references'];before=terminal.load(controls['external-pre-stop.json']);external=terminal.load(release['audit']);a_ref=controls['admission.json']
 finished=terminal.load(controls['terminal.json'])if 'terminal.json'in controls else None
 failed=terminal.load(controls['failed-terminal.json'])if 'failed-terminal.json'in controls else None
 outcome=terminal.classify(exited['actual_exit']['exit_code'],release,external,finished,failed,controls['external-pre-stop.json'],release['audit'],a_ref)
 require(outcome==audit['outcome']and outcome['producer_qualified']is audit['qualified']is audit['model_qualified'],'No terminal outcome promoted')
 import nemotron9b_package_protocol_v15 as p
 require(set(before['rows'])=={'controller',*spec['stages']},'Complete independent stage snapshots')
 rows=[];summaries=[]
 for stage in ['controller',*spec['stages']]:
  row,summary=terminal.retained_stage(spec,stage,SPEC,REVIEW,p,Path(spec['control_root']),before['rows'][stage]);rows.append(row);summaries.append(summary)
 require(rows==ext['services']and summaries==audit['services'],'Every new service derived from actual frozen evidence')
 for name,ref in controls.items():require(terminal.ref(ref['path'])==ref and value['source_sha256'].get(ref['path'])==ref['sha256'],'Every original control evidence pin')
 for ref in [PRIOR,SPEC,REVIEW,LAUNCH,INTENT,MAINTENANCE,ext['actual_exit'],ext['terminal_review'],ext['release']]:require(value['source_sha256'].get(ref['path'])==ref['sha256'],'Exact completed history source references')
 require(audit['finalizer_self_exit_claimed']is False and audit['next_admission_must_verify_finalizer_exit']is True,'Finalizer self-exit still needs external proof')
 maintenance_receipt=maintenance.validate_maintenance(spec,terminal.load)
 for name,sha in maintenance.MAINTENANCE_PINS.items():require(value['source_sha256'].get(name)==sha,'Every actual maintenance evidence pin retained')
 owners=[(maintenance_receipt['owner'],'nemotron9b_r7_prelaunch_maintenance'),(launch['coordinator_owner'],'nemotron9b_package_r7_external_launcher'),(before['auditor'],'nemotron9b_package_r7_original_pre_stop_auditor')]
 for ref in audit['recovery_receipts']:
  receipt=terminal.load(ref)
  require(receipt['reviewer']=='/root'and receipt['execution_spec']==SPEC and receipt['source_review']==REVIEW and receipt['actual_tool_exit_observed']is True,'Actual recovery binding')
  owners.append((receipt['owner'],'nemotron9b_package_r7_supported_recovery_auditor'))
 require(audit['standalone_owners']==[dict(owner=owner,role=role)for owner,role in owners],'No invented standalone owners')
 processes=terminal.extend_processes(original['processes'],owners)+[dict(ext['finalizer_owner'],role='nemotron9b_package_r7_finalizer',next_admission_must_prove_gone=True)]
 require(value['processes']==processes,'Every standalone owner derives from actual evidence')
 require(audit['complete_rows']==1034 and audit['complete_processes']==len(value['processes']),'Actual complete counts')
 return validate_extension(original,value,spec,ext['release'],ext['terminal_review'],ext['actual_exit'],outcome,ext['finalizer_owner'])

def text(path,expected=None):
 if Path(path)!=CURRENT:return prior.text(path,expected)
 require(type(expected)is str and re.fullmatch('[0-9a-f]{64}',expected),'Exact completed r7 hash required')
 from nemotron9b_package_metadata_v15 import stable_bytes
 raw=stable_bytes(CURRENT,32*2**20);require(hashlib.sha256(raw).hexdigest()==expected,'Exact completed r7 bytes');validate(json.loads(raw));return raw

def load(ref):
 if Path(ref['path'])!=CURRENT:return prior.load(ref)
 require(set(ref)in ({'path','sha256'},{'path','sha256','bytes'}),'Exact current reference');raw=text(ref['path'],ref['sha256'])
 require('bytes'not in ref or type(ref['bytes'])is int and len(raw)==ref['bytes'],'Exact optional current size');return json.loads(raw)

def physical_drain(value):
 validate(value);older=old_drain.physical_drain(prior.load(PRIOR))
 from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain as added_drain
 newer=added_drain(dict(rows=value['rows'][981:],processes=value['processes'][196:]))
 return dict(older=older,newer=newer)
