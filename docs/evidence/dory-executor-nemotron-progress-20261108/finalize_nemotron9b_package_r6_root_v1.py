"""Finalize actual r6 only after root observes its external tool exit and release.

Never starts/stops/releases a model. Source mode cannot create final history.
"""
import argparse,copy,datetime,hashlib,json,os,re,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import nemotron9b_package_protocol_v14 as p
import nemotron9b_package_r6_completed_history_v1 as history
import nemotron9b_r4_terminal_metadata_v2 as m
from nemotron9b_package_controller_v14 import acquire,release
from research.finalize_smollm3_sio14_root_20261108_v1 import physical_drain as added_drain,life
SPEC=history.SPEC;REVIEW=history.REVIEW;LIVE=history.LAUNCH;PRIOR=history.PRIOR
C=R/'nemotron9b-package-r6-20261108-control'
AUDIT=Q/'nemotron9b-package-r6-20261108-root-terminal-v1.json';CATALOG=history.CURRENT

def record(path,value):
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();m.require(len(raw)<32*2**20,'Bounded finalizer32MiB output')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(directory)
 finally:os.close(directory)
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def setup():
 spec=p.metadata.load(SPEC);review=p.metadata.load(REVIEW);live=p.metadata.load(LIVE)
 m.require(spec['stages']==list(p.STAGES)and len(spec['stages'])==52 and set(spec['units'])=={'controller',*spec['stages']},'All52originalstages/controller')
 m.require(live['specification']==SPEC and live['review']==REVIEW and live['actual_tool_session']==46838 and live['terminal_exit_observed']is False,'Exact actual r6 launch')
 m.require(spec['final_predecessor_catalog']==PRIOR and spec['control_root']==str(C),'Actual925/182prior history and r6 namespace')
 m.require(review['passed']is review['source_only']is review['independent_review']is True and review['execution_spec_sha256']==SPEC['sha256']and review['source_sha256']==spec['source_sha256'],'Exact root source review')
 return spec,review,live

def check_current():
 spec,review,live=setup();rows=[]
 for stage in spec['stages']:
  path=C/(stage+'-retained-exit.json')
  if not path.exists():continue
  saved=m.load(m.ref(path));row,summary=m.retained_stage(spec,stage,SPEC,REVIEW,p,C,saved)
  m.require(saved['successful']is True and summary['worker_reported_complete']is True and summary['worker_reported_passed']is True,'Actual retained successful worker metadata')
  rows.append(dict(stage=stage,retained_exit=m.ref(path),worker_result=summary['worker_result']))
 return dict(schema='nemotron9b_r6_partial_terminal_source_check_v1',count=len(rows),checked=rows,execution_spec=SPEC,source_review=REVIEW,terminal_finalization=False,catalog_written=False,model_qualified=False,new_speedup_claim=False)

def main(args):
 m.require(args.actual_exit and args.actual_exit_sha256,'Real externally observed exit path/hash required')
 exit_ref=dict(path=args.actual_exit,sha256=args.actual_exit_sha256);exited=m.load(exit_ref);spec,review,live=setup()
 exit_code=m.validate_actual_exit(exited,dict(owner=live['coordinator_owner']),SPEC,REVIEW,46838)
 m.require(exited['schema']=='nemotron9b_package_r6_actual_external_exit_v1'and re.fullmatch('[0-9a-f]{6}',exited['actual_exit']['tool_chunk']),'Exact observed r6 tool completion')
 p.low.gone(live['coordinator_owner'])
 m.require(not AUDIT.exists()and not CATALOG.exists(),'New immutable finalizer outputs')
 fd,finalizer,lock=acquire()
 try:
  p.low.active_heavy();prior=history.prior.load(PRIOR);result=copy.deepcopy(prior)
  intent=m.load(m.ref(C/'external-launch-intent.json'));before_ref=m.ref(C/'external-pre-stop.json');before=m.load(before_ref)
  release_ref=m.ref(C/'root-release.json');released=m.load(release_ref);external_ref=released['audit'];external=m.load(external_ref)
  m.require(intent['execution_spec']==SPEC and intent['source_review']==REVIEW and intent['external_owner']==live['coordinator_owner']and before['launch_intent']==m.ref(C/'external-launch-intent.json'),'Actual original independent intent/pre-stop')
  m.require(lock==intent['lock_identity']and p.canonical()==intent['canonical_sha256'],'Actual serial lock and unchanged canonical inputs')
  admission_ref=m.ref(C/'admission.json');admission=m.load(admission_ref)
  m.require(admission['execution_spec']==SPEC and admission['source_review']==REVIEW and admission['final_predecessor_catalog']==PRIOR,'Actual admission and complete prior')
  terminal=m.maybe(C/'terminal.json');failed=m.maybe(C/'failed-terminal.json')
  outcome=m.classify(exit_code,released,external,terminal,failed,before_ref,external_ref,admission_ref)
  m.require(set(before['rows'])=={'controller',*spec['stages']},'All53independent original stage snapshots')
  summaries=[]
  for stage in ['controller',*spec['stages']]:
   row,summary=m.retained_stage(spec,stage,SPEC,REVIEW,p,C,before['rows'][stage]);result['rows'].append(row);summaries.append(summary)
  if released['successful']:
   m.require(all(row['successful']is True and row['identity_verified']is True for row in before['rows'].values()),'All53original successful exits')
   m.require(all(not row['never_launched']for row in summaries)and all(row['worker_reported_complete']is True and row['worker_reported_passed']is True and row['normal_retained_exit_success']is True for row in summaries[1:]),'All52actual complete workers')
  owners=[(live['coordinator_owner'],'nemotron9b_package_r6_external_launcher'),(before['auditor'],'nemotron9b_package_r6_original_pre_stop_auditor')]
  recovery_refs=[]
  for path in args.recovery_receipt:
   ref=m.ref(path);value=m.load(ref)
   m.require(value['reviewer']=='/root'and value['execution_spec']==SPEC and value['source_review']==REVIEW and value['actual_tool_exit_observed']is True,'Exact actual supported recovery receipt')
   m.require(type(value['actual_exit']['exit_code'])is int and type(value['actual_exit']['session_id'])is int and re.fullmatch('[0-9a-f]{6}',value['actual_exit']['tool_chunk']),'Actual recovery code/session/chunk')
   p.low.gone(value['owner']);owners.append((value['owner'],'nemotron9b_package_r6_supported_recovery_auditor'));recovery_refs.append(ref)
  result['processes']=m.extend_processes(prior['processes'],owners)
  # No current unit may be active. Every earlier row, including the exact Mini
  # missing-ACK exception, and each new53row is physically checked here.
  p.low.lock_proof(finalizer,lock);p.sources(spec)
  old_observed=history.old_drain.physical_drain(prior)
  new_observed=added_drain(dict(rows=result['rows'][925:],processes=result['processes'][182:]))
  m.require(p.canonical()==intent['canonical_sha256'],'Canonical unchanged after physical drain')
  result['processes'].append(dict(finalizer,role='nemotron9b_package_r6_finalizer',next_admission_must_prove_gone=True))
  if 'process_predecessors'in result:result['process_predecessors']=copy.deepcopy(result['processes'])
  controls={path.name:m.ref(path)for path in sorted(C.iterdir())if path.is_file()and path.suffix in ('.json','.log')}
  audit=dict(schema='nemotron9b_package_r6_root_terminal_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_spec=SPEC,source_review=REVIEW,previous_catalog=PRIOR,root_live=LIVE,actual_external_exit=exit_ref,original_tool_exit=exited['actual_exit'],release=release_ref,external_audit=external_ref,
   recovery_receipts=recovery_refs,standalone_owners=[dict(owner=owner,role=role)for owner,role in owners],finalizer_owner=finalizer,finalizer_self_exit_claimed=False,next_admission_must_verify_finalizer_exit=True,
   outcome=outcome,original_controller_error=None if failed is None else failed['error'],all_53_planned_services_retained=True,services=summaries,complete_rows=978,complete_processes=len(result['processes']),physical_observations=dict(prior=old_observed,added=new_observed),physical_drain_passed=True,source_references=controls,qualified=outcome['producer_qualified'],model_qualified=outcome['producer_qualified'],new_speedup_claim=False,all_15_model_families_remain_in_scope=True)
  audit_ref=record(AUDIT,audit);pins=dict(spec['source_sha256'])
  for item in [SPEC,REVIEW,LIVE,exit_ref,audit_ref,release_ref,m.ref(Path(__file__)),m.ref(Path(history.__file__)),m.ref(Path(m.__file__)),*controls.values(),*recovery_refs]:
   m.require(item['path']not in pins or pins[item['path']]==item['sha256'],'No conflicting source reference');pins[item['path']]=item['sha256']
  for name,digest in pins.items():m.require(name not in result['source_sha256']or result['source_sha256'][name]==digest,'Original source rebound');result['source_sha256'][name]=digest
  result['root_release_refs'].append(release_ref);result.update(schema=history.SCHEMA,complete_count=978,appended_from=PRIOR,appended_terminal_audit=audit_ref,latest_producer='nemotron9b_package_r6',latest_producer_qualified=outcome['producer_qualified'],latest_producer_exit=exit_ref,latest_producer_terminal_audit=audit_ref,root_finalized=True,all_planned_launches_accounted_for=True)
  result['nemotron9b_package_r6_append']=dict(services=result['rows'][925:],processes=result['processes'][182:],actual_exit=exit_ref,terminal_review=audit_ref,release=release_ref,execution_spec=SPEC,source_review=REVIEW,launch=LIVE,all_planned_services=53,finalizer_owner=finalizer)
  history.validate_extension(prior,result,spec,release_ref,audit_ref,exit_ref,outcome,finalizer)
  history.validate(result)
  catalog_ref=record(CATALOG,result)
  # Return the digest of the bytes actually written; never route this large
  # completed catalog through an unrelated legacy8MiB final-print helper.
  print(json.dumps(dict(audit=audit_ref,catalog=catalog_ref,rows=978,processes=len(result['processes']),outcome=outcome,finalizer_owner=finalizer,actual_self_exit_claimed=False)),flush=True)
 finally:release(fd)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check-current',action='store_true');parser.add_argument('--actual-exit');parser.add_argument('--actual-exit-sha256');parser.add_argument('--recovery-receipt',action='append',default=[]);args=parser.parse_args()
 m.require(sys.dont_write_bytecode and __debug__,'Bytecode disabled, assertions enabled')
 if args.check_current:print(json.dumps(check_current()))
 else:main(args)
