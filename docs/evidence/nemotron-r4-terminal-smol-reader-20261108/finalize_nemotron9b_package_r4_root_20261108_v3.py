"""Independent r4 terminal finalizer. Requires the real external exit receipt.

No model/native work, stop/release operation, automatic successor or default exit
code. --check-current only checks retained metadata and never creates a catalog.
"""
import argparse,copy,datetime,fcntl,hashlib,json,os,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import nemotron9b_package_protocol_v10 as p
import smollm3_token_capture_predecessors_v7 as predecessors
import nemotron9b_r4_terminal_metadata_v2 as m
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain,life
SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v10-r4-20261108-spec-v2.json'),sha256='a782f6a824bbf2877fa95997f58a1e0dbc441fcaf30a98b0087a4c2a8713c8e5')
REVIEW=dict(path=str(Q/'nemotron9b-package-runtime-v10-r4-20261108-review-v2.json'),sha256='f3878615dad67046f73e3bbe035826f14acc980ccfc071d72f629de456ea0d1e')
LIVE=dict(path=str(Q/'nemotron9b-package-runtime-v10-r4-20261108-root-live.json'),sha256='575d9d1d912377085971d3f61a2ad5f85563f4f180d0d57952f7179838813b98')
DIAGNOSTICS=[dict(path=str(Q/'smollm3-guard-source-preparation-20261108-v1.json'),sha256='91aae5222f7f07c8479796a265d0175b1e71f9c64e2f2fcc11ee944b2548fffb'),dict(path=str(Q/'smollm3-title-guard-source-preparation-20261108-v1.json'),sha256='4bf051f8776ce44ae96a75de154b3b10c26397996dc5f2ba3ac9291420644593')]
C=R/'nemotron9b-package-r4-20261108-control'
AUDIT=Q/'nemotron9b-package-r4-20261108-root-terminal-v1.json';CATALOG=Q/'nemotron9b-package-r4-20261108-complete-catalog-v1.json'

def record(path,value):
 raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();m.require(len(raw)<32*2**20,'Original32MiB finalizer output cap')
 with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())

def refs_in(value):
 if type(value)is dict:
  if type(value.get('path'))is str and type(value.get('sha256'))is str:yield dict(path=value['path'],sha256=value['sha256'])
  for child in value.values():yield from refs_in(child)
 elif type(value)is list:
  for child in value:yield from refs_in(child)

def setup():
 spec=m.load(SPEC);review=m.load(REVIEW);live=m.load(LIVE)
 m.require(spec['stages']==list(p.STAGES)and len(spec['stages'])==52 and set(spec['units'])=={'controller',*spec['stages']},'All original52stages plus controller')
 m.require(live['execution_spec']==SPEC and live['source_review']==REVIEW and live['session']==50501,'Exact actual r4 live launch')
 m.require(spec['control_root']==str(C)and live['completed_predecessor_services']==785 and live['recorded_predecessor_processes']==147,'Exact r4 predecessor scale')
 # Source review shape remains verified by its frozen hash and exact live link.
 return spec,review,live

def check_current():
 spec,review,live=setup();checked=[]
 for stage in spec['stages']:
  path=C/(stage+'-retained-exit.json')
  if not path.exists():continue
  saved=m.load(m.ref(path));row,summary=m.retained_stage(spec,stage,SPEC,REVIEW,p,C,saved)
  m.require(saved['successful']is True and summary['worker_reported_complete']is True and summary['worker_reported_passed']is True,'Current retained successful worker metadata')
  checked.append(dict(stage=stage,retained_exit=m.ref(path),worker_result=summary['worker_result']))
 output=dict(schema='nemotron9b_r4_partial_terminal_metadata_source_check_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),finalizer_source=m.ref(Path(__file__)),helper_source=m.ref(Path(m.__file__)),execution_spec=SPEC,source_review=REVIEW,checked=checked,count=len(checked),terminal_finalization=False,model_qualified=False,catalog_written=False)
 path=Q/'nemotron9b-r4-finalizer-partial-metadata-check-20261108-v1.json';result=record(path,output);print(json.dumps(dict(result=result,count=len(checked),terminal_finalization=False)))

def main(args):
 m.require(args.actual_exit and args.actual_exit_sha256,'Real external exit path and hash required; never infer from stale status')
 exit_ref=dict(path=args.actual_exit,sha256=args.actual_exit_sha256);exited=m.load(exit_ref);spec,review,live=setup()
 exit_code=m.validate_actual_exit(exited,live,SPEC,REVIEW,50501);life.gone(live['owner'])
 m.require(not AUDIT.exists()and not CATALOG.exists(),'No overwrite of prior/partial finalization')
 prior=m.load(spec['final_predecessor_catalog']);m.require(len(prior['rows'])==785 and len(prior['processes'])==147,'Preserve exact finalized Smol history')
 result=copy.deepcopy(prior);intent=m.load(m.ref(C/'external-launch-intent.json'));before_ref=m.ref(C/'external-pre-stop.json');before=m.load(before_ref);release_ref=m.ref(C/'root-release.json');release=m.load(release_ref);external_audit_ref=release['audit'];external_audit=m.load(external_audit_ref)
 m.require(intent['execution_spec']==SPEC and intent['source_review']==REVIEW and intent['external_owner']==live['owner']and before['launch_intent']==m.ref(C/'external-launch-intent.json'),'Original independent external launch/pre-stop')
 m.require(p.canonical()==intent['canonical_sha256'],'Three canonical files unchanged')
 admission_ref=m.ref(C/'admission.json');admission=m.load(admission_ref)
 m.require(admission['execution_spec']==SPEC and admission['source_review']==REVIEW and admission['final_predecessor_catalog']==spec['final_predecessor_catalog'],'Original admission spec/review/predecessors')
 terminal=m.maybe(C/'terminal.json');failed=m.maybe(C/'failed-terminal.json')
 outcome=m.classify(exit_code,release,external_audit,terminal,failed,before_ref,external_audit_ref,admission_ref)
 m.require(set(before['rows'])=={'controller',*spec['stages']},'Every planned service has original independent snapshot')
 summaries=[]
 for stage in ['controller',*spec['stages']]:
  row,summary=m.retained_stage(spec,stage,SPEC,REVIEW,p,C,before['rows'][stage]);result['rows'].append(row);summaries.append(summary)
 if release['successful']:
  m.require(all(x['successful']is True and x['identity_verified']is True for x in before['rows'].values()),'Original53independent successful retained exits')
  m.require(all(not x['never_launched']for x in summaries)and all(x['worker_reported_complete']is True and x['worker_reported_passed']is True and x['normal_retained_exit_success']is True for x in summaries[1:]),'All52complete successful worker results')
 owners=[(live['owner'],'nemotron9b_package_r4_external_launcher'),(before['auditor'],'nemotron9b_package_r4_original_pre_stop_auditor')];diagnostic_refs=[]
 for prep_ref in DIAGNOSTICS:
  prep=m.load(prep_ref);diagnostic_refs.append(prep_ref)
  for row in prep['completed_diagnostics']:
   observed=m.load(row['result']);m.require(row['owner']==observed['owner']and row['exact_owner_gone']is True and row['actual_exit']['exit_code']==0,'Actual completed source-diagnostic owner/exit')
   life.gone(row['owner']);owners.append((row['owner'],'completed_source_diagnostic_for_smol_guard'));diagnostic_refs.append(row['result'])
  for item in refs_in(prep):
   m.require(Path(item['path']).suffix not in ['.raw','.pt','.safetensors','.neff','.so'],'No new model/native payload reference read in source diagnostics')
   m.require(m.ref(item['path'])==item,'Exact retained source diagnostic evidence');diagnostic_refs.append(item)
 recovery_refs=[];unknown_recovery_owner=False
 for path in args.recovery_receipt:
  row_ref=m.ref(path);row=m.load(row_ref);m.require(row['reviewer']=='/root'and row['execution_spec']==SPEC and row['source_review']==REVIEW and row['actual_tool_exit_observed']is True,'Actual supported recovery receipt')
  recovery_refs.append(row_ref)
  if row.get('owner')is None:unknown_recovery_owner=True
  else:life.gone(row['owner']);owners.append((row['owner'],'nemotron9b_package_r4_supported_recovery_auditor'))
 for path in args.completed_diagnostic_receipt:
  row_ref=m.ref(path);row=m.load(row_ref);observed=m.load(row['result'])
  m.require(row['reviewer']=='/root'and row['execution_spec']==SPEC and row['source_review']==REVIEW and row['actual_tool_exit_observed']is True and row['actual_exit']['exit_code']==0 and row['owner']==observed['owner']and row['exact_owner_gone']is True,'Actual completed r4 recovery diagnostic')
  life.gone(row['owner']);owners.append((row['owner'],'nemotron9b_package_r4_recovery_source_diagnostic'));diagnostic_refs.extend([row_ref,row['result']])
  for item in refs_in(observed):
   m.require(Path(item['path']).suffix not in ['.raw','.pt','.safetensors','.neff','.so'],'Recovery diagnostic metadata only')
   m.require(m.ref(item['path'])==item,'Exact recovery diagnostic provenance');diagnostic_refs.append(item)
 result['processes']=m.extend_processes(prior['processes'],owners)
 m.require(result['rows'][:785]==prior['rows']and result['processes'][:147]==prior['processes']and len(result['rows'])==838,'Exact append-only full history')
 predecessors.validate_rows(result['rows'],result['processes'])
 fd=os.open(life.LOCK,os.O_RDWR|os.O_NOFOLLOW)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);p.sources(spec);observations=physical_drain(result)
  m.require(p.canonical()==intent['canonical_sha256'],'Canonical unchanged after independent drain')
  controls={x.name:m.ref(x)for x in sorted(C.iterdir())if x.is_file()and x.suffix in ['.json','.log']}
  audit=dict(schema='nemotron9b_package_r4_root_terminal_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_spec=SPEC,source_review=REVIEW,previous_catalog=spec['final_predecessor_catalog'],root_live=LIVE,actual_external_exit=exit_ref,original_tool_exit=exited['actual_exit'],release=release_ref,external_audit=external_audit_ref,recovery_receipts=recovery_refs,recovery_auditor_identity_not_captured=unknown_recovery_owner,recovery_exit_is_experiment_success=False,outcome=outcome,original_controller_error=None if failed is None else failed['error'],all_53_planned_services_retained=True,services=summaries,complete_rows=len(result['rows']),complete_processes=len(result['processes']),physical_observations=observations,physical_drain_passed=True,source_references=controls,source_diagnostics=diagnostic_refs,qualified=outcome['producer_qualified'],model_qualified=outcome['producer_qualified'],new_speedup_claim=False,all_15_model_families_remain_in_scope=True)
  audit_ref=record(AUDIT,audit);pins=dict(spec['source_sha256'])
  for item in [SPEC,REVIEW,LIVE,exit_ref,audit_ref,release_ref,m.ref(Path(__file__)),m.ref(Path(m.__file__)),*controls.values(),*diagnostic_refs,*recovery_refs]:
   m.require(item['path']not in pins or pins[item['path']]==item['sha256'],'No conflicting retained source reference');pins[item['path']]=item['sha256']
  for name,digest in pins.items():m.require(name not in result['source_sha256']or result['source_sha256'][name]==digest,'Historical source pin changed');result['source_sha256'][name]=digest
  result['root_release_refs'].append(release_ref);result.update(schema='root_post_nemotron_package_complete_predecessors_v1',complete_count=len(result['rows']),appended_from=spec['final_predecessor_catalog'],appended_terminal_audit=audit_ref,latest_producer='nemotron9b_package_r4',latest_producer_qualified=outcome['producer_qualified'],latest_producer_exit=exit_ref,latest_producer_terminal_audit=audit_ref,root_finalized=True,all_planned_launches_accounted_for=True)
  result['nemotron9b_package_r4_append']=dict(services=result['rows'][785:],processes=result['processes'][147:],actual_exit=exit_ref,terminal_review=audit_ref,all_planned_services=53)
  catalog_ref=record(CATALOG,result);print(json.dumps(dict(audit=audit_ref,catalog=catalog_ref,rows=len(result['rows']),processes=len(result['processes']),outcome=outcome)))
 finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check-current',action='store_true');parser.add_argument('--actual-exit');parser.add_argument('--actual-exit-sha256');parser.add_argument('--recovery-receipt',action='append',default=[]);parser.add_argument('--completed-diagnostic-receipt',action='append',default=[]);args=parser.parse_args()
 if args.check_current:check_current()
 else:main(args)
