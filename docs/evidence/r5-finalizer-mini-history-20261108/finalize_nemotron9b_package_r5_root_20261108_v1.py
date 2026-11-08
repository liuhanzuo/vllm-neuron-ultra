"""Independent r5 terminal finalizer. Requires the real external exit receipt.

No model/native work, stop/release operation, automatic successor or default exit
code. --check-current only checks retained metadata and never creates a catalog.
"""
import argparse,copy,datetime,fcntl,hashlib,json,os,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import nemotron9b_package_protocol_v13 as p
import smollm3_token_capture_predecessors_v7 as predecessors
import nemotron9b_r4_terminal_metadata_v2 as m
from research.finalize_smollm3_sio9_root_20261107_v1 import physical_drain,life
SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v13-r5-20261108-spec.json'),sha256='efd052d53047131747ef620c6e916262599fc270930fe03ca7722e8629dffc8b')
REVIEW=dict(path=str(Q/'nemotron9b-package-runtime-v13-r5-20261108-review.json'),sha256='2fad11788fb7d61dcdc67405273381202f31c1191ba59a09884774df14bcb909')
LIVE=dict(path=str(Q/'nemotron9b-package-v13-r5-root-live-20261108-v1.json'),sha256='3d8ba9ba5f41d1133b8122d01ca0621a9885d3693bc76c0207fd9e5abceb6fb1')
DIAGNOSTICS=[]  # Earlier diagnostic owners are already retained in the prior catalog.
PRIOR=dict(path=str(Q/'nemotron9b-package-post-smol-maintenance-r4-20261108-catalog-v2.json'),sha256='1bc9c18fb1c809018be394928e8b52bdce8c49a6b854a1b66dcada9ce386edd6')
C=R/'nemotron9b-package-r5-20261108-control'
AUDIT=Q/'nemotron9b-package-r5-20261108-root-terminal-v1.json';CATALOG=Q/'nemotron9b-package-r5-20261108-complete-catalog-v1.json'

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
 m.require(live['execution_spec']==SPEC and live['source_review']==REVIEW and live['session']==92674,'Exact actual r5 live launch')
 m.require(spec['control_root']==str(C)and live['final_predecessor_catalog']==PRIOR and spec['final_predecessor_catalog']==PRIOR and live['native_authorized']is True,'Exact r5 immutable maintenance predecessor')
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
 output=dict(schema='nemotron9b_r5_partial_terminal_metadata_source_check_v1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),finalizer_source=m.ref(Path(__file__)),helper_source=m.ref(Path(m.__file__)),execution_spec=SPEC,source_review=REVIEW,checked=checked,count=len(checked),terminal_finalization=False,model_qualified=False,catalog_written=False)
 path=Q/'nemotron9b-r5-finalizer-partial-metadata-check-20261108-v1.json';result=record(path,output);print(json.dumps(dict(result=result,count=len(checked),terminal_finalization=False)))


def validate_catalog_extension(prior,result,spec,release_ref,audit_ref,exit_ref,outcome):
 """Pure append validation; it does not manufacture actual phase/exit evidence."""
 changed={'schema','rows','processes','process_predecessors','source_sha256','root_release_refs','complete_count','appended_from','appended_terminal_audit','latest_producer','latest_producer_qualified','latest_producer_exit','latest_producer_terminal_audit','root_finalized','all_planned_launches_accounted_for'}
 added={'nemotron9b_package_r5_append'}
 m.require(set(result)==set(prior)|added,'Only exact r5 append field added')
 m.require(all(result[k]==v for k,v in prior.items()if k not in changed),'Every original service outcome and maintenance field retained')
 m.require(len(prior['rows'])==863 and len(prior['processes'])==168 and result['rows'][:863]==prior['rows'] and result['processes'][:168]==prior['processes'],'All actual preceding services and processes retained')
 new_rows=result['rows'][863:];new_processes=result['processes'][168:]
 m.require(len(new_rows)==53 and [row['unit']for row in new_rows]==[spec['units'][stage]for stage in ['controller',*spec['stages']]],'Exactly all53 planned r5 units in original order')
 m.require(len({row['unit']for row in result['rows']})==len(result['rows'])==916==result['complete_count'],'Unique complete current service inventory')
 m.require(bool(new_processes) and len({(row['pid'],row['start_ticks'])for row in result['processes']})==len(result['processes']),'All new actual owners uniquely retained')
 m.require(all(result['source_sha256'].get(k)==v for k,v in prior['source_sha256'].items()),'No original source pin omitted or rebound')
 m.require(result['root_release_refs']==prior['root_release_refs']+[release_ref],'Only actual current release appended')
 m.require(result['appended_from']==PRIOR and result['appended_terminal_audit']==audit_ref and result['latest_producer_exit']==exit_ref and result['latest_producer_terminal_audit']==audit_ref and result['latest_producer_qualified']==outcome['producer_qualified'],'Exact actual r5 terminal references')
 m.require(result['schema']=='root_post_nemotron_package_complete_predecessors_v1' and result['latest_producer']=='nemotron9b_package_r5' and result['root_finalized'] is result['all_planned_launches_accounted_for'] is True,'Finalized r5 schema; actual completion required separately')
 m.require(result['nemotron9b_package_r5_append']==dict(services=new_rows,processes=new_processes,actual_exit=exit_ref,terminal_review=audit_ref,all_planned_services=53),'Exact appended service/process slice and proof references')
 if 'process_predecessors'in prior:m.require(result['process_predecessors']==result['processes'],'Complete process alias retained')
 return True

def main(args):
 m.require(args.actual_exit and args.actual_exit_sha256,'Real external exit path and hash required; never infer from stale status')
 exit_ref=dict(path=args.actual_exit,sha256=args.actual_exit_sha256);exited=m.load(exit_ref);spec,review,live=setup()
 exit_code=m.validate_actual_exit(exited,live,SPEC,REVIEW,92674)
 m.require(exited.get('schema')=='nemotron9b_package_r5_actual_external_exit_v1' and __import__('re').fullmatch('[0-9a-f]{6}',exited['actual_exit']['tool_chunk']),'Exact current root exit receipt schema and actual tool chunk')
 life.gone(live['owner'])
 m.require(not AUDIT.exists()and not CATALOG.exists(),'No overwrite of prior/partial finalization')
 prior=p.metadata.load(spec['final_predecessor_catalog']);m.require(len(prior['rows'])==863 and len(prior['processes'])==168 and prior['root_finalized'] and prior['all_planned_launches_accounted_for'],'Preserve exact finalized Smol history')
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
 owners=[(live['owner'],'nemotron9b_package_r5_external_launcher'),(before['auditor'],'nemotron9b_package_r5_original_pre_stop_auditor')];diagnostic_refs=[]
 for prep_ref in DIAGNOSTICS:
  prep=m.load(prep_ref);diagnostic_refs.append(prep_ref)
  for row in prep['completed_diagnostics']:
   observed=m.load(row['result']);actual=row.get('actual_exit',row.get('actual_tool_exit'));m.require(type(actual)is dict and actual['exit_code']==0 and (row.get('actual_exit')is None or row.get('actual_tool_exit')is None or row['actual_exit']==row['actual_tool_exit']),'Original retained diagnostic exit field, no conflicting records');m.require(row['owner']==observed['owner']and row['exact_owner_gone']is True,'Actual completed source-diagnostic owner/exit')
   life.gone(row['owner']);owners.append((row['owner'],'completed_source_diagnostic_for_smol_guard'));diagnostic_refs.append(row['result'])
  for item in refs_in(prep):
   m.require(Path(item['path']).suffix not in ['.raw','.pt','.safetensors','.neff','.so'],'No new model/native payload reference read in source diagnostics')
   m.require(m.ref(item['path'])==item,'Exact retained source diagnostic evidence');diagnostic_refs.append(item)
 recovery_refs=[];unknown_recovery_owner=False
 for path in args.recovery_receipt:
  row_ref=m.ref(path);row=m.load(row_ref);m.require(row['reviewer']=='/root'and row['execution_spec']==SPEC and row['source_review']==REVIEW and row['actual_tool_exit_observed']is True,'Actual supported recovery receipt')
  recovery_refs.append(row_ref)
  m.require(row.get('owner')is not None,'Every actual r5 recovery auditor identity must be captured')
  life.gone(row['owner']);owners.append((row['owner'],'nemotron9b_package_r5_supported_recovery_auditor'))
 for path in args.completed_diagnostic_receipt:
  row_ref=m.ref(path);row=m.load(row_ref);observed=m.load(row['result'])
  m.require(row['reviewer']=='/root'and row['execution_spec']==SPEC and row['source_review']==REVIEW and row['actual_tool_exit_observed']is True and row['actual_exit']['exit_code']==0 and row['owner']==observed['owner']and row['exact_owner_gone']is True,'Actual completed r5 recovery diagnostic')
  life.gone(row['owner']);owners.append((row['owner'],'nemotron9b_package_r5_recovery_source_diagnostic'));diagnostic_refs.extend([row_ref,row['result']])
  for item in refs_in(observed):
   m.require(Path(item['path']).suffix not in ['.raw','.pt','.safetensors','.neff','.so'],'Recovery diagnostic metadata only')
   m.require(m.ref(item['path'])==item,'Exact recovery diagnostic provenance');diagnostic_refs.append(item)
 result['processes']=m.extend_processes(prior['processes'],owners)
 if 'process_predecessors'in result:result['process_predecessors']=copy.deepcopy(result['processes'])
 m.require(result['rows'][:863]==prior['rows']and result['processes'][:168]==prior['processes']and len(result['rows'])==916,'Exact append-only full history')
 predecessors.validate_rows(result['rows'],result['processes'])
 fd=os.open(life.LOCK,os.O_RDWR|os.O_NOFOLLOW)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);p.sources(spec);observations=physical_drain(result)
  m.require(p.canonical()==intent['canonical_sha256'],'Canonical unchanged after independent drain')
  controls={x.name:m.ref(x)for x in sorted(C.iterdir())if x.is_file()and x.suffix in ['.json','.log']}
  audit=dict(schema='nemotron9b_package_r5_root_terminal_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_spec=SPEC,source_review=REVIEW,previous_catalog=spec['final_predecessor_catalog'],root_live=LIVE,actual_external_exit=exit_ref,original_tool_exit=exited['actual_exit'],release=release_ref,external_audit=external_audit_ref,recovery_receipts=recovery_refs,recovery_auditor_identity_not_captured=unknown_recovery_owner,recovery_exit_is_experiment_success=False,outcome=outcome,original_controller_error=None if failed is None else failed['error'],all_53_planned_services_retained=True,services=summaries,complete_rows=len(result['rows']),complete_processes=len(result['processes']),physical_observations=observations,physical_drain_passed=True,source_references=controls,source_diagnostics=diagnostic_refs,qualified=outcome['producer_qualified'],model_qualified=outcome['producer_qualified'],new_speedup_claim=False,all_15_model_families_remain_in_scope=True)
  audit_ref=record(AUDIT,audit);pins=dict(spec['source_sha256'])
  for item in [SPEC,REVIEW,LIVE,exit_ref,audit_ref,release_ref,m.ref(Path(__file__)),m.ref(Path(m.__file__)),*controls.values(),*diagnostic_refs,*recovery_refs]:
   m.require(item['path']not in pins or pins[item['path']]==item['sha256'],'No conflicting retained source reference');pins[item['path']]=item['sha256']
  for name,digest in pins.items():m.require(name not in result['source_sha256']or result['source_sha256'][name]==digest,'Historical source pin changed');result['source_sha256'][name]=digest
  result['root_release_refs'].append(release_ref);result.update(schema='root_post_nemotron_package_complete_predecessors_v1',complete_count=len(result['rows']),appended_from=spec['final_predecessor_catalog'],appended_terminal_audit=audit_ref,latest_producer='nemotron9b_package_r5',latest_producer_qualified=outcome['producer_qualified'],latest_producer_exit=exit_ref,latest_producer_terminal_audit=audit_ref,root_finalized=True,all_planned_launches_accounted_for=True)
  result['nemotron9b_package_r5_append']=dict(services=result['rows'][863:],processes=result['processes'][168:],actual_exit=exit_ref,terminal_review=audit_ref,all_planned_services=53)
  validate_catalog_extension(prior,result,spec,release_ref,audit_ref,exit_ref,outcome)
  catalog_ref=record(CATALOG,result);print(json.dumps(dict(audit=audit_ref,catalog=catalog_ref,rows=len(result['rows']),processes=len(result['processes']),outcome=outcome)))
 finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--check-current',action='store_true');parser.add_argument('--actual-exit');parser.add_argument('--actual-exit-sha256');parser.add_argument('--recovery-receipt',action='append',default=[]);parser.add_argument('--completed-diagnostic-receipt',action='append',default=[]);args=parser.parse_args()
 if args.check_current:check_current()
 else:main(args)
