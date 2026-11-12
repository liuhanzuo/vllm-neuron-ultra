import argparse,copy,datetime,fcntl,hashlib,json,os,pathlib,resource,signal,sys
R=pathlib.Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(R),str(Q)]
P='minicpm5-formal-import-r5-20261112'
def ref(p):
 p=pathlib.Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pending={}
def read(r):
 if r['path'] in pending:
  raw=pending[r['path']];assert hashlib.sha256(raw).hexdigest()==r['sha256'];return json.loads(raw)
 p=pathlib.Path(r['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'];return json.loads(raw)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--independent-review',required=True);args=parser.parse_args()
 assert sys.dont_write_bytecode and __debug__
 resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(600,605));signal.alarm(600)
 fd=os.open(R/'serial-enrollment.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
 try:
  assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(66306,8294421);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
  from research import prepare_minicpm5_pv_sdk_import_diagnostic_v1 as original
  from research import minicpm5_fixed_input_pv_catalog_auth_v2 as compatible
  from research import minicpm5_sdk_mapping_outer_source_entry_v9 as outer
  from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
  from qwen35_sampler_runtime_io_v12 import process_api
  process_api().active_heavy();tip=latest.latest_handoff()
  request=read(ref(Q/(P+'-request.json')));assert tip['path']==request['current_handoff']['path']
  independent_ref=ref(args.independent_review);independent=read(independent_ref)
  assert independent.get('blockers')==[] and (independent.get('passed') is True or independent.get('source_accepted') is True or independent.get('source_only_accepted') is True)
  prep_ref=ref(Q/(P+'-preparation-receipt.json'));prep=read(prep_ref)
  metadata_exit_ref=ref(Q/(P+'-metadata-root-actual-exit.json'));metadata_exit=read(metadata_exit_ref)
  assert metadata_exit['actual_tool_exit_observed'] and metadata_exit['owners_gone'] and metadata_exit['original_lock_reacquired_and_released']
  assert metadata_exit['preparation']==prep_ref
  assert prep['catalog_rows']==155 and prep['complete_source_files']>=48220
  assert independent['preparation']==prep_ref and independent['source_only_accepted'] is True
  assert len(prep['artifacts'])==16
  values={}
  for name,row in prep['artifacts'].items():
   value=read(row['reference']);raw=launch.encoded(value)
   assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['reference']['sha256'];values[name]=value
  pins=values['complete-source-map'];assert original.r.digest(pins)==prep['complete_source_map_sha256']
  inherited=read(ref(Q/'minicpm5-formal-import-r2-20261112-complete-source-map.json'))
  assert all(pins[p]==sha for p,sha in inherited.items())
  assert values['complete-source-contract']==read(ref(Q/'minicpm5-formal-import-r2-20261112-complete-source-contract.json'))
  def save(name,value):
   target=Q/(P+'-'+name+'.json');raw=launch.encoded(value)
   assert not target.exists() and str(target) not in pending
   pending[str(target)]=raw
   return dict(path=str(target),sha256=hashlib.sha256(raw).hexdigest())
  policies={}
  for role in ('outer','sdk'):
   candidate=values[role+'-native-policy-draft'];prior=read(ref(Q/('minicpm5-formal-import-r1-20261112-'+role+'-native-policy-draft.json')))
   assert candidate['source_map_sha256']==original.r.digest(pins)
   assert {k:v for k,v in candidate.items() if k!='source_map_sha256'}=={k:v for k,v in prior.items() if k!='source_map_sha256'}
   approved=copy.deepcopy(candidate);approved.update(reviewer='/root',accepted=True);policies[role]=save(role+'-native-policy-accepted',approved)
  pv=values['pv-spec'];pv_ref=prep['artifacts']['pv-spec']['reference']
  pv_review=copy.deepcopy(values['pv-review-draft']);pv_review.update(reviewer='/root',passed=True,blockers=[])
  pv_review_ref=save('pv-review-accepted',pv_review)
  compatible.validate(original.auth,pv_ref,pv_review_ref,pv,pv_review,read=read,guard=lambda *a,**kw:None)
  candidate=values['original-preparation'];assert original.serial.exact(pv,candidate['pv_spec_candidate'])
  diag=copy.deepcopy(values['diagnostic-spec']);diag.update(pv_spec=pv_ref,pv_review=pv_review_ref)
  assert diag==dict(schema=original.diagnostic.SPEC_SCHEMA,pv_spec=pv_ref,pv_review=pv_review_ref,worker=pv['worker'],limits=pv['limits'],artifact_limits=pv['artifact_limits'],source_extension=pv['source_extension'])
  diag_ref=save('diagnostic-spec-final',diag)
  diag_review=copy.deepcopy(values['diagnostic-review-draft']);diag_review.update(reviewer='/root',passed=True,blockers=[],spec=diag_ref)
  diag_review_ref=save('diagnostic-review-accepted',diag_review);original.diagnostic.validate_spec(diag_ref,diag_review_ref,diag,diag_review)
  sdk=copy.deepcopy(values['sdk-entry-template']);sdk['native_authority'].update(policies['sdk']);sdk_ref=save('sdk-entry-template-final',sdk)
  spec=copy.deepcopy(values['outer-spec']);spec.update(diagnostic_spec=diag_ref,diagnostic_review=diag_review_ref,sdk_entry_template=sdk_ref)
  spec_ref=save('outer-spec-final',spec)
  review=copy.deepcopy(values['outer-review-draft']);review.update(reviewer='/root',passed=True,blockers=[],spec=spec_ref)
  review_ref=save('outer-review-accepted',review)
  plan=copy.deepcopy(values['outer-entry-plan']);plan['native_authority'].update(policies['outer']);plan['successor_input']=json.dumps(dict(spec=spec_ref,review=review_ref),sort_keys=True,separators=(',',':'))
  plan_ref=save('outer-entry-plan-final',plan)
  launch.separate_authorities(plan,sdk)
  assert plan['bootstrap_sources']==sdk['bootstrap_sources']==spec['source_extension'] and len(spec['source_extension'])==172
  for path,sha in spec['source_extension'].items():assert pins[path]==sha==hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
  assert plan['entry_module']=='research.minicpm5_sdk_mapping_outer_source_entry_v9' and sdk['entry_module']=='research.minicpm5_sdk_mapping_worker_source_entry_v6'
  entry=prep['entry_source'];literal=pathlib.Path(entry['path']).read_bytes();argv=launch.launch(literal,entry,plan,plan_ref)
  assert len(literal)==37178 and argv[7]==literal.decode() and entry['sha256']=='bbf5affece28e3d6356ce27d48ccdb7f4d707fbf6717808a69ca1f61222763d0'
  from types import SimpleNamespace
  def read_source(path,sha):
   raw=pending[path] if path in pending else pathlib.Path(path).read_bytes()
   assert hashlib.sha256(raw).hexdigest()==sha
   return raw,None
  fake=SimpleNamespace(role='outer312',history_owner=None,plan=plan,read_source=read_source)
  assert outer.validate_inputs(fake,plan['successor_input'])[1:]==(spec,review)
  assert latest.latest_handoff()==tip
  result=dict(schema='minicpm5_formal_import_root_concrete_binding_review_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,blockers=[],preparation=prep_ref,preparation_actual_exit=metadata_exit_ref,independent_content_review=independent_ref,catalog_rows=155,complete_source_files=len(pins),complete_source_map=prep['artifacts']['complete-source-map']['reference'],complete_source_contract=prep['artifacts']['complete-source-contract']['reference'],native_policy=policies,pv_spec=pv_ref,pv_review=pv_review_ref,diagnostic_spec=diag_ref,diagnostic_review=diag_review_ref,sdk_entry_template=sdk_ref,outer_spec=spec_ref,outer_review=review_ref,outer_entry_plan=plan_ref,entry_source=entry,launch_environment_correction=dict(remove=['LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH']),binder_change='Root uses accepted exact completed-slot catalog helper shared by prepare/runtime; all prior sources retained. Outer9 keeps accepted tuple/list and owner-wire fixes.',native_policy_change='Only full source_map_sha256 changed before root acceptance; original native files and evidence retained.',runtime_launched=False,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False,original_nine_native_checks_pending=9,original_scientific_rows_still_pending=1284)
  root_ref=save('root-concrete-review',result);argv_ref=save('launch-argv',argv)
  for path,raw in pending.items():
   with pathlib.Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  print(json.dumps(dict(root_review=root_ref,launch_argv=argv_ref,plan=plan_ref)))
 finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
if __name__=='__main__':main()
