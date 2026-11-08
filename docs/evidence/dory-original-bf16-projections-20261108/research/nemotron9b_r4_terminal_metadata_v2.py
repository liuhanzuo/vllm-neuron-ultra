"""Terminal metadata checks shared by the r4 finalizer and source tests.

Inert on import. No service operations, native imports, release writes or outcome
promotion. Read inputs are retained metadata; actual physical drain is separate.
"""
import copy,hashlib,json,os,stat
from pathlib import Path

def require(value,message):
 if not value:raise RuntimeError(message)

def ref(path):
 path=Path(path);before=path.lstat();cap=(64 if path.suffix=='.log'else 32)*2**20
 require(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=cap,'Bounded ordinary terminal metadata')
 digest=hashlib.sha256()
 with path.open('rb')as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
 after=path.lstat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Terminal metadata changed while hashing')
 return dict(path=str(path),sha256=digest.hexdigest())

def load(row):
 path=Path(row['path']);require(ref(path)==row,'Exact terminal reference');require(path.stat().st_size<=32*2**20,'Bounded terminal JSON')
 raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==row['sha256'],'Terminal read matches reference');return json.loads(raw)

def maybe(path):return load(ref(path))if Path(path).exists()else None

def validate_actual_exit(exited,live,spec_ref,review_ref,session):
 require(exited['reviewer']=='/root'and exited['actual_tool_exit_observed']is True,'Actual root-observed external tool exit')
 require(exited['execution_spec']==spec_ref and exited['source_review']==review_ref,'Exit binds exact experiment specification/review')
 require(exited['owner']==live['owner']and exited['exact_owner_gone']is True,'Exit binds exact original owner and gone observation')
 tool=exited['actual_exit'];require(tool['session_id']==session and type(tool['exit_code'])is int and type(tool['tool_chunk'])is str and tool['tool_chunk'],'Original external session/code/chunk')
 return tool['exit_code']

def classify(exit_code,release,audit,terminal,failed,pre_stop_ref,audit_ref,admission_ref):
 require(release['released']is True and release['all_current_groups_recursively_drained']is True and release['canonical_unchanged']is True,'Released exact physically drained diagnostic')
 for key in ['successful','failed','model_qualified','explicit_failed_release']:require(type(release[key])is bool,'Explicit release outcome')
 require(release['successful']is not release['failed']and release['model_qualified']is release['successful']and release['explicit_failed_release']is release['failed'],'Release outcome consistency')
 require(release['audit']==audit_ref and audit['pre_stop']==pre_stop_ref and audit['canonical_unchanged']is True,'Release/audit/pre-stop reference chain')
 require(type(audit['passed'])is bool and audit['passed']is release['successful']and audit['model_qualified']is audit['passed'],'Original audit outcome consistency')
 require(release['automatic_successor_launch']is False and release['new_speedup_claim']is False and audit['automatic_successor_launch']is False and audit['new_speedup_claim']is False,'No automatic successor or inferred speedup')
 scientific=False
 if release['successful']:
  require(failed is None and terminal is not None and terminal['passed']is True and terminal['diagnostic_complete']is True and terminal['all_groups_physically_drained']is True and terminal['successful_units_retained_for_external']is True and terminal['admission']==admission_ref,'Original complete controller terminal')
  require(audit['independent_retained_exits']is True and not audit.get('recovery_only',False)and not audit['failures']and audit['completion']['full_model_qualified']is True and audit['completion']['awaiting_external_raw_byte_audit']is False and audit['completion']['external_raw_byte_audit'],'Successful independent full/raw audit')
  scientific=True
 else:
  require(release['explicit_failed_release']is True,'Failed release remains explicitly failed')
  if failed is not None:require(failed['passed']is False and failed['all_started_groups_physically_drained']is True,'Original controller failure retained')
  require(audit['passed']is False,'Recovery never upgrades original failed audit')
 return dict(original_tool_exit_code=exit_code,external_scientific_audit_passed=scientific,producer_qualified=bool(exit_code==0 and scientific),failed_external_exit=exit_code!=0,failed_release=release['failed'],new_speedup_claim=False)

def retained_stage(spec,stage,spec_ref,review_ref,p,control,saved,exists=None,read=None):
 control=Path(control);exists=exists or (lambda path:path.exists());read=read or (lambda path:load(ref(path)))
 unit=spec['units'][stage];launch=control/('external-launch-intent.json'if stage=='controller'else stage+'-launch.json');ownerpath=control/('external-controller-owner.json'if stage=='controller'else stage+'-owner.json')
 common=dict(unit=unit,recorded_cgroups=['/sys/fs/cgroup'+p.group(spec,stage)],historical_invocation_unknown=False)
 require(saved['stage']==stage,'Snapshot belongs to exact stage')
 if not exists(launch):
  absent=[launch,ownerpath,control/(stage+'-lease.json')]
  require(all(not exists(x)for x in absent)and saved['owner']is None,'Never-launched absence evidence')
  props=saved['properties'];require(props['LoadState']=='not-found'and props['MainPID']=='0'and props['ControlPID']=='0'and not props['InvocationID'],'Never-launched retained absence')
  return dict(common,owner=None,captured_process_owner=None,never_launched=True,invocation_id=None,installed_argv=None,absent_records=[str(x)for x in absent]),dict(stage=stage,never_launched=True)
 intent=read(launch);require(intent['command']==p.service_command(spec,stage,spec_ref,review_ref),'Exact durable original launch command')
 require(exists(ownerpath),'Launched stage needs its actual retained owner; do not invent one')
 raw_owner=read(ownerpath);owner=raw_owner['owner']if stage=='controller'else raw_owner
 if stage=='controller':require(raw_owner['launch_intent']==ref(launch),'Controller acknowledgement binds actual launch')
 require(owner['argv']==p.argv(spec,stage,spec_ref,review_ref)and owner['cgroup']=='0::'+p.group(spec,stage),'Retained owner argv/group')
 require(saved['owner']==owner,'Independent snapshot owner matches retained owner')
 candidates=[saved]
 for suffix in ['-retained-exit.json','-cleanup-before-stop.json']:
  path=control/(stage+suffix)
  if stage!='controller'and exists(path):
   value=read(path);require(value['stage']==stage and value['owner']==owner,'Original stage exit identity');candidates.append(value)
 valid=[]
 for value in candidates:
  props=value['properties']
  if props['LoadState']=='loaded':
   p._service_binding(spec,stage,spec_ref,review_ref,owner,props);require(props['MainPID']=='0'and props['ControlPID']=='0'and props['ActiveState']in ['active','inactive','failed']and props['SubState']in ['exited','dead','failed'],'Retained original process exit before cleanup')
   valid.append(value)
 require(valid,'Actual retained installed command/invocation before service GC')
 original=valid[0];props=original['properties'];installed=p.service_command(spec,stage,spec_ref,review_ref);installed=' '.join(installed[installed.index('/usr/bin/env'):])
 row=dict(common,owner=dict(pid=owner['pid'],start_ticks=owner['start_ticks'],unit=unit,script=spec['controller']if stage=='controller'else spec['worker']),captured_process_owner=dict(owner,unit=unit),never_launched=False,invocation_id=owner['invocation_id'],installed_argv=installed,absent_records=[])
 summary=dict(stage=stage,never_launched=False,owner=owner,original_exit_properties=props,original_exit_success=props['Result']=='success'and props['ExecMainCode']=='1'and props['ExecMainStatus']=='0')
 wp=control/(stage+'-worker-result.json');ep=control/(stage+'-retained-exit.json')
 if stage!='controller'and exists(wp):
  w=read(wp);require(w['phase']==stage and w['producer']==p.producer(owner)and w['lease']==ref(control/(stage+'-lease.json')),'Actual worker result producer/lease')
  summary.update(worker_result=ref(wp),worker_reported_complete=w['complete'],worker_reported_passed=w['passed'])
 if stage!='controller'and exists(ep):
  e=read(ep);summary.update(retained_exit=ref(ep),normal_retained_exit_success=e['successful']);require(type(e['successful'])is bool,'Retained phase result boolean')
  if e['successful']:require(e['identity_verified']is True and summary['original_exit_success'],'Successful phase retains exact original successful exit')
 return row,summary

def extend_processes(prior,owners):
 out=copy.deepcopy(prior);seen={(x['pid'],x['start_ticks'])for x in prior}
 for owner,role in owners:
  require(type(owner['pid'])is int and owner['pid']>0 and type(owner['start_ticks'])is str and owner['start_ticks'].isdigit()and owner['argv'],'Actual standalone process identity')
  key=(owner['pid'],owner['start_ticks'])
  if key in seen:
   existing=next(x for x in out if(x['pid'],x['start_ticks'])==key)
   require(all(existing.get(k)==owner.get(k)for k in ['pid','start_ticks','ppid','argv','cgroup']),'Repeated owner identity agrees exactly');continue
  out.append(dict(owner,role=role));seen.add(key)
 return out


ordinary_retained_stage = retained_stage

def retained_stage(spec,stage,spec_ref,review_ref,p,control,saved,exists=None,read=None):
 """Keep explicit stopped-live failures separate from retained normal exits."""
 control=Path(control);exists=exists or (lambda path:path.exists());read=read or (lambda path:load(ref(path)))
 cleanup_path=control/(stage+'-cleanup-before-stop.json')
 if stage=='controller' or not exists(cleanup_path) or exists(control/(stage+'-retained-exit.json')):
  return ordinary_retained_stage(spec,stage,spec_ref,review_ref,p,control,saved,exists,read)
 cleanup=read(cleanup_path)
 if cleanup['properties']['MainPID']=='0':
  return ordinary_retained_stage(spec,stage,spec_ref,review_ref,p,control,saved,exists,read)
 owner=read(control/(stage+'-owner.json'));props=cleanup['properties']
 require(saved['stage']==cleanup['stage']==stage and cleanup['owner']==saved['owner']==owner,'Stopped-live exact recorded stage/owner')
 require(cleanup['identity_verified']is True and cleanup['successful']is False,'Original stopped-live identity verified, never successful')
 require(owner['argv']==p.argv(spec,stage,spec_ref,review_ref)and owner['cgroup']=='0::'+p.group(spec,stage),'Original stopped-live argv/group')
 p._service_binding(spec,stage,spec_ref,review_ref,owner,props)
 require(props['LoadState']=='loaded'and props['MainPID']==str(owner['pid'])and props['ControlPID']=='0'and props['ActiveState']=='active'and props['SubState']=='running','Original live command snapshot, not an exit')
 intent=read(control/(stage+'-launch.json'));require(intent['command']==p.service_command(spec,stage,spec_ref,review_ref),'Original stopped-live launch command')
 def absent(value):
  return value['LoadState']=='not-found'and value['MainPID']=='0'and value['ControlPID']=='0'and value['ActiveState']=='inactive'and value['SubState']=='dead'and not value['InvocationID']
 require(absent(saved['properties'])and saved['successful']is False,'Independent post-stop absent snapshot, never a successful exit')
 failed=read(control/'failed-terminal.json')
 require(failed['passed']is False and failed['all_started_groups_physically_drained']is True,'Original failed controller and drain')
 matching=[x['drain']for x in failed['cleanup']if x['phase']==stage];require(len(matching)==1,'Exactly one original phase cleanup result')
 drain=matching[0];require(drain['physically_drained']is True and drain['successful_exit']is False and len(drain['samples'])==2,'Two retained failed-stop drain samples')
 for sample in drain['samples']:
  require(absent(sample['properties'])and sample['group']['group']==p.group(spec,stage)and sample['group'].get('absent')is True,'Exact empty absent original service group')
 installed=p.service_command(spec,stage,spec_ref,review_ref);installed=' '.join(installed[installed.index('/usr/bin/env'):])
 unit=spec['units'][stage]
 row=dict(unit=unit,recorded_cgroups=['/sys/fs/cgroup'+p.group(spec,stage)],historical_invocation_unknown=False,owner=dict(pid=owner['pid'],start_ticks=owner['start_ticks'],unit=unit,script=spec['worker']),captured_process_owner=dict(owner,unit=unit),never_launched=False,invocation_id=owner['invocation_id'],installed_argv=installed,absent_records=[])
 summary=dict(stage=stage,never_launched=False,owner=owner,original_live_properties=props,original_exit_properties=None,original_exit_success=False,normal_retained_exit_success=False,worker_reported_complete=False,worker_reported_passed=False,interrupted_while_live=True,original_cleanup=ref(cleanup_path),failed_terminal=ref(control/'failed-terminal.json'),original_stop_drain=drain)
 return row,summary
