"""Add the actual r7 retained-failed compile exit without changing older readers.

Metadata only. The original running snapshot, failed terminal snapshots and
empty-group observations all stay authoritative; no success is inferred.
"""
from nemotron9b_r4_terminal_metadata_v2 import *
import nemotron9b_r4_terminal_metadata_v2 as original

SPEC_SHA='005718e2c909d58682bb99ede64bbd8ed0b7afc609dfb164398166218afc9a20'
REVIEW_SHA='69f832d6f7b7f822d52be4c715475c630a26f584056b2de86f23ac51758ded86'


def retained_stage(spec,stage,spec_ref,review_ref,p,control,saved,exists=None,read=None):
 control=Path(control);exists=exists or(lambda path:path.exists());read=read or(lambda path:load(ref(path)))
 if stage!='compile' or control.name!='nemotron9b-package-r7-20261108-control':
  return original.retained_stage(spec,stage,spec_ref,review_ref,p,control,saved,exists,read)
 require(spec_ref['sha256']==SPEC_SHA and review_ref['sha256']==REVIEW_SHA,'Only original r7 compile failure')
 require(not exists(control/'compile-retained-exit.json'),'No fabricated normal retained exit')
 cleanup_path=control/'compile-cleanup-before-stop.json';cleanup=read(cleanup_path)
 owner=read(control/'compile-owner.json');props=cleanup['properties'];ended=saved['properties']
 require(saved['stage']==cleanup['stage']==stage and cleanup['owner']==saved['owner']==owner,'Original running/terminal owner')
 require(cleanup['identity_verified']is True and cleanup['successful']is False and saved['identity_verified']is True and saved['successful']is False,'Failed stage remains failed with exact identity')
 require(owner['argv']==p.argv(spec,stage,spec_ref,review_ref)and owner['cgroup']=='0::'+p.group(spec,stage),'Original command and cgroup')
 p._service_binding(spec,stage,spec_ref,review_ref,owner,props)
 require(props['LoadState']=='loaded'and props['MainPID']==str(owner['pid'])and props['ControlPID']=='0'and props['ActiveState']=='active'and props['SubState']=='running','Original pre-stop observation was live')
 require(read(control/'compile-launch.json')['command']==p.service_command(spec,stage,spec_ref,review_ref),'Original durable launch')
 def failed_exit(value):
  p._service_binding(spec,stage,spec_ref,review_ref,owner,value)
  require(value['LoadState']=='loaded'and value['MainPID']==value['ControlPID']=='0'and value['ActiveState']==value['SubState']=='failed','Retained failed service has no running process')
  require((value['Result'],value['ExecMainCode'],value['ExecMainStatus'])==('timeout','2','15'),'Actual timeout and SIGTERM retained; never success')
 failed_exit(ended)
 failed=read(control/'failed-terminal.json');require(failed['passed']is False and failed['all_started_groups_physically_drained']is True,'Original controller failure and complete drain')
 matches=[x['drain']for x in failed['cleanup']if x['phase']==stage];require(len(matches)==1,'Exactly one original compile cleanup')
 drain=matches[0];require(drain['physically_drained']is True and drain['successful_exit']is False and len(drain['samples'])==2,'Two independent failed-stop observations')
 for sample in drain['samples']:
  failed_exit(sample['properties']);require(sample['properties']==ended,'Original terminal properties agree exactly')
  require(sample['group']==dict(group=p.group(spec,stage),absent=True),'Original group physically absent')
 installed=p.service_command(spec,stage,spec_ref,review_ref);installed=' '.join(installed[installed.index('/usr/bin/env'):]);unit=spec['units'][stage]
 row=dict(unit=unit,recorded_cgroups=['/sys/fs/cgroup'+p.group(spec,stage)],historical_invocation_unknown=False,owner=dict(pid=owner['pid'],start_ticks=owner['start_ticks'],unit=unit,script=spec['worker']),captured_process_owner=dict(owner,unit=unit),never_launched=False,invocation_id=owner['invocation_id'],installed_argv=installed,absent_records=[])
 summary=dict(stage=stage,never_launched=False,owner=owner,original_live_properties=props,original_exit_properties=ended,original_exit_success=False,normal_retained_exit_success=False,worker_reported_complete=False,worker_reported_passed=False,interrupted_while_live=True,retained_failed_service=True,original_cleanup=ref(cleanup_path),failed_terminal=ref(control/'failed-terminal.json'),original_stop_drain=drain)
 return row,summary
