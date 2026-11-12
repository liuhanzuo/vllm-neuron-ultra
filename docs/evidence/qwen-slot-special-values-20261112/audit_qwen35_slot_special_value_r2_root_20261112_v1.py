"""Root terminal audit of fixed six-call diagnostic; never model qualification."""
import argparse,datetime,fcntl,json,os,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
import qwen35_slot_special_value_serial_v2 as stage
from qwen35_sampler_runtime_io_v12 import read,ref,once,process_api
from qwen35_sampler_runtime_history_v12 import expected_running
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal,terminal_unit
assert sys.dont_write_bytecode and __debug__
parser=argparse.ArgumentParser();parser.add_argument('--terminal-chunk',required=True);parser.add_argument('--exit-code',type=int,required=True);args=parser.parse_args()
actual=dict(kind='tool_session',session_id=29233,terminal_chunk=args.terminal_chunk,exit_code=args.exit_code)
binding_ref=ref(Q/'qwen35-slot-special-value-r2-root-binding-20261112-v1.json');binding=read(binding_ref)
spec=read(binding['spec'],64*2**20);review=read(binding['review'],64*2**20);stage.validate(spec,review,binding['spec'])
root=Path(spec['output_root']);work=root/stage.diagnostic.NAME
admission_ref=ref(root/'admission.json');admission=read(admission_ref)
terminal_ref=ref(root/'outer-terminal.json');terminal=read(terminal_ref)
running_ref=ref(spec['running_handoff_path']);running=read(running_ref)
live_ref=ref(Q/'qwen35-slot-special-value-r2-root-live-20261112-v1.json');live=read(live_ref)
owner_ref=ref(root/'supervisor-owner.json');owner=read(owner_ref)
assert live['actual_tool']==dict(kind='tool_session',session_id=29233)
assert admission['outer_owner']==terminal['outer_owner']==live['outer_owner']
assert admission['spec']==binding['spec'] and admission['review']==binding['review']
assert running==expected_running(read(admission['freshness']['latest_handoff']),running['updated_at'],live['actual_tool'],admission_ref,admission,owner_ref,owner)
ack=read(ref(root/'running-ack.json'));assert ack['running']['reference']==running_ref and ack['running']['value']==running and ack['running']['observation']['path']==running_ref['path'];assert ack['admission']==admission_ref and ack['coordinator']==owner
exit_ref=ref(work/'native-process-exit.json');native=read(exit_ref)
assert native['popen_wait_observed'] and native['self_exit_claimed'] is False
assert native['owner']==read(ref(work/'native-owner.json')) and native['parent_observer']==owner['process']
assert native['owner']['ppid']==owner['process']['pid'] and native['owner']['cgroup']==owner['process']['cgroup']
owners=[admission['outer_owner'],owner['process'],native['owner']]
report_ref=ref(work/'report.json');report=read(report_ref)
assert report['qualification_matrix_completed'] is report['full_model_qualified'] is report['model_speedup_claim'] is False
ends={terminal_unit(t):t for t in terminal['complete_owned_unit_terminals']};assert set(ends)==set(admission['units'].values())
p=process_api();observations=[];artifacts={};summary=None;fd=os.open(R/'serial-enrollment.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
 assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(66306,8294421);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(2):
  p.active_heavy()
  for current in owners:p.gone(current)
  observation={}
  for unit,saved in ends.items():
   props=stage.properties(unit);assert p.terminal(props);verify_current_terminal(saved,props,p)
   observation[unit]=dict(properties=props,recursive_group=p.empty_group(p.GROUP_PREFIX+unit+'.service'))
  observations.append(observation)
 assert observations[0]==observations[1]
 stage.diagnostic.verify_inputs()
 if actual['exit_code']==0:
  assert native['actual_returncode']==0 and report['complete'] and report['runtime_calls']==6 and report['error'] is None
  result=read(ref(root/'supervisor-result.json'));stage.audit_results(root,result)
  summary=stage.validate_layout(root,stage.diagnostic.NAME,read(ref(work/'leaf-result.json')),native)
  assert summary==result['layouts'][0]
 else:
  assert native['actual_returncode']!=0 and not report['complete'];assert not (root/'supervisor-result.json').exists()
 for name in ['model.pt','graph.neff','model/graph.hlo','model/metadata.json','command.txt']:
  path=work/'artifacts/identity'/name
  if path.exists():artifacts[name]=stage.diagnostic.file_ref(path)
 for profile in stage.diagnostic.PROFILES:
  for key,payload in stage.diagnostic.inputs(profile).items():
   r=report['inputs'][profile][key];assert r['path']==str(work/(profile+'-'+key+'.raw')) and stage.diagnostic.checked(r,stage.diagnostic.BYTES)==payload
 reconciliation=stage.reconcile(terminal,running,actual,actual_owner_gone=True,actual_service_drained=True)
 assert reconciliation['passed']==(actual['exit_code']==0)
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
v=dict(schema='qwen35_slot_special_value_root_actual_exit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tool=actual,actual_tool_exit_observed=True,actual_outer_exit_observed=True,root_live=live_ref,root_binding=binding_ref,spec=binding['spec'],review=binding['review'],admission=admission_ref,outer_terminal=terminal_ref,running_handoff=running_ref,reconciliation=reconciliation,owners=owners,owners_gone=True,units=sorted(ends),physically_drained=True,two_current_service_observations=observations,original_lock_reacquired_and_released=True,lock_identity=stage.LOCK_IDENTITY,native_exit=exit_ref,report=report_ref,native_log=ref(work/'native.log'),artifacts=artifacts,diagnostic_complete=report['complete'],summary=summary,runtime_calls=report['runtime_calls'],measured_calls=0,calls=[dict(arm=x['arm'],profile=x['profile'],different_bytes=x['different_bytes'],bitwise_equal=x['bitwise_equal'],output=x['output'],differences=x['differences']) for x in report['calls']],failure=report['error'],primitive_qualified=False,qualification_matrix_completed=False,model_executed=False,full_model_qualified=False,new_speedup_claim=False,auditor_source=ref(Path(__file__).resolve()))
receipt=once(Q/'qwen35-slot-special-value-r2-root-actual-exit-20261112-v1.json',v)
print(json.dumps(dict(result=receipt,actual_tool=actual,physically_drained=True,runtime_calls=report['runtime_calls'],calls=v['calls'])))
