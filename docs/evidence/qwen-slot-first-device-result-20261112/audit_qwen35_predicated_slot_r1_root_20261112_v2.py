"""Root audit of actual slot r1 failure at the first reference correctness call."""
import datetime, fcntl, json, os, sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research'
sys.path[:0]=[str(Q),str(R)]
import qwen35_predicated_slot_serial_v3 as stage
from qwen35_sampler_runtime_io_v12 import read,ref,once,process_api
from qwen35_sampler_runtime_history_v12 import expected_running
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal,terminal_unit
assert sys.dont_write_bytecode and __debug__
actual=dict(kind='tool_session',session_id=23987,terminal_chunk='57d44c',exit_code=1)
binding=read(ref(Q/'qwen35-predicated-slot-r1-20261112-root-binding-v3.json'))
spec=read(binding['spec'],64*2**20);review=read(binding['review'],64*2**20)
stage.validate(spec,review,binding['spec'])
root=Path(spec['output_root']);admission_ref=ref(root/'admission.json');admission=read(admission_ref)
terminal_ref=ref(root/'outer-terminal.json');terminal=read(terminal_ref)
running_ref=ref(spec['running_handoff_path']);running=read(running_ref)
live_ref=ref(Q/'qwen35-predicated-slot-r1-root-live-20261112-v1.json');live=read(live_ref)
owner_ref=ref(root/'supervisor-owner.json');owner=read(owner_ref)
assert live['actual_tool']==dict(kind='tool_session',session_id=23987)
assert admission['outer_owner']==terminal['outer_owner']==live['outer_owner']
assert admission['spec']==binding['spec'] and admission['review']==binding['review']
assert running==expected_running(read(admission['freshness']['latest_handoff']),running['updated_at'],
 dict(kind='tool_session',session_id=23987),admission_ref,admission,owner_ref,owner)
ack=read(ref(root/'running-ack.json'))
assert ack['running']['reference']==running_ref and ack['running']['value']==running
assert ack['running']['observation']['path']==running_ref['path']
assert ack['admission']==admission_ref and ack['coordinator']==owner
exit_ref=ref(root/'conv/native-process-exit.json');native=read(exit_ref)
assert native['actual_returncode']==1 and native['popen_wait_observed'] and native['self_exit_claimed'] is False
assert native['owner']==read(ref(root/'conv/native-owner.json'))
assert native['parent_observer']==owner['process']
assert native['owner']['ppid']==owner['process']['pid'] and native['owner']['cgroup']==owner['process']['cgroup']
owners=[admission['outer_owner'],owner['process'],native['owner']]
report_ref=ref(root/'conv/report.json');report=read(report_ref)
assert report['complete'] is report['passed'] is False
assert report['runtime_counts']==dict(correctness=1,measured=0,reset=0,warmup=0)
assert len(report['checks'])==1 and len(report['graphs'])==2
assert [(x['mode'],x['variant']) for x in report['graphs']]==[('standalone','reference'),('standalone','candidate')]
assert not (root/'supervisor-result.json').exists() and not (root/'conv/leaf-result.json').exists()
assert not any((root/name).exists() for name in stage.LAYOUTS if name!='conv')
log_ref=ref(root/'conv/native.log')
assert Path(log_ref['path']).read_text().endswith('RuntimeError: Full byte comparison failed: standalone/reference/correctness\n')
ends={terminal_unit(t):t for t in terminal['complete_owned_unit_terminals']}
assert set(ends)==set(admission['units'].values())
p=process_api();observations=[]
fd=os.open(R/'serial-enrollment.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
try:
 assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(66306,8294421)
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for _ in range(2):
  p.active_heavy()
  for current in owners:p.gone(current)
  observation={}
  for unit,saved in ends.items():
   props=stage.properties(unit);assert p.terminal(props)
   verify_current_terminal(saved,props,p)
   observation[unit]=dict(properties=props,recursive_group=p.empty_group(p.GROUP_PREFIX+unit+'.service'))
  observations.append(observation)
 assert observations[0]==observations[1]
 artifacts=[]
 for graph in report['graphs']:
  info=read(ref(root/'conv'/('artifact-standalone-'+graph['variant']+'.json')))
  assert info==graph['artifact'];stage.artifact_record(info,'conv',root/'conv');artifacts.append(info)
 reconciliation=stage.reconcile(terminal,running,actual,actual_owner_gone=True,actual_service_drained=True)
 assert reconciliation['passed'] is False
finally:
 fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
v=dict(schema='qwen35_predicated_slot_root_actual_exit_v1',reviewer='/root',
 observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tool=actual,
 actual_tool_exit_observed=True,actual_outer_exit_observed=True,root_live=live_ref,
 spec=binding['spec'],review=binding['review'],admission=admission_ref,outer_terminal=terminal_ref,
 running_handoff=running_ref,reconciliation=reconciliation,owners=owners,owners_gone=True,
 units=sorted(ends),physically_drained=True,two_current_service_observations=observations,
 original_lock_reacquired_and_released=True,lock_identity=stage.LOCK_IDENTITY,native_exit=exit_ref,
 report=report_ref,native_log=log_ref,artifacts=artifacts,first_check=report['checks'][0],
 preceding_root_audit_attempt=dict(source=ref(Q/'audit_qwen35_predicated_slot_r1_root_20261112_v1.py'),actual_tool=dict(kind='tool_call',terminal_chunk='c0519d',exit_code=1),failure='Root auditor incorrectly compared running envelope to only2fields; real envelope also contains observation. Failed before lock/physical audit; v2 checks all observed binding fields.'),
 runtime_counts=report['runtime_counts'],compiled_graphs=2,completed_layouts=0,
 failure='First conv standalone reference correctness call failed full byte comparison; candidate call and timing not reached.',
 candidate_correctness_executed=False,root_cause_determined=False,primitive_qualified=False,
 model_executed=False,full_model_qualified=False,new_speedup_claim=False,auditor_source=ref(Path(__file__).resolve()))
receipt=once(Q/'qwen35-predicated-slot-r1-root-actual-exit-20261112-v2.json',v)
print(json.dumps(dict(result=receipt,actual_tool=actual,physically_drained=True,compiled_graphs=2,measured_calls=0)))
