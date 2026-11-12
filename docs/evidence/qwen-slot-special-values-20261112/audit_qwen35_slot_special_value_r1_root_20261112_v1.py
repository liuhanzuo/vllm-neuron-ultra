"""Root actual terminal audit: identity compiled, HLO parser failed, zero calls."""
import datetime,fcntl,json,os,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
import qwen35_slot_special_value_serial_v1 as stage
from qwen35_sampler_runtime_io_v12 import read,ref,once,process_api
from qwen35_sampler_runtime_history_v12 import expected_running
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal,terminal_unit
assert sys.dont_write_bytecode and __debug__
actual=dict(kind='tool_session',session_id=31827,terminal_chunk='436e73',exit_code=1)
binding_ref=ref(Q/'qwen35-slot-special-value-r1-root-binding-20261112-v1.json');binding=read(binding_ref)
spec=read(binding['spec'],64*2**20);review=read(binding['review'],64*2**20);stage.validate(spec,review,binding['spec'])
root=Path(spec['output_root']);work=root/stage.diagnostic.NAME
admission_ref=ref(root/'admission.json');admission=read(admission_ref)
terminal_ref=ref(root/'outer-terminal.json');terminal=read(terminal_ref)
running_ref=ref(spec['running_handoff_path']);running=read(running_ref)
live_ref=ref(Q/'qwen35-slot-special-value-r1-root-live-20261112-v1.json');live=read(live_ref)
owner_ref=ref(root/'supervisor-owner.json');owner=read(owner_ref)
assert live['actual_tool']==dict(kind='tool_session',session_id=31827)
assert admission['outer_owner']==terminal['outer_owner']==live['outer_owner']
assert admission['spec']==binding['spec'] and admission['review']==binding['review']
assert running==expected_running(read(admission['freshness']['latest_handoff']),running['updated_at'],live['actual_tool'],admission_ref,admission,owner_ref,owner)
ack=read(ref(root/'running-ack.json'));assert ack['running']['reference']==running_ref and ack['running']['value']==running and ack['running']['observation']['path']==running_ref['path'];assert ack['admission']==admission_ref and ack['coordinator']==owner
exit_ref=ref(work/'native-process-exit.json');native=read(exit_ref)
assert native['actual_returncode']==1 and native['popen_wait_observed'] and native['self_exit_claimed'] is False
assert native['owner']==read(ref(work/'native-owner.json')) and native['parent_observer']==owner['process']
assert native['owner']['ppid']==owner['process']['pid'] and native['owner']['cgroup']==owner['process']['cgroup']
owners=[admission['outer_owner'],owner['process'],native['owner']]
report_ref=ref(work/'report.json');report=read(report_ref)
assert report['complete'] is False and report['error']=="ValueError('One complete NKI backend config in identity HLO')"
assert report['runtime_calls']==0 and report['graphs']==report['calls']==[]
assert report['qualification_matrix_completed'] is report['full_model_qualified'] is report['model_speedup_claim'] is False
assert not (root/'supervisor-result.json').exists() and not (work/'leaf-result.json').exists() and not (work/'artifact-identity.json').exists()
log_ref=ref(work/'native.log');log=Path(log_ref['path']).read_text();assert 'Compiler status PASS' in log and 'qwen35_conv_identity_dma' in log and log.endswith('ValueError: One complete NKI backend config in identity HLO\n')
ends={terminal_unit(t):t for t in terminal['complete_owned_unit_terminals']};assert set(ends)==set(admission['units'].values())
p=process_api();observations=[];artifacts={};fd=os.open(R/'serial-enrollment.lock',os.O_RDWR|os.O_NOFOLLOW|os.O_CLOEXEC)
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
 for name in ['model.pt','graph.neff','model/graph.hlo','model/metadata.json','command.txt']:
  path=work/'artifacts/identity'/name;artifacts[name]=stage.diagnostic.file_ref(path);assert artifacts[name]['bytes']>0
 for profile in stage.diagnostic.PROFILES:
  for key,payload in stage.diagnostic.inputs(profile).items():
   r=report['inputs'][profile][key];assert r['path']==str(work/(profile+'-'+key+'.raw')) and stage.diagnostic.checked(r,stage.diagnostic.BYTES)==payload
 reconciliation=stage.reconcile(terminal,running,actual,actual_owner_gone=True,actual_service_drained=True);assert reconciliation['passed'] is False
finally:fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
v=dict(schema='qwen35_slot_special_value_root_actual_exit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tool=actual,actual_tool_exit_observed=True,actual_outer_exit_observed=True,root_live=live_ref,root_binding=binding_ref,spec=binding['spec'],review=binding['review'],admission=admission_ref,outer_terminal=terminal_ref,running_handoff=running_ref,reconciliation=reconciliation,owners=owners,owners_gone=True,units=sorted(ends),physically_drained=True,two_current_service_observations=observations,original_lock_reacquired_and_released=True,lock_identity=stage.LOCK_IDENTITY,native_exit=exit_ref,report=report_ref,native_log=log_ref,artifacts=artifacts,compiler_log_pass=True,identity_artifact_saved=True,identity_hlo_proof_accepted=False,graphs_loaded=0,runtime_calls=0,measured_calls=0,reference_executed=False,candidate_executed=False,identity_executed=False,preceding_invocation_failure=live['preceding_invocation_failure'],failure='New identity graph compiled and saved; strict HLO backend-config extractor rejected actual encoding before any graph load or runtime call.',primitive_qualified=False,qualification_matrix_completed=False,model_executed=False,full_model_qualified=False,new_speedup_claim=False,auditor_source=ref(Path(__file__).resolve()))
receipt=once(Q/'qwen35-slot-special-value-r1-root-actual-exit-20261112-v1.json',v)
print(json.dumps(dict(result=receipt,actual_tool=actual,physically_drained=True,compiled_artifact_saved=True,runtime_calls=0)))
