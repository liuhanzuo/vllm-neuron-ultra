"""Root concrete review plus unchanged full History precheck for the complete integer-reference matrix r1."""
import copy, datetime, fcntl, json, os, resource, signal, sys, time
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research'
sys.path[:0]=[str(Q),str(R)]
assert sys.dont_write_bytecode and __debug__
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
resource.setrlimit(resource.RLIMIT_CPU,(600,605));signal.alarm(600)
import qwen35_integer_reference_serial_v1 as stage
from qwen35_package_preflight_history_v2 import History
from qwen35_sampler_runtime_io_v12 import read,ref,once,loaded_local_sources,process_api
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
from prepare_qwen35_sampler_v9_candidate_v3 import encode
source_review=ref(Q/'qwen35-integer-reference-root-source-review-20261112-v1.json')
assert read(source_review)['source_accepted']
candidate_ref=ref(Q/'qwen35-integer-reference-slot-r1-20261112-binding-candidate.json')
candidate=read(candidate_ref)
spec_ref=candidate['spec_ref'];spec=read(spec_ref,64*2**20)
catalog=read(spec['current_serial_catalog']);tip=latest.latest_handoff()
assert len(catalog['runs'])==157 and catalog['runs'][-1]['tool']==dict(kind='tool_session',session_id=22831,terminal_chunk='a07595',exit_code=0)
assert tip['path']==catalog['current_handoff']['path']
assert not read(catalog['current_handoff'])['running']
assert candidate['source_files']==len(spec['source_sha256']) and candidate['catalog_rows']==157
draft=read(candidate['review_draft_ref'],64*2**20)
assert draft['passed'] is draft['launch_authorized'] is False
assert draft['spec']==spec_ref and draft['source_sha256']==spec['source_sha256']
review=dict(draft,passed=True,blockers=[],launch_authorized=True)
stage.validate(spec,review,spec_ref);stage.verify_delivery(spec)
fd=os.open(R/'serial-enrollment.lock',os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW)
try:
 assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(66306,8294421)
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);process_api().active_heavy()
 stage.resources(spec,reserve=True)
 assert latest.latest_handoff()==tip
 with Path(candidate['paths']['review']).open('xb') as stream:
  stream.write(encode(review));stream.flush();os.fsync(stream.fileno())
 review_ref=ref(candidate['paths']['review'],64*2**20)
 assert stage.load_spec(spec_ref,review_ref)==spec
 started=time.monotonic();last=[-1.0]
 def guard():
  now=time.monotonic();assert now-started<600
  assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(66306,8294421)
  if now-last[0]>=.5:assert latest.latest_handoff()==tip;last[0]=now
  assert not any(n.split('.')[0] in ('torch','numpy','neuronxcc','torch_neuronx','nki') for n in sys.modules)
 print('Concrete source review accepted; checking complete original History',flush=True)
 bound=History(spec,spec_ref,review_ref,stage.names(spec),guard).resolve()
 bound.unchanged();fresh=bound.freshness()
 assert fresh['latest_handoff']==catalog['current_handoff']
 assert len(bound.supplement['scope']['children'])==874
 loaded_local_sources(spec['source_sha256']);resources=stage.resources(spec,reserve=True)
 assert latest.latest_handoff()==tip
 value=dict(schema='qwen35_integer_reference_root_concrete_binding_v1',reviewer='/root',passed=True,
  observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate=candidate_ref,
  source_review=source_review,spec=spec_ref,review=review_ref,history_overlay=spec['history_overlay'],
  catalog=spec['current_serial_catalog'],idle_handoff=catalog['current_handoff'],
  source_files=len(spec['source_sha256']),catalog_rows=157,input_extension_count=len(spec['input_extension']),
  full_source_history_resolution_passed=True,lfm_children=874,original_failed_statuses_preserved=True,
  original_lock_reacquired=True,resources=resources,physical_pair_performed=False,native_executed=False,
  runtime_launched=False,full_model_qualified=False,new_speedup_claim=False)
 result=once(Q/'qwen35-integer-reference-r1-root-binding-20261112-v1.json',value)
 print(json.dumps(dict(binding=result,spec=spec_ref,review=review_ref)),flush=True)
finally:
 fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
