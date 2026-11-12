"""One root-owned metadata scan with unchanged complete History and retained lock.

No native libraries, model payloads, service, compiler or successor are launched.
"""
import copy,datetime,hashlib,json,os,resource,signal,sys,time
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
from qwen35_sampler_runtime_io_v12 import RetainedLock,read,ref,once,identity,process_api,loaded_local_sources
from qwen35_sampler_runtime_contract_v12 import merge_sources
from qwen35_package_preflight_history_v2 import History,Authority
from prepare_qwen35_sampler_r6_candidate_v1 import JsonMetadata,catalog_shape
from prepare_qwen35_sampler_v9_candidate_v3 import encode,object_ref,merge_inputs
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
import qwen35_completed_allocation_inventory_v1 as scanner
assert sys.dont_write_bytecode and __debug__
resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30));resource.setrlimit(resource.RLIMIT_CPU,(900,905));resource.setrlimit(resource.RLIMIT_FSIZE,(64*2**20,64*2**20));signal.alarm(900)
self_ref=ref(Path(__file__).resolve());prefix='qwen35-completed-allocation-r1-20261112'
for suffix in ('spec','review','overlay','admission','result','terminal'):
 assert not (Q/(prefix+'-'+suffix+'.json')).exists()
metadata=JsonMetadata(lambda:None)
def get(r):return metadata.read(r)['value']
seed_ref=ref(Q/'qwen35-slot-special-value-r2-v2-20261112-spec.json',64*2**20);assert seed_ref['sha256']=='92a857b9715b84104a7981cba6fdf324438437a4df8eb823642ee5f5a91aeb8d';seed=get(seed_ref)
seed_binding_ref=ref(Q/'qwen35-slot-special-value-r2-root-binding-20261112-v1.json');seed_binding=get(seed_binding_ref);assert seed_binding['spec']==seed_ref and seed_binding['passed'] and seed_binding['full_source_history_resolution_passed']
old_overlay=get(seed['history_overlay']);old_catalog=get(seed['current_serial_catalog']);base=get(old_overlay['base_plan'])
cat_ref=ref(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v104.json');assert cat_ref['sha256']=='9bc186b643d554423a35390389cc28ea245c6fc383c02e41c37528731cffb4a5';catalog=get(cat_ref);catalog_shape(catalog)
assert len(catalog['runs'])==153 and catalog['runs'][:-1]==old_catalog['runs'] and catalog['completed_sdk']==old_catalog['completed_sdk']
hand_ref=catalog['current_handoff'];handoff=get(hand_ref);assert handoff['running'] is False and handoff['active_serial'] is None and handoff['active_heavy_units']==handoff['live_tool_sessions']==[]
ended_ref=ref(Q/'qwen35-slot-special-value-r2-root-actual-exit-20261112-v1.json');ended=get(ended_ref);assert ended_ref==catalog['runs'][-1]['terminal'] and ended['actual_tool']==dict(kind='tool_session',session_id=29233,terminal_chunk='bc15e6',exit_code=0) and ended['owners_gone'] and ended['physically_drained']
review_ref=ref(Q/'qwen35-completed-allocation-inventory-root-source-review-20261112-v1.json');review=get(review_ref);assert review['source_accepted'] is True and review['blockers']==[]
delivery_ref=review['delivery'];delivery=get(delivery_ref);assert delivery_ref['sha256']=='8fb9d8c1286883d37380ce65006e96482f5773d2f39fec18ba48c26eab720d32'
evidence=get({k:delivery['evidence'][k] for k in ('path','sha256')})
for row in evidence['full12_evidence']:get({k:row[k] for k in ('path','sha256')})
for row in delivery['new_sources']+[delivery['interface'],delivery['reused_algorithm_source']]:
 payload=Path(row['path']).read_bytes();assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
get(seed['lfm_r2_scope'])
recovered={p:s for p,s in seed['source_sha256'].items() if p not in old_overlay['source_extension']};assert len(recovered)==44532 and merge_sources(recovered,old_overlay['source_extension'])==seed['source_sha256']
new_refs=[seed_binding_ref,cat_ref,hand_ref,ended_ref,review_ref,delivery_ref,self_ref]+[{k:r[k] for k in ('path','sha256')} for r in delivery['new_sources']+[delivery['evidence'],delivery['interface']]]
pins=merge_sources(seed['source_sha256'],*(r['source_sha256'] for r in catalog['runs'] if r['source_sha256']),{r['path']:r['sha256'] for r in new_refs})
ordinary={p:v for p,v in metadata.observed.items() if v['identity']['size']<=8*2**20}
overlay=copy.deepcopy(old_overlay);overlay.update(current_serial_catalog=cat_ref,output_root='/mnt/nvme/minicpm5-layer1-operator-native-qwen-allocation-r1-20261112',source_extension={p:s for p,s in pins.items() if p not in recovered})
merge_inputs(overlay['input_extension'],{p:v for p,v in ordinary.items() if p not in base['inputs']});assert not set(overlay['input_extension'])&set(base['inputs'])
overlay_ref=once(Q/(prefix+'-overlay.json'),overlay)
inputs=copy.deepcopy(seed['input_extension']);merge_inputs(inputs,ordinary)
spec=dict(schema='qwen35_completed_allocation_root_spec_v1',source_sha256=pins,input_extension=inputs,history_overlay=overlay_ref,current_serial_catalog=cat_ref,lfm_r2_scope=seed['lfm_r2_scope'],target=scanner.TARGET,scanner_limits=scanner.LIMITS,source_review=review_ref,source_delivery=delivery_ref,native_execution=False)
spec_path=Q/(prefix+'-spec.json');spec_ref=object_ref(str(spec_path),spec)
with spec_path.open('xb') as f:f.write(encode(spec));f.flush();os.fsync(f.fileno())
assert ref(spec_path,64*2**20)==spec_ref
runtime_review=dict(schema='qwen35_sampler_runtime_source_review_v1',reviewer='/root',passed=True,blockers=[],spec=spec_ref,source_sha256=pins,launch_authorized=False)
# This source-only metadata review cannot authorize any native/runtime stage.
review_path=Q/(prefix+'-review.json')
with review_path.open('xb') as f:f.write(encode(runtime_review));f.flush();os.fsync(f.fileno())
runtime_review_ref=ref(review_path,64*2**20)
assert all(identity(p)==v['identity'] for p,v in metadata.observed.items())
lock=RetainedLock();lock.acquire();started=time.monotonic();initial_tip=latest.latest_handoff();assert initial_tip['path']==hand_ref['path']
completed=False;result_ref=None;error=None;history=None;last=[-1.0]
try:
 def guard():
  lock.check();assert time.monotonic()-started<880
  assert not any(n.split('.')[0] in ('torch','torch_xla','numpy','neuronxcc','torch_neuronx','ctypes','_ctypes') for n in sys.modules)
  now=time.monotonic()
  if now-last[0]>=.5:
   assert latest.latest_handoff()==initial_tip;last[0]=now
  return dict(admitted=True,exclusive=True,completed_tree=scanner.TARGET,target_not_active=True)
 guard();process_api().active_heavy()
 history=History(spec,spec_ref,runtime_review_ref,dict(coordinator='trn1-qwen35-completed-allocation-r1-20261112'),guard).resolve()
 pair=history.physical_pair(Authority(history,guard,lambda:[]));history.unchanged();fresh=history.freshness();assert fresh['latest_handoff']==hand_ref
 process_api().active_heavy()
 for owner in ended['owners']:process_api().gone(owner)
 root_identity=scanner.metadata(os.lstat(scanner.TARGET));admission_ref=once(Q/(prefix+'-admission.json'),dict(schema='qwen35_completed_allocation_metadata_admission_v1',spec=spec_ref,review=runtime_review_ref,catalog=cat_ref,idle_handoff=hand_ref,owner=lock.owner,retained_fd=lock.fd,initial_lock_proof=lock.initial,physical_pair=pair,freshness=fresh,root_identity=root_identity,source_history_resolved=True,lfm_children=len(history.supplement['scope']['children']),native_executed=False))
 print(json.dumps(dict(event='complete_history_metadata_inventory_admitted',admission=admission_ref,owner=lock.owner,catalog_rows=153)),flush=True)
 result=scanner.inventory(scanner.TARGET,root_identity,guard);history.unchanged();guard();process_api().active_heavy()
 result.update(admission=admission_ref,owner=lock.owner,source=ref(Path(scanner.__file__)),full_source_history_resolved=True)
 result_ref=once(Q/(prefix+'-result.json'),result);completed=True
finally:
 # No child or unit has been created. Retain the original lease throughout all metadata IO.
 lock.check();process_api().active_heavy();owner=lock.owner;lock.release(True)
 terminal=once(Q/(prefix+'-terminal.json'),dict(schema='qwen35_completed_allocation_metadata_terminal_v1',spec=spec_ref,review=runtime_review_ref,catalog=cat_ref,idle_handoff=hand_ref,owner=owner,result=result_ref,complete=completed,no_children_or_units_created=True,original_lock_released=True,native_executed=False,self_exit_claimed=False,actual_external_exit_pending=True,source=self_ref))
print(json.dumps(dict(result=result_ref,terminal=terminal,summary=result['summary'],elapsed=result['elapsed_seconds'])),flush=True)
