"""Root concrete composition for complete integer-reference matrix; metadata only, no native or lock."""
import copy, hashlib, json, os, sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research'
sys.path[:0]=[str(Q),str(R)]
assert sys.dont_write_bytecode and __debug__
import qwen35_integer_reference_serial_contract_v1 as stage
from prepare_qwen35_sampler_r6_candidate_v1 import JsonMetadata
from prepare_qwen35_sampler_v9_candidate_v3 import encode,object_ref,merge_inputs,catalog_shape
from qwen35_sampler_runtime_contract_v12 import merge_sources
from qwen35_sampler_runtime_io_v12 import ref,identity
reader=JsonMetadata(lambda:None)
def read(r):return reader.read(r)['value']
def at(name):return ref(Q/name,64*2**20)
seed_ref=at('qwen35-slot-select-probe-r2-20261112-spec.json')
assert seed_ref['sha256']=='8b97a181a3fef0d720df17ba229ea3ce30a142b7390aae76bfb010d637e847c0'
seed=read(seed_ref);seed_review_ref=at('qwen35-slot-select-probe-r2-20261112-review.json');seed_review=read(seed_review_ref)
seed_binding_ref=at('qwen35-slot-select-probe-r2-root-binding-20261112-v1.json');seed_binding=read(seed_binding_ref)
assert seed_binding_ref['sha256']=='aa78eff5892e6ef9886329d6e7461c5c650efced5d152a0c17ff5678accf8aa0'
assert seed_binding['passed'] and seed_binding['full_source_history_resolution_passed'] and seed_binding['spec']==seed_ref and seed_binding['review']==seed_review_ref
assert seed_review['passed'] and seed_review['launch_authorized'] and not seed_review['blockers'] and seed_review['source_sha256']==seed['source_sha256']
overlay=read(seed['history_overlay']);old_catalog=read(seed['current_serial_catalog']);base=read(overlay['base_plan'])
cat_ref=at('minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v109.json');catalog=read(cat_ref)
assert cat_ref['sha256']=='6d881d5b266221dcf016c7f8e81c50504d6d2e042550ec4e1e6dd1ac17e95446'
catalog_shape(catalog);assert len(catalog['runs'])==157 and catalog['runs'][:len(old_catalog['runs'])]==old_catalog['runs'] and catalog['completed_sdk']==old_catalog['completed_sdk']
hand_ref=catalog['current_handoff'];hand=read(hand_ref)
assert hand_ref['sha256']=='ca1cefe8b5dfd1b9701a677366a39e3989138ace1059ba12dc005d311b2568da'
assert hand['serial_execution'] is True and hand['running'] is False and hand['active_serial'] is None and hand['live_tool_sessions']==hand['active_heavy_units']==[]
terminal_ref=at('qwen35-slot-select-probe-r2-root-actual-exit-20261112-v1.json');terminal=read(terminal_ref)
assert terminal_ref['sha256']=='6fe2fcba9ea46142aa1f9910e7bebb1f093e6e6859bab33aee77223f1fe86227'
row=catalog['runs'][-1]
assert row['terminal']==terminal_ref and row['tool']==terminal['actual_tool']==dict(kind='tool_session',session_id=22831,terminal_chunk='a07595',exit_code=0)
assert terminal['physically_drained'] and terminal['owners_gone'] and terminal['original_lock_reacquired_and_released']
assert row['owners']==terminal['owners'] and row['units']==terminal['units'] and terminal['lock_identity']==stage.LOCK_IDENTITY
assert terminal['diagnostic_complete'] and not terminal['qualification_matrix_completed'] and not terminal['full_model_qualified']
acceptance_ref=at('qwen35-integer-reference-root-source-review-20261112-v1.json');acceptance=read(acceptance_ref)
assert acceptance['reviewer']=='/root' and acceptance['source_accepted'] is True and acceptance['blockers']==[]
delivery_ref=acceptance['delivery'];delivery=read(delivery_ref)
assert delivery['schema']=='qwen35_integer_reference_serial_source_delivery_v1' and delivery['source_only'] is True
assert delivery['workload']==stage.protocol.workload() and delivery['limits']==stage.LIMITS
read(seed['lfm_r2_scope']);read({k:stage.MANIFEST[k] for k in ('path','sha256')})
for r in stage.protocol.REFERENCE_EVIDENCE:read(r)
recovered={p:s for p,s in seed['source_sha256'].items() if p not in overlay['source_extension']}
assert len(recovered)==44532 and merge_sources(recovered,overlay['source_extension'])==seed['source_sha256'] and len(base['inputs'])==8791
metadata_refs=[seed_binding_ref,cat_ref,hand_ref,terminal_ref,acceptance_ref,delivery_ref]
metadata_refs.append(at('accept_qwen35_integer_reference_root_r1_20261112_v1.py'))
pins=merge_sources(seed['source_sha256'],*(row['source_sha256'] for row in catalog['runs'] if row['source_sha256']),delivery['static_local_source_sha256'],delivery['sdk_source_sha256'],{r['path']:r['sha256'] for r in metadata_refs},{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
ordinary={p:copy.deepcopy(v) for p,v in reader.observed.items() if v['identity']['size']<=8*2**20}
preparation_only={p:copy.deepcopy(v) for p,v in reader.observed.items() if p not in ordinary}
assert seed_ref['path'] in preparation_only and seed_review_ref['path'] in preparation_only and overlay['base_plan']['path'] in preparation_only
run='r1-20261112';prefix='qwen35-integer-reference-slot-'+run
paths=dict(overlay=str(Q/(prefix+'-history-overlay.json')),spec=str(Q/(prefix+'-spec.json')),review_draft=str(Q/(prefix+'-review-draft.json')),review=str(Q/(prefix+'-review.json')),candidate=str(Q/(prefix+'-binding-candidate.json')),output='/mnt/nvme/neuron-work/'+prefix,overlay_output='/mnt/nvme/minicpm5-layer1-operator-native-integer-reference-'+run,running=str(Q/'continuation-handoff-20261112-v580-qwen35-integer-reference-r1-running.json'))
overlay=copy.deepcopy(overlay);overlay.update(current_serial_catalog=cat_ref,output_root=paths['overlay_output'],source_extension={p:s for p,s in pins.items() if p not in recovered})
merge_inputs(overlay['input_extension'],{p:v for p,v in ordinary.items() if p not in base['inputs']});assert not set(overlay['input_extension'])&set(base['inputs'])
overlay_ref=object_ref(paths['overlay'],overlay,ordinary=True)
inputs=copy.deepcopy(seed['input_extension']);merge_inputs(inputs,ordinary)
assert all(v['identity']['size']<=8*2**20 for v in inputs.values())
spec=dict(schema='qwen35_integer_reference_serial_spec_v1',run_name=run,output_root=paths['output'],history_overlay=overlay_ref,current_serial_catalog=cat_ref,source_sha256=pins,input_extension=inputs,running_handoff_path=paths['running'],lfm_r2_scope=copy.deepcopy(seed['lfm_r2_scope']),source_delivery=delivery_ref,primitive_manifest=stage.MANIFEST,workload=stage.protocol.workload(),limits=stage.LIMITS,lock_identity=stage.LOCK_IDENTITY,native_python=stage.PYTHON,automatic_successor=False)
spec_ref=object_ref(paths['spec'],spec)
draft=dict(schema='qwen35_sampler_runtime_source_review_v1',reviewer='/root',passed=False,blockers=['Root concrete source and full History precheck pending'],spec=spec_ref,source_sha256=pins,launch_authorized=False)
stage.validate(spec,dict(draft,passed=True,blockers=[],launch_authorized=True),spec_ref)
stage.verify_delivery(spec);stage.resources(spec,reserve=True)
assert len(set(paths.values()))==len(paths) and all(not Path(p).exists() and not Path(p).is_symlink() for p in paths.values())
assert all(identity(p)==v['identity'] for p,v in reader.observed.items())
for key,value in [('overlay',overlay),('spec',spec),('review_draft',draft)]:
 with Path(paths[key]).open('xb') as stream:stream.write(encode(value));stream.flush();os.fsync(stream.fileno())
assert ref(paths['overlay'],64*2**20)==overlay_ref and ref(paths['spec'],64*2**20)==spec_ref
candidate=dict(schema='qwen35_integer_reference_root_binding_candidate_v1',paths=paths,spec_ref=spec_ref,overlay_ref=overlay_ref,review_draft_ref=ref(paths['review_draft'],64*2**20),source_review=acceptance_ref,preparation_only_inputs=preparation_only,catalog_rows=157,source_files=len(pins),root_review_pending=True,full_history_precheck_pending=True,physical_pair_performed=False,lock_acquired=False,native_executed=False,compiler_executed=False,runtime_launched=False)
assert all(identity(p)==v['identity'] for p,v in reader.observed.items())
with Path(paths['candidate']).open('xb') as stream:stream.write(encode(candidate));stream.flush();os.fsync(stream.fileno())
print(json.dumps(dict(candidate=ref(paths['candidate']),spec=spec_ref,source_files=len(pins),input_extensions=len(inputs),catalog_rows=157)))
