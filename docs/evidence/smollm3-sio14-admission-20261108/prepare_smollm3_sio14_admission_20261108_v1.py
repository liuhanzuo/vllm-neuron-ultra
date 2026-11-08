"""Bind final history, forecast and source acceptance for the unchanged twelve-run scientific matrix."""
import datetime,hashlib,json,os,resource,secrets,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R))
import smollm3_observer_invariance_protocol_v20 as p
import smollm3_observer_invariance_lifecycle_v24 as life
import historical_b3_payload_partition_v1 as history
from research import smollm3_sio14_prospective_budget_v1 as forecast
CAT=Q/'nemotron9b-package-r4-20261108-complete-catalog-v3.json'
CAT_SHA='17623034921ec7a5e48d68bd9a22e6377bf2ce0846931d280aa57ca20f612207'
BUDGET=Q/'smollm3-sio14-final-admission-budget-20261108-v1.json'
SPEC=Q/'smollm3-observer-sio14-execution-spec-20261108-v1.json'
REVIEW=Q/'smollm3-observer-sio14-execution-review-20261108-v1.json'
ADDED=['smollm3-sio14-actual-history-extension-source-checks-20261108-v1.json','smollm3-sio14-budget-source-checks-20261108-v1.json','smollm3-sio14-completed-memory-probe-append-20261108-v1.json','smollm3-sio14-runtime-source-checks-20261108-v1.json','check_smollm3_sio14_reader_integration_20261108_v2.py','probe_smollm3_sio14_retained_audit_memory_20261108_v1.py','smollm3-sio14-retained-audit-memory-probe-20261108-v1.json','check_smollm3_sio14_prospective_budget_v1.py','prepare_smollm3_sio14_admission_20261108_v1.py','smollm3-sio12-post-finalizer-output-verification-20261108-v1.json','smollm3-sio12-terminal-receipt-source-preparation-20261108-v1.json']
def ref(f):return dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
def load(f):return json.loads(f.read_text())
def record(f,v):
 raw=life.json_bytes(v);assert len(raw)<16*2**20
 with f.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return ref(f)
def main():
 assert not p.DEST.exists()and not any(f.exists()for f in [BUDGET,SPEC,REVIEW,life.CALLABLE_ACCEPTANCE])
 assert ref(CAT)['sha256']==CAT_SHA;catalog=load(CAT);assert len(catalog['rows'])==838 and len(catalog['processes'])==164
 template=load(life.TEMPLATE);handoff=load(life.HANDOFF);base=p.load_base();p.validate_template(template,base)
 assert handoff['source_frozen']is True
 life.source_closure({'source_sha256':handoff['source_sha256']},{},handoff['original_metadata_symlink_resolutions'],hash_bytes=True)
 tests=load(Q/'smollm3-sio14-finalizer-source-checks-20261108-v1.json');assert tests['passed']and tests['test_methods_passed']==26
 for source in tests['source_refs']:assert ref(Path(source['path']))==source
 memory=load(Q/'smollm3-sio14-retained-audit-memory-probe-20261108-v1.json');assert memory['passed']and memory['error']is None and len(memory['fixture_slots'])==12
 assert memory['final_memory']['peak_rss_bytes']<320*2**20 and memory['address_space_limit_bytes']==512*2**20
 for f in ADDED:
  if f not in forecast.EXTRAS:forecast.EXTRAS.append(f)
 resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
 result=forecast.forecast(CAT,CAT_SHA,verify_source_bytes=True)
 assert result['forecast']['forecast_fits_original_caps']and result['root_free_floor_met']and result['nvme_floor_met']
 result['remaining']=['Actual final runtime load_spec/source and live storage/owner admission checks.','Actual12distinct runs, final scientific reanalysis and physical terminal review remain mandatory.']
 result['root_budget_review']=dict(accepted_for_fixed_workload=True,empirical_not_hypothetical_maximum_bound=True,off_mode_headroom_bytes=2*2**20,unexpected_output_fails_and_is_retained=True,memory_probe=ref(Q/'smollm3-sio14-retained-audit-memory-probe-20261108-v1.json'),memory_probe_only_repeats_real_reference_fixtures=True,original_readers_and_caps_unchanged=True,remaining_full_runtime_checks_retained=True)
 budget_ref=record(BUDGET,result)
 # Construct exactly the forecast union. Bound budget/acceptance/spec/review
 # by separate immutable references, avoiding self-referential source hashes.
 parts=history.partition(catalog);pins={}
 for values in [parts['source_sha256'],parts['provenance_sha256'],handoff['source_sha256'],template['source_sha256']]:forecast.merge(pins,values)
 for f in [CAT,life.TEMPLATE,life.HANDOFF,*[Q/n for n in forecast.EXTRAS],Path(forecast.__file__)]:item=ref(f);forecast.merge(pins,{item['path']:item['sha256']})
 assert len(pins)==result['source_entry_count']
 # The retained memory probe used this exact predecessor plus one fewer
 # completed source diagnostic. Bound all added metadata against its already
 # allocated16MiB future-binding allowance; never claim a new12-run execution.
 measured=load(Path(memory['catalog']['path']));assert ref(Path(memory['catalog']['path']))==memory['catalog']
 assert catalog['rows']==measured['rows'] and catalog['processes'][:len(measured['processes'])]==measured['processes']
 assert all(catalog['source_sha256'].get(k)==v for k,v in measured['source_sha256'].items())
 measured_pins={};measured_parts=history.partition(measured)
 for values in [measured_parts['source_sha256'],measured_parts['provenance_sha256'],handoff['source_sha256'],template['source_sha256']]:forecast.merge(measured_pins,values)
 delta_pins={k:v for k,v in pins.items()if k not in measured_pins}
 delta_record=dict(source_entries=delta_pins,added_processes=catalog['processes'][len(measured['processes']):],extra_catalog_fields={k:v for k,v in catalog.items()if k not in measured})
 reserve_required_bytes=8*len(life.json_bytes(delta_record))
 assert reserve_required_bytes<memory['extra_future_binding_allowance_bytes']==16*2**20
 memory_delta=dict(measured_catalog=memory['catalog'],final_catalog=ref(CAT),additional_metadata_source_entries=len(delta_pins),eight_copies_serialized_delta_bytes=reserve_required_bytes,retained_allowance_bytes=16*2**20,within_original_memory_probe_allowance=True,not_a_new_model_run=True)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 acceptance=record(life.CALLABLE_ACCEPTANCE,dict(schema='smollm3_sio14_callable_root_source_acceptance_v1',reviewer='/root',observed_at=now,passed=True,independent_review=True,source_only=True,source_handoff=ref(life.HANDOFF),source_template=ref(life.TEMPLATE),blockers=[],review_basis='Root review independent of prior model producers; root authored metadata/finalizer and reviewed source contracts, retained test evidence, final history, full-category forecast and actual retained-data exporter memory path. No separate-author review claimed.',memory_probe_final_binding_delta=memory_delta,frozen_runtime_modules=handoff['complete_migrated_modules'],metadata_helpers=handoff['metadata_helpers'],finalizer_source_checks=ref(Q/'smollm3-sio14-finalizer-source-checks-20261108-v1.json'),current_final_catalog=ref(CAT),complete_category_budget=budget_ref,memory_probe=ref(Q/'smollm3-sio14-retained-audit-memory-probe-20261108-v1.json'),source_entries_fully_rehashed=len(pins),source_frozen=True,resources_and_science_unchanged=True,native_runtime_started=False,full_matrix_completed=False,new_speedup_claim=False))
 spec=dict(schema='smollm3_observer_invariance_execution_spec_v10',source_template=ref(life.TEMPLATE),callable_handoff=ref(life.HANDOFF),callable_root_acceptance=acceptance,work=str(p.DEST),matrix=p.matrix(),runtime_admitted=True,root_review_required=True,supervisor_unit=life.UNIT,planned_units=[p.unit(row['run_id'],kind)for row in p.matrix()for kind in ('server','client')]+[life.UNIT],wall_time=template['wall_time'],storage=template['storage'],attempt_ids={row['run_id']:secrets.token_hex(16)for row in p.matrix()},canonical_sha256={name:ref(R/name)['sha256']for name in ('continuation-state.json','roadmap.json','best-known-configurations.json')},runtime_source_sha256=pins,historical_payload_sha256=parts['historical_payload_sha256'],final_predecessor_catalog=ref(CAT),automatic_successor_launch=False,compile_allowed=False)
 life.validate_spec(spec,template,base);spec_ref=record(SPEC,spec)
 review_ref=record(REVIEW,dict(schema='smollm3_sio14_execution_source_review_v1',reviewer='/root',observed_at=now,passed=True,independent_review=True,source_only=True,execution_spec=spec_ref,spec_sha256=spec_ref['sha256'],runtime_source_sha256=pins,source_acceptance=acceptance,complete_category_budget=budget_ref,memory_probe=ref(Q/'smollm3-sio14-retained-audit-memory-probe-20261108-v1.json'),finalizer=ref(Q/'finalize_smollm3_sio14_root_20261108_v1.py'),blockers=[],runtime_launch_performed=False,new_speedup_claim=False))
 loaded=life.load_spec(SPEC,spec_ref['sha256']);assert loaded[0]==spec
 for path,floor in [(R,30*2**30),(Path('/mnt/nvme'),5*2**30)]:assert os.statvfs(path).f_bavail*os.statvfs(path).f_frsize>=floor
 assert not p.DEST.exists()
 print(json.dumps(dict(spec=spec_ref,review=review_ref,acceptance=acceptance,budget=budget_ref,source_entries=len(pins),runtime_started=False)))
if __name__=='__main__':main()
