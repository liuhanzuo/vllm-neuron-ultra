"""Actual SIGXCPU failure audit; no missing child wait/trace is promoted to pass."""
import copy,datetime,hashlib,json,signal,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
from qwen35_sampler_runtime_io_v12 import ref,read,once,RetainedLock,process_api
from research import minicpm5_formal_literal_codec_v1 as codec
from research import audit_minicpm5_formal_r4_completion_v1 as prior
from research import minicpm5_sdk_mapping_observer_logic_v2 as logic
from research import minicpm5_sdk_mapping_nocache_launch_v3 as launch
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
assert sys.dont_write_bytecode and __debug__
prefix='minicpm5-formal-import-r5-20261112'
live_ref=ref(Q/(prefix+'-root-live.json'));assert live_ref['sha256']=='57572cb9676c44349432f6182b70e702bee3d4e4c9e4ffbbddd6dc466bdcf34e';live=read(live_ref)
wait_ref=ref(Q/(prefix+'-launcher-wait.json'));waited=read(wait_ref)
actual=prior.tool_terminal(58506,'d8b842',152,live,waited)
assert waited['actual_returncode']==-signal.SIGXCPU and waited['pid']==live['owner']['pid']==948519
review=read(live['root_review']);spec=read(review['outer_spec']);plan=read(review['outer_entry_plan']);sdk_plan=read(review['sdk_entry_template']);pv=read(review['pv_spec'])
literal=Path(codec.LITERAL['path']).read_text();codec.literal_bytes(literal)
assert live['owner']['argv']==launch.launch(literal.encode(),review['entry_source'],plan,review['outer_entry_plan'])
assert spec['limits']==dict(address_space_bytes=2**31,cpu=6,cpu_seconds=600,hard_cpu_seconds=605,nice=19,output_file_bytes=2**21,wall_seconds=900)
output=Path(pv['output_root']);assert output==Q/'minicpm5-fixed-input-pv-sdk-import-formal-r5-20261112'
assert sorted(x.name for x in output.iterdir())==['admission.json','phase-1-owner.json']
saved={}
for name,kind in [('admission.json','admission'),('phase-1-owner.json','owner')]:
 p=output/name;b=p.read_bytes();saved[name]=dict(reference=ref(p),bytes=len(b),value=codec.decode(kind,b,literal,wire_sha256=hashlib.sha256(b).hexdigest()))
admission=saved['admission.json']['value'];sdk_owner=saved['phase-1-owner.json']['value']
assert saved['admission.json']['bytes']==44438 and saved['phase-1-owner.json']['bytes']==990
assert admission['lease']['outer_owner']==live['owner'] and admission['lease']['lock_identity']==dict(device=66306,inode=8294421) and admission['lease']['unit']==spec['unit']
assert admission['source_entry_plan']==sdk_plan and admission['source_entry_binding']['outer_spec']==review['outer_spec'] and admission['source_entry_binding']['outer_review']==review['outer_review']
assert admission['diagnostic_spec']==review['diagnostic_spec'] and admission['diagnostic_review']==review['diagnostic_review']
assert sdk_owner['argv']==launch.admitted_sdk(literal.encode(),review['entry_source'],admission,saved['admission.json']['reference'],admission['source_entry_binding']['outer_deadline'])
assert sdk_owner['pid']==960133 and sdk_owner['ppid']==live['owner']['pid'] and sdk_owner['cgroup']==live['owner']['cgroup']
assert logic.valid_owner(sdk_owner,sdk_owner['pid'])==sdk_owner
assert admission['witness']['source_history_and_catalog_reconstructed'] is True and admission['witness']['complete_source_files']==48223
owners=[live['owner'],sdk_owner,live['launcher_owner']]
cat_ref=admission['witness']['current_serial_catalog'];cat=read(cat_ref);assert len(cat['runs'])==155
hand_ref=cat['current_handoff'];hand=read(hand_ref);assert latest.latest_handoff()['path']==hand_ref['path']
log_ref=ref(Q/(prefix+'-tool.log'))
api=process_api();api.user_bus();matching={review['outer_entry_plan']['path'],str(output/'admission.json')}
observations=[prior.observe_drained(api,owners,[sdk_owner['pid']],sdk_owner['pid'],spec['unit'],matching) for _ in range(2)]
assert observations[0]==observations[1]
lock=RetainedLock();lock.acquire()
try:
 proof=prior.observe_drained(api,owners,[sdk_owner['pid']],sdk_owner['pid'],spec['unit'],matching);assert proof==observations[0]
 for name,row in saved.items():assert ref(output/name)==row['reference']
 assert ref(Q/(prefix+'-tool.log'))==log_ref and ref(Q/(prefix+'-launcher-wait.json'))==wait_ref and latest.latest_handoff()['path']==hand_ref['path']
 ended=dict(schema='minicpm5_formal_r5_sigxcpu_root_actual_exit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tool=actual,actual_tool_exit_observed=True,actual_outer_exit_observed=True,actual_outer_returncode=-24,signal='SIGXCPU',root_live=live_ref,root_review=live['root_review'],launcher_wait=wait_ref,log=log_ref,output_root=str(output),output_files={name:{k:v for k,v in row.items() if k!='value'} for name,row in saved.items()},owners=owners,units=[spec['unit']],owners_gone=True,physically_drained=True,two_current_observations=observations,post_lock_observation=proof,original_lock_reacquired_and_released=True,lock_identity=dict(device=66306,inode=8294421),original_lock_proof=lock.initial,original_limits=spec['limits'],failure='Actual outer terminated by SIGXCPU after durable admission and SDK owner, without durable child terminal/observer/trace. Original CPU soft/hard limits600/605s unchanged.',sdk_child_started=True,sdk_child_exit_status=None,sdk_import_completed=None,sdk_import_completion_verified=False,observer_complete_payload_available=False,observer_started=None,source_gate_release_completed=None,exact_interrupted_instruction_unknown=True,python_finally_cleanup_claimed=False,known_sdk_pid_start_delay_seconds=(int(sdk_owner['start_ticks'])-int(live['owner']['start_ticks']))/100,known_original_descriptor_release_physically_verified=True,all_observer_task_exits_reconstructed=False,original_nine_native_checks_passed=False,scientific_rows_completed=0,original_scientific_rows_still_pending=1284,native_ABI_qualified=False,full_model_qualified=False,new_speedup_claim=False,auditor_source=ref(Path(__file__).resolve()))
 ended_ref=once(Q/'minicpm5-formal-r5-root-actual-exit-20261112-v1.json',ended)
 pins=dict(spec['source_extension'])
 for p in [Path(__file__).resolve(),Q/'launch_minicpm5_formal_r5_root_20261112_v1.py',Q/'run_minicpm5_formal_r5_metadata_root_20261112_v1.py',Q/'finalize_minicpm5_formal_r5_root_20261112_v2.py']:
  pins[str(p)]=ref(p)['sha256']
 for path,sha in pins.items():assert ref(Path(path))['sha256']==sha
 cat['runs'].append(dict(terminal=ended_ref,tool=actual,owners=owners,units=[spec['unit']],source_sha256=pins))
 idle=copy.deepcopy(hand);idle.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous=hand_ref,previous_handoff=hand_ref,running=False,active_serial=None,live_tool_sessions=[],active_heavy_units=[],goal_status='active',this_turn_classification='progress',complete_current_serial_catalog=cat_ref,complete_current_serial_catalog_next=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v107.json'),latest_verified_publication=ref(Q/'qwen-allocation-mini-r5-publication-verification-20261112-v1.json'),publication_current_status_stale=True)
 idle['mini_formal_import_previous']=idle.get('mini_formal_import_current')
 idle['mini_formal_import_current']=dict(actual_exit=ended_ref,actual_exit_code=152,failure=ended['failure'],sdk_child_started=True,sdk_import_completion_verified=False,complete_observer_available=False,next='Source-only duplicate verification/CPU-cost review under unchanged600s limit. Do not rerun unchanged formal input.')
 idle['next_serial_work']='Review Qwen12call select diagnostic and bind all156 completed rows if accepted. Package25phase source needs readiness-log pending fix and independent review; Mini SIGXCPU source analysis in progress, unchanged limits and all scientific gates retained.'
 idle_ref=once(Q/'continuation-handoff-20261112-v576-mini-formal-r5-ended.json',idle);cat['current_handoff']=idle_ref;catalog_shape(cat)
 newcat=once(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v107.json',cat)
 print(json.dumps(dict(actual_exit=ended_ref,catalog=newcat,handoff=idle_ref,rows=len(cat['runs']),native_checks_passed=False,known_owners_gone=True)))
finally:lock.release(True)
