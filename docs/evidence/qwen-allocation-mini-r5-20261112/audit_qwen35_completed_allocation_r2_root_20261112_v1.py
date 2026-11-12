"""Join the actual external exit to the completed metadata-only allocation scan."""
import copy,datetime,hashlib,json,sys
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path[:0]=[str(Q),str(R)]
from qwen35_sampler_runtime_io_v12 import RetainedLock,process_api,ref,read,once
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
assert sys.dont_write_bytecode
actual=dict(kind='tool_session',session_id=1710,terminal_chunk='4b463a',exit_code=0)
prefix='qwen35-completed-allocation-r2-20261112'
terminal_ref=ref(Q/(prefix+'-terminal.json'));terminal=read(terminal_ref)
assert terminal['complete'] is True and terminal['original_lock_released'] and terminal['no_children_or_units_created'] and terminal['native_executed'] is False
live_ref=ref(Q/'qwen35-completed-allocation-r2-root-live-20261112-v1.json');live=read(live_ref)
assert live['actual_tool']==dict(kind='tool_session',session_id=1710) and live['owner']==terminal['owner'] and live['source']==terminal['source']
result_ref=terminal['result'];result=read(result_ref);admission_ref=result['admission'];admission=read(admission_ref)
assert result['complete'] and result['passes']==2 and 0<result['elapsed_seconds']<120
assert result['full_source_history_resolved'] and admission['source_history_resolved']
assert result['owner']==admission['owner']==terminal['owner']
assert result['native_executed'] is False and result['regular_payloads_opened'] is False
assert result['historical_compiler_workdir_attributed'] is result['historical_cache_attributed'] is False
assert result['source']['sha256']=='22b6ced5e718172418e709ba0f0e8f1c1a5c5483d917008b54b4392d3ce289f1'
assert admission['spec']==terminal['spec'] and admission['review']==terminal['review']
cat_ref=terminal['catalog'];cat=read(cat_ref);hand_ref=cat['current_handoff'];hand=read(hand_ref)
assert len(cat['runs'])==154 and admission['catalog']==cat_ref and terminal['idle_handoff']==admission['idle_handoff']==hand_ref
assert latest.latest_handoff()['path']==hand_ref['path'] and hand['running'] is False
# This small actual tree has only three regular files. Both largest-file lists
# therefore preserve every file row, allowing full independent stat accounting.
s=result['summary'];files=s['largest_logical_files']
assert s['entries']==s['unique_inodes']==4 and s['entry_kinds']==dict(directory=1,file=3)
assert len(files)==3 and files==s['largest_allocated_files']
root=dict(result['root_identity'],path='.',kind='directory');rows={r['path']:r for r in [root]+files}
assert len(rows)==4 and all(r['nlink']==1 for r in files)
assert len({(r['device'],r['inode']) for r in rows.values()})==4
assert all(rows['model.pt'][k]==v for k,v in result['model_anchor'].items())
encode=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
digest=hashlib.sha256(b''.join(encode(rows[p])+b'\n' for p in sorted(rows))).hexdigest()
assert digest==s['metadata_inventory_sha256']
assert sum(r['blocks']*512 for r in rows.values())==s['allocated_bytes']==s['path_counted_allocated_bytes']==294494208
assert sum(r['size'] for r in files)==s['regular_file_path_logical_bytes']==294483317
assert max(r['size'] for r in files)==s['max_file_logical_bytes']==294473003
assert max(r['blocks']*512 for r in files)==s['max_file_allocated_bytes']==294473728
assert s['external_hardlinked_inodes']==0 and s['reclaimable_bytes_claimed'] is s['symlinks_followed'] is s['symlink_targets_read'] is False
lock=RetainedLock();lock.acquire()
try:
 p=process_api();observations=[]
 for _ in range(2):
  p.gone(terminal['owner']);p.active_heavy();observations.append(dict(owner_gone=True,active_heavy_absent=True))
 assert latest.latest_handoff()['path']==hand_ref['path']
 v=dict(schema='qwen35_completed_allocation_root_actual_exit_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tool=actual,actual_tool_exit_observed=True,actual_outer_exit_observed=True,terminal=terminal_ref,admission=admission_ref,live_observation=live_ref,owners=[terminal['owner']],owners_gone=True,units=[],physically_drained=True,no_children_or_units_created=True,original_lock_reacquired_and_released=True,lock_identity=dict(device=66306,inode=8294421),two_current_observations=observations,scanner_executed=True,result=result_ref,full_source_history_resolved=True,model_or_neuron_native_executed=False,stdlib_ctypes_imported=True,independent_all_four_metadata_rows_reconstructed=True,independent_metadata_inventory_sha256=digest,allocated_bytes=s['allocated_bytes'],regular_file_path_logical_bytes=s['regular_file_path_logical_bytes'],reclaimable_bytes_claimed=False,historical_compiler_workdir_attributed=False,historical_cache_attributed=False,auditor_source=ref(Path(__file__).resolve()),full_model_qualified=False,new_speedup_claim=False)
 ended_ref=once(Q/'qwen35-completed-allocation-r2-root-actual-exit-20261112-v1.json',v)
 delivery=read(ref(Q/'qwen35-slot-special-value-serial-source-delivery-20261112-v2.json'));pins=dict(delivery['static_local_source_sha256'])
 for path in [Q/'qwen35_completed_allocation_inventory_v1.py',Q/'run_qwen35_completed_allocation_root_r2_20261112_v2.py',Path(__file__).resolve()]:pins[str(path)]=ref(path)['sha256']
 for path,sha in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 cat['runs'].append(dict(terminal=ended_ref,tool=actual,owners=[terminal['owner']],units=[],source_sha256=pins))
 idle=copy.deepcopy(hand);idle.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous=hand_ref,previous_handoff=hand_ref,running=False,active_serial=None,live_tool_sessions=[],active_heavy_units=[],goal_status='active',this_turn_classification='progress',complete_current_serial_catalog=cat_ref,complete_current_serial_catalog_next=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v106.json'),latest_verified_publication=ref(Q/'qwen-special-r2-results-publication-verification-20261112-v1.json'),publication_current_status_stale=True)
 idle['qwen35_completed_allocation_current']=dict(actual_exit=ended_ref,result=result_ref,scanner_executed=True,allocated_bytes=s['allocated_bytes'],new_compile_storage_budget_proven=False)
 idle['next_serial_work']='Prepare Mini formal r5 against all155 completed rows with exact completed-slot source successor; Qwen package25phase and select/bitcast12call runtime remain source-only pending root review and finite budgets.'
 idle_ref=once(Q/'continuation-handoff-20261112-v575-qwen35-allocation-r2-ended.json',idle);cat['current_handoff']=idle_ref;catalog_shape(cat)
 newcat=once(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v106.json',cat)
 print(json.dumps(dict(actual_exit=ended_ref,catalog=newcat,handoff=idle_ref,rows=len(cat['runs']),allocated_bytes=s['allocated_bytes'])))
finally:lock.release(True)
