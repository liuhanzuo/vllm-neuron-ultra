"""Record actual r2 precheck failure without admitting any model/native stage."""
import copy
import datetime
import json
from pathlib import Path
import sys

R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031'); Q=R/'research'
sys.path[:0]=[str(Q),str(R)]
assert sys.dont_write_bytecode and __debug__
from qwen35_package_serial_runtime_v4 import contract,io,budget
from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

actual=dict(kind='tool_session',session_id=53698,terminal_chunk='ca4d13',exit_code=1)
spec_ref=dict(path=str(Q/'qwen35-package-full-r2-20261112-spec.json'),sha256='9f9b6f438c469c5bf4d3e28896f9f9c6f0f68549fc2d9d383047985ae49637d2')
spec=io.read(spec_ref,64*2**20)
review_ref=io.ref(Q/'qwen35-package-full-r2-20261112-review.json',64*2**20)
cat_ref=spec['current_serial_catalog']; oldcat=io.read(cat_ref)
idle_ref=oldcat['current_handoff'];idle=io.read(idle_ref)
assert latest.latest_handoff()==dict(path=idle_ref['path'],identity=io.identity(idle_ref['path']))
assert len(oldcat['runs'])==159 and not idle['live_tool_sessions'] and not idle['active_heavy_units']
assert not (Q/'qwen35-package-full-r2-root-binding-20261112-v1.json').exists()
assert not Path(spec['running_handoff_path']).exists()
diag_ref=io.ref(Q/'qwen35-package-full-r2-root-reader-diagnostic-20261112-v1.json');diag=io.read(diag_ref)
assert diag['spec']==spec_ref and diag['host_reader_v4_passed']
assert diag['inherited_partition_reader_failure']==dict(type='RuntimeError',message='Ordinary bounded metadata')
overlay=io.read(spec['history_overlay'])
prior={p:sha for p,sha in spec['source_sha256'].items() if p not in overlay['source_extension']}
assert len(prior)==44532
for row in oldcat['runs']:
    for p,sha in row['source_sha256'].items():
        assert p not in prior or prior[p]==sha
        prior[p]=sha
assert all(spec['source_sha256'].get(p)==sha for p,sha in prior.items())
delta={p:sha for p,sha in spec['source_sha256'].items() if p not in prior}
assert dict(prior,**delta)==spec['source_sha256']
for path,sha in delta.items():
    assert Path(path).suffix in ('.py','.json','.md','.txt','.patch','.sh','.toml','.cfg','.ini','.jinja','.log')
    assert io.ref(path)['sha256']==sha
declared_delta=len(delta)
own_ref=io.ref(Path(__file__).resolve());delta[own_ref['path']]=own_ref['sha256']
lock=io.RetainedLock();lock.acquire()
try:
    io.process_api().active_heavy();budget.fresh_paths(spec)
    expected=str(Q/'accept_qwen35_package_full_root_r2_20261112_v1.py').encode(); matching=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if expected in argv:matching.append(int(proc.name))
    assert not matching
    assert latest.latest_handoff()==dict(path=idle_ref['path'],identity=io.identity(idle_ref['path']))
    receipt=io.once(Q/'qwen35-package-full-r2-root-precheck-failure-20261112-v1.json',dict(
        schema='qwen35_package_full_root_precheck_failure_v1',reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_tool=actual,
        actual_tool_exit_observed=True,spec=spec_ref,review=review_ref,
        prepare_actual_exit=io.ref(Q/'qwen35-package-full-r2-root-preparation-tool-exit-20261112-v1.json'),
        failed_source=io.ref(Q/'accept_qwen35_package_full_root_r2_20261112_v1.py'),
        runtime_history_source=io.ref(Q/'qwen35_package_serial_runtime_v4/history.py'),
        failure=dict(type='RuntimeError',message='Ordinary bounded metadata',line=77,
            path=diag['host_source'],applied_cap_bytes=8*2**20,original_spec_envelope_bytes=64*2**20),
        exact_reader_diagnostic=diag_ref,host_reader_v4_passed=True,
        full_history_resolution_passed=False,physical_pair_performed=False,
        six_runtime_trees_absent=True,no_running_handoff=True,actual_binding_absent=True,
        original_lock_reacquired_and_released=True,lock_identity=contract.LOCK_IDENTITY,
        owners=[],units=[],precheck_owner_identity_not_recorded=True,
        exact_precheck_argv_matches_after_terminal=matching,
        source_composition=dict(prior_sources=len(prior),specified_sources=len(spec['source_sha256']),
            newly_verified_declared_sources=declared_delta,recorder_sources=1,all_previous_rows_preserved=True),
        native_executed=False,compiler_executed=False,model_executed=False,runtime_admitted=False,
        full_model_qualified=False,new_speedup_claim=False,recorder_source=own_ref))
    handoff=copy.deepcopy(idle)
    handoff.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous=idle_ref,
        previous_handoff=idle_ref,publication_only_since_bound_terminal=False,complete_current_serial_catalog=cat_ref,
        complete_current_serial_catalog_next=str(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v113.json'),
        goal_status='active',this_turn_classification='progress',running=False,active_serial=None,
        live_tool_sessions=[],active_heavy_units=[])
    handoff['qwen35_package_full_current']=dict(status='r2_full_partition_precheck_failed_before_native',
        actual_exit=receipt,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
    handoff['latest_storage_reclaim']=io.ref(Q/'completed-mock-tmp-archive-20261112-v1.json')
    handoff['next_serial_work']='Review runtime v5 complete partition reader routing for the exact old r10 spec; preserve all source categories, aliases, Python-byte fixtures, original limits and both actual precheck failures. Fresh r3 binding must pass full History and original physical checks before native work. No new model speedup.'
    new_idle=io.once(Q/'continuation-handoff-20261112-v585-qwen-package-partition-precheck-failed.json',handoff)
    catalog=dict(oldcat,current_handoff=new_idle,runs=oldcat['runs']+[dict(terminal=receipt,tool=actual,owners=[],units=[],source_sha256=delta)])
    catalog_shape(catalog);assert catalog['runs'][:-1]==oldcat['runs']
    newcat=io.once(Q/'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v113.json',catalog)
finally:lock.release(True)
print(json.dumps(dict(actual_exit=receipt,catalog=newcat,idle=new_idle,rows=len(catalog['runs']))))
