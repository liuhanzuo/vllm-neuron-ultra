"""Finish catalog publication after v1's duplicate base-map serialization overflow.

The original 8 MiB receipt cap remains. Only the source extension belongs in the
new catalog row; the complete base map remains authenticated through base_plan.
No experiment is rerun and the original audit failure remains recorded.
"""
import datetime
import json
from pathlib import Path
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]
import qwen35_slot_select_probe_serial_v1 as stage
from qwen35_sampler_runtime_io_v12 import read, ref, once, RetainedLock, process_api
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal, terminal_unit
from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest

assert sys.dont_write_bytecode and __debug__
receipt_ref = ref(Q / 'qwen35-slot-select-probe-r2-root-actual-exit-20261112-v1.json')
receipt = read(receipt_ref)
assert receipt['actual_tool'] == dict(kind='tool_session', session_id=22831, terminal_chunk='a07595', exit_code=0)
assert receipt['runtime_calls'] == 12 and receipt['diagnostic_complete'] and receipt['reconciliation']['passed']
spec = read(receipt['spec'], 64 * 2**20)
overlay = read(spec['history_overlay'], 64 * 2**20)
base = read(overlay['base_plan'], 64 * 2**20)
seed_ref = dict(path=str(Q / 'qwen35-slot-special-value-r2-v2-20261112-spec.json'), sha256='92a857b9715b84104a7981cba6fdf324438437a4df8eb823642ee5f5a91aeb8d')
seed = read(seed_ref, 64 * 2**20)
seed_overlay = read(seed['history_overlay'], 64 * 2**20)
recovered = {path: sha for path, sha in seed['source_sha256'].items() if path not in seed_overlay['source_extension']}
assert len(recovered) == 44532 and all(recovered.get(path) == sha for path, sha in base['source_sha256'].items())
combined = dict(recovered)
for path, sha in overlay['source_extension'].items():
    assert path not in combined or combined[path] == sha
    combined[path] = sha
assert combined == spec['source_sha256']
oldcat = read(spec['current_serial_catalog'])
idle_ref = ref(Q / 'continuation-handoff-20261112-v578-qwen35-slot-select-r2-ended.json')
idle = read(idle_ref)
assert idle['qwen35_slot_select_current']['actual_exit'] == receipt_ref and not idle['running']
assert latest.latest_handoff()['path'] == idle_ref['path']
lock = RetainedLock()
lock.acquire()
try:
    terminal = read(receipt['outer_terminal'])
    p = process_api()
    observations = []
    for _ in range(2):
        p.active_heavy()
        for owner in receipt['owners']:
            p.gone(owner)
        observed = {}
        for saved in terminal['complete_owned_unit_terminals']:
            unit = terminal_unit(saved)
            props = stage.properties(unit)
            verify_current_terminal(saved, props, p)
            observed[unit] = dict(properties=props, recursive_group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
        observations.append(observed)
    assert observations[0] == observations[1]
    root = Path(spec['output_root'])
    stage.audit_results(root, read(ref(root / 'supervisor-result.json')))
    failure_ref = once(Q / 'qwen35-slot-select-probe-r2-root-audit-publication-failure-20261112-v1.json', dict(
        schema='qwen35_slot_select_probe_root_audit_publication_failure_v1', reviewer='/root',
        actual_auditor_tool=dict(kind='tool_session', session_id=88614, terminal_chunk='9e75c0', exit_code=1),
        failed_source=ref(Q / 'audit_qwen35_slot_select_probe_r2_root_20261112_v1.py'),
        reason='v1 wrote actual result and idle handoff, then attempted to repeat the complete base source map in one new catalog row; the unchanged 8 MiB receipt cap rejected publication before file creation.',
        experiment_actual_tool=receipt['actual_tool'], existing_actual_exit=receipt_ref, existing_idle_handoff=idle_ref,
        model_or_device_failure=False, original_receipt_limit_changed=False, device_calls_repeated=False))
    pins = dict(overlay['source_extension'])
    for script in (Path(__file__).resolve(), Q / 'audit_qwen35_slot_select_probe_r2_root_20261112_v1.py', Q / 'ack_qwen35_slot_select_probe_r2_root_20261112_v1.py', Q / 'finalize_qwen35_slot_select_probe_r2_root_20261112_v2.py'):
        row = ref(script)
        assert row['path'] not in pins or pins[row['path']] == row['sha256']
        pins[row['path']] = row['sha256']
    for path, sha in pins.items():
        assert ref(path)['sha256'] == sha
    newcat = dict(oldcat, current_handoff=idle_ref, runs=oldcat['runs'] + [dict(
        terminal=receipt_ref, tool=receipt['actual_tool'], owners=receipt['owners'], units=receipt['units'], source_sha256=pins)])
    catalog_shape(newcat)
    assert newcat['runs'][:-1] == oldcat['runs']
    catalog_ref = once(Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v108.json', newcat)
    finished = once(Q / 'qwen35-slot-select-probe-r2-root-finalization-20261112-v3.json', dict(
        schema='qwen35_slot_select_probe_root_finalization_v3', reviewer='/root', passed=True,
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_exit=receipt_ref,
        prior_audit_failure=failure_ref, original_cap_preserved=True, all_predecessor_rows_preserved=True,
        complete_source_map_reconstructed=True, base_sources=len(base['source_sha256']), recovered_sources=len(recovered),
        metadata_precheck_failure=dict(tool_chunk='a37f80', exit_code=1, source=ref(Q / 'finalize_qwen35_slot_select_probe_r2_root_20261112_v2.py'), reason='Base plan alone omitted29 recovered sources; failed before lock or write. v3 reconstructs full44532 recovered partition from the already authenticated seed.'),
        full_sources=len(combined), new_row_source_extension=len(pins), catalog=catalog_ref,
        catalog_rows=len(newcat['runs']), idle_handoff=idle_ref, independent_raw_validator_repeated=True,
        two_current_service_observations=observations, original_lock_reacquired=True, owners_gone=True,
        physically_drained=True, no_device_rerun=True, full_model_qualified=False, new_speedup_claim=False,
        finalizer_source=ref(Path(__file__).resolve())))
finally:
    lock.release(True)
print(json.dumps(dict(finalization=finished, catalog=catalog_ref, rows=len(newcat['runs']), runtime_calls=receipt['runtime_calls'])))
