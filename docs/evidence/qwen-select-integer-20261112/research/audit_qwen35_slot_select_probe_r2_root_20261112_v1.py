"""Reconcile the actual twelve-call tool terminal and independently reread bytes.

No native imports or experiment launch. Run only after the external tool exits.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]
import qwen35_slot_select_probe_serial_v1 as stage
from qwen35_sampler_runtime_io_v12 import read, ref, once, process_api
from qwen35_sampler_runtime_history_v12 import expected_running
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal, terminal_unit
from prepare_qwen35_sampler_r6_candidate_v1 import catalog_shape

assert sys.dont_write_bytecode and __debug__
parser = argparse.ArgumentParser()
parser.add_argument('--terminal-chunk', required=True)
parser.add_argument('--exit-code', type=int, required=True)
args = parser.parse_args()
actual = dict(kind='tool_session', session_id=22831, terminal_chunk=args.terminal_chunk, exit_code=args.exit_code)
binding_ref = ref(Q / 'qwen35-slot-select-probe-r2-root-binding-20261112-v1.json')
binding = read(binding_ref)
spec = read(binding['spec'], 64 * 2**20)
review = read(binding['review'], 64 * 2**20)
stage.validate(spec, review, binding['spec'])
root = Path(spec['output_root'])
work = root / stage.diagnostic.NAME
admission_ref = ref(root / 'admission.json')
admission = read(admission_ref)
terminal_ref = ref(root / 'outer-terminal.json')
terminal = read(terminal_ref)
running_ref = ref(spec['running_handoff_path'])
running = read(running_ref)
live_ref = ref(Q / 'qwen35-slot-select-probe-r2-root-live-20261112-v1.json')
live = read(live_ref)
owner_ref = ref(root / 'supervisor-owner.json')
owner = read(owner_ref)
assert live['actual_tool'] == dict(kind='tool_session', session_id=22831)
assert admission['outer_owner'] == terminal['outer_owner'] == live['outer_owner']
assert admission['spec'] == binding['spec'] and admission['review'] == binding['review']
assert running == expected_running(read(admission['freshness']['latest_handoff']), running['updated_at'], live['actual_tool'], admission_ref, admission, owner_ref, owner)
ack = read(ref(root / 'running-ack.json'))
assert ack['running']['reference'] == running_ref and ack['running']['value'] == running
assert ack['running']['observation']['path'] == running_ref['path']
assert ack['admission'] == admission_ref and ack['coordinator'] == owner
native_ref = ref(work / 'native-process-exit.json')
native = read(native_ref)
assert native['popen_wait_observed'] and native['self_exit_claimed'] is False
assert native['owner'] == read(ref(work / 'native-owner.json'))
assert native['owner']['ppid'] == owner['process']['pid'] and native['owner']['cgroup'] == owner['process']['cgroup']
owners = [admission['outer_owner'], owner['process'], native['owner']]
report_ref = ref(work / 'report.json')
report = read(report_ref)
assert all(report[k] is False for k in ('qualification_matrix_completed', 'full_model_qualified', 'model_speedup_claim'))
ends = {terminal_unit(t): t for t in terminal['complete_owned_unit_terminals']}
assert set(ends) == set(admission['units'].values())
observations, raw_rows = [], []
summary = None
fd = os.open(R / 'serial-enrollment.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
try:
    assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (66306, 8294421)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p = process_api()
    for _ in range(2):
        p.active_heavy()
        for current in owners:
            p.gone(current)
        observation = {}
        for unit, saved in ends.items():
            props = stage.properties(unit)
            assert p.terminal(props)
            verify_current_terminal(saved, props, p)
            observation[unit] = dict(properties=props, recursive_group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
        observations.append(observation)
    assert observations[0] == observations[1]
    stage.diagnostic.verify_inputs()
    if actual['exit_code'] == 0:
        assert native['actual_returncode'] == 0 and report['complete'] and report['runtime_calls'] == 12 and report['error'] is None
        assert native['parent_observer'] == owner['process']
        result = read(ref(root / 'supervisor-result.json'))
        stage.audit_results(root, result)
        summary = stage.validate_layout(root, stage.diagnostic.NAME, read(ref(work / 'leaf-result.json')), native)
        assert summary == result['layouts'][0]
    else:
        assert terminal['failed'] and not (root / 'supervisor-result.json').exists()
    for row in report['calls']:
        # Construct the oracle directly from independently checked saved input bytes.
        inputs = report['inputs'][row['profile']]
        cache = stage.diagnostic.checked(inputs['cache'], 98304)
        updates = stage.diagnostic.checked(inputs['updates'], 98304)
        assert dict(cache=cache, updates=updates) == stage.diagnostic.input_data(row['profile'])
        slots = struct.unpack('<4q', stage.diagnostic.checked(report['slots'][row['mapping']], 32))
        expected = bytearray(cache)
        width = 98304 // 4
        for source, dest in enumerate(slots):
            if dest != -1:
                assert 0 <= dest < 4
                expected[dest * width:(dest + 1) * width] = updates[source * width:(source + 1) * width]
        output = stage.diagnostic.checked(row['output'], 98304)
        assert len(output) == len(expected) == 98304
        offsets = [i for i, (a, b) in enumerate(zip(output, expected)) if a != b]
        assert b''.join(struct.pack('<I', i) for i in offsets) == stage.diagnostic.checked(row['differences'], 4 * 98304)
        assert row['different_bytes'] == len(offsets) and row['bitwise_equal'] == (not offsets)
        assert row['first16_different_bytes'] == offsets[:16]
        assert row['expected_sha256'] == hashlib.sha256(expected).hexdigest()
        raw_rows.append(dict(arm=row['arm'], profile=row['profile'], mapping=row['mapping'], different_bytes=len(offsets), bitwise_equal=not offsets, output=row['output'], differences=row['differences']))
    reconciliation = stage.reconcile(terminal, running, actual, actual_owner_gone=True, actual_service_drained=True)
    receipt = once(Q / 'qwen35-slot-select-probe-r2-root-actual-exit-20261112-v1.json', dict(
        schema='qwen35_slot_select_probe_root_actual_exit_v1', reviewer='/root', observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_tool=actual, actual_tool_exit_observed=True, actual_outer_exit_observed=True, root_live=live_ref, root_binding=binding_ref,
        spec=binding['spec'], review=binding['review'], admission=admission_ref, outer_terminal=terminal_ref, running_handoff=running_ref,
        reconciliation=reconciliation, owners=owners, owners_gone=True, units=sorted(ends), physically_drained=True,
        two_current_service_observations=observations, original_lock_reacquired_and_released=True, lock_identity=stage.LOCK_IDENTITY,
        native_exit=native_ref, report=report_ref, native_log=ref(work / 'native.log'), diagnostic_complete=report['complete'], summary=summary,
        runtime_calls=report['runtime_calls'], measured_calls=0, calls=raw_rows, failure=report['error'],
        primitive_qualified=False, qualification_matrix_completed=False, model_executed=False, full_model_qualified=False,
        new_speedup_claim=False, auditor_source=ref(Path(__file__).resolve())))
    oldcat_ref = spec['current_serial_catalog']
    catalog = read(oldcat_ref)
    previous_runs = copy.deepcopy(catalog['runs'])
    pins = dict(spec['source_sha256'])
    for script in (Path(__file__).resolve(), Q / 'ack_qwen35_slot_select_probe_r2_root_20261112_v1.py'):
        source = ref(script)
        assert source['path'] not in pins or pins[source['path']] == source['sha256']
        pins[source['path']] = source['sha256']
    catalog['runs'].append(dict(terminal=receipt, tool=actual, owners=owners, units=sorted(ends), source_sha256=pins))
    idle = copy.deepcopy(running)
    idle.update(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous=running_ref, previous_handoff=running_ref,
        running=False, active_serial=None, live_tool_sessions=[], active_heavy_units=[], goal_status='active', this_turn_classification='progress',
        complete_current_serial_catalog=oldcat_ref, complete_current_serial_catalog_next=str(Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v108.json'),
        publication_current_status_stale=True, new_speedup_claim=False)
    idle['qwen35_slot_select_current'] = dict(actual_exit=receipt, status='diagnostic_completed' if reconciliation['passed'] else 'diagnostic_failed',
        runtime_calls=report['runtime_calls'], diagnostic_complete=report['complete'], full_model_qualified=False, new_speedup_claim=False)
    idle['qwen35_package_serial_runtime_v2'] = dict(source_review=ref(Q / 'qwen35-package-serial-runtime-v2-root-source-review-20261112-v1.json'), actual_full_run_pending=True)
    idle['next_serial_work'] = 'Review twelve-call saved raw outputs and actual HLO before any primitive successor; preserve original eight layouts and full model gates. Package readiness fix reviewed; bind finite storage budget before full package run. Mini SIGXCPU source successor remains pending.'
    idle_ref = once(Q / 'continuation-handoff-20261112-v578-qwen35-slot-select-r2-ended.json', idle)
    catalog['current_handoff'] = idle_ref
    catalog_shape(catalog)
    assert catalog['runs'][:-1] == previous_runs
    newcat = once(Q / 'minicpm5-operator-post-r2-serial-workloads-root-catalog-20261112-v108.json', catalog)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
print(json.dumps(dict(actual_exit=receipt, catalog=newcat, handoff=idle_ref, rows=len(catalog['runs']), runtime_calls=report['runtime_calls'], calls=raw_rows)))
