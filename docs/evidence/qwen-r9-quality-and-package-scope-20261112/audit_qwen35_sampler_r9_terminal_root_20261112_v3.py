"""Independent saved-result/drain audit after root observes session 78094 exit.

No model/native imports or experiment launch. Actual raw-vector and sampler
replays are authenticated against their native producers and external waits.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
C = Path('/mnt/nvme/neuron-work/qw-r9-v11-20261112')
O = Path('/mnt/nvme/neuron-work/qwen35-sampler-pair-r9-v11-20261112')
DEST = Q / 'qwen35-sampler-r9-root-actual-exit-20261112-v1.json'
SUCCESS = dict(ExecMainCode=1, ExecMainStatus=0, Result='success')
sys.path[:0] = [str(Q), str(R)]


def raw(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_size <= 64 * 2**20
        with os.fdopen(os.dup(fd), 'rb') as stream:
            data = stream.read(64 * 2**20 + 1)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        assert identity(before) == identity(after) == identity(path.lstat())
        assert len(data) == before.st_size
        return data
    finally:
        os.close(fd)


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(raw(path)).hexdigest())


def read(path):
    return json.loads(raw(path))


def bound(reference):
    assert ref(reference['path']) == {k: reference[k] for k in ('path', 'sha256')}
    return read(reference['path'])


def science(completed, successful, owners):
    from qwen35_sampler_full_checks_v1 import original_validators, check_model_metadata, check_quality_pair
    from qwen35_sampler_pair_protocol_v1 import (BASE, SAMPLER_GATES, expected_case,
        host_identity, check_capture_accounting, compare_full_http)
    from qwen35_sampler_pair_driver_v1 import (MODEL_FILES, authenticated, root_observation,
        attached_worker, cleaned_server, check_initial_pair)
    validators = original_validators()
    contract = read(O / 'contract.json')
    run_id = contract['run_id']
    case = expected_case()
    assert contract['case'] == case
    summary = dict(model_phases={}, vector_reproductions={}, paired_http=None)

    def native(arm, phase, path):
        stage = C / (arm + '-' + phase)
        result = read(path)
        authenticated(result, run_id, arm=arm, phase=phase)
        child = read(stage / 'child-exit.json')
        leaf = read(stage / 'leaf-result.json')
        assert child['child_returncode'] == 0 and child['stop_requested'] is False
        assert result['actual_child_exit'] == child
        assert result['actual_service_terminal'] == read(stage / 'root-terminal.json')
        assert result['actual_service_terminal']['systemd_exit_status'] == SUCCESS
        assert leaf['actual_process'] == child['child_bootstrap_owner']
        for key, value in leaf['result'].items():
            assert result[key] == value
        return result

    for arm in ('baseline', 'candidate'):
        rows = []
        for phase, filename in MODEL_FILES.items():
            if arm + '-' + phase not in completed or not (O / arm / 'model' / (phase + '-execution.json')).exists():
                assert not successful
                continue
            native(arm, phase, O / arm / 'model' / (phase + '-execution.json'))
            check_model_metadata(read(O / arm / 'model' / filename), phase, validators)
            rows.append(phase)
        summary['model_phases'][arm] = rows
        if (O / arm / 'model/vectors203.json').exists():
            vectors = native(arm, 'vectors203', O / arm / 'model/vectors203.json')
            assert vectors['complete'] is True and vectors['frames'] == vectors['identical_vectors'] == 203
            assert vectors['phases'] == [dict(phase=p, frames=n, identical_vectors=n)
                                        for p, n in [('numeric', 32), ('cold', 84), ('long', 87)]]
            summary['vector_reproductions'][arm] = 203
        else:
            assert not successful
    preflight = read(C / 'native-preflight.json')
    child = read(C / 'shared-native_preflight/child-exit.json')
    leaf = read(C / 'shared-native_preflight/leaf-result.json')
    assert preflight['actual_child_exit'] == child and child['child_returncode'] == 0
    assert preflight['actual_service_terminal']['systemd_exit_status'] == SUCCESS
    assert preflight['run_id'] == read(C / 'admission.json')['attempt_id']
    assert leaf['actual_process'] == child['child_bootstrap_owner']
    assert all(preflight[k] == v for k, v in leaf['result'].items())
    assert preflight['passed'] and preflight['actual_rng_unchanged_by_serialization']
    assert len(preflight['actual_native_serialization_cases']) == 42
    assert all(x['passed'] for x in preflight['actual_native_serialization_cases'])
    assert len(preflight['worker_import_pickle']) == 4 and all(x['passed'] for x in preflight['worker_import_pickle'])
    assert preflight['overlapping_layout_negative_control']['passed'] is True
    assert preflight['general_overlapping_layout_support_claimed'] is False
    assert preflight['rejection_controls']['passed'] is True
    assert [x['control'] for x in preflight['rejection_controls']['controls']] == SAMPLER_GATES['controls']
    assert all(x['passed'] for x in preflight['rejection_controls']['controls'])
    summary['native_preflight42_passed'] = True
    if not successful:
        return summary

    host_sources = contract['host_source_sha256']
    root_owner = contract['root_admission']['root_owner']
    captures = {}
    for arm in ('baseline', 'candidate'):
        capture = read(O / arm / 'capture/packets/capture-receipt.json')
        check_capture_accounting(capture)
        diagnostic = read(O / arm / 'diagnostic-http.json')
        assert diagnostic['complete'] and diagnostic['requests'] == 5 and diagnostic['output_tokens'] == 160
        assert diagnostic['sampling'] == dict(temperature=.7, top_p=.9, top_k=40)
        assert diagnostic['timing_eligible'] is False
        assert [len(wave['requests']) for wave in diagnostic['waves']] == [1, 4]
        assert all(row['usage']['prompt_tokens'] == 128 and row['usage']['completion_tokens'] == 32
                   for wave in diagnostic['waves'] for row in wave['requests'])
        replay = native(arm, 'sampler_replay', O / arm / 'sampler-replay.json')
        assert replay['passed'] and replay['all_own_tokens_and_rng_reproduced'] and replay['all_distribution_gates_passed']
        assert replay['packets'] == capture['packets'] == len(capture['files']) == len(replay['proofs'])
        assert replay['rows'] == capture['captured_rows'] == sum(x['rows'] for x in replay['proofs'])
        assert replay['fast_path_rows'] == sum(x['fast_rows'] for x in replay['proofs']) > 0
        assert sum(x['bytes'] for x in capture['files']) == capture['total_packet_bytes']
        for index, (item, proof) in enumerate(zip(capture['files'], replay['proofs'])):
            assert item['file'] == 'capture-%04d.bin' % index
            packet = raw(O / arm / 'capture/packets' / item['file'])
            assert len(packet) == item['bytes'] and hashlib.sha256(packet).hexdigest() == item['sha256']
            assert proof['passed'] and proof['own_token_and_rng_reproduced'] and proof['vocabulary'] == 248320
            assert proof['arm'] == arm and proof['call_id'] == item['call_id'] and proof['rows'] == item['rows']
            for name in ('total_variation_max', 'max_probability_error'):
                assert math.isfinite(proof[name]) and 0 <= proof[name] <= 1e-6
        captures[arm] = dict(packets=capture['packets'], rows=capture['captured_rows'])
        for phase, host_phase in [('capture', 'correctness_capture'), ('quality', 'measured'), ('http', 'measured')]:
            directory = O / arm / phase
            live = read(directory / 'server-live-root-audit.json')
            terminal = read(directory / 'server-terminal-root-audit.json')
            attached = read(directory / 'worker-attachment.json')
            identity = host_identity(arm, host_phase, host_sources)
            root_observation(live, run_id, root_owner, host_sources)
            root_observation(terminal, run_id, root_owner, host_sources)
            attached_worker(attached, live, identity)
            cleaned_server(terminal, live)
            assert terminal['actual_child_exit'] == read(C / (arm + '-' + phase) / 'child-exit.json')
            assert terminal['systemd_exit_status'] == SUCCESS
            owners.extend([live['api_process_owner'], live['worker_owner']])
        smoke = read(O / arm / 'quality/smoke.json')
        assert smoke['passed'] and smoke['all_identical_rows_match']
        assert all(len(smoke[k]) == 4 for k in ('serial', 'concurrent', 'identical_prompt_outputs'))
    controls = native('shared', 'sampler_controls', O / 'sampler-controls.json')
    assert controls['passed'] and [x['control'] for x in controls['controls']] == SAMPLER_GATES['controls']
    assert len(controls['controls']) == 13 and all(x['passed'] for x in controls['controls'])
    quality = check_quality_pair(read(O / 'baseline/quality/quality96.json'), read(O / 'candidate/quality/quality96.json'),
        read(BASE / 'quality96.json'), read(R / 'quality-suite.json'), ref(R / 'quality-suite.json')['sha256'], validators)
    assert quality == read(O / 'quality-pair.json')
    check_initial_pair(read(O / 'baseline/http/initial120.json'), read(O / 'candidate/http/initial120.json'))
    left, right = (read(O / arm / 'http/workloads.json') for arm in ('baseline', 'candidate'))
    paired = compare_full_http(left, right, case, left['inputs'], read(O / 'baseline/http/host.json'), read(O / 'candidate/http/host.json'))
    driver = read(O / 'driver-result.json')
    assert driver['run_id'] == run_id and driver['execution_finished'] and driver['gates_checked_by_driver']
    assert driver['paired_http'] == paired
    assert driver['full_model_qualified'] is False and driver['independent_final_audit_required'] is True
    coordinator = read(C / 'coordinator-result.json')
    assert coordinator['driver_result'] == driver and coordinator['all_model_and_serving_phases_completed'] is True
    summary.update(captures=captures, sampler_controls=13, quality=quality, paired_http=paired)
    return summary


def canonical_process_owner(owner):
    # proc() represents start ticks as text; live API observations use integers.
    # Preserve the original observation separately and compare the same domain.
    ticks = owner['start_ticks']
    assert type(ticks) in (str, int) and str(ticks).isdigit() and int(ticks) > 0
    assert type(owner['pid']) is int and owner['pid'] > 0
    return {**owner, 'start_ticks': str(ticks)}


def terminal_unit(saved):
    if saved.get('server_owner') is not None:
        assert saved['process_owner'] is not None and saved.get('unacknowledged_terminal') is None
        return saved['server_owner']['unit']
    if saved.get('unacknowledged_terminal') is not None:
        assert saved.get('process_owner') is None
        unack = saved['unacknowledged_terminal']
        assert unack['captured_live_owner'] is False and unack['start_ticks_observed'] is False
        return unack['unit']
    assert saved.get('never_loaded') is True and saved['exit_observed'] is False
    assert saved['actual_process_owner_observed'] is False
    assert saved.get('process_owner') is None and saved.get('systemd_exit_status') is None
    return saved['unit']


def verify_current_terminal(saved, value, process_api):
    assert saved['cleanup_complete'] is True and saved['remaining_processes'] == []
    if saved.get('never_loaded'):
        assert saved['exit_observed'] is False
        assert value['LoadState'] == 'not-found' and value['InvocationID'] == ''
        return
    assert saved['exit_observed'] is True
    if saved.get('server_owner') is not None:
        invocation = saved['server_owner']['invocation_id']
        pid = saved['process_owner']['pid']
    else:
        unack = saved['unacknowledged_terminal']
        invocation, pid = unack['invocation_id'], unack['exec_main_pid']
        assert type(pid) is int and pid >= 0
        assert pid == 0 or process_api.proc(pid) is None
    if value['LoadState'] == 'not-found':
        assert value['InvocationID'] == ''
    else:
        assert value['InvocationID'] == invocation and value['ExecMainPID'] == str(pid)
        assert value['ExecStart'] == saved['properties']['ExecStart']
        assert {k: (int(value[k]) if k != 'Result' else value[k])
                for k in ('ExecMainCode', 'ExecMainStatus', 'Result')} == saved['systemd_exit_status']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-chunk', required=True)
    parser.add_argument('--exit-code', required=True, type=int)
    args = parser.parse_args()
    assert __debug__ and not DEST.exists()
    a = read(C / 'admission.json')
    spec, review = bound(a['spec']), bound(a['review'])
    terminal = read(C / 'outer-terminal.json')
    co = read(C / 'coordinator-owner.json')
    assert terminal['admission'] == ref(C / 'admission.json') and terminal['whole_owned_scope_drained'] is True
    assert terminal['outer_owner'] == a['outer_owner']
    assert terminal['original_lock_identity'] == dict(device=66306, inode=8294421)
    successful = args.exit_code == 0
    if successful:
        assert terminal['failed'] is False and terminal['failure'] is None and terminal['result_present'] is True
    owners = [a['outer_owner'], co['process']]
    service_ends = {terminal_unit(x): x for x in terminal['complete_owned_unit_terminals']}
    assert len(service_ends) == len(terminal['complete_owned_unit_terminals'])
    assert set(service_ends) <= set(a['units'].values())
    completed, stages = set(), []
    for stage in sorted(d for d in C.iterdir() if d.is_dir() and (d / 'phase-spec.json').exists()):
        phase = read(stage / 'phase-spec.json')
        assert phase['admission'] == ref(C / 'admission.json') and phase['coordinator'] == co
        for name, key in [('supervisor-owner.json', 'process'), ('child-bootstrap-owner.json', None)]:
            if (stage / name).exists():
                owner = read(stage / name)
                owners.append(owner[key] if key else owner)
        if not (stage / 'child-exit.json').exists():
            assert not successful
            continue
        child = read(stage / 'child-exit.json')
        assert child['phase_spec'] == ref(stage / 'phase-spec.json')
        assert child['supervisor'] == read(stage / 'supervisor-owner.json')
        assert child['child_bootstrap_owner'] == read(stage / 'child-bootstrap-owner.json')
        assert child['actual_popen_wait_observed'] is True and child['self_exit_claimed'] is False
        api = phase['phase'] in ('capture', 'quality', 'http')
        if successful:
            assert child['child_returncode'] in ((0, -15) if api else (0,))
            assert child['stop_requested'] is api
        leaf_ref = None
        if (stage / 'leaf-result.json').exists():
            leaf = read(stage / 'leaf-result.json')
            assert leaf['phase_spec'] == ref(stage / 'phase-spec.json')
            assert leaf['actual_process'] == child['child_bootstrap_owner']
            if successful:
                assert child['child_returncode'] == 0
            leaf_ref = ref(stage / 'leaf-result.json')
            if child['child_returncode'] == 0:
                completed.add(stage.name)
        stages.append(dict(stage=stage.name, actual_child_returncode=child['child_returncode'],
                           child_exit=ref(stage / 'child-exit.json'), leaf_result=leaf_ref))
    if successful:
        from qwen35_sampler_runtime_contract_v11 import PHASES
        assert len(stages) == len(PHASES) == 20 and len(service_ends) == 21
        assert all(x['systemd_exit_status'] == SUCCESS for x in service_ends.values())
    for arm in ('baseline', 'candidate'):
        for phase in ('capture', 'quality', 'http'):
            live_path = O / arm / phase / 'server-live-root-audit.json'
            if live_path.exists():
                live = read(live_path)
                owners.extend([live['api_process_owner'], live['worker_owner']])
    fd = os.open(R / 'serial-enrollment.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    samples = []
    try:
        st = os.fstat(fd)
        assert (st.st_dev, st.st_ino) == (66306, 8294421)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scientific = science(completed, successful, owners)
        # Preserve every process incarnation, including API exec with changed argv.
        owner_observations = list(owners)
        unique = {}
        for original in owners:
            owner = canonical_process_owner(original)
            unique[(owner['pid'], owner['start_ticks'], tuple(owner['argv']))] = owner
        owners = list(unique.values())
        import minicpm5_layer0_execution_protocol_v4 as p
        from qwen35_sampler_runtime_service_v11 import properties
        for _ in range(2):
            p.active_heavy()
            for owner in owners:
                p.gone(owner)
            current = {}
            for unit in sorted(a['units'].values()):
                value = properties(unit)
                assert p.terminal(value)
                group = p.GROUP_PREFIX + unit + '.service'
                assert value['ControlGroup'] in ('', group)
                if unit in service_ends:
                    saved = service_ends[unit]
                    verify_current_terminal(saved, value, p)
                else:
                    assert value['LoadState'] == 'not-found' and value['InvocationID'] == ''
                current[unit] = dict(properties=value, group=p.empty_group(group))
            samples.append(current)
        assert samples[0] == samples[1]
        inputs = {str(path): ref(path) for base in (C, O) for pattern in ('*.json', '*.log') for path in base.rglob(pattern)}
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    value = dict(schema='qwen35_sampler_r9_root_actual_exit_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_tool=dict(kind='tool_session', session_id=78094, terminal_chunk=args.terminal_chunk, exit_code=args.exit_code),
        actual_exit_code=args.exit_code, actual_tool_exit_observed=True, admission=ref(C / 'admission.json'),
        spec=a['spec'], review=a['review'], outer_terminal=ref(C / 'outer-terminal.json'), inputs=inputs,
        owner_observations=owner_observations, owners=owners, owners_gone=True, all_declared_units=sorted(a['units'].values()), physically_drained=True,
        actual_service_terminations=[dict(unit=k, actual_exit=v.get('systemd_exit_status'),
            exit_observed=v['exit_observed'], never_loaded=v.get('never_loaded', False),
            unacknowledged_terminal=v.get('unacknowledged_terminal')) for k, v in service_ends.items()],
        two_complete_current_service_observations=samples, completed_stages=stages, scientific=scientific,
        lock_identity=dict(device=66306, inode=8294421), original_lock_observed_free=True,
        original_lock_nonblocking_acquired_and_released=True, failure=terminal['failure'],
        full_model_qualified=successful, paired_throughput_qualified=successful,
        new_speedup_claim=False, auditor_source=ref(Path(__file__).resolve()))
    with DEST.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(result=ref(DEST), actual_exit=args.exit_code, phases=len(stages),
                          full_model_qualified=successful, owners=len(owners), units=len(a['units']))))


if __name__ == '__main__':
    main()
