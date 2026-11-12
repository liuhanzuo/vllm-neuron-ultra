"""Root post-tool-exit audit: saved metadata plus current complete drain."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
C = Path('/mnt/nvme/neuron-work/qw-r8-v10-20261112')
O = Path('/mnt/nvme/neuron-work/qwen35-sampler-pair-r8-v10-20261112')
OUT = Q / 'qwen35-sampler-r8-root-actual-exit-20261112-v1.json'


def read(path):
    path = Path(path)
    assert path.suffix in ('.json', '.log', '.py') and path.stat().st_size < 32 * 2**20
    return json.loads(path.read_text())


def ref(path):
    path = Path(path)
    assert path.suffix in ('.json', '.log', '.py') and path.stat().st_size < 32 * 2**20
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not OUT.exists()
    a = read(C / 'admission.json')
    spec = read(a['spec']['path'])
    terminal = read(C / 'outer-terminal.json')
    co = read(C / 'coordinator-owner.json')
    assert ref(a['spec']['path']) == a['spec'] and ref(a['review']['path']) == a['review']
    assert terminal['failed'] and terminal['whole_owned_scope_drained']
    assert terminal['admission'] == ref(C / 'admission.json')
    owners = [a['outer_owner'], co['process']]
    completed = []
    inputs = {str(p): ref(p) for p in C.rglob('*.json')}
    inputs.update({str(p): ref(p) for p in O.rglob('*.json')})
    inputs.update({str(p): ref(p) for p in C.rglob('*.log')})
    service_ends = {x['server_owner']['unit']: x for x in terminal['complete_owned_unit_terminals']}
    for child_file in sorted(C.glob('*/child-exit.json')):
        child = read(child_file)
        stage = child_file.parent.name
        assert child['actual_popen_wait_observed'] and not child['self_exit_claimed']
        assert child['phase_spec'] == ref(child_file.parent / 'phase-spec.json')
        assert child['supervisor'] == read(child_file.parent / 'supervisor-owner.json')
        assert child['child_bootstrap_owner'] == read(child_file.parent / 'child-bootstrap-owner.json')
        expected_exit = -9 if stage == 'baseline-numeric' else 0
        assert child['child_returncode'] == expected_exit
        owners += [child['supervisor']['process'], child['child_bootstrap_owner']]
        completed.append(dict(stage=stage, actual_child_returncode=expected_exit,
                              child_exit=ref(child_file), leaf_result=(
                                  ref(child_file.parent / 'leaf-result.json')
                                  if (child_file.parent / 'leaf-result.json').exists() else None)))
    assert len(owners) == 8 and len({(x['pid'], x['start_ticks']) for x in owners}) == 8
    assert len(completed) == 3 and len(service_ends) == 4 and len(spec['limits']) > 0
    execution = read(O / 'baseline/model/state_copy-execution.json')
    assert execution['returncode'] == 0 and execution['fresh_execution'] is True
    assert execution['actual_child_exit'] == read(C / 'baseline-state_copy/child-exit.json')
    assert not (O / 'baseline/model/real-logits-validation.json').exists()
    assert not (C / 'baseline-numeric/leaf-result.json').exists()
    native = read(C / 'native-preflight.json')
    assert native['passed'] and len(native['actual_native_serialization_cases']) == 42
    assert all(x['passed'] for x in native['actual_native_serialization_cases'])
    log = (C / 'baseline-numeric/phase.log').read_text()
    assert 'ValueError: actual storage floor/reservation failed' in log
    assert 'qwen35_sampler_runtime_service_v10.py' in log and 'supervise' in log
    assert read(C / 'baseline-numeric/child-exit.json')['stop_requested'] is False
    lock = R / 'serial-enrollment.lock'
    fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    samples = []
    try:
        s = os.fstat(fd)
        assert (s.st_dev, s.st_ino) == (66306, 8294421)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path[:0] = [str(Q), str(R)]
        import minicpm5_layer0_execution_protocol_v4 as p
        from qwen35_sampler_runtime_service_v10 import properties
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
                    assert saved['cleanup_complete'] and saved['exit_observed'] and saved['remaining_processes'] == []
                    if value['LoadState'] == 'not-found':
                        assert value['InvocationID'] == ''
                    else:
                        assert value['InvocationID'] == saved['server_owner']['invocation_id']
                        assert value['ExecMainPID'] == str(saved['process_owner']['pid'])
                        assert value['ExecStart'] == saved['properties']['ExecStart']
                        assert {k: (int(value[k]) if k != 'Result' else value[k])
                                for k in ('ExecMainCode', 'ExecMainStatus', 'Result')} == saved['systemd_exit_status']
                else:
                    assert value['LoadState'] == 'not-found' and value['InvocationID'] == ''
                current[unit] = dict(properties=value, group=p.empty_group(group))
            samples.append(current)
        assert samples[0] == samples[1]
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    value = dict(schema='qwen35_sampler_r8_root_actual_exit_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_tool=dict(kind='tool_session', session_id=43414, terminal_chunk='cb7e3c', exit_code=1),
        actual_exit_code=1, actual_tool_exit_observed=True, admission=ref(C / 'admission.json'),
        spec=a['spec'], review=a['review'], outer_terminal=ref(C / 'outer-terminal.json'), inputs=inputs,
        lock_identity=dict(device=66306, inode=8294421), original_lock_observed_free=True,
        original_lock_nonblocking_acquired_and_released=True, owners=owners, owners_gone=True,
        all_declared_units=sorted(a['units'].values()), physically_drained=True,
        actual_service_terminations=[dict(unit=k, actual_exit=v['systemd_exit_status']) for k, v in service_ends.items()],
        two_complete_current_service_observations=samples, completed_stages=completed,
        native_preflight42_passed=True, baseline_state_copy_passed=True,
        numeric_phase_completed=False, root_raw_model_payloads_read=False,
        failure=terminal['failure'], failure_detail=dict(phase='baseline_numeric',
            type='ValueError', cause='supervisor storage-floor guard rejected current free disk',
            actual_child_returncode=-9, supervisor_cleanup_SIGKILL_path_present=True,
            exact_failing_filesystem_and_transient_free_bytes_not_recorded=True,
            OOM_not_established=True,
            phase_log=ref(C / 'baseline-numeric/phase.log'),
            API_never_started=True, short_IPC_actual_verification_pending=True),
        quality96_HTTP_and_sampler_capture_not_completed=True, full_model_qualified=False,
        new_speedup_claim=False, auditor_source=ref(Path(__file__).resolve()))
    with OUT.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(result=ref(OUT), owners=len(owners), units=len(a['units']),
                          phases=len(completed), storage_guard_failure=True, actual_exit=1)))


if __name__ == '__main__':
    main()
