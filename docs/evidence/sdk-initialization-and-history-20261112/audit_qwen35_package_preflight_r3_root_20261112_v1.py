"""Root audit after observing the actual external packaged-preflight exit."""
import argparse
import datetime
import fcntl
import os
from pathlib import Path
import sys

R = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = R / 'research'
sys.path[:0] = [str(Q), str(R)]
import qwen35_package_preflight_serial_v4 as stage
from qwen35_sampler_runtime_io_v12 import read, ref, once, process_api
from audit_qwen35_sampler_r10_terminal_root_20261112_v2 import verify_current_terminal, terminal_unit
from qwen35_package_tokenizer_checks_v3 import compare_encodings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session-id', type=int, required=True)
    parser.add_argument('--terminal-chunk', required=True)
    parser.add_argument('--exit-code', type=int, required=True)
    args = parser.parse_args()
    assert sys.dont_write_bytecode and __debug__ and args.session_id > 0
    actual = dict(kind='tool_session', session_id=args.session_id,
                  terminal_chunk=args.terminal_chunk, exit_code=args.exit_code)
    binding = read(ref(Q / 'qwen35-package-preflight-r3-20261112-root-binding-v4.json'))
    spec = read(binding['spec'], 64 * 2**20)
    review = read(binding['review'], 64 * 2**20)
    stage.validate(spec, review, binding['spec'])
    root = Path(spec['output_root'])
    admission_ref = ref(root / 'admission.json')
    admission = read(admission_ref)
    terminal_ref = ref(root / 'outer-terminal.json')
    terminal = read(terminal_ref)
    running_ref = ref(spec['running_handoff_path'])
    running = read(running_ref)
    live_ref = ref(Q / 'qwen35-package-preflight-r3-root-live-20261112-v1.json')
    live = read(live_ref)
    assert live['actual_tool'] == dict(kind='tool_session', session_id=args.session_id)
    assert admission['outer_owner'] == live['outer_owner'] == terminal['outer_owner']
    assert admission['spec'] == binding['spec'] and admission['review'] == binding['review']
    owners = [admission['outer_owner']]
    supervisor = read(ref(root / 'supervisor-owner.json'))
    supervisor_ref = ref(root / 'supervisor-owner.json')
    assert running == stage.expected_running(read(admission['freshness']['latest_handoff']),
        running['updated_at'], dict(kind='tool_session', session_id=args.session_id),
        admission_ref, admission, supervisor_ref, supervisor)
    ack = read(ref(root / 'running-ack.json'))
    assert ack['running']['reference'] == running_ref and ack['running']['value'] == running
    assert ack['admission'] == admission_ref and ack['coordinator'] == supervisor
    owners.append(supervisor['process'])
    native_exit = None
    if (root / 'native-process-exit.json').exists():
        native_exit = read(ref(root / 'native-process-exit.json'))
        assert native_exit['popen_wait_observed'] and native_exit['self_exit_claimed'] is False
        if native_exit['owner'] is not None:
            owners.append(native_exit['owner'])
    success = args.exit_code == 0
    science = dict(native_preflight_passed=False, model_constructed=False,
                   weights_read=False, compiled=False, device_graph_executed=False)
    if success:
        assert native_exit and native_exit['actual_returncode'] == 0
        summary = read(ref(root / 'supervisor-result.json'))
        result = read(summary['native_result'], 16 * 2**20)
        assert summary['passed'] is summary['complete'] is True
        assert summary['admission'] == admission_ref and summary['owner'] == supervisor
        assert result['passed'] is result['complete'] is True
        assert result['source_bundle'] == stage.BUNDLE
        worker_input = read({k: admission['worker_input'][k] for k in ('path', 'sha256')})
        assert result['root_binding'] == worker_input['binding']
        assert result['full_model_qualification'] is result['inherited_performance_claim'] is False
        assert result['parent']['pid'] == native_exit['owner']['pid']
        assert str(result['parent']['start_ticks']) == native_exit['owner']['start_ticks']
        assert result['parent']['ppid'] == supervisor['process']['pid']
        assert result['child']['ppid'] == result['parent']['pid']
        assert result['child']['pid'] != result['parent']['pid']
        assert result['child']['origins'] == result['parent']['origins']
        assert result['child']['tokenization'] == result['parent']['tokenization']
        assert result['child']['configuration'] == result['parent']['configuration'] or (
            result['child']['configuration']['configuration_sha256'] ==
            result['parent']['configuration']['configuration_sha256'])
        for label in ('parent', 'child'):
            item = result[label]
            assert item['native_imports_executed'] and item['registration_idempotent']
            assert all(item[k] is False for k in ('model_constructed', 'weights_read', 'device_graph_executed', 'compiled'))
            assert item['split'] == dict(cpu=True, dtype='bfloat16', elements=6144, transpose=True, equal_coordinates=True)
            probe = item['optional_cpu_probe']
            assert probe['supports_sve'] is False and probe['original_numpy_fallback_observed']
            assert probe['optional_probe_subprocess_remains_forbidden'] and probe['fabricated_command_output'] is False
            assert len(probe['events']) == 1 and probe['events'][0]['denied_before_exec']
            assert probe['events'][0]['subprocess_started'] is False
            assert probe['events'][0]['source']['sha256'] == stage.worker.NUMPY_SVE_SHA256
            tokenization = item['tokenization']
            assert tokenization['passed'] and tokenization['native_tokenizer_executed']
            assert tokenization['comparison'] == compare_encodings(tokenization['original'], tokenization['view'])
            config = item['configuration']
            assert all(config[k] for k in ('root_save_load_equal', 'json_constructor_equal', 'saved_body_admission', 'changed_saved_body_rejected'))
            assert all(item['sampler'][k] for k in ('standard_preserved', 'fast_per_instance_reference', 'same_source_configuration_compiler_environment'))
        owners.append({k: result['child'][k] for k in ('pid', 'ppid', 'start_ticks')})
        science.update(native_preflight_passed=True, actual_spawn_passed=True,
                       tokenization_comparison=result['parent']['tokenization']['comparison'],
                       configuration_sha256=result['parent']['configuration']['configuration_sha256'],
                       result=summary['native_result'])
    helper_reports = {}
    for label in ('parent', 'spawn'):
        helper_path = root / 'native' / (label + '-pjrt-helper.json')
        if not helper_path.exists():
            assert not success, 'Successful parent/spawn requires both actual helper receipts'
            continue
        helper = read(ref(helper_path))
        assert helper['label'] == label and helper['command'] == [stage.worker.PJRT_HELPER]
        assert helper['sdk_command'] == ['libneuronpjrt-path']
        assert helper['self_exit_claimed'] is helper['fabricated_command_output'] is False
        assert helper['popen_wait_observed'] and type(helper['actual_returncode']) is int
        assert helper['same_owned_cgroup_required'] and helper['whole_recursive_cgroup_drain_pending']
        assert helper['environment'] == stage.worker.helper_environment(str(root / 'native'))
        assert helper['library'] == stage.worker.library_identity()
        owner = helper['owner']
        if owner is not None:
            assert owner['ppid'] == helper['parent_pid']
            assert owner['cgroup'] == native_exit['owner']['cgroup']
            owners.append(owner)
        if helper['failure'] is None:
            assert owner is not None and helper['actual_returncode'] == 0
            assert helper['stdout'] == stage.worker.PJRT_LIBRARY + '\n' and helper['stderr'] == ''
            assert owner['argv'] == ['/home/ubuntu/venvs/vllm_neuron/bin/python3', stage.worker.PJRT_HELPER]
            assert owner['executable'] == str(Path(stage.PYTHON).resolve())
        if success:
            assert helper == result['parent' if label == 'parent' else 'child']['required_pjrt_helper']
            assert helper['failure'] is None
            assert helper['parent_pid'] == result['parent' if label == 'parent' else 'child']['pid']
        helper_reports[label] = dict(reference=ref(helper_path), value=helper)
    ends = {terminal_unit(t): t for t in terminal['complete_owned_unit_terminals']}
    assert set(ends) == set(admission['units'].values())
    observations = []
    fd = os.open(R / 'serial-enrollment.lock', os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (66306, 8294421)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = process_api()
        for _ in range(2):
            p.active_heavy()
            for owner in owners:
                p.gone(dict(owner, start_ticks=str(owner['start_ticks'])))
            snapshot = {}
            for unit, saved in ends.items():
                props = stage.properties(unit)
                assert p.terminal(props)
                verify_current_terminal(saved, props, p)
                if success:
                    assert saved['systemd_exit_status'] == dict(ExecMainCode=1, ExecMainStatus=0, Result='success')
                snapshot[unit] = dict(properties=props, group=p.empty_group(p.GROUP_PREFIX + unit + '.service'))
            observations.append(snapshot)
        assert observations[0] == observations[1]
        reconciliation = stage.reconcile(terminal, running, actual, actual_owner_gone=True, actual_service_drained=True)
        assert reconciliation['passed'] is success
        inputs = {str(path): ref(path, 16 * 2**20) for path in root.rglob('*')
                  if not path.is_symlink() and path.is_file() and path.suffix in ('.json', '.log')}
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    value = dict(schema='qwen35_package_preflight_root_actual_exit_v1', reviewer='/root',
        observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=actual,
        actual_tool_exit_observed=True, actual_outer_exit_observed=True, root_live=live_ref,
        spec=binding['spec'], review=binding['review'], admission=admission_ref,
        outer_terminal=terminal_ref, running_handoff=running_ref, reconciliation=reconciliation,
        owners=owners, owners_gone=True, units=sorted(ends), physically_drained=True,
        two_current_service_observations=observations, original_lock_reacquired_and_released=True,
        lock_identity=stage.LOCK_IDENTITY, native_preflight_passed=success, scientific=science,
        failure=terminal['failure'], inputs=inputs, required_helper_reports=helper_reports, full_model_qualified=False,
        new_speedup_claim=False, auditor_source=ref(Path(__file__).resolve()))
    output = once(Q / 'qwen35-package-preflight-r3-root-actual-exit-20261112-v1.json', value)
    print(dict(result=output, native_preflight_passed=success, owners=len(owners), full_model_qualified=False))


if __name__ == '__main__':
    main()
