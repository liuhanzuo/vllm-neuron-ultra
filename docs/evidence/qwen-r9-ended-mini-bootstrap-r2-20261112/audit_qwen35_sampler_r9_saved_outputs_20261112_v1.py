"""Post-failure saved-data audit; never rewrites or upgrades the failed run."""
import datetime
import json
import math
from pathlib import Path
from audit_qwen35_sampler_r9_terminal_root_20261112_v3 import R, Q, C, O, raw, ref, read, bound

def audit_saved_outputs(completed, owners):
    successful = False
    from qwen35_sampler_full_checks_v1 import original_validators, check_model_metadata, check_quality_pair
    from qwen35_sampler_pair_protocol_v1 import (BASE, SAMPLER_GATES, expected_case,
        host_identity, check_capture_accounting, compare_full_http)
    from qwen35_sampler_pair_driver_v1 import (MODEL_FILES, authenticated, root_observation,
        attached_worker, cleaned_server)
    from qwen35_sampler_pair_driver_v2 import check_initial_pair
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
    assert not (O / 'driver-result.json').exists()
    assert not (C / 'coordinator-result.json').exists()
    summary['initial_greedy_pairs'] = 19
    summary['initial_usage_optional_field_fix_applied_only_in_auditor'] = True
    summary['original_outer_exit_code'] = 1
    summary['original_driver_completed'] = False
    summary['original_final_artifact_reuse_check_ran'] = False
    summary['full_model_qualified'] = False
    summary['paired_throughput_qualified'] = False
    summary['timings_are_independently_checked_observations_only'] = True
    summary.update(captures=captures, sampler_controls=13, quality=quality, paired_http=paired)
    return summary


def main():
    ended_path = Q / 'qwen35-sampler-r9-root-actual-exit-20261112-v1.json'
    assert ref(ended_path)['sha256'] == 'bec84cd8758f18215b452dd606816faac021a1fa846270991865e4ed0cfb9975'
    ended = read(ended_path)
    assert ended['actual_tool'] == dict(kind='tool_session', session_id=78094, terminal_chunk='e05d64', exit_code=1)
    assert ended['physically_drained'] and ended['owners_gone']
    for reference in ended['inputs'].values():
        assert ref(reference['path']) == reference
    completed = {row['stage'] for row in ended['completed_stages']
                 if row['actual_child_returncode'] == 0 and row['leaf_result'] is not None}
    owners = []
    scientific = audit_saved_outputs(completed, owners)
    assert len(owners) == 12
    value = dict(schema='qwen35_sampler_r9_saved_outputs_after_failed_summary_v1',
        reviewer='/root', observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_failed_exit=ref(ended_path), scientific=scientific,
        original_outputs_preserved=True, no_experiment_rerun=True,
        source_correction=ref(Q / 'qwen35_sampler_pair_driver_v2.py'),
        auditor_source=ref(Path(__file__).resolve()),
        full_model_qualified=False, new_qualified_speedup=False,
        limitations=['Original outer exited 1; driver/coordinator result and final artifact reuse check absent.',
                     'Saved HTTP rates are descriptive; sequential trials are not a statistical significance test.',
                     'Packaged Qwen3.5 source has not inherited this research run qualification.'])
    dest=Q / 'qwen35-sampler-r9-saved-output-root-audit-20261112-v1.json'
    with dest.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(result=ref(dest),quality=scientific['quality'],
        http=scientific['paired_http']['workload_comparison'],full_model_qualified=False)))

if __name__ == '__main__':
    main()
