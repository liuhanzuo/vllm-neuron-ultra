"""Mandatory original sampler gates before package quality and timed HTTP.

Actual worker/process authentication and post-exit draining remain with the
existing root. Metadata checks cannot turn mocks into fresh execution proof.
"""
import hashlib
import json
from pathlib import Path

from .runtime import (validate_case, require, require_execution, read, write_new,
                      check_environment, server_arguments)
from .capture_worker import capture_identity, worker_name, ENV_PREFIX, REFERENCE, CANDIDATE
from qwen35_sampler_pair_protocol_v1 import check_capture_accounting, SAMPLER_GATES


def capture_server_arguments(case, arm):
    return server_arguments(case) + ['--worker-cls', worker_name(arm)]


def capture_environment(case, arm, directory, artifact_manifest_sha256):
    """Additional exact diagnostic bindings for the original root's clean env."""
    identity = capture_identity(case, arm, artifact_manifest_sha256)
    directory = Path(directory)
    require(directory.is_absolute() and directory.name == 'capture' and directory.parent.name == arm,
            'Exact per-arm capture directory required')
    return {ENV_PREFIX + 'ROOT': str(directory / 'packets'),
            ENV_PREFIX + 'RECEIPT': str(directory / 'worker-attachment.json'),
            ENV_PREFIX + 'IDENTITY': json.dumps(identity, sort_keys=True)}


def check_capture_attachment(attachment, live, case, arm, directory, manifest_sha256):
    """Call on the root's actual ready API/worker observation before diagnostics."""
    identity = capture_identity(case, arm, manifest_sha256)
    require(live['observation_kind'] == 'live_ready' and live['server_active'] is True
            and live['ready'] is True and live['exit_observed'] is False, 'Live capture server required')
    require(live['proc_owner_matches'] is True and live['invocation_matches'] is True
            and live['worker_in_server_cgroup'] is True, 'Root must observe actual capture owner')
    require(attachment['pid'] == live['worker_owner']['pid']
            and type(live['worker_owner']['start_ticks']) is int and live['worker_owner']['start_ticks'] > 0,
            'Actual capture worker PID/start identity mismatch')
    proof = live['startup_config_proof']
    require(proof['actual_api_argv'] == live['api_process_owner']['argv'] == capture_server_arguments(case, arm),
            'Actual package capture worker argv changed')
    env = proof['actual_api_initial_environment']
    require(all(env.get(k) == v for k, v in capture_environment(case, arm, directory, manifest_sha256).items()),
            'Actual capture environment binding changed')
    from .capture_worker import check_environment as check_capture_environment
    check_capture_environment(env, arm)
    require(env.get('NEURON_COMPILED_ARTIFACTS') == case['artifact'], 'Live capture artifact changed')
    for key in ('arm', 'phase', 'case', 'operation_class', 'reference_class', 'worker_class',
                'artifact_manifest_sha256', 'observation_enabled', 'timing_eligible'):
        require(attachment[key] == identity[key], 'Actual capture attachment mismatch: ' + key)
    require(attachment['host_identity_sha256'] == identity['sha256']
            and attachment['operator_installed_by_package'] is True
            and attachment['observer_only_attachment'] is True and attachment['debug_behavior'] == 'unchanged',
            'Capture changed the package operation or DEBUG behavior')
    return dict(attachment_consistent=True, root_observation_authentication_required=True, timing_eligible=False)


def run_capture_diagnostic(case, output, execution_guard):
    require_execution(execution_guard)
    validate_case(case); check_environment(case)
    require(not Path(output).exists(), 'Fresh diagnostic output required')
    from benchmark_workloads import inputs
    from bench_http import run_round
    prompt = inputs(case)[128]
    sampling = {"temperature": .7, "top_p": .9, "top_k": 40}
    # Original finite five requests: no seed or retry; B1/B4 quota is checked later.
    waves = [run_round(8010, case["served_name"], [prompt] * n, 32, False, **sampling) for n in (1, 4)]
    rows = [row for wave in waves for row in wave["requests"]]
    require(len(rows) == 5 and all(row["usage"]["prompt_tokens"] == 128
            and row["usage"]["completion_tokens"] == 32 for row in rows), "diagnostic token budget mismatch")
    result = {"complete": True, "waves": waves, "requests": 5, "output_tokens": 160,
              "sampling": sampling, "timing_eligible": False, "full_model_qualified": False}
    write_new(output, result)
    return result


def run_sampler_stage(phase, case, output_root, execution_guard, arm=None):
    """Each call in a new admitted process; capture server has already drained."""
    require_execution(execution_guard)
    validate_case(case); check_environment(case)
    root = Path(output_root)
    from vllm_neuron.models.qwen35.identity import ARTIFACT_MANIFEST
    manifest_sha256 = hashlib.sha256((Path(case['artifact']) / ARTIFACT_MANIFEST).read_bytes()).hexdigest()
    if phase == 'sampler_replay':
        require(arm in ('baseline', 'candidate'), 'Exact replay arm required')
        output = root / arm / 'sampler-replay.json'
        require(not output.exists(), 'Fresh sampler replay output required')
        attachment = read(root / arm / 'capture/worker-attachment.json')
        identity = capture_identity(case, arm, manifest_sha256)
        require(attachment['host_identity_sha256'] == identity['sha256']
                and attachment['case'] == case and attachment['artifact_manifest_sha256'] == manifest_sha256,
                'Replay must use this new package capture')
        require(attachment['operation_class'] == identity['operation_class']
                and attachment['reference_class'] == REFERENCE, 'Actual captured package operation changed')
        from .sampler_leaves import replay_captures
        result = replay_captures(root / arm / 'capture/packets', execution_guard)
        require(all(row['arm'] == arm for row in result['proofs']), 'Replay crossed capture arms')
        result.update(arm=arm, capture_identity_sha256=identity['sha256'])
    elif phase == 'sampler_controls':
        require(arm is None, 'Thirteen controls run once on the actual package implementation')
        output = root / 'sampler-controls.json'
        require(not output.exists(), 'Fresh sampler controls output required')
        from .sampler_leaves import rejection_controls
        result = rejection_controls(execution_guard)
    else:
        raise ValueError('Unknown sampler gate')
    result.update(case=case, phase=phase, artifact_manifest_sha256=manifest_sha256,
                  candidate_operation=CANDIDATE, reference_operation=REFERENCE,
                  fresh_execution_proven=False, full_model_qualified=False, timing_eligible=False)
    if phase == 'sampler_replay':
        check_replay_result(result, read(root / arm / 'capture/packets/capture-receipt.json'),
                            case, arm, manifest_sha256)
    else:
        check_controls_result(result, case, manifest_sha256)
    write_new(output, result)
    return result


def check_replay_result(result, capture, case, arm, manifest_sha256):
    check_capture_accounting(capture)
    require(capture['arm'] == arm and result['arm'] == arm and result['case'] == case
            and result['phase'] == 'sampler_replay', 'Replay arm/case/stage changed')
    require(result['artifact_manifest_sha256'] == manifest_sha256
            and result['capture_identity_sha256'] == capture_identity(case, arm, manifest_sha256)['sha256'],
            'Replay is not tied to the new package capture')
    require(result['passed'] is True and result['packets'] == capture['packets']
            and result['rows'] == capture['captured_rows'] and result['all_own_tokens_and_rng_reproduced'] is True
            and result['all_distribution_gates_passed'] is True and result['fast_path_rows'] > 0,
            'Complete actual sampler replay is mandatory')
    require(result['candidate_operation'] == CANDIDATE and result['reference_operation'] == REFERENCE
            and result['timing_eligible'] is False, 'Package replay implementation/timing changed')
    rows = result['proofs']
    require(len(rows) == result['packets'] == len(capture['files']), 'Every captured packet must be replayed')
    for row, packet in zip(rows, capture['files']):
        require(row['passed'] is True and row['arm'] == arm and row['call_id'] == packet['call_id']
                and row['rows'] == packet['rows'] and row['vocabulary'] == 248320
                and row['own_token_and_rng_reproduced'] is True and row['timing_eligible'] is False,
                'Packet/row/vocabulary/own-token-RNG proof incomplete')
        require(0 <= row['total_variation_max'] <= SAMPLER_GATES['total_variation_max']
                and 0 <= row['max_probability_error'] <= SAMPLER_GATES['max_probability_error_max'],
                'Full-vocabulary distribution threshold exceeded')
    require(sum(row['rows'] for row in rows) == result['rows']
            and sum(row['fast_rows'] for row in rows) == result['fast_path_rows'], 'Replay accounting mismatch')


def check_controls_result(result, case, manifest_sha256):
    require(result['case'] == case and result['phase'] == 'sampler_controls'
            and result['artifact_manifest_sha256'] == manifest_sha256, 'Controls package case changed')
    require(result['passed'] is True and result['candidate_operation'] == CANDIDATE
            and result['reference_operation'] == REFERENCE and result['timing_eligible'] is False,
            'Controls must execute the actual package candidate and original reference')
    rows = result['controls']
    require(len(rows) == 13 and [r['control'] for r in rows] == SAMPLER_GATES['controls'],
            'All thirteen named rejection controls are mandatory, in original order')
    require(all(r['passed'] is True and r['reference_calls'] == 1 and r['token_and_rng_equal'] is True for r in rows),
            'Fallback must reproduce output/error/global and request RNG with one original call')
    row_fallback = {'excessive_kth_ties', 'nonfinite_max_or_kth', 'cdf_near_boundary', 'cdf_boundary_tie'}
    require(all(r['whole_call'] == (r['control'] not in row_fallback) for r in rows), 'Control fallback scope changed')


def check_saved_sampler_gates(case, root):
    """Mandatory metadata prerequisite for quality/timing; root proves real exits."""
    validate_case(case)
    root = Path(root)
    from vllm_neuron.models.qwen35.identity import ARTIFACT_MANIFEST
    manifest_sha256 = hashlib.sha256((Path(case['artifact']) / ARTIFACT_MANIFEST).read_bytes()).hexdigest()
    for arm in ('baseline', 'candidate'):
        capture = read(root / arm / 'capture/packets/capture-receipt.json')
        check_replay_result(read(root / arm / 'sampler-replay.json'), capture, case, arm, manifest_sha256)
    check_controls_result(read(root / 'sampler-controls.json'), case, manifest_sha256)
    return dict(metadata_gates_passed=True, actual_capture_arms=2, mandatory_B1_calls_per_arm=16,
        mandatory_B4_calls_per_arm=16, controls=13, fresh_execution_proven=False,
        full_model_qualified=False, timing_eligible=False)
