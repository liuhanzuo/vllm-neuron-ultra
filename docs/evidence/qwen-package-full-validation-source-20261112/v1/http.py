"""Original clients on an explicitly root-owned package server at port 8010.

The root launches/stops each fresh server using runtime.server_arguments and
authenticates live/terminal observations. These functions never launch a server.
"""
import hashlib
import json
from pathlib import Path

from .runtime import (ROOT, BASE, PYTHON, configuration, validate_case, read, require,
                      require_execution, check_environment, write_new, server_arguments)


def run_http_stage(phase, case, output, execution_guard):
    require_execution(execution_guard)
    validate_case(case)
    check_environment(case)
    output = Path(output)
    require(not output.exists(), 'Historical HTTP output reuse forbidden')
    if phase == 'smoke':
        import orchestrate
        result = orchestrate.quality(case)
        require(result['passed'] and result['all_identical_rows_match']
                and all(len(result[k]) == 4 for k in ('serial', 'concurrent', 'identical_prompt_outputs')),
                'Complete original smoke gate required')
        write_new(output, result)
    elif phase == 'quality96':
        from evaluate_quality import evaluate
        result = evaluate(case, output)
    elif phase == 'initial120':
        from bench_http import benchmark
        from orchestrate import prompt_ids
        prompt = prompt_ids(case, '请详细介绍一下法国的首都。')
        result = benchmark(8010, case['served_name'], [prompt], 4, output_tokens=120, rounds=3)
        write_new(output, result)
    elif phase == 'workloads':
        from benchmark_workloads import evaluate
        result = evaluate(case, output)
    else:
        raise ValueError('Unknown HTTP stage')
    return result


def check_matched_hosts(left, right, case, directories):
    """Check complete fresh observed API settings; root authenticates observations.

    This is a metadata consistency check, never a replacement for independent
    live-owner/worker/cgroup identity and actual post-stop drain evidence.
    """
    validate_case(case)
    settings = []
    for arm, report in (('baseline', left), ('candidate', right)):
        require(report['observation_kind'] == 'live_ready' and report['ready'] is True
                and report['server_active'] is True and report['exit_observed'] is False,
                'Actual live observation required')
        proof = report['startup_config_proof']
        env = proof['actual_api_initial_environment']
        argv = proof['actual_api_argv']
        require(type(env) is dict and all(type(k) is str and type(v) is str for k, v in env.items()),
                'Complete observed environment required')
        require(argv == report['api_process_owner']['argv'] == server_arguments(case),
                'Actual package serve argv changed')
        require(env.get('NEURON_CPU_SAMPLER_FAST') == ('0' if arm == 'baseline' else '1'),
                'Actual package sampler arm differs')
        required = dict(NEURON_QWEN35='1', NEURON_ON_DEVICE_SAMPLING_DISABLED='1',
            XLA_DISABLE_FUNCTIONALIZATION='1', NEURON_QWEN35_ALLOW_COMPILE='0',
            NEURON_COMPILED_ARTIFACTS=case['artifact'],
            NEURON_COMPILE_CACHE_URL=case['compiler_cache'], BASE_COMPILE_WORK_DIR=case['compiler_workdir'])
        require(all(env.get(k) == v for k, v in required.items()), 'Package load-only settings changed')
        require(not any(k.startswith(('TRN1_QWEN35', 'QWEN35_SAMPLER_')) or k == 'TRN1_CPU_SAMPLER_FAST'
                        for k in env), 'Research host/capture hooks cannot enter measured package arms')
        directory = Path(directories[arm])
        require(directory.is_absolute() and directory.name == 'http' and directory.parent.name == arm,
                'Exact fresh measured arm directory required')
        require(env.get('TMPDIR') == str(directory / 'tmp')
                and env.get('VLLM_RPC_BASE_PATH') == str(directory / 'rpc'), 'Unbound per-arm temp/RPC paths')
        settings.append({k: v for k, v in env.items()
                         if k not in ('NEURON_CPU_SAMPLER_FAST', 'TMPDIR', 'VLLM_RPC_BASE_PATH')})
    require(settings[0] == settings[1], 'Host/debug/thread/environment mismatch')
    require(left['api_process_owner'] != right['api_process_owner'], 'Fresh separate API invocations required')
    return dict(matched=True, observation_enabled=False, actual_process_authentication_required=True)


def check_initial_pair(left, right):
    from qwen35_gdn_serving_checks_v2 import valid_rate
    for report in (left, right):
        require(report["batch"] == 4 and report["sampling"] == {} and report["output_tokens_per_request"] == 120
                and len(report["rounds"]) == 3 and len(report["prompt_inputs"]) == 1, "initial HTTP protocol changed")
    require(left["prompt_inputs"] == right["prompt_inputs"], "initial prompt mismatch")
    def groups(report):
        return [r[k] for r in report["rounds"] for k in ("single", "concurrent")] + [report["streaming"]]
    for index, (a, b) in enumerate(zip(groups(left), groups(right))):
        count = 4 if index == 6 or index % 2 else 1
        valid_rate(a); valid_rate(b)
        require(len(a["requests"]) == len(b["requests"]) == count, "initial HTTP group incomplete")
        for x, y in zip(a["requests"], b["requests"]):
            expected_usage = {"prompt_tokens": len(left["prompt_inputs"][0]), "completion_tokens": 120,
                              "total_tokens": len(left["prompt_inputs"][0]) + 120}
            require(x["usage"] == y["usage"], "initial usage differs between arms")
            for row in (x, y):
                require(isinstance(row["usage"], dict) and all(
                    type(row["usage"].get(key)) is int and row["usage"][key] == count
                    for key, count in expected_usage.items()), "initial token budget changed")
            require(x["text"] == y["text"], "initial greedy text changed")


def check_saved_quality(case, baseline, candidate):
    """Run after both fresh quality arms and BEFORE either timed HTTP arm."""
    validate_case(case)
    from .checks import check_quality_pair, original_validators
    base, cand = Path(baseline), Path(candidate)
    suite_path = ROOT / 'quality-suite.json'
    return check_quality_pair(read(base / 'quality/quality96.json'), read(cand / 'quality/quality96.json'),
        read(BASE / 'quality96.json'), read(suite_path), hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        original_validators(), case)


def compare_saved_pair(case, baseline, candidate, hosts, http_directories):
    """All original quality and HTTP gates; actual root exits are audited separately."""
    validate_case(case)
    from qwen35_gdn_serving_checks_v2 import compare
    base, cand = Path(baseline), Path(candidate)
    quality = check_saved_quality(case, base, cand)
    matched = check_matched_hosts(hosts['baseline'], hosts['candidate'], case, http_directories)
    check_initial_pair(read(base / 'http/initial120.json'), read(cand / 'http/initial120.json'))
    left, right = read(base / 'http/workloads.json'), read(cand / 'http/workloads.json')
    result = compare(left, right, case, case, left['inputs'])
    require(result['identical_greedy_texts'] == 57, 'All 57 greedy HTTP texts must match')
    return dict(quality=quality, matched_hosts=matched, paired_http=result,
                fresh_execution_proven=False, full_model_qualified=False,
                independent_root_final_audit_required=True, automatic_best_promotion=False)
