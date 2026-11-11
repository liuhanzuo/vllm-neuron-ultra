"""Additional stdlib checks; preserves the already completed v1 evidence."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as N

import deepseek_cpu_capture_bridge_v1 as b
import verify_deepseek_cold_boundary_neutrality_v1 as verifier

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('negative control accepted')


def main():
    prior_path = ROOT / 'research/deepseek-cpu-capture-bridge-source-checks-20261111-v1.json'
    prior = json.loads(prior_path.read_text())
    assert prior['passed']
    for kind in ('source_sha256', 'successor_sha256'):
        for path, expected in prior[kind].items():
            assert sha(path) == expected, path
    checks = ['all_previously_reviewed_v1_sources_and_evidence_preserved']

    old = ROOT / 'validate_deepseek_cold_kv_v2.py'
    new = ROOT / 'validate_deepseek_cold_boundary_v2.py'
    for name in ('main', 'reference'):
        left = [ast.dump(n, include_attributes=False) for n in ast.walk(function(old, name)) if isinstance(n, ast.For)]
        right = [ast.dump(n, include_attributes=False) for n in ast.walk(function(new, name)) if isinstance(n, ast.For)]
        assert left == right
    assert 'from deepseek_cold_boundary_lifecycle_v1 import observe as lifecycle_observer' in new.read_text()
    assert "report['raw_logits_sha256'] = digest(output.with_suffix('.pt'))" in new.read_text()
    checks.append('v2_driver_preserves_original_loops_and_binds_relocated_lifecycle_plus_raw_logit_hash')

    lifecycle = ROOT / 'deepseek_cold_boundary_lifecycle_v1.py'
    tree = ast.parse(lifecycle.read_text())
    assert not any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                   and n.value.id == 'p' and n.attr == 'DEST' for n in ast.walk(tree))
    text = lifecycle.read_text()
    for statement in ('p.check_copy(copy)', 'p.check_lifecycle(report)', 'model.warmup()',
                      'clear(model, copy[\'bindings\'])', 'graph_instrumented=True',
                      'raw_dump_bytes=2*STATE_BYTES', 'path.relative_to(destination)'):
        assert statement in text, statement
    checks.append('relocated_lifecycle_retains_original_validators_and_never_uses_old_destination')

    # Exercise the actual new join against the actual old frame enumerator,
    # isolated from all protocol imports and all tensor files.
    contract = ROOT / 'deepseek_cold_kv_contract_v2.py'
    code = compile(ast.Module(body=[function(contract, 'frames')], type_ignores=[]), str(contract), 'exec')
    env = {'PREFIXES': b.PREFIXES}; exec(code, env)
    stub = ModuleType('deepseek_cold_kv_contract_v2'); stub.frames = env['frames']
    assert stub.__name__ not in sys.modules
    sys.modules[stub.__name__] = stub
    try:
        for backend in ('native', 'device'):
            rows = [dict(frame=f) for f in stub.frames(backend)]
            old_manifest = dict(complete=True, frames=rows)
            new_manifest = dict(complete=True, frames=list(rows))
            assert len(verifier._join_cache_frames(old_manifest, new_manifest, backend)) == (42 if backend == 'native' else 84)
            reject(lambda: verifier._join_cache_frames(old_manifest, dict(new_manifest, frames=rows[:-1]), backend))
            reject(lambda: verifier._join_cache_frames(old_manifest, dict(new_manifest, frames=rows[::-1]), backend))
            reject(lambda: verifier._join_cache_frames(old_manifest, dict(new_manifest, complete=False), backend))
    finally:
        del sys.modules[stub.__name__]
    checks.append('full_raw_cache_comparator_pairs_all42_84_and_rejects_omission_reordering_incomplete_capture')

    # Actual old codec scope arithmetic, evaluated with stdlib constants only.
    env = dict(PREFIXES=b.PREFIXES, LAYERS=36, HEADS=8, HEAD_DIM=128, old=N(CRITERIA=prior['metadata_neutrality_control']))
    exec(compile(ast.Module(body=[function(contract, 'scope')], type_ignores=[]), str(contract), 'exec'), env)
    scope = env['scope']()
    boundaries = b.boundary_byte_budget()
    lifecycle_bytes = 2 * 144 * 4 * 4 * 4224 * 128 * 2
    logits = (42 + 84) * 151936 * 4
    journal = scope['append_only_cache_journal_bytes']
    fallback = scope['worst_case_full_cache_payload_bytes']
    plan_path = ROOT / 'deepseek-cold-kv-diagnostic-v2/plan.json'
    plan = json.loads(plan_path.read_text())
    weight_paths = [p for p in plan['large_file_identities'] if 'deepseek-qwen3-exact-input-cpu32-shards/rank' in p]
    weight_copy = sum(plan['large_file_identities'][p][2] for p in weight_paths)
    assert len(weight_paths) == 2 and weight_copy == 16382184072
    serialization_reserve = 2 << 30
    budget = dict(
        calculation_source_sha256={str(contract): sha(contract), str(plan_path): sha(plan_path)},
        boundary_tensor_bytes=boundaries['combined_boundary_tensor_bytes'],
        native_and_device_raw_logits_bytes=logits,
        all144_state_before_after_warmup_snapshots_bytes=lifecycle_bytes,
        full_history_KV_append_only_bytes=journal,
        full_history_KV_all_replacements_worst_case_bytes=fallback,
        serialization_and_json_reserved_bytes=serialization_reserve,
        output_bytes_if_all_old_prefixes_remain_identical=journal+lifecycle_bytes+logits+boundaries['combined_boundary_tensor_bytes']+serialization_reserve,
        output_bytes_worst_case=fallback+lifecycle_bytes+logits+boundaries['combined_boundary_tensor_bytes']+serialization_reserve,
        native_KV_worst_case_bytes=fallback//3,
        device_KV_worst_case_bytes=fallback*2//3,
        two_historical_TP_shard_weight_files_bytes=weight_copy,
        additional_weight_copy_required='Unknown: immutable sharing of unchanged weight entries must first be verified; reserve full copy otherwise.',
        new_compiler_scratch_bytes=None,
        new_NEFF_and_serialized_graph_bytes=None,
        old_model_pt_bytes=176072171,
        old_model_pt_is_not_new_artifact_bound=True,
        original_existing_artifacts_and_raw_evidence_preserved=True,
        proposed_placement='Root: native payload and lifecycle snapshots; NVMe: device payload. New compile weights/scratch/artifact require separate fresh bounds.',
        root_reported_free_gib=28.67, nvme_reported_free_gib=44, nvme_required_free_floor_gib=30,
        free_space_source='Parent status message, not independently remeasured by this source-only task.',
        launch_admitted=False,
        reason='No total execution admission until new compile scratch/artifact bounds and weight sharing/copy are established and all free-space floors rechecked.')
    checks.append('total_payload_budget_includes_full_KV_fallback_lifecycle_logits_weights_and_unknown_compile_costs')
    outputs = [ROOT / name for name in ('validate_deepseek_cold_boundary_v2.py',
        'deepseek_cold_boundary_lifecycle_v1.py', 'verify_deepseek_cold_boundary_neutrality_v1.py', Path(__file__).name)]
    for path in outputs:
        compile(path.read_text(), str(path), 'exec')
    assert not any(n == 'torch' or n.startswith(('torch.', 'numpy', 'neuronx_distributed')) for n in sys.modules)
    result = dict(passed=True, scope='source plus inert stdlib metadata fixtures only', checks=checks,
                  predecessor_check_sha256=sha(prior_path), source_sha256={str(p): sha(p) for p in outputs},
                  execution_storage_budget=budget, runtime_launched=False, raw_payloads_read=False,
                  actual_capture_neutrality_proven=False, full_model_qualified=False, new_speedup_claim=False)
    destination = ROOT / 'research/deepseek-cold-boundary-integration-source-checks-20261111-v2.json'
    assert not destination.exists(), 'preserve prior check evidence'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
