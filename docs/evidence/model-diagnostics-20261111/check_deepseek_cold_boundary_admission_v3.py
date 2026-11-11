"""Stdlib negative controls for caller-bound neutrality input hashes only."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import verify_deepseek_cold_boundary_neutrality_v2 as v

ROOT = Path(__file__).resolve().parent


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('unadmitted or changed metadata was accepted')


def main():
    prior_path = ROOT / 'research/deepseek-cold-boundary-integration-source-checks-20261111-v2.json'
    prior = json.loads(prior_path.read_text())
    for path, expected in prior['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    expected = hashlib.sha256(prior_path.read_bytes()).hexdigest()
    binding = {str(prior_path.resolve()): expected}
    assert v.admitted_digest(prior_path, binding) == expected
    assert v.pinned_read(prior_path, binding) == prior
    reject(lambda: v.admitted_digest(prior_path, {}))
    reject(lambda: v.admitted_digest(prior_path, {str(prior_path.resolve()): '0' * 64}))
    reject(lambda: v.pinned_read(prior_path, {str(prior_path.resolve()): expected[:-1]}))
    source = ROOT / 'verify_deepseek_cold_boundary_neutrality_v2.py'
    tree = ast.parse(source.read_text())
    verify = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    assert [arg.arg for arg in verify.args.kwonlyargs] == ['admitted_sha256']
    raw_paths = next(n for n in verify.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'raw_paths' for t in n.targets))
    assert len(raw_paths.value.elts) == 4
    admission = next(n for n in verify.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'admitted_raw' for t in n.targets))
    loads = [n for n in ast.walk(verify) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'torch' and n.func.attr == 'load']
    assert loads and all(admission.lineno < n.lineno for n in loads)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'read'
                   for n in ast.walk(verify))
    cache = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compare_cache_backend')
    manifest_assignments = [n for n in cache.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id in ('old_manifest', 'new_manifest') for t in n.targets)]
    assert len(manifest_assignments) == 2 and all(n.value.func.id == 'pinned_read' for n in manifest_assignments)
    assert not any(n == 'torch' or n.startswith(('torch.', 'numpy', 'neuronx_distributed')) for n in sys.modules)
    result = dict(passed=True, scope='actual source AST and small JSON only; no tensor or native execution',
                  checks=['prior_v2_sources_unchanged', 'exact_admitted_digest_and_metadata_accepted',
                          'missing_changed_and_malformed_admission_hashes_rejected',
                          'all_four_raw_logits_admitted_before_deserialization',
                          'outer_reports_and_both_original_and_new_manifests_require_caller_pins'],
                  source_sha256={str(source): hashlib.sha256(source.read_bytes()).hexdigest(),
                                 str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  predecessor_check_sha256=expected, runtime_launched=False, raw_payloads_read=False,
                  actual_capture_neutrality_proven=False)
    output = ROOT / 'research/deepseek-cold-boundary-admission-source-checks-20261111-v3.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
