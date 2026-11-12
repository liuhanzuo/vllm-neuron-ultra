"""Exclusive source-only evidence capture; never a formal execution/approval."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

from research import check_minicpm5_formal_literal_boundary_v1 as checks

NEW = (
    'minicpm5_formal_literal_codec_v1.py',
    'minicpm5_formal_literal_result_v1.py',
    'minicpm5_sdk_mapping_observer_backend_v3.py',
    'minicpm5_sdk_mapping_observed_adapter_v4.py',
    'minicpm5_sdk_mapping_source_adapter_v2.py',
    'minicpm5_sdk_mapping_nocache_launch_v3.py',
    'minicpm5_sdk_mapping_outer_source_entry_v6.py',
    'minicpm5_sdk_mapping_worker_source_entry_v6.py',
    'check_minicpm5_formal_literal_boundary_v1.py',
    'run_minicpm5_formal_literal_source_checks_v1.py',
)
PRESERVED = (
    'minicpm5_sdk_mapping_nocache_entry_v6.py',
    'minicpm5_sdk_mapping_outer_source_entry_v5.py',
    'minicpm5_sdk_mapping_worker_source_entry_v5.py',
    'minicpm5_sdk_mapping_source_adapter_v1.py',
    'minicpm5_sdk_mapping_nocache_launch_v2.py',
    'minicpm5_sdk_mapping_observed_adapter_v3.py',
    'minicpm5_sdk_mapping_observer_backend_v2.py',
    'minicpm5_sdk_mapping_observer_logic_v1.py',
)


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Source evidence never overwrites a prior file')
    suite = unittest.defaultTestLoader.loadTestsFromModule(checks)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    original = checks.saved('minicpm5-formal-import-r2-20261112-outer-entry-plan-final.json')['bootstrap_sources']
    original.update(checks.saved('minicpm5-formal-import-r2-20261112-sdk-entry-template-final.json')['bootstrap_sources'])
    preserved = [ref(checks.Q / name) for name in PRESERVED]
    frozen_unchanged = all(row['sha256'] == original[row['path']] for row in preserved)
    native_loaded = sorted(checks.NATIVE.intersection(sys.modules))
    evidence = dict(schema='minicpm5_formal_literal_source_checks_v1',
        scope='stdlib/source methods and mocked OS/native/auth ports; disposable temporary Store trees only',
        source_checks_passed=result.wasSuccessful() and frozen_unchanged and not native_loaded,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        complete_unittest_log=log.getvalue(), source_files=[ref(checks.Q / name) for name in NEW],
        frozen_original_files=preserved, frozen_original_files_match_saved_r2_bootstrap=frozen_unchanged,
        measured_fixture_serializations=checks.SIZING, native_modules_loaded=native_loaded,
        runtime_executed=False, runtime_binding_created=False, current_lock_observed=False,
        experiment_rerun=False, model_executed=False, full_model_qualified=False, new_speedup_claim=False,
        evidence_limits=[
            'Saved r2 owner/plan/references are real historical inputs; unsaved dynamic proof/types/time/SDK owner fields are explicitly synthetic fixtures.',
            'Measurements are actual codec/original Store bytes for these inputs, not the unknown failed r2 admission total.',
            'Normal complete result references roundtrip every original result field.',
            'Missing observer receipt reports retain complete terminal wire but explicitly unavailable observer payload with hash/bytes; no whole observer reconstruction is claimed.',
            'An overlong error class name or failed stderr stream can prevent fallback output; original persistence exception is retained without truncation or a fabricated receipt.',
        ])
    raw = checks.codec.encoded(evidence)
    with args.output.open('xb') as stream:
        stream.write(raw)
    print(log.getvalue(), end='')
    print(json.dumps(dict(evidence=ref(args.output.resolve()), source_checks_passed=evidence['source_checks_passed']), sort_keys=True))
    return 0 if evidence['source_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
