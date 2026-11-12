"""Exclusive source/mock evidence writer; no runtime binding or approval."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

from research import check_minicpm5_formal_admission_identity_v1 as checks

NEW = ('minicpm5_sdk_mapping_outer_source_entry_v8.py',
       'minicpm5_sdk_mapping_source_adapter_v3.py',
       'minicpm5_sdk_mapping_observed_adapter_v5.py',
       'minicpm5_sdk_mapping_observer_backend_v4.py',
       'minicpm5_sdk_mapping_observer_logic_v2.py',
       'prepare_minicpm5_catalog_formal_binding_v2.py',
       'check_minicpm5_formal_admission_identity_v1.py',
       'run_minicpm5_formal_admission_identity_checks_v1.py')
PRESERVED = ('minicpm5_sdk_mapping_nocache_entry_v6.py',
             'minicpm5_sdk_mapping_outer_source_entry_v7.py',
             'minicpm5_sdk_mapping_source_adapter_v2.py',
             'minicpm5_sdk_mapping_observed_adapter_v4.py',
             'minicpm5_sdk_mapping_observer_backend_v2.py',
             'minicpm5_sdk_mapping_observer_backend_v3.py',
             'minicpm5_sdk_mapping_observer_logic_v1.py',
             'minicpm5_formal_literal_codec_v1.py',
             'minicpm5_formal_literal_result_v1.py',
             'minicpm5_sdk_mapping_worker_source_entry_v6.py',
             'minicpm5_pv_sdk_import_diagnostic_v1.py',
             'prepare_minicpm5_catalog_formal_binding_v1.py')


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Never replace existing source evidence')
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(checks))
    original = json.loads((checks.Q / 'minicpm5-formal-import-r4-20261112-complete-source-map.json').read_text())
    preserved = [ref(checks.Q / name) for name in PRESERVED]
    unchanged = all(original.get(row['path']) == row['sha256'] for row in preserved)
    loaded = sorted(checks.BLOCKED.intersection(sys.modules))
    evidence = dict(schema='minicpm5_formal_admission_identity_source_checks_v1',
        interpreter=sys.executable, python_version=list(sys.version_info[:3]),
        scope='saved r4 JSON and full frozen literal; source/stdlib mocks and disposable Store directories',
        passed=result.wasSuccessful() and unchanged and not loaded,
        tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
        skipped=len(result.skipped), unittest_log=log.getvalue(),
        sources=[ref(checks.Q / name) for name in NEW], preserved_sources=preserved,
        preserved_sources_match_actual_r4_complete_map=unchanged,
        actual_saved_admission=ref(checks.ADMISSION_PATH),
        actual_r4_first_failed_predicate='metadata.read(admission_ref) == admission',
        exact_type_mismatch_paths=[
            'source_entry_binding.outer_types_source_observation.current_source_identity',
            'source_entry_binding.outer_types_source_observation.execution_receipt.identity'],
        saved_wire_bytes=len(checks.RAW), complete_expanded_admission_bytes=len(checks.codec.encoded(checks.SAVED)),
        frozen_literal_bytes=len(checks.LITERAL.encode()),
        observer_budget_fixture_measurements=checks.MEASUREMENTS,
        binding_fixture_measurements=checks.catalog_checks.MEASUREMENTS,
        native_modules_loaded=loaded, native_executed=False, lock_observed=False,
        runtime_binding_created=False, current_pid_inspected=False, model_executed=False,
        full_model_qualified=False, new_speedup_claim=False,
        limits=[
            'Actual r4 failed before SDK start; no actual SDK process or native import success is claimed.',
            'Original and successor source methods run with explicit mocks; no lock, native loader or process operation is invoked.',
            'The original predicate short-circuited at its first equality; later current-proc equality was never evaluated in that failed call.',
            'The source-memory reconstruction changes only the two tuples required by the frozen producer; it reproduces actual saved wire bytes exactly.',
            'Descriptor-plan mock and prospective SDK owners are explicit fixtures, not reconstructed runtime observations.',
            'All budgets, early source fence, 900-second deadline, original equality, task/wait/cleanup semantics and 1,284 scientific rows remain.'])
    with args.output.open('xb') as stream:
        stream.write(checks.codec.encoded(evidence))
    print(log.getvalue(), end='')
    print(json.dumps(dict(evidence=ref(args.output), passed=evidence['passed']), sort_keys=True))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
