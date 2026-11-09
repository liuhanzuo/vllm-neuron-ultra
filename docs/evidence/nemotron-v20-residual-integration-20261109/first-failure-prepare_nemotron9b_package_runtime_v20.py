"""Prepare v20 only after the actual native r2 and separate saved review exit.

No launch, cleanup, tensor reads, source approval, or runtime admission. The
builder retains the original r9 science and all old plus Dory source pins.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

R = Path(__file__).resolve().parents[1]
Q = R / 'research'
sys.path.insert(0, str(R))
import nemotron9b_package_protocol_v20 as p
import nemotron9b_residual_completion_v1 as completion
from research import nemotron9b_residual_source_partition_v1 as partition
from research.prepare_nemotron9b_package_runtime_v13 import merge, observer_policy

PARENT = dict(partition.PARENT)
CATALOG = dict(completion.base_history.CATALOG)
FINALIZER = dict(completion.base_history.EXIT)
TEST_REFS = (
    dict(path=str(Q/'nemotron9b-residual-partition-test-actual-exit-20261109-v1.json'),
         sha256='d0c63096fc0416e267dd4a7343dad3f4f98e59cec8ed634b8cd2f7c2700c425d'),
    dict(path=str(Q/'nemotron9b-v20-integration-test-actual-exit-20261109-v1.json'),
         sha256='2721639dc7b3082f17f1a9df931d130859e68ecf734f65b689c208f15ef3934d'),
)
CHANGED = {'schema', 'run_id', 'controller', 'worker', 'units', 'control_root',
           'evidence_root', 'work_root', 'source_sha256', 'runtime_readiness',
           'external_observer_contract', 'generated_compiler_link_successor'}
ADDED = {'required_residual_source_anchor', 'required_residual_completion',
         'dory_historical_payload_sha256', 'residual_successor'}


def encoded(path, value):
    return p.metadata.encode(path, value)


def value_ref(path, value):
    return dict(path=str(path), sha256=hashlib.sha256(encoded(path, value)).hexdigest())


def validate_parent(parent):
    # Exact identity also prevents changing scientific fields not named below.
    p.require(parent == p.metadata.load(PARENT), 'Exact actual admitted r9 parent')
    p.require(parent['schema'] == 'nemotron9b_package_runtime_v18'
              and parent['root_finalized'] is True, 'Original admitted r9 schema')
    p.require(parent['package_commit'] == p.COMMIT and parent['package_root'] == str(p.PACKAGE),
              'Original model implementation')
    p.require(parent['stages'] == list(p.STAGES) and len(p.STAGES) == 52
              and parent['resources'] == {x:p.resources(x) for x in (*p.STAGES, 'controller')},
              'All original stages and resources')
    p.require(parent['disk_budget'] == dict(root_floor=p.ROOT_FLOOR, nvme_floor=p.NVME_FLOOR,
              root_cap=p.ROOT_CAP, nvme_cap=p.NVME_CAP, caps=p.CAPS), 'Unreduced disk policy')
    p.require(parent['retirement_policy'] == 'none' and parent['automatic_successor_launch'] is False
              and parent['speedup_denominator'] is None, 'No automatic restart, cleanup or failed denominator')


def compose(parent, catalog, run_id, observer_ref, pins, review_exit_ref):
    p.require(isinstance(review_exit_ref, dict) and set(review_exit_ref) == {'path', 'sha256'},
              'Actual completed residual review exit required')
    validate_parent(parent)
    p.require(re.fullmatch('[a-z0-9-]{1,40}', run_id) and run_id != parent['run_id'],
              'Fresh explicit run identifier')
    p.require(catalog == p.metadata.load(CATALOG), 'Exact original base catalog')
    completed = completion.residual.completed(review_exit_ref)
    sources = merge(parent['source_sha256'], completed['source_sha256'], pins,
                    {ref['path']:ref['sha256'] for ref in (PARENT, CATALOG, FINALIZER, observer_ref)})
    result = copy.deepcopy(parent)
    result.update(
        schema='nemotron9b_package_runtime_v20', run_id=run_id,
        controller=str(R/'nemotron9b_package_controller_v20.py'),
        worker=str(R/'nemotron9b_package_worker_v20.py'),
        units={stage:'trn1-latest-verify-nemotron9b-package-'+run_id+'-'+stage.replace('_', '-')
               for stage in (*p.STAGES, 'controller')},
        control_root=str(R/('nemotron9b-package-'+run_id+'-control')),
        evidence_root=str(R/('nemotron9b-package-'+run_id+'-evidence')),
        work_root='/mnt/nvme/nemotron9b-package-'+run_id,
        source_sha256=sources, external_observer_contract=copy.deepcopy(observer_ref),
        required_residual_source_anchor=copy.deepcopy(completion.residual.PRODUCER),
        required_residual_completion=copy.deepcopy(review_exit_ref),
        dory_historical_payload_sha256=copy.deepcopy(completed['historical_binary_sha256']),
        residual_successor=dict(parent_spec=PARENT, actual_completed_review=copy.deepcopy(review_exit_ref),
                                original_failures_retained=True, full_model_qualified=False),
        generated_compiler_link_successor=dict(parent_spec=PARENT, integration_tests=list(TEST_REFS),
            original_failures_retained=True, model_code_unchanged=True,
            metadata_preparation_is_not_runtime_admission=True),
        runtime_readiness=dict(ready=False, source_compatible=True, blockers=[
            'Separate root review of exact generated specification',
            'Fresh complete physical drain, observer, storage and payload checks',
            'All52 original stages remain unexecuted in this new run']))
    p.require(set(result) == set(parent) | ADDED, 'Only reviewed successor fields added')
    p.require(all(result[k] == value for k, value in parent.items() if k not in CHANGED),
              'Every other original scientific/input/provenance field unchanged')
    p.require(set(p.required_sources()) <= set(sources), 'Complete new source closure')
    p.require(all(sources.get(ref['path']) == ref['sha256'] for ref in TEST_REFS),
              'Both actual source test results retained')
    history = completion.validate_spec(result, catalog)
    p.require(len(set(result['units'].values())) == 53
              and not set(result['units'].values()) & {row['unit'] for row in history['rows']},
              'Unique never-reused service names')
    p.alias_metadata.validate_spec(result)
    p.metadata.classify_sources(result, catalog)
    return result


def current_source_checks(spec, catalog, pending):
    parts = p.metadata.classify_sources(spec, catalog)
    for name, digest in parts['text_sha256'].items():
        if name in pending:
            p.require(value_ref(Path(name), pending[name])['sha256'] == digest,
                      'Exact unpublished metadata bytes')
            continue
        actual = (p.alias_metadata.source_sha(name, spec['metadata_aliases'], p.metadata.text)
                  if name in spec['metadata_aliases'] else p.metadata.sha(name, digest))
        p.require(actual == digest, 'Source fingerprint changed: '+name)
    for name, digest in parts['python_bytes_sha256'].items():
        p.metadata.python_source_bytes(name, digest)
    for name, row in spec['payloads'].items():
        p.require(p.file_stat(name) == row['stat'], 'Original payload identity changed')
    for name, identity in spec['runtime_binary_stat'].items():
        p.require(p.file_stat(name) == identity, 'Original runtime binary identity changed')
    return dict(source_entries_rehashed=len(parts['text_sha256'])+len(parts['python_bytes_sha256']),
                historical_binary_pins=len(parts['historical_payload_sha256']), historical_raw_reads=0,
                payload_bytes_read=0, payload_stats_checked=len(spec['payloads']),
                runtime_binary_stats_checked=len(spec['runtime_binary_stat']))


def build(run_id, review_exit_ref):
    # This mandatory real completion check precedes all output and preparation.
    completion.residual.completed(review_exit_ref)
    parent = p.metadata.load(PARENT)
    catalog = p.metadata.load(CATALOG)
    pins = {}
    for ref, count in zip(TEST_REFS, (19, 26)):
        tests = p.metadata.load(ref)
        p.require(tests['all_passed'] is True and tests['tests'] == count
                  and tests['actual_external_exit_observed'] is True and tests['exit_code'] == 0,
                  'Actual partition and integration test results')
        pins = merge(pins, tests['source_sha256'], {ref['path']:ref['sha256']})
    for name, digest in pins.items():
        p.metadata.sha(name, digest)
    for name in [*p.required_sources(), str(Path(__file__).resolve()),
                 str(Q/'check_prepare_nemotron9b_package_runtime_v20.py')]:
        pins = merge(pins, {name:p.metadata.sha(name)})
    observer = observer_policy(parent, p.metadata.load(parent['external_observer_contract']),
                               pins[str(R/'nemotron9b_package_window_v20.py')])
    prefix = Q/('nemotron9b-package-runtime-v20-'+run_id)
    observer_path = Path(str(prefix)+'-observer-contract.json')
    spec_path = Path(str(prefix)+'-spec.json')
    checks_path = Path(str(prefix)+'-preparation-checks.json')
    p.require(not any(path.exists() for path in (observer_path, spec_path, checks_path)),
              'New immutable preparation files')
    spec = compose(parent, catalog, run_id, value_ref(observer_path, observer), pins, review_exit_ref)
    for field in ('control_root', 'evidence_root', 'work_root'):
        p.require(not Path(spec[field]).exists(), 'Never reuse experiment output roots')
    p.storage(spec, reserve=True)
    p.compiler_links.verify_assets(p.metadata.load(p.compiler_links.ASSET_REF), hash_content=True)
    stats = current_source_checks(spec, catalog, {str(observer_path):observer})
    from nemotron9b_package_evidence_v20 import validate_payload_closure
    validate_payload_closure(spec)
    encoded(observer_path, observer)
    encoded(spec_path, spec)
    p.once(observer_path, observer)
    p.once(spec_path, spec)
    output = dict(schema='nemotron9b_v20_spec_preparation_checks_v1', reviewer='/root',
        specification=p.ref(spec_path), observer_contract=p.ref(observer_path), catalog=CATALOG,
        parent=PARENT, tests=list(TEST_REFS), source=p.ref(Path(__file__).resolve()), stats=stats,
        actual_residual_review_exit=review_exit_ref, original_stage_count=52,
        all_other_parent_fields_preserved=True, original_payload_and_runtime_stats_unchanged=True,
        fresh_physical_drain_proven=False, native_libraries_imported=False,
        execution_review_created=False, runtime_admitted=False, native_started=False, new_speedup_claim=False)
    p.once(checks_path, output)
    return dict(specification=p.ref(spec_path), observer_contract=p.ref(observer_path),
                checks=p.ref(checks_path), runtime_admitted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--completed-residual-review-exit', required=True)
    parser.add_argument('--completed-residual-review-exit-sha256', required=True)
    args = parser.parse_args()
    p.require(sys.dont_write_bytecode and __debug__, 'Bytecode disabled, assertions enabled')
    print(json.dumps(build(args.run_id, dict(path=args.completed_residual_review_exit,
                                           sha256=args.completed_residual_review_exit_sha256))))
