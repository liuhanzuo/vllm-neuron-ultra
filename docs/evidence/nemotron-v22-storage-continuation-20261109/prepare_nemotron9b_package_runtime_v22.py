"""Build a future Nemotron spec only from actual completed storage maintenance.

The original r9 model/science remains unchanged. This prepares metadata only;
it never launches, cleans storage, approves execution, or invents completion.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

R=Path(__file__).resolve().parents[1];Q=R/'research'
sys.path[:0]=[str(R),str(Q)]
import nemotron9b_package_protocol_v22 as p
import nemotron9b_completed_storage_completion_v1 as completion
import prepare_nemotron9b_package_runtime_v20 as previous

PARENT,CATALOG,FINALIZER=previous.PARENT,previous.CATALOG,previous.FINALIZER
merge,observer_policy=previous.merge,previous.observer_policy
TEST_REFS=(*previous.TEST_REFS,
    dict(path=str(Q/'nemotron9b-storage-completion-tests-actual-exit-20261109-v1.json'),
         sha256='fe948d491ffd31138affaed7cbfa40cc6f10ee3b6756102db0b46fa11bdc137e'),
    dict(path=str(Q/'nemotron9b-storage-metadata-tests-actual-exit-20261109-v1.json'),
         sha256='da7a85ef6dd845c20b0b14491fc6ea28cceccdd08eb8e7cfc4ff085bd4af1345'),
    dict(path=str(Q/'nemotron9b-v22-integration-tests-actual-exit-20261109-v1.json'),
         sha256='3812bc27633f86d8ed9950c0ac5c4fce23bebc3d9e3fd0b129fca0ec888270d2'))
TEST_COUNTS=(19,26,28,9,27)
CHANGED=set(previous.CHANGED)
ADDED=set(previous.ADDED)|{'required_completed_storage','completed_storage_successor'}


def encoded(path,value):return p.metadata.encode(path,value)


def value_ref(path,value):return dict(path=str(path),sha256=hashlib.sha256(encoded(path,value)).hexdigest())


def compose(parent,catalog,run_id,observer_ref,pins,storage_exit_ref):
    p.require(isinstance(storage_exit_ref,dict) and set(storage_exit_ref)=={'path','sha256'},
              'Actual completed storage external exit required')
    previous.validate_parent(parent)
    p.require(parent['stages']==list(p.STAGES)
              and parent['resources']=={stage:p.resources(stage)for stage in (*p.STAGES,'controller')},
              'All original52 stages and resources')
    p.require(re.fullmatch('[a-z0-9-]{1,40}',run_id) and run_id!=parent['run_id'], 'Fresh explicit run identifier')
    p.require(catalog==p.metadata.load(CATALOG), 'Exact original base catalog')
    completed=completion.maintenance.completed(storage_exit_ref)
    review_exit_ref=completed['history']['completed_residual_review']
    sources=merge(parent['source_sha256'],completed['source_sha256'],pins,
                  {ref['path']:ref['sha256']for ref in (PARENT,CATALOG,FINALIZER,observer_ref)})
    result=copy.deepcopy(parent)
    result.update(schema='nemotron9b_package_runtime_v22',run_id=run_id,
        controller=str(R/'nemotron9b_package_controller_v22.py'),
        worker=str(R/'nemotron9b_package_worker_v22.py'),
        units={stage:'trn1-latest-verify-nemotron9b-package-'+run_id+'-'+stage.replace('_','-')
               for stage in (*p.STAGES,'controller')},
        control_root=str(R/('nemotron9b-package-'+run_id+'-control')),
        evidence_root=str(R/('nemotron9b-package-'+run_id+'-evidence')),
        work_root='/mnt/nvme/nemotron9b-package-'+run_id,
        source_sha256=sources,external_observer_contract=copy.deepcopy(observer_ref),
        required_residual_source_anchor=copy.deepcopy(completion.residual.PRODUCER),
        required_residual_completion=copy.deepcopy(review_exit_ref),
        dory_historical_payload_sha256=copy.deepcopy(completed['historical_binary_sha256']),
        required_completed_storage=copy.deepcopy(storage_exit_ref),
        residual_successor=dict(parent_spec=PARENT,actual_completed_review=copy.deepcopy(review_exit_ref),
                                original_failures_retained=True,full_model_qualified=False),
        completed_storage_successor=dict(actual_completed_maintenance=copy.deepcopy(storage_exit_ref),
            original_model_and_science_retained=True,fresh_capacity_check_still_required=True),
        generated_compiler_link_successor=dict(parent_spec=PARENT,integration_tests=list(TEST_REFS),
            original_failures_retained=True,model_code_unchanged=True,
            metadata_preparation_is_not_runtime_admission=True),
        runtime_readiness=dict(ready=False,source_compatible=True,blockers=[
            'Separate root review of exact generated specification',
            'Fresh full349-owner/1149-service physical drain, observer, storage and payload checks',
            'All52 original stages remain unexecuted in this new run']))
    p.require(set(result)==set(parent)|ADDED, 'Only reviewed completion and successor fields added')
    p.require(all(result[key]==value for key,value in parent.items()if key not in CHANGED),
              'Every other original scientific/input/provenance field unchanged')
    p.require(set(p.required_sources())<=set(sources), 'Complete new source closure')
    p.require(all(sources.get(ref['path'])==ref['sha256']for ref in TEST_REFS), 'All actual source test results retained')
    history=completion.validate_spec(result,catalog)
    p.require(len(set(result['units'].values()))==53
              and not set(result['units'].values())&{row['unit']for row in history['rows']},
              'Unique never-reused service names')
    p.alias_metadata.validate_spec(result)
    p.metadata.classify_sources(result,catalog)
    return result


def current_source_checks(spec,catalog,pending):
    parts=p.metadata.classify_sources(spec,catalog)
    for name,digest in parts['text_sha256'].items():
        if name in pending:
            p.require(value_ref(Path(name),pending[name])['sha256']==digest, 'Exact unpublished metadata bytes')
            continue
        actual=(p.alias_metadata.source_sha(name,spec['metadata_aliases'],p.metadata.text)
                if name in spec['metadata_aliases'] else p.metadata.sha(name,digest))
        p.require(actual==digest, 'Source fingerprint changed: '+name)
    for name,digest in parts['python_bytes_sha256'].items():p.metadata.python_source_bytes(name,digest)
    for name,row in spec['payloads'].items():p.require(p.file_stat(name)==row['stat'], 'Original payload identity changed')
    for name,identity in spec['runtime_binary_stat'].items():
        p.require(p.file_stat(name)==identity, 'Original runtime binary identity changed')
    return dict(source_entries_rehashed=len(parts['text_sha256'])+len(parts['python_bytes_sha256']),
        historical_binary_pins=len(parts['historical_payload_sha256']),historical_raw_reads=0,payload_bytes_read=0,
        payload_stats_checked=len(spec['payloads']),runtime_binary_stats_checked=len(spec['runtime_binary_stat']))


def build(run_id,storage_exit_ref):
    # Real successful maintenance, native and saved-review exits precede any output.
    actual=completion.maintenance.completed(storage_exit_ref)
    parent,catalog=p.metadata.load(PARENT),p.metadata.load(CATALOG)
    pins={}
    for ref,count in zip(TEST_REFS,TEST_COUNTS):
        tests=p.metadata.load(ref)
        p.require(tests['all_passed'] is True and tests['tests']==count
                  and tests['actual_external_exit_observed'] is True and tests['exit_code']==0,
                  'Actual completed source validation suites')
        pins=merge(pins,tests['source_sha256'],{ref['path']:ref['sha256']})
    for name,digest in pins.items():p.metadata.sha(name,digest)
    for name in [*p.required_sources(),str(Path(__file__).resolve()),str(Path(previous.__file__).resolve()),
                 str(Q/'check_prepare_nemotron9b_package_runtime_v22.py')]:
        pins=merge(pins,{name:p.metadata.sha(name)})
    observer=observer_policy(parent,p.metadata.load(parent['external_observer_contract']),
                             pins[str(R/'nemotron9b_package_window_v22.py')])
    prefix=Q/('nemotron9b-package-runtime-v22-'+run_id)
    observer_path=Path(str(prefix)+'-observer-contract.json')
    spec_path=Path(str(prefix)+'-spec.json')
    checks_path=Path(str(prefix)+'-preparation-checks.json')
    p.require(not any(path.exists()for path in (observer_path,spec_path,checks_path)), 'Fresh immutable preparation files')
    spec=compose(parent,catalog,run_id,value_ref(observer_path,observer),pins,storage_exit_ref)
    for field in ('control_root','evidence_root','work_root'):
        p.require(not Path(spec[field]).exists(), 'Never reuse experiment output roots')
    p.storage(spec,reserve=True)
    p.compiler_links.verify_assets(p.metadata.load(p.compiler_links.ASSET_REF),hash_content=True)
    stats=current_source_checks(spec,catalog,{str(observer_path):observer})
    from nemotron9b_package_evidence_v22 import validate_payload_closure
    validate_payload_closure(spec)
    encoded(observer_path,observer);encoded(spec_path,spec)
    p.once(observer_path,observer);p.once(spec_path,spec)
    output=dict(schema='nemotron9b_v22_spec_preparation_checks_v1',reviewer='/root',
        specification=p.ref(spec_path),observer_contract=p.ref(observer_path),catalog=CATALOG,parent=PARENT,
        tests=list(TEST_REFS),source=p.ref(Path(__file__).resolve()),stats=stats,
        actual_residual_review_exit=actual['history']['completed_residual_review'],
        actual_completed_storage_exit=storage_exit_ref,original_stage_count=52,
        all_other_parent_fields_preserved=True,original_payload_and_runtime_stats_unchanged=True,
        fresh_physical_drain_proven=False,native_libraries_imported=False,
        execution_review_created=False,runtime_admitted=False,native_started=False,new_speedup_claim=False)
    p.once(checks_path,output)
    return dict(specification=p.ref(spec_path),observer_contract=p.ref(observer_path),
                checks=p.ref(checks_path),runtime_admitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--completed-storage-exit',required=True)
    parser.add_argument('--completed-storage-exit-sha256',required=True)
    args=parser.parse_args()
    p.require(sys.dont_write_bytecode and __debug__, 'Bytecode disabled and original assertions enabled')
    print(json.dumps(build(args.run_id,dict(path=args.completed_storage_exit,sha256=args.completed_storage_exit_sha256))))
