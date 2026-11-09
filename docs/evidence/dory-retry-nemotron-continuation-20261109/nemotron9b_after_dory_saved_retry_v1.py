"""Metadata completion proof for a separately admitted full saved-review retry.

No producer or reviewer is launched. Actual failed r2 and successful r3 exits
must both exist. The original seven numerical failures remain failures; this
only permits later diagnostic analysis and a freshly admitted serial successor.
"""
import copy
from pathlib import Path
import stat

import dory_saved_review_retry_contract_v1 as retry

old = retry.original
require, merge, external, append_owner = old.require, old.merge, old.external, old.append_owner
Q, BASE, PRODUCER = old.Q, old.BASE, old.PRODUCER
OUT = retry.NEXT_OUT
PREPARATION = dict(path=str(Q/'dory-saved-review-retry-source-preparation-review-20261109-v1.json'),
    sha256='9636499abbae1322190f35e2060a641dc41f837969596afced5f36056089f5b4')
QUERY_PREPARATION = dict(path=str(Q/'dory-saved-review-service-query-source-preparation-review-20261109-v2.json'),
    sha256='c4ee8608f4a56ca8717c99ed8b23d215e92819fd85e624abb0fc9cdab9f691bb')
OUTPUTS = dict(admission='admission.json', predecessor_history='predecessor-history.json',
    predecessor_drain='predecessor-drain.json', source_files='source-files.json',
    frozen_contract='post-exit-contract.json', saved_review='saved-residual-review.json',
    query_authentication='service-query-authentication.json', result='result.json')
Reader = old.Reader


def output_names():
    names = set()
    for path in OUT.iterdir():
        info = path.lstat()
        require(path.resolve() == path and stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                and 0 < info.st_size <= 32*2**20, 'Bounded ordinary complete retry output')
        names.add(path.name)
    return names


def query_scope(record):
    accounting = record['guard_accounting']
    require(record['complete'] is True and record['dependency_observation'] == retry.QUERY_OBSERVATION
            and record['authenticated_files_before_import'] == 41
            and record['authenticated_bytes_before_import'] == 4326816
            and record['original_import_policy_restored'] is True and record['model_speedup_claim'] is False,
            'Complete query pre-import authentication and restored import policy')
    require(accounting['closed'] is True and accounting['poisoned'] is False
            and accounting['poll_results_cached'] is False and accounting['guard_frequency_reduced'] is False
            and type(accounting['original_service_checks']) is int
            and accounting['original_service_checks'] >= 41
            and type(accounting['persistent_service_checks']) is int
            and accounting['persistent_service_checks'] > 0
            and accounting['successful_guards'] ==
                accounting['original_service_checks'] + accounting['persistent_service_checks'],
            'Every original and fresh persistent guard completed without failure or reduced cadence')
    audit = record['final_dependency_audit']
    require(type(audit['new_native_files']) is int and audit['new_native_files'] == 9
            and isinstance(audit['authenticated_imports'], dict) and audit['authenticated_imports'],
            'Final nine-file native mapping and authenticated import audit')


def extend(previous, actual_exit_ref, read, *, gone=old.low.gone, directories=None, names=None):
    """previous must come from retry.failed_history; injected readers are tests."""
    receipt = read(actual_exit_ref)
    external(receipt, 'saved-review-retry')
    spec_ref, source_ref = receipt['spec'], receipt['source_review']
    spec = read(spec_ref)
    parent = retry.validate_spec(spec, read)
    require(previous['parent'] == parent and previous['failed_review_complete'] is False
            and spec['failed_review_exit'] == previous['history']['failed_residual_saved_review'],
            'Completed retry retains the exact actual failed prior review')
    prep = read(PREPARATION)
    require(prep['source_preparation_passed'] is True and prep['tests_passed'] == 32
            and prep['runtime_admitted'] is False
            and all(spec['source_sha256'].get(p) == digest for p, digest in prep['source_sha256'].items()),
            'Exact separately prepared retry implementation retained')
    query_prep = read(QUERY_PREPARATION)
    for name in ('dory_service_query_auth_v1.py', 'dory_saved_review_service_query_v1.py',
                 'dory_saved_review_guard_v1.py'):
        path = str(Q/name)
        require(spec['source_sha256'].get(path) == query_prep['source_sha256'][path],
                'Exact prepared query authentication and complete guard implementation')
    source_review = read(source_ref)
    require(source_review['reviewer'] == '/root' and source_review['passed'] is True
            and source_review['blockers'] == [] and source_review['spec_sha256'] == spec_ref['sha256']
            and source_review['source_sha256'] == spec['source_sha256'],
            'Final exact runtime source review, separate from preparation')
    for key, name in OUTPUTS.items():
        require(receipt[key]['path'] == str(OUT/name), 'Exact retry output namespace: '+key)
    records = {key: read(receipt[key]) for key in OUTPUTS}
    actual_names = output_names() if names is None else set(names)
    require(actual_names == set(OUTPUTS.values()),
            'No missing, failed, cleanup-failed or unknown retry output')
    owner, admission, result = receipt['owner'], records['admission'], records['result']
    expected = ['/usr/bin/python3', '-B', str(retry.EXECUTOR), '--spec', spec_ref['path'],
        '--spec-sha256', spec_ref['sha256'], '--review', source_ref['path'], '--review-sha256', source_ref['sha256']]
    require(owner['argv'] == expected and old.owner_key(owner) not in
            {old.owner_key(p) for p in previous['history']['processes']},
            'Exact direct retry invocation and a distinct actual owner')
    require(owner == admission['owner'] == result['owner']
            and admission['spec_sha256'] == spec_ref['sha256'] and admission['source_review'] == source_ref
            and admission['parent_review_spec'] == retry.PARENT_SPEC
            and admission['failed_review_exit'] == result['failed_review_exit'] == spec['failed_review_exit']
            and admission['mode'] == result['mode'] == 'review'
            and admission['review'] is result['review'] is True
            and admission['performance_measurement'] is False and admission['budget'] == spec['budget'],
            'Actual retry admission, result and unchanged complete budget')
    require(result['complete'] is True and result['actual_self_exit_claimed'] is False
            and result['full_model_qualified'] is result['new_speedup_claim'] is False,
            'Successful saved-review result cannot claim its own exit or model qualification')
    prior = previous['history']
    require(records['predecessor_history'] == prior
            and admission['predecessor_history'] == receipt['predecessor_history']
            and admission['canonical_sha256'] == prior['canonical_sha256']
            and (len(prior['rows']),len(prior['processes']),len(prior['known_admitted_directories'])) == (1149,348,37),
            'Full original native and failed-review history, never a truncated snapshot')
    drain = records['predecessor_drain']
    require(admission['predecessor_drain_record'] == receipt['predecessor_drain']
            and admission['predecessor_drain'] == drain and drain['complete'] is True
            and (drain['services'],drain['processes']) == (1149,348)
            and [r['unit'] for r in drain['observations']] == [r['unit'] for r in prior['rows']]
            and all(len(r['samples']) == 2 for r in drain['observations']),
            'Every historical service/owner physically drained at retry admission')
    sources = records['source_files']
    require(sources['source_and_binary_files_authenticated'] is True
            and {p:r['sha256'] for p,r in sources['files'].items()} == spec['source_sha256'],
            'Complete actual retry source and dependency authentication ledger')
    frozen = records['frozen_contract']
    require(frozen['schema'] == 'dory_residual_post_exit_contract_v1'
            and frozen['producer_spec'] == spec['producer_spec'] and frozen['actual_exit'] == spec['native_exit']
            and frozen['raw_file_count'] == 6275 and frozen['raw_bytes'] == 6051509248
            and frozen['original_pointwise_failures'] == 7 and frozen['raw_payloads_authenticated'] is False
            and frozen['full_model_qualified'] is frozen['new_speedup_claim'] is False,
            'Original full post-exit frozen contract and diagnostic limits retained')
    saved = records['saved_review']
    old.saved_scope(saved)
    producer = read(spec['producer_spec'])
    require({p:r['sha256'] for p,r in saved['source_file_review']['files'].items()} == producer['source_sha256'],
            'Both archives and original complete producer sources independently reviewed')
    auth = records['query_authentication']
    query_scope(auth)
    observation = read(retry.QUERY_OBSERVATION)
    pins = observation['source_and_binary_pins']
    require(all(spec['source_sha256'].get(p) == row['sha256'] for p,row in pins.items())
            and all(p in pins for p in auth['final_dependency_audit']['authenticated_imports'].values()),
            'Every authenticated query import maps to the pinned dependency observation')
    gone(owner)
    history = copy.deepcopy(prior)
    append_owner(history, owner)
    require(OUT.name not in history['known_admitted_directories'], 'Retry directory appended once')
    history['known_admitted_directories'].append(OUT.name)
    actual_dirs = ({p.name for p in BASE.iterdir() if p.is_dir() and (p/'admission.json').is_file()}
                   if directories is None else set(directories))
    require(actual_dirs == set(history['known_admitted_directories'])
            and (len(history['processes']),len(actual_dirs)) == (349,38),
            'All historical and successful-retry owners/directories, no unknown admission')
    history.update(completed_residual_review=copy.deepcopy(actual_exit_ref),
        completed_residual_native=copy.deepcopy(spec['native_exit']),current_physical_release_verified=False)
    merged = {}
    merge(merged, producer['source_sha256'])
    merge(merged, spec['source_sha256'])
    return history, merged


def completed(actual_exit_ref):
    reader = Reader()
    receipt = reader(actual_exit_ref)
    external(receipt, 'saved-review-retry')
    spec = reader(receipt['spec'])
    retry.validate_spec(spec, reader)
    # Exclude only this exact already validated retry namespace from the prior
    # history view; all other unexpected admissions remain fatal.
    directories = {p.name for p in BASE.iterdir() if p.is_dir() and (p/'admission.json').is_file()}
    require(OUT.name in directories, 'Actual retry admission directory required')
    previous = retry.failed_history(spec['failed_review_exit'], reader, directories=directories-{OUT.name})
    externally_bound = {actual_exit_ref['path'], receipt['spec']['path']}
    require(all(path in externally_bound or spec['source_sha256'].get(path) == digest
                for path, digest in reader.pins.items()),
            'Every prior terminal record was pinned in the admitted retry source closure')
    history, pins = extend(previous, actual_exit_ref, reader, directories=directories)
    merge(pins, reader.pins)
    binaries = {p:s for p,s in pins.items() if Path(p).suffix == '.bin'}
    producer = reader(PRODUCER)
    require(binaries == {p:s for p,s in producer['source_sha256'].items() if Path(p).suffix == '.bin'}
            and len(binaries) == 6, 'Only the original six Dory evidence binaries')
    return dict(history=history,source_sha256=pins,historical_binary_sha256=binaries,
        metadata_files_read=len(reader.sizes),metadata_bytes_read=reader.total,
        tensor_payloads_read=False,fresh_physical_drain_verified=False,
        runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Requires actual failed r2 and successful r3 tool exits; no automatic launch')
