"""Prepare a saved-review retry only after an actual failed r2 tool exit.

The current r2 remains live. No receipt is fabricated here and no launch or
specification is emitted. Completed native replay and every original failure
remain in the chain; a failed saved review never becomes model qualification.
"""
import copy
from pathlib import Path
import re
import stat
import sys

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import nemotron9b_after_dory_residual_review_v1 as original
from dory_generation_execution_provenance_v1 import inspect_receipts, owner_key

Q, BASE = original.Q, original.BASE
require = original.require
PARENT_SPEC = dict(path=str(Q/'dory-residual-saved-review-r2-runtime-spec-20261109-v1.json'),
    sha256='957360f389de29cd3ff29ce7e2711b79691c02da370f6f973945270d9faa44bc')
PARENT_REVIEW = dict(path=str(Q/'dory-residual-saved-review-r2-source-review-20261109-v1.json'),
    sha256='94b68d5721a9dab5a0da9a0b454cae804246f7d295aa9545b50c3d590b3e6bcf')
OUT = BASE/'residual-replay-review-r2'
NEXT_OUT = BASE/'residual-replay-review-r3'
EXECUTOR = Q/'run_dory_saved_review_retry_v1.py'
ANCHORS = {
    'admission.json': '448ca54c27f9c5a7fc3ebfabeeb0c779982052da425782c4c44df1ce13b3f1d2',
    'predecessor-history.json': '94df382063d8c8b8723094870c03addcc4d048651a56ddea28cc06267a3a628b',
    'source-files.json': '17da2947b7f76e45182edd4440d963582ed34e6c71e02e098bcd35eaa627ba04',
    'predecessor-drain.json': '8bca57fa83446964c58e614b8a22e9a8f0eba8aac593d49f04b335a1538c657d',
    'post-exit-contract.json': 'cf2359e65cce6ae077a3967a75c46c0c1822704a3299e564c09e7e51f80ccdd3',
}
ALLOWED_OUTPUTS = set(ANCHORS) | {'failure.json', 'saved-residual-review.json', 'result.json'}
QUERY_OBSERVATION = dict(path=str(Q/'dory-service-query-dependency-observation-20261109-v1.json'),
    sha256='11de174a895bf018dc209814be7d349b7c7cf88413038fd29d2537eeae8e41c4')
SCHEMA = 'dory_saved_review_retry_v1'


def validate_failed_receipt(receipt, owner):
    require(receipt['schema'] == 'dory_residual_saved_review_r2_failed_actual_exit_v1'
            and receipt['reviewer'] == '/root'
            and receipt['actual_external_exit_observed'] is True
            and receipt['session_id'] == 86437
            and type(receipt['exit_code']) is int and receipt['exit_code'] != 0
            and isinstance(receipt['tool_chunk'], str)
            and re.fullmatch('[0-9a-f]+', receipt['tool_chunk']),
            'Actual nonzero original saved-review tool exit required')
    require(receipt['owner'] == owner and owner_key(owner) == (36934, '403068361')
            and receipt['spec'] == PARENT_SPEC and receipt['source_review'] == PARENT_REVIEW,
            'Exact originally observed saved-review owner and inputs')
    wanted = ['/usr/bin/python3', '-B', str(Q/'run_dory_residual_serial_v4.py'),
        '--mode', 'review', '--spec', PARENT_SPEC['path'], '--spec-sha256', PARENT_SPEC['sha256'],
        '--review', PARENT_REVIEW['path'], '--review-sha256', PARENT_REVIEW['sha256']]
    require(owner['argv'] == wanted and owner['ppid'] == 433847
            and owner['cgroup'] == '0::/user.slice/user-1000.slice/session-56.scope',
            'Exact actual original invocation and owner identity')
    outputs = receipt['output_files']
    require(set(ANCHORS) <= set(outputs) <= ALLOWED_OUTPUTS,
            'Every original output preserved; no unknown output omitted')
    for name, ref in outputs.items():
        require(set(ref) == {'path', 'sha256'} and ref['path'] == str(OUT/name)
                and isinstance(ref['sha256'], str) and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
                'Exact retained failed-review output reference')
        if name in ANCHORS:
            require(ref['sha256'] == ANCHORS[name], 'Original admitted output pin cannot be rebound')
    require(receipt['full_model_qualified'] is receipt['new_speedup_claim'] is False
            and receipt['saved_review_completed'] is False,
            'Failed external exit cannot certify saved review or a model')
    return outputs


def ordinary_outputs():
    names = set()
    for path in OUT.iterdir():
        info = path.lstat()
        require(path.resolve() == path and stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1 and 0 < info.st_size <= 32*2**20,
                'All failed-review outputs remain bounded ordinary files')
        names.add(path.name)
    return names


def failed_history(exit_ref, read, *, gone=original.low.gone,
                   lookup=original.current_process, output_names=None, directories=None):
    """Authenticate only terminal JSON; future lock owner must physically drain."""
    receipt = read(exit_ref)
    admission = read(dict(path=str(OUT/'admission.json'), sha256=ANCHORS['admission.json']))
    output_refs = validate_failed_receipt(receipt, admission['owner'])
    # Refuse a live exact owner even if an input receipt claims a failed exit.
    gone(admission['owner'])
    names = ordinary_outputs() if output_names is None else set(output_names)
    require(names == set(output_refs), 'Complete exact failed-review output inventory')
    outputs = {name: read(ref) for name, ref in output_refs.items()}
    parent = original.validate(read(PARENT_SPEC), 'review')
    review = read(PARENT_REVIEW)
    require(review['reviewer'] == '/root' and review['passed'] is True and review['blockers'] == []
            and review['spec_sha256'] == PARENT_SPEC['sha256']
            and review['source_sha256'] == parent['source_sha256'],
            'Exact original reviewed source closure')
    require(admission == outputs['admission.json']
            and admission['spec_sha256'] == PARENT_SPEC['sha256']
            and admission['mode'] == 'review' and admission['review'] is True
            and admission['performance_measurement'] is False and admission['budget'] == parent['budget']
            and parent['review_output'] == str(OUT), 'Original admitted saved-review scope')
    start = original.anchor(read)
    require(parent['producer_spec'] == original.PRODUCER
            and all(parent['source_sha256'].get(p) == digest
                    for p, digest in start['spec']['source_sha256'].items()),
            'All original native producer pins retained')
    native_ref = parent['native_exit']
    require(parent['source_sha256'].get(native_ref['path']) == native_ref['sha256'],
            'Actual successful native exit remains source-pinned')
    native = read(native_ref)
    original.external(native, 'native')
    require(native['spec'] == original.PRODUCER and native['owner'] == start['admission']['owner']
            and native['child_owner'] == start['child'], 'Exact completed native producer owners')
    def producer_record(name):
        path = str(Path(start['spec']['output'])/name)
        require(path in parent['source_sha256'], 'Every producer terminal record remains pinned')
        return read(dict(path=path, sha256=parent['source_sha256'][path]))
    records = [producer_record(n) for n in ('admission.json', 'child.json', 'child-owner.json',
                                           'child-exit.json', 'result.json')]
    require(records[0] == start['admission'] and records[2] == start['child']
            and producer_record('predecessor-history.json') == start['history'],
            'Actual native terminal records equal their original anchor')
    inspect_receipts(original.PRODUCER['sha256'], *records, native, process_lookup=lookup)
    require(not any((Path(start['spec']['output'])/n).exists() for n in
            ('failure.json', 'cleanup-pending.json', 'diagnostic/residual-replay-failure.json')),
            'Native failure cannot be promoted by a retry')
    history = copy.deepcopy(start['history'])
    for owner in (native['owner'], native['child_owner']):
        gone(owner)
        original.append_owner(history, owner)
    history['known_admitted_directories'].append(Path(start['spec']['output']).name)
    require(outputs['predecessor-history.json'] == history
            and admission['predecessor_history'] == output_refs['predecessor-history.json']
            and admission['canonical_sha256'] == history['canonical_sha256'],
            'Original saved review followed the exact complete native history')
    drain = outputs['predecessor-drain.json']
    require(admission['predecessor_drain_record'] == output_refs['predecessor-drain.json']
            and admission['predecessor_drain'] == drain and drain['complete'] is True
            and (drain['services'], drain['processes']) == (1149, 347),
            'Original admitted drain record is preserved')
    source_files = outputs['source-files.json']
    require(source_files['source_and_binary_files_authenticated'] is True
            and {p: row['sha256'] for p, row in source_files['files'].items()} == parent['source_sha256'],
            'Original complete source-file authentication remains recorded')
    original.append_owner(history, admission['owner'])
    require(OUT.name not in history['known_admitted_directories'], 'Failed review owner/directory appended once')
    history['known_admitted_directories'].append(OUT.name)
    actual = ({p.name for p in BASE.iterdir() if p.is_dir() and (p/'admission.json').is_file()}
              if directories is None else set(directories))
    require(actual == set(history['known_admitted_directories'])
            and (len(history['rows']), len(history['processes']), len(actual)) == (1149, 348, 37),
            'All native and failed-review services, owners and admitted directories retained')
    history.update(failed_residual_saved_review=copy.deepcopy(exit_ref),
                   completed_residual_native=copy.deepcopy(native_ref),
                   current_physical_release_verified=False)
    return dict(history=history, parent=parent, receipt=receipt, output_refs=output_refs,
                failed_review_complete=False, raw_tensor_bytes_read=0,
                full_model_qualified=False, new_speedup_claim=False)


def validate_spec(spec, read):
    parent = original.validate(read(PARENT_SPEC), 'review')
    require(spec['schema'] == SCHEMA and spec['parent_review_spec'] == PARENT_SPEC
            and spec['output'] == str(NEXT_OUT) and spec['executor'] == str(EXECUTOR)
            and spec['reviewer'] == parent['reviewer']
            and spec['producer_spec'] == parent['producer_spec']
            and spec['native_exit'] == parent['native_exit']
            and spec['budget'] == parent['budget']
            and spec['query_dependency_observation'] == QUERY_OBSERVATION,
            'Exact retry entries, producer, query observation and unchanged budgets')
    require(spec['diagnostic_only'] is True and spec['runtime_admitted'] is True
            and all(spec[k] is False for k in ('automatic_launch', 'full_model_qualified', 'new_speedup_claim')),
            'Separately reviewed diagnostic retry cannot claim model or automatic admission')
    pins = spec['source_sha256']
    require(all(pins.get(p) == sha for p, sha in parent['source_sha256'].items()),
            'Every original source/input pin retained without rebinding')
    for ref in (PARENT_SPEC, PARENT_REVIEW, QUERY_OBSERVATION, spec['failed_review_exit']):
        require(pins.get(ref['path']) == ref['sha256'], 'All exact retry prerequisites are source-pinned')
    required = {str(Q/n) for n in ('dory_saved_review_retry_contract_v1.py',
        'run_dory_saved_review_retry_v1.py', 'dory_service_query_auth_v1.py',
        'dory_saved_review_service_query_v1.py', 'dory_saved_review_guard_v1.py')}
    require(required <= set(spec['essential_sources']) == set(pins),
            'Complete original and successor source closure required')
    require(type(spec['source_read_cap_bytes']) is int
            and parent['source_read_cap_bytes'] <= spec['source_read_cap_bytes']
            <= parent['source_read_cap_bytes'] + 64*2**20,
            'Bounded original source authentication plus retry metadata/dependencies')
    return parent


if __name__ == '__main__':
    raise SystemExit('No launch or receipt creation; requires an actual failed saved-review exit')
