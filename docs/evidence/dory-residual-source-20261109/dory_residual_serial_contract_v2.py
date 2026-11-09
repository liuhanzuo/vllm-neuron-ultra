"""Fixed full-residual serial inputs, metadata admission and child ownership.

No executable specification is created by this module. Post-r9 receipts and a
successful complete metadata preflight must exist before native admission.
"""
import copy
import hashlib
import os
from pathlib import Path

from dory_first5_serial_contract_v1 import ORIGINAL, PLAN
from dory_layer0_serial_contract_v2 import require, read_pinned
from dory_residual_saved_complete_v1 import validate_producer
from dory_generation_execution_provenance_v1 import inspect_receipts, current_process

Q = Path(__file__).resolve().parent
PROPOSAL = dict(path=str(Q/'dory-residual-resource-proposal-20261109-v1.json'),
    sha256='6dfbd595c598745a10c30157ebb904b2755831fd38ab8ae127688aee9bf792ee')
RAW_BYTES = 5867691008
METADATA_BUDGET = dict(root_floor=28*2**30, nvme_floor=8*2**30,
    nvme_new_bytes=128*2**20, wall_seconds=3600, cpu_seconds=3400,
    address_space_bytes=4*2**30, single_file_bytes=64*2**20, log_bytes=32*2**20,
    cpus=[4], initial_memory_available_bytes=4*2**30, memory_available_floor_bytes=3*2**30)
MODES = ('metadata', 'native', 'review')


def load(ref):
    return read_pinned(ref['path'], ref['sha256'], 32*2**20)


def reviewed_ref(spec, ref):
    require(spec['source_sha256'].get(ref['path']) == ref['sha256'],
            'Exact completed metadata reference in reviewed source closure')
    return load(ref)


def validate(spec, mode):
    require(mode in MODES and spec['execution_mode'] == mode, 'Explicit serial mode')
    old, proposal = load(ORIGINAL), load(PROPOSAL)
    require(spec['schema'] == 'dory_residual_complete_replay_diagnostic_v1'
            and spec['diagnostic_only'] is True and spec['runtime_admitted'] is True
            and spec['review_only_spec'] is (mode == 'review'), 'Exact admitted residual diagnostic')
    for key in ('base', 'environment', 'official_source', 'checkpoint', 'checkpoint_sha256',
                'previous_loader_proof', 'previous_token_captures', 'tokenizer_directory', 'long_source'):
        require(spec[key] == old[key], 'Unchanged original '+key)
    for key, digest in old['source_sha256'].items():
        require(spec['source_sha256'].get(key) == digest, 'All original source/input pins retained')
    require(spec['selection_plan'] == PLAN and spec['resource_proposal'] == PROPOSAL,
            'Original full event selection and explicit resource proposal')
    for ref in (PLAN, PROPOSAL):
        require(spec['source_sha256'].get(ref['path']) == ref['sha256'], 'Pinned original plan/budget')
    for field, name in (('metadata_output', 'residual-metadata-r1'), ('output', 'residual-replay-r1'),
                        ('review_output', 'residual-replay-review-r1')):
        require(spec[field] == str(Path(old['base'])/name), 'Fresh fixed '+field)
    require(spec['worker'] == str(Q/'dory_residual_worker_v2.py')
            and spec['executor'] == str(Q/'run_dory_residual_serial_v2.py')
            and spec['reviewer'] == str(Q/'dory_residual_saved_complete_v1.py'), 'Reviewed concrete entries')
    budget = (METADATA_BUDGET if mode == 'metadata' else
              proposal['proposed_native_budget' if mode == 'native' else 'proposed_saved_review_budget'])
    require(spec['budget'] == budget and spec['expected_original_raw_bytes'] == RAW_BYTES
            and spec['payload_review_budget'] == proposal['proposed_payload_review_budget'],
            'Full original storage, memory and runtime budgets')
    needed = [str(Q/name) for name in ('dory_after_nemotron_r9_v2.py',
        'dory_residual_serial_contract_v2.py', 'dory_residual_worker_v2.py',
        'run_dory_residual_serial_v2.py', 'review_dory_layer0_serial_snapshot_v1.py',
        'run_dory_layer0_serial_v2.py')]
    require(set(needed) <= set(spec['essential_sources']) <= set(spec['source_sha256']),
            'Complete serial source closure')
    # Validate the complete producer closure even in metadata/review mode.
    producer = dict(spec, review_only_spec=False, budget=proposal['proposed_native_budget'])
    validate_producer(producer)
    require(all(spec[k] is False for k in ('full_model_qualified', 'new_speedup_claim', 'automatic_launch')),
            'Diagnostic preparation cannot claim qualification or auto-launch')
    if mode != 'metadata':
        ref = spec['metadata_preparation_exit']
        require(spec['source_sha256'].get(ref['path']) == ref['sha256'],
                'Actual completed metadata preflight required')
    return spec


def check_metadata_result(result):
    summary = result['summary']
    source = result['source_file_authentication']
    require(source['source_and_binary_files_authenticated'] is True
            and type(source['file_count']) is int and source['file_count'] > 0
            and type(source['total_bytes_read']) is int and source['total_bytes_read'] > 0
            and isinstance(source['bytes_by_suffix'], dict) and source['bytes_by_suffix']
            and all(isinstance(k, str) and type(v) is int and v >= 0
                    for k, v in source['bytes_by_suffix'].items())
            and sum(source['bytes_by_suffix'].values()) == source['total_bytes_read'],
            'Source-file authentication has an explicit complete byte ledger')
    require(result['complete'] is True and result['mode'] == 'metadata'
            and result['actual_self_exit_claimed'] is False and result['native_started'] is False
            and result['tensor_payload_validation_performed'] is False
            and 'raw_tensor_bytes_read' not in result
            and summary['raw_tensor_bytes_read'] == 0
            and summary['events'] == 6200 and summary['native_successful_calls_required'] == 6185
            and summary['native_rejections_required'] == 3 and summary['native_injections_required'] == 1
            and summary['selected_calls'] == 89 and summary['original_pointwise_failures'] == 7
            and summary['tensor_labels'] == 114 and summary['cache_tensor_labels'] == 112
            and summary['original_raw_archive_bytes'] == RAW_BYTES
            and summary['original_selection_plan'] == PLAN
            and summary['metadata_only'] is True and summary['native_execution'] is False
            and summary['raw_tensor_bytes_verified'] is False
            and result['full_model_qualified'] is result['new_speedup_claim'] is False,
            'Complete event metadata scope; source hashing is not tensor/model qualification')
    return result


def metadata_history(spec, gone):
    receipt = reviewed_ref(spec, spec['metadata_preparation_exit'])
    require(receipt['reviewer'] == '/root' and receipt['actual_external_exit_observed'] is True
            and type(receipt['session_id']) is int and receipt['session_id'] > 0
            and receipt['exit_code'] == 0 and receipt['tool_chunk'], 'Actual successful preflight tool exit')
    prep = validate(reviewed_ref(spec, receipt['spec']), 'metadata')
    admission = reviewed_ref(spec, receipt['admission'])
    result = check_metadata_result(reviewed_ref(spec, receipt['result']))
    sources = reviewed_ref(spec, receipt['source_files'])
    require(result['source_file_authentication'] == source_accounting(sources),
            'Metadata byte ledger matches actual admitted source-file authentication')
    require(receipt['spec']['sha256'] == admission['spec_sha256']
            and receipt['owner'] == admission['owner'] == result['owner']
            and admission['mode'] == 'metadata' and admission['performance_measurement'] is False
            and receipt['predecessor_history'] == admission['predecessor_history'], 'One admitted completed preflight')
    require(Path(receipt['admission']['path']) == Path(prep['metadata_output'])/'admission.json'
            and Path(receipt['result']['path']) == Path(prep['metadata_output'])/'result.json'
            and Path(receipt['source_files']['path']) == Path(prep['metadata_output'])/'source-files.json',
            'Exact successful preflight output namespace')
    h = copy.deepcopy(reviewed_ref(spec, receipt['predecessor_history']))
    require(h['schema'] == 'dory_post_nemotron_r9_history_v1' and len(h['rows']) == h['complete_count'] == 1149
            and len(h['r9_stages']) == 53 and h['r9_actual_exit'] == prep['r9_actual_exit']
            and h['r9_additional_actual_exits'] == prep['r9_additional_actual_exits']
            and h['r9_recovery_operations'] == prep['r9_recovery_operations'],
            'Actual post-r9 snapshot, never pre-r9 service/process totals')
    require(admission['canonical_sha256'] == h['canonical_sha256'], 'Original canonical snapshot')
    for path, digest in h['predecessor_source_sha256'].items():
        require(spec['source_sha256'].get(path) == digest, 'Post-r9 metadata closure retained')
    require(all((x['pid'], x['start_ticks']) != (receipt['owner']['pid'], receipt['owner']['start_ticks'])
                for x in h['processes']), 'Preflight owner appended exactly once')
    gone(receipt['owner'])
    h['processes'].append(receipt['owner'])
    h['known_admitted_directories'].append(Path(prep['metadata_output']).name)
    h['residual_metadata_preparation_exit'] = spec['metadata_preparation_exit']
    if spec['execution_mode'] == 'review':
        producer = reviewed_ref(spec, spec['producer_spec'])
        native = reviewed_ref(spec, spec['native_exit'])
        validate(producer, 'native')
        require(producer['output'] == spec['output']
                and producer['metadata_preparation_exit'] == spec['metadata_preparation_exit'], 'Same complete producer')
        root = Path(producer['output'])
        import json
        records = [json.loads((root/name).read_text()) for name in
                   ('admission.json', 'child.json', 'child-owner.json', 'child-exit.json', 'result.json')]
        inspect_receipts(spec['producer_spec']['sha256'], *records, native, process_lookup=current_process)
        require(load(records[0]['predecessor_history']) == h, 'Native used this exact post-r9/preflight history')
        h['processes'].extend([native['owner'], native['child_owner']])
        h['known_admitted_directories'].append(root.name)
    base = Path(spec['base'])
    actual = {x.name for x in base.iterdir() if x.is_dir() and (x/'admission.json').is_file()}
    require(actual == set(h['known_admitted_directories']), 'No missing or unknown intervening Dory admission')
    return h


def source_accounting(sources):
    require(sources['source_and_binary_files_authenticated'] is True and sources['files'],
            'Actual complete source-file authentication required')
    by_suffix = {}
    for name, row in sources['files'].items():
        require(type(row['bytes']) is int and row['bytes'] >= 0, 'Actual nonnegative file byte count')
        suffix = Path(name).suffix or '<none>'
        by_suffix[suffix] = by_suffix.get(suffix, 0)+row['bytes']
    require(sum(by_suffix.values()) == sources['total_bytes_read'], 'Exact source-file byte total')
    return dict(source_and_binary_files_authenticated=True, file_count=len(sources['files']),
                total_bytes_read=sources['total_bytes_read'], bytes_by_suffix=by_suffix)


def validate_child(spec_path, spec, admission, lookup, lock_check):
    validate(spec, 'native')
    wanted = [str(Path(spec['environment'])/'bin/python'), '-B', spec['worker'], '--spec', str(spec_path),
              '--admission', str(Path(spec['output'])/'admission.json')]
    actual = lookup(os.getpid())
    require(admission['spec_sha256'] == hashlib.sha256(Path(spec_path).read_bytes()).hexdigest()
            and admission['mode'] == 'native' and admission['review'] is False
            and admission['performance_measurement'] is False and admission['budget'] == spec['budget']
            and actual is not None and admission['command'] == wanted == actual['argv']
            and actual['ppid'] == admission['owner']['pid'], 'Actual admitted native parent/child command')
    lock_check(admission['owner'], admission['lock'])
    h, drain = load(admission['predecessor_history']), load(admission['predecessor_drain_record'])
    require(drain == admission['predecessor_drain'] and drain['complete'] is True
            and drain['services'] == len(h['rows']) == 1149
            and drain['processes'] == len(h['processes'])
            and h['residual_metadata_preparation_exit'] == spec['metadata_preparation_exit'],
            'Exact actual post-r9/preflight history and physical release, dynamic process count')
    return actual
