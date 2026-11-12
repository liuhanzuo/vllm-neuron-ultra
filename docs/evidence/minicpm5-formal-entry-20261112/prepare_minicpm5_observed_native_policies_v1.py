"""Prepare unaccepted native policy inputs from the real completed r2 run.

JSON/source metadata only. Never sets reviewer=/root or accepted=True, launches
an entry, claims current library bytes, or replaces full diagnostic binding.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path[:0] = [str(ROOT), str(Q)]
from research import minicpm5_bootstrap_observation_v2 as discovery
from research import minicpm5_sdk_mapping_history_gate_v1 as history

ACTUAL_EXIT = dict(path=str(Q / 'minicpm5-bootstrap-observation-r2-root-actual-exit-20261112-v1.json'),
    sha256='33ee19570567fe19ba3abf63c533dccb956857f5e9ce5679ddef627ecfa4fba5')
NEW_SOURCES = (
    'minicpm5_sdk310_composite_origins_v2.py',
    'minicpm5_sdk_mapping_native_authenticator_v3.py',
    'minicpm5_sdk_mapping_worker_native_v4.py',
    'minicpm5_sdk_mapping_worker_source_entry_v5.py',
    'minicpm5_sdk_mapping_outer_source_entry_v5.py',
    'minicpm5_sdk_mapping_nocache_entry_v6.py',
    'prepare_minicpm5_observed_native_policies_v1.py',
)
require = discovery.require


def build(read, complete_pins, source_manifest_ref):
    """Return input drafts; later original runtime reconstructs full pins again."""
    ended = read(ACTUAL_EXIT)
    require(ended['schema'] == 'minicpm5_bootstrap_observation_root_actual_exit_v2'
            and ended['reviewer'] == '/root'
            and ended['actual_tool'] == dict(kind='tool_session', session_id=50185,
                                            terminal_chunk='bf14fc', exit_code=0)
            and ended['actual_tool_exit_observed'] is True and ended['physically_drained'] is True
            and ended['owners_gone'] is True and ended['lock_reacquired_and_released'] is True
            and ended['lock_identity'] == dict(device=66306, inode=8294421)
            and ended['full_six_raw_maps_replayed'] is True
            and ended['source_execution_gaps_recomputed'] is True,
            'Exact completed original dual-bootstrap and independent drain/map audit')
    require(all(ended[k] is False for k in ('accepted_native_authority_created',
            'native_ABI_qualified', 'full_model_qualified', 'new_speedup_claim')),
            'Completed discovery cannot become native ABI or model qualification')
    result = read(ended['result'])
    require(result['complete'] is True and result['schema'] == 'minicpm5_bootstrap_observation_result_v2'
            and result['role_order'] == ['sdk310', 'outer312']
            and result['plan'] == ended['plan'] and result['review'] == ended['review'],
            'Actual result and reviewed plan remain exact')
    require(all(result[k] is False for k in ('accepted_native_authority_created',
            'native_ABI_qualified', 'full_model_qualified', 'new_speedup_claim')),
            'Actual result still makes no native ABI or model qualification claim')
    plan, review = read(result['plan']), read(result['review'])
    discovery.validate(plan, review, result['plan'])
    table = discovery.candidates(read)
    manifest = read(source_manifest_ref)
    require(manifest['schema'] == 'minicpm5_observed_formal_entry_source_manifest_v1'
            and set(manifest['new_runtime_source_sha256']) == {str(Q / name) for name in NEW_SOURCES},
            'Exact local integration sources, with no implicit runtime admission')
    for source_map in (plan['source_sha256'], manifest['retained_source_sha256'],
                       manifest['new_runtime_source_sha256']):
        require(all(complete_pins.get(path) == sha for path, sha in source_map.items()),
                'Every original and new selected source retained in complete prospective map')
    require(type(complete_pins) is dict and len(complete_pins) > len(plan['source_sha256']),
            'Bootstrap source subset is not the full original diagnostic source map')
    for role in ('sdk310', 'outer312'):
        for path, row in table[role]['pure_source_candidates'].items():
            require(path not in complete_pins or complete_pins[path] == row['sha256'],
                    'Original signed nonlocal pure-source policy never replaces inherited pins')
    policies = {}
    for row in result['children']:
        role = row['role']
        observed = read(row['result'])
        require(all(observed[k] is False for k in ('accepted_native_authority_created',
                'native_ABI_qualified', 'full_model_qualified', 'new_speedup_claim')),
                'Role observation cannot claim a native/model success')
        require(observed['role'] == role and observed['complete'] is True
                and observed['plan'] == result['plan']
                and observed['missing_content_origins'] is False
                and observed['source_execution_provenance_gaps'] == []
                and row['missing_content_origins'] is False
                and row['source_execution_provenance_gaps'] == []
                and observed['ffi_failure'] is None and observed['ffi_definitions'] is not None,
                'Actual completed role has no observed source/native content gap')
        audited = ended['selected_providers_and_gaps'][role]
        for key in ('selected_stdlib_providers', 'map_summaries', 'known_pure_source_observations',
                    'ffi_failure', 'missing_content_origins', 'types_source_observation',
                    'source_execution_provenance_gaps'):
            require(audited[key] == observed[key], 'Full saved role equals root independent audit')
        mappings = observed['map_summaries']
        require(set(mappings) == {'earliest', 'pre_ctypes', 'post_ctypes'}
                and all(item['gaps'] == [] and item['accepted_native_authority_created'] is False
                        for item in mappings.values()), 'All three stages remain complete discovery evidence')
        before = mappings['pre_ctypes']['known_selected_closure']['paths']
        after = mappings['post_ctypes']['known_selected_closure']['paths']
        require(len(before) == len(set(before)) and len(after) == len(set(after)),
                'Unique observed native provider sets')
        selected = table[role]
        require(set(after) <= set(selected['native_candidates']),
                'Only actual mapped known original candidates may become prospective policy inputs')
        if role == 'sdk310':
            def member(row):
                return {k: row[k] for k in ('path', 'bytes', 'sha256', 'package', 'version')}
            require(set(before) < set(after) and selected['ctypes_extension'] in set(after) - set(before),
                    'Actual separate dynamic ctypes transition')
            policies[role] = dict(schema='minicpm5_sdk310_native_source_review_v3',
                role=role, python=selected['python'], executable=selected['executable'],
                ctypes_root='/usr/lib/python3.10/ctypes', ctypes_extension=selected['ctypes_extension'],
                ffi_provider=selected['ffi_candidate'], package_origins=copy.deepcopy(table['package_origins']),
                native_files={path: member(selected['native_candidates'][path]) for path in after},
                pure_sources={path: member(value) for path, value in selected['pure_source_candidates'].items()},
                mapped_before_ctypes=list(before), mapped_after_ctypes=list(after))
        else:
            types = observed['types_source_observation']
            require(set(before) == set(after) == set(selected['native_candidates'])
                    and types['executed_through_pinned_loader'] is True
                    and types['original_execution_provenance_gap'] is False
                    and types['preexisting_module_reloaded'] is False,
                    'Actual original outer closure and completed pinned types execution')
            policies[role] = dict(schema='minicpm5_sdk_mapping_native_source_review_v1',
                origin_evidence=copy.deepcopy(table['input_refs']['outer_saved_origin']),
                native_files=copy.deepcopy(selected['native_candidates']))
            # Original authority contract pins the path/hash pair exactly.
            policies[role]['origin_evidence'] = {k: policies[role]['origin_evidence'][k]
                                               for k in ('path', 'bytes', 'sha256')}
        policies[role].update(reviewer=None, accepted=False,
            source_map_sha256=history.digest(complete_pins),
            observed_bootstrap_result=copy.deepcopy(row['result']),
            observed_bootstrap_actual_exit=copy.deepcopy(ACTUAL_EXIT))
    require(set(policies) == {'sdk310', 'outer312'}, 'Both distinct native policies prepared')
    return dict(schema='minicpm5_observed_native_policy_inputs_v1', actual_bootstrap_exit=copy.deepcopy(ACTUAL_EXIT),
        nonlocal_pure_source_sha256={role: {path: row['sha256'] for path, row in
            table[role]['pure_source_candidates'].items()} for role in ('sdk310', 'outer312')},
        nonlocal_sources_authenticated_by_original_origin_policy=True,
        source_manifest=copy.deepcopy(source_manifest_ref), prospective_source_map_sha256=history.digest(complete_pins),
        policies=policies, source_metadata_only=True, runtime_admitted=False,
        complete_original_runtime_source_map_authenticated_here=False,
        current_native_bytes_read_here=False, actual_current_maps_observed_here=False,
        accepted_native_authority_created=False, native_ABI_qualified=False,
        full_model_qualified=False, new_speedup_claim=False,
        pending=['Root binds/reviews complete original diagnostic source map and current serial history.',
                 'Root reviews separate native policies and explicit outer/SDK entry plans.',
                 'Actual run rehashes current whole files, full signed indexes, ELF and maps, checks ABI and original nine.',
                 'Original fixed-input 1284-row scientific plan, model cold/long/slot/quality/HTTP gates remain unchanged.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('complete-source-map', 'source-manifest'):
        parser.add_argument('--' + name, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    def read(reference):
        path = Path(reference['path'])
        require(path.suffix == '.json' and path.is_absolute() and path.resolve() == path,
                'Only canonical saved JSON metadata')
        before = path.stat()
        require(0 < before.st_size <= 16 * 2**20, 'Bounded complete JSON metadata')
        data = path.read_bytes()
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after) and len(data) == before.st_size
                and hashlib.sha256(data).hexdigest() == reference['sha256']
                and reference.get('bytes', len(data)) == len(data), 'Entire exact stable metadata')
        return json.loads(data)
    source_map = dict(path=args.complete_source_map, sha256=args.complete_source_map_sha256)
    manifest = dict(path=args.source_manifest, sha256=args.source_manifest_sha256)
    value = build(read, read(source_map), manifest)
    value['prospective_source_map'] = source_map
    destination = Path(args.destination)
    require(destination.parent == Q and destination.suffix == '.json'
            and destination.name.startswith('minicpm5-observed-native-policy-inputs-'),
            'One new unaccepted policy input package only')
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    require(len(raw) <= 2**20, 'Small policy preparation metadata')
    with destination.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(path=str(destination), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())))


if __name__ == '__main__':
    main()
