"""Validate completed Smol producers without rewriting their release schemas.

Pure metadata validation: the package coordinator still must perform its fresh
physical drain. Historical failure, observer completion, and model qualification
remain different facts. No framework, payload, process or service I/O here.
"""
import copy
from pathlib import Path
import re
from minicpm5_layer0_runtime_auth_v1 import require
from nemotron9b_completed_smol_metadata_v1 import catalog_number

FIELDS = ('execution_spec', 'exited', 'release', 'terminal', 'live', 'controller', 'supervisor')

def validate(spec, catalog, load):
    selected = spec.get('required_smol_predecessor_chain')
    require(type(selected) is list and 1 <= len(selected) <= 16, 'Explicit bounded completed Smol chain')
    current, current_ref = catalog, spec['final_predecessor_catalog']
    result = []; references = []
    def document(ref):
        require(type(ref) is dict and set(ref) == {'path', 'sha256'} and
                type(ref['path']) is str and type(ref['sha256']) is str and
                re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'Exact predecessor document reference')
        require(spec['source_sha256'].get(ref['path']) == ref['sha256'], 'Explicit predecessor document source pin')
        value = load(ref); references.append(copy.deepcopy(ref)); return value
    for item in selected:
        number = catalog_number(current_ref['path'])
        require(number is not None, 'Selected producer is a completed Smol catalog')
        require(type(item) is dict and set(item) == {'catalog', 'terminal_audit', *FIELDS}, 'Exact Smol chain selection fields')
        require(item['catalog'] == current_ref and item['terminal_audit'] == current['appended_terminal_audit'], 'Latest actual Smol producer selected')
        require(current['schema'] == f'root_post_smollm3_sio{number}_complete_predecessors_v1' and
                current['reviewer'] == '/root' and current['root_finalized'] is True and
                current['all_planned_launches_accounted_for'] is True, 'Finalized Smol catalog')
        require(spec['source_sha256'].get(current_ref['path']) == current_ref['sha256'], 'Current catalog source pin')
        audit = document(item['terminal_audit'])
        require(audit['schema'] == f'smollm3_sio{number}_root_terminal_audit_v1' and
                audit['reviewer'] == '/root' and audit['passed'] is True and audit['all_groups_drained'] is True and
                audit['all_planned_launches_accounted_for'] is True, 'Actual root terminal audit and recorded drain')
        require(all(audit['references'][key] == item[key] for key in FIELDS), 'Exact original audit references')
        values = {key: document(item[key]) for key in FIELDS}
        execution, exited, release, terminal, live, controller, supervisor = (values[key] for key in FIELDS)
        require(current['appended_from'] == audit['references']['previous_catalog'] == execution['final_predecessor_catalog'], 'Exact previous catalog reference')
        previous_ref = current['appended_from']; previous = document(previous_ref)
        prior_rows, prior_processes = previous['rows'], previous['processes']
        rows, processes = current['rows'], current['processes']
        require(rows[:len(prior_rows)] == prior_rows and processes[:len(prior_processes)] == prior_processes, 'Retain every historical row and process')
        require(len(rows) == len(prior_rows) + 25 == current['complete_count'] == audit['complete_rows'] and
                len(processes) == len(prior_processes) + 1 == audit['complete_processes'], 'Actual complete Smol extension counts')
        require(len({row['unit'] for row in rows}) == len(rows) and
                [row['unit'] for row in rows[len(prior_rows):]] == execution['planned_units'], 'Exact 25 planned units')
        require(all(current['source_sha256'].get(k) == v for k, v in previous['source_sha256'].items()), 'Retain all historical source pins')
        require(all(spec['source_sha256'].get(k) == v for k, v in current['source_sha256'].items()), 'Full completed Smol source closure')
        require(current['root_release_refs'] == previous['root_release_refs'], 'Keep original Nemotron release history without schema relabeling')
        owner = live['owner']
        require(owner == exited['owner'] == release['external_owner'] == terminal['external_owner'], 'Same actual external producer')
        require(live['reviewer'] == exited['reviewer'] == '/root' and live['actual_exit_observed'] is False and
                live['execution_spec'] == exited['execution_spec'] == item['execution_spec'], 'Original live/exit execution binding')
        require(processes[-1] == dict(owner, role='external_launcher_and_final_auditor',
                self_exit_not_yet_observed=True, next_admission_must_prove_gone=True), 'Original external process history retained')
        require(exited['actual_exit_observed_by_root'] is True and exited['terminal'] is True and exited['gone'] is True and
                exited['release'] == item['release'], 'Independent actual external exit')
        handle = exited['execution_handle']
        require(type(exited['exit_code']) is int and handle['kind'] == 'tool_session' and
                type(handle['session_id']) is int and handle['session_id'] == live['session'] and
                handle['actual_completion_observed'] is True and handle['exit_code'] == exited['exit_code'] and
                type(handle['tool_chunk']) is str and re.fullmatch('[0-9a-f]{6}', handle['tool_chunk']), 'Actual tool completion authority')
        require(release['released'] is True and release['all_planned_groups_drained'] is True and
                release['next_admission_must_verify_external_exit'] is True and
                terminal['all_planned_groups_drained'] is True and terminal['external_self_exit_not_claimed'] is True, 'Original release schema and bounded self-claim')
        complete = terminal['scientific_error'] is None and controller['protocol_completed'] is True
        props = supervisor['properties']
        original_zero = props['Result'] == 'success' and props['ExecMainCode'] == '1' and props['ExecMainStatus'] == '0'
        lifecycle = terminal['original_error'] is None and original_zero and controller['original_error'] is None and controller['cleanup_errors'] == []
        qualified = complete and lifecycle
        require(terminal['original_supervisor_exit_zero'] is original_zero and terminal['protocol_completed'] is complete and
                terminal['lifecycle_qualified'] is lifecycle and terminal['qualified'] is qualified and
                audit['experiment_qualified'] is qualified and audit['protocol_completed'] is complete and
                exited['exit_code'] == audit['actual_external_exit_code'] == (0 if qualified else 1) and
                release['failed'] is (not qualified), 'Original success and failure outcomes remain exact')
        require(audit['original_controller_error'] == terminal['original_controller_error'] == controller['original_error'] and
                audit['original_controller_cleanup_errors'] == terminal['original_controller_cleanup_errors'] == controller['cleanup_errors'], 'Preserve original errors')
        require(audit['observer_api_invariance_passed'] == terminal['observer_api_invariance_passed'] and
                (type(terminal['observer_api_invariance_passed']) is bool if complete else terminal['observer_api_invariance_passed'] is None), 'No invented observer invariance')
        require(audit['full_model_qualified'] is False and audit['new_speedup_claim'] is False and
                terminal['full_model_qualified'] is False and terminal['new_speedup_claim'] is False, 'Observer diagnostic grants no model speedup')
        result.append(dict(catalog=copy.deepcopy(current_ref), actual_exit=copy.deepcopy(item['exited']),
            terminal_audit=copy.deepcopy(item['terminal_audit']), original_exit_code=exited['exit_code'],
            experiment_qualified=qualified, protocol_completed=complete,
            full_model_qualified=False, new_speedup_claim=False))
        current, current_ref = previous, previous_ref
    require(catalog_number(current_ref['path']) is None, 'No omitted completed Smol predecessor')
    require(spec['required_predecessor_releases'] == current['root_release_refs'] == catalog['root_release_refs'], 'Unchanged older release chain')
    return dict(producers=result, predecessor_anchor=copy.deepcopy(current_ref), references=references,
                source_only=True, fresh_physical_drain_proven=False, scientific_success_inferred=False)
