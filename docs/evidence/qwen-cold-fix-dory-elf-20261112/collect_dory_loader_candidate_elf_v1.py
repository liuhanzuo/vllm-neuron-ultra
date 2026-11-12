"""Finite Dory three-object ELF metadata successor; inert until root admission.

Reuses collect_dory_loader_route_sources_v2 retained-lock/history/physical and
empty-child drain structure. Only three explicitly pinned native files may be
read, as data; no native interpreter/packages, readelf/ldconfig, subprocesses,
gzip, archives, directory search or automatic candidate expansion. Existing24
ELF records and the27 content candidates come from saved authenticated JSON.

No current plan, catalog, handoff, admission, or physical binding is prepared by
this source delivery. The future root review must bind content_scope, input_table,
full local source pins, current serial history, original lock policy and budgets.
Actual loader selection/ABI/full-model claims remain false even if this finishes.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
BUDGET = dict(address_space_bytes=2 * 2**30, cpu_seconds=120, wall_seconds=180,
              single_file_bytes=8 * 2**20, output_bytes=16 * 2**20,
              log_bytes=8 * 2**20, cpus=[5], nice=19,
              root_floor_bytes=28 * 2**30, nvme_floor_bytes=30 * 2**30,
              memory_available_floor_bytes=2**30)
INPUT_TABLE = dict(path=str(Q/'dory-loader-candidate-elf-inputs-20261111-v1.json'),
                   bytes=79651, sha256='db254b660a0f29effc9e4bb1ee0784a7dbf66177a632a94b2e600ab713123fe6')
POLICY = {'schema': 'dory_loader_candidate_ELF_scope_v1', 'candidate_files': 27, 'new_native_files': 3, 'new_native_total_bytes': 2422457, 'new_native_read_cap_bytes': 4194304, 'new_native_per_file_cap_bytes': 4194304, 'metadata_input_total_cap_bytes': 33554432, 'metadata_input_per_file_cap_bytes': 16777216, 'graph_edge_cap': 1755, 'uncovered_item_cap': 10000, 'filesystem_enumeration': False, 'automatic_candidate_expansion': False, 'native_or_target_execution': False, 'readelf_subprocess': False, 'all_fifteen_runtime_selections_must_be_resolved': False}
ENTRYPOINT_SOURCES = {
    str(Q / 'dory_stage_a_native_resolver_v4.py'): 'e872f26fafd78e72b77847353a782a526e4093d6575ee033b94c4d4298bd1eb9',
    str(Q / 'dory_serial_stage_worker_v4.py'): 'ef56bb3049fd352a576c380e45094b0248882407b7532484c3b87952b97fe614',
    str(Q / 'dory_stage_a_capture_core_v1.py'): '9341ac49a187638a6c5eec2eb989238bb04cfcf37accd481c38a3c4f6555cd17',
    str(Q / 'dory_stage_a_worker_preparation_v1.py'): 'aae622346332916fddd3ec84e9da3bad1241f0ad64a5129d797f60f7b01a2ddd',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def validate_reference(ref):
    require(type(ref) is dict and {'path', 'sha256'} <= set(ref) <= {'path', 'sha256', 'bytes'},
            'Exact metadata reference fields')
    path = ref['path']
    require(type(path) is str and Path(path).is_absolute() and str(Path(path)) == path
            and '\\' not in path and '\0' not in path
            and all(part not in ('', '.', '..') for part in path.split('/')[1:]),
            'Canonical absolute reference spelling')
    require(type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}', ref['sha256']),
            'Complete expected SHA256')
    require('bytes' not in ref or type(ref['bytes']) is int and ref['bytes'] >= 0,
            'Nonnegative exact reference length')


def validate_plan(plan, review, plan_ref, policy, content_scope):
    require(plan['schema'] == 'dory_loader_candidate_ELF_plan_v1'
            and plan['runtime_bound'] is True and plan['automatic_successor_launch'] is False,
            'One fresh root-bound finite three-object metadata collection')
    out = Path(plan['output_root'])
    require(out.parent == Path('/mnt/nvme') and str(out) == plan['output_root']
            and re.fullmatch(r'dory-loader-candidate-elf-[a-zA-Z0-9-]+', out.name),
            'Fixed fresh output namespace')
    require(plan['budget'] == BUDGET and plan['lock_policy'] == policy,
            'Exact bounded resource and current descriptor policies')
    require(review['schema'] == 'dory_loader_candidate_ELF_review_v1'
            and review['reviewer'] == '/root' and review['passed'] is True
            and review['blockers'] == [] and review['plan'] == plan_ref,
            'Exact accepted root source review')
    for name in ('source_sha256', 'lock_policy', 'current_serial_catalog',
                 'current_handoff', 'predecessor_exit'):
        require(review[name] == plan[name], 'Exact reviewed ' + name)
    require(review['latest_handoff_accepted'] is True
            and review['current_physical_history_accepted'] is True,
            'Root accepts latest explicit history')
    for name in ('current_serial_catalog', 'current_handoff', 'predecessor_exit'):
        validate_reference(plan[name])
        require(Path(plan[name]['path']).suffix == '.json', 'History inputs are JSON metadata only')
    require(type(plan['predecessor_expected_exit_code']) is int
            and type(plan['predecessor_owners']) is list and 0 < len(plan['predecessor_owners']) <= 20000
            and type(plan['predecessor_services']) is list and len(plan['predecessor_services']) <= 10000,
            'Bounded actual predecessor history')
    require(str(Path(__file__).resolve()) in plan['source_sha256'], 'Collector is included in source closure')
    require(plan['content_scope'] == review['content_scope'] == content_scope,
            'Exact reviewed three-object content scope; no automatic expansion')
    require(str(Q / 'dory_loader_candidate_elf_v1.py') in plan['source_sha256'],
            'Pure ELF helper included in authenticated source closure')
    require(all(plan['source_sha256'].get(path) == wanted
                for path, wanted in ENTRYPOINT_SOURCES.items()),
            'Original Stage A entrypoint and two-loop/56-history/TP1 source pins retained')
    require(plan['input_table'] == review['input_table'] == INPUT_TABLE
            and review['candidate_content_sources_reviewed'] is True,
            'Exact independent candidate-source review; runtime binding gaps may remain')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'review', 'review-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(sys.executable == '/usr/bin/python3' and sys.dont_write_bytecode
            and sys.flags.no_user_site and __debug__, 'System Python -B -s with assertions only')
    require(not any(name.split('.')[0] in ('torch', 'vllm', 'numpy', 'transformers', 'tokenizers')
                    for name in sys.modules), 'No target packages in ELF metadata collector')
    sys.path[:0] = [str(ROOT), str(Q)]
    import dory_serial_stage_contract_v2 as c
    plan_ref = dict(path=args.plan, sha256=args.plan_sha256)
    review_ref = dict(path=args.review, sha256=args.review_sha256)
    validate_reference(plan_ref); validate_reference(review_ref)
    plan = c.load(plan_ref, 8 * c.MIB); review = c.load(review_ref, 8 * c.MIB)
    c.verify_sources(plan['source_sha256'])
    import dory_loader_candidate_elf_v1 as elfcore
    import minicpm5_layer0_external_supervisor_v4 as external
    import dory_descriptor_inventory_runtime_v2 as descriptor
    import run_dory_serial_stage_root_v5 as lifecycle
    import dory_native_inventory_finish_v4 as finish
    out = validate_plan(plan, review, plan_ref, descriptor.POLICY, POLICY)
    c.verify_loaded(plan['source_sha256'])
    require(not out.exists(), 'Never reuse an existing collection output')
    p = external.p
    p.user_bus()
    fd, owner, lock, first = descriptor.acquire(p, plan_ref, review_ref)
    started = time.monotonic(); last = [0.0]; created = False
    observed = []; result_ref = None; metadata_read_bytes = 0
    proof = descriptor.Guard(p, owner, lock, fd, out, plan_ref, review_ref, BUDGET,
                             role='parent', created=lambda: created, first_observation=first)
    result = dict(schema='dory_loader_candidate_ELF_result_v1', complete=False,
                  plan=plan_ref, review=review_ref, owner=owner, lock_identity=lock,
                  workload_subprocesses_started=0, native_payload_bytes_read=0,
                  target_source_executed=False, target_packages_imported=False,
                  package_archive_payload_read=False, original_environment_modified=False,
                  loader_selection_established=False, native_ABI_qualified=False,
                  full_model_qualified=False, new_speedup_claim=False,
                  actual_process_exit_claimed=False)

    def guard():
        require(time.monotonic() - started < BUDGET['wall_seconds'], 'Original wall budget')
        if time.monotonic() - last[0] >= .5:
            proof(); p.clear_window(); p.active_heavy()
            for path, floor in ((ROOT, BUDGET['root_floor_bytes']),
                                (Path('/mnt/nvme'), BUDGET['nvme_floor_bytes'])):
                require(lifecycle.free(path) >= floor, 'Original storage floor')
            require(lifecycle.memory_free() >= BUDGET['memory_available_floor_bytes'],
                    'Reviewed available-memory floor')
            if created:
                require(lifecycle.storage_usage(out) <= BUDGET['output_bytes'], 'Bounded owned output')
            last[0] = time.monotonic()

    budget = finish.Budget(guard)

    def pinned(ref):
        nonlocal metadata_read_bytes
        validate_reference(ref)
        require(Path(ref['path']).suffix == '.json', 'Only explicit JSON metadata in pinned reader')
        read_cap = min(POLICY['metadata_input_per_file_cap_bytes'],
                       POLICY['metadata_input_total_cap_bytes'] - metadata_read_bytes)
        require(read_cap > 0, 'Remaining metadata byte budget before opening input')
        record, raw = finish.file_bytes(ref, budget, read_cap, retain=True)
        metadata_read_bytes += record['bytes']
        require(metadata_read_bytes <= POLICY['metadata_input_total_cap_bytes'], 'Finite complete metadata read budget')
        observed.append(record)
        return finish.decode(raw)

    def current_handoff():
        # Same finite state check as cache_v3; never enumerate environment files.
        guard()
        latest = max(Q.glob('continuation-handoff-*.json'),
                     key=lambda path: (path.stat().st_mtime_ns, str(path)), default=None)
        require(latest is not None and str(latest) == plan['current_handoff']['path'],
                'Latest handoff path still exactly matches the admitted plan')
        pinned(plan['current_handoff'])  # Full expected hash, length and stable identity.

    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(sig, frame):
        raise InterruptedError('Source collector termination signal ' + str(sig))

    try:
        for sig in saved_handlers:
            signal.signal(sig, interrupted)
        os.sched_setaffinity(0, set(BUDGET['cpus'])); os.nice(BUDGET['nice'])
        resource.setrlimit(resource.RLIMIT_AS, (BUDGET['address_space_bytes'],) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (BUDGET['cpu_seconds'], BUDGET['cpu_seconds'] + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (BUDGET['single_file_bytes'],) * 2)
        guard()
        require(lifecycle.free('/mnt/nvme') >= BUDGET['nvme_floor_bytes'] + BUDGET['output_bytes'],
                'Full bounded output reserved above original floor')
        out.mkdir(mode=0o700); created = True
        first_ref = c.save(out / 'initial-lock-proof.json',
                           descriptor.initial_record(first, owner, lock, fd, plan_ref, review_ref))
        proof.initial_persisted(first_ref)
        require(pinned(plan_ref) == plan and pinned(review_ref) == review, 'Plan/review unchanged after acquisition')
        c.verify_sources(plan['source_sha256'], guard); c.verify_loaded(plan['source_sha256'])
        serial = pinned(plan['current_serial_catalog']); pinned(plan['current_handoff'])
        require(serial['all_later_serial_runs_included'] is True
                and serial['current_handoff'] == plan['current_handoff']
                and type(serial['runs']) is list and len(serial['runs']) <= 20000,
                'Explicit latest root-reviewed complete serial history')
        for row in serial['runs']:
            pinned(row['terminal'])
            require(type(row['owners']) is list and len(row['owners']) <= 20000, 'Finite historical owners')
            for previous in row['owners']:
                guard(); p.gone(previous)
        history = lifecycle.preflight_history(plan, p)
        guard()
        c.save(out / 'owner.json', dict(owner=owner, lock_identity=lock, retained_fd=fd,
                                       plan=plan_ref, review=review_ref, history=history))
        table = pinned(INPUT_TABLE)
        require(table['schema'] == 'dory_loader_candidate_ELF_inputs_v1'
                and table['policy'] == POLICY and table['remaining_actual_loader_gaps'] == 15,
                'Exact fixed candidate table with unresolved selections preserved')
        design_inputs = {name: pinned(ref) for name, ref in table['input_refs'].items()}
        authority_inputs = {name: pinned(ref) for name, ref in table['authority_inputs'].items()}
        design = design_inputs['design']; root_review = design_inputs['root_review']
        require(root_review['passed'] is True and root_review['runtime_admitted'] is False
                and root_review['design']['sha256'] == table['input_refs']['design']['sha256'],
                'Accepted source design is not native/runtime admission')
        require(table['authority_inputs'] == design['pinned_saved_metadata']
                and table['candidates'] == design['content_authority']['candidate_table'],
                'All27 exact candidate content origins return to accepted pinned metadata')
        require(authority_inputs['inventory_result']['outputs']['files']['sha256']
                == table['authority_inputs']['native_files']['sha256'], 'Original inventory output binding')
        native = {row['path']: row for row in authority_inputs['native_files']}
        provenance = authority_inputs['site_provenance']
        packages = {}
        for package in authority_inputs['package_result']['packages']:
            for comparison in package['comparisons']:
                ref = comparison['current']; member = comparison['member']
                require(comparison['current_matches'] is True and ref['bytes'] == member['bytes']
                        and ref['sha256'] == member['sha256'], 'Independent package/member content match')
                require(ref['path'] not in packages, 'Unique package target')
                packages[ref['path']] = ref
        metadata = {}
        saved_candidates = {}
        for row in design_inputs['external']['files']:
            path = row['reference']['path']
            require(path not in metadata, 'Unique saved ELF record')
            parsed = elfcore.parse_saved_readelf(row['elf_command']['stdout'])
            require(parsed['needed'] == row['elf']['needed']
                    and parsed['interpreter'] == row['elf']['interpreter'], 'Saved ELF parser crosscheck')
            metadata[path] = parsed; saved_candidates[path] = row['dependencies']
        require(metadata == table['saved_elf_metadata']
                and saved_candidates == table['saved_dependency_candidates'] and len(metadata) == 24,
                'Unchanged24 complete saved metadata records')
        missing = [row['reference'] for row in table['candidates']
                   if row['reference']['path'] not in metadata]
        require(missing == table['new_native_reads'] and len(missing) == 3
                and sum(ref['bytes'] for ref in missing) == POLICY['new_native_total_bytes'],
                'Exactly three finite missing ELF objects; no dynamic expansion')
        # Authenticate every new expected ref from an independent prior source,
        # BEFORE opening any of the three native files as data.
        for ref in missing:
            path = ref['path']
            if path in native:
                require(native[path] == ref and provenance[path]['kind'] == 'upstream_wheel_member'
                        and provenance[path]['evidence']['archive_sha256'] == ref['sha256'],
                        'Missing torch object rooted in authenticated wheel member')
            else:
                require(packages.get(path) == ref, 'Missing system object rooted in authenticated deb member')
        current_handoff(); guard()
        collected = []; result['native_payload_attempted_refs'] = []
        result['native_payload_bytes_verified'] = 0
        for ref in missing:
            guard(); result['native_payload_attempted_refs'].append(deepcopy(ref))
            # Null on an interrupted/failed read; never misreport attempted bytes as zero.
            result['native_payload_bytes_read'] = None
            observation, raw = finish.file_bytes(ref, budget,
                POLICY['new_native_per_file_cap_bytes'], retain=True, budget_key='system_hash_bytes')
            result['native_payload_bytes_verified'] += observation['bytes']
            require(result['native_payload_bytes_verified'] <= POLICY['new_native_read_cap_bytes'],
                    'Exact bounded native data reads')
            parsed = elfcore.parse_elf64(raw)
            metadata[ref['path']] = parsed
            collected.append(dict(reference=deepcopy(ref), observation=observation, elf=parsed))
            del raw
        result['native_payload_bytes_read'] = result['native_payload_bytes_verified']
        graph = elfcore.candidate_graph(table, metadata)
        result.update(complete=True, input_table=INPUT_TABLE, new_elf_objects=collected,
                      candidate_graph=graph, consumed_metadata=observed,
                      metadata_bytes_read=metadata_read_bytes,
                      original_runtime_loader_gaps_remaining=15,
                      content_scope=POLICY, automatic_scope_expansion=False,
                      model_constraints_preserved=dict(num_loops=2,
                          total_independent_per_loop_attention_histories=56,
                          tensor_parallel_size=1, model_executed_here=False))
        current_handoff(); c.verify_sources(plan['source_sha256'], guard)
        c.verify_loaded(plan['source_sha256']); guard()
    except BaseException as error:
        result.update(complete=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for sig in saved_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if created:
                result.update(elapsed_seconds=time.monotonic() - started, descriptor_guard=proof.summary())
                result_ref = c.save(out / 'result.json', result)
                require(lifecycle.storage_usage(out) <= BUDGET['output_bytes'], 'Final output cap includes result')
        finally:
            # No workload child can exist in this entry point. Reuse the same
            # release-after-drain lifecycle over that empty child set.
            try:
                lifecycle.release_after_drain(None, None, p, SimpleNamespace(release_lock=os.close),
                                              fd, out if created else None)
            finally:
                for sig, handler in saved_handlers.items():
                    signal.signal(sig, handler)
    if result_ref is not None:
        print(json.dumps(result_ref), flush=True)


if __name__ == '__main__':
    main()
