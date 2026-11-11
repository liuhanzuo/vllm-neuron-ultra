"""One admitted, finite saved-event/source read; inert until explicit CLI launch.

No target interpreter, native payload, wheel/archive payload, workload child,
imported target source, model, or environment scan. The existing serial-history
helpers retain their process/service observations. Root supplies fresh reviewed
plan/history references; this source does not bind a currently live handoff.

Plan schema dory_loader_route_source_plan_v2 requires runtime_bound=true,
automatic_successor_launch=false, output_root, source_sha256, budget=BUDGET,
lock_policy=descriptor.POLICY, current_serial_catalog, current_handoff,
predecessor_exit, predecessor_expected_exit_code, predecessor_owners, and
predecessor_services. The root review schema is
dory_loader_route_source_review_v2; it binds plan, source_sha256, lock_policy,
both current-history references and predecessor_exit, with reviewer=/root,
passed=true, blockers=[], latest_handoff_accepted=true, and
current_physical_history_accepted=true. References contain path/sha256 and may
include bytes. Root prepares those concrete files only after the live work ends.
V2 reuses the V1 lifecycle and adds the exact reviewed source_selection_policy;
only authenticated source-name candidates are selected, never filesystem scans.
A missing reference in this limited set does not prove whole-package absence.
"""
import argparse
from copy import deepcopy
import hashlib
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
SITE = Path('/mnt/nvme/venvs/dory-official-cpu-20261108-v1/lib/python3.12/site-packages')
ENTRYPOINT_SOURCES = {
    str(Q / 'dory_stage_a_native_resolver_v4.py'): 'e872f26fafd78e72b77847353a782a526e4093d6575ee033b94c4d4298bd1eb9',
    str(Q / 'dory_serial_stage_worker_v4.py'): 'ef56bb3049fd352a576c380e45094b0248882407b7532484c3b87952b97fe614',
    str(Q / 'dory_stage_a_capture_core_v1.py'): '9341ac49a187638a6c5eec2eb989238bb04cfcf37accd481c38a3c4f6555cd17',
    str(Q / 'dory_stage_a_worker_preparation_v1.py'): 'aae622346332916fddd3ec84e9da3bad1241f0ad64a5129d797f60f7b01a2ddd',
}
SOURCE_CAP = 512 * 2**10
TOTAL_SOURCE_CAP = 8 * 2**20
BUDGET = dict(address_space_bytes=2 * 2**30, cpu_seconds=600, wall_seconds=900,
              single_file_bytes=8 * 2**20, output_bytes=32 * 2**20,
              log_bytes=8 * 2**20, cpus=[5], nice=19,
              root_floor_bytes=28 * 2**30, nvme_floor_bytes=30 * 2**30,
              memory_available_floor_bytes=2**30)
EVENTS = dict(path='/mnt/nvme/dory-serial-inventory-r5-20261111/events.jsonl.gz',
              bytes=6131851,
              sha256='37fe06a4bfde3f911e37b41c3d4aea46571c749f42bfcf51c40851711313095a',
              uncompressed_bytes=40170999,
              uncompressed_sha256='c8991e63329cb54bf394098e7153c7b05ca89f80a827ff7b2756493ee337de80',
              events=70764)
INPUTS = {
    'previous_source_result': dict(path='/mnt/nvme/dory-loader-route-sources-r1-20261111/result.json',
        bytes=54368, sha256='5fae7953b883f17949b6c1438e78957558f7ba04604e5315339ed3580a733481'),
    'previous_source_exit': dict(path=str(Q / 'dory-loader-route-sources-r1-root-actual-exit-20261111-v1.json'),
        bytes=5441, sha256='5cc698cc618c69a6e90f3d40da9c484a4940f38f9620b4974afa502cb0d5f1d2'),
    'original_imports': dict(path='/mnt/nvme/dory-official-cpu-20261108-v1/qualify-r2/qualification/imports.json',
        bytes=2470, sha256='8daae2c286aeabd38dbcfc8c614c4d4039a124b2cb5ad573ae85f8be26ec50ea'),
    'r5_exit': dict(path=str(Q / 'dory-serial-inventory-r5-actual-exit-20261111-v1.json'),
                    bytes=8157, sha256='e6cff0181f6a1d95c7f816c6d906e9bf170dd4768a5cc52a56f94f26b170ec89'),
    'remaining': dict(path=str(Q / 'dory-remaining-loader-gaps-after-cache-20261111-v1.json'),
                      bytes=37912, sha256='41e6d4600ec218dfc56e2555333795c68fc3a5e4e4b279142356ae6877583681'),
    'manifest': dict(path=str(Q / 'dory-full-wheel-source-manifest-20261111-v1.json'),
                     sha256='f4bf58eea14dd3d73856f6a5e4b14be1423560f059aedeef2afd5ef6f8656cea'),
}
ARCHIVES = {
    'torch': dict(path='/mnt/nvme/dory-torch-cpython-archive-verification-r1-20261111/torch.whl',
                  bytes=196253793, sha256='a09987c95ec4cffdb6df798d3d641558110a334cfccce22f2f046d83142bc260'),
    'vllm': dict(path='/mnt/nvme/dory-official-cpu-20261108-v1/wheels/vllm-0.1.dev1+g75fc05a53.cpu-cp312-cp312-linux_x86_64.whl',
                 bytes=132685835, sha256='37eae6e5e8f345b22d2c6147befa1c9f6b725c9fe88f83c336ae83f3184821f7'),
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


def validate_plan(plan, review, plan_ref, policy, selection_policy):
    require(plan['schema'] == 'dory_loader_route_source_plan_v2'
            and plan['runtime_bound'] is True and plan['automatic_successor_launch'] is False,
            'One fresh root-bound source collection')
    out = Path(plan['output_root'])
    require(out.parent == Path('/mnt/nvme') and str(out) == plan['output_root']
            and re.fullmatch(r'dory-loader-route-sources-[a-zA-Z0-9-]+', out.name),
            'Fixed fresh output namespace')
    require(plan['budget'] == BUDGET and plan['lock_policy'] == policy,
            'Exact bounded resource and current descriptor policies')
    require(review['schema'] == 'dory_loader_route_source_review_v2'
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
    require(plan['source_selection_policy'] == review['source_selection_policy'] == selection_policy,
            'Exact reviewed finite source selection; no automatic expansion')
    require(str(Q / 'dory_loader_route_selection_v2.py') in plan['source_sha256'],
            'Pure selection helper included in authenticated source closure')
    require(all(plan['source_sha256'].get(path) == wanted
                for path, wanted in ENTRYPOINT_SOURCES.items()),
            'Original Stage A entrypoint and two-loop/56-history/TP1 source pins retained')
    return out


def write_source(path, raw, guard):
    """Preserve exact authenticated source bytes; never import or execute them."""
    require(len(raw) <= SOURCE_CAP, 'Bounded saved source')
    guard()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(path.parent.resolve() == path.parent, 'Canonical source output parent')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            guard()
            written = os.write(fd, raw[offset:offset + 65536])
            require(written > 0, 'Complete source output write')
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'review', 'review-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    require(sys.executable == '/usr/bin/python3' and sys.dont_write_bytecode
            and sys.flags.no_user_site and __debug__, 'System Python -B -s with assertions only')
    require(not any(name.split('.')[0] in ('torch', 'vllm', 'numpy', 'transformers', 'tokenizers')
                    for name in sys.modules), 'No native target packages in collector')
    sys.path[:0] = [str(ROOT), str(Q)]
    import dory_serial_stage_contract_v2 as c
    plan_ref = dict(path=args.plan, sha256=args.plan_sha256)
    review_ref = dict(path=args.review, sha256=args.review_sha256)
    validate_reference(plan_ref); validate_reference(review_ref)
    plan = c.load(plan_ref, 8 * c.MIB); review = c.load(review_ref, 8 * c.MIB)
    c.verify_sources(plan['source_sha256'])
    import dory_loader_route_selection_v2 as selector
    require(selector.POLICY['per_source_bytes'] == SOURCE_CAP
            and selector.POLICY['total_source_bytes'] == TOTAL_SOURCE_CAP,
            'One consistent finite source budget')
    import minicpm5_layer0_external_supervisor_v4 as external
    import dory_descriptor_inventory_runtime_v2 as descriptor
    import run_dory_serial_stage_root_v5 as lifecycle
    import dory_native_inventory_finish_v4 as finish
    out = validate_plan(plan, review, plan_ref, descriptor.POLICY, selector.POLICY)
    c.verify_loaded(plan['source_sha256'])
    require(not out.exists(), 'Never reuse an existing collection output')
    p = external.p
    p.user_bus()
    fd, owner, lock, first = descriptor.acquire(p, plan_ref, review_ref)
    started = time.monotonic(); last = [0.0]; created = False
    observed = []; result_ref = None
    proof = descriptor.Guard(p, owner, lock, fd, out, plan_ref, review_ref, BUDGET,
                             role='parent', created=lambda: created, first_observation=first)
    result = dict(schema='dory_loader_route_source_result_v2', complete=False,
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
        validate_reference(ref)
        require(Path(ref['path']).suffix == '.json', 'Only explicit JSON metadata outside the fixed sources/events')
        record, raw = finish.file_bytes(ref, budget, 8 * c.MIB, retain=True)
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
        inputs = {name: pinned(ref) for name, ref in INPUTS.items()}
        previous = inputs['previous_source_exit']
        require(previous['actual_tool_exit_observed'] is True
                and previous['actual_exit_code'] == previous['actual_tool']['exit_code'] == 0
                and previous['result']['path'] == INPUTS['previous_source_result']['path']
                and previous['result']['sha256'] == INPUTS['previous_source_result']['sha256']
                and inputs['previous_source_result']['complete'] is True,
                'Prior five-source collection has an actual successful root-observed exit')
        imports = inputs['original_imports']
        require(imports['platform'] == 'CpuPlatform'
                and imports['cpu_capabilities']['architecture'] == 'x86_64'
                and imports['cpu_capabilities']['avx512_f'] is True
                and imports['cpu_capabilities']['avx512_bf16'] is False
                and imports['ld_preload'] == '/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:'
                    '/mnt/nvme/venvs/dory-official-cpu-20261108-v1/lib/libiomp5.so',
                'Original saved platform/capability/preload evidence, not current loader selection')
        audit = inputs['r5_exit']['complete_events_audit']
        expected_audit_ref = dict(audit['compressed'], uncompressed_bytes=audit['raw_bytes'],
                                 uncompressed_sha256=audit['raw_sha256'], events=audit['events'])
        require(inputs['r5_exit']['stage_success_observed'] is True
                and inputs['r5_exit']['actual_tool']['exit_code'] == 0
                and audit['complete'] is True and expected_audit_ref == EVENTS,
                'Complete successful r5 audit authenticates exact saved event stream')
        remaining = inputs['remaining']
        require(remaining['remaining_loader_and_cache_gaps'] == 15
                and set(remaining['remaining']) == {'vllm_missing_loader_edges', 'torch_ambiguous_libgomp_edges'}
                and len(remaining['remaining']['vllm_missing_loader_edges']) == 13
                and len(remaining['remaining']['torch_ambiguous_libgomp_edges']) == 2,
                'Only the reviewed 13 plus 2 loader gaps remain')
        packages = {}
        for row in inputs['manifest']['packages']:
            name = row['name']
            if name in ARCHIVES:
                require(name not in packages, 'Unique selected archive manifest row')
                require(dict(path=row['retained_archive'], bytes=row['archive_bytes'], sha256=row['sha256'])
                        == ARCHIVES[name], 'Existing exact archive authority, without archive reads')
                packages[name] = row
        require(set(packages) == set(ARCHIVES), 'Both original archive authorities present')
        current_handoff()
        selected = selector.select_targets(finish.gzip_records(EVENTS, budget, lines=True),
                                  expected_events=EVENTS['events'])
        result.update(event_stream=EVENTS, authenticated_event_stream_exhausted=True,
                      selected_members=selected, package_authorities=packages,
                      fixed_inputs=INPUTS, source_selection_policy=selector.POLICY,
                      original_entrypoint_source_pins=ENTRYPOINT_SOURCES,
                      original_import_metadata=imports)
        sources = []; total = 0; findings = []; finding_count = 0; finding_bytes = 0
        for row in selected:
            name = str(Path(row['reference']['path']).relative_to(SITE))
            record, raw = finish.file_bytes(row['reference'], budget, SOURCE_CAP, retain=True)
            require(record['identity'] == row['identity'], 'Original saved source identity unchanged')
            observed.append(record); total += len(raw)
            require(total <= TOTAL_SOURCE_CAP, 'Actual aggregate selected-source cap')
            finding = selector.inspect_source(name, raw)  # AST parse only; no target execution.
            finding_count += len(finding['direct_import_statements']) + len(finding['textual_mentions'])
            finding_bytes += len(json.dumps(finding, ensure_ascii=False).encode('utf-8'))
            require(finding_count <= selector.POLICY['total_findings_cap']
                    and finding_bytes <= selector.POLICY['total_findings_bytes'],
                    'Bounded aggregate source-reference findings')
            findings.append(finding)
            saved = write_source(out / 'sources' / name, raw, guard)
            require(saved['sha256'] == record['sha256'] and saved['bytes'] == record['bytes'],
                    'Saved complete source bytes match authenticated input')
            sources.append(dict(reference=finish.result_ref(record), identity=record['identity'],
                                saved_source=saved, provenance=row['provenance']))
        for record in observed:
            path = Path(record['path'])
            require(path.resolve() == path and finish.ident(path.lstat()) == record['identity'],
                    'All consumed input identities unchanged at completion')
        c.verify_sources(plan['source_sha256'], guard); c.verify_loaded(plan['source_sha256'])
        require(not any(name.split('.')[0] in ('torch', 'vllm', 'numpy', 'transformers', 'tokenizers')
                        for name in sys.modules), 'No target package imported by source collector')
        result.update(complete=True, sources=sources, source_files=len(sources), source_bytes=total,
                      source_findings=findings, findings_count=finding_count,
                      findings_bytes=finding_bytes, consumed_inputs=observed,
                      whole_package_source_search=False, missing_match_proves_absence=False,
                      actual_import_or_call_observed=False, automatic_scope_expansion=False,
                      model_constraints_preserved=dict(num_loops=2,
                          total_independent_per_loop_attention_histories=56,
                          tensor_parallel_size=1, model_executed_here=False))
        current_handoff()
        guard()
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
