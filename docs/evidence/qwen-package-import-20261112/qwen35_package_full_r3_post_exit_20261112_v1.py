"""Explicit external root post-exit observer; import performs no runtime work.

Supply --request and --request-sha256 after the real tool terminal is observed.
This separately reviewed source lives outside experiment ROOT. It never edits
the frozen runtime/spec, skips loaded-source checks, or grants qualification.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q = ROOT / 'research'
SELF = Path('/home/ubuntu/qwen35_package_full_r3_post_exit_20261112_v1.py')
CAP = 8 * 2**20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reference(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}
            and type(value['path']) is str and Path(value['path']).is_absolute()
            and str(Path(value['path'])) == value['path'] and '..' not in Path(value['path']).parts
            and type(value['sha256']) is str and re.fullmatch('[a-f0-9]{64}', value['sha256']),
            'exact absolute SHA reference required')


def stable_source(ref, cap=4 * 2**20):
    reference(ref); path = Path(ref['path'])
    require(path.resolve() == path and not any(p.is_symlink() for p in (path, *path.parents)),
            'canonical source path required')
    def identity():
        s = path.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= cap,
                'bounded singleton regular source required')
        return dict(device=s.st_dev, inode=s.st_ino, size=s.st_size,
                    mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)
    before = identity(); fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino) == (before['device'], before['inode']), 'source open identity changed')
        payload = bytearray()
        while len(payload) <= cap:
            block = os.read(fd, min(2**20, cap + 1 - len(payload)))
            if not block: break
            payload.extend(block)
        require(len(payload) == before['size'] and identity() == before, 'source identity/length changed')
        require(hashlib.sha256(payload).hexdigest() == ref['sha256'], 'source digest changed')
        return bytes(payload), before
    finally:
        os.close(fd)


def encode(value):
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    require(len(payload) <= CAP, 'original 8 MiB publication cap')
    return payload


def predicted(path, value):
    return dict(path=str(path), sha256=hashlib.sha256(encode(value)).hexdigest())


def actual_tool(value, wanted):
    require(set(value) == {'schema', 'reviewer', 'actual_tool', 'completed', 'observation_source', 'observed_unix_ns'}
            and value['schema'] == 'qwen35_package_external_tool_result_v1'
            and value['reviewer'] == '/root' and value['completed'] is True
            and value['observation_source'] == 'actual_external_tool_result'
            and type(value['observed_unix_ns']) is int and value['observed_unix_ns'] > 0,
            'real externally supplied terminal observation required')
    tool = value['actual_tool']
    require(set(tool) == {'kind', 'session_id', 'terminal_chunk', 'exit_code'}
            and tool['kind'] == 'tool_session' and type(tool['session_id']) is int and tool['session_id'] > 0
            and type(tool['terminal_chunk']) is str and re.fullmatch('[a-zA-Z0-9_-]{1,128}', tool['terminal_chunk'])
            and type(tool['exit_code']) is int
            and {k: tool[k] for k in ('kind', 'session_id')} == wanted, 'exact actual tool identity/exit required')
    return copy.deepcopy(tool)


def collect_scope(spec, admission_ref, coordinator, owners, declared_services, properties, process, guard):
    """Use original complete declaration logic; absence never becomes success."""
    expected = set(spec['units']); require(expected, 'complete expected units required')
    retained, passes = {}, []
    for _ in range(2):
        guard(); process.active_heavy()
        for owner in owners: process.gone(owner)
        declarations = []
        services = declared_services(spec['runtime_spec'], admission_ref, coordinator,
                                     retained=retained, observations=declarations)
        selected = {service.unit: service for service in services}
        require(len(selected) == len(services) and set(selected) <= expected, 'no duplicate/foreign declared service')
        seen = set(selected) | {row['unit'] for row in declarations}
        require(seen == expected, 'complete declared and unstarted scope required')
        current, terminals, historical = {}, {}, {}
        for unit in sorted(expected):
            guard(); service = selected.get(unit)
            if service is not None and service.started:
                terminal = service.drained()
                require(terminal['cleanup_complete'] is True and terminal['remaining_processes'] == [],
                        'every started service must drain')
                terminals[unit] = copy.deepcopy(terminal)
                if service.historical_terminal_reuse is not None:
                    historical[unit] = copy.deepcopy(service.historical_terminal_reuse)
            props = properties(unit)
            if service is None or not service.started:
                require(props['LoadState'] == 'not-found' and props['InvocationID'] == '',
                        'undeclared/unstarted unit must remain absent')
            current[unit] = dict(properties=props,
                recursive_empty_group=process.empty_group(process.GROUP_PREFIX + unit + '.service'))
        passes.append(dict(current=current, actual_terminals=terminals,
                           historical_terminal_reuse=historical, declaration_observations=declarations))
    require(passes[0]['current'] == passes[1]['current'], 'two stable complete current scope observations required')
    guard()
    return passes


def source_delta(spec_sources, extension, old_catalog, external):
    def merge(*maps):
        result = {}
        for values in maps:
            for path, sha in values.items():
                reference(dict(path=path, sha256=sha))
                require(path not in result or result[path] == sha, 'source composition conflict')
                result[path] = sha
        return result
    recovered = {p: sha for p, sha in spec_sources.items() if p not in extension}
    require(merge(recovered, extension) == spec_sources, 'complete recovered/overlay source composition')
    prior = merge(recovered, *(row['source_sha256'] for row in old_catalog['runs']))
    require(all(spec_sources.get(p) == sha for p, sha in prior.items()), 'every old catalog source retained')
    executed = merge(spec_sources, external)
    delta = {p: sha for p, sha in executed.items() if p not in prior}
    require(merge(prior, delta) == executed, 'exact full executed source map reconstructed')
    return delta, dict(recovered_sources=len(recovered), retained_prior_sources=len(prior),
        runtime_sources=len(spec_sources), post_audit_sources=len(executed), new_row_sources=len(delta),
        all_previous_rows_retained=True, exact_full_mapping_reconstructed=True)


def audit_outcome(tool, terminal, spec_ref, review_ref, external_ref, guard, audit):
    result = dict(actual_outer_exit_code=tool['exit_code'], outer_reported_failed=terminal['failed'],
                  stage25=None, stage25_error=None, all25_stages_audited=False,
                  full_model_qualified=False, new_speedup_claim=False)
    if tool['exit_code'] == 0 and terminal['failed'] is False and terminal['result_present'] is True:
        try:
            result['stage25'] = audit(spec_ref, review_ref, external_ref, guard)
            result['all25_stages_audited'] = True
        except Exception as error:
            result['stage25_error'] = dict(type=type(error).__name__, message=str(error))
    result['status'] = ('stage25_audited_pending_root_qualification' if result['all25_stages_audited'] else
                        'independent_stage25_failed' if result['stage25_error'] else 'outer_failed_or_inconsistent')
    return result


def append_catalog(old, idle_ref, terminal_ref, tool, owners, units, delta, catalog_shape):
    require(len(old['runs']) == 160, 'exact retained r3 predecessor has 160 rows')
    catalog_shape(old)
    result = copy.deepcopy(old)
    result.update(current_handoff=idle_ref, runs=copy.deepcopy(old['runs']) + [dict(
        terminal=terminal_ref, tool=tool, owners=owners, units=sorted(units), source_sha256=delta)])
    require(result['runs'][:-1] == old['runs'], 'every original catalog row retained in order')
    catalog_shape(result); encode(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True); parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    require(sys.dont_write_bytecode and __debug__, 'python -B without optimization required')
    request_ref = dict(path=args.request, sha256=args.request_sha256)
    request_raw, request_identity = stable_source(request_ref, CAP); request = json.loads(request_raw)
    require(set(request) == {'schema', 'binding', 'root_live', 'admission', 'running_handoff',
        'external_tool_result', 'observer_source', 'ack_source', 'outputs'}
        and request['schema'] == 'qwen35_package_r3_post_exit_request_v1', 'exact explicit request required')
    require(request['observer_source']['path'] == str(SELF) == str(Path(__file__).resolve())
            and ROOT not in SELF.parents, 'separately reviewed external observer path required')
    source_refs = [request['observer_source'], request['ack_source']]
    source_before = {r['path']: stable_source(r)[1] for r in source_refs}
    sys.path[:0] = [str(Q), str(ROOT)]
    from qwen35_package_serial_runtime_v5 import io, contract, adapter, history, wiring, final_audit
    from prepare_qwen35_package_serial_candidate_v4 import catalog_shape
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    observed_metadata = {}
    def read(r, cap=CAP):
        reference(r); before = io.identity(r['path']); value = io.read(r, cap)
        require(io.identity(r['path']) == before, 'metadata changed across authenticated read')
        require(r['path'] not in observed_metadata or observed_metadata[r['path']] == before,
                'previously observed metadata identity changed')
        observed_metadata[r['path']] = before
        return value
    for name in ('binding', 'root_live', 'admission', 'running_handoff', 'external_tool_result'): reference(request[name])
    binding = read(request['binding']); spec_ref = binding['spec']; review_ref = binding['review']
    require(binding['passed'] is True and binding['full_source_history_resolution_passed'] is True,
            'successful exact root binding required')
    spec = read(spec_ref, 64 * 2**20); contract.validate_spec(spec, read(review_ref, 64 * 2**20), spec_ref)
    require(spec['run_name'] == 'pkg-r3-20261112', 'explicit r3 run required')
    control = contract.control_root(spec); admission = read(request['admission'])
    require(io.ref(control / 'admission.json') == request['admission'], 'exact owned admission path')
    adapter.admit_shape(admission, spec, spec_ref, review_ref)
    coordinator_ref = io.ref(control / 'coordinator-owner.json'); coordinator = read(coordinator_ref)
    live = read(request['root_live']); running = read(request['running_handoff'])
    require(io.ref(spec['running_handoff_path']) == request['running_handoff'], 'exact running handoff')
    require(live['root_binding'] == request['binding'] and live['coordinator_owner'] == coordinator_ref
            and live['running_handoff'] == request['running_handoff'] and live['ack_source'] == request['ack_source']
            and live['outer_owner'] == admission['outer_owner'], 'root-live identity chain required')
    wanted = running['qwen35_sampler_current_execution']['tool']
    require(live['actual_tool'] == wanted, 'live and running tool identity required')
    previous = read(admission['freshness']['latest_handoff'])
    require(running == history.expected_running(previous, running['updated_at'], wanted,
            request['admission'], admission, coordinator_ref, coordinator), 'complete inherited running handoff required')
    ack_ref = io.ref(control / 'running-ack.json'); ack = read(ack_ref)
    running_observation = dict(path=request['running_handoff']['path'], identity=io.identity(request['running_handoff']['path']))
    require(ack == dict(admission=request['admission'], coordinator_owner=coordinator_ref,
            coordinator=coordinator, running=dict(reference=request['running_handoff'], observation=running_observation, value=running)),
            'complete actual running ACK required')
    external = read(request['external_tool_result']); tool = actual_tool(external, wanted)
    terminal_ref = io.ref(control / 'outer-terminal.json'); terminal = read(terminal_ref)
    require(terminal['schema'] == 'qwen35_sampler_outer_observed_terminal_v1'
            and terminal['admission'] == request['admission'] and terminal['outer_owner'] == admission['outer_owner']
            and terminal['original_lock_identity'] == contract.LOCK_IDENTITY and terminal['self_exit_claimed'] is False
            and terminal['whole_owned_scope_drained'] is True and type(terminal['failed']) is bool,
            'authentic full outer terminal required; missing evidence remains unresolved')
    active_scope_failure = read(terminal['active_scope_failure']) if terminal['active_scope_failure'] is not None else None
    oldcat = read(spec['current_serial_catalog']); overlay = read(spec['history_overlay'])
    require(oldcat['current_handoff'] == admission['freshness']['latest_handoff'], 'executed predecessor catalog chain')
    outputs = request['outputs']
    require(set(outputs) == {'external_tool_exit', 'actual_exit', 'idle_handoff', 'catalog'}
            and len(set(outputs.values())) == 4, 'four distinct explicit output paths required')
    for value in outputs.values():
        path = Path(value)
        require(path.parent == Q and path.suffix == '.json' and path.resolve() == path
                and not path.exists() and not path.is_symlink(), 'fresh canonical explicit metadata output required')
    owners = [admission['outer_owner'], coordinator['process']]; phase_evidence = []
    for arm, phase in contract.PHASES:
        directory = control / (arm + '-' + phase); evidence = dict(arm=arm, phase=phase, references={})
        values = {}
        for name in ('phase-spec', 'supervisor-owner', 'child-bootstrap-owner', 'child-exit', 'root-terminal',
                     'api-failure', 'api-failure-complete'):
            path = directory / (name + '.json')
            if not path.exists(): continue
            r = io.ref(path); value = read(r); evidence['references'][name] = r; values[name] = value
            if name in ('supervisor-owner', 'child-bootstrap-owner'):
                owner = value['process'] if name == 'supervisor-owner' else value
                if owner not in owners: owners.append(owner)
        if evidence['references']:
            require('phase-spec' in values, 'recorded child/terminal requires original phase declaration')
            phase_spec = values['phase-spec']
            require(phase_spec['admission'] == request['admission'] and phase_spec['coordinator'] == coordinator
                    and (phase_spec['arm'], phase_spec['phase']) == (arm, phase), 'exact recorded phase scope')
            if 'child-bootstrap-owner' in values:
                require('supervisor-owner' in values, 'child identity requires original supervisor identity')
                child = values['child-bootstrap-owner']; supervisor = values['supervisor-owner']['process']
                require(child['ppid'] == supervisor['pid'] and child['cgroup'] == supervisor['cgroup'],
                        'original direct child parent/cgroup required')
            if 'child-exit' in values:
                code = values['child-exit']
                require(code['phase_spec'] == evidence['references']['phase-spec']
                        and code['supervisor'] == values.get('supervisor-owner')
                        and code['child_bootstrap_owner'] == values.get('child-bootstrap-owner')
                        and code['actual_popen_wait_observed'] is True, 'actual child exit binding required')
            phase_evidence.append(evidence)
    prefix = []; gap = False
    for index, (arm, stage) in enumerate(wiring.STAGES[:-1], 1):
        path = wiring.stage_receipt(spec, index)
        if not path.exists(): gap = True; continue
        require(not gap, 'completed scientific stages must remain a contiguous real prefix')
        r = io.ref(path); value = read(r)
        require(value['index'] == index and (value['arm'], value['stage']) == (arm, stage)
                and value['case'] == spec['case'] and value['completed'] is True, 'authentic completed prefix entry')
        read(value['result']); prefix.append(r)
    frozen = [(request_ref, request_identity), *[(r, source_before[r['path']]) for r in source_refs]]
    def sources_unchanged():
        for r, identity in frozen:
            require(stable_source(r, CAP if r == request_ref else 4 * 2**20)[1] == identity, 'external execution source/request changed')
    process = io.process_api(); lock = io.RetainedLock(); lock.acquire()
    # This observer never starts services. On any failed proof it publishes no
    # idle/catalog; releasing its own audit lease is not a drain-success claim.
    try:
        def guard():
            lock.check(); require(latest.latest_handoff() == running_observation, 'unknown later handoff')
            require(all(io.identity(path) == identity for path, identity in observed_metadata.items()),
                    'authenticated input metadata changed during audit')
            require(not any(n.split('.')[0] in ('torch', 'torch_neuronx', 'numpy', 'neuronxcc', 'vllm', 'transformers') for n in sys.modules), 'no native imports in observer')
            return dict(admitted=True, exclusive=True)
        observations = collect_scope(dict(runtime_spec=spec, units=contract.units(spec['run_name']).values()),
            request['admission'], coordinator, owners, adapter.declared_services, adapter.properties, process, guard)
        source_pins = {r['path']: r['sha256'] for r in source_refs}
        delta, composition = source_delta(spec['source_sha256'], overlay['source_extension'], oldcat, source_pins)
        external_value = dict(schema='qwen35_package_root_actual_tool_exit_v1', reviewer='/root',
            spec=spec_ref, admission=request['admission'], outer_owner=admission['outer_owner'], tool=wanted,
            exit_code=tool['exit_code'], completed=True, observation_source='actual_external_tool_result',
            observed_unix_ns=external['observed_unix_ns'])
        guard(); sources_unchanged(); external_ref = io.once(Path(outputs['external_tool_exit']), external_value)
        outcome = audit_outcome(tool, terminal, spec_ref, review_ref, external_ref, guard, final_audit.audit)
        # A stage25 error is independent of the real outer exit and is retained.
        guard(); sources_unchanged()
        process.active_heavy()
        for owner in owners: process.gone(owner)
        after = {r['path']: stable_source(r)[1] for r in source_refs}
        require(after == source_before, 'external source identity changed after execution')
        value = dict(schema='qwen35_package_root_actual_exit_v1', reviewer='/root',
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), actual_tool=tool,
            actual_tool_exit_observed=True, external_tool_result=request['external_tool_result'], external_tool_exit=external_ref,
            root_binding=request['binding'], root_live=request['root_live'], spec=spec_ref, review=review_ref,
            admission=request['admission'], coordinator_owner=coordinator_ref, running_handoff=request['running_handoff'],
            running_ack=ack_ref, outer_terminal=terminal_ref, original_outer_terminal=terminal,
            original_active_scope_failure=active_scope_failure,
            owners=owners, owners_gone=True, all_declared_units=sorted(contract.units(spec['run_name']).values()),
            physically_drained=True, two_current_scope_observations=observations,
            recorded_phase_evidence=phase_evidence, completed_stage_prefix=prefix, completed_stages=len(prefix),
            outcome=outcome, stage25=outcome['stage25'], source_composition=composition,
            external_sources=source_refs, external_source_identities_before=source_before,
            external_source_identities_after=after, auditor_request=request_ref,
            audit_lock_identity=contract.LOCK_IDENTITY, full_model_qualified=False, new_speedup_claim=False)
        terminal_out = predicted(outputs['actual_exit'], value)
        idle = copy.deepcopy(running)
        idle.update(updated_at=value['observed_at'], previous=request['running_handoff'],
            previous_handoff=request['running_handoff'], running=False, active_serial=None,
            live_tool_sessions=[], active_heavy_units=[], goal_status='active', this_turn_classification='progress',
            complete_current_serial_catalog=spec['current_serial_catalog'], complete_current_serial_catalog_next=outputs['catalog'],
            publication_current_status_stale=True, new_speedup_claim=False)
        idle['qwen35_package_r3_current'] = dict(actual_exit=terminal_out, status=outcome['status'],
            completed_stages=len(prefix), all25_stages_audited=outcome['all25_stages_audited'],
            full_model_qualified=False, new_speedup_claim=False)
        idle_ref = predicted(outputs['idle_handoff'], idle)
        catalog = append_catalog(oldcat, idle_ref, terminal_out, tool, owners,
                                 contract.units(spec['run_name']).values(), delta, catalog_shape)
        catalog_ref = predicted(outputs['catalog'], catalog)
        guard(); sources_unchanged(); io.loaded_local_sources(spec['source_sha256'])
        require(io.once(Path(outputs['actual_exit']), value) == terminal_out, 'actual-exit publication hash')
        require(io.once(Path(outputs['idle_handoff']), idle) == idle_ref, 'idle publication hash')
        require(io.once(Path(outputs['catalog']), catalog) == catalog_ref, 'catalog publication hash')
        sources_unchanged()
    finally:
        lock.release(True)
    print(json.dumps(dict(actual_exit=terminal_out, idle_handoff=idle_ref, catalog=catalog_ref, outcome=outcome)))


if __name__ == '__main__':
    main()
