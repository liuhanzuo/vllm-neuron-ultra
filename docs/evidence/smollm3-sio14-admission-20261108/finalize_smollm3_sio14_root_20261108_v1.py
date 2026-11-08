"""Finalize sio14 only after its authenticated actual external exit.

Source-prepared only: there is no bound runtime specification or exit yet.
Native/model payloads are never read. New experiment output is hashed in a
bounded streaming inventory after independent exit/drain proof.

This is separate from the frozen running experiment. Both successful and
unsuccessful scientific outcomes retain the same complete process history.
"""
import argparse
import copy
import re
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
import stat
import smollm3_sio14_catalog_transport_review_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smollm3_observer_invariance_lifecycle_v24 as life
import smollm3_observer_invariance_protocol_v20 as protocol
import smollm3_observer_invariance_analysis_v20 as analysis
import smollm3_token_capture_predecessors_v7 as predecessors

# Set only from explicit, authenticated root references in main().
SPEC = SPEC_SHA = REVIEW = REVIEW_SHA = LIVE_SHA = None
SESSION = PRIOR_ROWS = PRIOR_PROCESSES = None
WORK = life.DEST
CONTROL = life.CONTROL
RESEARCH = ROOT / 'research'


def reference(path):
    raw = life.read_bytes(path) if Path(path).is_relative_to(WORK) else protocol.bounded_source(path)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def load(path):
    raw = life.read_bytes(path) if Path(path).is_relative_to(WORK) else protocol.bounded_source(path)
    value = life.decode_bytes(path, raw) if Path(path).is_relative_to(WORK) else json.loads(raw)
    return value, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def record(path, value):
    assert path.parent == RESEARCH and not path.exists()
    raw = life.json_bytes(value)
    assert len(raw) < 16 * 2**20
    pending = path.with_name('.pending-' + path.name + '-' + os.urandom(16).hex())
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        assert stream.write(raw) == len(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
        life.rename_no_replace(pending, path)
        os.fsync(parent)
    finally:
        os.close(parent)
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def validate_outcome(terminal, release, controller, supervisor, exited, tool_chunk):
    """Neither physical absence nor a completed matrix implies invariance."""
    qualified = terminal['qualified']
    assert type(qualified) is bool
    assert release['released'] is True and release['failed'] is (not qualified)
    assert release['all_planned_groups_drained'] is terminal['all_planned_groups_drained'] is True
    assert terminal['cleanup_errors'] == []
    assert type(controller['cleanup_errors']) is list
    assert terminal['original_controller_error'] == controller['original_error']
    assert terminal['original_controller_cleanup_errors'] == controller['cleanup_errors']
    assert terminal['request_failure_audit']['evidence_audit_errors'] == []
    assert terminal['full_model_qualified'] is terminal['new_speedup_claim'] is False
    assert terminal['external_self_exit_not_claimed'] is True
    assert release['next_admission_must_verify_external_exit'] is True
    original_exit_ok = (supervisor['success'] is True and
                       supervisor['properties']['Result'] == 'success' and
                       supervisor['properties']['ExecMainCode'] == '1' and
                       supervisor['properties']['ExecMainStatus'] == '0')
    assert terminal['original_supervisor_exit_zero'] is original_exit_ok
    protocol_complete = (terminal['scientific_error'] is None and
                         controller['protocol_completed'] is True)
    lifecycle_ok = (terminal['original_error'] is None and original_exit_ok and
                    controller['original_error'] is None and controller['cleanup_errors'] == [])
    assert terminal['protocol_completed'] is protocol_complete
    assert terminal['lifecycle_qualified'] is lifecycle_ok
    assert qualified is (protocol_complete and lifecycle_ok)
    assert exited['actual_exit_observed_by_root'] is True and exited['gone'] is True
    assert type(exited['exit_code']) is int
    assert exited['exit_code'] == (0 if qualified else 1)
    assert type(tool_chunk) is str and re.fullmatch('[0-9a-f]{6}', tool_chunk)
    assert exited['execution_handle'] == dict(kind='tool_session', session_id=SESSION,
        actual_completion_observed=True, exit_code=exited['exit_code'], tool_chunk=tool_chunk)
    if not protocol_complete:
        assert terminal['observer_api_invariance_passed'] is None
    else:
        assert type(terminal['observer_api_invariance_passed']) is bool
    return qualified, protocol_complete


def validate_extension(prior, candidate, spec, owner):
    assert prior['root_finalized'] is True and prior['reviewer'] == '/root'
    assert len(prior['rows']) == PRIOR_ROWS and len(prior['processes']) == PRIOR_PROCESSES
    assert prior['complete_count'] == PRIOR_ROWS and PRIOR_ROWS >= 682 and PRIOR_PROCESSES >= 106
    assert candidate['rows'][:PRIOR_ROWS] == prior['rows']
    assert candidate['processes'][:PRIOR_PROCESSES] == prior['processes']
    assert len(candidate['rows']) == candidate['complete_count'] == PRIOR_ROWS + 25
    assert len(candidate['processes']) == PRIOR_PROCESSES + 1
    assert all(candidate['source_sha256'][k] == v for k, v in prior['source_sha256'].items())
    assert all(candidate['source_sha256'][k] == v for k, v in spec['runtime_source_sha256'].items())
    assert candidate['historical_payload_sha256'] == spec['historical_payload_sha256']
    assert candidate['previous_catalog'] == spec['final_predecessor_catalog']
    changed = {'schema', 'rows', 'processes', 'process_predecessors', 'complete_count', 'source_sha256',
        'historical_payload_sha256', 'added_planned_units', 'previous_catalog', 'root_finalized',
        'requires_root_terminal_review', 'physical_absence_is_not_success', 'external_self_exit_claimed',
        'new_source_and_evidence_pins'}
    assert all(candidate.get(k) == v for k, v in prior.items() if k not in changed)
    assert candidate['root_finalized'] is False
    assert candidate['all_planned_launches_accounted_for'] is True
    assert candidate['requires_root_terminal_review'] is True
    assert candidate['external_self_exit_claimed'] is False
    added = candidate['rows'][PRIOR_ROWS:]
    assert [row['unit'] for row in added] == spec['planned_units']
    assert len({row['unit'] for row in candidate['rows']}) == PRIOR_ROWS + 25
    assert all(not row.get('launch_without_authenticated_owner') for row in added)
    expected = dict(owner, role='external_launcher_and_final_auditor',
                    self_exit_not_yet_observed=True, next_admission_must_prove_gone=True)
    assert candidate['processes'][-1] == expected
    if 'process_predecessors' in candidate:
        assert candidate['process_predecessors'] == candidate['processes']
    predecessors.validate_rows(candidate['rows'], candidate['processes'])
    return added


def validate_identity(row, realpath):
    assert type(row) is dict and set(row) == {'device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'realpath'}
    assert row['realpath'] == realpath and Path(realpath).is_absolute()
    assert all(type(row[k]) is int and row[k] > 0 for k in ('device', 'inode', 'mtime_ns', 'ctime_ns'))
    assert type(row['size']) is int and row['size'] >= 0


def validate_proof(spec, base, aliases, source, payload, final):
    """Bind inherited byte proof to frozen inputs; never accept two matching fakes."""
    assert source['source_pins_verified'] is True
    assert final['full_source_rehash'] is final['full_payload_rehash'] is True
    assert final['measurement_window'] is False and final['within_external_audit_seconds'] == 1800
    assert set(final['source_stats']) == set(source['source_stats']) == set(spec['runtime_source_sha256'])
    assert len(final['source_stats']) == len(spec['runtime_source_sha256'])
    assert final['source_stats'] == source['source_stats']
    for name, row in final['source_stats'].items():
        target = aliases[name]['target'] if name in aliases else name
        validate_identity(row, target)
    expected_sha = {name: row['sha256'] for name, row in base['payloads'].items()}
    expected_stats = {name: row['identity'] for name, row in base['payloads'].items()}
    assert len(expected_sha) == 28
    assert final['payload_sha256'] == payload['verified_sha256'] == expected_sha
    assert payload['stats'] == expected_stats
    for row in expected_stats.values():
        validate_identity(row, row['realpath'])
    # The fourteen earlier Phi raw fingerprints stay separate; they were not
    # included in this experiment's new 28-payload full byte rehash.
    assert not set(spec['historical_payload_sha256']) & set(spec['runtime_source_sha256'])
    return expected_stats


def validate_owners(v, refs, spec_ref, review_ref, added):
    owner = v['live']['owner']
    assert refs['live']['sha256'] == LIVE_SHA
    assert type(owner['pid']) is int and owner['pid'] > 0
    assert type(owner['start_ticks']) is str and owner['start_ticks'].isdigit()
    assert owner == v['terminal']['external_owner'] == v['release']['external_owner'] == v['exited']['owner']
    assert v['live']['session'] == SESSION and v['live']['actual_exit_observed'] is False
    assert owner['argv'][owner['argv'].index('--review')+1] == review_ref['path']
    assert owner['argv'][owner['argv'].index('--review-sha')+1] == review_ref['sha256']
    for value in (v['live'], v['exited']):
        assert value['reviewer'] == '/root' and value['execution_spec'] == spec_ref
    assert v['exited']['terminal'] is True and v['exited']['release'] == refs['release']
    assert v['terminal']['successor_catalog'] == str(CONTROL / 'successor-catalog.json')
    for row in added:
        captured = row['captured_process_owner']
        if captured is None:
            assert row['owner'] is None and row['never_launched'] is True
            assert row['invocation_id'] is None and row['installed_argv'] is None
        else:
            script = life.CONTROLLER if row['unit'] == life.UNIT else life.WRAPPER
            assert row['owner'] == dict(pid=captured['pid'], start_ticks=captured['start_ticks'],
                unit=captured['unit'], script=str(script))
            assert captured['unit'] == row['unit'] and captured['invocation_id'] == row['invocation_id']
            assert captured['cgroup'] == '0::' + life.GROUP_PREFIX + row['unit'] + '.service'
            assert row['never_launched'] is False
    supervisor = v['supervisor_owner']
    intent = v['supervisor_launch']
    assert v['supervisor']['owner'] == supervisor == added[-1]['captured_process_owner']
    assert supervisor['unit'] == intent['unit'] == life.UNIT == added[-1]['unit']
    assert supervisor['argv'] == intent['argv']
    assert supervisor['cgroup'] == '0::' + life.GROUP_PREFIX + life.UNIT + '.service'
    props = v['supervisor']['properties']
    assert props['ExecMainPID'] == str(supervisor['pid'])
    assert props['InvocationID'] == supervisor['invocation_id'] == added[-1]['invocation_id']
    assert life.installed(props) == ' '.join(intent['execargv']) == added[-1]['installed_argv']
    life.verify_retained_binding(intent, supervisor, v['supervisor_binding'])
    return owner


def validate_original_rows(spec, base, candidate, read_json, exists):
    """Derive original evidence paths from the frozen matrix, not catalog input."""
    added = candidate['rows'][PRIOR_ROWS:]
    assert len(added) == 25
    arguments = types.SimpleNamespace(spec=str(SPEC), spec_sha=SPEC_SHA,
        review=str(REVIEW), review_sha=REVIEW_SHA)
    from smollm3_observer_invariance_audit_v25 import supervisor_launch
    expected = []
    for run in spec['matrix']:
        for kind in ('server', 'client'):
            directory = WORK / run['run_id']
            intent = life.worker_command(SPEC, SPEC_SHA, REVIEW, REVIEW_SHA, base,
                run['run_id'], kind, spec['attempt_ids'][run['run_id']])
            expected.append((intent['unit'], directory, kind, intent, life.WRAPPER))
    expected.append((life.UNIT, CONTROL, 'supervisor', supervisor_launch(arguments), life.CONTROLLER))
    proofs = []
    for row, (unit, directory, kind, expected_intent, script) in zip(added, expected):
        assert row['unit'] == unit
        assert row['recorded_cgroups'] == ['/sys/fs/cgroup' + life.GROUP_PREFIX + unit + '.service']
        assert row['historical_invocation_unknown'] is False
        launch_path = directory / (kind + '-launch.json')
        owner_path = directory / (kind + '-owner.json')
        binding_path = directory / (kind + '-service-binding.json')
        if kind != 'supervisor':
            assert row['runtime_evidence_root'] == str(directory)
            assert row['successful_execution_proven'] is False
        if row['never_launched']:
            assert row['absent_records'] == [str(launch_path), str(owner_path)]
            assert row['owner'] is row['captured_process_owner'] is None
            assert row['invocation_id'] is row['installed_argv'] is None
            assert not any(exists(path) for path in (launch_path, owner_path, binding_path))
            proofs.append(dict(unit=unit, never_launched=True,
                absent_records=[str(launch_path), str(owner_path), str(binding_path)]))
            continue
        assert row['never_launched'] is False and row['absent_records'] == []
        intent, launch_ref = read_json(launch_path)
        owner, owner_ref = read_json(owner_path)
        binding, binding_ref = read_json(binding_path)
        for ref in (launch_ref, owner_ref, binding_ref):
            assert candidate['source_sha256'].get(ref['path']) == ref['sha256']
        assert intent == expected_intent
        assert row['captured_process_owner'] == owner
        assert row['owner'] == dict(pid=owner['pid'], start_ticks=owner['start_ticks'],
            unit=unit, script=str(script))
        assert owner['unit'] == unit and owner['argv'] == intent['argv']
        assert owner['cgroup'] == '0::' + life.GROUP_PREFIX + unit + '.service'
        assert row['invocation_id'] == owner['invocation_id']
        assert row['installed_argv'] == ' '.join(intent['execargv'])
        if kind != 'supervisor':
            assert owner['attempt_id'] == intent['attempt_id']
        life.verify_retained_binding(intent, owner, binding)
        proofs.append(dict(unit=unit, never_launched=False, original_owner=owner_ref,
            original_launch=launch_ref, original_binding=binding_ref))
    return proofs


def physical_drain(candidate):
    for process in candidate['processes']:
        life.gone(process)
    observations = []
    for row in candidate['rows']:
        name = row['unit']
        samples = []
        for index in range(2):
            value = life.props(name)
            assert life.terminal(value) and value['Id'] == name + '.service'
            if row['owner'] is None:
                assert row['never_launched'] is True and value['LoadState'] == 'not-found'
                assert not value['InvocationID'] and all(not Path(p).exists() for p in row['absent_records'])
            else:
                life.gone(row['owner'])
                if value['LoadState'] == 'not-found':
                    assert not value['InvocationID'] and value['ExecMainPID'] == '0'
                else:
                    assert value['InvocationID'] == row['invocation_id']
                    assert value['ExecMainPID'] == str(row['owner']['pid'])
                    assert life.installed(value) == row['installed_argv']
            paths = set(row['recorded_cgroups']) | {'/sys/fs/cgroup' + life.GROUP_PREFIX + name + '.service'}
            groups = []
            for path in sorted(paths):
                group = path.removeprefix('/sys/fs/cgroup')
                assert group == life.GROUP_PREFIX + name + '.service'
                groups.append(life.cgroup_inventory(group, require_empty=True))
            samples.append(dict(properties=value, groups=groups))
            if index == 0:
                time.sleep(.02)
        assert samples[0] == samples[1]
        observations.append(dict(unit=name, samples=samples))
    running = life.command(['systemctl', '--user', 'list-units', '--type=service',
        '--state=running,activating,deactivating', '--plain', '--no-legend', '--no-pager'])
    assert not [line for line in running.splitlines() if line.split() and line.split()[0].startswith('trn1-')]
    return observations


def validate_matrix_api(spec, base, captures, matrix):
    """Recompute saved API comparisons; mode/producer proof stays a separate audit."""
    assert set(captures) == {run['run_id'] for run in spec['matrix']}
    requests = {}; ids = set()
    for run in spec['matrix']:
        name = run['run_id']
        assert captures[name]['run'] == run
        requests[name] = analysis.validate_capture(base, name, captures[name])
        for request in requests[name]:
            assert request['request_id'] not in ids
            ids.add(request['request_id'])
    assert len(ids) == matrix['requests'] == 264
    assert sum(len(row['token_ids']) for rows in requests.values() for row in rows) == matrix['output_tokens'] == 33696
    pairs = []; repeats = []
    for role in protocol.ROLES:
        runs = [run for run in spec['matrix'] if run['role'] == role]
        for pair in (0, 1):
            off = next(run['run_id'] for run in runs if run['pair'] == pair and run['mode'] == 'off')
            on = next(run['run_id'] for run in runs if run['pair'] == pair and run['mode'] == 'on')
            pairs.append(dict(role=role, pair=pair, off=off, on=on,
                comparison=analysis.compare_requests(requests[off], requests[on])))
        for mode in ('off', 'on'):
            left, right = [run['run_id'] for run in runs if run['mode'] == mode]
            repeats.append(dict(role=role, mode=mode, left=left, right=right,
                comparison=analysis.compare_requests(requests[left], requests[right])))
    assert matrix['pairs'] == pairs and matrix['repeats'] == repeats
    assert matrix['historical'] == {name:value['historical'] for name,value in captures.items()}
    equal = all(row['comparison']['equal'] for row in pairs + repeats)
    assert matrix['observer_api_invariance_passed'] is matrix['passed'] is equal
    assert matrix['protocol_passed'] is matrix['diagnostic_complete'] is True
    assert matrix['full_model_qualified'] is matrix['new_speedup_claim'] is False
    return dict(captures=12, requests=264, output_tokens=33696, paired_comparisons=6,
        repeat_comparisons=6, api_invariance=equal, saved_api_comparisons_recomputed=True,
        observer_producer_and_lifecycle_proof_independently_recomputed=False)



def output_inventory(root):
    """Preserve every experiment-output byte, including interrupted writes.

    Model/input paths are outside this private root. No raw bytes enter source
    pins. Per-file and aggregate reads are bounded by original output budgets.
    """
    root = Path(root)
    before_root = life.directory_identity(root)
    paths = sorted(root.rglob('*'))
    assert len(paths) <= 100000
    files = []; dirs = {}; sockets = []; total = 0
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            dirs[str(path)] = life.directory_identity(path)
            continue
        before = path.lstat()
        if stat.S_ISSOCK(before.st_mode):
            relative = path.relative_to(root)
            assert re.fullmatch('t[0-9]{2}', relative.parts[0])
            assert relative.parts[0][1:] in {r['run_id'][1:] for r in protocol.matrix()}
            assert before.st_uid == os.getuid() and len(os.fsencode(path)) <= 107
            sockets.append(dict(path=str(path), identity=transport.store.identity(before)))
            continue
        before = transport.store.ordinary(path)
        assert before.st_size <= 64 * 2**20
        total += before.st_size
        assert total <= protocol.OUTPUT_CAP
        digest = hashlib.sha256(); count = 0
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            assert transport.store.identity(os.fstat(fd)) == transport.store.identity(before)
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                while True:
                    raw = stream.read(min(2**20, 64 * 2**20 + 1 - count))
                    if not raw: break
                    count += len(raw); assert count <= 64 * 2**20
                    digest.update(raw)
            assert count == before.st_size
            assert transport.store.identity(os.fstat(fd)) == transport.store.identity(before)
        finally:
            os.close(fd)
        assert transport.store.identity(transport.store.ordinary(path)) == transport.store.identity(before)
        files.append(dict(path=str(path), sha256=digest.hexdigest(), bytes=count,
                          identity=transport.store.identity(before), pending=path.name.startswith('.pending-')))
    assert paths == sorted(root.rglob('*'))
    assert life.directory_identity(root) == before_root
    for name, identity in dirs.items(): assert life.directory_identity(name) == identity
    for item in files:
        assert transport.store.identity(transport.store.ordinary(item['path'])) == item['identity']
    for item in sockets:
        current = Path(item['path']).lstat()
        assert stat.S_ISSOCK(current.st_mode) and transport.store.identity(current) == item['identity']
    return dict(files=files, bytes=total, directories=dirs, sockets=sockets, all_output_bytes_retained=True,
                pending_files=[item for item in files if item['pending']], model_payload_reads=False,
                raw_output_added_to_source_pins=False)


def validate_extra_evidence(args, closure, inventory, exit_ref):
    """Extra bytes remain evidence and require a concrete root disposition."""
    required = closure['extra_evidence_requires_review'] or bool(inventory['pending_files'])
    if not required:
        assert args.extra_evidence_review is None and args.extra_evidence_review_sha256 is None
        return None
    assert args.extra_evidence_review is not None
    path = Path(args.extra_evidence_review)
    assert path.parent == RESEARCH and path.suffix == '.json'
    value, ref = load(path)
    assert ref['sha256'] == args.extra_evidence_review_sha256
    assert value['reviewer'] == '/root' and value['passed'] is True
    assert value['actual_external_exit'] == exit_ref
    assert value['catalog_reference'] == {k:closure['catalog_reference'][k] for k in ('path','sha256')}
    assert value['closure_sha256'] == protocol.digest(closure)
    assert value['inventory_sha256'] == protocol.digest(inventory)
    assert value['retain_all_bytes'] is True and value['scientific_success_inferred'] is False
    assert type(value['disposition']) is str and value['disposition'].strip()
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    global SPEC, SPEC_SHA, REVIEW, REVIEW_SHA, LIVE_SHA, SESSION, PRIOR_ROWS, PRIOR_PROCESSES
    for field in ('spec', 'review', 'live', 'external-exit'):
        parser.add_argument('--' + field, required=True)
        parser.add_argument('--' + field + '-sha256', required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--session', required=True, type=int)
    parser.add_argument('--extra-evidence-review')
    parser.add_argument('--extra-evidence-review-sha256')
    parser.add_argument('--tool-chunk', required=True)
    args = parser.parse_args()
    for field in ('spec', 'review', 'live', 'external_exit'):
        path = Path(getattr(args, field))
        assert path.parent == RESEARCH and path.suffix == '.json' and path.resolve() == path
        assert re.fullmatch('[0-9a-f]{64}', getattr(args, field + '_sha256'))
    assert re.fullmatch('[0-9a-f]{64}', args.candidate_sha256)
    assert type(args.session) is int and args.session > 0
    SPEC, SPEC_SHA = Path(args.spec), args.spec_sha256
    REVIEW, REVIEW_SHA = Path(args.review), args.review_sha256
    LIVE_SHA, SESSION = args.live_sha256, args.session
    assert bool(args.extra_evidence_review) == bool(args.extra_evidence_review_sha256)
    assert re.fullmatch('[0-9a-f]{6}', args.tool_chunk)
    assert sys.dont_write_bytecode and __debug__
    protocol.performance_guard()
    audit_path = RESEARCH / 'smollm3-observer-sio14-root-terminal-audit-20261108-v1.json'
    catalog_path = RESEARCH / 'root-post-smol-sio14-complete-predecessors-20261108-v1.json'
    assert not audit_path.exists() and not catalog_path.exists()
    fd = os.open(life.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec, spec_ref = load(SPEC)
        assert spec_ref['sha256'] == SPEC_SHA
        review, review_ref = load(REVIEW)
        assert review_ref['sha256'] == REVIEW_SHA
        assert review['reviewer'] == '/root' and review['passed'] is review['independent_review'] is True
        assert review['execution_spec'] == spec_ref and review['spec_sha256'] == SPEC_SHA
        assert review['runtime_source_sha256'] == spec['runtime_source_sha256']
        plan = life.reference(spec['source_template'])
        base = life.reference(plan['base_plan'])
        handoff = life.reference(spec['callable_handoff'])
        assert plan['base_plan']['sha256'] == protocol.BASE_SHA
        assert protocol.digest(base) == protocol.BASE_CANONICAL_SHA
        life.validate_spec(spec, plan, base)
        prior = life.reference(spec['final_predecessor_catalog'])
        PRIOR_ROWS, PRIOR_PROCESSES = len(prior['rows']), len(prior['processes'])
        for path in (Path(__file__).resolve(), Path(transport.__file__).resolve()):
            assert spec['runtime_source_sha256'].get(str(path)) == reference(path)['sha256']
        assert review['passed'] is True and review['independent_review'] is True
        references = dict(execution_spec=spec_ref, execution_review=review_ref,
            previous_catalog=spec['final_predecessor_catalog'], base=plan['base_plan'],
            source_template=spec['source_template'], callable_handoff=spec['callable_handoff'])
        values = {}
        for key, path in {
            'candidate': CONTROL / 'successor-catalog.json',
            'terminal': CONTROL / 'external-terminal.json',
            'release': CONTROL / 'external-release.json',
            'controller': CONTROL / 'supervisor-terminal.json',
            'supervisor': CONTROL / 'supervisor-original-exit.json',
            'supervisor_owner': CONTROL / 'supervisor-owner.json',
            'supervisor_launch': CONTROL / 'supervisor-launch.json',
            'supervisor_binding': CONTROL / 'supervisor-service-binding.json',
            'identity': CONTROL / 'final-identity-rehash.json',
            'source': CONTROL / 'source-admission.json',
            'payload': CONTROL / 'payload-admission.json',
            'live': Path(args.live),
            'exited': Path(args.external_exit),
        }.items():
            values[key], references[key] = load(path)
        assert references['exited']['sha256'] == args.external_exit_sha256
        v = values
        catalog_ref = dict(path=str(CONTROL / 'successor-catalog.json'), sha256=args.candidate_sha256)
        assert references['candidate'] == catalog_ref
        expanded, closure = transport.collect(WORK, catalog_ref)
        assert expanded == v['candidate']
        owner = v['live']['owner']
        qualified, complete = validate_outcome(v['terminal'], v['release'], v['controller'], v['supervisor'], v['exited'], args.tool_chunk)
        payload_stats = validate_proof(spec, base, handoff['original_metadata_symlink_resolutions'],
            v['source'], v['payload'], v['identity'])
        added = validate_extension(prior, v['candidate'], spec, owner)
        validate_owners(v, references, spec_ref, review_ref, added)
        original_rows = validate_original_rows(spec, base, v['candidate'], load,
            lambda path: path.exists() or path.is_symlink())
        assert life.source_closure(plan, spec['runtime_source_sha256'],
            handoff['original_metadata_symlink_resolutions'], hash_bytes=False) == v['identity']['source_stats']
        life.verify_stat_closure(payload_stats)
        captures = []; capture_values = {}; api_reaudit = None
        for row in spec['matrix']:
            path = WORK / row['run_id'] / 'capture-result.json'
            if path.exists():
                capture, ref = load(path)
                captures.append(dict(run_id=row['run_id'], capture=ref, requests=len(capture['requests'])))
                capture_values[row['run_id']] = capture
        if complete:
            matrix, references['matrix'] = load(CONTROL / 'matrix-result.json')
            assert matrix['observer_api_invariance_passed'] is v['terminal']['observer_api_invariance_passed']
            assert len(captures) == 12 and sum(row['requests'] for row in captures) == 264
            assert all(row['owner'] is not None for row in added)
            api_reaudit = validate_matrix_api(spec, base, capture_values, matrix)
        drains = physical_drain(v['candidate'])
        output_budget = life.monitor('r00', reserve=False)
        inventory = output_inventory(WORK)
        extra_review = validate_extra_evidence(args, closure, inventory, references['exited'])
        if extra_review is not None: references['extra_evidence_review'] = extra_review
        for name, digest in spec['canonical_sha256'].items():
            assert reference(ROOT / name)['sha256'] == digest
        audit = dict(schema='smollm3_sio14_root_terminal_audit_v1', reviewer='/root', passed=True,
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), references=references,
            source=reference(Path(__file__).resolve()), transport_closure=closure,
            output_inventory=inventory, output_budget=output_budget, extra_evidence_review=extra_review, complete_rows=PRIOR_ROWS + 25, complete_processes=PRIOR_PROCESSES + 1,
            all_planned_launches_accounted_for=True, all_groups_drained=True,
            launched_units=sum(row['owner'] is not None for row in added),
            never_launched_units=sum(row['never_launched'] for row in added),
            original_row_evidence=original_rows,
            completed_captures=captures, api_reaudit=api_reaudit, physical_observations=drains,
            actual_external_exit_code=v['exited']['exit_code'], experiment_qualified=qualified,
            protocol_completed=complete,
            original_controller_error=copy.deepcopy(v['controller']['original_error']),
            original_controller_cleanup_errors=copy.deepcopy(v['controller']['cleanup_errors']),
            external_cleanup_errors=copy.deepcopy(v['terminal']['cleanup_errors']),
            controller_cleanup_failure_recovered_by_external=bool(v['controller']['cleanup_errors']),
            observer_api_invariance_passed=v['terminal']['observer_api_invariance_passed'],
            full_model_qualified=False, new_speedup_claim=False, canonical_unchanged=True,
            canonical_sha256=spec['canonical_sha256'],
            scope='Terminal metadata, physical drain, authenticated chunk closure and bounded experiment-output inventory; original model/input payload byte proof retained from frozen external auditor')
        record(audit_path, audit)
        result = copy.deepcopy(prior)
        for key in ('rows', 'processes', 'complete_count', 'source_sha256', 'historical_payload_sha256'):
            result[key] = copy.deepcopy(v['candidate'][key])
        if 'process_predecessors' in result:
            result['process_predecessors'] = copy.deepcopy(result['processes'])
        result.update(schema='root_post_smollm3_sio14_complete_predecessors_v1', reviewer='/root',
            root_finalized=True, root_finalization_observed_at=audit['observed_at'],
            appended_from=spec['final_predecessor_catalog'], appended_terminal_audit=reference(audit_path))
        for ref in list(references.values()) + [audit['source'], reference(audit_path)]:
            pins = result['source_sha256']
            assert ref['path'] not in pins or pins[ref['path']] == ref['sha256']
            pins[ref['path']] = ref['sha256']
        for name, digest in closure['authenticated_source_sha256_additions'].items():
            pins = result['source_sha256']
            assert name not in pins or pins[name] == digest
            pins[name] = digest
        assert all(result['source_sha256'].get(k) == v for k, v in prior['source_sha256'].items())
        catalog_reference = record(catalog_path, result)
        print(json.dumps(dict(audit=reference(audit_path), catalog=catalog_reference,
            experiment_qualified=qualified, complete_rows=PRIOR_ROWS + 25, complete_processes=PRIOR_PROCESSES + 1)))
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
