"""Fixed full-v2 sequence on original Phase; no queue, retry or reused graphs."""
from pathlib import Path

from .contract import require, native_environment, phase_environment, digest, control_root
from .io import once, read, ref

ARMS = ('baseline', 'candidate')
STAGES = (('shared', 'prepare'), ('shared', 'build')) + tuple(
    (arm, name) for arm in ARMS for name in ('state_copy', 'numeric', 'cold', 'long', 'vectors203')) + (
    ('baseline', 'capture'), ('baseline', 'sampler_replay'), ('candidate', 'capture'), ('candidate', 'sampler_replay'),
    ('shared', 'sampler_controls'), ('shared', 'sampler_pair_gate'),
    ('baseline', 'quality'), ('candidate', 'quality'), ('shared', 'quality_pair_gate'),
    ('baseline', 'http'), ('candidate', 'http'), ('shared', 'full_pair_gate'), ('external', 'independent_final_audit'))


def build_inventory(spec):
    return Path(spec['output_root']) / 'build-inventory.json'


def manifest_sha(spec):
    report = read(ref(build_inventory(spec)))
    require(report['case'] == spec['case'] and report['complete'] is True and len(report['compiled_graph_files']) == 12,
            'complete fresh twelve-graph inventory required')
    path = Path(spec['case']['artifact']) / 'qwen35-ultra-artifact.json'
    value = ref(path)
    require(value['sha256'] == report['artifact_manifest_sha256'], 'new package manifest changed')
    return value['sha256']


def immutable_build_rows(spec):
    """Stat identities only; compiled/checkpoint/tensor payloads are not opened."""
    from .budget import reviewed_budget
    from .io import identity
    import stat
    reviewed = reviewed_budget(spec); result = {}
    for key, kind in (("model_path", "view"), ("artifact", "artifact")):
        root = Path(spec["case"][key]); pending = [root]; count = 0
        while pending:
            path = pending.pop(); count += 1
            require(count <= reviewed["classes"][kind]["entries"], "bounded immutable build inventory")
            st = path.lstat()
            result[str(path)] = dict(device=st.st_dev, inode=st.st_ino, mode=st.st_mode, nlink=st.st_nlink,
                size=st.st_size, blocks=st.st_blocks, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
            if stat.S_ISDIR(st.st_mode): pending.extend(path.iterdir())
            else: require(stat.S_ISREG(st.st_mode) or kind == "view" and stat.S_ISLNK(st.st_mode), "ordinary immutable build entry")
    inventory = read(ref(build_inventory(spec)))
    for row in inventory["compiled_graph_files"]:
        path = Path(row["path"])
        require(Path(spec["case"]["compiler_workdir"]) in path.parents and path.name == "graph.neff", "declared fresh graph metadata")
        result[str(path)] = identity(path)
        require(result[str(path)]["size"] == row["bytes"], "compiled graph changed after real build")
    require(len(inventory["compiled_graph_files"]) == 12, "all twelve compiled graph identities")
    return result


def seal_build(spec):
    rows = immutable_build_rows(spec)
    require(immutable_build_rows(spec) == rows, "two stable immutable build inventories required")
    return once(Path(spec["output_root"]) / "build-immutable-identities.json",
        dict(case=spec["case"], manifest_sha256=manifest_sha(spec), rows=rows,
             binary_contents_rehashed_here=False, actual_build_wait_and_drain_observed=True))


def check_build_seal(spec):
    expected = ref(Path(spec["output_root"]) / "build-immutable-identities.json")
    completed = stage_receipt(spec, 2)
    if completed.exists():
        result = read(read(ref(completed))["result"])
        require(result["immutable_build"] == expected, "completed build seal reference changed")
    value = read(expected)
    require(value["case"] == spec["case"] and value["manifest_sha256"] == manifest_sha(spec)
            and value["actual_build_wait_and_drain_observed"] is True and value["rows"] == immutable_build_rows(spec),
            "prepared view/artifact/twelve graph identities changed after drained fresh build")
    return value


def phase_directory(spec, arm, name):
    root = Path(spec['output_root'])
    if name in ('prepare', 'build', 'sampler_controls'): return root / 'shared'
    if name in ('state_copy', 'numeric', 'cold', 'long', 'vectors203'): return root / arm / 'model'
    if name == 'sampler_replay': return root / arm
    require(name in ('capture', 'quality', 'http'), 'fixed phase directory')
    return root / arm / name


def child_environment(spec, phase):
    arm = 'baseline' if phase['arm'] == 'shared' else phase['arm']
    env = native_environment(spec['case'], arm=arm, compile_allowed=phase['phase'] == 'build')
    if phase['phase'] == 'capture':
        from qwen35_package_full_validation_v2.sampler import capture_environment
        env.update(capture_environment(spec['case'], arm, phase['directory'], manifest_sha(spec)))
        require(all(len((key + '=' + value).encode()) < 64 * 1024 for key, value in env.items()), 'bounded direct-child environment strings')
    return phase_environment(spec, phase, env)


def api_arguments(spec, arm, name):
    if name == 'capture':
        from qwen35_package_full_validation_v2.sampler import capture_server_arguments
        return capture_server_arguments(spec['case'], arm)
    require(name in ('quality', 'http'), 'exact plain API phase')
    from qwen35_package_full_validation_v2.runtime import server_arguments
    return server_arguments(spec['case'])


def stage_receipt(spec, index):
    return Path(spec['output_root']) / 'stages' / ('%02d.json' % index)


def require_prefix(spec, index):
    """The actual leaf/server may only follow the fixed completed prefix."""
    for n in range(1, index):
        row = read(ref(stage_receipt(spec, n)))
        require(row['index'] == n and (row['arm'], row['stage']) == STAGES[n - 1]
                and row['completed'] is True and row['case'] == spec['case'], 'fixed prior completed stage missing')
        # Ref authentication also prevents replacing a prior completed result.
        read(row['result'])
    require(not stage_receipt(spec, index).exists(), 'no stage restart or reuse')


def require_phase(spec, arm, name, directory):
    require(Path(directory) == phase_directory(spec, arm, name), 'exact fixed package stage output')
    index = STAGES.index((arm, name)) + 1
    require_prefix(spec, index)
    return index


def leaf(spec, phase, guard):
    from qwen35_package_full_validation_v2 import entry
    name, arm = phase['phase'], phase['arm']
    directory = Path(phase['directory']); root = Path(spec['output_root']); case = spec['case']
    require_phase(spec, arm, name, directory)
    guard()
    if name == 'prepare':
        result = entry.prepare_view(case, guard)
        once(root / 'prepare-view.json', result)
    elif name == 'build':
        result = entry.build_fresh(case, build_inventory(spec), guard)
        require(result['complete'] is True and len(result['compiled_graph_files']) == 12
                and result['external_compiler_exit_audit_required'] is True, 'actual twelve-graph build required')
    elif name in ('state_copy', 'numeric', 'cold', 'long'):
        result = entry.run_model_stage(name, directory, build_inventory(spec), guard)
    elif name == 'vectors203':
        result = entry.compare_vectors(directory, guard)
        require(result['frames'] == result['identical_vectors'] == 203, 'all 203 original raw vectors required')
        once(directory / 'vectors203.json', result)
    elif name in ('sampler_replay', 'sampler_controls'):
        result = entry.run_sampler_stage(name, case, root, guard, None if arm == 'shared' else arm)
    else:
        raise ValueError('undeclared package leaf')
    guard()
    return result


def client_operation(adapter, function, *args):
    """Original operation-boundary failure latch; no instant cancellation claim."""
    phase = adapter.active
    require(phase is not None, 'client requires actual active owned API')
    phase.reject_api_failure(); adapter.observer.check(full=True)
    result = function(*args)
    require(adapter.active is phase, 'active API changed during operation')
    phase.reject_api_failure(); adapter.observer.check(full=True)
    return result


def run_server(adapter, arm, name, hosts):
    from .adapter import Phase
    from qwen35_package_full_validation_v2 import sampler, http
    spec = adapter.spec; case = spec['case']; root = Path(spec['output_root'])
    directory = phase_directory(spec, arm, name)
    manager = Phase(adapter, arm, name, directory, adapter.admission['attempt_id'])
    result = {}; live = None
    try:
        with manager:
            live = adapter.server_audit(manager, 'live', manager.run_id)
            once(directory / 'root-live.json', live)
            guard = lambda: adapter.guard('client_' + name, manager.run_id)
            if name == 'capture':
                sampler.check_capture_attachment(manager.attachment, live, case, arm, directory, manifest_sha(spec))
                result = client_operation(adapter, sampler.run_capture_diagnostic, case, root / arm / 'diagnostic-http.json', guard)
            elif name == 'quality':
                for stage in ('smoke', 'quality96'):
                    client_operation(adapter, http.run_http_stage, stage, case, directory / (stage + '.json'), guard)
                result = {'smoke': ref(directory / 'smoke.json'), 'quality96': ref(directory / 'quality96.json')}
            else:
                hosts[arm] = live
                if arm == 'candidate':
                    http.check_matched_hosts(hosts['baseline'], live, case, {a: str(root / a / 'http') for a in ARMS})
                for stage in ('initial120', 'workloads'):
                    client_operation(adapter, http.run_http_stage, stage, case, directory / (stage + '.json'), guard)
                result = {'initial120': ref(directory / 'initial120.json'), 'workloads': ref(directory / 'workloads.json')}
    finally:
        # The original context drains even on a diagnostic/client exception.
        terminal = adapter.server_audit(manager, 'terminal', manager.run_id)
        once(directory / 'root-terminal.json', terminal)
    require(terminal.get('stop_outcome_accepted') is True and terminal.get('cleanup_complete') is True,
            'actual API child wait, supervisor terminal and recursive drain required before next stage')
    return dict(result=result, live=ref(directory / 'root-live.json'), terminal=ref(directory / 'root-terminal.json'))


def run(adapter):
    from .adapter import Phase
    from qwen35_package_full_validation_v2 import sampler, http
    spec = adapter.spec; case = spec['case']; root = Path(spec['output_root'])
    require(not root.exists(), 'fresh output root; no resume')
    root.mkdir(); (root / 'stages').mkdir(); (root / 'shared').mkdir()
    for arm in ARMS:
        (root / arm).mkdir(); (root / arm / 'model').mkdir()
        once(root / arm / 'model/case.json', case)
        for name in ('capture', 'quality', 'http'):
            directory = root / arm / name
            directory.mkdir(); (directory / 'tmp').mkdir(); (directory / 'rpc').mkdir()
    hosts = {}; receipts = []
    for index, (arm, name) in enumerate(STAGES[:-1], 1):
        require_prefix(spec, index)
        adapter.guard('before_' + name, adapter.admission['attempt_id'])
        require(adapter.active is None, 'prior actual phase must be completely drained')
        if name in ('capture', 'quality', 'http'):
            result = run_server(adapter, arm, name, hosts)
        elif name == 'sampler_pair_gate':
            result = sampler.check_saved_sampler_gates(case, root)
        elif name == 'quality_pair_gate':
            result = http.check_saved_quality(case, root / 'baseline', root / 'candidate')
        elif name == 'full_pair_gate':
            result = http.compare_saved_pair(case, root / 'baseline', root / 'candidate', hosts,
                {a: str(root / a / 'http') for a in ARMS})
        else:
            result = Phase(adapter, arm, name, phase_directory(spec, arm, name), adapter.admission['attempt_id']).leaf()
            if name == 'build': result['immutable_build'] = seal_build(spec)
        require(adapter.active is None, 'result cannot advance while phase still owned/live')
        adapter.guard('after_' + name, adapter.admission['attempt_id'])
        result_ref = once(root / 'stages' / ('%02d-result.json' % index), result)
        receipts.append(once(stage_receipt(spec, index), dict(index=index, arm=arm, stage=name, case=case,
            result=result_ref, completed=True, independent_final_audit_required=True, full_model_qualified=False)))
    pending = once(root / 'independent-final-audit-required.json', dict(index=25, stage=STAGES[-1][1],
        stage_receipts=receipts, completed=False, actual_external_tool_exit_required=True,
        full_model_qualified=False, automatic_best_promotion=False))
    return dict(stage_receipts=receipts, completed_stages=24, final_audit_pending=pending,
                full_model_qualified=False, performance_qualified=False, automatic_best_promotion=False)
