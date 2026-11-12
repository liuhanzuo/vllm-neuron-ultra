"""Independent actual API/loaded-worker observations, without plain worker hooks."""
import copy
from pathlib import Path
import os
import re
import stat
import urllib.error
import urllib.request

from .contract import STARTUP_FILES, LIMITS, digest, require
from .io import once, read_if_present, process_api
from .wiring import api_arguments, child_environment, manifest_sha


def bounded_proc(path, cap):
    with Path(path).open('rb') as stream:
        value = stream.read(cap + 1)
    require(len(value) <= cap, 'bounded actual process metadata')
    return value


def environment(owner):
    raw = bounded_proc(Path('/proc', str(owner['pid']), 'environ'), 256 * 1024)
    rows = [row.decode().split('=', 1) for row in raw.split(b'\0') if row]
    require(all(len(row) == 2 for row in rows) and len({row[0] for row in rows}) == len(rows), 'unique actual API environment entries')
    return dict(rows)


def group_processes(group):
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    pending = [root]; pids = set(); groups = 0
    while pending:
        path = pending.pop(); groups += 1
        require(groups <= 256 and path.is_dir() and not path.is_symlink(), 'finite ordinary owned cgroup tree')
        rows = bounded_proc(path / 'cgroup.procs', 65536).decode().split()
        require(all(row.isdigit() for row in rows), 'actual cgroup PID list')
        pids.update(map(int, rows)); require(len(pids) <= 256, 'original TasksMax process bound')
        for child in path.iterdir():
            require(not child.is_symlink(), 'ordinary owned cgroup entries')
            if child.is_dir(): pending.append(child)
    owners = [process_api().proc(pid) for pid in sorted(pids)]
    require(all(owner is not None for owner in owners), 'owned process disappeared during readiness observation')
    return owners


def listener_owned(api):
    rows = bounded_proc('/proc/net/tcp', 2 * 1024**2).decode().splitlines()[1:]
    endpoint = '0100007F:%04X' % 8010
    inodes = [row.split()[9] for row in rows if len(row.split()) >= 10
              and row.split()[1] == endpoint and row.split()[3] == '0A']
    require(len(inodes) == 1, 'one actual bound localhost API listener required')
    path = Path('/proc', str(api['pid']), 'fd')
    descriptors = list(path.iterdir()); require(len(descriptors) <= 4096, 'bounded actual API descriptors')
    sockets = []
    for fd in descriptors:
        try: target = os.readlink(fd)
        except FileNotFoundError: continue
        if target.startswith('socket:['): sockets.append(target)
    require('socket:[' + inodes[0] + ']' in sockets, 'healthy endpoint must be owned by exact API process')
    return {'address': '127.0.0.1', 'port': 8010, 'socket_inode': inodes[0], 'api_pid': api['pid']}


def completed_engine_line(log, worker):
    """Pinned EngineCore logs this only after executor.load_model + warmup.

    Observe bounded live stdout; do not demand static log size while active.
    The original supervisor owns/forwards the stream and owns its finite cap.
    """
    before = Path(log).lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= LIMITS['server_log_bytes'], 'bounded owned regular server log')
    fd = os.open(log, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), 'server log inode changed')
        data = b''
        while len(data) < before.st_size:
            block = os.read(fd, min(65536, before.st_size - len(data)))
            require(block, 'server log truncated while reading'); data += block
        after = os.fstat(fd)
        require(after.st_size >= before.st_size and after.st_size <= LIMITS['server_log_bytes'], 'server log truncated/exceeded cap')
    finally:
        os.close(fd)
    clean = re.sub(r'\x1b\[[0-9;]*m', '', data.decode('utf-8', errors='replace'))
    expression = r'^\(EngineCore_DP([0-9]+) pid=([0-9]+)\) .*init engine \(profile, create kv cache, warmup model\) took [0-9]+\.[0-9]+ seconds$'
    lines = [match for line in clean.splitlines() if (match := re.fullmatch(expression, line))]
    # EngineCore and API readiness run independently of supervisor PIPE forwarding.
    # A missing completion line is pending inside the original startup deadline.
    if not lines: return None
    require(len(lines) == 1 and lines[0].group(1) == '0' and int(lines[0].group(2)) == worker['pid'],
            'unique same-PID completed EngineCore load/warmup log required')
    return {'path': str(log), 'device': before.st_dev, 'inode': before.st_ino,
            'line': lines[0].group(0), 'observed_prefix_bytes': before.st_size, 'worker_pid': worker['pid']}


def plain_worker(manager, api):
    owners = group_processes(manager.service.group)
    engines = [row for row in owners if row['argv'] == ['VLLM::EngineCore']]
    require(len(engines) == 1, 'one actual default EngineCore in owned cgroup required')
    worker = engines[0]
    require(worker['cgroup'] == '0::' + manager.service.group and worker['ppid'] == api['pid']
            and int(worker['start_ticks']) >= int(api['start_ticks']), 'fresh direct loaded worker PID/start/argv/cgroup required')
    loaded = completed_engine_line(manager.phase['log'], worker)
    require(process_api().proc(worker['pid']) == worker, 'actual worker identity changed while observing readiness')
    if loaded is None: return None
    return worker, {'loaded_worker': worker, 'load_completion': loaded,
        'worker_class_from_pinned_platform': 'vllm_neuron.worker.neuron_worker.NeuronWorker',
        'load_order_from_pinned_executor_and_engine': True, 'custom_worker_or_attachment': False}


def api_ready(manager):
    manager.reject_api_failure(); manager.adapter.observer.check(); manager.service.live()
    intent = read_if_present(manager.stage / 'api-exec-intent.json')
    if intent is None: return None
    require(intent['phase_spec'] == manager.phase_ref and intent['bootstrap_owner'] == manager.child, 'exact acknowledged API exec transition')
    current = process_api().proc(manager.child['pid'])
    require(current is not None and all(current[k] == manager.child[k] for k in ('start_ticks', 'ppid', 'cgroup')), 'same child identity survives API exec')
    if current['argv'] == manager.child['argv']: return None
    spec = manager.adapter.spec
    args = api_arguments(spec, manager.arm, manager.name)
    require(current['argv'] == intent['argv'] == args, 'exact package API argv required')
    actual = environment(current); expected = child_environment(spec, manager.phase)
    require(actual == expected and digest(actual) == intent['environment_sha256'], 'actual clean package API exec environment changed')
    require(not any(k.startswith(('VLLM_DP_', 'TRN1_QWEN35', 'QWEN35_SAMPLER_')) or k in
        ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'VLLM_SERVER_DEV_MODE', 'TRN1_CPU_SAMPLER_FAST', 'TRN1_SAMPLER_DIAGNOSTIC_DIR') for k in actual), 'no legacy/DP overrides')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8010/health', timeout=2) as response:
            body = response.read(4097)
            require(response.status == 200 and len(body) <= 4096, 'actual bounded health response')
    except (urllib.error.URLError, TimeoutError): return None
    endpoint = listener_owned(current)
    attachment = None
    if manager.name == 'capture':
        attachment = read_if_present(manager.directory / 'worker-attachment.json')
        require(attachment is not None, 'healthy capture requires its completed attachment')
        worker = process_api().proc(attachment['pid'])
        require(worker is not None and worker['cgroup'] == '0::' + manager.service.group
                and worker['ppid'] == current['pid'] and worker['argv'] == ['VLLM::EngineCore']
                and int(worker['start_ticks']) >= int(current['start_ticks']), 'actual capture worker PID/start/argv/parent/cgroup')
        loaded = completed_engine_line(manager.phase['log'], worker)
        if loaded is None: return None
        worker_proof = {'loaded_worker': worker, 'capture_attachment_observed': True, 'custom_worker_or_attachment': True,
                        'load_completion': loaded}
    else:
        require(not (manager.directory / 'worker-attachment.json').exists()
                and not any(k.startswith('QWEN35_PACKAGE_CAPTURE_') for k in actual), 'plain quality/HTTP cannot carry capture attachment')
        observed = plain_worker(manager, current)
        if observed is None: return None
        worker, worker_proof = observed
    settings = {'case': spec['case'], 'arm': manager.arm, 'phase': manager.name,
                'manifest_sha256': manifest_sha(spec), 'compile_allowed': False}
    proof = {'actual_api_initial_environment_sha256': digest(actual), 'actual_api_initial_environment': copy.deepcopy(actual),
        'actual_api_argv': current['argv'], 'effective_host_settings': settings,
        'startup_source_sha256': {path: spec['source_sha256'][path] for path in STARTUP_FILES},
        'dp_size_derived_from_pinned_source_clean_environment_and_argv': 1,
        'dp_rank_derived_from_pinned_source_clean_environment_and_argv': 0,
        'parallel_config_object_directly_observed': False, 'owned_listener': endpoint, 'worker_proof': worker_proof}
    if manager.name == 'capture':
        from qwen35_package_full_validation_v2.sampler import check_capture_attachment
        live = {'observation_kind': 'live_ready', 'server_active': True, 'ready': True, 'exit_observed': False,
            'proc_owner_matches': True, 'invocation_matches': True, 'worker_in_server_cgroup': True,
            'worker_owner': {**worker, 'start_ticks': int(worker['start_ticks'])}, 'api_process_owner': current,
            'startup_config_proof': proof}
        check_capture_attachment(attachment, live, spec['case'], manager.arm, manager.directory, manifest_sha(spec))
        if manager.startup_proof is None:
            packet_dir = manager.directory / 'packets'
            require(sorted(p.name for p in packet_dir.iterdir()) == ['writer-owner.json'], 'zero packets/receipt before diagnostic')
            require(read_if_present(packet_dir / 'writer-owner.json') == {'pid': attachment['pid'], 'arm': manager.arm, 'phase': 'correctness_capture'}, 'exact capture writer owner')
            proof.update(written_packets_before_diagnostic=0, receipt_before_diagnostic=False,
                zero_seen_count_claimed=False, actual_B1_B4_each16_still_required=True)
    if manager.startup_proof is None:
        manager.startup_proof = proof
        once(manager.stage / 'startup-config-proof.json', proof)
    else:
        require(all(manager.startup_proof[key] == proof[key] for key in ('actual_api_initial_environment', 'actual_api_argv', 'effective_host_settings', 'owned_listener')),
                'actual API startup binding changed')
        require(manager.startup_proof['worker_proof']['loaded_worker'] == worker, 'loaded worker changed after first readiness')
    require(process_api().proc(current['pid']) == current and process_api().proc(worker['pid']) == worker, 'readiness process identities changed')
    manager.api_owner, manager.attachment = current, attachment
    manager.reject_api_failure()
    return {'attachment': attachment, 'worker': worker, 'api_owner': current}
