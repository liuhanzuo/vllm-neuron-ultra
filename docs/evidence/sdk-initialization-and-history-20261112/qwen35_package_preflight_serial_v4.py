"""One future root-owned serial service for the immutable packaged preflight v5.

Import and --help are inert. Current spec/review/tool/handoff bindings are never
selected here. The external root observes the outer's actual tool termination.
"""
from __future__ import annotations
import argparse
import base64
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import subprocess
import sys
import time
import zlib

Q = Path(__file__).resolve().parent
ROOT = Q.parent
for directory in (Q, ROOT):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import qwen35_package_native_preflight_v5 as worker
from qwen35_sampler_runtime_contract_v12 import LOCK_PATH, LOCK_IDENTITY, reference, pin_map, require
from qwen35_sampler_runtime_io_v12 import (RetainedLock, identity, loaded_local_sources, once, process_api,
    raw, read, ref, service_owner, wait_until, descriptor, observed_stat, PublicationRace, transient)
from qwen35_sampler_runtime_service_v12 import Service, properties, verify_properties
from qwen35_sampler_runtime_history_v12 import expected_running, check_running
from qwen35_package_preflight_history_v2 import History, Authority
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained

SELF = str(Path(__file__).resolve())
PYTHON = worker.PYTHON
MIB, GIB = 2**20, 2**30
BUNDLE = dict(path=str(Q / 'qwen35-package-native-preflight-source-bundle-20261112-v5.json'), bytes=24356,
    sha256='75244d0b6af718436c77fc20c44fedf77a2bb1cd302dfdd9942cfd67566e333d')
LIMITS = dict(root_floor_bytes=28*GIB, nvme_floor_bytes=30*GIB, output_bytes=64*MIB,
    file_bytes=16*MIB, log_bytes=8*MIB, max_entries=512, memory_available_floor_bytes=2*GIB,
    outer_seconds=1800, supervisor_seconds=840, native_seconds=660, ack_seconds=180,
    cpus=[6, 7], nice=19, metadata_chunk_bytes=512*1024, complete_history_bytes=64*MIB)
PROPERTIES = dict(KillMode='control-group', RemainAfterExit='yes', SendSIGKILL='yes',
    WorkingDirectory=str(ROOT), TimeoutStopSec='15', LimitFSIZE=str(16*MIB), TasksMax='256',
    MemoryMax=str(12*GIB), MemoryHigh=str(10*GIB), MemorySwapMax='0', RuntimeMaxSec='900')


def names(spec):
    require(type(spec['run_name']) is str and re.fullmatch('[a-z0-9][a-z0-9-]{0,39}', spec['run_name']), 'Bounded fresh run name')
    return {'coordinator': 'trn1-qwen35-sampler-' + spec['run_name'] + '-package-native-preflight'}


def validate(spec, review, spec_ref):
    fields = {'schema', 'run_name', 'output_root', 'history_overlay', 'current_serial_catalog', 'source_sha256',
        'input_extension', 'running_handoff_path', 'lfm_r2_scope', 'bundle', 'limits', 'lock_identity', 'native_python', 'automatic_successor'}
    require(type(spec) is dict and set(spec) == fields and spec['schema'] == 'qwen35_package_preflight_serial_spec_v4', 'Exact single-stage spec')
    selected = names(spec)
    require(spec['output_root'] == '/mnt/nvme/neuron-work/qwen35-package-preflight-' + spec['run_name'], 'Fresh preflight-only output namespace')
    require(spec['bundle'] == BUNDLE and spec['limits'] == LIMITS and spec['native_python'] == PYTHON
        and spec['lock_identity'] == LOCK_IDENTITY and spec['automatic_successor'] is False, 'Immutable worker, original lock and exact resource policy')
    for key in ('history_overlay', 'current_serial_catalog', 'lfm_r2_scope'): reference(spec[key])
    target = Path(spec['running_handoff_path'])
    require(target.parent == Q and re.fullmatch('continuation-handoff-[a-z0-9-]+\\.json', target.name), 'Existing running-handoff namespace')
    pin_map(spec['source_sha256'])
    require(type(spec['input_extension']) is dict, 'Complete explicit metadata identities')
    for path, row in spec['input_extension'].items():
        require(type(row) is dict and set(row) == {'sha256', 'identity'} and type(row['identity']) is dict, 'Exact current input identity')
        reference(dict(path=path, sha256=row['sha256']))
    # Same exact root review envelope already used by Qwen runtime v12.
    require(type(review) is dict and set(review) == {'schema', 'reviewer', 'passed', 'blockers', 'spec', 'source_sha256', 'launch_authorized'}
        and review['schema'] == 'qwen35_sampler_runtime_source_review_v1' and review['reviewer'] == '/root'
        and review['passed'] is True and review['blockers'] == [] and review['spec'] == spec_ref
        and review['source_sha256'] == spec['source_sha256'] and review['launch_authorized'] is True,
        'Existing exact independent root source review contract')
    return selected


def environment(work):
    minimal = dict(HOME='/home/ubuntu', LANG='C.UTF-8', XDG_RUNTIME_DIR='/run/user/1000',
        DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus',
        PATH='/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin')
    return worker.environment(str(work), minimal)


def supervisor_argv(binding):
    return [PYTHON, '-I', '-B', SELF, '--mode', 'supervisor', '--binding', binding['path'], '--binding-sha256', binding['sha256']]


def native_argv(binding):
    return [PYTHON, '-I', '-B', str(Q / 'qwen35_package_native_preflight_v5.py'), '--execute-native',
        '--input', binding['path'], '--input-sha256', binding['sha256']]


def bound_ref(path):
    value = ref(path, 4*MIB)
    return dict(value, bytes=identity(path)['size'])


def history_parts(value):
    """Lossless bounded transport for the complete existing physical proof."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(len(payload) <= LIMITS['complete_history_bytes'], 'Complete history proof expansion ceiling')
    packed = zlib.compress(payload, 6)
    require(len(packed) <= 12*MIB, 'Complete history compressed ceiling; never truncate')
    size = LIMITS['metadata_chunk_bytes']
    parts = [dict(index=i//size, data=base64.b64encode(packed[i:i+size]).decode()) for i in range(0, len(packed), size)]
    return dict(encoding='zlib-base64-json-v1', raw_bytes=len(payload), raw_sha256=hashlib.sha256(payload).hexdigest(),
        compressed_bytes=len(packed), compressed_sha256=hashlib.sha256(packed).hexdigest(), chunks=len(parts)), parts


def join_history(manifest, parts):
    require(manifest['encoding'] == 'zlib-base64-json-v1' and manifest['chunks'] == len(parts)
        and 0 < len(parts) <= 24 and 0 < manifest['raw_bytes'] <= LIMITS['complete_history_bytes'], 'Complete bounded history transport')
    blocks = []
    for index, row in enumerate(parts):
        require(set(row) == {'index', 'data'} and row['index'] == index, 'Every history chunk in order')
        block = base64.b64decode(row['data'], validate=True)
        require(0 < len(block) <= LIMITS['metadata_chunk_bytes'], 'History chunk bound')
        blocks.append(block)
    packed = b''.join(blocks)
    require(len(packed) == manifest['compressed_bytes'] <= 12*MIB and hashlib.sha256(packed).hexdigest() == manifest['compressed_sha256'], 'Complete compressed proof hash')
    decoder = zlib.decompressobj(); payload = decoder.decompress(packed, manifest['raw_bytes'] + 1)
    require(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data and len(payload) == manifest['raw_bytes']
        and hashlib.sha256(payload).hexdigest() == manifest['raw_sha256'], 'Complete lossless proof')
    return json.loads(payload)


def save_history(root, value):
    manifest, parts = history_parts(value)
    require(join_history(manifest, parts) == value, 'Complete history serialization roundtrip')
    references = []
    for index, part in enumerate(parts):
        path = root / ('history-part-%03d.json' % index); once(path, part); references.append(bound_ref(path))
    manifest['references'] = references
    path = root / 'history-manifest.json'; once(path, manifest)
    return [bound_ref(path), *references]


def scan_output_once(root):
    """Count symlink inodes; never follow/read original tokenizer targets."""
    import stat
    root = Path(root); entries = allocated = logical = 0
    allowed = {str(root / 'native/shared-hf-config' / name): '/mnt/nvme/models/Qwen3.5-2B/' + name
        for name in ('tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.jinja')}
    pending = [root] if root.exists() else []
    while pending:
        path = pending.pop()
        try: info = observed_stat(path)
        except FileNotFoundError:
            if transient(path): raise PublicationRace('Known atomic output publication finished')
            raise
        entries += 1; allocated += info.st_blocks * 512
        require(entries <= LIMITS['max_entries'], 'Finite output entry count')
        if stat.S_ISLNK(info.st_mode):
            require(str(path) in allowed and os.readlink(path) == allowed[str(path)], 'Only exact prepared tokenizer aliases allowed')
        elif stat.S_ISDIR(info.st_mode): pending.extend(path.iterdir())
        else:
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Singleton regular output file')
            cap = LIMITS['log_bytes'] if path.suffix == '.log' else LIMITS['file_bytes']
            require(info.st_size <= cap, 'Per-file output/log ceiling')
            logical += info.st_size
    require(max(allocated, logical) <= LIMITS['output_bytes'], 'Whole single-stage output ceiling')
    return dict(entries=entries, allocated_bytes=allocated, logical_bytes=logical)


def scan_output(root):
    for attempt in range(8):
        try: return scan_output_once(root)
        except PublicationRace:
            if attempt == 7: raise ValueError('Atomic output publication did not settle')
            time.sleep(.01)


def resources(spec, reserve=False):
    usage = scan_output(spec['output_root']); result = {}
    for path, key in ((ROOT, 'root_floor_bytes'), (Path('/mnt/nvme'), 'nvme_floor_bytes')):
        fs = os.statvfs(path); available = fs.f_bavail * fs.f_frsize
        extra = max(0, LIMITS['output_bytes'] - usage['allocated_bytes']) if reserve and path.name == 'nvme' else 0
        require(available >= LIMITS[key] + extra, 'Original storage floor plus remaining reservation: ' + str(path))
        result[str(path)] = available
    require(Path('/mnt/nvme').is_mount() and ROOT.stat().st_dev != Path('/mnt/nvme').stat().st_dev, 'Actual distinct NVMe filesystem')
    text = Path('/proc/meminfo').read_text()
    available = int(next(line.split()[1] for line in text.splitlines() if line.startswith('MemAvailable:'))) * 1024
    require(available >= LIMITS['memory_available_floor_bytes'], 'Original memory available floor')
    return dict(storage=usage, available_disk_bytes=result, available_memory_bytes=available)


def make_worker_input(spec, owner, group, history):
    value = dict(schema=worker.SCHEMA, bundle=copy.deepcopy(BUNDLE), work=str(Path(spec['output_root']) / 'native'),
        binding=dict(owner=dict(pid=owner['pid'], start_ticks=int(owner['start_ticks'])),
            lock=dict(path=str(LOCK_PATH), **LOCK_IDENTITY), history=copy.deepcopy(history), cgroup=group))
    return worker.validate_spec(value)


def load_spec(spec_ref, review_ref):
    spec, review = read(spec_ref, 64*MIB), read(review_ref, 64*MIB)
    validate(spec, review, spec_ref)
    worker.validate_bundle(json.loads(worker.checked(BUNDLE)))
    loaded_local_sources(spec['source_sha256'])
    return spec


def check_ack(spec, admission_ref, admission, owner_ref, owner):
    """Reuse the exact established root running-handoff shape and latest check."""
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    root = Path(spec['output_root']); ack = read(ref(root / 'running-ack.json'))
    require(ack['admission'] == admission_ref and ack['coordinator_owner'] == owner_ref and ack['coordinator'] == owner, 'Exact root running ACK')
    running = ack['running']; value = read(running['reference'])
    prior = read(admission['freshness']['latest_handoff'])
    expected = expected_running(prior, value['updated_at'], value['qwen35_sampler_current_execution']['tool'],
        admission_ref, admission, owner_ref, owner)
    require(value == expected and running['value'] == value and latest.latest_handoff() == running['observation'], 'Complete current actual-tool running binding')
    return running


def capture_native(command, env, root, guard):
    """Wait the real native parent; the root also drains the entire service."""
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    owner = None; log_bytes = 0; start = time.monotonic(); selector = selectors.DefaultSelector()
    try:
        require(child.stdout is not None, 'Owned native log pipe')
        selector.register(child.stdout, selectors.EVENT_READ)
        def executed():
            current = process_api().proc(child.pid)
            return current if current is not None and current['argv'] == command else None
        owner = wait_until(executed, guard, 10, message='Actual native child exec identity not observed')
        require(owner['ppid'] == os.getpid() and owner['argv'] == command
            and owner['cgroup'] == process_api().proc(os.getpid())['cgroup'], 'Actual native parent PID/argv/cgroup')
        once(root / 'native-owner.json', owner)
        with (root / 'native.log').open('xb') as stream:
            while selector.get_map() or child.poll() is None:
                guard(); require(time.monotonic() - start < LIMITS['native_seconds'], 'Native process wall deadline')
                for key, _ in selector.select(.1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block: selector.unregister(key.fileobj); continue
                    log_bytes += len(block); require(log_bytes <= LIMITS['log_bytes'], 'Native combined log ceiling')
                    stream.write(block); stream.flush()
            code = child.wait(timeout=10)
        value = dict(owner=owner, argv=command, actual_returncode=code, popen_wait_observed=True, log_bytes=log_bytes,
            parent_observer=process_api().proc(os.getpid()), self_exit_claimed=False, whole_cgroup_drain_pending=True)
        once(root / 'native-process-exit.json', value)
        require(code == 0, 'Actual native preflight process failed')
        return value
    finally:
        selector.close()
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        if child.stdout is not None: child.stdout.close()
        if not (root / 'native-process-exit.json').exists():
            once(root / 'native-process-exit.json', dict(owner=owner, argv=command, actual_returncode=child.returncode,
                popen_wait_observed=True, log_bytes=log_bytes, self_exit_claimed=False,
                interrupted_or_failed=True, whole_cgroup_drain_pending=True))


def supervisor(binding_ref):
    admission = read(binding_ref); spec = load_spec(admission['spec'], admission['review'])
    require(sys.version_info[:2] == (3, 10) and Path(sys.executable).absolute() == Path(PYTHON), 'Exact original native interpreter')
    os.sched_setaffinity(0, set(LIMITS['cpus'])); os.setpriority(os.PRIO_PROCESS, 0, LIMITS['nice'])
    resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS['file_bytes'],)*2)
    root = Path(spec['output_root']); started = time.monotonic()
    unit = names(spec)['coordinator']; command = supervisor_argv(binding_ref); env = environment(root / 'native')
    require(all(os.environ.get(k) == v for k, v in env.items()), 'Exact clean supervisor/native environment')
    own = process_api().proc(os.getpid()); props = verify_properties(properties(unit), unit, command, env, PROPERTIES, root / 'service.log')
    require(own['argv'] == command and own['cgroup'] == '0::' + process_api().GROUP_PREFIX + unit + '.service', 'Actual supervisor identity')
    owner = dict(process=own, service_owner=service_owner(unit, own, props))
    owner_ref = once(root / 'supervisor-owner.json', owner)
    running = None
    def guard():
        descriptor(admission['outer_owner'], admission['retained_fd'])
        require(time.monotonic() - started < LIMITS['supervisor_seconds'], 'Finite supervisor window')
        require(process_api().proc(os.getpid()) == own, 'Supervisor identity unchanged')
        if running is not None: check_ack(spec, binding_ref, admission, owner_ref, owner)
    wait_until(lambda: True if (root / 'running-ack.json').exists() else None, guard, LIMITS['ack_seconds'], message='Root actual tool/running ACK missing')
    running = check_ack(spec, binding_ref, admission, owner_ref, owner)
    guard()
    input_ref = admission['worker_input']; value = json.loads(worker.checked(input_ref)); worker.validate_spec(value)
    require(value == make_worker_input(spec, admission['outer_owner'], process_api().GROUP_PREFIX + unit + '.service', admission['worker_history']), 'Exact v5 root owner/FLOCK/cgroup/history input')
    native_exit = capture_native(native_argv(input_ref), env, root, guard)
    result_ref = ref(root / 'native/result.json', LIMITS['file_bytes']); result = read(result_ref, LIMITS['file_bytes'])
    require(result['schema'] == worker.SCHEMA + '-result' and result['passed'] is result['complete'] is True
        and result['root_binding'] == value['binding'] and result['source_bundle'] == BUNDLE, 'Exact complete native v5 result')
    require(result['parent']['pid'] == native_exit['owner']['pid']
        and result['parent']['start_ticks'] == int(native_exit['owner']['start_ticks'])
        and result['parent']['ppid'] == own['pid']
        and result['child']['ppid'] == result['parent']['pid'] and result['child']['pid'] != result['parent']['pid'],
        'V5 evidence belongs to the actual waited native parent and its spawn child')
    guard(); loaded_local_sources(spec['source_sha256'])
    once(root / 'supervisor-result.json', dict(complete=True, passed=True, admission=binding_ref, owner=owner,
        native_result=result_ref, native_process_exit=ref(root / 'native-process-exit.json'), self_exit_claimed=False,
        full_model_qualified=False, automatic_successor=False))
    return 0


def owned_terminal(service):
    """Re-observe the original no-install proof without requiring a loaded unit."""
    if service is None or not service.started: return None
    if service.terminal is not None and service.terminal.get('never_loaded') is True:
        # Original cleanup rechecks not-found, all captured identities absent,
        # and the exact empty group. A now-installed unit takes its original
        # authenticated cleanup path; an observed unit cannot disappear here.
        return service.cleanup()
    return service.drained()


def outer(spec_ref, review_ref):
    spec = load_spec(spec_ref, review_ref); root = Path(spec['output_root'])
    require(root.parent.resolve() == root.parent and not root.exists() and not Path(spec['running_handoff_path']).exists(), 'Fresh ordinary one-stage output/handoff only')
    lock = RetainedLock(); lock.acquire()
    started = time.monotonic(); history = None; service = None; admission_ref = None; result = None; failure = None; running = None
    interrupted = [False]; handlers = {}; last_check = [-1.0]
    for sig in (signal.SIGINT, signal.SIGTERM): handlers[sig] = signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    def guard():
        lock.check(); require(not interrupted[0] and time.monotonic() - started < LIMITS['outer_seconds'], 'Finite/interruption-free outer work')
        if time.monotonic() - last_check[0] >= .5:
            resources(spec, reserve=True); last_check[0] = time.monotonic()
        if running is not None:
            from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
            require(latest.latest_handoff() == running['observation'], 'Unknown/replaced running handoff')
    try:
        guard(); process_api().active_heavy()
        history = History(spec, spec_ref, review_ref, names(spec), guard).resolve()
        pair = history.physical_pair(Authority(history, guard, lambda: [])); freshness = history.freshness()
        history.unchanged(); guard()
        unit = names(spec)['coordinator']; group = process_api().GROUP_PREFIX + unit + '.service'
        empty = properties(unit); require(empty['LoadState'] == 'not-found' and not empty['InvocationID'], 'Unused single service namespace')
        process_api().empty_group(group)
        root.mkdir(mode=0o700)
        worker_history = save_history(root, dict(physical_pair=pair, current_serial_catalog=spec['current_serial_catalog'],
            history_overlay=spec['history_overlay'], resolved_plan_sha256=history.resolved['plan_sha256'],
            metadata_identities=history.meta.identities, freshness=freshness))
        input_value = make_worker_input(spec, lock.owner, group, worker_history)
        once(root / 'worker-input.json', input_value); input_ref = bound_ref(root / 'worker-input.json')
        admission = dict(schema='qwen35_package_preflight_serial_admission_v4', spec=spec_ref, review=review_ref,
            outer_owner=lock.owner, retained_fd=lock.fd, initial_descriptor_proof=lock.initial, lock_identity=LOCK_IDENTITY,
            current_serial_catalog=spec['current_serial_catalog'], history_overlay=spec['history_overlay'],
            freshness=freshness, units=names(spec), worker_input=input_ref, worker_history=worker_history,
            full_model_qualified=False, self_exit_claimed=False, automatic_successor=False)
        admission_ref = once(root / 'admission.json', admission)
        service = Service(unit, supervisor_argv(admission_ref), environment(root / 'native'), PROPERTIES, root / 'service.log')
        history.unchanged(); guard(); service.start(guard)
        owner_ref = wait_until(lambda: ref(root / 'supervisor-owner.json') if (root / 'supervisor-owner.json').exists() else None,
            guard, 60, message='Supervisor exact owner ACK missing')
        require(read(owner_ref) == service.owner, 'Independently observed service owner equals worker ACK')
        print(json.dumps(dict(event='qwen35_package_preflight_running_handoff_required', admission=admission_ref,
            coordinator_owner=owner_ref, coordinator=service.owner, running_handoff_path=spec['running_handoff_path'],
            actual_external_tool_session_required=True, native_work_released=False)), flush=True)
        wait_until(lambda: True if Path(spec['running_handoff_path']).exists() else None, guard, LIMITS['ack_seconds'], message='Root actual tool/running handoff missing')
        running = check_running(history, admission_ref, admission, owner_ref, service.owner)
        once(root / 'running-ack.json', dict(admission=admission_ref, coordinator_owner=owner_ref, coordinator=service.owner, running=running))
        while True:
            guard(); history.unchanged()
            process_api().active_heavy((unit,))
            value = service.observe()
            if process_api().terminal(value): break
            time.sleep(.5)
        terminal = service.cleanup()  # Includes natural-exit resource-tracker drain before any success claim.
        require(terminal['systemd_exit_status'] == dict(ExecMainCode=1, ExecMainStatus=0, Result='success'), 'Actual preflight service failed')
        result = read(ref(root / 'supervisor-result.json'))
        require(result['complete'] is result['passed'] is True and result['admission'] == admission_ref and result['owner'] == service.owner, 'Exact supervisor result')
        history.unchanged(); guard()
    except BaseException as error:
        failure = error
    finally:
        def cleanup():
            lock.check()
            if service is not None and service.started: service.cleanup()
        def verify():
            lock.check()
            terminal = owned_terminal(service)
            return dict(physically_drained=True, terminal=terminal)
        def pending(value): print(json.dumps(dict(event='package_preflight_cleanup_blocked', serial_lease_retained=True, details=value)), flush=True)
        proof = retain_until_drained(cleanup, verify, pending, lambda: time.sleep(1))
        try:
            if admission_ref is not None:
                once(root / 'outer-terminal.json', dict(schema='qwen35_package_preflight_outer_terminal_v4', admission=admission_ref,
                    outer_owner=lock.owner, complete_owned_unit_terminals=[] if proof['terminal'] is None else [proof['terminal']],
                    whole_owned_scope_drained=True, result_present=result is not None, failed=failure is not None,
                    failure=None if failure is None else dict(type=type(failure).__name__, message=str(failure)[:2048]),
                    storage=scan_output(root), full_model_qualified=False, self_exit_claimed=False,
                    actual_external_tool_exit_pending=True, automatic_successor=False))
        finally:
            lock.release(proof['physically_drained'])
            for sig, handler in handlers.items(): signal.signal(sig, handler)
    if failure is not None: raise failure
    return 0


def reconcile(terminal, running, external_tool, *, actual_owner_gone, actual_service_drained):
    """Pure reconciliation of the external root's actual terminal, never self-exit."""
    require(type(external_tool) is dict and set(external_tool) == {'kind', 'session_id', 'exit_code', 'terminal_chunk'}
        and external_tool['kind'] == 'tool_session' and type(external_tool['session_id']) is int and external_tool['session_id'] > 0
        and type(external_tool['exit_code']) is int and 0 <= external_tool['exit_code'] <= 255
        and re.fullmatch('[a-f0-9]{3,64}', external_tool['terminal_chunk']), 'Actual external tool terminal fields required')
    require(terminal['schema'] == 'qwen35_package_preflight_outer_terminal_v4' and terminal['self_exit_claimed'] is False
        and terminal['actual_external_tool_exit_pending'] is True and terminal['whole_owned_scope_drained'] is True
        and actual_owner_gone is True and actual_service_drained is True, 'Independent post-tool owner/service drain required')
    execution = running['qwen35_sampler_current_execution']
    require(execution['admission'] == terminal['admission'] and execution['outer_owner'] == terminal['outer_owner']
        and execution['tool'] == {'kind': external_tool['kind'], 'session_id': external_tool['session_id']},
        'Actual tool terminal must belong to this original running execution')
    success = external_tool['exit_code'] == 0 and terminal['failed'] is False and terminal['result_present'] is True
    require(external_tool['exit_code'] != 0 or success, 'Actual exit zero cannot conceal failed/missing preflight result')
    return dict(schema='qwen35_package_preflight_reconciliation_v4', actual_tool=copy.deepcopy(external_tool),
        outer_owner=copy.deepcopy(terminal['outer_owner']), passed=success, actual_outer_exit_observed=True,
        independent_post_exit_drain=True, full_model_qualified=False, automatic_successor=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('outer', 'supervisor'), default='outer')
    for name in ('spec', 'spec-sha256', 'review', 'review-sha256', 'binding', 'binding-sha256'): parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    require(sys.dont_write_bytecode and __debug__, 'Explicit -B and original assertions')
    if args.mode == 'outer':
        require(all((args.spec, args.spec_sha256, args.review, args.review_sha256)) and not args.binding and not args.binding_sha256, 'Explicit root-reviewed future spec only')
        return outer(reference(dict(path=args.spec, sha256=args.spec_sha256)), reference(dict(path=args.review, sha256=args.review_sha256)))
    require(args.binding and args.binding_sha256 and not any((args.spec, args.spec_sha256, args.review, args.review_sha256)), 'Only exact admitted supervisor binding')
    return supervisor(reference(dict(path=args.binding, sha256=args.binding_sha256)))


if __name__ == '__main__': raise SystemExit(main())
