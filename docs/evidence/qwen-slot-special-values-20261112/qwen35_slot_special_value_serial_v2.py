"""Fixed independent special-value diagnostic; no original eight-layout qualification.

Imports and --help are inert. This source does not select a current catalog or
handoff. Original History, retained descriptor, Service and cleanup are reused.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time

Q = Path(__file__).resolve().parent
ROOT = Q.parent
for directory in (Q, ROOT):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
# -X applies before any local import, so -B cannot silently read an old pyc.
if __name__ == '__main__' and '--help' not in sys.argv:
    if not (sys.dont_write_bytecode and __debug__ and sys.flags.isolated
            and sys.pycache_prefix and not Path(sys.pycache_prefix).exists()):
        raise RuntimeError('Use -I -B -X pycache_prefix=<output>/bytecode-never-created')
from qwen35_slot_special_value_serial_contract_v2 import (LIMITS, PROPERTIES, PYTHON, SITE, SDK_ROOTS,
    LAYOUTS, MIB, GIB, protocol, diagnostic, names, validate, verify_delivery, environment, bytecode_prefix,
    check_environment, check_process_settings, update_json, current_layout, scan_output, resources, validate_report)
from qwen35_sampler_runtime_contract_v12 import LOCK_IDENTITY, require, reference
from qwen35_sampler_runtime_io_v12 import (RetainedLock, identity, loaded_local_sources, once,
    process_api, raw, read, ref, service_owner, wait_until, descriptor)
from qwen35_sampler_runtime_service_v12 import Service, properties, verify_properties
from qwen35_sampler_runtime_history_v12 import check_running
from qwen35_package_preflight_history_v2 import History, Authority
from qwen35_package_preflight_serial_v2 import save_history, check_ack, owned_terminal, bound_ref, join_history
from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained

from qwen35_predicated_slot_sdk_environment_v1 import SDKEnvironment

SELF = str(Path(__file__).resolve())


def load_spec(spec_ref, review_ref):
    spec, review = read(spec_ref, 64*MIB), read(review_ref, 64*MIB)
    validate(spec, review, spec_ref); verify_delivery(spec)
    loaded_local_sources(spec['source_sha256'])
    return spec


def entry_argv(mode, binding, work):
    return [PYTHON, '-I', '-B', '-X', 'pycache_prefix=' + str(bytecode_prefix(work)), SELF,
        '--mode', mode, '--binding', binding['path'], '--binding-sha256', binding['sha256']]


def supervisor_argv(binding):
    admission = read(binding); spec = read(admission['spec'], 64*MIB)
    return entry_argv('supervisor', binding, spec['output_root'])


def loaded_sources(pins):
    local = loaded_local_sources(pins)
    for module in dict(sys.modules).values():
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and any(Path(path).is_relative_to(SITE / name) for name in SDK_ROOTS):
            require(path in pins and ref(path, 4*MIB)['sha256'] == pins[path], 'Actually imported SDK Python source pin')
    return local


def group_members(group):
    p = process_api()
    require(group.startswith(p.GROUP_PREFIX + 'trn1-qwen35-sampler-') and '..' not in Path(group).parts, 'Exact owned service cgroup')
    root = Path('/sys/fs/cgroup') / group.lstrip('/'); pending = [root]; members = set(); count = 0
    while pending:
        path = pending.pop(); count += 1
        require(count <= 256 and not path.is_symlink(), 'Bounded ordinary recursive cgroups')
        text = (path / 'cgroup.procs').read_text(); require(len(text) <= 65536, 'Bounded cgroup process metadata')
        members.update(int(value) for value in text.split())
        for child in path.iterdir():
            require(not child.is_symlink(), 'No cgroup symlinks')
            if child.is_dir(): pending.append(child)
    return members


def check_group(group):
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    expected = {'memory.max': 6*GIB, 'memory.high': 5*GIB, 'memory.swap.max': 0, 'pids.max': 256}
    for key, wanted in expected.items():
        require((root / key).read_text().strip() == str(wanted), 'Exact cgroup resource: ' + key)
    events = dict(line.split() for line in (root / 'memory.events').read_text().splitlines())
    require(all(int(events.get(key, '0')) == 0 for key in ('oom', 'oom_kill', 'oom_group_kill')), 'No OOM may be hidden by a later successful exit')
    for pid in group_members(group):
        try: cpus = os.sched_getaffinity(pid)
        except ProcessLookupError: continue
        require(bool(cpus) and cpus <= set(LIMITS['cpus']), 'All compiler/native descendants remain on assigned CPUs')
    return expected


def only_supervisor(group, own):
    require(process_api().proc(own['pid']) == own and group_members(group) == {own['pid']}, 'No prior leaf/compiler descendant before the next layout')
    return True


def artifact_record(info, name, work):
    require(name == diagnostic.NAME and set(info) == {'arm','artifact','hlo','hlo_proof'}
        and info['arm'] == 'identity', 'Only one new identity artifact; original arms remain immutable inputs')
    require(info['artifact']['path'] == str(Path(work)/'artifacts/identity/model.pt')
        and info['hlo']['path'] == str(Path(work)/'artifacts/identity/model/graph.hlo'), 'Exact identity artifact paths')
    diagnostic.checked(info['artifact'])
    require(info['hlo_proof'] == diagnostic.identity_hlo_proof(diagnostic.checked(info['hlo'],16*MIB)), 'Complete identity HLO evidence')
    return info


def validate_layout(root, name, receipt, native_exit):
    work = root/name; result_ref = ref(work/'report.json',16*MIB); result = read(result_ref,16*MIB)
    binding_ref = ref(work/'layout-binding.json'); binding = read(binding_ref)
    info=read(ref(work/'artifact-identity.json')); artifact_record(info,name,work)
    summary=validate_report(result,name,work)
    require(next(row for row in result['graphs'] if row['arm']=='identity')['artifact']==info['artifact'], 'Saved identity artifact is actual called graph')
    require(receipt['layout'] == name and receipt['binding'] == binding_ref and receipt['report'] == result_ref
        and receipt['owner'] == native_exit['owner'] and binding['layout'] == name
        and native_exit['argv'] == entry_argv('layout',binding_ref,work)
        and native_exit['owner']['ppid'] == binding['supervisor']['process']['pid']
        and native_exit['owner']['cgroup'] == binding['supervisor']['process']['cgroup']
        and receipt['diagnostic_complete'] is True and receipt['qualification_matrix_completed'] is False
        and receipt['self_exit_claimed'] is False and native_exit['actual_returncode'] == 0
        and native_exit['popen_wait_observed'] is True and native_exit['self_exit_claimed'] is False,
        'Independent diagnostic completion belongs to actual waited leaf; no qualification')
    return dict(summary,report=result_ref,leaf_result=ref(work/'leaf-result.json'),process_exit=ref(work/'native-process-exit.json'))


def audit_results(root, result):
    require(result['counts'] == dict(diagnostic_calls=6,graphs=3)
        and [row['layout'] for row in result['layouts']] == list(LAYOUTS)
        and result['qualification_matrix_completed'] is False, 'One independent six-call diagnostic')
    name,=LAYOUTS
    require(validate_layout(root,name,read(ref(root/name/'leaf-result.json')),read(ref(root/name/'native-process-exit.json')))
        == result['layouts'][0], 'Independent full raw and difference revalidation')
    scan_output(root)


def capture_layout(command, env, work, binding_ref, own, guard, started):
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
    owner = None; log_bytes = 0; selector = selectors.DefaultSelector(); waited = False
    def bounded():
        guard(); require(time.monotonic() - started < 900, 'Per-layout wall deadline includes launch, compile, calls and drain')
    try:
        require(child.stdout is not None, 'Owned native log pipe'); selector.register(child.stdout, selectors.EVENT_READ)
        def executed():
            current = process_api().proc(child.pid)
            return current if current is not None and current['argv'] == command else None
        owner = wait_until(executed, bounded, 10, message='Actual leaf exec identity not observed')
        require(owner['ppid'] == own['pid'] and owner['cgroup'] == own['cgroup'], 'Real direct fresh child in the owned cgroup')
        once(work / 'native-owner.json', owner)
        once(work / 'layout-release.json', dict(binding=binding_ref, owner=owner))
        with (work / 'native.log').open('xb') as stream:
            while selector.get_map() or child.poll() is None:
                bounded()
                for key, _ in selector.select(.1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block: selector.unregister(key.fileobj); continue
                    log_bytes += len(block); require(log_bytes <= LIMITS['log_bytes'], 'Native combined log cap')
                    stream.write(block); stream.flush()
            code = child.wait(timeout=10); waited = True
        bounded()
        value = dict(owner=owner, argv=command, actual_returncode=code, popen_wait_observed=True, log_bytes=log_bytes,
            parent_observer=own, self_exit_claimed=False, whole_cgroup_drain_pending=True)
        once(work / 'native-process-exit.json', value)
        require(code == 0, 'Actual slot leaf failed; no retry or reduced workload')
        only_supervisor(own['cgroup'].removeprefix('0::'), own)
        return value
    finally:
        selector.close()
        if child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
            waited = True
        if not waited: child.wait(timeout=10)
        if child.stdout is not None: child.stdout.close()
        if not (work / 'native-process-exit.json').exists():
            once(work / 'native-process-exit.json', dict(owner=owner, argv=command, actual_returncode=child.returncode,
                popen_wait_observed=True, log_bytes=log_bytes, self_exit_claimed=False,
                interrupted_or_failed=True, whole_cgroup_drain_pending=True))


def run_layouts(spec, admission_ref, owner, guard):
    root = Path(spec['output_root']); summaries = []; started = time.monotonic()
    for name in LAYOUTS:
        guard(); require(time.monotonic() - started < 7200, 'Whole eight-layout wall deadline')
        only_supervisor(owner['process']['cgroup'].removeprefix('0::'), owner['process'])
        clock = dict(name=name, started=time.monotonic(), active=True)
        update_json(root / 'current-layout.json', clock, 4096)
        work = root / name; work.mkdir(mode=0o700)
        for directory in ('tmp', 'cache'): (work / directory).mkdir()
        binding = dict(schema='qwen35_slot_special_value_layout_binding_v2', admission=admission_ref, layout=name,
            supervisor=owner, started=clock['started'])
        binding_ref = once(work / 'layout-binding.json', binding)
        native = capture_layout(entry_argv('layout', binding_ref, work), environment(work), work,
            binding_ref, owner['process'], guard, clock['started'])
        receipt = read(ref(work / 'leaf-result.json'))
        summaries.append(validate_layout(root, name, receipt, native)); guard(); current_layout(root)
        update_json(root / 'current-layout.json', dict(clock, active=False), 4096)
    require(time.monotonic() - started < 7200, 'Whole eight-layout completion deadline')
    return summaries


def read_history_reference(row):
    require(set(row) == {'path', 'bytes', 'sha256'} and identity(row['path'])['size'] == row['bytes'], 'Complete history reference bytes')
    return read({key: row[key] for key in ('path', 'sha256')})


def supervisor(binding_ref):
    admission = read(binding_ref); spec = load_spec(admission['spec'], admission['review']); root = Path(spec['output_root'])
    require(sys.version_info[:2] == (3,10) and Path(sys.executable).absolute() == Path(PYTHON), 'Original interpreter')
    os.sched_setaffinity(0, set(LIMITS['cpus'])); os.setpriority(os.PRIO_PROCESS, 0, LIMITS['nice'])
    resource.setrlimit(resource.RLIMIT_CORE, (0,0)); resource.setrlimit(resource.RLIMIT_FSIZE, (GIB,GIB))
    check_environment(root); started = time.monotonic(); unit = names(spec)['coordinator']; env = environment(root)
    command = supervisor_argv(binding_ref); own = process_api().proc(os.getpid())
    props = verify_properties(properties(unit), unit, command, env, PROPERTIES, root / 'service.log')
    require(own['argv'] == command and own['cgroup'] == '0::' + process_api().GROUP_PREFIX + unit + '.service', 'Actual supervisor identity')
    owner = dict(process=own, service_owner=service_owner(unit, own, props)); owner_ref = once(root / 'supervisor-owner.json', owner)
    running = None; last_check = [-1.0]
    def guard():
        require(time.monotonic() - started < 7200, 'Supervisor total 7200 second deadline')
        if time.monotonic() - last_check[0] >= .5:
            descriptor(admission['outer_owner'], admission['retained_fd'])
            require(process_api().proc(os.getpid()) == own, 'Supervisor identity retained')
            check_environment(root); check_group(own['cgroup'].removeprefix('0::')); resources(spec)
            if running is not None: check_ack(spec, binding_ref, admission, owner_ref, owner)
            last_check[0] = time.monotonic()
    wait_until(lambda: True if (root / 'running-ack.json').exists() else None, guard, LIMITS['ack_seconds'], message='Root actual tool/running ACK missing')
    running = check_ack(spec, binding_ref, admission, owner_ref, owner)
    # Reconstruct all history chunks before releasing the first leaf; no projection.
    manifest = read_history_reference(admission['worker_history'][0]); parts = [read_history_reference(row) for row in admission['worker_history'][1:]]
    require(manifest['references'] == admission['worker_history'][1:], 'All original complete-history chunk references')
    history = join_history(manifest, parts)
    require(history['current_serial_catalog'] == spec['current_serial_catalog'] and history['history_overlay'] == spec['history_overlay'], 'Same complete root-owned history')
    summaries = run_layouts(spec, binding_ref, owner, guard)
    guard(); loaded_local_sources(spec['source_sha256'])
    once(root / 'supervisor-result.json', dict(complete=True, passed=True, admission=binding_ref, owner=owner,
        layouts=summaries, counts=dict(diagnostic_calls=6, graphs=3), qualification_matrix_completed=False, self_exit_claimed=False,
        full_model_qualified=False, automatic_successor=False))
    return 0


def layout(binding_ref):
    binding = read(binding_ref); admission_ref = binding['admission']; admission = read(admission_ref)
    spec = load_spec(admission['spec'], admission['review']); root = Path(spec['output_root']); name = binding['layout']; work = root / name
    require(set(binding) == {'schema','admission','layout','supervisor','started'}
        and binding['schema'] == 'qwen35_slot_special_value_layout_binding_v2' and name in LAYOUTS
        and binding_ref == ref(work / 'layout-binding.json'), 'Exact original bound layout')
    check_environment(work)
    require(sys.version_info[:2] == (3,10) and Path(sys.executable).absolute() == Path(PYTHON), 'Original leaf interpreter')
    own = process_api().proc(os.getpid()); supervisor_owner = binding['supervisor']; unit = names(spec)['coordinator']
    require(own['argv'] == entry_argv('layout', binding_ref, work) and own['ppid'] == supervisor_owner['process']['pid']
        and own['cgroup'] == supervisor_owner['process']['cgroup'], 'Actual leaf parent and cgroup')
    service = Service(unit, supervisor_argv(admission_ref), environment(root), PROPERTIES, root / 'service.log', owner=supervisor_owner)
    sdk_environment = SDKEnvironment(environment(work), os.environ, spec['source_sha256'])
    last_check = [-1.0]
    def guard(event):
        sdk_environment.check(event)
        if event.startswith('before_compile:') and sdk_environment.module is None:
            sdk_environment.install_context_fix(sys.modules.get('torch_neuronx.contexts'))
        require(time.monotonic() - binding['started'] < 900, 'Leaf wall deadline')
        if time.monotonic() - last_check[0] >= .5:
            descriptor(admission['outer_owner'], admission['retained_fd']); service.live(); check_process_settings(work)
            require(process_api().proc(os.getpid()) == own, 'Exact leaf identity retained')
            check_group(own['cgroup'].removeprefix('0::'))
            check_ack(spec, admission_ref, admission, ref(root / 'supervisor-owner.json'), supervisor_owner)
            last_check[0] = time.monotonic()
        if not event.startswith('before_call:'): loaded_sources(spec['source_sha256'])
    wait_until(lambda: True if (work / 'layout-release.json').exists() else None, lambda: guard('await_parent'), 10, message='Native parent release missing')
    require(read(ref(work / 'layout-release.json')) == dict(binding=binding_ref, owner=own), 'Parent observed this exact leaf before native imports')
    guard('before_leaf')
    from research.qwen35_slot_special_value_leaf_v2 import run_diagnostic
    def save(report): update_json(work / 'report.json', report, 16*MIB)
    def artifact(info):
        artifact_record(info, name, work)
        once(work / 'artifact-identity.json', info)
    try:
        report = run_diagnostic(work, guard=guard, save_report=save, record_artifact=artifact)
    except BaseException:
        try:
            sdk_environment.close()
        except BaseException:
            pass  # Keep the original failed leaf exception after attempting exact restoration.
        raise
    else:
        sdk_environment.close()
    guard('after_leaf'); loaded_sources(spec['source_sha256'])
    validate_report(report, name, work)
    once(work / 'leaf-result.json', dict(layout=name, binding=binding_ref, owner=own, report=ref(work / 'report.json',16*MIB),
        diagnostic_complete=True, qualification_matrix_completed=False, self_exit_claimed=False, full_model_qualified=False))
    return 0


def outer(spec_ref, review_ref):
    spec = load_spec(spec_ref, review_ref); root = Path(spec['output_root'])
    require(root.parent.resolve() == root.parent and not root.exists() and not Path(spec['running_handoff_path']).exists(), 'Fresh ordinary one-stage output/handoff only')
    lock = RetainedLock(); lock.acquire()
    started = time.monotonic(); history = None; service = None; admission_ref = None; result = None; failure = None; running = None
    interrupted = [False]; handlers = {}; last_check = [-1.0]; last_handoff = [-1.0]
    for sig in (signal.SIGINT, signal.SIGTERM): handlers[sig] = signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    def guard():
        lock.check(); require(not interrupted[0] and time.monotonic() - started < LIMITS['outer_seconds'], 'Finite/interruption-free outer work')
        if time.monotonic() - last_check[0] >= .5:
            resources(spec, reserve=True); last_check[0] = time.monotonic()
        if running is not None and time.monotonic() - last_handoff[0] >= .5:
            last_handoff[0] = time.monotonic()
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
        for child in ('tmp', 'cache'): (root / child).mkdir()
        worker_history = save_history(root, dict(physical_pair=pair, current_serial_catalog=spec['current_serial_catalog'],
            history_overlay=spec['history_overlay'], resolved_plan_sha256=history.resolved['plan_sha256'],
            metadata_identities=history.meta.identities, freshness=freshness))
        admission = dict(schema='qwen35_slot_special_value_serial_admission_v2', spec=spec_ref, review=review_ref,
            outer_owner=lock.owner, retained_fd=lock.fd, initial_descriptor_proof=lock.initial, lock_identity=LOCK_IDENTITY,
            current_serial_catalog=spec['current_serial_catalog'], history_overlay=spec['history_overlay'],
            freshness=freshness, units=names(spec), worker_history=worker_history, workload=diagnostic.workload(),
            full_model_qualified=False, self_exit_claimed=False, automatic_successor=False)
        admission_ref = once(root / 'admission.json', admission)
        service = Service(unit, supervisor_argv(admission_ref), environment(root), PROPERTIES, root / 'service.log')
        history.unchanged(); guard(); service.start(guard)
        owner_ref = wait_until(lambda: ref(root / 'supervisor-owner.json') if (root / 'supervisor-owner.json').exists() else None,
            guard, 60, message='Supervisor exact owner ACK missing')
        require(read(owner_ref) == service.owner, 'Independently observed service owner equals worker ACK')
        print(json.dumps(dict(event='qwen35_slot_special_value_running_handoff_required', admission=admission_ref,
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
        audit_results(root, result)
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
        def pending(value): print(json.dumps(dict(event='predicated_slot_cleanup_blocked', serial_lease_retained=True, details=value)), flush=True)
        proof = retain_until_drained(cleanup, verify, pending, lambda: time.sleep(1))
        try:
            if admission_ref is not None:
                once(root / 'outer-terminal.json', dict(schema='qwen35_slot_special_value_outer_terminal_v2', admission=admission_ref,
                    outer_owner=lock.owner, complete_owned_unit_terminals=[] if proof['terminal'] is None else [proof['terminal']],
                    whole_owned_scope_drained=True, result_present=result is not None, failed=failure is not None,
                    failure=None if failure is None else dict(type=type(failure).__name__, message=str(failure)[:2048]),
                    storage=scan_output(root), qualification_matrix_completed=False, full_model_qualified=False, self_exit_claimed=False,
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
    require(terminal['schema'] == 'qwen35_slot_special_value_outer_terminal_v2' and terminal['self_exit_claimed'] is False
        and terminal['actual_external_tool_exit_pending'] is True and terminal['whole_owned_scope_drained'] is True
        and actual_owner_gone is True and actual_service_drained is True, 'Independent post-tool owner/service drain required')
    execution = running['qwen35_sampler_current_execution']
    require(execution['admission'] == terminal['admission'] and execution['outer_owner'] == terminal['outer_owner']
        and execution['tool'] == {'kind': external_tool['kind'], 'session_id': external_tool['session_id']},
        'Actual tool terminal must belong to this original running execution')
    success = external_tool['exit_code'] == 0 and terminal['failed'] is False and terminal['result_present'] is True
    require(external_tool['exit_code'] != 0 or success, 'Actual exit zero cannot conceal failed/missing slot result')
    return dict(schema='qwen35_slot_special_value_reconciliation_v2', actual_tool=copy.deepcopy(external_tool),
        outer_owner=copy.deepcopy(terminal['outer_owner']), passed=success, actual_outer_exit_observed=True,
        independent_post_exit_drain=True, qualification_matrix_completed=False, full_model_qualified=False, automatic_successor=False)



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('outer','supervisor','layout'), default='outer')
    for name in ('spec','spec-sha256','review','review-sha256','binding','binding-sha256'): parser.add_argument('--'+name)
    args = parser.parse_args(argv)
    require(sys.dont_write_bytecode and __debug__, 'Explicit -B and original assertions')
    if args.mode == 'outer':
        require(all((args.spec,args.spec_sha256,args.review,args.review_sha256)) and not args.binding and not args.binding_sha256, 'Explicit future root spec/review only')
        spec_ref = reference(dict(path=args.spec,sha256=args.spec_sha256)); spec = read(spec_ref,64*MIB)
        require(sys.pycache_prefix == str(bytecode_prefix(spec['output_root'])), 'Exact output-bound absent bytecode prefix')
        return outer(spec_ref, reference(dict(path=args.review,sha256=args.review_sha256)))
    require(args.binding and args.binding_sha256 and not any((args.spec,args.spec_sha256,args.review,args.review_sha256)), 'Only original admitted binding')
    return (supervisor if args.mode == 'supervisor' else layout)(reference(dict(path=args.binding,sha256=args.binding_sha256)))


if __name__ == '__main__': raise SystemExit(main())
