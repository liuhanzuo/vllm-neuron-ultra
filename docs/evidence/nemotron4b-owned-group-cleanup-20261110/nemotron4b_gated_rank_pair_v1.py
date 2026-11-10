"""Concurrent TP2 child lifecycle for the future Nano4B root launcher.

Both exact rank identities must be captured and recorded before either is
released. This is a source-prepared library, not a launcher: the caller owns
serial admission, source/environment authentication, fresh output, native
helper records, and whole service/cgroup drain. Tests execute stdlib fixtures.
"""
import hashlib
import json
import os
import selectors
import signal
import subprocess
import time

from nemotron4b_process_handshake_v1 import (
    encode, owner, receive, require, scope, send,
)

ROLES = ('rank-0', 'rank-1')


def validate_ready(ready, role, nonce, actual, argv, cgroup, parent):
    require(set(ready) == {'schema', 'nonce', 'role', 'owner', 'child_ready_ns'}
            and ready['schema'] == 'nemotron4b_process_ready_v1'
            and ready['nonce'] == nonce and ready['role'] == role,
            'Exact rank ready message')
    require(actual == ready['owner'] and actual['ppid'] == parent['pid']
            and actual['argv'] == argv and actual['cgroup'] == cgroup,
            'Actual expected rank identity')
    require(type(ready['child_ready_ns']) is int
            and 0 < ready['child_ready_ns'] <= time.monotonic_ns(),
            'Actual rank ready timestamp')


def capture_and_release(rows, *, nonce, expected_cgroup, guard, record_pair, deadline, opened, capture_state):
    parent = owner(os.getpid())
    for role in ROLES:
        row = rows[role]
        ready = receive(row['ready_read'], deadline)
        actual = owner(row['child'].pid)
        validate_ready(ready, role, nonce, actual, row['argv'], expected_cgroup, parent)
        row['ready'] = ready
        row['first_actual_capture'] = actual
        row['capture_ns'] = time.monotonic_ns()
        os.close(row['ready_read'])
        opened.remove(row['ready_read'])

    guard()
    for role in ROLES:
        row = rows[role]
        second = owner(row['child'].pid)
        require(second == row['first_actual_capture'], 'Both rank identities stable before record')
        row['second_actual_capture'] = second
    require(owner(os.getpid()) == parent, 'Parent identity stable before pair record')
    # A real root must persist this complete actual pair before permitting imports.
    recorded = json.loads(encode(dict(schema='nemotron4b_pair_capture_v1', nonce=nonce, parent=parent,
        ranks={role: {key: rows[role][key] for key in (
            'argv', 'ready', 'first_actual_capture', 'second_actual_capture', 'capture_ns')}
            for role in ROLES})))
    capture_state['recorded'] = recorded
    receipt = record_pair(json.loads(encode(recorded)))
    require(type(receipt) is dict and receipt and len(encode(receipt)) <= 8192,
            'Caller must acknowledge durable pair capture')
    capture_state['receipt'] = json.loads(encode(receipt))
    guard()
    for role in ROLES:
        require(owner(rows[role]['child'].pid) == rows[role]['second_actual_capture'],
                'Both ranks remain exact after record and guard')
    require(owner(os.getpid()) == parent, 'Parent remains exact after record and guard')
    guarded = time.monotonic_ns()

    # No rank has received a grant before all checks above pass.
    for role in ROLES:
        row = rows[role]
        # Once released, the other rank may exit normally; never require it live.
        require(owner(row['child'].pid) == row['second_actual_capture'],
                'Unreleased rank remains exact at grant')
        grant = dict(schema='nemotron4b_process_release_v1', nonce=nonce, role=role,
            owner=row['second_actual_capture'], parent=parent,
            ready_sha256=hashlib.sha256(encode(row['ready'])).hexdigest(),
            capture_ns=row['capture_ns'], guard_complete_ns=guarded,
            grant_ns=time.monotonic_ns())
        send(row['release_write'], grant, deadline)
        row['grant'] = grant
        os.close(row['release_write'])
        opened.remove(row['release_write'])
    return recorded, receipt


def collect_pair(rows, *, deadline, caps, guard):
    with selectors.DefaultSelector() as selector:
        for role in ROLES:
            for name in ('stdout', 'stderr'):
                stream = getattr(rows[role]['child'], name)
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, (role, name))
        while True:
            guard()
            all_exited = True
            for role in ROLES:
                row = rows[role]
                code = row['child'].poll()
                if code is None:
                    all_exited = False
                else:
                    row.setdefault('exit_ns', time.monotonic_ns())
                    require(code == 0, role+' nonzero actual exit: '+str(code))
            if not selector.get_map() and all_exited:
                return
            remaining = deadline-time.monotonic()
            require(remaining > 0, 'TP2 pair execution/output deadline')
            if not selector.get_map():
                # Pipes may close before processes exit. Keep waiting bounded.
                for role in ROLES:
                    child = rows[role]['child']
                    if child.poll() is None:
                        try:
                            child.wait(timeout=min(.05, max(.001, remaining)))
                        except subprocess.TimeoutExpired:
                            pass
                        break
                continue
            for key, _ in selector.select(min(.05, remaining)):
                role, name = key.data
                buffer = rows[role]['output'][name]
                data = os.read(key.fileobj.fileno(), min(4096, caps[name]-len(buffer)+1))
                if not data:
                    selector.unregister(key.fileobj)
                else:
                    buffer.extend(data)
                    require(len(buffer) <= caps[name], role+' '+name+' cap exceeded')


def signal_owned(row, requested):
    child = row['child']
    if child.poll() is not None:
        return
    require(row.get('pidfd') is not None, 'Missing PID handle; outer group cleanup required')
    signal.pidfd_send_signal(row['pidfd'], requested)
    row['cleanup_signals'].append(signal.Signals(requested).name)


def cleanup(rows, opened):
    errors = []
    for fd in reversed(opened):
        try:
            os.close(fd)
        except OSError as error:
            errors.append(repr(error))
    # Signal all live ranks first so a blocked peer cannot delay the other.
    for role, row in rows.items():
        if 'child' in row:
            try:
                signal_owned(row, signal.SIGTERM)
            except BaseException as error:
                errors.append(role+': '+repr(error))
    deadline = time.monotonic()+1
    for role, row in rows.items():
        if 'child' not in row:
            continue
        try:
            child = row['child']
            try:
                child.wait(timeout=max(.001, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                signal_owned(row, signal.SIGKILL)
                child.wait(timeout=2)
            row.setdefault('exit_ns', time.monotonic_ns())
        except BaseException as error:
            errors.append(role+': '+repr(error))
        finally:
            for name in ('stdout', 'stderr'):
                stream = getattr(row['child'], name)
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as error:
                        errors.append(role+': '+repr(error))
            if row.get('pidfd') is not None:
                try:
                    os.close(row['pidfd'])
                except BaseException as error:
                    errors.append(role+': '+repr(error))
    return errors


def run(commands, *, environments, cwd, nonce, expected_cgroup, guard, record_pair,
        handshake_seconds=10, execution_seconds=300, stdout_cap=65536, stderr_cap=65536):
    """Start both gated ranks, capture both, then collect concurrent bounded IO.

    commands maps exactly rank-0/rank-1 to reviewed argv builders accepting
    (ready_write_fd, release_read_fd). record_pair must durably bind the actual
    identities before release; its receipt is retained without inventing any
    actual external exit. Direct-child success is not whole-tree success.
    """
    require(set(commands) == set(environments) == set(ROLES), 'Exactly both TP2 ranks')
    for role in ROLES:
        scope(nonce, role)
        require(callable(commands[role]) and type(environments[role]) is dict
                and all(type(k) is str and type(v) is str for k, v in environments[role].items()),
                'Explicit rank command and environment')
    require(callable(guard) and callable(record_pair), 'Root guard and durable capture callback')
    require(type(handshake_seconds) is int and 0 < handshake_seconds <= 20
            and type(execution_seconds) is int and 0 < execution_seconds <= 900,
            'Explicit bounded pair budgets')
    require(type(stdout_cap) is int and type(stderr_cap) is int
            and 0 < stdout_cap <= 2**20 and 0 < stderr_cap <= 2**20,
            'Explicit bounded rank output caps')
    opened, rows = [], {}
    failure = None
    capture_state = dict(recorded=None, receipt=None)
    try:
        # Resolve every command before creating either child.
        for role in ROLES:
            guard()
            ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
            opened.extend((ready_read, ready_write))
            release_read, release_write = os.pipe2(os.O_CLOEXEC)
            opened.extend((release_read, release_write))
            argv = commands[role](ready_write, release_read)
            require(type(argv) is list and len(argv) > 1
                    and all(type(x) is str and x for x in argv) and argv[0].startswith('/'),
                    'Exact explicit absolute rank command')
            rows[role] = dict(argv=argv, ready_read=ready_read, ready_write=ready_write,
                release_read=release_read, release_write=release_write,
                output=dict(stdout=bytearray(), stderr=bytearray()), cleanup_signals=[])
        for role in ROLES:
            row = rows[role]
            guard()
            child = subprocess.Popen(row['argv'], env=environments[role], cwd=cwd,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(row['ready_write'], row['release_read']), close_fds=True)
            row['child'] = child
            # An unreaped direct child cannot have its PID reused.
            row['pidfd'] = os.pidfd_open(child.pid, 0)
            for fd in (row['ready_write'], row['release_read']):
                os.close(fd)
                opened.remove(fd)
        capture_and_release(rows, nonce=nonce,
            expected_cgroup=expected_cgroup, guard=guard, record_pair=record_pair,
            deadline=time.monotonic()+handshake_seconds, opened=opened, capture_state=capture_state)
        collect_pair(rows, deadline=time.monotonic()+execution_seconds,
            caps=dict(stdout=stdout_cap, stderr=stderr_cap), guard=guard)
        guard()
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error)[:2048])
    finally:
        cleanup_errors = cleanup(rows, opened)
    participants = {}
    for role, row in rows.items():
        child = row.get('child')
        participants[role] = dict(
            argv=row['argv'], launched=child is not None,
            # PID alone is diagnostic; exact identity exists only after capture.
            launched_pid=None if child is None else child.pid,
            ready=row.get('ready'), first_actual_capture=row.get('first_actual_capture'),
            second_actual_capture=row.get('second_actual_capture'), grant=row.get('grant'),
            child_release_observed=False,
            actual_exit_code=None if child is None else child.returncode,
            actual_exit_observed=child is not None and child.returncode is not None,
            child_reaped=child is not None and child.returncode is not None,
            exit_ns=row.get('exit_ns'), cleanup_signals=row['cleanup_signals'],
            output={name: bytes(value) for name, value in row['output'].items()},
            output_may_be_partial=failure is not None,
        )
    passed = failure is None and not cleanup_errors and set(participants) == set(ROLES)
    passed = passed and all(row['child_reaped'] and row['actual_exit_code'] == 0
                           and row['grant'] is not None for row in participants.values())
    return dict(schema='nemotron4b_gated_rank_pair_v1', nonce=nonce,
        direct_children_successful=passed, ranks=participants,
        pair_capture=capture_state['recorded'], pair_capture_receipt=capture_state['receipt'],
        failure=failure, cleanup_errors=cleanup_errors,
        owned_group_drained_proven=False, runtime_admitted=False,
        model_or_native_runtime_qualified=False, full_model_qualified=False,
        new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Source preparation only; exact root/worker/helper and serial admission still required')
