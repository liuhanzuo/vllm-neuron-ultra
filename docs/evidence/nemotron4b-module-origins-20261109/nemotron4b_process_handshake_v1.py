"""Stdlib process identity handshake, before a future native/helper entrypoint.

No subprocess launch, source authentication, systemd mutation or serial
admission occurs here. A reviewed successor must call child_wait before native
imports, bind the wrapper's actual argv, and authenticate the eventual original
helper source separately. This is incompatible with frozen v2's direct-helper
argv contract; v2 is deliberately not modified or claimed executable.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import stat
import struct
import sys
import time

ROLES = ('native-meta', 'rank-0', 'rank-1', 'helper-0', 'helper-1')
MAX_FRAME = 4096


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def scope(nonce, role):
    require(type(nonce) is str and re.fullmatch('[0-9a-f]{32}', nonce), 'Exact attempt nonce')
    require(role in ROLES, 'Exact prepared participant role')


def bounded(path, cap):
    with Path(path).open('rb') as f:
        raw = f.read(cap+1)
    require(len(raw) <= cap, 'Bounded actual process metadata')
    return raw


def owner(pid):
    require(type(pid) is int and pid > 0, 'Exact actual positive PID')
    p = Path('/proc')/str(pid)
    def status():
        fields = bounded(p/'stat', 8192).decode().rsplit(') ', 1)[1].split()
        require(fields[0] not in ('Z', 'X', 'x'), 'Actual participant must be live')
        return fields[19], int(fields[1])
    before = status()
    argv = bounded(p/'cmdline', 16384)
    require(argv.endswith(b'\0') and argv != b'\0', 'Actual complete argv')
    group = bounded(p/'cgroup', 4096).decode().strip()
    require(group.startswith('0::/') and '\n' not in group, 'Actual single cgroup-v2 identity')
    require(status() == before, 'Actual process identity stable during capture')
    return dict(pid=pid, start_ticks=before[0], ppid=before[1],
                argv=argv[:-1].decode().split('\0'), cgroup=group)


def pipe_fd(fd, access):
    require(type(fd) is int and fd >= 3 and stat.S_ISFIFO(os.fstat(fd).st_mode),
            'Inherited anonymous pipe descriptor')
    require(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == access,
            'Correct pipe direction')
    require(str(Path('/proc/self/fd', str(fd)).readlink()).startswith('pipe:['),
            'Anonymous pipe, not a filesystem FIFO')
    os.set_inheritable(fd, False)


def send(fd, value, deadline):
    """Send one bounded atomic frame. Caller closes the write end afterwards."""
    pipe_fd(fd, os.O_WRONLY)
    raw = encode(value)
    frame = struct.pack('!I', len(raw))+raw
    require(0 < len(raw) and len(frame) <= MAX_FRAME <= os.fpathconf(fd, 'PC_PIPE_BUF'),
            'One bounded atomic handshake frame')
    require(time.monotonic() < deadline, 'Handshake send deadline')
    require(select.select([], [fd], [], max(0, deadline-time.monotonic()))[1], 'Handshake send timeout')
    require(os.write(fd, frame) == len(frame), 'Complete atomic handshake write')


def receive(fd, deadline):
    """Require the complete single frame and writer EOF within the deadline."""
    pipe_fd(fd, os.O_RDONLY)
    def exact(size):
        parts = []
        while size:
            left = deadline-time.monotonic()
            require(left > 0 and select.select([fd], [], [], max(0, left))[0], 'Handshake receive timeout')
            raw = os.read(fd, size)
            require(raw, 'Handshake pipe closed before complete message')
            parts.append(raw)
            size -= len(raw)
        return b''.join(parts)
    size = struct.unpack('!I', exact(4))[0]
    require(0 < size <= MAX_FRAME-4, 'Bounded declared handshake length')
    raw = exact(size)
    left = deadline-time.monotonic()
    require(left > 0 and select.select([fd], [], [], max(0, left))[0], 'Handshake writer did not close')
    require(os.read(fd, 1) == b'', 'Only one handshake frame; trailing bytes rejected')
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate handshake field')
            result[key] = value
        return result
    result = json.loads(raw, object_pairs_hook=pairs)
    require(type(result) is dict and encode(result) == raw, 'Canonical handshake object')
    return result


def child_wait(ready_fd, release_fd, *, nonce, role, deadline):
    """No caller native/helper work may occur before this function returns."""
    scope(nonce, role)
    require(ready_fd != release_fd, 'Distinct handshake pipe ends')
    require(not {'torch', 'transformers', 'vllm', 'neuronx_distributed', 'libneuronxla'} & set(sys.modules),
            'Child handshake must precede native and SDK helper imports')
    actual = owner(os.getpid())
    ready = dict(schema='nemotron4b_process_ready_v1', nonce=nonce, role=role,
                 owner=actual, child_ready_ns=time.monotonic_ns())
    try:
        send(ready_fd, ready, deadline)
    finally:
        os.close(ready_fd)
    try:
        grant = receive(release_fd, deadline)
    finally:
        os.close(release_fd)
    require(set(grant) == {'schema', 'nonce', 'role', 'owner', 'parent', 'ready_sha256',
                           'capture_ns', 'guard_complete_ns', 'grant_ns'}, 'Exact release fields')
    require(grant['schema'] == 'nemotron4b_process_release_v1'
            and grant['nonce'] == nonce and grant['role'] == role
            and grant['ready_sha256'] == hashlib.sha256(encode(ready)).hexdigest(),
            'Exact original ready message and scope bound to release')
    require(owner(os.getpid()) == grant['owner'] == actual,
            'Actual child identity unchanged at release')
    require(grant['parent']['pid'] == actual['ppid'] and owner(actual['ppid']) == grant['parent'],
            'Actual parent identity unchanged at release')
    released = time.monotonic_ns()
    require(all(type(grant[k]) is int for k in ('capture_ns', 'guard_complete_ns', 'grant_ns'))
            and 0 < ready['child_ready_ns'] <= grant['capture_ns'] <= grant['guard_complete_ns']
            <= grant['grant_ns'] <= released, 'Actual capture/guard/grant/release ordering')
    return dict(ready=ready, grant=grant, child_release_ns=released,
                native_or_helper_started_by_this_module=False, runtime_admitted=False)


def parent_release(pid, ready_fd, release_fd, *, nonce, role, expected_argv,
                   expected_cgroup, guard, deadline):
    """Capture actual identity twice and run the caller guard before release.

    The caller owns process launch, a PID handle, all exits and group cleanup.
    On failure this function closes both pipe ends, sends no successful grant
    and never signals any process. It cannot establish whole-tree drain.
    """
    scope(nonce, role)
    require(ready_fd != release_fd, 'Distinct handshake pipe ends')
    try:
        guard()
        ready = receive(ready_fd, deadline)
        require(set(ready) == {'schema', 'nonce', 'role', 'owner', 'child_ready_ns'}
                and ready['schema'] == 'nemotron4b_process_ready_v1'
                and ready['nonce'] == nonce and ready['role'] == role,
                'Exact child ready message scope')
        first = owner(pid)
        captured = time.monotonic_ns()
        parent = owner(os.getpid())
        require(first == ready['owner'] and first['pid'] == pid
                and first['ppid'] == os.getpid() and first['argv'] == expected_argv
                and first['cgroup'] == expected_cgroup, 'Actual expected child identity')
        require(type(ready['child_ready_ns']) is int and 0 < ready['child_ready_ns'] <= captured,
                'Actual child ready precedes parent capture')
        guard()
        guarded = time.monotonic_ns()
        second = owner(pid)
        require(second == first and owner(os.getpid()) == parent,
                'Parent and child identities stable after guard')
        grant = dict(schema='nemotron4b_process_release_v1', nonce=nonce, role=role,
            owner=second, parent=parent, ready_sha256=hashlib.sha256(encode(ready)).hexdigest(),
            capture_ns=captured, guard_complete_ns=guarded, grant_ns=time.monotonic_ns())
        send(release_fd, grant, deadline)
        return dict(ready=ready, first_actual_capture=first, second_actual_capture=second,
                    grant=grant, child_release_observed=False,
                    native_or_helper_executed=False, runtime_admitted=False)
    finally:
        os.close(ready_fd)
        os.close(release_fd)


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires an immutable reviewed launcher/worker successor')
