"""Bounded supervisor for a future Nano4B gated worker/helper subprocess.

The caller supplies a reviewed exact command builder/environment and fresh
source/resource/serial guard. This module grants no runtime admission. Every
spawned child gets a kernel PID handle while unreaped; failure cleanup targets
only that child. Full owned service/cgroup descendant drain remains the outer
root supervisor's responsibility.
"""
import os
import selectors
import signal
import subprocess
import time

from nemotron4b_process_handshake_v1 import parent_release, require, scope


def signal_child(child, pidfd, requested):
    """Never target a process by an unowned numeric PID."""
    if child.poll() is not None:
        return False
    signal.pidfd_send_signal(pidfd, requested)
    return True


def collect(child, deadline, caps, guard, result):
    with selectors.DefaultSelector() as selector:
        for name in result:
            stream = getattr(child, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            guard()
            remaining = deadline-time.monotonic()
            require(remaining > 0, 'Gated child output/exit deadline')
            for key, _ in selector.select(min(remaining, .05)):
                name = key.data
                data = os.read(key.fileobj.fileno(), min(4096, caps[name]-len(result[name])+1))
                if not data:
                    selector.unregister(key.fileobj)
                else:
                    result[name].extend(data)
                    require(len(result[name]) <= caps[name], 'Gated child '+name+' cap exceeded')
        while child.poll() is None:
            guard()
            require(time.monotonic() < deadline, 'Gated child exit deadline')
            # Poll waits are bounded; do not busy-spin after pipes reach EOF.
            try: child.wait(timeout=min(.05, max(.001, deadline-time.monotonic())))
            except subprocess.TimeoutExpired: pass
    return {name: bytes(value) for name, value in result.items()}


def run(command, *, environment, cwd, nonce, role, expected_cgroup, guard,
        handshake_seconds=10, execution_seconds=20, stdout_cap=4096, stderr_cap=16384):
    """Return actual lifecycle/outcome, including failed-child cleanup evidence.

    command(ready_write_fd, release_read_fd) must return the reviewed exact
    Python wrapper argv, never an installed helper argv invented after exec.
    A failed result is not success even if forced cleanup drains the child.
    """
    scope(nonce, role)
    require(type(handshake_seconds) is int and 0 < handshake_seconds <= 20
            and type(execution_seconds) is int and 0 < execution_seconds <= 900,
            'Explicit bounded handshake and execution budgets')
    require(type(stdout_cap) is int and type(stderr_cap) is int
            and 0 < stdout_cap <= 2**20 and 0 < stderr_cap <= 2**20,
            'Explicit bounded output caps')
    child = None
    pidfd = None
    opened = []
    proof = None
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    failure = None
    cleanup_errors = []
    cleanup_signals = []
    actual_exit = None
    exit_ns = None
    try:
        guard()
        ready_read, ready_write = os.pipe2(os.O_CLOEXEC)
        opened.extend((ready_read, ready_write))
        release_read, release_write = os.pipe2(os.O_CLOEXEC)
        opened.extend((release_read, release_write))
        argv = command(ready_write, release_read)
        require(type(argv) is list and len(argv) > 1 and all(type(x) is str and x for x in argv)
                and argv[0].startswith('/'), 'Exact explicit absolute command builder')
        guard()
        child = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(ready_write, release_read),
            close_fds=True)
        # No wait/poll before pidfd_open: our unreaped child cannot have its PID reused.
        pidfd = os.pidfd_open(child.pid, 0)
        for fd in (ready_write, release_read):
            os.close(fd);opened.remove(fd)
        try:
            proof = parent_release(child.pid, ready_read, release_write,
                nonce=nonce, role=role, expected_argv=argv, expected_cgroup=expected_cgroup,
                guard=guard, deadline=time.monotonic()+handshake_seconds)
        finally:
            # parent_release closes both ends even when capture/guard fails.
            opened.remove(ready_read);opened.remove(release_write)
        collect(child, time.monotonic()+execution_seconds,
                {'stdout': stdout_cap, 'stderr': stderr_cap}, guard, buffers)
        actual_exit = child.wait(timeout=.1)
        exit_ns = time.monotonic_ns()
        require(actual_exit == 0, 'Gated child nonzero actual exit')
        guard()
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error)[:2048])
    finally:
        for fd in reversed(opened):
            try: os.close(fd)
            except OSError as error: cleanup_errors.append(repr(error))
        if child is not None:
            try:
                if child.poll() is None:
                    require(pidfd is not None, 'PID handle missing: outer owned-group cleanup required')
                    if signal_child(child, pidfd, signal.SIGTERM):
                        cleanup_signals.append('SIGTERM')
                    try: child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        if signal_child(child, pidfd, signal.SIGKILL):
                            cleanup_signals.append('SIGKILL')
                        child.wait(timeout=2)
                actual_exit = child.returncode
                exit_ns = exit_ns or time.monotonic_ns()
            except BaseException as error:
                cleanup_errors.append(repr(error))
            finally:
                for name in ('stdout', 'stderr'):
                    stream = getattr(child, name)
                    if stream is not None: stream.close()
        if pidfd is not None:
            os.close(pidfd)
    return dict(schema='nemotron4b_gated_child_supervision_v1', nonce=nonce, role=role,
        successful=(failure is None and not cleanup_errors and actual_exit == 0 and proof is not None),
        handshake=proof, output={name:bytes(value) for name,value in buffers.items()},
        output_may_be_partial=failure is not None,
        actual_exit_code=actual_exit, actual_exit_observed=actual_exit is not None,
        exit_ns=exit_ns, failure=failure, cleanup_signals=cleanup_signals,
        cleanup_errors=cleanup_errors, child_reaped=child is not None and child.returncode is not None,
        owned_group_drained_proven=False, runtime_admitted=False,
        full_model_qualified=False, new_speedup_claim=False)


if __name__ == '__main__':
    raise SystemExit('Source preparation only; requires a reviewed exact caller and complete root lifecycle')
