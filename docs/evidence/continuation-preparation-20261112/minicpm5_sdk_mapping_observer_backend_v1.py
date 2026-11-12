"""Lazy future native ports and narrow original-Store adapter. SOURCE ONLY.

No top-level ctypes import, OS inspection, file read, process action or admission.
Caller must already own original full-auth outer/worker/gate and provide the
separately reviewed native authenticator. No current authority is synthesized.
The unchanged pure logic owns all wait handling once the ports are handed over.
"""
import copy
from research import minicpm5_sdk_mapping_observer_logic_v1 as logic

require = logic.require
CTYPES_ROOT = '/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/ctypes'
OUTER_EXE = '/home/ubuntu/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12'
LIBC = '/usr/lib/x86_64-linux-gnu/libc.so.6'
PTRACE_CONT, PTRACE_SYSCALL = 7, 24
PTRACE_GETEVENTMSG, PTRACE_SEIZE = 0x4201, 0x4206
PTRACE_INTERRUPT, PTRACE_GET_SYSCALL_INFO = 0x4207, 0x420e
OPTIONS_VALUE = 1 | 2 | 4 | 8 | 16 | 64 | (1 << 20)
WAIT_FLAGS = 0x40000000 | 1  # Linux __WALL | WNOHANG, always exact requested tid.
ORIGINAL_NAMES = frozenset(('admission.json', 'phase-1-owner.json',
                            'phase-1-summary.json', 'phase-1-exit.json'))
OBSERVER_NAMES = frozenset((logic.TRACE_NAME, logic.RECEIPT_NAME))


def _load_ctypes():
    import ctypes  # Only reached from make_ports after external authentication.
    return ctypes


def _load_os():
    import os
    return os


def _load_time():
    import time
    return time


def validate_authority_receipt(receipt, authority):
    """Shape/binding check only; external authenticator must verify real bytes.

This does not convert arbitrary caller booleans into authentication. The callback
itself must be reviewed and source-pinned by the future original full-auth outer.
File provenance, hashes, current identities and complete dependency checks belong
to that callback, using existing source and native-byte authentication primitives.
"""
    require(type(receipt) is dict and receipt.get('schema') == 'mini_mapping_observer_native_authority_v1'
            and receipt.get('authority') == authority, 'explicit_native_authority_binding')
    for flag in ('current_bytes_authenticated', 'current_identities_authenticated',
                 'pure_sources_authenticated', 'dependency_closure_verified',
                 'original_nine_native_checks_preserved'):
        require(receipt.get(flag) is True, 'missing_native_authority_'+flag)
    require(receipt.get('unresolved') == [] and receipt.get('outer_executable') == OUTER_EXE
            and receipt.get('ctypes_root') == CTYPES_ROOT and receipt.get('libc_path') == LIBC
            and receipt.get('abi') == 'linux-x86_64-lp64', 'exact_observer_runtime_authority')
    require(type(receipt.get('evidence')) is dict and receipt['evidence'], 'native_authority_evidence_required')


def make_ports(child, owner, authority, guard, *, authenticate, verify_loaded_native,
               inspect_owner, owned_group_members, ctypes_loader=_load_ctypes,
               os_loader=_load_os, time_loader=_load_time):
    """Future explicit entry, never called by preparation with real loaders.

authenticate(authority, guard) must complete before ctypes (including ctypes'
own PyDLL(None)) loads. verify_loaded_native must revalidate actual imported
module origins/provider identities after import/CDLL and before any ptrace call.
No CDLL(None), find_library, external command or native probing fallback exists.
Failures before return occur before SEIZE; the caller still owns ordinary cleanup.
"""
    require(authority is not None and callable(authenticate) and callable(verify_loaded_native),
            'explicit_native_authority_and_verifiers_required')
    guard(); receipt = authenticate(copy.deepcopy(authority), guard)
    validate_authority_receipt(receipt, authority)
    guard(); ctypes = ctypes_loader()
    require(ctypes.__file__ == CTYPES_ROOT+'/__init__.py', 'exact_ctypes_origin')
    require(ctypes.sizeof(ctypes.c_int) == 4 and ctypes.sizeof(ctypes.c_long) == 8
            and ctypes.sizeof(ctypes.c_ulong) == 8 and ctypes.sizeof(ctypes.c_void_p) == 8,
            'exact_linux_x86_64_lp64_ffi_sizes')
    libc = ctypes.CDLL(receipt['libc_path'], use_errno=True)
    require(libc._name == LIBC, 'exact_libc_handle_request')
    require(verify_loaded_native(ctypes, libc, copy.deepcopy(receipt), guard) is True,
            'loaded_native_provider_verification_required')
    guard()
    ptrace = libc.ptrace
    ptrace.restype = ctypes.c_long
    ptrace.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    os_api, time_api = os_loader(), time_loader()
    require(child.pid == owner['pid'] and child.returncode is None
            and inspect_owner(child.pid) == owner, 'exact_original_direct_child_before_handoff')
    require(child.stdin is not None and not child.stdin.closed, 'original_open_stdin_gate')
    require(child._waitpid_lock.acquire(False), 'exclusive_popen_waitpid_handoff')
    try:
        fd = child.stdin.fileno()
        os_api.set_blocking(fd, False)
        return NativePorts(child, owner, authority, receipt, ctypes, libc, ptrace,
                           os_api, time_api, inspect_owner, owned_group_members, fd)
    except BaseException:
        # No SEIZE has occurred; restore ordinary owner cleanup on factory failure.
        child._waitpid_lock.release()
        raise


class NativePorts:
    """Known tasks only. One retained Popen handle and exclusive waitpid lock.

Never call Popen.poll/wait/send_signal/communicate or original polling cleanup
while this object owns the wait lock. cleanup_pending keeps this object and the
outer's original serial descriptor alive; it is not permission to drop either.
"""
    def __init__(self, child, owner, authority, receipt, ctypes, libc, ptrace,
                 os_api, time_api, inspect_owner, group_members, gate_fd):
        self.child, self.owner = child, copy.deepcopy(owner)
        self.authority, self.receipt = copy.deepcopy(authority), copy.deepcopy(receipt)
        self.ctypes, self.libc, self.ptrace = ctypes, libc, ptrace
        self.os, self.time = os_api, time_api
        self.inspect_owner, self.group_members = inspect_owner, group_members
        self.gate_fd = gate_fd; self.gate_closed = False
        self.known = {owner['pid']: copy.deepcopy(owner)}
        self.waited_terminals = {}
        self.wait_lock_owned = True; self.seized = False

    def validate_authority(self, authority, sdk_owner):
        validate_authority_receipt(self.receipt, authority)
        require(authority == self.authority and sdk_owner == self.owner, 'same_factory_native_authority')

    def _known(self, tid):
        require(self.wait_lock_owned and type(tid) is int and tid in self.known,
                'only_known_owned_task_operations')

    def _call(self, request, tid, address=0, data=0):
        self._known(tid)
        require(request in (PTRACE_CONT, PTRACE_SYSCALL, PTRACE_GETEVENTMSG,
                            PTRACE_SEIZE, PTRACE_INTERRUPT, PTRACE_GET_SYSCALL_INFO),
                'read_only_observer_ptrace_requests')
        self.ctypes.set_errno(0)
        result = self.ptrace(self.ctypes.c_int(request), self.ctypes.c_int(tid),
                             self.ctypes.c_void_p(address),
                             self.ctypes.c_void_p(data) if type(data) is int else data)
        if result == -1:
            actual_errno = self.ctypes.get_errno()
            raise OSError(actual_errno, 'ptrace request failed', request)
        require(type(result) is int, 'actual_ptrace_integer_result')
        return result

    def seize(self, tid, options):
        require(tid == self.owner['pid'] and not self.seized and tuple(options) == logic.OPTIONS,
                'once_only_direct_sdk_seize')
        require(self.inspect_owner(tid) == self.owner, 'sdk_owner_before_seize')
        require(self._call(PTRACE_SEIZE, tid, 0, OPTIONS_VALUE) == 0, 'seize_zero_result')
        self.seized = True

    def interrupt(self, tid):
        require(self.seized and self._call(PTRACE_INTERRUPT, tid) == 0, 'interrupt_actual_result')

    def wait_known(self, tid):
        self._known(tid)
        # Preserve EINTR/ECHILD and every raw status. No wait(-1), no defaults.
        pair = self.os.waitpid(tid, WAIT_FLAGS)
        if (type(pair) is tuple and len(pair) == 2 and pair[0] == tid
                and logic.wait_event(pair[1])['kind'] == 'terminal'):
            self.waited_terminals[tid] = pair[1]
        return pair

    def syscall_info(self, tid):
        buffer = self.ctypes.create_string_buffer(88)
        length = self._call(PTRACE_GET_SYSCALL_INFO, tid, 88, self.ctypes.byref(buffer))
        # Kernel returns the available information size, not necessarily bytes
        # copied when the caller's buffer is short: reject any value above 88.
        require(24 <= length <= 88, 'whole_kernel_syscall_info_fits_buffer')
        return dict(data=buffer.raw[:length], reported_length=length)

    def event_message(self, tid):
        value = self.ctypes.c_ulong(0)
        require(self._call(PTRACE_GETEVENTMSG, tid, 0, self.ctypes.byref(value)) == 0,
                'geteventmsg_actual_result')
        born = value.value
        require(type(born) is int and born > 0 and born not in self.known, 'new_kernel_event_tid')
        # The pure logic applies the normal/cleanup task caps. One extra event
        # beyond its hard cap remains explicit there and prohibits lock release.
        require(len(self.known) <= logic.CLEANUP_TASK_CAP, 'backend_kernel_task_bound')
        self.known[born] = None
        return born

    def identity(self, tid):
        self._known(tid)
        actual = self.inspect_owner(tid)
        if self.known[tid] is None and actual is not None:
            self.known[tid] = logic.valid_owner(actual, tid)
        return copy.deepcopy(actual)

    def resume(self, tid, mode, signal):
        require(mode in ('syscall', 'cont') and type(signal) is int and 0 <= signal <= 64,
                'exact_resume_and_signal')
        request = PTRACE_SYSCALL if mode == 'syscall' else PTRACE_CONT
        require(self._call(request, tid, 0, signal) == 0, 'resume_actual_result')

    def write_gate(self, view):
        require(self.wait_lock_owned and not self.gate_closed, 'live_gate_owner')
        return self.os.write(self.gate_fd, view)

    def close_gate(self):
        if not self.gate_closed:
            # No buffered writes are ever used by these ports; close sends EOF.
            self.child.stdin.close(); self.gate_closed = True

    def signal_owned(self, owner, signal):
        tid = owner['pid']; self._known(tid)
        require(signal in (logic.SIGTERM, logic.SIGKILL)
                and self.known[tid] == owner and self.inspect_owner(tid) == owner,
                'exact_known_owner_before_cleanup_signal')
        self.os.kill(tid, signal)

    def now(self):
        return self.time.monotonic()

    def idle(self, seconds):
        require(type(seconds) in (int, float) and 0 <= seconds <= .01, 'bounded_observer_idle')
        self.time.sleep(seconds)

    def owned_group_members(self):
        return self.group_members(self.owner['pid'])

    def finish_wait_handoff(self, result):
        """Explicit outer call only after actual logic result; no serial release."""
        terminal = result.get('sdk_terminal')
        require(self.wait_lock_owned and result.get('physically_drained') is True
                and type(terminal) is dict and terminal.get('parent_observed_wait') is True
                and terminal.get('pid') == self.child.pid
                and logic.wait_event(terminal['wait_status']) ==
                    dict(kind='terminal', code=self.child.returncode)
                and terminal.get('returncode') == self.child.returncode
                and self.waited_terminals.get(self.child.pid) == terminal.get('wait_status')
                and self.gate_closed, 'actual_terminal_and_drain_before_wait_handoff')
        require(set(self.waited_terminals) == set(self.known)
                and all(self.inspect_owner(tid) is None for tid in self.known)
                and self.owned_group_members() == [], 'actual_all_known_gone_before_handoff')
        self.child._waitpid_lock.release(); self.wait_lock_owned = False


def make_store_class(original):
    """Original Store IO/accounting, six names, one extra runtime receipt.

Pure class construction. Real inventory/write actions occur only when called
later by the admitted outer. The original module and frozen sources are untouched.
"""
    require(original.RUNTIME_METADATA_CAP == logic.RUNTIME_METADATA_CAP
            and original.serial.CAPS == logic.CAPS, 'unchanged_original_store_caps')
    runtime = frozenset(original.RUNTIME) | {logic.RECEIPT_NAME}

    class ObserverStore(original.Store):
        @staticmethod
        def legal(name):
            return name in ORIGINAL_NAMES | OBSERVER_NAMES

        def _runtime_bytes(self):
            return sum(row[2] for name, row in self.rows.items()
                       if name in runtime and not name.endswith('-summary.json'))

        def inventory(self):
            result = super().inventory()
            require(self._runtime_bytes() <= logic.RUNTIME_METADATA_CAP,
                    'observer_receipt_in_original_runtime_envelope')
            return result

        def write(self, name, raw):
            require(self.legal(name) and type(raw) is bytes, 'six_exact_artifact_names')
            self.inventory()  # Reserve against actual shared totals before write.
            if name in runtime and not name.endswith('-summary.json'):
                require(self._runtime_bytes()+len(raw) <= logic.RUNTIME_METADATA_CAP,
                        'joint_runtime_metadata_reservation')
            if name in ('phase-1-summary.json', logic.RECEIPT_NAME):
                require(len(raw) <= logic.SUMMARY_CAP, 'whole_summary_receipt_16k')
            return super().write(name, raw)

        def validate_observer_budget(self, limits, caps, extra):
            require(limits == logic.LIMITS and caps == logic.CAPS
                    and extra == dict(trace_bytes=logic.TRACE_CAP, summary_bytes=logic.SUMMARY_CAP,
                        runtime_metadata_bytes=logic.RUNTIME_METADATA_CAP,
                        trace_name=logic.TRACE_NAME, receipt_name=logic.RECEIPT_NAME),
                    'exact_original_observer_budget')
            self.inventory()
            # Both future writes fit before observation; original Store retains
            # exclusive creation, actual inode/allocation checks and sticky guard.
            require(self.logical_metadata+logic.TRACE_CAP+logic.SUMMARY_CAP <= caps['metadata_bytes']
                    and self._runtime_bytes()+logic.SUMMARY_CAP <= logic.RUNTIME_METADATA_CAP
                    and self.allocated+logic.TRACE_CAP+logic.SUMMARY_CAP+2*65536 <= caps['allocated_bytes'],
                    'whole_trace_and_receipt_reserved_in_original_totals')
    return ObserverStore
