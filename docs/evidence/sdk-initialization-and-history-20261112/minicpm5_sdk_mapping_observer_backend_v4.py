"""Original native factory with explicit complete-owner budgeting successor.

Native operations, authority and wait ownership remain original. This factory
selects the narrow NativePorts identity override; Store uses the frozen lossless
literal boundary implementation. No factory or OS operation runs on import.
"""
import copy
from research import minicpm5_sdk_mapping_observer_backend_v2 as original
from research import minicpm5_sdk_mapping_observer_backend_v3 as literal_backend
from research import minicpm5_sdk_mapping_observer_logic_v2 as logic

require = original.require
CTYPES_ROOT, LIBC = original.CTYPES_ROOT, original.LIBC
_load_ctypes, _load_os = original._load_ctypes, original._load_os
_load_time, _load_signal = original._load_time, original._load_signal
validate_authority_receipt = original.validate_authority_receipt
ORIGINAL_NAMES, OBSERVER_NAMES = original.ORIGINAL_NAMES, original.OBSERVER_NAMES
make_store_class = literal_backend.make_store_class


def make_ports(child, owner, authority, guard, *, authenticate, verify_loaded_native,
               inspect_owner, owned_group_members, ctypes_loader=_load_ctypes,
               os_loader=_load_os, time_loader=_load_time, signal_loader=_load_signal):
    """Future explicit entry, never called by preparation with real loaders.

authenticate(authority, guard) must complete before ctypes (including ctypes'
own PyDLL(None)) loads. verify_loaded_native must revalidate actual imported
module origins/provider identities after import/CDLL and before any ptrace call.
No CDLL(None), find_library, external command or native probing fallback exists.
This factory never acquires the Popen wait lock. Before any ownership transfer,
the caller must persistently bind the returned object and then explicitly call
begin_wait_handoff(). All factory failure/return/assignment windows therefore
remain before handoff and SEIZE, with ordinary original cleanup still available.
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
    fd = child.stdin.fileno()
    os_api.set_blocking(fd, False)
    return NativePorts(child, owner, authority, receipt, ctypes, libc, ptrace,
                       os_api, time_api, inspect_owner, owned_group_members, fd,
                       signal_loader())


class NativePorts(original.NativePorts):
    def identity(self, tid):
        self._known(tid)
        actual = self.inspect_owner(tid)
        if self.known[tid] is None and actual is not None:
            self.known[tid] = logic.valid_owner(actual, tid)
        return copy.deepcopy(actual)
