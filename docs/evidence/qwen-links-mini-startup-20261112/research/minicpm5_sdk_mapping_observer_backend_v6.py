"""Original observer ports and Store semantics with three explicit wire encodings.

Native factory/wait operations retain the reviewed v4/v2 bodies. Persisted
admission, owner and exit and full owner validation use the pinned v7 codec;
full in-memory values do not change.
"""
import copy
from research import minicpm5_sdk_mapping_observer_backend_v2 as previous
from research import minicpm5_formal_literal_codec_v2 as codec
from research import minicpm5_formal_literal_result_v2 as literal_result
from research import minicpm5_formal_literal_budget_v2 as prepared
from research import minicpm5_sdk_mapping_observer_logic_v3 as owner_logic

ORIGINAL_NAMES = previous.ORIGINAL_NAMES
OBSERVER_NAMES = previous.OBSERVER_NAMES
logic = previous.logic
require = previous.require


def make_store_class(original):
    base = previous.make_store_class(original)

    class LiteralObserverStore(base):
        KINDS = {'admission.json': 'admission', 'phase-1-owner.json': 'owner', 'phase-1-exit.json': 'exit'}

        def __init__(self, root, guard, *, literal):
            super().__init__(root, guard)
            codec.literal_bytes(literal, guard)
            self.literal = literal
            self.runtime_reservation = self.reserved_admission = self.reserved_argv = None

        def prospective_admission(self, value):
            raw = codec.encode('admission', value, self.literal, self.guard)
            return raw, dict(path=str(self.root / 'admission.json'), sha256=codec.digest(raw, self.guard))

        def reserve_runtime(self, value, reference, argv):
            require(self.runtime_reservation is None, 'One complete pre-spawn reservation')
            inventory = self.inventory()
            require(inventory['names'] == [], 'Reserve all original runtime artifacts before admission write')
            raw, expected = self.prospective_admission(value)
            require(reference == expected, 'Reservation binds actual proposed wire reference')
            budget = prepared.combined_budget(raw, argv, self.literal, str(self.root))
            require(self.logical_metadata + budget['total_runtime_bound'] + logic.TRACE_CAP + logic.SUMMARY_CAP
                    <= original.serial.CAPS['metadata_bytes'], 'Original complete metadata reservation')
            # Preserve the original conservative 64KiB per-file allocation margin.
            require(self.allocated + budget['total_runtime_bound'] + logic.TRACE_CAP + logic.SUMMARY_CAP
                    + 6 * 65536 <= original.serial.CAPS['allocated_bytes'], 'Original complete allocation reservation')
            self.runtime_reservation = budget
            self.reserved_admission = dict(reference)
            self.reserved_argv = list(argv)
            return dict(budget)

        def before_spawn(self, admission, reference, argv):
            require(self.runtime_reservation is not None and self.reserved_admission == reference
                    and self.reserved_argv == argv, 'Complete actual reservation required before SDK Popen')
            raw, expected = self.prospective_admission(admission)
            require(expected == reference, 'In-memory admission unchanged since reservation')
            persisted = self.reference('admission.json')
            require({key: persisted[key] for key in ('path', 'sha256')} == reference
                    and self.read(persisted) == raw, 'Actual durable wire admission matches complete reservation')
            self.guard()

        def write(self, name, raw):
            if name in self.KINDS:
                require(self.runtime_reservation is not None, 'No boundary artifact before whole reservation')
                raw = codec.encode(self.KINDS[name], codec.parsed(raw), self.literal, self.guard)
                key = {'admission.json': 'admission_wire_bytes', 'phase-1-owner.json': 'owner_wire_bound',
                       'phase-1-exit.json': 'exit_wire_bound'}[name]
                require(len(raw) <= self.runtime_reservation[key], 'Actual whole boundary bytes exceed reservation')
            return super().write(name, raw)

        def load(self, name):
            reference = self.reference(name)
            raw = self.read(reference)  # Original complete readback/hash/identity/guard.
            if name in self.KINDS:
                value = codec.decode(self.KINDS[name], raw, self.literal,
                                     wire_sha256=reference['sha256'], check=self.guard)
            else:
                value = codec.parsed(raw)
            return value, reference

    return LiteralObserverStore


# The native factory is the unchanged v4 body; only identity uses the v7 codec.
original = previous
CTYPES_ROOT, LIBC = original.CTYPES_ROOT, original.LIBC
_load_ctypes, _load_os = original._load_ctypes, original._load_os
_load_time, _load_signal = original._load_time, original._load_signal
validate_authority_receipt = original.validate_authority_receipt


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
            self.known[tid] = owner_logic.valid_owner(actual, tid)
        return copy.deepcopy(actual)
