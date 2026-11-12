"""Pure injected-port mapping observer logic; no native/OS implementation.

This module cannot launch, attach, authenticate a native library or release a
lock by itself. A future separately reviewed adapter must supply every port,
the original full-auth worker/gate, native authority, and the original Store
with its extended narrow name whitelist and aggregate accounting. No current
specification, admission or runtime backend is supplied here.
"""
import copy
import json


LIMITS = dict(address_space_bytes=2147483648, cpu=6, cpu_seconds=600,
              hard_cpu_seconds=605, nice=19, output_file_bytes=2097152,
              wall_seconds=900)
CAPS = dict(raw_files=4392, raw_bytes=12288000, metadata_bytes=8*2**20,
            allocated_bytes=32*2**20, one_file_bytes=2*2**20)
GATE_CAP = 16*2**20
TRACE_CAP = 2*2**20
SUMMARY_CAP = 16*1024
RUNTIME_METADATA_CAP = 79008
TASK_CAP = 16                    # Cumulative, not merely simultaneous tasks.
CLEANUP_TASK_CAP = 32            # Kernel-discovered tasks only, never proc adoption.
EVENT_CAP = 2000000              # Non-memory syscall stops are counted, not logged.
STEP_CAP = 4000000
CLEANUP_EVENT_CAP = 8192
CLEANUP_STEP_CAP = 20000
CLEANUP_SECONDS = 10             # Each original TERM/KILL window.
WRITE_CHUNK = 65536
ARCH_X86_64 = 0xc000003e
MEMORY_CALLS = {9: 'mmap', 10: 'mprotect', 11: 'munmap', 12: 'brk', 25: 'mremap'}
NONRETURNING_CALLS = frozenset((60, 231))  # Linux x86_64 exit, exit_group.
RESTART_RETURNS = frozenset((-512, -513, -514, -516))
SIGTRAP, SIGKILL, SIGTERM = 5, 9, 15
SYSCALL_STOP = SIGTRAP | 0x80
GROUP_SIGNALS = frozenset((19, 20, 21, 22))
FORK_EVENTS = frozenset((1, 2, 3))
EVENT_EXEC, EVENT_EXIT, EVENT_STOP = 4, 6, 128
OPTIONS = ('TRACESYSGOOD', 'TRACEFORK', 'TRACEVFORK', 'TRACECLONE',
           'TRACEEXEC', 'TRACEEXIT', 'EXITKILL')
TRACE_NAME = 'phase-1-memory-syscalls.jsonl'
RECEIPT_NAME = 'phase-1-observer-exit.json'


class ObservationError(ValueError):
    pass


def require(value, reason):
    if not value:
        raise ObservationError(reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False)+'\n').encode('utf-8')


def wait_event(status):
    """Decode an actual Linux wait status; never substitute ECHILD or a poll."""
    require(type(status) is int and 0 <= status < 2**32, 'invalid_wait_status')
    if status & 0xff == 0x7f:
        sig, event = (status >> 8) & 0xff, status >> 16
        require(1 <= sig <= 64 or sig == SYSCALL_STOP, 'invalid_stop_signal')
        return dict(kind='stop', signal=sig, event=event)
    if status & 0x7f == 0:
        require(status & ~0xff00 == 0, 'invalid_exit_status')
        return dict(kind='terminal', code=(status >> 8) & 0xff)
    require(1 <= (status & 0x7f) <= 64 and status >> 8 == 0,
            'invalid_signal_terminal_status')
    return dict(kind='terminal', code=-(status & 0x7f))


def syscall_info(packet):
    """Parse bounded GET_SYSCALL_INFO bytes, not tracee memory/registers.

Future native ports must return exactly the number of bytes the kernel wrote.
No native binding, pointer layout implementation or runtime ABI claim is here.
"""
    require(type(packet) is dict and set(packet) == {'data', 'reported_length'},
            'syscall_info_packet')
    raw, length = packet['data'], packet['reported_length']
    require(type(raw) is bytes and type(length) is int and len(raw) == length
            and 24 <= length <= 88, 'syscall_info_whole_bounded_bytes')
    arch = int.from_bytes(raw[4:8], 'little')
    require(arch == ARCH_X86_64, 'unsupported_syscall_arch')
    result = dict(op=raw[0], arch=arch, raw_hex=raw.hex(), reported_length=length,
                  ip=int.from_bytes(raw[8:16], 'little'),
                  sp=int.from_bytes(raw[16:24], 'little'))
    if result['op'] == 1:
        require(length >= 80, 'short_syscall_entry')
        result.update(nr=int.from_bytes(raw[24:32], 'little'),
                      args=[int.from_bytes(raw[i:i+8], 'little') for i in range(32, 80, 8)])
    elif result['op'] == 2:
        require(length >= 33 and raw[32] in (0, 1), 'short_or_invalid_syscall_exit')
        result.update(rval=int.from_bytes(raw[24:32], 'little', signed=True),
                      is_error=bool(raw[32]))
        require(not result['is_error'] or -4095 <= result['rval'] < 0,
                'invalid_error_return')
    else:
        raise ObservationError('unsupported_syscall_info_operation')
    return result


def valid_owner(owner, pid):
    require(type(owner) is dict and owner.get('pid') == pid
            and type(owner.get('start_ticks')) is str and owner['start_ticks'].isdigit()
            and type(owner.get('ppid')) is int and type(owner.get('cgroup')) is str
            and type(owner.get('argv')) is list, 'bounded_task_identity')
    require(len(encoded(owner)) <= 2048, 'task_identity_budget')
    return copy.deepcopy(owner)


class ObservedSDK:
    """One once-only SDK observation, one wait owner, no implicit native port.

Ports: validate_authority(authority, sdk_owner), identity(tid), seize(pid, OPTIONS),
interrupt(tid), wait_known(tid), syscall_info(tid), event_message(tid),
resume(tid, 'syscall'|'cont', signal), write_gate(memoryview), close_gate(),
signal_owned(exact_owner, signal), now(), idle(seconds), owned_group_members().
wait_known is *only* waitpid(tid, __WALL|WNOHANG), with real errors preserved.

Store: validate_observer_budget(limits, caps, extra), write(name, whole_bytes).
The future Store adapter must count both added artifacts in all original totals;
receipt counts against the 79,008-byte runtime envelope, trace against metadata.
No fallback writer, process-wide wait, Popen poll/wait or lock-release port exists.
"""
    def __init__(self, child, owner, gate, authority, guard, store, ports):
        require(type(gate) is bytes and 0 < len(gate) <= GATE_CAP, 'whole_source_gate_budget')
        require(type(child.pid) is int and child.pid > 0 and child.returncode is None,
                'original_live_direct_child_handle')
        self.child, self.sdk = child, child.pid
        self.guard, self.store, self.ports = guard, store, ports
        self.authority, self.gate = copy.deepcopy(authority), gate
        self.tasks = {self.sdk: self._task(valid_owner(owner, self.sdk), None)}
        self.state = 'new'; self.failure = None; self.cleanup_errors = []
        self.trace = bytearray(); self.trace_records = 0; self.trace_dropped = 0
        self.offset = 0; self.gate_closed = False; self.gate_eof = False
        self.ready = False; self.seized = False; self.force_kill = False
        self.orphan_used = False; self.events = 0; self.cleanup_events = 0
        self.signalled = set(); self.ownership_complete = True; self.untracked_kernel_event = None
        self.steps = 0; self.cleanup_steps = 0; self.last_time = None
        self.started = None; self.terminal = None; self.outcome = None
        self.trace_saved = False; self.receipt_saved = False
        self.counts = dict(syscall_entry=0, syscall_exit=0, non_memory_entry=0,
                           non_memory_exit=0, signal_stops=0, fork_events=0,
                           terminal_events=0, attach_orphan_exit=0)

    @staticmethod
    def _task(owner, parent_event):
        return dict(owner=owner, parent_event=parent_event, state='new',
                    entry=None, syscall_seen=False, last_wait_status=None,
                    wait_events=0, terminal_status=None, exit_code=None)

    def _clock(self):
        value = self.ports.now()
        require(type(value) in (int, float) and value >= 0
                and (self.last_time is None or value >= self.last_time), 'monotonic_clock')
        self.last_time = value
        return value

    def _fail(self, reason):
        if self.failure is None:
            self.failure = str(reason)[:160]

    def _cleanup_error(self, reason):
        if len(self.cleanup_errors) < 8:
            self.cleanup_errors.append(str(reason)[:160])

    def _record(self, kind, *, cleanup=False, **fields):
        raw = encoded(dict(kind=kind, **fields))
        if len(self.trace)+len(raw) > TRACE_CAP:
            self.trace_dropped += 1
            self._fail('trace_budget_exceeded')
            if not cleanup:
                raise ObservationError('trace_budget_exceeded')
            return
        self.trace.extend(raw); self.trace_records += 1

    def _identity(self, tid):
        task = self.tasks[tid]; actual = self.ports.identity(tid)
        require(actual == task['owner'], 'known_task_identity_changed')

    def _resume(self, tid, signal=0, *, cleanup=False):
        self.ports.resume(tid, 'cont' if cleanup else 'syscall', signal)
        self.tasks[tid]['state'] = 'running'

    def _close_gate(self, *, whole=False):
        if not self.gate_closed:
            self.ports.close_gate(); self.gate_closed = True
        if whole:
            require(self.offset == len(self.gate), 'gate_eof_requires_every_byte')
            self.gate_eof = True

    def _pump_gate(self):
        if not self.ready or self.gate_closed or self.tasks[self.sdk]['state'] == 'terminal':
            return False
        view = memoryview(self.gate)[self.offset:self.offset+WRITE_CHUNK]
        try:
            count = self.ports.write_gate(view)
        except BlockingIOError:
            return False
        require(type(count) is int and 0 < count <= len(view), 'gate_short_write_contract')
        self.offset += count
        if self.offset == len(self.gate):
            self._close_gate(whole=True)
        return True

    def _syscall(self, tid, status):
        task = self.tasks[tid]; info = syscall_info(self.ports.syscall_info(tid))
        if info['op'] == 1:
            require(task['entry'] is None, 'entry_overwrites_missing_exit')
            task['syscall_seen'] = True
            task['entry'] = dict(info=info, wait_status=status)
            self.counts['syscall_entry'] += 1
            if info['nr'] in MEMORY_CALLS:
                self._record('memory_entry', tid=tid, wait_status=status, info=info,
                             syscall=MEMORY_CALLS[info['nr']])
            else:
                self.counts['non_memory_entry'] += 1
        else:
            self.counts['syscall_exit'] += 1
            entry = task['entry']
            if entry is None:
                require(tid == self.sdk and not task['syscall_seen']
                        and not self.orphan_used and not self.gate_eof,
                        'exit_without_entry')
                self.orphan_used = True; task['syscall_seen'] = True
                self.counts['attach_orphan_exit'] += 1
                self._record('attach_boundary_unpaired_exit', tid=tid,
                             wait_status=status, info=info, attributed=False)
            else:
                task['entry'] = None
                nr = entry['info']['nr']
                if nr in MEMORY_CALLS:
                    restarted = info['rval'] in RESTART_RETURNS and info['is_error']
                    self._record('memory_exit', tid=tid, wait_status=status,
                                 entry_wait_status=entry['wait_status'], info=info,
                                 entry_info=entry['info'], syscall=MEMORY_CALLS[nr],
                                 internal_restart=restarted,
                                 errno=(-info['rval'] if info['is_error'] and not restarted else None))
                else:
                    self.counts['non_memory_exit'] += 1
        self._resume(tid)

    def _new_task(self, parent_tid, status, event, *, cleanup=False):
        tid = self.ports.event_message(parent_tid)
        require(type(tid) is int and tid > 0 and tid not in self.tasks, 'unique_kernel_task_event')
        if len(self.tasks) >= CLEANUP_TASK_CAP:
            self.ownership_complete = False
            self.untracked_kernel_event = dict(tid=tid, parent_tid=parent_tid, wait_status=status)
            raise ObservationError('cleanup_task_budget_exceeded')
        # Insert the kernel-proven identity before further fallible metadata IO.
        task = self._task(None, dict(parent_tid=parent_tid, wait_status=status, event=event))
        self.tasks[tid] = task
        task['owner'] = valid_owner(self.ports.identity(tid), tid)
        require(task['owner']['cgroup'] == self.tasks[self.sdk]['owner']['cgroup'],
                'kernel_child_original_cgroup')
        self.counts['fork_events'] += 1
        self._record('kernel_task', tid=tid, parent_tid=parent_tid, wait_status=status,
                     event=event, owner=task['owner'], cleanup=cleanup)
        if len(self.tasks) > TASK_CAP:
            self.force_kill = True
            raise ObservationError('observation_task_budget_exceeded')

    def _consume(self, tid, status, *, cleanup=False):
        task = self.tasks[tid]
        require(task['state'] != 'terminal', 'event_after_terminal')
        event = wait_event(status)
        task['last_wait_status'] = status; task['wait_events'] += 1
        if event['kind'] == 'terminal':
            task.update(state='terminal', terminal_status=status, exit_code=event['code'])
            self.counts['terminal_events'] += 1
            # Reaping is irreversible: preserve terminal facts before any
            # fallible logging, serialization, identity lookup or Store call.
            if tid == self.sdk:
                self.child.returncode = event['code']
                self.terminal = dict(pid=tid, wait_status=status, returncode=event['code'],
                                     parent_observed_wait=True, self_exit_claimed=False)
                if not self.gate_eof:
                    self._fail('sdk_exited_before_complete_gate_eof')
            entry = task['entry']; task['entry'] = None
            if entry is not None:
                nr = entry['info']['nr']
                if nr in NONRETURNING_CALLS:
                    self._record('nonreturning_syscall_terminal', tid=tid, wait_status=status,
                                 entry_wait_status=entry['wait_status'], syscall_number=nr,
                                 paired_exit_invented=False, cleanup=cleanup)
                else:
                    self._fail('terminal_with_missing_syscall_exit')
            self._record('terminal', tid=tid, wait_status=status, returncode=event['code'], cleanup=cleanup)
            return
        task['state'] = 'stopped'
        sig, kind = event['signal'], event['event']
        if kind in FORK_EVENTS:
            require(sig == SIGTRAP, 'fork_event_signal')
            self._new_task(tid, status, kind, cleanup=cleanup)
            self._resume(tid, cleanup=cleanup)
        elif kind == EVENT_EXEC:
            raise ObservationError('unexpected_exec_event')
        elif kind == EVENT_EXIT:
            require(sig == SIGTRAP, 'exit_event_signal')
            self._record('preterminal_stop', tid=tid, wait_status=status, cleanup=cleanup)
            self._resume(tid, cleanup=cleanup)
        elif kind == EVENT_STOP:
            if sig in GROUP_SIGNALS:
                require(cleanup, 'unsupported_group_stop')
                self._resume(tid, cleanup=True)
            else:
                require(sig == SIGTRAP, 'unexpected_event_stop')
                initial = tid == self.sdk and not self.ready
                require(initial or task['parent_event'] is not None and task['wait_events'] == 1
                        or cleanup, 'unexpected_interrupt_stop')
                if initial:
                    self.ready = True
                self._record('initial_stop', tid=tid, wait_status=status, cleanup=cleanup)
                self._resume(tid, cleanup=cleanup)
        elif kind != 0:
            raise ObservationError('unknown_ptrace_event')
        elif sig == SYSCALL_STOP:
            if cleanup:
                self._resume(tid, cleanup=True)
            else:
                require(self.ready, 'syscall_before_initial_interrupt_stop')
                self._syscall(tid, status)
        else:
            # A plain SIGTRAP is a real delivery stop, not a syscall/event stop.
            require(1 <= sig <= 64, 'unknown_signal_stop')
            self.counts['signal_stops'] += 1
            self._record('signal_delivery', tid=tid, wait_status=status,
                         delivered_signal=sig, cleanup=cleanup)
            self._resume(tid, sig, cleanup=cleanup)

    def _sweep(self, *, cleanup=False):
        progress = False
        for tid in list(self.tasks):
            if self.tasks[tid]['state'] == 'terminal':
                continue
            try:
                require((self.cleanup_events < CLEANUP_EVENT_CAP) if cleanup else
                        (self.events < EVENT_CAP), 'cleanup_event_budget_exceeded' if cleanup
                        else 'observation_event_budget_exceeded')
                pair = self.ports.wait_known(tid)
                require(type(pair) is tuple and len(pair) == 2, 'actual_wait_pair')
                got, status = pair
                if type(got) is int and got not in (0, tid):
                    self.ownership_complete = False
                require(type(got) is int and (got == tid or got == 0 and status == 0),
                        'wait_returned_unknown_task')
                if got == 0:
                    continue
                progress = True
                if cleanup:
                    self.cleanup_events += 1
                else:
                    self.events += 1
                self._consume(tid, status, cleanup=cleanup)
            except BaseException as error:
                if not cleanup:
                    raise
                self._cleanup_error(type(error).__name__+':'+str(error))
                self.force_kill = True
        return progress

    def _all_terminal(self):
        return all(t['state'] == 'terminal' for t in self.tasks.values())

    def _signal_tasks(self, sig):
        for tid, task in list(self.tasks.items()):
            if task['state'] == 'terminal' or (tid, sig) in self.signalled:
                continue
            try:
                require(task['owner'] is not None, 'no_identity_for_signal')
                self._identity(tid)
                self.ports.signal_owned(copy.deepcopy(task['owner']), sig)
                self.signalled.add((tid, sig))
                if task['state'] == 'stopped':
                    self._resume(tid, cleanup=True)
            except BaseException as error:
                self._cleanup_error(type(error).__name__+':'+str(error))

    def _cleanup(self):
        self.state = 'cleanup'
        try:
            self._close_gate()
        except BaseException as error:
            self._cleanup_error(type(error).__name__+':'+str(error))
        for sig in (SIGTERM, SIGKILL):
            if sig == SIGTERM and self.force_kill:
                continue
            if self._all_terminal():
                break
            try:
                deadline = self._clock()+CLEANUP_SECONDS
                self._signal_tasks(sig)
                while not self._all_terminal() and self._clock() < deadline:
                    self.cleanup_steps += 1
                    require(self.cleanup_steps <= CLEANUP_STEP_CAP, 'cleanup_step_budget_exceeded')
                    self._sweep(cleanup=True)
                    # Covers newly discovered, kernel-owned tasks as well.
                    self._signal_tasks(sig)
                    if sig == SIGTERM and self.force_kill:
                        break
                    self.ports.idle(.01)
            except BaseException as error:
                self._cleanup_error(type(error).__name__+':'+str(error))
        # No guard or writer retry is required to continue actual cleanup.

    def _drained(self):
        if (not self._all_terminal() or self.terminal is None or not self.gate_closed
                or not self.ownership_complete):
            return False
        try:
            for task in self.tasks.values():
                if self.ports.identity(task['owner']['pid']) is not None:
                    return False
            return self.ports.owned_group_members() == []
        except BaseException as error:
            self._cleanup_error(type(error).__name__+':'+str(error))
            return False

    def _result(self, drained):
        return dict(schema='minicpm5_sdk_mapping_observer_logic_result_v1',
                    observation_complete=self.failure is None and drained and self.gate_eof,
                    trace_complete=self.failure is None, failure=self.failure,
                    physically_drained=drained, lock_release_permitted=drained,
                    sdk_terminal=copy.deepcopy(self.terminal),
                    gate_bytes_written=self.offset, gate_bytes=len(self.gate), gate_eof=self.gate_eof,
                    counters=copy.deepcopy(self.counts), wait_events=self.events,
                    cleanup_wait_events=self.cleanup_events, trace_bytes=len(self.trace),
                    trace_records=self.trace_records, trace_records_dropped=self.trace_dropped,
                    task_terminals={str(tid): {key: task[key] for key in
                        ('last_wait_status', 'wait_events', 'terminal_status', 'exit_code')}
                        for tid, task in self.tasks.items()},
                    cleanup_errors=list(self.cleanup_errors),
                    ownership_complete=self.ownership_complete,
                    untracked_kernel_event=copy.deepcopy(self.untracked_kernel_event),
                    trace_saved=self.trace_saved, exit_receipt_saved=self.receipt_saved,
                    model_executed=False, full_model_qualified=False, new_speedup_claim=False)

    def run(self):
        require(self.state == 'new', 'once_only_observer')
        self.state = 'observing'
        try:
            self.started = self._clock()
            self.guard()
            self.ports.validate_authority(copy.deepcopy(self.authority),
                                          copy.deepcopy(self.tasks[self.sdk]['owner']))
            self.store.validate_observer_budget(copy.deepcopy(LIMITS), copy.deepcopy(CAPS),
                dict(trace_bytes=TRACE_CAP, summary_bytes=SUMMARY_CAP,
                     runtime_metadata_bytes=RUNTIME_METADATA_CAP,
                     trace_name=TRACE_NAME, receipt_name=RECEIPT_NAME))
            self._identity(self.sdk)
            self.ports.seize(self.sdk, OPTIONS); self.seized = True
            self.ports.interrupt(self.sdk)
            while not self._all_terminal():
                self.guard()
                self.steps += 1
                require(self.steps <= STEP_CAP, 'observation_step_budget_exceeded')
                require(self._clock()-self.started < LIMITS['wall_seconds'], 'original_wall_exceeded')
                progress = self._sweep()
                if self.failure is not None:
                    raise ObservationError(self.failure)
                progress = self._pump_gate() or progress
                if not progress:
                    self.ports.idle(.01)
            require(self.gate_eof, 'incomplete_gate_eof')
        except BaseException as error:
            self._fail(type(error).__name__+':'+str(error))
            self._cleanup()
        drained = self._drained()
        if not drained:
            self._fail('physical_drain_not_proven')
        self.state = 'drained' if drained else 'cleanup_pending'
        if drained:
            try:
                # This is the original guarded writer contract, exactly once.
                self.guard()
                self.store.write(TRACE_NAME, bytes(self.trace)); self.trace_saved = True
                receipt = self._result(drained)
                receipt.pop('exit_receipt_saved')
                receipt['self_persistence_claimed'] = False
                raw = encoded(receipt)
                require(len(raw) <= SUMMARY_CAP, 'observer_receipt_16k_budget')
                self.store.write(RECEIPT_NAME, raw); self.receipt_saved = True
            except BaseException as error:
                self._fail('persistence_failed:'+type(error).__name__)
        self.outcome = self._result(drained)
        return copy.deepcopy(self.outcome)
