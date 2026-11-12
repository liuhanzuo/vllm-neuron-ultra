"""Exact frozen-literal wire budget; complete observer owners stay in memory.

All wait, identity matching, cleanup, trace and resource behavior is inherited
unchanged. Only the two original methods that invoke valid_owner are copied to
bind the explicit successor helper. No IO, native import or runtime AST.
"""
import copy
from research import minicpm5_sdk_mapping_observer_logic_v1 as previous
from research import minicpm5_formal_literal_codec_v1 as codec

require, encoded, ObservationError = previous.require, previous.encoded, previous.ObservationError
LIMITS, CAPS = previous.LIMITS, previous.CAPS
GATE_CAP, TASK_CAP, CLEANUP_TASK_CAP = previous.GATE_CAP, previous.TASK_CAP, previous.CLEANUP_TASK_CAP


def valid_owner(owner, pid):
    require(type(owner) is dict and owner.get('pid') == pid
            and type(owner.get('start_ticks')) is str and owner['start_ticks'].isdigit()
            and type(owner.get('ppid')) is int and type(owner.get('cgroup')) is str
            and type(owner.get('argv')) is list, 'bounded_task_identity')
    if len(encoded(owner)) <= 2048:
        return previous.valid_owner(owner, pid)
    require(len(owner['argv']) == 15 and type(owner['argv'][7]) is str,
            'exact_full_literal_task_identity')
    try:
        wire = codec.encode('owner', owner, owner['argv'][7])
    except ValueError as error:
        raise ObservationError('task_identity_literal_codec') from error
    require(len(wire) <= 2048, 'task_identity_budget')
    return copy.deepcopy(owner)


class ObservedSDK(previous.ObservedSDK):
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

