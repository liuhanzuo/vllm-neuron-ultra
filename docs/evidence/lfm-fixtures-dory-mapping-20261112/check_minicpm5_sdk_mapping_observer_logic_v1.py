"""Adversarial schedules for the pure observer; every OS/native port is mocked.

No experimental modules, framework imports, process inspection, locks, native
libraries or payload files are accessed. Output files exist only in FakeStore.
"""
import copy
import ast
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import minicpm5_sdk_mapping_observer_logic_v1 as m


PID, CHILD = 101, 102
OWNER = dict(pid=PID, start_ticks='200', ppid=99,
             argv=['/sdk/python', '-B', 'original-worker'], cgroup='original')
CHILD_OWNER = dict(pid=CHILD, start_ticks='201', ppid=PID,
                   argv=OWNER['argv'][:], cgroup='original')


def stop(sig=5, event=0):
    return (event << 16) | (sig << 8) | 0x7f


def info(op, nr=9, args=(0, 1048576, 3, 34, 2**64-1, 0), rval=0,
         error=False, arch=0xc000003e):
    raw = bytearray(80 if op == 1 else 33)
    raw[0] = op; raw[4:8] = arch.to_bytes(4, 'little')
    raw[8:16] = (123456).to_bytes(8, 'little')
    raw[16:24] = (987654).to_bytes(8, 'little')
    if op == 1:
        raw[24:32] = nr.to_bytes(8, 'little')
        for i, value in enumerate(args):
            raw[32+8*i:40+8*i] = value.to_bytes(8, 'little')
    else:
        raw[24:32] = rval.to_bytes(8, 'little', signed=True); raw[32] = int(error)
    return dict(data=bytes(raw), reported_length=len(raw))


def entry(nr=9, **kw):
    return dict(status=stop(133), info=info(1, nr=nr, **kw))


def leave(rval=0, error=False):
    return dict(status=stop(133), info=info(2, rval=rval, error=error))


INITIAL = dict(status=stop(5, 128))


def normal(code=1):
    return [INITIAL, entry(), leave(-12, True), entry(231), dict(status=stop(5, 6)),
            dict(status=code << 8)]


class FakeChild:
    pid = PID
    returncode = None
    def poll(self):
        raise AssertionError('Popen poll is forbidden')
    def wait(self, *args, **kwargs):
        raise AssertionError('Popen wait is forbidden')


class FakeStore:
    def __init__(self):
        self.files = {}; self.attempts = []; self.fail_name = None; self.budget_seen = None
    def validate_observer_budget(self, limits, caps, extra):
        self.budget_seen = (limits, caps, extra)
    def write(self, name, raw):
        self.attempts.append(name)
        if name == self.fail_name:
            raise ValueError('original sticky Store rejects write')
        assert name not in self.files
        self.files[name] = raw


class FakePorts:
    def __init__(self, events=None):
        self.owners = {PID: copy.deepcopy(OWNER), CHILD: copy.deepcopy(CHILD_OWNER)}
        self.queues = {PID: deque(copy.deepcopy(normal() if events is None else events))}
        self.last = {}; self.dead = set(); self.log = []; self.written = bytearray()
        self.closed = False; self.clock = 0.; self.write_schedule = deque()
        self.seize_error = None; self.authority_error = None; self.term_ignored = False
        self.kill_ignored = False; self.extra_group = []; self.on_wait = None
    def validate_authority(self, authority, owner):
        self.log.append(('authority', copy.deepcopy(authority), copy.deepcopy(owner)))
        if self.authority_error: raise self.authority_error
        assert owner['ppid'] == 99
    def identity(self, tid):
        return None if tid in self.dead else copy.deepcopy(self.owners.get(tid))
    def seize(self, tid, options):
        self.log.append(('seize', tid, options))
        if self.seize_error: raise self.seize_error
    def interrupt(self, tid): self.log.append(('interrupt', tid))
    def wait_known(self, tid):
        self.log.append(('wait', tid))
        if self.on_wait: self.on_wait(self, tid)
        queue = self.queues.setdefault(tid, deque())
        if not queue: return (0, 0)
        item = queue.popleft()
        if isinstance(item, BaseException): raise item
        if item is None: return (0, 0)
        self.last[tid] = item
        if 'returned_pid' in item: return item['returned_pid'], item['status']
        if m.wait_event(item['status'])['kind'] == 'terminal': self.dead.add(tid)
        return tid, item['status']
    def syscall_info(self, tid): return copy.deepcopy(self.last[tid]['info'])
    def event_message(self, tid): return self.last[tid]['new_tid']
    def resume(self, tid, mode, signal): self.log.append(('resume', tid, mode, signal))
    def write_gate(self, view):
        assert not self.closed
        count = self.write_schedule.popleft() if self.write_schedule else len(view)
        if isinstance(count, BaseException): raise count
        if 0 < count <= len(view): self.written.extend(view[:count])
        self.log.append(('write', count)); return count
    def close_gate(self): self.closed = True; self.log.append(('close',))
    def signal_owned(self, owner, sig):
        tid = owner['pid']; assert owner == self.owners[tid]
        self.log.append(('signal', tid, sig))
        if sig == 15 and self.term_ignored or sig == 9 and self.kill_ignored: return
        self.queues[tid] = deque([dict(status=sig)])
    def now(self): self.clock += .1; return self.clock
    def idle(self, seconds): self.clock += seconds
    def owned_group_members(self): return copy.deepcopy(self.extra_group)


def fixture(events=None, gate=b'whole original source gate', guard=None):
    ports = FakePorts(events); store = FakeStore(); child = FakeChild()
    engine = m.ObservedSDK(child, OWNER, gate, {'mock_authority': 'future input'},
                          guard or (lambda: None), store, ports)
    return engine, ports, store, child


def records(store):
    return [json.loads(line) for line in store.files.get(m.TRACE_NAME, b'').splitlines()]


class ProtocolTests(unittest.TestCase):
    def test_actual_nonzero_sdk_exit_is_complete_observation(self):
        e,p,s,c = fixture(); result=e.run()
        self.assertTrue(result['observation_complete']); self.assertEqual(c.returncode,1)
        self.assertEqual(result['sdk_terminal']['wait_status'],256)
        self.assertEqual(bytes(p.written),e.gate); self.assertTrue(p.closed)
        self.assertTrue(result['exit_receipt_saved']); self.assertEqual(len(s.files),2)
        self.assertFalse(json.loads(s.files[m.RECEIPT_NAME])['self_persistence_claimed'])
    def test_mapping_errno_comes_from_actual_info(self):
        e,p,s,c=fixture(); e.run()
        row=next(x for x in records(s) if x['kind']=='memory_exit')
        self.assertEqual(row['errno'],12); self.assertEqual(row['info']['rval'],-12)
        self.assertEqual(row['entry_wait_status'],stop(133)); self.assertEqual(row['wait_status'],stop(133))
        self.assertEqual(bytes.fromhex(row['info']['raw_hex']),info(2,rval=-12,error=True)['data'])
    def test_brk_old_break_is_not_invented_enomem(self):
        e,p,s,c=fixture([INITIAL,entry(12),leave(0x10000),dict(status=0)]); e.run()
        row=next(x for x in records(s) if x['kind']=='memory_exit')
        self.assertIsNone(row['errno']); self.assertEqual(row['info']['rval'],0x10000)
    def test_internal_restart_not_reported_as_errno(self):
        e,p,s,c=fixture([INITIAL,entry(),leave(-512,True),dict(status=0)]); e.run()
        row=next(x for x in records(s) if x['kind']=='memory_exit')
        self.assertTrue(row['internal_restart']); self.assertIsNone(row['errno'])
    def test_non_target_stops_counted_without_individual_trace(self):
        events=[INITIAL]+[item for _ in range(30) for item in (entry(0),leave(5))]+[dict(status=0)]
        e,p,s,c=fixture(events); result=e.run()
        self.assertTrue(result['observation_complete'])
        self.assertEqual(result['counters']['non_memory_entry'],30)
        self.assertEqual(result['counters']['non_memory_exit'],30)
        self.assertEqual([r['kind'] for r in records(s)],['initial_stop','terminal'])
    def test_short_writes_and_eagain_never_truncate_gate(self):
        gate=b'x'*150000
        events=[INITIAL]+[None]*10+normal()[1:]
        e,p,s,c=fixture(events,gate); p.write_schedule=deque([BlockingIOError(),3,100,65536,100,65536])
        result=e.run(); self.assertTrue(result['observation_complete'])
        self.assertEqual(bytes(p.written),gate); self.assertEqual(result['gate_bytes_written'],len(gate))
    def test_sdk_exit_before_full_gate_is_incomplete(self):
        e,p,s,c=fixture([INITIAL,dict(status=7<<8)],b'x'*150000)
        result=e.run(); self.assertFalse(result['observation_complete'])
        self.assertFalse(result['gate_eof']); self.assertEqual(c.returncode,7)
        self.assertTrue(result['physically_drained']); self.assertLess(len(p.written),150000)
    def test_initial_orphan_exit_before_eof_once(self):
        e,p,s,c=fixture([INITIAL,leave(3),entry(),leave(-12,True),dict(status=0)],b'x'*100000)
        result=e.run(); self.assertTrue(result['observation_complete'])
        self.assertEqual(result['counters']['attach_orphan_exit'],1)
        row=next(x for x in records(s) if x['kind']=='attach_boundary_unpaired_exit')
        self.assertFalse(row['attributed'])
    def test_initial_orphan_after_eof_fails(self):
        e,p,s,c=fixture([INITIAL,leave(3),dict(status=0)])
        result=e.run(); self.assertIn('exit_without_entry',result['failure'])
        self.assertFalse(result['observation_complete']); self.assertEqual(result['counters']['attach_orphan_exit'],0)
    def test_second_orphan_exit_fails(self):
        e,p,s,c=fixture([INITIAL,leave(3),leave(3)],b'x'*300000)
        result=e.run(); self.assertIn('exit_without_entry',result['failure'])
        self.assertEqual(result['counters']['attach_orphan_exit'],1)
    def test_first_entry_closes_attach_exemption(self):
        e,p,s,c=fixture([INITIAL,entry(0),leave(3),leave(3)],b'x'*400000)
        self.assertIn('exit_without_entry',e.run()['failure'])
    def test_overwritten_entry_is_incomplete(self):
        e,p,s,c=fixture([INITIAL,entry(),entry()])
        self.assertIn('entry_overwrites_missing_exit',e.run()['failure'])
    def test_terminal_with_missing_memory_exit_is_incomplete(self):
        e,p,s,c=fixture([INITIAL,entry(),dict(status=0)])
        result=e.run(); self.assertFalse(result['observation_complete'])
        self.assertEqual(result['failure'],'terminal_with_missing_syscall_exit')
        self.assertEqual(c.returncode,0); self.assertEqual(result['sdk_terminal']['wait_status'],0)
    def test_plain_sigtrap_is_delivered(self):
        e,p,s,c=fixture([INITIAL,dict(status=stop(5)),dict(status=0)])
        self.assertTrue(e.run()['observation_complete'])
        self.assertIn(('resume',PID,'syscall',5),p.log)
    def test_group_stop_fails_and_actually_reaps(self):
        e,p,s,c=fixture([INITIAL,dict(status=stop(19,128))])
        result=e.run(); self.assertIn('unsupported_group_stop',result['failure'])
        self.assertTrue(result['physically_drained']); self.assertEqual(c.returncode,-15)
    def test_ptrace_event_exit_is_only_a_stop(self):
        e,p,s,c=fixture([INITIAL,dict(status=stop(5,6)),None,dict(status=3<<8)])
        seen=[]
        p.on_wait=lambda _,tid:seen.append(c.returncode)
        result=e.run(); self.assertEqual(seen,[None,None,None,None])
        self.assertEqual(c.returncode,3); self.assertEqual(result['sdk_terminal']['wait_status'],768)
    def test_echild_is_never_success(self):
        e,p,s,c=fixture([INITIAL,ChildProcessError('ECHILD')])
        result=e.run(); self.assertFalse(result['observation_complete'])
        self.assertIn('ChildProcessError',result['failure']); self.assertEqual(c.returncode,-15)
    def test_attach_denied_uses_true_untraced_terminal(self):
        e,p,s,c=fixture(); p.seize_error=PermissionError('EPERM')
        result=e.run(); self.assertFalse(result['observation_complete'])
        self.assertEqual(c.returncode,-15); self.assertEqual(result['sdk_terminal']['wait_status'],15)
        self.assertFalse(any(x[0]=='interrupt' for x in p.log))
    def test_authority_failure_never_seizes(self):
        e,p,s,c=fixture(); p.authority_error=ValueError('not authenticated')
        result=e.run(); self.assertFalse(any(x[0]=='seize' for x in p.log))
        self.assertTrue(result['physically_drained'])
    def test_wrong_arch_fails(self):
        e,p,s,c=fixture([INITIAL,entry(9,arch=3)])
        self.assertIn('unsupported_syscall_arch',e.run()['failure'])
    def test_whole_info_length_required(self):
        packet=info(1); packet['data']=packet['data'][:-1]
        with self.assertRaises(m.ObservationError): m.syscall_info(packet)
    def test_invalid_wait_not_a_terminal(self):
        for raw in (0xffff,2**32,-1,True,0x100000):
            with self.subTest(raw=raw),self.assertRaises(m.ObservationError):m.wait_event(raw)
    def test_term_then_kill_requires_actual_terminal(self):
        e,p,s,c=fixture([INITIAL,leave()]); p.term_ignored=True
        result=e.run(); self.assertEqual(c.returncode,-9)
        self.assertIn(('signal',PID,15),p.log); self.assertIn(('signal',PID,9),p.log)
        self.assertEqual(result['sdk_terminal']['wait_status'],9)
    def test_signals_alone_do_not_prove_drain(self):
        e,p,s,c=fixture([INITIAL,leave()]); p.term_ignored=True; p.kill_ignored=True
        result=e.run(); self.assertFalse(result['physically_drained'])
        self.assertFalse(result['lock_release_permitted']); self.assertIsNone(c.returncode)
        self.assertIsNone(result['sdk_terminal']); self.assertEqual(s.files,{})
    def test_unknown_group_is_not_adopted(self):
        e,p,s,c=fixture(); p.extra_group=[{'pid':999}]
        result=e.run(); self.assertFalse(result['physically_drained'])
        self.assertNotIn(999,e.tasks); self.assertEqual(s.files,{})
    def test_store_failure_never_uses_fallback(self):
        e,p,s,c=fixture(); s.fail_name=m.RECEIPT_NAME
        result=e.run(); self.assertTrue(result['physically_drained'])
        self.assertFalse(result['exit_receipt_saved']); self.assertTrue(result['trace_saved'])
        self.assertEqual(s.attempts,[m.TRACE_NAME,m.RECEIPT_NAME])
    def test_sticky_guard_cleanup_then_no_writer(self):
        calls=[0]
        def guard():
            calls[0]+=1
            if calls[0]>=3: raise ValueError('sticky guard')
        e,p,s,c=fixture(guard=guard); result=e.run()
        self.assertTrue(result['physically_drained']); self.assertEqual(c.returncode,-15)
        self.assertFalse(result['exit_receipt_saved']); self.assertEqual(s.attempts,[])
    def test_trace_overflow_explicit_incomplete(self):
        e,p,s,c=fixture()
        with patch.object(m,'TRACE_CAP',150):result=e.run()
        self.assertFalse(result['trace_complete']); self.assertGreater(result['trace_records_dropped'],0)
        self.assertLessEqual(len(e.trace),150); self.assertTrue(result['physically_drained'])
    def test_wait_budget_does_not_discard_consumed_terminal(self):
        e,p,s,c=fixture([INITIAL,dict(status=0)])
        with patch.object(m,'EVENT_CAP',1):result=e.run()
        self.assertFalse(result['observation_complete']); self.assertEqual(c.returncode,-15)
        self.assertEqual(result['sdk_terminal']['wait_status'],15)
    def test_wall_budget_unchanged(self):
        e,p,s,c=fixture([INITIAL]+[None]*4)
        old=p.now; bumped=[False]
        def now():
            value=old()
            if not bumped[0] and not e.ready and e.steps==1:
                p.clock+=901; value=p.clock; bumped[0]=True
            return value
        p.now=now
        result=e.run(); self.assertIn('original_wall_exceeded',result['failure'])
        self.assertEqual(m.LIMITS['wall_seconds'],900); self.assertTrue(result['physically_drained'])
    def test_no_second_run(self):
        e,p,s,c=fixture(); e.run()
        with self.assertRaises(m.ObservationError):e.run()
    def test_trace_failure_after_reap_preserves_exit_before_logging(self):
        e,p,s,c=fixture([INITIAL,entry(231),dict(status=4<<8)])
        def before_wait(_,tid):
            if e.tasks[tid]['entry'] is not None:
                e.trace=bytearray(m.TRACE_CAP)
        p.on_wait=before_wait
        result=e.run()
        self.assertEqual(c.returncode,4); self.assertEqual(result['sdk_terminal']['wait_status'],1024)
        self.assertFalse(result['trace_complete']); self.assertTrue(result['physically_drained'])
        self.assertFalse(any(x[0]=='signal' for x in p.log))
    def test_event_budget_is_checked_before_terminal_is_reaped(self):
        e,p,s,c=fixture([INITIAL,dict(status=6<<8)])
        p.term_ignored=True
        with patch.object(m,'EVENT_CAP',1):result=e.run()
        self.assertEqual(c.returncode,6); self.assertEqual(result['sdk_terminal']['wait_status'],1536)
        self.assertEqual(result['wait_events'],1); self.assertEqual(result['cleanup_wait_events'],1)
        self.assertFalse(result['observation_complete']); self.assertTrue(result['physically_drained'])
    def test_unknown_wait_return_cannot_allow_release(self):
        e,p,s,c=fixture([INITIAL,dict(status=0,returned_pid=999)])
        result=e.run(); self.assertFalse(result['ownership_complete'])
        self.assertFalse(result['lock_release_permitted']); self.assertNotIn(999,e.tasks)
    def test_invalid_gate_write_counts_stop_and_cleanup(self):
        for count in (0,-1,True,1000000):
            with self.subTest(count=count):
                e,p,s,c=fixture(); p.write_schedule=deque([count])
                result=e.run(); self.assertIn('gate_short_write_contract',result['failure'])
                self.assertFalse(result['gate_eof']); self.assertTrue(result['physically_drained'])
    def test_closed_gate_failure_blocks_release(self):
        e,p,s,c=fixture()
        def close(): raise ValueError('close failed')
        p.close_gate=close
        result=e.run(); self.assertFalse(result['physically_drained'])
        self.assertFalse(result['lock_release_permitted']); self.assertEqual(s.files,{})
    def test_unexpected_exec_fails_without_changing_owner(self):
        e,p,s,c=fixture([INITIAL,dict(status=stop(5,4))])
        result=e.run(); self.assertIn('unexpected_exec_event',result['failure'])
        self.assertEqual(e.tasks[PID]['owner'],OWNER); self.assertTrue(result['physically_drained'])
    def test_cleanup_event_exhaustion_never_reaps_then_discards(self):
        e,p,s,c=fixture([INITIAL,leave()])
        with patch.object(m,'CLEANUP_EVENT_CAP',0):result=e.run()
        self.assertFalse(result['physically_drained']); self.assertIsNone(c.returncode)
        self.assertFalse(result['exit_receipt_saved'])
    def test_budget_contract_keeps_original_caps(self):
        e,p,s,c=fixture(); e.run(); limits,caps,extra=s.budget_seen
        self.assertEqual(limits,dict(address_space_bytes=2147483648,cpu=6,cpu_seconds=600,
            hard_cpu_seconds=605,nice=19,output_file_bytes=2097152,wall_seconds=900))
        self.assertEqual(caps,dict(raw_files=4392,raw_bytes=12288000,metadata_bytes=8388608,
            allocated_bytes=33554432,one_file_bytes=2097152))
        self.assertEqual(extra['runtime_metadata_bytes'],79008)
        self.assertEqual(extra['trace_bytes'],2097152); self.assertEqual(extra['summary_bytes'],16384)
    def test_no_runtime_or_process_backend_in_source(self):
        tree=ast.parse(Path(m.__file__).read_text())
        imports=[node.names[0].name for node in tree.body if isinstance(node,ast.Import)]
        self.assertEqual(imports,['copy','json'])
        forbidden={'poll','wait','waitpid','Popen','CDLL','PyDLL','killpg','system','open'}
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(node.func,ast.Attribute) and node.func.attr in forbidden
                             or isinstance(node.func,ast.Name) and node.func.id in forbidden
                             for node in calls))


class ForkTests(unittest.TestCase):
    def setup_fork(self, child_events):
        events=[INITIAL,entry(56),dict(status=stop(5,3),new_tid=CHILD),leave(CHILD),
                None,None,entry(231),dict(status=0)]
        e,p,s,c=fixture(events)
        # Child's stop is available before the parent's event is collected.
        p.queues[CHILD]=deque(copy.deepcopy(child_events))
        return e,p,s,c
    def test_child_ready_before_parent_event_only_waited_after_event(self):
        e,p,s,c=self.setup_fork([INITIAL,entry(231),dict(status=0)])
        result=e.run(); self.assertTrue(result['observation_complete'])
        self.assertEqual(list(e.tasks),[PID,CHILD]); self.assertEqual(result['counters']['fork_events'],1)
        first_child_wait=next(i for i,x in enumerate(p.log) if x==('wait',CHILD))
        parent_resumes=[i for i,x in enumerate(p.log[:first_child_wait]) if x==('resume',PID,'syscall',0)]
        self.assertGreaterEqual(len(parent_resumes),3)
    def test_child_birth_orphan_exit_fails_without_invented_entry(self):
        e,p,s,c=self.setup_fork([INITIAL,leave(0)])
        result=e.run(); self.assertIn('exit_without_entry',result['failure'])
        self.assertTrue(result['physically_drained']); self.assertEqual(result['counters']['attach_orphan_exit'],0)
        self.assertFalse(any(row['kind']=='memory_entry' and row.get('tid')==CHILD for row in records(s)))
    def test_normal_task_budget_still_tracks_child_for_cleanup(self):
        e,p,s,c=self.setup_fork([INITIAL,entry(231),dict(status=0)])
        with patch.object(m,'TASK_CAP',1):result=e.run()
        self.assertIn('observation_task_budget_exceeded',result['failure'])
        self.assertIn(CHILD,e.tasks); self.assertTrue(result['physically_drained'])
        self.assertIn(('signal',CHILD,9),p.log)
    def test_hard_task_budget_does_not_claim_complete_ownership(self):
        e,p,s,c=self.setup_fork([INITIAL,dict(status=0)])
        with patch.object(m,'CLEANUP_TASK_CAP',1):result=e.run()
        self.assertFalse(result['ownership_complete']); self.assertFalse(result['physically_drained'])
        self.assertEqual(result['untracked_kernel_event']['tid'],CHILD); self.assertEqual(s.files,{})
    def test_duplicate_kernel_task_rejected(self):
        e,p,s,c=fixture([INITIAL,dict(status=stop(5,3),new_tid=PID)])
        self.assertIn('unique_kernel_task_event',e.run()['failure'])
    def test_parent_terminal_does_not_imply_child_terminal(self):
        e,p,s,c=self.setup_fork([INITIAL,None,None,None,dict(status=0)])
        result=e.run(); self.assertTrue(result['physically_drained'])
        self.assertEqual(result['task_terminals'][str(CHILD)]['terminal_status'],0)
    def test_event_budget_cleanup_still_discovers_kernel_child(self):
        e,p,s,c=self.setup_fork([INITIAL,dict(status=0)])
        p.term_ignored=True
        with patch.object(m,'EVENT_CAP',2):result=e.run()
        self.assertIn(CHILD,e.tasks); self.assertTrue(result['physically_drained'])
        self.assertEqual(result['counters']['fork_events'],1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
