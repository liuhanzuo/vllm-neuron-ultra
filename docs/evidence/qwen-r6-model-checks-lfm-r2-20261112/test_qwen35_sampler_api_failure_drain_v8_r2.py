"""Stdlib fake process/pipe/clock coverage; no child launch or native imports."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from contextlib import ExitStack

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import qwen35_sampler_runtime_service_v8 as service
import qwen35_sampler_runtime_adapter_v8 as adapter
import qwen35_sampler_runtime_contract_v8 as contract


class Clock:
    def __init__(self): self.now = 0.
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class World:
    def __init__(self, stage, blocks, *, exit_code=1, hang=False, ignore_term=False, duration=100):
        self.stage = Path(stage); self.clock = Clock(); self.blocks = list(blocks)
        self.exit_code, self.hang, self.ignore_term = exit_code, hang, ignore_term
        self.code = None; self.eof = False; self.registered = False
        self.signals = []; self.output = bytearray(); self.waits = []; self.hook = lambda: None
        self.phase_ref = {'path': str(self.stage/'phase-spec.json'), 'sha256': 'a'*64}
        self.owner = {'process': {'pid': 10}, 'service_owner': {'unit': 'synthetic'}}
        self.phase = {'control': str(self.stage), 'phase': 'capture', 'coordinator': {'pid': 9},
                      'properties': {'RuntimeMaxSec': str(duration)}}
        self.child = {'pid': 11, 'ppid': 10, 'start_ticks': '110', 'cgroup': '0::/synthetic',
                      'argv': service.child_process(self.phase, self.phase_ref, True)}
        self.stdout = NS(fileno=lambda: 123, close=lambda: None)
        self.process = NS(pid=11, stdout=self.stdout, poll=self.poll, wait=self.wait)
        self.selector = NS(register=self.register, unregister=self.unregister,
            get_map=lambda: {123: True} if self.registered else {}, select=self.select, close=lambda: None)
        self.p = NS(proc=lambda pid: copy.deepcopy(self.child))
        (self.stage/'supervisor-release.json').write_text(json.dumps({'phase_spec': self.phase_ref,
            'supervisor': self.owner, 'coordinator': self.phase['coordinator']}))
    def register(self, *args): self.registered = True
    def unregister(self, *args): self.registered = False
    def poll(self):
        if self.eof and not self.hang and self.code is None: self.code = self.exit_code
        return self.code
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.poll() is None: raise subprocess.TimeoutExpired('mock', timeout)
        return self.code
    def select(self, timeout):
        self.clock.now += timeout; self.hook()
        if self.blocks or not self.hang or self.code is not None:
            return [(NS(fileobj=self.stdout), 1)]
        return []
    def read(self, fd, count):
        if self.blocks:
            block = self.blocks.pop(0)
            if len(block) > count:
                self.blocks.insert(0, block[count:]); block = block[:count]
            return block
        self.eof = True; return b''
    def write(self, fd, block): self.output.extend(block); return len(block)
    def killpg(self, pid, sig):
        self.signals.append((self.clock.now, sig))
        if sig == signal.SIGKILL or not self.ignore_term:
            self.code = -sig
    def run(self, *, guard=lambda: None, cap=None):
        context = {'guard': guard, 'service': NS(owner=self.owner, group='/synthetic'), 'child_environment': {}}
        with ExitStack() as stack:
            for target, name, value in [
                (service.subprocess, 'Popen', lambda *a, **k: self.process),
                (service.selectors, 'DefaultSelector', lambda: self.selector),
                (service.signal, 'signal', lambda *a: None),
                (service, 'os', NS(read=self.read, write=self.write, killpg=self.killpg)),
                (service, 'process_api', lambda: self.p),
                (service.time, 'monotonic', self.clock.monotonic),
                (service.time, 'sleep', self.clock.sleep), (service.time, 'time_ns', lambda: 1),
            ]: stack.enter_context(patch.object(target, name, value))
            if cap is not None: stack.enter_context(patch.dict(service.LIMITS, server_log_bytes=cap))
            return service.supervise(self.phase_ref, lambda *a, **k: (self.phase, context))
    def receipt(self, name): return json.loads((self.stage/name).read_text())


class FailureDrainTests(unittest.TestCase):
    def world(self, blocks, **kwargs):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        return World(tmp.name, blocks, **kwargs)
    def test_natural_exit_drains_later_traceback_blocks(self):
        blocks = [b'Traceback (most recent call last):\n', b'  later_frame\n', b'RootError: exact cause\n']
        w = self.world(blocks)
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        self.assertEqual(bytes(w.output), b''.join(blocks)); self.assertEqual(w.signals, [])
        self.assertEqual(w.receipt('child-exit.json')['child_returncode'], 1)
        done = w.receipt('api-failure-complete.json')
        self.assertTrue(done['log_eof_observed']); self.assertFalse(done['traceback_completeness_claimed'])
    def test_marker_cannot_be_success_even_if_child_exits_zero(self):
        w = self.world([b'Traceback (most recent call last):\n', b'handled\n'], exit_code=0)
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        self.assertEqual(w.receipt('child-exit.json')['child_returncode'], 0)
        self.assertTrue(w.receipt('api-failure-complete.json')['failed'])
    def test_all_existing_markers_fail_and_drain(self):
        for marker in service.API_ERROR_MARKERS:
            with self.subTest(marker=marker):
                w = self.world([marker, b' remaining diagnostic\n'])
                with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
                self.assertTrue(bytes(w.output).endswith(b' remaining diagnostic\n'))
    def test_marker_matches_across_pipe_blocks_and_case(self):
        w = self.world([b'TRACEBACK (most ', b'recent call last):\n', b'Error: root\n'])
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        self.assertEqual(w.receipt('api-failure.json')['marker'], service.API_ERROR_MARKERS[0].decode())
    def test_repeated_markers_cannot_reset_deadline(self):
        w = self.world([service.API_ERROR_MARKERS[0]]*5, hang=True)
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        f = w.receipt('api-failure.json')
        self.assertAlmostEqual(f['observed_monotonic'], .1)
        self.assertAlmostEqual(f['deadline_monotonic'], 45.1)
        self.assertEqual([sig for _, sig in w.signals], [signal.SIGTERM])
    def test_hung_process_uses_term_then_kill_within_original_budget(self):
        w = self.world([service.API_ERROR_MARKERS[0]], hang=True, ignore_term=True)
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        f = w.receipt('api-failure.json')
        self.assertEqual([sig for _, sig in w.signals], [signal.SIGTERM, signal.SIGKILL])
        self.assertLess(w.clock.now, f['deadline_monotonic'])
        self.assertEqual(w.receipt('child-exit.json')['child_returncode'], -signal.SIGKILL)
    def test_phase_remaining_time_caps_cleanup(self):
        w = self.world([service.API_ERROR_MARKERS[0]], hang=True, ignore_term=True, duration=30)
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        f = w.receipt('api-failure.json')
        self.assertEqual(f['deadline_monotonic'], 15.)
        self.assertLess(w.clock.now, 15.)
    def test_explicit_stop_after_failure_does_not_accept_or_skip_tail(self):
        w = self.world([service.API_ERROR_MARKERS[0], b'RootError: preserve\n'])
        def hook():
            if (w.stage/'api-failure.json').exists():
                (w.stage/'stop-request.json').write_text(json.dumps({'phase_spec': w.phase_ref,
                    'supervisor': w.owner, 'coordinator': w.phase['coordinator']}))
        w.hook = hook
        with self.assertRaisesRegex(ValueError, 'actual server'): w.run()
        self.assertTrue(bytes(w.output).endswith(b'RootError: preserve\n'))
        self.assertTrue(w.receipt('child-exit.json')['stop_requested'])
        self.assertTrue(w.receipt('api-failure-complete.json')['failed'])
    def test_log_cap_is_not_relaxed_during_collection(self):
        marker = service.API_ERROR_MARKERS[0]; w = self.world([marker, b'x'*30])
        with self.assertRaisesRegex(ValueError, 'finite phase log cap'): w.run(cap=len(marker)+5)
        self.assertEqual([sig for _, sig in w.signals], [signal.SIGKILL])
        self.assertFalse((w.stage/'api-failure-complete.json').exists())
    def test_guard_failure_still_interrupts_collection(self):
        w = self.world([service.API_ERROR_MARKERS[0], b'x'])
        def guard():
            if (w.stage/'api-failure.json').exists(): raise ValueError('resource failure')
        with self.assertRaisesRegex(ValueError, 'resource failure'): w.run(guard=guard)
        self.assertEqual([sig for _, sig in w.signals], [signal.SIGKILL])
    def test_no_marker_keeps_original_success(self):
        w = self.world([b'normal output\n'], exit_code=0)
        self.assertEqual(w.run(), 0)
        self.assertFalse((w.stage/'api-failure.json').exists())
    def test_changed_failure_owner_or_deadline_is_rejected(self):
        w = self.world([]); f = service.api_failure_receipt(w.phase_ref,w.owner,w.child,
            service.API_ERROR_MARKERS[0], 1.,100.,10)
        service.validate_api_failure(f,w.phase_ref,w.owner,w.child)
        for key, value in [('child_bootstrap_owner',{}), ('deadline_monotonic',100.),
                           ('failed',False),('marker','unknown'),('observed_monotonic',float('nan'))]:
            changed = copy.deepcopy(f); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                service.validate_api_failure(changed,w.phase_ref,w.owner,w.child)
    def test_client_failure_prevents_operation_or_acceptance(self):
        for fail_on in (1, 2):
            calls = []; checks = []
            def reject():
                checks.append(1)
                if len(checks)==fail_on: raise ValueError('latched')
            phase = NS(reject_api_failure=reject)
            clients = adapter.FailureCheckedClients(NS(active=phase), NS(diagnostic=lambda *a: calls.append('client')))
            with self.subTest(fail_on=fail_on), self.assertRaisesRegex(ValueError, 'latched'):
                clients.diagnostic('case','output')
            self.assertEqual(calls, [] if fail_on==1 else ['client'])
    def test_client_success_preserves_exact_arguments_result_and_two_checks(self):
        checks=[]; calls=[]; result=object()
        def original(*args,**kwargs): calls.append((args,kwargs)); return result
        clients=adapter.FailureCheckedClients(NS(active=NS(reject_api_failure=lambda:checks.append(1))),
                                             NS(workloads=original))
        self.assertIs(clients.workloads('case','output',key=4), result)
        self.assertEqual(calls,[(('case','output'),{'key':4})]);self.assertEqual(len(checks),2)
    def test_phase_reject_waits_for_supervisor_without_signalling(self):
        phase=adapter.Phase.__new__(adapter.Phase); clock=Clock(); observations=[]
        failure={'deadline_monotonic':1.}
        phase.api_failure=lambda:failure
        phase.adapter=NS(observer=NS(check=lambda:None))
        phase.service=NS(observe=lambda:observations.append(clock.now) or {'terminal':clock.now>=.3})
        with patch.object(adapter.time,'monotonic',clock.monotonic),patch.object(adapter.time,'sleep',clock.sleep),\
             patch.object(adapter,'process_api',return_value=NS(terminal=lambda v:v['terminal'])),\
             self.assertRaisesRegex(ValueError,'actual server'):
            phase.reject_api_failure()
        self.assertGreaterEqual(clock.now,.3);self.assertLess(clock.now,1.)
    def test_phase_collection_deadline_never_renews(self):
        phase=adapter.Phase.__new__(adapter.Phase);clock=Clock()
        phase.adapter=NS(observer=NS(check=lambda:None));phase.service=NS(observe=lambda:{})
        with patch.object(adapter.time,'monotonic',clock.monotonic),patch.object(adapter.time,'sleep',clock.sleep),\
             patch.object(adapter,'process_api',return_value=NS(terminal=lambda v:False)):
            phase.wait_failed_api({'deadline_monotonic':.25})
            phase.wait_failed_api({'deadline_monotonic':.25})
        self.assertEqual(clock.now,.25)
    def test_cleanup_still_runs_when_collection_guard_fails(self):
        phase=adapter.Phase.__new__(adapter.Phase);events=[];end={'cleanup_complete':True}
        phase.end=None;phase.child={'pid':4};phase.stage=Path('/synthetic')
        phase.service=NS(started=True,cleanup=lambda:events.append('cleanup') or end)
        phase.adapter=NS(observer=NS(current='phase'),active=phase)
        phase.api_failure=lambda:{'deadline_monotonic':1.}
        def fail(_):raise ValueError('original resource failure')
        phase.wait_failed_api=fail
        with patch.object(adapter,'process_api',return_value=NS(gone=lambda child:events.append('gone'))),\
             patch.object(adapter,'once',side_effect=lambda *a:events.append('terminal')),\
             self.assertRaisesRegex(ValueError,'original resource failure'):
            phase.finish()
        self.assertEqual(events,['cleanup','gone','terminal'])
        self.assertIs(phase.end,end);self.assertIsNone(phase.adapter.active)
        self.assertIsNone(phase.adapter.observer.current)

    def test_exit_before_pipe_eof_still_forwards_remaining_blocks(self):
        blocks=[service.API_ERROR_MARKERS[0],b'late traceback frame\n',b'RootError: root\n']
        w=self.world(blocks)
        def hook():
            if (w.stage/'api-failure.json').exists():w.code=1
        w.hook=hook
        with self.assertRaisesRegex(ValueError,'actual server'):w.run()
        self.assertEqual(bytes(w.output),b''.join(blocks));self.assertEqual(w.signals,[])
        self.assertTrue(w.receipt('api-failure-complete.json')['log_eof_observed'])

    def test_original_limits_phases_and_science_references_unchanged(self):
        import qwen35_sampler_runtime_contract_v6 as old
        for name in ('LIMITS','PHASES','LOCK_IDENTITY','MODEL_PROPERTIES','SERVER_PROPERTIES','CPU_PROPERTIES'):
            self.assertEqual(getattr(contract,name),getattr(old,name))
        self.assertEqual(contract.STARTUP_FILES,old.STARTUP_FILES)


if __name__=='__main__': unittest.main(verbosity=2)
