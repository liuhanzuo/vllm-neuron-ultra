"""Actual stdlib-only fixtures for bounded gated-child outcomes and cleanup."""
import json
import os
from pathlib import Path
import sys
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_child_supervisor_v1 as supervisor

NONCE='3'*32


def child_fixture():
    ready,release,mode=int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
    if mode=='early-exit':
        os.close(ready);os.close(release)
        return
    proof=handshake.child_wait(ready,release,nonce=NONCE,role='helper-1',deadline=time.monotonic()+5)
    if mode=='nonzero':
        os.write(2,b'fixture failed after actual grant\n')
        raise SystemExit(17)
    if mode=='stdout-overflow':
        os.write(1,b'x'*8192)
        return
    if mode=='stderr-overflow':
        os.write(2,b'x'*32768)
        return
    if mode=='timeout':
        os.write(1,b'fixture reached post-grant wait\n')
        time.sleep(5)
        return
    print(json.dumps(dict(child_release_ns=proof['child_release_ns'],fixture_completed=True,
                         actual_SDK_helper_executed=False)),flush=True)


class Supervision(unittest.TestCase):
    def invoke(self,mode='normal',guard=lambda:None):
        def command(ready,release):
            return [sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),
                    '--fixture-child',str(ready),str(release),mode]
        return supervisor.run(command,environment={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
            cwd=str(Path(__file__).resolve().parent),nonce=NONCE,role='helper-1',
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],guard=guard,
            handshake_seconds=2,execution_seconds=1,stdout_cap=4096,stderr_cap=16384)

    def assert_drained_child(self,result):
        self.assertTrue(result['child_reaped'])
        self.assertTrue(result['actual_exit_observed'])
        self.assertEqual(result['cleanup_errors'],[])
        self.assertFalse(result['owned_group_drained_proven'])
        self.assertFalse(result['runtime_admitted'])
        self.assertFalse(result['full_model_qualified'])
        self.assertFalse(result['new_speedup_claim'])

    def test_successful_actual_short_lived_fixture(self):
        result=self.invoke()
        self.assert_drained_child(result)
        self.assertTrue(result['successful'])
        self.assertEqual(result['actual_exit_code'],0)
        self.assertEqual(result['cleanup_signals'],[])
        self.assertFalse(result['output_may_be_partial'])
        record=json.loads(result['output']['stdout'])
        self.assertTrue(record['fixture_completed'])
        self.assertFalse(record['actual_SDK_helper_executed'])
        self.assertLessEqual(result['handshake']['grant']['grant_ns'],record['child_release_ns'])
        self.assertLess(record['child_release_ns'],result['exit_ns'])

    def test_nonzero_actual_exit_is_retained_as_failure(self):
        result=self.invoke('nonzero')
        self.assert_drained_child(result)
        self.assertFalse(result['successful'])
        self.assertEqual(result['actual_exit_code'],17)
        self.assertEqual(result['output']['stderr'],b'fixture failed after actual grant\n')
        self.assertEqual(result['cleanup_signals'],[])

    def test_early_zero_exit_before_capture_never_qualifies(self):
        result=self.invoke('early-exit')
        self.assert_drained_child(result)
        self.assertFalse(result['successful'])
        self.assertEqual(result['actual_exit_code'],0)
        self.assertIsNone(result['handshake'])

    def test_stdout_and_stderr_cap_failure_is_bounded_and_reaped(self):
        for mode,name,cap in (('stdout-overflow','stdout',4096),('stderr-overflow','stderr',16384)):
            with self.subTest(mode=mode):
                result=self.invoke(mode)
                self.assert_drained_child(result)
                self.assertFalse(result['successful'])
                self.assertEqual(len(result['output'][name]),cap+1)
                self.assertIn(name+' cap exceeded',result['failure']['message'])

    def test_timeout_cleanup_cannot_be_promoted_to_success(self):
        start=time.monotonic();result=self.invoke('timeout')
        self.assertLess(time.monotonic()-start,4)
        self.assert_drained_child(result)
        self.assertFalse(result['successful'])
        self.assertEqual(result['cleanup_signals'],['SIGTERM'])
        self.assertEqual(result['actual_exit_code'],-15)
        self.assertEqual(result['output']['stdout'],b'fixture reached post-grant wait\n')
        self.assertIn('deadline',result['failure']['message'])

    def test_guard_failure_before_spawn_creates_no_child(self):
        def guard():raise RuntimeError('guard stopped launch')
        result=self.invoke(guard=guard)
        self.assertFalse(result['successful'])
        self.assertFalse(result['child_reaped'])
        self.assertFalse(result['actual_exit_observed'])
        self.assertIsNone(result['actual_exit_code'])
        self.assertEqual(result['cleanup_signals'],[])

    def test_guard_failure_after_spawn_still_reaps_owned_child(self):
        count=0
        def guard():
            nonlocal count
            count+=1
            if count==4:raise RuntimeError('guard denied grant')
        result=self.invoke(guard=guard)
        self.assert_drained_child(result)
        self.assertFalse(result['successful'])
        self.assertIsNone(result['handshake'])
        self.assertEqual(result['failure']['message'],'guard denied grant')

    def test_no_fd_leak_across_success_failure_and_timeout(self):
        before=len(list(Path('/proc/self/fd').iterdir()))
        for mode in ('normal','nonzero','early-exit'):
            self.assert_drained_child(self.invoke(mode))
        self.assertEqual(len(list(Path('/proc/self/fd').iterdir())),before)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture-child':child_fixture()
    else:unittest.main(verbosity=2)
