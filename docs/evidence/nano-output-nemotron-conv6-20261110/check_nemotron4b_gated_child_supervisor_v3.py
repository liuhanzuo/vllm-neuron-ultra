"""Real stdlib child regressions for early failure output preservation.

Uses generated process behavior only, never the installed helper/SDK/model.
The original v2 lost early stderr; v3 preserves available bounded pipe bytes
after owned-child cleanup and separately reports EOF, truncation and timeout.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

import nemotron4b_process_handshake_v1 as handshake
import nemotron4b_gated_child_supervisor_v2 as previous
import nemotron4b_gated_child_supervisor_v3 as supervisor

NONCE='a'*32
EARLY_OUT=b'pre-gate-output\x00\xff\n'
EARLY_ERR=b'pre-gate-diagnostic\xfe\n'
LATE_OUT=b'post-gate-output\x00\xff\n'
LATE_ERR=b'post-gate-diagnostic\xfe\n'


def fixture():
    mode=sys.argv[2];root=Path(sys.argv[3])
    ready=int(sys.argv[sys.argv.index('--ready-fd')+1])
    release=int(sys.argv[sys.argv.index('--release-fd')+1])
    if mode in('early-exit','malformed','capture-denied','early-overflow'):
        os.write(1,EARLY_OUT);os.write(2,EARLY_ERR)
    if mode=='early-exit':os._exit(37)
    if mode=='early-overflow':
        os.write(1,b'x'*512);os.write(2,b'y'*512);os._exit(38)
    if mode=='malformed':
        os.write(ready,b'bad');os.close(ready);os.close(release);os._exit(39)
    handshake.child_wait(ready,release,nonce=NONCE,role='helper-0',
                         deadline=time.monotonic()+5)
    (root/'body-started').write_text('generated fixture')
    os.write(1,LATE_OUT);os.write(2,LATE_ERR)
    if mode=='late-exit':os._exit(41)
    if mode=='late-overflow':os.write(2,b'z'*1024)
    if mode=='timeout':
        time.sleep(3);(root/'body-finished').touch()


class ChildOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nano-child-output-fixture-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        from nemotron4b_worker_receipts_v1 import ReceiptWriter
        self.writer=ReceiptWriter(self.root,allowed_names=['capture.json'],guard=lambda:None)

    def invoke(self,mode,implementation=supervisor,**changes):
        cache=self.root/'private-cache';cache.mkdir(mode=0o700)
        def command(ready,release):
            return [sys.executable,'-B','-S','-X','pycache_prefix='+str(cache),
                str(Path(__file__).resolve()),'--fixture',mode,str(self.root),
                '--ready-fd',str(ready),'--release-fd',str(release)]
        def record(value):
            if mode=='capture-denied':raise OSError('generated capture rejected')
            self.assertFalse((self.root/'body-started').exists())
            return self.writer.write('capture.json',value)
        kwargs=dict(environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',
            PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0'),
            cwd=str(self.root),nonce=NONCE,role='helper-0',
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],guard=lambda:None,
            record_capture=record,handshake_seconds=2,execution_seconds=2,
            stdout_cap=4096,stderr_cap=4096)
        kwargs.update(changes)
        start=time.monotonic();result=implementation.run(command,**kwargs)
        self.assertLess(time.monotonic()-start,5)
        self.assertTrue(result['child_reaped']);self.assertEqual(result['cleanup_errors'],[])
        self.assertFalse(result['runtime_admitted']);self.assertFalse(result['owned_group_drained_proven'])
        self.assertEqual(list(cache.iterdir()),[])
        return result

    def test_original_v2_reproduces_lost_pre_handshake_output(self):
        result=self.invoke('early-exit',implementation=previous)
        self.assertFalse(result['successful'])
        self.assertEqual(result['output'],{'stdout':b'','stderr':b''})

    def test_early_exit_preserves_exact_binary_output_without_grant(self):
        result=self.invoke('early-exit')
        self.assertFalse(result['successful']);self.assertIsNone(result['handshake'])
        self.assertEqual(result['output'],{'stdout':EARLY_OUT,'stderr':EARLY_ERR})
        self.assertTrue(result['output_pipe_capture_complete'])
        self.assertTrue(result['output_may_be_partial'])
        self.assertFalse((self.root/'body-started').exists())
        self.assertIn(result['actual_exit_code'],(37,-15))

    def test_malformed_handshake_preserves_diagnostics(self):
        result=self.invoke('malformed')
        self.assertFalse(result['successful']);self.assertIsNone(result['handshake'])
        self.assertEqual(result['output']['stderr'],EARLY_ERR)
        self.assertTrue(result['output_pipe_capture_complete'])
        self.assertFalse((self.root/'capture.json').exists())

    def test_rejected_parent_record_still_preserves_early_output(self):
        result=self.invoke('capture-denied')
        self.assertFalse(result['successful'])
        self.assertEqual(result['failure']['message'],'generated capture rejected')
        self.assertTrue(result['output']['stdout'].startswith(EARLY_OUT))
        self.assertTrue(result['output']['stderr'].startswith(EARLY_ERR))
        self.assertFalse((self.root/'body-started').exists())

    def test_normal_capture_and_complete_output(self):
        result=self.invoke('normal')
        self.assertTrue(result['successful'],result)
        self.assertEqual(result['actual_exit_code'],0)
        self.assertIsNotNone(result['handshake']['capture_record'])
        self.assertEqual(result['output'],{'stdout':LATE_OUT,'stderr':LATE_ERR})
        self.assertTrue(result['output_pipe_capture_complete'])
        self.assertFalse(result['output_may_be_partial'])

    def test_post_grant_nonzero_exit_keeps_output_and_actual_code(self):
        result=self.invoke('late-exit')
        self.assertFalse(result['successful']);self.assertEqual(result['actual_exit_code'],41)
        self.assertEqual(result['output'],{'stdout':LATE_OUT,'stderr':LATE_ERR})
        self.assertTrue(result['output_pipe_capture_complete'])

    def test_pre_handshake_overflow_is_bounded_and_explicitly_incomplete(self):
        result=self.invoke('early-overflow',stdout_cap=64,stderr_cap=64)
        self.assertFalse(result['successful'])
        self.assertEqual(result['output']['stdout'],(EARLY_OUT+b'x'*512)[:64])
        self.assertEqual(result['output']['stderr'],(EARLY_ERR+b'y'*512)[:64])
        self.assertTrue(all(result['output_capture']['truncated'].values()))
        self.assertFalse(result['output_pipe_capture_complete'])

    def test_post_handshake_overflow_preserves_prefix_and_other_pipe(self):
        result=self.invoke('late-overflow',stderr_cap=64)
        self.assertFalse(result['successful'])
        self.assertEqual(result['output']['stderr'],(LATE_ERR+b'z'*1024)[:64])
        self.assertEqual(result['output']['stdout'],LATE_OUT)
        self.assertTrue(result['output_capture']['truncated']['stderr'])
        self.assertFalse(result['output_pipe_capture_complete'])

    def test_timeout_reaps_owned_child_and_retains_output(self):
        result=self.invoke('timeout',execution_seconds=1)
        self.assertFalse(result['successful']);self.assertTrue(result['cleanup_signals'])
        self.assertEqual(result['output'],{'stdout':LATE_OUT,'stderr':LATE_ERR})
        self.assertTrue(result['output_pipe_capture_complete'])
        self.assertFalse((self.root/'body-finished').exists())

    def test_guard_denial_after_capture_does_not_suppress_failure_diagnostics(self):
        def guard():
            if (self.root/'capture.json').exists():raise ValueError('generated post-capture denial')
        result=self.invoke('normal',guard=guard)
        self.assertFalse(result['successful']);self.assertFalse((self.root/'body-started').exists())
        self.assertEqual(result['failure']['message'],'generated post-capture denial')
        self.assertTrue(result['output_pipe_capture_complete'])


class DrainTests(unittest.TestCase):
    def test_open_writer_produces_bounded_incomplete_capture_without_waiting_for_descendant(self):
        readfd,writefd=os.pipe()
        stream=os.fdopen(readfd,'rb');self.addCleanup(stream.close);self.addCleanup(os.close,writefd)
        os.write(writefd,b'pending')
        child=types.SimpleNamespace(stdout=stream,stderr=None)
        buffers={'stdout':bytearray(),'stderr':bytearray()}
        capture=dict(eof={'stdout':False,'stderr':True},
                     truncated={'stdout':False,'stderr':False},drain_timed_out=False)
        before=time.monotonic()
        supervisor.drain_after_exit(child,before+.1,{'stdout':64,'stderr':64},buffers,capture)
        self.assertLess(time.monotonic()-before,.5)
        self.assertEqual(bytes(buffers['stdout']),b'pending')
        self.assertTrue(capture['drain_timed_out']);self.assertFalse(capture['eof']['stdout'])

    def test_exact_cap_with_eof_is_complete_and_not_truncated(self):
        readfd,writefd=os.pipe();os.write(writefd,b'x'*64);os.close(writefd)
        stream=os.fdopen(readfd,'rb');self.addCleanup(stream.close)
        child=types.SimpleNamespace(stdout=stream,stderr=None)
        buffers={'stdout':bytearray(),'stderr':bytearray()}
        capture=dict(eof={'stdout':False,'stderr':True},
                     truncated={'stdout':False,'stderr':False},drain_timed_out=False)
        supervisor.drain_after_exit(child,time.monotonic()+.1,{'stdout':64,'stderr':64},buffers,capture)
        self.assertEqual(bytes(buffers['stdout']),b'x'*64)
        self.assertTrue(capture['eof']['stdout']);self.assertFalse(capture['truncated']['stdout'])


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture':fixture()
    else:unittest.main(verbosity=2)
