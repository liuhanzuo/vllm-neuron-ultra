"""Stdlib-only generated pipe fixtures and short-lived subprocess checks.

No native SDK/model/helper binary, service, serial lock or raw model data is
used. Every fixture child is waited/reaped before its observer parent exits.
"""
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron4b_process_handshake_v1 as handshake

NONCE = '1'*32


def child_fixture():
    ready, release = int(sys.argv[2]), int(sys.argv[3])
    mode = sys.argv[4]
    if mode == 'early-exit':
        os.close(ready);os.close(release)
        return
    nonce = '2'*32 if mode == 'wrong-nonce' else NONCE
    proof = handshake.child_wait(ready, release, nonce=nonce,
        role='helper-0', deadline=time.monotonic()+5)
    print(json.dumps(dict(proof=proof, generated_fixture_completed=True,
                         actual_SDK_helper_executed=False)), flush=True)


class PipeProtocol(unittest.TestCase):
    def setUp(self):
        self.fds = []
        self.addCleanup(self.close)

    def close(self):
        for fd in self.fds:
            try: os.close(fd)
            except OSError: pass

    def pipe(self):
        pair = os.pipe2(os.O_CLOEXEC)
        self.fds.extend(pair)
        return pair

    def packet(self, raw):
        read, write = self.pipe()
        os.write(write, raw)
        os.close(write);self.fds.remove(write)
        return read

    def test_one_canonical_frame_roundtrip_and_no_inheritance(self):
        read, write = self.pipe()
        value = dict(message='bounded fixture', count=3)
        handshake.send(write, value, time.monotonic()+1)
        os.close(write);self.fds.remove(write)
        self.assertEqual(handshake.receive(read,time.monotonic()+1),value)
        self.assertFalse(os.get_inheritable(read))

    def test_wrong_pipe_direction_and_regular_fd_rejected(self):
        read, write = self.pipe()
        with self.assertRaisesRegex(ValueError,'direction'):
            handshake.send(read,{},time.monotonic()+1)
        with self.assertRaisesRegex(ValueError,'direction'):
            handshake.receive(write,time.monotonic()+1)
        fd=os.open('/dev/null',os.O_RDONLY);self.fds.append(fd)
        with self.assertRaisesRegex(ValueError,'pipe descriptor'):
            handshake.receive(fd,time.monotonic()+1)

    def test_truncation_oversize_zero_trailing_duplicate_noncanonical_rejected(self):
        cases = [b'\x00',struct.pack('!I',0),struct.pack('!I',4093),
                 struct.pack('!I',10)+b'{}',struct.pack('!I',2)+b'{}x',
                 struct.pack('!I',13)+b'{"x":1,"x":2}',
                 struct.pack('!I',8)+b'{"x": 1}']
        for raw in cases:
            with self.subTest(raw=raw),self.assertRaises(ValueError):
                handshake.receive(self.packet(raw),time.monotonic()+1)

    def test_missing_writer_close_and_silent_writer_timeout(self):
        for complete in (False,True):
            read,write=self.pipe()
            if complete:os.write(write,struct.pack('!I',2)+b'{}')
            started=time.monotonic()
            with self.subTest(complete=complete),self.assertRaisesRegex(ValueError,'timeout|did not close'):
                handshake.receive(read,started+.02)
            self.assertLess(time.monotonic()-started,.5)

    def test_expired_send_and_oversized_frame_rejected(self):
        read,write=self.pipe()
        for value,deadline in (({},time.monotonic()-1),({'x':'a'*5000},time.monotonic()+1)):
            with self.assertRaises(ValueError):handshake.send(write,value,deadline)
        self.assertFalse(select.select([read],[],[],0)[0])

    def test_invalid_nonce_or_role_rejected(self):
        for nonce,role in (('x'*32,'helper-0'),('1'*31,'helper-0'),(NONCE,'helper-2'),(True,'rank-0')):
            with self.subTest(nonce=nonce,role=role),self.assertRaises(ValueError):
                handshake.scope(nonce,role)

    def test_actual_self_identity_matches_proc(self):
        value=handshake.owner(os.getpid())
        self.assertEqual(value['pid'],os.getpid())
        self.assertEqual(value['ppid'],os.getppid())
        self.assertEqual(value['argv'],Path('/proc/self/cmdline').read_bytes()[:-1].decode().split('\0'))
        self.assertEqual(value['cgroup'],Path('/proc/self/cgroup').read_text().strip())

    def test_dead_or_invalid_pid_rejected(self):
        for pid in (0,-1,True):
            with self.subTest(pid=pid),self.assertRaises(ValueError):handshake.owner(pid)
        with self.assertRaises(FileNotFoundError):handshake.owner(2147483647)


class ActualFixture(unittest.TestCase):
    def setUp(self):
        self.children=[]
        self.fds=[]
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for fd in self.fds:
            try:os.close(fd)
            except OSError:pass
        for child in self.children:
            if child.poll() is None:child.kill()
            child.communicate(timeout=3)
            self.assertIsNotNone(child.returncode)

    def spawn(self,mode='normal'):
        read,ready=os.pipe2(os.O_CLOEXEC)
        release,write=os.pipe2(os.O_CLOEXEC)
        argv=[sys.executable,'-I','-S','-B',str(Path(__file__).resolve()),
              '--fixture-child',str(ready),str(release),mode]
        child=subprocess.Popen(argv,pass_fds=(ready,release),stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,close_fds=True)
        self.children.append(child)
        os.close(ready);os.close(release)
        self.fds.extend((read,write))
        return child,read,write,argv

    def release(self,child,read,write,argv,guard=lambda:None,**changes):
        kwargs=dict(nonce=NONCE,role='helper-0',expected_argv=argv,
            expected_cgroup=handshake.owner(os.getpid())['cgroup'],guard=guard,
            deadline=time.monotonic()+3)
        kwargs.update(changes)
        try:return handshake.parent_release(child.pid,read,write,**kwargs)
        finally:
            self.fds.remove(read);self.fds.remove(write)

    def outcome(self,child):
        stdout,stderr=child.communicate(timeout=3)
        return child.returncode,stdout,stderr

    def test_actual_child_remains_waiting_until_parent_grant_and_then_exits(self):
        child,read,write,argv=self.spawn()
        self.assertTrue(select.select([read],[],[],2)[0])
        self.assertIsNone(child.poll())
        self.assertFalse(select.select([child.stdout],[],[],0)[0])
        guards=[]
        def guard():
            self.assertIsNone(child.poll());guards.append(time.monotonic_ns())
        proof=self.release(child,read,write,argv,guard)
        code,stdout,stderr=self.outcome(child)
        self.assertEqual((code,stderr),(0,b''))
        result=json.loads(stdout)
        self.assertTrue(result['generated_fixture_completed'])
        self.assertFalse(result['actual_SDK_helper_executed'])
        self.assertEqual(len(guards),2)
        self.assertEqual(proof['ready'],result['proof']['ready'])
        self.assertEqual(proof['grant'],result['proof']['grant'])
        self.assertEqual(proof['first_actual_capture'],proof['second_actual_capture'])
        self.assertLessEqual(proof['grant']['grant_ns'],result['proof']['child_release_ns'])
        self.assertFalse(proof['native_or_helper_executed'])
        self.assertFalse(proof['runtime_admitted'])

    def test_short_lived_child_repeated_capture_without_polling_race(self):
        for _ in range(8):
            child,read,write,argv=self.spawn()
            proof=self.release(child,read,write,argv)
            code,stdout,stderr=self.outcome(child)
            self.assertEqual((code,stderr),(0,b''))
            self.assertEqual(json.loads(stdout)['proof']['grant'],proof['grant'])

    def test_child_early_exit_cannot_get_release(self):
        child,read,write,argv=self.spawn('early-exit')
        with self.assertRaisesRegex(ValueError,'closed'):self.release(child,read,write,argv)
        code,stdout,_=self.outcome(child)
        self.assertEqual((code,stdout),(0,b''))

    def test_wrong_nonce_argv_or_cgroup_stops_child_before_fixture(self):
        for mode,change in (('wrong-nonce',{}),('normal',{'expected_argv':['wrong']}),
                            ('normal',{'expected_cgroup':'0::/wrong'})):
            child,read,write,argv=self.spawn(mode)
            with self.subTest(mode=mode,change=change),self.assertRaises(ValueError):
                self.release(child,read,write,argv,**change)
            code,stdout,_=self.outcome(child)
            self.assertNotEqual(code,0);self.assertEqual(stdout,b'')

    def test_guard_failure_at_either_checkpoint_prevents_release(self):
        for fail_at in (1,2):
            child,read,write,argv=self.spawn()
            count=0
            def guard():
                nonlocal count
                count+=1
                if count==fail_at:raise RuntimeError('Lost source or resource admission')
            with self.subTest(fail_at=fail_at),self.assertRaisesRegex(RuntimeError,'Lost'):
                self.release(child,read,write,argv,guard)
            code,stdout,_=self.outcome(child)
            self.assertNotEqual(code,0);self.assertEqual(stdout,b'')

    def test_child_rejects_forged_or_misbound_release(self):
        for field,value in (('nonce','2'*32),('ready_sha256','0'*64),('parent',{'pid':1})):
            child,read,write,argv=self.spawn()
            ready=handshake.receive(read,time.monotonic()+3)
            os.close(read);self.fds.remove(read)
            now=time.monotonic_ns()
            grant=dict(schema='nemotron4b_process_release_v1',nonce=NONCE,role='helper-0',
                owner=ready['owner'],parent=handshake.owner(os.getpid()),
                ready_sha256=hashlib.sha256(handshake.encode(ready)).hexdigest(),
                capture_ns=now,guard_complete_ns=now,grant_ns=now)
            grant[field]=value
            handshake.send(write,grant,time.monotonic()+3)
            os.close(write);self.fds.remove(write)
            code,stdout,_=self.outcome(child)
            self.assertNotEqual(code,0);self.assertEqual(stdout,b'')


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture-child':
        child_fixture()
    else:
        unittest.main(verbosity=2)
