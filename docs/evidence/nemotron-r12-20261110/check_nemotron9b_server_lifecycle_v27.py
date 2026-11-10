"""Regression controls execute actual HTTP lifecycle with mock kernel/HTTP only."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nemotron9b_package_protocol_v27 as p
from nemotron9b_package_startup_v27 import publish_prepared
import nemotron9b_package_worker_v27 as worker
import nemotron9b_package_server_lifecycle_v27 as life


class Harness:
    def __init__(self,root,*,crash=None,stop_code=-15,timeout=False,mutation=None,exec_missing=False,intent_crash=None,signal_failure=False):
        self.spec=dict(control_root=str(root),evidence_root=str(root),work_root=str(root))
        self.spec_ref=dict(path='spec',sha256='a');self.review_ref=dict(path='review',sha256='b')
        self.owner=dict(pid=1234,start_ticks='100',ppid=1,argv=['worker'],cgroup='0::/mock',invocation_id='i')
        self.child_owner=dict(pid=1235,start_ticks='101',ppid=1234,argv=['mock-worker','--server-child'],cgroup='0::/mock')
        self.admission=dict(controller=dict(pid=1200),execution_spec=self.spec_ref,source_review=self.review_ref)
        p.once(root/'admission.json',self.admission)
        self.exe=dict(realpath='/python',device=1,inode=2,size=3,mtime_ns=4,ctime_ns=5)
        self.intent_crash=intent_crash;self.signal_failure=signal_failure
        self.executed=False;self.quality_called=False;self.observations=[];self.signals=[];self.crash=crash
        self.stop_code=stop_code;self.timeout=timeout;self.mutation=mutation;self.exec_missing=exec_missing
        outer=self
        class Child:
            pid=1235
            returncode=None
            waits=0
            def poll(self):return self.returncode
            def wait(self,timeout):
                self.waits+=1
                if outer.timeout and self.waits==1:raise subprocess.TimeoutExpired('mock',timeout)
                self.returncode=-9 if outer.timeout else outer.stop_code
                return self.returncode
        self.child=Child()

    def proc(self,pid):
        return dict(self.child_owner,argv=['mock-server'])if self.executed else self.child_owner.copy()

    def observe(self,pid):
        process=self.proc(pid)
        if self.executed and self.mutation in ('pid','start_ticks','ppid','argv','cgroup'):
            process[self.mutation]={'pid':9999,'start_ticks':'999','ppid':99,'argv':['unexpected'],'cgroup':'0::/other'}[self.mutation]
        row=dict(process=process,pgid=1235,sid=1235,exe=self.exe.copy(),live=True)
        if self.executed and self.mutation in ('pgid','sid'):row[self.mutation]=999
        if self.executed and self.mutation=='exe':row['exe']['inode']=999
        self.observations.append(copy.deepcopy(row));return row

    def run(self):
        once=p.once;root=Path(self.spec['control_root'])
        def record(path,value):
            once(path,value)
            if Path(path).name=='quality-server-stop-intent.json'and self.intent_crash is not None:
                self.child.returncode=self.intent_crash
            if Path(path).name=='quality-server-ack.json':
                if not self.exec_missing:
                    once(root/'quality-server-exec.json',dict(original_owner=self.child_owner,argv=['mock-server'],ack=p.ref(path)))
                    publish_prepared(root/'quality-server-prepared')
                self.executed=True
        def send_signal(fd,sig,info,flags):
            if self.signal_failure:raise ProcessLookupError('mock child vanished')
            self.signals.append((fd,sig))
        def quality(spec):
            self.quality_called=True
            if self.crash is not None:self.child.returncode=self.crash
            if self.mutation=='after-response-start':self.child_owner=dict(self.child_owner,start_ticks='bad')
            return dict(complete=True,passed=True)
        def guard():life.controller_observe(self.spec,'quality',self.owner)
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*args):return False
        class Socket(Response):
            def connect_ex(self,address):return 111
        with ExitStack()as stack:
            for ctx in (
                patch.object(p,'argv',return_value=['mock-worker']),patch.object(p,'server_argv',return_value=['mock-server']),
                patch.object(p,'group',return_value='/mock'),patch.object(p.low,'proc',side_effect=self.proc),
                patch.object(p.low,'gone'),patch.object(p,'file_stat',return_value=self.exe),
                patch.object(p,'once',side_effect=record),patch.object(life,'observe',side_effect=self.observe),
                patch.object(worker.subprocess,'Popen',return_value=self.child),patch.object(worker.socket,'socket',return_value=Socket()),
                patch.object(worker.urllib.request,'urlopen',return_value=Response()),patch.object(worker,'quality',side_effect=quality),
                patch.object(worker.os,'pidfd_open',return_value=77),patch.object(worker.os,'close'),
                patch.object(life.signal,'pidfd_send_signal',side_effect=send_signal),
            ):stack.enter_context(ctx)
            return worker.http_stage(self.spec,'quality',self.owner,guard,self.spec_ref,self.review_ref)

    def check(self,result):
        with patch.object(p,'argv',return_value=['mock-worker']),patch.object(p,'server_argv',return_value=['mock-server']), \
             patch.object(p,'group',return_value='/mock'),patch.object(p,'file_stat',return_value=self.exe),patch.object(p.low,'gone'):
            return life.validate_chain(self.spec,'quality',self.owner,result,self.admission)


class ActualHttpRegression(unittest.TestCase):
    def test_root_nq1_crash_after_last_response_rejected_and_original_status_retained(self):
        for code in (-11,17,0,-15):
            with self.subTest(code=code),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp),crash=code)
                with self.assertRaises(RuntimeError):h.run()
                row=p.read(Path(tmp)/'quality-server-exit.json')
                self.assertEqual(row['returncode'],code);self.assertEqual(row['pre_stop_returncode'],code)
                self.assertFalse(row['qualification_passed']);self.assertFalse(row['signal_sent'])
                self.assertIsNone(row['stop_intent']);self.assertEqual(h.signals,[])
                self.assertTrue(h.quality_called);self.assertGreater(len(h.observations),1)

    def test_actual_success_requires_observed_exec_and_explicit_stop(self):
        for code in (0,-15):
            with self.subTest(code=code),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp),stop_code=code);result=h.run();h.check(result)
                self.assertTrue(result['passed']);self.assertEqual(h.signals,[(77,signal.SIGTERM)])
                row=p.read(Path(tmp)/'quality-server-exit.json');self.assertTrue(row['qualification_passed'])
                self.assertIsNone(row['pre_stop_returncode']);self.assertEqual(row['returncode'],code)
                self.assertGreaterEqual(len(h.observations),6)

    def test_explicit_signal_does_not_make_abnormal_exit_successful(self):
        for code in (-11,17,-9):
            with self.subTest(code=code),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp),stop_code=code)
                with self.assertRaises(RuntimeError):h.run()
                row=p.read(Path(tmp)/'quality-server-exit.json')
                self.assertTrue(row['signal_sent']);self.assertFalse(row['qualification_passed']);self.assertEqual(row['returncode'],code)

    def test_shutdown_timeout_and_kill_are_failed_even_after_all_requests(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp),timeout=True)
            with self.assertRaises(RuntimeError):h.run()
            row=p.read(Path(tmp)/'quality-server-exit.json')
            self.assertTrue(row['workload_complete']);self.assertTrue(row['escalated']);self.assertFalse(row['qualification_passed'])
            self.assertEqual(row['returncode'],-9);self.assertEqual(h.signals,[(77,signal.SIGTERM),(77,signal.SIGKILL)])

    def test_wrong_actual_post_exec_identity_cannot_send_requests_or_signal_replacement(self):
        for field in ('pid','start_ticks','ppid','argv','cgroup','pgid','sid','exe'):
            with self.subTest(field=field),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp),mutation=field)
                with self.assertRaises((AssertionError,RuntimeError)):h.run()
                self.assertFalse(h.quality_called);self.assertEqual(h.signals,[])
                self.assertFalse(p.read(Path(tmp)/'quality-server-exit.json')['qualification_passed'])

    def test_preexec_intent_missing_is_not_substituted_by_http_health(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp),exec_missing=True)
            with self.assertRaises((RuntimeError,FileNotFoundError)):h.run()
            self.assertFalse(h.quality_called)
            self.assertFalse(p.read(Path(tmp)/'quality-server-exit.json')['qualification_passed'])

    def test_post_response_identity_change_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp),mutation='after-response-start')
            with self.assertRaises((AssertionError,RuntimeError)):h.run()
            self.assertTrue(h.quality_called);self.assertEqual(h.signals,[])

    def test_terminal_receipt_tampering_and_false_flag_rejected(self):
        mutations=(('exit','returncode',17),('exit','returncode',True),('exit','pre_stop_returncode',0),
            ('exit','signal_sent',False),('exit','escalated',True),('exit','qualification_passed',False),
            ('exit','cleanup_only',True),('stop-intent','pre_stop_returncode',17),
            ('stop-intent','reason','cleanup'),('observed','observer','worker'))
        for name,key,value in mutations:
            with self.subTest(receipt=name,key=key),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp));result=h.run();path=Path(tmp)/('quality-server-'+name+'.json')
                row=p.read(path);row[key]=value;p.save(path,row)
                with self.assertRaises(RuntimeError):h.check(result)
                # Even replacing the result digest cannot turn false evidence
                # into success: all downstream cross-references remain bound.
                result['server_lifecycle'][name]=p.ref(path)
                with self.assertRaises(RuntimeError):h.check(result)

    def test_missing_postexec_observation_rejected_at_terminal(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp));result=h.run();result['server_lifecycle'].pop('post-exec')
            with self.assertRaises(RuntimeError):h.check(result)

    def test_exit_between_durable_intent_and_signal_is_not_explicit_stop(self):
        for code in (0,-11,17):
            with self.subTest(code=code),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp),intent_crash=code)
                with self.assertRaises(RuntimeError):h.run()
                row=p.read(Path(tmp)/'quality-server-exit.json')
                self.assertEqual(row['pre_stop_returncode'],code);self.assertEqual(row['returncode'],code)
                self.assertIsNotNone(row['stop_intent']);self.assertFalse(row['signal_sent'])
                self.assertFalse(row['qualification_passed']);self.assertEqual(h.signals,[])

    def test_pidfd_signal_failure_cannot_claim_explicit_stop(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp),signal_failure=True)
            with self.assertRaises(RuntimeError):h.run()
            row=p.read(Path(tmp)/'quality-server-exit.json')
            self.assertFalse(row['signal_sent']);self.assertFalse(row['qualification_passed'])
            self.assertIsNotNone(row['signal_error']);self.assertIsNone(row['returncode'])

    def test_controller_checks_actual_identity_before_ack(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp));result=h.run()
            c=Path(tmp);actual=p.read(c/'quality-server-post-exec.json')['observation']
            bad=copy.deepcopy(actual);bad['process']['start_ticks']='reused'
            # Pretend ACK is absent without deleting any evidence.
            real_exists=Path.exists
            def exists(path):return False if path.name=='quality-server-observed.json'else real_exists(path)
            with patch.object(Path,'exists',exists),patch.object(life,'observe',return_value=bad), \
                 patch.object(p,'group',return_value='/mock'),patch.object(p,'server_argv',return_value=['mock-server']), \
                 patch.object(p,'file_stat',return_value=h.exe),patch.object(p,'once')as once:
                with self.assertRaises(RuntimeError):life.controller_observe(h.spec,'quality',h.owner)
                once.assert_not_called()

    def test_missing_preparation_marker_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp));result=h.run()
            (Path(tmp)/'quality-server-prepared').rmdir()
            with self.assertRaises(RuntimeError):h.check(result)

    def test_missing_preparation_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            h=Harness(Path(tmp));result=h.run()
            result['server_lifecycle'].pop('preparation')
            with self.assertRaises(RuntimeError):h.check(result)

    def test_preparation_identity_deadline_or_false_status_rejected(self):
        for key,value in [('owner',{}),('status','merely_exists'),('budget_seconds',600.),
                          ('elapsed_seconds',300.),('elapsed_seconds',True),('elapsed_seconds',-1.),
                          ('full_model_qualified',True),('last',dict(pid=9,argv=[]))]:
            with self.subTest(key=key),tempfile.TemporaryDirectory()as tmp:
                h=Harness(Path(tmp));result=h.run();path=Path(tmp)/'quality-server-preparation.json'
                row=p.read(path);row[key]=value;p.save(path,row)
                result['server_lifecycle']['preparation']=p.ref(path)
                with self.assertRaises(RuntimeError):h.check(result)

    def test_no_native_libraries_imported(self):
        self.assertFalse({'torch','numpy','transformers','vllm','neuronxcc'}&set(sys.modules))


if __name__=='__main__':
    p.quiet();unittest.main()
