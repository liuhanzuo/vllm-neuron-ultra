"""Post-freeze source/mock audit of v5 helper races and cleanup; no processes."""
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parent
sys.path.insert(0, str(Q))
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch','torch_xla','torch_neuronx','neuronxcc','numpy','vllm','transformers'}:
            raise AssertionError('No native import: ' + name)
sys.meta_path.insert(0, RejectNative())
import qwen35_package_native_preflight_v5 as w
import subprocess


class CleanupAudit(unittest.TestCase):
    def run_case(self, root, *, observation_error=False, exited=False, timeout=False,
                 term_wait_timeout=False, final_wait_timeout=False):
        spec = dict(work=str(root), binding={'cgroup':'/mock'})
        (root/'tmp').mkdir()
        helper = w.RequiredPjrtHelper(spec, {'pjrt_library_identity':w.library_identity()}, 'parent')
        events = []; clock = [0.]
        pipes = {key:SimpleNamespace(fileno=lambda fd=100+i:fd, close=lambda key=key:events.append(('close',key)))
                 for i,key in enumerate(('stdout','stderr'))}
        child = SimpleNamespace(pid=900, stdout=pipes['stdout'], stderr=pipes['stderr'], returncode=0 if exited else None)
        def wait(timeout):
            events.append(('wait',timeout))
            if final_wait_timeout or term_wait_timeout and child.returncode is None:
                raise subprocess.TimeoutExpired('mock-helper',timeout)
            child.returncode = child.returncode if child.returncode is not None else -15
            return child.returncode
        def terminate():
            events.append('terminate')
            if not term_wait_timeout: child.returncode=-15
        def kill(): events.append('kill'); child.returncode=-9
        child.poll=lambda:child.returncode;child.wait=wait;child.terminate=terminate;child.kill=kill
        owner=dict(pid=900,start_ticks=123,ppid=os.getpid(),argv=['/home/ubuntu/venvs/vllm_neuron/bin/python3',w.PJRT_HELPER],
            cgroup='0::/mock',executable=str(Path(w.PYTHON).resolve()))
        registered={}
        def register(pipe,event,key):registered[key]=SimpleNamespace(fileobj=pipe,data=key)
        selector=SimpleNamespace(register=register,get_map=lambda:registered,select=lambda *args:[],close=lambda:events.append('selector-close'))
        with patch.object(w.sys,'addaudithook') as hook:w.install_runtime_guard((),[],helper)
        guard=hook.call_args.args[0]
        def launch(argv,**kwargs):
            events.append('Popen')
            guard('subprocess.Popen',(argv[0],argv,kwargs['cwd'],kwargs['env']))
            return child
        def observe(pid):
            events.append('observe')
            if observation_error: raise FileNotFoundError('process disappeared before live identity capture')
            return owner
        def monotonic():
            value=clock[0]; clock[0]+=31 if timeout else 0
            return value
        with patch.object(subprocess,'Popen',side_effect=launch),patch.object(w,'helper_process',side_effect=observe),\
                patch.object(w.os,'sched_getaffinity',return_value={6,7}),patch.object(w,'check_binding',return_value={}),\
                patch.object(w.selectors,'DefaultSelector',return_value=selector),patch.object(w.time,'monotonic',side_effect=monotonic),\
                patch('builtins.print'):
            error=FileNotFoundError if observation_error else subprocess.TimeoutExpired if final_wait_timeout else RuntimeError
            if final_wait_timeout:error=subprocess.TimeoutExpired
            with self.assertRaises(error):helper.capture()
        self.assertFalse(helper.active)
        receipt=root/'parent-pjrt-helper.json'
        return events,json.loads(receipt.read_text()) if receipt.exists() else None

    def test_fast_exited_helper_identity_failure_is_waited_and_recorded_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            events,result=self.run_case(Path(directory),observation_error=True,exited=True)
            self.assertEqual([event for event in events if isinstance(event,tuple) and event[0]=='wait'],[('wait',1)])
            self.assertNotIn('terminate',events);self.assertNotIn('kill',events)
            self.assertEqual(result['actual_returncode'],0);self.assertEqual(result['failure']['type'],'FileNotFoundError')
            self.assertTrue(result['popen_wait_observed']);self.assertFalse(result['self_exit_claimed'])

    def test_observation_failure_terminates_then_waits_actual_child(self):
        with tempfile.TemporaryDirectory() as directory:
            events,result=self.run_case(Path(directory),observation_error=True)
            self.assertLess(events.index('terminate'),events.index(('wait',5)))
            self.assertNotIn('kill',events);self.assertEqual(result['actual_returncode'],-15)
            self.assertTrue(result['popen_wait_observed']);self.assertIsNotNone(result['failure'])

    def test_observation_failure_escalates_term_timeout_to_kill_and_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            events,result=self.run_case(Path(directory),observation_error=True,term_wait_timeout=True)
            self.assertEqual([item for item in events if item in ('terminate','kill') or isinstance(item,tuple) and item[0]=='wait'],
                ['terminate',('wait',5),'kill',('wait',5)])
            self.assertEqual(result['actual_returncode'],-9);self.assertTrue(result['popen_wait_observed'])
            self.assertTrue(result['whole_recursive_cgroup_drain_pending'])

    def test_runtime_timeout_escalates_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            events,result=self.run_case(Path(directory),timeout=True,term_wait_timeout=True)
            self.assertIn('kill',events);self.assertEqual(result['failure']['type'],'RuntimeError')
            self.assertEqual(result['actual_returncode'],-9);self.assertTrue(result['popen_wait_observed'])

    def test_even_kill_wait_failure_cannot_emit_success(self):
        with tempfile.TemporaryDirectory() as directory:
            events,result=self.run_case(Path(directory),observation_error=True,term_wait_timeout=True,final_wait_timeout=True)
            self.assertIn('kill',events);self.assertIsNone(result)
            # The exception exits native nonzero; original root retains lock until recursive service drain.

    def test_required_evidence_rejects_missing_failed_unwaited_wrong_sdk_environment(self):
        helper=w.RequiredPjrtHelper({'work':'/mock'}, {'pjrt_library_identity':{}}, 'parent')
        good=dict(failure=None,actual_returncode=0,popen_wait_observed=True)
        for events,environment in [([],{'PJRT_DEVICE':'NEURON','NEURON_LIBRARY_PATH':w.PJRT_LIBRARY}),
                ([dict(good,failure={})],{'PJRT_DEVICE':'NEURON','NEURON_LIBRARY_PATH':w.PJRT_LIBRARY}),
                ([dict(good,popen_wait_observed=False)],{'PJRT_DEVICE':'NEURON','NEURON_LIBRARY_PATH':w.PJRT_LIBRARY}),
                ([good],{'PJRT_DEVICE':'CPU','NEURON_LIBRARY_PATH':w.PJRT_LIBRARY}),
                ([good],{'PJRT_DEVICE':'NEURON','NEURON_LIBRARY_PATH':'/foreign.so'})]:
            helper.events=events
            with patch.dict(os.environ,environment,clear=True),self.assertRaises(RuntimeError):helper.evidence()
        helper.events=[good]
        with patch.dict(os.environ,{'PJRT_DEVICE':'NEURON','NEURON_LIBRARY_PATH':w.PJRT_LIBRARY},clear=True):
            self.assertEqual(helper.evidence(),good)


if __name__=='__main__':unittest.main(verbosity=2)
