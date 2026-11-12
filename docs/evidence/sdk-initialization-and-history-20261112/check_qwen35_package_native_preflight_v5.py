# SPDX-License-Identifier: Apache-2.0
"""Source/mock author checks; rejects native imports and real process/device work."""
from __future__ import annotations
import ast
import builtins
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / 'qwen35_package_native_preflight_v5.py'
REPO = Path('/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107')


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'torch_xla', 'torch_neuronx', 'neuronxcc', 'neuronx_distributed',
                'neuronx_distributed_inference', 'vllm', 'transformers', 'numpy'}:
            raise AssertionError('Real native import forbidden: ' + name)
        return None


sys.meta_path.insert(0, RejectNative())
spec = importlib.util.spec_from_file_location('qwen_preflight_source', WORKER)
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
spec = importlib.util.spec_from_file_location('qwen_tokenizer_source', ROOT / 'qwen35_package_tokenizer_checks_v3.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
spec = importlib.util.spec_from_file_location('qwen_preflight_prepare', ROOT / 'prepare_qwen35_package_native_preflight_v5.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
sys.path.insert(0, str(REPO))
from vllm_neuron.models.qwen35 import identity, prepare_view
from vllm_neuron.models.qwen35.provenance import EXPECTED_MODEL, ORIGINAL_CONFIG


def fake_spec():
    return dict(schema=w.SCHEMA, bundle={'path': '/source.json', 'bytes': 1, 'sha256': '0'*64},
        work='/tmp/source-only-native-preflight-mock', binding=dict(owner={'pid': 42, 'start_ticks': 77},
        lock={'path': '/tmp/original.lock', 'device': 1, 'inode': 5}, history=[{'path': '/history.json', 'bytes': 2, 'sha256': '0'*64}], cgroup='/mock'))


def normalized_body():
    body = {k: copy.deepcopy(v) for k, v in EXPECTED_MODEL.items() if k not in ('dtype', 'torch_dtype')}
    neuron = dict(w.NEURON_KWARGS, max_batch_size=4, tkg_batch_size=4, max_length=4096,
        buckets=[4096], save_sharded_checkpoint=False, skip_sharding=False, skip_warmup=False,
        logical_nc_config=1, start_rank_id=0, local_ranks_size=2, ep_degree=1, world_size=2,
        quantize_clamp_bound=float('inf'))
    return dict(body, neuron_config=neuron, metadata=None)


class PureBoundaryTests(unittest.TestCase):
    def test_import_is_stdlib_only_and_inert(self):
        tree = ast.parse(WORKER.read_text())
        for n in tree.body:
            if isinstance(n, ast.Import):
                self.assertFalse(any(x.name.split('.')[0] in w.NATIVE_ROOTS for x in n.names))
            if isinstance(n, ast.ImportFrom): self.assertNotIn((n.module or '').split('.')[0], w.NATIVE_ROOTS)
        self.assertFalse(any(name == x or name.startswith(x + '.') for name in sys.modules for x in w.NATIVE_ROOTS))

    def test_no_execute_prints_only_schema(self):
        with patch.object(sys, 'argv', ['preflight']), patch('builtins.print') as output, patch.object(w, 'execute', side_effect=AssertionError('native')), patch.object(w, 'read_small', side_effect=AssertionError('IO')):
            w.main()
        self.assertEqual(json.loads(output.call_args.args[0])['schema'], w.SCHEMA)

    def test_explicit_hash_is_checked_before_execute(self):
        with patch.object(sys, 'argv', ['preflight', '--execute-native', '--input', '/input.json', '--input-sha256', 'wrong']), patch.object(w, 'read_small', return_value=b'{}'), patch.object(w, 'execute') as execute:
            with self.assertRaises(RuntimeError): w.main()
            execute.assert_not_called()

    def test_schema_rejects_missing_history_owner_and_reused_shape(self):
        w.validate_spec(fake_spec())
        for replacement in (dict(history=[]), dict(owner={'pid': True, 'start_ticks': 77}), dict(cgroup='/../foreign')):
            value = fake_spec(); value['binding'].update(replacement)
            with self.assertRaises(RuntimeError): w.validate_spec(value)
        value = fake_spec(); value['extra_permission'] = True
        with self.assertRaises(RuntimeError): w.validate_spec(value)

    def test_environment_is_offline_bounded_and_removes_model_flags(self):
        result = w.environment('/work', {'PATH': '/bin', 'NEURON_LFM3': '1', 'NEURON_QWEN35_ALLOW_COMPILE': '1', 'PYTHONPATH': '/foreign'})
        self.assertNotIn('NEURON_LFM3', result); self.assertNotIn('NEURON_QWEN35_ALLOW_COMPILE', result)
        self.assertNotIn('PYTHONPATH', result)
        self.assertEqual(result['OMP_NUM_THREADS'], '1'); self.assertEqual(result['HF_HUB_OFFLINE'], '1')
        self.assertEqual(result['NEURON_CPU_SAMPLER_FAST'], '0')
        with self.assertRaises(RuntimeError): w.environment('/work', {'TRN1_QWEN35_TEXT_ADAPTER': '1'})

    def test_native_requires_external_binding_before_import_or_mkdir(self):
        spec = fake_spec()
        with patch.object(w, 'checked', return_value=b'{}'), patch.object(w, 'validate_bundle', return_value={}), patch.object(w, 'check_binding', side_effect=RuntimeError('lock stale')), patch.object(w.Path, 'mkdir') as mkdir, patch.object(w, 'native_checks') as native:
            with self.assertRaisesRegex(RuntimeError, 'lock stale'): w.execute(spec)
            mkdir.assert_not_called(); native.assert_not_called()

    def test_lock_is_never_acquired_released_or_replaced(self):
        tree = ast.parse(WORKER.read_text())
        calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for forbidden in ('fcntl.flock', 'fcntl.lockf', 'subprocess.run', 'os.system', 'shutil.rmtree'):
            self.assertNotIn(forbidden, calls)
        self.assertIn("Path('/proc/locks').read_text()", WORKER.read_text())

    def test_existing_lock_and_resource_observations_are_required(self):
        value = fake_spec()
        def read_text(path, *args, **kwargs):
            name = str(path)
            values = {'/proc/locks': '1: FLOCK ADVISORY WRITE 42 00:01:5 0 EOF\n',
                '/proc/self/cgroup': '0::/mock\n', '/sys/fs/cgroup/mock/memory.max': str(12 * 2**30),
                '/sys/fs/cgroup/mock/memory.swap.max': '0', '/sys/fs/cgroup/mock/pids.max': '256'}
            return values[name]
        status = SimpleNamespace(st_dev=1, st_ino=5, st_mode=0o100600)
        with patch.object(w, 'process_start', return_value=77), patch.object(w.Path, 'stat', return_value=status), patch.object(w.Path, 'resolve', lambda path: path), patch.object(w.Path, 'read_text', read_text), patch.object(w, 'checked', return_value=b'{}'), patch.object(w.os, 'sched_getaffinity', return_value={0, 1}):
            report = w.check_binding(value)
            self.assertEqual(report['owner'], value['binding']['owner'])
            with patch.object(w, 'process_start', return_value=78):
                with self.assertRaisesRegex(RuntimeError, 'no longer live'): w.check_binding(value)
            with patch.object(w.Path, 'read_text', return_value=''):
                with self.assertRaisesRegex(RuntimeError, 'kernel FLOCK'): w.check_binding(value)
            with patch.object(w.os, 'sched_getaffinity', return_value=set(range(9))):
                with self.assertRaisesRegex(RuntimeError, 'CPU affinity'): w.check_binding(value)

    def test_input_reads_refuse_native_or_model_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for extension in ('.so', '.pt', '.bin', '.safetensors', '.neff', '.whl'):
                path = root / ('fake' + extension)
                with self.assertRaises(RuntimeError): w.read_small(path)
            path = root / 'text.json'; path.write_text('{}')
            self.assertEqual(w.checked({'path': str(path), 'bytes': 2, 'sha256': hashlib.sha256(b'{}').hexdigest()}), b'{}')
            with self.assertRaises(RuntimeError): w.checked({'path': str(path), 'bytes': 3, 'sha256': '0'*64})

    def test_fresh_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'result.json'; w.once(path, {'passed': True})
            with self.assertRaises(FileExistsError): w.once(path, {'passed': False})
            self.assertEqual(json.loads(path.read_text()), {'passed': True})


class OptionalProbeTests(unittest.TestCase):
    def probe_function(self):
        # Compile only the exact pinned stdlib function, not NumPy or native imports.
        payload=w.read_small(w.NUMPY_SVE_SOURCE); self.assertEqual(w.sha(payload),w.NUMPY_SVE_SHA256)
        node=next(node for node in ast.parse(payload).body if isinstance(node,ast.FunctionDef) and node.name=='check_support_sve')
        namespace={'__name__':'numpy.testing._private.utils','__file__':w.NUMPY_SVE_SOURCE}
        exec(compile(ast.Module(body=[node],type_ignores=[]),w.NUMPY_SVE_SOURCE,'exec'),namespace)
        return namespace

    def guarded_call(self,namespace,args=('lscpu',['lscpu'],None,None),machine='x86_64',events=None):
        import subprocess
        events=[] if events is None else events
        with patch.object(w.sys,'addaudithook') as hook:
            w.install_runtime_guard((),events)
        guard=hook.call_args.args[0]
        def run(command,**kwargs):
            self.assertEqual(command,'lscpu'); self.assertEqual(kwargs,dict(capture_output=True,text=True))
            guard('subprocess.Popen',args)
            raise AssertionError('No command may actually run')
        with patch.object(subprocess,'run',side_effect=run),patch.object(w.os,'uname',return_value=SimpleNamespace(machine=machine)):
            value=namespace['check_support_sve']()
        namespace['_SUPPORTS_SVE']=value
        return value,events

    def test_original_numpy_oserror_fallback_records_denied_exec(self):
        namespace=self.probe_function(); value,events=self.guarded_call(namespace)
        self.assertIs(value,False); self.assertEqual(len(events),1); self.assertFalse(events[0]['subprocess_started'])
        with patch.dict(sys.modules,{'numpy.testing._private.utils':SimpleNamespace(**namespace)}):
            evidence=w.optional_probe_evidence(events)
            self.assertTrue(evidence['original_numpy_fallback_observed']); self.assertFalse(evidence['fabricated_command_output'])

    def test_real_stdlib_popen_audit_arguments_are_denied_before_fork(self):
        import subprocess
        namespace=self.probe_function(); events=[]
        with patch.object(w.sys,'addaudithook') as hook:w.install_runtime_guard((),events)
        guard=hook.call_args.args[0]
        with patch.object(subprocess.sys,'audit',side_effect=lambda event,*args:guard(event,args)),\
                patch.object(subprocess._posixsubprocess,'fork_exec',side_effect=AssertionError('No process may launch')) as fork,\
                patch.object(subprocess.Popen,'_posix_spawn',side_effect=AssertionError('No process may launch')) as spawn:
            self.assertIs(namespace['check_support_sve'](),False)
        fork.assert_not_called(); spawn.assert_not_called(); self.assertEqual(len(events),1)

    def test_other_architecture_commands_and_arguments_stay_hard_denied(self):
        for args,machine in [(('lscpu',['lscpu'],None,None),'aarch64'),(('/usr/bin/lscpu',['lscpu'],None,None),'x86_64'),
                (('lscpu',['lscpu','-J'],None,None),'x86_64'),(('lscpu',['lscpu'],'/tmp',None),'x86_64'),
                (('lscpu',['lscpu'],None,{}),'x86_64'),(('sh',['sh','-c','lscpu'],None,None),'x86_64')]:
            with self.subTest(args=args,machine=machine),self.assertRaisesRegex(RuntimeError,'Subprocess/network forbidden'):
                self.guarded_call(self.probe_function(),args,machine)

    def test_same_command_outside_exact_function_is_still_hard_denied(self):
        events=[]
        with patch.object(w.sys,'addaudithook') as hook:w.install_runtime_guard((),events)
        for event,args in [('subprocess.Popen',('lscpu',['lscpu'],None,None)),('os.system',('lscpu',)),('socket.connect',('',))]:
            with self.subTest(event=event),self.assertRaisesRegex(RuntimeError,'Subprocess/network forbidden'):hook.call_args.args[0](event,args)
        self.assertEqual(events,[])

    def test_source_module_function_and_repetition_cannot_widen_fallback(self):
        for key,value in [('__name__','foreign'),('__file__','/foreign.py')]:
            namespace=self.probe_function(); namespace[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):self.guarded_call(namespace)
        namespace=self.probe_function()
        with patch.object(w,'read_small',return_value=b'changed'),self.assertRaisesRegex(RuntimeError,'source changed'):
            self.guarded_call(namespace)
        with self.assertRaisesRegex(RuntimeError,'only once'):self.guarded_call(self.probe_function(),events=[{'denied':True}])

    def test_result_requires_observed_false_fallback(self):
        for module,events in [(None,[]),(SimpleNamespace(__file__=w.NUMPY_SVE_SOURCE,_SUPPORTS_SVE=True),[{}]),
                (SimpleNamespace(__file__=w.NUMPY_SVE_SOURCE,_SUPPORTS_SVE=False),[])]:
            with patch.dict(sys.modules,{'numpy.testing._private.utils':module}),self.assertRaises(RuntimeError):w.optional_probe_evidence(events)

    def test_no_other_worker_behavior_changed(self):
        old=ast.parse((ROOT/'qwen35_package_native_preflight_v3.py').read_text()); new=ast.parse(WORKER.read_text())
        old_functions={node.name:ast.dump(node) for node in old.body if isinstance(node,ast.FunctionDef)}
        new_functions={node.name:ast.dump(node) for node in new.body if isinstance(node,ast.FunctionDef)}
        for name in old_functions.keys()-{'validate_bundle','install_runtime_guard','native_checks','execute'}:
            self.assertEqual(old_functions[name],new_functions[name],name)
        self.assertEqual(w.LIMITS,dict(memory_max=12*2**30,memory_swap_max=0,pids_max=256,cpu_count=8,wall_seconds=600,cpu_seconds=480,file_bytes=16*2**20))


class RequiredHelperTests(unittest.TestCase):
    def initializer(self):
        import subprocess
        node=next(node for node in ast.parse(w.read_small(w.PJRT_INITIALIZER)).body if isinstance(node,ast.ClassDef) and node.name=='Initializer')
        namespace=dict(__name__='torch_neuronx.initialization.initializer',__file__=w.PJRT_INITIALIZER,
            os=os,logger=SimpleNamespace(info=Mock(),warning=Mock()),subprocess=subprocess,
            _remove_unused_gpu_flags=lambda:None,_disable_functionalization=lambda:None,
            libneuronxla=SimpleNamespace(configure_environment=lambda:None),_patch_zero1_redundancy_optimizer=lambda:None,
            set_rt_root_comm_id=lambda:None,is_torchelastic_launched=lambda:False)
        exec(compile(ast.Module(body=[node],type_ignores=[]),w.PJRT_INITIALIZER,'exec'),namespace)
        return namespace

    def exercise(self,root,code=0,stdout=None,stderr=b'',owner_change=None,late=False):
        import subprocess
        spec=fake_spec(); spec['work']=str(root); (root/'tmp').mkdir()
        bundle=dict(pjrt_library_identity=w.library_identity()); helper=w.RequiredPjrtHelper(spec,bundle,'parent')
        namespace=self.initializer(); buffers={'stdout':(w.PJRT_LIBRARY+'\n').encode() if stdout is None else stdout,'stderr':stderr}
        pipes={name:SimpleNamespace(fileno=lambda fd=100+i:fd,close=Mock()) for i,name in enumerate(buffers)}
        child=SimpleNamespace(pid=333,stdout=pipes['stdout'],stderr=pipes['stderr'],returncode=code,
            poll=lambda:code,wait=Mock(return_value=code),terminate=Mock(),kill=Mock())
        owner=dict(pid=333,start_ticks=4,ppid=os.getpid(),argv=['/home/ubuntu/venvs/vllm_neuron/bin/python3',w.PJRT_HELPER],cgroup='0::/mock',executable=str(Path(w.PYTHON).resolve()))
        if owner_change:owner.update(owner_change)
        registered={}; data={100:iter([buffers['stdout'],b'']),101:iter([buffers['stderr'],b''])}
        def register(pipe,event,name):registered[pipe.fileno()]=SimpleNamespace(fileobj=pipe,data=name)
        selector=SimpleNamespace(register=register,get_map=lambda:registered,
            select=lambda *a:[(value,None) for value in list(registered.values())],unregister=lambda pipe:registered.pop(pipe.fileno()),close=Mock())
        original_read=os.read
        def read(fd,size):return next(data[fd]) if fd in data else original_read(fd,size)
        with patch.object(w.sys,'addaudithook') as hook:w.install_runtime_guard((),[],helper)
        guard=hook.call_args.args[0]
        def popen(command,**kwargs):
            self.assertEqual(command,[w.PJRT_HELPER]);self.assertEqual(kwargs['env'],w.helper_environment(str(root)))
            self.assertTrue(kwargs['close_fds']);self.assertEqual(kwargs['cwd'],str(root/'tmp'))
            guard('subprocess.Popen',(command[0],command,kwargs['cwd'],kwargs['env']))
            return child
        env=w.environment(str(root),dict(PATH=w.PJRT_PATH))
        with patch.dict(os.environ,env,clear=True),patch.object(subprocess,'run',helper.run),\
                patch.object(subprocess,'Popen',side_effect=popen) as launch,patch.object(w,'check_binding',return_value={}),\
                patch.object(w,'helper_process',return_value=owner),patch.object(w.os,'sched_getaffinity',return_value={6,7}),\
                patch.object(w.selectors,'DefaultSelector',return_value=selector),patch.object(w.os,'read',side_effect=read),\
                patch.object(w.time,'monotonic',side_effect=[0.,31.,32.] if late else None,return_value=1.),patch('builtins.print'):
            if code or stdout is not None or stderr or owner_change or late:
                with self.assertRaises(RuntimeError):namespace['Initializer']()
            else:
                namespace['Initializer']();self.assertEqual(os.environ['NEURON_LIBRARY_PATH'],w.PJRT_LIBRARY)
                self.assertEqual(helper.evidence()['actual_returncode'],0)
                with self.assertRaisesRegex(RuntimeError,'No repeated'):namespace['Initializer']()
        launch.assert_called_once(); child.wait.assert_called_once();self.assertFalse(helper.active)
        receipt=json.loads((root/'parent-pjrt-helper.json').read_text())
        self.assertEqual(receipt['actual_returncode'],code);self.assertTrue(receipt['popen_wait_observed'])
        self.assertFalse(receipt['fabricated_command_output']);self.assertTrue(receipt['whole_recursive_cgroup_drain_pending'])
        return receipt

    def test_exact_original_initializer_calls_real_capture_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt=self.exercise(Path(directory));self.assertIsNone(receipt['failure'])

    def test_failure_output_cgroup_argv_and_timeout_fail_without_retry(self):
        cases=[dict(code=7),dict(stdout=b'/fake/lib.so\n'),dict(stdout=(w.PJRT_LIBRARY+'\nextra').encode()),dict(stderr=b'error'),
            dict(owner_change={'cgroup':'0::/foreign'}),dict(owner_change={'argv':['foreign']}),dict(owner_change={'executable':'/foreign/python'}),dict(late=True)]
        for index,case in enumerate(cases):
            with tempfile.TemporaryDirectory() as directory,self.subTest(case=index):
                self.assertIsNotNone(self.exercise(Path(directory),**case)['failure'])

    def test_helper_allowance_rejects_general_subprocess_even_same_command(self):
        helper=w.RequiredPjrtHelper(fake_spec(),{'pjrt_library_identity':{}},'parent')
        helper.active=True;helper.thread=w.threading.get_ident()
        args=(w.PJRT_HELPER,[w.PJRT_HELPER],str(Path(fake_spec()['work'])/'tmp'),helper.env)
        self.assertFalse(helper.permits_audit(args,sys._getframe()))
        with patch.object(w.sys,'addaudithook') as hook:w.install_runtime_guard((),[],helper)
        with self.assertRaisesRegex(RuntimeError,'Subprocess/network forbidden'):hook.call_args.args[0]('subprocess.Popen',args)
        helper.active=False
        self.assertFalse(helper.permits_audit(args,sys._getframe()))

    def test_exact_caller_kwargs_path_and_source_required(self):
        import subprocess
        helper=w.RequiredPjrtHelper(fake_spec(),{'pjrt_library_identity':{}},'parent');namespace=self.initializer()
        with patch.object(subprocess,'run',helper.run),patch.dict(os.environ,dict(PATH='/foreign'),clear=True),\
                patch.object(helper,'capture',side_effect=AssertionError('No launch')):
            with self.assertRaisesRegex(RuntimeError,'fixed PATH'):namespace['Initializer']()
        source=w.read_small(w.PJRT_INITIALIZER).decode().replace('stdout=subprocess.PIPE','stdout=subprocess.PIPE, shell=True')
        tree=ast.parse(source);node=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Initializer')
        exec(compile(ast.Module(body=[node],type_ignores=[]),w.PJRT_INITIALIZER,'exec'),namespace)
        with patch.object(subprocess,'run',helper.run),patch.object(helper,'capture',side_effect=AssertionError('No launch')):
            with self.assertRaisesRegex(RuntimeError,'Exact pinned SDK helper'):namespace['Initializer']()
        other=Mock(side_effect=RuntimeError('other process forbidden'));helper.original_run=other
        with self.assertRaisesRegex(RuntimeError,'other process'):helper.run(['sh','-c','bad'])
        other.assert_called_once()

    def test_clean_spawn_environment_is_restored_even_if_start_fails(self):
        clean={'PATH':w.PJRT_PATH,'NEURON_QWEN35':'1'}
        initialized=dict(clean,PJRT_DEVICE='NEURON',NEURON_LIBRARY_PATH=w.PJRT_LIBRARY,NEURON_RT_ROOT_COMM_ID='localhost:62182')
        for failed in (False,True):
            observed=[]
            def start():
                observed.append(dict(os.environ))
                if failed:raise RuntimeError('spawn failed')
            with patch.dict(os.environ,initialized,clear=True):
                if failed:
                    with self.assertRaises(RuntimeError):w.start_fresh_spawn(SimpleNamespace(start=start),clean)
                else:w.start_fresh_spawn(SimpleNamespace(start=start),clean)
                self.assertEqual(dict(os.environ),initialized)
            self.assertEqual(observed,[clean])

    def test_original_helper_function_returns_exact_actual_library_location(self):
        path=Path(w.SITE_PACKAGES)/'libneuronxla/libneuronpjrt_path.py'
        tree=ast.parse(w.read_small(path)); body=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
        namespace=dict(os=os,__file__=str(path))
        exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
        with patch('builtins.print') as output:namespace['main']()
        output.assert_called_once_with(w.PJRT_LIBRARY)
        self.assertEqual(namespace['libneuronpjrt_path'](),w.library_identity()['path'])

    def test_missing_chat_patch_is_original_warning_and_does_not_raise(self):
        source=REPO/'vllm_neuron/platform.py';tree=ast.parse(source.read_text())
        klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='NeuronPlatform')
        method=copy.deepcopy(next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='_apply_chat_completion_stream_override'))
        method.decorator_list=[];namespace=dict(logger=SimpleNamespace(warning=Mock(),error=Mock(),debug=Mock(),info=Mock()))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
        original_import=builtins.__import__
        def import_stub(name,*args,**kwargs):
            if name=='vllm.entrypoints.openai.serving_chat':return SimpleNamespace(OpenAIServingChat=object)
            if name=='vllm_neuron.patches.chat_completion_stream_v0_11_0':raise ImportError('No module named vllm_neuron.patches')
            return original_import(name,*args,**kwargs)
        with patch.object(builtins,'__import__',side_effect=import_stub):
            self.assertIsNone(namespace['_apply_chat_completion_stream_override'](SimpleNamespace(_get_vllm_version=lambda:'0.11.0')))
        namespace['logger'].warning.assert_called_once();namespace['logger'].error.assert_not_called()

    def test_frozen_v4_unchanged_and_only_required_helper_integration_changed(self):
        frozen=json.loads((ROOT/'qwen35-package-native-preflight-source-bundle-20261112-v4.json').read_text())
        for row in frozen['worker_sources']:self.assertEqual(p.reference(Path(row['path'])),row)
        old={node.name:ast.dump(node) for node in ast.parse((ROOT/'qwen35_package_native_preflight_v4.py').read_text()).body if isinstance(node,ast.FunctionDef)}
        new={node.name:ast.dump(node) for node in ast.parse(WORKER.read_text()).body if isinstance(node,ast.FunctionDef)}
        for name in old.keys()-{'validate_bundle','install_runtime_guard','native_checks','execute','optional_probe_evidence'}:
            self.assertEqual(old[name],new[name],name)


class ConfigAndNativeSourceTests(unittest.TestCase):
    def test_actual_package_scope_rejects_all_nine_mutations(self):
        body = normalized_body(); identity.validate_scope(body, body['neuron_config'])
        self.assertEqual(len(w.negative_scope_checks(identity, body)), 9)
        self.assertEqual(body['neuron_config']['tp_degree'], 2)

    def test_config_fixture_authenticates_exact_official_transform(self):
        actual = prepare_view.compatible_model_config(copy.deepcopy(ORIGINAL_CONFIG))
        self.assertEqual(actual, EXPECTED_MODEL)
        self.assertEqual(hashlib.sha256(json.dumps(actual, indent=2).encode()).hexdigest(), identity.CONFIG_VIEW_SHA256)

    def test_real_config_roundtrip_path_uses_whole_body_not_subset(self):
        source = ast.get_source_segment(WORKER.read_text(), next(n for n in ast.parse(WORKER.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'config_checks'))
        for call in ('load_pretrained_config(str(config_dir))', 'config.to_json_string()', 'config.save(root_path)',
                     'Qwen35InferenceConfig.load(root_path)', 'Qwen35InferenceConfig.from_json_string(serialized)',
                     'artifacts.admit_saved_configuration(bound_path, binding)'):
            self.assertIn(call, source)
        self.assertNotIn('build_identity(', source); self.assertNotIn('load_or_compile(', source)

    def test_no_model_constructor_compile_or_weight_loader_calls(self):
        tree = ast.parse(WORKER.read_text())
        names = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
        for name in names:
            self.assertFalse(name.endswith(('.compile', '.load_weights', '.load_state_dict', '.verify_view', '.build_identity', '.load_or_compile')))
        self.assertFalse(any(name in ('NeuronQwen35ForCausalLM', 'modeling.NeuronQwen35ForCausalLM', 'NeuronQwen35Model') for name in names))

    def test_spawn_is_real_context_process_and_joined(self):
        source = WORKER.read_text()
        self.assertIn("multiprocessing.get_context('spawn')", source)
        self.assertIn('context.Process(target=spawn_worker', source)
        self.assertIn('child.exitcode == 0', source)
        self.assertIn("spawned['ppid'] == parent['pid']", source)
        self.assertIn('child.terminate()', source); self.assertIn('child.kill()', source)

    def test_sampler_is_per_instance_and_does_not_hash_binary_or_weights(self):
        source = WORKER.read_text()
        self.assertIn('standard = Sampler()', source); self.assertIn('fast = Sampler()', source)
        self.assertIn('fast.topk_topp_sampler.reference is reference', source)
        self.assertIn("modules['identity'].same_identity(graph_inputs, fast_inputs)", source)
        self.assertNotIn('sampler_fast=', source)
        self.assertIn('full_artifact_identity_not_computed=True', source)

    def test_config_path_runs_admission_with_mock_sdk_and_real_package_rules(self):
        from vllm_neuron.models.qwen35 import artifacts
        calls = []
        class Config:
            def __init__(self, neuron=None, load_config=None, body=None):
                self.body = copy.deepcopy(body or normalized_body()); self.metadata = self.body['metadata']
                if load_config: load_config(self)
            def to_json_string(self):
                return json.dumps(dict(self.body, metadata=self.metadata), sort_keys=True)
            def save(self, path):
                calls.append('save'); Path(path).mkdir(exist_ok=False)
                (Path(path) / 'neuron_config.json').write_text(self.to_json_string())
            @classmethod
            def load(cls, path):
                calls.append('load'); return cls(body=json.loads((Path(path)/'neuron_config.json').read_text()))
            @classmethod
            def from_json_string(cls, text):
                calls.append('json'); return cls(body=json.loads(text))
        def load_hf(path):
            calls.append('hf')
            self.assertEqual(json.loads((Path(path)/'config.json').read_text()), EXPECTED_MODEL)
            return lambda obj: None
        modules = dict(identity=identity, artifacts=artifacts,
            modeling=SimpleNamespace(Qwen35InferenceConfig=Config, NeuronConfig=lambda **kwargs: kwargs),
            hf_adapter=SimpleNamespace(load_pretrained_config=load_hf))
        official = ROOT / 'Qwen__Qwen3.5-2B.config.json'
        raw = official.read_bytes()
        bundle = dict(view_config=EXPECTED_MODEL, view_config_sha256=identity.CONFIG_VIEW_SHA256,
            neuron_kwargs=w.NEURON_KWARGS, official_config={'path': str(official), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / 'parent'; output.mkdir()
            body, report = w.config_checks(modules, output, bundle)
            self.assertEqual(len(report['invalid_scope_rejected']), 9)
            self.assertTrue(report['saved_body_admission']); self.assertFalse(report['weights_admitted'])
            self.assertIn('sdk-positive-infinity', json.dumps(report, allow_nan=False))
            self.assertEqual(calls, ['hf', 'save', 'load', 'json', 'save', 'json', 'json'])
            self.assertEqual(report['invalid_sdk_constructor_or_scope_rejected'], ['padding', 'quantization'])
            self.assertEqual(body['neuron_config']['max_length'], 4096)

    def test_config_result_serializes_sdk_infinity_as_typed_canonical(self):
        body = normalized_body()
        result = {'complete_configuration_canonical': identity.canonical_configuration(body)}
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn('sdk-positive-infinity', encoded)
        self.assertNotIn(': Infinity', encoded)


class FakeTokenizer:
    """Synthetic IDs test projection mechanics, never real tokenizer equivalence."""
    def apply_chat_template(self, messages, **kwargs):
        text = messages[0]['content']
        digest = hashlib.sha256(text.encode()).digest()
        size = max(24, len(text.encode()) // 5)
        return [151644, 872, 198] + [digest[i % len(digest)] + 100 for i in range(size)] + [151645, 198]
    def encode(self, text, **kwargs):
        return list(hashlib.sha256(text.encode()).digest())


class TokenizerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / 'source-bundle.json'
            p.prepare(dest)
            cls.bundle = json.loads(dest.read_text())
        cls.inputs = cls.bundle['tokenizer']
        cls.encoded = t.encode_protocol(FakeTokenizer(), cls.inputs)

    def comparable(self):
        result = copy.deepcopy(self.encoded)
        # Fake IDs deliberately do not certify the saved real token sequences.
        result['saved_quality'] = {r['id']: r['input_ids'][:] for r in result['quality']}
        result['saved_workloads'] = copy.deepcopy(result['workloads'])
        result['saved_six'] = copy.deepcopy(result['six_hf'])
        result['saved_long'] = copy.deepcopy(result['long'])
        return result

    def test_complete_original_protocol_functions_and_known_inputs(self):
        result = self.encoded
        self.assertEqual(len({row['id'] for row in result['quality']}), 96)
        self.assertEqual({k: len(v) for k, v in result['workloads'].items()}, {'32': 32, '128': 128, '512': 512, '1536': 1536})
        self.assertEqual(result['diagnostic'], result['workloads']['128'])
        self.assertEqual(len(result['smoke']), 4); self.assertEqual(len(result['six_hf']), 6)
        self.assertEqual([len(x) for x in result['long']], [2060, 2060])
        self.assertEqual(len(result['template_calls']), 111)
        self.assertTrue(all(row['kwargs'] == t.TEMPLATE_ARGS for row in result['template_calls']))
        suite = json.loads(t.checked_protocol(self.inputs, 'quality_suite'))
        self.assertEqual([r['messages'][0]['content'] for r in result['template_calls'][:96]], [r['prompt'] for r in suite['cases']])
        self.assertEqual(result['template_calls'][97]['messages'][0]['content'], '请详细介绍一下法国的首都。')
        left = self.comparable(); report = t.compare_encodings(left, copy.deepcopy(left))
        self.assertFalse(report['execution_scope']['http_or_model_execution'])
        self.assertTrue(report['unavailable_reference_evidence'])
        # Keep nested complete parent/child evidence within worker and reader ceilings.
        process = dict(original=left, view=left)
        self.assertLess(len(json.dumps(process, indent=2).encode()), 4 * 2**20)
        self.assertLess(len(json.dumps(dict(parent=process, child=process), indent=2).encode()), w.LIMITS['file_bytes'])

    def test_equal_lengths_cannot_hide_one_changed_token(self):
        left = self.comparable()
        for key in ('quality', 'workloads', 'initial', 'diagnostic', 'smoke', 'six_hf', 'long'):
            right = copy.deepcopy(left)
            ids = right[key][0]['input_ids'] if key == 'quality' else right[key]['128'] if key == 'workloads' else right[key][0] if key in ('smoke', 'six_hf', 'long') else right[key]
            ids[0] += 1
            with self.assertRaisesRegex(RuntimeError, 'Complete original/view token IDs differ'):
                t.compare_encodings(left, right)

    def test_saved_ids_must_equal_both_current_tokenizers(self):
        for key in ('saved_quality', 'saved_workloads', 'saved_six', 'saved_long'):
            left = self.comparable()
            ids = next(iter(left[key].values())) if isinstance(left[key], dict) else left[key][0]
            ids[0] += 1
            with self.assertRaises(RuntimeError): t.compare_encodings(left, copy.deepcopy(left))
        with self.assertRaises(RuntimeError): t.compare_encodings(self.encoded, copy.deepcopy(self.encoded))

    def test_missing_quality_case_and_changed_template_are_rejected(self):
        left = self.comparable(); left['quality'].pop()
        with self.assertRaisesRegex(RuntimeError, 'Missing quality'): t.compare_encodings(left, copy.deepcopy(left))
        recorder = t.RecordingTokenizer(FakeTokenizer())
        for argument in t.TEMPLATE_ARGS:
            flags = dict(t.TEMPLATE_ARGS); flags[argument] = not flags[argument]
            with self.assertRaisesRegex(RuntimeError, 'arguments changed'):
                recorder.apply_chat_template([{'role': 'user', 'content': 'test'}], **flags)
        with self.assertRaisesRegex(RuntimeError, 'encoding arguments'):
            recorder.encode('test', add_special_tokens=True)
        left = self.comparable(); right = copy.deepcopy(left)
        right['template_calls'][0]['kwargs']['enable_thinking'] = True
        with self.assertRaisesRegex(RuntimeError, 'template_calls'): t.compare_encodings(left, right)

    def test_long_projection_retains_actual_prefix_and_body_statements(self):
        source = t.checked_protocol(self.inputs, 'long_source').decode()
        class SequenceTokenizer:
            def apply_chat_template(self, messages, **kwargs): return [7, 8, 9]
            def encode(self, text, **kwargs): return [10, 11]
        result = t.long_reconstruction(source, SequenceTokenizer())
        expected = ([7, 8, 9] + [10, 11] * 1031)[:2060]
        self.assertEqual(result, [expected, expected])
        changed = source.replace('body=tok.encode(paragraph,add_special_tokens=False)', 'body=model.encode(paragraph)')
        with self.assertRaisesRegex(RuntimeError, 'Native/model work'): t.long_reconstruction(changed, SequenceTokenizer())

    def test_protocol_hash_change_is_rejected_before_execution(self):
        inputs = copy.deepcopy(self.inputs); inputs['sources']['quality_source']['sha256'] = '0'*64
        with self.assertRaisesRegex(RuntimeError, 'protocol/source changed'):
            t.encode_protocol(FakeTokenizer(), inputs)
        with tempfile.TemporaryDirectory() as root:
            suite = json.loads(t.checked_protocol(self.inputs, 'quality_suite'))
            suite['cases'].pop(); path = Path(root) / 'suite.json'; path.write_text(json.dumps(suite))
            inputs = copy.deepcopy(self.inputs); inputs['sources']['quality_suite'] = p.reference(path)
            with self.assertRaisesRegex(RuntimeError, 'Complete unique quality96'):
                t.encode_protocol(FakeTokenizer(), inputs)

    def test_preparation_never_reads_real_tokenizer_payloads(self):
        original_open = builtins.open
        def guarded_open(file, *args, **kwargs):
            self.assertFalse(str(file).startswith('/mnt/nvme/models/'), 'Preparation read model/tokenizer directory')
            return original_open(file, *args, **kwargs)
        import io
        original_io_open = io.open
        def guarded_io(file, *args, **kwargs):
            self.assertFalse(str(file).startswith('/mnt/nvme/models/'), 'Preparation read model/tokenizer directory')
            return original_io_open(file, *args, **kwargs)
        with tempfile.TemporaryDirectory() as root, patch('builtins.open', guarded_open), patch('io.open', guarded_io):
            dest = Path(root) / 'bundle.json'; p.prepare(dest)
            with self.assertRaises(FileExistsError): p.prepare(dest)
        self.assertFalse(self.inputs['tokenizer_payload_read_during_preparation'])
        self.assertEqual(set(self.inputs['files']), {'config.json', *t.TOKENIZER_NAMES})
        self.assertFalse(any(name == x or name.startswith(x + '.') for name in sys.modules for x in w.NATIVE_ROOTS))

    def test_payload_digest_and_lineage_use_only_temporary_fixture_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); original = root / 'original'; historical = root / 'historical'; work = root / 'work'; view = work / 'shared-hf-config'
            for path in (original, historical, view): path.mkdir(parents=True)
            config = b'{}'; (view / 'config.json').write_bytes(config); (historical / 'config.json').write_bytes(config)
            descriptors = {}
            for name in ('config.json', *t.TOKENIZER_NAMES):
                raw = config if name == 'config.json' else ('small mock ' + name).encode()
                (original / name).write_bytes(raw)
                descriptors[name] = dict(size=len(raw), algorithm='sha256', digest=hashlib.sha256(raw).hexdigest())
                if name != 'config.json': (historical / name).symlink_to(original / name)
            bundle = dict(tokenizer=dict(files=descriptors), view_config_sha256=hashlib.sha256(config).hexdigest())
            with patch.object(t, 'ORIGINAL_ROOT', str(original)), patch.object(t, 'HISTORICAL_VIEW', str(historical)):
                receipt = t.lineage(bundle, work)
                self.assertEqual(len(receipt['pinned_files']), 6); self.assertFalse(receipt['model_weights_read'])
                self.assertEqual(t.lineage(bundle, work), receipt)
                target = view / t.TOKENIZER_NAMES[0]; target.unlink(); target.write_bytes(b'foreign')
                with self.assertRaisesRegex(RuntimeError, 'Cannot overwrite'): t.lineage(bundle, work)
                target.unlink()
                target = historical / t.TOKENIZER_NAMES[0]; target.unlink(); target.symlink_to(original / 'config.json')
                with self.assertRaisesRegex(RuntimeError, 'Historical tokenizer lineage changed'): t.lineage(bundle, work)
            wrong = dict(descriptors['config.json'], digest='0'*64)
            with self.assertRaisesRegex(RuntimeError, 'bytes changed'): t.verify_payload(original / 'config.json', wrong)
            blob = dict(size=2, algorithm='git-blob-sha1', digest=hashlib.sha1(b'blob 2\0{}').hexdigest())
            self.assertEqual(t.verify_payload(original / 'config.json', blob)['digest'], blob['digest'])
            with self.assertRaisesRegex(RuntimeError, 'Not an allowed'): t.verify_payload(original / 'model.safetensors', wrong)

    def test_audit_allowlist_is_exact_and_original_files_read_only(self):
        with patch.object(w.sys, 'addaudithook') as install:
            w.install_runtime_guard(t.tokenizer_paths('/work'))
        guard = install.call_args.args[0]
        guard('open', (t.ORIGINAL_ROOT + '/tokenizer.json', 'r', os.O_RDONLY))
        for name, flags in [(t.ORIGINAL_ROOT + '/foreign.json', os.O_RDONLY),
                (t.ORIGINAL_ROOT + '/tokenizer.json', os.O_WRONLY),
                (t.ORIGINAL_ROOT + '/sub/../foreign.json', os.O_RDONLY),
                ('/tmp/model.safetensors', os.O_RDONLY), ('/dev/neuron0', os.O_RDONLY)]:
            with self.assertRaises(RuntimeError): guard('open', (name, 'r', flags))
        for event in ('subprocess.Popen', 'socket.connect', 'os.system'):
            with self.assertRaises(RuntimeError): guard(event, ())

    def test_all_prepared_aliases_reject_text_and_numeric_writes(self):
        with patch.object(w.sys, 'addaudithook') as install:
            w.install_runtime_guard(t.tokenizer_paths('/work'))
        guard = install.call_args.args[0]
        modes = ('w', 'wb', 'w+', 'a', 'ab', 'a+', 'x', 'xb', 'x+', 'r+', 'rb+')
        flags = (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND)
        for root in (t.ORIGINAL_ROOT, t.HISTORICAL_VIEW, '/work/shared-hf-config'):
            names = ('config.json', *t.TOKENIZER_NAMES) if root != '/work/shared-hf-config' else t.TOKENIZER_NAMES
            for name in names:
                for path in (root + '/' + name, root + '/nested/../' + name):
                    for mode in modes:
                        with self.assertRaisesRegex(RuntimeError, 'read-only'): guard('open', (path, mode, 0))
                    for flag in flags:
                        with self.assertRaisesRegex(RuntimeError, 'read-only'): guard('open', (path, None, flag))
                    guard('open', (path, 'r', os.O_RDONLY))
                    guard('open', (path, 'rb', os.O_RDONLY))
        guard('open', ('/work/shared-hf-config/config.json', 'x', os.O_WRONLY | os.O_CREAT | os.O_EXCL))

    def test_resolved_aliases_and_real_temporary_reads_preserve_fresh_config_writes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); original = root / 'original'; view = root / 'work/shared-hf-config'
            original.mkdir(); view.mkdir(parents=True)
            allowlist = {str(view / 'config.json')}
            for name in t.TOKENIZER_NAMES:
                target = original / name; target.write_bytes(b'synthetic tokenizer fixture')
                (view / name).symlink_to(target)
                allowlist.update((str(target), str(view / name)))
            with patch.object(w.sys, 'addaudithook') as install: w.install_runtime_guard(allowlist)
            guard = install.call_args.args[0]
            for name in t.TOKENIZER_NAMES:
                alias = root / ('alternate-' + name); alias.symlink_to(view / name)
                for flag in (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND):
                    with self.assertRaisesRegex(RuntimeError, 'read-only'): guard('open', (str(alias), None, flag))
                for mode in ('w', 'a', 'x', 'r+'):
                    with self.assertRaisesRegex(RuntimeError, 'read-only'): guard('open', (str(alias), mode, 0))
                # Use the captured guard before real reads of temporary files only.
                guard('open', (str(alias), 'rb', os.O_RDONLY))
                self.assertEqual(alias.read_bytes(), b'synthetic tokenizer fixture')
                guard('open', (str(view / name), 'r', os.O_RDONLY))
                self.assertEqual((view / name).read_text(), 'synthetic tokenizer fixture')
            config = view / 'config.json'
            guard('open', (str(config), 'x', os.O_WRONLY | os.O_CREAT | os.O_EXCL))
            with config.open('x') as stream: stream.write('{}')
            guard('open', (str(config), 'r', os.O_RDONLY)); self.assertEqual(config.read_text(), '{}')
            config.unlink(); config.symlink_to(original / t.TOKENIZER_NAMES[0])
            with self.assertRaisesRegex(RuntimeError, 'read-only'): guard('open', (str(config), 'w', os.O_WRONLY))

    def test_v2_is_frozen_and_only_guard_behavior_changes(self):
        frozen = json.loads((ROOT / 'qwen35-package-native-preflight-source-bundle-20261112-v2.json').read_text())
        for row in frozen['worker_sources']:
            self.assertEqual(p.reference(Path(row['path'])), row)
        original = ast.parse((ROOT / 'qwen35_package_native_preflight_v2.py').read_text())
        source = (ROOT / 'qwen35_package_native_preflight_v3.py').read_text().replace('qwen35_package_tokenizer_checks_v3.py', 'qwen35_package_tokenizer_checks_v2.py').replace('qwen35_tokenizer_preflight_v3', 'qwen35_tokenizer_preflight_v2')
        current = ast.parse(source)
        original_functions = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)}
        current_functions = {n.name: ast.dump(n) for n in current.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(original_functions), set(current_functions))
        for name in original_functions.keys() - {'install_runtime_guard'}:
            self.assertEqual(original_functions[name], current_functions[name], name)
        self.assertEqual((ROOT / 'qwen35_package_tokenizer_checks_v2.py').read_bytes(),
            (ROOT / 'qwen35_package_tokenizer_checks_v3.py').read_bytes())

    def test_v1_frozen_sources_and_unchanged_checks_preserved(self):
        frozen = json.loads((ROOT / 'qwen35-package-native-preflight-source-bundle-20261112-v1.json').read_text())
        for row in frozen['worker_sources']:
            self.assertEqual(p.reference(Path(row['path'])), row)
        original = {n.name: ast.dump(n) for n in ast.parse((ROOT / 'qwen35_package_native_preflight_v1.py').read_text()).body if isinstance(n, ast.FunctionDef)}
        current = {n.name: ast.dump(n) for n in ast.parse(WORKER.read_text()).body if isinstance(n, ast.FunctionDef)}
        changed = {'validate_bundle', 'install_runtime_guard', 'native_checks', 'execute'}
        for name in original.keys() - changed: self.assertEqual(original[name], current[name], name)
        previous = p.literal_assignments(ROOT / 'qwen35_package_native_preflight_v1.py')
        now = p.literal_assignments(WORKER)
        for name in ('SOURCE_COMMIT', 'SOURCE_DELIVERY_SHA256', 'PYTHON', 'SITE_PACKAGES', 'CHECKOUT', 'COMPILE_ENV', 'NATIVE_ROOTS'):
            self.assertEqual(previous[name], now[name])


if __name__ == '__main__':
    unittest.main(verbosity=2)
