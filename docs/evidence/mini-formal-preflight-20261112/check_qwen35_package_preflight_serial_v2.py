"""Source/mock lifecycle checks; no native, service, lock or model payload work."""
import ast
import copy
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'torch', 'torch_xla', 'torch_neuronx', 'neuronxcc', 'neuronx_distributed', 'neuronx_distributed_inference', 'vllm', 'transformers', 'numpy'}:
            raise AssertionError('Native import forbidden: ' + name)
sys.meta_path.insert(0, RejectNative())
import qwen35_package_preflight_serial_v2 as s
import qwen35_package_preflight_history_v2 as h


def spec_fixture():
    spec = dict(schema='qwen35_package_preflight_serial_spec_v2', run_name='source-fixture',
        output_root='/mnt/nvme/neuron-work/qwen35-package-preflight-source-fixture', history_overlay={'path': '/overlay.json', 'sha256': '1'*64},
        current_serial_catalog={'path': '/catalog.json', 'sha256': '2'*64}, source_sha256={str(Q / 'qwen35_package_preflight_serial_v2.py'): '3'*64},
        input_extension={}, running_handoff_path=str(Q / 'continuation-handoff-source-fixture.json'),
        lfm_r2_scope={'path': '/lfm-scope.json', 'sha256': '4'*64}, bundle=copy.deepcopy(s.BUNDLE), limits=copy.deepcopy(s.LIMITS),
        lock_identity=copy.deepcopy(s.LOCK_IDENTITY), native_python=s.PYTHON, automatic_successor=False)
    ref = dict(path='/future-spec.json', sha256='5'*64)
    review = dict(schema='qwen35_sampler_runtime_source_review_v1', reviewer='/root', passed=True, blockers=[],
        spec=ref, source_sha256=spec['source_sha256'], launch_authorized=True)
    return spec, ref, review


class ContractTests(unittest.TestCase):
    def test_exact_original_resources_and_single_stage(self):
        spec, ref, review = spec_fixture(); self.assertEqual(len(s.validate(spec, review, ref)), 1)
        self.assertEqual(s.LIMITS['root_floor_bytes'], 28*2**30); self.assertEqual(s.LIMITS['nvme_floor_bytes'], 30*2**30)
        self.assertEqual(s.PROPERTIES['MemorySwapMax'], '0'); self.assertEqual(s.PROPERTIES['MemoryMax'], str(12*2**30))
        for key, value in [('native_python', '/foreign/python'), ('automatic_successor', True), ('limits', dict(s.LIMITS, root_floor_bytes=1)),
                ('lock_identity', dict(inode=123, device=66306)), ('bundle', dict(s.BUNDLE, sha256='0'*64))]:
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)): s.validate(dict(spec, **{key: value}), review, ref)
        with self.assertRaises((ValueError, RuntimeError)): s.validate(spec, dict(review, spec={'path': '/other', 'sha256': '0'*64}), ref)

    def test_clean_pinned_native_command_and_worker_binding(self):
        spec, _, _ = spec_fixture(); owner = dict(pid=42, start_ticks='77')
        refs = [dict(path='/complete-current-proof.json', bytes=2, sha256='0'*64)]
        value = s.make_worker_input(spec, owner, '/mock', refs)
        self.assertEqual(value['schema'], 'qwen35-package-native-preflight-v3')
        self.assertEqual(value['binding']['owner'], dict(pid=42, start_ticks=77))
        self.assertEqual(value['binding']['history'], refs)
        env = s.environment(value['work']); self.assertNotIn('PYTHONPATH', env)
        self.assertFalse(any(k.startswith('TRN1_') or 'ALLOW_COMPILE' in k or k == 'NEURON_COMPILED_ARTIFACTS' for k in env))
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'): self.assertEqual(env[key], '1')
        command = s.native_argv(refs[0]); self.assertEqual(command[:3], [s.PYTHON, '-I', '-B'])
        self.assertIn('qwen35_package_native_preflight_v3.py', command[3])

    def test_lossless_whole_history_transport_and_mutation_negatives(self):
        # All 874 child objects remain present; no fixed current catalog is used.
        value = dict(catalog=[dict(run=i) for i in range(200)], children=[dict(pid=i+100, start_ticks=str(i+5), argv=['retained']) for i in range(874)])
        manifest, parts = s.history_parts(value); self.assertEqual(s.join_history(manifest, parts), value)
        for mutate in ('missing', 'index', 'byte'):
            changed = copy.deepcopy(parts)
            if mutate == 'missing': changed.pop()
            elif mutate == 'index': changed[0]['index'] += 1
            else: changed[0]['data'] = 'AAAA' + changed[0]['data'][4:]
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, RuntimeError)): s.join_history(manifest, changed)

    def test_output_scan_counts_only_exact_symlink_inodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); view = root / 'native/shared-hf-config'; view.mkdir(parents=True)
            for name in ('tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'chat_template.jinja'):
                (view / name).symlink_to('/mnt/nvme/models/Qwen3.5-2B/' + name)
            (view / 'config.json').write_text('{}')
            original_read = Path.read_bytes
            def guarded_read(path):
                self.assertFalse(str(path).startswith('/mnt/nvme/models/'))
                return original_read(path)
            with patch.object(Path, 'read_bytes', guarded_read):
                self.assertLess(s.scan_output(root)['logical_bytes'], 100)
            (view / 'extra').symlink_to('/mnt/nvme/models/Qwen3.5-2B/model.safetensors')
            with self.assertRaises((ValueError, RuntimeError)): s.scan_output(root)

    def test_known_atomic_publication_race_retries_without_relaxation(self):
        with patch.object(s, 'scan_output_once', side_effect=[s.PublicationRace('known'), {'entries': 1}]), patch.object(s.time, 'sleep'):
            self.assertEqual(s.scan_output('/tmp/fake'), {'entries': 1})
        with patch.object(s, 'scan_output_once', side_effect=ValueError('foreign hardlink')):
            with self.assertRaisesRegex(ValueError, 'foreign hardlink'): s.scan_output('/tmp/fake')

    def test_history_reuses_full_original_collectors_and_witness_dispatch(self):
        from qwen35_sampler_runtime_history_v12 import Bundle, Authority
        self.assertIs(h.History.physical_pair, Bundle.physical_pair)
        self.assertIs(h.History.freshness, Bundle.freshness); self.assertIs(h.History.unchanged, Bundle.unchanged)
        self.assertIs(s.Authority, Authority)
        source = Path(h.__file__).read_text()
        for text in ('overlays.resolve(', 'extend_scope(', 'build(self.meta.read', 'witnesses.executed_extension(',
                'sources.verify_text_sources(', "resolved['plan']['source_sha256']", "original_model_exit_code'] == 1"):
            self.assertIn(text, source)
        self.assertNotIn('minicpm5-operator-post-r2-serial-workloads-root-catalog-', source)

    def test_frozen_v3_worker_sources_and_no_native_imports(self):
        raw = Path(s.BUNDLE['path']).read_bytes()
        self.assertEqual(len(raw), s.BUNDLE['bytes']); self.assertEqual(hashlib.sha256(raw).hexdigest(), s.BUNDLE['sha256'])
        for row in json.loads(raw)['worker_sources']:
            actual = Path(row['path']).read_bytes()
            self.assertEqual(len(actual), row['bytes']); self.assertEqual(hashlib.sha256(actual).hexdigest(), row['sha256'])
        self.assertFalse(any(name == root or name.startswith(root+'.') for name in sys.modules for root in s.worker.NATIVE_ROOTS))

    def test_help_is_inert(self):
        with patch.object(s, 'outer', side_effect=AssertionError('runtime')), patch.object(s, 'supervisor', side_effect=AssertionError('runtime')), patch('sys.stdout', new=io.StringIO()):
            with self.assertRaises(SystemExit) as stopped: s.main(['--help'])
        self.assertEqual(stopped.exception.code, 0)


class OuterTests(unittest.TestCase):
    def exercise(self, root, *, launch_failure=False, delayed_drain=False, result_failed=False):
        spec, spec_ref, _ = spec_fixture(); spec['output_root'] = str(root); spec['running_handoff_path'] = str(root.parent / 'running.json')
        events = []; owner = dict(pid=42, start_ticks='77', ppid=1, argv=['outer'], cgroup='0::/outer')
        lock = SimpleNamespace(owner=owner, fd=9, initial={'real': 'descriptor'}, acquire=lambda: events.append('acquire'),
            check=lambda: None, release=lambda value: events.append(('release', value)))
        histogram = SimpleNamespace(resolve=lambda: histogram, physical_pair=lambda authority: {'both': 'complete'},
            freshness=lambda: {'latest_handoff': {'path': '/previous.json', 'sha256': '0'*64}}, unchanged=lambda: None,
            resolved={'plan_sha256': 'a'*64}, meta=SimpleNamespace(identities={}))
        terminal = dict(systemd_exit_status=dict(ExecMainCode=1, ExecMainStatus=0, Result='success'), cleanup_complete=True)
        observation = {'ActiveState': 'active', 'SubState': 'exited'}
        service = SimpleNamespace(started=False, terminal=None, owner={'process': {'pid': 43}, 'service_owner': {'unit': s.names(spec)['coordinator']}})
        attempts = [0]
        def start(guard):
            events.append('start'); service.started = True
            if launch_failure: raise ValueError('ambiguous launch')
            s.once(root / 'supervisor-owner.json', service.owner)
            Path(spec['running_handoff_path']).write_text('{}')
            s.once(root / 'supervisor-result.json', dict(complete=not result_failed, passed=not result_failed,
                admission=s.ref(root / 'admission.json'), owner=service.owner))
            return service.owner
        def cleanup(): events.append('cleanup'); return terminal
        def drained():
            attempts[0] += 1; events.append('drain')
            if delayed_drain and attempts[0] == 1: raise RuntimeError('resource tracker still live')
            return terminal
        service.start = start; service.observe = lambda: observation; service.cleanup = cleanup; service.drained = drained
        api = SimpleNamespace(GROUP_PREFIX='/mock/', active_heavy=lambda *args: [], empty_group=lambda *args: {}, terminal=lambda value: True)
        with patch.object(s, 'load_spec', return_value=spec), patch.object(s, 'RetainedLock', return_value=lock), \
                patch.object(s, 'resources', return_value={}), patch.object(s, 'History', return_value=histogram), \
                patch.object(s, 'Service', return_value=service), patch.object(s, 'process_api', return_value=api), \
                patch.object(s, 'properties', return_value={'LoadState': 'not-found', 'InvocationID': ''}), \
                patch.object(s, 'check_running', return_value={'observation': {}, 'value': {}}), \
                patch.object(s.time, 'sleep', side_effect=lambda delay: events.append('retry')), \
                patch('sys.stdout', new=io.StringIO()):
            # The inherited latest checker is only needed after root ACK.
            from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
            with patch.object(latest, 'latest_handoff', return_value={}):
                if launch_failure or result_failed:
                    with self.assertRaises((ValueError, RuntimeError)): s.outer(spec_ref, {'path': '/review.json', 'sha256': 'b'*64})
                else: self.assertEqual(s.outer(spec_ref, {'path': '/review.json', 'sha256': 'b'*64}), 0)
        return events, json.loads((root / 'outer-terminal.json').read_text())

    def test_success_orders_root_ack_actual_terminal_and_drain_before_unlock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'output'; events, terminal = self.exercise(root)
            self.assertEqual(events[0], 'acquire'); self.assertEqual(events[-1], ('release', True))
            self.assertLess(events.index('drain'), len(events)-1)
            self.assertFalse(terminal['failed']); self.assertTrue(terminal['actual_external_tool_exit_pending'])
            self.assertFalse(terminal['self_exit_claimed'])
            value = json.loads((root / 'worker-input.json').read_text())
            self.assertEqual(value['binding']['owner'], {'pid': 42, 'start_ticks': 77})
            self.assertGreaterEqual(len(value['binding']['history']), 2)

    def test_partial_launch_failure_retains_lock_until_tracker_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            events, terminal = self.exercise(Path(directory) / 'output', launch_failure=True, delayed_drain=True)
            self.assertEqual(events.count('start'), 1); self.assertIn('retry', events)
            self.assertEqual(events[-1], ('release', True)); self.assertEqual(events.count('drain'), 2)
            self.assertTrue(terminal['failed']); self.assertFalse(terminal['result_present'])

    def test_failed_result_is_not_promoted_by_service_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            events, terminal = self.exercise(Path(directory) / 'output', result_failed=True)
            self.assertTrue(terminal['failed']); self.assertEqual(events[-1], ('release', True))

    def test_external_reconciliation_requires_same_actual_tool_and_drain(self):
        terminal = dict(schema='qwen35_package_preflight_outer_terminal_v2', self_exit_claimed=False,
            actual_external_tool_exit_pending=True, whole_owned_scope_drained=True, failed=False, result_present=True,
            admission={'path': '/admission.json', 'sha256': 'a'*64}, outer_owner={'pid': 42, 'start_ticks': '77'})
        tool = dict(kind='tool_session', session_id=123, exit_code=0, terminal_chunk='abcdef')
        running = dict(qwen35_sampler_current_execution=dict(admission=terminal['admission'], outer_owner=terminal['outer_owner'],
            tool={'kind': 'tool_session', 'session_id': 123}))
        self.assertTrue(s.reconcile(terminal, running, tool, actual_owner_gone=True, actual_service_drained=True)['passed'])
        for changed, gone, drain in [(dict(tool, session_id=124), True, True), (tool, False, True), (tool, True, False)]:
            with self.assertRaises((ValueError, RuntimeError)): s.reconcile(terminal, running, changed, actual_owner_gone=gone, actual_service_drained=drain)
        failed = s.reconcile(dict(terminal, failed=True, result_present=False), running, dict(tool, exit_code=1), actual_owner_gone=True, actual_service_drained=True)
        self.assertFalse(failed['passed']); self.assertEqual(failed['actual_tool']['exit_code'], 1)


class NativeProcessTests(unittest.TestCase):
    def test_supervisor_releases_only_exact_v3_binding_and_waited_owner(self):
        for changed in (None, 'input', 'result_owner'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); spec, spec_ref, _ = spec_fixture(); spec['output_root'] = str(root)
                owner = dict(pid=42, start_ticks='77'); unit = s.names(spec)['coordinator']; group = '/mock/' + unit + '.service'
                proof = root / 'history.json'; proof.write_text('{}')
                history = [s.bound_ref(proof)]; value = s.make_worker_input(spec, owner, group, history)
                if changed == 'input': value['binding']['owner']['pid'] += 1
                s.once(root / 'worker-input.json', value)
                admission = dict(spec=spec_ref, review={'path': '/review.json', 'sha256': 'a'*64}, outer_owner=owner,
                    retained_fd=9, worker_input=s.bound_ref(root / 'worker-input.json'), worker_history=history)
                admission_ref = s.once(root / 'admission.json', admission)
                own = dict(pid=os.getpid(), start_ticks='88', ppid=1, argv=s.supervisor_argv(admission_ref), cgroup='0::'+group)
                native = dict(pid=333, start_ticks='99', ppid=own['pid'], argv=s.native_argv(admission['worker_input']), cgroup=own['cgroup'])
                api = SimpleNamespace(GROUP_PREFIX='/mock/', proc=lambda pid: own)
                def native_mock(command, env, output, guard):
                    self.assertEqual(command, s.native_argv(admission['worker_input'])); guard()
                    result = dict(schema=s.worker.SCHEMA+'-result', passed=True, complete=True, root_binding=value['binding'], source_bundle=s.BUNDLE,
                        parent=dict(pid=333 if changed != 'result_owner' else 334, start_ticks=99, ppid=own['pid']), child=dict(pid=444, ppid=333))
                    (root / 'native').mkdir(); s.once(root / 'native/result.json', result)
                    receipt = dict(owner=native, actual_returncode=0, popen_wait_observed=True)
                    s.once(root / 'native-process-exit.json', receipt); return receipt
                (root / 'running-ack.json').write_text('{}')
                with patch.object(s, 'load_spec', return_value=spec), patch.object(s.sys, 'version_info', (3, 10)), \
                        patch.object(s.sys, 'executable', s.PYTHON), patch.object(s.os, 'sched_setaffinity'), patch.object(s.os, 'setpriority'), \
                        patch.object(s.resource, 'setrlimit'), patch.object(s, 'process_api', return_value=api), \
                        patch.object(s, 'properties', return_value={}), patch.object(s, 'verify_properties', return_value={'InvocationID': 'a'*32}), \
                        patch.object(s, 'descriptor'), patch.object(s, 'check_ack', return_value={}), \
                        patch.object(s, 'loaded_local_sources'), patch.dict(os.environ, s.environment(root/'native'), clear=True), \
                        patch.object(s, 'capture_native', side_effect=native_mock) as native_call:
                    if changed:
                        with self.assertRaises((ValueError, RuntimeError)): s.supervisor(admission_ref)
                        self.assertFalse((root / 'supervisor-result.json').exists())
                        if changed == 'input': native_call.assert_not_called()
                    else:
                        self.assertEqual(s.supervisor(admission_ref), 0); native_call.assert_called_once()
                        self.assertTrue(json.loads((root / 'supervisor-result.json').read_text())['passed'])

    def test_real_child_command_wait_and_nonzero_status_are_not_service_ack(self):
        for code in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); command = [s.PYTHON, '-I', '-B', '/pinned-worker.py']
                stdout = SimpleNamespace(fileno=lambda: 99, close=Mock())
                child = SimpleNamespace(pid=333, stdout=stdout, poll=lambda: code, wait=Mock(return_value=code), returncode=code)
                active = [True]
                selector = SimpleNamespace(register=lambda *a: None, get_map=lambda: {99: True} if active[0] else {},
                    select=lambda *a: [(SimpleNamespace(fileobj=stdout), None)], unregister=lambda *a: active.__setitem__(0, False), close=Mock())
                own = dict(pid=os.getpid(), cgroup='0::/owned')
                native = dict(pid=333, ppid=os.getpid(), start_ticks='99', argv=command, cgroup='0::/owned')
                api = SimpleNamespace(proc=lambda pid: native if pid == 333 else own)
                original_read = os.read; blocks = iter([b'fixture log', b''])
                def read_pipe(fd, size): return next(blocks) if fd == 99 else original_read(fd, size)
                with patch.object(s.subprocess, 'Popen', return_value=child) as popen, patch.object(s.selectors, 'DefaultSelector', return_value=selector), \
                        patch.object(s, 'process_api', return_value=api), patch.object(s.os, 'read', side_effect=read_pipe):
                    if code:
                        with self.assertRaisesRegex((ValueError, RuntimeError), 'Actual native'): s.capture_native(command, {'OFFLINE': '1'}, root, lambda: None)
                    else: self.assertEqual(s.capture_native(command, {'OFFLINE': '1'}, root, lambda: None)['actual_returncode'], 0)
                self.assertTrue(popen.call_args.kwargs['close_fds']); child.wait.assert_called_once()
                receipt = json.loads((root / 'native-process-exit.json').read_text())
                self.assertEqual(receipt['actual_returncode'], code); self.assertTrue(receipt['popen_wait_observed'])
                self.assertTrue(receipt['whole_cgroup_drain_pending'])


class ActualServiceCompositionTests(unittest.TestCase):
    def new_service(self, root):
        return s.Service('trn1-qwen35-sampler-source-fixture-package-native-preflight',
            ['/pinned/python', '-I', '-B', '/pinned/supervisor.py'], {'PATH': '/usr/bin:/bin'},
            s.PROPERTIES, root / 'service.log')

    def test_install_failure_uses_two_original_never_loaded_checks_without_hold(self):
        import qwen35_sampler_runtime_service_v12 as original
        with tempfile.TemporaryDirectory() as directory:
            service = self.new_service(Path(directory)); empty = {'LoadState': 'not-found', 'InvocationID': ''}
            api = SimpleNamespace(GROUP_PREFIX=original.process_api().GROUP_PREFIX,
                empty_group=Mock(return_value={'group': service.group, 'absent': True}),
                command=Mock(side_effect=ValueError('systemd-run failed before installation')))
            with patch.object(original, 'properties', return_value=empty), patch.object(original, 'process_api', return_value=api):
                with self.assertRaisesRegex(ValueError, 'before installation'): service.start(lambda: None)
                self.assertTrue(service.started)
                waited = Mock(side_effect=AssertionError('Never-loaded proof must not enter the cleanup hold'))
                proof = s.retain_until_drained(service.cleanup,
                    lambda: dict(physically_drained=True, terminal=s.owned_terminal(service)), Mock(), waited)
                waited.assert_not_called()
                self.assertTrue(proof['terminal']['never_loaded']); self.assertTrue(proof['terminal']['cleanup_complete'])
                self.assertFalse(proof['terminal']['exit_observed'])
                self.assertEqual(api.empty_group.call_count, 3)  # Fresh start check, cleanup, independent verification.
                api.command.assert_called_once()

    def test_installed_failed_unit_keeps_original_identity_exit_and_drained_path(self):
        import qwen35_sampler_runtime_service_v12 as original
        with tempfile.TemporaryDirectory() as directory:
            service = self.new_service(Path(directory)); installed = [False]
            command = original.command(service.unit, service.argv, service.env, service.props, service.log)
            value = dict(service.props, Id=service.unit+'.service', LoadState='loaded', InvocationID='a'*32,
                RuntimeMaxUSec='15min', TimeoutStopUSec='15s', ExecStopPost='', Environment='',
                ExecStart='{ argv[]='+' '.join(command[command.index('/usr/bin/env'):])+' ; }',
                ControlGroup=service.group, StandardOutput='append', StandardError='inherit',
                MainPID='0', ControlPID='0', Job='', ActiveState='failed', SubState='failed',
                ExecMainPID='987', ExecMainCode='1', ExecMainStatus='7', Result='exit-code')
            def props(unit): return value if installed[0] else {'LoadState': 'not-found', 'InvocationID': ''}
            api = SimpleNamespace(GROUP_PREFIX=original.process_api().GROUP_PREFIX, command=lambda *args, **kwargs: installed.__setitem__(0, True),
                empty_group=Mock(return_value={'group': service.group, 'absent': True}), proc=lambda pid: None,
                invocation=lambda value: value == 'a'*32, terminal=lambda current: current['MainPID'] == '0')
            with patch.object(original, 'properties', side_effect=props), patch.object(original, 'process_api', return_value=api):
                with self.assertRaisesRegex(ValueError, 'before actual owner acknowledgment'): service.start(lambda: None)
                self.assertIsNotNone(service.unacknowledged); self.assertEqual(service.installed_invocation, 'a'*32)
                service.cleanup()
                with patch.object(service, 'cleanup', wraps=service.cleanup) as cleanup:
                    terminal = s.owned_terminal(service); cleanup.assert_not_called()
                self.assertEqual(terminal['systemd_exit_status'], dict(ExecMainCode=1, ExecMainStatus=7, Result='exit-code'))
                self.assertTrue(terminal['exit_observed']); self.assertFalse(terminal.get('never_loaded', False))

    def test_captured_or_installed_identity_cannot_be_relabelled_never_loaded(self):
        import qwen35_sampler_runtime_service_v12 as original
        with tempfile.TemporaryDirectory() as directory:
            for attribute, value in [('installed_invocation', 'a'*32), ('pending_process', {'pid': 12}),
                    ('unacknowledged', {'exec_main_pid': 12}), ('owner', {'process': {'pid': 12}})]:
                service = self.new_service(Path(directory)); service.started = True
                service.terminal = {'never_loaded': True}; setattr(service, attribute, value)
                api = SimpleNamespace(empty_group=Mock())
                with self.subTest(attribute=attribute), patch.object(original, 'properties', return_value={'LoadState': 'not-found'}), \
                        patch.object(original, 'process_api', return_value=api):
                    with self.assertRaisesRegex(ValueError, 'cannot disappear before audit'): s.owned_terminal(service)
                    api.empty_group.assert_not_called()

    def test_frozen_v1_and_all_other_adapter_functions_are_unchanged(self):
        delivery = json.loads((Q / 'qwen35-package-preflight-serial-source-delivery-20261112-v1.json').read_text())
        for row in delivery['new_sources']:
            payload = Path(row['path']).read_bytes()
            self.assertEqual(len(payload), row['bytes']); self.assertEqual(hashlib.sha256(payload).hexdigest(), row['sha256'])
        before = (Q / 'qwen35_package_preflight_serial_v1.py').read_text()
        after = Path(s.__file__).read_text()
        names = ['qwen35_package_preflight_serial', 'qwen35_package_preflight_history', 'check_qwen35_package_preflight_serial',
            'prepare_qwen35_package_preflight_serial', 'qwen35_package_preflight_serial_spec', 'qwen35_package_preflight_serial_admission',
            'qwen35_package_preflight_outer_terminal', 'qwen35_package_preflight_reconciliation']
        for name in names: after = after.replace(name+'_v2', name+'_v1')
        old = {node.name: ast.dump(node) for node in ast.parse(before).body if isinstance(node, ast.FunctionDef)}
        new = {node.name: ast.dump(node) for node in ast.parse(after).body if isinstance(node, ast.FunctionDef)}
        for name in old.keys() - {'outer'}: self.assertEqual(old[name], new[name], name)
        self.assertEqual((Q / 'qwen35_package_preflight_history_v1.py').read_bytes(), (Q / 'qwen35_package_preflight_history_v2.py').read_bytes())


if __name__ == '__main__': unittest.main(verbosity=2)
