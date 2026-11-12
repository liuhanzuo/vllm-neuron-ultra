"""SOURCE/MOCK only. No native imports, subprocess, real proc/systemd or lock calls."""
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

HERE = Path(__file__).resolve().parent
Q = HERE.parent
sys.path[:0] = [str(Q), str(Q.parent)]

class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_neuronx', 'neuronx_distributed', 'neuronx_distributed_inference', 'vllm', 'vllm_neuron', 'transformers', 'numpy'}:
            raise AssertionError('native import forbidden in SOURCE/MOCK: ' + fullname)

sys.meta_path.insert(0, NativeBlocker())
from qwen35_package_serial_runtime_v8 import contract as c, io, budget as b, wiring as w, adapter as a, service as svc, readiness as ready
from qwen35_package_full_validation_v2 import entry, sampler, http


def spec_fixture():
    case = json.loads((Q / 'qwen35_package_full_validation_v2/case.example.json').read_text())
    pin = {'path': '/tmp/unbound-metadata.json', 'sha256': 'a' * 64}
    return dict(schema='qwen35_package_serial_spec_v1', run_name='mock-r1', output_root='/mnt/nvme/neuron-work/qwp-mock-r1',
        case=case, history_overlay=pin, current_serial_catalog=pin, frozen_science_delivery=c.SCIENCE_DELIVERY,
        source_sha256={'/tmp/source.py': 'a' * 64}, input_extension={pin['path']: {'sha256': pin['sha256'], 'identity': {}}}, running_handoff_path=str(Q / 'continuation-handoff-mock.json'),
        limits=c.LIMITS, lock_identity=c.LOCK_IDENTITY, native_python=c.PYTHON, automatic_successor=False,
        lfm_r2_scope=pin, resource_budget=pin, build_file_limit_bytes=32 * c.MIB,
        binary_inputs={path: {'kind': 'binary_payload', 'identity': {}} for path in c.BINARY_PATHS})


def reviewed_fixture(spec):
    pins = json.loads((HERE / 'source_files.json').read_text())
    return dict(schema='qwen35_package_reviewed_budget_v1', reviewer='/root', passed=True, blockers=[], case=spec['case'],
        output_root=spec['output_root'], control_root=str(c.control_root(spec)),
        classes={k: dict(allocated_bytes=64 * c.MIB, single_file_bytes=32 * c.MIB, entries=1000) for k in b.CLASSES},
        headroom_bytes={'root': c.MIB, 'nvme': c.MIB},
        parent_identities={str(p.parent): {'realpath': str(p.parent), 'device': 1, 'inode': 2} for p in b.roots(spec)},
        source_files={name: dict(pin=row, identity={'realpath': '/mnt/nvme/models/Qwen3.5-2B/' + name, 'size': row['size']}) for name, row in pins.items()},
        build_file_limit_bytes=spec['build_file_limit_bytes'])


def node(path, name):
    tree = ast.parse(path.read_text())
    result = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    return ast.dump(result, include_attributes=False)


def method(path, class_name, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    value = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(value, include_attributes=False)


class Contracts(unittest.TestCase):
    def test_import_has_no_native(self):
        self.assertFalse(any(n.split('.')[0] in {'torch', 'vllm', 'vllm_neuron'} for n in sys.modules))

    def test_exact_25_order_matches_frozen_delivery(self):
        original = json.loads((Q / 'qwen35_package_full_validation_v2/stage_order.json').read_text())['stages']
        names = {'prepare_view': 'prepare', 'build12': 'build', 'correctness_capture': 'capture', 'matched_http': 'http'}
        actual = [(x.get('arm', 'external' if x['stage'] == 'independent_final_audit' else 'shared'), names.get(x['stage'], x['stage'])) for x in original]
        self.assertEqual(list(w.STAGES), actual)
        self.assertEqual(len(w.STAGES), 25)
        self.assertEqual(set(c.PHASES), set(w.STAGES) - {('external', 'independent_final_audit'), ('shared', 'sampler_pair_gate'), ('shared', 'quality_pair_gate'), ('shared', 'full_pair_gate')})

    def test_valid_future_spec_is_not_runtime_admission(self):
        spec = spec_fixture(); ref = {'path': '/tmp/future-spec.json', 'sha256': 'b' * 64}
        review = dict(schema='qwen35_package_serial_source_review_v1', reviewer='/root', passed=True, blockers=[], spec=ref, source_sha256=spec['source_sha256'], launch_authorized=True)
        self.assertEqual(c.validate_spec(spec, review, ref), c.units(spec['run_name']))
        review['launch_authorized'] = False
        with self.assertRaises(ValueError): c.validate_spec(spec, review, ref)

    def test_missing_budget_fails_before_lock(self):
        with patch.object(a, 'prelock', side_effect=ValueError('unbound budget')), patch.object(a, 'RetainedLock') as lock:
            with self.assertRaisesRegex(ValueError, 'unbound budget'): a.outer({}, {})
            lock.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'unbound'): b.reviewed_budget({})

    def test_budgets_require_independent_full_binding(self):
        spec = spec_fixture(); value = reviewed_fixture(spec)
        self.assertEqual(b.validate_budget(spec, value), value)
        for key in b.CLASSES:
            bad = copy.deepcopy(value); bad['classes'][key]['allocated_bytes'] = 0
            with self.assertRaises(ValueError): b.validate_budget(spec, bad)
        for field in ('case', 'output_root', 'control_root'):
            bad = copy.deepcopy(value); bad[field] = None
            with self.assertRaises(ValueError): b.validate_budget(spec, bad)

    def test_no_guessed_compile_budget(self):
        spec = spec_fixture(); value = reviewed_fixture(spec)
        bad = copy.deepcopy(value); bad['build_file_limit_bytes'] += 1
        with self.assertRaises(ValueError): b.validate_budget(spec, bad)
        bad = copy.deepcopy(value); bad['classes']['build_auxiliary']['allocated_bytes'] = c.MIB
        with self.assertRaises(ValueError): b.validate_budget(spec, bad)

    def test_all_six_rpc_paths_and_byte_limit(self):
        spec = spec_fixture(); paths = c.ipc_paths(spec)
        self.assertEqual(len(paths), 6)
        self.assertLessEqual(max(row['path_bytes'] for row in paths), 107)
        spec['output_root'] = '/mnt/nvme/neuron-work/qwen35-package-full-v2-20261112-r1'
        with self.assertRaisesRegex(ValueError, '107'): c.ipc_paths(spec)

    def test_timed_environment_only_three_differences(self):
        spec = spec_fixture(); envs = []
        for arm in w.ARMS:
            phase = dict(arm=arm, phase='http', directory=str(w.phase_directory(spec, arm, 'http')), control=str(c.control_root(spec) / (arm + '-http')))
            envs.append(w.child_environment(spec, phase))
        changed = {k for k in envs[0] if envs[0][k] != envs[1][k]}
        self.assertEqual(changed, {'NEURON_CPU_SAMPLER_FAST', 'TMPDIR', 'VLLM_RPC_BASE_PATH'})
        self.assertFalse(any(k.startswith(('TRN1_QWEN35', 'QWEN35_SAMPLER_', 'QWEN35_PACKAGE_CAPTURE_')) for k in envs[0]))
        self.assertEqual(envs[0]['NEURON_QWEN35_ALLOW_COMPILE'], '0')

    def test_compile_only_build_child_not_supervisor(self):
        spec = spec_fixture()
        for arm, name in c.PHASES:
            phase = dict(arm=arm, phase=name, directory=str(w.phase_directory(spec, arm, name)), control=str(c.control_root(spec) / (arm + '-' + name)))
            with patch.object(w, 'manifest_sha', return_value='a' * 64): env = w.child_environment(spec, phase)
            self.assertEqual(env['NEURON_QWEN35_ALLOW_COMPILE'], '1' if name == 'build' else '0')
        self.assertEqual(c.native_environment(spec['case'])['NEURON_QWEN35_ALLOW_COMPILE'], '0')

    def test_capture_json_uses_direct_popen_env(self):
        spec = spec_fixture(); phase = dict(arm='candidate', phase='capture', directory=str(w.phase_directory(spec, 'candidate', 'capture')), control=str(c.control_root(spec) / 'candidate-capture'))
        with patch.object(w, 'manifest_sha', return_value='a' * 64): env = w.child_environment(spec, phase)
        self.assertIn(' ', env['QWEN35_PACKAGE_CAPTURE_IDENTITY'])
        with self.assertRaises(ValueError): svc.command('unit', [], env, {}, Path('/tmp/mock.log'))
        svc.command('unit', [], c.native_environment(spec['case']), {}, Path('/tmp/mock.log'))

    def test_plain_argv_has_no_custom_worker(self):
        spec = spec_fixture()
        self.assertNotIn('--worker-cls', w.api_arguments(spec, 'baseline', 'http'))
        self.assertNotIn('--worker-cls', w.api_arguments(spec, 'candidate', 'quality'))
        self.assertIn('--worker-cls', w.api_arguments(spec, 'baseline', 'capture'))

    def test_original_resource_envelopes(self):
        self.assertEqual(c.phase_properties('baseline', 'numeric'), c.MODEL_PROPERTIES)
        self.assertEqual(c.phase_properties('candidate', 'http'), c.SERVER_PROPERTIES)
        spec = spec_fixture(); build = c.phase_properties('shared', 'build', spec)
        self.assertEqual(build['MemoryMax'], str(26 * c.GIB)); self.assertEqual(build['MemoryHigh'], str(25 * c.GIB))
        self.assertEqual(build['MemorySwapMax'], str(12 * c.GIB)); self.assertEqual(build['RuntimeMaxSec'], '14400')
        self.assertEqual(sum(c.budget().values()), 2 * c.GIB)
        self.assertEqual(c.LIMITS['root_floor_bytes'], 28 * c.GIB); self.assertEqual(c.LIMITS['nvme_floor_bytes'], 30 * c.GIB)

    def test_remaining_reserve_continuous(self):
        spec = spec_fixture(); value = reviewed_fixture(spec)
        cap = 2 * c.GIB + sum(r['allocated_bytes'] for r in value['classes'].values())
        for used in (0, 100, cap):
            result = b.remaining_reserve(spec, {'allocated_bytes': used}, value)
            self.assertEqual(result, {'root': c.MIB, 'nvme': cap - used + c.MIB})

    def test_pinned_startup_source_closure_exists(self):
        self.assertTrue(all(Path(path).is_file() for path in c.STARTUP_FILES))
        self.assertTrue(any(path.endswith('executor/uniproc_executor.py') for path in c.STARTUP_FILES))
        self.assertTrue(any(path.endswith('worker/neuronx_distributed_model_runner.py') for path in c.STARTUP_FILES))

    def test_typed_payload_map_is_separate(self):
        spec = spec_fixture(); self.assertEqual(len(spec['binary_inputs']), 6)
        self.assertTrue(all(p.endswith('.pt') for p in spec['binary_inputs']))
        text = (HERE / 'history.py').read_text()
        self.assertNotIn('binary_inputs', text)
        self.assertIn('identity(path) == row[\'identity\']', (HERE / 'budget.py').read_text())


class LifecyclePreservation(unittest.TestCase):
    def test_original_service_and_wait_are_exact_ast(self):
        old = Q / 'qwen35_sampler_runtime_service_v12.py'; new = HERE / 'service.py'
        for name in ('Service', 'supervise', 'api_failure_receipt', 'validate_api_failure', 'failure_signal', 'command', 'verify_properties'):
            self.assertEqual(node(old, name), node(new, name), name)

    def test_original_lock_and_atomic_publication_are_exact_ast(self):
        old = Q / 'qwen35_sampler_runtime_io_v12.py'; new = HERE / 'io.py'
        for name in ('RetainedLock', 'descriptor', 'PublicationRace', 'transient', 'observed_stat', 'once', 'identity', 'raw', 'Metadata'):
            self.assertEqual(node(old, name), node(new, name), name)

    def test_whole_history_authority_and_running_ack_exact_ast(self):
        old = Q / 'qwen35_sampler_runtime_history_v12.py'; new = HERE / 'history.py'
        for name in ('Authority', 'expected_running', 'check_running'):
            self.assertEqual(node(old, name), node(new, name), name)

    def test_original_monitor_reconstruction_and_emergency_hold_exact_ast(self):
        old = Q / 'qwen35_sampler_runtime_adapter_v12.py'; new = HERE / 'adapter.py'
        for name in ('LeaseObserver', 'ActiveScopeMonitor', 'declared_services', 'retain_until_drained', 'admit_shape', 'load_context'):
            self.assertEqual(node(old, name), node(new, name), name)
        for name in ('api_failure', 'wait_failed_api', 'reject_api_failure', 'finish', '__exit__'):
            self.assertEqual(method(old, 'Phase', name), method(new, 'Phase', name), name)
        self.assertEqual(method(old, 'Adapter', 'server_audit'), method(new, 'Adapter', 'server_audit'))

    def test_build_requires_compiler_drain_before_cleanup(self):
        old_tree = ast.parse((Q / 'qwen35_sampler_runtime_adapter_v12.py').read_text())
        new_tree = ast.parse((HERE / 'adapter.py').read_text())
        def leaf(tree):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Phase')
            return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'leaf')
        old, new = leaf(old_tree), leaf(new_tree)
        body = next(n for n in new.body if isinstance(n, ast.Try)).body
        added = body.pop()
        self.assertEqual(ast.dump(added, include_attributes=False), ast.dump(ast.parse('if self.name == "build": self.service.drained()').body[0], include_attributes=False))
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_retained_lock_never_releases_without_drain(self):
        lock = io.RetainedLock(); lock.fd = 4
        with patch.object(io.fcntl, 'flock') as flock:
            with self.assertRaises(ValueError): lock.release(False)
            flock.assert_not_called()

    def test_lease_failure_latches(self):
        obs = a.LeaseObserver({'outer_owner': {}, 'retained_fd': 4}, {})
        error = ValueError('changed descriptor')
        with patch.object(a, 'descriptor', side_effect=error) as check:
            for _ in range(2):
                with self.assertRaises(ValueError) as raised: obs.check()
                self.assertIs(raised.exception, error)
            self.assertEqual(check.call_count, 1)

    def test_failure_deadline_not_renewed(self):
        value = svc.api_failure_receipt({}, {}, {}, svc.API_ERROR_MARKERS[0], 10, 100, 32)
        self.assertEqual(value['deadline_monotonic'], 55)
        self.assertEqual(svc.failure_signal(value, 30, False, False), svc.signal.SIGTERM)
        self.assertEqual(svc.failure_signal(value, 45, True, False), svc.signal.SIGKILL)
        self.assertIsNone(svc.failure_signal(value, 50, True, True))


class Readiness(unittest.TestCase):
    def test_loaded_worker_requires_same_pid_log_and_ownership(self):
        worker = dict(pid=123, start_ticks='20', ppid=111, argv=['VLLM::EngineCore'], cgroup='0::/mock')
        manager = types.SimpleNamespace(service=types.SimpleNamespace(group='/mock'), phase={'log': '/tmp/mock.log'})
        api = dict(pid=111, start_ticks='10')
        with patch.object(ready, 'group_processes', return_value=[worker]), patch.object(ready, 'completed_engine_line', return_value={}), patch.object(ready, 'process_api', return_value=types.SimpleNamespace(proc=lambda pid: worker)):
            observed, proof = ready.plain_worker(manager, api)
            self.assertEqual(observed, worker); self.assertFalse(proof['custom_worker_or_attachment'])
            worker['ppid'] = 112
            with self.assertRaises(ValueError): ready.plain_worker(manager, api)

    def test_no_health_only_plain_ready(self):
        worker = dict(pid=123, start_ticks='20', ppid=111, argv=['VLLM::EngineCore'], cgroup='0::/mock')
        manager = types.SimpleNamespace(service=types.SimpleNamespace(group='/mock'), phase={'log': '/tmp/mock.log'})
        with patch.object(ready, 'group_processes', return_value=[worker]), patch.object(ready, 'completed_engine_line', return_value=None), patch.object(ready, 'process_api', return_value=types.SimpleNamespace(proc=lambda pid: worker)):
            self.assertIsNone(ready.plain_worker(manager, dict(pid=111, start_ticks='10')))

    def test_exact_log_pid_and_completed_load_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'server.log'
            path.write_text('\x1b[1;36m(EngineCore_DP0 pid=123)\x1b[0m INFO [core.py:210] init engine (profile, create kv cache, warmup model) took 12.34 seconds\n')
            proof = ready.completed_engine_line(path, {'pid': 123})
            self.assertIn('warmup model', proof['line'])
            with self.assertRaises(ValueError): ready.completed_engine_line(path, {'pid': 124})

    def test_plain_ready_authenticates_api_environment_and_listener(self):
        spec = spec_fixture(); spec['source_sha256'].update({path: 'a'*64 for path in c.STARTUP_FILES})
        phase = dict(arm='baseline', phase='http', directory=str(w.phase_directory(spec, 'baseline', 'http')),
                     control=str(c.control_root(spec) / 'baseline-http'))
        child = dict(pid=111, start_ticks='10', ppid=90, cgroup='0::/mock', argv=['bootstrap'])
        api = {**child, 'argv': w.api_arguments(spec, 'baseline', 'http')}
        worker = dict(pid=123, start_ticks='20', ppid=111, cgroup='0::/mock', argv=['VLLM::EngineCore'])
        env = w.child_environment(spec, phase)
        manager = types.SimpleNamespace(reject_api_failure=lambda:None, adapter=types.SimpleNamespace(spec=spec, observer=types.SimpleNamespace(check=lambda:None)),
            service=types.SimpleNamespace(group='/mock', live=lambda:None), stage=Path('/tmp/mock-stage'), phase_ref={'mock':True},
            child=child, phase=phase, arm='baseline', name='http', directory=Path(phase['directory']), startup_proof=None)
        intent = dict(phase_spec=manager.phase_ref, bootstrap_owner=child, argv=api['argv'], environment_sha256=c.digest(env))
        response = types.SimpleNamespace(status=200, read=lambda n:b'')
        opener = Mock(); opener.__enter__ = Mock(return_value=response); opener.__exit__ = Mock(return_value=False)
        with patch.object(ready, 'read_if_present', return_value=intent), patch.object(ready, 'process_api', return_value=types.SimpleNamespace(proc=lambda pid:api if pid==111 else worker)), \
             patch.object(ready, 'environment', return_value=env) as actual_env, patch.object(ready.urllib.request, 'urlopen', return_value=opener), \
             patch.object(ready, 'listener_owned', return_value={'mock':True}) as listener, patch.object(ready, 'plain_worker', return_value=(worker, {'loaded_worker':worker})), \
             patch.object(ready, 'manifest_sha', return_value='a'*64), patch.object(ready, 'once'):
            observed = ready.api_ready(manager)
            self.assertEqual(observed['api_owner'], api); self.assertIsNone(observed['attachment'])
            actual_env.return_value = {**env, 'NEURON_CPU_SAMPLER_FAST':'1'}
            with self.assertRaisesRegex(ValueError, 'environment changed'): ready.api_ready(manager)
            actual_env.return_value = env; listener.side_effect = ValueError('listener not owned')
            with self.assertRaisesRegex(ValueError, 'listener not owned'): ready.api_ready(manager)

    def test_duplicate_or_unowned_worker_rejected(self):
        manager = types.SimpleNamespace(service=types.SimpleNamespace(group='/mock'), phase={})
        worker = dict(pid=123, argv=['VLLM::EngineCore'])
        for values in ([], [worker, worker]):
            with patch.object(ready, 'group_processes', return_value=values):
                with self.assertRaises(ValueError): ready.plain_worker(manager, {'pid': 111})


class Storage(unittest.TestCase):
    def fixture(self, temp):
        root = Path(temp); spec = spec_fixture()
        spec['output_root'] = str(root / 'output')
        for key in b.CASE_KEYS: spec['case'][key] = str(root / key)
        control = root / 'control'; reviewed = reviewed_fixture(spec)
        return spec, reviewed, control

    def scan(self, spec, reviewed, control):
        with patch.object(io, 'control_root', return_value=control), patch.object(b, 'control_root', return_value=control), patch.object(b, 'reviewed_budget', return_value=reviewed), patch.object(b, 'unchanged_inputs', side_effect=lambda spec, value: value):
            return io.storage_scan(spec)

    def test_all_six_trees_and_directories_are_charged(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, reviewed, control = self.fixture(temp)
            paths = [Path(spec['output_root']), control, *(Path(spec['case'][k]) for k in b.CASE_KEYS)]
            for path in paths: path.mkdir()
            (Path(spec['case']['artifact']) / 'mock-graph.dat').write_bytes(b'x' * 5000)
            result, inventory = self.scan(spec, reviewed, control)
            self.assertEqual(result['entries'], 7)
            self.assertEqual(result['allocated_bytes'], sum(Path(row[0]).lstat().st_blocks * 512 for row in inventory))
            self.assertGreater(result['classes']['artifact'], 0)

    def test_artifact_symlink_and_hardlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, reviewed, control = self.fixture(temp)
            artifact = Path(spec['case']['artifact']); artifact.mkdir()
            target = Path(temp) / 'target'; target.write_bytes(b'mock')
            link = artifact / 'link'; link.symlink_to(target)
            with self.assertRaises(ValueError): self.scan(spec, reviewed, control)
            link.unlink(); os.link(target, link)
            with self.assertRaises(ValueError): self.scan(spec, reviewed, control)

    def test_view_link_requires_exact_original_target(self):
        spec = spec_fixture(); reviewed = reviewed_fixture(spec)
        path = Path(spec['case']['model_path']) / 'LICENSE'
        row = reviewed['source_files']['LICENSE']
        with patch.object(b.os, 'readlink', return_value=row['identity']['realpath']), patch.object(io, 'identity', return_value=row['identity']), patch.object(Path, 'lstat', return_value=types.SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)):
            self.assertEqual(b.check_view_link(path, spec, reviewed)['target_identity'], row['identity'])
        with patch.object(b.os, 'readlink', return_value='/tmp/arbitrary'):
            with self.assertRaises(ValueError): b.check_view_link(path, spec, reviewed)
        with self.assertRaises(ValueError): b.check_view_link(path.with_name('config.json'), spec, reviewed)

    def test_entry_class_and_single_file_caps_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, reviewed, control = self.fixture(temp)
            artifact = Path(spec['case']['artifact']); artifact.mkdir(); (artifact / 'mock.dat').write_bytes(b'x' * 5000)
            for field, limit in (('single_file_bytes', 1), ('allocated_bytes', 1), ('entries', 1)):
                caps = copy.deepcopy(reviewed); caps['classes']['artifact'][field] = limit
                with self.assertRaises(ValueError): self.scan(spec, caps, control)

    def test_build_logs_do_not_consume_original_science_log_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, reviewed, control = self.fixture(temp)
            stage = control / 'shared-build'; stage.mkdir(parents=True); (stage / 'phase.log').write_bytes(b'mock')
            result, _ = self.scan(spec, reviewed, control)
            self.assertGreater(result['classes']['build_auxiliary'], 0)
            self.assertEqual(result['classes']['all_logs_bytes'], 0)

    def test_only_declared_uuid4_socket_names(self):
        spec = spec_fixture(); root = c.server_temporary_roots(spec)[0]
        self.assertTrue(io.ipc_socket_path(root / c.IPC_UUID, spec))
        self.assertFalse(io.ipc_socket_path(root / 'arbitrary.sock', spec))
        self.assertFalse(io.ipc_socket_path(root / 'nested' / c.IPC_UUID, spec))

    def test_runtime_observation_reserves_remaining_even_without_initial_flag(self):
        spec = spec_fixture(); reviewed = reviewed_fixture(spec)
        usage = {'allocated_bytes': 0, 'classes': {}, 'entries': 0}
        free_nvme = c.LIMITS['nvme_floor_bytes'] + 100
        def statvfs(path):
            free = free_nvme if str(path) == '/mnt/nvme' else 100 * c.GIB
            return types.SimpleNamespace(f_bavail=free, f_frsize=1, f_fsid=1)
        with patch.object(io, 'storage', return_value=usage), patch.object(b, 'reviewed_budget', return_value=reviewed), patch.object(io.os, 'statvfs', side_effect=statvfs):
            with self.assertRaisesRegex(ValueError, 'reservation'): io.resources(spec, reserve=False)

    def test_original_publication_race_only_is_retried(self):
        with patch.object(io, 'storage_scan', side_effect=[io.PublicationRace(), ({'ok': True}, [])]), patch.object(io.time, 'sleep'):
            self.assertEqual(io.storage({}), {'ok': True})
        with patch.object(io, 'storage_scan', side_effect=FileNotFoundError('unknown disappearance')):
            with self.assertRaises(FileNotFoundError): io.storage({})


class FullSequenceMock(unittest.TestCase):
    def exercise(self, temp, fail=None):
        spec = spec_fixture(); spec['output_root'] = str(Path(temp) / 'out'); calls = []
        observer = types.SimpleNamespace(check=lambda **kw: {'admitted': True, 'exclusive': True})
        adapter = types.SimpleNamespace(spec=spec, admission={'attempt_id': 'mock-attempt'}, observer=observer, active=None,
            guard=lambda *args: {'admitted': True, 'exclusive': True})
        def mark(name, arm='shared'):
            calls.append((arm, name))
            if fail == (arm, name): raise ValueError('injected failure')
        class FakePhase:
            def __init__(self, adapter, arm, name, directory, run_id, host=None):
                self.adapter=adapter; self.arm=arm; self.name=name; self.directory=Path(directory); self.run_id=run_id; self.attachment={}
            def leaf(self):
                w.require_phase(spec, self.arm, self.name, self.directory)
                mark(self.name, self.arm)
                return {'mock_only': True, 'actual_service_terminal': {'cleanup_complete': True}}
            def __enter__(self):
                w.require_phase(spec, self.arm, self.name, self.directory); adapter.active=self
                mark(self.name, self.arm); return self.attachment
            def __exit__(self, *args): adapter.active=None
            def reject_api_failure(self): pass
        adapter.server_audit=lambda manager, state, run: {'mock_only': True, 'stop_outcome_accepted': True, 'cleanup_complete': True}
        def stage(name, case, output, guard):
            guard(); io.once(output, {'mock_only': True, 'stage': name}); return {}
        def diagnostic(case, output, guard):
            guard(); io.once(output, {'mock_only': True}); return {}
        def gate(name):
            def call(*args): mark(name); return {'mock_only': True}
            return call
        with patch.object(a, 'Phase', FakePhase), patch.object(w, 'seal_build', return_value={'mock_only': True}), patch.object(w, 'manifest_sha', return_value='a'*64), \
             patch.object(sampler, 'check_capture_attachment', return_value={}), patch.object(sampler, 'run_capture_diagnostic', side_effect=diagnostic), \
             patch.object(sampler, 'check_saved_sampler_gates', side_effect=gate('sampler_pair_gate')), \
             patch.object(http, 'run_http_stage', side_effect=stage), patch.object(http, 'check_matched_hosts', return_value={}), \
             patch.object(http, 'check_saved_quality', side_effect=gate('quality_pair_gate')), \
             patch.object(http, 'compare_saved_pair', side_effect=gate('full_pair_gate')):
            if fail:
                with self.assertRaisesRegex(ValueError, 'injected failure'): w.run(adapter)
                self.assertEqual(calls[-1], fail)
                return calls
            result=w.run(adapter)
        self.assertEqual(calls, list(w.STAGES[:-1]))
        self.assertEqual(result['completed_stages'], 24); self.assertFalse(result['full_model_qualified'])
        self.assertFalse(w.stage_receipt(spec, 25).exists())
        return calls

    def test_complete_24_then_external_stage25_pending(self):
        with tempfile.TemporaryDirectory() as temp: self.exercise(temp)

    def test_failed_build_cannot_enter_science(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(self.exercise(temp, ('shared', 'build')), list(w.STAGES[:2]))

    def test_failed_sampler_gate_cannot_enter_quality_or_http(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(len(self.exercise(temp, ('shared', 'sampler_pair_gate'))), 18)

    def test_failed_quality_gate_cannot_enter_timed_http(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(len(self.exercise(temp, ('shared', 'quality_pair_gate'))), 21)

    def test_missing_prior_stage_blocks_direct_leaf(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = spec_fixture(); spec['output_root']=temp
            with self.assertRaises(FileNotFoundError): w.require_phase(spec, 'baseline', 'numeric', w.phase_directory(spec, 'baseline', 'numeric'))

    def test_vector_return_is_saved_and_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            spec=spec_fixture(); spec['output_root']=temp
            directory=Path(temp)/'baseline/model'; directory.mkdir(parents=True)
            phase=dict(phase='vectors203', arm='baseline', directory=str(directory))
            with patch.object(w, 'require_phase'), patch.object(entry, 'compare_vectors', return_value={'frames':203,'identical_vectors':203}):
                w.leaf(spec, phase, lambda:{'admitted':True,'exclusive':True})
            self.assertTrue((directory/'vectors203.json').exists())

    def test_immutable_graph_or_view_change_is_rejected(self):
        spec = spec_fixture()
        record = {'case': spec['case'], 'manifest_sha256': 'a'*64, 'actual_build_wait_and_drain_observed': True, 'rows': {'graph': {'inode': 1}}}
        with patch.object(w, 'read', return_value=record), patch.object(w, 'ref', return_value={}), patch.object(w, 'manifest_sha', return_value='a'*64), patch.object(w, 'immutable_build_rows', return_value={'graph': {'inode': 2}}):
            with self.assertRaisesRegex(ValueError, 'identities changed'): w.check_build_seal(spec)

    def test_stage25_refuses_unadmitted_external_guard_before_inputs(self):
        from qwen35_package_serial_runtime_v8 import final_audit
        with patch.object(final_audit, 'Bundle') as bundle:
            with self.assertRaisesRegex(ValueError, 'serial audit authority'):
                final_audit.audit({}, {}, {}, lambda: None)
            bundle.assert_not_called()

    def test_stage25_requires_separate_external_actual_tool_exit(self):
        text=(HERE/'final_audit.py').read_text()
        for evidence in ('process_api().gone(admission[\'outer_owner\'])', 'actual_external_tool_result', 'declared_services', 'bundle.physical_pair', 'actual_popen_wait_observed'):
            self.assertIn(evidence, text)
        self.assertIn('full_model_qualified=False', text)


class DelayedLogReadiness(unittest.TestCase):
    LINE = '(EngineCore_DP0 pid=123) INFO [core.py:210] init engine (profile, create kv cache, warmup model) took 1.23 seconds\n'

    def test_delayed_complete_line_becomes_proof_for_same_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase-server.log'; path.write_text('(APIServer pid=111) INFO ready\n')
            self.assertIsNone(ready.completed_engine_line(path, {'pid': 123}))
            with path.open('a') as stream: stream.write(self.LINE)
            self.assertEqual(ready.completed_engine_line(path, {'pid': 123})['worker_pid'], 123)

    def test_partial_forwarded_line_remains_pending_until_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase-server.log'; path.write_text(self.LINE[:-10])
            self.assertIsNone(ready.completed_engine_line(path, {'pid': 123}))
            with path.open('a') as stream: stream.write(self.LINE[-10:])
            self.assertEqual(ready.completed_engine_line(path, {'pid': 123})['worker_pid'], 123)

    def test_duplicate_wrong_pid_or_wrong_rank_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase-server.log'
            for line in (self.LINE * 2, self.LINE.replace('pid=123', 'pid=124'), self.LINE.replace('EngineCore_DP0', 'EngineCore_DP1')):
                path.write_text(line)
                with self.assertRaisesRegex(ValueError, 'unique same-PID'): ready.completed_engine_line(path, {'pid': 123})

    def test_log_file_errors_do_not_become_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'phase-server.log'
            with self.assertRaises(FileNotFoundError): ready.completed_engine_line(path, {'pid': 123})
            target = Path(temp) / 'other.log'; target.write_text(self.LINE); path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'regular server log'): ready.completed_engine_line(path, {'pid': 123})
            path.unlink(); path.write_text(self.LINE)
            with patch.object(ready.os, 'read', return_value=b''):
                with self.assertRaisesRegex(ValueError, 'truncated'): ready.completed_engine_line(path, {'pid': 123})

    def run_api_pending(self, name):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as temp:
            spec = spec_fixture(); spec['output_root'] = temp
            spec['source_sha256'].update({path: 'a'*64 for path in c.STARTUP_FILES})
            directory = w.phase_directory(spec, 'baseline', name); directory.mkdir(parents=True)
            if name == 'capture':
                (directory / 'packets').mkdir(); (directory / 'packets/writer-owner.json').write_text('{}')
            phase = dict(arm='baseline', phase=name, directory=str(directory), control=str(c.control_root(spec) / ('baseline-' + name)), log=str(Path(temp) / 'phase-server.log'))
            child = dict(pid=111, start_ticks='10', ppid=90, cgroup='0::/mock', argv=['bootstrap'])
            api = {**child, 'argv': w.api_arguments(spec, 'baseline', name)}
            worker = dict(pid=123, start_ticks='20', ppid=111, cgroup='0::/mock', argv=['VLLM::EngineCore'])
            with patch.object(w, 'manifest_sha', return_value='a'*64): env = w.child_environment(spec, phase)
            manager = types.SimpleNamespace(reject_api_failure=lambda:None,
                adapter=types.SimpleNamespace(spec=spec, observer=types.SimpleNamespace(check=lambda:None)),
                service=types.SimpleNamespace(group='/mock', live=lambda:None), stage=Path(temp) / 'control',
                phase_ref={'mock':True}, child=child, phase=phase, arm='baseline', name=name, directory=directory, startup_proof=None)
            intent = dict(phase_spec=manager.phase_ref, bootstrap_owner=child, argv=api['argv'], environment_sha256=c.digest(env))
            def metadata(path):
                return {'api-exec-intent.json': intent, 'worker-attachment.json': {'pid':123},
                    'writer-owner.json': {'pid':123, 'arm':'baseline', 'phase':'correctness_capture'}}[path.name]
            opener = Mock(); opener.__enter__ = Mock(return_value=types.SimpleNamespace(status=200, read=lambda n:b'')); opener.__exit__ = Mock(return_value=False)
            with ExitStack() as stack:
                stack.enter_context(patch.object(ready, 'read_if_present', side_effect=metadata))
                stack.enter_context(patch.object(ready, 'process_api', return_value=types.SimpleNamespace(proc=lambda pid:api if pid == 111 else worker)))
                stack.enter_context(patch.object(ready, 'environment', return_value=env))
                stack.enter_context(patch.object(ready.urllib.request, 'urlopen', return_value=opener))
                stack.enter_context(patch.object(ready, 'listener_owned', return_value={'mock':True}))
                stack.enter_context(patch.object(ready, 'group_processes', return_value=[worker]))
                stack.enter_context(patch.object(ready, 'completed_engine_line', side_effect=[None, {'worker_pid':123}]))
                stack.enter_context(patch.object(ready, 'manifest_sha', return_value='a'*64))
                stack.enter_context(patch.object(w, 'manifest_sha', return_value='a'*64))
                stack.enter_context(patch.object(sampler, 'check_capture_attachment', return_value={}))
                publish = stack.enter_context(patch.object(ready, 'once'))
                self.assertIsNone(ready.api_ready(manager)); self.assertIsNone(manager.startup_proof); publish.assert_not_called()
                observed = ready.api_ready(manager)
                self.assertEqual(observed['worker'], worker); self.assertEqual(observed['api_owner'], api)
                self.assertIsNotNone(manager.startup_proof); publish.assert_called_once()

    def test_plain_pending_propagates_before_startup_proof(self):
        self.run_api_pending('http')

    def test_capture_pending_propagates_before_startup_proof(self):
        self.run_api_pending('capture')

    def test_original_phase_wait_retries_pending_without_cleanup(self):
        phase = object.__new__(a.Phase)
        phase.start = Mock(); phase.finish = Mock(); phase.attachment = 'observed attachment'
        phase.adapter = types.SimpleNamespace(observer=types.SimpleNamespace(check=lambda:None))
        phase.api_ready = Mock(side_effect=[None, {'ready': True}])
        with patch.object(a, 'endpoint_open', return_value=False), patch.object(a, 'wait_until', wraps=io.wait_until) as wait, patch.object(io.time, 'sleep'):
            self.assertEqual(phase.__enter__(), phase.attachment)
            self.assertEqual(wait.call_args.args[2], 14100); self.assertEqual(phase.api_ready.call_count, 2)
            phase.start.assert_called_once(); phase.finish.assert_not_called()

    def test_pending_deadline_failure_still_enters_original_cleanup(self):
        phase = object.__new__(a.Phase)
        phase.start = Mock(); phase.finish = Mock(); phase.attachment = None
        phase.adapter = types.SimpleNamespace(observer=types.SimpleNamespace(check=lambda:None))
        phase.api_ready = Mock(return_value=None)
        with patch.object(a, 'endpoint_open', return_value=False), patch.object(io.time, 'monotonic', side_effect=[100, 14201]):
            with self.assertRaisesRegex(ValueError, 'actual package server'): phase.__enter__()
        phase.finish.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
