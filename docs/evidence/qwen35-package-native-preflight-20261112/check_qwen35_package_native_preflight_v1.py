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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
WORKER = ROOT / 'qwen35_package_native_preflight_v1.py'
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
