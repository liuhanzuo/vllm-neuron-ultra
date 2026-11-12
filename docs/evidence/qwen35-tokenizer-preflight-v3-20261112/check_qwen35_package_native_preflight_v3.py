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
WORKER = ROOT / 'qwen35_package_native_preflight_v3.py'
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
spec = importlib.util.spec_from_file_location('qwen_preflight_prepare', ROOT / 'prepare_qwen35_package_native_preflight_v3.py')
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
        source = WORKER.read_text().replace('qwen35_package_tokenizer_checks_v3.py', 'qwen35_package_tokenizer_checks_v2.py').replace('qwen35_tokenizer_preflight_v3', 'qwen35_tokenizer_preflight_v2')
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
