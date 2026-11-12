# SPDX-License-Identifier: Apache-2.0
"""Standard-library AST and mocked control-flow tests; never imports native libraries."""
import ast
import builtins
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from vllm_neuron.models.qwen35 import register, identity as i, prepare_view, helpers, state, validation as q
from vllm_neuron.models.qwen35 import artifacts as a
from vllm_neuron.models.qwen35.provenance import ORIGINAL_CONFIG, EXPECTED_MODEL
P = Path(i.__file__).parent
CONTRACT = json.loads(Path(__file__).with_name('core_contract.json').read_text())


def node(filename, name):
    return next(n for n in ast.parse((P / filename).read_text()).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)


class UndoSplit(ast.NodeTransformer):
    def visit_Call(self, value):
        self.generic_visit(value)
        if isinstance(value.func, ast.Name) and value.func.id == 'split_last_dim':
            assert len(value.args) == 2 and not value.keywords
            return ast.Call(func=ast.Attribute(value=ast.Name(id='torch', ctx=ast.Load()), attr='split', ctx=ast.Load()),
                            args=value.args, keywords=[ast.keyword(arg='dim', value=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(1)))])
        return value


def digest(value):
    return hashlib.sha256(ast.dump(UndoSplit().visit(value), include_attributes=False).encode()).hexdigest()


class ScientificSourceTests(unittest.TestCase):
    def test_every_original_math_and_adapter_body_is_preserved(self):
        for row in CONTRACT['entries']:
            with self.subTest(name=row['name']):
                self.assertEqual(digest(node(row['target'], row['name'])), row['ast_sha256'])

    def test_mutated_dtype_split_index_and_decay_are_detected(self):
        for filename, name, old, new in [
            ('modeling.py', 'HybridCache', 'dtypes = [nc.torch_dtype, torch.float32]', 'dtypes = [nc.torch_dtype, torch.bfloat16]'),
            ('modeling.py', 'NeuronQwen35Model', 'self.n_positions:', 'self.n_positions + 1:'),
            ('text_core.py', 'LinearAttention', '[self.key_dim, self.key_dim, self.value_dim]', '[self.value_dim, self.key_dim, self.key_dim]'),
            ('primitives.py', 'delta_prefill', 'running = state.float()', 'running = state.to(dtype)')]:
            source = (P / filename).read_text(); self.assertIn(old, source)
            tree = ast.parse(source.replace(old, new, 1))
            changed = next(x for x in tree.body if getattr(x, 'name', None) == name)
            expected = next(r['ast_sha256'] for r in CONTRACT['entries'] if r['name'] == name)
            self.assertNotEqual(digest(changed), expected)

    def test_exact_one_split_no_dynamic_rewrite_or_legacy_import(self):
        calls = []
        for source in P.glob('*.py'):
            tree = ast.parse(source.read_text())
            for item in ast.walk(tree):
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                    self.assertNotIn(item.func.id, ('eval', 'exec'))
                    if item.func.id == 'split_last_dim': calls.append(item)
                if isinstance(item, ast.ImportFrom) and not item.level:
                    self.assertFalse((item.module or '').startswith(('qwen35_', 'frontend_split_', 'orchestrate')))
            self.assertNotIn('/home/ubuntu/experiments', source.read_text())
        self.assertEqual(len(calls), 1)

    def test_disabled_registration_is_native_free(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            self.assertFalse(name.startswith(('torch', 'neuron', 'transformers', 'vllm.')))
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {}, clear=True), patch('builtins.__import__', side_effect=guarded):
            register()

    def test_split_boundaries_and_api(self):
        self.assertEqual(helpers.boundaries([1024, 1024, 1024]), (1024, 2048))
        calls = []
        tensor = SimpleNamespace(shape=(4, 1, 3072))
        with patch.dict(sys.modules, {'torch': SimpleNamespace(tensor_split=lambda *args, **kw: calls.append((args, kw)))}):
            helpers.split_last_dim(tensor, [1024] * 3)
            self.assertEqual(calls, [((tensor, (1024, 2048)), {'dim': -1})])
            with self.assertRaises(AssertionError): helpers.split_last_dim(tensor, [1024, 1024])
        for widths in ([0, 2], [1], [True, 2]):
            with self.assertRaises(AssertionError): helpers.boundaries(widths)

    def test_original_config_transform_and_checkpoint_identity(self):
        original = copy.deepcopy(ORIGINAL_CONFIG)
        actual = prepare_view.compatible_model_config(original)
        self.assertEqual(original, ORIGINAL_CONFIG)
        self.assertEqual(actual, EXPECTED_MODEL)
        self.assertEqual(hashlib.sha256(json.dumps(actual, indent=2).encode()).hexdigest(), i.CONFIG_VIEW_SHA256)
        self.assertEqual(i.SOURCE_FILES['model.safetensors-00001-of-00001.safetensors']['digest'],
                         'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1')
        original['text_config']['hidden_size'] = 1
        with self.assertRaises(i.AdmissionError): prepare_view.compatible_model_config(original)

    def test_single_token_prefill_uses_original_two_token_route(self):
        class Input:
            def __init__(self, rows, dtype='int64'): self.rows, self.shape, self.dtype = rows, (len(rows), len(rows[0])), dtype
            def __eq__(self, value): return SimpleNamespace(all=lambda: all(x == value for r in self.rows for x in r))
            def expand(self, b, t): return Input([self.rows[0] for _ in range(b)], self.dtype)
        calls = []
        class Base:
            def forward(self, **kwargs): calls.append(kwargs); return kwargs
        def pad(input_value, shape, value_to_add=0, **kw):
            fill = kw.get('value', value_to_add)
            return Input([r + [fill] * shape[1] for r in input_value.rows], input_value.dtype)
        namespace = {'NeuronBaseForCausalLM': Base, 'NeuronQwen35Model': object(),
                     'F': SimpleNamespace(pad=pad), 'torch': SimpleNamespace(tensor=lambda rows, dtype: Input([rows], dtype))}
        exec(compile(ast.Module(body=[node('modeling.py', 'NeuronQwen35ForCausalLM')], type_ignores=[]), '<source-only>', 'exec'), namespace)
        model = namespace['NeuronQwen35ForCausalLM'](); model.config = SimpleNamespace(pad_token_id=248044)
        ids = Input([[7]]); mask = Input([[1]])
        result = model.forward(input_ids=ids, position_ids=Input([[0]]), attention_mask=mask)
        self.assertEqual(result['input_ids'].rows, [[7, 248044]])
        self.assertEqual(result['attention_mask'].rows, [[1, 0]])
        result = model.forward(input_ids=ids, position_ids=Input([[1]]), attention_mask=mask)
        self.assertIs(result['input_ids'], ids)
        with self.assertRaises(ValueError): model.forward(input_ids=ids)

    def test_slot_writer_preserves_inactive_and_other_request_histories(self):
        class Array:
            def __init__(self, rows): self.rows, self.shape, self.ndim, self.device = rows, (len(rows), 2), 2, 'mock'
            def __getitem__(self, key): return Array(self.rows[key]) if isinstance(key, slice) else self.rows[key]
        class Mask:
            def __init__(self, flags): self.flags = flags
            def reshape(self, shape): return self
        class Indices:
            def __init__(self, count): self.count = count
            def __eq__(self, slot): return Mask([n == slot for n in range(self.count)])
        fake = SimpleNamespace(arange=lambda count, device: Indices(count),
            where=lambda mask, update, cache: Array([copy.deepcopy(update.rows[0] if flag else old) for flag, old in zip(mask.flags, cache.rows)]))
        namespace = {'torch': fake}
        exec(compile(ast.Module(body=[node('text_core.py', 'store_slots')], type_ignores=[]), '<source-only>', 'exec'), namespace)
        write = namespace['store_slots']; initial = Array([[10, 11], [20, 21], [30, 31], [40, 41]])
        result = write(initial, Array([[1, 2], [90, 91], [3, 4]]), [3, -1, 1])
        self.assertEqual(result.rows, [[10, 11], [3, 4], [30, 31], [1, 2]])
        result = write(result, Array([[0, 0]]), [3])
        self.assertEqual(result.rows, [[10, 11], [3, 4], [30, 31], [0, 0]])
        self.assertEqual(initial.rows[3], [40, 41])


class RegistrationTests(unittest.TestCase):
    def prepared(self, collision=None):
        calls = []
        model = object(); config = SimpleNamespace(model_type='qwen35_neuron_text'); wrapper = object()
        types = {} if collision != 'nxdi' else {'qwen35neurontext': {'causal-lm': object()}}
        constants = ModuleType('neuronx_distributed_inference.utils.constants'); constants.MODEL_TYPES = types
        def install_nxdi():
            types['qwen35neurontext'] = {'causal-lm': model}; calls.append('nxdi')
        ns = dict(_INSTALLED=False, selected_variant=lambda: i.VARIANT, require=i.require, SCHEMA=i.SCHEMA,
            loader=SimpleNamespace(QWEN35_ARTIFACT_ADMISSION_SCHEMA=i.SCHEMA if collision != 'loader' else 'wrong'),
            CONFIG_MAPPING={'qwen35_neuron_text': object()} if collision == 'config' else {},
            ModelRegistry=SimpleNamespace(get_supported_archs=lambda: ['Qwen35NeuronTextForCausalLM'] if collision == 'vllm' else [],
                register_model=lambda *args: calls.append('vllm')),
            AutoConfig=SimpleNamespace(register=lambda *args: calls.append('config')),
            NeuronQwen35ForCausalLM=model, Qwen35NeuronTextConfig=config, Qwen35NeuronTextForCausalLM=wrapper,
            install_nxdi=install_nxdi)
        exec(compile(ast.Module(body=[node('registration.py', 'install')], type_ignores=[]), '<source-only>', 'exec'), ns)
        return ns, {constants.__name__: constants}, calls

    def test_collisions_fail_before_any_registration_mutation(self):
        for collision in ('nxdi', 'config', 'vllm', 'loader'):
            ns, modules, calls = self.prepared(collision)
            with patch.dict(sys.modules, modules), self.assertRaises(i.AdmissionError): ns['install']()
            self.assertEqual(calls, [])

    def test_install_is_idempotent_and_spawn_can_register_again(self):
        for _ in range(2):
            ns, modules, calls = self.prepared()
            with patch.dict(sys.modules, modules): ns['install'](); ns['install']()
            self.assertEqual(calls, ['nxdi', 'config', 'vllm'])


class ArtifactLifecycleTests(unittest.TestCase):
    def test_compile_publishes_only_after_success_and_no_failure_retry(self):
        from contextlib import nullcontext
        for fail in ('compile', 'load', None):
            calls = []
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'artifact'
                config = SimpleNamespace(to_json_string=lambda: '{"metadata": null}')
                identity = {'configuration': {'metadata': None}, 'variant': i.VARIANT}
                class Model:
                    def __init__(self, *args): calls.append('construct')
                    def compile(self, directory):
                        calls.append('compile')
                        if fail == 'compile': raise RuntimeError('mock compile failure')
                    def load(self, directory):
                        calls.append('load')
                        self_manifest = Path(directory) / i.ARTIFACT_MANIFEST
                        assert not self_manifest.exists()
                        if fail == 'load': raise ValueError('mock load failure')
                owner = SimpleNamespace()
                env = {'NEURON_COMPILED_ARTIFACTS': str(path), 'NEURON_QWEN35_ALLOW_COMPILE': '1'}
                with patch.dict(os.environ, env, clear=True), patch.object(a, 'build_identity', return_value=identity), patch.object(a, 'serial_compiler', return_value=nullcontext()), patch.object(a, 'admit_saved_configuration'), patch.object(a, 'artifact_files', return_value={'model.pt': 'mock text digest', 'neuron_config.json': 'mock text digest'}):
                    if fail:
                        with self.assertRaises((RuntimeError, ValueError)): a.load_or_compile(owner, 'fake', Model, config, {})
                        self.assertFalse((path / i.ARTIFACT_MANIFEST).exists())
                        with self.assertRaises(i.AdmissionError): a.load_or_compile(owner, 'fake', Model, config, {})
                    else:
                        result = a.load_or_compile(owner, 'fake', Model, config, {})
                        self.assertEqual(result, (True, str(path), config))
                        self.assertTrue(json.loads((path / i.ARTIFACT_MANIFEST).read_text())['complete'])
                self.assertEqual(calls.count('compile'), 1)

    def test_cached_load_valueerror_propagates_without_compile_fallback(self):
        calls = []
        def failed_load(*args): calls.append('load'); raise ValueError('corrupted mock runtime')
        owner = SimpleNamespace(_load_compiled_model=failed_load)
        config = SimpleNamespace(to_json_string=lambda: '{"metadata": null}')
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'NEURON_COMPILED_ARTIFACTS': temporary}, clear=True), patch.object(a, 'build_identity', return_value={}), patch.object(a, 'admit_existing'):
            with self.assertRaises(ValueError): a.load_or_compile(owner, 'fake', lambda *args: self.fail('compile called'), config, {})
        self.assertEqual(calls, ['load'])


class FakeState:
    def __init__(self, shape, dtype, rank, ident):
        self.shape, self.dtype, self.device, self._cdata = shape, dtype, SimpleNamespace(type='privateuseone', index=rank), ident
        self.dirty, self.copies = True, 0
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def view(self, dtype): return self
    def numpy(self): return SimpleNamespace(tobytes=lambda: b'dirty' if self.dirty else b'zero')
    def copy_(self, value): self.dirty = value.dirty; self.copies += 1
    def numel(self): return math.prod(self.shape)
    def element_size(self): return 4 if self.dtype == 'torch.float32' else 2


def mock_states():
    states = []
    for rank in (0, 1):
        row = {}
        for layer, kind in enumerate(EXPECTED_MODEL['layer_types']):
            pairs = [((4, 3072, 4), 'torch.bfloat16'), ((4, 8, 128, 128), 'torch.float32')] if kind == 'linear_attention' else [((4, 1, 4096, 256), 'torch.bfloat16')] * 2
            for offset, (shape, dtype) in enumerate(pairs):
                index = layer * 2 + offset
                row['kv_mgr.past_key_values.' + str(index)] = FakeState(shape, dtype, rank, 1 + rank * 48 + index)
        states.append(row)
    return SimpleNamespace(state=states)


class StateTests(unittest.TestCase):
    def test_clear_is_raw_zero_in_place_without_graph_initialization(self):
        model = mock_states(); before = state.binding_metadata(model)
        objects = [id(v) for row in model.state for v in row.values()]
        def zero(shape, dtype, device):
            self.assertEqual(device, 'cpu')
            out = FakeState(shape, dtype, 0, 0); out.dirty = False; return out
        fake = SimpleNamespace(zeros=zero, uint8='uint8', count_nonzero=lambda v: int(v.dirty))
        with patch.dict(sys.modules, {'torch': fake}):
            for clear in range(2):
                for row in model.state:
                    for value in row.values(): value.dirty = True
                result = state.clear_states(model, before)
                self.assertEqual(result['states_written'], 96)
                self.assertEqual(result['states_nonzero_before_clear'], 96)
                self.assertTrue(result['bindings_unchanged'])
                self.assertFalse(result['graph_initialization_called'])
                self.assertEqual(objects, [id(v) for row in model.state for v in row.values()])
                self.assertTrue(all(not v.dirty and v.copies == clear + 1 for row in model.state for v in row.values()))
            with self.assertRaises(i.AdmissionError): state.clear_states(model, [])

    def test_failed_device_copy_is_detected(self):
        model = mock_states()
        first = next(iter(model.state[0].values())); first.copy_ = lambda value: None
        def zero(shape, dtype, device):
            out = FakeState(shape, dtype, 0, 0); out.dirty = False; return out
        with patch.dict(sys.modules, {'torch': SimpleNamespace(zeros=zero, uint8='uint8', count_nonzero=lambda v: int(v.dirty))}):
            with self.assertRaises(i.AdmissionError): state.clear_states(model)


class AdmissionTests(unittest.TestCase):
    def test_config_comparison_preserves_types_and_rejects_nonfinite(self):
        self.assertNotEqual(i.canonical_configuration({'x': True}), i.canonical_configuration({'x': 1}))
        self.assertNotEqual(i.canonical_configuration({'x': 1}), i.canonical_configuration({'x': 1.0}))
        i.canonical_configuration({'neuron_config': {'quantize_clamp_bound': float('inf')}})
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(i.AdmissionError): i.canonical_configuration({'x': value})

    def test_identity_comparison_keeps_sdk_infinity_scoped_and_boolean_distinct(self):
        valid = {'configuration': {'metadata': None, 'neuron_config': {'quantize_clamp_bound': float('inf')}}, 'batch': 1}
        self.assertTrue(i.same_identity(valid, copy.deepcopy(valid)))
        self.assertFalse(i.same_identity(valid, {**valid, 'batch': True}))
        with self.assertRaises(i.AdmissionError):
            i.same_identity({**valid, 'outside': float('inf')}, {**valid, 'outside': float('inf')})

    def test_existing_view_rejected_before_hashing(self):
        with tempfile.TemporaryDirectory() as path, patch.object(prepare_view, 'verify_files') as verify:
            with self.assertRaises(i.AdmissionError): prepare_view.prepare(path, path)
            verify.assert_not_called()

    def test_scope_rejects_dimensions_quantization_sampler_and_missing_bucket(self):
        neuron = dict(tp_degree=2, batch_size=4, max_batch_size=4, tkg_batch_size=4, ctx_batch_size=1,
            seq_len=4096, max_length=4096, max_context_length=4096, padding_side='right', apply_seq_ids_mask=True,
            is_continuous_batching=True, enable_bucketing=True, buckets=[4096], context_encoding_buckets=i.BUCKETS,
            save_sharded_checkpoint=False, skip_sharding=False, skip_warmup=False, logical_nc_config=1,
            start_rank_id=0, local_ranks_size=2, ep_degree=1, world_size=2, torch_dtype='bfloat16')
        # Execute the actual pinned SDK normalizer with fake HF/config classes.
        fixture_path = Path(__file__).with_name('hf_config_loader_fixture.py')
        fixture_tree = ast.parse(fixture_path.read_text())
        fixture_fn = next(n for n in fixture_tree.body if isinstance(n, ast.FunctionDef))
        expected = json.loads(fixture_path.with_suffix('.json').read_text())
        self.assertEqual(hashlib.sha256(ast.dump(fixture_fn, include_attributes=False).encode()).hexdigest(),
                         expected['function_ast_sha256'])
        class HF:
            transformers_version = None
            def __init__(self): self.__dict__.update(copy.deepcopy(EXPECTED_MODEL))
            def to_dict(self): return copy.deepcopy(EXPECTED_MODEL)
        normalized = SimpleNamespace(neuron_config=SimpleNamespace(overrides_torch_dtype=True, torch_dtype='bfloat16'))
        namespace = {'SimpleNamespace': SimpleNamespace, 'PretrainedConfig': HF,
                     'to_torch_dtype': lambda value: value}
        exec(compile(fixture_tree, '<actual-sdk-normalizer-mocked>', 'exec'), namespace)
        namespace['load_pretrained_config'](hf_config=HF())(normalized)
        self.assertNotIn('dtype', vars(normalized)); self.assertNotIn('torch_dtype', vars(normalized))
        normalized_model = vars(normalized)
        i.validate_scope(normalized_model, neuron)
        for key, value in [('tp_degree', 1), ('max_length', 2048), ('context_encoding_buckets', [128, 4096]),
                           ('torch_dtype', 'float32'), ('quantized', True), ('on_device_sampling_config', {})]:
            with self.subTest(field=key), self.assertRaises(i.AdmissionError):
                i.validate_scope(normalized_model, {**neuron, key: value})

    def test_loader_branch_precedes_generic_deletion_and_fallback(self):
        path = P.parents[1] / 'worker/neuronx_distributed_model_loader.py'
        tree = ast.parse(path.read_text()); cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NeuronModelBase')
        method = next(n for n in cls.body if getattr(n, 'name', None) == '_load_weights_common')
        branch = next(n for n in method.body if isinstance(n, ast.If) and 'qwen35_neuron_text' in ast.unparse(n.test))
        import_node, return_node = branch.body
        self.assertEqual(import_node.module, 'vllm_neuron.models.qwen35.artifacts')
        self.assertIsInstance(return_node, ast.Return)
        self.assertLess(branch.lineno, next(n.lineno for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == '_get_compiled_model_path'))
        self.assertNotIn('except', ast.unparse(branch))

    def test_source_identity_includes_runner_loader_and_existing_sampler(self):
        paths = i.source_identity()
        for name in ['worker/cpu_sampler.py', 'worker/neuronx_distributed_model_loader.py',
                     'worker/neuronx_distributed_model_runner.py', 'models/qwen35/primitives.py']:
            self.assertIn(name, paths)
        self.assertFalse((P / 'cpu_sampler.py').exists())

    def test_host_sampler_arms_share_one_compiled_identity(self):
        env = {'NEURON_QWEN35': '1', 'NEURON_ON_DEVICE_SAMPLING_DISABLED': '1',
               'XLA_DISABLE_FUNCTIONALIZATION': '1', 'NEURON_CPU_SAMPLER_FAST': '0'}
        with patch.dict(os.environ, env, clear=True), patch.object(i, 'validate_scope'), patch.object(i, 'verify_view', return_value={'pinned': True}), patch.object(i, 'software_identity', return_value={'pinned': True}), patch.object(i, 'hardware_identity', return_value={'pinned': True}), patch.object(i, 'source_identity', return_value={'worker/cpu_sampler.py': 'same-source-sha256'}):
            original = i.build_identity('mock-view', {'metadata': None})
            os.environ['NEURON_CPU_SAMPLER_FAST'] = '1'
            fast = i.build_identity('mock-view', {'metadata': None})
        self.assertEqual(original, fast)
        self.assertNotIn('sampler_fast', original)
        self.assertIn('worker/cpu_sampler.py', original['sources'])

    def test_saved_configuration_is_exact_except_embedded_metadata(self):
        identity = {'configuration': {'metadata': None, 'neuron_config': {'quantize_clamp_bound': float('inf')}, 'batch': 4}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'neuron_config.json'
            saved = {**identity['configuration'], 'metadata': {'qwen35_ultra': identity}}
            path.write_text(json.dumps(saved)); a.admit_saved_configuration(temporary, identity)
            saved['batch'] = 1; path.write_text(json.dumps(saved))
            with self.assertRaises(i.AdmissionError): a.admit_saved_configuration(temporary, identity)

    def test_legacy_incomplete_mismatched_artifact_never_loads(self):
        owner = SimpleNamespace(_load_compiled_model=lambda *args: self.fail('must reject before native load'))
        config = SimpleNamespace(to_json_string=lambda: json.dumps({'metadata': None}))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'NEURON_COMPILED_ARTIFACTS': temporary}, clear=True), patch.object(a, 'build_identity', return_value={'new': True}):
            path = Path(temporary) / i.ARTIFACT_MANIFEST
            for manifest in (None, {'complete': False}, {'complete': True, 'identity': {'old': True}, 'files': {}}):
                if manifest is not None: path.write_text(json.dumps(manifest))
                with self.assertRaises(i.AdmissionError): a.load_or_compile(owner, 'fake', object(), config, {})

    def test_no_missing_artifact_created_without_compile_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fresh'
            config = SimpleNamespace(to_json_string=lambda: '{"metadata": null}')
            with patch.dict(os.environ, {'NEURON_COMPILED_ARTIFACTS': str(path)}, clear=True), patch.object(a, 'build_identity', return_value={}):
                with self.assertRaises(i.AdmissionError): a.load_or_compile(object(), 'fake', object(), config, {})
            self.assertFalse(path.exists())

    def test_long_protocol_and_strict_numerical_gate(self):
        frames = q.expected_long_frames()
        self.assertEqual(len(frames), 87)
        self.assertEqual(frames[-1], {'fixture': 1, 'position': 32, 'slot': 3, 'phase': 'resume_inactive'})
        row = dict(cosine=.999, kl_reference_to_neuron=.001, max_abs_error=.1, reference_margin=1., argmax_match=True)
        q.strict_rows([row] * 32, 32)
        q.strict_rows([{**row, "kl_reference_to_neuron": -2.19e-9}] * 32, 32)
        for key, bad in [('cosine', .99), ('kl_reference_to_neuron', .03), ('max_abs_error', 2.1), ('argmax_match', False), ('cosine', float('nan'))]:
            with self.assertRaises(AssertionError): q.strict_rows([{**row, key: bad}] * 32, 32)



if __name__ == '__main__':
    unittest.main()
