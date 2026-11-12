"""SOURCE/STDLIB/MOCK checks only; no tensor/native/model/graph execution."""
import ast
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
Q = HERE.parent
PACKAGE = HERE / 'vllm_neuron/models/qwen35'
PLAN = json.loads((Q / 'qwen35-cache-writer-package-source-plan-20261112-v1.json').read_text())
ORIGINAL = Path(PLAN['sources']['package_text']['path']).parent


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'neuronxcc', 'torch_neuronx',
                'neuronx_distributed', 'neuronx_distributed_inference', 'transformers', 'vllm', 'vllm_neuron'):
            raise AssertionError('native/model import forbidden: ' + fullname)


sys.meta_path.insert(0, NoNative())
PACKAGE_NAME = '_isolated_qwen_writer'
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE), str(ORIGINAL)]
sys.modules[PACKAGE_NAME] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE_NAME + '.' + name, PACKAGE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


identity = load('identity')
writer = load('cache_writer')


def config(prefill=False):
    nc = types.SimpleNamespace(tp_degree=2, max_batch_size=4, ctx_batch_size=1,
        tkg_batch_size=4, max_length=4096, batch_size=1 if prefill else 4,
        is_prefill_stage=prefill, torch_dtype='torch.bfloat16')
    return types.SimpleNamespace(**copy.deepcopy(writer.EXPECTED_MODEL), neuron_config=nc,
        metadata={'qwen35_ultra': {'schema': identity.SCHEMA, 'cache_writer': writer.cache_writer_identity()}})


class TensorShape:
    def __init__(self, shape, dtype, root=None):
        self.shape, self.dtype = tuple(shape), dtype
        self.root = root if root is not None else self
        self.calls = []
    @property
    def device(self):
        raise AssertionError('device must not select trace backend')
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        self.root.calls.append(tuple(shape))
        if -1 in shape:
            elements = 1
            for n in self.shape: elements *= n
            shape = tuple(elements // shape[0] if n == -1 else n for n in shape)
        return TensorShape(shape, self.dtype, self.root)


class SelectionAndIdentity(unittest.TestCase):
    def test_import_and_default_do_not_import_native(self):
        self.assertFalse(any(n.split('.')[0] in ('torch', 'numpy', 'neuronxcc') for n in sys.modules))
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(writer.selected_cache_writer(), writer.ORIGINAL)
            self.assertEqual(writer.cache_writer_identity()['decode_writer'], writer.ORIGINAL)

    def test_unknown_modes_fail(self):
        for mode in ('', 'auto', 'nki', 'NKI', 'torch_where '):
            with self.subTest(mode=mode), patch.dict(os.environ, {writer.WRITER_ENV: mode}):
                with self.assertRaises(identity.AdmissionError): writer.selected_cache_writer()

    def test_original_and_prefill_bind_original_callable(self):
        original = lambda *args: 'original'
        for mode in (writer.ORIGINAL, writer.CANDIDATE):
            for prefill in (False, True):
                with self.subTest(mode=mode, prefill=prefill), patch.dict(os.environ, {writer.WRITER_ENV: mode}):
                    result = writer.resolve_cache_writer(config(prefill), prefill, original)
                    self.assertIs(result, writer.store_slots_nki_decode if mode == writer.CANDIDATE and not prefill else original)

    def test_parent_spawn_mode_or_source_mismatch_rejected_in_both_phases(self):
        for prefill in (False, True):
            with patch.dict(os.environ, {writer.WRITER_ENV: writer.CANDIDATE}): c = config(prefill)
            with patch.dict(os.environ, {writer.WRITER_ENV: writer.ORIGINAL}):
                with self.assertRaises(identity.AdmissionError): writer.resolve_cache_writer(c, prefill, lambda: None)
            with patch.dict(os.environ, {writer.WRITER_ENV: writer.CANDIDATE}):
                c.metadata['qwen35_ultra']['cache_writer']['source_sha256']['cache_writer.py'] = '0' * 64
                with self.assertRaises(identity.AdmissionError): writer.resolve_cache_writer(c, prefill, lambda: None)

    def test_missing_old_or_changed_metadata_rejected(self):
        with patch.dict(os.environ, {writer.WRITER_ENV: writer.ORIGINAL}):
            for metadata in (None, {}, {'qwen35_ultra': {'schema': identity.SCHEMA}},
                             {'qwen35_ultra': {'schema': 'foreign', 'cache_writer': writer.cache_writer_identity()}}):
                c = config(); c.metadata = metadata
                with self.assertRaises(identity.AdmissionError): writer.resolve_cache_writer(c, False, lambda: None)

    def test_static_tp_batch_precision_layer_layout_and_phase_are_exact(self):
        with patch.dict(os.environ, {writer.WRITER_ENV: writer.CANDIDATE}):
            for name, value in (('tp_degree', 1), ('max_batch_size', 8), ('batch_size', 1),
                                ('ctx_batch_size', 4), ('max_length', 2048), ('torch_dtype', 'torch.float32'),
                                ('is_prefill_stage', 0)):
                c = config(); setattr(c.neuron_config, name, value)
                with self.subTest(name=name), self.assertRaises(identity.AdmissionError):
                    writer.resolve_cache_writer(c, False, lambda: None)
            c = config(); c.layer_types = ['linear_attention'] * 24
            with self.assertRaises(identity.AdmissionError): writer.resolve_cache_writer(c, False, lambda: None)

    def test_parent_identity_records_writer_and_rejects_old_or_cross_arm_artifact(self):
        with patch.object(identity, 'selected_variant', return_value='original'), \
             patch.object(identity, 'validate_scope'), patch.object(identity, 'verify_view', return_value={}), \
             patch.object(identity, 'software_identity', return_value={}), \
             patch.object(identity, 'hardware_identity', return_value={}), \
             patch.object(identity, 'source_identity', return_value={}):
            results = []
            for mode in (writer.ORIGINAL, writer.CANDIDATE):
                with patch.dict(os.environ, {writer.WRITER_ENV: mode}):
                    results.append(identity.build_identity('/not-read', {'metadata': None, 'neuron_config': {}}))
        self.assertFalse(identity.same_identity(*results))
        old = dict(results[0]); old.pop('cache_writer')
        self.assertFalse(identity.same_identity(old, results[0]))
        actual = copy.deepcopy(results[0]); actual['cache_writer']['source_sha256']['slot_cache_nki_predicated.py'] = '0' * 64
        self.assertFalse(identity.same_identity(actual, results[0]))
        self.assertEqual({k: v for k, v in results[0].items() if k != 'cache_writer'},
                         {k: v for k, v in results[1].items() if k != 'cache_writer'})

    def test_new_top_level_modules_are_in_existing_package_source_closure(self):
        with patch.object(identity, 'digest_file', side_effect=lambda path: 'f' * 64):
            pins = identity.source_identity()
        self.assertIn('models/qwen35/cache_writer.py', pins)
        self.assertIn('models/qwen35/slot_cache_nki_predicated.py', pins)

    def test_binding_is_static_and_forward_never_reselects_environment(self):
        with patch.dict(os.environ, {writer.WRITER_ENV: writer.CANDIDATE}):
            bound = writer.resolve_cache_writer(config(), False, lambda: None)
        with patch.dict(os.environ, {writer.WRITER_ENV: 'invalid-after-binding'}):
            self.assertIs(bound, writer.store_slots_nki_decode)
            # The callable contains no environment or backend-selection logic.
            fn = next(n for n in ast.parse((PACKAGE/'cache_writer.py').read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'store_slots_nki_decode')
            self.assertFalse(any(isinstance(n, ast.Name) and n.id in ('os', 'selected_cache_writer', 'cache_writer_identity')
                                 for n in ast.walk(fn)))


class DecodeBoundary(unittest.TestCase):
    def invoke(self, shape, dtype='torch.bfloat16', slots_dtype='torch.int32', update_shape=None):
        cache = TensorShape(shape, dtype); updates = TensorShape(update_shape or shape, dtype)
        slots = TensorShape((4,), slots_dtype); calls = []
        fake = types.ModuleType(PACKAGE_NAME + '.slot_cache_nki_predicated')
        def kernel(c, u, s):
            calls.append((c, u, s))
            return TensorShape(c.shape, c.dtype)
        fake.slot_cache_gather_predicated = kernel
        with patch.dict(sys.modules, {fake.__name__: fake}):
            output = writer.store_slots_nki_decode(cache, updates, slots)
        return cache, updates, slots, calls, output

    def test_all_eight_geometry_cases_preserve_payload_dtype_and_actual_int32_slots(self):
        cases = [((4, 3072, 4), 'torch.bfloat16'), ((4, 8, 128, 128), 'torch.float32')]
        cases += [((4, 1, n, 256), 'torch.bfloat16') for n in writer.BUCKETS]
        for shape, dtype in cases:
            with self.subTest(shape=shape):
                cache, updates, slots, calls, output = self.invoke(shape, dtype)
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0][2], slots)
                self.assertEqual((output.shape, output.dtype), (shape, dtype))
                self.assertEqual(cache.calls, [(4, -1)])
                self.assertEqual(updates.calls, [(4, -1)])

    def test_prefill_one_row_and_other_unqualified_geometry_fail_without_fallback(self):
        for shape in ((1, 3072, 4), (4, 1, 64, 256), (4, 2, 128, 256), (4, 8, 64, 128)):
            with self.subTest(shape=shape), self.assertRaises(identity.AdmissionError): self.invoke(shape)
        with self.assertRaises(identity.AdmissionError): self.invoke((4, 3072, 4), update_shape=(1, 3072, 4))

    def test_int64_slots_wrong_payload_dtype_and_mismatched_updates_fail(self):
        with self.assertRaises(identity.AdmissionError): self.invoke((4, 3072, 4), slots_dtype='torch.int64')
        with self.assertRaises(identity.AdmissionError): self.invoke((4, 3072, 4), dtype='torch.float32')
        with self.assertRaises(identity.AdmissionError):
            writer.store_slots_nki_decode(TensorShape((4,3072,4), 'torch.bfloat16'),
                TensorShape((4,3072,4), 'torch.float32'), TensorShape((4,), 'torch.int32'))


class OriginalComputationPreserved(unittest.TestCase):
    def test_all_original_sources_still_match_the_twenty_two_plan_pins(self):
        for key, row in PLAN['sources'].items():
            raw = Path(row['path']).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (row['bytes'], row['sha256']), key)

    def test_kernel_is_exact_previously_reviewed_source(self):
        self.assertEqual((PACKAGE/'slot_cache_nki_predicated.py').read_bytes(),
                         Path(PLAN['sources']['nki_kernel']['path']).read_bytes())

    def test_text_math_and_original_writer_equal_after_only_four_callable_substitutions(self):
        old = ast.parse((ORIGINAL/'text_core.py').read_text()); new = ast.parse((PACKAGE/'text_core.py').read_text())
        class Restore(ast.NodeTransformer):
            calls = 0
            fields = 0
            def visit_Assign(self, n):
                if len(n.targets) == 1 and ast.unparse(n.targets[0]) == 'self.cache_writer':
                    self.fields += 1; return None
                return self.generic_visit(n)
            def visit_Call(self, n):
                if ast.unparse(n.func) == 'self.cache_writer':
                    self.calls += 1; n.func = ast.Name(id='store_slots', ctx=ast.Load())
                return self.generic_visit(n)
        restore = Restore(); new = restore.visit(new)
        self.assertEqual((restore.calls, restore.fields), (4, 2))
        self.assertEqual(ast.dump(new, include_attributes=False), ast.dump(old, include_attributes=False))

    def test_model_math_sharding_fp32_cache_slice_and_tail_are_unchanged(self):
        old = (ORIGINAL/'modeling.py').read_text(); new = (PACKAGE/'modeling.py').read_text()
        new = new.replace('from .text_core import Layer, RMSNorm, LinearAttention, store_slots\nfrom .cache_writer import resolve_cache_writer',
                          'from .text_core import Layer, RMSNorm, LinearAttention')
        block = """        # Resolve once after TP sharding. Prefill remains the original writer.
        writer = resolve_cache_writer(c, self.prefill, store_slots)
        for layer in self.layers:
            attention = layer.linear_attn if layer.kind == 'linear_attention' else layer.self_attn
            attention.cache_writer = writer
"""
        self.assertEqual(new.count(block), 1); new = new.replace(block, '')
        self.assertEqual(new, old)

    def test_existing_artifact_admission_math_and_sources_remain_exact(self):
        old = (ORIGINAL/'identity.py').read_text(); new = (PACKAGE/'identity.py').read_text()
        new = new.replace('    from .cache_writer import cache_writer_identity\n', '')
        new = new.replace('"sources": source_identity(), "cache_writer": cache_writer_identity(),',
                          '"sources": source_identity(),')
        self.assertEqual(new, old)


class WriterPairContract(unittest.TestCase):
    @staticmethod
    def module():
        spec = importlib.util.spec_from_file_location('_writer_pair_contract', HERE/'pair_contract.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def fixture(self):
        result = {}
        for arm, mode in zip(('baseline', 'candidate'), (writer.ORIGINAL, writer.CANDIDATE)):
            with patch.dict(os.environ, {writer.WRITER_ENV: mode}):
                result[arm] = dict(identity={'sources': {'unchanged.py': 'a'*64},
                    'cache_writer': writer.cache_writer_identity()}, sampler_fast='1',
                    artifact='/future/'+arm+'-artifact', compiler_cache='/future/'+arm+'-cache',
                    compiler_workdir='/future/'+arm+'-workdir')
        return result

    def test_same_sampler_and_six_separate_trees_with_pending_qualification(self):
        value = self.module().validate_pair(self.fixture())
        self.assertEqual(value['graphs_required_per_arm'], 12)
        self.assertTrue(value['fresh_tree_checks_still_required'])
        self.assertFalse(value['full_model_qualified'])

    def test_sampler_confounding_shared_nested_trees_and_source_drift_rejected(self):
        for mode in ('sampler', 'shared', 'nested', 'source', 'wrong-writer'):
            arms = self.fixture()
            if mode == 'sampler': arms['baseline']['sampler_fast'] = '0'
            elif mode == 'shared': arms['candidate']['artifact'] = arms['baseline']['artifact']
            elif mode == 'nested': arms['candidate']['artifact'] = arms['baseline']['artifact'] + '/nested'
            elif mode == 'source': arms['candidate']['identity']['sources']['unchanged.py'] = 'b'*64
            else: arms['candidate']['identity']['cache_writer']['mode'] = writer.ORIGINAL
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.module().validate_pair(arms)

    def test_all_twelve_actual_graph_positions_required(self):
        rows = [dict(tag=tag, index=i, bucket=b) for tag in ('context_encoding_model', 'token_generation_model')
                for i,b in enumerate(writer.BUCKETS)]
        self.module().validate_graph_inventory(rows)
        for mode in ('missing', 'duplicate', 'wrong-bucket', 'bool-index'):
            value = copy.deepcopy(rows)
            if mode == 'missing': value.pop()
            elif mode == 'duplicate': value[-1] = value[0]
            elif mode == 'wrong-bucket': value[-1]['bucket'] = 8192
            else: value[0]['index'] = False
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.module().validate_graph_inventory(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
