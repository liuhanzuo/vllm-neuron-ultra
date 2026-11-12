"""Execute complete original/patched loader modules with isolated dependency stubs.

No native or model execution. The Sampler name is deliberately NOT patched;
Python's real from-import semantics select either a ModuleType or its class.
"""
import hashlib
import importlib.abc
import os
from pathlib import Path
import re
import sys
import types
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
ORIGINAL = Q / 'vllm-neuron-053/vllm_neuron/worker/neuronx_distributed_model_loader.py'
CANDIDATE = Q / 'vllm053_model_loader_sampler_import_candidate_v1.py'
OLD_IMPORT = b'from vllm.v1.sample import sampler as Sampler\n'
NEW_IMPORT = b'from vllm.v1.sample.sampler import Sampler\n'
DENIED = {'torch', 'torch_xla', 'torch_neuronx', 'neuronx_distributed',
          'neuronx_distributed_inference', 'numpy', 'ctypes', '_ctypes', 'nki',
          'neuronxcc', 'transformers', 'vllm', 'vllm_neuron', 'regex'}


class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in DENIED:
            raise RuntimeError('Unstubbed native/application import denied: ' + fullname)
        return None


class BaseModule:
    pass


class SamplerSentinel:
    instances = 0

    def __init__(self):
        type(self).instances += 1


def dependency_stubs():
    modules = {}

    def module(name, **attributes):
        if name not in modules:
            item = types.ModuleType(name)
            item.__path__ = []
            modules[name] = item
            if '.' in name:
                parent, leaf = name.rsplit('.', 1)
                setattr(module(parent), leaf, item)
        modules[name].__dict__.update(attributes)
        return modules[name]

    class Config:
        vocab_size = 32000

        def get_text_config(self):
            return self

    module('regex').__dict__.update(re.__dict__)
    module('torch', Tensor=type('Tensor', (), {}), dtype=type('dtype', (), {}))
    module('torch.nn', Module=BaseModule)
    module('neuronx_distributed_inference.models.config',
           **{name: type(name, (), {}) for name in (
               'ChunkedPrefillConfig', 'FusedSpecNeuronConfig', 'NeuronConfig',
               'OnDeviceSamplingConfig')})
    module('neuronx_distributed_inference.modules.lora_serving',
           LoraServingConfig=type('LoraServingConfig', (), {}))
    module('neuronx_distributed_inference.utils.constants', MODEL_TYPES={})
    module('neuronx_distributed_inference.utils.hf_adapter',
           load_pretrained_config=lambda *a, **kw: None)
    module('transformers', AutoModelForCausalLM=type('AutoModelForCausalLM', (), {}),
           PretrainedConfig=Config)
    module('vllm.config', **{name: type(name, (), {}) for name in (
        'CacheConfig', 'ModelConfig', 'ParallelConfig', 'SchedulerConfig', 'SpeculativeConfig')})
    module('vllm.model_executor.layers.logits_processor',
           LogitsProcessor=lambda *a, **kw: object())
    module('vllm.v1.outputs', SamplerOutput=type('SamplerOutput', (), {}))
    module('vllm.v1.sample.sampler', Sampler=SamplerSentinel)
    module('vllm_neuron.worker.constants', NEURON_MULTI_MODAL_MODELS=[],
           TORCH_DTYPE_TO_NEURON_AMP={})
    return modules, Config()


def load_complete(path):
    module = types.ModuleType('isolated_model_loader_under_test')
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


class Regression(unittest.TestCase):
    def setUp(self):
        self.assertFalse(any(n.split('.')[0] in DENIED for n in sys.modules))
        self.modules, self.config = dependency_stubs()
        self.modules_patch = patch.dict(sys.modules, self.modules)
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.blocker = NativeBlocker()
        sys.meta_path.insert(0, self.blocker)
        self.addCleanup(lambda: sys.meta_path.remove(self.blocker))
        SamplerSentinel.instances = 0

    def test_candidate_only_changes_import(self):
        old = ORIGINAL.read_bytes()
        self.assertEqual(old.count(OLD_IMPORT), 1)
        self.assertEqual(CANDIDATE.read_bytes(), old.replace(OLD_IMPORT, NEW_IMPORT))
        sample_init = Q / 'vllm016-source/vllm/v1/sample/__init__.py'
        self.assertEqual(sample_init.read_bytes(), b'')
        sampler = Q / 'vllm016-source/vllm/v1/sample/sampler.py'
        self.assertIn('\nclass Sampler(nn.Module):', sampler.read_text())

    def test_original_cpu_switch_raises_module_not_callable(self):
        loader = load_complete(ORIGINAL)
        self.assertIsInstance(loader.Sampler, types.ModuleType)
        with patch.dict(os.environ, {'NEURON_ON_DEVICE_SAMPLING_DISABLED': '1'}):
            with self.assertRaisesRegex(TypeError, "'module' object is not callable"):
                loader.NeuronCausalLM(self.config)
        self.assertEqual(SamplerSentinel.instances, 0)

    def test_fixed_cpu_switch_constructs_the_class(self):
        loader = load_complete(CANDIDATE)
        self.assertIs(loader.Sampler, SamplerSentinel)
        with patch.dict(os.environ, {'NEURON_ON_DEVICE_SAMPLING_DISABLED': '1'}):
            model = loader.NeuronCausalLM(self.config)
        self.assertTrue(model.on_device_sampling_disabled)
        self.assertIsInstance(model.sampler, SamplerSentinel)
        self.assertEqual(SamplerSentinel.instances, 1)

    def test_both_default_paths_avoid_optional_constructor(self):
        for path in (ORIGINAL, CANDIDATE):
            loader = load_complete(path)
            with patch.dict(os.environ, {}, clear=True):
                model = loader.NeuronCausalLM(self.config)
            self.assertFalse(model.on_device_sampling_disabled)
            self.assertFalse(hasattr(model, 'sampler'))
        self.assertEqual(SamplerSentinel.instances, 0)

    def test_both_explicit_device_paths_avoid_optional_constructor(self):
        for path in (ORIGINAL, CANDIDATE):
            loader = load_complete(path)
            with patch.dict(os.environ, {'NEURON_ON_DEVICE_SAMPLING_DISABLED': '0'}):
                model = loader.NeuronCausalLM(self.config)
            self.assertFalse(model.on_device_sampling_disabled)
        self.assertEqual(SamplerSentinel.instances, 0)

    def test_unstubbed_native_import_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Unstubbed native/application import denied'):
            __import__('torch_xla')


if __name__ == '__main__':
    unittest.main(verbosity=2)
