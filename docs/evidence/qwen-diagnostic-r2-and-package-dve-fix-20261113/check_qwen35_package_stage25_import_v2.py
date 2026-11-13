"""Actual plain-python pure-package import probe; never a native/model probe."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
ALLOWED = {'vllm_neuron', 'vllm_neuron.models', 'vllm_neuron.models.qwen35',
    'vllm_neuron.models.qwen35.provenance', 'vllm_neuron.models.qwen35.identity',
    'vllm_neuron.models.qwen35.validation'}
NATIVE = {'ctypes', '_ctypes', 'torch', 'torch_xla', 'torch_neuronx', 'neuronxcc',
    'numpy', 'ml_dtypes', 'transformers', 'vllm', 'libneuronxla', 'nki', 'safetensors',
    'neuronx_distributed', 'neuronx_distributed_inference'}


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in NATIVE or (fullname.startswith('vllm_neuron') and fullname not in ALLOWED):
            raise AssertionError('Not an approved pure stage25 import: ' + fullname)


def main():
    assert sys.dont_write_bytecode and __debug__
    sys.meta_path.insert(0, RejectNative())
    import qwen35_package_post_audit_v2 as subject
    manifest = Q / 'qwen35_package_serial_runtime_v9/input_pins.json'
    raw = manifest.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '3a2d7bd900c8b3b4a19aea195e7c89112022d81249f79cd8291c9999f8938b84'
    spec = dict(source_sha256={path:row['sha256'] for path,row in json.loads(raw)['files'].items()})
    tests = []
    with (patch('subprocess.Popen', side_effect=AssertionError('No process launch')),
          patch('fcntl.flock', side_effect=AssertionError('No original-lock operation')),
          patch.object(subject.Bundle, 'resolve', side_effect=AssertionError('No actual History'))):
        missing = dict(source_sha256={})
        try: subject.bind_pure_package_sources(missing)
        except ValueError: tests.append('missing_source_pin_rejected')
        else: raise AssertionError('Missing source pin admitted')
        foreign = ModuleType('vllm_neuron'); foreign.__file__ = '/different-checkout/vllm_neuron/__init__.py'
        with patch.dict(sys.modules, {'vllm_neuron': foreign}):
            try: subject.bind_pure_package_sources(spec)
            except ValueError: tests.append('preloaded_foreign_checkout_rejected')
            else: raise AssertionError('Foreign checkout admitted')
        loaded = subject.bind_pure_package_sources(spec)
        assert set(loaded) == ALLOWED
        assert all(Path(path).is_relative_to(subject.PACKAGE) for path in loaded.values())
        tests.append('all_six_actual_pure_modules_from_exact_package_checkout')
        assert subject.bind_pure_package_sources(spec) == loaded
        tests.append('same_executed_pure_module_identity_rechecked')
        assert not any(name.split('.')[0] in NATIVE for name in sys.modules)
        tests.append('no_native_imports_history_locks_or_payloads')
    print(json.dumps(dict(source_only=True, interpreter=sys.executable, tests=tests,
        pure_modules=loaded, native_executed=False, history_executed=False, runtime_admitted=False), indent=2))


if __name__ == '__main__':
    main()
