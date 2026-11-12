# SPDX-License-Identifier: Apache-2.0
"""Run tests with a finder that rejects every real native dependency import."""
import importlib.abc
from pathlib import Path
import sys
import unittest


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_xla', 'torch_neuronx', 'neuronxcc',
            'neuronx_distributed', 'neuronx_distributed_inference', 'transformers', 'vllm', 'numpy'}:
            raise AssertionError('Native dependency import forbidden in source tests: ' + fullname)
        return None


if __name__ == '__main__':
    sys.meta_path.insert(0, DenyNative())
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
