"""Execute the unchanged driver_v5.run against synthetic Python-only ports.

No native package is imported, no model is allocated, and no model/raw file is
opened. The actual driver source executes with a private research package and
explicit loader/source/scientific stubs; this does not patch installed sources.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

DRIVER = Path(__file__).with_name('minicpm5_layer1_operator_native_driver_v5.py')


class InjectedGateFailure(RuntimeError):
    pass


class Harness:
    def __init__(self, *, sources_checked=True, fail_loaded_check=None):
        self.events = []
        self.loads = []
        self.scientific_calls = []
        self.checks = 0
        self.fail_loaded_check = fail_loaded_check
        self.native_bundle = object()
        self.torch = object()  # A sentinel, not a native library.
        self.source = 'synthetic, never compiled model source'
        self.return_value = object()
        package = Path('/synthetic/minicpm5-transformers')
        self.native = SimpleNamespace(__file__=str(package / 'models/llama/modeling_llama.py'))
        attention_type = type('SyntheticAttention', (), {'__module__': 'synthetic_native_attention'})
        self.model = SimpleNamespace(model=SimpleNamespace(layers=[None, SimpleNamespace(self_attn=attention_type())]))
        self.context = SimpleNamespace(_sources_checked=sources_checked, plan={'synthetic': True},
            guard=self.context_guard, verify_loaded_sources=self.verify_loaded_sources, activate=self.activate)
        self.guard = lambda: self.events.append('guard')

        def require(condition, message):
            if not condition:
                raise RuntimeError(message)
        scientific = ModuleType('research.minicpm5_layer1_operator_native_driver_v4')
        scientific.require = require
        scientific.run_loaded = self.run_loaded
        source_helper = ModuleType('research.minicpm5_native_python_sources_v1')
        source_helper.PACKAGE = package
        source_helper.require_package_coverage = self.package_coverage
        research = ModuleType('research')
        research.__path__ = []
        research.minicpm5_layer1_operator_native_driver_v4 = scientific
        research.minicpm5_native_python_sources_v1 = source_helper
        loader = ModuleType('minicpm5_layer0_native_adapter_v2')
        loader.load = self.load
        trace = ModuleType('minicpm5_layer0_python_trace_source_v1')
        trace.source_text = self.source_text
        self.stubs = {
            'research': research,
            scientific.__name__: scientific,
            source_helper.__name__: source_helper,
            loader.__name__: loader,
            trace.__name__: trace,
        }
        self.driver = ModuleType('synthetic_driver_v5_integration_target')
        # Execute the full unmodified driver, substituting only its dependencies.
        with mock.patch.dict(sys.modules, self.stubs):
            exec(compile(DRIVER.read_bytes(), str(DRIVER), 'exec'), self.driver.__dict__)
        self.driver.importlib = SimpleNamespace(import_module=self.import_module)

    def context_guard(self, role):
        assert role == 'native'
        self.events.append('context.guard')

    def package_coverage(self, plan, guard):
        assert plan is self.context.plan and guard is self.guard
        self.events.append('package.coverage')

    @contextlib.contextmanager
    def activate(self):
        self.events.append('activate.enter')
        try:
            yield
        finally:
            self.events.append('activate.exit')

    def import_module(self, name):
        # Neither name is ever forwarded to the real import machinery.
        assert name in ('transformers.models.llama.modeling_llama', 'synthetic_native_attention')
        self.events.append('import.native' if name.startswith('transformers.') else 'import.model')
        return self.native

    def verify_loaded_sources(self):
        self.checks += 1
        self.events.append('loaded.check.%d' % self.checks)
        if self.checks == self.fail_loaded_check:
            raise InjectedGateFailure('synthetic source failure %d' % self.checks)

    def load(self, context):
        assert context is self.context
        self.events.append('loader.load')
        self.loads.append(context)
        return self.torch, self.model, object(), object()

    def source_text(self, context, module, role):
        assert context is self.context and module is self.native and role == 'native'
        self.events.append('source.text')
        return self.source

    def run_loaded(self, *args):
        self.events.append('scientific.run_loaded')
        self.scientific_calls.append(args)
        return self.return_value

    def run(self):
        with mock.patch.dict(sys.modules, self.stubs):
            return self.driver.run(self.context, self.guard, self.native_bundle)


class DriverGateChecks(unittest.TestCase):
    def test_incomplete_source_authentication_prevents_import_and_load(self):
        case = Harness(sources_checked=False)
        with self.assertRaisesRegex(RuntimeError, 'complete source authentication'):
            case.run()
        self.assertEqual(case.events, ['guard', 'context.guard'])
        self.assertEqual(case.loads, [])
        self.assertEqual(case.scientific_calls, [])

    def test_first_loaded_source_failure_prevents_model_load(self):
        case = Harness(fail_loaded_check=1)
        with self.assertRaisesRegex(InjectedGateFailure, 'failure 1'):
            case.run()
        self.assertEqual(case.loads, [])
        self.assertEqual(case.scientific_calls, [])
        self.assertEqual(case.events[-2:], ['loaded.check.1', 'activate.exit'])

    def test_second_loaded_source_failure_prevents_scientific_capture(self):
        case = Harness(fail_loaded_check=2)
        with self.assertRaisesRegex(InjectedGateFailure, 'failure 2'):
            case.run()
        self.assertEqual(case.loads, [case.context])
        self.assertEqual(case.scientific_calls, [])
        self.assertEqual(case.events[-3:], ['source.text', 'loaded.check.2', 'activate.exit'])
        self.assertLess(case.events.index('loaded.check.1'), case.events.index('loader.load'))

    def test_success_forwards_original_scientific_arguments_by_identity(self):
        case = Harness()
        self.assertIs(case.run(), case.return_value)
        self.assertEqual(len(case.scientific_calls), 1)
        expected = (case.torch, case.model, case.native, case.context, case.guard, case.native_bundle, case.source)
        self.assertEqual(len(case.scientific_calls[0]), len(expected))
        for actual, wanted in zip(case.scientific_calls[0], expected):
            self.assertIs(actual, wanted)
        self.assertEqual(case.events, ['guard', 'context.guard', 'package.coverage', 'activate.enter',
            'import.native', 'loaded.check.1', 'loader.load', 'import.model', 'source.text',
            'loaded.check.2', 'scientific.run_loaded', 'activate.exit'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    before_native = {name for name in sys.modules if name == 'torch' or name == 'transformers'
                     or name.startswith(('torch.', 'transformers.'))}
    assert not before_native
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DriverGateChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after_native = {name for name in sys.modules if name == 'torch' or name == 'transformers'
                    or name.startswith(('torch.', 'transformers.'))}
    assert not after_native
    reference = lambda path: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report = dict(schema='minicpm5_native_driver_v5_pure_integration_checks_v1', source_only=True,
        driver_source=reference(DRIVER), test_source=reference(Path(__file__)), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), passed=result.wasSuccessful(),
        actual_unchanged_driver_run_exercised=True,only_synthetic_dependency_ports_used=True,
        incomplete_auth_prevents_import_and_load=True if result.wasSuccessful() else None,
        first_loaded_gate_prevents_model_load=True if result.wasSuccessful() else None,
        second_loaded_gate_prevents_scientific_capture=True if result.wasSuccessful() else None,
        scientific_arguments_and_return_forwarded_unchanged=True if result.wasSuccessful() else None,
        native_imports_performed=False,model_execution_performed=False,model_or_raw_bytes_read=False,
        lock_or_service_operations_performed=False,runtime_admitted=False,
        full_model_qualified=False,new_speedup_claim=False)
    with Path(args.result).open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
