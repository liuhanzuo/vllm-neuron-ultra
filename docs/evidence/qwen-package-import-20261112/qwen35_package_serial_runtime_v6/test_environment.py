"""SOURCE/STDLIB MOCK only: package resolution without executing its code."""
import ast
import hashlib
from importlib.machinery import PathFinder
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from test_source import HERE, Q, c, a, w, svc, spec_fixture


class PackageEnvironment(unittest.TestCase):
    def expected(self):
        return ':'.join(map(str, (c.PACKAGE, c.Q, c.ROOT)))

    def test_explicit_checkout_first_for_both_arms_and_compile_modes(self):
        spec = spec_fixture()
        for arm in ('baseline', 'candidate'):
            for compile_allowed in (False, True):
                env = c.native_environment(spec['case'], arm=arm, compile_allowed=compile_allowed)
                self.assertEqual(env['PYTHONPATH'], self.expected())
                self.assertEqual(env['NEURON_QWEN35_ALLOW_COMPILE'], str(int(compile_allowed)))

    def test_all_phase_children_capture_and_build_keep_exact_pythonpath(self):
        spec = spec_fixture()
        for arm, name in c.PHASES:
            phase = dict(arm=arm, phase=name, directory=str(w.phase_directory(spec, arm, name)),
                         control=str(c.control_root(spec) / (arm + '-' + name)))
            with patch.object(w, 'manifest_sha', return_value='a' * 64): env = w.child_environment(spec, phase)
            self.assertEqual(env['PYTHONPATH'], self.expected(), (arm, name))
            self.assertEqual(env['NEURON_QWEN35_ALLOW_COMPILE'], '1' if name == 'build' else '0')

    def test_coordinator_and_supervisor_services_receive_same_checkout(self):
        spec = spec_fixture(); ref = {'path': '/mock/admission.json', 'sha256': 'a' * 64}
        co = a.coordinator_service(spec, ref)
        self.assertEqual(co.env['PYTHONPATH'], self.expected())
        phase = dict(unit=c.units(spec['run_name'])['shared:prepare'], properties={}, log='/mock/log')
        supervisor = a.phase_service(spec, phase, ref)
        self.assertEqual(supervisor.env['PYTHONPATH'], self.expected())

    def test_actual_pathfinder_resolves_top_models_and_leaf_without_importing_package(self):
        self.assertNotIn('vllm_neuron', sys.modules)
        search = self.expected().split(':') + list(sys.path)
        parent = PathFinder.find_spec('vllm_neuron', search)
        self.assertEqual(parent.origin, str(c.PACKAGE / 'vllm_neuron/__init__.py'))
        expected = c.PACKAGE / 'vllm_neuron'
        for fullname, leaf in (('vllm_neuron.models', 'models'),
                               ('vllm_neuron.models.qwen35', 'qwen35'),
                               ('vllm_neuron.models.qwen35.prepare_view', 'prepare_view.py')):
            parent = PathFinder.find_spec(fullname, parent.submodule_search_locations)
            self.assertIsNotNone(parent, fullname)
            expected /= leaf
            self.assertEqual(parent.origin, str(expected if leaf.endswith('.py') else expected / '__init__.py'))
        self.assertNotIn('vllm_neuron', sys.modules)

    def test_old_checkout_cannot_shadow_explicit_package_first(self):
        old = '/home/ubuntu/src/vllm-neuron'
        spec = PathFinder.find_spec('vllm_neuron', self.expected().split(':') + [old])
        self.assertEqual(spec.origin, str(c.PACKAGE / 'vllm_neuron/__init__.py'))

    def test_api_exec_and_supervisor_popen_use_unchanged_child_environment(self):
        # Verify the actual source forwarding edges without invoking bootstrap,
        # service supervision, native imports or a subprocess.
        adapter_tree = ast.parse((HERE / 'adapter.py').read_text())
        service_tree = ast.parse((HERE / 'service.py').read_text())
        exec_calls = [n for n in ast.walk(adapter_tree) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == 'execvpe']
        self.assertEqual(len(exec_calls), 1)
        self.assertEqual(ast.dump(exec_calls[0].args[2]), ast.dump(ast.parse('context["child_environment"]', mode='eval').body))
        popens = [n for n in ast.walk(service_tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'Popen']
        self.assertEqual(len(popens), 1)
        self.assertEqual(ast.dump(next(k.value for k in popens[0].keywords if k.arg == 'env')), ast.dump(ast.Name(id='env', ctx=ast.Load())))
        assignments = [n for n in ast.walk(service_tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'env' for t in n.targets)]
        self.assertTrue(any(ast.dump(n.value) == ast.dump(ast.parse('context["child_environment"]', mode='eval').body) for n in assignments))


class Preservation(unittest.TestCase):
    def test_contract_changes_only_one_environment_entry(self):
        old = (Q / 'qwen35_package_serial_runtime_v5/contract.py').read_text()
        new = (HERE / 'contract.py').read_text()
        before = '"PYTHONPATH": str(Q) + ":" + str(ROOT)'
        after = '"PYTHONPATH": str(PACKAGE) + ":" + str(Q) + ":" + str(ROOT)'
        self.assertEqual(old.count(before), 1); self.assertEqual(new, old.replace(before, after))

    def test_other_production_sources_identical_and_run_namespace_only(self):
        old = Q / 'qwen35_package_serial_runtime_v5'
        for name in c.RUNTIME_FILES:
            if name in ('contract.py', 'run.py'): continue
            self.assertEqual((HERE / name).read_bytes(), (old / name).read_bytes(), name)
        self.assertEqual((HERE / 'run.py').read_text(), (old / 'run.py').read_text().replace(
            'qwen35_package_serial_runtime_v5', 'qwen35_package_serial_runtime_v6'))

    def test_all_frozen_v1_through_v5_members_unchanged(self):
        values = json.loads((HERE / 'predecessor_pins.json').read_text())['files']
        for path, pin in values.items():
            raw = Path(path).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), pin['sha256'], path)
            self.assertEqual(len(raw), pin['bytes'], path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
