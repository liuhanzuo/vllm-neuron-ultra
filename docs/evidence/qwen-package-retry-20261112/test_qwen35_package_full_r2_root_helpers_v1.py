"""SOURCE/STDLIB MOCK only; never call either helper's actual main body."""
import ast
import copy
import datetime
import importlib.abc
import importlib.util
import io as stdio
import json
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parent


class NoRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.startswith(('qwen35_package_serial_runtime_', 'prepare_qwen35_package_serial_candidate_', 'research'))
                or fullname.split('.')[0] in {'torch', 'numpy', 'neuronxcc', 'vllm', 'transformers', 'torch_neuronx'}):
            raise AssertionError('actual runtime/History/native import forbidden: ' + fullname)


sys.meta_path.insert(0, NoRuntimeImports())


def load(kind):
    path = Q / (kind + '_qwen35_package_full_root_r2_20261112_v1.py')
    spec = importlib.util.spec_from_file_location('synthetic_' + kind, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


PREPARE, ACCEPT = load('prepare'), load('accept')


def argv(module):
    result = []
    for name, (path, sha) in module.EXACT_REFS.items():
        result += ['--' + name, path, '--' + name + '-sha256', sha or 'a' * 64]
    return result


def tree(kind, version):
    return ast.parse((Q / (kind + '_qwen35_package_full_root_' + version + '_20261112_v1.py')).read_text())


def main_body(kind):
    return next(node.body for node in tree(kind, 'r2').body if isinstance(node, ast.FunctionDef) and node.name == 'main')


class ExactReferences(unittest.TestCase):
    def test_import_and_help_do_not_import_runtime_or_history(self):
        self.assertFalse(any(name.startswith(('qwen35_package_serial_runtime_', 'research.')) for name in sys.modules))
        for module in (PREPARE, ACCEPT):
            with patch('sys.stdout', new=stdio.StringIO()):
                with self.assertRaises(SystemExit) as exit: module.main(['--help'])
            self.assertEqual(exit.exception.code, 0)

    def test_all_unknown_hashes_must_be_supplied_and_return_exact_refs(self):
        for module in (PREPARE, ACCEPT):
            result = module.selected_refs(module.parser().parse_args(argv(module)))
            self.assertEqual(set(result), set(module.EXACT_REFS))
            for name, (path, sha) in module.EXACT_REFS.items():
                self.assertEqual(result[name], {'path': path, 'sha256': sha or 'a' * 64})

    def test_missing_or_wrong_cli_references_fail_before_runtime(self):
        for module in (PREPARE, ACCEPT):
            for name in module.EXACT_REFS:
                original = module.parser().parse_args(argv(module))
                for field, bad in ((name.replace('-', '_'), '/tmp/foreign.json'),
                                   (name.replace('-', '_') + '_sha256', 'not-a-sha')):
                    args = copy.deepcopy(original); setattr(args, field, bad)
                    with self.assertRaises(ValueError): module.selected_refs(args)
            args = argv(module)[:-2]
            with patch('sys.stderr', new=stdio.StringIO()):
                with self.assertRaises(SystemExit): module.main(args)

    def test_catalog_and_idle_wrong_well_formed_digest_fail(self):
        for module in (PREPARE, ACCEPT):
            for name in ('catalog', 'idle'):
                args = module.parser().parse_args(argv(module)); setattr(args, name + '_sha256', '0' * 64)
                with self.assertRaisesRegex(ValueError, 'frozen r2 prerequisite'): module.selected_refs(args)

    def test_exact_catalog_idle_and_fresh_r2_names_are_frozen(self):
        for module in (PREPARE, ACCEPT):
            self.assertEqual(module.EXACT_REFS['catalog'][1], '7d3d1ac0b8ca22ec6ce334baffb8599e967081624f9caf658116905665c0a256')
            self.assertEqual(module.EXACT_REFS['idle'][1], 'fdd95f3f235e3c4c16daf0c60372977f7bf9b01c8bfbbed40309a1460b02e7dc')
        prepare = ast.unparse(ast.Module(body=main_body('prepare'), type_ignores=[]))
        accept = ast.unparse(ast.Module(body=main_body('accept'), type_ignores=[]))
        for text in (prepare, accept):
            self.assertIn('pkg-r2-20261112', text)
            self.assertIn('v585-qwen35-package-full-r2-running.json', text)
            self.assertIn('qwen35_package_serial_runtime_v4', text)
            self.assertNotIn('qwen35-package-full-r1-', text)
        self.assertIn('prepare_qwen35_package_serial_candidate_v3', prepare)


class OriginalFlowPreservation(unittest.TestCase):
    def test_complete_accept_lock_try_finally_body_is_exact_except_r2_output(self):
        old = next(node for node in tree('accept', 'r1').body if isinstance(node, ast.Try))
        new = next(node for node in main_body('accept') if isinstance(node, ast.Try))
        class ReverseOutput(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    node.value = node.value.replace('qwen35-package-full-r2-root-binding', 'qwen35-package-full-r1-root-binding')
                return node
        new = ReverseOutput().visit(copy.deepcopy(new))
        self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_prepare_publication_and_all_observed_identity_checks_are_exact(self):
        old = tree('prepare', 'r1').body
        new = main_body('prepare')
        def suffix(nodes):
            index = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'candidate' for t in node.targets))
            return nodes[index:]
        self.assertEqual(ast.dump(ast.Module(body=suffix(old), type_ignores=[]), include_attributes=False),
                         ast.dump(ast.Module(body=suffix(new), type_ignores=[]), include_attributes=False))

    def test_original_accept_resource_and_time_limits_retained(self):
        old, new = tree('accept', 'r1').body, main_body('accept')
        for target in ('setrlimit', 'alarm', 'validate_spec', 'fresh_paths', 'unchanged_inputs'):
            def calls(nodes):
                return [ast.dump(n, include_attributes=False) for node in nodes for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == target]
            self.assertEqual(calls(old), calls(new), target)


class MockCompleteAcceptance(unittest.TestCase):
    def exercise(self, failure=None):
        # Execute only the preserved try/finally AST with synthetic collaborators.
        # No actual Bundle, lock, source readers, filesystem or process code is imported.
        node = next(node for node in main_body('accept') if isinstance(node, ast.Try))
        tip = {'path': '/synthetic/idle'}; catalog = {'current_handoff': tip, 'runs': []}
        bound = types.SimpleNamespace(unchanged=Mock(), freshness=Mock(return_value={'latest_handoff': tip}),
            supplement={'scope': {'children': list(range(874))}})
        if failure == 'children': bound.supplement['scope']['children'].pop()
        if failure == 'freshness': bound.freshness.return_value = {'latest_handoff': {'path': '/wrong'}}
        if failure == 'identity': bound.unchanged.side_effect = ValueError('synthetic changed input identity')
        lock = types.SimpleNamespace(check=Mock(), release=Mock())
        output = stdio.BytesIO()
        stream = types.SimpleNamespace(write=output.write, flush=lambda: None, fileno=lambda: 0)
        file_context = types.SimpleNamespace()
        class Context:
            def __enter__(self): return stream
            def __exit__(self, *args): return False
        path = lambda value: types.SimpleNamespace(open=lambda mode: Context())
        calls = []
        def bundle(spec, review, guard):
            calls.append(('bundle', spec, review)); guard()
            def resolve():
                calls.append(('resolve',))
                if failure == 'history': raise ValueError('synthetic complete History failure')
                return bound
            return types.SimpleNamespace(resolve=resolve)
        namespace = dict(io=types.SimpleNamespace(
            process_api=lambda: types.SimpleNamespace(active_heavy=Mock()), ref=lambda *args: {'path': '/synthetic/review', 'sha256': 'b'*64},
            loaded_local_sources=Mock(side_effect=ValueError('synthetic source mismatch') if failure == 'source' else None),
            resources=Mock(return_value={'synthetic': True}), once=Mock(return_value={'path': '/synthetic/binding', 'sha256': 'c'*64})),
            latest=types.SimpleNamespace(latest_handoff=lambda: tip), tip=tip,
            json=json, review={'synthetic': True}, Path=path, candidate={'review_path':'/synthetic/review','preservation':{}},
            os=types.SimpleNamespace(fsync=lambda fd: None), time=time, lock=lock, sys=types.SimpleNamespace(modules={}),
            Bundle=bundle, spec_ref={'path':'/synthetic/spec','sha256':'a'*64}, candidate_ref={'path':'/synthetic/candidate','sha256':'d'*64},
            catalog=catalog,spec={'source_sha256':{'synthetic.py':'e'*64},'history_overlay':{},'current_serial_catalog':{},'input_extension':{}},
            Q=Path('/synthetic'), datetime=datetime, print=lambda *args, **kw: None)
        code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])), '<synthetic-accept-try>', 'exec')
        if failure:
            with self.assertRaises((ValueError, AssertionError)): exec(code, namespace)
            namespace['io'].once.assert_not_called()
        else:
            exec(code, namespace); namespace['io'].once.assert_called_once()
            self.assertEqual(namespace['io'].once.call_args.args[1]['lfm_children'], 874)
            self.assertTrue(namespace['io'].once.call_args.args[1]['full_source_history_resolution_passed'])
            namespace['io'].loaded_local_sources.assert_called_once_with(namespace['spec']['source_sha256'])
        lock.release.assert_called_once_with(True)
        self.assertEqual(calls[1], ('resolve',))

    def test_success_requires_complete_history_874_and_sources(self): self.exercise()
    def test_history_failure_does_not_publish_and_releases_lock(self): self.exercise('history')
    def test_873_children_do_not_publish_and_release_lock(self): self.exercise('children')
    def test_identity_failure_does_not_publish_and_releases_lock(self): self.exercise('identity')
    def test_source_failure_does_not_publish_and_releases_lock(self): self.exercise('source')
    def test_freshness_failure_does_not_publish_and_releases_lock(self): self.exercise('freshness')


if __name__ == '__main__':
    unittest.main(verbosity=2)
