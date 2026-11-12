"""Mocked module/import/source ports only. No original interpreter execution."""
import sys

FORBIDDEN = {'torch', 'numpy', 'ctypes', '_ctypes', 'transformers', 'vllm',
             'neuron', 'neuronxcc', 'neuronx_distributed', 'neuronx_distributed_inference', 'nki'}


class NoNative:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in FORBIDDEN:
            raise AssertionError('Forbidden native import: ' + name)


sys.meta_path.insert(0, NoNative())
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

Q = Path(__file__).parent
sys.path.insert(0, str(Q.parent))
from research import minicpm5_bootstrap_types_source_v1 as core


def world(*, preexisting=False, source='value = 7\n'):
    events = []
    raw = source.encode() + b'\n' * (core.PIN['bytes'] - len(source.encode()))
    runtime = NS(modules={}, meta_path=[])

    class SourceLoader:
        def __init__(self, name, path):
            self.name, self.path = name, path
        def exec_module(self, module):
            exec(self.get_code(module.__name__), module.__dict__)

    def spec_for(name, path=None, target=None):
        events.append(('find', name))
        if name == 'types':
            return NS(name=name, origin=core.PIN['path'], loader=SourceLoader(name, core.PIN['path']))
        return None

    external = NS(SourceFileLoader=SourceLoader, PathFinder=NS(find_spec=spec_for))

    def module(loader):
        return NS(__name__='types', __file__=core.PIN['path'], __loader__=loader,
                  __spec__=NS(name='types', origin=core.PIN['path'], loader=loader))

    if preexisting:
        runtime.modules['types'] = module(SourceLoader('types', core.PIN['path']))
    previous = NS(find_spec=lambda name, path=None, target=None: events.append(('previous', name)))
    runtime.meta_path = [previous]
    entry = NS(finder=previous, source_bytes=0, sources={})
    def check():
        assert runtime.meta_path[0] is entry.finder
        events.append('guard')
    def read_source(path, sha, cap):
        assert (path, sha, cap) == (core.PIN['path'], core.PIN['sha256'], core.PIN['bytes'])
        events.append('authenticated-source-port')
        return raw, ('mock-dev', 'mock-inode', len(raw))
    entry.check, entry.read_source = check, read_source
    tracker = core.TypesSourceObservation(runtime)

    def import_types():
        if 'types' in runtime.modules:
            events.append('reuse-preexisting')
            return runtime.modules['types']
        spec = runtime.meta_path[0].find_spec('types')
        result = module(spec.loader)
        runtime.modules['types'] = result
        try:
            spec.loader.exec_module(result)
        except BaseException:
            del runtime.modules['types']
            raise
        return result

    def import_json_chain():
        for name in ('json', 'json.decoder', 're', 'enum'):
            events.append(('simulated-import', name))
            runtime.meta_path[0].find_spec(name)
        return import_types()

    return NS(runtime=runtime, external=external, entry=entry, events=events, tracker=tracker,
              module=module, import_types=import_types, import_json_chain=import_json_chain)


def install(w):
    w.tracker.install_before_json(w.entry, w.external, 16 * 2**20)


class TypesSourceTests(unittest.TestCase):
    def test_json_chain_first_load_records_only_completed_pinned_execution(self):
        w = world()
        install(w)
        self.assertNotIn(core.PIN['path'], w.entry.sources)
        module = w.import_json_chain()
        w.tracker.after_json()
        value = w.tracker.finish(dict(core.PIN))
        self.assertEqual(module.value, 7)
        self.assertFalse(value['first_checkpoint']['present'])
        self.assertTrue(value['after_json']['present'])
        self.assertTrue(value['executed_through_pinned_loader'])
        self.assertFalse(value['original_execution_provenance_gap'])
        self.assertIs(w.runtime.modules['types'], module)
        self.assertEqual(w.entry.source_bytes, 2 * core.PIN['bytes'])
        self.assertFalse(value['accepted_native_authority_created'])

    def test_preexisting_module_gets_current_source_correlation_not_execution_receipt(self):
        w = world(preexisting=True)
        before = w.runtime.modules['types']
        install(w)
        self.assertIs(w.import_json_chain(), before)
        w.tracker.after_json()
        value = w.tracker.finish(dict(core.PIN))
        self.assertTrue(value['current_source_bytes_verified'])
        self.assertTrue(value['loaded_provider_metadata_matches'])
        self.assertTrue(value['original_execution_provenance_gap'])
        self.assertFalse(value['executed_through_pinned_loader'])
        self.assertFalse(value['interpreter_startup_cause_established'])
        self.assertFalse(value['preexisting_module_reloaded'])
        self.assertIsNone(value['execution_receipt'])
        self.assertNotIn(core.PIN['path'], w.entry.sources)
        self.assertIs(w.runtime.modules['types'], before)

    def test_get_code_without_exec_never_becomes_execution_receipt(self):
        w = world()
        install(w)
        loader = w.runtime.meta_path[0].find_spec('types').loader
        loader.get_code('types')
        self.assertIsNone(w.tracker.execution_receipt)
        self.assertNotIn(core.PIN['path'], w.entry.sources)

    def test_exec_failure_does_not_create_receipt(self):
        w = world(source="raise ValueError('synthetic types execution failure')\n")
        install(w)
        with self.assertRaisesRegex(ValueError, 'synthetic types'):
            w.import_types()
        self.assertIsNone(w.tracker.executed_module)
        self.assertNotIn(core.PIN['path'], w.entry.sources)

    def test_wrong_current_source_hash_rejects_before_compilation(self):
        w = world()
        install(w)
        def bad_reader(*args):
            raise ValueError('Executed source bytes match reviewed SHA256')
        w.entry.read_source = bad_reader
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            w.import_types()
        self.assertNotIn(core.PIN['path'], w.entry.sources)

    def test_source_cap_preserved_and_exhaustion_has_no_receipt(self):
        w = world()
        with self.assertRaisesRegex(ValueError, 'Unchanged original'):
            w.tracker.install_before_json(w.entry, w.external, 32 * 2**20)
        install(w)
        w.entry.source_bytes = 16 * 2**20
        with self.assertRaisesRegex(ValueError, 'whole-source byte cap'):
            w.import_types()
        self.assertNotIn(core.PIN['path'], w.entry.sources)

    def test_wrong_preexisting_file_spec_or_loader_never_attested(self):
        for wrong in ('file', 'origin', 'loader', 'name'):
            w = world(preexisting=True)
            module = w.runtime.modules['types']
            if wrong == 'file': module.__file__ = '/unreviewed/types.py'
            if wrong == 'origin': module.__spec__.origin = '/unreviewed/types.py'
            if wrong == 'loader': module.__spec__.loader = object()
            if wrong == 'name': module.__spec__.name = 'other'
            install(w)
            w.import_json_chain()
            w.tracker.after_json()
            with self.assertRaises(ValueError): w.tracker.finish(dict(core.PIN))
            self.assertNotIn(core.PIN['path'], w.entry.sources)

    def test_wrong_selected_source_provider_is_rejected(self):
        w = world()
        install(w)
        original = w.external.PathFinder.find_spec
        def wrong(name, path=None, target=None):
            spec = original(name, path, target)
            spec.origin = '/unreviewed/types.py'
            return spec
        w.external.PathFinder.find_spec = wrong
        with self.assertRaisesRegex(ValueError, 'reviewed ordinary'):
            w.import_types()

    def test_module_replacement_and_synthetic_preexisting_receipt_rejected(self):
        for mode in ('executed', 'preexisting', 'fake-receipt'):
            w = world(preexisting=mode != 'executed')
            install(w)
            w.import_types()
            w.tracker.after_json()
            if mode == 'fake-receipt': w.entry.sources[core.PIN['path']] = dict(core.PIN)
            else: w.runtime.modules['types'] = w.module(w.external.SourceFileLoader('types', core.PIN['path']))
            with self.assertRaises(ValueError): w.tracker.finish(dict(core.PIN))

    def test_early_pin_must_match_later_authenticated_table(self):
        w = world()
        install(w)
        w.import_types()
        w.tracker.after_json()
        pin = dict(core.PIN, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'authenticated candidate'):
            w.tracker.finish(pin)

    def test_existing_module_is_not_reloaded_by_explicit_loader_call(self):
        w = world(preexisting=True)
        install(w)
        loader = w.runtime.meta_path[0].find_spec('types').loader
        with self.assertRaisesRegex(ValueError, 'Never reload'):
            loader.exec_module(w.runtime.modules['types'])
        self.assertNotIn(core.PIN['path'], w.entry.sources)

    def test_component_has_no_import_or_direct_file_process_native_operations(self):
        tree = ast.parse((Q / 'minicpm5_bootstrap_types_source_v1.py').read_text())
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {'open', 'eval', '__import__'})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {'Popen', 'run', 'read_bytes', 'write', 'flock', 'kill', 'setrlimit', 'CDLL'})


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(TypesSourceTests))
    assert not FORBIDDEN.intersection(sys.modules)
    print('Mocked module/source/import ports only; no original runtime or native observation.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
