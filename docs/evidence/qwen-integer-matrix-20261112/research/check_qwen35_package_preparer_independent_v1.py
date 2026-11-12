"""Independent synthetic checks; no current binding, History or native work.

Default subject is frozen v1. --candidate-module permits a named source successor
without editing these tests. Only StableReaders checks use temporary JSON files.
"""
import copy
import hashlib
import importlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

Q = Path(__file__).resolve().parent
sys.path.insert(0, str(Q))


class RejectNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_xla', 'torch_neuronx', 'numpy',
                'nki', 'neuronxcc', 'ctypes', '_ctypes', 'vllm', 'vllm_neuron',
                'transformers', 'neuronx_distributed', 'neuronx_distributed_inference'}:
            raise AssertionError('Native import forbidden: ' + fullname)


sys.meta_path.insert(0, RejectNative())
module_name = 'prepare_qwen35_package_serial_candidate_v1'
if '--candidate-module' in sys.argv:
    index = sys.argv.index('--candidate-module')
    module_name = sys.argv[index + 1]
    del sys.argv[index:index + 2]
if module_name not in {'prepare_qwen35_package_serial_candidate_v1',
                       'prepare_qwen35_package_serial_candidate_v2'}:
    raise ValueError('Only named source review subjects')
p = importlib.import_module(module_name)
import test_prepare_qwen35_package_serial_candidate_v1 as fixture
fixture.p = p


class IndependentPreparation(unittest.TestCase):
    def test_exact_source_union_and_recovered_partition_both_versions(self):
        for version in (2, 3):
            f = fixture.Fixture(version=version)
            out = f.run()
            seed, old_overlay = f.value(f.seed_ref), f.value(f.overlay_ref)
            runtime = f.value(f.request['runtime_delivery'])
            science = f.value(p.c.SCIENCE_DELIVERY)
            expected = dict(seed['source_sha256'])
            for row in f.value(f.catalog_ref)['runs']:
                expected.update(row['source_sha256'])
            for mapping in (runtime['files'],
                    f.value(dict(path=str(f.directory/'input_pins.json')))['files'],
                    science['files'], f.request['preparation_sources']):
                expected.update({path: value['sha256'] if isinstance(value, dict) else value
                                 for path, value in mapping.items()})
            for name in ('input_sources.json', 'capture_sources.json'):
                expected.update(f.value(dict(path=str(p.c.Q/'qwen35_package_full_validation_v2'/name))))
            for ref in (f.request['runtime_delivery'], p.c.SCIENCE_DELIVERY):
                expected[ref['path']] = ref['sha256']
            self.assertEqual(out['spec']['source_sha256'], expected)
            recovered = set(seed['source_sha256']) - set(old_overlay['source_extension'])
            self.assertEqual(out['overlay']['source_extension'],
                             {path: sha for path, sha in expected.items() if path not in recovered})

    def test_request_inputs_and_base_reference_remain_immutable(self):
        f = fixture.Fixture()
        request, metadata = copy.deepcopy(f.request), copy.deepcopy(f.metadata)
        out = f.run()
        self.assertEqual(f.request, request)
        self.assertEqual(f.metadata, metadata)
        self.assertEqual(out['overlay']['base_plan'], f.base_ref)
        for ref in (f.seed_ref, f.overlay_ref):
            for path, row in f.value(ref)['input_extension'].items():
                self.assertEqual(out['spec']['input_extension'][path], row)
        raw = f.binary_base_input['identity']['realpath']
        self.assertNotIn(raw, {row[0] for row in f.reads})
        self.assertNotIn(raw, out['spec']['input_extension'])

    def test_every_runtime_file_required_in_v2_and_v3(self):
        for version in (2, 3):
            for name in (*p.c.RUNTIME_FILES, 'input_pins.json'):
                f = fixture.Fixture(version=version)
                del f.value(f.request['runtime_delivery'])['files'][str(f.directory/name)]
                with self.subTest(version=version, name=name), self.assertRaises(ValueError):
                    f.run()

    def test_v3_still_requires_all_five_v2_pure_import_dependencies(self):
        paths = (str(p.c.HERE/'__init__.py'), str(p.c.HERE/'contract.py'),
                 str(p.c.Q/'qwen35_sampler_pair_protocol_v1.py'),
                 str(p.c.Q/'qwen35_package_full_validation_v2/__init__.py'),
                 str(p.c.Q/'qwen35_package_full_validation_v2/runtime.py'))
        for path in paths:
            f = fixture.Fixture(version=3)
            f.request['preparation_sources'].pop(path, None)
            f.value(p.c.SCIENCE_DELIVERY)['files'].pop(path, None)
            f.value(dict(path=str(f.directory/'input_pins.json')))['files'].pop(path, None)
            with self.subTest(path=path), self.assertRaises(ValueError):
                f.run()

    def test_false_review_and_serialized_hashes_both_versions(self):
        for version in (2, 3):
            out = fixture.Fixture(version=version).run()
            for row in (out, out['review']):
                self.assertIs(row['passed'], False)
                self.assertIs(row['launch_authorized'], False)
            with self.assertRaises(ValueError):
                p.c.validate_spec(out['spec'], out['review'], out['spec_ref'])
            for name in ('overlay', 'spec', 'review'):
                self.assertEqual(json.loads(out[name+'_bytes']), out[name])
                self.assertEqual(hashlib.sha256(out[name+'_bytes']).hexdigest(), out[name+'_ref']['sha256'])

    def test_closed_guard_prevents_all_injected_reads(self):
        f = fixture.Fixture()
        with patch.object(f, 'guard', side_effect=PermissionError('closed source review window')):
            with self.assertRaises(PermissionError):
                f.run()
        self.assertEqual(f.reads, [])
        self.assertEqual(f.stats, [])

    def test_six_binary_payloads_only_statted_twice(self):
        f = fixture.Fixture()
        out = f.run()
        opened = {row[0] for row in f.reads}
        for path in p.c.BINARY_PATHS:
            self.assertEqual(f.stats.count(path), 2)
            self.assertNotIn(path, opened)
            self.assertNotIn(path, out['spec']['input_extension'])
            self.assertEqual(out['spec']['binary_inputs'][path], f.request['binary_inputs'][path])

    def test_catalog_original_256_row_boundary(self):
        out = fixture.Fixture(old_count=2, current_count=256).run()
        self.assertEqual(out['preservation']['current_catalog_rows'], 256)
        with self.assertRaises(ValueError):
            fixture.Fixture(old_count=2, current_count=257).run()

    def test_retained_but_unopened_reference_paths_cannot_be_outputs(self):
        for target in ('recovery', 'native_manifest', 'completed_sdk', 'later_terminal'):
            f = fixture.Fixture()
            old_overlay, catalog = f.value(f.overlay_ref), f.value(f.catalog_ref)
            retained = {'recovery': old_overlay['sdk_operator_recovery'],
                        'native_manifest': old_overlay['native_manifest'],
                        'completed_sdk': catalog['completed_sdk'],
                        'later_terminal': catalog['runs'][-1]['terminal']}[target]
            f.request['future']['overlay_path'] = retained['path']
            with self.subTest(target=target), self.assertRaises(ValueError):
                f.run()

    def test_overflowed_json_numbers_are_rejected_at_any_depth(self):
        with tempfile.TemporaryDirectory(prefix='qwen-package-independent-json-') as directory:
            path = Path(directory)/'number.json'
            for payload in (b'{"x":1e999}', b'{"x":[-1e999]}', b'{"x":{"y":1e999}}'):
                path.write_bytes(payload)
                ref = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    p.StableReaders(lambda:None).metadata(ref)


if __name__ == '__main__':
    unittest.main(verbosity=2)
