"""Synthetic stdlib-only preparation tests: no current bindings or payload IO."""
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class NativeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'transformers', 'vllm', 'vllm_neuron',
                'torch_neuronx', 'neuronx_distributed', 'neuronx_distributed_inference'}:
            raise AssertionError('native import forbidden: '+fullname)


sys.meta_path.insert(0, NativeBlocker())
import prepare_qwen35_package_serial_candidate_v5 as p
from qwen35_package_full_validation_v2.runtime import configuration


def ident(path, size=3, inode=1):
    return dict(realpath=path, device=1, inode=inode, size=size, mtime_ns=2, ctime_ns=3)


class Fixture:
    """Synthetic graph only. Readers deliberately substitute fake file records."""
    def __init__(self, old_count=2, current_count=5, version=2):
        self.metadata, self.sources, self.identities = {}, {}, {}
        self.reads, self.stats, self.guards = [], [], 0
        self.directory = p.c.Q/('qwen35_package_serial_runtime_v'+str(version))
        self.seed_actual_ref = dict(path=str(p.c.Q/'mock-seed-actual.json'), sha256='e'*64)
        tool = dict(kind='tool_session', session_id=7, terminal_chunk='mock', exit_code=0)
        owners, units = [{'pid': 123, 'start_ticks': 321}], ['mock-unit']
        old_sources = {}
        for name in ('base-source.py', 'recovery-only.py', 'seed-extension.py'):
            ref = self.source(str(p.c.Q/name))
            old_sources[ref['path']] = ref['sha256']
        self.base_input = self.saved_input(str(p.c.Q/'mock-base-input.json'))
        self.binary_base_input = dict(sha256='3'*64, identity=ident(str(p.c.Q/'mock-original.raw')))
        base_inputs = {self.base_input[0]: self.base_input[1], self.binary_base_input['identity']['realpath']: self.binary_base_input}
        self.base_ref = self.meta('mock-base.json', dict(source_sha256={str(p.c.Q/'base-source.py'): old_sources[str(p.c.Q/'base-source.py')]}, inputs=base_inputs))
        self.idle_ref = self.meta('mock-idle.json', dict(serial_execution=True, running=False, live_tool_sessions=[], active_heavy_units=[]))
        rows = []
        for index in range(current_count):
            ref = self.seed_actual_ref if index == 0 else dict(path=str(p.c.Q/f'mock-exit-{index}.json'), sha256=f'{index:064x}')
            extra = self.source(str(p.c.Q/f'catalog-source-{index}.py'))
            rows.append(dict(terminal=ref, tool=copy.deepcopy(tool), owners=copy.deepcopy(owners), units=copy.deepcopy(units),
                             source_sha256={extra['path']: extra['sha256']}))
        def catalog(rows):
            return dict(schema='minicpm5_operator_later_serial_metadata_catalog_v1', reviewer='/root', passed=True, blockers=[],
                completed_sdk=dict(path=str(p.c.Q/'mock-completed-sdk.json'), sha256='c'*64), current_handoff=self.idle_ref,
                all_later_serial_runs_included=True, runs=rows)
        self.old_catalog_ref = self.meta('mock-old-catalog.json', catalog(copy.deepcopy(rows[:old_count])))
        self.catalog_ref = self.meta('mock-current-catalog.json', catalog(copy.deepcopy(rows)))
        overlay_input = self.saved_input(str(p.c.Q/'mock-overlay-input.json'))
        self.overlay_ref = self.meta('mock-overlay.json', dict(schema='minicpm5_layer1_operator_native_small_overlay_v2',
            base_plan=self.base_ref, sdk_operator_recovery=dict(path=str(p.c.Q/'mock-recovery.json'), sha256='4'*64),
            current_serial_catalog=self.old_catalog_ref, native_manifest=dict(path=str(p.c.Q/'mock-native-manifest.json'), sha256='5'*64),
            source_extension={str(p.c.Q/'seed-extension.py'): old_sources[str(p.c.Q/'seed-extension.py')]},
            input_extension={overlay_input[0]: overlay_input[1]}, output_root='/mnt/nvme/minicpm5-layer1-operator-native-mock-old'))
        self.scope_ref = self.meta('mock-lfm-scope.json', {'scope': 'mock'})
        seed_input = self.saved_input(str(p.c.Q/'mock-seed-input.json'))
        self.seed_ref = self.meta('mock-seed-spec.json', dict(schema='qwen35_sampler_runtime_spec_v1', run_name='mock-old',
            output_root='/mnt/nvme/neuron-work/old-mock', limits=copy.deepcopy(p.c.LIMITS), lock_identity=copy.deepcopy(p.c.LOCK_IDENTITY),
            native_python=p.c.PYTHON, automatic_successor=False, history_overlay=self.overlay_ref, current_serial_catalog=self.old_catalog_ref,
            source_sha256=old_sources, input_extension={seed_input[0]: seed_input[1], self.base_input[0]: self.base_input[1]},
            lfm_r2_scope=self.scope_ref, running_handoff_path=str(p.c.Q/'continuation-handoff-mock-old.json')))
        self.seed_review_ref = self.meta('mock-seed-review.json', dict(schema='qwen35_sampler_runtime_source_review_v1', reviewer='/root',
            passed=True, blockers=[], launch_authorized=True, source_sha256=old_sources, spec=self.seed_ref))
        self.meta('mock-seed-actual.json', dict(reviewer='/root', spec=self.seed_ref, review=self.seed_review_ref, actual_exit_code=0,
            actual_tool_exit_observed=True, actual_tool=tool, owners=owners, all_declared_units=units,
            owners_gone=True, physically_drained=True, full_model_qualified=True, paired_throughput_qualified=True), fixed=self.seed_actual_ref)
        input_pins = {}
        for path in (*p.c.STARTUP_FILES, str(p.c.Q/'qwen35_sampler_history_supplement_v1.py'),
                     str(p.c.Q/'qwen35_sampler_torch_namespace_sources_v1.py'), str(p.c.Q/'qwen35_sampler_pair_protocol_v1.py')):
            r = self.source(path)
            input_pins[path] = dict(sha256=r['sha256'], bytes=self.identities[path]['size'])
        input_manifest_ref = self.meta(str(self.directory/'input_pins.json'), dict(source_only=True, binary_payloads_read=False, files=input_pins))
        original_name = 'model-00001.safetensors'
        original_path = '/mnt/nvme/models/Qwen3.5-2B/'+original_name
        originals = {original_name: dict(size=500, sha256='9'*64)}
        source_files_ref = self.meta(str(self.directory/'source_files.json'), originals)
        runtime_files = {}
        for name in (*p.c.RUNTIME_FILES, 'input_pins.json', *(('source_text.py',) if version in (5, 6) else ())):
            path = str(self.directory/name)
            r = self.source(path) if path not in self.sources else dict(path=path, sha256=hashlib.sha256(self.sources[path]).hexdigest())
            runtime_files[path] = dict(sha256=r['sha256'], bytes=self.identities[path]['size'])
        runtime_ref = self.meta(str(self.directory/'source_delivery.json'), dict(schema='qwen35_package_serial_runtime_source_delivery_v'+str(version),
            source_only=True, launch_authorized=False, runtime_admitted=False, frozen_science_delivery=copy.deepcopy(p.c.SCIENCE_DELIVERY), files=runtime_files))
        runtime_review_ref = self.meta('mock-runtime-review.json', dict(schema='qwen35_package_serial_runtime_v'+str(version)+'_root_source_review_v1',
            reviewer='/root', passed=True, blockers=[], source_only=True, runtime_admitted=False, source_delivery=dict(runtime_ref, bytes=123)))
        science_files = {}
        for name in ('input_sources.json', 'capture_sources.json'):
            sub = self.source(str(p.c.Q/('science-'+name+'.py')))
            r = self.meta(str(p.c.Q/'qwen35_package_full_validation_v2'/name), {sub['path']: sub['sha256']})
            science_files[r['path']] = dict(sha256=r['sha256'], bytes=self.identities[r['path']]['size'])
        for name in ('__init__.py', 'runtime.py'):
            r = self.source(str(p.c.Q/'qwen35_package_full_validation_v2'/name))
            science_files[r['path']] = dict(sha256=r['sha256'], bytes=self.identities[r['path']]['size'])
        self.meta(p.c.SCIENCE_DELIVERY['path'], dict(files=science_files), fixed=p.c.SCIENCE_DELIVERY)
        prep_sources = {}
        self.helper = str(p.c.Q/'mock-root-preparation-helper.py')
        for path in (p.SELF, self.helper, str(p.c.HERE/'__init__.py'), str(p.c.HERE/'contract.py')):
            r = self.source(path)
            prep_sources[path] = r['sha256']
        case = dict(name='qwen35-package-full-v2-20261112', served_name='qwen3.5-2b-text', batch=4, tp=2,
            max_model_len=4096, cpu=True, adapter='qwen35_text', package_schema='vllm-neuron-ultra-qwen35-split-v1',
            neuron_configuration=configuration(), model_path='/mnt/nvme/models/qwen35-package-full-v2-mock',
            artifact='/mnt/nvme/neuron-artifacts/qwen35-package-full-v2-mock',
            compiler_cache='/mnt/nvme/neuron-work/qwen35-package-full-v2-mock-cache',
            compiler_workdir='/mnt/nvme/neuron-work/qwen35-package-full-v2-mock-work')
        self.identities[original_path] = ident(original_path, size=500)
        future = dict(run_name='mock-new', output_root='/mnt/nvme/neuron-work/qwp-mock-new',
            overlay_output_root='/mnt/nvme/minicpm5-layer1-operator-native-mock-new',
            running_handoff_path=str(p.c.Q/'continuation-handoff-mock-new.json'),
            overlay_path=str(p.c.Q/'mock-new-overlay.json'), spec_path=str(p.c.Q/'mock-new-spec.json'),
            review_candidate_path=str(p.c.Q/'mock-new-review-candidate.json'))
        budget = dict(schema='qwen35_package_reviewed_budget_v1', reviewer='/root', passed=True, blockers=[], case=case,
            output_root=future['output_root'], control_root='/mnt/nvme/neuron-work/qw-mock-new',
            classes={k: dict(allocated_bytes=64*p.MIB, single_file_bytes=32*p.MIB, entries=1000)
                     for k in ('view', 'artifact', 'compiler_cache', 'compiler_workdir', 'build_auxiliary')},
            headroom_bytes={'root': p.MIB, 'nvme': p.MIB},
            parent_identities={path: dict(realpath=path, device=1, inode=1)
                               for path in ('/mnt/nvme/neuron-work', '/mnt/nvme/neuron-artifacts', '/mnt/nvme/models')},
            source_files={original_name: dict(pin=originals[original_name], identity=self.identities[original_path])},
            build_file_limit_bytes=32*p.MIB)
        budget_ref = self.meta('mock-budget.json', budget)
        binaries = {}
        for path in p.c.BINARY_PATHS:
            self.identities[path] = ident(path, size=1000000000)
            binaries[path] = dict(kind='binary_payload', identity=self.identities[path])
        self.request = dict(seed_spec=self.seed_ref, seed_review=self.seed_review_ref, seed_actual_exit=self.seed_actual_ref,
            current_catalog=self.catalog_ref, idle_handoff=self.idle_ref, runtime_directory=str(self.directory),
            runtime_delivery=runtime_ref, runtime_review=runtime_review_ref, resource_budget=budget_ref, case=case,
            binary_inputs=binaries, preparation_sources=prep_sources, required_helpers=[self.helper], extra_metadata=[], future=future)

    def source(self, path):
        if path not in self.sources:
            self.sources[path] = ('# synthetic source '+path+'\n').encode()
            self.identities[path] = ident(path, len(self.sources[path]), len(self.identities)+1)
        return dict(path=path, sha256=hashlib.sha256(self.sources[path]).hexdigest())

    def meta(self, name, value, fixed=None):
        path = name if name.startswith('/') else str(p.c.Q/name)
        raw = p.encode(value)
        self.sources[path] = raw
        ref = fixed or dict(path=path, sha256=hashlib.sha256(raw).hexdigest())
        self.metadata[path] = dict(value=copy.deepcopy(value), ref=copy.deepcopy(ref))
        self.identities[path] = ident(path, len(raw), len(self.identities)+1)
        return copy.deepcopy(ref)

    def saved_input(self, path):
        ref = self.meta(path, {'test': path})
        return path, dict(sha256=ref['sha256'], identity=copy.deepcopy(self.identities[path]))

    def read_metadata(self, ref, cap):
        self.reads.append((ref['path'], cap, 'metadata'))
        row = self.metadata[ref['path']]
        assert ref == row['ref'], 'fixture ref mismatch'
        return dict(value=copy.deepcopy(row['value']), identity=copy.deepcopy(self.identities[ref['path']]))

    def read_source(self, ref, cap):
        self.reads.append((ref['path'], cap, 'source'))
        return dict(payload=self.sources[ref['path']], identity=copy.deepcopy(self.identities[ref['path']]))

    def identity(self, path):
        self.stats.append(path)
        return copy.deepcopy(self.identities[path])

    def guard(self):
        self.guards += 1

    def value(self, ref):
        return self.metadata[ref['path']]['value']

    def run(self, **kwargs):
        # All input IO is injected; even accidental ordinary filesystem reads,
        # writes, native imports, process launches and locks must fail here.
        with patch('builtins.open', side_effect=AssertionError('unexpected file IO')), \
             patch('os.open', side_effect=AssertionError('unexpected fd IO')), \
             patch('os.system', side_effect=AssertionError('unexpected command')), \
             patch('subprocess.Popen', side_effect=AssertionError('unexpected process')), \
             patch('fcntl.flock', side_effect=AssertionError('unexpected lock')), \
             patch.object(Path, 'is_symlink', return_value=False):
            return p.prepare(self.request, kwargs.get('read_metadata', self.read_metadata),
                kwargs.get('read_source', self.read_source), kwargs.get('identity_only', self.identity), self.guard)


class Preparation(unittest.TestCase):
    def test_all_original_sources_inputs_and_recovered_partition_retained(self):
        f = Fixture(); out = f.run()
        old, overlay, base = (f.value(x) for x in (f.seed_ref, f.overlay_ref, f.base_ref))
        for path, sha in old['source_sha256'].items():
            self.assertEqual(out['spec']['source_sha256'][path], sha)
        for old_inputs in (old['input_extension'], overlay['input_extension']):
            for path, row in old_inputs.items():
                self.assertEqual(out['spec']['input_extension'][path], row)
        self.assertEqual(out['overlay']['base_plan'], f.base_ref)
        self.assertEqual(f.value(f.base_ref)['inputs'], base['inputs'])
        recovered_path = str(p.c.Q/'recovery-only.py')
        self.assertIn(recovered_path, out['spec']['source_sha256'])
        self.assertNotIn(recovered_path, out['overlay']['source_extension'])
        self.assertEqual(out['preservation']['recovery_only_sources'], 1)
        self.assertNotIn(f.binary_base_input['identity']['realpath'], out['spec']['input_extension'])

    def test_variable_catalog_lengths_and_no_full_map_duplication(self):
        for old, current in ((2, 2), (2, 5), (3, 8)):
            with self.subTest(old=old, current=current):
                f = Fixture(old, current); out = f.run()
                self.assertEqual(out['preservation']['old_catalog_rows'], old)
                self.assertEqual(out['preservation']['current_catalog_rows'], current)
                self.assertEqual(out['catalog_source_extension'], out['overlay']['source_extension'])
                self.assertLess(len(out['catalog_source_extension']), len(out['spec']['source_sha256']))

    def test_candidate_never_authorizes_launch(self):
        out = Fixture().run()
        for value in (out, out['review']):
            self.assertIs(value['passed'], False); self.assertIs(value['launch_authorized'], False)
        self.assertFalse(out['history_resolved']); self.assertFalse(out['native_executed'])
        with self.assertRaises(ValueError):
            p.c.validate_spec(out['spec'], out['review'], out['spec_ref'])
        self.assertEqual(json.loads(out['review_bytes']), out['review'])
        for name in ('spec', 'overlay', 'review'):
            self.assertEqual(hashlib.sha256(out[name+'_bytes']).hexdigest(), out[name+'_ref']['sha256'])
            self.assertTrue(out[name+'_bytes'].endswith(b'\n'))

    def test_v3_explicit_delivery_and_all_prep_import_sources_retained(self):
        f = Fixture(version=3); out = f.run()
        self.assertIn(str(f.directory/'run.py'), out['spec']['source_sha256'])
        for path, sha in f.request['preparation_sources'].items():
            self.assertEqual(out['spec']['source_sha256'][path], sha)

    def test_v4_explicit_delivery_retains_all_sources_inputs_and_false_review(self):
        f = Fixture(version=4); out = f.run()
        self.assertEqual(out['schema'], 'qwen35_package_serial_candidate_v5')
        self.assertIn(str(f.directory/'history.py'), out['spec']['source_sha256'])
        for path, sha in f.value(f.seed_ref)['source_sha256'].items():
            self.assertEqual(out['spec']['source_sha256'][path], sha)
        for inputs in (f.value(f.seed_ref)['input_extension'], f.value(f.overlay_ref)['input_extension']):
            for path, row in inputs.items():
                self.assertEqual(out['spec']['input_extension'][path], row)
        self.assertFalse(out['passed']); self.assertFalse(out['launch_authorized'])
        with self.assertRaises(ValueError):
            p.c.validate_spec(out['spec'], out['review'], out['spec_ref'])
        self.assertFalse(set(p.c.BINARY_PATHS) & {row[0] for row in f.reads})

    def test_v4_rejects_mismatched_review_or_delivery_schema(self):
        for target in ('runtime_review', 'runtime_delivery'):
            f = Fixture(version=4)
            f.value(f.request[target])['schema'] = f.value(f.request[target])['schema'].replace('_v4', '_v3')
            with self.subTest(target=target), self.assertRaises(ValueError):
                f.run()

    def test_v4_retains_exact_ordinary_and_explicit_seed_envelopes(self):
        f = Fixture(version=4)
        f.identities[f.seed_ref['path']]['size'] = 8804235
        out = f.run()
        self.assertEqual(p.ORDINARY_CAP, 8*1024**2)
        self.assertEqual(p.ENVELOPE_CAP, 64*1024**2)
        self.assertNotIn(f.seed_ref['path'], out['spec']['input_extension'])
        self.assertNotIn(f.seed_ref['path'], out['overlay']['input_extension'])
        self.assertEqual([row[1:] for row in f.reads if row[0] == f.seed_ref['path']],
                         [(p.ENVELOPE_CAP, 'metadata')])
        f = Fixture(version=4)
        f.identities[f.catalog_ref['path']]['size'] = 8804235
        with self.assertRaisesRegex(ValueError, 'metadata cap'):
            f.run()

    def test_v4_runtime_file_and_delivery_pins_cannot_be_rebound(self):
        for target in ('review_ref', 'runtime_source'):
            f = Fixture(version=4)
            if target == 'review_ref':
                f.value(f.request['runtime_review'])['source_delivery']['sha256'] = 'f'*64
            else:
                f.value(f.request['runtime_delivery'])['files'][str(f.directory/'history.py')]['sha256'] = 'f'*64
            with self.subTest(target=target), self.assertRaises(ValueError):
                f.run()

    def test_unknown_future_runtime_version_is_not_implicitly_accepted(self):
        f = Fixture(version=7)
        with self.assertRaisesRegex(ValueError, 'only explicit reviewed'):
            f.run()

    def test_v5_complete_new_source_module_retained_and_false_candidate(self):
        f = Fixture(version=5); out = f.run()
        path = str(f.directory/'source_text.py')
        self.assertEqual(out['spec']['source_sha256'][path], hashlib.sha256(f.sources[path]).hexdigest())
        self.assertEqual(out['schema'], 'qwen35_package_serial_candidate_v5')
        self.assertFalse(out['passed']); self.assertFalse(out['launch_authorized'])
        self.assertFalse(out['history_resolved']); self.assertFalse(out['native_executed'])
        self.assertFalse(set(p.c.BINARY_PATHS) & {row[0] for row in f.reads})

    def test_v5_new_verifier_module_cannot_be_omitted_or_mutated(self):
        for mode in ('missing', 'changed'):
            f = Fixture(version=5); path = str(f.directory/'source_text.py')
            if mode == 'missing': del f.value(f.request['runtime_delivery'])['files'][path]
            else: f.sources[path] += b'changed'
            with self.subTest(mode=mode), self.assertRaises(ValueError): f.run()

    def test_v5_mismatched_review_and_delivery_schemas_rejected(self):
        for target in ('runtime_review', 'runtime_delivery'):
            f = Fixture(version=5)
            f.value(f.request[target])['schema'] = f.value(f.request[target])['schema'].replace('_v5', '_v4')
            with self.subTest(target=target), self.assertRaises(ValueError): f.run()

    def test_v5_caps_and_original_source_input_preservation(self):
        f = Fixture(version=5); out = f.run()
        self.assertEqual((p.ORDINARY_CAP, p.ENVELOPE_CAP), (8*1024**2, 64*1024**2))
        for path, sha in f.value(f.seed_ref)['source_sha256'].items():
            self.assertEqual(out['spec']['source_sha256'][path], sha)
        for ref in (f.seed_ref, f.overlay_ref):
            for path, value in f.value(ref)['input_extension'].items():
                self.assertEqual(out['spec']['input_extension'][path], value)
        f.identities[f.catalog_ref['path']]['size'] = p.ORDINARY_CAP + 1
        with self.assertRaisesRegex(ValueError, 'metadata cap'): f.run()

    def test_v6_complete_new_source_module_retained_and_false_candidate(self):
        f = Fixture(version=6); out = f.run()
        path = str(f.directory/'source_text.py')
        self.assertEqual(out['spec']['source_sha256'][path], hashlib.sha256(f.sources[path]).hexdigest())
        self.assertEqual(out['schema'], 'qwen35_package_serial_candidate_v5')
        self.assertFalse(out['passed']); self.assertFalse(out['launch_authorized'])
        self.assertFalse(out['history_resolved']); self.assertFalse(out['native_executed'])
        self.assertFalse(set(p.c.BINARY_PATHS) & {row[0] for row in f.reads})

    def test_v6_new_verifier_module_cannot_be_omitted_or_mutated(self):
        for mode in ('missing', 'changed'):
            f = Fixture(version=6); path = str(f.directory/'source_text.py')
            if mode == 'missing': del f.value(f.request['runtime_delivery'])['files'][path]
            else: f.sources[path] += b'changed'
            with self.subTest(mode=mode), self.assertRaises(ValueError): f.run()

    def test_v6_mismatched_review_and_delivery_schemas_rejected(self):
        for target in ('runtime_review', 'runtime_delivery'):
            f = Fixture(version=6)
            f.value(f.request[target])['schema'] = f.value(f.request[target])['schema'].replace('_v6', '_v5')
            with self.subTest(target=target), self.assertRaises(ValueError): f.run()

    def test_v6_caps_and_original_source_input_preservation(self):
        f = Fixture(version=6); out = f.run()
        self.assertEqual((p.ORDINARY_CAP, p.ENVELOPE_CAP), (8*1024**2, 64*1024**2))
        for path, sha in f.value(f.seed_ref)['source_sha256'].items():
            self.assertEqual(out['spec']['source_sha256'][path], sha)
        for ref in (f.seed_ref, f.overlay_ref):
            for path, value in f.value(ref)['input_extension'].items():
                self.assertEqual(out['spec']['input_extension'][path], value)
        f.identities[f.catalog_ref['path']]['size'] = p.ORDINARY_CAP + 1
        with self.assertRaisesRegex(ValueError, 'metadata cap'): f.run()

    def test_arbitrary_runtime_directory_rejected(self):
        f = Fixture(); f.request['runtime_directory'] = '/tmp/arbitrary-runtime'
        with self.assertRaisesRegex(ValueError, 'only explicit'):
            f.run()

    def test_catalog_removal_reorder_and_duplicate_rejected(self):
        for mutation in ('remove', 'reorder', 'duplicate'):
            f = Fixture(); rows = f.value(f.catalog_ref)['runs']
            if mutation == 'remove': rows.pop(1)
            elif mutation == 'reorder': rows[0], rows[1] = rows[1], rows[0]
            else: rows.append(copy.deepcopy(rows[0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                f.run()

    def test_original_failed_rows_not_recast_or_dropped(self):
        f = Fixture()
        for ref in (f.old_catalog_ref, f.catalog_ref):
            f.value(ref)['runs'][1]['tool']['exit_code'] = 1
        out = f.run()
        self.assertEqual(out['preservation']['current_catalog_rows'], 5)
        self.assertEqual(f.value(f.catalog_ref)['runs'][1]['tool']['exit_code'], 1)

    def test_catalog_source_conflict_rejected(self):
        f = Fixture(); f.value(f.catalog_ref)['runs'][-1]['source_sha256'][str(p.c.Q/'base-source.py')] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'source pin conflict'):
            f.run()

    def test_actual_seed_row_and_review_must_match(self):
        for target in ('review', 'actual', 'catalog'):
            f = Fixture()
            if target == 'review': f.value(f.seed_review_ref)['launch_authorized'] = False
            elif target == 'actual': f.value(f.seed_actual_ref)['physically_drained'] = False
            else: f.value(f.catalog_ref)['runs'][0]['tool']['session_id'] = 999
            with self.subTest(target=target), self.assertRaises(ValueError):
                f.run()

    def test_incompatible_python_seed_input_rejected_not_deleted(self):
        f = Fixture(); path = str(p.c.Q/'old-helper.py')
        f.value(f.seed_ref)['input_extension'][path] = dict(sha256='a'*64, identity=ident(path))
        with self.assertRaisesRegex(ValueError, 'extension inputs must be JSON'):
            f.run()
        self.assertIn(path, f.value(f.seed_ref)['input_extension'])

    def test_original_identity_conflict_rejected(self):
        f = Fixture(); old = f.value(f.overlay_ref)['input_extension']
        old[f.base_input[0]] = copy.deepcopy(f.base_input[1]); old[f.base_input[0]]['identity']['inode'] += 1
        with self.assertRaisesRegex(ValueError, 'identity conflict'):
            f.run()

    def test_ordinary_cap_and_exact_large_preparation_exclusions(self):
        f = Fixture()
        for ref in (f.seed_ref, f.seed_review_ref, f.base_ref):
            f.identities[ref['path']]['size'] = p.ORDINARY_CAP+1
        out = f.run()
        self.assertEqual(set(out['preparation_only_references']), {x['path'] for x in (f.seed_ref, f.seed_review_ref, f.base_ref)})
        self.assertFalse(set(out['preparation_only_references']) & set(out['spec']['input_extension']))
        f = Fixture(); f.identities[f.catalog_ref['path']]['size'] = p.ORDINARY_CAP+1
        with self.assertRaisesRegex(ValueError, 'metadata cap'):
            f.run()

    def test_missing_helper_and_own_source_rejected(self):
        for path_kind in ('helper', 'self'):
            f = Fixture(); del f.request['preparation_sources'][f.helper if path_kind == 'helper' else p.SELF]
            with self.assertRaisesRegex(ValueError, 'every explicit root helper'):
                f.run()

    def test_missing_runtime_source_rejected(self):
        f = Fixture(); del f.value(f.request['runtime_delivery'])['files'][str(f.directory/'history.py')]
        with self.assertRaisesRegex(ValueError, 'runtime file closure'):
            f.run()

    def test_source_digest_and_metadata_identity_mutations_rejected(self):
        f = Fixture(); f.sources[f.helper] += b'# changed\n'
        with self.assertRaisesRegex(ValueError, 'authenticated source'):
            f.run()
        f = Fixture()
        def mutate(path):
            value = f.identity(path)
            if path == f.idle_ref['path']: value['inode'] += 1
            return value
        with self.assertRaisesRegex(ValueError, 'changed before return'):
            f.run(identity_only=mutate)

    def test_all_six_binaries_only_statted_never_read(self):
        f = Fixture(); f.run()
        self.assertTrue(set(p.c.BINARY_PATHS).issubset(f.stats))
        self.assertFalse(set(p.c.BINARY_PATHS) & {row[0] for row in f.reads})
        f = Fixture(); f.request['binary_inputs'].pop(p.c.BINARY_PATHS[0])
        with self.assertRaisesRegex(ValueError, 'six exact binary'):
            f.run()

    def test_binary_identity_change_rejected(self):
        f = Fixture()
        def mutate(path):
            value = f.identity(path)
            if path == p.c.BINARY_PATHS[0]: value['size'] += 1
            return value
        with self.assertRaisesRegex(ValueError, 'binary stat identity changed'):
            f.run(identity_only=mutate)

    def test_budget_case_and_auxiliary_limit_rejected(self):
        for field in ('case', 'headroom', 'auxiliary', 'filelimit', 'parents', 'sources'):
            f = Fixture(); value = f.value(f.request['resource_budget'])
            if field == 'case': value['case']['tp'] = 8
            elif field == 'headroom': value['headroom_bytes']['nvme'] = 0
            elif field == 'auxiliary': value['classes']['build_auxiliary']['allocated_bytes'] = 1
            elif field == 'filelimit': value['build_file_limit_bytes'] = 1
            elif field == 'parents': value['parent_identities'].pop('/mnt/nvme/models')
            else: value['source_files'] = {}
            with self.subTest(field=field), self.assertRaises(ValueError):
                f.run()

    def test_future_overwrite_and_ipc_path_rejected(self):
        f = Fixture(); f.request['future']['spec_path'] = f.seed_ref['path']
        with self.assertRaisesRegex(ValueError, 'cannot overwrite'):
            f.run()
        f = Fixture(); future = f.request['future']; future['run_name'] = 'x'*40
        future['output_root'] = '/mnt/nvme/neuron-work/qwp-'+future['run_name']
        with self.assertRaisesRegex(ValueError, 'IPC path'):
            f.run()

    def test_runtime_review_cannot_bind_other_delivery(self):
        f = Fixture(); f.value(f.request['runtime_review'])['source_delivery']['sha256'] = '1'*64
        with self.assertRaisesRegex(ValueError, 'root-selected'):
            f.run()

    def test_binary_cannot_be_disguised_as_preparation_source(self):
        f = Fixture(); f.request['preparation_sources'][p.c.BINARY_PATHS[0]] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'never open binary'):
            f.run()
        self.assertFalse(set(p.c.BINARY_PATHS) & {row[0] for row in f.reads})

    def test_seed_science_and_package_configuration_unchanged(self):
        for target in ('limits', 'case'):
            f = Fixture()
            if target == 'limits': f.value(f.seed_ref)['limits']['metadata_bytes'] *= 2
            else: f.request['case']['neuron_configuration']['seq_len'] = 2048
            with self.subTest(target=target), self.assertRaises(ValueError):
                f.run()

    def test_input_manifest_observation_cannot_rebind_source_identity(self):
        f = Fixture(); path = str(f.directory/'input_pins.json')
        def reader(ref, cap):
            value = f.read_metadata(ref, cap)
            if ref['path'] == path: value['identity']['inode'] += 1
            return value
        with self.assertRaisesRegex(ValueError, 'observations must agree'):
            f.run(read_metadata=reader)

    def test_no_native_or_execution_and_guard_rechecked(self):
        f = Fixture(); f.run()
        self.assertGreater(f.guards, 50)
        self.assertFalse(any(x.split('.')[0] in {'torch', 'vllm', 'numpy'} for x in sys.modules))

    def test_all_four_future_metadata_paths_reject_all_retained_refs(self):
        for output in ('overlay_path', 'spec_path', 'review_candidate_path', 'running_handoff_path'):
            for origin in ('sdk_operator_recovery', 'native_manifest', 'completed_sdk', 'later_terminal'):
                with self.subTest(output=output, origin=origin):
                    f = Fixture()
                    # Use a handoff-shaped reference so even running_handoff_path
                    # is otherwise valid; rejection must come from collision.
                    path = str(p.c.Q/'continuation-handoff-retained-ref.json')
                    ref = dict(path=path, sha256='d'*64)
                    if origin in ('sdk_operator_recovery', 'native_manifest'):
                        f.value(f.overlay_ref)[origin] = ref
                    elif origin == 'completed_sdk':
                        for cat in (f.old_catalog_ref, f.catalog_ref):
                            f.value(cat)['completed_sdk'] = ref
                    else:
                        f.value(f.catalog_ref)['runs'][-1]['terminal'] = ref
                    f.request['future'][output] = path
                    with self.assertRaisesRegex(ValueError, 'cannot overwrite'):
                        f.run()

    def test_annotated_and_lexical_alias_retained_refs_reject_collision(self):
        for path_kind in ('annotated', 'lexical_alias'):
            f = Fixture()
            target = f.request['future']['overlay_path']
            source_path = str(p.c.Q/'unused'/'..'/Path(target).name) if path_kind == 'lexical_alias' else target
            f.value(f.base_ref)['nested_original_evidence'] = [{'ref': dict(path=source_path, sha256='d'*64, bytes=12)}]
            with self.subTest(path_kind=path_kind), self.assertRaisesRegex(ValueError, 'cannot overwrite'):
                f.run()

    def test_injected_metadata_nonfinite_exponent_value_rejected(self):
        f = Fixture()
        f.value(f.scope_ref)['original_diagnostic'] = [{'nested_number': float('inf')}]
        with self.assertRaisesRegex(ValueError, 'finite JSON numbers'):
            f.run()

    def test_nested_finite_json_numbers_remain_accepted(self):
        f = Fixture()
        f.value(f.scope_ref)['original_diagnostic'] = [1e308, {'tiny': -1e-308}, 12345678901234567890]
        out = f.run()
        self.assertFalse(out['launch_authorized'])


class StableReader(unittest.TestCase):
    def test_overflow_exponents_rejected_at_every_json_depth(self):
        payloads = (b'1e999', b'-1e999', b'{"number":1e999}', b'[-1e999]', b'{"nested":[{"value":1e999}]}')
        with tempfile.TemporaryDirectory(prefix='qwen-preparer-source-mock-') as temp:
            path = Path(temp)/'overflow.json'
            for payload in payloads:
                path.write_bytes(payload)
                ref = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
                with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, 'finite JSON numbers'):
                    p.StableReaders(lambda: None).metadata(ref)

    def test_real_small_temp_json_only_and_duplicate_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qwen-preparer-source-mock-') as temp:
            path = Path(temp)/'fixture.json'
            for payload, good in ((b'{"value":1}\n', True), (b'{"a":1,"a":2}', False), (b'{"a":NaN}', False)):
                path.write_bytes(payload)
                ref = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())
                reader = p.StableReaders(lambda: None)
                if good:
                    self.assertEqual(reader.metadata(ref)['value'], {'value': 1})
                else:
                    with self.assertRaises(ValueError): reader.metadata(ref)

    def test_changed_identity_and_digest_rejected(self):
        with tempfile.TemporaryDirectory(prefix='qwen-preparer-source-mock-') as temp:
            path = Path(temp)/'fixture.json'; path.write_bytes(b'{}')
            ref = dict(path=str(path), sha256=hashlib.sha256(b'{}').hexdigest())
            reader = p.StableReaders(lambda: None); reader.metadata(ref)
            path.write_bytes(b'{ }')
            with self.assertRaises(ValueError): reader.metadata(ref)
            new = dict(path=str(path), sha256=hashlib.sha256(b'{ }').hexdigest())
            with self.assertRaisesRegex(ValueError, 'cannot be rebound'): reader.metadata(new)

    def test_caps_and_non_json_payload_rejected_before_open(self):
        reader = p.StableReaders(lambda: None)
        with self.assertRaisesRegex(ValueError, 'JSON metadata'):
            reader.metadata(dict(path='/tmp/never-open.pt', sha256='a'*64))
        with tempfile.TemporaryDirectory(prefix='qwen-preparer-source-mock-') as temp:
            path = Path(temp)/'fixture.json'; path.write_bytes(b'{}')
            ref = dict(path=str(path), sha256=hashlib.sha256(b'{}').hexdigest())
            with self.assertRaisesRegex(ValueError, 'finite envelope'): reader.metadata(ref, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
