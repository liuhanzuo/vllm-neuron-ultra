"""Bounded stdlib/source/mock checks. No native import, History or real lock."""
import ast
import copy
import hashlib
import importlib.abc
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

Q = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Q), str(Q.parent)]


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_neuronx', 'torch_xla', 'neuronxcc', 'neuronx_distributed',
                'neuronx_distributed_inference', 'numpy', 'vllm', 'vllm_neuron', 'transformers', 'ctypes'}:
            raise AssertionError('Native import forbidden: ' + fullname)


sys.meta_path.insert(0, NoNative())
from qwen35_writer_pair_validation_v1 import runtime, http, geometry, capture_worker
from qwen35_writer_pair_serial_runtime_v2 import contract, wiring, io, budget, adapter, qualification


def cases(sampler='1'):
    base = json.loads((Q / 'qwen35_package_full_validation_v2/case.example.json').read_text())
    values = {}
    for arm in runtime.ARMS:
        value = copy.deepcopy(base)
        value.update(name='qwen35-writer-pair-v1-20261112', writer_arm=arm,
                     writer_mode=runtime.WRITERS[arm], sampler_fast=sampler)
        for key in ('model_path', *runtime.TREE_KEYS):
            value[key] = str(Path(value[key]).parent / ('qwen35-writer-pair-v1-' + arm + '-' + key))
        values[arm] = value
    return values


def writer_identity(mode):
    return dict(schema='package', sources={'same': 'a' * 64}, cache_writer=dict(
        schema='qwen35-cache-writer-v1', mode=mode, decode_writer=mode, prefill_writer='torch_where',
        slots_dtype='torch.int32', graphs_per_artifact={'prefill': 6, 'decode': 6}))


class Cases(unittest.TestCase):
    def test_both_explicit_equal_sampler_modes_accepted(self):
        for mode in ('0', '1'):
            runtime.validate_pair_cases(cases(mode))

    def test_sampler_confounded_writer_comparison_rejected(self):
        value = cases(); value['baseline']['sampler_fast'] = '0'
        with self.assertRaisesRegex(ValueError, 'sampler'): runtime.validate_pair_cases(value)

    def test_old_single_case_and_wrong_arm_rejected(self):
        value = cases()
        for change in ('old', 'cross'):
            selected = copy.deepcopy(value)
            if change == 'old': selected.pop('candidate')
            else: selected['candidate']['writer_arm'] = 'baseline'
            with self.subTest(change=change), self.assertRaises(ValueError): runtime.validate_pair_cases(selected)

    def test_every_shared_build_tree_is_rejected(self):
        for key in ('model_path', *runtime.TREE_KEYS):
            value = cases(); value['candidate'][key] = value['baseline'][key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'independent'): runtime.validate_pair_cases(value)

    def test_environment_uses_same_sampler_and_different_writer_and_artifacts(self):
        value = cases('0'); envs = [runtime.native_environment(value[a]) for a in runtime.ARMS]
        for arm, env in zip(runtime.ARMS, envs):
            self.assertEqual(env['NEURON_CPU_SAMPLER_FAST'], '0')
            self.assertEqual(env['NEURON_QWEN35_CACHE_WRITER'], runtime.WRITERS[arm])
            self.assertEqual(env['NEURON_COMPILED_ARTIFACTS'], value[arm]['artifact'])
            self.assertTrue(env['PYTHONPATH'].startswith(str(runtime.CHECKOUT) + ':'))

    def test_two_complete_builds_and_cross_writer_identity(self):
        value = cases(); reports = {}
        for arm in runtime.ARMS:
            reports[arm] = dict(case=value[arm], artifact_identity=writer_identity(runtime.WRITERS[arm]),
                compiled_graph_files=[dict(tag=tag, index=i, bucket=b) for tag in
                    ('context_encoding_model', 'token_generation_model') for i, b in enumerate(runtime.BUCKETS)])
        def manifest(path):
            arm = next(a for a in runtime.ARMS if str(path).startswith(value[a]['artifact']))
            return dict(complete=True, identity=reports[arm]['artifact_identity'])
        with patch.object(runtime, 'read_build_inventory', side_effect=lambda key: reports[key]), \
             patch.object(runtime, 'read', side_effect=manifest):
            self.assertEqual(runtime.check_build_pair(value, dict(baseline='baseline', candidate='candidate'))['graphs_observed'], 24)
            reports['candidate']['artifact_identity'] = reports['baseline']['artifact_identity']
            with self.assertRaises(ValueError): runtime.check_build_pair(value, dict(baseline='baseline', candidate='candidate'))


class Hosts(unittest.TestCase):
    def reports(self):
        value = cases(); hosts = {}; directories = {}
        for i, arm in enumerate(runtime.ARMS):
            directory = Path('/mock/pair') / arm / 'http'; directories[arm] = str(directory)
            env = runtime.native_environment(value[arm])
            env.update(TMPDIR=str(directory / 'tmp'), VLLM_RPC_BASE_PATH=str(directory / 'rpc'))
            argv = runtime.server_arguments(value[arm])
            hosts[arm] = dict(observation_kind='live_ready', ready=True, server_active=True, exit_observed=False,
                startup_config_proof=dict(actual_api_initial_environment=env, actual_api_argv=argv),
                api_process_owner=dict(argv=argv, pid=10 + i))
        return value, hosts, directories

    def test_matched_hosts_keep_sampler_in_compared_environment(self):
        value, hosts, directories = self.reports()
        self.assertTrue(http.check_matched_hosts(hosts['baseline'], hosts['candidate'], value, directories)['matched'])
        hosts['candidate']['startup_config_proof']['actual_api_initial_environment']['NEURON_CPU_SAMPLER_FAST'] = '0'
        with self.assertRaises(ValueError): http.check_matched_hosts(hosts['baseline'], hosts['candidate'], value, directories)

    def test_artifact_swap_debug_threads_and_capture_hook_rejected(self):
        for key, wrong in [('NEURON_COMPILED_ARTIFACTS', '/wrong'), ('OMP_NUM_THREADS', '2'),
                           ('VLLM_LOGGING_LEVEL', 'DEBUG'), ('QWEN35_PACKAGE_CAPTURE_ROOT', '/bad')]:
            value, hosts, directories = self.reports()
            hosts['candidate']['startup_config_proof']['actual_api_initial_environment'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                http.check_matched_hosts(hosts['baseline'], hosts['candidate'], value, directories)

    def test_capture_identity_same_operation_on_both_writer_arms(self):
        for mode in ('0', '1'):
            values = cases(mode)
            identities = [capture_worker.capture_identity(values[a], a, 'a' * 64) for a in runtime.ARMS]
            self.assertEqual(identities[0]['operation_class'], identities[1]['operation_class'])
            self.assertNotEqual(identities[0]['writer_mode'], identities[1]['writer_mode'])
            for arm in runtime.ARMS:
                capture_worker.check_environment(runtime.native_environment(values[arm]), arm, mode)


class Lifecycle(unittest.TestCase):
    def test_stage_mapping_keeps_all_original_requirements_and_two_builds(self):
        from qwen35_package_serial_runtime_v6.wiring import STAGES as old
        self.assertEqual(len(wiring.STAGES), 28)
        self.assertEqual(wiring.STAGES[5:], old[2:])
        self.assertEqual(wiring.STAGES[:5], (('baseline', 'prepare'), ('baseline', 'build'),
            ('candidate', 'prepare'), ('candidate', 'build'), ('shared', 'build_pair_gate')))
        self.assertEqual(set(contract.PHASES), set(wiring.STAGES) - {
            ('shared', 'build_pair_gate'), ('shared', 'sampler_pair_gate'), ('shared', 'quality_pair_gate'),
            ('shared', 'full_pair_gate'), ('external', 'independent_final_audit')})

    def test_case_inventory_directory_and_api_paths_are_per_arm(self):
        spec = dict(writer_cases=cases(), output_root='/mock/pair')
        self.assertNotEqual(wiring.build_inventory(spec, 'baseline'), wiring.build_inventory(spec, 'candidate'))
        for arm in runtime.ARMS:
            self.assertEqual(wiring.phase_directory(spec, arm, 'build'), Path('/mock/pair') / arm / 'build')
            self.assertIn(spec['writer_cases'][arm]['model_path'], wiring.api_arguments(spec, arm, 'http'))
        self.assertEqual(len(set(budget.roots(dict(spec, run_name='mock')))), 10)

    def test_compile_mutation_freezes_each_arm_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = dict(output_root=temp, writer_cases=cases())
            for arm in runtime.ARMS:
                spec['writer_cases'][arm]['compiler_workdir'] = str(Path(temp) / (arm + '-work'))
                spec['writer_cases'][arm]['compiler_cache'] = str(Path(temp) / (arm + '-cache'))
                self.assertTrue(io.compiler_mutation(Path(temp) / (arm + '-work/layout_opt/temporary'), spec))
            (Path(temp) / 'baseline').mkdir(); (Path(temp) / 'baseline/build-immutable-identities.json').write_text('{}')
            self.assertFalse(io.compiler_mutation(Path(temp) / 'baseline-work/layout_opt/temporary', spec))
            self.assertTrue(io.compiler_mutation(Path(temp) / 'candidate-work/layout_opt/temporary', spec))
            self.assertFalse(io.compiler_mutation(Path(temp) / 'foreign/anything', spec))

    def test_complete_driver_sequence_with_real_local_receipt_writes_only(self):
        from qwen35_writer_pair_validation_v1 import sampler
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'pair'; value = cases()
            spec = dict(writer_cases=value, case=value['baseline'], output_root=str(root))
            owner = types.SimpleNamespace(spec=spec, active=None, admission={'attempt_id': 'mock'}, guard=Mock())
            order = []
            class FakePhase:
                def __init__(self, adapter, arm, name, directory, run_id): self.arm, self.name = arm, name
                def leaf(self):
                    order.append((self.arm, self.name))
                    return {'mock_only': True}
            def server(_owner, arm, name, hosts):
                order.append((arm, name)); return {'mock_only': True}
            def gate(name):
                return lambda *a, **k: (order.append(('shared', name)) or {'mock_only': True})
            with patch.object(qualification, 'check_geometry_qualification', return_value={'mock_only': True}), \
                 patch.object(adapter, 'Phase', FakePhase), patch.object(wiring, 'run_server', side_effect=server), \
                 patch.object(wiring, 'seal_build', side_effect=lambda _s, a: {'arm': a}), \
                 patch.object(runtime, 'check_build_pair', side_effect=gate('build_pair_gate')), \
                 patch.object(sampler, 'check_saved_sampler_gates', side_effect=gate('sampler_pair_gate')), \
                 patch.object(http, 'check_saved_quality', side_effect=gate('quality_pair_gate')), \
                 patch.object(http, 'compare_saved_pair', side_effect=gate('full_pair_gate')):
                result = wiring.run(owner)
            self.assertEqual(order, list(wiring.STAGES[:-1]))
            self.assertEqual(result['completed_stages'], 27)
            self.assertFalse(result['full_model_qualified'])
            self.assertFalse((root / 'stages/28.json').exists())
            for i in range(1, 28):
                row = json.loads((root / 'stages' / ('%02d.json' % i)).read_text())
                self.assertEqual(row['writer_cases'], value)

    def test_no_geometry_qualification_stops_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'pair'; owner = types.SimpleNamespace(spec={'output_root': str(root)})
            with patch.object(qualification, 'check_geometry_qualification', side_effect=ValueError('missing actual geometry')):
                with self.assertRaisesRegex(ValueError, 'missing actual geometry'): wiring.run(owner)
            self.assertFalse(root.exists())


class Preservation(unittest.TestCase):
    def test_original_model_scientific_bodies_are_byte_identical(self):
        new = Q / 'qwen35_writer_pair_validation_v1'; old = Q / 'qwen35_package_full_validation_v2'
        for name in ('numeric.py', 'cold.py', 'long.py', 'state_copy.py', 'state_observation.py', 'capture_writer.py'):
            self.assertEqual((new / name).read_bytes(), (old / name).read_bytes(), name)

    def test_vector_bitwise_gate_unchanged(self):
        old = ast.parse((Q / 'qwen35_package_full_validation_v2/entry.py').read_text())
        new = ast.parse((Q / 'qwen35_writer_pair_validation_v1/entry.py').read_text())
        function = lambda x: ast.dump(next(n for n in x.body if isinstance(n, ast.FunctionDef) and n.name == 'compare_vectors'))
        self.assertEqual(function(old), function(new))

    def test_original_service_history_and_cleanup_unchanged_except_namespace(self):
        old = Q / 'qwen35_package_serial_runtime_v6'; new = Q / 'qwen35_writer_pair_serial_runtime_v1'
        for name in ('service.py', 'history.py', 'source_text.py', 'run.py'):
            expected = (old / name).read_text().replace('qwen35_package_serial_runtime_v6', new.name).replace(
                'qwen35_package_full_validation_v2', 'qwen35_writer_pair_validation_v1')
            self.assertEqual(expected, (new / name).read_text(), name)

    def test_every_scientific_source_parses_without_native_import(self):
        for directory in ('qwen35_writer_pair_validation_v1', 'qwen35_writer_pair_serial_runtime_v1'):
            for path in (Q / directory).glob('*.py'): ast.parse(path.read_text(), filename=str(path))
        self.assertFalse(any(name.split('.')[0] in ('torch', 'numpy', 'vllm', 'neuronxcc') for name in sys.modules))

    def test_every_port_edit_and_append_reconstructs_the_successor(self):
        for name in ('qwen35_writer_pair_validation_v1', 'qwen35_writer_pair_serial_runtime_v1'):
            directory = Q / name
            parents = json.loads((directory / 'parent_sources.json').read_text())
            edits = json.loads((directory / 'port_edits.json').read_text())
            appends = directory / 'port_appends.json'
            appends = json.loads(appends.read_text()) if appends.exists() else {}
            for file, row in parents.items():
                source = Path(row['source']).read_bytes()
                self.assertEqual(hashlib.sha256(source).hexdigest(), row['sha256'])
                self.assertEqual(len(source), row['bytes'])
                expected = source.decode()
                if name.endswith('serial_runtime_v1'):
                    expected = expected.replace('qwen35_package_serial_runtime_v6', name).replace(
                        'qwen35_package_full_validation_v2', 'qwen35_writer_pair_validation_v1')
                for edit in edits.get(file, []):
                    self.assertEqual(expected.count(edit['before']), edit['count'], file)
                    expected = expected.replace(edit['before'], edit['after'], edit['count'])
                self.assertEqual(expected + appends.get(file, ''), (directory / file).read_text(), file)

    def test_geometry_and_capture_sources_are_in_the_complete_input_closure(self):
        directory = Q / 'qwen35_writer_pair_validation_v1'
        inputs = json.loads((directory / 'input_sources.json').read_text())
        self.assertEqual(geometry.verify_sources(), json.loads((directory / 'geometry_sources.json').read_text()))
        for name in ('capture_sources.json', 'geometry_sources.json'):
            sources = json.loads((directory / name).read_text())
            for path, sha in sources.items():
                self.assertEqual(inputs[path], sha, path)
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), sha, path)
        for name in ('cache_writer.py', 'slot_cache_nki_predicated.py', 'text_core.py'):
            path = str(runtime.CHECKOUT / 'vllm_neuron/models/qwen35' / name)
            self.assertIn(path, geometry.verify_sources())

    def test_geometry_int32_and_noncontiguous_prefix_have_no_dtype_fallback(self):
        source = (Q / 'qwen35_writer_pair_validation_v1/geometry.py').read_text()
        self.assertIn('torch.tensor(mapping, dtype=torch.int32)', source)
        self.assertIn('writer(cache[:, :, :spec[\'bucket\'], :], updates, slots)', source)
        self.assertIn("torch.cat((prefix, cache[:, :, spec['bucket']:, :]), dim=2)", source)
        self.assertNotIn('cache.device', source)
        self.assertEqual(len(geometry.MAPPINGS), 625)
        self.assertEqual(len(geometry.LAYOUTS), 8)

    def test_byte_comparison_rejects_signed_zero_and_nan_payload_changes(self):
        for left, right in ((b'\0\0', b'\0\x80'), (b'\xc1\x7f', b'\xc2\x7f')):
            self.assertTrue(geometry.byte_proof(left, left)['identical'])
            self.assertFalse(geometry.byte_proof(left, right)['identical'])


class GeometryMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {}
        for name in geometry.LAYOUTS:
            spec = geometry.layout(name); rows = []
            for mode, variant, profile in itertools.product(geometry.MODES, geometry.VARIANTS, geometry.PROFILES):
                full = math.prod(spec['maximum_shape']) * (2 if spec['dtype'] == 'bfloat16' else 4)
                def proof(size):
                    return dict(bytes_compared=size, actual_sha256='a' * 64, expected_sha256='a' * 64, identical=True)
                proofs = dict(old=proof(full), full_new=proof(full), host_inputs_unchanged=[True] * (2 if mode == 'persistent' else 3))
                if mode == 'persistent': proofs['device_full_state'] = proof(full)
                if spec['bucket'] is not None: proofs['tail'] = proof(4 * (4096 - spec['bucket']) * 256 * 2)
                for i, mapping in enumerate(geometry.MAPPINGS):
                    rows.append(dict(mode=mode, variant=variant, profile=profile, mapping_index=i,
                                     slots=list(mapping), passed=True, proofs=proofs))
            cls.reports[name] = dict(layout=name, geometry=spec, complete=True, slots_dtype='torch.int32',
                candidate_byte_checks_passed=True, source_sha256={'mock.py': 'b' * 64}, original_raw_failures=0,
                graphs=[dict(mode=m, variant=v, calls=1250, actual_slots_dtype='torch.int32')
                        for m, v in itertools.product(geometry.MODES, geometry.VARIANTS)], calls=rows)

    def test_all_eight_complete_layouts_all_slots_and_full_buffers_required(self):
        result = geometry.check_complete_geometry(self.reports)
        self.assertEqual(result['calls'], 40000)
        self.assertFalse(result['full_model_qualified'])

    def test_complete_geometry_metadata_fits_the_existing_reader_envelope(self):
        sources = json.loads((Q / 'qwen35_writer_pair_validation_v1/geometry_sources.json').read_text())
        for name, report in self.reports.items():
            complete = dict(report, source_sha256=sources)
            self.assertLess(len(json.dumps(complete, indent=2).encode()), contract.LIMITS['metadata_bytes'], name)

    def changed(self, name, row_index, change, expected_error):
        report = dict(self.reports[name]); report['calls'] = list(report['calls'])
        row = copy.deepcopy(report['calls'][row_index]); change(row); report['calls'][row_index] = row
        reports = dict(self.reports); reports[name] = report
        with self.assertRaisesRegex(ValueError, expected_error): geometry.check_complete_geometry(reports)

    def test_small_prefix_cannot_masquerade_as_full_new_state(self):
        self.changed('kv128', 0, lambda r: r['proofs']['full_new'].update(bytes_compared=4 * 128 * 256 * 2), 'storage proof')

    def test_tail_and_persistent_state_cannot_be_omitted(self):
        self.changed('kv128', 0, lambda r: r['proofs'].pop('tail'), 'proof required')
        self.changed('conv', 2500, lambda r: r['proofs'].pop('device_full_state'), 'proof required')

    def test_int64_result_is_rejected(self):
        reports = dict(self.reports); reports['conv'] = dict(reports['conv'], slots_dtype='torch.int64')
        with self.assertRaises(ValueError): geometry.check_complete_geometry(reports)

    def test_duplicate_mapping_rejected(self):
        self.changed('conv', 1, lambda r: r.update(mapping_index=0, slots=[-1] * 4), 'Repeated or omitted')

    def test_false_passed_flag_rejected(self):
        self.changed('conv', 1250, lambda r: r['proofs']['full_new'].update(actual_sha256='b' * 64), 'storage proof')

    def test_host_input_mutation_rejected(self):
        self.changed('conv', 1250, lambda r: r['proofs'].update(host_inputs_unchanged=[True, False, True]), 'Passed flag')

    def test_original_raw_failures_retained_candidate_raw_failures_rejected(self):
        reports = dict(self.reports); report = dict(reports['conv']); report['calls'] = list(report['calls'])
        row = copy.deepcopy(report['calls'][0]); row['passed'] = False
        row['proofs']['full_new'].update(actual_sha256='b' * 64, identical=False)
        report['calls'][0] = row; report['original_raw_failures'] = 1; reports['conv'] = report
        self.assertTrue(geometry.check_complete_geometry(reports)['raw_original_failures_retained'])
        self.changed('conv', 1250, lambda r: (r.update(passed=False),
                     r['proofs']['full_new'].update(actual_sha256='b' * 64, identical=False)), 'Required geometry')


class Qualification(unittest.TestCase):
    def fixture(self):
        def ref(name): return dict(path='/mock/' + name + '.json', sha256='a' * 64)
        qref, eref, cref = ref('qualification'), ref('exit'), ref('catalog')
        report_refs = {n: ref(n) for n in geometry.LAYOUTS}
        sources = {'/mock/geometry.py': 'b' * 64}
        qual = dict(schema='qwen35_writer_geometry_independent_qualification_v1', reviewer='/root', passed=True,
                    blockers=[], reports=report_refs, actual_exit=eref, source_sha256=sources, full_model_qualified=False)
        end = dict(schema='qwen35_writer_geometry_root_actual_exit_v1', reviewer='/root',
            actual_tool_exit_observed=True, actual_tool={'exit_code': 0}, physically_drained=True, owners_gone=True,
            reports=report_refs, source_sha256=sources, full_model_qualified=False)
        values = {qref['path']: qual, eref['path']: end, cref['path']: {'runs': [{'terminal': eref}]}}
        values.update({r['path']: {'source_sha256': sources} for r in report_refs.values()})
        spec = dict(geometry_qualification=qref, current_serial_catalog=cref, source_sha256=sources,
            input_extension={r['path']: {'sha256': r['sha256']} for r in [qref, eref, *report_refs.values()]})
        return spec, values, qual, end

    def test_requires_actual_prior_catalog_exit_and_all_bound_sources(self):
        spec, values, _, _ = self.fixture()
        with patch.object(qualification, 'read', side_effect=lambda r: values[r['path']]), \
             patch.object(geometry, 'check_complete_geometry', return_value={'mock_only': True}):
            self.assertFalse(qualification.check_geometry_qualification(spec)['full_model_qualified'])
            values[spec['current_serial_catalog']['path']]['runs'] = []
            with self.assertRaisesRegex(ValueError, 'serial history'): qualification.check_geometry_qualification(spec)

    def test_actual_nonzero_undrained_or_old_integer_schema_rejected(self):
        for mode in ('exit', 'drain', 'old'):
            spec, values, _, end = self.fixture()
            if mode == 'exit': end['actual_tool']['exit_code'] = 1
            elif mode == 'drain': end['physically_drained'] = False
            else: end['schema'] = 'qwen35_integer_reference_root_actual_exit_v1'
            with patch.object(qualification, 'read', side_effect=lambda r: values[r['path']]), \
                 patch.object(geometry, 'check_complete_geometry') as check:
                with self.assertRaises(ValueError): qualification.check_geometry_qualification(spec)
                check.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
