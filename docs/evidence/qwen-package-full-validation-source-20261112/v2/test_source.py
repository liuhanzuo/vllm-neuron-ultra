"""Stdlib-only source/metadata/mock tests; no native or payload execution."""
import ast
import builtins
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

D = Path(__file__).resolve().parent
ROOT = D.parent.parent
CHECKOUT = Path('/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107')
sys.path[:0] = [str(D.parent), str(ROOT), str(CHECKOUT)]
NATIVE = ('torch', 'neuron', 'libneuron', 'transformers', 'vllm.', 'ctypes')
ORIGINAL_IMPORT = builtins.__import__


def no_native(name, *args, **kwargs):
    if name.startswith(NATIVE):
        raise AssertionError('Native import forbidden in source checks: ' + name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


with patch('builtins.__import__', side_effect=no_native):
    from qwen35_package_full_validation_v2 import runtime as r, entry as e, http as h, checks as c
    from qwen35_package_full_validation_v2.state_observation import graph_routes
    from qwen35_gdn_serving_checks_v2 import compare as compare_workloads

CASE = r.read(D / 'case.example.json')


def node(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if getattr(n, 'name', None) == name)


def dump(value):
    return ast.dump(value, include_attributes=False)


def original_report(phase):
    value = copy.deepcopy(r.read(r.BASE / e.MODEL_FILES[phase]))
    if phase == 'numeric':
        value.update(model=CASE['model_path'], artifacts=CASE['artifact'])
    elif phase != 'state_copy':
        value['case'] = CASE
    return value


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.block = patch('builtins.__import__', side_effect=no_native)
        self.block.start(); self.addCleanup(self.block.stop)

    def test_original_sources_and_package_unchanged(self):
        pins = r.read(D / 'input_sources.json')
        for name, expected in pins.items():
            raw = Path(name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (expected['bytes'], expected['sha256']), name)
        plan = r.read(r.BASE / 'plan.json')['source_sha256']
        for row in r.read(D / 'port_edits.json'):
            self.assertEqual(hashlib.sha256(Path(row['source']).read_bytes()).hexdigest(), plan[row['source']])

    def test_only_enumerated_entry_edits_full_source_preserved(self):
        for row in r.read(D / 'port_edits.json'):
            source = Path(row['source']).read_text()
            for edit in row['replacements']:
                self.assertEqual(source.count(edit['before']), 1)
                source = source.replace(edit['before'], edit['after'], 1)
            self.assertEqual(dump(ast.parse(source)), dump(ast.parse(Path(row['target']).read_text())))
        for filename, name, original in [('numeric.py', 'metrics', 'validate_real_logits.py'),
            ('long.py', 'passed', 'validate_qwen35_long_decode.py'),
            ('http.py', 'check_initial_pair', 'research/qwen35_sampler_pair_driver_v3.py'),
            ('state_observation.py', 'graph_routes', 'qwen35_split_state_checks_v4.py'),
            ('state_observation.py', 'raw_zero_states', 'qwen35_split_state_checks_v4.py'),
            ('state_observation.py', 'verify_cold_reproduction', 'qwen35_state_inplace_reset_v1.py')]:
            self.assertEqual(dump(node(D / filename, name)), dump(node(ROOT / original, name)))

    def test_source_reduction_dtype_slot_and_threshold_mutations_detected(self):
        for filename, old, new in [('long.py', 'STEPS = 1541', 'STEPS = 128'),
            ('long.py', 'slots[1:]', 'slots[:1]'), ('cold.py', 'range(4)', 'range(3)'),
            ('cold.py', 'range(3)', 'range(2)'), ('state_copy.py', 'range(2)', 'range(1)'),
            ('state_copy.py', 'torch.int32', 'torch.int16'), ('numeric.py', '>=.995', '>=.95')]:
            raw = (D / filename).read_text(); self.assertIn(old, raw)
            self.assertNotEqual(dump(ast.parse(raw)), dump(ast.parse(raw.replace(old, new, 1))))

    def test_original_validator_port_changes_only_case_binding(self):
        old = (ROOT / 'research/qwen35_sampler_full_checks_v1.py').read_text()
        old = old.replace('def check_model_metadata(report, phase, validators):',
                          'def check_model_metadata(report, phase, validators, case):').replace('    case = expected_case()\n', '')
        old = old.replace('def check_quality_pair(reference, candidate, historical, suite, suite_sha256, validators):',
                          'def check_quality_pair(reference, candidate, historical, suite, suite_sha256, validators, case):')
        old = old.replace('    for report in (historical, reference, candidate):',
                          '    for report, report_case in ((historical, expected_case()), (reference, case), (candidate, case)):')
        old = old.replace('report["case"] == expected_case()', 'report["case"] == report_case')
        self.assertEqual(dump(ast.parse(old)), dump(ast.parse((D / 'checks.py').read_text())))

    def test_complete_configuration_and_package_serve_entry(self):
        self.assertEqual(r.validate_case(CASE), CASE)
        self.assertEqual(r.configuration(), r.read(D / 'neuron_config.json'))
        argv = r.server_arguments(CASE)
        self.assertEqual(argv[:3], [r.PYTHON, '-m', 'vllm_neuron.models.qwen35.serve'])
        self.assertEqual(json.loads(argv[-1]), {'override_neuron_config': r.configuration()})
        for key, value in [('batch', 1), ('tp', 1), ('max_model_len', 2048),
                           ('model_path', r.read(r.BASE / 'case.json')['model_path']),
                           ('artifact', r.read(r.BASE / 'case.json')['artifact'])]:
            case = copy.deepcopy(CASE); case[key] = value
            with self.assertRaises(ValueError): r.validate_case(case)
        case = copy.deepcopy(CASE); case['neuron_configuration']['context_encoding_buckets'].pop()
        with self.assertRaises(ValueError): r.validate_case(case)
        case = copy.deepcopy(CASE); case['neuron_configuration']['tp_degree'] = 2.0
        with self.assertRaises(ValueError): r.validate_case(case)

    def test_actual_loader_default_and_override_config_match_build(self):
        source = CHECKOUT / 'vllm_neuron/worker/neuronx_distributed_model_loader.py'
        nodes = [node(source, n) for n in ('_get_default_neuron_config',
                  '_validate_override_neuron_config', '_get_neuron_config_after_override')]
        future = ast.parse('from __future__ import annotations').body
        ns = dict(ceil=math.ceil, logger=SimpleNamespace(debug=lambda *a: None),
            TORCH_DTYPE_TO_NEURON_AMP={'bf16': 'bfloat16'}, OnDeviceSamplingConfig=lambda **kw: kw)
        exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(source), 'exec'), ns)
        model = SimpleNamespace(dtype='bf16', max_prompt_length=None)
        scheduler = SimpleNamespace(chunked_prefill_enabled=False, max_num_seqs=4, max_model_len=4096)
        cache = SimpleNamespace(block_size=4096, num_gpu_blocks_override=None, enable_prefix_caching=False)
        defaults = ns['_get_default_neuron_config'](model, cache, SimpleNamespace(tensor_parallel_size=2),
                                                    scheduler, None, None)
        self.assertEqual((defaults['pa_num_blocks'], defaults['pa_block_size']), (4, 4096))
        overrides = ns['_validate_override_neuron_config'](r.configuration(), model)
        serving = ns['_get_neuron_config_after_override'](defaults, overrides)
        extra_defaults = dict(quantized_checkpoints_path=None, quantization_type='per_tensor_symmetric',
                              quantization_dtype='int8')
        self.assertEqual(serving, {**r.configuration(), **extra_defaults})
        # These three normalized fields equal the pinned SDK constructor defaults.
        sdk = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/config.py')
        constructor = next(n for n in node(sdk, 'NeuronConfig').body if getattr(n, 'name', None) == '__init__')
        found = {}
        for n in ast.walk(constructor):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'pop' and len(n.args) == 2:
                if isinstance(n.args[0], ast.Constant) and n.args[0].value in extra_defaults:
                    found[n.args[0].value] = ast.literal_eval(n.args[1])
        self.assertEqual(found, extra_defaults)

    def test_guard_rejection_precedes_native_import_payload_or_writes(self):
        bad = lambda: dict(admitted=False, exclusive=False)
        for call in (lambda: e.prepare_view(CASE, bad), lambda: e.build_fresh(CASE, '/unused', bad),
                     lambda: e.run_model_stage('cold', '/unused', '/unused', bad),
                     lambda: e.compare_vectors('/unused', bad),
                     lambda: h.run_http_stage('quality96', CASE, '/unused', bad)):
            with self.assertRaises(ValueError): call()

    def test_load_cannot_compile_and_calls_real_package_entry(self):
        with patch.dict(os.environ, {'NEURON_QWEN35_ALLOW_COMPILE': '1'}, clear=True):
            with self.assertRaises(ValueError): r.load_admitted('model', 'artifact')
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            fake = ModuleType('vllm_neuron.models.qwen35.build')
            fake.build = lambda *args: calls.append(args) or 'mock loaded model'
            with patch.dict(sys.modules, {fake.__name__: fake}), patch.dict(os.environ,
                    {'NEURON_QWEN35_ALLOW_COMPILE': '0', 'NEURON_COMPILED_ARTIFACTS': tmp}, clear=True):
                self.assertEqual(r.load_admitted('new-view', tmp), 'mock loaded model')
                self.assertEqual(calls, [('new-view', r.configuration())])

    def test_build_rejects_existing_artifact_cache_or_workdir_before_native_import(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'check_environment'):
            case = copy.deepcopy(CASE)
            for key in ('artifact', 'compiler_cache', 'compiler_workdir'):
                case[key] = str(Path(tmp) / key)
            for key in ('artifact', 'compiler_cache', 'compiler_workdir'):
                path = Path(case[key]); path.mkdir()
                with self.assertRaises(ValueError):
                    e.build_fresh(case, Path(tmp) / 'build.json', lambda: dict(admitted=True, exclusive=True))
                path.rmdir()

    def test_actual_twelve_route_shape_rejects_missing_bucket_and_wrong_batch(self):
        mapping = {}
        for tag, b in [('context_encoding_model', 1), ('token_generation_model', 4)]:
            for i, bucket in enumerate(r.BUCKETS):
                mapping[str([[b, bucket if b == 1 else 1], [b, bucket]])] = (tag, i)
        model = SimpleNamespace(traced_model=SimpleNamespace(nxd_model=SimpleNamespace(input_shape_map=mapping)))
        self.assertEqual(len(graph_routes(model)['graphs']), 12)
        key = next(iter(mapping)); saved = mapping.pop(key)
        with self.assertRaises(AssertionError): graph_routes(model)
        mapping[str([[2, 128], [2, 128]])] = saved
        with self.assertRaises(AssertionError): graph_routes(model)

    def test_full_metadata_positive_mocks_do_not_claim_execution(self):
        validators = c.original_validators()
        for phase in e.MODEL_FILES:
            result = c.check_model_metadata(original_report(phase), phase, validators, CASE)
            self.assertTrue(result['metadata_gate_passed'])
            self.assertFalse(result['fresh_execution_proven'])
            self.assertFalse(result['full_model_qualified'])

    def test_numeric_cold_long_slot_and_state_copy_negative_metadata(self):
        validators = c.original_validators()
        mutations = [
            ('numeric', lambda x: x['rows'].pop()),
            ('numeric', lambda x: x['rows'][0].update(cosine=.994)),
            ('numeric', lambda x: x.update(artifacts='old-research-artifact')),
            ('cold', lambda x: x['experiments'].pop()),
            ('cold', lambda x: x['experiments'][0]['rows'].pop()),
            ('cold', lambda x: x['state_reset_receipts'].pop()),
            ('cold', lambda x: x['prior_partial_reproduction'].update(original_partial_vectors_reproduced_exactly=71)),
            ('cold', lambda x: x['graph_routes']['graphs'].pop()),
            ('long', lambda x: x.update(teacher_forced_decode_steps=128)),
            ('long', lambda x: x['rows'].pop()),
            ('long', lambda x: x['rows'][-1].update(phase='partial_batch')),
            ('long', lambda x: x['rows'][-1].update(slot=1)),
            ('state_copy', lambda x: x.update(fill_roundtrip_comparisons=191)),
            ('state_copy', lambda x: x.update(graph_models_loaded=1)),
        ]
        for phase, mutate in mutations:
            report = original_report(phase); mutate(report)
            with self.subTest(phase=phase), self.assertRaises((ValueError, AssertionError)):
                c.check_model_metadata(report, phase, validators, CASE)

    def test_quality_regrade_all96_honest_case_and_negative_text_prompt_rows(self):
        historical = r.read(r.BASE / 'quality96.json'); suite = r.read(ROOT / 'quality-suite.json')
        digest = hashlib.sha256((ROOT / 'quality-suite.json').read_bytes()).hexdigest()
        left = copy.deepcopy(historical); left['case'] = CASE
        check = lambda right: c.check_quality_pair(left, right, historical, suite, digest, c.original_validators(), CASE)
        self.assertEqual(check(copy.deepcopy(left))['all_texts_equal'], 96)
        for mutate in (lambda x: x['rows'].pop(), lambda x: x.update(case=historical['case']),
                       lambda x: x['rows'][0].update(text=x['rows'][0]['text'] + ' changed'),
                       lambda x: x['rows'][0]['prompt_ids'].pop(), lambda x: x.update(correct=70)):
            right = copy.deepcopy(left); mutate(right)
            with self.assertRaises((ValueError, AssertionError)): check(right)

    def host_pair(self):
        hosts = {}; dirs = {}
        for arm in ('baseline', 'candidate'):
            directory = Path('/mnt/nvme/neuron-work/qwen35-package-full-v2-run') / arm / 'http'
            dirs[arm] = str(directory)
            env = dict(NEURON_QWEN35='1', NEURON_ON_DEVICE_SAMPLING_DISABLED='1',
                XLA_DISABLE_FUNCTIONALIZATION='1', NEURON_QWEN35_ALLOW_COMPILE='0',
                NEURON_CPU_SAMPLER_FAST='0' if arm == 'baseline' else '1',
                NEURON_COMPILED_ARTIFACTS=CASE['artifact'], NEURON_COMPILE_CACHE_URL=CASE['compiler_cache'],
                BASE_COMPILE_WORK_DIR=CASE['compiler_workdir'], TMPDIR=str(directory / 'tmp'),
                VLLM_RPC_BASE_PATH=str(directory / 'rpc'), VLLM_LOGGING_LEVEL='DEBUG', OMP_NUM_THREADS='1')
            argv = r.server_arguments(CASE)
            hosts[arm] = dict(observation_kind='live_ready', ready=True, server_active=True, exit_observed=False,
                startup_config_proof=dict(actual_api_initial_environment=env, actual_api_argv=argv),
                api_process_owner=dict(pid=101 if arm == 'baseline' else 102, argv=argv))
        return hosts, dirs

    def test_host_pair_rejects_debug_threads_hooks_old_argv_and_temp_alias(self):
        hosts, dirs = self.host_pair()
        self.assertTrue(h.check_matched_hosts(hosts['baseline'], hosts['candidate'], CASE, dirs)['matched'])
        for key, value in [('VLLM_LOGGING_LEVEL', 'ERROR'), ('OMP_NUM_THREADS', '2'),
            ('QWEN35_SAMPLER_CAPTURE_ROOT', '/tmp/capture'), ('NEURON_CPU_SAMPLER_FAST', '0'),
            ('NEURON_QWEN35_ALLOW_COMPILE', '1'), ('TMPDIR', '/tmp/unbound')]:
            pair = copy.deepcopy(hosts); pair['candidate']['startup_config_proof']['actual_api_initial_environment'][key] = value
            with self.assertRaises(ValueError): h.check_matched_hosts(pair['baseline'], pair['candidate'], CASE, dirs)
        pair = copy.deepcopy(hosts); pair['candidate']['startup_config_proof']['actual_api_argv'][2] = 'server_with_qwen35_split_v4'
        with self.assertRaises(ValueError): h.check_matched_hosts(pair['baseline'], pair['candidate'], CASE, dirs)

    def test_http_all124_requests30848_tokens57_greedy_and_negative_budgets(self):
        left = r.read(r.BASE / 'workloads.json'); left['case'] = CASE
        result = compare_workloads(left, copy.deepcopy(left), CASE, CASE, left['inputs'])
        self.assertEqual((result['request_pairs'], result['identical_greedy_texts']), (124, 57))
        for mutate in (lambda x: x['workloads']['mixed_lengths']['rounds'].pop(),
            lambda x: x['workloads']['short_decode']['rounds'][0]['single']['requests'][0]['usage'].update(completion_tokens=511),
            lambda x: x['inputs']['1536'].pop()):
            right = copy.deepcopy(left); mutate(right)
            with self.assertRaises(AssertionError): compare_workloads(left, right, CASE, CASE, left['inputs'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
