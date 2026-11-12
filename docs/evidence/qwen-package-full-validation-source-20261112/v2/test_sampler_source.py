"""Sampler port source/mock tests only. No torch, packets, models or HTTP run."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

D = Path(__file__).resolve().parent
sys.path[:0] = [str(D.parent), str(D.parent.parent), '/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107']
from qwen35_package_full_validation_v2.test_source import no_native, node, dump, CASE, SourceTests

with patch('builtins.__import__', side_effect=no_native):
    from qwen35_package_full_validation_v2 import sampler as s, capture_worker as w, runtime as r, http as h, entry as e
    from qwen35_package_full_validation_v2 import sampler_leaves
    from qwen35_sampler_native_capture_v1 import CaptureBudget

Q = D.parent
P = Path('/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107')
MANIFEST = 'a' * 64
CONTROL_NAMES = ['partial_explicit_generators', 'all_explicit_generators', 'k_absent', 'k_zero',
                 'k_above64', 'non_float32', 'invalid_p', 'processed_logits', 'processed_logprobs',
                 'excessive_kth_ties', 'nonfinite_max_or_kth', 'cdf_near_boundary', 'cdf_boundary_tie']


def mock_replay(arm='candidate'):
    sizes = [1] * 16 + [4] * 16
    files = [dict(file='capture-%04d.bin' % i, rows=n, bytes=n * 248320 * 4 + 4096,
                  sha256='b' * 64, call_id=i + 1) for i, n in enumerate(sizes)]
    capture = dict(phase='correctness_capture', arm=arm, packets=32,
        captured_calls_by_rows={'1': 16, '2': 0, '3': 0, '4': 16}, captured_rows=80,
        processed_logit_bytes=80 * 248320 * 4, total_packet_bytes=sum(x['bytes'] for x in files),
        timing_eligible=False, files=files)
    proofs = [dict(passed=True, arm=arm, call_id=i + 1, rows=n, vocabulary=248320, fast_rows=n,
        total_variation_max=1e-8, max_probability_error=1e-7, own_token_and_rng_reproduced=True,
        explicit_generator_fallback_tested=False, timing_eligible=False, full_model_qualified=False)
        for i, n in enumerate(sizes)]
    replay = dict(case=CASE, arm=arm, phase='sampler_replay', artifact_manifest_sha256=MANIFEST,
        capture_identity_sha256=w.capture_identity(CASE, arm, MANIFEST)['sha256'], passed=True, packets=32,
        rows=80, all_own_tokens_and_rng_reproduced=True, all_distribution_gates_passed=True, fast_path_rows=80,
        candidate_operation=w.CANDIDATE, reference_operation=w.REFERENCE, timing_eligible=False, proofs=proofs,
        fresh_execution_proven=False, full_model_qualified=False)
    return replay, capture


def mock_controls():
    return dict(case=CASE, phase='sampler_controls', artifact_manifest_sha256=MANIFEST,
        passed=True, candidate_operation=w.CANDIDATE, reference_operation=w.REFERENCE, timing_eligible=False,
        fresh_execution_proven=False, full_model_qualified=False,
        controls=[dict(control=name, passed=True, reference_calls=1, reference_exception=None,
                       whole_call=i < 9, token_and_rng_equal=True) for i, name in enumerate(CONTROL_NAMES)])


def env(arm):
    return dict(NEURON_QWEN35='1', NEURON_ON_DEVICE_SAMPLING_DISABLED='1', NEURON_QWEN35_ALLOW_COMPILE='0',
                NEURON_CPU_SAMPLER_FAST='0' if arm == 'baseline' else '1')


def runner(arm):
    reference_type = type('TopKTopPSampler', (), {'__module__': 'vllm.v1.sample.ops.topk_topp_sampler'})
    candidate_type = type('BoundedTopKTopPSampler', (), {'__module__': 'vllm_neuron.worker.cpu_sampler'})
    reference = reference_type(); operation = reference
    if arm == 'candidate':
        operation = candidate_type(); operation.reference = reference
    return SimpleNamespace(cpu_sampler=SimpleNamespace(topk_topp_sampler=operation))


class SamplerSourceTests(unittest.TestCase):
    def setUp(self):
        self.block = patch('builtins.__import__', side_effect=no_native)
        self.block.start(); self.addCleanup(self.block.stop)

    def test_v1_delivery_and_all_original_files_still_frozen(self):
        path = Q / 'qwen35_package_full_validation_v1/source_delivery.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '792bd1d9d7e720e6db6bf7d7c05f08cf09355926ea001508082c5f638b577c9a')
        for name, ref in json.loads(path.read_text())['files'].items():
            raw = Path(name).read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (ref['bytes'], ref['sha256']))

    def test_research_source_pins_join_actual_r10_and_exact13_controls(self):
        record = r.read(D / 'r10_sampler_review.json')
        self.assertEqual(record['actual_research_controls'], CONTROL_NAMES)
        self.assertEqual(record['actual_research_controls_returncode'], 0)
        for path, digest in record['science_sources_joined_to_actual_r10_spec'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        for path, digest in r.read(D / 'capture_sources.json').items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        for arm, row in record['actual_research_arms'].items():
            self.assertEqual((row['packets'], row['rows'], row['actual_child_returncode']), (32, 80, 0))
        self.assertFalse(record['package_results_inherited'])

    def test_entire_replay_and_controls_source_only_changes_package_import(self):
        original = (Q / 'qwen35_sampler_native_replay_v1.py').read_text()
        new = (D / 'sampler_replay.py').read_text()
        changed = 'from vllm_neuron.worker.cpu_sampler import BoundedTopKTopPSampler, _shortlist'
        self.assertEqual(new.count(changed), 2)
        restored = new.replace(changed, 'from qwen35_sampler_bounded_operation_v1 import BoundedTopKTopPSampler, _shortlist')
        self.assertEqual(dump(ast.parse(restored)), dump(ast.parse(original)))
        for name in ('replay_captures', 'rejection_controls'):
            old = node(Q / 'qwen35_sampler_native_leaves_v2.py', name)
            new = node(D / 'sampler_leaves.py', name)
            for item in ast.walk(new):
                if isinstance(item, ast.ImportFrom) and item.module == 'sampler_replay':
                    item.module, item.level = 'qwen35_sampler_native_replay_v1', 0
            self.assertEqual(dump(old), dump(new))

    def test_original_writer_and_actual_package_math_AST_preserved(self):
        for name in ('nvme_destination', 'atomic_metadata', 'CaptureFileWriter'):
            self.assertEqual(dump(node(D / 'capture_writer.py', name)), dump(node(Q / 'qwen35_sampler_worker_v1.py', name)))
        for name in ('_shortlist', 'BoundedTopKTopPSampler'):
            self.assertEqual(dump(node(P / 'vllm_neuron/worker/cpu_sampler.py', name)),
                             dump(node(Q / 'qwen35_sampler_bounded_operation_v1.py', name)))
        self.assertEqual(s.SAMPLER_GATES['controls'], CONTROL_NAMES)
        self.assertEqual(s.SAMPLER_GATES['total_variation_max'], 1e-6)
        self.assertEqual(s.SAMPLER_GATES['max_probability_error_max'], 1e-6)

    def test_capture_observes_existing_operation_no_double_wrap(self):
        for arm in ('baseline', 'candidate'):
            model_runner = runner(arm); operation = model_runner.cpu_sampler.topk_topp_sampler
            def observe(runner_value, sampler, observed_arm):
                self.assertEqual(observed_arm, arm)
                return SimpleNamespace(install=lambda: setattr(sampler, '_qwen35_pair_observed', True))
            attachment = w.attach(model_runner, arm, observe, env(arm))
            self.assertIs(model_runner.cpu_sampler.topk_topp_sampler, operation)
            self.assertTrue(attachment['observer_only_attachment'])
            with self.assertRaises(ValueError): w.attach(model_runner, arm, observe, env(arm))
        tree = node(D / 'capture_worker.py', 'attach')
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == 'topk_topp_sampler'
                             and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)))

    def test_capture_rejects_wrong_operation_nested_reference_or_activation(self):
        observer = lambda *args: SimpleNamespace(install=lambda: None)
        for arm in ('baseline', 'candidate'):
            with self.assertRaises(ValueError): w.attach(runner(arm), arm, observer, env('candidate' if arm == 'baseline' else 'baseline'))
            other = runner('candidate' if arm == 'baseline' else 'baseline')
            with self.assertRaises(ValueError): w.attach(other, arm, observer, env(arm))
        value = runner('candidate'); value.cpu_sampler.topk_topp_sampler.reference = runner('candidate').cpu_sampler.topk_topp_sampler
        with self.assertRaises(ValueError): w.attach(value, 'candidate', observer, env('candidate'))
        for key in ('TRN1_CPU_SAMPLER_FAST', 'QWEN35_SAMPLER_CAPTURE_ROOT'):
            bad = env('candidate'); bad[key] = '1'
            with self.assertRaises(ValueError): w.attach(runner('candidate'), 'candidate', observer, bad)

    def test_spawn_worker_names_are_stable_and_measured_entry_has_no_worker(self):
        module = ModuleType('vllm_neuron.worker.neuron_worker')
        module.NeuronWorker = type('NeuronWorker', (), {})
        with patch.dict(sys.modules, {module.__name__: module}):
            for name, arm in w.EXPORTS.items():
                cls = getattr(w, name)
                self.assertIs(getattr(w, name), cls)
                self.assertEqual(cls.__module__ + '.' + cls.__qualname__, w.worker_name(arm))
                self.assertEqual(s.capture_server_arguments(CASE, arm)[-2:], ['--worker-cls', w.worker_name(arm)])
        self.assertNotIn('--worker-cls', r.server_arguments(CASE))

    def test_complete_mock_replay_and_controls_never_claim_execution(self):
        for arm in ('baseline', 'candidate'):
            replay, capture = mock_replay(arm)
            s.check_replay_result(replay, capture, CASE, arm, MANIFEST)
            self.assertFalse(replay['fresh_execution_proven']); self.assertFalse(replay['full_model_qualified'])
        controls = mock_controls(); s.check_controls_result(controls, CASE, MANIFEST)
        self.assertFalse(controls['fresh_execution_proven'])

    def test_replay_rejects_partial_quota_rows_distribution_rng_or_old_operation(self):
        mutations = [lambda x: x['proofs'].pop(), lambda x: x.update(fast_path_rows=0),
            lambda x: x.update(all_own_tokens_and_rng_reproduced=False), lambda x: x.update(all_distribution_gates_passed=False),
            lambda x: x.update(candidate_operation='qwen35_sampler_bounded_operation_v1.BoundedTopKTopPSampler'),
            lambda x: x.update(artifact_manifest_sha256='c' * 64), lambda x: x.update(arm='baseline'),
            lambda x: x['proofs'][0].update(vocabulary=64), lambda x: x['proofs'][0].update(rows=2),
            lambda x: x['proofs'][0].update(total_variation_max=1.1e-6),
            lambda x: x['proofs'][0].update(max_probability_error=1.1e-6),
            lambda x: x['proofs'][0].update(total_variation_max=float('nan')),
            lambda x: x['proofs'][0].update(own_token_and_rng_reproduced=False)]
        for mutate in mutations:
            replay, capture = mock_replay(); mutate(replay)
            with self.assertRaises(ValueError): s.check_replay_result(replay, capture, CASE, 'candidate', MANIFEST)
        for mutate in (lambda x: x['captured_calls_by_rows'].update({'1': 15}),
                       lambda x: x['captured_calls_by_rows'].update({'4': 0}),
                       lambda x: x.update(total_packet_bytes=257 * 1024**2)):
            replay, capture = mock_replay(); mutate(capture)
            with self.assertRaises(ValueError): s.check_replay_result(replay, capture, CASE, 'candidate', MANIFEST)

    def test_controls_reject_missing_renamed_fallback_rng_or_multiple_reference_calls(self):
        for mutate in (lambda x: x['controls'].pop(), lambda x: x['controls'].reverse(),
            lambda x: x['controls'][0].update(control='partial_generators_reduced'),
            lambda x: x['controls'][0].update(token_and_rng_equal=False),
            lambda x: x['controls'][0].update(reference_calls=2),
            lambda x: x['controls'][-1].update(whole_call=True), lambda x: x.update(passed=False)):
            controls = mock_controls(); mutate(controls)
            with self.assertRaises(ValueError): s.check_controls_result(controls, CASE, MANIFEST)

    def test_guard_precedes_native_replay_controls_or_diagnostics(self):
        bad = lambda: dict(admitted=False, exclusive=False)
        calls = [lambda: sampler_leaves.replay_captures('/unused', bad),
                 lambda: sampler_leaves.rejection_controls(bad),
                 lambda: s.run_capture_diagnostic(CASE, '/unused', bad),
                 lambda: e.run_sampler_stage('sampler_replay', CASE, '/unused', bad, 'candidate'),
                 lambda: e.run_sampler_stage('sampler_controls', CASE, '/unused', bad)]
        for call in calls:
            with self.assertRaises(ValueError): call()

    def test_missing_sampler_gate_blocks_every_quality_and_http_client(self):
        with patch.object(h, 'check_environment'), patch.object(s, 'check_saved_sampler_gates', side_effect=ValueError('missing actual sampler gates')) as gate:
            for phase in ('smoke', 'quality96', 'initial120', 'workloads'):
                folder = 'quality' if phase in ('smoke', 'quality96') else 'http'
                with self.assertRaisesRegex(ValueError, 'missing actual sampler gates'):
                    h.run_http_stage(phase, CASE, '/unused/baseline/' + folder + '/report.json',
                                     lambda: dict(admitted=True, exclusive=True))
            self.assertEqual(gate.call_count, 4)
            with self.assertRaisesRegex(ValueError, 'missing actual sampler gates'):
                h.check_saved_quality(CASE, '/unused/baseline', '/unused/candidate')
        with patch.object(h, 'check_environment'), patch.object(s, 'check_saved_sampler_gates', return_value={}), \
                patch.object(h, 'check_saved_quality', side_effect=ValueError('quality pair missing')) as quality:
            for phase in ('initial120', 'workloads'):
                with self.assertRaisesRegex(ValueError, 'quality pair missing'):
                    h.run_http_stage(phase, CASE, '/unused/baseline/http/report.json',
                                     lambda: dict(admitted=True, exclusive=True))
            self.assertEqual(quality.call_count, 2)

    def test_capture_diagnostic_keeps_original_five_requests160_tokens_without_seed(self):
        calls = []
        inputs = ModuleType('benchmark_workloads'); inputs.inputs = lambda case: {128: [42] * 128}
        bench = ModuleType('bench_http')
        def run_round(port, model, prompts, tokens, stream, **sampling):
            calls.append((port, model, len(prompts), tokens, stream, sampling))
            return {'requests': [{'usage': {'prompt_tokens': 128, 'completion_tokens': 32}} for _ in prompts]}
        bench.run_round = run_round
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {inputs.__name__: inputs, bench.__name__: bench}), patch.object(s, 'check_environment'):
            report = s.run_capture_diagnostic(CASE, Path(tmp) / 'diagnostic.json', lambda: dict(admitted=True, exclusive=True))
        self.assertEqual([x[2] for x in calls], [1, 4])
        self.assertTrue(all(x[0] == 8010 and x[3:5] == (32, False)
                            and x[5] == dict(temperature=.7, top_p=.9, top_k=40) for x in calls))
        self.assertEqual((report['requests'], report['output_tokens']), (5, 160))
        self.assertFalse(report['timing_eligible'])

    def test_original_capture_budget_is_bounded_and_no_missing_quota_retry(self):
        budget = CaptureBudget()
        for n in range(1, 5):
            self.assertEqual([budget.select(n, 248320) for _ in range(20)], [True] * 16 + [False] * 4)
        with self.assertRaises(ValueError): budget.select(1, 64)
        source = (D / 'sampler.py').read_text()
        self.assertNotIn('while ', ast.get_source_segment(source, node(D / 'sampler.py', 'run_capture_diagnostic')))

    def test_measured_host_rejects_package_capture_environment(self):
        hosts, directories = SourceTests().host_pair()
        hosts['candidate']['startup_config_proof']['actual_api_initial_environment']['QWEN35_PACKAGE_CAPTURE_ROOT'] = '/unused'
        with self.assertRaises(ValueError): h.check_matched_hosts(hosts['baseline'], hosts['candidate'], CASE, directories)


if __name__ == '__main__':
    unittest.main(verbosity=2)
