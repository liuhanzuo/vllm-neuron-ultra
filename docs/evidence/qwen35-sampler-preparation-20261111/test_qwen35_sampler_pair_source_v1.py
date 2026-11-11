"""Stdlib/fake-object tests only. No native sampler or model qualification."""
import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
real_import = builtins.__import__


def no_native(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "transformers", "vllm", "vllm_neuron", "neuronx_distributed_inference", "numpy"}:
        raise AssertionError("native import during source-only tests: " + name)
    return real_import(name, *args, **kwargs)


builtins.__import__ = no_native
import qwen35_sampler_pair_protocol_v1 as protocol
import qwen35_sampler_instance_v1 as integration
import qwen35_sampler_native_capture_v1 as capture
import qwen35_sampler_native_replay_v1 as replay
import qwen35_sampler_full_checks_v1 as full_checks
import qwen35_sampler_native_leaves_v1 as leaves
import qwen35_sampler_pair_driver_v1 as driver
import qwen35_sampler_worker_v1 as workers
import qwen35_gdn_serving_checks_v2 as http_checks
builtins.__import__ = real_import


class FakeLogits:
    def __init__(self, rows, processed=False):
        self.shape = (rows, protocol.VOCAB)
        self.processed = processed


class FakeOperation:
    def __init__(self):
        self.calls = 0
        self.logprobs_mode = "raw_logprobs"

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, logits, generators, k, p):
        if not logits.processed:
            raise AssertionError("capture/input boundary was too early")
        self.calls += 1
        return ("output", self.calls)


class FakeSampler:
    def __init__(self):
        self.topk_topp_sampler = FakeOperation()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, logits, metadata):
        if metadata.all_greedy:
            return "greedy"
        processed = FakeLogits(logits.shape[0], True)
        return self.topk_topp_sampler(processed, metadata.generators, metadata.top_k, metadata.top_p)


class FakeBounded:
    def __init__(self, reference):
        self.reference = reference


class FakeSnapshots:
    def __init__(self):
        self.observed = []

    def before(self, arm, call_id, logits, generators, k, p, metadata, context, operation, reference_forward):
        if not logits.processed or generators is not metadata.generators or k is not metadata.top_k or p is not metadata.top_p:
            raise AssertionError("not actual fully processed invocation")
        if "observed_operation" in reference_forward:
            raise AssertionError("reference method identity contaminated by observer")
        self.observed.append((arm, call_id, logits.shape, context, metadata))
        return call_id

    def after(self, snapshot, generators, result):
        return ("synthetic-source-test-packet:" + str(snapshot)).encode()


class SourceTests(unittest.TestCase):
    def setUp(self):
        builtins.__import__ = no_native
        self.case = protocol.expected_case()

    def tearDown(self):
        builtins.__import__ = real_import

    def test_no_native_imports_and_source_syntax(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "vllm", "numpy")))
        for path in HERE.glob("qwen35_sampler_*_v1.py"):
            ast.parse(path.read_text(), filename=str(path))
        for name in (protocol.__name__, integration.__name__, capture.__name__, replay.__name__,
                     full_checks.__name__, leaves.__name__, driver.__name__, workers.__name__):
            importlib.reload(sys.modules[name])

    def test_stable_worker_fqn_pickles_across_fresh_stdlib_interpreters(self):
        code = '''import base64, pickle, sys, types
sys.path.insert(0, sys.argv[1])
for name in ("vllm_neuron", "vllm_neuron.worker", "vllm_neuron.worker.neuron_worker"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
sys.modules["vllm_neuron.worker.neuron_worker"].NeuronWorker = type("StubNeuronWorker", (), {})
import qwen35_sampler_worker_v1 as workers
if sys.argv[2] == "encode":
    classes = [getattr(workers, name) for name in workers.EXPORTS]
    assert all(c.__qualname__ == c.__name__ and c.__module__ == workers.__name__ for c in classes)
    print(base64.b64encode(pickle.dumps(classes)).decode())
else:
    classes = pickle.loads(base64.b64decode(sys.argv[3]))
    assert [c.__name__ for c in classes] == list(workers.EXPORTS)
    assert all(getattr(workers, c.__name__) is c for c in classes)
    print("fresh-import-worker-resolution-passed")
assert not any(name in sys.modules for name in ("torch", "transformers", "numpy"))
'''
        encoded = subprocess.run([sys.executable, "-B", "-c", code, str(HERE), "encode"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        decoded = subprocess.run([sys.executable, "-B", "-c", code, str(HERE), "decode", encoded], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        self.assertEqual(decoded, "fresh-import-worker-resolution-passed")

    def test_native_leaves_fail_before_native_import_without_root_admission(self):
        self.assertIsInstance(leaves.state_copy_body(), ast.Module)
        for fn, args in ((leaves.run_model_stage, ("numeric", "/tmp/not-a-run")),
                         (leaves.compare_vectors, (protocol.BASE, "/tmp/not-a-run")),
                         (leaves.replay_captures, ("/tmp/not-a-run",)), (leaves.rejection_controls, ())):
            with self.assertRaises(ValueError): fn(*args, lambda: {"admitted": False, "exclusive": False})
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "vllm")))

    def test_cli_preserves_frozen_entry_and_requires_real_attachment(self):
        args = driver.server_arguments(self.case, "candidate", "measured", {"cpu": True})
        self.assertEqual(args[0], str(protocol.ROOT / "server_with_qwen35_split_v4.py"))
        self.assertEqual(args[-2:], ["--worker-cls", "qwen35_sampler_worker_v1.CandidateMeasuredWorker"])
        identity = protocol.host_identity("candidate", "measured", {"source": "a" * 64})
        receipt = {**identity, "pid": 321,
                   "worker_class": workers.worker_name("candidate", "measured"),
                   "host_identity_sha256": identity["sha256"], "operator_replaced": True}
        live = {"observation_kind": "live_ready", "exit_observed": False, "server_active": True, "ready": True,
                "server_owner": {"unit": "synthetic", "pid": 123, "start_ticks": 456, "invocation_id": "d" * 32},
                "worker_owner": {"pid": 321, "start_ticks": 654}, "proc_owner_matches": True,
                "invocation_matches": True, "worker_in_server_cgroup": True}
        driver.attached_worker(receipt, live, identity)
        bad_live = {**live, "exit_observed": True, "returncode": 0}
        with self.assertRaises(ValueError): driver.attached_worker(receipt, bad_live, identity)
        receipt["worker_class"] = "vllm_neuron.worker.neuron_worker.NeuronWorker"
        with self.assertRaises(ValueError): driver.attached_worker(receipt, live, identity)

    def test_root_terminal_owner_cleanup_and_pinned_observer_required(self):
        server_owner = {"unit": "synthetic-server", "pid": 12, "start_ticks": 34, "invocation_id": "a" * 32}
        live = {"server_owner": server_owner, "observed_unix_ns": 100}
        terminal = {"observation_kind": "terminal_cleanup", "server_active": False, "remaining_processes": [],
                    "cleanup_complete": True, "server_owner": server_owner, "exit_observed": True,
                    "stop_requested": True, "stop_outcome_accepted": True, "observed_unix_ns": 101,
                    "systemd_exit_status": {"ExecMainCode": 2, "ExecMainStatus": 15, "Result": "success"}}
        driver.cleaned_server(terminal, live)
        for key, value in (("remaining_processes", [12]), ("exit_observed", False),
                           ("server_owner", {**server_owner, "start_ticks": 35}), ("cleanup_complete", False)):
            with self.assertRaises(ValueError): driver.cleaned_server({**terminal, key: value}, live)
        observation = {"run_id": "synthetic", "root_owner": server_owner, "observer_role": "root_owner_auditor",
                       "observer_source": "missing", "observer_source_sha256": None, "observed_unix_ns": 100}
        with self.assertRaises(ValueError): driver.root_observation(observation, "synthetic", server_owner, {})

    def test_original_native_clients_never_use_orchestrator_cached_run_case(self):
        tree = ast.parse((HERE / "qwen35_sampler_pair_driver_v1.py").read_text())
        native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NativeClients")
        calls = [n for n in ast.walk(native) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr == "run_case" for n in calls))
        self.assertTrue(any(isinstance(n.func, ast.Name) and n.func.id == "benchmark" for n in calls))
        self.assertEqual(sum(isinstance(n.func, ast.Name) and n.func.id == "evaluate" for n in calls), 2)

    def test_frozen_case_and_separate_host_identity(self):
        record = protocol.read_small(protocol.BASE / "qualified-baseline-record-v1.json")
        before = deepcopy(record)
        for arm in protocol.ARMS:
            value = protocol.host_identity(arm, "measured", {"source": "a" * 64})
            self.assertFalse(value["observation_enabled"])
            self.assertEqual(protocol.expected_case(record), self.case)
        self.assertEqual(record, before)
        bad = deepcopy(record)
        bad["case"]["max_model_len"] = 2048
        with self.assertRaises(ValueError): protocol.expected_case(bad)

    def test_full_science_not_diagnostic_substitution(self):
        contract = protocol.finite_protocol()
        self.assertFalse(contract["runtime_admission"])
        self.assertFalse(contract["compile_permitted"])
        self.assertEqual(contract["full_gates_each_arm"]["full_vector_reproduction"]["frames"], 203)
        self.assertEqual(contract["full_gates_each_arm"]["cold"]["prefixes"][-1], 2052)
        self.assertEqual(contract["full_gates_each_arm"]["long"]["teacher_forced_steps"], 1541)
        self.assertEqual(contract["full_gates_each_arm"]["quality"]["questions"], 96)
        self.assertEqual(contract["full_gates_each_arm"]["http"]["recorded_requests"], 124)
        self.assertEqual(contract["capture"]["max_processed_logit_bytes_per_arm"], 158924800)
        self.assertFalse(contract["sampler_gates"]["historical_missing_penalties_or_rng_may_be_inferred"])

    def test_measured_baseline_untouched_candidate_per_instance(self):
        a, b = SimpleNamespace(cpu_sampler=FakeSampler()), SimpleNamespace(cpu_sampler=FakeSampler())
        original = a.cpu_sampler.topk_topp_sampler
        baseline_dict = dict(vars(a.cpu_sampler))
        integration.attach(a, "baseline", "measured", environ={})
        self.assertEqual(vars(a.cpu_sampler), baseline_dict)
        integration.attach(b, "candidate", "measured", operation_factory=FakeBounded, environ={})
        self.assertIs(a.cpu_sampler.topk_topp_sampler, original)
        self.assertIsInstance(b.cpu_sampler.topk_topp_sampler, FakeBounded)
        self.assertNotIn("forward", vars(b.cpu_sampler))
        with self.assertRaises(ValueError):
            integration.attach(b, "candidate", "measured", operation_factory=FakeBounded, environ={})

    def test_worker_factory_preserves_original_configuration(self):
        seen = []
        class BaseWorker:
            def get_neuronx_distributed_model_runner(self, config, device):
                seen.append((config, device))
                return SimpleNamespace(cpu_sampler=FakeSampler())
        original = BaseWorker.get_neuronx_distributed_model_runner
        cls = integration.worker_class(BaseWorker, "candidate", "measured", operation_factory=FakeBounded, environ={})
        worker = cls()
        config, device = object(), object()
        runner = worker.get_neuronx_distributed_model_runner(config, device)
        self.assertEqual(seen, [(config, device)])
        self.assertIs(BaseWorker.get_neuronx_distributed_model_runner, original)
        self.assertIsInstance(runner.cpu_sampler.topk_topp_sampler, FakeBounded)

    def test_reject_old_activation_and_capture_in_timing(self):
        runner = SimpleNamespace(cpu_sampler=FakeSampler())
        for key in ("TRN1_CPU_SAMPLER_FAST", "NEURON_CPU_SAMPLER_FAST"):
            with self.assertRaises(ValueError):
                integration.attach(runner, "baseline", "measured", environ={key: "0"})
        with self.assertRaises(ValueError):
            integration.attach(runner, "baseline", "measured", observer_factory=lambda *args: None, environ={})
        with self.assertRaises(ValueError):
            integration.attach(runner, "baseline", "correctness_capture", environ={})

    def test_actual_processed_boundary_single_original_call_and_cap(self):
        sampler, other, snapshots, stored = FakeSampler(), FakeSampler(), FakeSnapshots(), []
        def writer(packet):
            stored.append(packet)
            return {"bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()}
        observer = capture.CaptureObserver(None, sampler, "baseline", writer, snapshots,
                                           context_fn=lambda runner, rows: {"actual_rows": rows})
        observer.install()
        metadata = SimpleNamespace(all_greedy=False, generators={}, top_k=object(), top_p=object())
        for rows in (1, 4):
            for _ in range(20): sampler(FakeLogits(rows), metadata)
        self.assertEqual(sampler.topk_topp_sampler.calls, 40)
        self.assertEqual(len(stored), 32)
        self.assertEqual(len(snapshots.observed), 32)
        self.assertEqual(observer.budget.receipt()["captured_calls_by_rows"], {"1": 16, "2": 0, "3": 0, "4": 16})
        self.assertEqual(other.topk_topp_sampler.calls, 0)
        self.assertNotIn("forward", vars(other))
        metadata.all_greedy = True
        self.assertEqual(sampler(FakeLogits(4), metadata), "greedy")
        self.assertEqual(len(stored), 32)
        self.assertIsNone(observer.active)

    def test_writer_failure_cannot_claim_capture(self):
        sampler = FakeSampler()
        observer = capture.CaptureObserver(None, sampler, "baseline", lambda data: {"bytes": 0},
                                           FakeSnapshots(), context_fn=lambda runner, rows: {})
        observer.install()
        metadata = SimpleNamespace(all_greedy=False, generators={}, top_k=None, top_p=None)
        with self.assertRaises(ValueError): sampler(FakeLogits(1), metadata)
        self.assertEqual(observer.budget.receipt()["packets"], 0)
        self.assertIsNone(observer.active)

    def test_budget_and_accounting_fail_closed(self):
        budget = capture.CaptureBudget()
        with self.assertRaises(ValueError): budget.select(5, protocol.VOCAB)
        with self.assertRaises(ValueError): budget.select(1, 128)
        budget.total_bytes = protocol.MAX_CAPTURE_BYTES_PER_ARM
        with self.assertRaises(ValueError): budget.check_bytes(b"x")
        rows = 80
        receipt = {"phase": "correctness_capture", "captured_calls_by_rows": {"1": 16, "2": 0, "3": 0, "4": 16},
                   "packets": 32, "captured_rows": rows, "processed_logit_bytes": rows * protocol.VOCAB * 4,
                   "total_packet_bytes": rows * protocol.VOCAB * 4 + 1234, "timing_eligible": False}
        protocol.check_capture_accounting(receipt)
        for key, value in (("captured_rows", 79), ("timing_eligible", True), ("total_packet_bytes", 1)):
            bad = deepcopy(receipt); bad[key] = value
            with self.assertRaises(ValueError): protocol.check_capture_accounting(bad)
        bad = deepcopy(receipt); bad["captured_calls_by_rows"]["4"] = 15
        with self.assertRaises(ValueError): protocol.check_capture_accounting(bad)

    def test_packet_framing_and_no_silent_truncation(self):
        builder = capture.PacketBuilder(SimpleNamespace(Tensor=type("FakeTensor", (), {})))
        header = {"schema": "qwen35_sampler_capture_v1", "timing_eligible": False, "actual": None,
                  "phase": "correctness_capture", "arm": "baseline"}
        packet = builder.finish(header)
        decoded, raw = capture.parse_packet(packet)
        self.assertEqual(decoded, header)
        self.assertEqual(len(raw), 0)
        for bad in (b"bad", packet[:12], capture.MAGIC + struct.pack("<Q", protocol.MAX_JSON_BYTES + 1)):
            with self.assertRaises(ValueError): capture.parse_packet(bad)
        with self.assertRaises(TypeError): builder.value(object())
        with self.assertRaises(ValueError): builder.value(float("nan"))
        with self.assertRaises(ValueError): builder.finish({"oversized": "x" * protocol.MAX_JSON_BYTES})

    def test_original_model_gates_metadata_only_and_reject_reductions(self):
        validators = full_checks.original_validators()
        files = {"state_copy": "state-copy.json", "numeric": "real-logits-validation.json",
                 "cold": "cold-buckets.json", "long": "long-decode-validation.json"}
        for phase, filename in files.items():
            report = protocol.read_small(protocol.BASE / filename)
            result = full_checks.check_model_metadata(report, phase, validators)
            self.assertFalse(result["fresh_execution_proven"])
            self.assertFalse(result["full_model_qualified"])
            if phase != "state_copy":
                if phase == "cold": report["experiments"].pop()
                else: report["rows"].pop()
                with self.assertRaises((ValueError, AssertionError)):
                    full_checks.check_model_metadata(report, phase, validators)

    def test_quality_saved_text_regrade_and_reject_changed_text(self):
        validators = full_checks.original_validators()
        quality = protocol.read_small(protocol.BASE / "quality96.json")
        suite_path = protocol.ROOT / "quality-suite.json"
        suite, digest = protocol.read_small(suite_path), hashlib.sha256(suite_path.read_bytes()).hexdigest()
        result = full_checks.check_quality_pair(quality, deepcopy(quality), deepcopy(quality), suite, digest, validators)
        self.assertEqual(result["all_texts_equal"], 96)
        self.assertFalse(result["fresh_execution_proven"])
        bad = deepcopy(quality); bad["rows"][0]["text"] += " changed"
        with self.assertRaises(ValueError):
            full_checks.check_quality_pair(quality, bad, quality, suite, digest, validators)

    def test_actual_bounded_forward_explicit_generators_short_circuit(self):
        tree = ast.parse((HERE / "qwen35_sampler_bounded_operation_v1.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BoundedTopKTopPSampler")
        fn = deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"))
        fn.returns = None
        for arg in fn.args.args: arg.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        namespace = {}
        exec(compile(module, "actual_bounded_forward_ast", "exec"), namespace)
        calls = []
        def reference(*args): calls.append(args); return "original-token-and-rng-path"
        def forbidden(*args): raise AssertionError("supported predicate must not run for explicit generators")
        fake = SimpleNamespace(reference=reference, _supported=forbidden)
        args = object(), {0: object()}, object(), object()
        self.assertEqual(namespace["forward"](fake, *args), "original-token-and-rng-path")
        self.assertEqual(calls, [args])
        calls.clear(); fake._supported = lambda *args: False
        args = object(), {}, None, None
        self.assertEqual(namespace["forward"](fake, *args), "original-token-and-rng-path")
        self.assertEqual(calls, [args])

    def http_fixture(self, candidate=False):
        inputs = {str(n): list(range(n)) for n in (32, 128, 512, 1536)}
        def group(budgets, stochastic):
            requests = []
            for n, m in budgets:
                requests.append({"input_tokens": n, "output_tokens": m,
                                 "usage": {"prompt_tokens": n, "completion_tokens": m, "total_tokens": n + m},
                                 "text": "changed unseeded" if candidate and stochastic else "same",
                                 "elapsed_s": 1., "stream_events_s": [], "ttft_s": None, "mean_token_interval_s": None})
            count = sum(m for _, m in budgets)
            elapsed = .5 if candidate else 1.
            return {"requests": requests, "output_tokens": count, "elapsed_s": elapsed, "throughput_tok_s": count / elapsed}
        report = {"complete": True, "case": self.case, "inputs": inputs, "workloads": {}}
        for name, (n, m) in http_checks.FIXED.items():
            stochastic = name == "short_stochastic"
            rounds = [{"single": group([(n, m)], stochastic), "concurrent": group([(n, m)] * 4, stochastic)} for _ in range(3)]
            report["workloads"][name] = {"rounds": rounds, "sampling": {"temperature": .7, "top_p": .9, "top_k": 40} if stochastic else {},
                "batch": 4, "output_tokens_per_request": m, "model": self.case["served_name"], "prompt_inputs": [inputs[str(n)]],
                "streaming": group([(n, m)] * 4, stochastic), "summary": {key + "_median_tok_s": rounds[0][key]["throughput_tok_s"] for key in ("single", "concurrent")}}
        rounds = [{"trial": i, **group(http_checks.MIXED, True)} for i in range(3)]
        report["workloads"]["mixed_lengths"] = {"rounds": rounds, "sampling": {"temperature": .7, "top_p": .9, "top_k": 40},
            "max_client_inflight": 4, "requests_per_round": 16, "summary": {"median_tok_s": rounds[0]["throughput_tok_s"]}}
        return report

    def test_full_http_rejects_incomplete_or_greedy_changed_or_observed_timing(self):
        old, new = self.http_fixture(), self.http_fixture(True)
        hosts = []
        for arm in protocol.ARMS:
            hosts.append({**protocol.host_identity(arm, "measured", {"source": "a" * 64}),
                          "graph_reuse_receipt_sha256": "b" * 64,
                          "effective_host_settings": {"DEBUG": "original", "threads": 1}, "frozen_model_case": self.case})
        result = protocol.compare_full_http(old, new, self.case, old["inputs"], *hosts)
        self.assertEqual(result["request_pairs"], 124)
        self.assertEqual(result["identical_greedy_texts"], 57)
        self.assertEqual(result["identical_stochastic_texts"], 0)
        bad = deepcopy(new); bad["workloads"]["short_decode"]["rounds"][0]["single"]["requests"][0]["text"] = "changed"
        with self.assertRaises(ValueError): protocol.compare_full_http(old, bad, self.case, old["inputs"], *hosts)
        bad = deepcopy(new); bad["workloads"]["mixed_lengths"]["rounds"].pop()
        with self.assertRaises(AssertionError): protocol.compare_full_http(old, bad, self.case, old["inputs"], *hosts)
        hosts[1]["observation_enabled"] = True
        with self.assertRaises(ValueError): protocol.compare_full_http(old, new, self.case, old["inputs"], *hosts)

    def test_finite_driver_synthetic_ports_and_no_timing_after_model_failure(self):
        owner = self
        events = []
        invalid_numeric = [False]
        invalid_client = [False]
        root_owner = {"unit": "synthetic-root", "pid": 99, "start_ticks": 88, "invocation_id": "c" * 32}
        def guard(stage, run_id):
            events.append(("guard", stage))
            return {"admitted": True, "exclusive": True, "run_id": run_id, "root_owner": root_owner}
        def reuse(case):
            self.assertEqual(case, self.case)
            return {"validated": True, "compile_permitted": False, "provenance_unchanged": True, "sha256": "b" * 64}
        def receipt(run_id, **extra):
            return {"run_id": run_id, "returncode": 0, "fresh_execution": True,
                    "source_test_fake_port": True, **extra}
        def model(arm, phase, directory, run_id):
            events.append(("model", arm, phase))
            value = protocol.read_small(protocol.BASE / driver.MODEL_FILES[phase])
            if invalid_numeric[0] and phase == "numeric": value["rows"].pop()
            driver.write_new(directory / driver.MODEL_FILES[phase], value)
            return receipt(run_id, arm=arm, phase=phase)
        def vectors(arm, before, after, run_id):
            events.append(("vectors", arm))
            return receipt(run_id, arm=arm, phase="vectors203", complete=True, frames=203, identical_vectors=203)
        class Server:
            def __init__(self, arm, phase, directory, worker, run_id, identity):
                self.arm, self.phase, self.directory = arm, phase, directory
                self.active, self.run_id = False, run_id
                self.server_owner = {"unit": "synthetic-" + str(directory), "pid": 123,
                                     "start_ticks": 456, "invocation_id": "d" * 32}
                self.attachment = {"pid": 321, "arm": arm, "phase": phase, "worker_class": worker,
                                   "host_identity_sha256": identity["sha256"], "operator_replaced": arm == "candidate",
                                   "observation_enabled": phase == "correctness_capture", "debug_behavior": "unchanged",
                                   "effective_host_settings": {"threads": 1, "DEBUG": "original"}}
            def __enter__(self):
                self.active = True
                events.append(("server_enter", self.arm, self.phase, self.directory.name))
                if self.phase == "correctness_capture":
                    driver.write_new(self.directory / "packets" / "capture-receipt.json", {
                        "phase": self.phase, "arm": self.arm, "captured_calls_by_rows": {"1": 16, "2": 0, "3": 0, "4": 16},
                        "packets": 32, "captured_rows": 80, "processed_logit_bytes": 80 * protocol.VOCAB * 4,
                        "total_packet_bytes": 80 * protocol.VOCAB * 4 + 1234, "timing_eligible": False})
                return self.attachment
            def __exit__(self, *exc):
                self.active = False
                events.append(("server_exit", self.arm, self.phase, self.directory.name))
        def audit_server(manager, kind, run_id):
            owner.assertEqual(manager.active, kind == "live")
            events.append(("server_audit", kind, manager.arm))
            common = {"run_id": run_id, "root_owner": root_owner, "server_owner": manager.server_owner,
                      "observer_role": "root_owner_auditor", "observer_source": "source", "observer_source_sha256": "a" * 64,
                      "observed_unix_ns": 100 if kind == "live" else 101}
            if kind == "live":
                return {**common, "observation_kind": "live_ready", "exit_observed": False, "server_active": True, "ready": True,
                        "worker_owner": {"pid": 321, "start_ticks": 654}, "proc_owner_matches": True,
                        "invocation_matches": True, "worker_in_server_cgroup": True}
            return {**common, "observation_kind": "terminal_cleanup", "exit_observed": True, "server_active": False,
                    "remaining_processes": [], "cleanup_complete": True, "stop_requested": True, "stop_outcome_accepted": True,
                    "systemd_exit_status": {"ExecMainCode": 1, "ExecMainStatus": 0, "Result": "success"}}
        def sampler(arm, directory, run_id):
            return receipt(run_id, arm=arm, phase="sampler_replay", passed=True, packets=32, rows=80,
                           all_own_tokens_and_rng_reproduced=True, all_distribution_gates_passed=True, fast_path_rows=80)
        def controls(run_id):
            return receipt(run_id, phase="sampler_controls", passed=True,
                           controls=[{"control": name, "passed": True} for name in protocol.SAMPLER_GATES["controls"]])
        class SyntheticClients:
            def diagnostic(self, case, output):
                events.append(("diagnostic", str(output)))
                if invalid_client[0]: raise RuntimeError("synthetic client failure")
            def smoke(self, case, output):
                return protocol.read_small(protocol.BASE / "smoke.json")
            def quality(self, case, output):
                events.append(("quality", str(output)))
                return protocol.read_small(protocol.BASE / "quality96.json")
            def initial(self, case, output):
                events.append(("initial", str(output)))
                report = owner.http_fixture()["workloads"]["short_decode"]
                report["output_tokens_per_request"] = 120
                groups = [g[k] for g in report["rounds"] for k in ("single", "concurrent")] + [report["streaming"]]
                for group in groups:
                    for row in group["requests"]:
                        row["usage"].update(completion_tokens=120, total_tokens=248)
                    group["output_tokens"] = len(group["requests"]) * 120
                    group["throughput_tok_s"] = group["output_tokens"] / group["elapsed_s"]
                return report
            def workloads(self, case, output):
                events.append(("workloads", str(output)))
                return owner.http_fixture("candidate" in output.parts)
        ports = driver.GuardedPorts(guard, reuse, model, vectors, Server, sampler, controls, audit_server)
        with tempfile.TemporaryDirectory(prefix="qwen35-source-fake-driver-") as tmp:
            with patch.object(driver, "nvme_destination", lambda path: Path(path)):
                destination = Path(tmp) / "synthetic-complete"
                result = driver.run(ports, destination, {"source": "a" * 64}, SyntheticClients())
                self.assertTrue(result["execution_finished"])
                self.assertFalse(result["full_model_qualified"])
                self.assertTrue(result["independent_final_audit_required"])
                self.assertEqual(sum(e[0] == "model" for e in events), 8)
                self.assertEqual(sum(e[0] == "vectors" for e in events), 2)
                self.assertEqual(sum(e[0] == "server_enter" for e in events), 6)
                self.assertEqual(sum(e[0] == "server_exit" for e in events), 6)
                self.assertEqual(sum(e[0] == "workloads" for e in events), 2)
                with self.assertRaises(ValueError): driver.run(ports, destination, {"source": "a" * 64}, SyntheticClients())
                invalid_numeric[0] = True
                events.clear()
                with self.assertRaises(AssertionError):
                    driver.run(ports, Path(tmp) / "synthetic-failure", {"source": "a" * 64}, SyntheticClients())
                self.assertFalse(any(e[0] in ("server_enter", "workloads", "initial", "quality") for e in events))
                invalid_numeric[0], invalid_client[0] = False, True
                events.clear()
                with self.assertRaisesRegex(RuntimeError, "synthetic client failure"):
                    driver.run(ports, Path(tmp) / "synthetic-client-failure", {"source": "a" * 64}, SyntheticClients())
                self.assertEqual(sum(e[0] == "server_enter" for e in events), 1)
                self.assertEqual(sum(e[0] == "server_exit" for e in events), 1)
                self.assertEqual([e[1] for e in events if e[0] == "server_audit"], ["live", "terminal"])
                self.assertFalse(any(e[0] in ("workloads", "initial", "quality") for e in events))
                terminal_path = Path(tmp) / "synthetic-client-failure" / "baseline" / "capture" / "server-terminal-root-audit.json"
                self.assertTrue(json.loads(terminal_path.read_text())["cleanup_complete"])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        builtins.__import__ = real_import
