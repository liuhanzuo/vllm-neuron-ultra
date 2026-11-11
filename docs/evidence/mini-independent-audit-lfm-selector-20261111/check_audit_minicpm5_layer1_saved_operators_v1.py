"""Pure post-exit auditor tests: AST-isolated functions, explicit mocks only.

Never import the audit runtime, resolve actual history, inspect processes or
locks, read native/model bytes, or launch work. The replay double uses twelve
in-memory 6-byte BF16 arrays (72 bytes total), never the real 1,284 captures.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import sys
from types import SimpleNamespace as NS
import unittest

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path.insert(0, str(ROOT))
from research import check_compare_minicpm5_layer1_saved_operators_v1 as prior

SOURCE = Q / "audit_minicpm5_layer1_saved_operators_v1.py"
TREE = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
compile(TREE, str(SOURCE), "exec")
require, digest, ref, extract = prior.require, prior.digest, prior.ref, prior.extract
CALLS = prior.saved.comparison.CALLS


def pure():
    old = prior.pure()
    life = extract(ast.parse((Q / "minicpm5_layer1_operator_native_lifecycle_core_v2.py").read_text()),
        ["ref", "process"], dict(require=require, Path=Path, re=re, copy=copy,
        PROCESS_KEYS=("pid", "start_ticks", "ppid", "argv", "cgroup")))
    old["r"].life.ref = life["ref"]; old["r"].life.process = life["process"]
    old["r"].LOCK_POLICY = {"original": "retained"}
    old["r"].completion = NS(physical=NS(PAIR_SCHEMA="synthetic-complete-pair"))
    previous = NS(**{key: old[key] for key in ("output_paths", "write_report", "audited", "run")},
                  SELF=str(prior.SOURCE), REPORT_SCHEMA="minicpm5_saved_operator_serial_call_report_v1",
                  SUMMARY_SCHEMA="minicpm5_saved_operator_serial_summary_v1", OUTPUT_RESERVE=16 * 2**20)
    namespace = dict(require=require, Path=Path, re=re, copy=copy, hashlib=hashlib, json=json, Q=Q, ROOT=ROOT,
                     SELF=str(SOURCE), LIMITS=prior.LIMITS, r=old["r"], core=old["core"], previous=previous,
                     reader=NS(comparison=prior.saved.comparison), audited=old["audited"],
                     COMPLETED_LFM_TEXT=ast.literal_eval(next(node.value for node in TREE.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "COMPLETED_LFM_TEXT" for target in node.targets))),
                     EXIT_SCHEMA="minicpm5_saved_operator_comparison_actual_exit_v1",
                     RESULT_SCHEMA="minicpm5_saved_operator_post_exit_independent_audit_v1")
    return extract(TREE, ["validate_spec", "comparison_terminal", "historical_freshness", "saved_storage", "expected_call",
        "report_references", "verify_reports", "expected_summary", "verify_summary", "audit", "run"], namespace)


def spec_fixture():
    spec = dict(schema="minicpm5_saved_operator_post_exit_audit_spec_v1", comparison_exit=ref("exit"),
        current_serial_catalog=ref("catalog"), source_extension={str(SOURCE): "c" * 64},
        output=str(Q / "minicpm5-layer1-saved-operator-comparison-synthetic-post-exit-audit-test.json"),
        limits=copy.deepcopy(prior.LIMITS))
    review = dict(schema="minicpm5_saved_operator_post_exit_audit_source_review_v1", reviewer="/root", passed=True,
                  blockers=[], spec=ref("audit-spec"), source_extension=spec["source_extension"])
    return spec, review


def terminal_fixture(ns):
    spec, _ = prior.spec_fixture()
    owner = dict(pid=123, start_ticks="456", ppid=100, cgroup="0::/synthetic.scope",
        argv=["python3", "-B", str(prior.SOURCE), "--spec", ref("spec")["path"], "--spec-sha256", ref("spec")["sha256"],
              "--review", ref("review")["path"], "--review-sha256", ref("review")["sha256"]])
    ended = dict(schema=ns["EXIT_SCHEMA"], reviewer="/root", spec=ref("spec"), review=ref("review"), live=ref("live"),
        result=dict(path=str(ns["previous"].output_paths(spec["output_prefix"])["summary"]), sha256="a" * 64),
        owner=owner, actual_tool=dict(kind="tool_session", session_id=42, terminal_chunk="abcdef", exit_code=0),
        actual_tool_exit_observed=True, exact_owner_gone=True, lock_identity=prior.LOCK,
        self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
    live = dict(schema="root_minicpm5_saved_operator_comparison_live_v1", reviewer="/root", owner=copy.deepcopy(owner),
        spec=ended["spec"], review=ended["review"], actual_tool=dict(kind="tool_session", session_id=42),
        actual_owner_observed=True, actual_tool_exit_observed=False, actual_executable="/synthetic/native-python",
        observed_at="2026-11-11T00:00:00+00:00", full_model_qualified=False, new_speedup_claim=False)
    summary = dict(schema=ns["previous"].SUMMARY_SCHEMA, complete=True, owner=copy.deepcopy(owner),
                   comparison_spec=ended["spec"], comparison_review=ended["review"])
    return ended, live, summary, spec


def replay_fixture(ns):
    """Tiny explicit reader double; actual frozen word-comparator arithmetic."""
    ended, _, summary, spec = terminal_fixture(ns)
    plan = dict(schema="synthetic-72-byte-test-plan", runtime_admitted=False,
                calls=[dict(call=call, native="virtual-native-%d" % call, sdk="virtual-sdk-%d" % call) for call in CALLS])
    arrays = {entry[side]: struct.pack("<HHH", 0x8000 if side == "sdk" else 0, 0x3f80, 0x4000)
              for entry in plan["calls"] for side in ("native", "sdk")}
    calls = []
    def compare_all(actual_plan, read, guard, sink, *, expected_plan_sha256):
        require(actual_plan == plan and expected_plan_sha256 == digest(plan), "Exact tiny test plan")
        reports = []
        for entry in actual_plan["calls"]:
            native, sdk = read(entry["native"]), read(entry["sdk"])
            metric = prior.saved.comparison.compare_words(native, sdk, "torch.bfloat16", guard)
            report = dict(call=entry["call"], entire_synthetic_metric=metric, cause_proven=False)
            reports.append(report); sink(report)
        return dict(synthetic_fixture_only=True, reports=reports, capture_files=12, capture_bytes=72)
    ns["reader"].compare_all = compare_all
    def read(key): calls.append(key); return arrays[key]
    reports = []
    recomputed = compare_all(plan, lambda key: arrays[key], lambda: None, reports.append, expected_plan_sha256=digest(plan))
    envelopes = [ns["expected_call"](ended["spec"], ended["review"], plan, report) for report in reports]
    paths = ns["previous"].output_paths(spec["output_prefix"])
    summary.update(reports=[dict(call=entry["call"], report=dict(path=str(paths[str(entry["call"])]), sha256=digest(envelope)),
        descriptor_call_sha256=digest(entry), comparison_sha256=digest(report))
        for entry, report, envelope in zip(plan["calls"], reports, envelopes)],
        comparison=recomputed, descriptor_plan_sha256=digest(plan),
        descriptor_plan_header={key: value for key, value in plan.items() if key != "calls"})
    return ended, spec, plan, summary, envelopes, read, calls


class AuditChecks(unittest.TestCase):
    def setUp(self): self.ns = pure()

    def test_01_exact_new_spec_original_limits_and_review(self):
        spec, review = spec_fixture()
        self.assertEqual(self.ns["validate_spec"](ref("audit-spec"), ref("audit-review"), spec, review), Path(spec["output"]))
        for fault in ("limits", "extra", "review", "binding", "output", "native_pin", "traversal", "missing_exit"):
            s, v = copy.deepcopy((spec, review))
            if fault == "limits": s["limits"]["wall_seconds"] += 1
            elif fault == "extra": s["partial"] = True
            elif fault == "review": v["passed"] = False
            elif fault == "binding": v["spec"] = ref("other")
            elif fault == "output": s["output"] = str(Q / "overwrite-summary.json")
            elif fault == "native_pin": s["source_extension"][str(ROOT / "native.so")] = "d" * 64
            elif fault == "traversal": s["source_extension"][str(ROOT / ".." / "outside.py")] = "d" * 64
            else: del s["comparison_exit"]
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                self.ns["validate_spec"](ref("audit-spec"), ref("audit-review"), s, v)

    def test_02_exact_external_terminal_live_and_whole_summary(self):
        ended, live, summary, spec = terminal_fixture(self.ns)
        check = lambda e, l, s: self.ns["comparison_terminal"](e, l, s, spec, "/synthetic/native-python")
        self.assertEqual(check(ended, live, summary), ended["actual_tool"])
        self.assertEqual(len(ended), 14)
        faults = ("failed", "bool_exit", "unobserved", "still_alive", "self_exit", "promotion", "extra", "missing_summary",
                  "incomplete", "wrong_session", "owner", "argv", "executable", "result_path", "summary_spec", "reviewer")
        for fault in faults:
            e, l, s = copy.deepcopy((ended, live, summary))
            if fault == "failed": e["actual_tool"]["exit_code"] = 1
            elif fault == "bool_exit": e["actual_tool"]["exit_code"] = False
            elif fault == "unobserved": e["actual_tool_exit_observed"] = False
            elif fault == "still_alive": e["exact_owner_gone"] = False
            elif fault == "self_exit": e["self_exit_claimed"] = True
            elif fault == "promotion": e["full_model_qualified"] = True
            elif fault == "extra": e["extra"] = True
            elif fault == "missing_summary": del e["result"]
            elif fault == "incomplete": s["complete"] = False
            elif fault == "wrong_session": l["actual_tool"]["session_id"] += 1
            elif fault == "owner": l["owner"]["start_ticks"] = "999"
            elif fault == "argv": e["owner"]["argv"][0] = "wrong-python"
            elif fault == "executable": l["actual_executable"] = "/wrong/python"
            elif fault == "result_path": e["result"] = ref("partial-call-only")
            elif fault == "summary_spec": s["comparison_spec"] = ref("other-spec")
            else: e["reviewer"] = "/self"
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): check(e, l, s)

    def test_03_only_six_ordered_original_report_paths(self):
        _, spec, _, summary, _, _, _ = replay_fixture(self.ns)
        self.assertEqual(len(self.ns["report_references"](spec, summary)), 6)
        for fault in ("missing", "extra", "reorder", "foreign", "bad_digest", "unknown", "wrong_call"):
            s = copy.deepcopy(summary)
            if fault == "missing": s["reports"].pop()
            elif fault == "extra": s["reports"].append(s["reports"][0])
            elif fault == "reorder": s["reports"].reverse()
            elif fault == "foreign": s["reports"][0]["report"] = ref("foreign")
            elif fault == "bad_digest": s["reports"][0]["comparison_sha256"] = "not-a-sha"
            elif fault == "unknown": s["reports"][0]["unknown"] = True
            else: s["reports"][0]["call"] = 7
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["report_references"](spec, s)

    def test_04_tiny_full_replay_signed_zero_and_all_six_envelopes(self):
        ended, spec, plan, summary, values, read, reads = replay_fixture(self.ns)
        recomputed, verified = self.ns["verify_reports"](ended, spec, plan, summary, values, read, lambda: None)
        self.assertEqual(recomputed, summary["comparison"]); self.assertEqual(verified, summary["reports"])
        self.assertEqual(len(reads), len(set(reads))); self.assertEqual(len(reads), 12)
        for report in recomputed["reports"]:
            metric = report["entire_synthetic_metric"]
            self.assertEqual(metric["different_words"], 1); self.assertEqual(metric["different_bits"], 1)
            self.assertEqual(metric["maximum_absolute_error"], 0.0); self.assertFalse(metric["all_bits_equal"])

    def test_05_every_metric_descriptor_field_type_and_digest_checked(self):
        ended, spec, plan, summary, values, read, _ = replay_fixture(self.ns)
        for fault in ("metric", "missing_metric", "unknown_metric", "bool_for_one", "negative_zero", "descriptor", "missing_envelope",
                      "unknown_envelope", "comparison_digest", "descriptor_digest", "plan_digest", "header", "summary_metric"):
            s, v = copy.deepcopy((summary, values))
            metric = v[-1]["comparison"]["entire_synthetic_metric"]
            if fault == "metric": metric["maximum_absolute_error"] = 1.0
            elif fault == "missing_metric": del metric["native_squared_norm"]
            elif fault == "unknown_metric": metric["invented"] = True
            elif fault == "bool_for_one": metric["different_words"] = True
            elif fault == "negative_zero": metric["maximum_absolute_error"] = -0.0
            elif fault == "descriptor": v[-1]["complete_call_capture_descriptors"]["native"] = "foreign"
            elif fault == "missing_envelope": del v[-1]["self_exit_claimed"]
            elif fault == "unknown_envelope": v[-1]["unknown"] = None
            elif fault == "comparison_digest": s["reports"][-1]["comparison_sha256"] = "f" * 64
            elif fault == "descriptor_digest": s["reports"][-1]["descriptor_call_sha256"] = "f" * 64
            elif fault == "plan_digest": s["descriptor_plan_sha256"] = "f" * 64
            elif fault == "header": s["descriptor_plan_header"]["schema"] = "other"
            else: s["comparison"]["reports"][-1]["entire_synthetic_metric"]["words"] = 1
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                self.ns["verify_reports"](ended, spec, plan, s, v, read, lambda: None)

    def test_06_historical_idle_chain_validated_without_current_tip_resolution(self):
        life = self.ns["r"].life
        publication = extract(ast.parse((Q / "minicpm5_layer1_operator_sdk_runtime_completion_v3.py").read_text()),
                              ["publication_view"], dict(copy=copy, require=require))["publication_view"]
        execution = extract(ast.parse((Q / "minicpm5_layer1_operator_native_current_execution_v1.py").read_text()),
            ["handoff_path", "freshness_shape", "validate_predecessor_chain"],
            dict(Path=Path, re=re, copy=copy, require=require, life=life, latest=NS(publication_view=publication)))
        life.ROOT = str(ROOT); self.ns["r"].execution = NS(**execution)
        bound = dict(path=str(Q / "continuation-handoff-synthetic-bound.json"), sha256="a" * 64)
        tip = dict(path=str(Q / "continuation-handoff-synthetic-publication.json"), sha256="b" * 64)
        serial = dict(reference=ref("catalog"), current_handoff=bound)
        freshness = dict(complete=True, historical_binding_unchanged=True, serial_catalog=serial["reference"],
                         bound_terminal=bound, latest_handoff=tip, publication_only_successors=[tip])
        base = dict(live_tool_sessions=[], active_heavy_units=[], serial_execution=True, mini_operator_current={"all_history": "retained"})
        pub = dict(copy.deepcopy(base), publication_only_since_bound_terminal=True, previous=bound, publication="presentation")
        values = {bound["path"]: base, tip["path"]: pub}; reads = []
        def read(reference): reads.append(reference); return copy.deepcopy(values[reference["path"]])
        self.assertEqual(self.ns["historical_freshness"](freshness, serial, read), freshness)
        self.assertEqual(reads, [tip, bound])
        for fault in ("live", "scope", "previous", "bound", "latest", "duplicate"):
            f = copy.deepcopy(freshness); old = copy.deepcopy(values)
            if fault == "live": values[bound["path"]]["live_tool_sessions"] = [1]
            elif fault == "scope": values[tip["path"]]["mini_operator_current"]["all_history"] = "dropped"
            elif fault == "previous": values[tip["path"]]["previous"] = ref("foreign")
            elif fault == "bound": f["bound_terminal"] = tip
            elif fault == "latest": f["latest_handoff"] = bound
            else: f["publication_only_successors"].append(tip)
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["historical_freshness"](f, serial, read)
            values = old

    def test_07_storage_keeps_checked_historical_telemetry_exact_usage_types(self):
        usage = dict(files=1093, total_allocated_bytes=37441536, physical_checks_performed=False, pending_destinations=[])
        storage = dict(available_bytes=dict(root=28 * 2**30 + 16 * 2**20, output=30 * 2**30),
                       usage=usage, maximum_check_interval_seconds=1)
        self.assertEqual(self.ns["saved_storage"](storage, usage), storage)
        for fault in ("root_floor", "output_floor", "interval", "bool_interval", "usage", "usage_type", "extra"):
            value = copy.deepcopy(storage)
            if fault == "root_floor": value["available_bytes"]["root"] -= 1
            elif fault == "output_floor": value["available_bytes"]["output"] -= 1
            elif fault == "interval": value["maximum_check_interval_seconds"] = 2
            elif fault == "bool_interval": value["maximum_check_interval_seconds"] = True
            elif fault == "usage": value["usage"]["files"] -= 1
            elif fault == "usage_type": value["usage"]["physical_checks_performed"] = 0
            else: value["extra"] = True
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["saved_storage"](value, usage)

    def test_08_complete_summary_reconstruction_preserves_failed_sdk_and_pending_self_exit(self):
        ended, spec, plan, partial, _, _, _ = replay_fixture(self.ns)
        context = dict(plan=plan, original_spec=spec, pins_plan={"source_sha256": {str(prior.SOURCE): "d" * 64}},
            comparison_source={"full_original_source": "retained"}, controls={"all_controls": True}, inventory={"all_files": True},
            sdk_controls={"all_controls": True}, native_trace={"native_line_table": "present"}, sdk_trace_proof={"sdk_line_table": "present"})
        original = dict(serial_freshness={"historical": "validated"}, complete_physical_pair={"historical": "validated"})
        summary = self.ns["expected_summary"](ended, original, context, partial["comparison"], partial["reports"], {"storage": "validated"})
        args = (context, partial["comparison"], partial["reports"], {"storage": "validated"})
        self.assertEqual(self.ns["verify_summary"](ended, summary, *args), summary)
        self.assertEqual(summary["original_sdk_model_exit_code"], 1)
        self.assertFalse(summary["original_sdk_model_runtime_successful"])
        self.assertTrue(summary["external_comparison_audit_pending"])
        # Exact producer dictionary schema comparison is source-only, not a
        # read of its real result or a claim to replay the 77.8 MB captures.
        producer = next(node for node in prior.TREE.body if isinstance(node, ast.FunctionDef) and node.name == "compare")
        result = next(node.value for node in ast.walk(producer) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets))
        self.assertEqual(set(summary), {keyword.arg for keyword in result.keywords})
        for fault in ("capture_files_read", "capture_bytes_read", "original_sdk_model_exit_code", "actual_comparison_tool_exit_verified",
                      "external_comparison_audit_pending", "model_reexecuted", "causal_operator_proven", "full_model_qualified",
                      "new_speedup_claim", "missing", "unknown", "controls", "native_trace", "sdk_trace"):
            bad = copy.deepcopy(summary)
            if fault == "missing": del bad["reviewer"]
            elif fault == "unknown": bad["unknown"] = True
            elif fault == "controls": bad["original_model_controls"]["all_controls"] = False
            elif fault == "native_trace": bad["native_trace_reconstructed_in_original_interpreter"].clear()
            elif fault == "sdk_trace": bad["authenticated_original_sdk_trace"].clear()
            elif fault == "original_sdk_model_exit_code": bad[fault] = True
            elif type(bad[fault]) is bool: bad[fault] = not bad[fault]
            else: bad[fault] -= 1
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["verify_summary"](ended, bad, *args)

    def test_09_bad_exit_stops_before_summary_or_reconstruction(self):
        spec, review = spec_fixture(); ended, _, _, _ = terminal_fixture(self.ns)
        ended["actual_tool"]["exit_code"] = 1
        loads = []; guards = []
        values = {ref("audit-spec")["path"]: spec, ref("audit-review")["path"]: review, spec["comparison_exit"]["path"]: ended}
        class VirtualPath:
            def exists(self): return False
        class Metadata:
            def read(self, reference): loads.append(reference["path"]); return values[reference["path"]]
        self.ns["previous"].ComparisonGuard = lambda lease: lambda **kw: guards.append(kw)
        self.ns["r"].Metadata = lambda guard: Metadata()
        self.ns["validate_spec"] = lambda *args: VirtualPath()
        raw = b"synthetic source bytes"; spec["source_extension"] = {str(SOURCE): hashlib.sha256(raw).hexdigest()}
        self.ns["r"].sources = NS(text=lambda path, sha: raw)
        self.ns["reconstruct"] = lambda *args: self.fail("Must not reconstruct after failed comparison")
        with self.assertRaisesRegex(RuntimeError, "successful tool terminal"):
            self.ns["audit"](ref("audit-spec"), ref("audit-review"), NS())
        self.assertEqual(loads, list(values)); self.assertTrue(guards)

    def test_10_runtime_gate_order_complete_sources_scope_and_frozen_replay(self):
        functions = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}
        calls = lambda name: sorted([(node.lineno, ast.unparse(node.func)) for node in ast.walk(functions[name]) if isinstance(node, ast.Call)])
        a = calls("audit"); raw = next(line for line, name in a if name == "verify_reports")
        for wanted in ("core.actual_tool", "r.p.gone", "reconstruct", "audited.pair", "report_references", "reader.validate_plan"):
            self.assertTrue(any(name == wanted and line < raw for line, name in a), wanted)
        for wanted in ("verify_summary", "audited.pair", "metadata.unchanged", "r.verify_loaded", "r.sources.verify_text_sources",
                       "r.sources.check_native_identities", "r.completion.latest.require_latest_terminal", "context['native_io'].unchanged",
                       "previous.check_sdk_controls"):
            self.assertTrue(any(name == wanted and line > raw for line, name in a), wanted)
        reconstruction = [name for _, name in calls("reconstruct")]
        for wanted in ("comparison_terminal", "previous.actual_auditor_terminal", "r.completion.completed", "r.overlays.resolve",
                       "previous.preparation.inherited_source_plan", "core.validate", "audited.validate_spec", "previous.source_plan",
                       "audited.merged_plan", "r.sources.validate_native_receipt", "r.sources.verify_text_sources", "r.verify_loaded",
                       "previous.preparation.completed_inherited_extension", "historical_freshness", "r.completion.physical.validate_pair",
                       "previous.audited_controls", "previous.check_sdk_controls", "native_io.final_inventory",
                       "audited.verifier.expected_trace", "previous.authenticated_sdk_trace", "reader.prepare"):
            self.assertIn(wanted, reconstruction)
        self.assertEqual(reconstruction.count("previous.catalog_rows"), 3)
        all_calls = [ast.unparse(node.func) for node in ast.walk(TREE) if isinstance(node, ast.Call)]
        self.assertNotIn("previous.compare", all_calls)
        self.assertNotIn("sdk_results_core.expected_trace_contract", all_calls)
        self.assertEqual(all_calls.count("reader.compare_all"), 1)
        self.assertNotIn("subprocess.Popen", all_calls)

    def test_11_mocked_entry_retains_original_limits_and_releases_on_failure(self):
        events = []
        self.ns["sys"] = NS(dont_write_bytecode=True, executable="/synthetic/native/python")
        self.ns["r"].life.PYTHON = "/synthetic/native/python"
        self.ns["os"] = NS(sched_setaffinity=lambda *a: events.append(("affinity", a)), nice=lambda *a: events.append(("nice", a)))
        self.ns["resource"] = NS(RLIMIT_AS=1, RLIMIT_CPU=2, RLIMIT_FSIZE=3, setrlimit=lambda *a: events.append(("limit", a)))
        self.ns["signal"] = NS(SIGALRM=1, signal=lambda *a: None, alarm=lambda n: events.append(("alarm", n)))
        self.ns["r"].p = NS(user_bus=lambda: events.append(("bus",)))
        self.ns["r"].Lease = lambda admission: NS(admission=admission)
        self.ns["audited"].support = NS(acquire=lambda: (99, {"pid": 123}, {"passed": True}),
            release_after_physical_drain=lambda fd, proof: events.append(("release", fd, proof)))
        def fail(*args): events.append(("audit",)); raise RuntimeError("synthetic stop before real work")
        self.ns["audit"] = fail
        with self.assertRaisesRegex(RuntimeError, "synthetic stop"):
            self.ns["run"](ref("audit-spec"), ref("audit-review"))
        self.assertEqual(events[-1][0:2], ("release", 99))
        self.assertTrue(events[-1][2]["no_independent_audit_children_or_services_started"])
        self.assertIn(("affinity", (0, {6})), events); self.assertIn(("nice", (19,)), events)
        self.assertIn(("alarm", 900), events); self.assertIn(("alarm", 0), events)
        self.assertEqual([entry[1][1] for entry in events if entry[0] == "limit"],
                         [(2 * 2**30, 2 * 2**30), (600, 605), (2 * 2**20, 2 * 2**20)])
        events.clear(); self.ns["sys"].executable = "/synthetic/wrong-python"
        with self.assertRaisesRegex(RuntimeError, "Original native interpreter"):
            self.ns["run"](ref("audit-spec"), ref("audit-review"))
        self.assertEqual(events, [])

    def test_12_frozen_original_sources_unchanged(self):
        hashes = {"minicpm5_layer1_saved_operator_comparison_v1.py": "8004bec5519ac732bc1a547d02e1a39ca0e63d9f5bdada7b06acece4446805ae",
                  "minicpm5_layer1_saved_operator_reader_v1.py": "53819b175232cf95e09a01d21dd06a18f3683cc8db559db9838cc2d0bd103ad0",
                  "compare_minicpm5_layer1_saved_operators_v1.py": "833ef426630820dd711af34e409b25b5d5d1f94a6f01e228843d45874fd3c2f4"}
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((Q / name).read_bytes()).hexdigest(), expected, name)

    def test_13_only_four_exact_completed_lfm_sdk_text_pins(self):
        spec, review = spec_fixture()
        expected = {
            "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/__init__.py": "3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288",
            "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/_ops.py": "3b58079d356d15cc73082c20b1279468ee9cfe7f5e1340d329a93b6e877a30d5",
            "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/_classes.py": "0db3dc302e28fd085065ea2883ce33fa4af406b9fd1439f2a431f053e9a764b9",
            "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/framework/_torch_convert.py": "2c510075ca5943ce655f3edd21553398ae6d83d05194ddaca21d8da3ff8ea729",
        }
        self.assertEqual(self.ns["COMPLETED_LFM_TEXT"], expected)
        spec["source_extension"].update(expected)
        self.ns["validate_spec"](ref("audit-spec"), ref("audit-review"), spec, review)
        for fault in ("wrong_sha", "neighbor_py", "native_extension", "other_venv", "parent_traversal"):
            s, v = copy.deepcopy((spec, review))
            path, sha = next(iter(expected.items()))
            if fault == "wrong_sha": s["source_extension"][path] = "f" * 64
            elif fault == "neighbor_py": s["source_extension"][path.replace("__init__.py", "unreviewed.py")] = sha
            elif fault == "native_extension": s["source_extension"][path.replace("__init__.py", "_C.so")] = sha
            elif fault == "other_venv": s["source_extension"][path.replace("vllm_neuron_053_py312", "other")] = sha
            else: s["source_extension"][path.replace("/torch/", "/torch/../torch/")] = sha
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                self.ns["validate_spec"](ref("audit-spec"), ref("audit-review"), s, v)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditChecks))
    print(json.dumps(dict(source_only=True, runtime_entry_imported=False, synthetic_arrays=12,
        synthetic_bytes_per_array=6, synthetic_bytes_per_replay=72, actual_capture_files_read=0,
        actual_native_or_model_bytes_read=0, actual_history_resolved=False, actual_lock_or_process_operations=False), sort_keys=True))
    raise SystemExit(not result.wasSuccessful())
