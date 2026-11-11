"""AST-isolated pure wrapper checks and bounded synthetic metadata only.

Never import the runtime entry, resolve real history, read tensor/native bytes,
inspect services/locks or write an output. Lifecycle calls are explicit mocks.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.abc
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import unittest

Q = Path(__file__).resolve().parent
ROOT = Q.parent
sys.path.insert(0, str(ROOT))


class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "torch_neuronx", "neuronx_distributed"}:
            raise AssertionError("Native/model import forbidden in source wrapper test")


sys.meta_path.insert(0, DenyNative())
from research.check_minicpm5_layer1_saved_operator_reader_v1 import metadata_fixture
from research import minicpm5_layer1_saved_operator_reader_v1 as saved

SOURCE = Q / "compare_minicpm5_layer1_saved_operators_v1.py"
TREE = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
compile(TREE, str(SOURCE), "exec")
LIMITS = dict(address_space_bytes=2 * 2**30, cpu_seconds=600, hard_cpu_seconds=605,
              output_file_bytes=2 * 2**20, wall_seconds=900, cpu=6, nice=19)
LOCK = dict(device=66306, inode=8294421)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ref(name): return {"path": str(Q / ("synthetic-comparison-test-" + name + ".json")), "sha256": "a" * 64}


def check_ref(value):
    require(type(value) is dict and set(value) == {"path", "sha256"} and Path(value["path"]).is_absolute()
            and re.fullmatch("[a-f0-9]{64}", value["sha256"]), "metadata reference")


def extract(tree, names, namespace):
    nodes = [next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name) for name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<isolated pure source checks>", "exec"), namespace)
    return namespace


def pure():
    tool_ns = extract(ast.parse((Q / "minicpm5_layer1_operator_native_post_exit_core_v4.py").read_text()), ["actual_tool"],
                      {"require": require, "re": re})
    r = NS(life=NS(ref=check_ref), ORIGINAL_LOCK=LOCK, digest=digest)
    ns = dict(require=require, re=re, Path=Path, copy=copy, json=json, hashlib=hashlib, Q=Q, ROOT=ROOT, SELF=str(SOURCE),
              LIMITS=LIMITS, r=r, core=NS(actual_tool=tool_ns["actual_tool"]), reader=saved,
              preparation=NS(__file__=str(Q / "prepare_minicpm5_layer1_operator_native_v7.py")),
              sdk_results_core=NS(__file__=str(Q / "minicpm5_layer1_operator_sdk_results_v2.py")),
              audited=NS(SELF=str(Q / "audit_minicpm5_layer1_operator_native_result_v4.py")))
    return extract(TREE, ["output_paths", "validate_spec", "audited_controls", "actual_auditor_terminal", "serial_scope", "catalog_rows",
                          "source_plan", "encoded", "write_report", "authenticated_sdk_trace", "run"], ns)


def spec_fixture():
    spec = dict(schema="minicpm5_saved_operator_serial_comparison_spec_v1", model_exit=ref("model"), recorder_exit=ref("recorder"),
                auditor_exit=ref("auditor"), current_serial_catalog=ref("catalog"), source_extension={str(SOURCE): "c" * 64},
                output_prefix=str(Q / "minicpm5-layer1-saved-operator-comparison-synthetic-source-test"), limits=copy.deepcopy(LIMITS))
    review = dict(schema="minicpm5_saved_operator_serial_comparison_source_review_v1", reviewer="/root", passed=True, blockers=[],
                  spec=ref("spec"), source_extension=spec["source_extension"])
    return spec, review


class WrapperChecks(unittest.TestCase):
    def setUp(self): self.ns = pure()

    def test_01_exact_spec_review_and_original_limits(self):
        spec, review = spec_fixture()
        paths = self.ns["validate_spec"](ref("spec"), ref("review"), spec, review)
        self.assertEqual(set(paths), {"4", "5", "6", "11", "12", "13", "summary"})
        for fault in ("limits", "unknown", "review", "spec", "output", "native"):
            s, v = copy.deepcopy((spec, review))
            if fault == "limits": s["limits"]["cpu_seconds"] += 1
            elif fault == "unknown": s["unknown"] = True
            elif fault == "review": v["passed"] = False
            elif fault == "spec": v["spec"] = ref("other")
            elif fault == "output": s["output_prefix"] = "/tmp/out"
            else: s["source_extension"][str(ROOT / "native.so")] = "a" * 64
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["validate_spec"](ref("spec"), ref("review"), s, v)

    def test_02_genuine_separate_auditor_terminal_only(self):
        mt = dict(kind="tool_session", session_id=10, terminal_chunk="abcdef", exit_code=0)
        rt = dict(mt, session_id=11)
        terminal = dict(schema="minicpm5_native_independent_audit_actual_exit_v1", reviewer="/root", audit_spec=ref("as"), audit_review=ref("ar"),
            result=ref("result"), owner={"pid": 123}, actual_tool=dict(mt, session_id=12), actual_tool_exit_observed=True,
            exact_owner_gone=True, lock_identity=LOCK, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
        self.ns["actual_auditor_terminal"](terminal, mt, rt)
        for fault in ("false_exit", "failed", "same_session", "missing_exit", "owner", "promoted", "extra"):
            bad = copy.deepcopy(terminal)
            if fault == "false_exit": bad["actual_tool"]["exit_code"] = False
            elif fault == "failed": bad["actual_tool"]["exit_code"] = 1
            elif fault == "same_session": bad["actual_tool"]["session_id"] = 11
            elif fault == "missing_exit": bad["actual_tool_exit_observed"] = False
            elif fault == "owner": bad["exact_owner_gone"] = False
            elif fault == "promoted": bad["full_model_qualified"] = True
            else: bad["extra"] = True
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["actual_auditor_terminal"](bad, mt, rt)

    def test_03_full_original_control_proof_required(self):
        frames = [{"frame": {"index": i}, "all_three_arms_exact": True} for i in saved.comparison.INDICES]
        proof = dict(schema="minicpm5_layer1_operator_native_independent_verification_v2", complete=True, frames=24,
            direct_attention_calls=72, raw_files=936, raw_bytes=28490724, arm_result_files=72, journals=72, journal_records=2736,
            every_raw_byte_independently_reread=True, every_journal_record_independently_verified=True,
            all_official_attention_and_full_cache_controls_exact=True, every_capture_event_shape_stride_offset_dtype_retained=True,
            original_scale_source_independently_bound=True, original_projection_matrix_hashes_bound=True,
            actual_process_exit_verified=False, fresh_physical_drain_performed=False, arithmetic_reexecuted=False,
            full_model_qualified=False, new_speedup_claim=False, frames_verified=frames)
        ranks = [ref("rank0"), ref("rank1")]
        report = dict(schema="minicpm5_native_post_exit_independent_audit_v1", complete=True, reviewer="/root", model_exit=ref("model"),
            recorder_exit=ref("recorder"), lock_identity=LOCK, sdk_rank_references=ranks, original_sdk_model_exit_code=1,
            actual_native_model_exit_verified=True, actual_native_recorder_exit_verified=True, every_raw_byte_reread_after_all_model_processes_exited=True,
            actual_auditor_tool_exit_verified=False, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False, result=proof)
        manifest = dict(total_raw_bytes=28490724, frames=frames)
        args = (ref("model"), ref("recorder"), ranks, manifest)
        self.ns["audited_controls"](report, *args)
        for fault in ("files", "frame", "controls", "sdk_success", "self_exit"):
            bad = copy.deepcopy(report)
            if fault == "files": bad["result"]["raw_files"] = 720
            elif fault == "frame": bad["result"]["frames_verified"].pop()
            elif fault == "controls": bad["result"]["all_official_attention_and_full_cache_controls_exact"] = False
            elif fault == "sdk_success": bad["original_sdk_model_exit_code"] = 0
            else: bad["actual_auditor_tool_exit_verified"] = True
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): self.ns["audited_controls"](bad, *args)

    def test_04_catalog_retains_rows_and_exact_terminal_tools(self):
        owner = {"pid": 100, "start_ticks": "1"}; tool = {"kind": "tool_session", "session_id": 42, "terminal_chunk": "abcdef", "exit_code": 0}
        old = dict(terminal=ref("failed-audit-v4"), owners=[owner], tool=dict(tool, exit_code=1), units=[], source_sha256={})
        new = dict(terminal=ref("new"), owners=[owner], tool=tool, units=[], source_sha256={})
        expected = [(ref("new"), owner, tool)]
        self.ns["catalog_rows"]({"runs": [old, new]}, {"runs": [old]}, expected)
        for rows in ([new], [old, old, new], [dict(old, tool=tool), new], [old, dict(new, tool=dict(tool, exit_code=1))]):
            with self.assertRaises(RuntimeError): self.ns["catalog_rows"]({"runs": rows}, {"runs": [old]}, expected)

    def test_05_complete_source_map_preserved_disjoint_extension(self):
        names = [str(SOURCE), str(Path(saved.__file__).resolve()), str(Path(saved.comparison.__file__).resolve()),
                 str(Q / "prepare_minicpm5_layer1_operator_native_v7.py"), self.ns["audited"].SELF]
        original = {str(ROOT / "unrelated-original.py"): "a" * 64, names[-1]: "b" * 64}
        inherited = dict(source_sha256=dict(original, **{str(ROOT / "completed-recorder.py"): "c" * 64}))
        self.ns["audited"].validate_source_binding = lambda *args: copy.deepcopy(inherited)
        merged = extract(ast.parse((Q / "audit_minicpm5_layer1_operator_native_result_v4.py").read_text()), ["merged_plan"],
                         {"copy": copy, "require": require})["merged_plan"]
        self.ns["audited"].merged_plan = merged
        resolved = dict(plan={"source_sha256": original}, plan_sha256="d" * 64)
        addition = {name: "e" * 64 for name in names if name not in inherited["source_sha256"]}
        args = (resolved, {"spec": ref("model-spec"), "review": ref("model-review")}, {"overlay": ref("overlay")},
                {"source_extension": {}}, {"source_binding": {"full": "synthetic-receipt"}})
        result, proof = self.ns["source_plan"](*args, {"source_extension": addition})
        self.assertTrue(all(result["source_sha256"][name] == sha for name, sha in original.items()))
        self.assertEqual(proof["complete_source_files"], len(result["source_sha256"]))
        self.assertTrue(proof["complete_source_reconstruction_required"])
        bad = dict(addition, **{next(iter(original)): "a" * 64})
        with self.assertRaises(RuntimeError): self.ns["source_plan"](*args, {"source_extension": bad})

    def test_06_exclusive_bounded_writer_uses_exact_original_encoding(self):
        writes = {}; checks = []
        class VirtualPath:
            def __init__(self, path): self.name = str(path)
            def __str__(self): return self.name
            def exists(self): return self.name in writes
        self.ns["Path"] = VirtualPath
        self.ns["r"].atomic_once = lambda path, value: writes.__setitem__(str(path), self.ns["encoded"](value))
        self.ns["audited"].outer = NS(reference=lambda path: dict(path=str(path), sha256=hashlib.sha256(writes[str(path)]).hexdigest()))
        path = str(Q / "synthetic-never-written.json")
        ref_value = self.ns["write_report"](path, {"complete": False}, lambda **kw: checks.append(kw))
        self.assertEqual(ref_value["sha256"], hashlib.sha256(writes[path]).hexdigest())
        with self.assertRaises(RuntimeError): self.ns["write_report"](path, {}, lambda **kw: None)
        with self.assertRaises(RuntimeError): self.ns["write_report"]("/not-created", {"text": "x" * LIMITS["output_file_bytes"]}, lambda **kw: None)
        self.assertEqual(len(writes), 1)

    def test_07_actual_runtime_gates_precede_raw_and_preserve_frozen_helpers(self):
        compare = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "compare")
        calls = [node for node in ast.walk(compare) if isinstance(node, ast.Call)]
        raw = next(node for node in calls if ast.unparse(node.func) == "reader.compare_all")
        required = ["actual_auditor_terminal", "core.validate", "preparation.inherited_source_plan", "preparation.completed_inherited_extension",
                    "audited_controls", "r.completion.completed", "r.sources.verify_text_sources", "r.verify_loaded", "catalog_rows",
                    "r.completion.latest.require_latest_terminal", "audited.pair", "check_sdk_controls", "native_io.final_inventory"]
        required.append("authenticated_sdk_trace")
        for name in required:
            self.assertTrue(any(ast.unparse(node.func) == name and node.lineno < raw.lineno for node in calls), name)
        self.assertEqual([ast.unparse(node.func) for node in calls].count("reader.compare_all"), 1)
        self.assertTrue(any(ast.unparse(node.func) == "audited.pair" and node.lineno > raw.lineno for node in calls))
        self.assertNotIn("verify_saved_driver", [ast.unparse(node.func) for node in calls])
        self.assertNotIn("sdk_results_core.expected_trace_contract", [ast.unparse(node.func) for node in calls])
        self.assertEqual(hashlib.sha256((Q / "minicpm5_layer1_saved_operator_comparison_v1.py").read_bytes()).hexdigest(),
                         "8004bec5519ac732bc1a547d02e1a39ca0e63d9f5bdada7b06acece4446805ae")
        self.assertEqual(hashlib.sha256((Q / "minicpm5_layer1_saved_operator_reader_v1.py").read_bytes()).hexdigest(),
                         "53819b175232cf95e09a01d21dd06a18f3683cc8db559db9838cc2d0bd103ad0")

    def test_08_mocked_entry_retains_and_releases_original_lock_on_failure(self):
        events = []
        self.ns["sys"] = NS(dont_write_bytecode=True, executable="/synthetic/native/python")
        self.ns["r"].life.PYTHON = "/synthetic/native/python"
        self.ns["os"] = NS(sched_setaffinity=lambda *a: events.append("affinity"), nice=lambda *a: events.append("nice"))
        self.ns["resource"] = NS(RLIMIT_AS=1, RLIMIT_CPU=2, RLIMIT_FSIZE=3, setrlimit=lambda *a: events.append(("limit", a)))
        self.ns["signal"] = NS(SIGALRM=1, signal=lambda *a: None, alarm=lambda n: events.append(("alarm", n)))
        self.ns["r"].p = NS(user_bus=lambda: events.append("bus"))
        self.ns["r"].LOCK_POLICY = {"original": True}
        self.ns["r"].Lease = lambda admission: NS(admission=admission)
        self.ns["audited"].support = NS(acquire=lambda: (99, {"pid": 123}, {"passed": True}),
            release_after_physical_drain=lambda fd, proof: events.append(("release", fd, proof)))
        def fail(*args): events.append("comparison"); raise RuntimeError("source-test stop before actual work")
        self.ns["compare"] = fail
        with self.assertRaisesRegex(RuntimeError, "source-test stop"):
            self.ns["run"](ref("spec"), ref("review"))
        self.assertEqual(events[-1][0:2], ("release", 99))
        self.assertTrue(events[-1][2]["no_comparison_children_or_services_started"])
        self.assertEqual([e[1][1] for e in events if type(e) is tuple and e[0] == "limit"],
                         [(2 * 2**30, 2 * 2**30), (600, 605), (2 * 2**20, 2 * 2**20)])

    def test_09_six_full_descriptor_report_budgets_metadata_only(self):
        plan = saved.prepare(*metadata_fixture()); sizes = []
        # Full metadata-only comparison structure with44 domains/frame and8
        # examples/domain. No tensor fixture, metrics calculation or raw reader.
        stat = dict(bytes=99999999, words=99999999, sha256="f" * 64, dtype="torch.bfloat16", finite_words=99999999,
                    zero_words=0, negative_zero_words=0, minimum=-3.4028234663852886e38, maximum=3.4028234663852886e38)
        example = dict(packed_index=99999999, native_bits=4294967295, sdk_bits=4294967295,
                       native_value=-3.4028234663852886e38, sdk_value=3.4028234663852886e38)
        metric = dict(words=99999999, different_words=99999999, different_bits=99999999, all_bits_equal=False,
            maximum_absolute_error=1.2345678901234567e99, squared_error_sum=1.2345678901234567e99,
            native_squared_norm=1.2345678901234567e99, relative_l2=1.2345678901234567e99, native_norm_zero=False,
            rmse=1.2345678901234567e99, native_sha256="f" * 64, sdk_sha256="e" * 64, example_cap=8,
            examples_are_not_coverage=True, examples=[copy.deepcopy(example) for _ in range(8)])
        for call in plan["calls"]:
            inventories = {side: {key: [dict(label=row["label"], path=row["path"], shape=row["spec"]["shape"], **stat) for row in rows]
                                  for key, rows in call[side].items()} for side in ("native", "sdk")}
            frames = []
            for case in [row for row in plan["contract"]["frames"] if row["call"] == call["call"]]:
                comparisons = [dict(boundary="attention_output_after_reduce", rank=i // 22, frame_index=case["frame"]["index"],
                                   sdk_physical_row=case["sdk_physical_row"], dtype="torch.float32", native_bf16_exact_bit_embedding=False, **metric)
                               for i in range(44)]
                domains = [dict(rank=i // 11, frame_index=case["frame"]["index"], label="o_proj_local_before_reduce",
                    domain="SDK-only intermediate; no saved native one-to-one counterpart", native_counterpart=False, **stat) for i in range(22)]
                frames.append(dict(frame=case["frame"], call=call["call"], position=case["position"], bucket=case["bucket"],
                    comparisons=comparisons, sdk_only_domains=domains, first_differing_saved_boundary_by_rank={"0": "q_projection", "1": "q_projection"},
                    causal_operator_proven=False))
            report = dict(schema="minicpm5_saved_operator_call_comparison_v1", contract_sha256="f" * 64, call=call["call"],
                complete_input_comparison=True, capture_files=214, capture_bytes=23920808, full_capture_inventory=inventories,
                frames=frames, no_new_attention_or_tensor_arithmetic_executed=True, cause_proven=False, runtime_admitted=False,
                actual_process_exit_verified=False, full_model_qualified=False, new_speedup_claim=False)
            output = dict(schema="minicpm5_saved_operator_serial_call_report_v1", comparison_spec=ref("spec"), comparison_review=ref("review"),
                descriptor_plan_sha256="f" * 64, call=call["call"], complete_call_capture_descriptors=call, comparison=report,
                actual_comparison_tool_exit_verified=False, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
            sizes.append(len(self.ns["encoded"](output)))
            self.assertLessEqual(sizes[-1], LIMITS["output_file_bytes"])
        type(self).measures = dict(six_metadata_only_envelope_bytes=sizes, maximum_envelope_bytes=max(sizes),
                                  descriptor_header_bytes=len(self.ns["encoded"]({k: v for k, v in plan.items() if k != "calls"})),
                                  actual_tensor_bytes_read=0, full_raw_fixture_constructed=False, lifecycle_operations_all_mocked=True)

    def test_10_mixed_interpreter_sdk_contract_requires_real_completed_evidence(self):
        inputs = metadata_fixture(); ranks = copy.deepcopy(inputs[4]); sdk_source = b"synthetic SDK source, never executed"
        verifier_source = b"synthetic original verifier, never executed"
        sdk_path = "/synthetic/sdk-source.py"; verifier_path = self.ns["sdk_results_core"].__file__
        source_ref = dict(path=sdk_path, sha256=hashlib.sha256(sdk_source).hexdigest())
        self.ns["reader"] = NS(comparison=saved.comparison, sdk_trace=NS(SOURCE_PATH=sdk_path, SOURCE_SHA=source_ref["sha256"],
            validate_sdk_capture_summary=saved.sdk_trace.validate_sdk_capture_summary))
        contract = dict(route="sdk_decode", source=source_ref, function="compute_for_token_gen",
            line_metadata=dict(first_line=10, co_linetable_sha256="d" * 64, co_lnotab_sha256="e" * 64),
            points=[dict(role=role, variable=variable, assignment_line=11 + index * 10, event_line=12 + index * 10)
                    for index, (role, variable) in enumerate((("qk0", "prior_scores"), ("qk1", "active_scores")))])
        for result in ranks:
            for row in result["records"]:
                full = row["result"]["observations"][-1]; summary = full["capture"]["summary"]
                summary["trace_contract"] = copy.deepcopy(contract)
                summary["scale_provenance"] = copy.deepcopy(summary["scale_provenance"])
                for point, operation in zip(contract["points"], summary["scale_provenance"]):
                    operation.update(source=source_ref, function=contract["function"], line_metadata=contract["line_metadata"],
                        assignment_line=point["assignment_line"], event_line=point["event_line"], local_name=point["variable"])
                full["trace"] = dict(contract=summary["trace_contract"], mode="full_trace", calls=1, other_phase_calls=0,
                    roles=["qk0", "qk1"], observed_scale_events=2, noop_scale_events=0, aten_dispatch_used=False)
        refs = [ref("original-sdk-rank0"), ref("original-sdk-rank1")]
        full_proof = dict(schema="minicpm5_layer1_operator_sdk_pair_verification_v1", complete=True, ranks=2, unique_checkpoints=24,
            physical_attention_invocations=60, raw_capture_files=564, raw_control_files=60, raw_bytes=63876480,
            every_raw_payload_reread=True, all_endpoint_controls_full_bytes_exact=True, both_rank_reduced_outputs_raw_exact=True,
            full_model_qualified=False, new_speedup_claim=False)
        verifier_sha = hashlib.sha256(verifier_source).hexdigest()
        proof = dict(schema="minicpm5_operator_sdk_failed_run_recovery_audit_v1", audit_complete=True, original_exit_code=1,
            original_runtime_successful=False, result=full_proof, every_raw_byte_reread_after_all_model_processes_exited=True,
            all_endpoint_controls_full_bytes_exact=True, all_capture_journals_verified=True, sdk_rope_payload_independently_reread=True,
            loaded_source_closure=dict(complete=True, source_sha256={verifier_path: verifier_sha}),
            original_model_outputs={"rank%d_result" % rank: value for rank, value in enumerate(refs)})
        binding = dict(actual_exit=ref("completed-sdk-audit-exit"), recorder_terminal=ref("completed-sdk-recorder-exit"))
        recovered = dict(binding=binding, metadata_complete=True, runtime_admitted=False, original_model_exit_code=1, recovery=proof,
            actual_exit=dict(raw_proof=full_proof, result=ref("sdk-recovery-result")),
            source_sha256={verifier_path: verifier_sha, sdk_path: source_ref["sha256"]},
            authenticated_metadata={value["path"]: value for value in refs})
        validate = extract(ast.parse((Q / "minicpm5_layer1_operator_sdk_failed_recovery_core_v1.py").read_text()),
                           ["validate_proof"], {"require": require})["validate_proof"]
        self.ns["r"].completion = NS(BINDING=binding, core=NS(validate_proof=validate))
        originals = {value["path"]: copy.deepcopy(result) for value, result in zip(refs, ranks)}
        read = lambda value: copy.deepcopy(originals[value["path"]])
        source = lambda path, sha: {sdk_path: sdk_source, verifier_path: verifier_source}[path]
        selected, evidence = self.ns["authenticated_sdk_trace"](recovered, ranks, refs, read, source, lambda: None)
        self.assertEqual(selected, contract); self.assertEqual(evidence["exact_saved_contract"], contract)
        self.assertEqual(len(evidence["rows"]), 12); self.assertFalse(evidence["source_recompiled_in_native_interpreter"])
        for fault in ("forgery", "mismatch", "line_drop", "missing_audit", "missing_verifier", "wrong_rank_ref", "incomplete_raw_proof"):
            rec = copy.deepcopy(recovered); values = copy.deepcopy(ranks); authenticated = read
            if fault in ("forgery", "mismatch"):
                values[1]["records"][-1]["result"]["observations"][-1]["capture"]["summary"]["trace_contract"]["line_metadata"]["co_linetable_sha256"] = "f" * 64
                if fault == "mismatch":
                    summary = values[1]["records"][-1]["result"]["observations"][-1]["capture"]["summary"]
                    for operation in summary["scale_provenance"]: operation["line_metadata"] = summary["trace_contract"]["line_metadata"]
                    altered = {value["path"]: result for value, result in zip(refs, values)}
                    authenticated = lambda value: copy.deepcopy(altered[value["path"]])
            elif fault == "line_drop":
                del values[0]["records"][0]["result"]["observations"][-1]["capture"]["summary"]["trace_contract"]["line_metadata"]["co_lnotab_sha256"]
                altered = {value["path"]: result for value, result in zip(refs, values)}
                authenticated = lambda value: copy.deepcopy(altered[value["path"]])
            elif fault == "missing_audit": rec["recovery"]["audit_complete"] = False
            elif fault == "missing_verifier": rec["recovery"]["loaded_source_closure"]["source_sha256"].clear()
            elif fault == "wrong_rank_ref": rec["recovery"]["original_model_outputs"]["rank0_result"] = ref("forged")
            else: rec["recovery"]["result"]["raw_capture_files"] = 1
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                self.ns["authenticated_sdk_trace"](rec, values, refs, authenticated, source, lambda: None)

    def test_11_original_catalog_accepts_failed_auditor_without_success_receipt(self):
        # Execute the unchanged original pure metadata catalog validator. The
        # failed terminal deliberately has no result/success receipt to load.
        binding = {"original_sdk": "synthetic"}; cat_ref = ref("complete-catalog"); handoff = ref("idle")
        failed_ref = ref("failed-auditor"); success_ref = ref("successful-retry")
        owner = dict(pid=123, start_ticks="456")
        tool = dict(kind="tool_session", session_id=1, terminal_chunk="abcdef", exit_code=1)
        failed = dict(schema="minicpm5_native_independent_audit_failed_actual_exit_v1", actual_tool=tool, result_absent=True)
        pin = str(Q / "audit_minicpm5_layer1_operator_native_result_v4.py")
        failed_row = dict(terminal=failed_ref, tool=tool, owners=[owner], units=[], source_sha256={pin: "a" * 64})
        good_row = dict(terminal=success_ref, tool=dict(tool, session_id=2, exit_code=0), owners=[dict(owner, pid=124)], units=[], source_sha256={pin: "a" * 64})
        catalog = dict(schema="minicpm5_operator_later_serial_metadata_catalog_v1", reviewer="/root", passed=True, blockers=[],
            completed_sdk=binding, current_handoff=handoff, all_later_serial_runs_included=True, runs=[failed_row, good_row])
        values = {cat_ref["path"]: catalog, handoff["path"]: dict(live_tool_sessions=[], active_heavy_units=[]),
                  failed_ref["path"]: failed, success_ref["path"]: {"schema": "synthetic-successful-terminal"}}
        loads = []
        def read(value): loads.append(value["path"]); return values[value["path"]]
        ns = dict(sources=NS(reference=check_ref, metadata=NS(pin_shape=lambda pins: None)), require=require, copy=copy, re=re,
                  MAX_SERIAL_RUNS=256, sdk=NS(BINDING=binding), physical_evidence=NS(unit_name=lambda unit: unit))
        validate = extract(ast.parse((Q / "minicpm5_layer1_operator_sdk_runtime_completion_v4.py").read_text()), ["serial_catalog"], ns)["serial_catalog"]
        checked = validate(dict(serial_predecessors=cat_ref, source_sha256={pin: "a" * 64}), read, lambda: None)
        self.assertEqual(checked["runs"][0], failed_row)
        self.assertEqual(checked["runs"][0]["tool"]["exit_code"], 1)
        self.assertEqual(set(loads), set(values))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WrapperChecks))
    if hasattr(WrapperChecks, "measures"): print(json.dumps(WrapperChecks.measures, sort_keys=True))
    raise SystemExit(not result.wasSuccessful())
