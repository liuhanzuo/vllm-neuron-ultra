"""Metadata gate checks using pinned original validator bodies, stdlib only.

Passing these checks is not a raw-tensor reproduction receipt or a new run.
The root must authenticate producers and separately compare all203 raw vectors.
"""
import ast
import collections
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re

from qwen35_sampler_pair_protocol_v1 import BASE, BUCKETS, PREFIXES, ROOT, expected_case, read_small, require

CRITERIA = {"cosine_min": .995, "kl_max_nats": .02, "max_abs_error_max": 2.,
            "argmax_match_fraction_min": .9, "all_reference_margins_gt_0.5_must_match": True}


def original_validators():
    record = read_small(BASE / "qualified-baseline-record-v1.json")
    raw = (BASE / "completion-metadata-audit-v1.json").read_bytes()
    require(len(raw) < 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == record["evidence_sha256"], "baseline audit changed")
    pins = json.loads(raw)["evidence_sha256"]
    namespace = {"ROOT": ROOT, "PREFIXES": list(PREFIXES), "json": json, "math": math,
                 "collections": collections, "Decimal": Decimal, "InvalidOperation": InvalidOperation,
                 "re": re, "NUM": r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"}
    required = {
        "qwen35_split_full_protocol_v4.py": ("strict_rows", "validate_states", "expected_long_frames"),
        "qwen35_state_reset_full_protocol_v1.py": ("check_state_copy",),
        "run_qwen35_cold_route_v3.py": ("check_result",),
        "evaluate_quality.py": ("decimal", "grade"),
    }
    for name, functions in required.items():
        path = ROOT / name
        require(path.stat().st_size < 300 * 1024, "source bound exceeded")
        source = path.read_bytes()
        require(hashlib.sha256(source).hexdigest() == pins[name], "frozen validator source changed: " + name)
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in functions]
        require({n.name for n in nodes} == set(functions), "missing frozen validator body")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def check_model_metadata(report, phase, validators, case):
    require(__debug__, "original validator assertions must remain enabled")
    if phase == "state_copy":
        validators["check_state_copy"](report)
    elif phase == "numeric":
        require(report["passed"] and report["criteria_set_before_measurement"] == CRITERIA, "numeric gate failed")
        require(report["model"] == case["model_path"] and report["artifacts"] == case["artifact"], "numeric graph identity changed")
        validators["strict_rows"](report["rows"], 32)
        old = read_small(BASE / "real-logits-validation.json")
        identity = lambda rows: [{k: r[k] for k in ("fixture", "phase", "slot", "step")} for r in rows]
        require(identity(report["rows"]) == identity(old["rows"]), "original numeric sequence changed")
    elif phase == "cold":
        validators["check_result"](report)
        require(report["case"] == case and report["criteria"] == CRITERIA, "cold case changed")
        require(report["all_outputs_finite"] and report["numerical_gate_passed"] and report["all_duplicate_slots_bitwise_equal"], "cold numerical/duplicate gate failed")
        rows = [row for ex in report["experiments"] for row in ex["rows"]]
        validators["strict_rows"]([r["metrics"] for r in rows], 84)
        for item in report["experiments"]:
            validators["validate_states"](item["initial_raw_zero_states"], zero=True)
            validators["validate_states"](item["state_after_two_decode_steps"])
            require(all(r["valid"] and r["elements"] == 248320 and r["nan_elements"] == r["inf_elements"] == 0 for r in item["rows"]), "cold vectors incomplete")
        receipts = report["state_reset_receipts"]
        require(len(receipts) == 7 and all(x["complete"] and x["states_written"] == 96 and x["bindings_unchanged"]
                and not x["graph_initialization_called"] for x in receipts), "cold reset contract changed")
        require(all(x["bindings"] == receipts[0]["bindings"] for x in receipts), "cold binding identity changed")
        require(all(x["states_nonzero_before_clear"] > 0 for x in receipts[1:]), "dirty-state check missing")
        reproduction = report["prior_partial_reproduction"]
        require(reproduction["passed"] and reproduction["original_partial_vectors_reproduced_exactly"] == 72
                and reproduction["state_writes"] == 672 and reproduction["same96_tensor_bindings_across_all_seven_clears"], "prior72/raw-zero proof missing")
        routes = {(r["route"]["tag"], r["route"]["index"]) for r in rows}
        require(routes == {(tag, i) for tag in ("context_encoding_model", "token_generation_model") for i in range(6)}, "missing graph route")
        require(report["graph_routes"]["complete"] and report["graph_routes"]["buckets"] == list(BUCKETS)
                and len(report["graph_routes"]["graphs"]) == 12, "graph route metadata incomplete")
    elif phase == "long":
        require(report["passed"] and report["neuron_vs_full_causal_oracle_passed"] and report["reference_incremental_crosscheck_passed"], "long oracle gate failed")
        require(report["case"] == case and report["criteria"] == CRITERIA and report["teacher_forced_decode_steps"] == 1541, "long protocol changed")
        require([{k: r[k] for k in ("fixture", "position", "slot", "phase")} for r in report["rows"]]
                == validators["expected_long_frames"](), "long/reuse/partial/resume sequence changed")
        validators["strict_rows"](report["rows"], 87)
    else:
        raise ValueError("unknown full model gate")
    return {"metadata_gate_passed": True, "phase": phase, "raw_tensor_reproduction_checked": False,
            "fresh_execution_proven": False, "full_model_qualified": False}


def check_quality_pair(reference, candidate, historical, suite, suite_sha256, validators, case):
    require(__debug__, "original validator assertions must remain enabled")
    fixtures = {r["id"]: r for r in suite["cases"]}
    require(len(fixtures) == 96, "quality suite incomplete")
    maps = []
    for report, report_case in ((historical, expected_case()), (reference, case), (candidate, case)):
        require(report["case"] == report_case and report["technical_complete"]
                and report["suite_sha256"] == suite_sha256 and report["protocol"] == suite["protocol"], "quality protocol mismatch")
        require(report["total"] == report["requested_cases"] == len(report["rows"]) == 96, "quality rows missing")
        rows = {r["id"]: r for r in report["rows"]}
        require(rows.keys() == fixtures.keys(), "quality identity mismatch")
        for name, row in rows.items():
            item = fixtures[name]
            grade = validators["grade"](item, row["text"])
            require(all(row[k] == value for k, value in grade.items()), "saved quality grading mismatch")
            require(row["expected"] == item["expected"] and row["group"] == item["group"], "quality fixture changed")
            require(row["usage"]["prompt_tokens"] == len(row["prompt_ids"]) and 0 < row["usage"]["completion_tokens"] <= item["max_tokens"], "quality token accounting changed")
            require(len(row["prompt_ids"]) + item["max_tokens"] <= 4096, "quality context overrun")
        require(sum(row["correct"] for row in rows.values()) == report["correct"] == 69, "quality baseline no longer reproduced")
        maps.append(rows)
    for name in fixtures:
        require(maps[0][name]["prompt_ids"] == maps[1][name]["prompt_ids"] == maps[2][name]["prompt_ids"], "quality prompt changed/truncated")
        require(maps[0][name]["text"] == maps[1][name]["text"] == maps[2][name]["text"], "sampler-only arm changed greedy quality text")
    return {"metadata_gate_passed": True, "quality_questions": 96, "correct": 69,
            "all_texts_equal": 96, "lost_correct_cases": [], "fresh_execution_proven": False,
            "full_model_qualified": False}
