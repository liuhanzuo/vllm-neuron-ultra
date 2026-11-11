"""Source-only sampler successor contract; no launcher, admission, or native import.

The frozen model case is identical for both arms. Host identity is separate and
must NEVER be written into the old artifact provenance or its source map.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "qwen35-state-reset-full-v1"
ARMS = ("baseline", "candidate")
PHASES = ("correctness_capture", "measured")
VOCAB = 248320
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
PREFIXES = (32, 128, 256, 512, 1024, 2048, 2052)
MAX_CALLS_PER_BATCH = 16
MAX_CAPTURE_BYTES_PER_ARM = 256 * 1024**2
MAX_PACKET_BYTES = 12 * 1024**2
MAX_JSON_BYTES = 1024**2
MAX_CAPTURE_ROWS_PER_ARM = MAX_CALLS_PER_BATCH * sum(range(1, 5))
MAX_PROCESSED_LOGIT_BYTES_PER_ARM = MAX_CAPTURE_ROWS_PER_ARM * VOCAB * 4

FULL_GATES = {
    "state_copy": {"bindings": 96, "fill_comparisons": 192, "zero_comparisons": 192},
    "numeric": {"full_vocabulary_frames": 32},
    "cold": {"full_vocabulary_frames": 84, "prefixes": list(PREFIXES),
             "zero_checks": 672, "duplicate_pairs": 42, "same_bindings": 96,
             "in_place_clears": 7, "original_partial_vectors_exact": 72,
             "graph_routes": 12, "buckets": list(BUCKETS)},
    "long": {"teacher_forced_steps": 1541, "full_vocabulary_frames": 87,
             "slot_reuse": True, "partial_batch": True, "resume_inactive": True,
             "original_full_causal_oracle_and_incremental_crosscheck": True},
    "full_vector_reproduction": {"frames": 203, "bitwise_equal_to_qualified_baseline": 203},
    "smoke": {"serial": 4, "concurrent": 4, "identical_prompt_outputs": 4},
    "quality": {"questions": 96, "baseline_correct": 69,
                "baseline_texts_reproduced": 96, "candidate_texts_equal": 96,
                "lost_correct_cases": 0, "concurrency": 4, "temperature": 0,
                "truncate": False, "regrade_all_saved_texts": True},
    "initial_http": {"output_tokens_per_request": 120, "rounds": 3,
                     "single_and_c4": True, "streaming_c4": True,
                     "warmup_tokens": 8, "greedy_texts_equal": True},
    "http": {"workloads": 5, "rounds": 3, "recorded_requests": 124,
             "recorded_output_tokens": 30848, "greedy_texts_equal": 57,
             "streaming_retained": True, "mixed_requests_per_round": 16,
             "mixed_concurrency": 4, "warmup_requests": 4, "warmup_tokens_each": 8,
             "original_seed_omission_preserved": True, "request_timeout_s": 240,
             "original_force_min_tokens_and_ignore_eos": True},
}

SAMPLER_GATES = {
    "mandatory_captured_calls_by_rows": {"1": 16, "4": 16},
    "optional_actual_batch_sizes": [2, 3],
    "full_vocab_distributions_not_topk_only": True,
    "distribution_support_equal": True,
    "total_variation_max": 1e-6,
    "max_probability_error_max": 1e-6,
    "own_operation_token_and_rng_reproduction": True,
    "explicit_generator_whole_call_token_and_rng_equal": True,
    "controls": ["partial_explicit_generators", "all_explicit_generators",
                 "k_absent", "k_zero", "k_above64", "non_float32",
                 "invalid_p", "processed_logits", "processed_logprobs",
                 "excessive_kth_ties", "nonfinite_max_or_kth",
                 "cdf_near_boundary", "cdf_boundary_tie"],
    "unseeded_token_sequence_equality_required": False,
    "historical_missing_penalties_or_rng_may_be_inferred": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_small(path):
    path = Path(path)
    require(path.stat().st_size <= 2 * 1024**2, "small metadata only")
    return json.loads(path.read_text())


def expected_case(record=None):
    record = read_small(BASE / "qualified-baseline-record-v1.json") if record is None else record
    require(record["model"] == "Qwen/Qwen3.5-2B", "wrong model")
    require(record["official_revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc", "revision changed")
    case = deepcopy(record["case"])
    require(case["name"] == "latest-qwen35-2b-text-bf16-cpu-tp2-b4-ctx4096-split-v4", "wrong baseline")
    require(case["tp"] == 2 and case["batch"] == 4 and case["max_model_len"] == 4096, "geometry changed")
    require(case["cpu"] is True and case["qwen35_tensor_split"] is True, "wrong graph/sampler mode")
    require(case["extra_overrides"]["context_encoding_buckets"] == list(BUCKETS), "missing bucket")
    require(not case.get("qwen35_gdn_prefill_v1") and not case.get("cpu_fast_sampler"), "unrelated optimization")
    return case


def host_identity(arm, phase, source_sha256):
    require(arm in ARMS and phase in PHASES, "unknown host arm or phase")
    require(source_sha256 and all(len(x) == 64 for x in source_sha256.values()), "missing host pins")
    value = {"schema": "qwen35_sampler_host_v1", "arm": arm, "phase": phase,
             "sampler": "original" if arm == "baseline" else "bounded_per_instance",
             "observation_enabled": phase == "correctness_capture",
             "debug_behavior": "unchanged", "source_sha256": dict(source_sha256)}
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def finite_protocol():
    """Scientific work only. This does not authorize or configure any execution."""
    return {
        "schema": "qwen35_sampler_pair_source_contract_v1", "source_only": True,
        "runtime_admission": False, "model_case_shared_unchanged": True,
        "fresh_server_per_arm_and_phase": True, "compile_permitted": False,
        "fresh_output_rule": "Call original benchmark/evaluate functions directly with new arm outputs. Never call orchestrate.run_case or reuse case-name HTTP JSON: the shared frozen case would reuse historical results.",
        "graph_reuse": "Validate frozen adapter-provenance, config, all12 graphs and model.pt separately; never rewrite fingerprints.",
        "weights": "original BF16", "recurrent_state": "FP32", "context": 4096,
        "arms": list(ARMS), "full_gates_each_arm": deepcopy(FULL_GATES),
        "sampler_gates": deepcopy(SAMPLER_GATES),
        "extra_diagnostic_requests": {"waves": [1, 4], "prompt_tokens": 128,
                                      "output_tokens_each": 32, "requests_per_arm": 5,
                                      "output_tokens_per_arm": 160, "temperature": .7,
                                      "top_p": .9, "top_k": 40, "seed_added": False,
                                      "retry_if_B1_or_B4_quota_missing": False,
                                      "original_full_http_replaced": False},
        "capture": {"selection": "first16 actual top-k/top-p calls for each row count1..4",
                    "max_calls_per_arm": 64, "max_rows_per_arm": MAX_CAPTURE_ROWS_PER_ARM,
                    "max_processed_logit_bytes_per_arm": MAX_PROCESSED_LOGIT_BYTES_PER_ARM,
                    "max_total_packet_bytes_per_arm": MAX_CAPTURE_BYTES_PER_ARM,
                    "max_single_packet_bytes": MAX_PACKET_BYTES, "max_packet_json_bytes": MAX_JSON_BYTES,
                    "cap_includes": "packet header, metadata, all tensors, logits, RNG, outputs",
                    "cap_overflow": "fail diagnostic; no silent truncation or fabricated completion",
                    "max_two_arm_packet_bytes": 2 * MAX_CAPTURE_BYTES_PER_ARM,
                    "source": "fresh actual processed logits/metadata/request-slot order/RNG before and after",
                    "timing_eligible": False},
        "remaining_resource_admission": "Root must account numerical payloads, HTTP, logs, transient peaks and writer bookkeeping in addition to packet caps.",
        "order": ["exclusive runtime admission by root", "immutable artifact reuse validation",
                  "full model gates on both arms", "separate bounded correctness captures",
                  "native distribution/own-operation/negative-control validation", "quality96 on both arms",
                  "unobserved matched initial HTTP and full5x3 HTTP on both arms", "independent final audit"],
        "timing_rules": "Identical host/debug/thread settings. No capture hooks in either measured arm. Ratios only after all gates. Report every workload; no universal multiplier.",
        "historical_limits": "The old GDN candidate remains disqualified. This does not recover absent historical penalties/RNG or extend model/multimodal coverage.",
    }


def check_capture_accounting(receipt):
    require(receipt["phase"] == "correctness_capture", "capture is not a timed run")
    counts = receipt["captured_calls_by_rows"]
    require(set(counts) == {"1", "2", "3", "4"}, "invalid batch accounting")
    require(all(type(n) is int and 0 <= n <= 16 for n in counts.values()), "capture count exceeds cap")
    require(counts["1"] == counts["4"] == 16, "actual B1/B4 coverage incomplete")
    require(receipt["packets"] == sum(counts.values()), "packet accounting mismatch")
    require(receipt["captured_rows"] == sum(int(k) * n for k, n in counts.items()), "row accounting mismatch")
    require(receipt["processed_logit_bytes"] == receipt["captured_rows"] * VOCAB * 4, "logits incomplete")
    require(receipt["processed_logit_bytes"] < receipt["total_packet_bytes"] <= MAX_CAPTURE_BYTES_PER_ARM,
            "total packet byte accounting invalid")
    require(receipt["timing_eligible"] is False, "capture overhead must not enter speedup")


def check_host_pair(left, right):
    require(left["arm"] == "baseline" and right["arm"] == "candidate", "arm identity mismatch")
    for receipt in (left, right):
        require(receipt["phase"] == "measured" and receipt["observation_enabled"] is False, "timing capture contamination")
        require(receipt["debug_behavior"] == "unchanged", "DEBUG changed")
    for key in ("graph_reuse_receipt_sha256", "effective_host_settings", "frozen_model_case"):
        require(left[key] == right[key], "unmatched setting: " + key)
    require(left["frozen_model_case"] == expected_case(), "frozen case modified")


def compare_full_http(reference, candidate, case, expected_inputs, host_left, host_right):
    """Use the original complete workload validator; no GDN graph import."""
    from qwen35_gdn_serving_checks_v2 import compare
    check_host_pair(host_left, host_right)
    require(case == expected_case(), "model case changed")
    result = compare(reference, candidate, case, case, expected_inputs)
    require(result["identical_greedy_texts"] == 57, "sampler-only change altered greedy outputs")
    return result
