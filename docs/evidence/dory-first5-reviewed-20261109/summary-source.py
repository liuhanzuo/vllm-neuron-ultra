"""Summarize finalized first-five diagnostics without changing model acceptance."""
import argparse
import hashlib
import json
from pathlib import Path


def load_ref(ref):
    path = Path(ref["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("Changed referenced evidence: " + str(path))
    return json.loads(data)


def summarize(exit_ref):
    receipt = load_ref(exit_ref)
    if not (receipt["actual_external_exit_observed"] is True and receipt["exit_code"] == 0
            and receipt["reviewer"] == "/root"):
        raise ValueError("Actual completed saved review required")
    report = load_ref(receipt["review"])
    result = load_ref(receipt["result"])
    if not (result["complete"] is True and report["complete"] is True
            and report["schema"] == "dory_first5_combined_saved_review_v1"
            and report["total_raw_output_bytes_verified"] == 191883264
            and report["original_calls"] == 6185 and report["original_marked_frames"] == 87
            and report["original_pointwise_failures"] == 7
            and report["recorded_source_checkpoint_exit_bound"] is True
            and report["checkpoint_review"]["checkpoint_bytes_read"] == 7950515472
            and report["full_model_qualified"] is False):
        raise ValueError("Complete original evidence and failed acceptance required")
    frames = report["frames"]
    labels = list(frames[0]["comparisons"])
    if len(frames) != 87 or len(labels) != 51 or len({f["event"] for f in frames}) != 87:
        raise ValueError("Exact original frame and boundary counts")
    aggregates = {}
    for label in labels:
        values = [f["comparisons"][label] for f in frames]
        if any(list(f["comparisons"]) != labels for f in frames):
            raise ValueError("Changed boundary order")
        aggregates[label] = dict(
            frames=87,
            different_frames=sum(v["bitwise_equal_words"] != v["elements"] for v in values),
            max_different_values=max(v["elements"] - v["bitwise_equal_words"] for v in values),
            max_abs=max(v["max_abs"] for v in values),
            max_relative_rms=max((v["relative_rms"] for v in values if v["relative_rms"] is not None), default=None),
            min_cosine=min((v["cosine"] for v in values if v["cosine"] is not None), default=None),
            undefined_relative_rms_frames=sum(v["relative_rms"] is None for v in values),
            undefined_cosine_frames=sum(v["cosine"] is None for v in values),
        )
    failures = []
    first_differences = {}
    for frame in frames:
        first = next((label for label in labels
                      if frame["comparisons"][label]["bitwise_equal_words"]
                      != frame["comparisons"][label]["elements"]), "all_equal")
        first_differences[first] = first_differences.get(first, 0) + 1
        if frame["original_pointwise_failed"]:
            failures.append(dict(event=frame["event"], fixture=frame["fixture"],
                position=frame["position"], original_FP32_metrics=frame["original_FP32_metrics"],
                first_observed_difference=first,
                boundaries={label:frame["comparisons"][label] for label in labels}))
    if len(failures) != 7 or aggregates["0-input"]["different_frames"] != 0:
        raise ValueError("Original seven failures and identical layer-zero inputs")
    max_frame = max(frames, key=lambda f: f["comparisons"]["4-residual_output"]["relative_rms"])
    max_layer4 = {k:max_frame[k] for k in ("event", "fixture", "position", "original_pointwise_failed", "original_FP32_metrics")}
    max_layer4["relative_rms"] = max_frame["comparisons"]["4-residual_output"]["relative_rms"]
    failed_relative = [f["boundaries"]["4-residual_output"]["relative_rms"] for f in failures]
    return dict(schema="dory_first5_reviewed_findings_v1", review_exit=exit_ref,
        saved_review=receipt["review"], original_calls=6185, marked_frames=87,
        observed_boundaries=51, original_pointwise_failures=7,
        raw_output_bytes_verified=191883264, checkpoint_bytes_verified=7950515472,
        boundary_aggregates=aggregates, first_observed_difference_counts=first_differences,
        failed_frames=failures, max_layer4_relative_rms_frame=max_layer4,
        failed_layer4_relative_rms_range=[min(failed_relative),max(failed_relative)],
        original_FP32_acceptance_passed=False,
        root_cause_established=False, full_model_qualified=False, new_speedup_claim=False,
        limitations=["First five physical layers of original CPU ReferenceAttention path only",
            "Float64 boundary statistics explain differences; original FP32 acceptance remains unchanged",
            "No same-input causal intervention at attention or residual boundaries",
            "No Neuron, production scheduler, generation, or HTTP performance qualification"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-exit", required=True)
    parser.add_argument("--review-exit-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = summarize(dict(path=args.review_exit, sha256=args.review_exit_sha256))
    with Path(args.output).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:value[k] for k in ["original_calls", "marked_frames", "observed_boundaries",
        "original_pointwise_failures", "first_observed_difference_counts", "new_speedup_claim"]}))


if __name__ == "__main__":
    main()
