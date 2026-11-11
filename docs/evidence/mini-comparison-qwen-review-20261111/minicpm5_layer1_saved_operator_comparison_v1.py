"""Pure complete saved-operator comparison; no file, native or model IO.

The future root-owned serial reader must authenticate actual terminal/audit
chains and every raw descriptor before supplying bytes. This core neither
authorizes reads nor turns an in-memory result into an executed experiment.
Payloads are canonical packed logical bytes. Original stride/offset metadata
belongs to the existing source/audit identity checks, never raw addressing.
"""
import hashlib
import json
import math
import struct

from research import minicpm5_layer1_operator_native_manifest_v1 as native
from research import minicpm5_layer1_operator_sdk_manifest_v1 as sdk

require = native.require
WIDTHS = {"torch.bfloat16": 2, "torch.float32": 4, "torch.int64": 8, "torch.bool": 1}
CALLS = (4, 5, 6, 11, 12, 13)
INDICES = tuple(range(4, 16)) + tuple(range(20, 32))
MAX_REPORT_BYTES = 2 * 2**20
MAX_CAPTURE_BYTES = 128 * 2**20
MAX_CALL_BYTES = 32 * 2**20
MAX_EXAMPLES = 8
# Canonical JSON digest of the original r3 manifest, including fixture/slot,
# source references, lengths, provenance and controls. This pins the intended
# input contract; it does not authenticate later output or authorize raw IO.
NATIVE_MANIFEST_SHA256 = "3671f05b8d8cb29e47a03a7efc26caf19e12937454f351b0c5dd3f08baa4ddc2"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def contract(native_manifest, sdk_manifest):
    """Bind complete source-derived geometry, including all original frames."""
    require(digest(native_manifest) == NATIVE_MANIFEST_SHA256, "exact original native manifest identity required")
    require(sdk_manifest == sdk.manifest(), "exact complete SDK manifest required")
    require(native_manifest["schema"] == native.SCHEMA and native_manifest["layer"] == 1
            and native_manifest["checkpoints"] == 24 and native_manifest["direct_attention_calls"] == 72
            and native_manifest["arms"] == list(native.ARMS), "complete original native manifest required")
    frames = native_manifest["frames"]
    require([r["frame"]["index"] for r in frames] == list(INDICES), "all24 exact native frame identities")
    rows = []
    for call, bucket in zip(CALLS, sdk.BUCKETS):
        group = [r for r in frames if r["call"] == call]
        require(len(group) == 4 and {r["frame"]["slot"] for r in group} == set(range(4)),
                "one original frame per physical slot")
        require({r["frame"]["step"] for r in group} == {CALLS.index(call) % 3 + 1}, "original decode step mapping")
        for case in group:
            length = case["cache_length"]
            require(type(length) is int and 2 <= length <= bucket
                    and case["logical_position"] == length - 1
                    and case["events"] == native.event_manifest(length), "complete full native length/events")
            require(case["frame"]["index"] in INDICES and case["frame"]["fixture"] in range(8), "original fixture")
            rows.append({"frame": case["frame"], "call": call, "bucket": bucket,
                         "position": length - 1, "length": length,
                         "sdk_physical_row": case["frame"]["slot"],
                         "q_head_ids": [list(range(8)), list(range(8, 16))], "kv_head_ids": [[0], [1]],
                         "native_events": case["events"]})
    require(len(rows) == 24 and {r["frame"]["index"] for r in rows} == set(INDICES), "all native frames consumed")
    native_bytes = sum(sum(v["bytes"] for e in r["native_events"] for v in native.leaves(e["value"])) for r in rows)
    sdk_bytes = sdk_manifest["total_capture_raw_bytes"]
    peak = max(sum(sum(v["bytes"] for e in r["native_events"] for v in native.leaves(e["value"]))
                   for r in rows if r["call"] == c) + sum(x["capture_raw_bytes"] for x in sdk_manifest["records"] if x["call"] == c)
               for c in CALLS)
    require(native_bytes == native_manifest["observer_raw_bytes"] and native_bytes + sdk_bytes <= MAX_CAPTURE_BYTES
            and peak <= MAX_CALL_BYTES, "full capture and one-call byte budgets")
    return {"schema": "minicpm5_saved_operator_comparison_contract_v1", "native_manifest_sha256": digest(native_manifest),
            "sdk_manifest_sha256": digest(sdk_manifest), "frames": rows, "calls": list(CALLS),
            "native_capture_files": 24 * 30, "sdk_capture_files": 12 * 47,
            "native_capture_bytes": native_bytes, "sdk_capture_bytes": sdk_bytes,
            "total_capture_bytes": native_bytes + sdk_bytes, "maximum_call_input_bytes": peak,
            "raw_addressing": "packed_logical_shape_only", "source_stride_offset_used_for_raw_addressing": False,
            "report_cap_bytes": MAX_REPORT_BYTES, "runtime_admitted": False, "full_model_qualified": False,
            "new_speedup_claim": False}


def native_specs(events):
    result = {}
    def visit(label, value, path=()):
        if type(value) is list:
            for index, child in enumerate(value): visit(label, child, path + (index,))
        elif "shape" in value:
            result[label, path] = value
    for event in events: visit(event["label"], event["value"])
    return result


def sdk_specs(bucket):
    return {(e["label"], tuple(t["path"])): t for e in sdk.events(bucket) for t in e["tensors"]}


def floating(raw, dtype):
    if dtype == "torch.bfloat16":
        for (word,) in struct.iter_unpack("<H", raw):
            yield struct.unpack("<f", struct.pack("<I", word << 16))[0]
    else:
        require(dtype == "torch.float32", "only declared floating dtype")
        for (value,) in struct.iter_unpack("<f", raw): yield value


def full_stats(raw, dtype, guard):
    """Full-domain finite/zero/range counts; no sampling and no normalization."""
    width = WIDTHS[dtype]
    require(type(raw) is bytes and len(raw) % width == 0, "whole packed words")
    words = len(raw) // width
    result = {"bytes": len(raw), "words": words, "sha256": sha(raw), "dtype": dtype}
    if dtype in ("torch.bfloat16", "torch.float32"):
        zeros = negative_zeros = 0
        lo = hi = None
        for index, value in enumerate(floating(raw, dtype)):
            if index % 4096 == 0: guard()
            require(math.isfinite(value), "nonfinite observed floating value; no NaN/Inf equivalence")
            zeros += int(value == 0.0)
            negative_zeros += int(value == 0.0 and math.copysign(1.0, value) < 0)
            lo = value if lo is None else min(lo, value)
            hi = value if hi is None else max(hi, value)
        result.update(finite_words=words, zero_words=zeros, negative_zero_words=negative_zeros, minimum=lo, maximum=hi)
    elif dtype == "torch.bool":
        require(set(raw) <= {0, 1}, "canonical whole boolean bytes")
        result.update(true_words=raw.count(1), false_words=raw.count(0))
    else:
        require(dtype == "torch.int64", "declared integer dtype")
    guard()
    return result


def validate_values(values, specs, guard):
    require(type(values) is dict and set(values) == set(specs), "every exact capture leaf, no missing or extra tensor")
    summaries = []
    for key, spec in specs.items():
        guard(); raw = values[key]
        require(type(raw) is bytes and len(raw) == spec["bytes"] == math.prod(spec["shape"]) * WIDTHS[spec["dtype"]],
                "full canonical tensor byte size/shape/dtype")
        summaries.append({"label": key[0], "path": list(key[1]), "shape": spec["shape"],
                          **full_stats(raw, spec["dtype"], guard)})
    return summaries


def packed_region(raw, shape, dtype, ranges):
    """Slice all specified coordinates using packed row-major addressing only."""
    require(len(shape) == len(ranges) and len(raw) == math.prod(shape) * WIDTHS[dtype], "packed tensor geometry")
    for size, span in zip(shape, ranges):
        require(type(span) is tuple and len(span) == 2 and all(type(x) is int for x in span)
                and 0 <= span[0] <= span[1] <= size, "complete declared region inside original shape")
    width = WIDTHS[dtype]
    strides = [math.prod(shape[index + 1:]) for index in range(len(shape))]
    output = bytearray()
    def walk(axis, offset):
        start, stop = ranges[axis]
        if axis == len(shape) - 1:
            output.extend(raw[(offset + start) * width:(offset + stop) * width])
        else:
            for value in range(start, stop): walk(axis + 1, offset + value * strides[axis])
    walk(0, 0)
    require(len(output) == math.prod(b - a for a, b in ranges) * width, "entire region consumed")
    return bytes(output)


def bf16_to_fp32_bits(raw):
    """Exact embedding, including signed zero; no arithmetic or rounded cast."""
    require(type(raw) is bytes and len(raw) % 2 == 0, "whole BF16 words")
    return b"".join(b"\x00\x00" + raw[i:i + 2] for i in range(0, len(raw), 2))


def compare_words(left, right, dtype, guard):
    require(type(left) is bytes and type(right) is bytes and len(left) == len(right)
            and dtype in ("torch.bfloat16", "torch.float32") and len(left) % WIDTHS[dtype] == 0,
            "same complete comparison domain and dtype")
    width = WIDTHS[dtype]; differences = bits = 0; examples = []; square_error = reference_norm = maximum = 0.0
    for index, (x, y) in enumerate(zip(floating(left, dtype), floating(right, dtype))):
        if index % 4096 == 0: guard()
        require(math.isfinite(x) and math.isfinite(y), "nonfinite compared value rejected")
        a = int.from_bytes(left[index * width:(index + 1) * width], "little")
        b = int.from_bytes(right[index * width:(index + 1) * width], "little")
        differences += int(a != b); bits += (a ^ b).bit_count()
        if a != b and len(examples) < MAX_EXAMPLES:
            examples.append({"packed_index": index, "native_bits": a, "sdk_bits": b,
                             "native_value": x, "sdk_value": y})
        delta = y - x; maximum = max(maximum, abs(delta)); square_error += delta * delta; reference_norm += x * x
    words = len(left) // width
    guard()
    return {"words": words, "different_words": differences, "different_bits": bits,
            "all_bits_equal": differences == 0, "maximum_absolute_error": maximum,
            "squared_error_sum": square_error, "native_squared_norm": reference_norm,
            "relative_l2": math.sqrt(square_error / reference_norm) if reference_norm else (0.0 if square_error == 0 else None),
            "native_norm_zero": reference_norm == 0, "rmse": math.sqrt(square_error / words) if words else 0.0,
            "native_sha256": sha(left), "sdk_sha256": sha(right), "example_cap": MAX_EXAMPLES,
            "examples_are_not_coverage": True, "examples": examples}


def compare_frame(case, native_values, sdk_values, guard):
    """Complete native/TP2 mapping for one frame; payload maps already complete."""
    require(callable(guard), "explicit future live serial guard")
    guard(); slot, length, bucket = case["sdk_physical_row"], case["length"], case["bucket"]
    previous = length - 1; ns = native_specs(case["native_events"]); ss = sdk_specs(bucket)
    n = lambda label, path=(): native_values[label, path]
    rows = []; domains = []
    for rank in (0, 1):
        values = sdk_values[rank]
        s = lambda label, path=(): values[label, path]
        def local(label, path=(), tail=None):
            spec = ss[label, path]; shape = spec["shape"]
            spans = [(slot, slot + 1)] + [(0, size) for size in shape[1:]]
            if tail is not None: spans[-1] = tail
            return packed_region(s(label, path), shape, spec["dtype"], spans)
        def native_heads(label, head_kind, path=(), length_range=None):
            spec = ns[label, path]; shape = spec["shape"]
            count = 8 if head_kind == "q" else 1
            spans = [(0, 1), (rank * count, (rank + 1) * count)] + [(0, size) for size in shape[2:]]
            if length_range is not None: spans[-1] = length_range
            return packed_region(n(label, path), shape, spec["dtype"], spans)
        def add(name, a, b, dtype="torch.bfloat16", embedding=False):
            rows.append({"boundary": name, "rank": rank, "frame_index": case["frame"]["index"],
                         "sdk_physical_row": slot, "dtype": dtype, "native_bf16_exact_bit_embedding": embedding,
                         **compare_words(a, b, dtype, guard)})
        # Inputs/caches are prerequisites, not arithmetic root-cause claims.
        require(n("normalized_input") == local("normalized_input"), "full normalized input differs before operator comparison")
        require(n("position_ids") == n("rope_position_ids") == local("position_ids") == local("rope_position_ids")
                == local("route_inputs_sdk_decode", ("position_ids",)) == struct.pack("<q", previous),
                "exact frame/slot position differs")
        require(n("attention_mask") == n("actual_eager_mask") == bytes(length * 2), "native full valid-domain mask must be +0")
        require(local("attention_mask") == local("prior_actual_mask")
                == local("route_inputs_sdk_decode", ("attention_mask",))
                == bytes([1]) * previous + bytes(bucket - previous),
                "SDK prior mask does not cover exactly saved history and full padding")
        for index, name in enumerate(("key", "value")):
            require(s("effective_prior_kv", (index,)) == s("route_inputs_sdk_decode", ("past_key_value", index)),
                    "SDK dispatch must consume the captured full prior view, including padding")
            require(n("native_effective_" + name) == n("cache_after", (index,)),
                    "native effective cache and saved updated cache differ")
            nr = packed_region(n("cache_before", (index,)), [1, 2, previous, 128], "torch.bfloat16",
                               [(0, 1), (rank, rank + 1), (0, previous), (0, 128)])
            sr = packed_region(s("effective_prior_kv", (index,)), [4, 1, bucket, 128], "torch.bfloat16",
                               [(slot, slot + 1), (0, 1), (0, previous), (0, 128)])
            require(nr == sr, "complete official prior cache differs")
            add("prior_" + name, nr, sr)
            padding = packed_region(s("effective_prior_kv", (index,)), [4, 1, bucket, 128], "torch.bfloat16",
                                    [(slot, slot + 1), (0, 1), (previous, bucket), (0, 128)])
            domains.append({"rank": rank, "frame_index": case["frame"]["index"], "label": "prior_" + name,
                            "domain": "SDK-only prior cache padding [position,bucket)", "native_counterpart": False,
                            **full_stats(padding, "torch.bfloat16", guard)})
        for name, width in (("q_projection", 1024), ("k_projection", 128), ("v_projection", 128)):
            add(name, n(name)[rank * width * 2:(rank + 1) * width * 2], local(name))
        for name, kind in (("q_rope_input", "q"), ("k_rope_input", "kv")):
            add(name, native_heads(name, kind), local(name))
        for name in ("rope_cos", "rope_sin"):
            add(name, n(name), local(name))
        for name, kind in (("q_post_rope", "q"), ("k_post_rope", "kv"), ("v_heads", "kv")):
            add(name, native_heads(name, kind), local(name))
        for short in ("q", "k", "v"):
            before = short + "_post_rope" if short != "v" else "v_heads"
            require(local(before) == local(short + "_after_prep") == local("route_inputs_sdk_decode", (short.upper(),)),
                    "SDK preparation/dispatch changed captured operand")
        for index, name in enumerate(("k_post_rope", "v_heads")):
            require(local(name) == local("returned_current_kv", (index,)), "SDK returned active cache identity differs")
        require(n("k_post_rope") == n("cache_update_active_key"), "native active cache update key differs")
        require(n("pv_context") == n("context_before_o"), "native context flattening changed packed values")
        for index, name in enumerate(("key", "value")):
            effective = packed_region(n("native_effective_" + name), [1, 2, length, 128], "torch.bfloat16",
                                      [(0, 1), (rank, rank + 1), (0, length), (0, 128)])
            prior = packed_region(s("effective_prior_kv", (index,)), [4, 1, bucket, 128], "torch.bfloat16",
                                  [(slot, slot + 1), (0, 1), (0, previous), (0, 128)])
            add("effective_" + name + "_after_cache_update", effective, prior + local("returned_current_kv", (index,)))
        for label, dtype in (("qk0_unscaled", "torch.bfloat16"), ("qk0_scaled", "torch.bfloat16"),
                             ("softmax_input", "torch.float32"), ("probabilities_fp32", "torch.float32"),
                             ("probabilities_bf16", "torch.bfloat16")):
            if label == "softmax_input":
                require(local("prior_mask_input") == local("qk0_scaled"), "actual mask consumes recorded scaled prior scores")
                add("prior_mask_valid_output", native_heads("softmax_input", "q", length_range=(0, previous)),
                    local("prior_mask_output", tail=(0, previous)))
                domains.append({"rank": rank, "frame_index": case["frame"]["index"], "label": "prior_mask_output",
                                "domain": "SDK-only masked prior score padding [position,bucket)", "native_counterpart": False,
                                **full_stats(local("prior_mask_output", tail=(previous, bucket)), "torch.bfloat16", guard)})
            sdk_active = {"qk0_unscaled": "qk1_unscaled", "qk0_scaled": "qk1_scaled",
                          "softmax_input": "softmax_active_input", "probabilities_fp32": "probabilities_active_fp32",
                          "probabilities_bf16": "probabilities_active_bf16"}[label]
            left = native_heads(label, "q")
            embedding = label == "softmax_input"
            if embedding: left = bf16_to_fp32_bits(left)
            # Canonical matched order is head then [all prior keys, active key].
            prior = local(label, tail=(0, previous)); active = local(sdk_active)
            width = WIDTHS[dtype]; joined = b"".join(prior[h * previous * width:(h + 1) * previous * width]
                                                   + active[h * width:(h + 1) * width] for h in range(8))
            add(label, left, joined, dtype, embedding)
            padding = local(label, tail=(previous, bucket))
            stat = full_stats(padding, dtype, guard)
            if label.startswith("probabilities"):
                require(stat["zero_words"] == stat["words"], "nonzero probability in SDK-only masked padding")
            domains.append({"rank": rank, "frame_index": case["frame"]["index"], "label": label,
                            "domain": "SDK-only prior padding [position,bucket)", "native_counterpart": False, **stat})
        context = native_heads("pv_context", "q")
        add("merged_pv_context", context, local("context_before_o"))
        for label in ("pv_prior", "pv_active", "o_proj_local_before_reduce"):
            domains.append({"rank": rank, "frame_index": case["frame"]["index"], "label": label,
                            "domain": "SDK-only intermediate; no saved native one-to-one counterpart",
                            "native_counterpart": False, **full_stats(local(label), "torch.bfloat16", guard)})
        require(local("o_proj_after_reduce") == local("attention_output"), "saved reduced endpoint identity changed")
        add("attention_output_after_reduce", n("attention_output"), local("attention_output"))
    # The two replicated reduced endpoints must remain the same original result.
    for label in ("o_proj_after_reduce", "attention_output"):
        require(sdk_values[0][label, ()] == sdk_values[1][label, ()], "actual TP2 reduced endpoints differ across ranks")
    first = {str(rank): next((r["boundary"] for r in rows if r["rank"] == rank and not r["all_bits_equal"]), None)
             for rank in (0, 1)}
    return {"frame": case["frame"], "call": case["call"], "position": previous, "bucket": bucket,
            "comparisons": rows, "sdk_only_domains": domains, "first_differing_saved_boundary_by_rank": first,
            "causal_operator_proven": False}


def compare_call(spec, call, native_values, sdk_values, guard):
    """One complete call, with every raw leaf supplied; no filesystem access.

    native_values maps original frame index to its30-leaf map. sdk_values maps
    rank0/1 to each47-leaf map. Keys are (event_label, nested_tensor_path_tuple).
    The caller must preserve original descriptor metadata separately and prove
    canonical packed layout using the existing independent output auditors.
    """
    require(call in CALLS and callable(guard), "declared original call and explicit guard")
    guard(); cases = [r for r in spec["frames"] if r["call"] == call]
    require(len(cases) == 4 and set(native_values) == {r["frame"]["index"] for r in cases}
            and set(sdk_values) == {0, 1}, "all four original frames and both ranks")
    inventories = {"native": {}, "sdk": {}}
    for case in cases:
        index = case["frame"]["index"]
        inventories["native"][str(index)] = validate_values(native_values[index], native_specs(case["native_events"]), guard)
    for rank in (0, 1):
        inventories["sdk"][str(rank)] = validate_values(sdk_values[rank], sdk_specs(cases[0]["bucket"]), guard)
    count = sum(len(v) for side in inventories.values() for v in side.values())
    total = sum(row["bytes"] for side in inventories.values() for v in side.values() for row in v)
    require(count == 4 * 30 + 2 * 47 and total <= spec["maximum_call_input_bytes"] <= MAX_CALL_BYTES,
            "every214 full capture leaves within one-call input bound")
    frames = [compare_frame(case, native_values[case["frame"]["index"]], sdk_values, guard) for case in cases]
    result = {"schema": "minicpm5_saved_operator_call_comparison_v1", "contract_sha256": digest(spec), "call": call,
              "complete_input_comparison": True, "capture_files": count, "capture_bytes": total,
              "full_capture_inventory": inventories, "frames": frames,
              "no_new_attention_or_tensor_arithmetic_executed": True, "cause_proven": False,
              "runtime_admitted": False, "actual_process_exit_verified": False, "full_model_qualified": False,
              "new_speedup_claim": False}
    require(len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode()) <= MAX_REPORT_BYTES,
            "complete per-call result exceeds bound; never truncate")
    guard(); return result


def summarize(spec, calls):
    """All six complete calls only; this does not replace actual-exit auditing."""
    require(type(calls) is list and [r["call"] for r in calls] == list(CALLS), "all six original calls in order")
    require(all(r["schema"] == "minicpm5_saved_operator_call_comparison_v1"
                and r["contract_sha256"] == digest(spec) and r["complete_input_comparison"] is True for r in calls),
            "same complete compared contract")
    require(sum(r["capture_files"] for r in calls) == spec["native_capture_files"] + spec["sdk_capture_files"] == 1284
            and sum(r["capture_bytes"] for r in calls) == spec["total_capture_bytes"], "complete whole capture counts")
    frames = [f for r in calls for f in r["frames"]]
    require(len(frames) == 24 and [f["frame"] for f in frames] == [c["frame"] for c in spec["frames"]]
            and [(f["call"], f["position"], f["bucket"]) for f in frames]
            == [(c["call"], c["position"], c["bucket"]) for c in spec["frames"]], "all24 exact compared frame identities")
    return {"schema": "minicpm5_saved_operator_comparison_summary_v1", "contract_sha256": digest(spec),
            "calls": 6, "frames": 24, "capture_files": 1284, "capture_bytes": spec["total_capture_bytes"],
            "first_differing_saved_boundaries": [{"frame": f["frame"], "by_rank": f["first_differing_saved_boundary_by_rank"]} for f in frames],
            "differences_identify_observed_boundaries_only": True, "causal_operator_proven": False,
            "full_model_qualified": False, "new_speedup_claim": False, "actual_execution_and_independent_audit_still_required": True}
