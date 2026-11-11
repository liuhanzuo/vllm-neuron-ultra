"""Future bounded CPU replay; import is stdlib only, no command-line entry.

This validates fresh captures, never reconstructs missing historical metadata.
The caller must admit native execution separately and supply an untouched
reference_factory(logprobs_mode) using the exact pinned vLLM environment.
"""
import hashlib
import math
import sys

from qwen35_sampler_native_capture_v1 import parse_packet
from qwen35_sampler_pair_protocol_v1 import MAX_PACKET_BYTES, SAMPLER_GATES, VOCAB, require


class Decoder:
    def __init__(self, torch, raw):
        self.torch, self.raw = torch, raw
        self.tensors = {}

    def tensor(self, spec):
        torch = self.torch
        require(set(spec) == {"tensor", "dtype", "shape", "original_stride", "original_device", "offset", "bytes", "sha256"},
                "unexpected tensor descriptor")
        require(type(spec["tensor"]) is int and spec["tensor"] not in self.tensors, "duplicate tensor descriptor")
        dtypes = {str(getattr(torch, name)): getattr(torch, name)
                  for name in ("float32", "float64", "bfloat16", "float16", "int64", "int32", "bool", "uint8")}
        require(spec["dtype"] in dtypes and spec["original_device"] == "cpu", "unexpected captured tensor")
        shape, stride = spec["shape"], spec["original_stride"]
        require(len(shape) == len(stride) <= 4 and all(type(n) is int and n >= 0 for n in shape + stride), "bad shape/stride")
        offset, length = spec["offset"], spec["bytes"]
        require(type(offset) is int and type(length) is int and 0 <= offset <= offset + length <= len(self.raw), "bad tensor range")
        raw = self.raw[offset:offset + length]
        require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "captured tensor hash mismatch")
        dtype = dtypes[spec["dtype"]]
        itemsize = torch.empty((), dtype=dtype).element_size()
        require(length == math.prod(shape) * itemsize, "captured tensor incomplete")
        allocation = (1 + sum((n - 1) * s for n, s in zip(shape, stride))) * itemsize if math.prod(shape) else 0
        require(0 <= allocation <= MAX_PACKET_BYTES, "captured stride allocation exceeds cap")
        value = torch.empty_strided(shape, stride, dtype=dtype, device="cpu")
        if length:
            canonical = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
            value.copy_(canonical)
        self.tensors[spec["tensor"]] = (offset, length)
        return value

    def value(self, value):
        if isinstance(value, list):
            return [self.value(x) for x in value]
        if isinstance(value, dict):
            if "tensor" in value:
                return self.tensor(value)
            if set(value) == {"mapping_items"}:
                pairs = value["mapping_items"]
                require(len({(type(k), k) for k, _ in pairs}) == len(pairs), "duplicate map key")
                return {k: self.value(v) for k, v in pairs}
            return {k: self.value(v) for k, v in value.items()}
        return value

    def check_coverage(self):
        require(set(self.tensors) == set(range(len(self.tensors))), "tensor index gap")
        cursor = 0
        # JSON key sorting changes visit order; compare sorted byte ranges.
        for offset, size in sorted(self.tensors.values()):
            require(offset == cursor, "overlapping/missing tensor payload")
            cursor += size
        require(cursor == len(self.raw), "unclaimed payload bytes")


def verify_packet(packet, reference_factory):
    import torch
    from qwen35_sampler_bounded_operation_v1 import BoundedTopKTopPSampler, _shortlist
    header, raw = parse_packet(packet)
    require(header["endian"] == sys.byteorder and header["torch_version"] == str(torch.__version__), "native environment changed")
    decoder = Decoder(torch, raw)
    record = decoder.value(header)
    decoder.check_coverage()
    logits = record["processed_logits"]
    require(logits.dtype == torch.float32 and logits.ndim == 2 and logits.shape[1] == VOCAB
            and 1 <= logits.shape[0] <= 4, "unexpected actual logit geometry")
    metadata = record["effective_sampling_metadata"]
    k, p = metadata["top_k"], metadata["top_p"]
    reference = reference_factory(record["reference_logprobs_mode"])
    require(reference.forward.__qualname__ == record["reference_forward"], "reference forward implementation changed")
    candidate = BoundedTopKTopPSampler(reference)
    require(record["arm"] in ("baseline", "candidate"), "wrong capture arm")
    expected_class = reference if record["arm"] == "baseline" else candidate
    require(record["operation_class"] == type(expected_class).__module__ + "." + type(expected_class).__name__, "capture operation identity changed")
    before, after = record["rng_before"], record["rng_after"]
    require(metadata["generator_rows"] == [x["row"] for x in before["generators"]], "effective generator mismatch")
    context = record["request_context"]
    require(len(context) == logits.shape[0] and [x["row"] for x in context] == list(range(logits.shape[0])), "request join incomplete")
    require(len({x["request_id"] for x in context}) == len({x["slot"] for x in context}) == logits.shape[0], "request slot join invalid")
    require(metadata["output_token_ids"] == [x["output_token_ids"] for x in context], "metadata/request history mismatch")

    def execute(operation):
        torch.set_rng_state(before["global_cpu"])
        generators = {}
        for item in before["generators"]:
            require(item["device"] == "cpu", "not a CPU request generator")
            generator = torch.Generator(device="cpu")
            generator.set_state(item["state"])
            require(generator.initial_seed() == item["initial_seed"], "generator seed/state mismatch")
            generators[item["row"]] = generator
        result = operation(logits.clone(), generators, None if k is None else k.clone(), None if p is None else p.clone())
        states = {i: g.get_state().clone() for i, g in generators.items()}
        return result, torch.get_rng_state().clone(), states

    def equal_output(a, b):
        return all((x is None and y is None) or (x is not None and y is not None and torch.equal(x, y)) for x, y in zip(a, b))

    saved_rng = torch.get_rng_state().clone()
    try:
        old, old_rng, old_generators = execute(reference)
        new, new_rng, new_generators = execute(candidate)
        own, own_rng, own_generators = (old, old_rng, old_generators) if record["arm"] == "baseline" else (new, new_rng, new_generators)
        require(equal_output(own, (record["sampled_token_ids"], record["processed_logits_or_logprobs_output"])), "own sampled token/output failed reproduction")
        require(torch.equal(own_rng, after["global_cpu"]), "own global RNG failed reproduction")
        require(set(own_generators) == {x["row"] for x in after["generators"]}, "after-generator rows changed")
        require(all(torch.equal(own_generators[x["row"]], x["state"]) for x in after["generators"]), "own request RNG failed reproduction")
        if before["generators"]:
            require(equal_output(old, new) and torch.equal(old_rng, new_rng)
                    and all(torch.equal(old_generators[i], new_generators[i]) for i in old_generators),
                    "explicit-generator whole-call token/RNG fallback changed")
        distribution = reference.apply_top_k_top_p(logits.clone(), k, p).softmax(dim=-1, dtype=torch.float32)
        bounded_distribution = distribution.clone()
        fast_rows = 0
        if not before["generators"] and candidate._supported(logits, k, p):
            for row, (ids, scores) in enumerate(_shortlist(logits, k, p)):
                if ids is not None:
                    bounded_distribution[row].zero_()
                    bounded_distribution[row, ids] = scores.softmax(dim=-1, dtype=torch.float32)
                    fast_rows += 1
        require(torch.isfinite(distribution).all().item() and torch.isfinite(bounded_distribution).all().item(), "nonfinite distribution")
        require(torch.equal(distribution > 0, bounded_distribution > 0), "support changed")
        error = (distribution.double() - bounded_distribution.double()).abs()
        tv, maximum = float(error.sum(dim=-1).max().item() / 2), float(error.max().item())
        require(tv <= SAMPLER_GATES["total_variation_max"] and maximum <= SAMPLER_GATES["max_probability_error_max"], "distribution mismatch")
        return {"passed": True, "call_id": record["call_id"], "arm": record["arm"],
                "rows": logits.shape[0], "vocabulary": VOCAB, "fast_rows": fast_rows,
                "total_variation_max": tv, "max_probability_error": maximum,
                "own_token_and_rng_reproduced": True, "explicit_generator_fallback_tested": bool(before["generators"]),
                "timing_eligible": False, "full_model_qualified": False}
    finally:
        torch.set_rng_state(saved_rng)


def verify_rejection_controls(reference_factory):
    """Finite native controls, separate from original HTTP bodies and timings.

Explicit seeds here exercise fallback only; they are never added to the HTTP
workloads. Invalid/nonfinite controls may retain a reference exception; that
is logged as such, not counted as a valid sampling distribution.
"""
    import torch
    from qwen35_sampler_bounded_operation_v1 import BoundedTopKTopPSampler, _shortlist
    saved_rng = torch.get_rng_state().clone()
    results = []
    try:
        for name in SAMPLER_GATES["controls"]:
            rows = 4 if name in ("partial_explicit_generators", "all_explicit_generators") else 1
            logits = torch.linspace(-5, 5, 256, dtype=torch.float32).repeat(rows, 1)
            k = torch.full((rows,), 40, dtype=torch.int32)
            p = torch.full((rows,), .9, dtype=torch.float32)
            mode = name if name in ("processed_logits", "processed_logprobs") else "raw_logprobs"
            generator_rows = list(range(rows)) if name == "all_explicit_generators" else [0] if name == "partial_explicit_generators" else []
            if name == "k_absent": k = None
            if name == "k_zero": k.fill_(0)
            if name == "k_above64": k.fill_(65)
            if name == "non_float32": logits = logits.double()
            if name == "invalid_p": p.fill_(0)
            if name == "excessive_kth_ties": logits.zero_()
            if name == "nonfinite_max_or_kth": logits[:, -1] = float("inf")
            if name in ("cdf_near_boundary", "cdf_boundary_tie"):
                logits.fill_(-100)
                count = 2 if name == "cdf_near_boundary" else 16
                logits[:, :count] = 0
                k.fill_(count)
                p.fill_(.5 if name == "cdf_near_boundary" else .45)
            probe = BoundedTopKTopPSampler(reference_factory(mode))
            row_fallback = name in ("excessive_kth_ties", "nonfinite_max_or_kth", "cdf_near_boundary", "cdf_boundary_tie")
            if row_fallback:
                require(probe._supported(logits, k, p), "row control failed to reach shortlist")
                require(all(ids is None and scores is None for ids, scores in _shortlist(logits, k, p)), "rejection control entered fast row path")
            elif not generator_rows:
                require(not probe._supported(logits, k, p), "unsupported control entered fast call path")

            def run(candidate):
                torch.manual_seed(20261111)
                generators = {i: torch.Generator(device="cpu").manual_seed(811 + i) for i in generator_rows}
                reference = reference_factory(mode)
                original_forward = reference.forward
                calls = []
                def counted(*args):
                    calls.append({"rows": args[0].shape[0], "generators": sorted(args[1])})
                    return original_forward(*args)
                reference.forward = counted
                operation = BoundedTopKTopPSampler(reference) if candidate else reference
                try:
                    output = operation(logits.clone(), generators, None if k is None else k.clone(), p.clone())
                    error = None
                except Exception as exc:
                    output, error = None, (type(exc).__name__, str(exc))
                return output, error, torch.get_rng_state().clone(), {i: g.get_state().clone() for i, g in generators.items()}, calls

            old, new = run(False), run(True)
            require(old[1] == new[1] and torch.equal(old[2], new[2]), "fallback error/global RNG changed: " + name)
            require(all(torch.equal(old[3][i], new[3][i]) for i in old[3]), "fallback request RNG changed: " + name)
            require(new[4] == old[4] == [{"rows": rows, "generators": generator_rows}], "fallback was not one original call: " + name)
            if old[0] is not None:
                require(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                            for a, b in zip(old[0], new[0])), "fallback output changed: " + name)
            results.append({"control": name, "passed": True, "reference_calls": 1,
                            "reference_exception": old[1], "whole_call": not row_fallback,
                            "token_and_rng_equal": True})
        return {"passed": True, "controls": results, "full_model_qualified": False, "timing_eligible": False}
    finally:
        torch.set_rng_state(saved_rng)
