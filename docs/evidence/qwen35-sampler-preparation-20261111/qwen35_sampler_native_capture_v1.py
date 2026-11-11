"""Finite, diagnostic-only capture at the ACTUAL processed-logit boundary.

No imports of torch/vLLM, no files, and no writer exist at module import time.
A future admitted caller supplies the writer. NativeSnapshotter imports torch
only when constructed. Every selected record retains the complete vocabulary;
records are not collected during either measured HTTP arm.
"""
import hashlib
import json
import math
import struct
import sys

from qwen35_sampler_pair_protocol_v1 import (
    MAX_CALLS_PER_BATCH, MAX_CAPTURE_BYTES_PER_ARM, MAX_JSON_BYTES,
    MAX_PACKET_BYTES, VOCAB, require,
)

MAGIC = b"QSPAIR1\n"
METADATA_FIELDS = (
    "temperature", "all_greedy", "all_random", "top_p", "top_k", "generators",
    "max_num_logprobs", "no_penalties", "prompt_token_ids", "frequency_penalties",
    "presence_penalties", "repetition_penalties", "output_token_ids",
    "allowed_token_ids_mask", "bad_words_token_ids", "logitsprocs",
)
REQUEST_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "min_tokens", "max_tokens",
    "ignore_eos", "seed", "presence_penalty", "frequency_penalty",
    "repetition_penalty", "stop_token_ids", "logits_processors", "allowed_token_ids",
    "bad_words", "logprobs", "prompt_logprobs",
)


def request_context(runner, rows):
    """The runner already reorders logits into input_batch.req_ids order."""
    require(not runner.is_chunked_prefill and not runner.is_prefix_caching
            and runner.speculative_config is None, "unexpected runner mode")
    require(runner.model.neuron_config.on_device_sampling_config is None, "not CPU sampling")
    order = list(runner.input_batch.req_ids)
    require(len(order) == rows and len(set(order)) == rows, "request/row order mismatch")
    records = []
    for index, request_id in enumerate(order):
        require(runner.input_batch.req_id_to_index[request_id] == index, "batch row mapping changed")
        state = runner.requests[request_id]
        prompt, output = list(state.prompt_token_ids), list(state.output_token_ids)
        require(0 < len(prompt) + len(output) <= 4096, "request history outside context4096")
        slot = int(runner.vllm_req_to_neuron_seq_id_mapping[request_id])
        require(0 <= slot < 4, "invalid Neuron slot")
        records.append({"row": index, "request_id": request_id, "slot": slot,
                        "prompt_token_ids": prompt, "output_token_ids": output,
                        "num_computed_tokens": int(runner.input_batch.num_computed_tokens_cpu[index]),
                        "sampling_params": {key: getattr(state.sampling_params, key) for key in REQUEST_FIELDS},
                        "all_stop_token_ids": sorted(state.sampling_params.all_stop_token_ids)})
    require(len({r["slot"] for r in records}) == rows, "duplicate active slots")
    return records


class CaptureBudget:
    def __init__(self):
        self.seen = {str(n): 0 for n in range(1, 5)}
        self.captured = {str(n): 0 for n in range(1, 5)}
        self.total_bytes = 0

    def select(self, rows, width):
        require(type(rows) is int and rows in range(1, 5) and width == VOCAB, "unexpected actual logit shape")
        self.seen[str(rows)] += 1
        return self.seen[str(rows)] <= MAX_CALLS_PER_BATCH

    def check_bytes(self, packet):
        require(type(packet) is bytes and 0 < len(packet) <= MAX_PACKET_BYTES, "packet exceeds finite cap")
        require(self.total_bytes + len(packet) <= MAX_CAPTURE_BYTES_PER_ARM, "capture byte cap exceeded")

    def commit(self, rows, packet):
        self.check_bytes(packet)
        require(self.captured[str(rows)] < MAX_CALLS_PER_BATCH, "batch capture cap exceeded")
        self.captured[str(rows)] += 1
        self.total_bytes += len(packet)

    def receipt(self):
        rows = sum(int(k) * n for k, n in self.captured.items())
        return {"phase": "correctness_capture", "seen_calls_by_rows": dict(self.seen),
                "captured_calls_by_rows": dict(self.captured), "packets": sum(self.captured.values()),
                "captured_rows": rows, "processed_logit_bytes": rows * VOCAB * 4,
                "total_packet_bytes": self.total_bytes, "timing_eligible": False}


class PacketBuilder:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.blobs = []
        self.blob_bytes = 0
        self.nodes = 0

    def tensor(self, value):
        torch = self.torch
        require(value.device.type == "cpu" and value.layout == torch.strided, "capture expects actual CPU strided tensor")
        length = int(value.numel() * value.element_size())
        require(0 <= length <= MAX_PACKET_BYTES - self.blob_bytes, "tensor capture cap exceeded")
        # An explicit fresh flat allocation normalizes singleton strides too;
        # contiguous().view(uint8) alone can fail on such tensors.
        flat = torch.empty((value.numel(),), dtype=value.dtype, device="cpu")
        flat.copy_(value.detach().reshape(-1))
        raw = flat.view(torch.uint8).numpy().tobytes()
        require(len(raw) == length, "tensor byte count mismatch")
        result = {"tensor": len(self.blobs), "dtype": str(value.dtype), "shape": list(value.shape),
                  "original_stride": list(value.stride()), "original_device": str(value.device),
                  "offset": self.blob_bytes, "bytes": length, "sha256": hashlib.sha256(raw).hexdigest()}
        self.blobs.append(raw)
        self.blob_bytes += length
        return result

    def value(self, value, depth=0):
        self.nodes += 1
        require(depth <= 12 and self.nodes <= 100000, "metadata bound exceeded")
        if isinstance(value, self.torch.Tensor):
            return self.tensor(value)
        if value is None or type(value) in (str, bool, int):
            require(type(value) is not str or len(value.encode()) <= MAX_JSON_BYTES, "string bound exceeded")
            return value
        if type(value) is float:
            require(math.isfinite(value), "nonfinite scalar metadata")
            return value
        if type(value) in (list, tuple):
            return [self.value(x, depth + 1) for x in value]
        if type(value) is dict:
            require(all(type(k) in (str, int) for k in value), "unsupported mapping key")
            # Preserve integer row keys instead of silently converting to strings.
            return {"mapping_items": [[k, self.value(v, depth + 1)] for k, v in value.items()]}
        raise TypeError("Uncaptured effective value type: " + type(value).__name__)

    def finish(self, header):
        data = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        require(len(data) <= MAX_JSON_BYTES, "capture JSON exceeds cap")
        require(len(MAGIC) + 8 + len(data) + self.blob_bytes <= MAX_PACKET_BYTES, "capture packet exceeds cap")
        return MAGIC + struct.pack("<Q", len(data)) + data + b"".join(self.blobs)


class NativeSnapshotter:
    def __init__(self, torch_module=None):
        if torch_module is None:
            import torch
            torch_module = torch
        self.torch = torch_module

    def rng(self, builder, generators):
        torch = self.torch
        require(all(type(i) is int and str(g.device) == "cpu" for i, g in generators.items()),
                "request generator must be CPU and row keyed")
        return {"global_cpu": builder.tensor(torch.get_rng_state()),
                "generators": [{"row": i, "device": str(g.device), "initial_seed": g.initial_seed(),
                                "state": builder.tensor(g.get_state())} for i, g in sorted(generators.items())]}

    def before(self, arm, call_id, logits, generators, k, p, metadata, context, operation, reference_forward):
        torch = self.torch
        require(logits.dtype == torch.float32 and tuple(logits.shape)[1:] == (VOCAB,), "unexpected processed logits")
        require(set(vars(metadata)) == set(METADATA_FIELDS), "SamplingMetadata schema changed")
        require(generators is metadata.generators and k is metadata.top_k and p is metadata.top_p,
                "not the original top-k/top-p call boundary")
        require(all(0 <= i < logits.shape[0] for i in generators), "generator row out of range")
        processors = metadata.logitsprocs
        # The original runner builds an empty processor container. Observe it;
        # never invent replacement defaults or add a min_tokens processor.
        require(not processors.non_argmax_invariant and not processors.argmax_invariant,
                "original empty processor chain changed; new design required")
        builder = PacketBuilder(torch)
        meta = {key: builder.value(getattr(metadata, key)) for key in METADATA_FIELDS
                if key not in ("generators", "logitsprocs")}
        meta["generator_rows"] = sorted(generators)
        meta["logitsprocs"] = {"non_argmax_invariant": [], "argmax_invariant": []}
        reference = getattr(operation, "reference", operation)
        header = {"schema": "qwen35_sampler_capture_v1", "arm": arm, "phase": "correctness_capture",
                  "call_id": call_id, "endian": sys.byteorder, "torch_version": str(torch.__version__),
                  "processed_logits": builder.tensor(logits), "effective_sampling_metadata": meta,
                  "request_context": builder.value(context), "rng_before": self.rng(builder, generators),
                  "operation_class": type(operation).__module__ + "." + type(operation).__name__,
                  "reference_logprobs_mode": reference.logprobs_mode,
                  "reference_forward": reference_forward, "timing_eligible": False}
        return builder, header

    def after(self, snapshot, generators, result):
        builder, header = snapshot
        header["rng_after"] = self.rng(builder, generators)
        require(type(result) is tuple and len(result) == 2, "top-k/top-p output contract changed")
        header["sampled_token_ids"] = builder.value(result[0])
        header["processed_logits_or_logprobs_output"] = builder.value(result[1])
        return builder.finish(header)


class CaptureObserver:
    """Per-instance hooks; the original operator and all RNG calls run once."""
    def __init__(self, runner, sampler, arm, writer, snapshotter, context_fn=request_context):
        require(arm in ("baseline", "candidate") and callable(writer), "missing capture dependency")
        self.runner, self.sampler, self.arm = runner, sampler, arm
        self.writer, self.snapshotter, self.context_fn = writer, snapshotter, context_fn
        self.budget, self.active, self.call_id = CaptureBudget(), None, 0

    def install(self):
        sampler = self.sampler
        require(not getattr(sampler, "_qwen35_pair_observed", False), "already observed")
        operation = sampler.topk_topp_sampler
        original_sampler, original_operation = sampler.forward, operation.forward
        reference_forward = getattr(operation, "reference", operation).forward.__qualname__

        def observed_sampler(logits, sampling_metadata):
            require(self.active is None, "concurrent/reentrant sampler capture unsupported")
            self.active = sampling_metadata
            try:
                return original_sampler(logits, sampling_metadata)
            finally:
                self.active = None

        def observed_operation(logits, generators, k, p):
            require(self.active is not None, "top-k called without actual SamplingMetadata")
            self.call_id += 1
            rows, width = logits.shape
            if not self.budget.select(rows, width):
                return original_operation(logits, generators, k, p)
            context = self.context_fn(self.runner, rows)
            snapshot = self.snapshotter.before(self.arm, self.call_id, logits, generators, k, p,
                                               self.active, context, operation, reference_forward)
            result = original_operation(logits, generators, k, p)
            packet = self.snapshotter.after(snapshot, generators, result)
            self.budget.check_bytes(packet)
            receipt = self.writer(packet)
            require(receipt == {"bytes": len(packet), "sha256": hashlib.sha256(packet).hexdigest()},
                    "writer did not confirm complete captured bytes")
            self.budget.commit(rows, packet)
            return result

        sampler.forward = observed_sampler
        operation.forward = observed_operation
        sampler._qwen35_pair_observed = True


def parse_packet(packet):
    """Small stdlib framing validator; this does not execute any native replay."""
    require(type(packet) is bytes and len(MAGIC) + 8 <= len(packet) <= MAX_PACKET_BYTES, "bad packet size")
    require(packet.startswith(MAGIC), "bad packet magic")
    offset = len(MAGIC) + 8
    size, = struct.unpack("<Q", packet[len(MAGIC):offset])
    require(0 < size <= MAX_JSON_BYTES and offset + size <= len(packet), "bad header size")
    header = json.loads(packet[offset:offset + size])
    require(header["schema"] == "qwen35_sampler_capture_v1" and header["timing_eligible"] is False, "wrong capture schema")
    require(header["phase"] == "correctness_capture" and header["arm"] in ("baseline", "candidate"), "wrong capture phase/arm")
    return header, memoryview(packet)[offset + size:]
