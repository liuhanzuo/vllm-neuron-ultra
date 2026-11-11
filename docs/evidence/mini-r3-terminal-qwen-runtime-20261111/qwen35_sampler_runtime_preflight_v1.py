"""Small real native preflight, additional to every frozen v1 full gate.

No native import or execution at module import. Called only by a separately
owned/admitted native leaf; this function cannot authorize or launch a service.
"""
import hashlib
import pickle

from qwen35_sampler_pair_protocol_v1 import require
from qwen35_sampler_native_leaves_v1 import require_execution


def run(execution_guard):
    require_execution(execution_guard)
    import torch
    from qwen35_sampler_native_capture_v1 import PacketBuilder, parse_packet
    from qwen35_sampler_native_replay_v1 import Decoder, verify_rejection_controls
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    import qwen35_sampler_worker_v1 as workers
    torch.set_num_threads(1)
    worker_proofs = []
    for name in workers.EXPORTS:
        require_execution(execution_guard)
        cls = getattr(workers, name)
        payload = pickle.dumps(cls, protocol=4)
        require(pickle.loads(payload) is cls and cls.__module__ == workers.__name__ and cls.__qualname__ == name,
                "actual native worker import/pickle failed")
        worker_proofs.append({"class": workers.__name__ + "." + name, "pickle_bytes": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest(), "passed": True})
    cases = []
    for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16, torch.int64, torch.int32, torch.uint8, torch.bool):
        base = torch.arange(24, dtype=torch.int64).reshape(4, 6).to(dtype)
        cases.extend([(str(dtype) + ":contiguous", base), (str(dtype) + ":slice", base[:, ::2]),
                      (str(dtype) + ":singleton_stride", base[::3, :1]),
                      (str(dtype) + ":singleton_stride0", torch.as_strided(base, (1,), (0,))),
                      (str(dtype) + ":empty", base[:0])])
    cases.extend([("fp32:scalar", torch.tensor(-0., dtype=torch.float32)),
                  ("fp32:masked", torch.tensor([[float("-inf"), -0., 1.]], dtype=torch.float32))])
    tensor_proofs = []
    saved_rng = torch.get_rng_state().clone()
    for name, value in cases:
        require_execution(execution_guard)
        builder = PacketBuilder(torch)
        header = {"schema": "qwen35_sampler_capture_v1", "phase": "correctness_capture", "arm": "baseline",
                  "timing_eligible": False, "value": builder.tensor(value)}
        packet = builder.finish(header)
        decoded_header, raw = parse_packet(packet)
        decoder = Decoder(torch, raw)
        decoded = decoder.value(decoded_header)["value"]
        decoder.check_coverage()
        require(decoded.shape == value.shape and decoded.dtype == value.dtype
                and decoded.stride() == value.stride(), "actual shape/dtype/stride roundtrip failed")
        x = torch.empty((value.numel(),), dtype=value.dtype).copy_(value.reshape(-1))
        y = torch.empty((decoded.numel(),), dtype=decoded.dtype).copy_(decoded.reshape(-1))
        require(torch.equal(x.view(torch.uint8), y.view(torch.uint8)), "actual serialized tensor bits changed")
        tensor_proofs.append({"case": name, "shape": list(value.shape), "stride": list(value.stride()),
                              "packet_bytes": len(packet), "passed": True})
    require(torch.equal(saved_rng, torch.get_rng_state()), "serializer changed actual CPU RNG")
    # No captured production field currently has a nonempty overlapping
    # expand producer (separately reviewed source trace). Preserve the frozen
    # decoder's explicit rejection; do not claim general overlap support or
    # silently canonicalize the original layout.
    expanded = torch.tensor([3.], dtype=torch.float32).expand(4)
    builder = PacketBuilder(torch)
    packet = builder.finish({"schema": "qwen35_sampler_capture_v1", "phase": "correctness_capture",
                             "arm": "baseline", "timing_eligible": False, "value": builder.tensor(expanded)})
    expanded_header, expanded_raw = parse_packet(packet)
    require(expanded_header["value"]["original_stride"] == [0], "expanded stride metadata must remain exact")
    rejected = None
    try:
        Decoder(torch, expanded_raw).value(expanded_header)
    except RuntimeError as error:
        require("single memory location" in str(error) or "overlap" in str(error), "unexpected expanded-layout failure")
        rejected = {"control": "nonempty_expanded_stride0_rejected", "passed": True,
                    "error_type": type(error).__name__, "error": str(error), "shape": [4], "stride": [0]}
    require(rejected is not None, "frozen decoder unexpectedly accepted unsupported overlapping stride0")
    require_execution(execution_guard)
    controls = verify_rejection_controls(TopKTopPSampler)
    require_execution(execution_guard)
    return {"schema": "qwen35_sampler_native_preflight_v1", "passed": True,
            "worker_import_pickle": worker_proofs, "actual_native_serialization_cases": tensor_proofs,
            "actual_rng_unchanged_by_serialization": True, "rejection_controls": controls,
            "overlapping_layout_negative_control": rejected, "general_overlapping_layout_support_claimed": False,
            "model_loaded": False, "device_execution": False, "timing_eligible": False,
            "full_model_qualified": False, "self_exit_claimed": False,
            "actual_process_exit_pending_root_observation": True}
