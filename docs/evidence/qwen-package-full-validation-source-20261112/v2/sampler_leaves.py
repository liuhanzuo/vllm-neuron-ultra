"""Original complete actual-packet replay and thirteen controls on package sampler."""
import hashlib
from .runtime import require_execution, read as read_small, require
from .capture_writer import nvme_destination
from qwen35_sampler_pair_protocol_v1 import MAX_PACKET_BYTES, check_capture_accounting


def replay_captures(directory, execution_guard):
    require_execution(execution_guard)
    directory = nvme_destination(directory)
    from .sampler_replay import verify_packet
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    receipt = read_small(directory / "capture-receipt.json")
    check_capture_accounting(receipt)
    require(0 < len(receipt["files"]) <= 64, "capture count exceeds cap")
    rows = []
    total = 0
    for index, item in enumerate(receipt["files"]):
        require_execution(execution_guard)
        require(item["file"] == "capture-%04d.bin" % index, "capture path/order changed")
        path = directory / item["file"]
        require(path.resolve() == path and path.stat().st_size == item["bytes"] <= MAX_PACKET_BYTES, "capture file changed")
        packet = path.read_bytes()
        require(hashlib.sha256(packet).hexdigest() == item["sha256"], "capture bytes changed")
        proof = verify_packet(packet, TopKTopPSampler)
        require(proof["call_id"] == item["call_id"] and proof["rows"] == item["rows"] and proof["arm"] == receipt["arm"], "capture receipt join failed")
        rows.append(proof)
        total += len(packet)
    require(total == receipt["total_packet_bytes"], "capture byte accounting mismatch")
    require({str(n): sum(row["rows"] == n for row in rows) for n in range(1, 5)}
            == receipt["captured_calls_by_rows"], "actual B1/B4 capture quotas not reproduced")
    return {"passed": True, "packets": len(rows), "rows": sum(x["rows"] for x in rows),
            "all_own_tokens_and_rng_reproduced": True, "all_distribution_gates_passed": True,
            "fast_path_rows": sum(x["fast_rows"] for x in rows), "proofs": rows,
            "full_model_qualified": False, "timing_eligible": False}


def rejection_controls(execution_guard):
    require_execution(execution_guard)
    from .sampler_replay import verify_rejection_controls
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    return verify_rejection_controls(TopKTopPSampler)
