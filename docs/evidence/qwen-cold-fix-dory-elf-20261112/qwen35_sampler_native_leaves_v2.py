"""Concrete native leaf bodies for the later guarded serial execution adapter.

No native imports at module import time. Each leaf requires the root's current
execution guard before native imports or payload reads. These functions do not
start subprocesses, create units, relax resource limits, or provide admission.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

from qwen35_sampler_pair_protocol_v1 import BASE, ROOT, MAX_PACKET_BYTES, check_capture_accounting, expected_case, read_small, require
from qwen35_sampler_full_checks_v1 import check_model_metadata, original_validators
from qwen35_sampler_worker_v1 import nvme_destination

PAYLOADS = (("numeric", "real-logits-validation.json.pt", "real-logits-validation.json", 32),
            ("cold", "cold-buckets.pt", "cold-buckets.json", 84),
            ("long", "long-decode-validation.pt", "long-decode-validation.json", 87))


def require_execution(execution_guard):
    require(callable(execution_guard), "root execution guard required")
    receipt = execution_guard()
    require(isinstance(receipt, dict) and receipt.get("admitted") is True and receipt.get("exclusive") is True,
            "root did not admit this native leaf")


def checked_source(name):
    path = ROOT / name
    pins = read_small(BASE / "plan.json")["source_sha256"]
    require(path.stat().st_size < 4 * 1024**2, "source bound exceeded")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pins[str(path)], "original model body changed")
    return raw


def state_copy_body():
    """Keep every scientific statement; replace only the old service-name guard.

    The new root guard authenticates the actual successor owner. It is never
    implemented by pretending that the old service names are running.
    """
    tree = ast.parse(checked_source("probe_qwen35_state_copy_v1.py"))
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    require(isinstance(original.body[0], ast.Assert)
            and "running_heavy_units" in ast.unparse(original.body[0])
            and "trn1-qwen35-state-reset-full-v1-worker.service" in ast.unparse(original.body[0]), "unexpected old guard")
    saved = original.body[0]
    original.body[0] = ast.Expr(value=ast.Call(func=ast.Name(id="execution_guard", ctx=ast.Load()), args=[], keywords=[]))
    body = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
    # Reverse exactly the sole AST edit to prove no reduced scientific body.
    new_guard = original.body[0]
    original.body[0] = saved
    source_main = next(n for n in ast.parse(checked_source("probe_qwen35_state_copy_v1.py")).body
                       if isinstance(n, ast.FunctionDef) and n.name == "main")
    require(ast.dump(original, include_attributes=False) == ast.dump(source_main, include_attributes=False), "state-copy scientific body changed")
    original.body[0] = new_guard
    return body


def run_model_stage(phase, directory, execution_guard):
    require_execution(execution_guard)
    directory = nvme_destination(directory)
    case = read_small(directory / "case.json")
    require(case == expected_case(), "frozen model case changed")
    filenames = {"state_copy": "state-copy.json", "numeric": "real-logits-validation.json",
                 "cold": "cold-buckets.json", "long": "long-decode-validation.json"}
    require(phase in filenames and not (directory / filenames[phase]).exists(), "invalid/reused model stage")
    # The frozen cold main owns its one strict installation; other bodies do not.
    if phase != "cold":
        from frontend_split_correction_v2 import install
        install("qwen35")
    import orchestrate
    from qwen35_split_runtime_v4 import install_runner_hooks
    install_runner_hooks(orchestrate)
    require(orchestrate.validate_cache(case), "frozen graph cache invalid; compilation forbidden")
    if phase == "numeric":
        checked_source("validate_real_logits.py")
        from validate_real_logits import neuron
        neuron(SimpleNamespace(model=case["model_path"], artifacts=case["artifact"], adapter="qwen35_text",
                               reference=str(ROOT / "qwen35-2b-hf-reference.pt"), output=str(directory / filenames[phase])))
    elif phase == "long":
        checked_source("validate_qwen35_long_decode.py")
        from validate_qwen35_long_decode import neuron
        neuron(case_path=directory / "case.json", result_dir=directory)
    elif phase == "cold":
        checked_source("diagnose_qwen35_state_reset_cold_v1.py")
        original = importlib.import_module("diagnose_qwen35_state_reset_cold_v1")
        namespace = {**vars(original), "DEST": directory}
        # New function globals redirect only the output directory. The original
        # module and its file are untouched, including all84 original frames.
        FunctionType(original.main.__code__, namespace, "main")()
    else:
        body = state_copy_body()
        original = importlib.import_module("probe_qwen35_state_copy_v1")
        namespace = {**vars(original), "DEST": directory, "execution_guard": lambda: require_execution(execution_guard)}
        exec(compile(body, "qwen35_sampler_state_copy_body_v1", "exec"), namespace)
        namespace["main"]()
    return check_model_metadata(read_small(directory / filenames[phase]), phase, original_validators())


def compare_vectors(before_dir, after_dir, execution_guard):
    require_execution(execution_guard)
    require(Path(before_dir) == BASE, "raw comparison must use the qualified baseline")
    after_dir = nvme_destination(after_dir)
    import torch
    torch.set_num_threads(1)
    validators = original_validators()
    phases = []
    for phase, payload, metadata, count in PAYLOADS:
        require_execution(execution_guard)
        old_report, new_report = read_small(BASE / metadata), read_small(after_dir / metadata)
        check_model_metadata(old_report, phase, validators)
        check_model_metadata(new_report, phase, validators)
        old = torch.load(BASE / payload, weights_only=True, map_location="cpu")
        new = torch.load(after_dir / payload, weights_only=True, map_location="cpu")
        require(len(old) == len(new) == count, "raw frame count changed")
        for a, b in zip(old, new):
            require(set(a) == set(b) and {k: v for k, v in a.items() if k != "logits"}
                    == {k: v for k, v in b.items() if k != "logits"}, "raw frame identity changed")
            x, y = a["logits"], b["logits"]
            require(x.shape == y.shape == (248320,) and x.dtype == y.dtype == torch.float32, "raw vector geometry changed")
            require(torch.isfinite(x).all().item() and torch.isfinite(y).all().item(), "nonfinite raw model vector")
            require(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8)), "host-only arm altered model vector bits")
        phases.append({"phase": phase, "frames": count, "identical_vectors": count})
        del old, new
    return {"complete": True, "frames": 203, "identical_vectors": 203, "phases": phases,
            "full_model_qualified": False}


def replay_captures(directory, execution_guard):
    require_execution(execution_guard)
    directory = nvme_destination(directory)
    from qwen35_sampler_native_replay_v1 import verify_packet
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
    from qwen35_sampler_native_replay_v1 import verify_rejection_controls
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    return verify_rejection_controls(TopKTopPSampler)
