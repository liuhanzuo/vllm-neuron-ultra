"""New runtime contract around the frozen sampler v1 scientific delivery.

Pure contract/source helpers. No default spec, handoff, unit, acquisition or IO
is created on import. A future root review binds every current history input.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

from qwen35_sampler_pair_protocol_v1 import ROOT, require

Q = ROOT / "research"
PYTHON = "/home/ubuntu/venvs/vllm_neuron/bin/python"
LOCK_PATH = ROOT / "serial-enrollment.lock"
LOCK_IDENTITY = {"device": 66306, "inode": 8294421}
GIB, MIB = 2**30, 2**20
LIMITS = {
    "root_floor_bytes": 28 * GIB, "nvme_floor_bytes": 30 * GIB,
    "total_allocated_output_bytes": 2 * GIB, "single_file_bytes": 128 * MIB,
    "model_single_file_bytes": 512 * MIB, "model_payload_bytes": 832 * MIB,
    "capture_payload_bytes": 512 * MIB, "all_logs_bytes": 512 * MIB,
    "all_metadata_bytes": 128 * MIB, "temporary_and_directory_bytes": 64 * MIB,
    "spec_envelope_bytes": 64 * MIB, "source_map_entries": 100000,
    "host_environment_string_bytes": 64 * 1024,
    "metadata_bytes": 8 * MIB, "max_output_entries": 1024,
    "server_log_bytes": 64 * MIB, "leaf_log_bytes": 8 * MIB,
    "coordinator_log_bytes": 16 * MIB, "outer_wall_seconds": 180000,
    "monitor_interval_seconds": .5, "startup_ack_seconds": 60,
    "root_running_ack_seconds": 180, "owned_cleanup_seconds": 45,
    "memory_available_floor_bytes": 2 * GIB,
}
COMMON_PROPERTIES = {"KillMode": "control-group", "RemainAfterExit": "yes", "SendSIGKILL": "yes",
                     "WorkingDirectory": str(ROOT), "TimeoutStopSec": "15",
                     "LimitFSIZE": str(128 * MIB), "TasksMax": "256"}
COORDINATOR_PROPERTIES = {**COMMON_PROPERTIES, "MemoryMax": str(4 * GIB), "MemoryHigh": str(3 * GIB),
                          "MemorySwapMax": str(GIB), "RuntimeMaxSec": "179000"}
SERVER_PROPERTIES = {**COMMON_PROPERTIES, "MemoryMax": str(26 * GIB), "MemoryHigh": str(25 * GIB),
                     "MemorySwapMax": str(12 * GIB), "RuntimeMaxSec": "14400"}
MODEL_PROPERTIES = {**COMMON_PROPERTIES, "MemoryMax": str(26 * GIB), "MemoryHigh": str(25 * GIB),
                    "MemorySwapMax": str(4 * GIB), "RuntimeMaxSec": "3600", "LimitFSIZE": str(512 * MIB)}
CPU_PROPERTIES = {**COMMON_PROPERTIES, "MemoryMax": str(4 * GIB), "MemoryHigh": str(3 * GIB),
                  "MemorySwapMax": str(GIB), "RuntimeMaxSec": "1800"}
PHASES = tuple((arm, phase) for arm in ("baseline", "candidate")
               for phase in ("state_copy", "numeric", "cold", "long", "vectors203",
                             "capture", "sampler_replay", "quality", "http")) + (
    ("shared", "native_preflight"), ("shared", "sampler_controls"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def reference(value):
    require(type(value) is dict and set(value) == {"path", "sha256"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and str(Path(value["path"])) == value["path"] and ".." not in Path(value["path"]).parts
            and type(value["sha256"]) is str and re.fullmatch("[a-f0-9]{64}", value["sha256"]),
            "exact absolute hash-authenticated reference required")
    return value


def pin_map(value):
    require(type(value) is dict and 0 < len(value) <= LIMITS["source_map_entries"]
            and len(json.dumps(value).encode()) <= LIMITS["spec_envelope_bytes"],
            "complete bounded source map required")
    for path, sha in value.items():
        reference({"path": path, "sha256": sha})


def merge_sources(*maps):
    result = {}
    for values in maps:
        pin_map(values)
        for path, sha in values.items():
            require(path not in result or result[path] == sha, "historical source conflict, never replace: " + path)
            result[path] = sha
    return result


def units(run_name):
    require(type(run_name) is str and re.fullmatch("[a-z0-9][a-z0-9-]{0,39}", run_name), "bounded root-selected run name")
    prefix = "trn1-qwen35-sampler-" + run_name
    return {"coordinator": prefix + "-coordinator", **{
        arm + ":" + phase: prefix + "-" + arm + "-" + phase.replace("_", "-") for arm, phase in PHASES}}


def phase_properties(arm, phase):
    require((arm, phase) in PHASES, "undeclared phase")
    if phase in ("capture", "quality", "http"):
        return copy.deepcopy(SERVER_PROPERTIES)
    if phase in ("state_copy", "numeric", "cold", "long"):
        result = copy.deepcopy(MODEL_PROPERTIES)
        if phase == "state_copy": result["RuntimeMaxSec"] = "900"
        return result
    result = copy.deepcopy(CPU_PROPERTIES)
    if phase == "sampler_controls": result["RuntimeMaxSec"] = "600"
    return result


def validate_spec(spec, review, spec_ref):
    reference(spec_ref)
    keys = {"schema", "run_name", "output_root", "history_overlay", "current_serial_catalog",
            "source_sha256", "input_extension", "frozen_science_delivery", "running_handoff_path",
            "limits", "lock_identity", "native_python", "automatic_successor"}
    require(type(spec) is dict and set(spec) == keys and spec["schema"] == "qwen35_sampler_runtime_spec_v1", "exact new runtime spec")
    names = units(spec["run_name"])
    require(spec["output_root"] == "/mnt/nvme/neuron-work/qwen35-sampler-pair-" + spec["run_name"], "fresh exact NVMe namespace")
    require(spec["limits"] == LIMITS and spec["lock_identity"] == LOCK_IDENTITY
            and spec["native_python"] == PYTHON and spec["automatic_successor"] is False, "original lock/scope/finite resource policy")
    for key in ("history_overlay", "current_serial_catalog", "frozen_science_delivery"):
        reference(spec[key])
    require(spec["frozen_science_delivery"] == {
        "path": str(Q / "qwen35-sampler-pair-source-delivery-20261111-v1.json"),
        "sha256": "2b50426e39728dc20800aa0030544af9b37307bca01c61e39de5dd4320bef04b"}, "frozen scientific v1 delivery retained")
    running = Path(spec["running_handoff_path"])
    require(running.parent == Q and re.fullmatch("continuation-handoff-[a-z0-9-]+\\.json", running.name), "future root-selected handoff path")
    pin_map(spec["source_sha256"])
    require(type(spec["input_extension"]) is dict, "complete current metadata identity extension")
    for path, row in spec["input_extension"].items():
        require(type(row) is dict and set(row) == {"sha256", "identity"}, "explicit metadata digest and stat identity")
        reference({"path": path, "sha256": row["sha256"]})
        require(type(row["identity"]) is dict, "actual metadata identity required")
    require(type(review) is dict and set(review) == {"schema", "reviewer", "passed", "blockers", "spec", "source_sha256", "launch_authorized"}
            and review["schema"] == "qwen35_sampler_runtime_source_review_v1" and review["reviewer"] == "/root"
            and review["passed"] is True and review["blockers"] == [] and review["spec"] == spec_ref
            and review["source_sha256"] == spec["source_sha256"] and review["launch_authorized"] is True,
            "future exact independent root review/admission required")
    return names


def native_environment(case, *, host_identity=None, host_receipt=None, capture_root=None):
    """Preserve the original original server settings in both sampler arms."""
    result = {"HOME": "/home/ubuntu", "LANG": "C.UTF-8", "XDG_RUNTIME_DIR": "/run/user/1000",
              "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus", "PYTHONDONTWRITEBYTECODE": "1",
              "PYTHONOPTIMIZE": "0", "PYTHONPATH": str(Q) + ":" + str(ROOT),
              "HF_HOME": "/mnt/nvme/hf", "NEURON_COMPILE_CACHE_URL": "/mnt/nvme/neuron-cache",
              "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
              "PATH": "/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin",
              "NEURON_COMPILED_ARTIFACTS": case["artifact"], "NEURON_RT_VISIBLE_CORES": "0-1",
              "TRN1_QWEN35_TEXT_ADAPTER": "1", "TRN1_QWEN35_SPLIT_SHA256": case["split_correction_sha256"],
              "TRN1_COMPILE_WORKERS": "1", "TORCHINDUCTOR_COMPILE_THREADS": "1",
              "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "XLA_DISABLE_FUNCTIONALIZATION": "1"}
    require((host_identity is None) == (host_receipt is None), "complete host worker binding")
    if host_identity is not None:
        result["QWEN35_SAMPLER_HOST_IDENTITY"] = json.dumps(host_identity, sort_keys=True)
        require(len(("QWEN35_SAMPLER_HOST_IDENTITY=" + result["QWEN35_SAMPLER_HOST_IDENTITY"]).encode())
                <= LIMITS["host_environment_string_bytes"], "host identity exceeds bounded Linux environment string")
        result["QWEN35_SAMPLER_HOST_RECEIPT"] = str(host_receipt)
    if capture_root is not None:
        require(host_identity is not None and host_identity["phase"] == "correctness_capture", "capture env only in diagnostic phase")
        result["QWEN35_SAMPLER_CAPTURE_ROOT"] = str(capture_root)
    return result


RUNTIME_FILES = ("qwen35_sampler_runtime_contract_v8.py", "qwen35_sampler_runtime_preflight_v1.py",
                 "qwen35_sampler_runtime_io_v8.py", "qwen35_sampler_runtime_history_v8.py",
                 "qwen35_sampler_runtime_service_v8.py", "qwen35_sampler_runtime_adapter_v8.py",
                 "run_qwen35_sampler_runtime_v8.py", "qwen35_sampler_torch_namespace_sources_v1.py",
                 "qwen35_sampler_native_leaves_v1.py", "qwen35_sampler_native_leaves_v2.py")
STARTUP_FILES = tuple("/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/vllm/" + name
                      for name in ("config/parallel.py", "engine/arg_utils.py", "envs.py", "v1/engine/core.py"))


def host_sources(spec, delivery):
    """Small actual host closure; all historical pins stay in the complete spec."""
    result = merge_sources(delivery["new_source_sha256"], delivery["prospective_host_dependency_sha256"])
    for path in (*[str(Q / name) for name in RUNTIME_FILES], *STARTUP_FILES):
        require(path in spec["source_sha256"], "new runtime source must be reviewed")
        require(path not in result or result[path] == spec["source_sha256"][path],
                "runtime additions cannot replace a frozen scientific source pin")
        result[path] = spec["source_sha256"][path]
    require(all(spec["source_sha256"].get(p) == s for p, s in result.items()), "host closure is exact authenticated subset")
    require(len(json.dumps(result).encode()) < LIMITS["host_environment_string_bytes"] - 4096, "compact host map required")
    return result


def control_root(spec):
    return Path(spec["output_root"] + "-runtime")


def budget():
    """Full two-arm serialized storage, with independent class and global caps."""
    parts = {key: LIMITS[key] for key in ("model_payload_bytes", "capture_payload_bytes", "all_logs_bytes",
                                         "all_metadata_bytes", "temporary_and_directory_bytes")}
    require(sum(parts.values()) == LIMITS["total_allocated_output_bytes"], "complete nonoverlapping budget")
    require(2 * (31791817 + 83461609 + 287083673) <= parts["model_payload_bytes"], "actual frozen serialization fits")
    require(6 * LIMITS["server_log_bytes"] + 14 * LIMITS["leaf_log_bytes"]
            + LIMITS["coordinator_log_bytes"] <= parts["all_logs_bytes"], "all21 finite unit logs fit")
    return parts
