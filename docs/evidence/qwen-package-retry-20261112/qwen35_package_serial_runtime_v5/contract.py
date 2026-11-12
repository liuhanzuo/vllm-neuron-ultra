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
# Stages 18,21,24 are coordinator metadata gates; stage25 is external root audit.
PHASES = (("shared", "prepare"), ("shared", "build")) + tuple(
    (arm, phase) for arm in ("baseline", "candidate")
    for phase in ("state_copy", "numeric", "cold", "long", "vectors203", "capture", "sampler_replay", "quality", "http")) + (("shared", "sampler_controls"),)
HERE = Path(__file__).resolve().parent
PACKAGE = Path("/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107")
SDK = Path("/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages")
SCIENCE_DELIVERY = {"path": str(Q / "qwen35_package_full_validation_v2/source_delivery.json"),
    "sha256": "020b4032a594a2a9dddee0e530e8b596456529540b4ed32937de49f9473e1b0d"}
BINARY_PATHS = tuple(str(ROOT / x) for x in (
    "qwen35-2b-hf-reference.pt", "qwen35-baseline/long-decode-hf-reference.pt",
    "qwen35-split-corrected-full-v4/cold-buckets.pt",
    "qwen35-state-reset-full-v1/real-logits-validation.json.pt",
    "qwen35-state-reset-full-v1/cold-buckets.pt",
    "qwen35-state-reset-full-v1/long-decode-validation.pt"))

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


def phase_properties(arm, phase, spec=None):
    require((arm, phase) in PHASES, "undeclared package phase")
    if phase == "build":
        require(spec is not None and type(spec.get("build_file_limit_bytes")) is int
                and spec["build_file_limit_bytes"] > 0, "reviewed finite build file limit required")
        return {**SERVER_PROPERTIES, "LimitFSIZE": str(spec["build_file_limit_bytes"])}
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
            "limits", "lock_identity", "native_python", "automatic_successor", "lfm_r2_scope",
            "case", "binary_inputs", "resource_budget", "build_file_limit_bytes"}
    require(type(spec) is dict and set(spec) == keys and spec["schema"] == "qwen35_package_serial_spec_v1", "exact package successor spec")
    from qwen35_package_full_validation_v2.runtime import validate_case
    validate_case(spec["case"])
    names = units(spec["run_name"])
    require(spec["limits"] == LIMITS and spec["lock_identity"] == LOCK_IDENTITY
            and spec["native_python"] == PYTHON and spec["automatic_successor"] is False,
            "unchanged original lock and scientific resource policy")
    require(spec["output_root"] == "/mnt/nvme/neuron-work/qwp-" + spec["run_name"], "fresh short output namespace")
    ipc_paths(spec)
    for key in ("history_overlay", "current_serial_catalog", "frozen_science_delivery", "lfm_r2_scope", "resource_budget"):
        reference(spec[key])
    require(spec["frozen_science_delivery"] == SCIENCE_DELIVERY, "exact full-v2 science delivery required")
    require(type(spec["build_file_limit_bytes"]) is int and 0 < spec["build_file_limit_bytes"] < 2**63, "explicit finite build file limit required")
    running = Path(spec["running_handoff_path"])
    require(running.parent == Q and re.fullmatch("continuation-handoff-[a-z0-9-]+\\.json", running.name), "future root-selected handoff path")
    pin_map(spec["source_sha256"])
    require(type(spec["input_extension"]) is dict, "complete metadata input extension required")
    for path, row in spec["input_extension"].items():
        require(type(row) is dict and set(row) == {"sha256", "identity"} and Path(path).suffix == ".json",
                "input_extension is JSON metadata only")
        reference({"path": path, "sha256": row["sha256"]})
        require(type(row["identity"]) is dict, "actual metadata identity required")
    budget_input = spec["input_extension"].get(spec["resource_budget"]["path"])
    require(budget_input is not None and budget_input["sha256"] == spec["resource_budget"]["sha256"],
            "reviewed resource budget retains original metadata identity/unchanged checks")
    require(type(spec["binary_inputs"]) is dict and set(spec["binary_inputs"]) == set(BINARY_PATHS), "all six separate scientific binary references required")
    for path, row in spec["binary_inputs"].items():
        require(type(row) is dict and set(row) == {"kind", "identity"} and row["kind"] == "binary_payload"
                and type(row["identity"]) is dict and path not in spec["input_extension"], "typed binary identity; never JSON-load payloads")
    require(type(review) is dict and set(review) == {"schema", "reviewer", "passed", "blockers", "spec", "source_sha256", "launch_authorized"}
            and review["schema"] == "qwen35_package_serial_source_review_v1" and review["reviewer"] == "/root"
            and review["passed"] is True and review["blockers"] == [] and review["spec"] == spec_ref
            and review["source_sha256"] == spec["source_sha256"] and review["launch_authorized"] is True,
            "future exact independent root admission required")
    return names


def native_environment(case, *, arm="baseline", compile_allowed=False):
    require(arm in ("baseline", "candidate"), "exact package sampler arm")
    return {"HOME": "/home/ubuntu", "LANG": "C.UTF-8", "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": "0", "PYTHONPATH": str(Q) + ":" + str(ROOT), "HF_HOME": "/mnt/nvme/hf",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/home/ubuntu/venvs/vllm_neuron/bin:/opt/aws/neuron/bin:/usr/local/bin:/usr/bin:/bin",
        "NEURON_COMPILED_ARTIFACTS": case["artifact"], "NEURON_COMPILE_CACHE_URL": case["compiler_cache"],
        "BASE_COMPILE_WORK_DIR": case["compiler_workdir"], "NEURON_RT_VISIBLE_CORES": "0-1",
        "NEURON_QWEN35": "1", "NEURON_ON_DEVICE_SAMPLING_DISABLED": "1",
        "NEURON_QWEN35_ALLOW_COMPILE": "1" if compile_allowed else "0",
        "NEURON_CPU_SAMPLER_FAST": "0" if arm == "baseline" else "1", "TRN1_COMPILE_WORKERS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1", "XLA_DISABLE_FUNCTIONALIZATION": "1"}


RUNTIME_FILES = ("contract.py", "io.py", "history.py", "service.py", "adapter.py", "run.py", "wiring.py",
    "readiness.py", "budget.py", "final_audit.py", "__init__.py", "source_files.json", "source_text.py")
STARTUP_FILES = tuple(str(SDK / "vllm" / name) for name in (
    "config/parallel.py", "engine/arg_utils.py", "envs.py", "v1/engine/core.py", "v1/engine/utils.py",
    "executor/uniproc_executor.py", "executor/executor_base.py", "worker/worker_base.py", "utils/__init__.py", "entrypoints/openai/api_server.py")) + tuple(
    str(PACKAGE / "vllm_neuron" / name) for name in ("platform.py", "worker/neuron_worker.py", "worker/neuronx_distributed_model_runner.py", "worker/neuronx_distributed_model_loader.py", "models/qwen35/serve.py"))


def host_sources(spec, delivery):
    from .io import read, ref
    result = {path: row["sha256"] for path, row in delivery["files"].items()}
    # Frozen science explicitly supplies its complete 60-input and 39-capture closures.
    for name in ("input_sources.json", "capture_sources.json"):
        path = Q / "qwen35_package_full_validation_v2" / name
        require(str(path) in result, "science input manifest must be in frozen delivery")
        values = read({"path": str(path), "sha256": result[str(path)]})
        values = {p: v["sha256"] if isinstance(v, dict) else v for p, v in values.items()}
        result = merge_sources(result, values)
    for path in (*[str(HERE / name) for name in RUNTIME_FILES], *STARTUP_FILES,
                 str(Q / "qwen35_sampler_history_supplement_v1.py"), str(Q / "qwen35_sampler_torch_namespace_sources_v1.py")):
        require(path in spec["source_sha256"], "runtime/startup source must be independently pinned")
        require(path not in result or result[path] == spec["source_sha256"][path], "frozen source conflict")
        result[path] = spec["source_sha256"][path]
    require(all(spec["source_sha256"].get(p) == sha for p, sha in result.items()), "complete authenticated host source subset")
    return result


def control_root(spec):
    return Path("/mnt/nvme/neuron-work") / ("qw-" + spec["run_name"])


IPC_PATH_LIMIT = 107
IPC_UUID = "00000000-0000-4000-8000-000000000000"
SERVER_PHASES = tuple((arm, phase) for arm, phase in PHASES if phase in ("capture", "quality", "http"))


def server_temporary_roots(spec):
    # Only RPC roots admit sockets; ordinary TMPDIR paths are separate.
    return tuple(Path(spec["output_root"]) / arm / phase / "rpc" for arm, phase in SERVER_PHASES)


def ipc_paths(spec):
    units(spec["run_name"])
    roots = server_temporary_roots(spec)
    require(len(roots) == len(set(roots)) == 6 and control_root(spec) != Path(spec["output_root"]), "six distinct RPC roots")
    result = []
    for (arm, phase), root in zip(SERVER_PHASES, roots):
        path = root / IPC_UUID
        size = len(str(path).encode("utf-8"))
        require(size <= IPC_PATH_LIMIT, "vLLM IPC path exceeds 107 bytes: " + str(path))
        result.append(dict(arm=arm, phase=phase, path=str(path), path_bytes=size, uuid_bytes=36, limit_bytes=IPC_PATH_LIMIT))
    return result


def phase_environment(spec, phase, environment):
    require((phase["arm"], phase["phase"]) in PHASES and phase["control"] == str(control_root(spec) / (phase["arm"] + "-" + phase["phase"])),
            "exact declared phase temporary directory")
    require("TMPDIR" not in environment and "VLLM_RPC_BASE_PATH" not in environment, "no inherited temporary/RPC path")
    result = dict(environment, TMPDIR=str(Path(phase["control"]) / "tmp"))
    if (phase["arm"], phase["phase"]) in SERVER_PHASES:
        ipc_paths(spec)
        directory = Path(spec["output_root"]) / phase["arm"] / phase["phase"]
        require(phase["directory"] == str(directory), "exact package server directory")
        result["TMPDIR"] = str(directory / "tmp")
        result["VLLM_RPC_BASE_PATH"] = str(directory / "rpc")
    return result


def budget():
    parts = {key: LIMITS[key] for key in ("model_payload_bytes", "capture_payload_bytes", "all_logs_bytes", "all_metadata_bytes", "temporary_and_directory_bytes")}
    require(sum(parts.values()) == LIMITS["total_allocated_output_bytes"], "original 2 GiB science budget retained")
    return parts
