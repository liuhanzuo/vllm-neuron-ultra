"""Finite serial driver source with explicit externally guarded execution ports.

No CLI, queue, admission, systemd policy, or resource allocation is installed.
run() is runnable only after root supplies GuardedPorts. Ports must run model
and replay leaves in separate admitted processes, and servers in fresh owned
contexts. NativeClients below calls the real original clients directly; it
never calls orchestrate.run_case or opens historical case-name HTTP outputs.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import uuid

from qwen35_sampler_pair_protocol_v1 import (
    ARMS, BASE, ROOT, SAMPLER_GATES, check_capture_accounting, compare_full_http,
    expected_case, finite_protocol, host_identity, read_small, require,
)
from qwen35_sampler_full_checks_v1 import check_model_metadata, check_quality_pair, original_validators
from qwen35_sampler_worker_v1 import nvme_destination, worker_name

MODEL_FILES = {"state_copy": "state-copy.json", "numeric": "real-logits-validation.json",
               "cold": "cold-buckets.json", "long": "long-decode-validation.json"}


@dataclass(frozen=True)
class GuardedPorts:
    # guard(stage, run_id) authenticates current exclusive owner, resources and
    # frozen sources. No prior resource observation counts as admission.
    guard: object
    # validate_reuse(case) checks the original graph source map/config/artifact
    # without editing/recompiling it and returns a hashed receipt.
    validate_reuse: object
    # model(arm, phase, dir, run_id) runs the unchanged full scientific body.
    model: object
    # vectors(arm, historical_dir, current_dir, run_id) compares all203 exactly.
    vectors: object
    # server(...) is an owned context manager; yields an authenticated host
    # attachment receipt AFTER the requested module-level worker was loaded.
    server: object
    # sampler_replay(arm, capture_dir, run_id) authenticates all finite packets
    # and calls the native replay leaf. controls(run_id) runs all14 controls.
    sampler_replay: object
    controls: object
    # Independent root-owned observation, queried while alive and again AFTER
    # context exit. The server port cannot self-certify its own cleanup.
    server_audit: object


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def server_arguments(case, arm, phase, overrides, port=8010):
    """Same frozen server entry and case flags, plus the stable host worker FQN.

    This is not a launch/admission spec. The guarded server port must add the
    original environment and research directory to PYTHONPATH in BOTH arms,
    keep DEBUG/thread settings identical, redirect logs to fresh NVMe paths,
    and pass a fresh capture directory only for correctness_capture.
    """
    require(case == expected_case() and port == 8010, "original client endpoint/case must be retained")
    return [str(ROOT / "server_with_qwen35_split_v4.py"),
            "--model", case["model_path"], "--served-model-name", case["served_name"],
            "--tensor-parallel-size", "2", "--max-model-len", "4096", "--max-num-seqs", "4",
            "--no-enable-prefix-caching", "--host", "127.0.0.1", "--port", str(port),
            "--additional-config", json.dumps({"override_neuron_config": overrides}),
            "--worker-cls", worker_name(arm, phase)]


class NativeClients:
    """Real original HTTP clients, lazy imports only after root admission."""
    def smoke(self, case, output):
        import orchestrate
        result = orchestrate.quality(case)
        write_new(output, result)
        return result

    def diagnostic(self, case, output):
        from benchmark_workloads import inputs
        from bench_http import run_round
        prompt = inputs(case)[128]
        sampling = {"temperature": .7, "top_p": .9, "top_k": 40}
        # Five extra diagnostic requests /160 output tokens. Full required
        #124-request HTTP runs remain separate and unchanged below.
        waves = [run_round(8010, case["served_name"], [prompt] * n, 32, False, **sampling) for n in (1, 4)]
        rows = [row for wave in waves for row in wave["requests"]]
        require(len(rows) == 5 and all(row["usage"]["prompt_tokens"] == 128
                and row["usage"]["completion_tokens"] == 32 for row in rows), "diagnostic token budget mismatch")
        result = {"complete": True, "waves": waves, "requests": 5, "output_tokens": 160,
                  "sampling": sampling, "timing_eligible": False, "full_model_qualified": False}
        write_new(output, result)
        return result

    def quality(self, case, output):
        from evaluate_quality import evaluate
        require(not output.exists(), "quality result reuse forbidden")
        return evaluate(case, output)

    def initial(self, case, output):
        from bench_http import benchmark
        from orchestrate import prompt_ids
        prompt = prompt_ids(case, "请详细介绍一下法国的首都。")
        result = benchmark(8010, case["served_name"], [prompt], 4, output_tokens=120, rounds=3)
        write_new(output, result)
        return result

    def workloads(self, case, output):
        from benchmark_workloads import evaluate
        require(not output.exists(), "workload result reuse forbidden")
        return evaluate(case, output)


def authenticated(receipt, run_id, **expected):
    require(receipt["run_id"] == run_id and receipt["returncode"] == 0
            and receipt["fresh_execution"] is True, "unauthenticated/stale execution receipt")
    require(all(receipt[k] == value for k, value in expected.items()), "execution phase/arm mismatch")


def owner_identity(owner):
    require(set(owner) == {"unit", "pid", "start_ticks", "invocation_id"}
            and isinstance(owner["unit"], str) and owner["unit"]
            and type(owner["pid"]) is int and owner["pid"] > 0
            and type(owner["start_ticks"]) is int and owner["start_ticks"] > 0
            and isinstance(owner["invocation_id"], str) and len(owner["invocation_id"]) == 32,
            "owner PID/start/invocation identity incomplete")


def root_observation(observation, run_id, root_owner, source_sha256):
    require(observation["run_id"] == run_id and observation["root_owner"] == root_owner,
            "observation came from a different run/root owner")
    require(observation["observer_role"] == "root_owner_auditor"
            and observation["observer_source"] in source_sha256
            and source_sha256[observation["observer_source"]] == observation["observer_source_sha256"],
            "root observer source not pinned")
    require(type(observation["observed_unix_ns"]) is int and observation["observed_unix_ns"] > 0,
            "missing actual observation time")


def attached_worker(receipt, live, identity):
    # No child/server exit code is accepted as a live-readiness claim.
    require(live["observation_kind"] == "live_ready" and live["exit_observed"] is False
            and "returncode" not in live and live["server_active"] is True and live["ready"] is True,
            "live server was incorrectly represented as exited")
    owner_identity(live["server_owner"])
    require(live["proc_owner_matches"] is True and live["invocation_matches"] is True
            and live["worker_in_server_cgroup"] is True, "root did not observe the live worker owner")
    require(type(live["worker_owner"]["start_ticks"]) is int and live["worker_owner"]["start_ticks"] > 0
            and receipt["pid"] == live["worker_owner"]["pid"], "worker PID/start identity mismatch")
    require(receipt["arm"] == identity["arm"] and receipt["phase"] == identity["phase"]
            and receipt["worker_class"] == worker_name(identity["arm"], identity["phase"])
            and receipt["host_identity_sha256"] == identity["sha256"], "configured worker was not actually loaded")
    require(receipt["operator_replaced"] == (identity["arm"] == "candidate")
            and receipt["observation_enabled"] == identity["observation_enabled"]
            and receipt["debug_behavior"] == "unchanged", "actual worker attachment mismatch")


def cleaned_server(terminal, live):
    require(terminal["observation_kind"] == "terminal_cleanup" and terminal["server_active"] is False
            and terminal["remaining_processes"] == [] and terminal["cleanup_complete"] is True,
            "root has not observed complete server cleanup")
    if live is not None:
        require(terminal["server_owner"] == live["server_owner"] and terminal["exit_observed"] is True
                and terminal["stop_requested"] is True and terminal["stop_outcome_accepted"] is True,
                "same live invocation was not observed to terminate")
        status = terminal["systemd_exit_status"]
        require(type(status["ExecMainCode"]) is int and type(status["ExecMainStatus"]) is int
                and isinstance(status["Result"], str), "actual server exit status missing")
        require(terminal["observed_unix_ns"] >= live["observed_unix_ns"], "terminal observation predates live owner")


def check_initial_pair(left, right):
    from qwen35_gdn_serving_checks_v2 import valid_rate
    for report in (left, right):
        require(report["batch"] == 4 and report["sampling"] == {} and report["output_tokens_per_request"] == 120
                and len(report["rounds"]) == 3 and len(report["prompt_inputs"]) == 1, "initial HTTP protocol changed")
    require(left["prompt_inputs"] == right["prompt_inputs"], "initial prompt mismatch")
    def groups(report):
        return [r[k] for r in report["rounds"] for k in ("single", "concurrent")] + [report["streaming"]]
    for index, (a, b) in enumerate(zip(groups(left), groups(right))):
        count = 4 if index == 6 or index % 2 else 1
        valid_rate(a); valid_rate(b)
        require(len(a["requests"]) == len(b["requests"]) == count, "initial HTTP group incomplete")
        for x, y in zip(a["requests"], b["requests"]):
            expected_usage = {"prompt_tokens": len(left["prompt_inputs"][0]), "completion_tokens": 120,
                              "total_tokens": len(left["prompt_inputs"][0]) + 120}
            require(x["usage"] == y["usage"], "initial usage differs between arms")
            for row in (x, y):
                require(isinstance(row["usage"], dict) and all(
                    type(row["usage"].get(key)) is int and row["usage"][key] == count
                    for key, count in expected_usage.items()), "initial token budget changed")
            require(x["text"] == y["text"], "initial greedy text changed")


def run(ports, output_root, source_sha256, clients=None):
    """One finite attempt, serial; any missing/failed gate stops before timing."""
    require(isinstance(ports, GuardedPorts) and all(callable(getattr(ports, k)) for k in ports.__dataclass_fields__), "root execution ports not bound")
    output_root = nvme_destination(output_root)
    require(not output_root.exists(), "fresh run output required; no resume or historical reuse")
    run_id = str(uuid.uuid4())
    root_owner = None

    def guard(stage):
        nonlocal root_owner
        result = ports.guard(stage, run_id)
        require(result["admitted"] is True and result["exclusive"] is True and result["run_id"] == run_id,
                "root has not admitted this exclusive stage")
        owner_identity(result["root_owner"])
        if root_owner is None:
            root_owner = result["root_owner"]
        require(result["root_owner"] == root_owner, "exclusive root owner changed")
        return result

    admission = guard("before_output_creation")
    output_root.mkdir(parents=False)
    case, validators = expected_case(), original_validators()
    clients = NativeClients() if clients is None else clients
    reuse = ports.validate_reuse(case)
    require(reuse["validated"] is True and reuse["compile_permitted"] is False
            and reuse["provenance_unchanged"] is True and len(reuse["sha256"]) == 64, "frozen graph reuse not validated")
    write_new(output_root / "contract.json", {"run_id": run_id, "case": case, "contract": finite_protocol(),
                                               "host_source_sha256": source_sha256, "graph_reuse": reuse,
                                               "root_admission": admission})

    @contextmanager
    def serving(arm, phase, directory, identity):
        manager = ports.server(arm, phase, directory, worker_name(arm, phase), run_id, identity)
        live = None
        try:
            with manager as attachment:
                live = ports.server_audit(manager, "live", run_id)
                root_observation(live, run_id, root_owner, source_sha256)
                attached_worker(attachment, live, identity)
                write_new(directory / "server-live-root-audit.json", live)
                yield attachment
        finally:
            # This runs after __exit__, including exceptions from HTTP clients.
            terminal = ports.server_audit(manager, "terminal", run_id)
            root_observation(terminal, run_id, root_owner, source_sha256)
            cleaned_server(terminal, live)
            write_new(directory / "server-terminal-root-audit.json", terminal)
    directories = {}
    for arm in ARMS:
        root = output_root / arm
        root.mkdir()
        for name in ("model", "capture", "quality", "http"):
            (root / name).mkdir()
        (root / "capture" / "packets").mkdir()
        write_new(root / "model" / "case.json", case)
        directories[arm] = root
        for phase, filename in MODEL_FILES.items():
            guard(arm + ":" + phase)
            receipt = ports.model(arm, phase, root / "model", run_id)
            authenticated(receipt, run_id, arm=arm, phase=phase)
            check_model_metadata(read_small(root / "model" / filename), phase, validators)
            write_new(root / "model" / (phase + "-execution.json"), receipt)
        guard(arm + ":vectors203")
        vectors = ports.vectors(arm, BASE, root / "model", run_id)
        authenticated(vectors, run_id, arm=arm, phase="vectors203")
        require(vectors["complete"] is True and vectors["frames"] == vectors["identical_vectors"] == 203, "all203 raw vectors must reproduce the qualified baseline")
        write_new(root / "model" / "vectors203.json", vectors)
    for arm in ARMS:
        root = directories[arm]
        guard(arm + ":correctness_capture")
        identity = host_identity(arm, "correctness_capture", source_sha256)
        with serving(arm, "correctness_capture", root / "capture", identity) as attached:
            require(attached["observation_enabled"] is True, "capture worker was not loaded")
            clients.diagnostic(case, root / "diagnostic-http.json")
        capture_receipt = read_small(root / "capture" / "packets" / "capture-receipt.json")
        check_capture_accounting(capture_receipt)
        guard(arm + ":sampler_replay")
        replay = ports.sampler_replay(arm, root / "capture" / "packets", run_id)
        authenticated(replay, run_id, arm=arm, phase="sampler_replay")
        require(replay["passed"] is True and replay["packets"] == capture_receipt["packets"]
                and replay["rows"] == capture_receipt["captured_rows"] and replay["all_own_tokens_and_rng_reproduced"] is True
                and replay["all_distribution_gates_passed"] is True and replay["fast_path_rows"] > 0, "complete selected sampler replay failed")
        write_new(root / "sampler-replay.json", replay)
    guard("sampler_rejection_controls")
    controls = ports.controls(run_id)
    authenticated(controls, run_id, phase="sampler_controls")
    require(controls["passed"] is True and [r["control"] for r in controls["controls"]] == SAMPLER_GATES["controls"]
            and all(r["passed"] for r in controls["controls"]), "native rejection controls incomplete")
    write_new(output_root / "sampler-controls.json", controls)
    qualities, reports, initials, measured_hosts = {}, {}, {}, {}
    for arm in ARMS:
        root = directories[arm]
        guard(arm + ":quality96")
        identity = host_identity(arm, "measured", source_sha256)
        with serving(arm, "measured", root / "quality", identity) as attached:
            require(attached["observation_enabled"] is False, "quality worker unexpectedly observed")
            smoke = clients.smoke(case, root / "quality" / "smoke.json")
            require(smoke["passed"] and smoke["all_identical_rows_match"]
                    and all(len(smoke[k]) == 4 for k in ("serial", "concurrent", "identical_prompt_outputs")), "smoke protocol incomplete")
            qualities[arm] = clients.quality(case, root / "quality" / "quality96.json")
    suite_path = ROOT / "quality-suite.json"
    quality_pair = check_quality_pair(qualities["baseline"], qualities["candidate"], read_small(BASE / "quality96.json"),
                                     read_small(suite_path), hashlib.sha256(suite_path.read_bytes()).hexdigest(), validators)
    write_new(output_root / "quality-pair.json", quality_pair)
    for arm in ARMS:
        root = directories[arm]
        guard(arm + ":matched_http")
        identity = host_identity(arm, "measured", source_sha256)
        with serving(arm, "measured", root / "http", identity) as attached:
            require(attached["observation_enabled"] is False and attached["debug_behavior"] == "unchanged", "timing contamination")
            measured_hosts[arm] = {**identity, **attached, "graph_reuse_receipt_sha256": reuse["sha256"], "frozen_model_case": case}
            initials[arm] = clients.initial(case, root / "http" / "initial120.json")
            reports[arm] = clients.workloads(case, root / "http" / "workloads.json")
        write_new(root / "http" / "host.json", measured_hosts[arm])
    check_initial_pair(initials["baseline"], initials["candidate"])
    paired = compare_full_http(reports["baseline"], reports["candidate"], case, reports["baseline"]["inputs"],
                               measured_hosts["baseline"], measured_hosts["candidate"])
    guard("before_final_receipt")
    final_reuse = ports.validate_reuse(case)
    require(final_reuse["validated"] is True and final_reuse["compile_permitted"] is False
            and final_reuse["provenance_unchanged"] is True and final_reuse["sha256"] == reuse["sha256"],
            "frozen artifact changed during paired attempt")
    result = {"run_id": run_id, "execution_finished": True, "gates_checked_by_driver": True,
              "paired_http": paired, "full_model_qualified": False,
              "independent_final_audit_required": True, "automatic_best_promotion": False}
    write_new(output_root / "driver-result.json", result)
    return result
