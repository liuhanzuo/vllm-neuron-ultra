"""Actual serial GuardedPorts adapter, with no selected current bindings.

The outer CLI owns the original descriptor. A dedicated coordinator supplies
stable root-owner identity; separately acknowledged phase supervisors Popen/wait
native leaves. Full history is retained independently of the small host map.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid

from qwen35_sampler_runtime_contract_v11 import (COORDINATOR_PROPERTIES, LIMITS, LOCK_IDENTITY, PYTHON, Q, ROOT, STARTUP_FILES,
    control_root, digest, native_environment, phase_environment, phase_properties, require, units)
from qwen35_sampler_runtime_io_v11 import (RetainedLock, descriptor, identity, loaded_local_sources, once, process_api,
    raw, read, read_if_present, ref, resources, service_owner, storage, wait_until)
from qwen35_sampler_runtime_history_v11 import Authority, Bundle, check_running
from qwen35_sampler_runtime_service_v11 import (CLI, Service, argv, expected_phase, properties, supervise,
                                                validate_api_failure, API_ERROR_MESSAGE)
from qwen35_sampler_pair_protocol_v1 import BASE, expected_case, host_identity


def admit_shape(admission, spec, spec_ref, review_ref):
    require(set(admission) == {"schema", "spec", "review", "attempt_id", "outer_owner", "retained_fd", "initial_descriptor_proof",
            "lock_identity", "current_serial_catalog", "history_overlay", "resolved_plan_sha256", "history_pair", "freshness",
            "units", "source_map_sha256", "host_source_sha256", "control_root", "output_root", "started_monotonic",
            "deadline_monotonic", "runtime_admitted", "automatic_successor", "self_exit_claimed"}
            and admission["schema"] == "qwen35_sampler_runtime_admission_v1", "exact runtime admission")
    require(admission["spec"] == spec_ref and admission["review"] == review_ref
            and admission["current_serial_catalog"] == spec["current_serial_catalog"]
            and admission["history_overlay"] == spec["history_overlay"]
            and admission["lock_identity"] == LOCK_IDENTITY and admission["units"] == units(spec["run_name"])
            and admission["source_map_sha256"] == digest(spec["source_sha256"])
            and admission["control_root"] == str(control_root(spec)) and admission["output_root"] == spec["output_root"]
            and admission["runtime_admitted"] is True and admission["automatic_successor"] is False
            and admission["self_exit_claimed"] is False, "original complete scope/lock/source/output binding")
    require(type(admission["retained_fd"]) is int and admission["retained_fd"] >= 0
            and admission["deadline_monotonic"] - admission["started_monotonic"] == LIMITS["outer_wall_seconds"],
            "retained descriptor and finite whole-attempt deadline")
    original = admission["initial_descriptor_proof"]
    require(original["passed"] is True and original["owner"] == admission["outer_owner"]
            and original["retained_fd"] == admission["retained_fd"] and original["lock"] == LOCK_IDENTITY
            and original["before"] == original["after"], "exact initial descriptor proof retained")


def coordinator_service(spec, admission_ref, owner=None):
    return Service(units(spec["run_name"])["coordinator"], argv("coordinator", admission_ref),
                   native_environment(expected_case()), COORDINATOR_PROPERTIES, control_root(spec) / "coordinator.log", owner=owner)


def phase_service(spec, phase, phase_ref, owner=None):
    return Service(phase["unit"], argv("supervisor", phase_ref), native_environment(expected_case()),
                   phase["properties"], phase["log"], parent=units(spec["run_name"])["coordinator"], owner=owner)


def graph_reuse(bundle, case):
    """Original frozen file identities; no model.pt read or graph recompilation."""
    require(case == bundle.delivery["baseline_case"] == expected_case(), "immutable baseline case required")
    wanted = bundle.delivery["graph_identity"]
    require(hashlib.sha256(raw(wanted["provenance_path"])).hexdigest() == wanted["provenance_file_sha256"],
            "original artifact-provenance bytes changed")
    artifact = Path(case["artifact"])
    config_bytes = raw(artifact / "neuron_config.json")
    require(hashlib.sha256(config_bytes).hexdigest() == wanted["neuron_config_sha256"], "original graph config changed")
    provenance = json.loads(raw(wanted["provenance_path"]))
    require(provenance["sha256"] == wanted["frozen_graph_fingerprint"], "immutable graph fingerprint changed")
    require(provenance["sources"] == {p: v["expected_sha256"] for p, v in wanted["source_pins"].items()},
            "complete original graph source map unchanged")
    for path, row in wanted["source_pins"].items():
        require(hashlib.sha256(raw(path, 4 * 1024**2)).hexdigest() == row["expected_sha256"], "frozen graph source changed")
    def original_stat(path):
        p = Path(path); s = p.stat()
        return {"realpath": str(p.resolve()), "device": s.st_dev, "inode": s.st_ino, "size": s.st_size,
                "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}
    observed = {}
    for row in bundle.delivery["large_artifact_metadata"]:
        require(original_stat(row["path"]) == row["expected_identity"], "original large-file identity changed")
        observed[row["path"]] = row["expected_identity"]
    for name, wanted_stat in bundle.delivery["historical_full_payload_stat_metadata"].items():
        path = str(BASE / name)
        require(original_stat(path) == wanted_stat, "original full model reference payload changed")
        observed[path] = wanted_stat
    value = {"case": case, "graph_provenance": provenance, "config_file_sha256": wanted["neuron_config_sha256"],
             "large_file_identities": observed, "all12_graph_routes_bound_by_original_model_pt": True,
             "large_contents_rehashed_here": False}
    return {"validated": True, "compile_permitted": False, "provenance_unchanged": True, "sha256": digest(value), "identity": value}


class LeaseObserver:
    def __init__(self, admission, spec):
        self.admission, self.spec = admission, spec
        self.coordinator = None; self.current = None; self.bundle = None; self.running = None
        self.last_slow = -float("inf"); self.failure = None

    def check(self, *, full=False):
        if self.failure is not None: raise self.failure
        try:
            descriptor(self.admission["outer_owner"], self.admission["retained_fd"])
            require(time.monotonic() < self.admission["deadline_monotonic"], "finite whole-attempt deadline exceeded")
            if self.coordinator is not None: self.coordinator.live()
            now = time.monotonic()
            if full or now - self.last_slow >= LIMITS["monitor_interval_seconds"]:
                resources(self.spec)
                allowed = [self.coordinator.unit] if self.coordinator is not None else []
                # A bootstrap observer authenticates its already running service
                # before a supervisor receipt exists; it did not call start().
                if self.current is not None and (self.current.started or self.current.owner is not None):
                    self.current.observe(); allowed.append(self.current.unit)
                process_api().active_heavy(allowed)
                if self.running is not None:
                    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
                    require(latest.latest_handoff() == self.running["observation"], "later/replaced running handoff detected")
                self.last_slow = now
            if full and self.bundle is not None:
                self.bundle.unchanged()
                for owner in self.bundle.scope["owners"]: process_api().gone(owner)
                loaded_local_sources(self.spec["source_sha256"])
            return {"admitted": True, "exclusive": True}
        except BaseException as error:
            self.failure = error; raise


def load_context(admission_ref):
    admission = read(admission_ref)
    spec_ref, review_ref = admission["spec"], admission["review"]
    spec = read(spec_ref, LIMITS["spec_envelope_bytes"])
    observer = LeaseObserver(admission, spec)
    # Runtime source review must be accepted before imports beyond the stdlib
    # source adapters. Full original lineage is resolved by the coordinator.
    bundle = Bundle(spec_ref, review_ref, observer.check)
    admit_shape(admission, spec, spec_ref, review_ref)
    require(str(control_root(spec) / "admission.json") == admission_ref["path"], "exact fresh admission path")
    return admission, bundle, observer


def worker_bootstrap(phase_ref, *, role):
    phase = read(phase_ref)
    admission, bundle, observer = load_context(phase["admission"])
    spec = bundle.spec; co = coordinator_service(spec, phase["admission"], phase["coordinator"])
    observer.coordinator = co; co.live()
    coordinator_ref = ref(control_root(spec) / "coordinator-owner.json")
    require(read(coordinator_ref) == phase["coordinator"], "actual coordinator owner receipt changed")
    expected = expected_phase(spec, phase["admission"], co.owner, phase["run_id"], phase["arm"], phase["phase"],
                              phase["directory"], phase["host_identity"])
    require(phase == expected and phase_ref["path"] == str(Path(phase["control"]) / "phase-spec.json"), "exact phase spec required")
    require(Path(phase["directory"]).resolve() == Path(phase["directory"])
            and (Path(spec["output_root"]) in Path(phase["directory"]).parents
                 or Path(phase["directory"]) == control_root(spec) / "shared-native_preflight"), "phase output namespace")
    phase_owner = read_if_present(Path(phase["control"]) / "supervisor-owner.json")
    service = phase_service(spec, phase, phase_ref, phase_owner)
    service.live(); observer.current = service
    # The exact, already authenticated current phase must be established
    # before the complete running-handoff reader invokes its active-unit guard.
    observer.running = check_running(bundle, phase["admission"], admission, coordinator_ref, co.owner)
    current = process_api().proc(os.getpid())
    if role == "supervisor":
        require(current == service.owner["process"] and current["argv"] == argv("supervisor", phase_ref), "actual supervisor owner")
    else:
        require(role in ("leaf", "api_bootstrap") and current["ppid"] == service.owner["process"]["pid"]
                and current["argv"] == argv(role, phase_ref) and current["cgroup"] == "0::" + service.group,
                "actual direct native bootstrap child")
        release = wait_until(lambda: read_if_present(Path(phase["control"]) / "child-release.json"), observer.check,
                             LIMITS["startup_ack_seconds"], message="independently observed native child ACK missing")
        require(release == {"phase_spec": phase_ref, "supervisor": service.owner, "child_bootstrap_owner": current,
                            "coordinator": co.owner}, "actual child PID/start/argv acknowledged before native imports")
    for path, sha in admission["host_source_sha256"].items():
        require(bundle.spec["source_sha256"].get(path) == sha and hashlib.sha256(raw(path, 4 * 1024**2)).hexdigest() == sha,
                "actual execution host source changed")
    loaded_local_sources(bundle.spec["source_sha256"])
    env = native_environment(expected_case(), host_identity=phase["host_identity"],
                             host_receipt=Path(phase["directory"]) / "worker-attachment.json" if phase["host_identity"] is not None else None,
                             capture_root=Path(phase["directory"]) / "packets" if phase["phase"] == "capture" else None)
    env = phase_environment(spec, phase, env)
    return phase, {"guard": observer.check, "service": service, "observer": observer, "child_environment": env,
                   "bundle": bundle, "admission": admission}


def native_leaf(phase_ref, *, api=False):
    phase, context = worker_bootstrap(phase_ref, role="api_bootstrap" if api else "leaf")
    guard = context["guard"]; stage = Path(phase["control"])
    guard()
    if api:
        from qwen35_sampler_pair_driver_v1 import server_arguments
        case = expected_case(); arm = phase["arm"]
        host_phase = "correctness_capture" if phase["phase"] == "capture" else "measured"
        args = [PYTHON, *server_arguments(case, arm, host_phase, {"on_device_sampling_config": None, **case["extra_overrides"]})]
        once(stage / "api-exec-intent.json", {"phase_spec": phase_ref, "bootstrap_owner": process_api().proc(os.getpid()),
                                             "argv": args, "environment_sha256": digest(context["child_environment"])})
        os.execvpe(PYTHON, args, context["child_environment"])
        raise AssertionError("exec returned")
    from qwen35_sampler_native_leaves_v2 import run_model_stage, compare_vectors, replay_captures, rejection_controls
    name, directory = phase["phase"], Path(phase["directory"])
    if name == "native_preflight":
        from qwen35_sampler_runtime_preflight_v1 import run
        result = run(guard)
    elif name in ("state_copy", "numeric", "cold", "long"):
        result = run_model_stage(name, directory, guard)
    elif name == "vectors203": result = compare_vectors(BASE, directory, guard)
    elif name == "sampler_replay": result = replay_captures(directory, guard)
    elif name == "sampler_controls": result = rejection_controls(guard)
    else: raise ValueError("undeclared leaf phase")
    guard(); loaded_local_sources(context["bundle"].spec["source_sha256"])
    once(stage / "leaf-result.json", {"schema": "qwen35_sampler_actual_leaf_result_v1", "phase_spec": phase_ref,
        "actual_process": process_api().proc(os.getpid()), "result": result, "self_exit_claimed": False})
    return 0


class Phase:
    def __init__(self, adapter, arm, name, directory, run_id, host=None):
        self.adapter = adapter; self.arm = arm; self.name = name; self.directory = Path(directory); self.run_id = run_id
        self.phase = expected_phase(adapter.spec, adapter.admission_ref, adapter.coordinator.owner,
                                    run_id, arm, name, directory, host)
        self.stage = Path(self.phase["control"]); self.phase_ref = None; self.service = None
        self.child = None; self.api_owner = None; self.attachment = None; self.live_observation = None; self.end = None
        self.startup_proof = None

    def start(self):
        a = self.adapter
        require(a.active is None and not self.stage.exists(), "strictly one fresh phase at a time")
        a.guard("before_phase_" + self.name, self.run_id)
        self.stage.mkdir(); (self.stage / "tmp").mkdir()
        self.phase_ref = once(self.stage / "phase-spec.json", self.phase)
        self.service = phase_service(a.spec, self.phase, self.phase_ref)
        a.active = self; a.observer.current = self.service
        try:
            self.service.start(a.observer.check)
            reported = wait_until(lambda: read_if_present(self.stage / "supervisor-owner.json"), a.observer.check,
                                  LIMITS["startup_ack_seconds"], message="supervisor actual owner receipt missing")
            require(reported == self.service.owner, "supervisor self-report disagrees with independent proc/systemd")
            once(self.stage / "supervisor-release.json", {"phase_spec": self.phase_ref, "supervisor": reported,
                                                          "coordinator": a.coordinator.owner})
            self.child = wait_until(lambda: read_if_present(self.stage / "child-bootstrap-owner.json"), a.observer.check,
                                    LIMITS["startup_ack_seconds"], message="native child actual owner receipt missing")
            api = self.name in ("capture", "quality", "http")
            expected = argv("api_bootstrap" if api else "leaf", self.phase_ref)
            require(self.child == process_api().proc(self.child["pid"]) and self.child["ppid"] == reported["process"]["pid"]
                    and self.child["argv"] == expected and self.child["cgroup"] == "0::" + self.service.group,
                    "independent direct native child PID/start/argv/cgroup mismatch")
            once(self.stage / "child-release.json", {"phase_spec": self.phase_ref, "supervisor": reported,
                "child_bootstrap_owner": self.child, "coordinator": a.coordinator.owner})
        except BaseException:
            self.finish(); raise
        return self

    def api_failure(self):
        if self.name not in ("capture", "quality", "http") or self.service is None or self.child is None:
            return None
        value = read_if_present(self.stage / "api-failure.json")
        if value is not None:
            validate_api_failure(value, self.phase_ref, self.service.owner, self.child)
        return value

    def wait_failed_api(self, failure):
        # Do not race the supervisor's bounded traceback/exit collection with
        # coordinator cleanup. This deadline is owned by the first failure,
        # shares the original45 seconds, and is never renewed by observations.
        while time.monotonic() < failure["deadline_monotonic"]:
            self.adapter.observer.check()
            if process_api().terminal(self.service.observe()): break
            time.sleep(min(.1, max(0., failure["deadline_monotonic"] - time.monotonic())))

    def reject_api_failure(self):
        failure = self.api_failure()
        if failure is not None:
            self.wait_failed_api(failure)
            raise ValueError(API_ERROR_MESSAGE)

    def finish(self):
        try:
            if self.end is not None:
                if self.service is not None and self.service.started:
                    self.service.cleanup() if self.end.get("never_loaded") else self.service.drained()
                return self.end
            if self.service is not None and self.service.started:
                try:
                    failure = self.api_failure()
                    if failure is not None: self.wait_failed_api(failure)
                finally:
                    # A resource/identity/deadline failure during collection
                    # must still enter the unchanged owned-service cleanup.
                    self.end = self.service.cleanup()
                    if self.child is not None: process_api().gone(self.child)
                    once(self.stage / "root-terminal.json", self.end)
            return self.end
        finally:
            if self.end is not None and self.end.get("cleanup_complete") is True:
                self.adapter.observer.current = None; self.adapter.active = None

    def leaf(self):
        self.start()
        try:
            code = wait_until(lambda: read_if_present(self.stage / "child-exit.json"), self.adapter.observer.check,
                              int(self.phase["properties"]["RuntimeMaxSec"]), message="actual child Popen exit missing")
            require(code["phase_spec"] == self.phase_ref and code["supervisor"] == self.service.owner
                    and code["child_bootstrap_owner"] == self.child and code["actual_popen_wait_observed"] is True
                    and code["child_returncode"] == 0 and code["stop_requested"] is False, "actual native child failed")
            result = read_if_present(self.stage / "leaf-result.json")
            require(result is not None and result["phase_spec"] == self.phase_ref
                    and result["actual_process"] == self.child and result["self_exit_claimed"] is False,
                    "real native result does not match acknowledged producer")
            # Child exit is observed before systemd main exit; wait for the
            # supervisor's independent actual exit before issuing cleanup.
            wait_until(lambda: True if process_api().terminal(self.service.observe()) else None, self.adapter.observer.check,
                       LIMITS["owned_cleanup_seconds"], message="supervisor did not exit after child wait")
        finally:
            self.finish()
        status = self.end["systemd_exit_status"]
        require(status == {"ExecMainCode": 1, "ExecMainStatus": 0, "Result": "success"}, "actual native service failed")
        return {**result["result"], "run_id": self.run_id, "arm": self.arm, "phase": self.name,
                "returncode": code["child_returncode"], "fresh_execution": True,
                "actual_child_exit": code, "actual_service_terminal": self.end}

    def api_ready(self):
        self.reject_api_failure()
        self.adapter.observer.check(); self.service.live()
        intent = read_if_present(self.stage / "api-exec-intent.json")
        if intent is None: return None
        require(intent["phase_spec"] == self.phase_ref and intent["bootstrap_owner"] == self.child, "exact API exec transition")
        current = process_api().proc(self.child["pid"])
        require(current is not None and current["start_ticks"] == self.child["start_ticks"]
                and current["ppid"] == self.child["ppid"] and current["cgroup"] == self.child["cgroup"],
                "same real child identity must survive exact API exec transition")
        if current["argv"] == self.child["argv"]:
            return None  # Intent is durable immediately before execvpe.
        require(current["argv"] == intent["argv"], "same real child must exec exact original API command")
        with Path("/proc", str(current["pid"]), "environ").open("rb") as stream:
            environ_raw = stream.read(256 * 1024 + 1)
        require(len(environ_raw) <= 256 * 1024, "bounded actual API initial environment")
        actual_environment = dict(row.decode().split("=", 1) for row in environ_raw.split(b"\0") if row)
        require(digest(actual_environment) == intent["environment_sha256"], "actual API environment differs from exact clean Popen/exec environment")
        require(not any(key.startswith("VLLM_DP_") or key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "VLLM_SERVER_DEV_MODE",
                    "TRN1_CPU_SAMPLER_FAST", "NEURON_CPU_SAMPLER_FAST", "TRN1_SAMPLER_DIAGNOSTIC_DIR")
                    for key in actual_environment), "no DP or legacy sampler environment override")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8010/health", timeout=2) as response:
                body = response.read(4097)
                require(response.status == 200 and len(body) <= 4096, "actual bounded API readiness response")
        except (urllib.error.URLError, TimeoutError): return None
        # Frozen record_attachment writes directly to its fresh path. /health
        # becomes ready only after worker initialization finishes, so read the
        # completed attachment after readiness, never parse an in-flight dump.
        attachment = read_if_present(self.directory / "worker-attachment.json")
        require(attachment is not None, "healthy API has no completed worker attachment")
        worker = process_api().proc(attachment["pid"])
        require(worker is not None and worker["cgroup"] == "0::" + self.service.group, "actual loaded worker outside owned group")
        self.api_owner, self.attachment = current, attachment
        if self.startup_proof is None:
            proof = {"dp_size_derived_from_pinned_source_clean_environment_and_argv": 1,
                     "dp_rank_derived_from_pinned_source_clean_environment_and_argv": 0,
                     "parallel_config_object_directly_observed": False, "observer_seen_or_call_id_directly_observed": False,
                     "actual_api_initial_environment_sha256": digest(actual_environment),
                     "dp_environment_overrides_present": False, "actual_api_argv": current["argv"],
                     "startup_source_sha256": {path: self.adapter.spec["source_sha256"][path] for path in STARTUP_FILES}}
            if self.name == "capture":
                packet_dir = self.directory / "packets"
                files = sorted(p.name for p in packet_dir.iterdir())
                require(files == ["writer-owner.json"], "no written capture packet/receipt may precede diagnostic HTTP")
                require(read_if_present(packet_dir / "writer-owner.json") == {"pid": attachment["pid"], "arm": self.arm,
                            "phase": "correctness_capture"}, "actual capture writer owner mismatch")
                proof.update(written_packets_before_diagnostic=0, receipt_before_diagnostic=False,
                             zero_seen_count_claimed=False, actual_B1_B4_each16_still_required=True)
            self.startup_proof = proof
            once(self.stage / "startup-config-proof.json", proof)
        self.reject_api_failure()
        return {"attachment": attachment, "worker": worker, "api_owner": current}

    def __enter__(self):
        # Never adopt an unidentified endpoint from another process.
        require(not endpoint_open(), "original endpoint already occupied")
        self.start()
        try:
            wait_until(self.api_ready, self.adapter.observer.check, expected_case()["startup_wait_limit_s"],
                       message="original server startup/worker attachment failed", interval=.5)
            return self.attachment
        except BaseException:
            self.finish(); raise

    def __exit__(self, kind, value, tb):
        try:
            self.reject_api_failure()
            if self.service is not None and self.service.started:
                self.service.live()
                once(self.stage / "stop-request.json", {"phase_spec": self.phase_ref, "supervisor": self.service.owner,
                                                        "coordinator": self.adapter.coordinator.owner})
                self.service.stop_requested = True
                wait_until(lambda: True if process_api().terminal(self.service.observe()) else None, self.adapter.observer.check,
                           LIMITS["owned_cleanup_seconds"], message="API supervisor graceful stop did not complete")
        finally: self.finish()
        return False


def endpoint_open():
    import socket
    sock = socket.socket(); sock.settimeout(.2)
    try: return sock.connect_ex(("127.0.0.1", 8010)) == 0
    finally: sock.close()


class Adapter:
    def __init__(self, admission_ref, admission, bundle, observer, coordinator):
        self.admission_ref, self.admission, self.bundle, self.observer, self.coordinator = admission_ref, admission, bundle, observer, coordinator
        self.spec = bundle.spec; self.active = None; self.run_id = None

    def guard(self, stage, run_id):
        require(type(run_id) is str and run_id, "explicit run identity")
        # The additional preflight uses the outer attempt UUID. The frozen
        # driver later generates its own UUID; bind that independently once.
        if run_id != self.admission["attempt_id"]:
            if self.run_id is None: self.run_id = run_id
            require(run_id == self.run_id, "paired driver run identity changed")
        if self.active is not None: self.active.reject_api_failure()
        self.observer.check(full=True)
        return {"admitted": True, "exclusive": True, "run_id": run_id,
                "root_owner": copy.deepcopy(self.coordinator.owner["service_owner"]), "stage": stage,
                "complete_current_serial_catalog": self.spec["current_serial_catalog"], "spec": self.admission["spec"]}

    def validate_reuse(self, case):
        self.observer.check(full=True); return graph_reuse(self.bundle, case)

    def model(self, arm, name, directory, run_id): return Phase(self, arm, name, directory, run_id).leaf()

    def vectors(self, arm, before, after, run_id):
        require(Path(before) == BASE, "qualified historical baseline only")
        return Phase(self, arm, "vectors203", after, run_id).leaf()

    def sampler_replay(self, arm, directory, run_id): return Phase(self, arm, "sampler_replay", directory, run_id).leaf()

    def controls(self, run_id):
        return Phase(self, "shared", "sampler_controls", Path(self.spec["output_root"]) / "baseline" / "capture", run_id).leaf()

    def server(self, arm, phase, directory, worker_fqn, run_id, host):
        from qwen35_sampler_worker_v1 import worker_name
        require(worker_fqn == worker_name(arm, phase) and host == host_identity(arm, phase, self.bundle.host), "exact original worker/host contract")
        name = "capture" if phase == "correctness_capture" else Path(directory).name
        require(name in ("capture", "quality", "http"), "finite server phase")
        return Phase(self, arm, name, directory, run_id, host)

    def server_audit(self, manager, state, run_id):
        require(isinstance(manager, Phase) and manager.adapter is self and manager.run_id == run_id, "actual owned manager required")
        source = str(Path(__file__).resolve())
        base = {"run_id": run_id, "root_owner": copy.deepcopy(self.coordinator.owner["service_owner"]),
                "observer_role": "root_owner_auditor", "observer_source": source,
                "observer_source_sha256": self.bundle.host[source], "observed_unix_ns": time.time_ns()}
        if state == "live":
            live = manager.api_ready(); require(live is not None, "actual live readiness missing")
            result = {**base, "observation_kind": "live_ready", "exit_observed": False,
                "server_active": True, "ready": True, "server_owner": manager.service.owner["service_owner"],
                "api_process_owner": live["api_owner"], "proc_owner_matches": True, "invocation_matches": True,
                "worker_in_server_cgroup": True, "worker_owner": {**live["worker"], "start_ticks": int(live["worker"]["start_ticks"])},
                "startup_config_proof": manager.startup_proof}
            manager.live_observation = result; return result
        if state == "terminal" and (manager.service is None or not manager.service.started
                                    or manager.end is not None and manager.end.get("never_loaded")):
            unit = manager.phase["unit"]; value = properties(unit)
            require(value["LoadState"] == "not-found" and value["InvocationID"] == "", "never-started unit unexpectedly exists")
            group = process_api().empty_group(process_api().GROUP_PREFIX + unit + ".service")
            return {**base, "observation_kind": "terminal_cleanup", "server_active": False,
                    "remaining_processes": [], "cleanup_complete": True, "exit_observed": False,
                    "server_never_started": True, "unstarted_group_observation": group}
        require(state == "terminal" and manager.end is not None and manager.end.get("cleanup_complete") is True,
                "independently observed actual terminal required")
        # Observe again after __exit__, not just trust the context's cached flag.
        observed = manager.service.drained()
        code = read_if_present(manager.stage / "child-exit.json")
        accepted = code is not None and code["actual_popen_wait_observed"] is True and code["phase_spec"] == manager.phase_ref \
            and code["supervisor"] == manager.service.owner and code["child_bootstrap_owner"] == manager.child \
            and code["stop_requested"] is True and code["child_returncode"] in (0, -signal.SIGTERM) \
            and observed["systemd_exit_status"] == {"ExecMainCode": 1, "ExecMainStatus": 0, "Result": "success"}
        return {**base, **observed, "observation_kind": "terminal_cleanup", "server_active": False,
                "stop_requested": manager.service.stop_requested, "stop_outcome_accepted": accepted,
                "actual_child_exit": code}

    def ports(self):
        from qwen35_sampler_pair_driver_v1 import GuardedPorts
        return GuardedPorts(self.guard, self.validate_reuse, self.model, self.vectors, self.server,
                            self.sampler_replay, self.controls, self.server_audit)


class FailureCheckedClients:
    """Original clients with failure checks at operation boundaries only.

    Never accepts an in-flight operation's results after a latched API error.
    It does not claim instant cancellation of an already submitted HTTP request;
    the supervisor's original finite cleanup window owns process termination.
    """
    METHODS = ("smoke", "diagnostic", "quality", "initial", "workloads")

    def __init__(self, adapter, original):
        self.adapter, self.original = adapter, original

    def __getattr__(self, name):
        require(name in self.METHODS, "only original finite client operations")
        def operation(*args, **kwargs):
            require(self.adapter.active is not None, "client requires exact active owned phase")
            phase = self.adapter.active
            phase.reject_api_failure()
            result = getattr(self.original, name)(*args, **kwargs)
            require(self.adapter.active is phase, "client active phase changed")
            phase.reject_api_failure()
            return result
        return operation


def coordinator(admission_ref):
    admission, bundle, observer = load_context(admission_ref)
    spec = bundle.spec; co = coordinator_service(spec, admission_ref)
    co.live()
    require(co.owner["process"] == process_api().proc(os.getpid()), "actual coordinator owns this root observation")
    observer.coordinator = co
    coordinator_ref = once(control_root(spec) / "coordinator-owner.json", co.owner)
    ack = wait_until(lambda: read_if_present(control_root(spec) / "running-ack.json"), observer.check,
                     LIMITS["root_running_ack_seconds"], message="external root running-handoff ACK missing")
    observed_running = check_running(bundle, admission_ref, admission, coordinator_ref, co.owner)
    require(ack == {"admission": admission_ref, "coordinator_owner": coordinator_ref, "coordinator": co.owner,
                    "running": observed_running}, "actual independent outer/root handoff ACK")
    observer.running = observed_running
    bundle.resolve(); observer.bundle = bundle
    require(bundle.resolved["plan_sha256"] == admission["resolved_plan_sha256"]
            and bundle.host == admission["host_source_sha256"], "complete resolved history/host closure unchanged")
    saved = read(admission["history_pair"])
    require(saved["resolved_plan_sha256"] == bundle.resolved["plan_sha256"]
            and saved["current_serial_catalog"] == spec["current_serial_catalog"], "outer full-history proof binding")
    # Independently decode the outer's complete saved pair and run both full
    # current physical passes again under exact coordinator-only authority.
    from research import minicpm5_layer1_operator_recovered_completion_v2 as completed
    required = {row["unit"] for row in bundle.scope["nemotron"]["history"]["rows"]}
    completed.physical.validate_pair(completed.audit.unpack(saved["paired_physical"]), required, bundle.scope["owners"],
                                    LOCK_IDENTITY, known_units=set(bundle.scope["complete_known_units"]))
    authority = Authority(bundle, observer.check, lambda: [co])
    once(control_root(spec) / "coordinator-history-pair.json", bundle.physical_pair(authority))
    adapter = Adapter(admission_ref, admission, bundle, observer, co)
    try:
        graph_reuse(bundle, expected_case())
        # Small separately owned native import/codec/control check happens
        # before the original two-arm model gates. Its results are additional.
        preflight = Phase(adapter, "shared", "native_preflight", control_root(spec) / "shared-native_preflight",
                          admission["attempt_id"]).leaf()
        require(preflight["passed"] is True, "native preflight failed before any model gates")
        once(control_root(spec) / "native-preflight.json", preflight)
        from qwen35_sampler_pair_driver_v1 import run
        from qwen35_sampler_pair_driver_v1 import NativeClients
        result = run(adapter.ports(), spec["output_root"], bundle.host, clients=FailureCheckedClients(adapter, NativeClients()))
        observer.check(full=True)
        require(adapter.active is None, "all phase units must be terminal")
        once(control_root(spec) / "coordinator-result.json", {"schema": "qwen35_sampler_coordinator_result_v1",
             "admission": admission_ref, "coordinator": co.owner, "driver_result": result,
             "native_preflight": ref(control_root(spec) / "native-preflight.json"),
             "all_model_and_serving_phases_completed": True, "self_exit_claimed": False,
             "full_model_qualified": False, "independent_final_audit_required": True})
        return 0
    finally:
        if adapter.active is not None: adapter.active.finish()


def declared_services(spec, admission_ref, coordinator_owner=None, retained=None, observations=None):
    """Outer recovery sees only declared phase specs, never guesses live jobs."""
    co = coordinator_service(spec, admission_ref, coordinator_owner)
    if retained is not None:
        previous = retained.setdefault(co.unit, co)
        require(previous.owner == coordinator_owner, "retained coordinator identity changed")
        co = previous
    if coordinator_owner is not None: co.started = True
    result = [co]
    for key in units(spec["run_name"]):
        if key == "coordinator": continue
        arm, name = key.split(":")
        path = control_root(spec) / (arm + "-" + name) / "phase-spec.json"
        if not path.exists():
            value = properties(units(spec["run_name"])[key])
            if observations is not None:
                observations.append({"unit": units(spec["run_name"])[key], "raw_properties": copy.deepcopy(value)})
            # The coordinator publishes phase-spec before installing its unit.
            # Recheck only that exact path after observing an installation.
            if not path.exists():
                require(value["LoadState"] == "not-found" and value["InvocationID"] == "", "undeclared phase unit appeared")
                require(retained is None or units(spec["run_name"])[key] not in retained,
                        "previously declared phase spec disappeared")
                try:
                    process_api().empty_group(process_api().GROUP_PREFIX + units(spec["run_name"])[key] + ".service")
                except RuntimeError as error:
                    # Only the observed absent->declared launch window changes
                    # the required proof from empty group to exact service.
                    # Other cgroup errors and undeclared descendants still fail.
                    if str(error) != "Live group descendants" or not path.exists(): raise
                    if observations is not None:
                        observations.append({"unit": units(spec["run_name"])[key],
                                             "declaration_published_during_absence_check": True})
                if not path.exists(): continue
        phase_ref = ref(path); phase = read(phase_ref)
        require(phase["admission"] == admission_ref and phase["arm"] == arm and phase["phase"] == name
                and phase["coordinator"] == coordinator_owner and phase == expected_phase(spec, admission_ref, coordinator_owner,
                    phase["run_id"], arm, name, phase["directory"], phase["host_identity"]), "outer authenticates exact declared phase")
        owner = read_if_present(path.parent / "supervisor-owner.json")
        service = phase_service(spec, phase, phase_ref, owner)
        if retained is not None:
            previous = retained.get(service.unit)
            if previous is not None:
                require(previous.declared_reference == phase_ref
                        and (previous.argv, previous.env, previous.props, previous.log, previous.parent)
                        == (service.argv, service.env, service.props, service.log, service.parent),
                        "retained phase declaration changed")
                require(owner is None or previous.owner in (None, owner), "phase owner receipt changed identity")
                if owner is not None and previous.owner is None:
                    require(previous.installed_invocation in (None, owner["service_owner"]["invocation_id"]),
                            "acknowledged invocation differs from observed installation")
                    if previous.pending_process is not None:
                        require(all(previous.pending_process[k] == owner["process"][k]
                                    for k in ("pid", "start_ticks", "ppid", "cgroup")), "acknowledged pending process changed")
                    previous.owner = copy.deepcopy(owner)
                service = previous
            else:
                service.declared_reference = copy.deepcopy(phase_ref)
                retained[service.unit] = service
        value = properties(service.unit)
        if observations is not None:
            observations.append({"unit": service.unit, "phase_reference": phase_ref, "raw_properties": copy.deepcopy(value)})
            service.observation_sink = lambda unit, actual: observations.append({"unit": unit, "raw_properties": actual})
        saved_terminal = read_if_present(path.parent / "root-terminal.json")
        require(owner is None or value["LoadState"] != "not-found" or saved_terminal is not None,
                "previously acknowledged service disappeared without saved terminal evidence")
        if owner is not None and saved_terminal is not None and service.saved_terminal is None:
            service.bind_saved_terminal(phase, phase_ref, ref(path.parent / "supervisor-owner.json"),
                                        ref(path.parent / "root-terminal.json"))
        if saved_terminal is not None and saved_terminal.get("unacknowledged_terminal") is not None:
            require(value["LoadState"] != "not-found", "previous unacknowledged terminal cannot be downgraded to never-loaded")
            service.unacknowledged = saved_terminal["unacknowledged_terminal"]
            require(service.unacknowledged["unit"] == service.unit
                    and service.unacknowledged["invocation_id"] == value["InvocationID"]
                    and service.unacknowledged["exec_main_pid"] == int(value["ExecMainPID"]), "retained unacknowledged terminal changed")
        service.started = service.started or value["LoadState"] != "not-found"
        if service.started: service.observe()
        result.append(service)
    return result


class ActiveScopeMonitor:
    """Bounded observations of declared scope; never a name-only exemption.

    Identity memory survives monitor calls and startup ACK publication. Only
    authenticated phase transitions may cause a retry. The original frozen
    scientific policy and all resource limits are unchanged.
    """
    ATTEMPTS = 4
    LIST_ARGV = ["systemctl", "--user", "list-units", "--type=service",
                 "--state=running,activating,deactivating", "--plain", "--no-legend", "--no-pager"]

    def __init__(self, spec, admission_ref, coordinator, guard):
        self.spec, self.admission_ref, self.coordinator, self.guard = spec, admission_ref, coordinator, guard
        self.retained = {coordinator.unit: coordinator}
        self.previous_live = set()
        self.failure_evidence = None
        self.observation = None

    def snapshot(self, label, events):
        self.guard()
        text = process_api().command(self.LIST_ARGV, cap=256 * 1024)
        require(type(text) is str and len(text.encode()) <= 256 * 1024, "bounded complete active-unit listing")
        names = [line.split()[0].removesuffix(".service") for line in text.splitlines()
                 if line.split() and line.split()[0].startswith("trn1-")]
        row = {"kind": label, "raw_systemctl_list": text, "units": sorted(names), "observed_unix_ns": time.time_ns()}
        events.append(row)
        require(len(names) == len(set(names)), "duplicate active heavy service")
        require(set(names) <= set(units(self.spec["run_name"]).values()), "unknown competing active heavy service")
        require(len(names) <= 2 and (not names or self.coordinator.unit in names),
                "coordinator plus at most one simultaneously active phase")
        return set(names)

    def scan(self, label, events):
        self.guard()
        rows = []; events.append({"kind": label, "services": rows})
        services = declared_services(self.spec, self.admission_ref, self.coordinator.owner, self.retained, rows)
        live = set(); values = {}
        for service in services:
            if not service.started: continue
            self.guard()
            value = service.observe()
            if not process_api().terminal(value):
                live.add(service.unit)
                if value["MainPID"] != "0":
                    expected = service.owner["process"] if service.owner is not None else service.pending_process
                    actual = process_api().proc(int(value["MainPID"]))
                    if expected is not None and actual != expected:
                        require(service.owner is None and service.pending_process is not None
                                and type(actual) is dict and actual.get("argv") == service.argv
                                and all(actual.get(k) == expected[k] for k in ("pid", "start_ticks", "ppid", "cgroup")),
                                "active process identity changed within complete scope observation")
                        # A single env->Python exec can occur after observe's
                        # proc read. Reuse its full property/invocation/command
                        # validator once; never rewrite or adopt an identity.
                        self.guard()
                        value = service.observe()
                        require(not process_api().terminal(value) and service.owner is not None
                                and service.owner["process"] == actual
                                and value["MainPID"] == str(actual["pid"])
                                and process_api().proc(actual["pid"]) == actual,
                                "expected env to Python exec did not converge within one reauthentication")
                        self.guard()
                        expected = service.owner["process"]
                    require(expected is not None and actual == expected,
                            "active process identity changed within complete scope observation")
            record = {"unit": service.unit, "properties": copy.deepcopy(value),
                      "phase_reference": copy.deepcopy(getattr(service, "declared_reference", None)),
                      "owner": copy.deepcopy(service.owner), "pending_process": copy.deepcopy(service.pending_process),
                      "installed_invocation": service.installed_invocation}
            rows.append(record); values[service.unit] = record
        return services, live, values

    def drain_removed(self, names, current_live, events):
        for unit in sorted(names - current_live):
            require(unit in self.retained and self.retained[unit].started,
                    "listed unit lacks authenticated current admission declaration")
            terminal = self.retained[unit].drained()
            events.append({"kind": "removed_unit_complete_terminal", "unit": unit, "terminal": terminal})

    def observe(self):
        events = []
        report = {"schema": "qwen35_sampler_active_scope_observation_v1", "admission": self.admission_ref,
                  "coordinator": copy.deepcopy(self.coordinator.owner), "events": events,
                  "observation_successful": False, "historical_r5_cause_proven": False}
        try:
            for attempt in range(self.ATTEMPTS):
                self.guard()
                before = self.snapshot("active_before", events)
                _, first_live, first = self.scan("scope_before", events)
                require(before <= set(first), "active unit has no authenticated declaration")
                after = self.snapshot("active_after", events)
                services, second_live, second = self.scan("scope_after", events)
                require(after <= set(second), "active unit has no authenticated declaration")
                self.drain_removed(before | first_live | self.previous_live, after | second_live, events)
                self.guard()
                if before == after == first_live == second_live and first == second:
                    self.previous_live = second_live
                    report.update(observation_successful=True, attempts=attempt + 1)
                    self.observation = report
                    return services, second[self.coordinator.unit]["properties"]
                require(attempt + 1 < self.ATTEMPTS, "active scope changed continuously within bounded retries")
                time.sleep(.01)
            raise AssertionError("finite active scope observation exhausted")
        except BaseException as error:
            report["failure"] = {"type": type(error).__name__, "message": str(error)[:2048]}
            self.failure_evidence = report
            raise


def retain_until_drained(lock, services, error):
    """No more scientific work. Retain the lease on unresolved owned cgroups."""
    print(json.dumps({"event": "qwen35_sampler_cleanup_blocked", "error_type": type(error).__name__,
                      "error": str(error)[:2048], "serial_lease_retained": True,
                      "automatic_successor": False}), flush=True)
    while True:
        try:
            lock.check()
            owned = services() if callable(services) else services
            for service in reversed(owned):
                if service.started: service.cleanup()
            return owned
        except BaseException:
            # Releasing a descriptor while an owned native descendant remains
            # would permit overlap. Work deadlines remain finite; this is only
            # an emergency ownership hold for root intervention.
            time.sleep(1)


def outer(spec_ref, review_ref):
    """Future explicit entry only. Never invoked by import or --help."""
    lock = RetainedLock(); lock.acquire()
    started = time.monotonic(); bundle = None; co = None; admission_ref = None; control = None
    all_services = []; drained = False; result = None; failed = None
    scope_monitor = None; scope_failure_ref = None
    interrupt = [False]; old_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, lambda *_: interrupt.__setitem__(0, True))
    last_resource = [-float("inf")]
    def guard():
        lock.check()
        require(not interrupt[0] and time.monotonic() - started < LIMITS["outer_wall_seconds"], "outer interruption/deadline")
        if bundle is not None and time.monotonic() - last_resource[0] >= LIMITS["monitor_interval_seconds"]:
            resources(bundle.spec); last_resource[0] = time.monotonic()
    try:
        bundle = Bundle(spec_ref, review_ref, guard)
        spec = bundle.spec; control = control_root(spec)
        require(not control.exists() and not Path(spec["output_root"]).exists() and not Path(spec["running_handoff_path"]).exists(),
                "fresh outputs and future handoff required; never resume/restart")
        resources(spec, reserve=True); process_api().active_heavy()
        bundle.resolve()
        authority = Authority(bundle, guard, lambda: [])
        physical = bundle.physical_pair(authority)
        freshness = bundle.freshness()
        graph_reuse(bundle, expected_case())
        for unit in bundle.names.values():
            value = properties(unit)
            require(value["LoadState"] == "not-found" and value["InvocationID"] == "", "all21 units must be fresh")
            process_api().empty_group(process_api().GROUP_PREFIX + unit + ".service")
        control.mkdir(); (control / "tmp").mkdir()
        physical_ref = once(control / "outer-history-pair.json", physical)
        admission = {"schema": "qwen35_sampler_runtime_admission_v1", "spec": spec_ref, "review": review_ref,
            "attempt_id": str(uuid.uuid4()), "outer_owner": lock.owner, "retained_fd": lock.fd,
            "initial_descriptor_proof": lock.initial, "lock_identity": LOCK_IDENTITY,
            "current_serial_catalog": spec["current_serial_catalog"], "history_overlay": spec["history_overlay"],
            "resolved_plan_sha256": bundle.resolved["plan_sha256"], "history_pair": physical_ref,
            "freshness": freshness, "units": bundle.names, "source_map_sha256": digest(spec["source_sha256"]),
            "host_source_sha256": bundle.host, "control_root": str(control), "output_root": spec["output_root"],
            "started_monotonic": started, "deadline_monotonic": started + LIMITS["outer_wall_seconds"],
            "runtime_admitted": True, "automatic_successor": False, "self_exit_claimed": False}
        admission_ref = once(control / "admission.json", admission)
        admit_shape(admission, spec, spec_ref, review_ref)
        co = coordinator_service(spec, admission_ref); all_services = [co]
        co.start(guard)
        coordinator_ref = wait_until(lambda: ref(control / "coordinator-owner.json") if (control / "coordinator-owner.json").exists() else None,
                                     guard, LIMITS["startup_ack_seconds"], message="coordinator owner ACK missing")
        require(read(coordinator_ref) == co.owner, "actual independently observed coordinator owner mismatch")
        print(json.dumps({"event": "qwen35_sampler_root_running_handoff_required", "admission": admission_ref,
                          "coordinator_owner": coordinator_ref, "coordinator": co.owner,
                          "running_handoff_path": spec["running_handoff_path"],
                          "actual_external_tool_session_required": True, "native_work_released": False}), flush=True)
        wait_until(lambda: True if Path(spec["running_handoff_path"]).exists() else None, guard,
                   LIMITS["root_running_ack_seconds"], message="root did not bind actual tool session/running handoff")
        running = check_running(bundle, admission_ref, admission, coordinator_ref, co.owner)
        once(control / "running-ack.json", {"admission": admission_ref, "coordinator_owner": coordinator_ref,
                                           "coordinator": co.owner, "running": running})
        scope_monitor = ActiveScopeMonitor(spec, admission_ref, co, guard)
        while True:
            guard()
            from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
            require(latest.latest_handoff() == running["observation"], "unknown later handoff during current execution")
            all_services, value = scope_monitor.observe()
            if process_api().terminal(value): break
            time.sleep(LIMITS["monitor_interval_seconds"])
        result = read_if_present(control / "coordinator-result.json")
        require(result is not None and result["admission"] == admission_ref and result["coordinator"] == co.owner,
                "coordinator did not produce its completed real result")
    except BaseException as error:
        failed = error
        if scope_monitor is not None and scope_monitor.failure_evidence is not None:
            scope_failure_ref = once(control / "active-scope-failure.json", scope_monitor.failure_evidence)
    finally:
        try:
            if admission_ref is not None:
                # An interrupted launch may exist before its owner file. The
                # original authenticated argv/invocation is captured before
                # cleanup; never use an unverified arbitrary service name.
                co_owner = co.owner if co is not None else None
                if co_owner is not None:
                    all_services = declared_services(bundle.spec, admission_ref, co_owner,
                        None if scope_monitor is None else scope_monitor.retained)
            for service in reversed(all_services):
                if service.started: service.cleanup()
            drained = True
        except BaseException as cleanup_error:
            failed = failed or cleanup_error
            # Catalog reconstruction itself may have failed on a vanished or
            # changed phase. Never fall back to an older short service list.
            def complete_scope():
                if admission_ref is not None and co is not None and co.owner is not None:
                    return declared_services(bundle.spec, admission_ref, co.owner,
                        None if scope_monitor is None else scope_monitor.retained)
                return all_services
            all_services = retain_until_drained(lock, complete_scope, cleanup_error)
            drained = True
        if drained:
            if control is not None and control.exists() and admission_ref is not None:
                terminal = {"schema": "qwen35_sampler_outer_observed_terminal_v1", "admission": admission_ref,
                    "outer_owner": lock.owner, "original_lock_identity": LOCK_IDENTITY,
                    "complete_owned_unit_terminals": [s.terminal for s in all_services if s.started],
                    "historical_terminal_reuses": [s.historical_terminal_reuse for s in all_services
                        if s.started and s.historical_terminal_reuse is not None],
                    "whole_owned_scope_drained": True, "result_present": result is not None,
                    "failed": failed is not None, "failure": None if failed is None else {"type": type(failed).__name__, "message": str(failed)[:2048]},
                    "active_scope_failure": scope_failure_ref,
                    "storage": storage(bundle.spec, terminal=True), "full_model_qualified": False,
                    "self_exit_claimed": False, "actual_external_tool_exit_pending": True,
                    "independent_final_audit_required": True, "automatic_successor": False}
                once(control / "outer-terminal.json", terminal)
            lock.release(True)
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)
    if failed is not None: raise failed
    final_co = next(service for service in all_services if service.unit == co.unit)
    require(final_co.owner == co.owner and final_co.terminal is not None
            and final_co.terminal["systemd_exit_status"] == {"ExecMainCode": 1, "ExecMainStatus": 0, "Result": "success"},
            "actual coordinator exit failed")
    return 0
