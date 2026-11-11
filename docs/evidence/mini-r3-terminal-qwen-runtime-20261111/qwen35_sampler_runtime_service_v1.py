"""Exact systemd ownership and real subprocess exit/drain for Qwen phases.

The systemd-run client exit only acknowledges installation. Completion always
requires actual captured main-process identity, its retained InvocationID,
actual systemd exit fields, the separately Popen-waited child, and a fully empty
recursive cgroup. No old service contract is weakened or monkeypatched.
"""
import copy
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

from qwen35_sampler_runtime_contract_v1 import (COORDINATOR_PROPERTIES, LIMITS, PYTHON, Q, ROOT,
    control_root, digest, native_environment, phase_properties, require, units)
from qwen35_sampler_runtime_io_v1 import (once, process_api, raw, read, read_if_present, ref,
                                        service_owner, wait_until)

CLI = str(Q / "run_qwen35_sampler_runtime_v1.py")
PROPERTY_KEYS = ("Id,LoadState,ActiveState,SubState,MainPID,ControlPID,Job,InvocationID,ControlGroup,Result,"
                 "ExecMainPID,ExecMainCode,ExecMainStatus,ExecStart,ExecStopPost,Environment,BindsTo,After,"
                 "MemoryMax,MemoryHigh,MemorySwapMax,TasksMax,KillMode,RemainAfterExit,SendSIGKILL,"
                 "WorkingDirectory,LimitFSIZE,RuntimeMaxUSec,TimeoutStopUSec,StandardOutput,StandardError")


def properties(unit):
    require(re.fullmatch(r"trn1-qwen35-sampler-[a-z0-9-]{1,180}", unit), "new exact Qwen unit namespace")
    value = process_api().command(["systemctl", "--user", "show", "--all", unit, "--property=" + PROPERTY_KEYS])
    return process_api().parse_properties(value)


def duration(value):
    # Actual systemd property text uses s/min/h; no arbitrary truthy coercion.
    total = 0.
    for number, suffix in re.findall(r"([0-9]+(?:\.[0-9]+)?)(us|ms|s|min|h|d)", value):
        total += float(number) * {"us": .000001, "ms": .001, "s": 1, "min": 60, "h": 3600, "d": 86400}[suffix]
    require(re.sub(r"[0-9]+(?:\.[0-9]+)?(?:us|ms|s|min|h|d)|\s", "", value) == "", "unexpected systemd duration")
    return total


def argv(mode, reference_value):
    require(mode in ("coordinator", "supervisor", "leaf", "api_bootstrap"), "fixed executable mode")
    return [PYTHON, "-B", CLI, "--mode", mode, "--binding", reference_value["path"],
            "--binding-sha256", reference_value["sha256"]]


def command(unit, expected_argv, env, props, log, parent=None):
    require(all(type(k) is str and type(v) is str and re.fullmatch(r"[A-Z_][A-Z0-9_]*", k)
                and not any(x in k + v for x in ("\n", "\r", "\0", " ")) for k, v in env.items()),
            "fixed simple systemd supervisor environment; host JSON belongs only in direct child Popen")
    settings = dict(props)
    settings.update(StandardOutput="append:" + str(log), StandardError="inherit")
    if parent is not None: settings.update(BindsTo=parent + ".service", After=parent + ".service")
    return ["systemd-run", "--user", "--quiet", "--unit=" + unit, "--service-type=exec",
            *[arg for key, value in settings.items() for arg in ("-p", key + "=" + value)],
            "/usr/bin/env", "-i", *[key + "=" + value for key, value in sorted(env.items())], *expected_argv]


def verify_properties(value, unit, expected_argv, env, props, log, parent=None):
    p = process_api(); expected = command(unit, expected_argv, env, props, log, parent)
    require(value["LoadState"] == "loaded" and value["Id"] == unit + ".service"
            and p.invocation(value["InvocationID"]), "actual loaded service invocation")
    for key, wanted in props.items():
        if key in ("RuntimeMaxSec", "TimeoutStopSec"):
            actual = key.replace("Sec", "USec")
            require(duration(value[actual]) == float(wanted), "actual finite service deadline changed")
        else: require(value[key] == wanted, "actual service resource changed: " + key)
    require(value["ExecStopPost"] == "" and value["Environment"] == "", "no hidden unit environment or cleanup command")
    installed = " ".join(expected[expected.index("/usr/bin/env"):])
    require(re.findall(r"argv\[\]=(.*?)\s*;", value["ExecStart"]) == [installed], "actual ExecStart changed")
    require(value["ControlGroup"] in ("", p.GROUP_PREFIX + unit + ".service"), "actual cgroup changed")
    require(value["StandardOutput"] == "append" and value["StandardError"] == "inherit", "actual file logging mode changed")
    if parent is not None:
        require(parent + ".service" in value["BindsTo"].split() and parent + ".service" in value["After"].split(),
                "phase must bind lifetime to actual coordinator")
    return value


class Service:
    def __init__(self, unit, expected_argv, env, props, log, *, parent=None, owner=None):
        self.unit, self.argv, self.env, self.props = unit, list(expected_argv), dict(env), dict(props)
        self.log, self.parent = Path(log), parent
        self.owner = copy.deepcopy(owner); self.started = owner is not None
        self.unacknowledged = None
        self.installed_invocation = None if owner is None else owner["service_owner"]["invocation_id"]
        self.pending_process = None
        self.stop_requested = False; self.terminal = None

    @property
    def group(self): return process_api().GROUP_PREFIX + self.unit + ".service"

    def observe(self):
        value = verify_properties(properties(self.unit), self.unit, self.argv, self.env, self.props, self.log, self.parent)
        require(self.installed_invocation in (None, value["InvocationID"]), "installed invocation changed during startup")
        self.installed_invocation = value["InvocationID"]
        if self.owner is None:
            if value["MainPID"] == "0":
                if value["ActiveState"] == "activating" and value["SubState"] in ("start", "start-pre", "start-post"):
                    return value
                require(process_api().terminal(value), "unit has neither a known pending state nor a terminal state")
                row = {"unit": self.unit, "invocation_id": value["InvocationID"],
                       "exec_main_pid": int(value["ExecMainPID"]), "captured_live_owner": False,
                       "start_ticks_observed": False}
                require(self.unacknowledged in (None, row), "unacknowledged terminal invocation changed")
                self.unacknowledged = row
                return value
            require(self.unacknowledged is None, "unacknowledged terminal cannot be restarted/adopted")
            owner = process_api().proc(int(value["MainPID"]))
            installed = command(self.unit, self.argv, self.env, self.props, self.log, self.parent)
            wrapper = installed[installed.index("/usr/bin/env"):]
            require(owner is not None and owner["argv"] in (self.argv, wrapper)
                    and owner["cgroup"] == "0::" + self.group,
                    "actual main PID/argv/cgroup mismatch")
            if self.pending_process is not None:
                require(all(owner[key] == self.pending_process[key] for key in ("pid", "start_ticks", "ppid", "cgroup")),
                        "installed env process changed identity before Python exec")
            if owner["argv"] == wrapper:
                self.pending_process = owner
                return value
            self.owner = {"process": owner, "service_owner": service_owner(self.unit, owner, value)}
        else:
            require(value["InvocationID"] == self.owner["service_owner"]["invocation_id"]
                    and value["ExecMainPID"] == str(self.owner["process"]["pid"]), "retained exact main invocation changed")
        return value

    def live(self):
        value = self.observe()
        require(self.owner is not None and value["ActiveState"] == "active" and value["SubState"] == "running"
                and value["MainPID"] == str(self.owner["process"]["pid"])
                and process_api().proc(self.owner["process"]["pid"]) == self.owner["process"], "same actual service must still be live")
        return value

    def start(self, guard):
        require(not self.started and self.owner is None, "no service restart or adoption")
        empty = properties(self.unit)
        require(empty["LoadState"] == "not-found" and empty["InvocationID"] == "", "fresh never-loaded service required")
        process_api().empty_group(self.group)
        require(not self.log.exists(), "fresh service log required")
        self.log.touch(mode=0o600, exist_ok=False)
        self.started = True  # Also own a partially successful systemd-run.
        process_api().command(command(self.unit, self.argv, self.env, self.props, self.log, self.parent), timeout=30)
        def acknowledged():
            value = self.observe()
            require(not process_api().terminal(value), "service failed before actual owner acknowledgment")
            return self.live() if self.owner is not None and (value["ActiveState"], value["SubState"]) == ("active", "running") else None
        wait_until(acknowledged,
                   guard, LIMITS["startup_ack_seconds"], message="actual unit owner did not acknowledge")
        return self.owner

    def drained(self):
        p = process_api(); first = self.observe()
        require(p.terminal(first), "owned service is not terminal")
        def absent():
            if self.owner is not None: p.gone(self.owner["process"])
            else:
                require(self.unacknowledged is not None, "explicit unacknowledged terminal identity required")
                pid = self.unacknowledged["exec_main_pid"]
                require(pid == 0 or p.proc(pid) is None, "uncaptured terminal PID must be entirely absent; no invented start ticks")
        absent(); group1 = p.empty_group(self.group)
        second = self.observe(); absent(); group2 = p.empty_group(self.group)
        require(first == second and group1 == group2, "two actual stable terminal/drained observations required")
        status = {"ExecMainCode": int(first["ExecMainCode"]), "ExecMainStatus": int(first["ExecMainStatus"]), "Result": first["Result"]}
        self.terminal = {"server_owner": copy.deepcopy(self.owner["service_owner"]) if self.owner is not None else None,
                         "process_owner": copy.deepcopy(self.owner["process"]) if self.owner is not None else None,
                         "unacknowledged_terminal": copy.deepcopy(self.unacknowledged),
                         "exit_observed": True, "systemd_exit_status": status, "properties": first,
                         "group_observations": [group1, group2], "remaining_processes": [], "cleanup_complete": True,
                         "observed_unix_ns": time.time_ns()}
        return self.terminal

    def cleanup(self):
        """Always authenticate invocation before any stop/kill; never global pkill."""
        p = process_api()
        if not self.started: return None
        value = properties(self.unit)
        if value["LoadState"] == "not-found":
            require(self.owner is None and self.unacknowledged is None
                    and self.installed_invocation is None and self.pending_process is None,
                    "previously captured or unacknowledged-terminal service cannot disappear before audit")
            group = p.empty_group(self.group)
            self.terminal = {"unit": self.unit, "never_loaded": True, "cleanup_complete": True,
                             "exit_observed": False, "actual_process_owner_observed": False,
                             "remaining_processes": [], "group_observation": group}
            return self.terminal
        self.observe()
        self.stop_requested = True
        if not p.terminal(value):
            self.observe()
            # Explicit stop can GC a transient unit and destroy the exact
            # terminal fields before audit. Signal the authenticated cgroup;
            # RemainAfterExit then retains natural terminal metadata.
            p.command(["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGTERM", self.unit], timeout=10)
        deadline = time.monotonic() + LIMITS["owned_cleanup_seconds"]
        killed = False
        while True:
            value = self.observe()
            if p.terminal(value):
                try: return self.drained()
                except ValueError: pass
                except RuntimeError: pass
            if not killed and time.monotonic() >= deadline - 15:
                self.observe()
                p.command(["systemctl", "--user", "kill", "--kill-whom=all", "--signal=SIGKILL", self.unit], timeout=10)
                killed = True
            require(time.monotonic() < deadline, "owned service cleanup incomplete; original lease must remain held")
            time.sleep(.1)


def child_process(phase, phase_ref, api=False):
    return argv("api_bootstrap" if api else "leaf", phase_ref)


def expected_phase(spec, admission_ref, coordinator, run_id, arm, phase, directory, host=None):
    require((arm + ":" + phase) in units(spec["run_name"]), "declared phase")
    stage = control_root(spec) / (arm + "-" + phase)
    return {"schema": "qwen35_sampler_phase_spec_v1", "admission": admission_ref, "coordinator": copy.deepcopy(coordinator),
            "run_id": run_id, "arm": arm, "phase": phase, "directory": str(directory), "control": str(stage),
            "unit": units(spec["run_name"])[arm + ":" + phase], "host_identity": copy.deepcopy(host),
            "properties": phase_properties(arm, phase), "log": str(stage / ("phase-server.log" if phase in ("capture", "quality", "http") else "phase.log")),
            "self_exit_claimed": False}


def supervise(phase_ref, bootstrap):
    """Real Popen owner/wait. Runs only in an admitted exact phase service."""
    phase, context = bootstrap(phase_ref, role="supervisor")
    stage = Path(phase["control"]); guard = context["guard"]; service = context["service"]
    owner = service.owner
    once(stage / "supervisor-owner.json", owner)
    release = wait_until(lambda: read_if_present(stage / "supervisor-release.json"), guard,
                         LIMITS["startup_ack_seconds"], message="independent supervisor owner ACK missing")
    require(release == {"phase_spec": phase_ref, "supervisor": owner, "coordinator": phase["coordinator"]}, "exact supervisor ACK required")
    api = phase["phase"] in ("capture", "quality", "http")
    env = context["child_environment"]
    process = None; child_owner = None; result = None; stop_requested = False
    stopped = [False]
    previous = signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    selector = selectors.DefaultSelector(); total_log = 0; log_tail = b""; log_cap = LIMITS["server_log_bytes" if api else "leaf_log_bytes"]
    start = time.monotonic(); child_argv = child_process(phase, phase_ref, api)
    try:
        guard()
        process = subprocess.Popen(child_argv, env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        selector.register(process.stdout, selectors.EVENT_READ)
        child_owner = wait_until(lambda: process_api().proc(process.pid), guard, 5, message="child owner disappeared before ACK")
        require(child_owner["ppid"] == owner["process"]["pid"] and child_owner["cgroup"] == "0::" + service.group
                and child_owner["argv"] == child_argv, "real direct child PID/start/argv/cgroup required")
        once(stage / "child-bootstrap-owner.json", child_owner)
        while process.poll() is None or selector.get_map():
            guard()
            require(time.monotonic() - start < int(phase["properties"]["RuntimeMaxSec"]) - 15, "finite native phase deadline")
            if stopped[0] or (stage / "stop-request.json").exists():
                request = read_if_present(stage / "stop-request.json")
                require(api and request == {"phase_spec": phase_ref, "supervisor": owner, "coordinator": phase["coordinator"]},
                        "only explicitly root-requested API stop is accepted")
                stop_requested = True; break
            for key, _ in selector.select(.1):
                block = os.read(key.fileobj.fileno(), min(65536, log_cap + 1 - total_log))
                if not block: selector.unregister(key.fileobj); continue
                total_log += len(block)
                require(total_log <= log_cap, "real native stdout exceeded finite phase log cap")
                # systemd stdout points at this phase's one bounded log.
                os.write(1, block)
                if api:
                    combined = log_tail + block
                    require(not any(marker in combined.lower() for marker in
                        (b"traceback (most recent call last)", b"unexpected runner mode", b"request/row order mismatch",
                         b"uncaptured effective value", b"dummy run failed", b"error during model execution")),
                        "actual server/capture/context exception observed; never consume and hide diagnostic quotas")
                    log_tail = combined[-4096:]
        if stop_requested and process.poll() is None:
            require(process_api().proc(process.pid)["start_ticks"] == child_owner["start_ticks"], "refuse PID-reused child signal")
            os.killpg(process.pid, signal.SIGTERM)
        try: code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            require(process_api().proc(process.pid)["start_ticks"] == child_owner["start_ticks"], "refuse PID-reused child kill")
            os.killpg(process.pid, signal.SIGKILL); code = process.wait(timeout=10)
        result = {"schema": "qwen35_sampler_actual_child_exit_v1", "phase_spec": phase_ref,
                  "supervisor": owner, "child_bootstrap_owner": child_owner, "child_returncode": code,
                  "actual_popen_wait_observed": True, "stop_requested": stop_requested,
                  "observed_unix_ns": time.time_ns(), "self_exit_claimed": False}
        once(stage / "child-exit.json", result)
        return 0 if stop_requested and code in (0, -signal.SIGTERM) else (0 if code == 0 else 1)
    finally:
        signal.signal(signal.SIGTERM, previous)
        selector.close()
        if process is not None:
            if process.poll() is None:
                current = process_api().proc(process.pid)
                require(current is not None and child_owner is not None and current["start_ticks"] == child_owner["start_ticks"],
                        "cannot authenticate remaining child; phase cgroup must be drained externally")
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait(timeout=10)
            process.stdout.close()
            if result is None and child_owner is not None:
                once(stage / "child-exit.json", {"schema": "qwen35_sampler_actual_child_exit_v1", "phase_spec": phase_ref,
                    "supervisor": owner, "child_bootstrap_owner": child_owner, "child_returncode": code,
                    "actual_popen_wait_observed": True, "stop_requested": stop_requested,
                    "observed_unix_ns": time.time_ns(), "self_exit_claimed": False})
