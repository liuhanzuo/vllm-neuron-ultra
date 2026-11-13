"""Original non-HTTP Phase lifetime bodies; no new scheduler or model loop."""
import time
from pathlib import Path
from .contract import LIMITS,require,expected_phase,require_phase
from .io import once,read_if_present,wait_until,process_api
from .service import argv,validate_api_failure,API_ERROR_MESSAGE
from .bootstrap import phase_service
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
        require_phase(a.spec, self.arm, self.name, self.directory)
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
            if self.name == "build":
                # Build success requires compiler descendants already gone;
                # cleanup must not turn an unfinished compiler into success.
                self.service.drained()
        finally:
            self.finish()
        status = self.end["systemd_exit_status"]
        require(status == {"ExecMainCode": 1, "ExecMainStatus": 0, "Result": "success"}, "actual native service failed")
        return {**result["result"], "run_id": self.run_id, "arm": self.arm, "phase": self.name,
                "returncode": code["child_returncode"], "fresh_execution": True,
                "actual_child_exit": code, "actual_service_terminal": self.end}

