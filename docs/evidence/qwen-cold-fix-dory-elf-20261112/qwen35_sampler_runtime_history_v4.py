"""Complete recovered/current history bridge for the Qwen host-only runtime.

Uses original latest v4 catalog and original two-pass physical bodies. No
current binding is selected or written here; all refs come from a future root
spec. The historical Mini output namespace remains a metadata-only view.
"""
import copy
from datetime import datetime
import hashlib
from pathlib import Path
import time

from qwen35_sampler_runtime_contract_v4 import (LIMITS, LOCK_IDENTITY, Q, RUNTIME_FILES,
    control_root, digest, host_sources, reference, require, units, validate_spec)
from qwen35_sampler_runtime_io_v4 import Metadata, identity, loaded_local_sources, process_api, raw, read


class Bundle:
    def __init__(self, spec_ref, review_ref, guard):
        self.spec_ref, self.review_ref = copy.deepcopy(spec_ref), copy.deepcopy(review_ref)
        self.guard = guard; self.meta = Metadata(guard)
        self.spec = read(spec_ref, LIMITS["spec_envelope_bytes"])
        self.review = read(review_ref, LIMITS["spec_envelope_bytes"])
        self.names = validate_spec(self.spec, self.review, spec_ref)
        self.binding_identities = {r["path"]: identity(r["path"]) for r in (spec_ref, review_ref)}
        self.resolved = None; self.delivery = None; self.host = None; self.partition = None

    def resolve(self, *, verify_sources=True):
        from research import minicpm5_layer1_operator_recovered_completion_v2 as completed
        from research import minicpm5_layer1_operator_native_overlay_v3 as overlays
        from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
        self.guard()
        recovered = completed.completed(completed.BINDING, self.meta.read, self.guard,
                                         lambda path, sha: sources.text(path, sha))
        overlay = self.meta.read(self.spec["history_overlay"])
        require(overlay["current_serial_catalog"] == self.spec["current_serial_catalog"], "complete latest catalog binding")
        resolved = overlays.resolve(overlay, recovered, self.meta.read, self.guard)
        # Latest recorder_v3/auditor_v4 receipts reconstruct the entire
        # inherited source plan. Use the accepted v7 witness dispatcher;
        # never feed these receipts into the older producer-only validator.
        # Do not call v7 preparation/binding or its idle-tip wrapper while
        # this coordinator is live. Outer freshness and running ACK are
        # independently enforced by this runtime's exact handoff contract.
        from research import prepare_minicpm5_layer1_operator_native_v7 as source_witness
        witness_cache, witnessed = {}, {}
        for row in resolved["current_scope"]["later_serial"]["runs"]:
            terminal = self.meta.read(row["terminal"])
            if terminal.get("schema") in ("minicpm5_native_model_exit_recorder_actual_exit_v1",
                                           "minicpm5_native_independent_audit_actual_exit_v1"):
                additions = source_witness.executed_extension(terminal, row, self.meta.read, self.guard, witness_cache)
                for path, sha in additions.items():
                    require(path not in witnessed or witnessed[path] == sha, "completed source witness conflict")
                    witnessed[path] = sha
        pins = self.spec["source_sha256"]
        require(all(pins.get(path) == sha for path, sha in resolved["plan"]["source_sha256"].items()),
                "every original/history/current source retained without replacement")
        require(all(pins.get(path) == sha for path, sha in witnessed.items()), "complete accepted recorder/auditor source witnesses retained")
        require(recovered["original_model_exit_code"] == 1 and resolved["runtime_admitted"] is False,
                "historical failed model remains failed; metadata is not runtime admission")
        require(not set(self.names.values()) & set(resolved["current_scope"]["complete_known_units"]),
                "every prospective Qwen unit is disjoint from the entire historical catalog")
        for path, row in self.spec["input_extension"].items():
            require(path not in resolved["plan"]["inputs"] or resolved["plan"]["inputs"][path] == row,
                    "input extension cannot replace original source/input identity")
            require(identity(path) == row["identity"], "future exact metadata input identity changed")
            self.meta.read({"path": path, "sha256": row["sha256"]})
        delivery = self.meta.read(self.spec["frozen_science_delivery"])
        self.host = host_sources(self.spec, delivery)
        for path, sha in self.host.items():
            self.guard()
            require(hashlib.sha256(raw(path, 4 * 1024**2)).hexdigest() == sha, "actual host text pin changed")
        partition = sources.classify(pins, resolved["plan"]["source_contract"], self.meta.read)
        if verify_sources: sources.verify_text_sources(partition, self.guard)
        self.resolved, self.delivery, self.partition = resolved, delivery, partition
        self.guard(); loaded_local_sources(pins)
        return self

    def unchanged(self):
        require(all(identity(path) == value for path, value in self.binding_identities.items()), "reviewed bindings changed")
        self.meta.unchanged()

    @property
    def scope(self):
        require(self.resolved is not None, "complete history not resolved")
        return self.resolved["current_scope"]

    def freshness(self):
        from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
        return latest.require_latest_terminal(self.resolved["current_serial_view"], self.meta.read, self.guard)

    def physical_pair(self, authority):
        from research import minicpm5_native_current_physical_v1 as physical
        from research import minicpm5_layer1_operator_recovered_completion_v2 as completed
        from research.audit_minicpm5_layer1_operator_sdk_result_v3 import packed_physical
        self.guard(); pair = physical.collect_pair(self.scope, self.guard, authority)
        required = {row["unit"] for row in self.scope["nemotron"]["history"]["rows"]}
        known = set(self.scope["complete_known_units"])
        require(completed.physical.validate_pair(pair, required, self.scope["owners"], LOCK_IDENTITY,
                                                known_units=known) == known, "entire original two-pass physical scope")
        return {"schema": "qwen35_sampler_full_history_pair_v1", "current_serial_catalog": self.spec["current_serial_catalog"],
                "history_overlay": self.spec["history_overlay"], "resolved_plan_sha256": self.resolved["plan_sha256"],
                "physical_pair_schema": completed.physical.PAIR_SCHEMA, "paired_physical": packed_physical(pair),
                "full_physical_passes": 2, "complete_known_units": sorted(known),
                "owners": copy.deepcopy(self.scope["owners"]), "lock_identity": copy.deepcopy(LOCK_IDENTITY),
                "original_failed_model_exit_code": 1, "full_model_qualified": False}


class Authority:
    """Exact constructed services only; never accepts arbitrary allowed units."""
    def __init__(self, bundle, guard, services):
        self.bundle, self.guard, self.services = bundle, guard, services
        self.lock_identity = copy.deepcopy(LOCK_IDENTITY)

    def check(self):
        self.guard()
        names = []
        for service in self.services():
            require(service.unit in self.bundle.names.values(), "undeclared physical exclusion")
            service.live()
            names.append(service.unit)
        require(len(names) == len(set(names)) <= 2, "coordinator and at most one admitted phase")
        return tuple(names)

    def disjoint(self, historical):
        require(not set(self.bundle.names.values()) & set(historical), "never hide a historical unit")
        self.check()

    def active_heavy(self):
        allowed = self.check(); value = process_api().active_heavy(allowed); self.check(); return value

    def require_running_names(self, names):
        require(type(names) is set and names <= {u + ".service" for u in self.check()},
                "unexpected running names in original physical body")


def expected_running(previous, updated_at, tool, admission_ref, admission, coordinator_ref, coordinator):
    """Pure future root-produced handoff shape, with complete inherited fields."""
    require(type(tool) is dict and set(tool) == {"kind", "session_id"} and tool["kind"] == "tool_session"
            and type(tool["session_id"]) is int and tool["session_id"] > 0, "actual externally observed tool session required")
    require(type(updated_at) is str and len(updated_at) < 48
            and datetime.fromisoformat(updated_at).utcoffset() is not None, "zoned root observation time required")
    require(previous["serial_execution"] is True and previous["live_tool_sessions"] == []
            and previous["active_heavy_units"] == [], "unchanged complete idle predecessor required")
    value = copy.deepcopy(previous)
    value.update(schema="trn1_continuation_handoff_qwen35_sampler_running_v1", updated_at=updated_at,
                 previous=copy.deepcopy(admission["freshness"]["latest_handoff"]),
                 live_tool_sessions=[tool["session_id"]], active_heavy_units=[coordinator["service_owner"]["unit"]],
                 publication_only_since_bound_terminal=False,
                 qwen35_sampler_current_execution={"schema": "qwen35_sampler_current_execution_v1",
                     "admission": copy.deepcopy(admission_ref), "spec": copy.deepcopy(admission["spec"]),
                     "review": copy.deepcopy(admission["review"]), "outer_owner": copy.deepcopy(admission["outer_owner"]),
                     "coordinator_owner": copy.deepcopy(coordinator_ref), "coordinator": copy.deepcopy(coordinator),
                     "complete_current_serial_catalog": copy.deepcopy(admission["current_serial_catalog"]),
                     "lock_identity": copy.deepcopy(LOCK_IDENTITY), "tool": copy.deepcopy(tool),
                     "phase_units": copy.deepcopy(admission["units"]), "automatic_successor": False})
    return value


def check_running(bundle, admission_ref, admission, coordinator_ref, coordinator):
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    path = bundle.spec["running_handoff_path"]
    value_ref = {"path": path, "sha256": hashlib.sha256(raw(path)).hexdigest()}
    value = read(value_ref)
    previous = bundle.meta.read(admission["freshness"]["latest_handoff"])
    expected = expected_running(previous, value["updated_at"], value["qwen35_sampler_current_execution"]["tool"],
                                admission_ref, admission, coordinator_ref, coordinator)
    require(value == expected, "running handoff must preserve every predecessor field and actual current identity")
    observation = {"path": path, "identity": identity(path)}
    require(latest.latest_handoff() == observation, "unknown later serial handoff detected")
    return {"reference": value_ref, "observation": observation, "value": value}
