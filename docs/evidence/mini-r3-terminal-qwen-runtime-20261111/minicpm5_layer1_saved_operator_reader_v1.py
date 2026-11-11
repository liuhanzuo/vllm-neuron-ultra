"""Explicit bounded capture-reader wiring; no import-time IO or admission.

All metadata and prerequisite references must be authenticated by the root's
original serial caller, after native model/recorder/auditor genuine exits and
the complete SDK recovered-data chain. This helper does not authenticate those
terminals, acquire locks, inspect services, scan directories or launch models.
Existing complete native936/SDK624 control audits remain prerequisites.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from research import minicpm5_layer1_saved_operator_comparison_v1 as comparison
from research import minicpm5_layer1_operator_native_results_v2 as native
from research import minicpm5_layer1_operator_native_audit_io_v1 as stable
import minicpm5_layer1_operator_sdk_trace_observer_v2 as sdk_trace

require = comparison.require
ANCHOR = "strict_anchor_s0"
MAX_METADATA_BYTES = 16 * 2**20
PREREQUISITES = (
    "native_model_exit", "native_recorder_terminal", "native_audit_result", "native_auditor_terminal",
    "native_plan", "native_driver", "sdk_recovery_actual_exit", "sdk_recovery_recorder_terminal",
    "sdk_plan", "sdk_rank0_result", "sdk_rank1_result", "source_review",
)


def reference(value):
    require(type(value) is dict and set(value) == {"path", "sha256"}, "Exact authenticated metadata reference")
    stable.canonical(value["path"])
    require(type(value["sha256"]) is str and re.fullmatch("[a-f0-9]{64}", value["sha256"]), "Complete metadata SHA256")
    return copy.deepcopy(value)


def descriptor(ref, path, size):
    """Pure original raw descriptor validation; original stride is not a byte address."""
    stable.canonical(path)
    require(type(size) is int and 0 < size <= comparison.native.MAX_TENSOR_BYTES
            and type(ref) is dict and set(ref) == {"path", "raw_bytes", "sha256", "identity"}
            and ref["path"] == path and type(ref["raw_bytes"]) is int and ref["raw_bytes"] == size,
            "Exact complete original capture descriptor path and extent")
    reference({"path": path, "sha256": ref["sha256"]})
    ident = ref["identity"]
    require(type(ident) is dict and set(ident) == {"realpath", "device", "inode", "size", "mtime_ns", "ctime_ns"}
            and ident["realpath"] == path and ident["size"] == size
            and all(type(ident[k]) is int and ident[k] >= 0 for k in ("device", "inode", "size", "mtime_ns", "ctime_ns"))
            and ident["inode"] > 0, "Full stable captured file identity, never path/hash alone")
    return copy.deepcopy(ref)


def leaf(label, path, wanted, value, filename):
    require(type(value) is dict and set(value) == {"metadata", "reference"}, "Exact captured tensor leaf")
    return {"label": label, "path": list(path), "spec": copy.deepcopy(wanted),
            "metadata": copy.deepcopy(value["metadata"]),
            "reference": descriptor(value["reference"], str(filename), wanted["bytes"])}


def native_descriptors(row, case, output_root, expected_trace):
    """Extract all30 descriptors without reading any raw bytes."""
    root = Path(stable.canonical(output_root)) / "native-operator" / ("frame-%02d" % case["frame"]["index"]) / "full_trace"
    require(row["frame"] == case["frame"] and row["call"] == case["call"] and row["arm"] == "full_trace"
            and row["source_inputs"] == case and row["official_full_bytes_exact"] == dict(attention=True, key=True, value=True),
            "Exact independently audited original full_trace frame and source manifest")
    capture = row["capture"]
    native.trace.validate_capture_summary(capture, expected_trace)
    require(capture["events"] == [e["label"] for e in case["events"]] and capture["layer_index"] == 1
            and capture["arm"] == "HF_native" and capture["rank"] == -1 and capture["call"] == case["call"]
            and capture["frames"] == [case["frame"]] and capture["phase"] == "decode"
            and capture["diagnostic_only"] is capture["full_model_endpoint_gate_still_required"] is True
            and capture["production_runtime_tested"] is False and len(capture["snapshots"]) == 30,
            "Complete original native event/frame scope")
    output, threads = [], set()
    def descend(value, wanted, label, context, path=()):
        if type(wanted) is list:
            require(type(value) is list and len(value) == len(wanted), "Complete nested native cache pair")
            for index, (child, spec) in enumerate(zip(value, wanted)):
                descend(child, spec, label, context, path + (index,))
        elif "shape" in wanted:
            native.capture_metadata(value["metadata"], wanted, context)
            output.append(leaf(label, path, wanted, value, root / ("observed-%02d.bin" % len(output))))
        else:
            require(type(value) is dict and wanted == {"json_only": True}, "Original metadata-only native event")
    for index, (event, record) in enumerate(zip(case["events"], capture["snapshots"])):
        require(set(record) == {"label", "context", "value"} and record["label"] == event["label"], "Exact native event schema/order")
        native.event_context(record["context"], index, case); threads.add(record["context"]["thread"])
        descend(record["value"], event["value"], event["label"], record["context"])
        if event["label"] == "qk0_scale_operation":
            require(record["value"] == capture["scale_provenance"][0], "Exact actual Python scale provenance")
        if event["label"] == "rotary_route":
            require(record["value"] == dict(function="native.apply_rotary_pos_emb",
                    position_source="original_rotary_module_on_saved_official_decode_input", direct_layer1_replay=True),
                    "Original direct native rotary provenance")
    require(len(output) == 30 and len(threads) == 1
            and sum(r["spec"]["bytes"] for r in output) == case["observer_raw_bytes"], "All30 native leaves and full bytes")
    return output


def sdk_descriptors(observation, rank, call, bucket, output_root, expected_trace):
    """SDK snapshots descriptor phase mirrors existing full auditor, without IO."""
    expected = comparison.sdk.call_manifest(rank, call["index"], bucket)
    frames = sorted(call["frames"], key=lambda frame: frame["slot"])
    require([frame["slot"] for frame in frames] == list(range(4)), "Every original SDK physical row")
    capture = observation["capture"]; summary = capture["summary"]; snapshot = capture["snapshots"]
    sdk_trace.validate_sdk_capture_summary(summary, expected_trace)
    require(observation["mode"] == "official_full" and summary["layer_index"] == 1 and summary["rank"] == rank
            and summary["arm"] == "C_Hd_control" and summary["call"] == call and summary["frames"] == frames
            and summary["events"] == [e["label"] for e in expected["tensor_events"]], "Exact SDK capture call/rank/physical frames")
    require(capture["full_backing_expectation"] == "unchanged_direct_attention_update_disabled"
            and snapshot["complete"] is True and snapshot["manifest"] == expected
            and snapshot["raw_bytes"] == expected["capture_raw_bytes"]
            and len(snapshot["records"]) == expected["event_count"], "Full SDK manifest and source backing contract")
    root = Path(stable.canonical(output_root)) / ("cpu-rank%d" % rank) / ANCHOR / ("call-%02d" % call["index"]) / "official_full" / "capture"
    def leaves(value, path=()):
        if type(value) is dict and set(value) == {"metadata", "reference"}: return [(path, value)]
        if type(value) is dict: return [r for key, child in value.items() for r in leaves(child, path + (key,))]
        if type(value) is list: return [r for index, child in enumerate(value) for r in leaves(child, path + (index,))]
        return []
    output, threads = [], set()
    for index, (event, record) in enumerate(zip(expected["tensor_events"], snapshot["records"])):
        require(set(record) == {"label", "context", "value"} and record["label"] == event["label"], "Exact SDK event schema/order")
        context = record["context"]; threads.add(context["thread"])
        require(context["event_index"] == index and context["arm"] == "C_Hd_control" and context["rank"] == rank
                and context["call"] == call and context["frames"] == frames and context["phase"] == "decode"
                and type(context["thread"]) is int and context["thread"] > 0, "Exact original SDK event identity")
        actual = leaves(record["value"]); paths = [tuple(t["path"]) for t in event["tensors"]]
        require(len(actual) == len(paths) and len({p for p, _ in actual}) == len(actual)
                and {p for p, _ in actual} == set(paths), "Every exact SDK nested tensor path")
        by_path = dict(actual)
        for tensor_index, wanted in enumerate(event["tensors"]):
            path = tuple(wanted["path"]); value = by_path[path]
            enriched = dict(context, event_label=event["label"], tensor_index=tensor_index, tensor_path=wanted["path"])
            native.capture_metadata(value["metadata"], wanted, enriched)
            output.append(leaf(event["label"], path, wanted, value,
                               root / ("%02d-%s-%02d.bin" % (index, event["label"], tensor_index))))
    require(len(output) == 47 and len(threads) == 1
            and sum(r["spec"]["bytes"] for r in output) == expected["capture_raw_bytes"], "All47 SDK leaves and full bytes")
    return output


def prepare(native_plan, sdk_plan, native_driver, native_full_rows, sdk_rank_results,
            native_trace, sdk_trace_contract, prerequisite_refs):
    """Pure descriptor plan. Supplied metadata/refs must already be authenticated.

    native_full_rows maps original frame index to authenticated full_trace JSON.
    refs identify real prerequisite artifacts but are not terminal validation:
    the root's caller must authenticate both lifecycle chains and audit controls.
    """
    require(type(prerequisite_refs) is dict and set(prerequisite_refs) == set(PREREQUISITES), "Every real prerequisite reference required")
    refs = {name: reference(prerequisite_refs[name]) for name in PREREQUISITES}
    manifest = native_plan["native_operator_manifest"]
    contract = comparison.contract(manifest, sdk_plan["sdk_operator_manifest"])
    require(type(native_full_rows) is dict and set(native_full_rows) == set(comparison.INDICES), "All24 original full_trace JSON inputs")
    require(native_driver["schema"] == "minicpm5_layer1_operator_native_driver_v4" and native_driver["complete"] is True
            and native_driver["manifest_sha256"] == comparison.digest(manifest) and len(native_driver["records"]) == 24
            and native_driver["raw_files"] == 936 and native_driver["raw_bytes"] == 28490724
            and native_driver["all_unobserved_noop_full_endpoints_and_full_caches_exact"] is True,
            "Original complete native driver and all three controls, not a substitute for independent audit")
    require(type(sdk_rank_results) is list and len(sdk_rank_results) == 2, "Both original SDK rank metadata inputs")
    for rank, result in enumerate(sdk_rank_results):
        require(result["schema"] == "minicpm5_layer1_operator_sdk_driver_v2" and result["complete"] is True
                and result["rank"] == rank and result["manifest"] == sdk_plan["sdk_operator_manifest"]
                and [r["result"]["call"]["index"] for r in result["records"]] == list(comparison.CALLS)
                and result["all_previous_controls_full_bytes_exact"] is True, "Both complete original SDK six-call metadata sets")
    calls = sdk_plan["calls"]
    require(len(calls) == 14 and [c["index"] for c in calls] == list(range(14)), "All14 original call definitions retained")
    native_rows, native_refs = {}, {}
    for report, case in zip(native_driver["records"], manifest["frames"]):
        require(report["frame"] == case["frame"] and report["all_three_arms_official_exact"] is True
                and [a["arm"] for a in report["arms"]] == list(comparison.native.ARMS), "Original native frame and all three controls")
        selected = report["arms"][2]; ref = selected["result"]; index = case["frame"]["index"]
        wanted = str(Path(native_plan["output_root"]) / "native-operator" / ("frame-%02d" % index) / "full_trace" / "result.json")
        require(type(ref) is dict and set(ref) == {"path", "sha256", "bytes"} and ref["path"] == wanted
                and type(ref["bytes"]) is int and 0 < ref["bytes"] <= 2**20, "Exact audited bounded full_trace metadata reference")
        reference({k: ref[k] for k in ("path", "sha256")})
        require(native_full_rows[index]["outputs"] == selected["outputs"], "Native full_trace endpoint descriptors retained")
        native_rows[index] = native_descriptors(native_full_rows[index], case, native_plan["output_root"], native_trace)
        native_refs[str(index)] = {"reference": copy.deepcopy(ref), "parsed_canonical_sha256": comparison.digest(native_full_rows[index])}
    output = []
    for call, bucket in zip(comparison.CALLS, comparison.sdk.BUCKETS):
        cases = [c for c in manifest["frames"] if c["call"] == call]; original = calls[call]
        require(original["phase"] == "decode" and original["frames"] == [c["frame"] for c in cases]
                and original["logical_positions"] == [c["logical_position"] for c in cases], "SDK call bound to exact native fixtures/slots/positions")
        ranks = {}
        for rank, result in enumerate(sdk_rank_results):
            row = result["records"][comparison.CALLS.index(call)]["result"]
            require(row["call"] == original and row["rank"] == rank and row["arm"] == ANCHOR
                    and [r["mode"] for r in row["observations"]] == ["official_unobserved", "official_noop", "official_full"],
                    "Exact original SDK call and all three observation controls")
            ranks[str(rank)] = sdk_descriptors(row["observations"][-1], rank, original, bucket, sdk_plan["output_root"], sdk_trace_contract)
        output.append({"call": call, "native": {str(c["frame"]["index"]): native_rows[c["frame"]["index"]] for c in cases}, "sdk": ranks})
    result = {"schema": "minicpm5_saved_operator_descriptor_plan_v1", "contract": contract,
              "prerequisite_refs": refs, "native_full_trace_metadata": native_refs,
              "authenticated_input_metadata_canonical_sha256": {
                  "native_plan": comparison.digest(native_plan), "sdk_plan": comparison.digest(sdk_plan),
                  "native_driver": comparison.digest(native_driver),
                  **{"sdk_rank%d_result" % rank: comparison.digest(row) for rank, row in enumerate(sdk_rank_results)}},
              "calls": output, "prerequisites_authenticated_here": False, "runtime_admitted": False,
              "raw_payloads_read": False, "full_model_qualified": False, "new_speedup_claim": False}
    validate_plan(result)
    require(len(json.dumps(result, separators=(",", ":"), allow_nan=False).encode()) <= MAX_METADATA_BYTES, "Bounded complete descriptor metadata, never truncate")
    return result


def validate_plan(plan):
    require(plan["schema"] == "minicpm5_saved_operator_descriptor_plan_v1"
            and [c["call"] for c in plan["calls"]] == list(comparison.CALLS), "Exact complete six-call descriptor plan")
    contract = plan["contract"]; all_refs = {}; nodes = set(); sizes = []
    require(contract["native_manifest_sha256"] == comparison.NATIVE_MANIFEST_SHA256
            and contract["sdk_manifest_sha256"] == comparison.digest(comparison.sdk.manifest()), "Original pinned geometry contract")
    for call in plan["calls"]:
        cases = [c for c in contract["frames"] if c["call"] == call["call"]]
        require(set(call["native"]) == {str(c["frame"]["index"]) for c in cases} and set(call["sdk"]) == {"0", "1"}, "Four exact frames and both ranks")
        count = total = 0
        for side, groups in (("native", call["native"]), ("sdk", call["sdk"])):
            for key, rows in groups.items():
                case = next((c for c in cases if str(c["frame"]["index"]) == key), None) if side == "native" else cases[0]
                specs = comparison.native_specs(case["native_events"]) if side == "native" else comparison.sdk_specs(case["bucket"])
                keys = [(r["label"], tuple(r["path"])) for r in rows]
                require(len(keys) == len(set(keys)) == len(specs) and set(keys) == set(specs), "Complete unique leaf domains in descriptor plan")
                for row in rows:
                    spec = specs[row["label"], tuple(row["path"])]; ref = row["reference"]
                    require(row["spec"] == spec, "Original full packed logical geometry retained")
                    descriptor(ref, ref["path"], spec["bytes"])
                    node = ref["identity"]["device"], ref["identity"]["inode"]
                    require(ref["path"] not in all_refs and node not in nodes, "No repeated capture path or inode across1284 leaves")
                    all_refs[ref["path"]] = copy.deepcopy(ref); nodes.add(node); count += 1; total += spec["bytes"]
        require(count == 214 and total <= contract["maximum_call_input_bytes"] <= comparison.MAX_CALL_BYTES, "All214 leaves within exact one-call budget")
        sizes.append(total)
    require(len(all_refs) == 1284 and sum(sizes) == contract["total_capture_bytes"] == 77790564, "All1284 files and77790564 original bytes")
    return all_refs


class StableCaptureIO:
    """No scans: read only exact descriptor map under an explicit caller guard.

    filesystem is mandatory; production caller explicitly supplies os, tests a
    tiny memory FS. NativeAuditIO's existing full no-follow parent-chain check
    is reused without its inventory/lifecycle methods. External complete audit
    and current physical guard are still prerequisites, not inferred here.
    """
    _directory = stable.NativeAuditIO._directory
    _parents = stable.NativeAuditIO._parents

    def __init__(self, allowed, guard, *, filesystem):
        require(type(allowed) is dict and 0 < len(allowed) <= 1284 and callable(guard), "Bounded exact descriptor allowlist and explicit guard")
        self.allowed = {path: descriptor(ref, path, ref["raw_bytes"]) for path, ref in allowed.items()}
        self.guard, self.fs, self._read = guard, filesystem, set()

    def read_bytes(self, ref, path, size):
        self.guard(); descriptor(ref, path, size)
        require(self.allowed.get(path) == ref and path not in self._read, "Only one complete read of each exact admitted capture")
        with self._parents(path) as (parent, name, parents_check):
            before = self.fs.stat(name, dir_fd=parent, follow_symlinks=False)
            def check(info):
                require(stat.S_ISREG(info.st_mode) and info.st_uid == 1000 and info.st_nlink == 1 and info.st_size == size
                        and stable.identity(path, info) == ref["identity"] and stable.tree_identity(info) == stable.tree_identity(before),
                        "Owned stable singleton regular capture, complete original identity")
            check(before)
            fd = self.fs.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
            try:
                check(self.fs.fstat(fd)); parts = []; count = 0; digest = hashlib.sha256()
                while True:
                    self.guard(); raw = self.fs.read(fd, min(65536, size - count + 1))
                    require(type(raw) is bytes, "Only raw bytes from explicitly supplied FS")
                    if not raw: break
                    count += len(raw); require(count <= size, "Complete capture cannot grow beyond exact extent")
                    parts.append(raw); digest.update(raw)
                require(count == size and digest.hexdigest() == ref["sha256"], "Whole capture extent/EOF/SHA256")
                check(self.fs.fstat(fd)); parents_check(); check(self.fs.stat(name, dir_fd=parent, follow_symlinks=False))
            finally: self.fs.close(fd)
        self.guard(); self._read.add(path)
        return b"".join(parts)


def compare_all(plan, read_bytes, guard, emit_report, *, expected_plan_sha256):
    """Six sequential complete calls; emits metadata only, retains no raw cache.

    Call only from the root's separately admitted serial operation. Any guard,
    read, geometry, comparison or sink exception aborts; no partial completion.
    emit_report receives a copy, so it cannot mutate retained summary evidence.
    """
    require(all(callable(f) for f in (read_bytes, guard, emit_report)), "Explicit stable reader, live guard and metadata sink")
    guard(); frozen = comparison.digest(plan)
    require(type(expected_plan_sha256) is str and re.fullmatch("[a-f0-9]{64}", expected_plan_sha256)
            and frozen == expected_plan_sha256, "Caller-authenticated original complete descriptor-plan canonical SHA")
    expected = validate_plan(plan)
    reader = native.Reader(read_bytes, guard); reports = []
    for call in plan["calls"]:
        guard(); require(comparison.digest(plan) == frozen, "Descriptor plan unchanged between complete calls")
        def read_groups(groups):
            return {int(index): {(r["label"], tuple(r["path"])):
                    reader.get(r["reference"], r["reference"]["path"], r["spec"]["bytes"], identity=True)
                    for r in rows} for index, rows in groups.items()}
        native_values = read_groups(call["native"]); sdk_values = read_groups(call["sdk"])
        guard(); require(comparison.digest(plan) == frozen, "Original descriptor plan retained across complete raw reads")
        report = comparison.compare_call(plan["contract"], call["call"], native_values, sdk_values, guard)
        del native_values, sdk_values
        reports.append(report); guard(); emit_report(copy.deepcopy(report)); guard()
    require(reader.seen == expected and len(reader.seen) == 1284 and sum(reader.bytes.values()) == 77790564
            and comparison.digest(plan) == frozen, "Every original capture independently reread once with no descriptor mutation")
    summary = comparison.summarize(plan["contract"], reports)
    summary.update(descriptor_plan_sha256=frozen, prerequisite_refs=copy.deepcopy(plan["prerequisite_refs"]),
                   source_geometry_retained_in_descriptor_plan=True, raw_addressing="canonical_packed_logical_shape_only",
                   capture_payloads_reread_in_this_pass=True, runtime_admitted=False,
                   prerequisite_terminal_authentication_by_caller_required=True)
    guard(); return summary
