"""Reviewed-entry candidate for serial saved-data comparison, never model work.

No import-time admission/IO. run() is the sole actual entry: original retained
lock, actual terminal reconstruction, complete inherited sources/catalog and
double physical observations precede the1284 capture reads. This process never
claims its own tool exit or model/performance qualification. No current spec,
catalog or future terminal references are created by this source delivery.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "research"
sys.path[:0] = [str(Q), str(ROOT)]
from research import audit_minicpm5_layer1_operator_native_result_v4 as audited
from research import prepare_minicpm5_layer1_operator_native_v7 as preparation
from research import minicpm5_layer1_saved_operator_reader_v1 as reader
from research import minicpm5_layer1_operator_sdk_results_v2 as sdk_results_core

r = audited.r
require = r.require
core = audited.core
SELF = str(Path(__file__).resolve())
LIMITS = copy.deepcopy(audited.LIMITS)
OUTPUT_RESERVE = 16 * 2**20
REPORT_SCHEMA = "minicpm5_saved_operator_serial_call_report_v1"
SUMMARY_SCHEMA = "minicpm5_saved_operator_serial_summary_v1"


def output_paths(prefix):
    path = Path(prefix)
    require(type(prefix) is str and str(path) == prefix and path.parent == Q
            and re.fullmatch(r"minicpm5-layer1-saved-operator-comparison-[a-z0-9][a-z0-9-]{0,63}", path.name),
            "Fresh bounded research-only saved-comparison output prefix")
    return {**{str(call): Path(prefix + "-call-%02d.json" % call) for call in reader.comparison.CALLS},
            "summary": Path(prefix + "-summary.json")}


def validate_spec(spec_ref, review_ref, spec, review):
    for ref in (spec_ref, review_ref): r.life.ref(ref)
    require(type(spec) is dict and set(spec) == {"schema", "model_exit", "recorder_exit", "auditor_exit",
            "current_serial_catalog", "source_extension", "output_prefix", "limits"}
            and spec["schema"] == "minicpm5_saved_operator_serial_comparison_spec_v1" and spec["limits"] == LIMITS,
            "Exact small comparison spec retains original audit limits")
    for name in ("model_exit", "recorder_exit", "auditor_exit", "current_serial_catalog"): r.life.ref(spec[name])
    require(type(review) is dict and set(review) == {"schema", "reviewer", "passed", "blockers", "spec", "source_extension"}
            and review["schema"] == "minicpm5_saved_operator_serial_comparison_source_review_v1"
            and review["reviewer"] == "/root" and review["passed"] is True and review["blockers"] == []
            and review["spec"] == spec_ref and review["source_extension"] == spec["source_extension"],
            "Independent exact comparison source review required")
    pins = spec["source_extension"]
    require(type(pins) is dict and 0 < len(pins) <= 2048, "Only bounded additive comparison text sources")
    for path, sha in pins.items():
        require(type(path) is str and Path(path).is_absolute() and str(Path(path)) == path and ".." not in Path(path).parts
                and path.startswith(str(ROOT) + "/") and Path(path).suffix in (".py", ".json", ".md", ".txt", ".sh")
                and type(sha) is str and re.fullmatch("[a-f0-9]{64}", sha), "Exact local text source addition")
    return output_paths(spec["output_prefix"])


def audited_controls(result, model_exit, recorder_exit, rank_refs, manifest):
    """Validate the real source-authenticated completed audit's full control proof.

    This consumes saved evidence, not a claim of redoing936 control payloads.
    The authentic auditor terminal and source witness are separately mandatory.
    """
    require(result["schema"] == "minicpm5_native_post_exit_independent_audit_v1" and result["complete"] is True
            and result["reviewer"] == "/root" and result["model_exit"] == model_exit and result["recorder_exit"] == recorder_exit
            and result["lock_identity"] == r.ORIGINAL_LOCK and result["sdk_rank_references"] == rank_refs
            and type(result["original_sdk_model_exit_code"]) is int and result["original_sdk_model_exit_code"] == 1,
            "Original completed independent native audit and failed SDK provenance")
    for key in ("actual_native_model_exit_verified", "actual_native_recorder_exit_verified", "every_raw_byte_reread_after_all_model_processes_exited"):
        require(result[key] is True, "Every original independent audit guarantee: " + key)
    for key in ("actual_auditor_tool_exit_verified", "self_exit_claimed", "full_model_qualified", "new_speedup_claim"):
        require(result[key] is False, "Auditor did not claim its own tool exit or qualification: " + key)
    proof = result["result"]
    require(proof["schema"] == "minicpm5_layer1_operator_native_independent_verification_v2" and proof["complete"] is True
            and [proof[k] for k in ("frames", "direct_attention_calls", "raw_files", "raw_bytes", "arm_result_files", "journals", "journal_records")]
            == [24, 72, 936, 28490724, 72, 72, 2736]
            and proof["raw_bytes"] == manifest["total_raw_bytes"], "Full native24/72/936 raw and every durable journal control")
    for key in ("every_raw_byte_independently_reread", "every_journal_record_independently_verified",
                "all_official_attention_and_full_cache_controls_exact", "every_capture_event_shape_stride_offset_dtype_retained",
                "original_scale_source_independently_bound", "original_projection_matrix_hashes_bound"):
        require(proof[key] is True, "Original complete native control proof: " + key)
    for key in ("actual_process_exit_verified", "fresh_physical_drain_performed", "arithmetic_reexecuted", "full_model_qualified", "new_speedup_claim"):
        require(proof[key] is False, "Independent pure proof preserves runtime/qualification boundary: " + key)
    require([row["frame"] for row in proof["frames_verified"]] == [row["frame"] for row in manifest["frames"]]
            and all(row["all_three_arms_exact"] is True for row in proof["frames_verified"]), "Every exact original frame and all three arms")
    return copy.deepcopy(proof)


def actual_auditor_terminal(value, model_tool, recorder_tool):
    """Same existing root auditor terminal schema, checked before reconstruction."""
    require(type(value) is dict and set(value) == {"schema", "reviewer", "audit_spec", "audit_review", "result", "owner", "actual_tool",
            "actual_tool_exit_observed", "exact_owner_gone", "lock_identity", "self_exit_claimed", "full_model_qualified", "new_speedup_claim"}
            and value["schema"] == "minicpm5_native_independent_audit_actual_exit_v1" and value["reviewer"] == "/root"
            and value["actual_tool_exit_observed"] is value["exact_owner_gone"] is True and value["lock_identity"] == r.ORIGINAL_LOCK
            and value["self_exit_claimed"] is value["full_model_qualified"] is value["new_speedup_claim"] is False,
            "Existing actual externally observed independent auditor terminal")
    tool = core.actual_tool(value["actual_tool"])
    require(tool["session_id"] != model_tool["session_id"]
            and (recorder_tool["kind"] == "tool_call" or tool["session_id"] != recorder_tool["session_id"]),
            "Independent audit has its own actual tool session")
    for name in ("audit_spec", "audit_review", "result"): r.life.ref(value[name])
    return tool


def serial_scope(resolved, serial, checked, recorder):
    owners = copy.deepcopy(resolved["current_scope"]["owners"]) + copy.deepcopy(serial["owners"]) + checked["model_owners"] + [recorder["owner"]]
    units = set(resolved["current_scope"]["complete_known_units"]) | set(serial["units"]) | {checked["contract"]["unit"]}
    for owner in owners:
        group = owner.get("cgroup")
        if type(group) is str and group.startswith("0::" + r.p.GROUP_PREFIX): units.add(r.completion.physical.group_unit(group[3:]))
    return dict(resolved["current_scope"], owners=owners, later_serial=serial, complete_known_units=sorted(units))


def catalog_rows(serial, older, expected):
    rows = {row["terminal"]["path"]: row for row in serial["runs"]}
    require(len(rows) == len(serial["runs"]) and all(rows.get(row["terminal"]["path"]) == row for row in older["runs"]),
            "Every exact old serial row preserved, no replacement or omission")
    for ref, owner, tool in expected:
        row = rows.get(ref["path"])
        require(row is not None and row["terminal"] == ref and row["tool"] == tool and owner in row["owners"],
                "Current full catalog binds exact true terminal, owner and tool")
    return rows


def source_plan(resolved, ended, model_spec, audit_spec, result, spec):
    inherited = audited.validate_source_binding(result["source_binding"], resolved["plan"], audit_spec["source_extension"],
        ended["spec"], ended["review"], model_spec["overlay"], audited.SELF)
    complete = audited.merged_plan(inherited, spec["source_extension"])
    for path in (SELF, str(Path(reader.__file__).resolve()), str(Path(reader.comparison.__file__).resolve()),
                 str(Path(preparation.__file__).resolve()), audited.SELF):
        require(path in complete["source_sha256"], "Every actual comparison entry/helper in full inherited source closure")
    return complete, dict(schema="minicpm5_saved_operator_complete_source_binding_v1",
        model_spec=copy.deepcopy(ended["spec"]), model_review=copy.deepcopy(ended["review"]), model_overlay=copy.deepcopy(model_spec["overlay"]),
        original_model_plan_sha256=resolved["plan_sha256"], original_model_source_map_sha256=r.digest(resolved["plan"]["source_sha256"]),
        original_model_source_files=len(resolved["plan"]["source_sha256"]),
        completed_auditor_source_binding_sha256=r.digest(result["source_binding"]),
        inherited_auditor_source_map_sha256=r.digest(inherited["source_sha256"]), inherited_auditor_source_files=len(inherited["source_sha256"]),
        new_source_map_sha256=r.digest(spec["source_extension"]), new_source_files=len(spec["source_extension"]),
        complete_source_map_sha256=r.digest(complete["source_sha256"]), complete_source_files=len(complete["source_sha256"]),
        inherited_sources_omitted=False, inherited_sources_replaced=False, complete_source_reconstruction_required=True)


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def write_report(path, value, guard):
    guard(force=True); raw = encoded(value)
    require(len(raw) <= LIMITS["output_file_bytes"], "Entire report exceeds original2MiB bound; never truncate")
    require(not Path(path).exists(), "Never overwrite an existing comparison artifact")
    r.atomic_once(Path(path), value); ref = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    require(audited.outer.reference(Path(path)) == ref, "Complete exclusive metadata output digest")
    guard(force=True); return ref


def check_sdk_controls(results, sdk_plan, guard):
    """Recheck all60 old SDK control identities without reading their payloads."""
    refs = {}
    for rank, result in enumerate(results):
        require(len(result["records"]) == 6, "All six original SDK control rows")
        for call, row in zip(reader.comparison.CALLS, result["records"]):
            require([item["variant"] for item in row["outputs"]] == list(reader.comparison.sdk.MODES), "All five original control variants")
            for item in row["outputs"]:
                path = str(Path(sdk_plan["output_root"]) / ("cpu-rank%d" % rank) / reader.ANCHOR /
                           ("call-%02d" % call) / item["variant"] / "attention.bf16")
                require(path not in refs, "Unique original control path")
                refs[path] = reader.descriptor(item["output"], path, 16384)
    require(len(refs) == 60, "All60 original SDK endpoint controls retained")
    io = reader.StableCaptureIO(refs, guard, filesystem=os)
    for path, ref in refs.items():
        guard()
        with io._parents(path) as (parent, name, parents_check):
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISREG(before.st_mode) and before.st_uid == 1000 and before.st_nlink == 1
                    and reader.stable.identity(path, before) == ref["identity"], "Original SDK control identity unchanged since complete recovery audit")
            parents_check()
            require(reader.stable.tree_identity(os.stat(name, dir_fd=parent, follow_symlinks=False)) == reader.stable.tree_identity(before),
                    "SDK control identity remains stable during no-byte check")
    return {"controls": 60, "raw_bytes_read": 0, "descriptors_sha256": r.digest(refs),
            "complete_old_byte_comparison_from_authenticated_recovery": True}


def authenticated_sdk_trace(recovered, results, rank_refs, read, read_source, guard):
    """Retain the exact SDK contract authenticated by its completed raw auditor.

    Native and SDK were captured under different Python compilers. Never
    recompile SDK under native Python, drop line tables, or trust unverified
    snapshot metadata. The completed SDK recovery audited snapshots against
    expected_trace_contract in its own interpreter and bound the exact rank
    JSON refs. Those genuine completed witnesses are mandatory here.
    """
    guard()
    require(recovered["binding"] == r.completion.BINDING and recovered["metadata_complete"] is True
            and recovered["runtime_admitted"] is False and type(recovered["original_model_exit_code"]) is int
            and recovered["original_model_exit_code"] == 1, "Actually completed SDK recovery, original failed runtime retained")
    proof = recovered["recovery"]
    require(proof["schema"] == "minicpm5_operator_sdk_failed_run_recovery_audit_v1" and proof["audit_complete"] is True
            and proof["original_exit_code"] == 1 and proof["original_runtime_successful"] is False,
            "Real independent SDK saved-data audit, never original model success")
    r.completion.core.validate_proof(proof["result"])
    require(recovered["actual_exit"]["raw_proof"] == proof["result"], "Genuine completed recovery exit binds exact complete raw proof")
    for key in ("every_raw_byte_reread_after_all_model_processes_exited", "all_endpoint_controls_full_bytes_exact",
                "all_capture_journals_verified", "sdk_rope_payload_independently_reread"):
        require(proof[key] is True, "Original full independent SDK observation audit: " + key)
    source_path = str(Path(sdk_results_core.__file__).resolve())
    loaded = proof["loaded_source_closure"]
    require(loaded["complete"] is True and loaded["source_sha256"].get(source_path) == recovered["source_sha256"].get(source_path)
            and source_path in loaded["source_sha256"], "Actual completed auditor executed the original full SDK snapshot verifier")
    verifier_sha = loaded["source_sha256"][source_path]
    require(hashlib.sha256(read_source(source_path, verifier_sha)).hexdigest() == verifier_sha,
            "Original SDK verifier that compared source/code/line contracts remains unchanged")
    expected_source = dict(path=reader.sdk_trace.SOURCE_PATH, sha256=reader.sdk_trace.SOURCE_SHA)
    require(recovered["source_sha256"].get(expected_source["path"]) == expected_source["sha256"]
            and hashlib.sha256(read_source(expected_source["path"], expected_source["sha256"])).hexdigest() == expected_source["sha256"],
            "Full original SDK source text/path/SHA retained without recompiling in another interpreter")
    require(type(results) is list and len(results) == len(rank_refs) == 2, "Both complete independently audited SDK rank results")
    selected = None; bound_rows = []
    for rank, (ref, result) in enumerate(zip(rank_refs, results)):
        require(ref == proof["original_model_outputs"]["rank%d_result" % rank]
                and recovered["authenticated_metadata"].get(ref["path"]) == ref
                and read(ref) == result, "Exact authenticated rank JSON from the completed independent audit, not caller-forged capture metadata")
        require(result["rank"] == rank and [row["result"]["call"]["index"] for row in result["records"]] == list(reader.comparison.CALLS),
                "Every original call in both SDK ranks")
        for row in result["records"]:
            observations = row["result"]["observations"]
            require([entry["mode"] for entry in observations] == ["official_unobserved", "official_noop", "official_full"],
                    "Complete original SDK observation controls")
            full = observations[-1]; value = full["capture"]["summary"]["trace_contract"]; trace = full["trace"]
            require(type(value) is dict and set(value) == {"route", "source", "function", "line_metadata", "points"}
                    and value["route"] == "sdk_decode" and value["source"] == expected_source and value["function"] == "compute_for_token_gen",
                    "Exact saved original SDK function/route/source contract")
            lines = value["line_metadata"]
            require(type(lines) is dict and set(lines) == {"first_line", "co_linetable_sha256", "co_lnotab_sha256"}
                    and type(lines["first_line"]) is int and lines["first_line"] > 0
                    and all(type(lines[key]) is str and re.fullmatch("[a-f0-9]{64}", lines[key]) for key in ("co_linetable_sha256", "co_lnotab_sha256")),
                    "Every originally saved line-table field retained, not normalized across interpreters")
            require(type(value["points"]) is list and len(value["points"]) == 2, "Both saved source-derived SDK scale points")
            for point, role, variable in zip(value["points"], ("qk0", "qk1"), ("prior_scores", "active_scores")):
                require(type(point) is dict and set(point) == {"role", "variable", "assignment_line", "event_line"}
                        and point["role"] == role and point["variable"] == variable
                        and type(point["assignment_line"]) is int and type(point["event_line"]) is int
                        and lines["first_line"] < point["assignment_line"] < point["event_line"],
                        "Complete original saved code-local/assignment/event mapping")
            require(trace["contract"] == value and trace["mode"] == "full_trace" and trace["calls"] == 1
                    and trace["other_phase_calls"] == 0 and trace["roles"] == ["qk0", "qk1"]
                    and trace["observed_scale_events"] == 2 and trace["noop_scale_events"] == 0 and trace["aten_dispatch_used"] is False,
                    "Actual original full SDK trace and saved summary agree")
            reader.sdk_trace.validate_sdk_capture_summary(full["capture"]["summary"], value)
            if selected is None: selected = copy.deepcopy(value)
            require(value == selected, "All12 independently audited SDK contracts exactly agree, including all line/code fields")
            bound_rows.append(dict(rank=rank, call=row["result"]["call"]["index"], rank_result=copy.deepcopy(ref), contract_sha256=r.digest(value)))
    require(len(bound_rows) == 12, "Complete original SDK two-rank six-call contract coverage")
    guard()
    return selected, dict(schema="minicpm5_saved_operator_authenticated_sdk_trace_v1", exact_saved_contract=selected,
        contract_sha256=r.digest(selected), all12_original_contracts_exact=True, rows=bound_rows,
        completed_recovery=copy.deepcopy(r.completion.BINDING), recovery_result=copy.deepcopy(recovered["actual_exit"]["result"]),
        complete_recovery_raw_proof_sha256=r.digest(proof["result"]),
        executed_original_snapshot_verifier=dict(path=source_path, sha256=verifier_sha),
        validation_basis="genuine_completed_independent_SDK_raw_audit_of_exact_hash_bound_rank_JSON",
        source_recompiled_in_native_interpreter=False, line_or_code_fields_dropped=False,
        new_raw_bytes_read=0, self_authenticated_capture_metadata=False)


class ComparisonGuard(audited.AuditGuard):
    """Reuse original lease/native-tree guard, adding only report-space reserve."""
    def __call__(self, *, force=False):
        super().__call__(force=force)
        require(self.storage["available_bytes"]["root"] >= 28 * 2**30 + OUTPUT_RESERVE,
                "Original root28GiB floor plus complete bounded report reserve")


def compare(spec_ref, review_ref, lease):
    guard = ComparisonGuard(lease); guard(force=True)
    metadata = r.Metadata(guard); lease.metadata = metadata
    spec = metadata.read(spec_ref); review = metadata.read(review_ref)
    paths = validate_spec(spec_ref, review_ref, spec, review)
    require(all(not path.exists() for path in paths.values()), "All six reports and summary have fresh immutable paths")
    for path, sha in spec["source_extension"].items():
        guard(); require(hashlib.sha256(r.sources.text(path, sha)).hexdigest() == sha, "Exact reviewed comparison source addition")
    ended = metadata.read(spec["model_exit"]); recorder = metadata.read(spec["recorder_exit"]); auditor = metadata.read(spec["auditor_exit"])
    mt = core.actual_tool(ended["actual_tool"]); rt = core.actual_tool(recorder["actual_tool"], allow_call=True)
    actual_auditor_terminal(auditor, mt, rt)
    require(ended["actual_exit_observed"] is True and type(ended["actual_exit_code"]) is int and ended["actual_exit_code"] == 0,
            "True model exit0 required before raw capture access")
    for owner in (ended["owner"], ended["coordinator"], ended["native"], recorder["owner"], auditor["owner"]): r.p.gone(owner)
    r.p.active_heavy()
    live = metadata.read(ended["live"]); model_spec = metadata.read(ended["spec"]); model_review = metadata.read(ended["review"])
    # Same original recovery/resolver as audit_v4. No model/native/tensor work.
    recovered = r.completion.completed(r.completion.BINDING, metadata.read, guard, lambda path, sha: r.sources.text(path, sha))
    preparation.preserved(recovered)
    overlay = metadata.read(model_spec["overlay"])
    resolved = r.overlays.resolve(overlay, recovered, metadata.read, guard)
    cache = {r.digest([ended["spec"], ended["review"], model_spec["overlay"]]):
             dict(resolved=resolved, overlay=copy.deepcopy(overlay), spec=copy.deepcopy(model_spec), review=copy.deepcopy(model_review))}
    checked_resolved, checked_spec = preparation.inherited_source_plan(ended, metadata.read, guard, cache)
    require(checked_resolved == resolved and checked_spec == model_spec, "Original accepted model review and complete reconstructed plan")
    records = {key: metadata.read(ref) for key, ref in ended["outputs"].items()}
    checked = core.validate(spec["model_exit"], ended, spec["recorder_exit"], recorder, live, records, resolved)
    a = records["admission"]; contract = checked["contract"]
    guard.bind(contract, resolved); r.validate_initial_descriptor(a); r.execution.validate_intent(records["intent"], a)
    require(a["overlay"] == model_spec["overlay"] and a["source_contract"] == resolved["plan"]["source_contract"]
            and a["sdk_operator_recovery"] == r.completion.BINDING and a["latest_completion"] == resolved["plan"]["latest_completion"]
            and a["completed_sdk_layout"] == resolved["plan"]["layer1_operator_boundary"]
            and a["serial_predecessors"] == resolved["plan"]["serial_predecessors"], "Complete original native admission/source lineage")
    audit_spec = metadata.read(auditor["audit_spec"]); audit_review = metadata.read(auditor["audit_review"]); audit_result = metadata.read(auditor["result"])
    audited.validate_spec(auditor["audit_spec"], auditor["audit_review"], audit_spec, audit_review)
    require(audit_spec["model_exit"] == spec["model_exit"] and audit_spec["recorder_exit"] == spec["recorder_exit"], "Completed audit of this exact native run")
    pins_plan, source_proof = source_plan(resolved, ended, model_spec, audit_spec, audit_result, spec)
    model_partition = r.sources.classify(resolved["plan"]["source_sha256"], resolved["plan"]["source_contract"], metadata.read)
    inherited = r.sources.validate_native_receipt(records["source_authentication"], model_partition, ended["owner"], r.ORIGINAL_LOCK, a["attempt_id"])
    require(audit_result["inherited_native_source_authentication"] == inherited, "Audit retains original full native-source identity receipt")
    partition = r.sources.classify(pins_plan["source_sha256"], pins_plan["source_contract"], metadata.read)
    r.sources.verify_text_sources(partition, guard); r.sources.check_native_identities(records["source_authentication"], model_partition, guard)
    r.verify_loaded(pins_plan, guard)
    old_view = dict(latest_completion=resolved["plan"]["latest_completion"], serial_predecessors=audit_spec["current_serial_catalog"], source_sha256=pins_plan["source_sha256"])
    old_serial = r.completion.latest.serial_catalog(old_view, metadata.read, guard)
    catalog_rows(old_serial, resolved["current_scope"]["later_serial"],
                 [(spec["model_exit"], ended["owner"], ended["actual_tool"]), (spec["recorder_exit"], recorder["owner"], recorder["actual_tool"])])
    view = dict(old_view, serial_predecessors=spec["current_serial_catalog"])
    serial = r.completion.latest.serial_catalog(view, metadata.read, guard)
    rows = catalog_rows(serial, old_serial, [(spec["auditor_exit"], auditor["owner"], auditor["actual_tool"])])
    # Existing successful auditor witness revalidates its actual terminal,
    # command/spec/review and complete inherited source map, recursively doing
    # the same for recorder_v3. The seeded cache avoids a second reconstruction.
    witnessed = preparation.completed_inherited_extension(auditor, rows[spec["auditor_exit"]["path"]], metadata.read, guard, cache)
    require(all(pins_plan["source_sha256"].get(path) == sha for path, sha in witnessed.items()), "Complete actual auditor source witness retained")
    freshness = r.completion.latest.require_latest_terminal(view, metadata.read, guard)
    observed = r.completion.latest.latest_handoff()
    require(observed["path"] == freshness["latest_handoff"]["path"], "Actual latest complete idle terminal bound before comparison")
    lease.execution_observation = copy.deepcopy(observed)
    scope = serial_scope(resolved, serial, checked, recorder); lease.scope = scope
    authority = r.current_authority.Authority(r, lease)
    original = records["drains"]; original_pair = r.completion.audit.unpack(original["paired_physical"])
    require(original["schema"] == "minicpm5_native_original_predecessor_pair_v1"
            and original["resolved_plan_sha256"] == resolved["plan_sha256"] and original["outer_owner"] == ended["owner"]
            and original["lock_identity"] == r.ORIGINAL_LOCK and original["physical_pair_schema"] == r.completion.physical.PAIR_SCHEMA
            and r.digest(original_pair) == original["complete_observation_sha256"], "Original complete saved predecessor physical binding")
    original_scope = resolved["current_scope"]
    r.completion.physical.validate_pair(original_pair, {row["unit"] for row in original_scope["nemotron"]["history"]["rows"]},
        original_scope["owners"], r.ORIGINAL_LOCK, known_units=original_scope["complete_known_units"])
    old_scope = serial_scope(resolved, old_serial, checked, recorder)
    old_pair = r.completion.audit.unpack(audit_result["current_physical_pair"])
    required = {row["unit"] for row in old_scope["nemotron"]["history"]["rows"]}
    known = set(old_scope["complete_known_units"])
    require(r.completion.physical.validate_pair(old_pair, required, old_scope["owners"], r.ORIGINAL_LOCK, known_units=known) == known
            and audit_result["physical_pair_schema"] == r.completion.physical.PAIR_SCHEMA
            and audit_result["current_unit_count"] == len(known) and audit_result["current_owner_count"] == len(old_scope["owners"]),
            "Saved complete independent audit physical evidence decoded in its original scope")
    owned = audited.outer.OwnedService(contract, ended["outputs"]["admission"], a, metadata)
    owned.attempted = True; owned.owner = records["coordinator_owner"]
    model_drain = owned.verify_drained()
    require(model_drain == audit_result["model_drain"], "Actual exact native service remains drained since independent audit")
    before_pair = audited.pair(scope, guard, authority)
    rank_refs = [recovered["recovery"]["original_model_outputs"]["rank%d_result" % rank] for rank in (0, 1)]
    controls = audited_controls(audit_result, spec["model_exit"], spec["recorder_exit"], rank_refs, resolved["plan"]["native_operator_manifest"])
    sdk_results = [metadata.read(ref) for ref in rank_refs]
    sdk_controls = check_sdk_controls(sdk_results, recovered["base_plan"], guard)
    native_io = audited.audit_io.NativeAuditIO(resolved["plan"], resolved["native_bundle"], guard)
    native_io.final_inventory(); inventory = native_io.unchanged()
    require(inventory == audit_result["inventory"], "Complete native tree remains the independently audited inventory")
    full_rows = {}
    for row in checked["driver"]["records"]:
        ref = row["arms"][2]["result"]
        full_rows[row["frame"]["index"]] = metadata.read({key: ref[key] for key in ("path", "sha256")})
    native_trace = audited.verifier.expected_trace(resolved["plan"], lambda path, sha: r.sources.text(path, sha), guard)
    sdk_trace, sdk_trace_proof = authenticated_sdk_trace(recovered, sdk_results, rank_refs, metadata.read,
                                                       lambda path, sha: r.sources.text(path, sha), guard)
    prereqs = dict(native_model_exit=spec["model_exit"], native_recorder_terminal=spec["recorder_exit"],
        native_audit_result=auditor["result"], native_auditor_terminal=spec["auditor_exit"],
        native_plan=model_spec["overlay"], native_driver=ended["outputs"]["native_result"],
        sdk_recovery_actual_exit=r.completion.BINDING["actual_exit"], sdk_recovery_recorder_terminal=r.completion.BINDING["recorder_terminal"],
        sdk_plan=recovered["base_plan_ref"], sdk_rank0_result=rank_refs[0], sdk_rank1_result=rank_refs[1], source_review=review_ref)
    plan = reader.prepare(resolved["plan"], recovered["base_plan"], checked["driver"], full_rows, sdk_results, native_trace, sdk_trace, prereqs)
    plan_sha = reader.comparison.digest(plan)
    allowed = reader.validate_plan(plan); io = reader.StableCaptureIO(allowed, guard, filesystem=os)
    reports = []
    def emit(report):
        call = report["call"]; entry = plan["calls"][reader.comparison.CALLS.index(call)]
        output = dict(schema=REPORT_SCHEMA, comparison_spec=spec_ref, comparison_review=review_ref,
            descriptor_plan_sha256=plan_sha, call=call, complete_call_capture_descriptors=entry, comparison=report,
            actual_comparison_tool_exit_verified=False, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)
        ref = write_report(paths[str(call)], output, guard); metadata.read(ref)
        reports.append(dict(call=call, report=ref, descriptor_call_sha256=r.digest(entry), comparison_sha256=r.digest(report)))
    summary = reader.compare_all(plan, io.read_bytes, guard, emit, expected_plan_sha256=plan_sha)
    require(len(io._read) == 1284 and io._read == set(allowed), "Only and every1284 original capture file read exactly once")
    require(native_io.unchanged() == inventory and check_sdk_controls(sdk_results, recovered["base_plan"], guard) == sdk_controls,
            "All original native tree and SDK60 control identities unchanged across comparison")
    guard(force=True); after_pair = audited.pair(scope, guard, authority)
    require(after_pair == before_pair and owned.verify_drained() == model_drain, "Complete original physical scope unchanged across all raw reads")
    require(r.completion.latest.require_latest_terminal(view, metadata.read, guard) == freshness,
            "Full current catalog freshness unchanged after all comparisons")
    r.sources.verify_text_sources(partition, guard); r.sources.check_native_identities(records["source_authentication"], model_partition, guard)
    r.verify_loaded(pins_plan, guard); metadata.unchanged(); guard(force=True)
    result = dict(schema=SUMMARY_SCHEMA, complete=True, reviewer="/root", comparison_spec=spec_ref, comparison_review=review_ref,
        owner=lease.admission["outer_owner"], lock_identity=r.ORIGINAL_LOCK, original_lock_policy=r.LOCK_POLICY,
        source=dict(path=SELF, sha256=pins_plan["source_sha256"][SELF]), complete_source_binding=source_proof,
        current_serial_catalog=spec["current_serial_catalog"], serial_freshness=freshness,
        complete_physical_pair=r.completion.audit.packed_physical(before_pair), physical_pair_schema=r.completion.physical.PAIR_SCHEMA,
        original_model_controls=controls, native_inventory=inventory, sdk_control_identity_check=sdk_controls,
        original_sdk_model_exit_code=1, original_sdk_model_runtime_successful=False,
        native_trace_reconstructed_in_original_interpreter=native_trace, authenticated_original_sdk_trace=sdk_trace_proof,
        descriptor_plan_sha256=plan_sha, descriptor_plan_header={key: value for key, value in plan.items() if key != "calls"},
        native_plan_reference_kind="original_overlay_reconstructs_complete_native_plan",
        native_driver_reference_kind="original_native_result_wrapper_embeds_driver",
        reports=reports, comparison=summary, capture_files_read=1284, capture_bytes_read=77790564,
        source_control_and_terminal_authentication_complete=True, actual_comparison_tool_exit_verified=False,
        external_comparison_audit_pending=True, self_exit_claimed=False, model_reexecuted=False,
        causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False, storage=guard.storage)
    return result, guard


def run(spec_ref, review_ref):
    require(sys.dont_write_bytecode and __debug__, "Original -B interpreter and enabled assertions")
    require(Path(sys.executable).resolve() == Path(r.life.PYTHON).resolve(),
            "Reconstruct native line/code contract only in the original native interpreter")
    r.p.user_bus(); fd, owner, first = audited.support.acquire()
    lease = r.Lease(dict(outer_owner=owner, lock_identity=r.ORIGINAL_LOCK, retained_fd=fd, lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first, unit="trn1-mini-layer1-saved-operator-comparison"))
    try:
        os.sched_setaffinity(0, {LIMITS["cpu"]}); os.nice(LIMITS["nice"])
        resource.setrlimit(resource.RLIMIT_AS, (LIMITS["address_space_bytes"],) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_seconds"], LIMITS["hard_cpu_seconds"]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS["output_file_bytes"],) * 2)
        signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError("Original saved comparison wall bound")))
        signal.alarm(LIMITS["wall_seconds"])
        result, guard = compare(spec_ref, review_ref, lease)
        spec = r.sources.load(spec_ref); path = output_paths(spec["output_prefix"])["summary"]
        # All six reports already have authenticated references; only the
        # root's later external terminal may claim this process actually exited.
        guard(force=True)
        ref = write_report(path, result, guard); print(json.dumps(ref), flush=True); return ref
    finally:
        signal.alarm(0)
        audited.support.release_after_physical_drain(fd, dict(physically_drained=True, no_comparison_children_or_services_started=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("spec", "review"):
        parser.add_argument("--" + name, required=True); parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    run(dict(path=args.spec, sha256=args.spec_sha256), dict(path=args.review, sha256=args.review_sha256))
