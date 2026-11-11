"""Independent post-exit all-capture replay of saved comparison reports.

Source-only until run() is explicitly admitted. Reuses the original retained
lock, metadata/source/physical machinery and immutable comparator/reader. No
model execution, successor launch, own-exit assertion or scope promotion.
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
import sys

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "research"
sys.path[:0] = [str(Q), str(ROOT)]
from research import compare_minicpm5_layer1_saved_operators_v1 as previous

r = previous.r
require = r.require
reader = previous.reader
core = previous.core
audited = previous.audited
LIMITS = copy.deepcopy(previous.LIMITS)
SELF = str(Path(__file__).resolve())
EXIT_SCHEMA = "minicpm5_saved_operator_comparison_actual_exit_v1"
RESULT_SCHEMA = "minicpm5_saved_operator_post_exit_independent_audit_v1"
# Exact text-only additions from the completed LFM namespace diagnostic source
# map. No general SDK/venv exemption; full source classification, content SHA,
# original contract and current serial catalog checks remain mandatory.
COMPLETED_LFM_TEXT = {
    "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/__init__.py": "3caf7f40140ede2465bde40b9003af10cbbc8f7bcf436fa1de026471daa1b288",
    "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/_ops.py": "3b58079d356d15cc73082c20b1279468ee9cfe7f5e1340d329a93b6e877a30d5",
    "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/torch/_classes.py": "0db3dc302e28fd085065ea2883ce33fa4af406b9fd1439f2a431f053e9a764b9",
    "/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages/nki/framework/_torch_convert.py": "2c510075ca5943ce655f3edd21553398ae6d83d05194ddaca21d8da3ff8ea729",
}


def validate_spec(spec_ref, review_ref, spec, review):
    for ref in (spec_ref, review_ref): r.life.ref(ref)
    require(type(spec) is dict and set(spec) == {"schema", "comparison_exit", "current_serial_catalog", "source_extension", "output", "limits"}
            and spec["schema"] == "minicpm5_saved_operator_post_exit_audit_spec_v1" and spec["limits"] == LIMITS,
            "Exact independent comparison audit spec and unchanged original limits")
    for name in ("comparison_exit", "current_serial_catalog"): r.life.ref(spec[name])
    require(type(review) is dict and set(review) == {"schema", "reviewer", "passed", "blockers", "spec", "source_extension"}
            and review["schema"] == "minicpm5_saved_operator_post_exit_audit_source_review_v1"
            and review["reviewer"] == "/root" and review["passed"] is True and review["blockers"] == []
            and review["spec"] == spec_ref and review["source_extension"] == spec["source_extension"], "Exact independent post-exit source review")
    pins = spec["source_extension"]
    require(type(pins) is dict and 0 < len(pins) <= 2048, "Bounded only-new audit source map")
    for path, sha in pins.items():
        require(type(path) is str and Path(path).is_absolute() and str(Path(path)) == path and ".." not in Path(path).parts
                and (path.startswith(str(ROOT) + "/") or COMPLETED_LFM_TEXT.get(path) == sha)
                and Path(path).suffix in (".py", ".json", ".md", ".txt", ".sh")
                and type(sha) is str and re.fullmatch("[a-f0-9]{64}", sha), "Exact additive local or completed LFM text source pin")
    path = Path(spec["output"])
    require(str(path) == spec["output"] and path.parent == Q
            and re.fullmatch(r"minicpm5-layer1-saved-operator-comparison-[a-z0-9-]+-post-exit-audit-[a-z0-9-]+\.json", path.name),
            "Separate immutable post-exit audit report namespace")
    return path


def comparison_terminal(ended, live, summary, spec, native_executable):
    """Exact14-field root external terminal; six partial calls never suffice."""
    require(type(ended) is dict and set(ended) == {"schema", "reviewer", "spec", "review", "live", "result", "owner", "actual_tool",
            "actual_tool_exit_observed", "exact_owner_gone", "lock_identity", "self_exit_claimed", "full_model_qualified", "new_speedup_claim"}
            and ended["schema"] == EXIT_SCHEMA and ended["reviewer"] == "/root"
            and ended["actual_tool_exit_observed"] is ended["exact_owner_gone"] is True and ended["lock_identity"] == r.ORIGINAL_LOCK
            and ended["self_exit_claimed"] is ended["full_model_qualified"] is ended["new_speedup_claim"] is False,
            "Genuine independently observed comparison success terminal, no absent/failed/self exit")
    tool = core.actual_tool(ended["actual_tool"])
    for name in ("spec", "review", "live", "result"): r.life.ref(ended[name])
    argv = ["python3", "-B", previous.SELF, "--spec", ended["spec"]["path"], "--spec-sha256", ended["spec"]["sha256"],
            "--review", ended["review"]["path"], "--review-sha256", ended["review"]["sha256"]]
    r.life.process(ended["owner"], argv)
    require(live["schema"] == "root_minicpm5_saved_operator_comparison_live_v1" and live["reviewer"] == "/root"
            and live["owner"] == ended["owner"] and live["spec"] == ended["spec"] and live["review"] == ended["review"]
            and live["actual_tool"] == dict(kind="tool_session", session_id=tool["session_id"])
            and live["actual_owner_observed"] is True and live["actual_tool_exit_observed"] is False
            and live["actual_executable"] == native_executable and live["full_model_qualified"] is live["new_speedup_claim"] is False,
            "Exact externally observed original live/owner/argv/native-interpreter/session chain")
    require(summary["schema"] == previous.SUMMARY_SCHEMA and summary["complete"] is True
            and summary["owner"] == ended["owner"] and summary["comparison_spec"] == ended["spec"] and summary["comparison_review"] == ended["review"]
            and ended["result"]["path"] == str(previous.output_paths(spec["output_prefix"])["summary"]),
            "Whole original summary bound to the true terminal and exact approved output")
    return tool


def historical_freshness(value, serial, read):
    """Authenticate old idle publication chain without pretending it is current."""
    r.execution.freshness_shape(value, serial["reference"])
    require(value["bound_terminal"] == serial["current_handoff"], "Original comparison freshness binds its actual full catalog")
    refs = [*value["publication_only_successors"], value["bound_terminal"]]
    chain = [(ref, read(ref)) for ref in refs]
    r.execution.validate_predecessor_chain(value, serial["reference"], chain[-1][1], chain)
    return copy.deepcopy(value)


def saved_storage(value, current_usage):
    require(type(value) is dict and set(value) == {"available_bytes", "usage", "maximum_check_interval_seconds"}
            and type(value["available_bytes"]) is dict and set(value["available_bytes"]) == {"root", "output"}
            and all(type(n) is int for n in value["available_bytes"].values())
            and value["available_bytes"]["root"] >= 28 * 2**30 + previous.OUTPUT_RESERVE
            and value["available_bytes"]["output"] >= 30 * 2**30
            and type(value["maximum_check_interval_seconds"]) is int and value["maximum_check_interval_seconds"] == 1
            and value["usage"] == current_usage and r.digest(value["usage"]) == r.digest(current_usage),
            "Historical measured storage bounds and unchanged complete native-tree usage")
    # Historical free-space values cannot be remeasured after the fact. Retain
    # the authenticated originals after checking their bound, never substitute.
    return copy.deepcopy(value)


def expected_call(spec_ref, review_ref, plan, report):
    call = report["call"]; entry = plan["calls"][reader.comparison.CALLS.index(call)]
    return dict(schema=previous.REPORT_SCHEMA, comparison_spec=spec_ref, comparison_review=review_ref,
        descriptor_plan_sha256=r.digest(plan), call=call, complete_call_capture_descriptors=entry, comparison=report,
        actual_comparison_tool_exit_verified=False, self_exit_claimed=False, full_model_qualified=False, new_speedup_claim=False)


def report_references(comparison_spec, summary):
    paths = previous.output_paths(comparison_spec["output_prefix"])
    require(type(summary["reports"]) is list and len(summary["reports"]) == 6
            and [row["call"] for row in summary["reports"]] == list(reader.comparison.CALLS), "Exactly six original ordered report references")
    refs = []
    for row in summary["reports"]:
        require(type(row) is dict and set(row) == {"call", "report", "descriptor_call_sha256", "comparison_sha256"}, "Exact complete report-reference record")
        r.life.ref(row["report"])
        require(row["report"]["path"] == str(paths[str(row["call"])])
                and all(type(row[key]) is str and re.fullmatch("[a-f0-9]{64}", row[key]) for key in ("descriptor_call_sha256", "comparison_sha256")),
                "Only original approved bounded report paths and complete digests")
        refs.append(row["report"])
    return refs


def verify_reports(ended, comparison_spec, plan, summary, report_values, read_bytes, guard):
    """Independent full reread and exact dictionary comparison, no writer."""
    require(type(report_values) is list and len(report_values) == 6
            and [row["call"] for row in summary["reports"]] == list(reader.comparison.CALLS), "All six complete original reports in order")
    paths = previous.output_paths(comparison_spec["output_prefix"]); verified = []; count = 0
    def sink(report):
        nonlocal count
        call = reader.comparison.CALLS[count]; original_ref = summary["reports"][count]
        require(report["call"] == call and original_ref["report"]["path"] == str(paths[str(call)]), "Exact approved report path/call")
        wanted = expected_call(ended["spec"], ended["review"], plan, report)
        require(report_values[count] == wanted and r.digest(report_values[count]) == r.digest(wanted),
                "Every saved report field and JSON type equals independent all-byte recomputation")
        entry = plan["calls"][count]
        require(original_ref == dict(call=call, report=original_ref["report"], descriptor_call_sha256=r.digest(entry), comparison_sha256=r.digest(report)),
                "Complete original descriptor/comparison digests reconstructed independently")
        verified.append(copy.deepcopy(original_ref)); count += 1
    recomputed = reader.compare_all(plan, read_bytes, guard, sink, expected_plan_sha256=r.digest(plan))
    require(count == 6 and summary["comparison"] == recomputed and r.digest(summary["comparison"]) == r.digest(recomputed)
            and summary["descriptor_plan_sha256"] == r.digest(plan)
            and summary["descriptor_plan_header"] == {key: value for key, value in plan.items() if key != "calls"},
            "Entire original comparison summary and complete reconstructed descriptor plan exact")
    return recomputed, verified


def reconstruct(ended, summary, spec, metadata, guard, lease):
    """Read-only original lineage reconstruction, reusing existing validators.

    No call to previous.compare (which writes reports), no copied lifecycle or
    monitor implementation, and no native/donor payload reads in this phase.
    """
    read = metadata.read
    original_spec = read(ended["spec"]); original_review = read(ended["review"])
    previous.validate_spec(ended["spec"], ended["review"], original_spec, original_review)
    live = read(ended["live"])
    comparison_terminal(ended, live, summary, original_spec, str(Path(r.life.PYTHON).resolve()))
    model = read(original_spec["model_exit"]); recorder = read(original_spec["recorder_exit"]); native_auditor = read(original_spec["auditor_exit"])
    mt = core.actual_tool(model["actual_tool"]); rt = core.actual_tool(recorder["actual_tool"], allow_call=True)
    previous.actual_auditor_terminal(native_auditor, mt, rt)
    for owner in (ended["owner"], model["owner"], model["coordinator"], model["native"], recorder["owner"], native_auditor["owner"]): r.p.gone(owner)
    recovered = r.completion.completed(r.completion.BINDING, read, guard, lambda path, sha: r.sources.text(path, sha))
    previous.preparation.preserved(recovered)
    model_spec = read(model["spec"]); model_review = read(model["review"]); overlay = read(model_spec["overlay"])
    resolved = r.overlays.resolve(overlay, recovered, read, guard)
    cache = {r.digest([model["spec"], model["review"], model_spec["overlay"]]):
             dict(resolved=resolved, overlay=copy.deepcopy(overlay), spec=copy.deepcopy(model_spec), review=copy.deepcopy(model_review))}
    checked_resolved, checked_spec = previous.preparation.inherited_source_plan(model, read, guard, cache)
    require(checked_resolved == resolved and checked_spec == model_spec, "Entire original model plan/source reconstruction")
    records = {name: read(ref) for name, ref in model["outputs"].items()}
    checked = core.validate(original_spec["model_exit"], model, original_spec["recorder_exit"], recorder, read(model["live"]), records, resolved)
    a = records["admission"]; contract = checked["contract"]; guard.bind(contract, resolved)
    r.validate_initial_descriptor(a); r.execution.validate_intent(records["intent"], a)
    require(a["overlay"] == model_spec["overlay"] and a["source_contract"] == resolved["plan"]["source_contract"]
            and a["sdk_operator_recovery"] == r.completion.BINDING and a["latest_completion"] == resolved["plan"]["latest_completion"]
            and a["completed_sdk_layout"] == resolved["plan"]["layer1_operator_boundary"]
            and a["serial_predecessors"] == resolved["plan"]["serial_predecessors"], "Complete unchanged model admission lineage")
    audit_spec = read(native_auditor["audit_spec"]); audit_review = read(native_auditor["audit_review"]); native_result = read(native_auditor["result"])
    audited.validate_spec(native_auditor["audit_spec"], native_auditor["audit_review"], audit_spec, audit_review)
    require(audit_spec["model_exit"] == original_spec["model_exit"] and audit_spec["recorder_exit"] == original_spec["recorder_exit"],
            "Exact completed native auditor prerequisite")
    comparison_plan, comparison_source = previous.source_plan(resolved, model, model_spec, audit_spec, native_result, original_spec)
    require(summary["complete_source_binding"] == comparison_source
            and summary["source"] == dict(path=previous.SELF, sha256=comparison_plan["source_sha256"][previous.SELF]),
            "Original complete comparison source map reconstructed, not trusted from its digest")
    pins_plan = audited.merged_plan(comparison_plan, spec["source_extension"])
    require(SELF in pins_plan["source_sha256"], "Actual independent audit entry pinned")
    model_partition = r.sources.classify(resolved["plan"]["source_sha256"], resolved["plan"]["source_contract"], read)
    inherited = r.sources.validate_native_receipt(records["source_authentication"], model_partition, model["owner"], r.ORIGINAL_LOCK, a["attempt_id"])
    require(native_result["inherited_native_source_authentication"] == inherited, "Original complete native source authentication")
    partition = r.sources.classify(pins_plan["source_sha256"], pins_plan["source_contract"], read)
    r.sources.verify_text_sources(partition, guard); r.sources.check_native_identities(records["source_authentication"], model_partition, guard)
    r.verify_loaded(pins_plan, guard)
    view = dict(latest_completion=resolved["plan"]["latest_completion"], source_sha256=pins_plan["source_sha256"])
    native_serial = r.completion.latest.serial_catalog(dict(view, serial_predecessors=audit_spec["current_serial_catalog"]), read, guard)
    previous.catalog_rows(native_serial, resolved["current_scope"]["later_serial"],
        [(original_spec["model_exit"], model["owner"], mt), (original_spec["recorder_exit"], recorder["owner"], rt)])
    original_serial = r.completion.latest.serial_catalog(dict(view, serial_predecessors=original_spec["current_serial_catalog"]), read, guard)
    rows = previous.catalog_rows(original_serial, native_serial, [(original_spec["auditor_exit"], native_auditor["owner"], native_auditor["actual_tool"])])
    witnesses = previous.preparation.completed_inherited_extension(native_auditor, rows[original_spec["auditor_exit"]["path"]], read, guard, cache)
    require(all(pins_plan["source_sha256"].get(path) == sha for path, sha in witnesses.items()), "All completed native auditor/recorder source witnesses")
    historical_freshness(summary["serial_freshness"], original_serial, read)
    current_view = dict(view, serial_predecessors=spec["current_serial_catalog"])
    current = r.completion.latest.serial_catalog(current_view, read, guard)
    previous.catalog_rows(current, original_serial, [(spec["comparison_exit"], ended["owner"], ended["actual_tool"])])
    freshness = r.completion.latest.require_latest_terminal(current_view, read, guard)
    observation = r.completion.latest.latest_handoff()
    require(observation["path"] == freshness["latest_handoff"]["path"], "Actual latest complete idle handoff bound for independent audit")
    lease.execution_observation = copy.deepcopy(observation)
    scope = previous.serial_scope(resolved, current, checked, recorder); lease.scope = scope
    original_scope = previous.serial_scope(resolved, original_serial, checked, recorder)
    native_scope = previous.serial_scope(resolved, native_serial, checked, recorder)
    original = records["drains"]; predecessor_pair = r.completion.audit.unpack(original["paired_physical"])
    require(original["schema"] == "minicpm5_native_original_predecessor_pair_v1" and original["resolved_plan_sha256"] == resolved["plan_sha256"]
            and original["outer_owner"] == model["owner"] and original["lock_identity"] == r.ORIGINAL_LOCK
            and original["physical_pair_schema"] == r.completion.physical.PAIR_SCHEMA
            and r.digest(predecessor_pair) == original["complete_observation_sha256"], "Original full predecessor pair identity")
    for observed_pair, old_scope in ((predecessor_pair, resolved["current_scope"]),
            (r.completion.audit.unpack(native_result["current_physical_pair"]), native_scope),
            (r.completion.audit.unpack(summary["complete_physical_pair"]), original_scope)):
        known = set(old_scope["complete_known_units"]); required = {row["unit"] for row in old_scope["nemotron"]["history"]["rows"]}
        require(r.completion.physical.validate_pair(observed_pair, required, old_scope["owners"], r.ORIGINAL_LOCK, known_units=known) == known,
                "Every whole saved physical tree authenticated in its original scope")
    require(native_result["current_owner_count"] == len(native_scope["owners"])
            and native_result["current_unit_count"] == len(native_scope["complete_known_units"])
            and native_result["physical_pair_schema"] == summary["physical_pair_schema"] == r.completion.physical.PAIR_SCHEMA,
            "Original full physical counts/schema retained after complete tree validation")
    owned = audited.outer.OwnedService(contract, model["outputs"]["admission"], a, metadata)
    owned.attempted = True; owned.owner = records["coordinator_owner"]
    model_drain = owned.verify_drained(); require(model_drain == native_result["model_drain"], "Exact native model remains physically drained")
    rank_refs = [recovered["recovery"]["original_model_outputs"]["rank%d_result" % rank] for rank in (0, 1)]
    controls = previous.audited_controls(native_result, original_spec["model_exit"], original_spec["recorder_exit"], rank_refs, resolved["plan"]["native_operator_manifest"])
    sdk_results = [read(ref) for ref in rank_refs]
    sdk_controls = previous.check_sdk_controls(sdk_results, recovered["base_plan"], guard)
    native_io = audited.audit_io.NativeAuditIO(resolved["plan"], resolved["native_bundle"], guard)
    native_io.final_inventory(); inventory = native_io.unchanged()
    require(inventory == native_result["inventory"] == summary["native_inventory"], "Original complete audited native inventory unchanged")
    full_rows = {row["frame"]["index"]: read({key: row["arms"][2]["result"][key] for key in ("path", "sha256")}) for row in checked["driver"]["records"]}
    native_trace = audited.verifier.expected_trace(resolved["plan"], lambda path, sha: r.sources.text(path, sha), guard)
    sdk_trace, sdk_trace_proof = previous.authenticated_sdk_trace(recovered, sdk_results, rank_refs, read, lambda path, sha: r.sources.text(path, sha), guard)
    refs = dict(native_model_exit=original_spec["model_exit"], native_recorder_terminal=original_spec["recorder_exit"],
        native_audit_result=native_auditor["result"], native_auditor_terminal=original_spec["auditor_exit"],
        native_plan=model_spec["overlay"], native_driver=model["outputs"]["native_result"], sdk_recovery_actual_exit=r.completion.BINDING["actual_exit"],
        sdk_recovery_recorder_terminal=r.completion.BINDING["recorder_terminal"], sdk_plan=recovered["base_plan_ref"],
        sdk_rank0_result=rank_refs[0], sdk_rank1_result=rank_refs[1], source_review=ended["review"])
    plan = reader.prepare(resolved["plan"], recovered["base_plan"], checked["driver"], full_rows, sdk_results, native_trace, sdk_trace, refs)
    return dict(original_spec=original_spec, resolved=resolved, records=records, model_partition=model_partition,
        pins_plan=pins_plan, partition=partition, comparison_source=comparison_source, scope=scope, current_view=current_view, freshness=freshness,
        owned=owned, model_drain=model_drain, recovered=recovered, sdk_results=sdk_results, sdk_controls=sdk_controls,
        native_io=native_io, inventory=inventory, controls=controls, native_trace=native_trace, sdk_trace_proof=sdk_trace_proof, plan=plan)


def expected_summary(ended, original, context, recomputed, reports, storage):
    """Reconstruct every stable field; historical telemetry is checked separately."""
    plan = context["plan"]; spec = context["original_spec"]
    return dict(schema=previous.SUMMARY_SCHEMA, complete=True, reviewer="/root", comparison_spec=ended["spec"], comparison_review=ended["review"],
        owner=ended["owner"], lock_identity=r.ORIGINAL_LOCK, original_lock_policy=r.LOCK_POLICY,
        source=dict(path=previous.SELF, sha256=context["pins_plan"]["source_sha256"][previous.SELF]), complete_source_binding=context["comparison_source"],
        current_serial_catalog=spec["current_serial_catalog"], serial_freshness=original["serial_freshness"],
        complete_physical_pair=original["complete_physical_pair"], physical_pair_schema=r.completion.physical.PAIR_SCHEMA,
        original_model_controls=context["controls"], native_inventory=context["inventory"], sdk_control_identity_check=context["sdk_controls"],
        original_sdk_model_exit_code=1, original_sdk_model_runtime_successful=False,
        native_trace_reconstructed_in_original_interpreter=context["native_trace"], authenticated_original_sdk_trace=context["sdk_trace_proof"],
        descriptor_plan_sha256=r.digest(plan), descriptor_plan_header={key: value for key, value in plan.items() if key != "calls"},
        native_plan_reference_kind="original_overlay_reconstructs_complete_native_plan", native_driver_reference_kind="original_native_result_wrapper_embeds_driver",
        reports=reports, comparison=recomputed, capture_files_read=1284, capture_bytes_read=77790564,
        source_control_and_terminal_authentication_complete=True, actual_comparison_tool_exit_verified=False,
        external_comparison_audit_pending=True, self_exit_claimed=False, model_reexecuted=False,
        causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False, storage=storage)


def verify_summary(ended, original, context, recomputed, reports, storage):
    wanted = expected_summary(ended, original, context, recomputed, reports, storage)
    require(original == wanted and r.digest(original) == r.digest(wanted),
            "Every complete saved summary field/type independently reconstructed/validated")
    return wanted


def audit(spec_ref, review_ref, lease):
    guard = previous.ComparisonGuard(lease); guard(force=True)
    metadata = r.Metadata(guard); lease.metadata = metadata
    spec = metadata.read(spec_ref); review = metadata.read(review_ref); target = validate_spec(spec_ref, review_ref, spec, review)
    require(not target.exists(), "Independent report must be a fresh exclusive output")
    for path, sha in spec["source_extension"].items():
        guard(); require(hashlib.sha256(r.sources.text(path, sha)).hexdigest() == sha, "Reviewed independent audit sources")
    ended = metadata.read(spec["comparison_exit"])
    # Reject nonzero/bool/missing terminal before loading any payload descriptor.
    core.actual_tool(ended["actual_tool"]); require(ended["schema"] == EXIT_SCHEMA and ended["actual_tool_exit_observed"] is True
        and ended["exact_owner_gone"] is True, "Actual completed comparison required, not partial report presence")
    r.p.gone(ended["owner"]); summary = metadata.read(ended["result"])
    require(metadata.identities[ended["result"]["path"]]["size"] <= LIMITS["output_file_bytes"], "Original complete summary remains within2MiB")
    context = reconstruct(ended, summary, spec, metadata, guard, lease)
    authority = r.current_authority.Authority(r, lease); before_pair = audited.pair(context["scope"], guard, authority)
    report_values = []
    for ref in report_references(context["original_spec"], summary):
        report_values.append(metadata.read(ref))
        require(metadata.identities[ref["path"]]["size"] <= LIMITS["output_file_bytes"], "Every complete original report remains within2MiB")
    allowed = reader.validate_plan(context["plan"]); io = reader.StableCaptureIO(allowed, guard, filesystem=os)
    recomputed, reports = verify_reports(ended, context["original_spec"], context["plan"], summary, report_values, io.read_bytes, guard)
    require(io._read == set(allowed) and len(io._read) == 1284, "Every complete capture independently read once in this post-exit process")
    require(context["native_io"].unchanged() == context["inventory"]
            and previous.check_sdk_controls(context["sdk_results"], context["recovered"]["base_plan"], guard) == context["sdk_controls"],
            "Original full native inventory and SDK control identities unchanged")
    guard(force=True); storage = saved_storage(summary["storage"], guard.storage["usage"])
    verify_summary(ended, summary, context, recomputed, reports, storage)
    after_pair = audited.pair(context["scope"], guard, authority)
    require(after_pair == before_pair and context["owned"].verify_drained() == context["model_drain"], "Complete physical scope unchanged across independent reread")
    require(r.completion.latest.require_latest_terminal(context["current_view"], metadata.read, guard) == context["freshness"], "Full current catalog freshness unchanged")
    r.sources.verify_text_sources(context["partition"], guard)
    r.sources.check_native_identities(context["records"]["source_authentication"], context["model_partition"], guard)
    r.verify_loaded(context["pins_plan"], guard); metadata.unchanged(); guard(force=True)
    result = dict(schema=RESULT_SCHEMA, complete=True, reviewer="/root", audit_spec=spec_ref, audit_review=review_ref,
        comparison_exit=spec["comparison_exit"], original_summary=ended["result"], original_reports=reports,
        audit_owner=lease.admission["outer_owner"], lock_identity=r.ORIGINAL_LOCK,
        source=dict(path=SELF, sha256=context["pins_plan"]["source_sha256"][SELF]),
        complete_source_binding=dict(original_comparison_source=context["comparison_source"],
            additional_sources_sha256=r.digest(spec["source_extension"]), additional_source_files=len(spec["source_extension"]),
            complete_source_map_sha256=r.digest(context["pins_plan"]["source_sha256"]), complete_source_files=len(context["pins_plan"]["source_sha256"]),
            complete_source_reconstruction_required=True, inherited_sources_replaced=False, inherited_sources_omitted=False),
        current_serial_catalog=spec["current_serial_catalog"], serial_freshness=context["freshness"],
        complete_current_physical_pair=r.completion.audit.packed_physical(before_pair), physical_pair_schema=r.completion.physical.PAIR_SCHEMA,
        capture_files_independently_reread=1284, capture_bytes_independently_reread=77790564, frames=24, calls=6, ranks=2,
        all_six_complete_reports_raw_recomputed_exact=True, complete_summary_reconstructed_exact=True,
        source_exit_and_full_control_lineage_verified=True, historical_storage_telemetry_bound_validated_not_remeasured=True,
        descriptor_plan_sha256=r.digest(context["plan"]), recomputed_scientific_summary_sha256=r.digest(recomputed),
        actual_comparison_tool_exit_verified=True, actual_auditor_tool_exit_verified=False, self_exit_claimed=False,
        model_reexecuted=False, causal_operator_proven=False, full_model_qualified=False, new_speedup_claim=False, storage=guard.storage)
    return result, guard


def run(spec_ref, review_ref):
    require(sys.dont_write_bytecode and __debug__ and Path(sys.executable).resolve() == Path(r.life.PYTHON).resolve(),
            "Original native interpreter, -B and assertions for exact native trace reconstruction")
    r.p.user_bus(); fd, owner, first = audited.support.acquire()
    lease = r.Lease(dict(outer_owner=owner, lock_identity=r.ORIGINAL_LOCK, retained_fd=fd, lock_policy=r.LOCK_POLICY,
        initial_descriptor_proof=first, unit="trn1-mini-layer1-saved-operator-independent-auditor"))
    try:
        os.sched_setaffinity(0, {LIMITS["cpu"]}); os.nice(LIMITS["nice"])
        resource.setrlimit(resource.RLIMIT_AS, (LIMITS["address_space_bytes"],) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_seconds"], LIMITS["hard_cpu_seconds"]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (LIMITS["output_file_bytes"],) * 2)
        signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError("Original independent comparison audit wall bound")))
        signal.alarm(LIMITS["wall_seconds"])
        result, guard = audit(spec_ref, review_ref, lease); spec = r.sources.load(spec_ref)
        ref = previous.write_report(Path(spec["output"]), result, guard); print(json.dumps(ref), flush=True); return ref
    finally:
        signal.alarm(0)
        audited.support.release_after_physical_drain(fd, dict(physically_drained=True, no_independent_audit_children_or_services_started=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("spec", "review"):
        parser.add_argument("--" + name, required=True); parser.add_argument("--" + name + "-sha256", required=True)
    args = parser.parse_args()
    run(dict(path=args.spec, sha256=args.spec_sha256), dict(path=args.review, sha256=args.review_sha256))
