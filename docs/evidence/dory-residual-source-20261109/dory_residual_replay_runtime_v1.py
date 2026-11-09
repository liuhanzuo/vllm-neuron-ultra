"""Prepared original complete Dory residual replay; no automatic entrypoint.

Runtime requires a separately finalized serial worker/spec after Nemotron r9
has actually terminated, been audited and physically released. Imports here
load no numerical framework. This diagnostic preserves the seven failures.
"""
import hashlib
import json
from pathlib import Path
import resource
import time

from dory_long_delta_writer_v1 import DeltaWriter
from dory_residual_event_stream_v1 import OriginalEventStream
from dory_residual_output_archive_v1 import ObservationArchive, original_archive_comparison
from dory_residual_request_banks_v1 import CACHE_IDS, Handle, ResidualRequestBanks
from dory_residual_replay_history_v1 import PHYSICAL_CALLS
from dory_residual_visit_capture_v1 import CaptureResidualVisits, require, selected_raw_bytes


class OriginalInjectedFailure(RuntimeError):
    pass


def run(env):
    torch, official, spec = (env[key] for key in ("torch", "official", "spec"))
    guard, raw, save, out = (env[key] for key in ("guard", "raw", "save", "out"))
    require(spec["schema"] == "dory_residual_complete_replay_diagnostic_v1"
            and spec["diagnostic_only"] is True and spec["runtime_admitted"] is True,
            "Separate finalized complete residual replay admission")
    require(env["control_active"][0] is False, "Only original official math may execute")
    guard()
    stream = OriginalEventStream(spec["selection_plan"], guard)
    budget = spec["budget"]
    require(type(spec["expected_original_raw_bytes"]) is int and spec["expected_original_raw_bytes"] > 0
            and budget["auxiliary_raw_bytes"] >= selected_raw_bytes(153) + 512*2**20
            and budget["nvme_new_bytes"] >= spec["expected_original_raw_bytes"]
                + budget["auxiliary_raw_bytes"] + 1024*2**20,
            "Full original raw data plus auxiliary failures/residuals and metadata reserved")
    require(all(cache.keys is None and cache.values is None for cache in env["caches"]),
            "No native backbone call before original complete replay")

    def describe(tensor):
        require(tensor.dtype in (torch.bfloat16, torch.float32),
                "Original supported observation dtype")
        return dict(shape=list(tensor.shape),
                    dtype="BF16" if tensor.dtype == torch.bfloat16 else "F32",
                    finite=bool(torch.isfinite(tensor).all()))

    def validate_cache(tensor, length):
        require(tensor.dtype == torch.bfloat16 and list(tensor.shape) == [4, length, 256],
                "Original complete BF16 per-loop cache shape")

    def select_copy(tensor, selectors):
        require(tensor.dtype == torch.bfloat16, "Original BF16 residual boundary")
        return tensor.index_select(0, selectors).detach().clone().contiguous()

    banks = ResidualRequestBanks(env["caches"], env["extract_layer_index"],
                                  validate_cache, lambda tensor: tensor.data_ptr())
    capture = CaptureResidualVisits(official, select_copy, guard)
    auxiliary = ObservationArchive(out, raw, describe, guard, budget["auxiliary_raw_bytes"])
    writer = DeltaWriter(out, torch, guard)
    counts = env["counts"]
    initial_ops = counts[0]
    initial_routes = sum(env["route_calls"].values())
    replay_files, selected_files, failure_files = [], [], []
    pending = None
    pending_outputs = {}
    successful_calls = rejections = injections = selected_rows = 0
    original_raw_bytes = residual_raw_bytes = 0
    started = time.monotonic()
    journal = (out/"residual-events.jsonl").open("x")

    def digest(tensor):
        guard()
        return hashlib.sha256(raw(tensor)).hexdigest()

    def inventory(exclude=None):
        return banks.inventory(digest, exclude)

    def event_record(action, record):
        guard()
        path = out/f'event-{action["event"]:06d}-replay.json'
        row = dict(event=action["event"], kind=action["kind"],
                   original_event=action["original_event"], slot=action["slot"],
                   epoch=action["epoch"], request=action["request"], **record)
        save(path, row)
        journal.write(json.dumps(dict(event=action["event"], file=path.name,
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()))+"\n")
        journal.flush()
        replay_files.append(path.name)
        return row

    def observed_tensors(values):
        return {f"layer-{physical}-loop-{loop}-{label}": tensor
                for (physical, loop, label), tensor in values.items()}

    def capture_failure(error, caches):
        # This callback runs inside the bank transaction, before invalidation.
        # It must not call bank.inventory(), which requires an idle transaction.
        tensors = observed_tensors(capture.partial_values)
        for key in CACHE_IDS:
            cache = caches[key]
            for kind, tensor in (("key", cache.keys), ("value", cache.values)):
                if tensor is not None:
                    tensors[f"cache-{key}-{kind}"] = tensor
        for label, tensor in pending_outputs.items():
            if tensor is not None:
                tensors[label] = tensor
        prefix = f'event-{pending["event"]:06d}-failure'
        path = out/(prefix+".json")
        failure = dict(error_type=type(error).__name__, error=str(error),
                       completed_visits=capture.completed, pending_visit=capture.pending_visit,
                       selected_boundaries=len(capture.partial_values),
                       partial_cache_lengths={
                           str(key): dict(key=None if c.keys is None else int(c.keys.shape[1]),
                                          value=None if c.values is None else int(c.values.shape[1]))
                           for key, c in caches.items()},
                       native_forward_success=False, complete=False)
        try:
            failure["archive"] = auxiliary.write(prefix+".bin", tensors)
            failure["complete"] = True
        except BaseException as archive_error:
            failure["archive_error_type"] = type(archive_error).__name__
            failure["archive_error"] = str(archive_error)
            failure["partial_archive"] = auxiliary.pending
            save(path, failure)
            failure_files.append(path.name)
            raise
        save(path, failure)
        failure_files.append(path.name)

    def native_forward(action, selectors):
        ids = stream.plan["original_prefix_plan"]["token_fixtures"][action["fixture"]]["input_ids"]
        begin, end = action["start"], action["end"]
        tokens = torch.tensor(ids[begin:end], dtype=torch.long)
        positions = torch.arange(begin, end, dtype=torch.long)
        rows = torch.tensor(selectors, dtype=torch.long)
        hidden, observations = capture.run(tokens, positions, rows)
        pending_outputs["hidden"] = hidden
        head_rows = (list(stream.history.reference_positions)
                     if action.get("phase") == "full_causal" else [end-begin-1])
        head = hidden.index_select(0, torch.tensor(head_rows, dtype=torch.long)).contiguous()
        logits = official.compute_logits(head)
        pending_outputs["logits"] = logits
        require(logits is not None, "Original output head produced logits")
        return hidden, logits, observations, list(head.shape)

    try:
        capture.install()
        for action in stream:
            guard()
            pending = action
            pending_outputs.clear()
            owner = Handle(action["slot"], action["epoch"], action["request"])
            kind = action["kind"]
            if kind == "start":
                before = inventory(owner.slot)
                banks.start(owner.slot, owner.epoch, owner.request)
                writer.reset(owner.slot)
                after = inventory(owner.slot)
                event_record(action, dict(other_slots_before=before, other_slots_after=after,
                                           original_native_start_executed=True))
                require(before == after, "Native request start preserves every other bank")
            elif kind == "release":
                before = inventory(owner.slot)
                banks.release(owner)
                writer.reset(owner.slot)
                after = inventory(owner.slot)
                event_record(action, dict(other_slots_before=before, other_slots_after=after,
                                           original_native_release_executed=True))
                require(before == after, "Native release preserves every other bank")
            elif action["native_rejection_required"]:
                before = inventory()
                before_ops = counts[0]
                forward_started = [False]
                def forbidden_forward():
                    forward_started[0] = True
                    return native_forward(action, [])
                try:
                    banks.execute(owner, action["start"], action["end"],
                                  forbidden_forward, capture_failure)
                except RuntimeError as error:
                    actual_error = dict(type=type(error).__name__, message=str(error))
                else:
                    raise RuntimeError("Original invalid request was accepted")
                after = inventory()
                event_record(action, dict(actual_error=actual_error,
                    original_error_messages=action["original_errors"],
                    internal_diagnostic_error_not_production_API_qualification=True,
                    projection_calls=counts[0]-before_ops,
                    native_forward_started=forward_started[0], before=before, after=after))
                require(not forward_started[0] and counts[0] == before_ops and before == after,
                        "Native invalid handle rejected without model work or mutation")
                rejections += 1
            elif action["native_injection_required"]:
                before = inventory(owner.slot)
                before_ops = counts[0]
                env["layer_calls"].clear()
                prior_failures = len(failure_files)
                expected_error = OriginalInjectedFailure("physical layer22 loop1")
                def inject(module, args, kwargs, output):
                    if kwargs.get("loop_idx", 0) == 1:
                        raise expected_error
                # The original fault is after layer22 output. The capture hooks
                # are already installed, so its final output is retained first.
                hook = official.backbone.layers[22].register_forward_hook(inject, with_kwargs=True)
                try:
                    try:
                        banks.execute(owner, action["start"], action["end"],
                                      lambda: native_forward(action, [0]), capture_failure)
                    except OriginalInjectedFailure as error:
                        require(error is expected_error, "Actual expected original injected error")
                    else:
                        raise RuntimeError("Original layer22 loop1 injection did not fire")
                finally:
                    hook.remove()
                require(len(failure_files) == prior_failures+1, "One complete native failure archive")
                failure = json.loads((out/failure_files[-1]).read_text())
                expected_lengths = action["original_observed"]["official_partial_lengths"]
                lengths = failure["partial_cache_lengths"]
                after = inventory(owner.slot)
                all_state = inventory()
                clear = all(c.keys is None and c.values is None for c in env["caches"])
                exact = (failure["complete"] is True and failure["completed_visits"] == 63
                         and failure["pending_visit"] is None and failure["selected_boundaries"] == 126
                         and set(lengths) == set(expected_lengths)
                         and all(lengths[key] == dict(key=value, value=value)
                                 for key, value in expected_lengths.items()))
                event_record(action, dict(failure_archive=failure_files[-1],
                    original_partial_lengths_and_counts_reproduced=exact, projection_calls=counts[0]-before_ops,
                    original_projection_calls=action["original_observed"]["official_projection_calls"],
                    other_slots_before=before, other_slots_after=after,
                    bound_caches_cleared=clear,
                    failed_request_alive=all_state[str(owner.slot)]["alive"],
                    failed_request_histories_empty=not all_state[str(owner.slot)]["histories"],
                    native_injected_failure_executed=True))
                require(exact and counts[0]-before_ops == 158 and before == after and clear
                        and not all_state[str(owner.slot)]["alive"]
                        and not all_state[str(owner.slot)]["histories"],
                        "Original native after-output failure and request invalidation reproduced")
                injections += 1
            else:
                require(action["native_call_required"], "Only original admitted native call")
                before_ops = counts[0]
                env["layer_calls"].clear()
                hidden, logits, values, head_shape = banks.execute(
                    owner, action["start"], action["end"],
                    lambda: native_forward(action, action["selected_local_rows"]), capture_failure)
                tensors = env["snapshot"](hidden, logits, banks.histories(owner))
                actual_archive = writer.write(action["event"], "official", owner.slot, owner.epoch,
                                              action["start"], action["end"], tensors)
                original_raw_bytes += actual_archive["bytes"]
                comparison = original_archive_comparison(actual_archive, action["original_archive"])
                selected = None
                if action["selected_local_rows"]:
                    selected = auxiliary.write(
                        f'event-{action["event"]:06d}-residual.bin', observed_tensors(values))
                    selected_rows += len(action["selected_local_rows"])
                    residual_raw_bytes += selected["bytes"]
                    selected_files.append(selected["file"])
                actual_top1 = logits.argmax(-1).tolist()
                actual_layers = {str(key): value for key, value in env["layer_calls"].items()}
                record = event_record(action, dict(
                    fixture=action["fixture"], phase=action["phase"], start=action["start"], end=action["end"],
                    original_archive_metadata_bound_by_event=action["original_event"],
                    actual_archive=actual_archive, archive_comparison=comparison,
                    selected_local_rows=action["selected_local_rows"], selected_archive=selected,
                    actual_top1=actual_top1, original_top1=action["original_top1"],
                    head_input_shape=head_shape, physical_layer_invocations=actual_layers,
                    actual_projection_calls=counts[0]-before_ops,
                    original_FP32_metrics=action["original_FP32_metrics"],
                    original_pointwise_failed=action["original_pointwise_failed"],
                    original_metric_recomputed=False, full_model_qualified=False))
                # Actual payloads and mismatching metadata have been retained.
                require(comparison["exact"] and actual_top1 == action["original_top1"]
                        and actual_layers == PHYSICAL_CALLS and counts[0]-before_ops == 281,
                        "All original output bytes, top1, layer visits and projection counts reproduced")
                require(selected is None or all(t["finite"] for t in selected["tensors"].values()),
                        "Selected residual observations finite")
                require(original_raw_bytes <= spec["expected_original_raw_bytes"],
                        "Original raw payload budget")
                successful_calls += 1
                del hidden, logits, values, tensors
                if successful_calls % 128 == 0:
                    print(json.dumps(dict(stage="complete_residual_replay", calls=successful_calls,
                                          selected_rows=selected_rows,
                                          raw_output_bytes=original_raw_bytes+auxiliary.bytes_written)), flush=True)
            require(env["control_active"][0] is False
                    and sum(env["route_calls"].values()) == initial_routes,
                    "No candidate execution or projection routing")
            pending = None
            pending_outputs.clear()
        require(stream.completed_summary is not None and stream.events_read == 6200
                and (successful_calls, rejections, injections) == (6185, 3, 1),
                "Every original native success, rejection and injection replayed")
        require(counts[0]-initial_ops == 6185*281+158
                and selected_rows == 153 and len(selected_files) == 89
                and residual_raw_bytes == selected_raw_bytes(153),
                "Complete original projections and all selected residual rows")
        require(original_raw_bytes == spec["expected_original_raw_bytes"]
                == stream.completed_summary["original_raw_archive_bytes"],
                "All original successful raw payload bytes retained")
        summary = dict(complete=True, diagnostic_only=True,
            original_event_metadata=stream.completed_summary,
            original_native_successful_calls=successful_calls, native_rejections=rejections,
            native_injections=injections, replay_files=replay_files,
            selected_files=selected_files, native_failure_archives=failure_files,
            original_successful_raw_bytes=original_raw_bytes,
            selected_residual_raw_bytes=residual_raw_bytes, auxiliary_raw_bytes=auxiliary.bytes_written,
            actual_official_projection_calls=counts[0]-initial_ops,
            all_original_successful_saved_outputs_reproduced=True,
            original_pointwise_failures=7, original_FP32_acceptance_passed=False,
            original_metrics_recomputed=False, independent_saved_byte_review_pending=True,
            production_request_lifecycle_qualified=False, generation_executed=False,
            neuron_executed=False, root_cause_established=False,
            full_model_qualified=False, new_speedup_claim=False,
            elapsed_seconds=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        save(out/"result.json", summary)
    except BaseException as error:
        save(out/"residual-replay-failure.json", dict(
            error_type=type(error).__name__, error=str(error), pending=pending,
            successful_native_calls=successful_calls, replay_files=replay_files,
            selected_files=selected_files, native_failure_archives=failure_files,
            original_raw_bytes=original_raw_bytes, auxiliary_raw_bytes=auxiliary.bytes_written,
            partial_auxiliary_archive=auxiliary.pending,
            original_pointwise_failures=7, root_cause_established=False,
            full_model_qualified=False, new_speedup_claim=False))
        raise
    finally:
        try:
            capture.close()
        finally:
            try:
                banks.clear_bound()
            finally:
                journal.close()


if __name__ == "__main__":
    raise RuntimeError("Source preparation only: complete native worker, source review, budgets and serial admission required")
