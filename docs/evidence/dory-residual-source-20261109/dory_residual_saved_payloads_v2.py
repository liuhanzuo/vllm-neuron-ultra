"""Prepared joined complete Dory saved-payload review, no runtime entrypoint.

This payload layer requires a separately authenticated original event stream and
externally frozen producer metadata after actual exit. Source/checkpoint/exit
provenance and serial admission belong to the future outer combined reviewer.
No original numerical failures become passes because replay bytes match.
"""
import hashlib
import json
from pathlib import Path

from dory_generation_journal_replay_v1 import SavedMetadata
from dory_residual_event_stream_v1 import OriginalEventStream
from dory_residual_saved_archives_v2 import CompleteArchiveReplay
from dory_residual_saved_observations_v1 import ResidualObservationReview, INJECTION_BYTES
from review_dory_long_matrix_saved_v3 import head_hash


VISIT_COUNTS = {str(i): 2 if 16 <= i < 56 else 1 for i in range(72)}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def inventory(reader, exclude=None):
    return {str(slot): dict(epoch=bank["epoch"], request=bank["request"],
        alive=bank["alive"], length=bank["length"],
        histories={label.removeprefix("cache-").replace("-key", "-0").replace("-value", "-1"):
                   head_hash(heads) for label, heads in bank["heads"].items()})
        for slot, bank in sorted(reader.banks.items()) if slot != exclude}


class ReplayRecordReview:
    def __init__(self, original, current, observations, guard):
        self.original, self.current, self.observations, self.guard = original, current, observations, guard
        self.poisoned = False
        self.replay_files, self.raw_files, self.failure_files = [], [], []
        self.projections = 0

    def accept(self, action, record, failure=None):
        require(not self.poisoned, "Failed joined payload review cannot resume")
        try:
            self._accept(action, record, failure)
        except BaseException:
            self.poisoned = True
            raise

    def _accept(self, action, record, failure):
        self.guard()
        require(self.original.events == self.current.events == action["event"],
                "Both original and new byte streams consume every event")
        for key in ("event", "kind", "original_event", "slot", "epoch", "request"):
            require(record[key] == action[key], "Saved replay belongs to original event and request")
        kind, slot = action["kind"], action["slot"]
        # Successful calls already authenticate every reconstructed full hash.
        # Rehash full inventories only at lifecycle boundaries, where their
        # hashes must be compared with recorded before/after state.
        if kind != "call":
            require(inventory(self.original) == inventory(self.current),
                    "Both independently reconstructed request inventories agree before event")
        if kind in ("start", "release"):
            require(failure is None and record["other_slots_before"] == inventory(self.current, slot),
                    "Lifecycle prior other-slot inventory matches reconstructed bytes")
            flag = "original_native_start_executed" if kind == "start" else "original_native_release_executed"
            require(record[flag] is True, "Original native lifecycle event recorded")
            self.original.accept(action); self.current.accept(action)
            require(record["other_slots_after"] == inventory(self.current, slot)
                    == record["other_slots_before"], "Lifecycle preserves all other actual saved histories")
        elif kind == "rejected_handle":
            require(failure is None and record["before"] == record["after"] == inventory(self.current)
                    and record["projection_calls"] == 0 and record["native_forward_started"] is False
                    and record["actual_error"] == {
                        "type": "RuntimeError", "message": "Stale, invalidated or foreign request handle"}
                    and record["original_error_messages"] == action["original_errors"]
                    and record["internal_diagnostic_error_not_production_API_qualification"] is True,
                    "Recorded original rejection has no forward or saved-state mutation")
            self.original.accept(action); self.current.accept(action)
            require(record["after"] == inventory(self.current), "All saved states persist after rejection")
        elif kind == "injected_failure":
            require(failure is not None
                    and record["other_slots_before"] == inventory(self.current, slot),
                    "Complete injected failure payload and independently reconstructed other slots")
            bank = self.current.banks[slot]
            require(bank["alive"] and bank["length"] == action["start"]
                    and (bank["epoch"], bank["request"]) == (action["epoch"], action["request"]),
                    "Injected failure binds to exact preceding successful cache owner")
            self.observations.injection(action, record, failure, bank["heads"])
            self.original.accept(action); self.current.accept(action)
            require(record["other_slots_after"] == inventory(self.current, slot)
                    and self.current.banks[slot]["alive"] is False
                    and not self.current.banks[slot]["heads"],
                    "Failure invalidation agrees with reconstructed lifecycle")
            self.failure_files.append(record["failure_archive"])
            self.raw_files.append(failure["archive"]["file"])
            self.projections += record["projection_calls"]
        else:
            require(kind == "call" and failure is None, "Only original successful call or lifecycle kind")
            for key in ("fixture", "phase", "start", "end", "selected_local_rows",
                        "original_FP32_metrics", "original_pointwise_failed"):
                require(record[key] == action[key], "Original complete call selection and metrics retained")
            require(record["original_archive_metadata_bound_by_event"] == action["original_event"]
                    and record["actual_top1"] == record["original_top1"] == action["original_top1"]
                    and record["head_input_shape"] == [33 if action["phase"] == "full_causal" else 1, 2560]
                    and record["physical_layer_invocations"] == VISIT_COUNTS
                    and record["actual_projection_calls"] == 281
                    and record["original_metric_recomputed"] is False
                    and record["full_model_qualified"] is False,
                    "Original complete forward/head shape, visits, projections and limited verdict")
            require(record["archive_comparison"] == dict(exact=True, header_differences=[],
                    missing_tensors=[], extra_tensors=[], changed_tensors=[], tensor_count=114,
                    independent_raw_review_complete=False),
                    "Producer archive comparison has no omitted or mismatched tensors")
            self.original.accept(action, action["original_archive"])
            self.current.accept(action, record["actual_archive"])
            require(record["actual_archive"] == action["original_archive"],
                    "Both actual saved byte streams match same original descriptors")
            selected = record["selected_archive"]
            if action["selected_local_rows"]:
                require(selected is not None, "Every selected original residual archived")
                self.observations.selected(action, selected)
                self.raw_files.append(selected["file"])
            else:
                require(selected is None, "No fabricated unselected residual archive")
            self.raw_files.append(record["actual_archive"]["file"])
            self.projections += 281
        if kind != "call":
            require(inventory(self.original) == inventory(self.current),
                    "Both independently reconstructed request inventories agree after event")
        self.replay_files.append(f'event-{action["event"]:06d}-replay.json')

    def finish(self, summary, stream_summary):
        require(not self.poisoned, "Failed joined payload review cannot finalize")
        original = self.original.finish()
        current = self.current.finish()
        observations = self.observations.finish()
        require(summary["complete"] is True and summary["diagnostic_only"] is True
                and summary["original_event_metadata"] == stream_summary
                and summary["original_native_successful_calls"] == 6185
                and summary["native_rejections"] == 3 and summary["native_injections"] == 1
                and summary["actual_official_projection_calls"] == self.projections == 6185*281+158
                and summary["replay_files"] == self.replay_files and len(self.replay_files) == 6200
                and summary["selected_files"] == self.observations.selected_files
                and summary["native_failure_archives"] == self.failure_files == ["event-006194-failure.json"],
                "Complete producer event/call/projection/file totals independently reconstructed")
        require(summary["original_successful_raw_bytes"] == original["raw_bytes_verified"]
                    == current["raw_bytes_verified"] == stream_summary["original_raw_archive_bytes"]
                and summary["selected_residual_raw_bytes"] == 175472640
                and summary["auxiliary_raw_bytes"] == 175472640+INJECTION_BYTES
                and summary["all_original_successful_saved_outputs_reproduced"] is True
                and summary["original_pointwise_failures"] == 7
                and summary["independent_saved_byte_review_pending"] is True,
                "Every original output, residual and failure byte included")
        false_fields = ("original_FP32_acceptance_passed", "original_metrics_recomputed",
            "production_request_lifecycle_qualified", "generation_executed", "neuron_executed",
            "root_cause_established", "full_model_qualified", "new_speedup_claim")
        require(all(summary[k] is False for k in false_fields), "Every original failed or unqualified verdict retained")
        return dict(complete=True, diagnostic_only=True, original_archives=original,
            replay_archives=current, observations=observations, replay_files=self.replay_files,
            original_events=6200, original_successful_calls=6185, original_pointwise_failures=7,
            both_original_and_new_successful_payloads_verified=True,
            source_checkpoint_exit_provenance_reviewed_here=False,
            original_top1_recomputed=False, original_FP32_metrics_recomputed=False,
            production_lifecycle_qualified=False, root_cause_established=False,
            full_model_qualified=False, new_speedup_claim=False)


def review_payloads(stream, source, metadata_pins, budget, guard):
    """A future outer reviewer must freeze every metadata pin before this call."""
    guard()
    require(type(stream) is OriginalEventStream and not stream.started,
            "Fresh authenticated original complete event stream")
    source = Path(source)
    require(source != stream.root, "Fresh replay directory distinct from retained original")
    require(not (source/"residual-replay-failure.json").exists(), "Failed producer cannot be promoted")
    metadata = SavedMetadata(source, metadata_pins, guard)
    raw = metadata.bytes("residual-events.jsonl", 4*2**20)
    require(raw.endswith(b"\n"), "Complete saved journal termination")
    entries = [json.loads(line) for line in raw.splitlines()]
    require(len(entries) == 6200, "All6200 saved replay journal entries")
    half = budget["combined_cache_bytes"]//2
    original = CompleteArchiveReplay(stream.root, guard,
        max_state_bytes=half, max_raw_bytes=budget["expected_original_raw_bytes"])
    current = CompleteArchiveReplay(source, guard,
        max_state_bytes=half, max_raw_bytes=budget["expected_original_raw_bytes"])
    observations = ResidualObservationReview(source, guard, list(stream.history.reference_positions),
        max_raw_bytes=budget["auxiliary_raw_bytes"], max_reference_bytes=budget["selected_reference_bytes"])
    joined = ReplayRecordReview(original, current, observations, guard)
    for action in stream:
        guard()
        entry = entries[action["event"]]
        name = f'event-{action["event"]:06d}-replay.json'
        require(set(entry) == {"event", "file", "sha256"}
                and entry["event"] == action["event"] and entry["file"] == name,
                "Every indexed saved replay event in original order")
        record = metadata.json(name, 8*2**20, expected_sha=entry["sha256"])
        failure = metadata.json("event-006194-failure.json", 8*2**20) if action["native_injection_required"] else None
        joined.accept(action, record, failure)
    require(stream.completed_summary is not None, "All original event bodies authenticated")
    summary = metadata.json("result.json", 8*2**20)
    result = joined.finish(summary, stream.completed_summary)
    expected_files = set(joined.replay_files + joined.raw_files + joined.failure_files
        + ["residual-events.jsonl", "result.json", "loader-proof.json", "loaded-parameters.json",
           "model.json", "tokenizer.json"])
    require({p.name for p in source.iterdir()} == expected_files,
            "No missing or unaccounted replay payload or metadata files")
    result["all_producer_output_names_accounted"] = True
    return result
