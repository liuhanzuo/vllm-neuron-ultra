"""Prepared independent selected-residual and injected-failure byte review.

Caller must bind original actions, freeze producer metadata and own a serial
post-exit review. This module does not execute a model or establish provenance,
production lifecycle qualification, original FP32 acceptance or a speedup.
"""
import hashlib
import math

from dory_generation_cache_replay_v1 import OrdinaryFiles
from review_dory_long_matrix_saved_v3 import finite, head_hash
from explain_dory_long_cache_differences_v1 import differences


VISITS = (tuple((i, 0) for i in range(16))
          + tuple((i, loop) for loop in range(2) for i in range(16, 56))
          + tuple((i, 0) for i in range(56, 72)))
BOUNDARIES = tuple(f"layer-{i}-loop-{loop}-{label}"
                   for i, loop in VISITS for label in ("input", "residual_output"))
CACHE_ORDER = (tuple(range(0, 16, 2)) + tuple(range(16, 56, 2))
               + tuple(range(88, 128, 2)) + tuple(range(56, 72, 2)))
ADVANCED = set(range(0, 56, 2)) | {88, 90, 92, 94}
PARTIAL_LENGTHS = {str(i): 34 if i in ADVANCED else 33 for i in CACHE_ORDER}
RESIDUAL_BYTES_PER_ROW = 224 * 2560 * 2
INJECTION_BYTES = 126*5120 + sum(2*4*n*256*2 for n in PARTIAL_LENGTHS.values())


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def fingerprint(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Exact saved SHA256")


class ResidualObservationReview:
    def __init__(self, root, guard, reference_positions, *, max_raw_bytes, max_reference_bytes):
        require(isinstance(reference_positions, list) and len(reference_positions) == 33
                and all(type(p) is int and 0 <= p < 2060 for p in reference_positions)
                and reference_positions == sorted(set(reference_positions)),
                "All33 original selected reference positions")
        require(type(max_raw_bytes) is int and max_raw_bytes > 0
                and type(max_reference_bytes) is int and max_reference_bytes > 0,
                "Explicit observation and retained-reference byte limits")
        self.files, self.guard = OrdinaryFiles(root, guard), guard
        self.positions = list(reference_positions)
        self.max_raw_bytes, self.max_reference_bytes = max_raw_bytes, max_reference_bytes
        self.references, self.frames, self.selected_files = {}, [], []
        self.bytes_read = self.selected_rows = self.residual_bytes = self.failure_bytes = 0
        self.last_selected_event = -1
        self.failure_review = None
        self.poisoned = False

    def _read(self, archive, name, layout):
        self.guard()
        require(set(archive) == {"file", "bytes", "tensors", "complete", "sha256"}
                and archive["complete"] is True and archive["file"] == name
                and list(archive["tensors"]) == list(layout),
                "Exact complete original observation archive and tensor order")
        offset = 0
        for label, shape in layout.items():
            desc = archive["tensors"][label]
            size = math.prod(shape)*2
            require(set(desc) == {"shape", "dtype", "finite", "offset", "bytes", "sha256"}
                    and desc["shape"] == shape and desc["dtype"] == "BF16"
                    and desc["finite"] is True and type(desc["offset"]) is int
                    and desc["offset"] == offset and type(desc["bytes"]) is int
                    and desc["bytes"] == size,
                    "Exact observation shape, dtype, finite claim and extent")
            fingerprint(desc["sha256"])
            offset += size
        require(type(archive["bytes"]) is int and archive["bytes"] == offset
                and self.bytes_read + offset <= self.max_raw_bytes,
                "Complete observation raw budget before open")
        fingerprint(archive["sha256"])
        values, whole = {}, hashlib.sha256()
        with self.files.open(name, offset) as stream:
            for label, desc in archive["tensors"].items():
                data, part = bytearray(), hashlib.sha256()
                while len(data) < desc["bytes"]:
                    self.guard()
                    block = stream.read(min(65536, desc["bytes"]-len(data)))
                    require(bool(block) and len(block) % 2 == 0 and finite(block, "BF16"),
                            "Complete finite actual observation bytes")
                    data.extend(block); part.update(block); whole.update(block)
                    self.bytes_read += len(block)
                require(part.hexdigest() == desc["sha256"], "Actual observation tensor SHA")
                values[label] = bytes(data)
            require(stream.read(1) == b"" and whole.hexdigest() == archive["sha256"],
                    "Whole saved observation SHA")
        return values

    def selected(self, action, archive):
        require(not self.poisoned, "Interrupted observation review cannot resume")
        try:
            return self._selected(action, archive)
        except BaseException:
            self.poisoned = True
            raise

    def _selected(self, action, archive):
        self.guard()
        event, fixture = action["event"], action["fixture"]
        require(type(event) is int and event > self.last_selected_event
                and type(fixture) is int and fixture in (0, 1)
                and action["kind"] == "call" and action["native_call_required"] is True,
                "Chronological original selected successful call")
        begin, end = action["start"], action["end"]
        require(type(begin) is type(end) is int and 0 <= begin < end <= 2060,
                "Original unshrunk selected call interval")
        reference = action["phase"] == "full_causal"
        rows = self.positions if reference else [end-begin-1]
        require(action["selected_local_rows"] == rows,
                "Original selected local rows without narrowing native forward")
        if reference:
            require(begin == 0 and end == 2060 and fixture not in self.references
                    and action["original_pointwise_failed"] is None,
                    "One original complete reference per fixture")
            require((len(self.references)+1)*33*RESIDUAL_BYTES_PER_ROW <= self.max_reference_bytes,
                    "All selected reference payloads fit retained-byte budget")
        else:
            require(fixture in self.references and end-1 in self.positions
                    and type(action["original_pointwise_failed"]) is bool,
                    "Original marked point belongs to prior full reference")
        name = f"event-{event:06d}-residual.bin"
        values = self._read(archive, name, {key: [len(rows), 2560] for key in BOUNDARIES})
        frame = None
        if reference:
            self.references[fixture] = values
        else:
            index = self.positions.index(end-1)
            comparisons = {}
            for key in BOUNDARIES:
                self.guard()
                reference_row = self.references[fixture][key][index*5120:(index+1)*5120]
                comparisons[key] = differences(reference_row, values[key])
            first = next((key for key, stats in comparisons.items()
                          if stats["bitwise_equal_words"] != stats["elements"]), None)
            frame = dict(event=event, fixture=fixture, position=end-1,
                         comparisons=comparisons, first_nonidentical_boundary=first,
                         original_FP32_metrics=action["original_FP32_metrics"],
                         original_pointwise_failed=action["original_pointwise_failed"],
                         original_metrics_recomputed=False, comparison_is_acceptance_metric=False,
                         root_cause_established=False, full_model_qualified=False)
            self.frames.append(frame)
        self.selected_rows += len(rows)
        self.residual_bytes += archive["bytes"]
        self.selected_files.append(name)
        self.last_selected_event = event
        return frame

    def injection(self, action, record, failure, previous_heads):
        require(not self.poisoned, "Interrupted observation review cannot resume")
        try:
            return self._injection(action, record, failure, previous_heads)
        except BaseException:
            self.poisoned = True
            raise

    def _injection(self, action, record, failure, previous_heads):
        self.guard()
        require(self.failure_review is None and action["event"] == 6194
                and action["kind"] == "injected_failure" and action["native_injection_required"] is True
                and (action["fixture"], action["start"], action["end"], action["layer"], action["loop"])
                    == (1, 33, 34, 22, 1) and action["after_output"] is True,
                "Original single event6194 after-output injection")
        require(action["original_observed"]["official_partial_lengths"] == PARTIAL_LENGTHS
                and action["original_observed"]["official_projection_calls"] == 158,
                "Original authenticated partial lengths and projection count")
        for key in ("event", "kind", "slot", "epoch", "request", "original_event"):
            require(record[key] == action[key], "Injected event bound to original owner and source")
        require(record["failure_archive"] == "event-006194-failure.json"
                and record["original_partial_lengths_and_counts_reproduced"] is True
                and record["projection_calls"] == record["original_projection_calls"] == 158
                and record["native_injected_failure_executed"] is True
                and record["bound_caches_cleared"] is True
                and record["failed_request_alive"] is False
                and record["failed_request_histories_empty"] is True
                and record["other_slots_before"] == record["other_slots_after"],
                "Original recorded failure counts, invalidation and other-slot stability")
        require(set(failure) == {"error_type", "error", "completed_visits", "pending_visit",
                                "selected_boundaries", "partial_cache_lengths",
                                "native_forward_success", "complete", "archive"}
                and failure["error_type"] == "OriginalInjectedFailure"
                and failure["error"] == "physical layer22 loop1"
                and failure["completed_visits"] == 63 and failure["pending_visit"] is None
                and failure["selected_boundaries"] == 126 and failure["complete"] is True
                and failure["native_forward_success"] is False,
                "Complete original after-output failure evidence")
        require(failure["partial_cache_lengths"] == {
                    key: dict(key=n, value=n) for key, n in PARTIAL_LENGTHS.items()},
                "Every actual partial cache length retained")
        layout = {key: [1, 2560] for key in BOUNDARIES[:126]}
        cache_labels = [f"cache-{i}-{kind}" for i in CACHE_ORDER for kind in ("key", "value")]
        require(set(previous_heads) == set(cache_labels), "All112 prior successful cache tensors")
        for i in CACHE_ORDER:
            for kind in ("key", "value"):
                key = f"cache-{i}-{kind}"
                require(len(previous_heads[key]) == 4
                        and all(len(head) == 33*512 for head in previous_heads[key]),
                        "Original four-head pre-failure histories at length33")
                layout[key] = [4, PARTIAL_LENGTHS[str(i)], 256]
        archive = failure["archive"]
        values = self._read(archive, "event-006194-failure.bin", layout)
        require(archive["bytes"] == INJECTION_BYTES, "Every failure residual and partial-cache byte")
        hashes = {}
        for i in CACHE_ORDER:
            for kind in ("key", "value"):
                self.guard()
                key = f"cache-{i}-{kind}"; width = PARTIAL_LENGTHS[str(i)]*512
                data = values[key]
                require(all(data[h*width:h*width+33*512] == previous_heads[key][h] for h in range(4)),
                        "Actual failed cache retains every reconstructed successful prefix")
                hashes[key] = sha(data)
        self.failure_bytes = archive["bytes"]
        self.failure_review = dict(event=6194, saved_bytes_verified=archive["bytes"],
            residual_boundaries=126, partial_cache_tensors=112, full_partial_cache_sha256=hashes,
            successful_prefixes_independently_verified=True,
            original_partial_lengths_and_counts_reproduced=True,
            original_failure_tensor_bytes_matched=False,
            recorded_cleanup_consistent=True, actual_cleanup_proven_by_payload_alone=False,
            production_lifecycle_qualified=False, full_model_qualified=False)
        return self.failure_review

    def finish(self):
        require(not self.poisoned, "Failed observations cannot finalize")
        require(len(self.references) == 2 and len(self.frames) == 87
                and len(self.selected_files) == 89 and self.selected_rows == 153
                and self.residual_bytes == 175472640
                and sum(row["original_pointwise_failed"] for row in self.frames) == 7
                and self.failure_review is not None and self.failure_bytes == INJECTION_BYTES,
                "Every selected residual, original marked failure and injected partial payload")
        return dict(complete=True, selected_calls=89, selected_rows=153,
            residual_bytes_verified=self.residual_bytes, injected_bytes_verified=self.failure_bytes,
            original_pointwise_failures=7, frames=self.frames, failure_review=self.failure_review,
            original_FP32_metrics_recomputed=False, source_checkpoint_exit_provenance_verified_here=False,
            native_model_executed=False, production_lifecycle_qualified=False,
            root_cause_established=False, full_model_qualified=False, new_speedup_claim=False)
