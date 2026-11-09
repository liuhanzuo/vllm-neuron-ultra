"""Synthetic observation bytes only; no original tensor or model execution."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import dory_residual_saved_observations_v1 as review


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Observations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.positions = list(range(33))
        self.reader = review.ResidualObservationReview(
            self.root, lambda: None, self.positions,
            max_raw_bytes=256*2**20, max_reference_bytes=128*2**20)

    def action(self, reference=False, fixture=0):
        return dict(event=1 if reference else 2, fixture=fixture, kind="call",
            native_call_required=True, start=0 if reference else 1, end=2060 if reference else 2,
            phase="full_causal" if reference else "decode",
            selected_local_rows=self.positions if reference else [0],
            original_pointwise_failed=None if reference else True,
            original_FP32_metrics=None if reference else {"max_abs_error": 1.25})

    def write(self, name, tensors):
        archive = dict(file=name, bytes=0, tensors={}, complete=True)
        whole = hashlib.sha256()
        with (self.root/name).open("wb") as f:
            for key, (shape, data) in tensors.items():
                f.write(data); whole.update(data)
                archive["tensors"][key] = dict(shape=shape, dtype="BF16", finite=True,
                    offset=archive["bytes"], bytes=len(data), sha256=sha(data))
                archive["bytes"] += len(data)
        archive["sha256"] = whole.hexdigest()
        return archive

    def residual(self, action, mutate=None):
        n = len(action["selected_local_rows"])
        tensors = {key: ([n, 2560], b"\0"*(n*5120)) for key in review.BOUNDARIES}
        if mutate:
            mutate(tensors)
        return self.write(f'event-{action["event"]:06d}-residual.bin', tensors)

    def seed_reference(self, fixture=0):
        # Individual comparison tests seed synthetic references directly.
        # The full-reference test below separately checks actual33-row reading.
        self.reader.references[fixture] = {key: b"\0"*(33*5120) for key in review.BOUNDARIES}

    def failure(self):
        visits = ([(i, 0) for i in range(16)]
                  + [(i, l) for l in range(2) for i in range(16, 56)]
                  + [(i, 0) for i in range(56, 72)])
        completed = visits[:63]
        advanced = {i+72*l for i, l in completed if i % 2 == 0}
        ids = list(range(0, 16, 2))+list(range(16, 56, 2))+list(range(88, 128, 2))+list(range(56, 72, 2))
        lengths = {str(i): 34 if i in advanced else 33 for i in ids}
        action = dict(event=6194, kind="injected_failure", native_injection_required=True,
            fixture=1, start=33, end=34, layer=22, loop=1, after_output=True,
            slot=1, epoch=3, request="reset-B", original_event={"path": "/synthetic/event.json", "sha256": "a"*64},
            original_observed=dict(official_partial_lengths=lengths, official_projection_calls=158))
        record = {k:action[k] for k in ("event", "kind", "slot", "epoch", "request", "original_event")}
        record.update(failure_archive="event-006194-failure.json",
            original_partial_lengths_and_counts_reproduced=True,
            projection_calls=158, original_projection_calls=158,
            native_injected_failure_executed=True, bound_caches_cleared=True,
            failed_request_alive=False, failed_request_histories_empty=True,
            other_slots_before={"0": {"length": 2}}, other_slots_after={"0": {"length": 2}})
        tensors = {}
        for i, loop in completed:
            for label in ("input", "residual_output"):
                tensors[f"layer-{i}-loop-{loop}-{label}"] = ([1, 2560], b"\0"*5120)
        previous = {}
        for cid in ids:
            n = lengths[str(cid)]
            for kind in ("key", "value"):
                key = f"cache-{cid}-{kind}"
                heads = [struct.pack("<H", 0x3f00+(cid+h)%100)*(33*256) for h in range(4)]
                previous[key] = heads
                data = b"".join(head + (b"\0"*512 if n == 34 else b"") for head in heads)
                tensors[key] = ([4, n, 256], data)
        archive = self.write("event-006194-failure.bin", tensors)
        failure = dict(error_type="OriginalInjectedFailure", error="physical layer22 loop1",
            completed_visits=63, pending_visit=None, selected_boundaries=126,
            partial_cache_lengths={k:dict(key=n, value=n) for k, n in lengths.items()},
            native_forward_success=False, complete=True, archive=archive)
        return action, record, failure, previous

    def test_full_reference_reads_all33_rows_and224_boundaries(self):
        action = self.action(True)
        archive = self.residual(action)
        self.assertIsNone(self.reader.selected(action, archive))
        self.assertEqual(len(self.reader.references[0]), 224)
        self.assertEqual(self.reader.bytes_read, 33*224*5120)
        self.assertEqual(self.reader.selected_rows, 33)

    def test_second_loop_difference_is_localized_and_original_failure_retained(self):
        self.seed_reference()
        action = self.action()
        label = "layer-16-loop-1-residual_output"
        def mutate(tensors):
            tensors[label] = ([1, 2560], b"\x80\x3f"+b"\0"*5118)
        frame = self.reader.selected(action, self.residual(action, mutate))
        self.assertEqual(frame["first_nonidentical_boundary"], label)
        self.assertEqual(frame["comparisons"][label]["max_abs"], 1.0)
        self.assertTrue(frame["original_pointwise_failed"])
        self.assertEqual(frame["original_FP32_metrics"], {"max_abs_error": 1.25})
        self.assertFalse(frame["root_cause_established"])

    def test_exact_rows_do_not_reclassify_original_failure(self):
        self.seed_reference(); action = self.action()
        frame = self.reader.selected(action, self.residual(action))
        self.assertIsNone(frame["first_nonidentical_boundary"])
        self.assertTrue(frame["original_pointwise_failed"])
        self.assertFalse(frame["comparison_is_acceptance_metric"])

    def test_reference_memory_budget_fails_before_payload_open(self):
        action = self.action(True); archive = self.residual(action)
        self.reader.max_reference_bytes = 1
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(RuntimeError, "retained-byte budget"):
                self.reader.selected(action, archive)

    def test_shrunken_reference_interval_is_rejected(self):
        action = self.action(True); action["end"] = 33
        with self.assertRaisesRegex(RuntimeError, "complete reference"):
            self.reader.selected(action, {})

    def test_wrong_selected_coordinate_is_rejected(self):
        action = self.action(); action["selected_local_rows"] = [1]
        with self.assertRaisesRegex(RuntimeError, "local rows"):
            self.reader.selected(action, {})

    def test_wrong_fixture_has_no_reference(self):
        self.seed_reference(0); action = self.action(fixture=1)
        with self.assertRaisesRegex(RuntimeError, "prior full reference"):
            self.reader.selected(action, {})

    def test_duplicate_selected_event_is_rejected(self):
        self.seed_reference(); action = self.action(); archive = self.residual(action)
        self.reader.selected(action, archive)
        with self.assertRaisesRegex(RuntimeError, "Chronological"):
            self.reader.selected(action, archive)

    def test_missing_loop_boundary_is_rejected_before_read(self):
        self.seed_reference(); action = self.action(); archive = self.residual(action)
        archive["tensors"].pop("layer-22-loop-1-residual_output")
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(RuntimeError, "tensor order"):
                self.reader.selected(action, archive)

    def test_incomplete_archive_is_not_promoted(self):
        self.seed_reference(); action = self.action(); archive = self.residual(action)
        archive["complete"] = False
        with self.assertRaisesRegex(RuntimeError, "complete original"):
            self.reader.selected(action, archive)

    def test_actual_corrupt_residual_bytes_rejected_and_reader_poisoned(self):
        self.seed_reference(); action = self.action(); archive = self.residual(action)
        p = self.root/archive["file"]; data = bytearray(p.read_bytes()); data[0] ^= 1; p.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "tensor SHA"):
            self.reader.selected(action, archive)
        with self.assertRaisesRegex(RuntimeError, "cannot resume"):
            self.reader.selected(action, archive)

    def test_nonfinite_residual_rejected_even_with_matching_hashes(self):
        self.seed_reference(); action = self.action()
        def mutate(tensors):
            tensors[next(iter(tensors))] = ([1, 2560], b"\x80\x7f"+b"\0"*5118)
        archive = self.residual(action, mutate)
        with self.assertRaisesRegex(RuntimeError, "finite actual"):
            self.reader.selected(action, archive)

    def test_total_raw_budget_fails_before_open(self):
        self.seed_reference(); action = self.action(); archive = self.residual(action)
        self.reader.max_raw_bytes = 1
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(RuntimeError, "raw budget"):
                self.reader.selected(action, archive)

    def test_original_fault_retains126_boundaries_and112_actual_prefixes(self):
        action, record, failure, previous = self.failure()
        result = self.reader.injection(action, record, failure, previous)
        self.assertEqual(result["residual_boundaries"], 126)
        self.assertEqual(result["partial_cache_tensors"], 112)
        self.assertTrue(result["successful_prefixes_independently_verified"])
        self.assertFalse(result["original_failure_tensor_bytes_matched"])
        self.assertFalse(result["actual_cleanup_proven_by_payload_alone"])

    def test_failure_cache_prefix_corruption_rejected_even_with_updated_hashes(self):
        action, record, failure, previous = self.failure()
        archive = failure["archive"]; desc = archive["tensors"]["cache-94-key"]
        p = self.root/archive["file"]; data = bytearray(p.read_bytes()); data[desc["offset"]] ^= 1
        p.write_bytes(data); desc["sha256"] = sha(data[desc["offset"]:desc["offset"]+desc["bytes"]])
        archive["sha256"] = sha(data)
        with self.assertRaisesRegex(RuntimeError, "reconstructed successful prefix"):
            self.reader.injection(action, record, failure, previous)

    def test_fault_before_layer22_output_cannot_substitute_original_timing(self):
        action, record, failure, previous = self.failure()
        failure["completed_visits"] = 62; failure["pending_visit"] = [22, 1]
        with self.assertRaisesRegex(RuntimeError, "after-output failure"):
            self.reader.injection(action, record, failure, previous)

    def test_unadvanced_second_loop_cache_must_stay_at33(self):
        action, record, failure, previous = self.failure()
        failure["partial_cache_lengths"]["96"]["key"] = 34
        with self.assertRaisesRegex(RuntimeError, "partial cache length"):
            self.reader.injection(action, record, failure, previous)

    def test_reported_cleanup_must_preserve_other_slots(self):
        action, record, failure, previous = self.failure()
        record["other_slots_after"]["0"]["length"] = 3
        with self.assertRaisesRegex(RuntimeError, "other-slot stability"):
            self.reader.injection(action, record, failure, previous)

    def test_failed_request_owner_cannot_be_relabelled(self):
        action, record, failure, previous = self.failure(); record["epoch"] += 1
        with self.assertRaisesRegex(RuntimeError, "original owner"):
            self.reader.injection(action, record, failure, previous)

    def test_hidden_output_cannot_be_invented_after_midbackbone_failure(self):
        action, record, failure, previous = self.failure()
        failure["archive"]["tensors"]["hidden"] = {}
        with self.assertRaisesRegex(RuntimeError, "tensor order"):
            self.reader.injection(action, record, failure, previous)

    def test_partial_selection_cannot_finalize(self):
        with self.assertRaisesRegex(RuntimeError, "Every selected"):
            self.reader.finish()

    def test_no_numerical_or_producer_imports(self):
        forbidden = {"torch", "numpy", "vllm", "dory_residual_replay_runtime_v1",
                     "dory_residual_output_archive_v1", "dory_long_delta_writer_v1"}
        self.assertFalse(forbidden.intersection(name.split(".")[0] for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
