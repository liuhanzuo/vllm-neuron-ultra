"""Synthetic joins of actual byte readers; no complete original/native replay."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_dory_residual_saved_archives_v2 as archives_fixture
import check_dory_residual_saved_observations_v1 as observation_fixture
from dory_residual_saved_archives_v2 import CompleteArchiveReplay, HISTORY_BYTES_PER_TOKEN
from dory_residual_saved_observations_v1 import ResidualObservationReview
from dory_residual_saved_payloads_v2 import ReplayRecordReview, inventory


class JoinedPayloads(unittest.TestCase):
    def setUp(self):
        self.fixture = archives_fixture.SavedArchives()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.original_dir = self.root/"original"
        self.original_dir.mkdir()
        args = dict(max_state_bytes=4*2060*HISTORY_BYTES_PER_TOKEN, max_raw_bytes=512*2**20)
        self.original = CompleteArchiveReplay(self.original_dir, lambda: None, **args)
        self.current = self.fixture.reader
        self.observations = ResidualObservationReview(self.root, lambda: None, list(range(33)),
            max_raw_bytes=256*2**20, max_reference_bytes=128*2**20)
        self.joined = ReplayRecordReview(self.original, self.current, self.observations, lambda: None)

    def action(self, kind, slot=0, epoch=1, request="A"):
        a = self.fixture.event(kind, slot, epoch, request)
        a["original_event"] = {"path": f'/synthetic/event-{a["event"]}.json', "sha256": "a"*64}
        return a

    def record(self, action):
        return {k:action[k] for k in ("event", "kind", "original_event", "slot", "epoch", "request")}

    def begin(self, slot=0, epoch=1, request="A"):
        a = self.action("start", slot, epoch, request)
        r = self.record(a)
        r.update(other_slots_before=inventory(self.current, slot),
                 other_slots_after=inventory(self.current, slot), original_native_start_executed=True)
        self.joined.accept(a, r)

    def call(self, **kwargs):
        a, archive = self.fixture.make(**kwargs)
        a.update(original_event={"path": f'/synthetic/event-{a["event"]}.json', "sha256": "a"*64},
            fixture=0, selected_local_rows=[], original_FP32_metrics=None, original_pointwise_failed=None)
        shutil.copyfile(self.root/archive["file"], self.original_dir/archive["file"])
        r = self.record(a)
        for key in ("fixture", "phase", "start", "end", "selected_local_rows",
                    "original_FP32_metrics", "original_pointwise_failed"):
            r[key] = a[key]
        r.update(original_archive_metadata_bound_by_event=a["original_event"],
            actual_archive=archive, archive_comparison=dict(exact=True, header_differences=[],
                missing_tensors=[], extra_tensors=[], changed_tensors=[], tensor_count=114,
                independent_raw_review_complete=False),
            selected_archive=None, actual_top1=[0], original_top1=[0], head_input_shape=[1, 2560],
            physical_layer_invocations={str(i): 2 if 16 <= i < 56 else 1 for i in range(72)},
            actual_projection_calls=281, original_metric_recomputed=False, full_model_qualified=False)
        return a, r

    def test_both_original_and_new_files_are_independently_consumed(self):
        self.begin(); a, r = self.call(); self.joined.accept(a, r)
        a, r = self.call(start=1, end=2); self.joined.accept(a, r)
        self.assertEqual(self.original.bytes_read, self.current.bytes_read)
        self.assertGreater(self.original.bytes_read, 0)
        self.assertEqual(inventory(self.original), inventory(self.current))
        self.assertEqual(self.joined.projections, 562)

    def test_changed_original_payload_is_not_trusted_from_old_metadata(self):
        self.begin(); a, r = self.call()
        p = self.original_dir/r["actual_archive"]["file"]
        data = bytearray(p.read_bytes()); data[0] = 1; p.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "Actual saved payload"):
            self.joined.accept(a, r)
        self.assertEqual(self.current.bytes_read, 0)
        self.assertTrue(self.joined.poisoned)

    def test_changed_new_payload_is_detected_after_original_passes(self):
        self.begin(); a, r = self.call()
        p = self.root/r["actual_archive"]["file"]
        data = bytearray(p.read_bytes()); data[0] = 1; p.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "Actual saved payload"):
            self.joined.accept(a, r)
        self.assertGreater(self.original.bytes_read, 0)
        with self.assertRaisesRegex(RuntimeError, "cannot resume"):
            self.joined.accept(a, r)

    def test_cross_request_record_is_rejected_before_bytes(self):
        self.begin(); a, r = self.call(); r["request"] = "B"
        with self.assertRaisesRegex(RuntimeError, "original event and request"):
            self.joined.accept(a, r)
        self.assertEqual(self.original.bytes_read, 0)

    def test_wrong_native_head_shape_is_rejected(self):
        self.begin(); a, r = self.call(); r["head_input_shape"] = [2, 2560]
        with self.assertRaisesRegex(RuntimeError, "forward/head"):
            self.joined.accept(a, r)

    def test_missing_recurrent_visit_is_rejected(self):
        self.begin(); a, r = self.call(); r["physical_layer_invocations"]["22"] = 1
        with self.assertRaisesRegex(RuntimeError, "forward/head"):
            self.joined.accept(a, r)

    def test_candidate_projection_count_cannot_substitute_official(self):
        self.begin(); a, r = self.call(); r["actual_projection_calls"] = 449
        with self.assertRaisesRegex(RuntimeError, "forward/head"):
            self.joined.accept(a, r)

    def test_original_failed_metric_cannot_be_relabelled(self):
        self.begin(); a, r = self.call(); a["original_pointwise_failed"] = True
        r["original_pointwise_failed"] = False
        with self.assertRaisesRegex(RuntimeError, "metrics retained"):
            self.joined.accept(a, r)

    def test_reset_and_stale_rejection_preserve_other_actual_slot_hashes(self):
        self.begin(); a, r = self.call(); self.joined.accept(a, r)
        self.begin(1, 1, "B")
        a, r = self.call(slot=1, request="B"); self.joined.accept(a, r)
        before = copy.deepcopy(self.current.banks[1])
        self.begin(0, 2, "C")
        a = self.action("rejected_handle", 0, 1, "A")
        a["original_errors"] = {"official": "old", "control": "old"}
        r = self.record(a)
        r.update(before=inventory(self.current), after=inventory(self.current), projection_calls=0,
            native_forward_started=False,
            actual_error={"type": "RuntimeError", "message": "Stale, invalidated or foreign request handle"},
            original_error_messages=a["original_errors"],
            internal_diagnostic_error_not_production_API_qualification=True)
        self.joined.accept(a, r)
        self.assertEqual(self.current.banks[1], before)
        self.assertEqual(inventory(self.original), inventory(self.current))

    def test_fabricated_other_slot_inventory_is_rejected(self):
        self.begin(); a, r = self.call(); self.joined.accept(a, r)
        a = self.action("start", 1, 1, "B"); r = self.record(a)
        r.update(other_slots_before={}, other_slots_after={}, original_native_start_executed=True)
        with self.assertRaisesRegex(RuntimeError, "reconstructed bytes"):
            self.joined.accept(a, r)

    def test_rejection_cannot_hide_native_forward_attempt(self):
        self.begin(); self.begin(0, 2, "C")
        a = self.action("rejected_handle", 0, 1, "A")
        a["original_errors"] = {"official": "old", "control": "old"}
        r = self.record(a)
        r.update(before=inventory(self.current), after=inventory(self.current), projection_calls=0,
            native_forward_started=True)
        with self.assertRaisesRegex(RuntimeError, "no forward"):
            self.joined.accept(a, r)

    def test_injection_reads_actual_failure_prefix_before_banks_are_cleared(self):
        fixture = observation_fixture.Observations()
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.root = self.root
        a, r, failure, previous = fixture.failure()
        for reader in (self.original, self.current):
            reader.events = 6194
            reader.banks[1] = dict(epoch=3, request="reset-B", alive=True, length=33,
                                   heads=copy.deepcopy(previous))
        r["other_slots_before"] = r["other_slots_after"] = {}
        self.joined.accept(a, r, failure)
        self.assertFalse(self.current.banks[1]["alive"])
        self.assertEqual(self.current.banks[1]["heads"], {})
        self.assertTrue(self.observations.failure_review["successful_prefixes_independently_verified"])
        self.assertEqual(self.joined.projections, 158)

    def test_incomplete_join_cannot_claim_full_replay(self):
        self.begin()
        with self.assertRaisesRegex(RuntimeError, "Only complete"):
            self.joined.finish({}, {})

    def test_successful_call_does_not_rehash_whole_inventories_redundantly(self):
        self.begin(); a, r = self.call()
        with patch("dory_residual_saved_payloads_v2.inventory", side_effect=AssertionError("redundant whole inventory")):
            self.joined.accept(a, r)
        self.assertEqual(self.original.calls, 1)
        self.assertEqual(self.current.calls, 1)

    def test_no_native_framework_or_producer_imports(self):
        forbidden = {"torch", "numpy", "vllm", "dory_residual_replay_runtime_v1",
                     "dory_residual_output_archive_v1", "dory_long_delta_writer_v1"}
        self.assertFalse(forbidden.intersection(name.split(".")[0] for name in sys.modules))


    def test_joined_reference_slot10_and11_are_preserved(self):
        for slot in (10, 11):
            request = f"full-causal-{slot-10}"
            self.begin(slot, 1, request)
            a, r = self.call(slot=slot, request=request)
            self.joined.accept(a, r)
            self.assertEqual(len(self.current.banks[slot]["heads"]), 112)
            a = self.action("release", slot, 1, request); r = self.record(a)
            r.update(other_slots_before=inventory(self.current, slot),
                     other_slots_after=inventory(self.current, slot), original_native_release_executed=True)
            self.joined.accept(a, r)
        self.assertEqual(inventory(self.current), {})
        self.assertEqual(self.current.calls, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
