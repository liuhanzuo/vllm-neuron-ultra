"""Complete-history metadata regressions; no raw tensor or native execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

from dory_residual_replay_history_v1 import (
    CACHE_LABELS, TENSOR_LABELS, EMPTY_SHA, PHYSICAL_CALLS, ResidualReplayHistory)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def inventory(history, exclude=None):
    result = {}
    for slot, bank in history.banks.items():
        if slot == exclude:
            continue
        official = {k: bank[k] for k in ("epoch", "request", "alive", "length")}
        official["histories"] = {label.removeprefix("cache-").replace("-key", "-0").replace("-value", "-1"): value
                                 for label, value in bank["hashes"].items()}
        result[str(slot)] = dict(official=official)
    return result


def start(history, slot=3, request="A"):
    old = history.banks.get(slot)
    other = inventory(history, slot)
    return dict(event=history.events, kind="start", slot=slot,
                epoch=history.epochs.get(slot, 0)+1, request=request,
                reset_without_release=old is not None,
                old_epoch=old["epoch"] if old else None,
                other_slots_before=other, other_slots_after=copy.deepcopy(other))


def call(history, slot=3, length=1, fixture=1):
    bank = history.banks[slot]
    begin, end = bank["length"], bank["length"]+length
    tensors = {}
    offset = 0
    for label in TENSOR_LABELS:
        whole = sha(f"{label}:whole:{end}")
        if label in CACHE_LABELS:
            previous = bank["hashes"].get(label)
            size = length*2048
            value = dict(shape=[4, end, 256], dtype="BF16", mode="append",
                         finite=True, prefix_valid=True, bytes=size,
                         sha256=whole if begin == 0 else sha(f"{label}:append:{begin}:{end}"),
                         full_sha256=whole,
                         prefix_sha256=previous if previous else EMPTY_SHA,
                         expected_previous=dict(length=begin, sha256=previous, finite=True) if previous else None)
        else:
            shape, dtype, size = (([length, 2560], "BF16", length*2560*2)
                                  if label == "hidden" else ([1, 131072], "F32", 131072*4))
            value = dict(shape=shape, dtype=dtype, mode="full", bytes=size,
                         sha256=whole, full_sha256=whole, finite=True)
        value["offset"] = offset
        offset += value["bytes"]
        tensors[label] = value
    archive = dict(slot=slot, epoch=bank["epoch"], start=begin, end=end, path="official",
                   file=f"{history.events:06d}-official.bin", tensors=tensors,
                   bytes=offset, sha256=sha("synthetic archive"))
    return dict(event=history.events, kind="call", slot=slot, epoch=bank["epoch"],
                request=bank["request"], fixture=fixture, start=begin, end=end,
                phase="long_decode", marked=False, finite_and_prefixes_valid=True,
                official_onednn_calls=281, control_onednn_calls=449,
                head_input_shape=[1, 2560], physical_layer_invocations=dict(PHYSICAL_CALLS),
                official_top1=[23], control_top1=[23], official=archive,
                exact={label: True for label in TENSOR_LABELS})


def injection(history, slot=3):
    fixture = json.loads((Path(__file__).resolve().parent /
        "dory-residual-original-injection-fixture-20261109-v1.json").read_text())
    bank = history.banks[slot]
    other = inventory(history, slot)
    return dict(event=history.events, kind="injected_failure", slot=slot,
                epoch=bank["epoch"], request=bank["request"], layer=22, loop=1,
                failed_histories_cleared=True, bound_caches_cleared=True,
                other_slots_before=other, other_slots_after=copy.deepcopy(other),
                observed=fixture["observed"])


def rejected(history, slot=3, epoch=None):
    bank = history.banks[slot]
    before = inventory(history)
    return dict(event=history.events, kind="rejected_handle", slot=slot,
                epoch=bank["epoch"] if epoch is None else epoch, request=bank["request"],
                label="synthetic", projection_calls=0, before=before, after=copy.deepcopy(before),
                errors=dict(official="invalid", control="invalid"))


class Metadata(unittest.TestCase):
    def setUp(self):
        self.h = ResidualReplayHistory({}, list(range(33)))
        self.h.accept(start(self.h))

    def reject_without_state_change(self, row, pattern):
        before = copy.deepcopy(self.h.__dict__)
        with self.assertRaisesRegex(RuntimeError, pattern):
            self.h.accept(row)
        self.assertEqual(self.h.__dict__, before)

    def test_two_loops_and_complete_outputs_are_required(self):
        action = self.h.accept(call(self.h, length=31))
        self.assertTrue(action["native_call_required"])
        self.assertEqual(len(action["original_archive"]["tensors"]), 114)
        self.assertEqual(len(self.h.banks[3]["hashes"]), 112)
        self.assertIn("cache-88-key", self.h.banks[3]["hashes"])
        self.assertIn("cache-126-value", self.h.banks[3]["hashes"])

    def test_missing_second_loop_tensor_not_hidden_by_same_total_count(self):
        row = call(self.h)
        value = row["official"]["tensors"].pop("cache-126-value")
        row["official"]["tensors"]["unobserved-placeholder"] = value
        self.reject_without_state_change(row, "Every original")

    def test_wrong_second_loop_prefix_rejected(self):
        self.h.accept(call(self.h))
        row = call(self.h)
        row["official"]["tensors"]["cache-88-key"]["prefix_sha256"] = "f"*64
        self.reject_without_state_change(row, "prefix SHA")

    def test_wrong_later_layer_previous_history_rejected(self):
        self.h.accept(call(self.h))
        row = call(self.h)
        row["official"]["tensors"]["cache-70-value"]["expected_previous"]["length"] = 0
        self.reject_without_state_change(row, "prior original")

    def test_changed_hidden_shape_rejected(self):
        row = call(self.h, length=31)
        row["official"]["tensors"]["hidden"]["shape"] = [1, 2560]
        self.reject_without_state_change(row, "hidden/logit")

    def test_changed_head_shape_rejected(self):
        row = call(self.h)
        row["head_input_shape"] = [33, 2560]
        self.reject_without_state_change(row, "head call shape")

    def test_incomplete_recurrent_visits_rejected(self):
        row = call(self.h)
        row["physical_layer_invocations"]["22"] = 1
        self.reject_without_state_change(row, "112 original")

    def test_noncontiguous_tensor_extents_rejected(self):
        row = call(self.h)
        row["official"]["tensors"]["cache-126-value"]["offset"] += 2
        self.reject_without_state_change(row, "Dense original")

    def test_whole_archive_extent_rejected(self):
        row = call(self.h)
        row["official"]["bytes"] -= 1
        self.reject_without_state_change(row, "Whole original")

    def test_whole_archive_sha_format_rejected(self):
        row = call(self.h)
        row["official"]["sha256"] = "missing"
        self.reject_without_state_change(row, "SHA256")

    def test_unselected_mark_rejected(self):
        row = call(self.h)
        row["marked"] = True
        self.reject_without_state_change(row, "Every original marked")

    def test_failure_metrics_preserved_without_promotion(self):
        row = call(self.h)
        row.update(marked=True, full_causal_vs_incremental=dict(max_abs_error=3.0))
        selected = {k: row[k] for k in ("fixture", "slot", "epoch", "start", "end", "phase")}
        selected.update(
            selected_local_rows=[0], original_FP32_metrics=copy.deepcopy(row["full_causal_vs_incremental"]),
            original_pointwise_failed=True,
            expected_original_cache={k: copy.deepcopy(row["official"]["tensors"][k])
                                     for k in CACHE_LABELS if k.split("-")[1] in ("0", "2", "4")})
        self.h.selected[row["event"]] = selected
        altered = copy.deepcopy(row)
        altered["full_causal_vs_incremental"]["max_abs_error"] = 1.0
        self.reject_without_state_change(altered, "FP32 metrics")
        result = self.h.accept(row)
        self.assertTrue(result["original_pointwise_failed"])
        self.assertEqual(result["original_FP32_metrics"]["max_abs_error"], 3.0)

    def test_unchanged_slot_inventory_covers_second_loop_hashes(self):
        self.h.accept(call(self.h))
        self.h.accept(start(self.h, slot=4, request="B"))
        row = start(self.h, slot=4, request="C")
        row["other_slots_before"]["3"]["official"]["histories"]["88-0"] = "f"*64
        row["other_slots_after"] = copy.deepcopy(row["other_slots_before"])
        self.reject_without_state_change(row, "112 original history")

    def test_missing_slot_inventory_hash_rejected(self):
        self.h.accept(call(self.h))
        row = start(self.h, slot=4, request="B")
        row["other_slots_before"]["3"]["official"]["histories"].pop("126-1")
        row["other_slots_after"] = copy.deepcopy(row["other_slots_before"])
        self.reject_without_state_change(row, "112 original history")

    def test_original_injection_requires_native_replay_and_exact_partial_state(self):
        self.h.accept(call(self.h, length=33))
        row = injection(self.h)
        changed = copy.deepcopy(row)
        changed["observed"]["official_partial_lengths"]["96"] = 34
        self.reject_without_state_change(changed, "partial-cache")
        action = self.h.accept(row)
        self.assertTrue(action["native_injection_required"])
        self.assertTrue(action["after_output"])
        self.assertEqual((action["start"], action["end"]), (33, 34))
        self.assertFalse(self.h.banks[3]["alive"])
        rejection = self.h.accept(rejected(self.h))
        self.assertTrue(rejection["native_rejection_required"])

    def test_stale_epoch_rejection_requires_native_call_attempt(self):
        self.h.accept(call(self.h))
        self.h.accept(start(self.h))
        action = self.h.accept(rejected(self.h, epoch=1))
        self.assertTrue(action["native_rejection_required"])
        self.assertFalse(action["native_call_required"])

    def test_native_injection_cannot_use_earlier_layer(self):
        self.h.accept(call(self.h, length=33))
        row = injection(self.h)
        row["layer"] = 4
        self.reject_without_state_change(row, "after-output")

    def test_request_fixture_may_change_only_after_reset(self):
        self.h.accept(call(self.h, fixture=0))
        self.reject_without_state_change(call(self.h, fixture=1), "keeps its fixture")
        self.h.accept(start(self.h))
        self.h.accept(call(self.h, fixture=1))

    def test_paired_top1_metadata_must_be_complete(self):
        row = call(self.h)
        row["control_top1"] = [24]
        self.reject_without_state_change(row, "paired top1")

    def test_skipped_event_cannot_advance_cursor(self):
        row = call(self.h)
        row["event"] += 1
        self.reject_without_state_change(row, "chronological")

    def test_partial_history_cannot_claim_complete(self):
        with self.assertRaisesRegex(RuntimeError, "Every original event"):
            self.h.finish()


class OriginalStartup(unittest.TestCase):
    def test_six_authenticated_original_events_cover_two_full114_tensor_calls(self):
        fixture = json.loads((Path(__file__).resolve().parent /
            "dory-residual-original-startup-fixture-20261109-v1.json").read_text())
        history = ResidualReplayHistory(
            {int(k): v for k, v in fixture["selected"].items()}, fixture["reference_positions"])
        calls = []
        for event in fixture["events"]:
            raw = event["raw_json"].encode()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), event["sha256"])
            action = history.accept(json.loads(raw))
            if action["native_call_required"]:
                calls.append(action)
        self.assertEqual((history.events, history.calls, history.starts, history.releases), (6, 2, 2, 2))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(row["original_archive"]["tensors"]) == 114 for row in calls))
        self.assertEqual(history.banks, {})
        with self.assertRaisesRegex(RuntimeError, "Every original event"):
            history.finish()

    def test_no_native_modules_imported(self):
        self.assertFalse(any(name.split(".")[0] in {"torch", "numpy", "vllm", "neuronxcc"}
                             for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
