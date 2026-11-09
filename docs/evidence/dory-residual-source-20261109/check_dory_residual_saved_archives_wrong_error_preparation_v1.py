"""Small synthetic saved-byte regressions; no real Dory data or native model."""
import copy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from dory_residual_saved_archives_v1 import CompleteArchiveReplay, HISTORY_BYTES_PER_TOKEN

IDS = sorted(set(range(0, 72, 2)) | set(range(88, 128, 2)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SavedArchives(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.reader = CompleteArchiveReplay(self.root, lambda: None,
                                           max_state_bytes=2060*4*HISTORY_BYTES_PER_TOKEN,
                                           max_raw_bytes=1024*2**20)
        self.oracle = {}

    def event(self, kind, slot=0, epoch=1, request="A"):
        return dict(event=self.reader.events, kind=kind, slot=slot, epoch=epoch, request=request)

    def begin(self, slot=0, epoch=1, request="A"):
        self.reader.accept(self.event("start", slot, epoch, request))

    def make(self, start=0, end=1, slot=0, epoch=1, request="A"):
        action = self.event("call", slot, epoch, request)
        action.update(start=start, end=end, native_call_required=True,
                      phase="decode", original_top1=[0])
        metadata = dict(file=f'{action["event"]:06d}-official.bin', slot=slot,
                        epoch=epoch, path="official", start=start, end=end, tensors={})
        data = bytearray()
        payloads = [("hidden", b"\0"*((end-start)*2560*2), [end-start, 2560], "BF16")]
        payloads += [("logits", b"\0"*(131072*4), [1, 131072], "F32")]
        for cid in IDS:
            for kind in ("key", "value"):
                # Independent oracle: materialize each full head in time order.
                whole_heads = [b"".join(struct.pack("<H", 0x3f00 + (cid+h*9+t+epoch)%100)*256
                                       for t in range(end)) for h in range(4)]
                payload = b"".join(head[start*512:] for head in whole_heads)
                payloads.append((f"cache-{cid}-{kind}", payload, [4, end, 256], "BF16"))
        for label, payload, shape, dtype in payloads:
            cache = label.startswith("cache-")
            desc = dict(shape=shape, dtype=dtype, mode="append" if cache else "full",
                        offset=len(data), bytes=len(payload), sha256=sha(payload), finite=True)
            if cache:
                cid = int(label.split("-")[1])
                whole_heads = [b"".join(struct.pack("<H", 0x3f00 + (cid+h*9+t+epoch)%100)*256
                                       for t in range(end)) for h in range(4)]
                full = b"".join(whole_heads)
                previous = b"".join(head[:start*512] for head in whole_heads)
                desc.update(full_sha256=sha(full), prefix_sha256=sha(previous), prefix_valid=True,
                            expected_previous=None if start == 0 else
                            dict(length=start, sha256=sha(previous), finite=True))
                self.oracle[(slot, epoch, label)] = whole_heads
            else:
                desc["full_sha256"] = sha(payload)
            metadata["tensors"][label] = desc
            data.extend(payload)
        metadata.update(bytes=len(data), sha256=sha(data))
        (self.root/metadata["file"]).write_bytes(data)
        action["original_archive"] = copy.deepcopy(metadata)
        return action, metadata

    def accept(self, **kwargs):
        action, meta = self.make(**kwargs)
        self.reader.accept(action, meta)
        return action, meta

    def attack(self, mutate, *, regex=None):
        action, meta = self.make()
        mutate(meta)
        action["original_archive"] = copy.deepcopy(meta)
        with self.assertRaisesRegex(RuntimeError, regex or "."):
            self.reader.accept(action, meta)
        self.assertTrue(self.reader.poisoned)

    def test_all112_histories_are_reconstructed_per_head_across_appends(self):
        self.begin()
        self.accept(end=2)
        self.accept(start=2, end=3)
        bank = self.reader.banks[0]
        self.assertEqual(len(bank["heads"]), 112)
        for label, heads in bank["heads"].items():
            self.assertEqual([bytes(h) for h in heads], self.oracle[(0, 1, label)])
        self.assertEqual((bank["length"], self.reader.calls), (3, 2))

    def test_other_slot_survives_reset_release_and_stale_rejection(self):
        self.begin(); self.accept(end=2)
        self.begin(1, 1, "B"); self.accept(slot=1, request="B")
        before = copy.deepcopy(self.reader.banks[1])
        self.begin(0, 2, "C")
        self.reader.accept(self.event("rejected_handle", 0, 1, "A"))
        self.accept(epoch=2, request="C")
        self.reader.accept(self.event("release", 0, 2, "C"))
        self.assertEqual(self.reader.banks[1], before)
        self.assertNotIn(0, self.reader.banks)

    def test_injection_marker_invalidates_only_its_owner(self):
        self.begin(); self.accept()
        self.begin(1, 1, "B"); self.accept(slot=1, request="B")
        before = copy.deepcopy(self.reader.banks[1])
        self.reader.accept(self.event("injected_failure"))
        self.assertFalse(self.reader.banks[0]["alive"])
        self.assertEqual(self.reader.banks[0]["heads"], {})
        self.reader.accept(self.event("rejected_handle"))
        self.assertEqual(self.reader.banks[1], before)

    def test_finite_hidden_corruption_is_detected_from_bytes(self):
        self.begin(); action, meta = self.make()
        p = self.root/meta["file"]; data = bytearray(p.read_bytes()); data[0] = 1; p.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, "Actual saved payload"):
            self.reader.accept(action, meta)
        self.assertTrue(self.reader.poisoned)

    def test_nan_logits_cannot_pass_even_with_matching_payload_hashes(self):
        self.begin()
        def mutate(meta):
            p = self.root/meta["file"]; data = bytearray(p.read_bytes())
            t = meta["tensors"]["logits"]
            data[t["offset"]:t["offset"]+4] = struct.pack("<I", 0x7fc00000)
            p.write_bytes(data); meta["sha256"] = sha(data)
            t["sha256"] = t["full_sha256"] = sha(data[t["offset"]:t["offset"]+t["bytes"]])
        self.attack(mutate, regex="Actual saved payload")

    def test_second_loop_payload_hash_cannot_replace_full_history_hash(self):
        self.begin()
        def mutate(meta):
            p = self.root/meta["file"]; data = bytearray(p.read_bytes())
            t = meta["tensors"]["cache-88-value"]; data[t["offset"]] ^= 1
            p.write_bytes(data); meta["sha256"] = sha(data)
            t["sha256"] = sha(data[t["offset"]:t["offset"]+t["bytes"]])
        self.attack(mutate, regex="Actual saved payload")

    def test_wrong_recorded_predecessor_is_detected(self):
        self.begin(); self.accept()
        action, meta = self.make(start=1, end=2)
        meta["tensors"]["cache-126-key"]["expected_previous"]["sha256"] = "a"*64
        action["original_archive"] = copy.deepcopy(meta)
        with self.assertRaisesRegex(RuntimeError, "Recorded predecessor"):
            self.reader.accept(action, meta)

    def test_omitted_second_loop_tensor_is_rejected_before_open(self):
        self.begin()
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not open")):
            self.attack(lambda m: m["tensors"].pop("cache-88-key"), regex="Every original114")

    def test_bad_offset_is_rejected_before_open(self):
        self.begin()
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not open")):
            self.attack(lambda m: m["tensors"]["logits"].update(offset=1), regex="geometry")

    def test_shared_memory_budget_counts_all_active_slots_before_read(self):
        self.begin(); self.accept()
        self.begin(1, 1, "B")
        self.reader.max_state_bytes = HISTORY_BYTES_PER_TOKEN
        action, meta = self.make(slot=1, request="B")
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not open")):
            with self.assertRaisesRegex(RuntimeError, "reconstruction budget"):
                self.reader.accept(action, meta)

    def test_raw_budget_rejects_before_open(self):
        self.begin(); self.reader.max_raw_bytes = 1
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not open")):
            self.attack(lambda m: None, regex="raw read budget")

    def test_symlink_payload_is_rejected(self):
        self.begin(); action, meta = self.make()
        original = self.root/meta["file"]; target = self.root/"other.bin"
        original.rename(target); original.symlink_to(target)
        with self.assertRaises(OSError):
            self.reader.accept(action, meta)
        self.assertTrue(self.reader.poisoned)

    def test_wrong_epoch_call_is_rejected(self):
        self.begin()
        action, meta = self.make(epoch=2, request="B")
        with self.assertRaisesRegex(RuntimeError, "Exact live"):
            self.reader.accept(action, meta)

    def test_fallback_does_not_turn_original_success_into_new_baseline(self):
        self.begin()
        self.attack(lambda m: m["tensors"]["cache-0-key"].update(mode="full", prefix_valid=False),
                    regex="geometry")

    def test_midread_failure_poison_prevents_partial_resume(self):
        self.begin(); action, meta = self.make()
        calls = [0]
        def guard():
            calls[0] += 1
            if calls[0] == 8:
                raise RuntimeError("injected reader interruption")
        self.reader.guard = guard
        with self.assertRaisesRegex(RuntimeError, "injected reader interruption"):
            self.reader.accept(action, meta)
        self.assertGreater(self.reader.bytes_read, 0)
        self.reader.guard = lambda: None
        with self.assertRaisesRegex(RuntimeError, "cannot resume"):
            self.reader.accept(action, meta)

    def test_partial_history_cannot_finalize(self):
        self.begin(); self.accept()
        with self.assertRaisesRegex(RuntimeError, "Only complete"):
            self.reader.finish()

    def test_full_reference_head_cannot_be_shrunk_to_one_row(self):
        self.begin(); action, meta = self.make()
        action["phase"] = "full_causal"; action["original_top1"] = [0]*33
        with patch.object(self.reader.files, "open", side_effect=AssertionError("must not open")):
            with self.assertRaisesRegex(RuntimeError, "geometry"):
                self.reader.accept(action, meta)

    def test_new_metadata_cannot_differ_from_original_bound_description(self):
        self.begin(); action, meta = self.make()
        meta["tensors"]["hidden"]["sha256"] = "a"*64
        with self.assertRaisesRegex(RuntimeError, "authenticated original"):
            self.reader.accept(action, meta)

    def test_trailing_file_byte_is_detected(self):
        self.begin(); action, meta = self.make()
        with (self.root/meta["file"]).open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(RuntimeError, "exact declared length"):
            self.reader.accept(action, meta)

    def test_valid_handle_cannot_be_labelled_rejected(self):
        self.begin()
        with self.assertRaisesRegex(RuntimeError, "actually stale"):
            self.reader.accept(self.event("rejected_handle"))

    def test_no_native_framework_or_producer_imported(self):
        forbidden = {"torch", "numpy", "vllm", "dory_long_delta_writer_v1",
                     "dory_residual_replay_runtime_v1", "dory_residual_output_archive_v1"}
        self.assertFalse(forbidden.intersection(name.split(".")[0] for name in sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
