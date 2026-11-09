"""Simulated ownership and failure tests; no native tensors or device work."""
from types import SimpleNamespace
import itertools
import unittest

from dory_residual_request_banks_v1 import (
    CACHE_IDS, Handle, FailureCaptureError, ResidualRequestBanks)


_addresses = itertools.count(1)


class Tensor:
    def __init__(self, length, marker):
        self.shape = (4, length, 256)
        self.dtype = "BF16"
        self.address = next(_addresses)
        self.marker = marker


def validate(tensor, length):
    if tensor.shape != (4, length, 256) or tensor.dtype != "BF16":
        raise ValueError("original BF16 cache shape")


def fingerprint(tensor):
    return tensor.marker


class BanksTests(unittest.TestCase):
    def setUp(self):
        self.caches = [SimpleNamespace(cache_id=key, keys=None, values=None) for key in CACHE_IDS]
        self.banks = ResidualRequestBanks(
            self.caches, lambda cache: cache.cache_id, validate, lambda tensor: tensor.address)
        self.calls = 0
        self.failures = []

    def forward(self, end, label):
        self.calls += 1
        for cache in self.caches:
            cache.keys = Tensor(end, (label, cache.cache_id, "key"))
            cache.values = Tensor(end, (label, cache.cache_id, "value"))
        return label

    def capture_failure(self, error, caches):
        self.failures.append((error, {key: (cache.keys, cache.values)
                                     for key, cache in caches.items()}))

    def execute(self, handle, start, end, label):
        return self.banks.execute(handle, start, end,
                                  lambda: self.forward(end, label), self.capture_failure)

    def inventory(self):
        return self.banks.inventory(fingerprint)

    def test_all56_histories_include_independent_second_loop(self):
        a = self.banks.start(0, 1, "A")
        result = self.execute(a, 0, 3, "A3")
        self.assertEqual(result, "A3")
        h = self.banks.histories(a)
        self.assertEqual(tuple(h), CACHE_IDS)
        self.assertNotEqual(h[16][0].address, h[88][0].address)
        self.assertEqual(len({t.address for pair in h.values() for t in pair}), 112)

    def test_interleaved_requests_rebind_every_history(self):
        a = self.banks.start(0, 1, "A")
        self.execute(a, 0, 3, "A3")
        b = self.banks.start(1, 1, "B")
        self.execute(b, 0, 5, "B5")
        before = self.banks.inventory(fingerprint, exclude=0)
        def advance_a():
            self.assertTrue(all(cache.keys.marker[0] == "A3" for cache in self.caches))
            return self.forward(4, "A4")
        self.banks.execute(a, 3, 4, advance_a, self.capture_failure)
        self.assertEqual(before, self.banks.inventory(fingerprint, exclude=0))
        self.assertEqual(self.inventory()["0"]["length"], 4)
        self.assertEqual(self.inventory()["1"]["length"], 5)

    def test_same_text_id_reset_invalidates_old_epoch_only(self):
        a = self.banks.start(0, 1, "A")
        self.execute(a, 0, 2, "A2")
        b = self.banks.start(1, 1, "B")
        self.execute(b, 0, 4, "B4")
        other = self.banks.inventory(fingerprint, exclude=0)
        old_histories = self.banks.histories(a)
        fresh = self.banks.start(0, 2, "A")
        self.assertEqual(dict(old_histories), {})
        self.assertEqual(dict(self.banks.histories(fresh)), {})
        self.assertEqual(other, self.banks.inventory(fingerprint, exclude=0))
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.execute(a, 2, 3, "invalid")
        self.assertEqual(self.calls, 2)

    def test_foreign_handle_rejected_before_cache_rebinding(self):
        a = self.banks.start(0, 1, "A")
        self.execute(a, 0, 2, "A2")
        before = [(cache.keys, cache.values) for cache in self.caches]
        with self.assertRaisesRegex(RuntimeError, "foreign"):
            self.execute(Handle(0, 1, "foreign"), 2, 3, "invalid")
        self.assertEqual(before, [(cache.keys, cache.values) for cache in self.caches])
        self.assertEqual(self.failures, [])
        self.assertEqual(self.calls, 1)

    def test_release_and_restart_preserve_epoch_history(self):
        a = self.banks.start(3, 1, "A")
        self.execute(a, 0, 2, "A2")
        self.banks.release(a)
        self.assertEqual(self.inventory(), {})
        b = self.banks.start(3, 2, "B")
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.execute(a, 2, 3, "invalid")
        self.execute(b, 0, 1, "B1")
        self.assertEqual(self.inventory()["3"]["epoch"], 2)

    def test_bad_epoch_does_not_invalidate_current_owner(self):
        a = self.banks.start(0, 1, "A")
        self.execute(a, 0, 2, "A2")
        before = self.inventory()
        for epoch in (1, 3):
            with self.assertRaisesRegex(RuntimeError, "exactly once"):
                self.banks.start(0, epoch, "B")
            self.assertEqual(self.inventory(), before)

    def test_noncontiguous_and_out_of_range_calls_do_not_execute(self):
        a = self.banks.start(0, 1, "A")
        for start, end in ((1, 2), (0, 0), (0, 262145), (False, 1)):
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                self.execute(a, start, end, "invalid")
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.failures, [])
        self.execute(a, 0, 1, "A1")

    def test_failed_request_keeps_partial_evidence_before_clear(self):
        a = self.banks.start(0, 1, "A")
        self.execute(a, 0, 3, "A3")
        b = self.banks.start(1, 1, "B")
        self.execute(b, 0, 4, "B4")
        other = self.banks.inventory(fingerprint, exclude=0)
        original = ValueError("layer22 loop1 after-output")
        def fail():
            self.caches[0].keys = Tensor(4, "partial")
            raise original
        with self.assertRaises(ValueError) as caught:
            self.banks.execute(a, 3, 4, fail, self.capture_failure)
        self.assertIs(caught.exception, original)
        self.assertEqual(self.failures[0][1][0][0].marker, "partial")
        self.assertTrue(all(cache.keys is None and cache.values is None for cache in self.caches))
        self.assertEqual(self.banks.inventory(fingerprint, exclude=0), other)
        self.assertFalse(self.inventory()["0"]["alive"])
        self.assertEqual(self.inventory()["0"]["histories"], {})
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            self.execute(a, 3, 4, "invalid")

    def test_failure_capture_error_cannot_skip_invalidation(self):
        a = self.banks.start(0, 1, "A")
        original = ValueError("native failure")
        retention = OSError("archive full")
        def forward():
            raise original
        def capture(error, caches):
            self.assertIs(error, original)
            raise retention
        with self.assertRaises(FailureCaptureError) as caught:
            self.banks.execute(a, 0, 1, forward, capture)
        self.assertIs(caught.exception.original, original)
        self.assertIs(caught.exception.retention, retention)
        self.assertIs(caught.exception.__cause__, original)
        self.assertFalse(self.inventory()["0"]["alive"])
        self.assertTrue(all(cache.keys is None and cache.values is None for cache in self.caches))

    def test_output_head_failure_also_aborts_after_complete_backbone(self):
        a = self.banks.start(0, 1, "A")
        def fail_head():
            self.forward(1, "complete_backbone")
            raise ValueError("head failure")
        with self.assertRaisesRegex(ValueError, "head failure"):
            self.banks.execute(a, 0, 1, fail_head, self.capture_failure)
        self.assertEqual(len(self.failures[0][1]), 56)
        self.assertFalse(self.inventory()["0"]["alive"])

    def test_missing_cache_cannot_commit(self):
        a = self.banks.start(0, 1, "A")
        def missing():
            self.forward(1, "A1")
            self.caches[-1].values = None
        with self.assertRaisesRegex(RuntimeError, "Every original"):
            self.banks.execute(a, 0, 1, missing, self.capture_failure)
        self.assertFalse(self.inventory()["0"]["alive"])

    def test_wrong_dtype_cannot_commit(self):
        a = self.banks.start(0, 1, "A")
        def wrong_dtype():
            self.forward(1, "A1")
            self.caches[0].keys.dtype = "F32"
        with self.assertRaisesRegex(ValueError, "BF16"):
            self.banks.execute(a, 0, 1, wrong_dtype, self.capture_failure)
        self.assertFalse(self.inventory()["0"]["alive"])

    def test_unadvanced_history_cannot_commit(self):
        a = self.banks.start(0, 1, "A")
        self.execute(a, 0, 1, "A1")
        def stale():
            self.forward(1, "stale")
        with self.assertRaisesRegex(ValueError, "shape"):
            self.banks.execute(a, 1, 2, stale, self.capture_failure)
        self.assertFalse(self.inventory()["0"]["alive"])

    def test_storage_alias_within_request_cannot_commit(self):
        a = self.banks.start(0, 1, "A")
        def alias():
            self.forward(1, "A1")
            self.caches[1].keys = self.caches[0].keys
        with self.assertRaisesRegex(RuntimeError, "112"):
            self.banks.execute(a, 0, 1, alias, self.capture_failure)

    def test_storage_alias_across_requests_cannot_commit(self):
        b = self.banks.start(1, 1, "B")
        self.execute(b, 0, 1, "B1")
        tensor = self.banks.histories(b)[0][0]
        before = self.banks.inventory(fingerprint, exclude=0)
        a = self.banks.start(0, 1, "A")
        def alias():
            self.forward(1, "A1")
            self.caches[0].keys = tensor
        with self.assertRaisesRegex(RuntimeError, "other slots"):
            self.banks.execute(a, 0, 1, alias, self.capture_failure)
        self.assertEqual(before, self.banks.inventory(fingerprint, exclude=0))

    def test_lifecycle_operations_inside_forward_are_rejected(self):
        a = self.banks.start(0, 1, "A")
        def illegal():
            self.banks.start(1, 1, "B")
        with self.assertRaisesRegex(RuntimeError, "during a forward"):
            self.banks.execute(a, 0, 1, illegal, self.capture_failure)
        self.assertEqual(set(self.inventory()), {"0"})
        self.assertFalse(self.inventory()["0"]["alive"])

    def test_recovery_requires_new_epoch_and_empty_histories(self):
        a = self.banks.start(0, 1, "A")
        def fail():
            raise ValueError("native failure")
        with self.assertRaises(ValueError):
            self.banks.execute(a, 0, 1, fail, self.capture_failure)
        b = self.banks.start(0, 2, "B")
        self.assertEqual(dict(self.banks.histories(b)), {})
        self.execute(b, 0, 2, "B2")
        self.assertTrue(self.inventory()["0"]["alive"])

    def test_duplicate_cache_ids_rejected(self):
        self.caches[-1].cache_id = self.caches[0].cache_id
        with self.assertRaisesRegex(RuntimeError, "56 distinct"):
            ResidualRequestBanks(self.caches, lambda c: c.cache_id, validate, lambda t: t.address)

    def test_boolean_cache_identity_rejected(self):
        self.caches[0].cache_id = False
        with self.assertRaisesRegex(RuntimeError, "56 distinct"):
            ResidualRequestBanks(self.caches, lambda c: c.cache_id, validate, lambda t: t.address)

    def test_nonempty_initial_cache_rejected(self):
        self.caches[0].keys = Tensor(1, "leftover")
        with self.assertRaisesRegex(RuntimeError, "empty before"):
            ResidualRequestBanks(self.caches, lambda c: c.cache_id, validate, lambda t: t.address)

    def test_handle_type_checks(self):
        for args in [(True, 1, "A"), (0, True, "A"), (-1, 1, "A"), (0, 1, "")]:
            with self.assertRaises(RuntimeError):
                Handle(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
