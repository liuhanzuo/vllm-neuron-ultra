"""Combined simulated Dory forward/capture/cache failure ordering checks.

No torch, original weights, numeric or performance claims. The after-output
injection models the retained original layer22/loop1 hook position.
"""
from types import SimpleNamespace
from pathlib import Path
import json
import unittest

from check_dory_residual_visit_capture_v1 import Tensor as Hidden, Handle as HookHandle
from check_dory_residual_request_banks_v1 import Tensor as CacheTensor, validate, fingerprint
from dory_residual_visit_capture_v1 import CaptureResidualVisits, visits
from dory_residual_request_banks_v1 import CACHE_IDS, ResidualRequestBanks, FailureCaptureError


class Block:
    def __init__(self, physical, caches):
        self.physical, self.caches = physical, caches
        self.pre, self.post = [], []

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        self.pre.append(hook)
        return HookHandle(self.pre, hook)

    def register_forward_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        self.post.append(hook)
        return HookHandle(self.post, hook)

    def __call__(self, *args, **kwargs):
        for hook in self.pre:
            assert hook(self, args, kwargs) is None
        positions, hidden = args
        loop = kwargs.get("loop_idx", 0)
        if self.physical % 2 == 0:
            cache = self.caches[self.physical + 72 * loop]
            old_length = 0 if cache.keys is None else cache.keys.shape[1]
            assert old_length == positions[0]
            end = positions[-1] + 1
            cache.keys = CacheTensor(end, (self.physical, loop, end, "key"))
            cache.values = CacheTensor(end, (self.physical, loop, end, "value"))
        result = Hidden(rows=hidden.shape[0], marker=(self.physical, loop))
        for hook in self.post:
            assert hook(self, args, kwargs, result) is None
        return result


class Model:
    def __init__(self, caches):
        config = SimpleNamespace(num_hidden_layers=72, n_input_layers=16,
                                 n_recurrent_layers=40, n_output_layers=16,
                                 n_recurrent_loops=2, hidden_size=2560)
        self.backbone = SimpleNamespace(
            config=config, num_loops=2, layers=[Block(i, caches) for i in range(72)])

    def __call__(self, tokens, positions):
        assert len(tokens) == len(positions)
        hidden = Hidden(rows=len(tokens), marker="embedding")
        for physical, loop in visits(self.backbone.config):
            kwargs = {"loop_idx": loop} if 16 <= physical < 56 else {}
            hidden = self.backbone.layers[physical](positions, hidden, **kwargs)
        return hidden


class InjectedFailure(RuntimeError):
    pass


class Integration(unittest.TestCase):
    def setUp(self):
        self.caches = {key: SimpleNamespace(cache_id=key, keys=None, values=None)
                       for key in CACHE_IDS}
        self.model = Model(self.caches)
        self.capture = CaptureResidualVisits(
            self.model,
            lambda tensor, selectors: Hidden(rows=len(selectors), marker=tensor.marker),
            lambda: None)
        self.capture.install()
        self.banks = ResidualRequestBanks(
            list(self.caches.values()), lambda c: c.cache_id, validate, lambda t: t.address)
        self.original_injection = json.loads((Path(__file__).resolve().parent /
            "dory-residual-original-injection-fixture-20261109-v1.json").read_text())
        self.retained = []
        self.head_calls = 0

    def tearDown(self):
        self.capture.close()

    def call(self, owner, start, end, retention=None):
        def forward():
            result = self.capture.run([1] * (end-start), list(range(start, end)), [end-start-1])
            self.head_calls += 1
            return result

        def retain(error, caches):
            self.retained.append(dict(
                error=error, completed=self.capture.completed,
                pending=self.capture.pending_visit,
                values=dict(self.capture.partial_values),
                lengths={key: None if c.keys is None else c.keys.shape[1]
                         for key, c in caches.items()}))
            if retention is not None:
                raise retention
        return self.banks.execute(owner, start, end, forward, retain)

    def inject_after_output(self, error):
        def fail(module, args, kwargs, output):
            if kwargs.get("loop_idx", 0) == 1:
                raise error
        # Installed after capture, just as a later native failure hook must be.
        return self.model.backbone.layers[22].register_forward_hook(fail, with_kwargs=True)

    def test_after_output_second_loop_failure_preserves126_boundaries_and_other_slot(self):
        a = self.banks.start(0, 1, "A")
        self.call(a, 0, 33)
        b = self.banks.start(1, 1, "B")
        self.call(b, 0, 31)
        before = self.banks.inventory(fingerprint, exclude=0)
        error = InjectedFailure("physical layer22 loop1")
        hook = self.inject_after_output(error)
        try:
            with self.assertRaises(InjectedFailure) as caught:
                self.call(a, 33, 34)
        finally:
            hook.remove()
        self.assertIs(caught.exception, error)
        retained = self.retained[-1]
        self.assertIs(retained["error"], error)
        self.assertEqual(retained["completed"], 63)
        self.assertIsNone(retained["pending"])
        self.assertEqual(len(retained["values"]), 126)
        self.assertIn((22, 1, "residual_output"), retained["values"])
        self.assertNotIn((23, 1, "input"), retained["values"])
        self.assertEqual(sum(n == 34 for n in retained["lengths"].values()), 32)
        self.assertEqual(sum(n == 33 for n in retained["lengths"].values()), 24)
        expected = self.original_injection["observed"]["official_partial_lengths"]
        self.assertEqual({str(k): v for k, v in retained["lengths"].items()}, expected)
        self.assertEqual(expected, self.original_injection["observed"]["control_partial_lengths"])
        self.assertFalse(self.capture.last_call_succeeded)
        self.assertEqual(self.head_calls, 2)
        self.assertTrue(all(c.keys is None and c.values is None for c in self.caches.values()))
        self.assertEqual(before, self.banks.inventory(fingerprint, exclude=0))

    def test_original_saved_failure_location_and_projection_count_match_visit_prefix(self):
        original = self.original_injection
        self.assertEqual((original["event"], original["layer"], original["loop"]), (6194, 22, 1))
        self.assertEqual(original["original_event"]["sha256"],
                         "cf89237ec379e81de9f6a0537a8efcdf4e6e51f175619d787452ededbf66ca66")
        prefix = visits(self.model.backbone.config)[:63]
        self.assertEqual(prefix[-1], (22, 1))
        derived_calls = sum(3 if layer % 2 == 0 else 2 for layer, _ in prefix)
        self.assertEqual(derived_calls, original["observed"]["official_projection_calls"])
        self.assertEqual(derived_calls, 158)

    def test_fresh_epoch_after_failure_runs_all112_visits_from_empty_history(self):
        a = self.banks.start(0, 1, "A")
        self.call(a, 0, 33)
        hook = self.inject_after_output(InjectedFailure("native injection"))
        try:
            with self.assertRaises(InjectedFailure):
                self.call(a, 33, 34)
        finally:
            hook.remove()
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            self.call(a, 33, 34)
        b = self.banks.start(0, 2, "B")
        hidden, values = self.call(b, 0, 31)
        self.assertEqual(hidden.shape, (31, 2560))
        self.assertEqual(len(values), 224)
        self.assertEqual(self.capture.completed, 112)
        self.assertEqual(set(self.banks.histories(b)), set(CACHE_IDS))
        self.assertEqual(self.head_calls, 2)

    def test_archive_error_preserves_original_error_and_cleans_up_partial_native_state(self):
        a = self.banks.start(0, 1, "A")
        self.call(a, 0, 33)
        original = InjectedFailure("native injection")
        archive_error = OSError("diagnostic archive full")
        hook = self.inject_after_output(original)
        try:
            with self.assertRaises(FailureCaptureError) as caught:
                self.call(a, 33, 34, retention=archive_error)
        finally:
            hook.remove()
        self.assertIs(caught.exception.original, original)
        self.assertIs(caught.exception.retention, archive_error)
        self.assertEqual(len(self.capture.partial_values), 126)
        self.assertFalse(self.banks.inventory(fingerprint)["0"]["alive"])
        self.assertTrue(all(c.keys is None and c.values is None for c in self.caches.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
