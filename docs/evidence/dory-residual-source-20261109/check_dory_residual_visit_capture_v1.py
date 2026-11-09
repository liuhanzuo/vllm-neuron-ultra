"""Stdlib event tests only: no torch, native model or device execution."""
from types import SimpleNamespace
import unittest

from dory_residual_visit_capture_v1 import (
    CaptureResidualVisits, selected_raw_bytes, visits)


class Tensor:
    ndim = 2

    def __init__(self, rows=3, width=2560, marker=None):
        self.shape = (rows, width)
        self.marker = marker


class Handle:
    def __init__(self, collection, hook):
        self.collection, self.hook = collection, hook

    def remove(self):
        self.collection.remove(self.hook)


class Block:
    def __init__(self, index):
        self.index = index
        self.pre, self.post = [], []
        self.calls = []
        self.fail = False
        self.fail_on_loop = None

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs is True
        self.pre.append(hook)
        return Handle(self.pre, hook)

    def register_forward_hook(self, hook, *, with_kwargs):
        assert with_kwargs is True
        self.post.append(hook)
        return Handle(self.post, hook)

    def __call__(self, *args, **kwargs):
        for hook in self.pre:
            assert hook(self, args, kwargs) is None
        self.calls.append((args, kwargs))
        if self.fail or self.fail_on_loop == kwargs.get("loop_idx", 0):
            raise ValueError("original layer failure")
        result = Tensor(marker=(self.index, kwargs.get("loop_idx", 0)))
        for hook in self.post:
            assert hook(self, args, kwargs, result) is None
        return result


class Model:
    def __init__(self):
        config = SimpleNamespace(num_hidden_layers=72, n_input_layers=16,
                                 n_recurrent_layers=40, n_output_layers=16,
                                 n_recurrent_loops=2, hidden_size=2560)
        self.backbone = SimpleNamespace(config=config, num_loops=2,
                                        layers=[Block(i) for i in range(72)])
        self.schedule = list(visits(config))
        self.final = None
        self.kwargs_mutation = None

    def __call__(self, tokens, positions):
        hidden = Tensor(marker="embedding")
        for physical, loop in self.schedule:
            kwargs = {"loop_idx": loop} if 16 <= physical < 56 else {}
            if self.kwargs_mutation:
                kwargs = self.kwargs_mutation(physical, loop, kwargs)
            hidden = self.backbone.layers[physical](positions, hidden, **kwargs)
        self.final = hidden
        return hidden


def copy_selected(tensor, selectors):
    return Tensor(rows=len(selectors), marker=tensor.marker)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.capture = CaptureResidualVisits(self.model, copy_selected, lambda: None)
        self.capture.install()

    def tearDown(self):
        self.capture.close()

    def test_two_recurrent_visits_have_distinct_keys_and_original_output(self):
        hidden, values = self.capture.run(None, None, [0, 2])
        self.assertIs(hidden, self.model.final)
        self.assertEqual(len(values), 224)
        self.assertEqual(len(self.model.backbone.layers[16].calls), 2)
        self.assertIn((16, 0, "input"), values)
        self.assertIn((16, 1, "input"), values)
        self.assertTrue(self.capture.last_call_succeeded)
        self.assertEqual(values[16, 0, "input"].shape, (2, 2560))

    def test_recurrent_transition_inputs_are_observed(self):
        self.capture.run(None, None, [2])
        order = self.capture.order
        self.assertEqual(order[31:34], [
            (15, 0, "residual_output"), (16, 0, "input"),
            (16, 0, "residual_output")])
        self.assertEqual(order[111:114], [
            (55, 0, "residual_output"), (16, 1, "input"),
            (16, 1, "residual_output")])
        self.assertEqual(order[191:194], [
            (55, 1, "residual_output"), (56, 0, "input"),
            (56, 0, "residual_output")])

    def test_repeat_requests_and_empty_selection_do_not_skip_forward(self):
        for selectors in ([0, 2], [2], []):
            _, values = self.capture.run(None, None, selectors)
            self.assertEqual(len(values), 224)
            self.assertEqual(values[0, 0, "input"].shape[0], len(selectors))
        self.assertEqual(len(self.model.backbone.layers[16].calls), 6)

    def test_missing_loop_is_rejected(self):
        self.model.schedule = [visit for visit in self.model.schedule if visit[1] == 0]
        with self.assertRaisesRegex(RuntimeError, "execution order"):
            self.capture.run(None, None, [0])
        self.assertFalse(self.capture.last_call_succeeded)

    def test_wrong_loop_keyword_is_rejected(self):
        self.model.kwargs_mutation = lambda physical, loop, kwargs: (
            {"loop_idx": 0} if loop == 1 else kwargs)
        with self.assertRaisesRegex(RuntimeError, "execution order"):
            self.capture.run(None, None, [0])

    def test_reordered_physical_layers_are_rejected(self):
        self.model.schedule[5:7] = self.model.schedule[5:7][::-1]
        with self.assertRaisesRegex(RuntimeError, "execution order"):
            self.capture.run(None, None, [0])

    def test_extra_visit_is_rejected(self):
        self.model.schedule.append((71, 0))
        with self.assertRaisesRegex(RuntimeError, "extra"):
            self.capture.run(None, None, [0])

    def test_missing_final_visit_is_rejected(self):
        self.model.schedule.pop()
        with self.assertRaisesRegex(RuntimeError, "224 boundaries"):
            self.capture.run(None, None, [0])

    def test_original_failure_propagates_with_partial_evidence(self):
        self.model.backbone.layers[22].fail = True
        with self.assertRaisesRegex(ValueError, "original layer failure"):
            self.capture.run(None, None, [0])
        self.assertFalse(self.capture.active)
        self.assertFalse(self.capture.last_call_succeeded)
        self.assertEqual(self.capture.completed, 22)
        self.assertEqual(self.capture.pending_visit, (22, 0))
        self.assertIn((22, 0, "input"), self.capture.partial_values)
        self.assertNotIn((22, 0, "residual_output"), self.capture.partial_values)
        with self.assertRaises(TypeError):
            self.capture.partial_values[(22, 0, "input")] = None

    def test_original_second_loop_layer22_failure_keeps_both_visit_identities(self):
        self.model.backbone.layers[22].fail_on_loop = 1
        with self.assertRaisesRegex(ValueError, "original layer failure"):
            self.capture.run(None, None, [0])
        self.assertEqual(self.capture.completed, 62)
        self.assertEqual(self.capture.pending_visit, (22, 1))
        self.assertIn((22, 0, "residual_output"), self.capture.partial_values)
        self.assertIn((22, 1, "input"), self.capture.partial_values)
        self.assertNotIn((22, 1, "residual_output"), self.capture.partial_values)

    def test_duplicate_physical_objects_are_rejected(self):
        self.capture.close()
        self.model.backbone.layers[71] = self.model.backbone.layers[70]
        with self.assertRaisesRegex(RuntimeError, "Distinct"):
            self.capture.install()
        self.assertEqual(self.capture.handles, [])

    def test_partial_installation_failure_cleans_up_registered_hooks(self):
        self.capture.close()
        def fail(*args, **kwargs):
            raise ValueError("registration failed")
        self.model.backbone.layers[3].register_forward_hook = fail
        with self.assertRaisesRegex(ValueError, "registration failed"):
            self.capture.install()
        self.assertEqual(self.capture.handles, [])
        self.assertTrue(all(not block.pre and not block.post
                            for block in self.model.backbone.layers))

    def test_malformed_selected_copy_is_rejected(self):
        self.capture.select_copy = lambda tensor, selectors: Tensor(rows=0)
        with self.assertRaisesRegex(RuntimeError, "every selected"):
            self.capture.run(None, None, [0])

    def test_architecture_change_is_rejected_before_model_call(self):
        self.model.backbone.config.n_recurrent_loops = 1
        with self.assertRaisesRegex(RuntimeError, "architecture"):
            self.capture.run(None, None, [0])
        self.assertEqual(self.model.backbone.layers[0].calls, [])

    def test_hook_cleanup_leaves_original_forward_available(self):
        self.capture.close()
        self.model(None, None)
        self.assertTrue(all(not b.pre and not b.post for b in self.model.backbone.layers))

    def test_boolean_loop_identity_is_rejected(self):
        self.model.kwargs_mutation = lambda physical, loop, kwargs: {"loop_idx": False}
        with self.assertRaisesRegex(RuntimeError, "Integer"):
            self.capture.run(None, None, [0])

    def test_raw_capture_size_is_bounded_and_not_a_runtime_reservation(self):
        self.assertEqual(selected_raw_bytes(153), 175472640)
        self.assertEqual(selected_raw_bytes(0), 0)
        with self.assertRaises(RuntimeError):
            selected_raw_bytes(-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
