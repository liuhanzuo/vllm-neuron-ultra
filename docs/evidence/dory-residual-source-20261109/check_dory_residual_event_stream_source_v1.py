"""Bounded source checks against retained host metadata; no native execution."""
from pathlib import Path
import unittest

from dory_residual_event_stream_v1 import OriginalEventStream


Q = Path("/home/ubuntu/experiments/trn1-latest-models-20261031/research")
PLAN = dict(path=str(Q / "dory-first5-propagation-plan-20261109-v1.json"),
            sha256="a6ff5ba1f8c237e62d536725f9b19eb374a69d9ab4b516cbfdf52fd891be7ecf")


class Stream(unittest.TestCase):
    def test_first_six_original_events_do_not_qualify_partial_stream(self):
        stream = OriginalEventStream(PLAN, lambda: None)
        iterator = iter(stream)
        actions = [next(iterator) for _ in range(6)]
        iterator.close()
        self.assertEqual([row["kind"] for row in actions],
                         ["start", "call", "release", "start", "call", "release"])
        self.assertEqual((stream.events_read, stream.calls_read), (6, 2))
        self.assertIsNone(stream.completed_summary)
        self.assertEqual(len(actions[1]["original_archive"]["tensors"]), 114)
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            next(iter(stream))

    def test_complete_index_and_selection_are_bound_without_claiming_body_validation(self):
        stream = OriginalEventStream(PLAN, lambda: None)
        self.assertEqual((len(stream.index), len(stream.call_index), len(stream.history.selected)),
                         (6200, 6185, 89))
        self.assertEqual(stream.events_read, 0)
        self.assertIsNone(stream.completed_summary)

    def test_changed_plan_hash_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            OriginalEventStream(dict(PLAN, sha256="0"*64), lambda: None)

    def test_guard_failure_prevents_source_read(self):
        def guard():
            raise RuntimeError("current serial owner unavailable")
        with self.assertRaisesRegex(RuntimeError, "serial owner"):
            OriginalEventStream(dict(path="/missing.json", sha256="0"*64), guard)


if __name__ == "__main__":
    unittest.main(verbosity=2)
