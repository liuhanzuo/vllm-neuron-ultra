"""Prepared authenticated Dory event stream; no model or tensor imports.

Only an admitted future serial worker may consume the full 6200-event stream.
A partial source check does not produce completion or authorize a native run.
"""
import hashlib
import json
from pathlib import Path

from dory_residual_replay_history_v1 import ResidualReplayHistory, check_sha
from dory_residual_visit_capture_v1 import require


def read_metadata(path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
            and path.suffix in (".json", ".jsonl"), "Canonical original metadata path")
    before = path.stat()
    require(before.st_size <= 8 * 2**20, "Bounded original metadata record")
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(tuple(getattr(before, k) for k in fields) == tuple(getattr(after, k) for k in fields)
            and len(raw) == before.st_size, "Stable original metadata identity")
    require(hashlib.sha256(raw).hexdigest() == check_sha(expected),
            "Exact original metadata fingerprint")
    return raw


class OriginalEventStream:
    def __init__(self, selection_plan, guard):
        self.guard = guard
        guard()
        raw = read_metadata(selection_plan["path"], selection_plan["sha256"])
        plan = json.loads(raw)
        require(plan["schema"] == "dory_first5_propagation_diagnostic_plan_v1"
                and plan["original_event_count"] == 6200
                and plan["original_saved_call_count"] == 6185
                and plan["selected_rows"] == 153,
                "Pinned original event selection; first-five execution is not reused")
        self.selection_plan = dict(selection_plan)
        self.plan = plan
        prefix = plan["original_prefix_plan"]
        self.root = Path(prefix["source"])
        require(self.root.resolve() == self.root and self.root.is_absolute(),
                "Canonical original event directory")
        guard()
        index_raw = read_metadata(prefix["event_index"]["path"], prefix["event_index"]["sha256"])
        index = [json.loads(line) for line in index_raw.splitlines()]
        require(len(index) == 6200, "Complete original6200-event index")
        for i, row in enumerate(index):
            require(type(row["event"]) is int and row["event"] == i
                    and row["file"] == f"event-{i:06d}.json",
                    "Every indexed original event in order")
            check_sha(row["sha256"])
        self.index = index
        self.call_index = plan["original_call_index"]
        require(len(self.call_index) == 6185
                and len({row["event"] for row in self.call_index}) == 6185
                and [row["event"] for row in self.call_index] == sorted(row["event"] for row in self.call_index)
                and all(row == index[row["event"]] for row in self.call_index),
                "All original6185 calls bound to complete event index")
        selected = {}
        for desc in plan["reference_and_marked_calls"]:
            path = Path(desc["event"]["path"])
            event = int(path.stem.removeprefix("event-"))
            require(event not in selected and 0 <= event < len(index)
                    and path == self.root / index[event]["file"]
                    and desc["event"]["sha256"] == index[event]["sha256"],
                    "Unique original reference/mark bound to indexed event")
            selected[event] = desc
        require(len(selected) == 89, "Every original89 reference/marked call selected")
        self.history = ResidualReplayHistory(selected, prefix["reference_positions"])
        self.started = False
        self.completed_summary = None
        self.metadata_bytes_read = len(raw) + len(index_raw)
        self.events_read = self.calls_read = 0

    def __iter__(self):
        require(not self.started, "Original event stream is single-use; no partial restart")
        self.started = True
        for item in self.index:
            self.guard()
            path = self.root / item["file"]
            raw = read_metadata(path, item["sha256"])
            self.metadata_bytes_read += len(raw)
            require(self.metadata_bytes_read <= 1024 * 2**20, "Complete original metadata byte budget")
            row = json.loads(raw)
            require(row["event"] == item["event"], "Original indexed body event number")
            action = self.history.accept(row)
            action["original_event"] = dict(path=str(path), sha256=item["sha256"])
            if action["native_call_required"]:
                require(self.calls_read < len(self.call_index)
                        and item == self.call_index[self.calls_read],
                        "Every original successful call in indexed order")
                self.calls_read += 1
            self.events_read += 1
            yield action
        require(self.calls_read == 6185 and self.events_read == 6200,
                "Only complete original stream can finalize")
        summary = self.history.finish()
        self.guard()
        self.completed_summary = dict(
            summary, original_selection_plan=self.selection_plan,
            metadata_bytes_read=self.metadata_bytes_read,
            raw_tensor_bytes_read=0, native_admitted=False)
