"""Raw binary evidence preservation checks, with stdlib byte fixtures."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from dory_residual_output_archive_v1 import ObservationArchive, original_archive_comparison


def tensor(payload, shape, dtype="BF16", finite=True):
    return dict(payload=payload, shape=shape, dtype=dtype, finite=finite)


class Archives(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.archive = ObservationArchive(
            self.root, lambda t: t["payload"],
            lambda t: {k: t[k] for k in ("shape", "dtype", "finite")}, lambda: None, 4096)

    def tearDown(self):
        self.temp.cleanup()

    def test_bf16_words_and_f32_outputs_are_preserved_without_conversion(self):
        bf16 = struct.pack("<HHHH", 0x3f80, 0x8000, 0x7fc1, 0x0001)
        fp32 = struct.pack("<ff", 1.25, -3.5)
        row = self.archive.write("event-000001-residual.bin", {
            "layer-0-loop-0-input": tensor(bf16, [1, 4], finite=False),
            "logits": tensor(fp32, [1, 2], dtype="F32")})
        actual = (self.root / row["file"]).read_bytes()
        self.assertEqual(struct.unpack("<HHHH", actual[:8]), (0x3f80, 0x8000, 0x7fc1, 0x0001))
        self.assertEqual(struct.unpack("<ff", actual[8:]), (1.25, -3.5))
        self.assertEqual(row["tensors"]["logits"]["offset"], 8)
        self.assertFalse(row["tensors"]["layer-0-loop-0-input"]["finite"])
        self.assertEqual(row["sha256"], hashlib.sha256(bf16+fp32).hexdigest())

    def test_limit_failure_retains_previous_tensor_bytes_and_descriptor(self):
        self.archive.byte_limit = 6
        with self.assertRaisesRegex(RuntimeError, "budget"):
            self.archive.write("failure.bin", {"first": tensor(b"abcd", [2]),
                                                "second": tensor(b"efgh", [2])})
        self.assertEqual((self.root / "failure.bin").read_bytes(), b"abcd")
        self.assertTrue(self.archive.failed)
        self.assertFalse(self.archive.pending["complete"])
        self.assertEqual(set(self.archive.pending["tensors"]), {"first"})
        self.assertEqual(self.archive.bytes_written, 4)
        with self.assertRaisesRegex(RuntimeError, "cannot be resumed"):
            self.archive.write("retry.bin", {})
        self.assertFalse((self.root / "retry.bin").exists())

    def test_raw_callback_failure_keeps_completed_prefix(self):
        def raw(value):
            if value["payload"] is None:
                raise ValueError("source tensor unavailable")
            return value["payload"]
        self.archive.raw = raw
        with self.assertRaisesRegex(ValueError, "source tensor"):
            self.archive.write("failure.bin", {"first": tensor(b"abcd", [2]),
                                                "second": tensor(None, [1])})
        self.assertEqual((self.root / "failure.bin").read_bytes(), b"abcd")
        self.assertEqual(self.archive.pending["completed_tensor_prefix_sha256"],
                         hashlib.sha256(b"abcd").hexdigest())

    def test_incorrect_payload_extent_is_rejected_before_tensor_write(self):
        with self.assertRaisesRegex(RuntimeError, "extent"):
            self.archive.write("wrong.bin", {"bad": tensor(b"ab", [2])})
        self.assertEqual((self.root / "wrong.bin").read_bytes(), b"")
        self.assertEqual(self.archive.pending["tensors"], {})

    def test_duplicate_filename_does_not_overwrite_previous_evidence(self):
        row = self.archive.write("once.bin", {"a": tensor(b"ab", [1])})
        with self.assertRaises(FileExistsError):
            self.archive.write("once.bin", {"a": tensor(b"cd", [1])})
        self.assertEqual((self.root / "once.bin").read_bytes(), b"ab")
        self.assertEqual(self.archive.bytes_written, 2)
        self.assertIsNone(self.archive.pending)
        self.assertTrue(row["complete"])

    def test_close_failure_cannot_mark_an_archive_complete(self):
        original_open = Path.open
        class CloseFailure:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self.stream
            def __exit__(self, *args):
                self.stream.close()
                raise OSError("archive close failed")
        def open_with_failure(path, *args, **kwargs):
            return CloseFailure(original_open(path, *args, **kwargs))
        with patch.object(Path, "open", open_with_failure):
            with self.assertRaisesRegex(OSError, "close failed"):
                self.archive.write("close-failure.bin", {"first": tensor(b"abcd", [2])})
        self.assertEqual((self.root / "close-failure.bin").read_bytes(), b"abcd")
        self.assertTrue(self.archive.failed)
        self.assertFalse(self.archive.pending["complete"])
        self.assertEqual(self.archive.completed, [])

    def test_zero_selected_rows_keep_their_explicit_descriptor(self):
        row = self.archive.write("empty-selection.bin", {"input": tensor(b"", [0, 2560])})
        self.assertEqual(row["tensors"]["input"]["shape"], [0, 2560])
        self.assertEqual(row["bytes"], 0)
        self.assertEqual((self.root / row["file"]).read_bytes(), b"")

    def test_archive_basename_cannot_escape_output_directory(self):
        for name in ("../escape.bin", "/tmp/escape.bin", "nested/file.bin"):
            with self.assertRaisesRegex(RuntimeError, "basename"):
                self.archive.write(name, {})

    def test_late_loop_tensor_change_is_not_hidden_by_equal_header(self):
        expected = {"file": "a.bin", "sha256": "same", "tensors": {
            "cache-0-key": {"sha256": "a"}, "cache-126-value": {"sha256": "b"}}}
        actual = copy.deepcopy(expected)
        actual["tensors"]["cache-126-value"]["sha256"] = "c"
        result = original_archive_comparison(actual, expected)
        self.assertFalse(result["exact"])
        self.assertEqual(result["header_differences"], [])
        self.assertEqual(result["changed_tensors"], ["cache-126-value"])

    def test_missing_and_extra_tensor_records_are_both_reported(self):
        result = original_archive_comparison(
            {"tensors": {"other": {}}, "bytes": 4},
            {"tensors": {"hidden": {}}, "bytes": 8})
        self.assertEqual(result["missing_tensors"], ["hidden"])
        self.assertEqual(result["extra_tensors"], ["other"])
        self.assertEqual(result["header_differences"], ["bytes"])
        self.assertFalse(result["independent_raw_review_complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
