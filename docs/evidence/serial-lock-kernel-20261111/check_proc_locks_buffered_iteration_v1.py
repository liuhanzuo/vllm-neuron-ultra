"""Source-directed in-memory seq cursor model; never reads or acquires locks.

The RawIOBase is a model of the relevant seq_read_iter position semantics,
not an execution of the kernel. TextIOWrapper and BufferedReader are real
stdlib objects. This demonstrates possibility, not the historical syscall
boundaries or concurrent mutations of Dory r3.
"""
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest

CAP = 2 * 2**20 + 1
A = "FLOCK  ADVISORY  WRITE 1298998 103:02:8294421 0 EOF"
B = "FLOCK  ADVISORY  WRITE 1298594 103:02:9303865 0 EOF"
BASE = [f"POSIX  ADVISORY  WRITE {2000+i} 103:02:{10000+i} 0 EOF" for i in range(27)] + [A, B]


class SeqCursorRaw(io.RawIOBase):
    def __init__(self, insertions=()):
        super().__init__()
        self.rows = list(BASE)
        self.index = 0
        self.insertions = list(insertions)
        self.calls = []

    def readable(self):
        return True

    def readinto(self, target):
        # Model list changes between raw reads, after the first iterator stop.
        if len(self.calls) == 1:
            self.rows[:0] = self.insertions
        start = self.index
        result = bytearray()
        while self.index < len(self.rows):
            row = f"{self.index+1}: {self.rows[self.index]}\n".encode()
            # This fixture fits both the modeled page and caller buffer.
            if len(result) + len(row) > min(4096, len(target)):
                break
            result.extend(row)
            self.index += 1
        target[:len(result)] = result
        self.calls.append(dict(requested=len(target), returned=len(result),
                               start_index=start, end_index=self.index,
                               table_rows=len(self.rows)))
        return len(result)


def read_fixture(insertions=()):
    raw = SeqCursorRaw(insertions)
    with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8") as stream:
        text = stream.read(CAP)
    return text, raw.calls


class Checks(unittest.TestCase):
    def test_unchanged_table(self):
        text, calls = read_fixture()
        self.assertEqual(len(text.splitlines()), 29)
        self.assertEqual([c['end_index'] for c in calls], [29, 29])
        self.assertEqual(calls[-1]['returned'], 0)

    def test_two_insertions_repeat_ab(self):
        text, calls = read_fixture(['NEW ONE', 'NEW TWO'])
        self.assertEqual(text.splitlines()[-4:], [f'28: {A}', f'29: {B}', f'30: {A}', f'31: {B}'])
        self.assertEqual([c['end_index'] for c in calls], [29, 31, 31])
        self.assertLess(len(text.encode()), 4096)

    def test_one_insertion_repeats_only_last_record(self):
        text, calls = read_fixture(['NEW ONE'])
        self.assertEqual(text.splitlines()[-3:], [f'28: {A}', f'29: {B}', f'30: {B}'])
        self.assertEqual(len(calls), 3)

    def test_no_duplicate_lock_in_any_table(self):
        raw = SeqCursorRaw(['NEW ONE', 'NEW TWO'])
        for _ in range(3):
            raw.readinto(bytearray(CAP))
            self.assertEqual(raw.rows.count(A), 1)
            self.assertEqual(raw.rows.count(B), 1)

    def test_one_raw_read_has_no_duplicate(self):
        raw = SeqCursorRaw(['NEW ONE', 'NEW TWO'])
        target = bytearray(CAP)
        n = raw.readinto(target)
        self.assertEqual(bytes(target[:n]).decode().splitlines()[-2:], [f'28: {A}', f'29: {B}'])
        self.assertEqual(len(raw.calls), 1)

    def test_original_unique_match_condition_rejects(self):
        text, _ = read_fixture(['NEW ONE', 'NEW TWO'])
        matches = [line.split() for line in text.splitlines()
                   if line.split()[1:] == A.split()]
        self.assertEqual(len(matches), 2)
        self.assertFalse(len(matches) == 1 and matches[0][6:] == ['0', 'EOF'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    text, calls = read_fixture(['NEW ONE', 'NEW TWO'])
    print(json.dumps(dict(schema='proc_locks_buffered_iteration_model_v1',
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        passed=result.wasSuccessful(), tests_run=result.testsRun,
        in_memory_only=True, kernel_execution=False, historical_cause_proven=False,
        actual_proc_locks_read=False, raw_calls=calls,
        output_bytes=len(text.encode()), final_four_rows=text.splitlines()[-4:]), indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
