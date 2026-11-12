"""Stdlib-only source/mock checks. No torch, numpy, NKI or device execution."""
import ast
import builtins
import collections
import copy
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import qwen35_predicated_slot_protocol_v1 as p
import qwen35_predicated_slot_device_leaf_v1 as leaf


class SourceChecks(unittest.TestCase):
    def test_exact_actual_layouts(self):
        expected = [("conv", "bfloat16", [4, 3072, 4], 98304),
                    ("recurrent", "float32", [4, 8, 128, 128], 2097152)]
        expected += [(f"kv{n}", "bfloat16", [4, 1, n, 256], 2048 * n) for n in p.BUCKETS]
        actual = [p.layout(row[0]) for row in p.LAYOUTS]
        self.assertEqual([(x["name"], x["dtype"], x["shape"], x["tensor_bytes"]) for x in actual], expected)
        self.assertTrue(all(x["rows"] == 4 and x["per_slot_width"] % 128 == 0 for x in actual))
        with self.assertRaises(ValueError):
            p.layout("kv64")

    def test_full_cartesian_correctness(self):
        maps = set(p.CORRECTNESS_MAPPINGS)
        self.assertEqual(len(p.CORRECTNESS_MAPPINGS), 625)
        self.assertEqual(maps, set(itertools.product((-1, 0, 1, 2, 3), repeat=4)))
        self.assertTrue(set(itertools.permutations(range(4))) <= maps)
        self.assertEqual(len(set(itertools.permutations(range(4)))), 24)
        self.assertIn((-1, -1, -1, -1), maps)
        self.assertTrue(all(tuple([slot] * 4) in maps for slot in range(4)))

    def test_timing_maps_are_explicit_and_separate(self):
        self.assertEqual(len(p.TIMING_MAPPINGS), 16)
        self.assertEqual(len(set(p.TIMING_MAPPINGS)), 16)
        self.assertTrue(set(p.TIMING_MAPPINGS) < set(p.CORRECTNESS_MAPPINGS))

    def test_schedule_covers_both_profiles_before_timing(self):
        schedule = list(p.sequences())
        self.assertEqual(len(schedule), 14)
        self.assertEqual([(x["phase"], x["profile"]) for x in schedule[:2]],
                         [("correctness", "raw"), ("correctness", "finite")])
        self.assertEqual([x["phase"] for x in schedule[2:4]], ["warmup"] * 2)
        self.assertEqual([x["phase"] for x in schedule[4:]], ["measured"] * 10)
        for sequence in schedule[:2]:
            self.assertEqual(sequence["mappings"], p.CORRECTNESS_MAPPINGS)

    def test_mock_schedule_pairs_identical_inputs_and_exact_counts(self):
        # Consume the same scientific schedule as the native leaf, without
        # native imports. This checks coverage/order, not kernel correctness.
        counts = collections.Counter()
        for mode in p.MODES:
            for sequence in p.sequences():
                arm_inputs = {}
                for variant in sequence["order"]:
                    if mode == "persistent_alias":
                        counts["reset"] += 1
                    arm_inputs[variant] = []
                    for step, mapping in enumerate(sequence["mappings"]):
                        counts[sequence["phase"]] += 1
                        arm_inputs[variant].append((sequence["profile"], sequence["initial_salt"],
                                                    sequence["update_salt_base"] + step, mapping))
                self.assertEqual(arm_inputs["reference"], arm_inputs["candidate"])
        self.assertEqual(dict(counts), {"correctness": 5000, "warmup": 128, "measured": 640, "reset": 28})
        self.assertEqual(p.counts()["per_layout"]["total_runtime_calls"], 5796)
        self.assertEqual(p.counts()["total"], {"graphs": 32, "correctness_calls": 40000,
                         "warmup_calls": 1024, "measured_calls": 5120, "reset_calls": 224,
                         "total_runtime_calls": 46368})

    def test_balanced_paired_order(self):
        orders = [p.paired_order(i) for i in range(p.MEASURED_CYCLES)]
        self.assertEqual(orders.count(("reference", "candidate")), 5)
        self.assertEqual(orders.count(("candidate", "reference")), 5)
        for invalid in (-1, True, "0"):
            with self.assertRaises(ValueError):
                p.paired_order(invalid)

    def test_timer_contains_only_invocation_and_output_materialization(self):
        events = []
        times = iter((100, 175))
        def clock():
            events.append("clock")
            return next(times)
        def invoke():
            events.append("invoke")
            return "pending"
        def materialize(value):
            self.assertEqual(value, "pending")
            events.append("all_outputs_to_host")
            return "complete"
        events.append("guard_oracle_inputs")
        output, elapsed = p.measured_call(invoke, materialize, clock)
        events.append("comparison_state_read_save")
        self.assertEqual(events, ["guard_oracle_inputs", "clock", "invoke", "all_outputs_to_host",
                                  "clock", "comparison_state_read_save"])
        self.assertEqual((output, elapsed), ("complete", 75))
        with self.assertRaises(RuntimeError):
            p.measured_call(lambda: None, lambda x: x, lambda: 0)

    def test_timer_does_not_hide_materialization_failure(self):
        def failed(_):
            raise LookupError("not synchronized")
        with self.assertRaises(LookupError):
            p.measured_call(lambda: object(), failed, lambda: 1)

    def test_full_byte_comparator_special_values(self):
        for zero, negzero, nan1, nan2 in ((b"\0\0", b"\0\x80", b"\xc1\x7f", b"\xc2\x7f"),
                                         (b"\0" * 4, b"\0\0\0\x80", b"\x01\0\xc0\x7f", b"\x02\0\xc0\x7f")):
            self.assertTrue(p.compare_storage((1,), "same", nan1, (1,), "same", nan1)["passed"])
            for a, b in ((zero, negzero), (nan1, nan2)):
                self.assertFalse(p.compare_storage((1,), "same", a, (1,), "same", b)["passed"])

    def test_full_byte_comparator_checks_tail_shape_dtype_and_length(self):
        original = b"x" * 2**20
        corrupt = original[:-1] + b"z"
        check = p.compare_storage((len(original),), "u8", corrupt, (len(original),), "u8", original)
        self.assertFalse(check["passed"])
        self.assertEqual(check["first_different_bytes"], [len(original) - 1])
        for shape, dtype, data in (((2,), "u8", original), ((1,), "i8", original), ((1,), "u8", original[:-1])):
            self.assertFalse(p.compare_storage(shape, dtype, data, (1,), "u8", original)["passed"])

    def test_pair_summary_requires_all_original_samples(self):
        rows = [{"round": i, "order": p.paired_order(i),
                 "elapsed_ns": {"reference": [20] * 16, "candidate": [10] * 16}}
                for i in range(10)]
        self.assertEqual(p.summarize_pairs(rows)["median_paired_ratio"], 2)
        self.assertFalse(p.summarize_pairs(rows)["model_speedup_claim"])
        with self.assertRaises(ValueError):
            p.summarize_pairs(rows[:-1])
        for mutation in ("order", "sample", "zero"):
            changed = copy.deepcopy(rows)
            if mutation == "order":
                changed[1]["order"] = p.VARIANTS
            elif mutation == "sample":
                changed[0]["elapsed_ns"]["candidate"].pop()
            else:
                changed[0]["elapsed_ns"]["candidate"][0] = 0
            with self.assertRaises(ValueError):
                p.summarize_pairs(changed)

    def test_original_storage_floors_preserved(self):
        r = p.RESOURCES
        self.assertEqual(r["root_floor_bytes"], 28 * 2**30)
        self.assertEqual(r["nvme_floor_bytes"], 30 * 2**30)
        self.assertEqual(r["full_workload_initial_free_bytes"],
                         r["output_volume_existing_free_reserve_bytes"] + r["output_cap_bytes_total"]
                         + r["additional_headroom_bytes"])
        self.assertEqual(r["full_workload_initial_free_bytes"], 40 * 2**30)
        self.assertEqual(r["lfm_separate_required_free_bytes_unchanged"], 54 * 2**30)
        self.assertFalse(p.workload()["compatible_current_serial_dispatcher"])

    def test_frozen_sources_are_current(self):
        self.assertEqual(p.verify_sources(), p.FROZEN_SOURCES)

    def test_no_native_module_scope_imports_or_cli(self):
        native = {"torch", "torch_neuronx", "numpy", "ml_dtypes", "neuronxcc"}
        for path in (Path(p.__file__), Path(leaf.__file__)):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    self.assertFalse({x.name.split(".")[0] for x in node.names} & native)
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module.split(".")[0], native)
                self.assertFalse(isinstance(node, ast.If) and "__main__" in ast.unparse(node.test))
            self.assertNotIn("subprocess", path.read_text())

    def test_guard_refusal_precedes_source_reads_and_native_imports(self):
        def refuse(_):
            raise PermissionError("mock owner not held")
        with mock.patch.object(p, "verify_sources", side_effect=AssertionError("too early")):
            with self.assertRaises(PermissionError):
                leaf.run_case("conv", "/not-created", guard=refuse,
                              save_report=lambda _: self.fail("unexpected report"),
                              record_artifact=lambda _: self.fail("unexpected artifact"))

    def test_native_boundary_refusal_saves_failed_report_without_native_import(self):
        events, reports = [], []
        native = ("torch", "numpy", "neuronxcc", "ml_dtypes")
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(native):
                raise AssertionError("native import during source checks")
            return original_import(name, *args, **kwargs)
        def guard(event):
            events.append(event)
            if event == "before_native_imports":
                raise PermissionError("mock native boundary closed")
        with tempfile.TemporaryDirectory(prefix="qwen-source-mock-") as temporary:
            with mock.patch.object(p, "verify_sources", return_value={}), mock.patch("builtins.__import__", guarded_import):
                with self.assertRaises(PermissionError):
                    leaf.run_case("conv", Path(temporary) / "mock-output", guard=guard,
                                  save_report=lambda x: reports.append(copy.deepcopy(x)),
                                  record_artifact=lambda _: self.fail("unexpected artifact"))
        self.assertEqual(events, ["before_source_verification", "before_output_creation", "before_native_imports"])
        self.assertFalse(reports[-1]["complete"])
        self.assertFalse(reports[-1]["passed"])
        self.assertIn("mock native boundary closed", reports[-1]["error"])


if __name__ == "__main__":
    unittest.main()
