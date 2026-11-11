"""Bounded, stdlib-only synthetic checks; never open recorded tensor payloads.

The frame fixture uses a TEST-ONLY bucket of 4, which the real contract rejects.
Original dimensions, two ranks, physical rows and all leaf names are retained.
Full six-call serialization checks construct metadata only, not raw captures.
"""
import copy
import importlib.abc
import json
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RejectNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "torch_neuronx", "neuronx_distributed"}:
            raise AssertionError("synthetic source test must not import native/model dependencies: " + fullname)


sys.meta_path.insert(0, RejectNativeImports())
from research import minicpm5_layer1_saved_operator_comparison_v1 as core


def guard():
    pass


def bf(value):
    packed = struct.pack("<f", value)
    assert packed[:2] == b"\0\0", "fixture must be exactly BF16 representable"
    return packed[2:]


def fixture():
    bucket, length, slot = 4, 3, 3
    events = core.native.event_manifest(length)
    ns = core.native_specs(events)
    ss = copy.deepcopy(core.sdk_specs(128))
    for (label, path), spec in ss.items():
        if label == "effective_prior_kv" or (label == "route_inputs_sdk_decode" and path[0] == "past_key_value"):
            spec["shape"][2] = bucket
        elif spec["dtype"] == "torch.bool" or (spec["selection"] is not None and spec["shape"][-1] == 128):
            spec["shape"][-1] = bucket
        spec["bytes"] = math.prod(spec["shape"]) * core.WIDTHS[spec["dtype"]]
    nv = {key: bytes(spec["bytes"]) for key, spec in ns.items()}
    sv = {r: {key: bytes(spec["bytes"]) for key, spec in ss.items()} for r in (0, 1)}
    def put(rank, label, value, path=()):
        key = label, path
        raw = sv[rank][key]
        rowbytes = len(raw) // 4
        assert len(value) == rowbytes
        sv[rank][key] = raw[:slot * rowbytes] + value + raw[(slot + 1) * rowbytes:]
    positions = struct.pack("<q", length - 1)
    nv["position_ids", ()] = nv["rope_position_ids", ()] = positions
    q = b"".join(bf(h + 1) * 128 for h in range(16))
    kv = bf(19) * 128 + bf(23) * 128
    for label in ("q_projection", "q_rope_input", "q_post_rope", "pv_context", "context_before_o"):
        nv[label, ()] = q
    for label in ("k_projection", "v_projection", "k_rope_input", "k_post_rope", "v_heads", "cache_update_active_key"):
        nv[label, ()] = kv
    effective = b"".join(bytes((length - 1) * 128 * 2) + kv[r * 256:(r + 1) * 256] for r in (0, 1))
    for index, name in enumerate(("key", "value")):
        nv["native_effective_" + name, ()] = nv["cache_after", (index,)] = effective
    scores = b"".join(bf(h * 8 + k + 1) for h in range(16) for k in range(length))
    for label in ("qk0_unscaled", "qk0_scaled", "softmax_input", "probabilities_bf16"):
        nv[label, ()] = scores
    nv["probabilities_fp32", ()] = core.bf16_to_fp32_bits(scores)
    for rank in (0, 1):
        for label in ("position_ids", "rope_position_ids"):
            put(rank, label, positions)
        put(rank, "route_inputs_sdk_decode", positions, ("position_ids",))
        mask = b"\1" * (length - 1) + bytes(bucket - length + 1)
        for label in ("attention_mask", "prior_actual_mask"):
            put(rank, label, mask)
        put(rank, "route_inputs_sdk_decode", mask, ("attention_mask",))
        qr, kr = q[rank * 2048:(rank + 1) * 2048], kv[rank * 256:(rank + 1) * 256]
        for label in ("q_projection", "q_rope_input", "q_post_rope", "q_after_prep", "context_before_o"):
            put(rank, label, qr)
        for label in ("k_projection", "v_projection", "k_rope_input", "k_post_rope", "v_heads", "k_after_prep", "v_after_prep"):
            put(rank, label, kr)
        for short, value in (("Q", qr), ("K", kr), ("V", kr)):
            put(rank, "route_inputs_sdk_decode", value, (short,))
        for i in (0, 1): put(rank, "returned_current_kv", kr, (i,))
        prior = b"".join(b"".join(bf(h * 8 + k + 1) for k in range(length - 1))
                         + bytes((bucket - length + 1) * 2) for h in range(rank * 8, (rank + 1) * 8))
        active = b"".join(bf(h * 8 + length) for h in range(rank * 8, (rank + 1) * 8))
        for label, active_label, fp32 in (
            ("qk0_unscaled", "qk1_unscaled", False), ("qk0_scaled", "qk1_scaled", False),
            ("softmax_input", "softmax_active_input", True),
            ("probabilities_fp32", "probabilities_active_fp32", True),
            ("probabilities_bf16", "probabilities_active_bf16", False)):
            put(rank, label, core.bf16_to_fp32_bits(prior) if fp32 else prior)
            put(rank, active_label, core.bf16_to_fp32_bits(active) if fp32 else active)
        put(rank, "prior_mask_input", prior)
        put(rank, "prior_mask_output", prior)
    case = {"frame": {"index": 4, "fixture": 0, "slot": slot, "step": 1}, "call": 4,
            "sdk_physical_row": slot, "length": length, "bucket": bucket, "native_events": events}
    return case, nv, sv, ss


class SyntheticChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q = ROOT / "research"
        cls.native_manifest = json.loads((q / "minicpm5-layer1-operator-native-r3-sourceclosure-20261111-manifest.json").read_text())
        cls.sdk_manifest = json.loads((q / "minicpm5-layer1-operator-sdk-event-manifest-20261111-v1.json").read_text())
        cls.contract = core.contract(cls.native_manifest, cls.sdk_manifest)
        cls.case, cls.nv, cls.sv, cls.ss = fixture()
        cls.fixture_bytes = sum(map(len, cls.nv.values())) + sum(len(raw) for rank in cls.sv.values() for raw in rank.values())
        assert cls.fixture_bytes < 512 * 1024
        cls.measurements = {"synthetic_payload_bytes": cls.fixture_bytes,
                            "largest_synthetic_leaf_bytes": max(len(raw) for rank in cls.sv.values() for raw in rank.values()),
                            "synthetic_bucket": 4, "recorded_payload_bytes_read": 0,
                            "whole_capture_77mb_fixture_constructed": False,
                            "real_contract_maximum_call_input_bytes": cls.contract["maximum_call_input_bytes"]}

    def run_frame(self, nv=None, sv=None, case=None):
        with patch.object(core, "sdk_specs", return_value=self.ss):
            return core.compare_frame(case or self.case, nv or self.nv, sv or self.sv, guard)

    def test_01_original_contract_full_topology(self):
        c = self.contract
        self.assertEqual(c["total_capture_bytes"], 77790564)
        self.assertEqual(c["native_capture_files"] + c["sdk_capture_files"], 1284)
        self.assertEqual([r["frame"]["index"] for r in c["frames"]], list(core.INDICES))
        group = [r for r in c["frames"] if r["call"] == 4]
        self.assertEqual([r["frame"]["index"] for r in sorted(group, key=lambda r: r["sdk_physical_row"])], [6, 5, 7, 4])
        for key, value in (("call", 5), ("cache_length", 24)):
            changed = copy.deepcopy(self.native_manifest); changed["frames"][0][key] = value
            with self.assertRaises(RuntimeError): core.contract(changed, self.sdk_manifest)
        for key, value in (("slot", 0), ("fixture", 3), ("index", 0)):
            changed = copy.deepcopy(self.native_manifest); changed["frames"][0]["frame"][key] = value
            with self.assertRaises(RuntimeError): core.contract(changed, self.sdk_manifest)

    def test_02_packed_addressing_and_bounds(self):
        raw = b"".join(struct.pack("<q", i) for i in range(24))
        selected = core.packed_region(raw, [2, 3, 4], "torch.int64", [(1, 2), (0, 3), (1, 3)])
        self.assertEqual(list(struct.unpack("<6q", selected)), [13, 14, 17, 18, 21, 22])
        with self.assertRaises(RuntimeError): core.packed_region(raw[:-1], [2, 3, 4], "torch.int64", [(0, 2), (0, 3), (0, 4)])
        with self.assertRaises(RuntimeError): core.packed_region(raw, [2, 3, 4], "torch.int64", [(0, 3), (0, 3), (0, 4)])

    def test_03_signed_zero_exact_embedding_and_metrics(self):
        raw = b"\0\0\0\x80" + bf(1) + bf(-2)
        self.assertEqual(core.bf16_to_fp32_bits(raw), struct.pack("<4f", 0., -0., 1., -2.))
        stat = core.full_stats(raw, "torch.bfloat16", guard)
        self.assertEqual((stat["zero_words"], stat["negative_zero_words"]), (2, 1))
        value = core.compare_words(b"\0\0", b"\0\x80", "torch.bfloat16", guard)
        self.assertEqual((value["different_words"], value["different_bits"], value["rmse"]), (1, 1, 0.0))
        value = core.compare_words(bf(0), bf(1), "torch.bfloat16", guard)
        self.assertIsNone(value["relative_l2"])
        value = core.compare_words(bf(1) + bf(2), bf(2) + bf(4), "torch.bfloat16", guard)
        self.assertEqual((value["squared_error_sum"], value["native_squared_norm"], value["relative_l2"]), (5., 5., 1.))
        value = core.compare_words(bf(1) * 19, bf(2) * 19, "torch.bfloat16", guard)
        self.assertEqual((value["words"], value["different_words"], len(value["examples"])), (19, 19, 8))

    def test_04_nonfinite_and_boolean_rejected(self):
        for dtype, raw in (("torch.bfloat16", b"\x80\x7f"), ("torch.bfloat16", b"\xc0\x7f"),
                           ("torch.float32", struct.pack("<f", float("inf"))),
                           ("torch.float32", struct.pack("<f", float("nan")))):
            with self.assertRaises(RuntimeError): core.full_stats(raw, dtype, guard)
            with self.assertRaises(RuntimeError): core.compare_words(raw, raw, dtype, guard)
        with self.assertRaises(RuntimeError): core.full_stats(b"\2", "torch.bool", guard)

    def test_05_small_complete_leaf_inventory(self):
        specs = core.native_specs(self.case["native_events"])
        self.assertEqual(len(core.validate_values(self.nv, specs, guard)), 30)
        self.assertEqual(len(core.validate_values(self.sv[0], self.ss, guard)), 47)
        missing = dict(self.nv); missing.pop(next(iter(missing)))
        extra = dict(self.nv); extra["unknown", ()] = b"\0\0"
        short = dict(self.nv); short["q_projection", ()] = short["q_projection", ()][:-2]
        for values in (missing, extra, short):
            with self.assertRaises(RuntimeError): core.validate_values(values, specs, guard)

    def test_06_rank_head_slot_and_prior_active_join(self):
        result = self.run_frame()
        self.assertTrue(all(r["all_bits_equal"] for r in result["comparisons"]))
        self.assertEqual(result["first_differing_saved_boundary_by_rank"], {"0": None, "1": None})
        self.assertEqual(len(result["comparisons"]), 44)
        domains = result["sdk_only_domains"]
        self.assertEqual(len(domains), 22)
        self.assertTrue(all(r["native_counterpart"] is False for r in domains))
        self.assertEqual({r["label"] for r in domains if "intermediate" in r["domain"]},
                         {"pv_prior", "pv_active", "o_proj_local_before_reduce"})
        embedding = [r for r in result["comparisons"] if r["boundary"] == "softmax_input"]
        self.assertTrue(all(r["native_bf16_exact_bit_embedding"] for r in embedding))
        self.assertTrue(all(r["words"] == 8 * 3 for r in embedding))
        self.measurements["small_frame_report_bytes"] = len(json.dumps(result, allow_nan=False).encode())

    def test_07_wrong_slot_rank_and_active_join_detected(self):
        case = copy.deepcopy(self.case); case["sdk_physical_row"] = 0
        with self.assertRaises(RuntimeError): self.run_frame(case=case)
        sv = {rank: dict(values) for rank, values in self.sv.items()}
        sv[0]["q_projection", ()] = self.sv[1]["q_projection", ()]
        result = self.run_frame(sv=sv)
        self.assertEqual(result["first_differing_saved_boundary_by_rank"]["0"], "q_projection")
        self.assertFalse(result["causal_operator_proven"])
        sv = {rank: dict(values) for rank, values in self.sv.items()}
        sv[0]["qk1_unscaled", ()] = bytes(len(sv[0]["qk1_unscaled", ()]))
        result = self.run_frame(sv=sv)
        rows = [r for r in result["comparisons"] if r["boundary"] == "qk0_unscaled" and r["rank"] == 0]
        self.assertEqual(rows[0]["different_words"], 8)
        self.assertEqual([r["packed_index"] for r in rows[0]["examples"]], list(range(2, 24, 3)))

    def test_08_padding_and_repeated_boundary_controls(self):
        for key, offset in ((("probabilities_bf16", ()), 3 * 8 * 4 * 2 + 2 * 2),
                            (("attention_output", ()), 3 * 2048 * 2),
                            (("route_inputs_sdk_decode", ("past_key_value", 0)), 0)):
            sv = {rank: dict(values) for rank, values in self.sv.items()}
            raw = sv[0][key]; sv[0][key] = raw[:offset] + bf(1) + raw[offset + 2:]
            with self.assertRaises(RuntimeError): self.run_frame(sv=sv)

    def test_09_all_six_call_report_budget_metadata_only(self):
        # Full report structure with eight examples per comparison. No complete
        # capture bytes are allocated, traversed or claimed as measurement.
        frame_template = self.run_frame()
        sample = core.compare_words(bf(1) * 8, bf(2) * 8, "torch.bfloat16", guard)
        calls = []
        sizes = []
        for call in core.CALLS:
            cases = [r for r in self.contract["frames"] if r["call"] == call]
            inventory = {"native": {}, "sdk": {}}
            def metadata(specs):
                output = []
                for key, s in specs.items():
                    row = {"label": key[0], "path": list(key[1]), "shape": s["shape"], "dtype": s["dtype"],
                           "bytes": s["bytes"], "words": s["bytes"] // core.WIDTHS[s["dtype"]], "sha256": "f" * 64}
                    row.update(finite_words=row["words"], zero_words=0, negative_zero_words=0,
                               minimum=-3.4028234663852886e38, maximum=3.4028234663852886e38,
                               true_words=0, false_words=0)
                    output.append(row)
                return output
            for case in cases:
                inventory["native"][str(case["frame"]["index"])] = metadata(core.native_specs(case["native_events"]))
            for rank in (0, 1): inventory["sdk"][str(rank)] = metadata(core.sdk_specs(cases[0]["bucket"]))
            frames = []
            for case in cases:
                frame = copy.deepcopy(frame_template)
                frame.update(frame=case["frame"], call=call, position=case["position"], bucket=case["bucket"])
                for row in frame["comparisons"]:
                    row.update(copy.deepcopy(sample)); row["frame_index"] = case["frame"]["index"]
                    for name in ("maximum_absolute_error", "squared_error_sum", "native_squared_norm", "relative_l2", "rmse"):
                        row[name] = 1.2345678901234567e99
                    for example in row["examples"]:
                        example.update(packed_index=99999999, native_bits=4294967295, sdk_bits=4294967295,
                                       native_value=-3.4028234663852886e38, sdk_value=3.4028234663852886e38)
                frames.append(frame)
            report = {"schema": "minicpm5_saved_operator_call_comparison_v1", "contract_sha256": core.digest(self.contract),
                      "call": call, "complete_input_comparison": True, "capture_files": 214,
                      "capture_bytes": sum(r["bytes"] for d in inventory.values() for rows in d.values() for r in rows),
                      "full_capture_inventory": inventory, "frames": frames,
                      "no_new_attention_or_tensor_arithmetic_executed": True, "cause_proven": False,
                      "runtime_admitted": False, "actual_process_exit_verified": False,
                      "full_model_qualified": False, "new_speedup_claim": False}
            sizes.append(len(json.dumps(report, separators=(",", ":"), allow_nan=False).encode()))
            self.assertLess(sizes[-1], core.MAX_REPORT_BYTES)
            calls.append(report)
        result = core.summarize(self.contract, calls)
        self.assertEqual((result["calls"], result["frames"], result["capture_files"]), (6, 24, 1284))
        self.assertFalse(result["new_speedup_claim"])
        with self.assertRaises(RuntimeError): core.summarize(self.contract, calls[:-1])
        swapped = list(calls); swapped[0], swapped[1] = swapped[1], swapped[0]
        with self.assertRaises(RuntimeError): core.summarize(self.contract, swapped)
        self.measurements.update(metadata_only_six_call_report_bytes=sizes,
                                 metadata_only_six_call_max_bytes=max(sizes),
                                 metadata_serialization_is_actual_capture_measurement=False)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(SyntheticChecks.measurements, sort_keys=True))
    raise SystemExit(not result.wasSuccessful())
