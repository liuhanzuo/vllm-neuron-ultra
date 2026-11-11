"""Source-only descriptor fixtures and tiny injected-FS bytes; no actual raw IO.

Full1284-leaf fixtures contain metadata only. The six-call orchestration check
mocks raw reads/computation and never claims actual payload coverage or metrics.
"""
import copy
import hashlib
import importlib.abc
import json
import math
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RejectNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "numpy", "transformers", "torch_neuronx", "neuronx_distributed"}:
            raise AssertionError("No native/model dependency permitted: " + fullname)


sys.meta_path.insert(0, RejectNativeImports())
from research import minicpm5_layer1_saved_operator_reader_v1 as reader
from research.check_minicpm5_layer1_operator_native_audit_io_v1 import MemoryFS
c = reader.comparison


def guard(): pass


def trace_fixture(sdk=False):
    expected = {"source": {"path": "/synthetic/source.py", "sha256": "a" * 64},
                "function": "synthetic_decode", "line_metadata": {"synthetic": True}}
    points = [{"role": role, "assignment_line": i * 2 + 1, "event_line": i * 2 + 2, "variable": role}
              for i, role in enumerate(("qk0", "qk1") if sdk else ("qk0",))]
    if sdk: expected.update(route="sdk_decode", points=points)
    else: expected["point"] = points[0]
    operations = []
    for point in points:
        operations.append(dict(observation="actual_python_local_after_assignment_before_next_statement", scaled_value_observed=True,
            scaled_value_recomputed=False, source=expected["source"], function=expected["function"],
            assignment_line=point["assignment_line"], event_line=point["event_line"], local_name=point["variable"],
            operator="python.divide" if sdk else "python.multiply", operator_identity_basis="pinned_source_AST",
            scalar=128 ** (.5 if sdk else -.5),
            scalar_origin="derived_from_pinned_expression_and_observed_head_dim" if sdk else "actual_function_argument",
            aten_operator_observed=False, route="sdk_decode" if sdk else "hf_eager", line_metadata=expected["line_metadata"],
            operand_binding_basis=reader.native.trace.OPERAND_BASIS, schema=reader.native.trace.SCHEMA))
    return expected, operations


def metadata(wanted, context):
    shape = wanted["shape"]
    # Deliberately huge source offset and non-packed source strides: descriptor
    # interpretation must retain them, never address packed payload with them.
    return dict(source_shape=shape, source_stride=[math.prod(shape[i + 1:]) * 3 for i in range(len(shape))],
                source_offset=987654321, source_dtype=wanted["dtype"], source_numel=math.prod(shape),
                selected_shape=shape, selected_bytes=wanted["bytes"], selection=wanted["selection"], cpu=True,
                owned_clone=True, original_version=None, version_counter_available=False, context=context)


def metadata_fixture():
    q = ROOT / "research"
    nm = json.loads((q / "minicpm5-layer1-operator-native-r3-sourceclosure-20261111-manifest.json").read_text())
    sm = json.loads((q / "minicpm5-layer1-operator-sdk-event-manifest-20261111-v1.json").read_text())
    ntrace, nops = trace_fixture(); strace, sops = trace_fixture(True)
    nplan = {"output_root": "/synthetic/native", "native_operator_manifest": nm}
    calls = [{"index": i} for i in range(14)]
    for ci in c.CALLS:
        cases = [r for r in nm["frames"] if r["call"] == ci]
        calls[ci].update(phase="decode", frames=[r["frame"] for r in cases], logical_positions=[r["logical_position"] for r in cases])
    splan = {"output_root": "/synthetic/sdk", "sdk_operator_manifest": sm, "calls": calls}
    inode = 1
    def ref(path, size):
        nonlocal inode
        value = dict(path=path, raw_bytes=size, sha256=hashlib.sha256(path.encode()).hexdigest(),
                     identity=dict(realpath=path, device=1, inode=inode, size=size, mtime_ns=1, ctime_ns=1))
        inode += 1
        return value
    full, reports = {}, []
    for case in nm["frames"]:
        index = case["frame"]["index"]; root = "/synthetic/native/native-operator/frame-%02d/full_trace" % index
        snapshots = []; ordinal = 0
        for event_index, event in enumerate(case["events"]):
            label = event["label"]
            routed = label in ("qk0_unscaled", "qk0_scaled", "qk0_scale_operation", "softmax_input", "probabilities_fp32", "probabilities_bf16", "pv_context")
            ctx = dict(arm="HF_native", rank=-1, call=case["call"], phase="decode", frames=[case["frame"]],
                       thread=123, route="hf_eager" if routed else None, event_index=event_index)
            def build(wanted):
                nonlocal ordinal
                if type(wanted) is list: return [build(w) for w in wanted]
                if "shape" not in wanted:
                    return nops[0] if label == "qk0_scale_operation" else dict(function="native.apply_rotary_pos_emb",
                        position_source="original_rotary_module_on_saved_official_decode_input", direct_layer1_replay=True)
                result = dict(metadata=metadata(wanted, ctx), reference=ref(root + "/observed-%02d.bin" % ordinal, wanted["bytes"]))
                ordinal += 1; return result
            snapshots.append(dict(label=label, context=ctx, value=build(event["value"])))
        capture = dict(complete=True, actual_route="hf_eager", original_layer_forward_calls=1, wrappers_return_original_objects=True,
            scale_observation_schema=reader.native.trace.SCHEMA, aten_dispatch_used=False, trace_contract=ntrace,
            scale_provenance=nops, events=[e["label"] for e in case["events"]], layer_index=1, arm="HF_native", rank=-1,
            call=case["call"], frames=[case["frame"]], phase="decode", diagnostic_only=True,
            full_model_endpoint_gate_still_required=True, production_runtime_tested=False, snapshots=snapshots)
        full[index] = dict(frame=case["frame"], call=case["call"], arm="full_trace", source_inputs=case, outputs={},
                           official_full_bytes_exact=dict(attention=True, key=True, value=True), capture=capture)
        arms = [{"arm": arm, "outputs": {}, "result": dict(path=root.replace("full_trace", arm) + "/result.json", sha256="b" * 64, bytes=100)}
                for arm in c.native.ARMS]
        reports.append(dict(frame=case["frame"], all_three_arms_official_exact=True, arms=arms))
    ndriver = dict(schema="minicpm5_layer1_operator_native_driver_v4", complete=True, manifest_sha256=c.digest(nm),
                   records=reports, raw_files=936, raw_bytes=28490724, all_unobserved_noop_full_endpoints_and_full_caches_exact=True)
    ranks = []
    for rank in (0, 1):
        records = []
        for ci, bucket in zip(c.CALLS, c.sdk.BUCKETS):
            call = calls[ci]; frames = sorted(call["frames"], key=lambda f: f["slot"]); expected = c.sdk.call_manifest(rank, ci, bucket)
            rows = []
            root = "/synthetic/sdk/cpu-rank%d/strict_anchor_s0/call-%02d/official_full/capture" % (rank, ci)
            for ei, event in enumerate(expected["tensor_events"]):
                ctx = dict(arm="C_Hd_control", rank=rank, call=call, frames=frames, phase="decode", thread=456, route="sdk_decode", event_index=ei)
                value = {}
                for ti, wanted in enumerate(event["tensors"]):
                    enriched = dict(ctx, event_label=event["label"], tensor_index=ti, tensor_path=wanted["path"])
                    entry = dict(metadata=metadata(wanted, enriched), reference=ref(root + "/%02d-%s-%02d.bin" % (ei, event["label"], ti), wanted["bytes"]))
                    if not wanted["path"]: value = entry; continue
                    target = value
                    for part in wanted["path"][:-1]: target = target.setdefault(part, {})
                    target[wanted["path"][-1]] = entry
                rows.append(dict(label=event["label"], context=ctx, value=value))
            summary = dict(complete=True, actual_route="sdk_decode", arm="C_Hd_control", rank=rank, phase="decode",
                original_layer_forward_calls=1, wrappers_return_original_objects=True, scale_observation_schema=reader.native.trace.SCHEMA,
                aten_dispatch_used=False, trace_contract=strace, scale_provenance=sops, events=[e["label"] for e in expected["tensor_events"]],
                layer_index=1, call=call, frames=frames)
            observation = dict(mode="official_full", capture=dict(summary=summary,
                full_backing_expectation="unchanged_direct_attention_update_disabled", snapshots=dict(complete=True,
                manifest=expected, raw_bytes=expected["capture_raw_bytes"], records=rows)))
            records.append({"result": dict(call=call, rank=rank, arm=reader.ANCHOR,
                observations=[{"mode": "official_unobserved"}, {"mode": "official_noop"}, observation])})
        ranks.append(dict(schema="minicpm5_layer1_operator_sdk_driver_v2", complete=True, rank=rank, manifest=sm,
                          records=records, all_previous_controls_full_bytes_exact=True))
    refs = {name: {"path": "/synthetic/prerequisites/" + name + ".json", "sha256": "c" * 64} for name in reader.PREREQUISITES}
    return nplan, splan, ndriver, full, ranks, ntrace, strace, refs


def small_fs(raw=b"\0\0\0\x80\x80\x3f"):
    path = "/synthetic/capture/tiny.bin"
    ref = dict(path=path, raw_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
               identity=dict(realpath=path, device=1, inode=19, size=len(raw), mtime_ns=1, ctime_ns=1))
    fs = MemoryFS(({"capture": ref}, {}, {}, [], SimpleNamespace(data={path: raw})))
    return ref, fs


class ReaderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = metadata_fixture()
        cls.plan = reader.prepare(*cls.inputs)
        cls.measures = {"metadata_plan_serialized_bytes": len(json.dumps(cls.plan, separators=(",", ":")).encode()),
                        "synthetic_raw_bytes_per_filesystem_fixture": 6, "recorded_raw_bytes_read": 0,
                        "full_capture_payload_fixture_constructed": False,
                        "full_six_call_test_is_mocked_metadata_controlflow_only": True}

    def test_01_complete_descriptors_no_payload(self):
        refs = reader.validate_plan(self.plan)
        self.assertEqual((len(refs), sum(r["raw_bytes"] for r in refs.values())), (1284, 77790564))
        self.assertLess(self.measures["metadata_plan_serialized_bytes"], reader.MAX_METADATA_BYTES)
        self.assertFalse(self.plan["runtime_admitted"])
        first = self.plan["calls"][0]["native"]["4"][0]
        self.assertEqual(first["metadata"]["source_offset"], 987654321)

    def test_02_metadata_shape_identity_order_fail_closed(self):
        np, sp, nd, full, ranks, nt, st, refs = self.inputs
        case = np["native_operator_manifest"]["frames"][0]
        for fault in ("shape", "stride", "offset", "path", "dtype", "order", "frame"):
            changed = copy.deepcopy(full[4]); value = changed["capture"]["snapshots"][0]["value"]
            if fault == "shape": value["metadata"]["selected_shape"] = [1]
            elif fault == "stride": value["metadata"]["source_stride"] = [-1, 1, 1]
            elif fault == "offset": value["metadata"]["source_offset"] = -1
            elif fault == "path": value["reference"]["path"] = "/synthetic/other.bin"
            elif fault == "dtype": value["metadata"]["source_dtype"] = "torch.float32"
            elif fault == "order": changed["capture"]["snapshots"].reverse()
            else: changed["frame"] = dict(case["frame"], slot=0)
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): reader.native_descriptors(changed, case, np["output_root"], nt)
        obs = ranks[0]["records"][0]["result"]["observations"][-1]
        for fault in ("context", "extra", "short", "rank", "slots"):
            changed = copy.deepcopy(obs)
            first = changed["capture"]["snapshots"]["records"][0]
            if fault == "context": first["value"]["metadata"]["context"]["tensor_index"] = 1
            elif fault == "extra": first["value"] = {"extra": first["value"]}
            elif fault == "short": changed["capture"]["snapshots"]["records"].pop()
            elif fault == "rank": changed["capture"]["summary"]["rank"] = 1
            else: changed["capture"]["summary"]["frames"].reverse()
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): reader.sdk_descriptors(changed, 0, sp["calls"][4], 2048, sp["output_root"], st)

    def test_03_missing_extra_duplicate_descriptors_rejected(self):
        for fault in ("missing", "extra", "duplicate", "geometry"):
            changed = copy.deepcopy(self.plan)
            rows = changed["calls"][0]["native"]["4"]
            if fault == "missing": rows.pop()
            elif fault == "extra": rows.append(copy.deepcopy(rows[0]))
            elif fault == "duplicate": rows[1]["reference"]["identity"]["inode"] = rows[0]["reference"]["identity"]["inode"]
            else: rows[0]["spec"]["shape"] = [1, 2, 1024]
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): reader.validate_plan(changed)

    def test_04_tiny_stable_reader_and_native_reader_reuse(self):
        ref, fs = small_fs(); io = reader.StableCaptureIO({ref["path"]: ref}, guard, filesystem=fs)
        self.assertEqual(fs.events, [])
        bounded = reader.native.Reader(io.read_bytes, guard)
        raw = bounded.get(ref, ref["path"], 6, identity=True)
        self.assertEqual(raw, b"\0\0\0\x80\x80\x3f")
        self.assertEqual(len(bounded.seen), 1)
        self.assertFalse(fs.handles)
        self.assertFalse(any(op == "scandir" for op, path in fs.events))
        with self.assertRaises(RuntimeError): io.read_bytes(ref, ref["path"], 6)

    def test_05_guard_and_unknown_path_precede_filesystem(self):
        ref, fs = small_fs()
        def deny(): raise RuntimeError("guard denied")
        io = reader.StableCaptureIO({ref["path"]: ref}, deny, filesystem=fs)
        with self.assertRaisesRegex(RuntimeError, "guard denied"): io.read_bytes(ref, ref["path"], 6)
        self.assertEqual(fs.events, [])
        io.guard = guard
        changed = copy.deepcopy(ref); changed["sha256"] = "a" * 64
        with self.assertRaises(RuntimeError): io.read_bytes(changed, ref["path"], 6)
        self.assertEqual(fs.events, [])

    def test_06_length_hash_identity_and_link_faults(self):
        for fault in ("short", "grown", "hash", "inode", "owner", "hardlink", "symlink", "parent_symlink"):
            ref, fs = small_fs(); node = fs.nodes[ref["path"]]
            if fault == "short": node.data = node.data[:-1]
            elif fault == "grown": node.data += b"x"
            elif fault == "hash": node.data = b"x" * 6
            elif fault == "inode": node.st_ino += 1
            elif fault == "owner": node.st_uid = 0
            elif fault == "hardlink": node.st_nlink = 2
            elif fault == "symlink": node.st_mode = stat.S_IFLNK | 0o777
            else: fs.nodes["/synthetic/capture"].st_mode = stat.S_IFLNK | 0o777
            io = reader.StableCaptureIO({ref["path"]: ref}, guard, filesystem=fs)
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): io.read_bytes(ref, ref["path"], 6)
            self.assertFalse(fs.handles); self.assertEqual(io._read, set())

    def test_07_midread_file_parent_and_guard_faults(self):
        for fault in ("mtime", "parent", "guard"):
            ref, fs = small_fs(); failed = [False]
            def active_guard():
                if failed[0]: raise RuntimeError("guard lost")
            def hook(op, path):
                if op == "read" and path == ref["path"]:
                    fs.hook = None
                    if fault == "mtime": fs.nodes[path].st_mtime_ns += 1
                    elif fault == "parent":
                        parent = "/synthetic/capture"; fs.nodes[parent] = copy.copy(fs.nodes[parent]); fs.nodes[parent].st_ino += 1
                    else: failed[0] = True
            fs.hook = hook
            io = reader.StableCaptureIO({ref["path"]: ref}, active_guard, filesystem=fs)
            with self.subTest(fault=fault), self.assertRaises(RuntimeError): io.read_bytes(ref, ref["path"], 6)
            self.assertFalse(fs.handles); self.assertEqual(io._read, set())

    def test_08_six_call_controlflow_metadata_only(self):
        # This deliberately mocks BOTH payload reader and scientific comparison.
        # It verifies grouping/retention/exhaustion only, never77.8MB data reads.
        visited = []; emitted = []
        class MetadataReader:
            def __init__(self, callback, guard): self.seen = {}; self.bytes = {}
            def get(self, ref, path, size, identity=False):
                self.seen[path] = ref; self.bytes[path] = size; visited.append(path); return None
        def fake_compare(contract, call, nv, sv, guard):
            cases = [r for r in contract["frames"] if r["call"] == call]
            self.assertEqual(len(nv), 4); self.assertEqual(set(sv), {0, 1})
            self.assertTrue(all(len(v) == 30 and set(v.values()) == {None} for v in nv.values()))
            self.assertTrue(all(len(v) == 47 and set(v.values()) == {None} for v in sv.values()))
            group = self.plan["calls"][c.CALLS.index(call)]
            size = sum(row["spec"]["bytes"] for side in ("native", "sdk") for rows in group[side].values() for row in rows)
            return {"schema": "minicpm5_saved_operator_call_comparison_v1", "contract_sha256": c.digest(contract),
                    "call": call, "complete_input_comparison": True, "capture_files": 214, "capture_bytes": size,
                    "frames": [{"frame": row["frame"], "call": call, "position": row["position"], "bucket": row["bucket"],
                                "first_differing_saved_boundary_by_rank": {"0": None, "1": None}} for row in cases]}
        def sink(report): emitted.append(report["call"]); report["frames"].clear()
        with patch.object(reader.native, "Reader", MetadataReader), patch.object(c, "compare_call", fake_compare):
            summary = reader.compare_all(self.plan, lambda *a: None, guard, sink, expected_plan_sha256=c.digest(self.plan))
        self.assertEqual(emitted, list(c.CALLS)); self.assertEqual(len(visited), 1284); self.assertEqual(len(set(visited)), 1284)
        self.assertEqual(summary["frames"], 24); self.assertFalse(summary["runtime_admitted"])

    def test_09_bad_plan_hash_or_read_failure_no_partial_output(self):
        read = []; outputs = []
        def failure(*args): read.append(args[1]); raise RuntimeError("synthetic read failure")
        with self.assertRaises(RuntimeError): reader.compare_all(self.plan, failure, guard, outputs.append, expected_plan_sha256="0" * 64)
        self.assertEqual(read, [])
        with self.assertRaisesRegex(RuntimeError, "synthetic read failure"):
            reader.compare_all(self.plan, failure, guard, outputs.append, expected_plan_sha256=c.digest(self.plan))
        self.assertEqual(len(read), 1); self.assertEqual(outputs, [])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReaderChecks))
    if hasattr(ReaderChecks, "measures"): print(json.dumps(ReaderChecks.measures, sort_keys=True))
    raise SystemExit(not result.wasSuccessful())
