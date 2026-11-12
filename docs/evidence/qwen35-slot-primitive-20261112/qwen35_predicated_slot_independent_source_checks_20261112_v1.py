"""Independent source AST projections on stdlib tiny fixtures; no native work.

This executes the leaf's actual validation and schedule statements with fake
tensors and graph handles. It cannot establish compiler/device correctness.
"""
import ast
import collections
import contextlib
import hashlib
import itertools
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import qwen35_predicated_slot_protocol_v1 as protocol

LEAF = ROOT / "qwen35_predicated_slot_device_leaf_v1.py"
assert hashlib.sha256(LEAF.read_bytes()).hexdigest() == "fbee83e19a4aea6f3e69a7702afafda5a29e8390a64425e49b0a03db81007137"
tree = ast.parse(LEAF.read_text())
run_case = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_case")
body = next(n for n in run_case.body if isinstance(n, ast.Try)).body
functions = {n.name: n for n in body if isinstance(n, ast.FunctionDef)}


class TinyTensor:
    def __init__(self, values, dtype="bfloat16", device="cpu"):
        self.values = list(values)
        self.dtype = dtype
        self.shape = (len(self.values),)
        self.device = SimpleNamespace(type=device)

    def clone(self):
        return TinyTensor(self.values, self.dtype, self.device.type)

    def detach(self):
        return self

    def cpu(self):
        return TinyTensor(self.values, self.dtype)

    def clamp_min(self, minimum):
        return TinyTensor([max(minimum, v) for v in self.values], self.dtype)

    def __getitem__(self, indices):
        return TinyTensor([self.values[i] for i in indices.values], self.dtype)

    def tolist(self):
        return list(self.values)

    def raw(self):
        code = "q" if self.dtype == "int64" else "I" if self.dtype == "float32" else "H"
        return struct.pack("<" + code * len(self.values), *self.values)


def fake_writer(cache, updates, slots):
    result = cache.clone()
    for row, destination in enumerate(slots.values):
        if destination >= 0:
            result.values[destination] = updates.values[row]
    return result


def compare(a, b):
    return protocol.compare_storage(a.shape, a.dtype, a.raw(), b.shape, b.dtype, b.raw())


def call_projection(mode, fault, dtype, phase="correctness"):
    initial = TinyTensor((1, 2, 3, 4), dtype)
    updates = TinyTensor((11, 22, 33, 44), dtype)
    slots = TinyTensor((2, 2, -1, 0), "int64")
    persistent = initial.clone()
    persistent.device.type = "privateuseone"
    events = []

    class Compiled:
        states = SimpleNamespace(_parameters={"cache": persistent})

        def __call__(self, *args):
            events.append("invoke")
            cache, u, s = args if mode == "standalone" else (self.states._parameters["cache"], *args)
            old = cache[s.clamp_min(0)].clone()
            new = fake_writer(cache, u, s)
            if mode == "persistent_alias" and fault != "stale_state":
                cache.values[:] = new.values
            if fault == "wrong_old_read":
                old = new[s.clamp_min(0)].clone()
            if fault == "mutate_updates":
                u.values[0] = 999
            if fault == "mutate_slots":
                s.values[0] = 1
            if fault == "mutate_cache":
                cache.values[0] = 999
            if fault == "wrong_output":
                new.values[-1] = 999
            if fault == "wrong_state_device":
                cache.device.type = "cpu"
            if fault == "wrong_state_parameters":
                self.states._parameters["extra"] = cache.clone()
            outputs = (new,) if mode == "standalone" else (old, new)
            if fault == "wrong_arity":
                outputs += (new,)
            return outputs

    report = {"checks": [], "runtime_counts": {k: 0 for k in ("correctness", "warmup", "measured", "reset")}}
    clock = iter((100, 175))
    namespace = {"torch": SimpleNamespace(Tensor=TinyTensor), "protocol": protocol,
                 "store_slots": fake_writer, "compare": compare, "report": report,
                 "guard": events.append, "time": SimpleNamespace(perf_counter_ns=lambda: next(clock))}
    projection = ast.Module(body=[functions[n] for n in ("materialize", "state", "call")], type_ignores=[])
    exec(compile(projection, str(LEAF), "exec"), namespace)
    handle = {"compiled": Compiled(), "mode": mode, "variant": "candidate",
              "expected_state": initial.clone(), "state_name": "cache"}
    previous_expected = handle["expected_state"]
    error = None
    try:
        elapsed = namespace["call"](handle, phase, updates, slots, cache=initial.clone())
    except RuntimeError as exc:
        error = str(exc)
    if fault == "none":
        assert error is None, error
        assert report["checks"][-1]["passed"]
        assert elapsed == (75 if phase == "measured" else None)
        if mode == "persistent_alias":
            assert handle["expected_state"].values == [44, 2, 22, 4]
            assert handle["expected_state"] is not previous_expected
    else:
        assert error is not None, (mode, fault)
        assert handle["expected_state"] is previous_expected
        assert previous_expected.values == [1, 2, 3, 4]
        failures = {k for row in report["checks"] for k, v in row["checks"].items() if not v["passed"]}
        target = {"stale_state": "persistent_state", "wrong_old_read": "old_read",
                  "mutate_updates": "updates_unchanged", "mutate_slots": "slots_unchanged",
                  "mutate_cache": "cache_input_unchanged", "wrong_output": "updated_output"}.get(fault)
        if target:
            assert target in failures, (fault, failures)
        if fault == "stale_state":
            assert report["checks"][-1]["checks"]["updated_output"]["passed"]
            assert report["checks"][-1]["checks"]["old_read"]["passed"]
    return {"mode": mode, "dtype": dtype, "phase": phase, "fault": fault,
            "expected_outcome": "accepted" if fault == "none" else "rejected", "passed": True}


def schedule_projection(layout_name):
    loop = next(n for n in body if isinstance(n, ast.With))
    records = collections.defaultdict(list)
    counts = collections.Counter()
    report = {"graphs": [], "timing": {}}

    class Descriptor:
        def __init__(self, profile, salt):
            self.data = (layout_name, profile, salt)

        def clone(self):
            return Descriptor(*self.data[1:])

    def compile_graph(mode, variant, initial, updates, slots):
        report["graphs"].append((mode, variant))
        return {"mode": mode, "variant": variant}

    def reset(handle, initial, detail):
        if handle["mode"] == "persistent_alias":
            counts["reset"] += 1
        records[(handle["mode"], handle["variant"])].append(("reset", initial.data, detail))

    def call(handle, phase, updates, slots, cache, detail):
        counts[phase] += 1
        records[(handle["mode"], handle["variant"])].append((phase, updates.data, slots, cache.data, detail))
        return 10 if phase == "measured" else None

    namespace = {"protocol": protocol, "torch": SimpleNamespace(inference_mode=contextlib.nullcontext,
                 tensor=lambda value, dtype: tuple(value), int64="int64"), "tensor": Descriptor,
                 "compile_graph": compile_graph, "reset": reset, "call": call, "report": report,
                 "save": lambda: None, "gc": SimpleNamespace(collect=lambda: None), "guard": lambda event: None}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(LEAF), "exec"), namespace)
    assert report["graphs"] == [(m, v) for m in protocol.MODES for v in protocol.VARIANTS]
    assert dict(counts) == {"correctness": 5000, "warmup": 128, "measured": 640, "reset": 28}
    for mode in protocol.MODES:
        assert records[(mode, "reference")] == records[(mode, "candidate")]
        for profile in ("raw", "finite"):
            mappings = [r[2] for r in records[(mode, "reference")] if r[0] == "correctness" and r[1][1] == profile]
            assert mappings == list(itertools.product(range(-1, 4), repeat=4))
        timing = report["timing"][mode]
        assert timing["complete"] and timing["median_paired_ratio"] == 1.0
        orders = collections.Counter(tuple(r["order"]) for r in timing["paired_rows"])
        assert orders == {("reference", "candidate"): 5, ("candidate", "reference"): 5}
    return {"layout": layout_name, "graphs": len(report["graphs"]), "mock_call_counts": dict(counts),
            "all_625_mappings_each_profile_mode_variant": True, "paired_inputs_equal": True,
            "balanced_order_and_complete_paired_summary": True, "passed": True}


cases = []
for dtype in ("bfloat16", "float32"):
    for mode in protocol.MODES:
        for phase in ("correctness", "measured"):
            cases.append(call_projection(mode, "none", dtype, phase))
        faults = ["mutate_updates", "mutate_slots", "wrong_output", "wrong_arity"]
        faults += (["mutate_cache"] if mode == "standalone" else
                   ["stale_state", "wrong_old_read", "wrong_state_device", "wrong_state_parameters"])
        for fault in faults:
            cases.append(call_projection(mode, fault, dtype))
schedules = [schedule_projection(name) for name, _, _ in protocol.LAYOUTS]
native = sorted(k for k in sys.modules if k.split(".")[0] in
                {"torch", "numpy", "ml_dtypes", "torch_neuronx", "neuronxcc", "nki"})
assert not native, native
result = {"schema": "qwen35_predicated_slot_independent_source_checks_v1", "passed": True,
          "scope": "Actual leaf AST validation/schedule projected onto tiny stdlib mocks only; no native, compiler, device or model work",
          "call_projection_cases": cases, "call_projection_count": len(cases),
          "fault_rejections": sum(c["fault"] != "none" for c in cases),
          "schedule_projections": schedules, "native_modules_loaded": native,
          "mock_total_graphs": sum(r["graphs"] for r in schedules),
          "mock_total_explicit_calls": sum(sum(r["mock_call_counts"].values()) for r in schedules),
          "real_device_calls": 0, "new_measured_speedup": None}
print(json.dumps(result, indent=2, sort_keys=True))
