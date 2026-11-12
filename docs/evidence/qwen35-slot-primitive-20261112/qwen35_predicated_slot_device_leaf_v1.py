"""Callable Qwen primitive leaf; intentionally no CLI or serial owner.

An existing owned serial runner must dispatch one exact layout per fresh process.
It must enforce the protocol's cgroup/time/storage caps, bind current history and
lock identity, retain descendants, and provide guard/save_report/record_artifact.
The callbacks do not by themselves establish ownership. No compatible existing
dispatcher accepts this leaf unchanged. Importing this file is stdlib-only.
"""
import gc
import json
from pathlib import Path
import time

import qwen35_predicated_slot_protocol_v1 as protocol


def run_case(name, output_dir, *, guard, save_report, record_artifact):
    """Future device call, never invoke during source review.

    guard(event: str): caller checks its existing ownership/resource mechanism;
      must raise on loss. An external monitor must enforce caps during trace.
    save_report(report: dict): caller durably/atomically saves the complete report.
    record_artifact(info: dict): caller retains path/size/hash in its artifact log.
    output_dir: caller-chosen, absent layout directory under its bounded run root.

    No retries, weights, package mutation, queue launch, or model qualification.
    """
    case = protocol.layout(name)
    if not all(callable(value) for value in (guard, save_report, record_artifact)):
        raise TypeError("Existing owner callbacks required")
    guard("before_source_verification")
    source_sha256 = protocol.verify_sources()
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError("Retain existing output; choose a fresh layout directory")
    report = {"schema": "qwen35_predicated_slot_primitive_result_v1", "case": case,
              "source_sha256": source_sha256, "complete": False, "passed": False,
              "graphs": [], "checks": [], "timing": {}, "controls": [],
              "runtime_counts": {key: 0 for key in ("correctness", "warmup", "measured", "reset")},
              "timing_scope": protocol.TIMING_SCOPE, "full_model_qualified": False,
              "model_speedup_claim": False, "lfm105_tasks_replaced": False,
              "required_resources": dict(protocol.RESOURCES)}

    def save():
        encoded = json.dumps(report, allow_nan=False).encode()
        if len(encoded) > protocol.RESOURCES["report_cap_bytes_per_layout"]:
            raise RuntimeError("Complete report exceeds fixed cap; do not truncate")
        save_report(report)

    try:
        guard("before_output_creation")
        directory.mkdir(parents=False, exist_ok=False)
        save()
        guard("before_native_imports")
        import numpy as np
        import torch
        import torch_neuronx
        from probe_slot_cache_nki_read import array_for
        from qwen35_text_port import store_slots
        from slot_cache_nki_predicated_torch import store_slots_nki_predicated

        torch.set_num_threads(1)
        dtype = torch.bfloat16 if case["dtype"] == "bfloat16" else torch.float32

        def tensor(profile, salt):
            array = array_for(case["dtype"], case["shape"], raw=profile == "raw", salt=salt)
            # Integer view and copy preserve BF16/FP32 exceptional payload bits.
            return torch.from_numpy(array.view(np.uint16).copy()).view(dtype)

        def raw_bytes(value):
            return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

        def compare(actual, expected):
            return protocol.compare_storage(tuple(actual.shape), str(actual.dtype), raw_bytes(actual),
                                            tuple(expected.shape), str(expected.dtype), raw_bytes(expected))

        # Byte-comparator negative controls: signed zero and NaN payload changes
        # must be rejected. These are not approximate floating point comparisons.
        word_dtype = np.uint16 if case["dtype"] == "bfloat16" else np.uint32
        pairs = ((0, 0x8000), (0x7fc1, 0x7fc2)) if case["dtype"] == "bfloat16" else (
            (0, 0x80000000), (0x7fc00001, 0x7fc00002))
        for left, right in pairs:
            a = torch.from_numpy(np.array([left], dtype=word_dtype).view(np.uint16).copy()).view(dtype)
            b = torch.from_numpy(np.array([right], dtype=word_dtype).view(np.uint16).copy()).view(dtype)
            passed = compare(a, a)["passed"] and not compare(a, b)["passed"]
            report["controls"].append({"left_word": left, "right_word": right, "passed": passed})
            if not passed:
                raise RuntimeError("Byte-comparator negative control failed")

        class Write(torch.nn.Module):
            def __init__(self, writer):
                super().__init__()
                self.writer = writer

            def forward(self, cache, updates, slots):
                return self.writer(cache, updates, slots)

        class ReadWrite(torch.nn.Module):
            def __init__(self, initial, writer):
                super().__init__()
                self.cache = torch.nn.Parameter(initial.clone(), requires_grad=False)
                self.writer = writer

            def forward(self, updates, slots):
                # The same gather as Qwen; this must observe the old state even
                # when the following writer output MUST_ALIASes self.cache.
                old = self.cache[slots.clamp_min(0)]
                new = self.writer(self.cache, updates, slots)
                return old, new

        def materialize(value):
            outputs = (value,) if isinstance(value, torch.Tensor) else tuple(value)
            return tuple(item.detach().cpu().clone() for item in outputs)

        def state(handle):
            values = dict(handle["compiled"].states._parameters.items())
            if list(values) != [handle["state_name"]]:
                raise RuntimeError("Persistent alias parameter set changed")
            value = values[handle["state_name"]]
            if value.device.type != "privateuseone":
                raise RuntimeError("Persistent state is not on the Neuron device")
            return value

        def compile_graph(mode, variant, initial, updates, slots):
            writer = store_slots if variant == "reference" else store_slots_nki_predicated
            source = Write(writer).eval() if mode == "standalone" else ReadWrite(initial, writer).eval()
            example = (initial.clone(), updates, slots) if mode == "standalone" else (updates, slots)
            aliases = {} if mode == "standalone" else {source.cache: 1}
            work = directory / f"{mode}-{variant}"
            guard(f"before_compile:{mode}:{variant}")
            work.mkdir(exist_ok=False)
            started = time.monotonic()
            compiled = torch_neuronx.trace(source, example, input_output_aliases=aliases,
                                          compiler_workdir=str(work), compiler_args=list(protocol.FLAGS))
            compile_seconds = time.monotonic() - started
            guard(f"after_compile:{mode}:{variant}")
            artifact = work / "model.pt"
            compiled.save(str(artifact))
            size = artifact.stat().st_size
            if size > protocol.RESOURCES["individual_model_artifact_cap_bytes"]:
                raise RuntimeError("Compiled artifact exceeds proposed fixed budget")
            info = {"path": str(artifact), "size_bytes": size, "sha256": protocol.digest_file(artifact),
                    "layout": name, "mode": mode, "variant": variant}
            record_artifact(info)
            handle = {"compiled": compiled, "expected_state": initial.clone(),
                      "mode": mode, "variant": variant}
            if mode == "persistent_alias":
                states = dict(compiled.states._parameters.items())
                if len(states) != 1:
                    raise RuntimeError("Expected exactly one actual alias state")
                state_name, value = next(iter(states.items()))
                if not compare(value, initial)["passed"]:
                    raise RuntimeError("Initial compiled state bytes differ")
                handle["state_name"] = state_name
                del states, value
            guard(f"before_device_load:{mode}:{variant}")
            torch_neuronx.move_trace_to_device(compiled, 0)
            if mode == "persistent_alias" and not compare(state(handle), initial)["passed"]:
                raise RuntimeError("Initial actual device state bytes differ")
            report["graphs"].append({"mode": mode, "variant": variant,
                                     "artifact": info, "compile_seconds": compile_seconds,
                                     "input_output_alias_indices": [1] if aliases else [],
                                     "state_name": handle.get("state_name")})
            save()
            return handle

        def call(handle, phase, updates, slots, *, cache=None, detail=None):
            mode = handle["mode"]
            before_updates, before_slots = updates.clone(), slots.clone()
            if mode == "standalone":
                before_cache = cache.clone()
                expected = (store_slots(before_cache, before_updates, before_slots),)
                args = (cache, updates, slots)
            else:
                previous = handle["expected_state"]
                expected = (previous[before_slots.clamp_min(0)].clone(),
                            store_slots(previous, before_updates, before_slots))
                args = (updates, slots)
            guard(f"before_call:{mode}:{handle['variant']}:{phase}")
            if phase == "measured":
                actual, elapsed = protocol.measured_call(lambda: handle["compiled"](*args),
                                                        materialize, time.perf_counter_ns)
            else:
                actual, elapsed = materialize(handle["compiled"](*args)), None
            report["runtime_counts"][phase] += 1
            if len(actual) != len(expected):
                raise RuntimeError("Unexpected compiled output arity")
            checks = {"updated_output": compare(actual[-1], expected[-1]),
                      "updates_unchanged": compare(updates, before_updates),
                      "slots_unchanged": compare(slots, before_slots)}
            if mode == "standalone":
                checks["cache_input_unchanged"] = compare(cache, before_cache)
            else:
                checks["old_read"] = compare(actual[0], expected[0])
                checks["persistent_state"] = compare(state(handle), expected[1])
            row = {"phase": phase, "mode": mode, "variant": handle["variant"],
                   "slots": before_slots.tolist(), "detail": detail, "checks": checks,
                   "elapsed_ns": elapsed, "passed": all(item["passed"] for item in checks.values())}
            report["checks"].append(row)
            if not row["passed"]:
                raise RuntimeError(f"Full byte comparison failed: {mode}/{handle['variant']}/{phase}")
            if mode == "persistent_alias":
                handle["expected_state"] = expected[1]
            return elapsed

        def reset(handle, initial, detail):
            if handle["mode"] == "persistent_alias":
                # Reset via an actual all-slot write, never substitute a host
                # shadow for checking real persistent device bytes.
                call(handle, "reset", initial.clone(), torch.tensor(protocol.TIMING_MAPPINGS[0], dtype=torch.int64),
                     detail=detail)

        with torch.inference_mode():
            for mode in protocol.MODES:
                initial = tensor("raw", 0)
                sample_updates = tensor("raw", 11)
                sample_slots = torch.tensor(protocol.TIMING_MAPPINGS[0], dtype=torch.int64)
                handles = {variant: compile_graph(mode, variant, initial, sample_updates, sample_slots)
                           for variant in protocol.VARIANTS}
                # All 625-map/raw+finite correctness succeeds before timing
                # this mode. Schedule is also exercised by stdlib mock checks.
                paired_rows = []
                for sequence in protocol.sequences():
                    phase, profile = sequence["phase"], sequence["profile"]
                    initial = tensor(profile, sequence["initial_salt"])
                    pair = {"round": sequence["round"], "order": list(sequence["order"]), "elapsed_ns": {}}
                    for variant in sequence["order"]:
                        handle = handles[variant]
                        reset(handle, initial, {"phase": phase, "profile": profile, "round": sequence["round"]})
                        samples = []
                        for step, mapping in enumerate(sequence["mappings"]):
                            samples.append(call(handle, phase, tensor(profile, sequence["update_salt_base"] + step),
                                                torch.tensor(mapping, dtype=torch.int64), cache=initial.clone(),
                                                detail={"profile": profile, "round": sequence["round"], "step": step}))
                        pair["elapsed_ns"][variant] = samples
                        save()
                    if phase == "measured":
                        paired_rows.append(pair)
                        report["timing"][mode] = {"paired_rows": paired_rows, "complete": False}
                    save()
                report["timing"][mode].update(complete=True, **protocol.summarize_pairs(paired_rows))
                del handles, handle
                gc.collect()
                guard(f"after_mode:{mode}")
        expected_counts = protocol.counts()["per_layout"]
        if len(report["graphs"]) != expected_counts["graphs"]:
            raise RuntimeError("Incomplete exact graph matrix")
        if any(report["runtime_counts"][key] != expected_counts[key + "_calls"]
               for key in report["runtime_counts"]):
            raise RuntimeError("Incomplete exact runtime call matrix")
        guard("before_completion")
        protocol.verify_sources()
        report.update(complete=True, passed=True)
        return report
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
