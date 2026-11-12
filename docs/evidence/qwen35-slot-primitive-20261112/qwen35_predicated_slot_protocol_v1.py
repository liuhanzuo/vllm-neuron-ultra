"""Inert exact-shape Qwen slot-writer workload. No runtime/queue ownership.

This is an independent primitive diagnostic, not an LFM phase or a Qwen model
qualification. Importing this module performs no IO or native imports.
"""
import hashlib
import itertools
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
BUCKETS = (128, 256, 512, 1024, 2048, 4096)
LAYOUTS = (
    ("conv", "bfloat16", (4, 3072, 4)),
    ("recurrent", "float32", (4, 8, 128, 128)),
    *((f"kv{length}", "bfloat16", (4, 1, length, 256)) for length in BUCKETS),
)
CORRECTNESS_MAPPINGS = tuple(itertools.product(range(-1, 4), repeat=4))
TIMING_MAPPINGS = (
    (0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1), (-1, -1, -1, -1),
    (3, -1, -1, -1), (-1, 2, -1, -1), (-1, -1, 1, -1), (-1, -1, -1, 0),
    (0, -1, 2, -1), (1, 0, -1, 3), (2, 2, 0, 3), (1, 0, 1, -1),
    (0, 0, 0, 0), (3, 3, 3, 3), (-1, 2, -1, 2), (3, -1, 3, -1),
)
PROFILES = ("raw", "finite")
MODES = ("standalone", "persistent_alias")
VARIANTS = ("reference", "candidate")
WARMUP_CYCLES = 2
MEASURED_CYCLES = 10
FLAGS = (
    "--target=trn1", "--auto-cast=none", "--model-type=transformer", "-O2", "--lnc=1",
    "--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma",
)
FROZEN_SOURCES = {
    "qwen35_text_port.py": "3b560a3dcbb925a4c010a679ebe8c5532178672d77edbe768d490e307cdd2e34",
    "qwen35_primitives.py": "4ce394ed810a743504391a91af6026fb30953bd0415ef63e8c883a2660d4ecbf",
    "qwen35_neuron_adapter.py": "de89afc16b64e06235d19246db486b465f5bb45ab4f4b1d6a67199439db89fe2",
    "slot_cache_nki_predicated.py": "b2f0ef3a727409c9816dc9c5e77cfd13df3a1777f4cdc31c8ce3e8cd373eca7e",
    "slot_cache_nki_predicated_torch.py": "33f4ffdd55d0a3eb3a1b7c9f3ddf2f7505161a7809a2fabb22956988475b7f67",
    "probe_slot_cache_nki_read.py": "10756354372e6479c762c89e8c1e37007a613fca27e00246819aa4f7f97f5ddc",
}
RESOURCES = {
    "interpreter": "/home/ubuntu/venvs/vllm_neuron/bin/python",
    "legacy_namespace": "neuronxcc.nki",
    "visible_core": "0",
    "cpus": 2,
    "torch_threads": 1,
    "memory_max_bytes": 6 * 2**30,
    "memory_high_bytes": 5 * 2**30,
    "swap_max_bytes": 0,
    "timeout_seconds_per_layout": 900,
    "timeout_seconds_total": 7200,
    "output_cap_bytes_per_layout": 2**30,
    "output_cap_bytes_total": 8 * 2**30,
    "individual_model_artifact_cap_bytes": 128 * 2**20,
    "report_cap_bytes_per_layout": 16 * 2**20,
    "root_floor_bytes": 28 * 2**30,
    "nvme_floor_bytes": 30 * 2**30,
    "output_volume_existing_free_reserve_bytes": 30 * 2**30,
    "additional_headroom_bytes": 2 * 2**30,
    "full_workload_initial_free_bytes": 40 * 2**30,
    "fresh_processes": 8,
    "max_live_graphs": 2,
    "max_graph_artifacts": 32,
    "lfm_separate_required_free_bytes_unchanged": 54 * 2**30,
}
TIMING_SCOPE = (
    "Synchronous host-call wall time: perf_counter_ns immediately before compiled invocation "
    "through detach().cpu().clone() for every returned tensor. Includes dispatch, input transfer, "
    "execution, output transfer and host copies. Guard, oracle, input construction, checks, "
    "persistent-state inspection, resets, compilation and report IO are outside the timer. "
    "A cycle is the sum of 16 separately timed calls with verification between calls. "
    "This is not device-only latency, uninterrupted throughput, TP2 or model speedup. "
    "A neutral wall-time result does not establish that the compiler already fuses the writer."
)


def layout(name):
    matches = [row for row in LAYOUTS if row[0] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown exact layout: {name!r}")
    name, dtype, shape = matches[0]
    return {"name": name, "dtype": dtype, "shape": list(shape), "rows": 4,
            "per_slot_width": math.prod(shape[1:]),
            "tensor_bytes": math.prod(shape) * (2 if dtype == "bfloat16" else 4)}


def counts():
    """Maximum successful-path call counts; any failure aborts without retries."""
    arms, modes = len(VARIANTS), len(MODES)
    per_layout = {
        "graphs": arms * modes,
        "correctness_calls": arms * modes * len(PROFILES) * len(CORRECTNESS_MAPPINGS),
        "warmup_calls": arms * modes * WARMUP_CYCLES * len(TIMING_MAPPINGS),
        "measured_calls": arms * modes * MEASURED_CYCLES * len(TIMING_MAPPINGS),
        # Only aliases reset. Correctness resets per profile; timing per cycle.
        "reset_calls": arms * (len(PROFILES) + WARMUP_CYCLES + MEASURED_CYCLES),
    }
    per_layout["total_runtime_calls"] = sum(v for k, v in per_layout.items() if k != "graphs")
    return {"per_layout": per_layout,
            "total": {key: value * len(LAYOUTS) for key, value in per_layout.items()}}


def paired_order(round_index):
    if type(round_index) is not int or round_index < 0:
        raise ValueError("Non-negative integer round required")
    return VARIANTS if round_index % 2 == 0 else tuple(reversed(VARIANTS))


def sequences():
    """Shared scientific schedule, identical for each exact layout and mode."""
    for profile_index, profile in enumerate(PROFILES):
        yield {"phase": "correctness", "profile": profile, "round": profile_index,
               "initial_salt": profile_index, "update_salt_base": 17,
               "order": VARIANTS, "mappings": CORRECTNESS_MAPPINGS}
    for phase, cycles in (("warmup", WARMUP_CYCLES), ("measured", MEASURED_CYCLES)):
        for round_index in range(cycles):
            yield {"phase": phase, "profile": "finite", "round": round_index,
                   "initial_salt": 101 + round_index,
                   "update_salt_base": 211 + round_index * 17,
                   "order": paired_order(round_index), "mappings": TIMING_MAPPINGS}


def compare_storage(actual_shape, actual_dtype, actual_raw, expected_shape, expected_dtype, expected_raw):
    """Complete raw storage comparison; never numeric equality or a sample/hash gate."""
    shape_equal, dtype_equal = actual_shape == expected_shape, actual_dtype == expected_dtype
    bitwise_equal = actual_raw == expected_raw
    row = {"passed": shape_equal and dtype_equal and bitwise_equal,
           "shape_equal": shape_equal, "dtype_equal": dtype_equal,
           "bitwise_equal": bitwise_equal, "bytes_compared": len(expected_raw),
           "actual_sha256": hashlib.sha256(actual_raw).hexdigest(),
           "expected_sha256": hashlib.sha256(expected_raw).hexdigest()}
    if not bitwise_equal:
        first = []
        for index, (left, right) in enumerate(zip(actual_raw, expected_raw)):
            if left != right:
                first.append(index)
                if len(first) == 16:
                    break
        row["first_different_bytes"] = first
    return row


def measured_call(invoke, materialize, clock_ns):
    """Only invocation and complete output materialization belong in this timer."""
    start = clock_ns()
    result = materialize(invoke())
    elapsed = clock_ns() - start
    if elapsed <= 0:
        raise RuntimeError("Timer did not advance")
    return result, elapsed


def summarize_pairs(rows):
    if len(rows) != MEASURED_CYCLES:
        raise ValueError("Incomplete paired timing")
    ratios = []
    for index, row in enumerate(rows):
        if row["round"] != index or tuple(row["order"]) != paired_order(index):
            raise ValueError("Changed paired order")
        for variant in VARIANTS:
            samples = row["elapsed_ns"][variant]
            if len(samples) != len(TIMING_MAPPINGS) or not all(type(x) is int and x > 0 for x in samples):
                raise ValueError("Incomplete positive per-call timings")
        ratios.append(sum(row["elapsed_ns"]["reference"]) / sum(row["elapsed_ns"]["candidate"]))
    return {"paired_reference_over_candidate": ratios, "median_paired_ratio": statistics.median(ratios),
            "scope": TIMING_SCOPE, "model_speedup_claim": False}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources():
    for name, expected in FROZEN_SOURCES.items():
        if digest_file(ROOT / name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    return dict(FROZEN_SOURCES)


def workload():
    return {"schema": "qwen35_predicated_slot_primitive_source_v1",
            "layouts": [layout(name) for name, _, _ in LAYOUTS],
            "correctness_mappings": [list(row) for row in CORRECTNESS_MAPPINGS],
            "timing_mappings": [list(row) for row in TIMING_MAPPINGS], "profiles": list(PROFILES),
            "modes": list(MODES), "variants": list(VARIANTS),
            "warmup_cycles": WARMUP_CYCLES, "measured_cycles": MEASURED_CYCLES,
            "counts": counts(), "resources": dict(RESOURCES), "timing_scope": TIMING_SCOPE,
            "full_model_qualified": False, "source_only": True,
            "compatible_current_serial_dispatcher": False,
            "lfm105_tasks_replaced": False}
