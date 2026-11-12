"""Unchanged complete scientific matrix with a separately named integer reference.

All schedule/timing/comparator functions below are the frozen original objects.
Only reference identity and additive source/evidence bindings are new.
"""
import copy
from pathlib import Path
import qwen35_predicated_slot_protocol_v1 as frozen

ROOT = Path(__file__).resolve().parent
BUCKETS, LAYOUTS = frozen.BUCKETS, frozen.LAYOUTS
CORRECTNESS_MAPPINGS, TIMING_MAPPINGS = frozen.CORRECTNESS_MAPPINGS, frozen.TIMING_MAPPINGS
PROFILES, MODES, VARIANTS = frozen.PROFILES, frozen.MODES, frozen.VARIANTS
WARMUP_CYCLES, MEASURED_CYCLES = frozen.WARMUP_CYCLES, frozen.MEASURED_CYCLES
FLAGS, RESOURCES, TIMING_SCOPE = frozen.FLAGS, dict(frozen.RESOURCES), frozen.TIMING_SCOPE
layout, counts, paired_order, sequences = frozen.layout, frozen.counts, frozen.paired_order, frozen.sequences
compare_storage, measured_call, digest_file = frozen.compare_storage, frozen.measured_call, frozen.digest_file
FROZEN_SOURCES = {**frozen.FROZEN_SOURCES, 'qwen35_integer_slot_reference_v1.py': '2d07a14d1e25fc6668f2df5104df9c110338a86a115b7135fb46f1fd4b5c3538'}
REFERENCE_EVIDENCE = [{'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-predicated-slot-r1-root-actual-exit-20261112-v2.json', 'sha256': '6e07d96c6da69cfeb141d020e251cac8376ae007113a240d6d204db387318769'}, {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-slot-select-probe-r2-root-actual-exit-20261112-v1.json', 'sha256': '6fe2fcba9ea46142aa1f9910e7bebb1f093e6e6859bab33aee77223f1fe86227'}, {'path': '/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-slot-select-probe-r2-root-finalization-20261112-v4.json', 'sha256': '90369dd4945978f90032136ee7cb329779c73fea1350d791de566750e1dc1fba'}]
REFERENCE_IMPLEMENTATION = dict(
    name='integer_bitcast_store_slots_v1', floating_storage_types=['bfloat16', 'float32'],
    integer_select_types=['s16', 's32'], numeric_conversion=False,
    candidate='unchanged_store_slots_nki_predicated', original_old_value_gather_unchanged=True,
    original_floating_reference_failed_status_preserved=True, original_model_baseline_speedup_claim=False)


def verify_sources():
    frozen.verify_sources()
    for name, expected in FROZEN_SOURCES.items():
        if digest_file(ROOT / name) != expected:
            raise RuntimeError('Frozen integer-reference source changed: ' + name)
    for row in REFERENCE_EVIDENCE:
        if digest_file(row['path']) != row['sha256']:
            raise RuntimeError('Original failed/reference diagnostic evidence changed')
    return dict(FROZEN_SOURCES)


def summarize_pairs(rows):
    return dict(frozen.summarize_pairs(rows), reference_implementation=copy.deepcopy(REFERENCE_IMPLEMENTATION))


def workload():
    result = frozen.workload()
    result.update(schema='qwen35_integer_reference_primitive_source_v1',
        reference_implementation=copy.deepcopy(REFERENCE_IMPLEMENTATION),
        reference_evidence=copy.deepcopy(REFERENCE_EVIDENCE))
    return result
