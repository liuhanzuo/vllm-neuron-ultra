"""Bounded stdlib-only semantic/index contract checks; no native imports."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

from spark26_native_contract_v1 import LAYER_TYPES, cache_layout, mask_oracle
from spark26_port_contract_v1 import SlotOracle, capacity, index_plan, specification, tp2_layout

ROOT = Path(__file__).resolve().parent


def main():
    checks = []
    def check(ok, label):
        if not ok:
            raise AssertionError(label)
        checks.append(label)
    def rejected(fn, label):
        try:
            fn()
        except (ValueError, RuntimeError):
            checks.append(label)
        else:
            raise AssertionError(label)

    spec = specification()
    # Compare against the pre-existing independently pinned native oracle.
    cases = [(0, 1), (0, 510), (0, 513), (510, 1), (511, 1),
             (512, 1), (513, 17), (1022, 1), (1023, 17), (4095, 1)]
    for layer, kind in enumerate(LAYER_TYPES):
        for seen, query in cases:
            for pad in (0, 2) if seen + query > 2 else (0,):
                plan = index_plan(seen, query, kind, pad)
                expected, raw_mask = mask_oracle(seen, query, kind, pad)
                actual_mask = bytes(int(v) for row in plan['mask'] for v in row)
                check(actual_mask == raw_mask, f'mask:{layer}:{seen}:{query}:{pad}')
                check(plan['keys'] == list(range(expected['kv_offset'], seen + query)),
                      f'kv-order:{layer}:{seen}:{query}:{pad}')
                stored = cache_layout(seen, query, kind)
                check(plan['retained'] == list(range(stored['stored_offset'], seen + query)),
                      f'stored-range:{layer}:{seen}:{query}:{pad}')
                check(len(set(plan['storage'])) == len(plan['storage']),
                      f'no-storage-collision:{layer}:{seen}:{query}:{pad}')

    oracle = SlotOracle()
    initial = [1, 510, 511, 1023]
    for slot, length in enumerate(initial):
        oracle.prefill(slot, length)
    for step in range(20):
        order = [3, 1, 0, 2] if step % 2 else [0, 2, 3, 1]
        out = oracle.decode(order, [initial[s] + step for s in order])
        for slot, layers in zip(order, out):
            p = initial[slot] + step
            for layer, values in enumerate(layers):
                first = max(0, p - 511) if LAYER_TYPES[layer] == 'sliding_attention' else 0
                check(values['values'] == [(1, k) for k in range(first, p + 1)],
                      f'heterogeneous-slot:{step}:{slot}:{layer}')
    old_epoch = oracle.epoch[2]
    oracle.prefill(2, 3)
    reused = oracle.decode([2], [3])[0]
    check(oracle.epoch[2] == old_epoch + 1, 'reuse-epoch')
    for layer, row in enumerate(reused):
        check(row['values'] == [(2, p) for p in range(4)], f'reuse-clears-old-history:{layer}')
    saved = copy.deepcopy(oracle.rows)
    rejected(lambda: oracle.decode([0, 0], [21, 21]), 'reject-duplicate-slot')
    rejected(lambda: oracle.decode([0, 1], [21, 999]), 'reject-stale-batch-before-mutation')
    check(saved == oracle.rows, 'rejected-batch-atomic')
    rejected(lambda: index_plan(4096, 1, 'full_attention'), 'reject-context-overflow')
    rejected(lambda: index_plan(0, 1, 'unknown'), 'reject-unknown-layer')
    rejected(lambda: oracle.prefill(4, 1), 'reject-slot-overflow')

    # Independently validate chronological recovery from the compact physical
    # ring and prefill replacement formulas described by the tensor source.
    for length in (1, 2, 510, 511, 512, 513, 1022, 1023, 4095):
        ring = {}
        for physical in range(511):
            absolute = length - 1 - ((length - 1 - physical) % 511)
            ring[physical] = absolute if absolute >= 0 else None
        recovered = [ring[p % 511] for p in range(max(0, length - 511), length)]
        check(recovered == list(range(max(0, length - 511), length)), f'physical-ring:{length}')
        # A full-attention cache must not inherit SWA's modulo operation.
        if length > 511:
            check(length % capacity('full_attention') != length % capacity('sliding_attention'),
                  f'negative-shared-swa-modulo:{length}')

    layout = tp2_layout()
    for field, start, end in [('q', 0, 4096), ('k', 4096, 5120), ('v', 5120, 6144),
                              ('gate', 0, 16), ('out_columns', 0, 4096), ('mlp', 0, 10240)]:
        joined = [i for rank in layout for i in range(*rank[field])]
        check(joined == list(range(start, end)), 'TP2-complete-nonoverlap:' + field)
    rank0 = [i for field in ('q', 'k', 'v') for i in range(*layout[0][field])]
    check(rank0 != list(range(3072)), 'negative-naive-packed-QKV-halving')

    names = ['spark26_port_contract_v1.py', 'spark26_text_port_v1.py',
             'spark26_neuron_adapter_v1.py', 'spark26_vllm_registration_v1.py',
             'check_spark26_port_source_v1.py']
    hashes = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        ast.parse(raw, filename=name)
        hashes[name] = hashlib.sha256(raw).hexdigest()
        checks.append('AST:' + name)
    adapter = ast.parse((ROOT / names[2]).read_text())
    check(isinstance(adapter.body[2], ast.Expr) and
          isinstance(adapter.body[2].value, ast.Call) and
          isinstance(adapter.body[2].value.func, ast.Name) and
          adapter.body[2].value.func.id == 'require_runtime_qualification',
          'adapter-guard-before-native-imports')
    from spark26_vllm_registration_v1 import install
    rejected(install, 'registration-fails-closed')
    check(not any(n in sys.modules for n in ('torch', 'numpy', 'transformers', 'neuronx_distributed_inference')),
          'no-native-modules-imported')
    return dict(passed=True, scope='stdlib index/mask/source checks only', checks=len(checks),
                specification=spec, source_sha256=hashes, native_tensor_execution=False,
                neuron_execution=False, weights_loaded=False, sdk_abi_qualified=False,
                full_model_qualified=False, speedup_claim=False)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
