"""Source-only Spark port contract. No framework imports or runtime admission."""
from pathlib import Path

from spark26_native_contract_v1 import LAYER_TYPES, config as official_config

ROOT = Path(__file__).resolve().parent
SDK_SITE = Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
SDK_VERSIONS = {
    'neuronx-distributed-inference': '0.8.16251+f3ca5575',
    'neuronx-distributed': '0.17.26814+4b18de63',
    'vllm-neuron': '0.2.2+lts', 'vllm': '0.11.0',
    'torch': '2.8.0', 'transformers': '4.57.6',
}
TARGET = {'batch_size': 4, 'max_context': 4096, 'tp_degree': 2}
UNVERIFIED_ABI = (
    'NxDI model tracing/output cache alias and sharded checkpoint loading',
    'vLLM request slots and cache_position mapping, including one-token prefill',
    'TP2 collectives, complete original-weight numerics and tied-head loading',
    'Neuron compilation, cold cache, long decode, slot reuse and HTTP qualification',
)


def require_runtime_qualification():
    """Deliberate unconditional guard: no environment flag bypass in this source version."""
    raise RuntimeError('Spark source candidate is not runtime admitted: ' + '; '.join(UNVERIFIED_ABI))


def capacity(kind, max_context=4096):
    if kind not in ('sliding_attention', 'full_attention'):
        raise ValueError('Unknown attention kind')
    if type(max_context) is not int or not 1 <= max_context <= 4096:
        raise ValueError('Unqualified context capacity')
    return 511 if kind == 'sliding_attention' else max_context


def index_plan(seen, query, kind, leading_pad=0):
    """Logical chronological KV and physical cache indices, independent of tensors."""
    if any(type(n) is not int for n in (seen, query, leading_pad)):
        raise ValueError('Integer positions required')
    if seen < 0 or query < 1 or seen + query > 4096 or not 0 <= leading_pad < seen + query:
        raise ValueError('Out-of-contract position or padding')
    cap = capacity(kind)
    first = max(0, seen - 511) if kind == 'sliding_attention' else 0
    keys = list(range(first, seen + query))
    retained = list(range(max(0, seen + query - cap), seen + query))
    mask = [[k >= leading_pad and k <= q and (kind == 'full_attention' or k > q - 512)
             for k in keys] for q in range(seen, seen + query)]
    return {'keys': keys, 'retained': retained, 'storage': [p % cap for p in retained], 'mask': mask}


def tp2_layout():
    """Source [all Q, all K, all V] rows must be split before rank slicing."""
    return [dict(rank=r, q=(r * 2048, (r + 1) * 2048),
                 k=(4096 + r * 512, 4096 + (r + 1) * 512),
                 v=(5120 + r * 512, 5120 + (r + 1) * 512),
                 gate=(r * 8, (r + 1) * 8), out_columns=(r * 2048, (r + 1) * 2048),
                 mlp=(r * 5120, (r + 1) * 5120)) for r in range(2)]


class SlotOracle:
    """Scalar cache simulation for lineage/slot checks; never a numerical model."""
    def __init__(self):
        self.rows = {}
        self.epoch = [0] * 4

    def prefill(self, slot, length, leading_pad=0):
        if type(slot) is not int or not 0 <= slot < 4:
            raise ValueError('Invalid slot')
        index_plan(0, length, 'full_attention', leading_pad)
        self.epoch[slot] += 1
        row = {'seen': length, 'leading_pad': leading_pad, 'cache': []}
        for kind in LAYER_TYPES:
            plan = index_plan(0, length, kind, leading_pad)
            row['cache'].append({i: (self.epoch[slot], p) for p, i in zip(plan['retained'], plan['storage'])})
        self.rows[slot] = row

    def decode(self, slots, positions):
        if len(slots) != len(positions) or len(set(slots)) != len(slots):
            raise ValueError('Duplicate slot or shape mismatch')
        # Validate the whole batch before mutating any cache.
        for slot, p in zip(slots, positions):
            if slot not in self.rows or p != self.rows[slot]['seen'] or p >= 4096:
                raise ValueError('Missing slot, stale position, or capacity exhausted')
        result = []
        for slot, p in zip(slots, positions):
            row = self.rows[slot]; layers = []
            for i, kind in enumerate(LAYER_TYPES):
                plan = index_plan(p, 1, kind, row['leading_pad']); cap = capacity(kind)
                old = [row['cache'][i][k % cap] for k in plan['keys'][:-1]]
                layers.append({'values': old + [(self.epoch[slot], p)], 'mask': plan['mask'][0]})
                row['cache'][i][p % cap] = (self.epoch[slot], p)
            row['seen'] += 1; result.append(layers)
        return result


def specification():
    return dict(status='source_only_not_runtime_admitted', sdk_site=str(SDK_SITE),
                sdk_versions=SDK_VERSIONS, target=TARGET, official_config=official_config(),
                physical_cache='27 layers of 511-entry ring; 9 layers of 4096 absolute entries',
                original_weight_numerics=False, native_import_tested=False,
                sdk_abi_tested=False, neuron_compilation=False, speedup_claim=False,
                runtime_blockers=list(UNVERIFIED_ABI))
