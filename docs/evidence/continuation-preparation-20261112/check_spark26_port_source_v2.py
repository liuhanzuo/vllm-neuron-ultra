"""Stdlib scalar/AST/mocked-interface checks, never native tensor/model tests."""
import ast
import contextlib
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from spark26_native_contract_v1 import LAYER_TYPES, mask_oracle
from spark26_port_contract_v1 import SlotOracle
from spark26_port_contract_v2 import (
    SDK_SITE, TRACE_INPUTS, alias_layout, checkpoint_schema, converted_key,
    index_plan, padded_decode_rows, scalar_masked_write, specification,
    tp2_layout, validate_host_request,
)

ROOT = Path(__file__).resolve().parent


def extracted(path, name, environment):
    """Execute only one selected function AST with stdlib mocks, never its module."""
    tree = ast.parse(path.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise AssertionError('Expected exactly one function ' + name)
    node = copy.deepcopy(functions[0]); node.decorator_list = []; node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(environment)
    exec(compile(module, str(path) + ':' + name, 'exec'), scope)
    return scope[name]


class Scalar(int):
    def long(self):
        return self


class Mask:
    def __init__(self, values):
        self.values = list(values)

    def __and__(self, other):
        return Mask([value and bool(other) for value in self.values])

    def __getitem__(self, item):
        return self


class Slots:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)

    def long(self):
        return self

    def __eq__(self, scalar):
        return Mask([value == scalar for value in self.values])

    def __getitem__(self, index):
        return Scalar(self.values[index])


class Rows:
    """Opaque row markers; this is not a numerical tensor implementation."""
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values), 1, 1, 1)
        self.device = 'scalar-mock'

    def __getitem__(self, index):
        return Rows(self.values[index])


class RowTorch:
    @staticmethod
    def arange(size, **kwargs):
        return Slots(range(size))

    @staticmethod
    def where(mask, new, old):
        return Rows([new.values[0] if selected else value for selected, value in zip(mask.values, old.values)])


class Weight:
    def __init__(self, shape, marker, dtype='bf16'):
        self.shape = tuple(shape); self.marker = marker; self.dtype = dtype

    def clone(self):
        return Weight(self.shape, self.marker, self.dtype)


class WeightTorch:
    bfloat16 = 'bf16'
    int32 = 'int32'

    @staticmethod
    def equal(a, b):
        return a.shape == b.shape and a.marker == b.marker and a.dtype == b.dtype

    @staticmethod
    def arange(size, dtype):
        return Weight((size,), 'ranks', dtype)


class Packed:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shape = (1, 1, len(self.rows))


class PackedTorch:
    @staticmethod
    def split(value, sizes, dim=0):
        if isinstance(sizes, int):
            if len(value.rows) % sizes:
                raise AssertionError('Nondivisible mocked shard')
            sizes = [sizes] * (len(value.rows) // sizes)
        result = []; start = 0
        for size in sizes:
            result.append(Packed(value.rows[start:start + size])); start += size
        if start != len(value.rows):
            raise AssertionError('Mock split did not consume all rows')
        return result

    @staticmethod
    def cat(values, dim=0, out=None):
        return Packed([row for value in values for row in value.rows])

    @staticmethod
    def no_grad():
        return contextlib.nullcontext()

    @staticmethod
    def tensor_split(value, boundaries, dim=2):
        bounds = (0, *boundaries, len(value.rows))
        return [Packed(value.rows[a:b]) for a, b in zip(bounds, bounds[1:])]


def main():
    checks = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    def rejects(fn, name):
        try:
            fn()
        except (ValueError, RuntimeError):
            checks.append(name)
        else:
            raise AssertionError(name)

    adapter = ROOT / 'spark26_neuron_adapter_v2.py'
    tensor_source = ROOT / 'spark26_text_port_v2.py'
    masked_write = extracted(tensor_source, 'masked_slot_write', {'torch': RowTorch})
    # Every placement and permutation of 0..4 distinct active slots amongst
    # four graph rows, including -1 rows before/after active slot0.
    arrangements = 0
    for active_count in range(5):
        for active_slots in itertools.permutations(range(4), active_count):
            for locations in itertools.combinations(range(4), active_count):
                slots = [-1] * 4
                for location, slot in zip(locations, active_slots):
                    slots[location] = slot
                old = [('old', s) for s in range(4)]
                new = [('new', row, slot) for row, slot in enumerate(slots)]
                wanted = scalar_masked_write(old, new, slots)
                got = masked_write(Rows(old), Rows(new), Slots(slots)).values
                check(got == wanted, 'masked-row-placement:' + str(slots))
                arrangements += 1
    check(arrangements == 209, 'all209-partial-batch-arrangements')
    old = ['old0', 'old1', 'old2', 'old3']
    naive = old.copy()
    for slot, value in zip([0, -1, -1, -1], ['active0', 'bad1', 'bad2', 'bad3']):
        naive[max(0, slot)] = value
    wanted = scalar_masked_write(old, ['active0', 'bad1', 'bad2', 'bad3'], [0, -1, -1, -1])
    check(naive != wanted and wanted[0] == 'active0', 'negative-clamp0-index-copy-corruption')
    rejects(lambda: scalar_masked_write(old, [1, 2], [0, 0]), 'reject-duplicate-active-slot')
    rejects(lambda: scalar_masked_write(old, [1], [-2]), 'reject-invalid-negative-slot')

    # Exercise changing active sets through the original independent logical
    # oracle and check that the actual masked writer AST preserves every idle row.
    oracle = SlotOracle()
    for slot, length in enumerate((1, 510, 511, 1023)):
        oracle.prefill(slot, length)
    patterns = [[0], [3, 0], [2], [3, 1, 0, 2], [1, 0], [3]]
    for step in range(12):
        slots = patterns[step % len(patterns)]
        positions = [oracle.rows[s]['seen'] for s in slots]
        graph_slots, graph_positions = padded_decode_rows(slots, positions)
        old_rows = copy.deepcopy(oracle.rows)
        oracle.decode(slots, positions)
        for layer, kind in enumerate(LAYER_TYPES):
            prior = [old_rows[s]['cache'][layer] for s in range(4)]
            updates = [oracle.rows[s]['cache'][layer] if s >= 0 else {'poison': 'dummy-clamped-slot0'}
                       for s in graph_slots]
            got = masked_write(Rows(prior), Rows(updates), Slots(graph_slots)).values
            check(got == [oracle.rows[s]['cache'][layer] for s in range(4)],
                  f'mixed-cache-idle-preservation:{step}:{layer}')
            for slot, p in zip(slots, positions):
                plan = index_plan(p, 1, kind)
                _, expected_mask = mask_oracle(p, 1, kind)
                check(bytes(int(x) for x in plan['mask'][0]) == expected_mask,
                      f'cross511-mask:{step}:{layer}:{slot}')
        check([graph_positions[s] for s in sorted(slots)] == [oracle.rows[s]['seen'] - 1 for s in sorted(slots)],
              'slot-sort-position-pairing:' + str(step))
    before_reuse = copy.deepcopy(oracle.rows)
    oracle.prefill(0, 1)
    for layer in range(36):
        prior = [before_reuse[s]['cache'][layer] for s in range(4)]
        got = masked_write(Rows(prior), Rows([oracle.rows[0]['cache'][layer]]), Slots([0])).values
        check(got == [oracle.rows[s]['cache'][layer] for s in range(4)], 'one-token-slot0-reuse:' + str(layer))

    host_cases = [([0], 1, [1]), ([3], 2, [1]), ([3, 0], 513, [511, 513]),
                  ([2, 1, 3, 0], 1024, [1, 511, 512, 1024]), ([1], 4096, [4096])]
    for slots, width, lengths in host_cases:
        tokens = [[7] * width for _ in slots]
        positions = [list(range(n)) + [0] * (width - n) for n in lengths]
        masks = [[1] * n + [0] * (width - n) for n in lengths]
        inferred = validate_host_request(tokens, positions, slots)
        explicit = validate_host_request(tokens, positions, slots, masks)
        check(inferred == explicit and inferred['lengths'] == lengths,
              'right-padding-host-lengths:' + str(lengths))
        check(inferred['pad_single_prompt'] == (width == 1), 'one-token-prefill-padding:' + str(lengths))
        # Graph prefill positions must be physical arange, not repeated zero pads.
        for length in lengths:
            graph_width = max(2, width)
            cache_positions = list(range(graph_width))
            check(cache_positions[:length] == list(range(length)), 'prefill-physical-positions:' + str(length))
    rejects(lambda: validate_host_request([[7, 7]], [[0, 0]], [0], [[0, 1]]), 'reject-leading-padding')
    rejects(lambda: validate_host_request([[7, 7, 7]], [[0, 2, 3]], [0]), 'reject-noncontiguous-prefill')
    rejects(lambda: validate_host_request([[7], [7]], [[0], [9]], [0, 1]), 'reject-mixed-prefill-decode')
    rejects(lambda: validate_host_request([[7]], [[4096]], [0]), 'reject-context-overflow')
    rejects(lambda: validate_host_request([[7]], [[1]], [-1]), 'reject-host-dummy-slot')

    # Execute only converter/hook AST against shape/identity markers, no weights.
    converter = extracted(adapter, 'convert_unprefixed_state_dict', {
        'checkpoint_schema': checkpoint_schema, 'converted_key': converted_key, 'torch': WeightTorch})
    tie = extracted(adapter, 'update_state_dict_for_tied_weights', {})
    schema = checkpoint_schema()
    state = {key: Weight(shape, key) for key, shape in schema.items()}
    state['lm_head.weight'] = state['embedding.weight']
    for omit_head in (False, True):
        supplied = {k: v for k, v in state.items() if not (omit_head and k == 'lm_head.weight')}
        converted = converter(supplied); tie(converted)
        check(len(converted) == 292, 'converted-key-count:' + str(omit_head))
        check(converted['lm_head.weight'].marker == converted['embedding.weight'].marker and
              converted['lm_head.weight'] is not converted['embedding.weight'], 'tied-values-independent-shards:' + str(omit_head))
        for layer in range(36):
            check(converted[f'layers.{layer}.self_attn.qkv_proj.Wqkv.weight'].shape == (6144, 2560),
                  f'packed-checkpoint-shape:{omit_head}:{layer}')
    bad = dict(state); bad['lm_head.weight'] = Weight((131072, 2560), 'wrong-head')
    rejects(lambda: converter(bad), 'reject-untied-original-head')
    bad = dict(state); bad['layers.0.self_attn.q_k_v_proj.weight'] = Weight((6143, 2560), 'wrong')
    rejects(lambda: converter(bad), 'reject-qkv-shape')
    bad = dict(state); bad['norm.weight'] = Weight((2560,), 'norm', 'fp32')
    rejects(lambda: converter(bad), 'reject-unplanned-weight-dtype')
    rejects(lambda: converter({'model.' + k: v for k, v in state.items()}), 'reject-unstripped-prefix')

    # Actual pinned SDK source functions, executed only on opaque row lists.
    trace_source = SDK_SITE / 'neuronx_distributed/trace/trace.py'
    gqa_source = SDK_SITE / 'neuronx_distributed_inference/modules/attention/gqa.py'
    shard = extracted(trace_source, 'create_local_weight_qkv', {'torch': PackedTorch, 'divide': lambda a, b: a // b})
    logger = SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None)
    split = extracted(gqa_source, '_split_fused_qkv', {'torch': PackedTorch, 'logger': logger})
    native_qkv = extracted(gqa_source, '_native_qkv_forward', {'logger': logger, 'is_lora_module': lambda x: False})
    head_counts = extracted(gqa_source, 'get_shardable_head_counts', {
        'get_number_of_extra_heads': lambda h, tp: (-h) % tp,
        'GQA': SimpleNamespace(REPLICATE_TO_TP_DEGREE='replicate', CONVERT_TO_MHA='mha')})
    check(head_counts(2, 16, 4, 'mha') == (16, 4), 'GQA-TP2-does-not-replicate-KV')
    all_rows = Packed(range(6144)); layout = tp2_layout()
    for rank in range(2):
        local = shard(rank, 2, all_rows, 0, 4096, 1024)
        expected = [row for field in ('q', 'k', 'v') for row in range(*layout[rank][field])]
        check(local.rows == expected, 'SDK-packed-sharding-rank:' + str(rank))
        calls = []
        def project(x):
            calls.append(x)
            return local
        owner = SimpleNamespace(fused_qkv=True, clip_qkv=None, Wqkv=project,
                                num_attention_heads=16, num_key_value_heads=4, head_dim=256, tp_degree=2)
        owner._split_fused_qkv = lambda value: split(owner, value)
        q, k, v = native_qkv(owner, 'opaque-hidden-input')
        check(len(calls) == 1 and [len(x.rows) for x in (q, k, v)] == [2048, 512, 512],
              'SDK-one-packed-projection-and-local-split:' + str(rank))

    aliases = alias_layout()
    check([row[0] for row in aliases] == list(range(1, 73)), '72-contiguous-cache-alias-outputs')
    check([row[3] for row in aliases] == [511 if kind == 'sliding_attention' else 4096
                                       for kind in LAYER_TYPES for _ in range(2)], 'mixed-cache-alias-capacities')
    parsed = ast.parse(adapter.read_text())
    model = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'NeuronSparkModel')
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    check(tuple(arg.arg for arg in forward.args.args[1:]) == TRACE_INPUTS and
          forward.args.vararg is None and forward.args.kwarg is None, 'exact-seven-trace-inputs')
    check(isinstance(parsed.body[2], ast.Expr) and parsed.body[2].value.func.id == 'require_runtime_admission',
          'adapter-runtime-guard-before-native-imports')
    check('index_copy' not in tensor_source.read_text(), 'no-clamped-dummy-index-copy-route')
    tensor_ast = ast.parse(tensor_source.read_text())
    decode = next(n for n in tensor_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'decode_cache')
    check(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'long'
              for n in ast.walk(decode)), 'decode-indices-explicit-long-cast')
    from spark26_vllm_registration_v2 import install
    rejects(install, 'registration-still-fails-before-binding')
    names = ['spark26_port_contract_v2.py', 'spark26_text_port_v2.py', 'spark26_neuron_adapter_v2.py',
             'spark26_vllm_registration_v2.py', 'check_spark26_port_source_v2.py']
    sources = {}
    for name in names:
        raw = (ROOT / name).read_bytes(); ast.parse(raw)
        sources[name] = hashlib.sha256(raw).hexdigest(); checks.append('AST:' + name)
    check(not any(name in sys.modules for name in ('torch', 'numpy', 'transformers', 'neuronx_distributed_inference')),
          'no-native-imports')
    return dict(passed=True, checks=len(checks), partial_batch_arrangements=arrangements,
                scope='stdlib scalar/AST and selected-function mocks only; no native tensor/model checks',
                specification=specification(), source_sha256=sources,
                sdk_mocked_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in (trace_source, gqa_source)},
                native_executed=False, weights_loaded=False, model_qualified=False,
                runtime_admitted=False, speedup_claim=False)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
