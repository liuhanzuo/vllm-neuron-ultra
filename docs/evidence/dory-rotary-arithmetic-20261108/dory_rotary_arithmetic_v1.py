"""Dory native RoPE arithmetic comparison preparation; no native import or CLI.

Only a future independently admitted serial worker may call run_comparison with
its actual torch module and live resource guard. This extracts unchanged pinned
function bodies; it does NOT import official vLLM or qualify backend dispatch.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

Q = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
PINS = {
    'base': (Q/'dory-rotary-source-20261108-v1/base.py', 'c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5'),
    'common': (Q/'dory-rotary-source-20261108-v1/common.py', 'af4825138c1d8ee3823bfad04dadd91f6657818f867756c676faeb9d14d64f02'),
    'candidate': (Q/'dory-native-reference-source-20261107-v1/dory_reference_candidate.py', '366dc0ab95bbeebb3bd2311f205e23afc63ce3fbbcfca321345185dcbc191716'),
    'config': (Q/'dory-checkpoint-metadata-20261108-v1/config.json', 'd2d086f5949598a043a00d70e2c28c45305bca7703682de3e4f990a7fd3279a4'),
}
SELECTION = (
    ('common', 'ApplyRotaryEmb', 'forward_static'),
    ('base', 'RotaryEmbeddingBase', '_compute_inv_freq'),
    ('base', 'RotaryEmbeddingBase', '_compute_cos_sin_cache'),
    ('base', 'RotaryEmbeddingBase', '_match_cos_sin_cache_dtype'),
    ('base', 'RotaryEmbedding', 'forward_static'),
    ('base', 'RotaryEmbedding', 'forward_native'),
    ('candidate', None, 'neox_rope_candidate'),
)
POSITIONS = (0, 1, 1023, 1024, 2048, 262143)
DTYPES = ('float32', 'bfloat16')
CASES = ('signed_ramp', 'binary_scale_cancellation')
THETAS = (10000, 10000000)
EXPECTED_COMPARISONS = 96  # 2 theta * 2 dtype * 2 fixtures * Q/K * 6 positions


def require(value, message):
    if not value:
        raise ValueError(message)


def read_pinned(path, sha256):
    require(path.is_absolute() and path.resolve() == path and
            not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical pinned source')
    with path.open('rb') as stream:
        raw = stream.read(65537)
    require(len(raw) <= 65536, 'Bounded pinned source')
    require(hashlib.sha256(raw).hexdigest() == sha256, 'Exact pinned source digest')
    return raw


def select_function(text, class_name, name):
    tree = ast.parse(text)
    parent = tree
    if class_name is not None:
        matches = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        require(len(matches) == 1, 'Unique original class')
        parent = matches[0]
    matches = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(matches) == 1, 'Unique original function')
    node = copy.deepcopy(matches[0])
    require(not node.decorator_list or (len(node.decorator_list) == 1 and
            isinstance(node.decorator_list[0], ast.Name) and node.decorator_list[0].id == 'staticmethod'),
            'Only staticmethod binding may be removed')
    node.decorator_list = []
    return node


def prepare_sources():
    texts, refs = {}, {}
    for key, (path, digest) in PINS.items():
        raw = read_pinned(path, digest)
        texts[key] = raw.decode('utf-8')
        refs[key] = dict(path=str(path), sha256=digest, bytes=len(raw))
    config = json.loads(texts['config'])
    require((config['num_attention_heads'], config['num_key_value_heads'], config['head_dim'],
             config['max_position_embeddings']) == (16, 4, 256, 262144), 'Original RoPE dimensions')
    require(tuple(config['rope_parameters'][key]['rope_theta'] for key in
                  ('sliding_attention', 'full_attention')) == THETAS, 'Both original theta values')
    require(all(v['rope_type'] == 'default' and v['partial_rotary_factor'] == 1.0
                for v in config['rope_parameters'].values()), 'Original full-dimensional default RoPE')
    records, nodes = [], {}
    for key, cls, name in SELECTION:
        node = select_function(texts[key], cls, name)
        ident = '.'.join(x for x in (key, cls, name) if x)
        nodes[ident] = node
        records.append(dict(identity=ident, source=refs[key], line=node.lineno,
                            end_line=node.end_lineno,
                            body_ast_sha256=hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()))
    return texts, nodes, dict(sources=refs, extracted_functions=records,
                             extraction_scope='Unchanged function bodies; no module import or backend dispatch')


def bind_functions(nodes, torch):
    """Definition binding only. Future annotations avoid native work at binding."""
    functions = {}
    for ident, node in nodes.items():
        namespace = dict(torch=torch)
        if ident.startswith('base.RotaryEmbedding.'):
            namespace['ApplyRotaryEmb'] = SimpleNamespace(
                forward_static=functions['common.ApplyRotaryEmb.forward_static'])
        tree = ast.Module(body=[ast.ImportFrom(module='__future__',
                          names=[ast.alias(name='annotations')], level=0), copy.deepcopy(node)], type_ignores=[])
        ast.fix_missing_locations(tree)
        exec(compile(tree, '<pinned-dory-rotary:' + ident + '>', 'exec', dont_inherit=True), namespace)
        functions[ident] = namespace[node.name]
    return functions


def plan():
    _, _, proof = prepare_sources()
    return dict(schema='dory_rotary_arithmetic_plan_v1', positions=list(POSITIONS),
                theta=list(THETAS), dtypes=list(DTYPES), fixtures=list(CASES),
                query_heads=16, key_heads=4, head_dim=256, original_cache_positions=262144,
                expected_position_comparisons=EXPECTED_COMPARISONS,
                acceptance='Every finite output value must match exactly AND packed output bytes must match; no tolerance relaxation',
                native_cache_policy='Build full original FP32 cache via unchanged upstream methods, then cast to configured dtype',
                native_cache_fp32_bytes=262144*256*4,
                scope='Extracted native arithmetic only; official module import/dispatch, CPU fused kernel and model remain unqualified',
                source_only=True, runtime_admitted=False, native_executed=False,
                numerical_parity_proven=False, native_reference_qualified=False, new_speedup_claim=False,
                provenance=proof)


def tensor_hash(tensor, torch):
    raw = bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())
    return hashlib.sha256(raw).hexdigest()


def input_fixture(case, heads, dtype, torch):
    count = len(POSITIONS) * heads * 256
    index = torch.arange(count, dtype=torch.int64)
    if case == 'signed_ramp':
        value = ((index % 509) - 254).to(torch.float32) / 32
    elif case == 'binary_scale_cancellation':
        pattern = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.00390625, -0.00390625,
                                0.5, -0.5, 1.0078125, -1.0078125, 128.0, -128.0,
                                0.99609375, -0.99609375, 0.03125, -0.03125], dtype=torch.float32)
        value = pattern[index % pattern.numel()]
    else:
        raise ValueError('Only frozen input fixtures')
    return value.to(dtype).reshape(len(POSITIONS), heads, 256)


def compare_position(actual, expected, torch):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype and
            actual.device.type == expected.device.type == 'cpu', 'Comparable original output tensors')
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    actual_hash, expected_hash = tensor_hash(actual, torch), tensor_hash(expected, torch)
    result = dict(shape=list(actual.shape), dtype=str(actual.dtype), all_finite=finite,
                  actual_sha256=actual_hash, expected_sha256=expected_hash,
                  bitwise_equal=actual_hash == expected_hash,
                  unequal_values=int((actual != expected).sum().item()))
    if finite:
        delta = (actual.double() - expected.double()).abs()
        denominator = expected.double().abs()
        nonzero = denominator != 0
        result.update(max_abs_error=delta.max().item(), mean_abs_error=delta.mean().item(),
                      max_relative_nonzero=(delta[nonzero]/denominator[nonzero]).max().item() if nonzero.any().item() else 0.0,
                      nonzero_against_zero=int(((actual != 0) & (expected == 0)).sum().item()))
    result['passed'] = finite and result['bitwise_equal'] and result['unequal_values'] == 0
    return result


def run_comparison(*, torch, guard):
    """Future serial runtime callback; does not provide admission or launch jobs.

    Requires CPU, one-thread eager execution; full native cache built sequentially.
    The caller must enforce measured memory/time limits, retained output and lease.
    A failed comparison is returned with passed=False and all diagnostic rows.
    """
    require(callable(guard), 'Actual serial runtime guard required')
    guard()
    require(torch.get_default_device().type == 'cpu' and torch.get_num_threads() == 1 and
            torch.get_num_interop_threads() == 1, 'Admitted one-thread CPU runtime')
    require(not torch.compiler.is_compiling(), 'Eager original arithmetic')
    texts, nodes, proof = prepare_sources()
    functions = bind_functions(nodes, torch)
    candidate = functions['candidate.neox_rope_candidate']
    rows, cases, cache_proofs = [], [], []
    positions = torch.tensor(POSITIONS, dtype=torch.long, device='cpu')
    with torch.inference_mode():
        for theta in THETAS:
            for dtype_name in DTYPES:
                guard()
                dtype = getattr(torch, dtype_name)
                reference = SimpleNamespace(head_size=256, rotary_dim=256, max_position_embeddings=262144,
                                            base=theta, is_neox_style=True, use_aiter=False)
                for name in ('_compute_inv_freq', '_compute_cos_sin_cache', '_match_cos_sin_cache_dtype'):
                    setattr(reference, name, MethodType(functions['base.RotaryEmbeddingBase.'+name], reference))
                reference.forward_static = functions['base.RotaryEmbedding.forward_static']
                reference.cos_sin_cache = reference._compute_cos_sin_cache().to(dtype)
                require(list(reference.cos_sin_cache.shape) == [262144, 256] and
                        reference.cos_sin_cache.dtype == dtype, 'Complete original cache')
                guard()
                for first in range(0, 262144, 1024):
                    require(bool(torch.isfinite(reference.cos_sin_cache[first:first+1024]).all().item()), 'Finite complete native cache')
                    guard()
                selected = reference.cos_sin_cache.index_select(0, positions)
                cache_proofs.append(dict(theta=theta, dtype=dtype_name, shape=[262144, 256],
                                         selected_positions_sha256=tensor_hash(selected, torch)))
                del selected
                for case in CASES:
                    guard()
                    q = input_fixture(case, 16, dtype, torch)
                    k = input_fixture(case, 4, dtype, torch)
                    before = [tensor_hash(t, torch) for t in (q, k)]
                    actual = candidate(q, k, positions, theta)
                    guard()
                    expected = functions['base.RotaryEmbedding.forward_native'](reference, positions, q, k)
                    guard()
                    require(before == [tensor_hash(t, torch) for t in (q, k)], 'Neither arithmetic path mutates original inputs')
                    cases.append(dict(theta=theta, dtype=dtype_name, fixture=case,
                                      input_sha256=dict(query=before[0], key=before[1])))
                    for kind, a, e in zip(('query', 'key'), actual, expected):
                        require(list(a.shape) == list(e.shape) == [6, 16 if kind == 'query' else 4, 256], 'Original Q/K output shape')
                        for i, position in enumerate(POSITIONS):
                            guard()
                            rows.append(dict(theta=theta, dtype=dtype_name, fixture=case, tensor=kind,
                                             position=position, **compare_position(a[i], e[i], torch)))
                    del q, k, actual, expected, a, e
                # Bound methods point back to the namespace; break that cycle so
                # the full cache is released before building the next case.
                reference.__dict__.clear()
                del reference
    guard()
    require(len(rows) == EXPECTED_COMPARISONS, 'Complete frozen comparison matrix')
    # Detect any source/config modification during the native comparison.
    require(prepare_sources()[2] == proof, 'Original pinned sources unchanged at completion')
    return dict(schema='dory_rotary_arithmetic_result_v1', complete=True,
                passed=all(row['passed'] for row in rows), comparisons=rows,
                input_cases=cases, cache_proofs=cache_proofs, provenance=proof,
                torch_version=str(torch.__version__), framework_file=str(torch.__file__),
                positions=list(POSITIONS), theta=list(THETAS), dtypes=list(DTYPES),
                acceptance='Exact finite values and packed bytes',
                native_arithmetic_executed=True, official_module_dispatch_executed=False,
                fused_cpu_kernel_executed=False, original_weights_loaded=False,
                full_model_executed=False, native_reference_qualified=False, new_speedup_claim=False)
