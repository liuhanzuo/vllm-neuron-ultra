"""Exact output ABI for a new DeepSeek CPU-sampling diagnostic graph.

Pure standard library. This is not a launcher or a model qualification result.
The original graph/config must never be loaded as a capture artifact.
"""
from copy import deepcopy

VERSION = 1
LAYERS = 36
KV_COUNT = 72
KV_SHAPE = (4, 4, 4224, 128)
PREFIXES = (32, 128, 256, 512, 1024, 2048, 2052)
FAILED_INDICES = (33, 35, 36, 38, 40, 42, 45, 47, 48, 50, 60, 62, 68, 70)
BOUNDARIES = (
    'layer_input', 'normalized_input', 'q_normalized', 'k_normalized',
    'query', 'key', 'value', 'attention_heads', 'attention_projected',
    'attention_residual', 'mlp_normalized', 'mlp_output', 'layer_output',
)
NAMES = tuple(f'layers.{layer}.{name}' for layer in range(LAYERS) for name in BOUNDARIES)
CAPTURE_COUNT = len(NAMES)
KV_OFFSET = 1 + CAPTURE_COUNT


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture_config():
    # SDK uses this count when placing KV output aliases. Actual values are
    # returned by our core override, not its first-output-only registry reader.
    return dict(modules_to_capture=[], max_intermediate_tensors=CAPTURE_COUNT,
                auto_capture_moe_tensors=False, capture_inputs=False)


def token_axis(name):
    return 2 if name.rsplit('.', 1)[-1] in ('query', 'key', 'value') else 1


def select_prediction(value, name, indices):
    """Preserve every feature; select exactly one actual query token per row.

Q/K norms are BSHD in the pinned model; prepared Q/K/V are BHSD. A gather
retains the token axis, so physical shard/head dimensions stay explicit.
"""
    axis = token_axis(name)
    shape = list(value.shape)
    require(len(shape) in (3, 4) and shape[0] == indices.shape[0], 'capture batch/layout mismatch')
    shape[axis] = 1
    index_shape = [indices.shape[0]] + [1] * (len(shape) - 1)
    return value.gather(axis, indices.reshape(index_shape).expand(shape))


def boundary_byte_budget():
    # Per TP rank: nine hidden/attention vectors plus separately sized Q/K/V.
    widths = dict(layer_input=4096, normalized_input=4096, q_normalized=2048,
                  k_normalized=512, query=2048, key=512, value=512,
                  attention_heads=2048, attention_projected=4096,
                  attention_residual=4096, mlp_normalized=4096,
                  mlp_output=4096, layer_output=4096)
    local = sum(widths.values()) * LAYERS * 2
    # Native has complete heads for five sharded boundaries plus attention_heads.
    sharded = ('q_normalized', 'k_normalized', 'query', 'key', 'value', 'attention_heads')
    native = local + sum(widths[name] for name in sharded) * LAYERS * 2
    return dict(device_bytes_per_prediction=local * 2, native_bytes_per_prediction=native,
                device84_tensor_bytes=local * 2 * 84, native42_tensor_bytes=native * 42,
                combined_boundary_tensor_bytes=local * 2 * 84 + native * 42,
                boundary_payload_admission_floor_bytes=1 << 30,
                excludes='KV journals, lifecycle snapshots, logits, compilation and serialization overhead')


def check_cpu_config(cfg):
    require(cfg.on_device_sampling_config is None, 'original CPU sampler required')
    require(cfg.output_logits is False, 'exactly one full-logit output required')
    require(cfg.tp_degree == 2, 'original TP2 required')
    require(cfg.batch_size in (1, 4), 'only original CTE B1/TKG B4 supported')
    require(str(cfg.torch_dtype) in ('bfloat16', 'torch.bfloat16'), 'original BF16 required')
    require(not cfg.async_mode and not cfg.sequence_parallel_enabled, 'sync, no sequence parallel')
    for key in ('enable_fused_speculation', 'enable_eagle_speculation', 'is_medusa',
                'is_block_kv_layout', 'quantized', 'kv_cache_quant'):
        require(not getattr(cfg, key, False), 'unsupported capture mode: ' + key)
    require(getattr(cfg, 'cp_degree', 1) == getattr(cfg, 'attention_dp_degree', 1) == 1,
            'original attention partitioning required')


def instrument_config_dict(original):
    """Copy a serialized original config; no file writes and no SDK imports."""
    copied = deepcopy(original)
    n = copied['neuron_config']
    require(n.get('tensor_capture_config') is None, 'expected original uncaptured config')
    require(n.get('on_device_sampling_config') is None and n.get('output_logits') is False,
            'original CPU sampler with one full-logit output required')
    require(n['batch_size'] == 4 and n['ctx_batch_size'] == 1 and n['tkg_batch_size'] == 4,
            'original B4/CTE B1/TKG B4 required')
    require(n['tp_degree'] == 2 and copied['num_hidden_layers'] == LAYERS, 'original model required')
    require(n['torch_dtype'] == 'bfloat16', 'original BF16 required')
    n['tensor_capture_config'] = capture_config()
    n['deepseek_cpu_capture_version'] = VERSION
    return copied


def split_outputs(outputs, *, device, validate=True):
    """Device hides aliased KV; CPU returns all 72 KV outputs after captures."""
    require(isinstance(outputs, (list, tuple)), 'capture graph must return an explicit output sequence')
    wanted = KV_OFFSET if device else KV_OFFSET + KV_COUNT
    require(len(outputs) == wanted, f'capture output ABI: expected {wanted}, got {len(outputs)}')
    logits, captures, caches = outputs[0], tuple(outputs[1:KV_OFFSET]), tuple(outputs[KV_OFFSET:])
    if validate:
        require(len(logits.shape) == 3 and logits.shape[-1] == 151936, 'one complete vocabulary output')
        require(str(logits.dtype) == 'torch.float32', 'original float32 logits required')
        for name, tensor in zip(NAMES, captures):
            require(len(tensor.shape) in (4, 5) and tensor.shape[-1] == 2,
                    name + ': selected tensor plus trailing TP-rank axis required')
            require(str(tensor.dtype) == 'torch.bfloat16', name + ': BF16 required')
            require(tensor.shape[token_axis(name)] == 1, name + ': prediction token only')
        for tensor in caches:
            require(tuple(tensor.shape) == KV_SHAPE and str(tensor.dtype) == 'torch.bfloat16',
                    'wrong KV offset, shape, or precision')
    return logits, captures, caches


def expected_alias_indices():
    return tuple(range(KV_OFFSET, KV_OFFSET + KV_COUNT))


def check_alias_indices(indices):
    require(tuple(indices) == expected_alias_indices(), 'KV aliases must follow one logits head and all captures')


def frames(backend):
    if backend == 'native':
        return [(n, f, s) for n in PREFIXES for f in range(2) for s in range(3)]
    require(backend == 'device', 'unknown backend')
    return [(n, slot, slot % 2, s) for n in PREFIXES for s in range(3) for slot in range(4)]


def check_neutrality_metadata(before, after):
    """No numerical thresholds are relaxed: every original metric must match."""
    a = [r for e in before['experiments'] for r in e['rows']]
    b = [r for e in after['experiments'] for r in e['rows']]
    for rows in (a, b):
        require([(r['prefix'], r['slot'], r['fixture'], r['step']) for r in rows] == frames('device'),
                'all original 84 ordered frames, four slots and two streams required')
    require(before['criteria'] == after['criteria'], 'original numerical thresholds required')
    criteria = before['criteria']
    failures = tuple(i for i, row in enumerate(a) if not row['valid'] or
                     row['metrics']['cosine'] < criteria['cosine_min'] or
                     row['metrics']['kl_reference_to_neuron'] > criteria['kl_max_nats'] or
                     row['metrics']['max_abs_error'] > criteria['max_abs_error_max'])
    require(failures == FAILED_INDICES, 'all original 14 failed frame indices required')
    for left, right in zip(a, b):
        require(left == right, 'capture changed original frame, route, or metrics')
    require(before['numerical_gate_passed'] is after['numerical_gate_passed'] is False,
            'original failed cold gate must remain failed')
    return dict(frames=84, original_metrics_exact=True, original14_failed_indices=list(failures), full_model_qualified=False)
