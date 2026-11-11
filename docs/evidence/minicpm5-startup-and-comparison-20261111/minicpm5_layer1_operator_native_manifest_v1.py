"""Exact metadata-only native layer1 decode capture contract; no admission/IO."""
import copy
import hashlib
import json
import math

SCHEMA = 'minicpm5_layer1_operator_native_manifest_v1'
ARMS = ('unobserved', 'noop_trace', 'full_trace')
CALLS = (4, 5, 6, 11, 12, 13)
WIDTHS = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.int64': 8}
MAX_TENSOR_BYTES = 16 * 2**20
MAX_OUTPUT_BYTES = 256 * 2**20
METADATA_RESERVE_BYTES = 8 * 2**20


def require(ok, message):
    if not ok: raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor(shape, dtype='torch.bfloat16', selection=None):
    size = math.prod(shape) * WIDTHS[dtype]
    require(0 < size <= MAX_TENSOR_BYTES, 'Full bounded tensor extent')
    return dict(shape=list(shape), dtype=dtype, bytes=size, selection=selection)


def event_manifest(length):
    require(type(length) is int and 2 <= length <= 4096, 'Original bounded native decode length')
    previous = length - 1
    selection = dict(kind='prediction_query_rows', axis=2, positions=[0])
    h = lambda: tensor([1, 1, 2048])
    q = lambda: tensor([1, 16, 1, 128])
    k = lambda: tensor([1, 2, 1, 128])
    mask = lambda: tensor([1, 1, 1, length])
    ids = lambda: tensor([1, 1], 'torch.int64')
    cache = lambda n: [tensor([1, 2, n, 128]), tensor([1, 2, n, 128])]
    score = lambda dtype='torch.bfloat16': tensor([1, 16, 1, length], dtype, selection)
    # This is the exact original observer order, including repeated complete
    # cache views and masks. No deduplication hides a boundary observation.
    events = [
        ('normalized_input', h()), ('attention_mask', mask()), ('position_ids', ids()),
        ('cache_before', cache(previous)), ('q_projection', h()),
        ('k_projection', tensor([1, 1, 256])), ('v_projection', tensor([1, 1, 256])),
        ('q_rope_input', q()), ('k_rope_input', k()), ('rope_position_ids', ids()),
        ('rotary_route', {'json_only': True}), ('rope_cos', tensor([1, 1, 128])),
        ('rope_sin', tensor([1, 1, 128])), ('q_post_rope', q()), ('k_post_rope', k()),
        ('v_heads', k()), ('cache_update_active_key', k()),
        ('native_effective_key', tensor([1, 2, length, 128])),
        ('native_effective_value', tensor([1, 2, length, 128])), ('actual_eager_mask', mask()),
        ('qk0_unscaled', score()), ('qk0_scaled', score()),
        ('qk0_scale_operation', {'json_only': True}), ('softmax_input', score()),
        ('probabilities_fp32', score('torch.float32')), ('probabilities_bf16', score()),
        ('pv_context', q()), ('context_before_o', h()),
        ('cache_after', cache(length)), ('attention_output', h()),
    ]
    return [dict(label=label, value=value) for label, value in events]


def leaves(value):
    if type(value) is list:
        for child in value: yield from leaves(child)
    elif 'shape' in value: yield value
    else: require(value == {'json_only': True}, 'Known metadata event')


def build(plan, native_result):
    require(len(plan['frames']) == 32 and len(plan['calls']) == 14
        and [f['id'] for f in plan['fixtures']] == list(range(8)), 'Unchanged original32/14/8 scope')
    require(native_result['schema'] == 'minicpm5_layer1_native_cache_result_v1'
        and native_result['complete'] is True and native_result['rank'] == -1
        and len(native_result['records']) == 32 and native_result['native_provenance'] == plan['native_provenance'],
        'Completed original native metadata and provenance')
    records = {(r['frame']['fixture'], r['frame']['step']): r for r in native_result['records']}
    require(len(records) == 32 and {r['frame']['index'] for r in records.values()} == set(range(32)),
        'All original native frames without duplicates')
    selected = [f for f in plan['frames'] if f['step'] in (1, 2, 3)]
    require(len(selected) == 24, 'Exactly24 original decode checkpoints')
    output = []
    for frame in selected:
        fixture = plan['fixtures'][frame['fixture']]
        length = len(fixture['input_ids']) + frame['step']
        current = records[frame['fixture'], frame['step']]
        previous = records[frame['fixture'], frame['step']-1]
        require(current['frame'] == frame and current['cache_length'] == length
            and previous['cache_length'] == length-1, 'Exact current and step-minus-one complete cache')
        calls = [c for c in plan['calls'] if frame in c['frames']]
        require(len(calls) == 1 and calls[0]['index'] in CALLS and calls[0]['phase'] == 'decode',
            'Exact original decode call mapping')
        ci = calls[0]['index']; position = calls[0]['logical_positions'][calls[0]['frames'].index(frame)]
        require(position == length-1, 'Original position and full history length agree')
        for record, expected_length in ((current, length), (previous, length-1)):
            for name in ('key', 'value'):
                require(record['cache_metadata'][name]['shape'] == [1, 2, expected_length, 128]
                    and record['cache_metadata'][name]['dtype'] == 'torch.bfloat16'
                    and record['outputs'][name]['raw_bytes'] == expected_length*512,
                    'Both complete official KV heads at every valid position')
        taps = {}
        for name in ('layers.1.input_layernorm', 'layers.1.self_attn'):
            entries = [e for e in current['tap_entries'] if e['name'] == name]
            require(len(entries) == 1 and entries[0]['shape'] == [1, 1, 2048]
                and entries[0]['bytes'] == 4096
                and current['tap_metadata'][name]['source_shape'] == [1, 1, 2048],
                'Full original B1 decode normalized input and attention output')
            taps[name] = copy.deepcopy(entries[0])
        events = event_manifest(length)
        raw_shapes = [v for event in events for v in leaves(event['value'])]
        controls = dict(attention=tensor([1, 1, 2048]), key=tensor([1, 2, length, 128]),
            value=tensor([1, 2, length, 128]))
        control_raw = sum(v['bytes'] for v in controls.values()) * len(ARMS)
        capture_raw = sum(v['bytes'] for v in raw_shapes)
        disk_bytes = sum(((v['bytes']+4095)//4096)*4096 for v in raw_shapes)
        disk_bytes += len(ARMS)*sum(((v['bytes']+4095)//4096)*4096 for v in controls.values())
        output.append(dict(frame=copy.deepcopy(frame), call=ci, logical_position=position,
            cache_length=length, previous_frame=copy.deepcopy(previous['frame']),
            current_outputs=copy.deepcopy(current['outputs']), previous_outputs={k:copy.deepcopy(previous['outputs'][k]) for k in ('key','value')},
            current_cache_metadata=copy.deepcopy(current['cache_metadata']),
            previous_cache_metadata=copy.deepcopy(previous['cache_metadata']),
            normalized_metadata=copy.deepcopy(current['tap_metadata']['layers.1.input_layernorm']),
            taps=taps, events=events, control_outputs=controls,
            observer_raw_bytes=capture_raw, control_raw_bytes=control_raw,
            total_raw_bytes=capture_raw+control_raw, raw_files=len(raw_shapes)+3*len(ARMS),
            raw_files_4k_allocation_bytes=disk_bytes))
    require({row['call'] for row in output} == set(CALLS), 'All six original call identities')
    total = sum(row['total_raw_bytes'] for row in output)
    allocation = sum(row['raw_files_4k_allocation_bytes'] for row in output)
    require(allocation+METADATA_RESERVE_BYTES < MAX_OUTPUT_BYTES, 'Original256MiB total reservation retained')
    return dict(schema=SCHEMA, layer=1, arms=list(ARMS), frames=output, checkpoints=24,
        direct_attention_calls=72, prefill_calls=0, full_model_forward_calls=0,
        native_provenance=copy.deepcopy(plan['native_provenance']),
        observer_raw_bytes=sum(r['observer_raw_bytes'] for r in output),
        control_raw_bytes=sum(r['control_raw_bytes'] for r in output), total_raw_bytes=total,
        raw_files=sum(r['raw_files'] for r in output), raw_files_4k_allocation_bytes=allocation,
        metadata_reserve_bytes=METADATA_RESERVE_BYTES, total_reserved_bytes=allocation+METADATA_RESERVE_BYTES,
        max_tensor_bytes=MAX_TENSOR_BYTES, total_output_cap_bytes=MAX_OUTPUT_BYTES,
        mask=dict(function='original_create_causal_mask', cache_layer_idx=1,
            dtype='torch.bfloat16', shape_rule='[1,1,1,current_cache_length]',
            input_mask='original_all_ones_int64_[1,current_cache_length]'),
        full_model_qualified=False, new_speedup_claim=False)
