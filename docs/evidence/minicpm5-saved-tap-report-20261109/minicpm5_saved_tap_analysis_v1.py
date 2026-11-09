"""Pure saved-tap validation and descriptive BF16 perturbation statistics.

No filesystem, model, framework or runtime admission. A caller must first
authenticate each bundle and maintain its serial resource/ownership guard.
Perturbations relative to a failing CPU anchor are not errors versus HF.
"""
import hashlib
import math
import re
import struct

ARMS = ('strict_anchor_s0', 'strict_layer0_full_bf16_prefill_only',
        'strict_layer0_full_bf16_both_phases')
TAPS = ('layers.0.self_attn', 'layers.0', 'layers.41.self_attn', 'layers.41')
STAGES = ('embed_tokens',) + tuple(name for i in range(42) for name in (
    f'layers.{i}.input_layernorm', f'layers.{i}.self_attn',
    f'layers.{i}.post_attention_layernorm', f'layers.{i}.mlp', f'layers.{i}')) + ('norm', 'lm_head')
WIDTH = 2048


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Literal SHA256')


def frame_order(call):
    rows = call['physical_rows']
    frames = call['call']['frames']
    require(len(rows) in (1, 4) and [r['frame'] for r in rows] == frames,
            'Complete original logical frame order')
    require(sorted(r['physical_row'] for r in rows) == list(range(len(rows))),
            'Complete physical-row permutation')
    require(all(type(f['index']) is int for f in frames)
            and len({f['index'] for f in frames}) == len(frames), 'Distinct integer frame indices')
    return [r['frame']['index'] for r in sorted(rows, key=lambda r:r['physical_row'])]


def validate_call(call):
    require(call['complete'] is True and call['arm'] in ARMS
            and type(call['rank']) is int and call['rank'] in (0, 1), 'Complete original arm/rank')
    require(tuple(call['stage_names']) == STAGES
            and set(call['stage_sha256']) == set(STAGES)
            and set(call['stage_metadata']) == set(STAGES), 'All213 original prediction stages')
    for value in call['stage_sha256'].values():
        digest(value)
    order = frame_order(call)
    require([r['frame'] for r in call['HF_metrics']] == call['call']['frames'],
            'Original endpoint frame association')
    bundle = call['selected_tap_bundle']
    require(bundle['encoding'] == 'raw_bf16_le' and bundle['raw_bytes'] == len(order)*WIDTH*2*4,
            'Exact four finite BF16 prediction taps')
    digest(bundle['sha256'])
    require([e['name'] for e in bundle['entries']] == list(TAPS), 'Four taps in original order')
    offset = 0
    for entry in bundle['entries']:
        require(entry['offset'] == offset and entry['raw_bytes'] == len(order)*WIDTH*2
                and entry['shape'] == [len(order), 1, WIDTH]
                and entry['dtype'] == 'torch.bfloat16', 'Exact contiguous selected tap geometry')
        require(entry['frame_indices'] == order
                and entry['source_metadata'] == call['stage_metadata'][entry['name']],
                'Tap bytes use physical, not logical, row order')
        digest(entry['sha256'])
        offset += entry['raw_bytes']
    require(offset == bundle['raw_bytes'], 'Complete gap-free bundle')
    for field in ('before_metadata', 'post_metadata'):
        caches = call[field]
        require(len(caches) == 84 and len({r['name'] for r in caches}) == 84, 'All84 cache records')
        for row in caches:
            digest(row['sha256'])
    require(call['full_model_qualified'] is call['new_speedup_claim'] is False,
            'Original diagnostic limits retained')
    return order


def compare_call(left, right):
    validate_call(left)
    validate_call(right)
    require(left['call'] == right['call'] and left['physical_rows'] == right['physical_rows'],
            'Same original call and physical frame mapping')
    changed = [name for name in STAGES if left['stage_sha256'][name] != right['stage_sha256'][name]]
    first = STAGES.index(changed[0]) if changed else len(STAGES)
    equal_after = [name for name in STAGES[first:] if name not in changed]
    caches = {}
    for field in ('before_metadata', 'post_metadata'):
        a, b = left[field], right[field]
        require([r['name'] for r in a] == [r['name'] for r in b], 'Same ordered cache names')
        caches[field] = [x['name'] for x,y in zip(a,b) if x['sha256'] != y['sha256']]
    return dict(call=left['call'], physical_frame_order=frame_order(left),
        first_recorded_different_stage=changed[0] if changed else None,
        differing_prediction_stages=changed, equal_stages_after_first_difference=equal_after,
        logits_hash_equal=left['output_metadata']['sha256'] == right['output_metadata']['sha256'],
        selected_tap_hash_equal={a['name']:a['sha256'] == b['sha256'] for a,b in zip(
            left['selected_tap_bundle']['entries'], right['selected_tap_bundle']['entries'])},
        cache_differing_hashes=caches, left_endpoint_metrics=left['HF_metrics'],
        right_endpoint_metrics=right['HF_metrics'], byte_payloads_revalidated_here=False,
        per_frame_internal_magnitude_known=False)


def decode_bundle(raw, call):
    """Decode only an already admitted, fully hashed bounded bundle."""
    order = validate_call(call)
    bundle = call['selected_tap_bundle']
    require(type(raw) is bytes and len(raw) == bundle['raw_bytes']
            and hashlib.sha256(raw).hexdigest() == bundle['sha256'], 'Exact complete bundle bytes')
    result = {}
    for entry in bundle['entries']:
        data = raw[entry['offset']:entry['offset']+entry['raw_bytes']]
        require(hashlib.sha256(data).hexdigest() == entry['sha256'], 'Exact selected tap bytes')
        words = [x[0] for x in struct.iter_unpack('<H', data)]
        values = [struct.unpack('<f', struct.pack('<I', x << 16))[0] for x in words]
        require(all(math.isfinite(x) for x in values), 'Finite BF16 values only')
        result[entry['name']] = {frame:values[i*WIDTH:(i+1)*WIDTH] for i,frame in enumerate(order)}
    return result


def perturbation(anchor, candidate):
    """Float64 descriptive statistics; never substitute for original FP32 gates."""
    require(len(anchor) == len(candidate) == WIDTH
            and all(math.isfinite(x) for x in (*anchor, *candidate)), 'Two finite2048-vectors')
    delta = [b-a for a,b in zip(anchor,candidate)]
    sq_a, sq_b, sq_d = (math.fsum(x*x for x in row) for row in (anchor,candidate,delta))
    return dict(elements=WIDTH, changed_values=sum(a != b for a,b in zip(anchor,candidate)),
        max_abs_delta=max(abs(x) for x in delta),
        mean_abs_delta=math.fsum(abs(x) for x in delta)/WIDTH,
        signed_mean_delta=math.fsum(delta)/WIDTH,
        delta_rms=math.sqrt(sq_d/WIDTH), anchor_rms=math.sqrt(sq_a/WIDTH),
        candidate_rms=math.sqrt(sq_b/WIDTH),
        relative_l2=math.sqrt(sq_d/sq_a) if sq_a else None,
        cosine=math.fsum(a*b for a,b in zip(anchor,candidate))/math.sqrt(sq_a*sq_b) if sq_a and sq_b else None,
        reference_is_failing_cpu_anchor=True, absolute_HF_error_measured=False,
        full_model_qualified=False, new_speedup_claim=False)
