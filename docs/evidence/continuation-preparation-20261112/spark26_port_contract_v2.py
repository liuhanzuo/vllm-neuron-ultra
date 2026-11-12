"""Stdlib source contract for Spark's fixed SDK ABI. Not runtime admission."""
from spark26_port_contract_v1 import SDK_SITE, SDK_VERSIONS, TARGET, capacity, index_plan, tp2_layout
from spark26_native_contract_v1 import LAYER_TYPES, state_schema

TRACE_INPUTS = ('input_ids', 'attention_mask', 'position_ids', 'seq_ids',
                'sampling_params', 'prev_hidden', 'adapter_ids')
UNVERIFIED = (
    'actual original-checkpoint load and TP2 sharding/numerical equivalence',
    'native tensor execution, tracing and mixed-cache output alias application',
    'Neuron compile/lowering, cold cache, long decode and request-slot reuse',
    'matched quality and HTTP performance qualification',
)


def require_runtime_admission():
    raise RuntimeError('Spark v2 remains source-only, not runtime admitted: ' + '; '.join(UNVERIFIED))


def checkpoint_schema():
    """SDK application_base strips model. before calling the conversion hook."""
    return {key.removeprefix('model.'): shape for key, shape in state_schema().items()}


def converted_key(key):
    if key.endswith('self_attn.q_k_v_proj.weight'):
        return key[:-len('q_k_v_proj.weight')] + 'qkv_proj.Wqkv.weight'
    return key


def alias_layout():
    """One logits output, then K,V for each layer; output_logits/capture disabled."""
    return [(1 + 2 * i + kv, i, 'K' if kv == 0 else 'V', capacity(kind))
            for i, kind in enumerate(LAYER_TYPES) for kv in (0, 1)]


def validate_host_request(token_rows, position_rows, slots, attention_rows=None):
    """Validate CPU request metadata before SDK padding; reject unsupported routes.

    Slots here are real slots 0..3. SDK-created -1 dummy rows exist only inside
    the compiled decode graph. Leading padding and chunked prefill are excluded.
    """
    batch = len(slots)
    if not 1 <= batch <= 4 or len(token_rows) != batch or len(position_rows) != batch:
        raise ValueError('Invalid request batch')
    if any(type(s) is not int or not 0 <= s < 4 for s in slots) or len(set(slots)) != batch:
        raise ValueError('Real slots must be unique integers in 0..3')
    width = len(token_rows[0])
    if not 1 <= width <= 4096:
        raise ValueError('Invalid request width')
    for tokens, positions in zip(token_rows, position_rows):
        if len(tokens) != width or len(positions) != width:
            raise ValueError('Ragged input or position shape')
        if any(type(t) is not int or not 0 <= t < 131072 for t in tokens):
            raise ValueError('Invalid token ID')
        if any(type(p) is not int or not 0 <= p < 4096 for p in positions):
            raise ValueError('Invalid cache position')
    prefill = all(row[0] == 0 for row in position_rows)
    if not prefill:
        if width != 1 or any(row[0] == 0 for row in position_rows):
            raise ValueError('Mixed prefill/decode or chunked input is unsupported')
        return {'prefill': False, 'lengths': [row[0] + 1 for row in position_rows],
                'pad_single_prompt': False, 'attention_rows': attention_rows}
    if attention_rows is not None and len(attention_rows) != batch:
        raise ValueError('Attention batch mismatch')
    lengths, masks = [], []
    for row_index, positions in enumerate(position_rows):
        if attention_rows is None:
            length = next((j for j in range(1, width) if positions[j] == 0), width)
            mask = [1] * length + [0] * (width - length)
        else:
            mask = list(attention_rows[row_index])
            if len(mask) != width or any(type(x) is not int or x not in (0, 1) for x in mask):
                raise ValueError('Attention mask must be binary with input width')
            length = sum(mask)
            if length < 1 or mask != [1] * length + [0] * (width - length):
                raise ValueError('Only nonempty right-padded prompts are supported')
        if positions[:length] != list(range(length)) or any(positions[length:]):
            raise ValueError('Prefill positions must be 0..length-1 followed by zero padding')
        lengths.append(length); masks.append(mask)
    return {'prefill': True, 'lengths': lengths, 'pad_single_prompt': width == 1,
            'attention_rows': masks}


def padded_decode_rows(slots, positions):
    """Scalar model of SDK _forward_with_pad and apply_seq_ids_mask=True."""
    if len(slots) != len(positions) or not slots or len(set(slots)) != len(slots):
        raise ValueError('Invalid decode request')
    if any(type(s) is not int or not 0 <= s < 4 for s in slots):
        raise ValueError('Invalid active slot')
    pairs = sorted(zip(slots, positions))
    first = pairs[0][1]
    by_slot = dict(pairs)
    return [s if s in by_slot else -1 for s in range(4)], [by_slot.get(s, first) for s in range(4)]


def scalar_masked_write(old_rows, new_rows, slots):
    """Physical-row selection oracle: a dummy row never has a target."""
    if len(new_rows) != len(slots):
        raise ValueError('Update shape mismatch')
    active = [s for s in slots if s != -1]
    if any(type(s) is not int or s < -1 or s >= len(old_rows) for s in slots) or len(active) != len(set(active)):
        raise ValueError('Duplicate or invalid real slot')
    by_slot = {s: value for s, value in zip(slots, new_rows) if s != -1}
    return [by_slot.get(s, old_rows[s]) for s in range(len(old_rows))]


def specification():
    return dict(status='source_only_not_runtime_admitted', sdk_site=str(SDK_SITE),
                sdk_versions=SDK_VERSIONS, target=TARGET,
                context_batch=1, decode_batch=4, apply_seq_ids_mask=True,
                trace_inputs=list(TRACE_INPUTS), cache_output_count=72,
                qkv='one GroupQueryAttention_QKV native packed projection; kernels disabled',
                runtime_pending=list(UNVERIFIED), native_executed=False,
                model_qualified=False, speedup_claim=False)
