"""Finite original88 saved-payload prefill comparison; root supplies serial guard.

Only run() imports NumPy. No model, attention, softmax, matmul or cache replay is
executed. Scores/probabilities cover the captured final prediction query only.
"""
import copy
import json
import math
from pathlib import Path
import re

import minicpm5_original88_first_five_analysis_v2 as five

p, parser, same, ROOT = five.p, five.parser, five.same, five.ROOT
FIRST_SOURCE_SHA = '516f4dfda864aaa91f08228666b1857ccbab4ef459b134f6c81109a7e99d67a1'
STAGES = ('active_value', 'effective_key', 'effective_value', 'returned_current_key',
          'returned_current_value', 'qk0_unscaled', 'qk0_scaled', 'softmax_input',
          'probabilities_fp32', 'probabilities_bf16', 'context_before_o', 'attention_output')
SCORES = STAGES[5:10]
ROLES = (('HF_native', -1), ('C_Hd_control', 0), ('C_Hd_control', 1),
         ('C_Hc_history_generation', 0), ('C_Hc_history_generation', 1))
DTYPE_BYTES = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.bool': 1, 'torch.int64': 8}


def tensor(value, shape, dtype, selected=None, selection=None):
    p.require(type(value) is dict and set(value) == {'metadata', 'reference'}, 'Actual captured tensor')
    meta = value['metadata']; selected = shape if selected is None else selected
    same(meta['source_shape'], shape, 'Exact source shape')
    same(meta['selected_shape'], selected, 'Exact stored clone shape')
    same(meta['selection'], selection, 'Exact saved selection')
    p.require(meta['source_dtype'] == dtype, 'Exact original dtype')
    p.require(len(meta['source_stride']) == len(shape) and
              all(type(x) is int and x >= 0 for x in meta['source_stride']) and
              type(meta['source_offset']) is int and meta['source_offset'] >= 0, 'Source layout retained')
    same([meta['selected_bytes'], value['reference']['bytes']],
         [math.prod(selected) * DTYPE_BYTES[dtype]] * 2, 'Contiguous saved byte size')
    p.require(re.fullmatch('[0-9a-f]{64}', value['reference']['sha256']) is not None, 'Captured hash')
    return value


def digest_chain(*values):
    """Metadata support only; never equate this check with a new raw comparison."""
    first = values[0]
    for value in values[1:]:
        same([value['reference']['sha256'], value['metadata']['selected_shape'], value['metadata']['source_dtype']],
             [first['reference']['sha256'], first['metadata']['selected_shape'], first['metadata']['source_dtype']],
             'Same captured tensor chain metadata')


def prepare(plan, reports, read_receipt):
    """All88 receipt admission inherited; select all40 prefill capture contexts."""
    snapshots = {}; observers = {}
    def read(ref):
        record = read_receipt(ref)
        snapshots[ref['path']] = {s['label']: s['value'] for s in record['observer']['snapshots']}
        observers[ref['path']] = record['observer']
        return record
    index = five.prepare_index(plan, reports, read); entries = {}
    frames = [f for f in plan['frames'] if f['step'] == 0]
    same([f['index'] for f in frames], [0, 1, 2, 3, 16, 17, 18, 19], 'All eight original prefills')
    for frame in frames:
        fixture = next(f for f in plan['fixtures'] if f['id'] == frame['fixture'])
        length = len(fixture['input_ids'])
        for arm, rank in ROLES:
            entry = copy.deepcopy(index[arm, rank, frame['index']]); call = plan['calls'][entry['call']]
            same(entry['frame'], frame, 'Complete prefill identity')
            same([entry['phase'], entry['physical_row'], call['phase']], ['prefill', 0, 'prefill'], 'Prefill row0')
            same(plan['graph_calls'][entry['call']]['call'], call, 'Original call binding')
            native = arm == 'HF_native'; width = length if native else plan['graph_calls'][entry['call']]['bucket']
            p.require(type(width) is int and width >= length > 0, 'Original native/bucket width')
            v = snapshots[entry['receipt']['path']]; o = observers[entry['receipt']['path']]
            same(o['actual_route'], 'hf_eager' if native else 'sdk_prefill', 'Actual eager/prefill route')
            p.require(v['cache_before' if native else 'effective_prior_kv'] is None, 'No consumed prefill prior')
            heads = 16 if native else 8; kvheads = 2 if native else 1
            kvshape = [1, kvheads, width, 128]
            positions = tensor(v['position_ids'], [1, width], 'torch.int64')
            masklabel = 'actual_eager_mask' if native else 'prefill_actual_mask'
            mask = tensor(v[masklabel], [1, 1, width, width], 'torch.bfloat16' if native else 'torch.bool')
            tensor(v['attention_mask'], [1, 1, width, width], 'torch.bfloat16' if native else 'torch.bool')
            digest_chain(mask, v['attention_mask'])
            active = tensor(v['v_heads'], kvshape, 'torch.bfloat16')
            if native:
                key = tensor(v['native_effective_key'], kvshape, 'torch.bfloat16')
                value = tensor(v['native_effective_value'], kvshape, 'torch.bfloat16')
                returned = v['cache_after']
            else:
                route = v['route_inputs_sdk_prefill']
                same([route['q_len'], route['bsz']], [width, 1], 'Actual route geometry')
                same(v['active_mask'], None, 'No prefill active branch')
                expected_flags = dict(attn_tkg_nki_kernel_enabled=False, attn_tkg_builtin_kernel_enabled=False,
                    attn_block_tkg_nki_kernel_enabled=False, attn_block_cte_nki_kernel_enabled=False,
                    flash_decoding_enabled=False, k_cache_transposed=False, sequence_parallel_enabled=False,
                    num_heads=8, num_key_value_heads=1, head_dim=128)
                same(v['dispatch_flags'], expected_flags, 'Original unfused TP2 route')
                tensor(route['attention_mask'], [1, 1, width, width], 'torch.bool')
                digest_chain(mask, route['attention_mask'])
                tensor(route['Q'], [1, heads, width, 128], 'torch.bfloat16')
                tensor(v['q_after_prep'], [1, heads, width, 128], 'torch.bfloat16')
                digest_chain(route['Q'], v['q_after_prep'])
                key = tensor(route['K'], kvshape, 'torch.bfloat16')
                value = tensor(route['V'], kvshape, 'torch.bfloat16')
                tensor(v['k_after_prep'], kvshape, 'torch.bfloat16')
                tensor(v['v_after_prep'], kvshape, 'torch.bfloat16')
                digest_chain(key, v['k_after_prep']); digest_chain(value, v['v_after_prep'], active)
                returned = v['returned_current_kv']
            p.require(type(returned) is list and len(returned) == 2, 'Actual returned K/V pair')
            for value0 in returned: tensor(value0, kvshape, 'torch.bfloat16')
            digest_chain(key, returned[0]); digest_chain(value, returned[1])
            payloads = dict(active_value=active, effective_key=key, effective_value=value,
                            returned_current_key=returned[0], returned_current_value=returned[1])
            selection = dict(kind='prediction_query_rows', axis=2, positions=[length-1])
            for stage in SCORES:
                dtype = 'torch.float32' if stage == 'probabilities_fp32' else 'torch.bfloat16'
                payloads[stage] = tensor(v[stage], [1, heads, width, width], dtype,
                                         [1, heads, 1, width], selection)
            if not native:
                for label in ('prefill_mask_input', 'prefill_mask_output'):
                    tensor(v[label], [1, heads, width, width], 'torch.bfloat16', [1, heads, 1, width], selection)
                digest_chain(v['prefill_mask_input'], v['qk0_scaled'])
                digest_chain(v['prefill_mask_output'], v['softmax_input'])
            operation = v['qk0_scale_operation']
            p.require(operation['scaled_value_observed'] is True and operation['scaled_value_recomputed'] is False,
                      'Scaled score is observed, not recalculated')
            for stage in STAGES[-2:]:
                features = 2048 if native or stage == 'attention_output' else 1024
                payloads[stage] = tensor(v[stage], [1, width, features], 'torch.bfloat16')
            entry.update(length=length, width=width, native=native, arm=arm, rank=rank, group=call['group'],
                         mask=mask, mask_label=masklabel, positions=positions, payloads=payloads,
                         scale_operation=operation, capture_label_names=list(v))
            entries[arm, rank, frame['index']] = entry
    p.require(len(entries) == 40, 'Eight native plus32 CPU prefills')
    return frames, entries


def decode(np, data, payload):
    meta = payload['metadata']; dtype = meta['source_dtype']
    storage = {'torch.bfloat16': '<u2', 'torch.float32': '<u4', 'torch.bool': 'u1', 'torch.int64': '<i8'}[dtype]
    raw = np.frombuffer(data, dtype=storage).reshape(meta['selected_shape'])
    values = (raw.astype(np.uint32) << 16).view(np.float32) if dtype == 'torch.bfloat16' else raw.view(np.float32) if dtype == 'torch.float32' else raw
    return raw, values


def identity_fields(entry):
    return {k: copy.deepcopy(entry[k]) for k in ('frame', 'call', 'group', 'arm', 'rank', 'physical_row', 'length', 'width', 'receipt')}


def observe_mask(np, entry, reader, guard):
    mask, positions = entry['mask'], entry['positions']; length = entry['length']; width = entry['width']
    raw, _ = decode(np, reader(mask, guard), mask); raw = raw[0, 0, :length, :]
    _, pos = decode(np, reader(positions, guard), positions); pos = pos[0, :length]
    allowed = (raw == 0) | (raw == 0x8000) if entry['native'] else raw == 1
    excluded = (raw == 0xff7f) | (raw == 0xff80) if entry['native'] else raw == 0
    unknown = ~(allowed | excluded)
    # This constructs an expected Boolean domain for validation, never an attention input.
    expected = np.arange(width)[None, :] <= np.arange(length)[:, None]
    missing = expected & ~allowed; extra = ~expected & allowed
    def examples(value): return np.argwhere(value)[:16].astype(int).tolist()
    mismatch = np.flatnonzero(pos != np.arange(length)).astype(int).tolist()
    counts = dict(unsupported=int(np.count_nonzero(unknown)), omitted=int(np.count_nonzero(missing)),
                  extra=int(np.count_nonzero(extra)), padding_allowed=int(np.count_nonzero(allowed[:, length:])))
    row = identity_fields(entry)
    row.update(mask_label=entry['mask_label'], mask_reference=mask['reference'], mask_layout=mask['metadata'],
               position_reference=positions['reference'], position_layout=positions['metadata'],
               logical_queries_checked=length, physical_columns_checked=width,
               checked_elements=length*width, padded_query_rows_checked=False,
               allowed_counts_per_query=np.count_nonzero(allowed, axis=1).astype(int).tolist(),
               position_mismatch_indices=mismatch, counts=counts,
               examples=dict(unsupported=examples(unknown), omitted=examples(missing), extra=examples(extra)),
               negative_zero_count=int(np.count_nonzero(raw == 0x8000)) if entry['native'] else 0,
               exclusion_encodings={str(int(x)): int(n) for x, n in zip(*np.unique(raw[excluded], return_counts=True))},
               canonical_mapping_eligible=not mismatch and not any(counts.values()))
    return row


def selection(entry, stage, rank):
    payload = entry['payloads'][stage]; native = entry['native']; length = entry['length']
    if stage in SCORES:
        start = rank*8 if native else 0; slices = [[0, 1], [start, start+8], [0, 1], [0, length]]
    elif stage in STAGES[:5]:
        start = rank if native else 0; slices = [[0, 1], [start, start+1], [0, length], [0, 128]]
    else:
        start = rank*1024 if native and stage == 'context_before_o' else 0
        size = 1024 if stage == 'context_before_o' else 2048
        slices = [[0, 1], [0, length], [start, start+size]]
    return payload, slices


def metrics(np, ar, a, br, b):
    same(list(a.shape), list(b.shape), 'Mapped numeric domain shapes')
    p.require(a.size > 0, 'Nonempty comparison')
    a = a.astype(np.float64); b = b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b); nonfinite = int(a.size-np.count_nonzero(finite))
    delta = np.abs(a[finite]-b[finite]); norm = float(np.sqrt(np.sum(a[finite]**2)*np.sum(b[finite]**2)))
    return dict(compared_elements=int(a.size), raw_exact=bool(np.array_equal(ar, br)),
                different_elements=int(np.count_nonzero(ar != br)), max_abs=float(delta.max()) if delta.size else None,
                mean_abs=float(delta.mean()) if delta.size else None,
                cosine=float(np.sum(a[finite]*b[finite]))/norm if norm > 0 and not nonfinite else None,
                nonfinite_elements=nonfinite)


def compare(np, left, right, stage, rank, eligible, reader, guard):
    lp, ls = selection(left, stage, rank); rp, rs = selection(right, stage, rank)
    row = dict(frame=left['frame'], call=left['call'], group=left['group'], phase='prefill', rank=rank,
               left_arm=left['arm'], right_arm=right['arm'], stage=stage,
               left_receipt=left['receipt'], right_receipt=right['receipt'],
               left_reference=lp['reference'], right_reference=rp['reference'],
               left_layout=lp['metadata'], right_layout=rp['metadata'], left_slice=ls, right_slice=rs,
               global_query_heads=list(range(rank*8, rank*8+8)), global_kv_head=rank,
               query_scope='prediction_query_only' if stage in SCORES else 'all_real_tokens',
               prediction_query=left['length']-1, mask_gate=eligible)
    if not eligible and stage in STAGES[5:]:
        row.update(status='blocked_mask_semantics', reason='At least one actual complete prefill mask/position check failed')
        return row
    ar, a = decode(np, reader(lp, guard), lp); br, b = decode(np, reader(rp, guard), rp)
    aselect = tuple(slice(*x) for x in ls); bselect = tuple(slice(*x) for x in rs)
    ar, a, br, b = ar[aselect], a[aselect], br[bselect], b[bselect]
    measured = metrics(np, ar, a, br, b)
    row.update(status='comparable_exact' if measured['raw_exact'] else 'comparable_different', reason=None, **measured)
    if stage in STAGES[-2:]:
        row['prediction_query_metrics'] = metrics(np, ar[:, -1:, :], a[:, -1:, :], br[:, -1:, :], b[:, -1:, :])
    return row


def physical_score_rows(np, entry, reader, guard):
    """Inspect saved complete key width; no softmax recompute/renormalization."""
    rows = []; length = entry['length']
    for stage in SCORES:
        payload = entry['payloads'][stage]; raw, a = decode(np, reader(payload, guard), payload)
        tail = a[..., length:]; tailraw = raw[..., length:]; finite = np.isfinite(tail)
        row = dict(identity_fields(entry), stage=stage, payload_reference=payload['reference'], layout=payload['metadata'],
                   query=length-1, query_scope='prediction_query_only', logical_key_range=[0, length],
                   padded_key_range=[length, entry['width']], padded_elements=int(tail.size),
                   padded_nonfinite=int(tail.size-np.count_nonzero(finite)), padded_nonzero=int(np.count_nonzero(tail)),
                   padded_finite_max_abs=float(np.max(np.abs(tail[finite]))) if np.any(finite) else None)
        if stage in ('qk0_unscaled', 'qk0_scaled'):
            row['padding_interpretation'] = 'Unmasked scores; zero is not required on excluded keys.'
        elif stage == 'softmax_input':
            row['padding_interpretation'] = 'Actual excluded-key sentinel counts; no cross-dtype sentinel equality claim.'
            row['padded_bf16_min_count'] = int(np.count_nonzero(tailraw == 0xff7f))
            row['padded_negative_infinity_count'] = int(np.count_nonzero(tailraw == 0xff80))
        else:
            row['padding_interpretation'] = 'Actual saved probability mass; never normalized after slicing.'
            row['full_sum_per_head'] = [float(x) if math.isfinite(float(x)) else None for x in np.sum(a.astype(np.float64), axis=-1).reshape(-1)]
            row['logical_sum_per_head'] = [float(x) if math.isfinite(float(x)) else None for x in np.sum(a[..., :length].astype(np.float64), axis=-1).reshape(-1)]
            row['padded_sum_per_head'] = [float(x) if math.isfinite(float(x)) else None for x in np.sum(tail.astype(np.float64), axis=-1).reshape(-1)]
            row['padded_probabilities_zero'] = bool(np.all(tail == 0))
        rows.append(row)
    return rows


def summarize(frames, rows, masks):
    per_frame = []
    for frame in frames:
        selected = [x for x in rows if x['frame']['index'] == frame['index']]
        selected_masks = [x for x in masks if x['frame']['index'] == frame['index']]
        p.require(len(selected) == 72 and len(selected_masks) == 5, 'Complete per-frame matrix')
        first = []
        for rank in (0, 1):
            for left, right in five.PAIRS:
                order = [next(x for x in selected if (x['rank'], x['left_arm'], x['right_arm'], x['stage']) == (rank, left, right, stage)) for stage in STAGES]
                bad = next((x for x in order if x['status'] != 'comparable_exact'), None)
                first.append(dict(rank=rank, left_arm=left, right_arm=right,
                                  first_nonexact_selected_stage=bad['stage'] if bad else None,
                                  status=bad['status'] if bad else 'all_selected_stages_exact',
                                  causal_operator_proven=False))
        per_frame.append(dict(frame=frame, rows=len(selected), mask_observations=len(selected_masks),
                              mask_gate=all(x['canonical_mapping_eligible'] for x in selected_masks),
                              exact=sum(x['status'] == 'comparable_exact' for x in selected),
                              different=sum(x['status'] == 'comparable_different' for x in selected),
                              blocked=sum(x['status'].startswith('blocked') for x in selected), first_observed=first))
    return dict(frames=per_frame, numeric_rows=len(rows), mask_observations=len(masks),
                implemented_stages=list(STAGES), first_observed_is_not_causal=True,
                score_probability_query_scope='Only prediction query L-1 was captured; not a complete score matrix.',
                all_other_score_queries='unavailable_in_original88',
                full_cpu_post_payload='unavailable_actual_post_payload_in_original88',
                prefill_prior='not_applicable_no_prior',
                decode='blocked_old_inclusive_prior_alignment; not analyzed by this prefill slice',
                full_attention_analyzed=False, full_model_qualified=False, causal_operator_proven=False, new_speedup_claim=False)


def run(bound, output_prefix, source_sha256, guard, execution_owner):
    """Root must hold serial lock and supply guard checking actual lock/window/drain.

    bound is the original five.load_finalized return. Root wrapper is responsible
    for admission and all later owner exits, and must observe this worker's exit.
    """
    guard(); source = Path(__file__).resolve()
    p.require(re.fullmatch('[0-9a-f]{64}', source_sha256) is not None and p.sha(source) == source_sha256, 'Actual analyzer code pin')
    p.require(p.sha(Path(five.__file__)) == FIRST_SOURCE_SHA, 'Frozen raw/metadata helper')
    prefix = Path(output_prefix)
    p.require(prefix.parent == ROOT/'research' and re.fullmatch('minicpm5-original88-prefill-attention-[a-z0-9-]+', prefix.name), 'Fresh finite output prefix')
    paths = {k: Path(str(prefix)+suffix) for k, suffix in (('report', '.json'), ('summary', '-summary.json'), ('failure', '-failure.json'))}
    p.require(all(not x.exists() for x in paths.values()), 'Never overwrite evidence')
    try:
        frames, entries = prepare(bound['plan'], bound['reports'], parser.checked_reference)
        guard()
        import numpy as np
        rows = []; masks = []; physical = []; pair_gates = []; consumed = {}
        for frame in frames:
            guard(); cache = {}
            def reader(payload, check):
                reference = payload['reference']; path = reference['path']
                if path not in cache:
                    cache[path] = (copy.deepcopy(reference), five.read_raw(payload, check))
                    consumed[path] = copy.deepcopy(reference)
                else:
                    check(); same(cache[path][0], reference, 'Same captured reference on cached read')
                return cache[path][1]
            frame_masks = {}
            for arm, rank in ROLES:
                entry = entries[arm, rank, frame['index']]
                row = observe_mask(np, entry, reader, guard); masks.append(row); frame_masks[arm, rank] = row
                physical.extend(physical_score_rows(np, entry, reader, guard))
            for rank in (0, 1):
                for left, right in five.PAIRS:
                    li = (left, -1 if left == 'HF_native' else rank); ri = (right, rank)
                    a = entries[li + (frame['index'],)]; b = entries[ri + (frame['index'],)]
                    eligible = frame_masks[li]['canonical_mapping_eligible'] and frame_masks[ri]['canonical_mapping_eligible']
                    pair_gates.append(dict(frame=frame, rank=rank, left_arm=left, right_arm=right, mask_gate=eligible,
                                           prior='not_applicable_no_prior', full_cpu_post='unavailable_actual_post_payload'))
                    for stage in STAGES: rows.append(compare(np, a, b, stage, rank, eligible, reader, guard))
            cache.clear()
        same([len(rows), len(masks), len(physical), len(pair_gates)], [576, 40, 200, 48], 'Complete finite output matrix')
        summary = summarize(frames, rows, masks); guard()
        p.require(p.sha(source) == source_sha256 and p.sha(Path(five.__file__)) == FIRST_SOURCE_SHA, 'Source unchanged')
        binding = dict(bound['binding'], code=dict(path=str(source), sha256=source_sha256),
                       first_five_library=dict(path=five.__file__, sha256=FIRST_SOURCE_SHA))
        report = dict(schema='minicpm5_original88_prefill_attention_analysis_v1', binding=binding,
            execution_owner=execution_owner, self_exit_claimed=False, root_must_observe_actual_exit=True,
            output_prefix=str(prefix), mask_observations=masks, pair_gates=pair_gates, rows=rows,
            physical_score_observations=physical, raw_inputs=list(consumed.values()), summary=summary,
            recorded_chain_support='Metadata digest chains checked; numerical comparisons use authenticated saved payloads only.',
            captured_scale_operations=[dict(identity_fields(e), operation=e['scale_operation']) for e in entries.values()],
            complete_selected_coverage=all(not x['blocked'] for x in summary['frames']),
            original_remaining_gates=copy.deepcopy(bound['plan']['remaining_gates']),
            original_logits_thresholds=dict(cosine_min=0.995, kl_max=0.02, max_abs_max=2, aggregate_argmax_min=0.9, hf_margin_gt_0_5_must_match=True),
            thresholds_changed=False, attention_recomputed=False, scores_recomputed=False, probabilities_renormalized=False,
            full_cpu_post_compared=False, full_attention_analyzed=False, full_model_qualified=False,
            new_speedup_claim=False, causal_operator_proven=False)
        p.once(paths['report'], report); p.once(paths['summary'], dict(binding=binding, **summary)); guard()
        return {k: dict(path=str(v), sha256=p.sha(v)) for k, v in paths.items() if k != 'failure'}
    except Exception as error:
        p.once(paths['failure'], dict(schema='minicpm5_original88_prefill_attention_failure_v1',
               error_type=type(error).__name__, error=str(error), execution_owner=execution_owner,
               source_sha256=source_sha256, self_exit_claimed=False, full_model_qualified=False))
        raise
