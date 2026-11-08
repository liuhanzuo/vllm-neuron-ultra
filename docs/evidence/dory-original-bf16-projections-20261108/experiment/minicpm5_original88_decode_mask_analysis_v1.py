"""Finite decode-mask saved-data slice; stdlib only, no K/V payload reads.

Actual raw IO is only under the inherited serial lease. Semantic failures remain
observations. Prefill and numeric K/V/score/probability work are deferred.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import struct
import sys
import minicpm5_original88_first_five_analysis_v2 as five
import minicpm5_original88_rope_analysis_v1 as rope

p, parser, post, same, ROOT = five.p, five.parser, five.post, five.same, five.ROOT
ref = rope.reference
DESIGN = ref('minicpm5-original88-cache-mask-four-boundary-interface-design-20261107-v1.json', '6e512fdc69a4ebd0eee07ac80dcb409a9a16129755cb55da8d678f6098088f5a')
REVIEW = ref('minicpm5-original88-cache-mask-root-design-review-20261107-v1.json', '1fb19fb4aef78ae56f701b0158324a58edc5c898c7e0a7a707cee2b74ad74d3a')
ROPE_RESULT = ref('minicpm5-original88-rope-r1-20261107-v1.json', 'b7e52f144a2b29c54257831beeb995d501fea512bb976c9c80f1076398e12175')
ROPE_EXIT = ref('minicpm5-original88-rope-root-exit-r1-20261107-v1.json', 'b47589c51768899202d740c3e3f93422d9cc9316425e391d5b93133179ec7e3d')
ROPE_REVIEW = ref('minicpm5-original88-rope-independent-metadata-review-20261107-v1.json', 'f0b4fa0089d08da833c60d2871e1b322651640f5e1186d2042d36d1133142d5c')
ROPE_SOURCE_SHA = 'ab3bf6a789df5440fd28a5c73c72c9bf0e284b1d62e0103e121e7aae4f4e1ad4'
ROLES = (('HF_native', -1), ('C_Hd_control', 0), ('C_Hd_control', 1), ('C_Hc_history_generation', 0), ('C_Hc_history_generation', 1))

def prerequisites(bound):
    p.require(p.sha(Path(rope.__file__)) == ROPE_SOURCE_SHA, 'Frozen RoPE helper')
    owners = rope.prerequisites(bound)
    design = p.read_ref(DESIGN); review = p.read_ref(REVIEW)
    p.require(review['reviewer'] == '/root' and review['passed'] is True and review['blockers'] == [], 'Root accepted finite decode slice')
    same(review['design'], DESIGN, 'Exact mask design review')
    for source in design['source_refs']:
        p.require(p.sha(Path(source['path'])) == source['sha256'], 'Frozen mask/cache source')
    result = p.read_ref(ROPE_RESULT); ended = p.read_ref(ROPE_EXIT)
    independent = p.read_ref(ROPE_REVIEW); live_ref = post.reference(ended['root_live']); live = p.read_ref(live_ref)
    p.require(independent['passed'] is True and independent['coverage']['rows'] == 1152 and any(x['path'] == ROPE_RESULT['path'] and x['sha256'] == ROPE_RESULT['sha256'] for x in independent['inputs']), 'Independently reviewed RoPE result')
    p.require(ended['schema'] == 'minicpm5_original88_rope_root_exit_v1' and ended['reviewer'] == '/root' and ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0, 'Root observed actual RoPE exit0')
    p.require(live['schema'] == 'minicpm5_original88_rope_root_live_v1' and live['reviewer'] == '/root' and live['actual_identity_verified'] is True, 'RoPE actual root live identity')
    owner = post.process(live['owner']); same(owner, ended['owner'], 'RoPE live/exit owner'); same(owner, result['execution_owner'], 'RoPE result owner')
    same(live['actual_execution_handle'], ended['actual_execution_handle'], 'RoPE live/exit handle')
    handle = ended['actual_execution_handle']
    p.require(set(handle) == {'kind', 'session_id'} and handle['kind'] == 'tool_session' and type(handle['session_id']) is int and handle['session_id'] > 0, 'Actual RoPE tool session')
    same(ended['outputs']['report'], ROPE_RESULT, 'Actual RoPE report output')
    same(live['source'], dict(path=rope.__file__, sha256=ROPE_SOURCE_SHA), 'RoPE actual code identity')
    same(result['binding']['code'], live['source'], 'RoPE result code binding')
    for key, value in bound['binding'].items(): same(result['binding'][key], value, 'RoPE finalized binding '+key)
    same(result['binding']['later_process_exit_supplement'], rope.SUPPLEMENT, 'RoPE same later owner supplement')
    p.require(result['complete_coverage'] is True and len(result['rows']) == 1152 and result['self_exit_claimed'] is False, 'Complete non-self-certified RoPE prerequisite')
    expected = ['/home/ubuntu/venvs/vllm_neuron/bin/python', '-B', rope.__file__]
    for key, option in (('finalization', 'finalization'), ('catalog', 'catalog'), ('finalizer_exit', 'finalizer-exit')):
        value = bound['binding'][key]; expected += ['--'+option, value['path'], '--'+option+'-sha256', value['sha256']]
    expected += ['--output-prefix', live['output_prefix'], '--source-sha256', ROPE_SOURCE_SHA]
    same(owner['argv'], expected, 'Actual RoPE argv'); same(result['output_prefix'], live['output_prefix'], 'Same RoPE output prefix')
    return owners + [owner], live_ref

def tensor(value, shape, dtype):
    p.require(type(value) is dict and set(value) == {'metadata', 'reference'}, 'One actual tensor required')
    meta = value['metadata']; same(meta['source_shape'], shape, 'Exact actual tensor shape'); same(meta['selected_shape'], shape, 'Full saved tensor shape')
    p.require(meta['selection'] is None and meta['source_dtype'] == dtype, 'Full actual mask/position capture dtype')
    size = {'torch.bool': 1, 'torch.bfloat16': 2, 'torch.int64': 8}[dtype]
    same([meta['selected_bytes'], value['reference']['bytes']], [math.prod(shape)*size]*2, 'Actual payload byte count')
    p.require(type(meta['source_stride']) is list and len(meta['source_stride']) == len(shape) and all(type(x) is int and x >= 0 for x in meta['source_stride']) and type(meta['source_offset']) is int and meta['source_offset'] >= 0, 'Original source layout')
    return value

def prepare_tasks(plan, reports, read_receipt):
    snapshots = {}
    def read(ref):
        record = read_receipt(ref); snapshots[ref['path']] = {s['label']: s['value'] for s in record['observer']['snapshots']}; return record
    index = five.prepare_index(plan, reports, read); tasks = []
    for frame in plan['frames']:
        call = next(c for c in plan['calls'] if any(parser.typed_equal(x, frame) for x in c['frames']))
        if call['phase'] == 'prefill': continue
        p.require(type(frame['step']) is int and frame['step'] in (1, 2, 3), 'Actual decode step')
        fixture = next(f for f in plan['fixtures'] if f['id'] == frame['fixture']); logical = len(fixture['input_ids']) + frame['step'] - 1
        same(plan['graph_calls'][call['index']]['call'], call, 'Original graph call/group binding')
        p.require(type(call['group']) is int and call['group'] in (0, 1), 'Original group')
        for arm, rank in ROLES:
            entry = index[arm, rank, frame['index']]; same(entry['frame'], frame, 'Exact complete frame identity')
            same([entry['call'], entry['phase']], [call['index'], 'decode'], 'Same original decode call')
            native = arm == 'HF_native'; batch = 1 if native else 4; row = 0 if native else frame['slot']
            same(entry['physical_row'], row, 'Actual sorted slot physical row')
            width = logical+1 if native else plan['graph_calls'][call['index']]['bucket']
            p.require(type(width) is int and width > logical, 'Physical bucket contains expected current position')
            values = snapshots[entry['receipt']['path']]; label = 'actual_eager_mask' if native else 'prior_actual_mask'
            mask = tensor(values[label], [batch, 1, 1, width], 'torch.bfloat16' if native else 'torch.bool')
            positions = tensor(values['position_ids'], [batch, 1], 'torch.int64')
            same(values['attention_mask']['reference']['sha256'], mask['reference']['sha256'], 'Observed mask input/final same recorded bytes')
            route_support = dict(actual_route='hf_eager', active_route_unmasked=None, active_key_count=None)
            if not native:
                route = values['route_inputs_sdk_decode']
                p.require(route['is_prefix_caching'] is False and route['active_mask'] is None and values['active_mask'] is None, 'Actual non-prefix unmasked active branch')
                tensor(route['position_ids'], [4, 1], 'torch.int64')
                same(route['position_ids']['reference']['sha256'], positions['reference']['sha256'], 'Actual route positions')
                tensor(route['attention_mask'], [4, 1, 1, width], 'torch.bool')
                same(route['attention_mask']['reference']['sha256'], mask['reference']['sha256'], 'Actual route prior mask')
                for name in ('K', 'V'): tensor(route[name], [4, 1, 1, 128], 'torch.bfloat16')
                flags = values['dispatch_flags']
                expected = dict(attn_tkg_nki_kernel_enabled=False, attn_tkg_builtin_kernel_enabled=False, attn_block_tkg_nki_kernel_enabled=False, attn_block_cte_nki_kernel_enabled=False,
                                flash_decoding_enabled=False, k_cache_transposed=False, sequence_parallel_enabled=False, num_heads=8, num_key_value_heads=1, head_dim=128)
                same(flags, expected, 'Original unfused TP2 single-active route')
                route_support = dict(actual_route='sdk_decode', active_route_unmasked=True, active_key_count=1, basis='Authenticated route with position sequence1, active_mask=None, prefix=False and frozen compute_for_token_gen; K/V shapes inspected, no K/V bytes read.')
            tasks.append(dict(frame=copy.deepcopy(frame), call=call['index'], group=call['group'], phase='decode', arm=arm, rank=rank, physical_row=row,
                              expected_position=logical, physical_key_width=width, mask_label=label, mask=mask, positions=positions, receipt=entry['receipt'], route_support=route_support))
    p.require(len(tasks) == 120 and len({(t['frame']['index'], t['arm'], t['rank']) for t in tasks}) == 120, 'All24native plus96CPU decode observations')
    return tasks

def observe(task, reader, guard):
    """Decode actual saved bits; never rebuild a mask or evaluate attention."""
    row = {k: copy.deepcopy(task[k]) for k in ('frame', 'call', 'group', 'phase', 'arm', 'rank', 'physical_row', 'expected_position', 'physical_key_width', 'mask_label', 'receipt', 'route_support')}
    row.update(mask_reference=task['mask']['reference'], position_reference=task['positions']['reference'], mask_layout=task['mask']['metadata'], position_layout=task['positions']['metadata'])
    position_bytes = reader(task['positions'], guard); mask_bytes = reader(task['mask'], guard)
    p.require(len(position_bytes) == task['positions']['reference']['bytes'] and len(mask_bytes) == task['mask']['reference']['bytes'], 'Full authenticated read lengths')
    physical = task['physical_row']; width = task['physical_key_width']; expected = task['expected_position']
    actual_position = struct.unpack_from('<q', position_bytes, physical*8)[0]; native = task['arm'] == 'HF_native'
    allowed = []; excluded = []; unsupported = []; negative_zero = []
    for key in range(width):
        if native:
            bits = struct.unpack_from('<H', mask_bytes, 2*(physical*width+key))[0]
            if bits in (0x0000, 0x8000):
                allowed.append(key)
                if bits == 0x8000: negative_zero.append(key)
            elif bits == 0xff7f: excluded.append(dict(index=key, encoding='bf16_min', bits=bits))
            elif bits == 0xff80: excluded.append(dict(index=key, encoding='negative_infinity', bits=bits))
            else:
                exponent = bits & 0x7f80; fraction = bits & 0x7f
                kind = 'nan' if exponent == 0x7f80 and fraction else 'positive_infinity' if bits == 0x7f80 else 'finite_bias'
                detail = dict(index=key, kind=kind, bits=bits)
                if kind == 'finite_bias': detail['value'] = struct.unpack('<f', struct.pack('<I', bits << 16))[0]
                unsupported.append(detail)
        else:
            value = mask_bytes[physical*width+key]
            if value == 1: allowed.append(key)
            elif value == 0: excluded.append(dict(index=key, encoding='false', bits=0))
            else: unsupported.append(dict(index=key, kind='invalid_bool_byte', bits=value))
    allow_set = set(allowed); required = list(range(expected+1 if native else expected))
    omitted = [key for key in required if key not in allow_set]; extra = [key for key in allowed if key not in required]
    reasons = []
    if actual_position != expected: reasons.append('position_mismatch')
    if unsupported: reasons.append('unsupported_mask_value')
    if omitted: reasons.append('omitted_required_key')
    if not native and expected in allow_set: reasons.append('current_also_allowed_in_prior')
    if any(key > expected for key in extra): reasons.append('later_key_allowed_in_prior')
    # For native, exactly-once here means its full joined mask permits [0,p].
    gate = not reasons
    row.update(actual_position=actual_position, position_matches=actual_position == expected, allowed_indices=allowed, allowed_count=len(allowed),
               excluded_indices=[x['index'] for x in excluded], exclusion_encodings=sorted({x['encoding'] for x in excluded}), unsupported_values=unsupported, negative_zero_indices=negative_zero,
               omitted_required_indices=omitted, allowed_at_current=expected in allow_set, allowed_indices_ge_current=[x for x in allowed if x >= expected],
               allowed_prior_count=sum(x < expected for x in allowed), allowed_later_indices=[x for x in allowed if x > expected],
               current_position_addressed_branch_count=(int(expected in allow_set) if native else int(expected in allow_set)+1),
               branch_count_meaning='Count of branches allowing address/logical label p, not identical current K/V copies; prior[p] may be stale or zero. No K/V values read.',
               current_exactly_once=gate, canonical_mapping_status='eligible' if gate else 'blocked', reasons=reasons, observation_complete=True)
    return row

def summarize(plan, rows):
    lookup = {(r['frame']['index'], r['arm'], r['rank']): r for r in rows}
    p.require(len(rows) == len(lookup) == 120, 'Complete unique observation matrix')
    comparisons = []
    for frame in plan['frames']:
        if frame['step'] == 0: continue
        for rank in (0, 1):
            for left, right in five.PAIRS:
                a = lookup[frame['index'], left, -1 if left == 'HF_native' else rank]; b = lookup[frame['index'], right, rank]
                comparisons.append(dict(frame=frame, rank=rank, left_arm=left, right_arm=right,
                                         canonical_mapping_status='eligible' if a['current_exactly_once'] and b['current_exactly_once'] else 'blocked',
                                         left_reasons=a['reasons'], right_reasons=b['reasons'], key_values_compared=False))
    p.require(len(comparisons) == 144, 'All24frames3pairs2ranks retained')
    summary = dict(observation_rows=120, native_rows=sum(r['arm'] == 'HF_native' for r in rows), cpu_rows=sum(r['arm'] != 'HF_native' for r in rows),
                   eligible_rows=sum(r['current_exactly_once'] for r in rows), blocked_rows=sum(not r['current_exactly_once'] for r in rows),
                   current_also_allowed_in_prior_rows=sum('current_also_allowed_in_prior' in r['reasons'] for r in rows),
                   later_key_allowed_rows=sum(bool(r['allowed_later_indices']) for r in rows), omitted_required_rows=sum(bool(r['omitted_required_indices']) for r in rows),
                   unsupported_value_rows=sum(bool(r['unsupported_values']) for r in rows), position_mismatch_rows=sum(not r['position_matches'] for r in rows),
                   prefill_status='deferred_not_analyzed', deferred_prefill_frames=[f for f in plan['frames'] if f['step'] == 0],
                   all32_frame_mask_semantics_complete=False, full_post_payload_status='unavailable_not_analyzed', full_model_qualified=False, causal_operator_proven=False)
    same([summary['native_rows'], summary['cpu_rows']], [24, 96], 'Complete arm partition')
    return summary, comparisons

def run(audit_ref, catalog_ref, exit_ref, prefix, source_sha256):
    source = Path(__file__).resolve()
    p.require(re.fullmatch('[0-9a-f]{64}', source_sha256) and p.sha(source) == source_sha256, 'Root actual mask-analysis code SHA')
    bound = five.load_finalized(audit_ref, catalog_ref, exit_ref); owners, rope_live = prerequisites(bound)
    spec, plan = bound['spec'], bound['plan']; prefix = Path(prefix)
    p.require(prefix.parent == ROOT/'research' and re.fullmatch('minicpm5-original88-decode-mask-[a-z0-9-]+', prefix.name), 'Fresh finite-mask report prefix')
    outputs = {name: Path(str(prefix)+suffix) for name, suffix in (('report', '.json'), ('summary', '-summary.json'), ('failure', '-failure.json'))}
    p.require(all(not path.exists() for path in outputs.values()), 'Never overwrite evidence')
    fd, owner, lock = post.external.acquire_lock()
    try:
        def guard():
            p.clear_window(); p.lock_proof(owner, lock); p.control_usage(spec); p.active_heavy()
            same({name: p.sha(ROOT/name) for name in spec['canonical_sha256']}, spec['canonical_sha256'], 'Canonical unchanged')
            p.require(p.sha(source) == source_sha256 and p.sha(Path(five.__file__)) == rope.FIRST_SOURCE_SHA and p.sha(Path(rope.__file__)) == ROPE_SOURCE_SHA, 'Analysis source remains pinned')
        guard(); p.gone(bound['finalizer_owner']); p.predecessor_drain(bound['catalog'])
        for previous in owners: p.gone(previous)
        tasks = prepare_tasks(plan, bound['reports'], parser.checked_reference)
        rows = [observe(task, five.read_raw, guard) for task in tasks]
        summary, comparisons = summarize(plan, rows); guard()
        binding = dict(bound['binding'], code=dict(path=str(source), sha256=source_sha256), design=DESIGN, design_review=REVIEW,
                       first_five_result=rope.FIRST_RESULT, first_five_independent_review=rope.FIRST_REVIEW, later_process_exit_supplement=rope.SUPPLEMENT,
                       rope_result=ROPE_RESULT, rope_independent_review=ROPE_REVIEW, rope_actual_exit=ROPE_EXIT, rope_actual_live=rope_live,
                       first_five_source=dict(path=five.__file__, sha256=rope.FIRST_SOURCE_SHA), rope_source=dict(path=rope.__file__, sha256=ROPE_SOURCE_SHA))
        report = dict(schema='minicpm5_original88_decode_mask_analysis_v1', binding=binding, execution_owner=owner, checked_prior_owners=owners, output_prefix=str(prefix),
                      rows=rows, comparisons=comparisons, summary=summary, complete_observation_coverage=True, self_exit_claimed=False, root_must_observe_actual_exit=True,
                      kv_payloads_read=0, numpy_used=False, original_remaining_gates=plan['remaining_gates'], thresholds_changed=False,
                      interpretation='Decode mask observations only; prefill is deferred. Addressing p in two branches does not prove equal K/V contents or a global cause; this slice cannot repair or qualify the complete model.',
                      full_four_boundary_implemented=False, full_model_qualified=False, new_speedup_claim=False, causal_operator_proven=False)
        p.once(outputs['report'], report); p.once(outputs['summary'], dict(binding=binding, **summary)); guard()
        return {name: dict(path=str(path), sha256=p.sha(path)) for name, path in outputs.items() if name != 'failure'}
    except Exception as error:
        p.once(outputs['failure'], dict(schema='minicpm5_original88_decode_mask_failure_v1', error_type=type(error).__name__, error=str(error), execution_owner=owner,
                                       code=dict(path=str(source), sha256=source_sha256), self_exit_claimed=False, full_model_qualified=False))
        raise
    finally:
        post.external.release_lock(fd)

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for name in ('finalization', 'catalog', 'finalizer-exit'):
        cli.add_argument('--'+name, required=True); cli.add_argument('--'+name+'-sha256', required=True)
    cli.add_argument('--output-prefix', required=True); cli.add_argument('--source-sha256', required=True)
    args = cli.parse_args(); p.require(sys.dont_write_bytecode and __debug__, 'python -B required')
    refs = [dict(path=getattr(args, n), sha256=getattr(args, n+'_sha256')) for n in ('finalization', 'catalog', 'finalizer_exit')]
    print(json.dumps(run(*refs, args.output_prefix, args.source_sha256), indent=2))
