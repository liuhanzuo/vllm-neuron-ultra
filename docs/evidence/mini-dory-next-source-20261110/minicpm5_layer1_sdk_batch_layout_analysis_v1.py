"""Pure-byte descriptive analysis; no model calls, file reads or qualification.

Caller authenticates cases, official targets and previous full controls under
its admitted reader. This module requires all twelve rank/call cases, all six
variants, complete four-row BF16 bytes and exact original control reproduction.
"""
import hashlib
import math
import struct

CALLS = (4, 5, 6, 11, 12, 13)
VARIANTS = ('own', 'b4_strided', 'b4_contiguous', 'b1_strided', 'b1_contiguous', 'restore')
PAIRS = (('layout_at_b4', 'b4_contiguous', 'b4_strided'),
         ('layout_at_b1', 'b1_contiguous', 'b1_strided'),
         ('batch_at_strided', 'b1_strided', 'b4_strided'),
         ('batch_at_contiguous', 'b1_contiguous', 'b4_contiguous'))


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def values(raw):
    require(type(raw) is bytes and len(raw) > 0 and len(raw) % 2 == 0, 'Complete BF16 bytes')
    words = [x[0] for x in struct.iter_unpack('<H', raw)]
    require(all(w & 0x7f80 != 0x7f80 for w in words), 'Finite actual BF16 words')
    return [struct.unpack('<f', struct.pack('<I', w << 16))[0] for w in words]


def distance(actual, reference):
    require(type(actual) is type(reference) is bytes and len(actual) == len(reference), 'Equal full vectors')
    a, b = values(actual), values(reference)
    num = math.fsum((x-y)**2 for x, y in zip(a, b))
    den = math.fsum(x*x for x in b)
    return dict(elements=len(a), raw_exact=actual == reference,
                changed_words=sum(actual[i:i+2] != reference[i:i+2] for i in range(0, len(actual), 2)),
                numeric_changed_elements=sum(x != y for x, y in zip(a, b)),
                l2_squared=num, reference_l2_squared=den,
                relative_l2=math.sqrt(num/den) if den else None,
                max_abs=max(abs(x-y) for x, y in zip(a, b)))


def aggregate(metrics):
    require(bool(metrics), 'Nonempty complete frame metrics')
    num = math.fsum(x['l2_squared'] for x in metrics)
    den = math.fsum(x['reference_l2_squared'] for x in metrics)
    return dict(frames=len(metrics), aggregate_relative_l2=math.sqrt(num/den) if den else None,
                l2_squared=num, reference_l2_squared=den,
                raw_exact_frames=sum(x['raw_exact'] for x in metrics),
                changed_words=sum(x['changed_words'] for x in metrics),
                max_abs=max(x['max_abs'] for x in metrics))


def analyze(cases, rows, controls):
    """controls[(rank, call_index)] contains authenticated complete byte controls.

    Required keys: call, original_call, original, previous_own, previous_both,
    previous_restore, official_target. Targets are complete physical-slot order.
    rows use {rank, call, variant, raw}; each raw is 16384 bytes, including B1
    outputs already assembled back into original slots 0,1,2,3.
    """
    require(type(cases) in (list, tuple) and len(cases) == 12, 'All two ranks and six calls')
    expected = {(r, c) for r in (0, 1) for c in CALLS}
    bycase = {}
    for case in cases:
        rank, ci = case['rank'], case['call']['index']
        require(type(rank) is int and (rank, ci) in expected and (rank, ci) not in bycase
                and case['arm'] == 'strict_anchor_s0' and case['call']['phase'] == 'decode', 'Unique original anchor cases')
        frames = case['physical_frame_order']
        require(len(frames) == 4 and [f['slot'] for f in frames] == [0, 1, 2, 3]
                and frames == sorted(case['call']['frames'], key=lambda f: f['slot']), 'Complete original physical frame order')
        bycase[rank, ci] = case
    require(set(bycase) == set(controls) == expected, 'Exact complete case and control sets')
    for ci in CALLS:
        left, right = bycase[0, ci], bycase[1, ci]
        require(left['call'] == right['call'] and left['physical_frame_order'] == right['physical_frame_order'], 'Same frame identity in both ranks')
    frames = [f for ci in CALLS for f in bycase[0, ci]['physical_frame_order']]
    require(len({f['index'] for f in frames}) == 24
            and {(f['fixture'], f['step']) for f in frames} == {(i, s) for i in range(8) for s in (1, 2, 3)}, 'All24 unique original decode frames')
    require(type(rows) is list and len(rows) == 72, 'All72 full replicated output tensors')
    outputs = {}
    for row in rows:
        key = row['rank'], row['call']['index']
        require(type(row['rank']) is int and key in bycase and row['call'] == bycase[key]['call']
                and row['variant'] in VARIANTS and key + (row['variant'],) not in outputs, 'Exact unique output provenance')
        require(type(row['raw']) is bytes and len(row['raw']) == 16384, 'Complete four-row output')
        values(row['raw'])
        outputs[key + (row['variant'],)] = row['raw']
    require(len(outputs) == 72, 'Every variant in every rank and call')
    for key, case in bycase.items():
        c = controls[key]
        require(c['call'] == case['call'] and c['original_call'] == case['original_call'], 'Original control provenance')
        for name in ('original', 'previous_own', 'previous_both', 'previous_restore', 'official_target'):
            require(type(c[name]) is bytes and len(c[name]) == 16384, 'Full old and official controls')
            values(c[name])
        require(digest(c['original']) == case['expected_attention']['logical_sha256'], 'Original observed output hash')
        require(c['original'] == c['previous_own'] == c['previous_restore']
                == outputs[key + ('own',)] == outputs[key + ('restore',)], 'All original/own/restore full bytes exact')
        require(c['previous_both'] == outputs[key + ('b4_strided',)], 'Actual previous both-official full bytes reproduced')
    for ci in CALLS:
        require(controls[0, ci]['official_target'] == controls[1, ci]['official_target'], 'Same official targets in both ranks')
        for mode in VARIANTS:
            require(outputs[0, ci, mode] == outputs[1, ci, mode], 'Replicated outputs identical in both ranks')
    records, interactions, pair_records = [], [], []
    for ci in CALLS:
        target = controls[0, ci]['official_target']
        for slot, frame in enumerate(bycase[0, ci]['physical_frame_order']):
            sl = slice(slot*4096, (slot+1)*4096)
            cells = {mode: outputs[0, ci, mode][sl] for mode in VARIANTS}
            official = target[sl]
            baseline = distance(cells['b4_strided'], official)
            for mode in VARIANTS:
                metric = distance(cells[mode], official)
                records.append(dict(call=ci, slot=slot, frame=dict(frame), variant=mode,
                    official_distance=metric, change_from_b4_strided=distance(cells[mode], cells['b4_strided']),
                    improved=metric['l2_squared'] < baseline['l2_squared'], worsened=metric['l2_squared'] > baseline['l2_squared']))
            for label, lhs, rhs in PAIRS:
                pair_records.append(dict(call=ci, slot=slot, frame=dict(frame), comparison=label,
                                         distance=distance(cells[lhs], cells[rhs])))
            a, b, c, d = [values(cells[m]) for m in VARIANTS[1:5]]
            delta = [math.fsum((w, -y, -z, x)) for x, y, z, w in zip(a, b, c, d)]
            den = math.fsum(x*x for x in values(official)); num = math.fsum(x*x for x in delta)
            interactions.append(dict(call=ci, slot=slot, frame=dict(frame), l2_squared=num,
                reference_l2_squared=den, relative_l2=math.sqrt(num/den) if den else None,
                max_abs=max(abs(x) for x in delta)))
    summary = []
    for mode in VARIANTS:
        selected = [x for x in records if x['variant'] == mode]
        summary.append(dict(variant=mode, **aggregate([x['official_distance'] for x in selected]),
                            improved_frames=sum(x['improved'] for x in selected), worsened_frames=sum(x['worsened'] for x in selected)))
    pair_summary = [dict(comparison=label, **aggregate([r['distance'] for r in pair_records if r['comparison'] == label])) for label, _, _ in PAIRS]
    num=math.fsum(x['l2_squared'] for x in interactions);den=math.fsum(x['reference_l2_squared'] for x in interactions)
    return dict(schema='minicpm5_layer1_sdk_batch_layout_analysis_v1', complete=True,
        complete_output_tensors=72, complete_output_bytes=72*16384, unique_frames=24,
        records=records, summary=summary, pair_records=pair_records, pair_summary=pair_summary,
        interactions=interactions, interaction_summary=dict(frames=24, aggregate_relative_l2=math.sqrt(num/den) if den else None,
            nonzero_frames=sum(x['l2_squared'] > 0 for x in interactions)),
        output_sha256={f'{r}:{ci}:{mode}': digest(raw) for (r, ci, mode), raw in outputs.items()},
        original_and_restore_controls_exact=True, previous_both_official_control_exact=True, both_ranks_identical=True,
        interaction_definition='b1_contiguous - b4_contiguous - b1_strided + b4_strided; not error contribution percentages',
        descriptive_precision='stdlib FP64 distances of complete BF16 rows',
        authenticated_file_provenance_checked_here=False, runtime_admitted=False, root_cause_proven=False,
        full_model_qualified=False, new_speedup_claim=False)
