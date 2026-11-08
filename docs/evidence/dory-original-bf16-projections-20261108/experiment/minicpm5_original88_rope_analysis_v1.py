"""Six actual RoPE tensors only; raw reads and NumPy are serial-runtime-only.

Reuses frozen first-five admission, typed indexing and raw IO without modifying
its globals. This module does not execute RoPE or construct coefficient values.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import re
import sys
import minicpm5_original88_first_five_analysis_v2 as five

p, parser, post, same = five.p, five.parser, five.post, five.same
ROOT = five.ROOT
LABELS = ('q_rope_input', 'k_rope_input', 'rope_cos', 'rope_sin', 'q_post_rope', 'k_post_rope')
BOUNDARIES = dict(zip(LABELS, ('qk_rope_inputs',)*2 + ('rope_parameters',)*2 + ('qk_post_rope',)*2))
PENDING = five.PENDING[3:]
FIRST_SOURCE_SHA = '516f4dfda864aaa91f08228666b1857ccbab4ef459b134f6c81109a7e99d67a1'

def reference(name, sha):
    return dict(path=str(ROOT/'research'/name), sha256=sha)

DESIGN = reference('minicpm5-original88-rope-three-boundary-interface-design-20261107-v1.json', 'dc7d06c11f0e4cb52c03f67d98346edf81f3b35cf56fcee9206d38bc47b3331e')
DESIGN_REVIEW = reference('minicpm5-original88-rope-three-boundary-root-design-review-20261107-v1.json', '9b526dc5b5a477d8e0cc6ea70d6265df05f8e159eeb2155e8092a9c1479e6e70')
FIRST_RESULT = reference('minicpm5-original88-first-five-r1-20261107-v2.json', '622782d1e297a3b3a59dfc95cfede70b48cbf1024c602bebb55a3d7cca8b44fd')
FIRST_REVIEW = reference('minicpm5-original88-first-five-independent-metadata-review-20261107-v1.json', '108c20d3f56d3fcbe702d945c1ac6aeade9243161474f841238d5016936812a9')
SUPPLEMENT = reference('root-post-mini-firstfive-additional-process-exits-20261107-v1.json', 'afa0914e75f2d1bd3f7ac100c122e3132c26c13b7cf17e5355cec0677fae6afb')

def prerequisites(bound):
    """Pinned JSON/source only; process checks are deferred until serial lease."""
    p.require(p.sha(Path(five.__file__)) == FIRST_SOURCE_SHA, 'Frozen first-five library')
    design = p.read_ref(DESIGN); review = p.read_ref(DESIGN_REVIEW)
    p.require(review['reviewer'] == '/root' and review['passed'] is True and review['blockers'] == [], 'Accepted RoPE design')
    same(review['design'], DESIGN, 'Exact reviewed design')
    for ref in design['source_refs'].values():
        p.require(p.sha(Path(ref['path'])) == ref['sha256'], 'Original RoPE source remains pinned')
    result = p.read_ref(FIRST_RESULT); independent = p.read_ref(FIRST_REVIEW)
    same(result['binding'], bound['binding'], 'First-five same finalized run')
    p.require(result['complete_coverage'] is True and len(result['rows']) == 960, 'Complete first-five prerequisite')
    p.require(independent['passed'] is True and independent['metadata_only'] is True and independent['coverage']['rows'] == 960, 'Accepted independent first-five metadata audit')
    p.require(any(row['path'] == FIRST_RESULT['path'] and row['sha256'] == FIRST_RESULT['sha256'] for row in independent['inputs']), 'Review binds first-five result')
    supplement = p.read_ref(SUPPLEMENT)
    p.require(supplement['schema'] == 'root_later_diagnostic_processes_exit_inventory_v1' and supplement['reviewer'] == '/root' and supplement['all_actual_exits_observed'] is True, 'Root later-process exit supplement')
    same(supplement['base_catalog'], bound['binding']['catalog'], 'Supplement same final444 catalog')
    owners = [post.process(row) for row in supplement['subsequent_root_processes']] + [post.process(supplement['probe_child'])]
    p.require(len(owners) == 5 and len({(o['pid'], o['start_ticks']) for o in owners}) == 5, 'All five subsequent owners')
    for ref in supplement['exit_evidence']:
        p.read_ref(post.reference(ref))
    return owners

def prepare_index(plan, reports, read_receipt):
    """Reuse original typed context/physical row checks; attach six snapshots."""
    snapshots = {}
    def reader(ref):
        record = read_receipt(ref)
        snapshots[ref['path']] = {row['label']: row for row in record['observer']['snapshots']}
        return record
    index = five.prepare_index(plan, reports, reader)
    for entry in index.values():
        observed = snapshots[entry['receipt']['path']]
        entry['rope'] = {label: observed.get(label) for label in (*LABELS, 'rotary_route', 'position_ids', 'rope_position_ids')}
    return index

def geometry(plan, entry, rank):
    p.require(type(rank) is int and rank in (0, 1), 'Exact TP2 rank')
    native = entry['context']['arm'] == 'HF_native'
    p.require(type(entry['context']['rank']) is int and entry['context']['rank'] == (-1 if native else rank), 'Exact capture rank')
    p.require(entry['phase'] in ('prefill', 'decode'), 'Original phase')
    fixture = next(x for x in plan['fixtures'] if x['id'] == entry['frame']['fixture'])
    length = len(fixture['input_ids']) if entry['phase'] == 'prefill' else 1
    sequence = length if native else 1 if entry['phase'] == 'decode' else plan['graph_calls'][entry['call']]['bucket']
    batch = 1 if native or entry['phase'] == 'prefill' else 4
    row = entry['physical_row']
    p.require(type(row) is int and row == (0 if batch == 1 else entry['frame']['slot']), 'Actual sorted-slot physical row')
    p.require(type(sequence) is int and sequence >= length and 0 <= row < batch, 'Real-token domain')
    return native, length, sequence, batch, row

def support(entry, batch, sequence):
    values = entry['rope']; native = entry['context']['arm'] == 'HF_native'
    for label in ('rotary_route', 'position_ids', 'rope_position_ids'):
        if values.get(label) is None:
            return 'missing RoPE support label ' + label
    expected = dict(function='native.apply_rotary_pos_emb', position_source='actual_original_model_call') if native else dict(function='SDK.apply_rotary_embedding/apply_rotary_pos_emb', polar=False)
    same(values['rotary_route']['value'], expected, 'Actual original non-polar rotary route')
    positions = []
    for label in ('position_ids', 'rope_position_ids'):
        value = values[label]['value']
        if type(value) is not dict or set(value) != {'metadata', 'reference'}:
            return 'incomplete RoPE support tensor ' + label
        meta = value['metadata']
        if meta['selection'] is not None:
            return 'partial RoPE support tensor ' + label
        same(meta['selected_shape'], [batch, sequence], 'Exact position domain')
        same(meta['source_shape'], meta['selected_shape'], 'Complete position capture')
        p.require(meta['source_dtype'] == 'torch.int64', 'Actual int64 position support')
        same([meta['selected_bytes'], value['reference']['bytes']], [batch*sequence*8]*2, 'Exact position bytes')
        p.require(re.fullmatch('[0-9a-f]{64}', value['reference']['sha256']) is not None, 'Position payload digest')
        positions.append(value['reference']['sha256'])
    same(positions[0], positions[1], 'Actual position_ids and rotary positions byte digest')
    return None

def selection(plan, entry, label, rank):
    p.require(label in LABELS, 'Only six actual RoPE labels')
    native, length, sequence, batch, row = geometry(plan, entry, rank)
    missing = support(entry, batch, sequence)
    if missing:
        return dict(blocked=missing)
    snapshot = entry['rope'].get(label)
    if snapshot is None:
        return dict(blocked='missing applicable label ' + label)
    payload = snapshot['value']
    if type(payload) is not dict or set(payload) != {'metadata', 'reference'}:
        return dict(blocked='label is not one complete tensor')
    meta = payload['metadata']; shape = meta['selected_shape']; dtype = meta['source_dtype']
    if meta['selection'] is not None or not parser.typed_equal(shape, meta['source_shape']):
        return dict(blocked='full tensor not captured')
    if dtype != 'torch.bfloat16':
        return dict(blocked='only actual BF16 capture admitted', dtype=dtype, status='dtype_mismatch')
    coefficient = label in ('rope_cos', 'rope_sin')
    heads = (16 if label.startswith('q_') else 2) if native else (8 if label.startswith('q_') else 1)
    expected = [batch, sequence, 128] if coefficient else [batch, heads, sequence, 128]
    same(shape, expected, 'Exact native L / CPU bucket or decode1, B/H/D geometry')
    p.require(type(meta['source_stride']) is list and len(meta['source_stride']) == len(shape) and all(type(x) is int and x >= 0 for x in meta['source_stride']) and type(meta['source_offset']) is int and meta['source_offset'] >= 0, 'Original source layout metadata')
    same([meta['selected_bytes'], payload['reference']['bytes']], [math.prod(shape)*2]*2, 'Exact contiguous BF16 clone bytes')
    local_heads = 8 if label.startswith('q_') else 1
    start = rank*local_heads if native else 0
    slices = [[row, row+1], [0, length], [0, 128]] if coefficient else [[row, row+1], [start, start+local_heads], [0, length], [0, 128]]
    return dict(payload=payload, shape=shape, dtype=dtype, slices=slices, row=row,
                layout=dict(shape=meta['source_shape'], stride=meta['source_stride'], offset=meta['source_offset'], selection=meta['selection']))

def schedule(plan, index):
    tasks = []
    for frame in plan['frames']:
        for rank in (0, 1):
            for left, right in five.PAIRS:
                a = index[left, -1 if left == 'HF_native' else rank, frame['index']]
                b = index[right, rank, frame['index']]
                same(a['frame'], frame, 'Exact left frame'); same(b['frame'], frame, 'Exact right frame')
                same([a['call'], a['phase']], [b['call'], b['phase']], 'Same original call and phase')
                for label in LABELS:
                    tasks.append(dict(frame=copy.deepcopy(frame), call=a['call'], phase=a['phase'], rank=rank, left_arm=left, right_arm=right,
                                      boundary=BOUNDARIES[label], label=label, left=selection(plan, a, label, rank), right=selection(plan, b, label, rank),
                                      left_receipt=a['receipt'], right_receipt=b['receipt']))
    p.require(len(tasks) == 1152 and len({(t['frame']['index'], t['rank'], t['left_arm'], t['right_arm'], t['label']) for t in tasks}) == 1152, 'All192 pairs times six components')
    return tasks

def view(np, data, chosen):
    """Read saved contiguous clone; keep singleton batch and actual head axis."""
    p.require(chosen['dtype'] == 'torch.bfloat16', 'BF16 view only')
    raw = np.frombuffer(data, dtype='<u2').reshape(chosen['shape'])
    exact = raw[tuple(slice(*span) for span in chosen['slices'])]
    return exact, (exact.astype(np.uint32) << 16).view(np.float32)

def compare(np, task, reader, guard):
    left, right = task['left'], task['right']
    row = {key: copy.deepcopy(task[key]) for key in ('frame', 'call', 'phase', 'rank', 'left_arm', 'right_arm', 'boundary', 'label', 'left_receipt', 'right_receipt')}
    row.update(left_layout=left.get('layout'), right_layout=right.get('layout'), left_slice=left.get('slices'), right_slice=right.get('slices'), left_dtype=left.get('dtype'), right_dtype=right.get('dtype'))
    if 'blocked' in left or 'blocked' in right:
        row.update(status='dtype_mismatch' if any(x.get('status') == 'dtype_mismatch' for x in (left, right)) else 'blocked_missing_or_unmapped', reason=left.get('blocked', right.get('blocked')))
        return row
    p.require(left['dtype'] == right['dtype'] == 'torch.bfloat16', 'Exact admitted BF16 pair')
    ar, a = view(np, reader(left['payload'], guard), left)
    br, b = view(np, reader(right['payload'], guard), right)
    same(list(a.shape), list(b.shape), 'Same complete logical RoPE domain')
    bit_equal = bool(np.array_equal(ar, br)); a = a.astype(np.float64); b = b.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b); nonfinite = int(a.size - np.count_nonzero(finite)); delta = np.abs(a[finite] - b[finite])
    dot = float(np.sum(a[finite]*b[finite], dtype=np.float64))
    norm = float(np.sqrt(np.sum(a[finite]*a[finite], dtype=np.float64)*np.sum(b[finite]*b[finite], dtype=np.float64)))
    row.update(status='comparable_exact' if bit_equal else 'comparable_different', reason=None, compared_elements=int(a.size), raw_exact=bit_equal,
               different_elements=int(np.count_nonzero(ar != br)), max_abs=float(delta.max()) if delta.size else None, mean_abs=float(delta.mean()) if delta.size else None,
               cosine=dot/norm if norm > 0 and nonfinite == 0 else None, nonfinite_elements=nonfinite)
    return row

def summarize(plan, rows):
    def counts(selected):
        return dict(rows=len(selected), exact=sum(x['status'] == 'comparable_exact' for x in selected), different=sum(x['status'] == 'comparable_different' for x in selected),
                    blocked=sum(x['status'] in ('blocked_missing_or_unmapped', 'dtype_mismatch') for x in selected), nonfinite_elements=sum(x.get('nonfinite_elements', 0) for x in selected))
    frames = []
    for frame in plan['frames']:
        selected = [row for row in rows if row['frame']['index'] == frame['index']]
        p.require(len(selected) == 36, 'All36 component rows per frame')
        frames.append(dict(frame=frame, **counts(selected)))
    boundaries = {name: counts([row for row in rows if row['boundary'] == name]) for name in dict.fromkeys(BOUNDARIES.values())}
    p.require(len(frames) == 32 and all(x['rows'] == 384 for x in boundaries.values()), 'All32 frames and192 pairs per boundary')
    return dict(frames=frames, boundaries=boundaries, pending_boundaries=list(PENDING), full_attention_analyzed=False)

def run(audit_ref, catalog_ref, exit_ref, output_prefix, source_sha256):
    source = Path(__file__).resolve()
    p.require(re.fullmatch('[0-9a-f]{64}', source_sha256) is not None and p.sha(source) == source_sha256, 'Root-provided actual code SHA')
    bound = five.load_finalized(audit_ref, catalog_ref, exit_ref); owners = prerequisites(bound)
    spec, plan = bound['spec'], bound['plan']; prefix = Path(output_prefix)
    p.require(prefix.parent == ROOT/'research' and re.fullmatch('minicpm5-original88-rope-[a-z0-9-]+', prefix.name), 'Fresh RoPE output prefix')
    paths = {name: Path(str(prefix)+suffix) for name, suffix in (('report', '.json'), ('csv', '.csv'), ('summary', '-frames.json'), ('failure', '-failure.json'))}
    p.require(all(not path.exists() for path in paths.values()), 'Never overwrite evidence')
    fd, owner, lock = post.external.acquire_lock()
    try:
        def guard():
            p.clear_window(); p.lock_proof(owner, lock); p.control_usage(spec); p.active_heavy()
            same({name: p.sha(ROOT/name) for name in spec['canonical_sha256']}, spec['canonical_sha256'], 'Canonical unchanged')
            p.require(p.sha(source) == source_sha256 and p.sha(Path(five.__file__)) == FIRST_SOURCE_SHA, 'Analysis code unchanged')
        guard(); p.gone(bound['finalizer_owner']); p.predecessor_drain(bound['catalog'])
        for previous in owners:
            p.gone(previous)
        tasks = schedule(plan, prepare_index(plan, bound['reports'], parser.checked_reference))
        import numpy as np
        rows = [compare(np, task, five.read_raw, guard) for task in tasks]
        summary = summarize(plan, rows); guard()
        binding = dict(bound['binding'], first_five_result=FIRST_RESULT, first_five_independent_review=FIRST_REVIEW, design=DESIGN, design_review=DESIGN_REVIEW, later_process_exit_supplement=SUPPLEMENT,
                       code=dict(path=str(source), sha256=source_sha256), first_five_library=dict(path=five.__file__, sha256=FIRST_SOURCE_SHA))
        report = dict(schema='minicpm5_original88_rope_analysis_v1', binding=binding, execution_owner=owner, checked_later_owners=owners, output_prefix=str(prefix),
                      self_exit_claimed=False, root_must_observe_actual_exit=True, frames=32, frame_rank_pairs=192, rows=rows, implemented_boundaries=list(dict.fromkeys(BOUNDARIES.values())),
                      component_labels=list(LABELS), pending_boundaries=list(PENDING), complete_coverage=all(x['blocked'] == 0 for x in summary['frames']),
                      full_attention_analyzed=False, original_remaining_gates=copy.deepcopy(plan['remaining_gates']), new_thresholds=False,
                      original_logits_thresholds=dict(cosine_min=0.995, kl_max=0.02, max_abs_max=2, aggregate_argmax_min=0.9, hf_margin_gt_0_5_must_match=True),
                      full_model_qualified=False, new_speedup_claim=False, causal_operator_proven=False)
        p.once(paths['report'], report); p.once(paths['summary'], dict(binding=binding, **summary))
        fields = list(rows[0])
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
        with paths['csv'].open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, separators=(',', ':')) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        guard()
        return {name: dict(path=str(path), sha256=p.sha(path)) for name, path in paths.items() if name != 'failure'}
    except Exception as error:
        p.once(paths['failure'], dict(schema='minicpm5_original88_rope_analysis_failure_v1', error_type=type(error).__name__, error=str(error), execution_owner=owner,
                                    source_sha256=source_sha256, later_process_exit_supplement=SUPPLEMENT, self_exit_claimed=False, full_model_qualified=False))
        raise
    finally:
        post.external.release_lock(fd)

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    for name in ('finalization', 'catalog', 'finalizer-exit'):
        cli.add_argument('--'+name, required=True); cli.add_argument('--'+name+'-sha256', required=True)
    cli.add_argument('--output-prefix', required=True); cli.add_argument('--source-sha256', required=True)
    args = cli.parse_args(); p.require(sys.dont_write_bytecode and __debug__, 'python -B required')
    refs = [dict(path=getattr(args, name), sha256=getattr(args, name+'_sha256')) for name in ('finalization', 'catalog', 'finalizer_exit')]
    print(json.dumps(run(*refs, args.output_prefix, args.source_sha256), indent=2))
