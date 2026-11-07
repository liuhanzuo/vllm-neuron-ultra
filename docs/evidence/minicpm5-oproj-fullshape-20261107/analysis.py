"""Full original-shape saved-context, layer0 output-projection intervention only.

Inherits v2 science/successor binding and failure protections.
All six formulas now execute at original L/E before slicing metrics.
Root supplies serial ownership/guard; torch is imported only inside run().
No model, attention, compiler, collectives, service or queue is launched.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

import minicpm5_original88_first_five_analysis_v2 as five

p, same, ROOT = five.p, five.same, five.ROOT
DESIGN = dict(path=str(ROOT/'research/minicpm5-saved-oproj-fullshape-intervention-design-20261107-v2.json'),
              sha256='dcfa57c14ff4bc3bbc05a7cbd74262c258ca6582ff9beb1d32af6fed3c3fe897')
FIRST_SHA = '516f4dfda864aaa91f08228666b1857ccbab4ef459b134f6c81109a7e99d67a1'
FORMULAS = ('hf_bf16_full', 'sdk_bf16_split', 'bf16_split_wide_sum',
            'fp32_full_final', 'fp32_split_final', 'fp32_split_round_local')


def prepare(bound, design, read_ref):
    """Pinned small JSON only; no payload stat/read or framework imports."""
    previous = read_ref(design['previous_projection']['report'])
    previous_exit = read_ref(design['previous_projection']['actual_exit'])
    p.require(previous_exit['actual_exit_observed'] is True and type(previous_exit['actual_exit_code']) is int and previous_exit['actual_exit_code'] == 0, 'Actual projection exit0')
    same(previous_exit['outputs']['report'], design['previous_projection']['report'], 'Actual previous projection output')
    same(previous_exit['owner'], previous['execution_owner'], 'Previous projection owner')
    p.require(previous['summary']['all_full_shape_controls_exact'] is True and previous['summary']['full_shape_controls'] == 24, 'All24 previous full controls')
    p.require(len(design['previous_projection']['shape1_failed']) == 5, 'Retain all five previous shape1 failures')
    original = read_ref(design['original_result'])
    for key, value in bound['binding'].items(): same(original['binding'][key], value, 'Same finalized original88')
    p.require(original['complete_selected_coverage'] is True and len(original['rows']) == 576, 'Complete saved comparison')
    exitref = design['references']['minicpm5-original88-prefill-attention-root-exit-r1-20261107-v1.json']
    ended = read_ref(exitref)
    p.require(ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int and ended['actual_exit_code'] == 0, 'Actual saved-analysis exit0')
    same(ended['outputs']['report'], design['original_result'], 'Exited report binding')
    same(ended['owner'], original['execution_owner'], 'Actual analysis owner')
    # The finalized original88 plan is a reviewed successor of the design's
    # original scientific input plan, with a new output namespace/source closure.
    # Authenticate the actual successor; never substitute the old plan in bound.
    original_plan = read_ref(design['original_plan'])
    same(design['original_plan'], p.plan_source.ORIGINAL, 'Exact inherited scientific plan reference')
    same(read_ref(bound['spec']['input_plan']), bound['plan'], 'Actual finalized successor input plan')
    same(p.digest(bound['plan']), bound['spec']['input_plan']['canonical_sha256'], 'Successor canonical digest')
    same(bound['plan']['output_root'], bound['spec']['output_root'], 'Actual successor output namespace')
    same(bound['spec']['science_handoff'], p.plan_source.SCIENCE, 'Actual reviewed science transform')
    same(bound['spec']['scientific_review'], p.plan_source.ACCEPT, 'Actual science acceptance')
    science = read_ref(p.plan_source.SCIENCE); accepted = read_ref(p.plan_source.ACCEPT)
    p.require(p.plan_source.validate_namespace_equivalence(original_plan, bound['plan'], science, accepted) is True,
              'Only exact reviewed namespace and additive source transform')
    for key, value in original_plan.items():
        if key not in ('output_root', 'source_sha256', 'writer_contexts'):
            same(bound['plan'][key], value, 'Unchanged scientific core '+key)
    official = bound['plan']['official_layer0_weights']; weight = design['weight']
    same(weight['tensor'], official['tensors'][weight['name']], 'Official output weight header')
    same(weight['file'], next(x for x in bound['plan']['model_files'] if x['path'] == official['path']), 'Original weight file binding')
    same(weight['tensor']['shape'], [2048, 2048], 'Full official output projection')
    same(weight['tensor']['dtype'], 'BF16', 'Official BF16 weight')
    same(weight['shard_axis'], 1, 'Shard input features, not output features')
    same(weight['rank_slices'], [[0, 1024], [1024, 2048]], 'Original TP2 feature order')
    p.require(weight['bias'] is False, 'Original bias-free projection')
    observed = {}
    for row in original['rows']:
        if row['stage'] not in ('context_before_o', 'attention_output'): continue
        for side in ('left', 'right'):
            arm = row[side+'_arm']; rank = -1 if arm == 'HF_native' else row['rank']
            key = (row['frame']['index'], arm, rank, row['stage'])
            value = dict(arm=arm, rank=rank, stage=row['stage'], reference=row[side+'_reference'],
                         metadata=row[side+'_layout'], receipt=row[side+'_receipt'])
            if key in observed: same(observed[key], value, 'Same repeated observed payload')
            observed[key] = value
    same([x['frame']['index'] for x in design['capture_inputs']], [0, 1, 2, 3, 16, 17, 18, 19], 'All8 prefills')
    contexts = []
    for row in design['capture_inputs']:
        frame = row['frame']; context = dict(frame=frame, call=row['call'], group=row['group'], prediction_query=row['prediction_query'], payloads={})
        call = bound['plan']['calls'][row['call']]
        same(call['frames'], [frame], 'Original complete prefill call')
        same(call['phase'], 'prefill', 'No decode intervention')
        length = row['prediction_query']+1; width = bound['plan']['graph_calls'][row['call']]['bucket']
        for value in row['payloads']:
            same(observed[frame['index'], value['arm'], value['rank'], value['stage']], value, 'Pinned actual saved payload')
            meta = value['metadata']; native = value['arm'] == 'HF_native'
            features = 2048 if native or value['stage'] == 'attention_output' else 1024
            sequence = length if native else width; shape = [1, sequence, features]
            same(meta['source_shape'], shape, 'Original full shape')
            same(meta['selected_shape'], shape, 'Full saved shape')
            same(meta['source_stride'], [sequence*features, features, 1], 'Original contiguous input/output layout')
            same(meta['source_offset'], 0, 'Original zero offset')
            p.require(meta['source_dtype'] == 'torch.bfloat16' and meta['selection'] is None, 'Full BF16 captured data')
            same([meta['selected_bytes'], value['reference']['bytes']], [sequence*features*2]*2, 'Exact saved byte count')
            context['payloads'][value['arm'], value['rank'], value['stage']] = value
        p.require(len(context['payloads']) == 10, 'Native2 plus CPU8 payloads')
        context.update(length=length, width=width); contexts.append(context)
    p.require(len(observed) == 80, 'Exactly80 relevant saved payloads')
    return contexts


def weight_tensor(torch, design, guard):
    """Read only pinned 8MiB matrix interval; never mmap/hash the full checkpoint."""
    spec = design['weight']; reference = spec['file']; path = Path(reference['path']); guard()
    same(five.identity(path), reference['identity'], 'Inherited official file identity unchanged')
    header = p.read_ref(spec['header']); fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        expected = [reference['identity'][k] for k in ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')]
        same([opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns], expected, 'Opened original checkpoint')
        prefix = os.pread(fd, 8, 0); p.require(len(prefix) == 8, 'Header prefix')
        header_len, = struct.unpack('<Q', prefix); p.require(0 < header_len < 1024**2, 'Bounded original header')
        data = os.pread(fd, header_len, 8); p.require(len(data) == header_len, 'Full bounded header')
        same(json.loads(data), header, 'Original header exactly reproduced')
        same(header[spec['name']], spec['tensor'], 'Original matrix interval')
        start, end = spec['tensor']['data_offsets']; count = end-start
        p.require(count == 2048*2048*2 and 8+header_len+end <= opened.st_size, '8MiB matrix bounds')
        buffer = bytearray(); digest = hashlib.sha256()
        while len(buffer) < count:
            guard(); part = os.pread(fd, min(1024**2, count-len(buffer)), 8+header_len+start+len(buffer))
            p.require(bool(part), 'Matrix interval not truncated'); buffer.extend(part); digest.update(part)
        after = os.fstat(fd)
        same([after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns], expected, 'Unchanged checkpoint descriptor')
    finally: os.close(fd)
    same(five.identity(path), reference['identity'], 'Unchanged checkpoint path')
    weight = torch.frombuffer(buffer, dtype=torch.bfloat16).reshape(2048, 2048)
    return weight, dict(file=reference, header=spec['header'], tensor=spec['tensor'], name=spec['name'],
        interval_sha256=digest.hexdigest(), bytes_read=count, full_file_rehashed=False,
        authentication='Original finalized input binding and unchanged file identity; exact header; new interval digest only.')


def digest(torch, value):
    raw = value.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast('B')).hexdigest()


def measure(torch, left, right):
    same(list(left.shape), list(right.shape), 'Exact comparison shape')
    p.require(left.dtype == right.dtype == torch.bfloat16, 'Final BF16 output comparison')
    ar = left.contiguous().view(torch.int16); br = right.contiguous().view(torch.int16)
    a = left.double(); b = right.double(); finite = torch.isfinite(a) & torch.isfinite(b)
    nonfinite = int(a.numel()-torch.count_nonzero(finite)); a = a[finite]; b = b[finite]
    delta = (a-b).abs(); norm = torch.sqrt(torch.sum(a*a)*torch.sum(b*b))
    return dict(raw_exact=bool(torch.equal(ar, br)), different_elements=int(torch.count_nonzero(ar != br)),
                compared_elements=left.numel(), nonfinite_elements=nonfinite,
                max_abs=float(delta.max()) if delta.numel() else None, mean_abs=float(delta.mean()) if delta.numel() else None,
                cosine=float(torch.sum(a*b)/norm) if float(norm) > 0 and not nonfinite else None)


def formulas(torch, x, weight, shards):
    """All six formulas share exactly x; each precision contrast reuses partials."""
    parts = [x[..., :1024].contiguous(), x[..., 1024:].contiguous()]
    bf = [torch.einsum('...m,mn->...n', part, shard.t()) for part, shard in zip(parts, shards)]
    fp = [torch.einsum('...m,mn->...n', part.float(), shard.float().t()) for part, shard in zip(parts, shards)]
    out = dict(hf_bf16_full=torch.nn.functional.linear(x, weight),
        sdk_bf16_split=bf[0]+bf[1], bf16_split_wide_sum=(bf[0].float()+bf[1].float()).bfloat16(),
        fp32_full_final=torch.nn.functional.linear(x.float(), weight.float()).bfloat16(),
        fp32_split_final=(fp[0]+fp[1]).bfloat16(),
        fp32_split_round_local=(fp[0].bfloat16().float()+fp[1].bfloat16().float()).bfloat16())
    p.require(list(x.shape[:1]) == [1] and x.shape[1] > 1 and x.shape[2] == 2048, 'Original full L/E, never isolated query')
    p.require(all(v.dtype == torch.bfloat16 and list(v.shape) == list(x.shape) for v in out.values()), 'Finite full-shape formulas')
    partials = [dict(rank=rank, input_shape=list(parts[rank].shape), input_stride=list(parts[rank].stride()),
                     weight_shape=list(shards[rank].shape), weight_stride=list(shards[rank].stride()),
                     bf16_sha256=digest(torch, b), fp32_sha256=digest(torch, f),
                     bf16_vs_fp32_cast=measure(torch, b, f.bfloat16())) for rank, (b, f) in enumerate(zip(bf, fp))]
    return out, partials


def run(bound, output_prefix, source_sha256, guard, execution_owner):
    """Root holds serial lock/drains, including projection session73920 owner.

    Exact endpoint reproduction is a measured gate, never relaxed on mismatch.
    This worker reproduces arithmetic in one process; it does not run Gloo.
    """
    guard(); source = Path(__file__).resolve()
    p.require(p.sha(source) == source_sha256 and p.sha(Path(five.__file__)) == FIRST_SHA, 'Frozen worker/helper')
    p.require(sys.executable == '/home/ubuntu/venvs/vllm_neuron/bin/python', 'Minimal SDK interpreter only')
    design = p.read_ref(DESIGN); contexts = prepare(bound, design, p.read_ref)
    prefix = Path(output_prefix)
    p.require(prefix.parent == ROOT/'research' and re.fullmatch('minicpm5-saved-oproj-fullshape-intervention-[a-z0-9-]+', prefix.name), 'Fresh finite output prefix')
    paths = {k: Path(str(prefix)+s) for k, s in (('report', '.json'), ('summary', '-summary.json'), ('failure', '-failure.json'))}
    p.require(all(not x.exists() for x in paths.values()), 'Never overwrite evidence')
    guard()
    import torch
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        p.require(not torch.distributed.is_initialized(), 'No distributed process group needed or admitted')
        with torch.inference_mode():
            weight, weight_proof = weight_tensor(torch, design, guard)
            shards = [weight[:, a:b].contiguous() for a, b in design['weight']['rank_slices']]
            controls = []; cases = []; frame_summaries = []
            for context in contexts:
                guard(); loaded = {}
                def load(arm, rank, stage):
                    key = (arm, rank, stage)
                    if key not in loaded:
                        payload = context['payloads'][key]
                        raw = five.read_raw(dict(metadata=payload['metadata'], reference=payload['reference']), guard)
                        loaded[key] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(payload['metadata']['selected_shape'])
                    return loaded[key]
                frame = context['frame']; length = context['length']; last = length-1
                native_x = load('HF_native', -1, 'context_before_o'); native_y = load('HF_native', -1, 'attention_output')
                torch.set_num_threads(2); guard()
                native_actual = torch.nn.functional.linear(native_x, weight)
                native_metric = measure(torch, native_actual, native_y)
                controls.append(dict(frame=frame, arm='HF_native', input_shape=list(native_x.shape), input_stride=list(native_x.stride()),
                    threads=2, operation='F.linear full2048, bias=None', measured=native_metric,
                    output_sha256=digest(torch, native_actual), interpreter_matches_original=False,
                    meaning='Endpoint reproduction in SDK interpreter; not identical native runtime proof.'))
                del native_actual
                torch.set_num_threads(1); cpu_metrics = {}; cpu_inputs = {}
                for arm in ('C_Hd_control', 'C_Hc_history_generation'):
                    rank_x = [load(arm, rank, 'context_before_o') for rank in (0, 1)]
                    guard()
                    partials = [torch.einsum('...m,mn->...n', x, w.t()) for x, w in zip(rank_x, shards)]
                    reduced = partials[0]+partials[1]
                    replica_metrics = [measure(torch, reduced, load(arm, rank, 'attention_output')) for rank in (0, 1)]
                    cpu_metrics[arm] = all(m['raw_exact'] and not m['nonfinite_elements'] for m in replica_metrics)
                    controls.append(dict(frame=frame, arm=arm, input_shapes=[list(x.shape) for x in rank_x], input_strides=[list(x.stride()) for x in rank_x],
                        threads=1, operation='Actual SDK einsum spelling, BF16 local outputs, standalone BF16 sum',
                        same_actual_gloo_executed=False, rank_output_metrics=replica_metrics,
                        local_partial_sha256=[digest(torch, x) for x in partials], output_sha256=digest(torch, reduced),
                        raw_exact=cpu_metrics[arm]))
                    cpu_inputs[arm] = torch.cat(rank_x, dim=-1).contiguous()
                    del partials, reduced
                full_gate = native_metric['raw_exact'] and not native_metric['nonfinite_elements'] and all(cpu_metrics.values())
                frame_cases = []
                input_rows = [('HF_native', native_x)] + list(cpu_inputs.items())
                for arm, x in input_rows:
                    torch.set_num_threads(2 if arm == 'HF_native' else 1)
                    guard(); result, partials = formulas(torch, x, weight, shards)
                    target = native_y if arm == 'HF_native' else load(arm, 0, 'attention_output')
                    baseline = 'hf_bf16_full' if arm == 'HF_native' else 'sdk_bf16_split'
                    row_control = measure(torch, result[baseline], target)
                    eligible = full_gate and row_control['raw_exact'] and not row_control['nonfinite_elements']
                    measured = {name: dict(output_sha256=digest(torch, output), vs_same_input_hf_bf16_full=measure(torch, output, result['hf_bf16_full']),
                        vs_saved_source_full=measure(torch, output, target),
                        real_tokens_vs_same_input_full=measure(torch, output[:, :length, :], result['hf_bf16_full'][:, :length, :]),
                        prediction_vs_same_input_full=measure(torch, output[:, last:last+1, :], result['hf_bf16_full'][:, last:last+1, :]),
                        real_tokens_vs_saved=measure(torch, output[:, :length, :], target[:, :length, :]),
                        prediction_vs_saved=measure(torch, output[:, last:last+1, :], target[:, last:last+1, :]),
                        real_tokens_vs_saved_native=measure(torch, output[:, :length, :], native_y),
                        prediction_vs_saved_native=measure(torch, output[:, last:last+1, :], native_y[:, last:last+1, :])) for name, output in result.items()}
                    contrasts = [dict(left=c['left'], right=c['right'], interpretation=c['scope'], measured=measure(torch, result[c['left']], result[c['right']]),
                        real_tokens=measure(torch, result[c['left']][:, :length, :], result[c['right']][:, :length, :]),
                        prediction_query=measure(torch, result[c['left']][:, last:last+1, :], result[c['right']][:, last:last+1, :])) for c in design['controlled_contrasts']]
                    case = dict(frame=frame, call=context['call'], group=context['group'], input_arm=arm, prediction_query=last,
                        input_sha256=digest(torch, x), input_shape=list(x.shape), input_stride=list(x.stride()), threads=2 if arm == 'HF_native' else 1,
                        arithmetic_scope='full_original_L_or_E_before_metric_slicing', logical_length=length, physical_width=x.shape[1],
                        full_shape_reproduction_gate=full_gate, full_shape_case_control=row_control,
                        original_execution_causal_eligible=eligible, original_cause_proven=False,
                        local_formula_trial_complete=True, formulas=measured, partials=partials, contrasts=contrasts,
                        shared_partial_provenance='Both sum-precision formulas reuse bf[0:2]; both local-cast formulas reuse fp[0:2]. No rerun between each paired formula.',
                        limit='Only these saved full contexts and layer0 projection; original local outputs were not captured; wider CPU math is not a device accumulator assertion. CPU-vs-saved-native metrics include both existing context and geometry differences; use same-input formulas for causal contrasts.')
                    cases.append(case); frame_cases.append(case)
                frame_summaries.append(dict(frame=frame, full_shape_reproduction_gate=full_gate,
                    fixed_input_cases=3, full_case_controls_exact=sum(c['full_shape_case_control']['raw_exact'] for c in frame_cases),
                    eligible_cases=sum(c['original_execution_causal_eligible'] for c in frame_cases),
                    same_bf16_partials_sum_precision_changed=sum(not c['contrasts'][1]['measured']['raw_exact'] for c in frame_cases),
                    same_fp32_partials_local_rounding_changed=sum(not c['contrasts'][2]['measured']['raw_exact'] for c in frame_cases)))
                loaded.clear()
            same([len(controls), len(cases)], [24, 24], 'All8frames3sources full controls/cases')
            guard(); p.require(p.sha(source) == source_sha256, 'Worker source unchanged')
            binding = dict(bound['binding'], design=DESIGN, code=dict(path=str(source), sha256=source_sha256),
                           original_result=design['original_result'], original_exit=design['references']['minicpm5-original88-prefill-attention-root-exit-r1-20261107-v1.json'],
                           previous_projection=design['previous_projection'],
                           original_scientific_plan=design['original_plan'], actual_successor_plan=bound['spec']['input_plan'],
                           science_transform=p.plan_source.SCIENCE, science_acceptance=p.plan_source.ACCEPT)
            summary = dict(frames=frame_summaries, full_shape_controls=24, fixed_input_cases=24, formula_results=144,
                all_full_shape_controls_exact=all(x['full_shape_reproduction_gate'] for x in frame_summaries),
                original_execution_eligible_cases=sum(c['original_execution_causal_eligible'] for c in cases),
                local_math_trial_complete=True, original_cause_proven=False, full_model_qualified=False, new_speedup_claim=False,
                arithmetic_scope='full_original_L_or_E', previous_shape1_failures=design['previous_projection']['shape1_failed'])
            report = dict(schema='minicpm5_saved_oproj_fullshape_intervention_v1', binding=binding, execution_owner=execution_owner,
                self_exit_claimed=False, root_must_observe_actual_exit=True, runtime=dict(python=sys.executable, torch_version=torch.__version__,
                    threads_native_control_and_formulas=2, threads_sdk_control_and_formulas=1, no_actual_collective=True, original_native_interpreter_different=True),
                weight=weight_proof, capture_inputs=design['capture_inputs'], full_shape_controls=controls, cases=cases, summary=summary,
                original_remaining_gates=bound['plan']['remaining_gates'], thresholds_changed=False, attention_recomputed=False,
                model_loaded=False, compiled=False, original_cause_proven=False, full_model_qualified=False, new_speedup_claim=False)
            p.once(paths['report'], report); p.once(paths['summary'], dict(binding=binding, **summary)); guard()
            return {k: dict(path=str(v), sha256=p.sha(v)) for k, v in paths.items() if k != 'failure'}
    except Exception as error:
        try:
            p.once(paths['failure'], dict(schema='minicpm5_saved_oproj_fullshape_intervention_failure_v1', error_type=type(error).__name__,
                error=str(error), execution_owner=execution_owner, source_sha256=source_sha256, self_exit_claimed=False, full_model_qualified=False))
        except BaseException as secondary:
            try: print('Secondary failure writing diagnostic: '+repr(secondary), file=sys.stderr)
            except BaseException: pass
        raise
    finally:
        primary = sys.exc_info()[1]
        try: torch.set_num_threads(original_threads)
        except BaseException as secondary:
            if primary is None: raise
            try: print('Secondary failure restoring thread count: '+repr(secondary), file=sys.stderr)
            except BaseException: pass
