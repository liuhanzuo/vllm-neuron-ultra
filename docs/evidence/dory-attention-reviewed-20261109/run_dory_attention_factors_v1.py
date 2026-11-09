"""Serial SDPA diagnosis on authenticated original Q and complete KV histories.

No weight/model modification or model qualification. The original 87 marks,
including four wider prefill calls and all seven failures, remain in scope.
"""
import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
import dory_after_first5_review_v1 as history
from dory_layer0_serial_contract_v2 import read_pinned, require
from dory_generation_execution_provenance_v1 import review_source_pins
from dory_first5_cache_replay_v1 import FirstFiveCacheReplay
from dory_generation_cache_replay_v1 import OrdinaryFiles
from review_dory_layer0_projection_saved_v2 import bound_metadata, row
from explain_dory_long_cache_differences_v1 import differences
from nemotron9b_package_controller_v16 import acquire, release
from run_dory_layer0_serial_v2 import save, free, memory_free

ARMS = [''.join(x) for x in itertools.product('IF', repeat=3)]
EXTRA = ['reference_full_extent_strided', 'reference_full_extent_contiguous', 'incremental_row_replay']
LAYERS = (0, 2, 4)
BUDGET = dict(wall_seconds=3600, cpu_seconds=3400, address_space_bytes=12*2**30,
              initial_memory_available_bytes=8*2**30, memory_available_floor_bytes=3*2**30,
              root_floor=28*2**30, nvme_floor=8*2**30, nvme_new_bytes=512*2**20, cpus=[4])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(spec, out, guard, receipt):
    import torch
    require(str(torch.__version__) == '2.14.0+cpu', 'Original CPU PyTorch version')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.set_grad_enabled(False)
    source = Path(spec['official_source'])/'tests/models/test_dory.py'
    raw_source = source.read_bytes()
    require(digest(raw_source) == spec['source_sha256'][str(source)], 'Original ReferenceAttention source')
    parsed = ast.parse(raw_source, filename=str(source))
    classes = [n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == 'ReferenceAttention']
    require(len(classes) == 1, 'Unique original reference attention class')
    namespace = dict(torch=torch, nn=torch.nn)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), 'exec'), namespace)
    ReferenceAttention = namespace['ReferenceAttention']
    contract = history.load(receipt['post_exit_contract'])
    producer = history.load(contract['producer_spec'])
    root = Path(producer['output'])/'diagnostic'
    read = bound_metadata(root, contract['metadata_pins'], guard)
    schedule = read(root/'replay-history.json', 64*2**20)
    model = read(root/'model.json')
    config = model['config']
    require((config['num_attention_heads'], config['num_key_value_heads'], config['head_dim'], config['sliding_window'])
            == (16, 4, 256, 1024) and all(config['layer_types'][i] == 'sliding_attention' for i in LAYERS),
            'Original attention geometry and window')
    plan = read_pinned(producer['first5_plan'], producer['source_sha256'][producer['first5_plan']])
    positions = plan['original_prefix_plan']['reference_positions']
    require(len(positions) == 33 and len(schedule['actions']) == 6200, 'Every original reference row and event')
    files = OrdinaryFiles(root, guard)
    cache = FirstFiveCacheReplay(root, guard)
    references, rows, full_records = {}, [], []
    raw_output_bytes, calls, exact_row_checks, exact_rows, full_checks, exact_full = 0, 0, 0, 0, 0, 0

    def raw(t):
        return t.detach().contiguous().view(torch.uint8).numpy().tobytes()

    def tensor(data, shape):
        t = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).clone().reshape(shape)
        require(bool(torch.isfinite(t).all()), 'Finite original BF16 inputs')
        return t

    def load_tensor(record, label):
        desc = record['tensors'][label]
        require(desc['dtype'] == 'BF16' and desc['finite'] is True, 'Original selected BF16 tensor')
        with files.open(desc['file'], desc['bytes']) as stream:
            data = stream.read(desc['bytes']+1)
        require(len(data) == desc['bytes'] and digest(data) == desc['sha256'], 'Authenticated original selected bytes')
        return data, desc

    def module():
        return ReferenceAttention(16, 256, .0625, 4, prefix='saved_attention_diagnostic', per_layer_sliding_window=1024)

    def sdpa(q, k, v, position):
        nonlocal calls
        guard()
        query = q.view(-1, 16, 256).transpose(0, 1)
        key_pos = torch.arange(k.shape[1])
        query_pos = torch.tensor([[position]])
        mask = (key_pos <= query_pos) & (key_pos > query_pos-1024)
        result = torch.nn.functional.scaled_dot_product_attention(query, k, v, attn_mask=mask,
                                                                 scale=.0625, enable_gqa=True)
        calls += 1
        return result.transpose(0, 1).reshape(1, 4096)

    def descriptor(t):
        return dict(shape=list(t.shape), stride=list(t.stride()), dtype='BF16', sha256=digest(raw(t)))

    def archive(name, outputs):
        nonlocal raw_output_bytes
        cursor, entries, whole = 0, {}, hashlib.sha256()
        with (out/name).open('xb') as stream:
            for key, value in outputs.items():
                guard()
                require(bool(torch.isfinite(value).all()), 'Finite actual attention output')
                blob = raw(value)
                entries[key] = dict(offset=cursor, bytes=len(blob), shape=list(value.shape),
                                    dtype='BF16', sha256=digest(blob))
                stream.write(blob)
                whole.update(blob)
                cursor += len(blob)
        raw_output_bytes += cursor
        require(raw_output_bytes <= 32*2**20, 'Original bounded attention output budget')
        return dict(file=name, bytes=cursor, sha256=whole.hexdigest(), tensors=entries)

    for action in schedule['actions']:
        guard()
        if not action['native_prefix_call']:
            cache.accept(action)
            continue
        record = read(root/f"event-{action['event']:06d}-replay.json")
        cache.accept(action, record)
        if not action['selected_local_rows']:
            continue
        fixture, event, end = action['fixture'], action['event'], action['end']
        bank = cache.banks[action['slot']]['heads']
        if action['phase'] == 'full_causal':
            require(fixture not in references and end == 2060 and action['selected_local_rows'] == positions,
                    'Two complete original full causal references')
            references[fixture] = {}
            for layer in LAYERS:
                qraw, qdesc = load_tensor(record, f'{layer}-query')
                expected, adesc = load_tensor(record, f'{layer}-attention')
                kv = {kind:tensor(b''.join(bank[f'cache-{layer}-{kind}']), [4, 2060, 256])
                      for kind in ('key', 'value')}
                qselected = tensor(qraw, [33, 4096])
                sparse = torch.zeros([2060, 4096], dtype=torch.bfloat16)
                sparse.index_copy_(0, torch.tensor(positions), qselected)
                m = module()
                m.start_pos = 0
                kflat, vflat = [kv[k].transpose(0, 1).contiguous().view(2060, 1024) for k in ('key', 'value')]
                guard()
                got = m(sparse, kflat, vflat).index_select(0, torch.tensor(positions))
                calls += 1
                comparisons = [differences(row(expected, 4096, i), row(raw(got), 4096, i)) for i in range(33)]
                full_checks += 33
                exact_full += sum(v['bitwise_equal_words'] == v['elements'] for v in comparisons)
                item = dict(fixture=fixture, event=event, layer=layer, positions=positions,
                    query_construction='Original 33 selected Q rows at original positions; all unobserved query rows explicitly zero',
                    whole_original_query_reproduced=False, original_selected_query=qdesc, original_selected_attention=adesc,
                    inputs=dict(query=descriptor(sparse), key=descriptor(m.keys), value=descriptor(m.values)),
                    output=archive(f'fixture-{fixture}-layer-{layer}-full-query-control.bin', dict(selected=got)),
                    comparisons_to_original_selected_attention=comparisons)
                save(out/f'fixture-{fixture}-layer-{layer}-full-query-control.json', item)
                full_records.append(item)
                references[fixture][layer] = dict(query=qselected, query_raw=qraw, expected=expected,
                    key=kv['key'], value=kv['value'], original_strided_key=m.keys, original_strided_value=m.values)
                del sparse, got, m
            print(json.dumps(dict(stage='full_query_shape_control', fixture=fixture, exact_selected_rows=exact_full,
                                  checked_selected_rows=full_checks)), flush=True)
            continue
        require(action['selected_local_rows'] == [end-action['start']-1], 'Exact original selected final row')
        position = end-1
        refindex = positions.index(position)
        for layer in LAYERS:
            guard()
            reference = references[fixture][layer]
            qraw, qdesc = load_tensor(record, f'{layer}-query')
            expected_i, adesc = load_tensor(record, f'{layer}-attention')
            qi = tensor(qraw, [1, 4096])
            qf = reference['query'][refindex:refindex+1].clone()
            expected_f = row(reference['expected'], 4096, refindex)
            ki, vi = [tensor(b''.join(bank[f'cache-{layer}-{kind}']), [4, end, 256]) for kind in ('key', 'value')]
            kf, vf = [reference[k][:, :end, :].contiguous() for k in ('key', 'value')]
            variables = [dict(I=qi, F=qf), dict(I=ki, F=kf), dict(I=vi, F=vf)]
            outputs = {arm:sdpa(*(choices[letter] for choices, letter in zip(variables, arm)), position)
                       for arm in ARMS}
            outputs['reference_full_extent_strided'] = sdpa(qf, reference['original_strided_key'], reference['original_strided_value'], position)
            outputs['reference_full_extent_contiguous'] = sdpa(qf, reference['key'], reference['value'], position)
            m = module()
            m.start_pos = position
            if position:
                m.keys, m.values = ki[:, :position, :].contiguous(), vi[:, :position, :].contiguous()
            krow, vrow = [t[:, position:position+1, :].transpose(0, 1).contiguous().view(1, 1024) for t in (ki, vi)]
            outputs['incremental_row_replay'] = m(qi, krow, vrow)
            calls += 1
            original_row_shape = end-action['start'] == 1
            if original_row_shape:
                exact_row_checks += 1
                exact_rows += raw(outputs['incremental_row_replay']) == expected_i
            begin = max(0, position-1023)
            visible = {}
            for label, left, right in [('key', ki, kf), ('value', vi, vf)]:
                a, b = left[:, begin:end, :].contiguous().view(torch.int16), right[:, begin:end, :].contiguous().view(torch.int16)
                changed = a != b
                by_position = changed.any(dim=0).any(dim=-1).nonzero().flatten()
                visible[label] = dict(start=begin, end=end, different_bf16_words=int(changed.sum()),
                    first_different_position=begin+int(by_position[0]) if len(by_position) else None,
                    incremental_sha256=digest(raw(left[:, begin:end, :])), reference_sha256=digest(raw(right[:, begin:end, :])))
            output = archive(f'event-{event:06d}-layer-{layer}-attention-factors.bin', outputs)
            values = {name:raw(value) for name, value in outputs.items()}
            item = dict(event=event, fixture=fixture, layer=layer, slot=action['slot'], epoch=action['epoch'],
                request=action['request'], position=position, original_query_rows=end-action['start'],
                original_query_shape_replayed=original_row_shape, original_query=qdesc, original_attention=adesc,
                reference_row_index=refindex, original_pointwise_failed=action['original_pointwise_failed'],
                original_FP32_metrics=action['original_FP32_metrics'],
                inputs={name:descriptor(t) for name, t in [('QI',qi),('QF',qf),('KI',ki),('KF',kf),('VI',vi),('VF',vf)]},
                visible_history_differences=visible, output=output,
                comparisons_to_original_incremental={name:differences(expected_i, blob) for name, blob in values.items()},
                comparisons_to_original_full_reference={name:differences(expected_f, blob) for name, blob in values.items()},
                comparisons_to_III={name:differences(values['III'], blob) for name, blob in values.items()},
                original_full_model_acceptance_changed=False, full_model_qualified=False)
            save(out/f'event-{event:06d}-layer-{layer}-attention-factors.json', item)
            rows.append(item)
        if len(rows)%30 == 0:
            print(json.dumps(dict(stage='attention_factors', marked_calls=len(rows)//3, exact_row_checks=exact_row_checks,
                                  exact_rows=exact_rows, sdpa_calls=calls)), flush=True)
    rebuilt = cache.finish()
    require(len(rows) == 261 and full_checks == 198 and exact_row_checks == 249 and calls == 2877,
            'All87 marks/all3 layers/8 factor arms/2 extent controls/row replay and6 full-query controls')
    require(sum(r['original_pointwise_failed'] for r in rows) == 21, 'All seven original failures in all three layers')
    result = dict(schema='dory_attention_factors_result_v1', complete=True, diagnostic_only=True,
        frames=rows, full_query_controls=full_records, cache_reconstruction=rebuilt, sdpa_calls=calls,
        original_marks=87, attention_layers=list(LAYERS), original_pointwise_failures=7,
        original_row_shape_checks=exact_row_checks, original_row_shape_exact=exact_rows,
        full_query_selected_row_checks=full_checks, full_query_selected_row_exact=exact_full,
        operator_controls_passed=exact_rows == 249 and exact_full == 198,
        raw_output_bytes=raw_output_bytes, original_FP32_acceptance_passed=False,
        model_weights_loaded=False, whole_model_executed=False, full_original_query_tensor_available=False,
        native_libraries_JIT_identity_proven=False, new_speedup_claim=False, full_model_qualified=False,
        final_seven_failure_root_cause_established=False, torch_version=str(torch.__version__))
    save(out/'result.json', result)
    return dict(complete=True, operator_controls_passed=result['operator_controls_passed'],
                exact_row_checks=exact_row_checks, exact_rows=exact_rows, full_checks=full_checks,
                exact_full=exact_full, raw_output_bytes=raw_output_bytes, sdpa_calls=calls)


def main():
    ap = argparse.ArgumentParser()
    for name in ('spec', 'spec-sha256', 'review', 'review-sha256'):
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    spec = read_pinned(args.spec, args.spec_sha256)
    review = read_pinned(args.review, args.review_sha256)
    require(spec['schema'] == 'dory_attention_factors_spec_v1' and spec['budget'] == BUDGET
            and spec['previous_review_exit'] == history.REVIEW_EXIT and spec['arms'] == ARMS
            and spec['layers'] == list(LAYERS) and spec['full_model_qualified'] is spec['new_speedup_claim'] is False,
            'Exact complete diagnostic scope')
    require(review['reviewer'] == '/root' and review['passed'] is True and review['blockers'] == []
            and review['spec_sha256'] == args.spec_sha256 and review['source_sha256'] == spec['source_sha256'],
            'Exact reviewed new source and original provenance')
    p = history.prior.p
    p.low.user_bus()
    fd, owner, lock = acquire()
    out = Path(spec['output'])
    require(not out.exists() and out.parent == Path(spec['base']), 'Fresh diagnostic output')
    started, last, initial = time.monotonic(), [0.], free('/mnt/nvme')
    canonical = p.canonical()
    created = False
    handlers = {}
    def interrupted(signum, frame):
        raise InterruptedError('Attention factor diagnosis interrupted')
    def guard():
        now = time.monotonic()
        require(now-started < BUDGET['wall_seconds'], 'Bounded diagnostic wall time')
        if now-last[0] >= .5:
            p.low.lock_proof(owner, lock)
            p.low.clear_window()
            p.low.active_heavy()
            require(free('/') >= BUDGET['root_floor']
                    and free('/mnt/nvme') >= max(BUDGET['nvme_floor'], initial-BUDGET['nvme_new_bytes'])
                    and memory_free() >= BUDGET['memory_available_floor_bytes'], 'Original resource floors')
            last[0] = now
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupted)
        guard()
        require(memory_free() >= BUDGET['initial_memory_available_bytes']
                and initial >= BUDGET['nvme_floor']+BUDGET['nvme_new_bytes'], 'Fresh complete reservation')
        sources = review_source_pins(spec['source_sha256'], spec['essential_sources'],
                                    max_total_bytes=spec['source_read_cap_bytes'], guard=guard)
        predecessors, receipt, previous = history.completed()
        require(spec['base'] == previous['base'] and spec['environment'] == previous['environment']
                and spec['official_source'] == previous['official_source'], 'Same original source and environment')
        physical = history.drain(predecessors)
        guard()
        out.mkdir()
        created = True
        os.sched_setaffinity(0, set(BUDGET['cpus']))
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS, (BUDGET['address_space_bytes'],)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (BUDGET['cpu_seconds'], BUDGET['cpu_seconds']+5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (128*2**20,)*2)
        save(out/'admission.json', dict(owner=owner, lock=lock, spec_sha256=args.spec_sha256,
            source_review=dict(path=args.review, sha256=args.review_sha256), budget=BUDGET,
            canonical_sha256=canonical, predecessor_history=save(out/'predecessor-history.json', predecessors),
            predecessor_drain=save(out/'predecessor-drain.json', physical), performance_measurement=False))
        save(out/'source-files.json', sources)
        print(json.dumps(dict(owner=owner, admission=str(out/'admission.json'))), flush=True)
        result = run(spec, out, guard, receipt)
        guard()
        require(p.canonical() == canonical, 'Original canonical files unchanged')
        result.update(owner=owner, elapsed_seconds=time.monotonic()-started, actual_self_exit_claimed=False,
                      full_model_qualified=False, new_speedup_claim=False)
        save(out/'terminal.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if created:
            save(out/'failure.json', dict(error_type=type(error).__name__, error=str(error), owner=owner))
        raise
    finally:
        release(fd)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()
