"""Independent post-exit input reconstruction and raw-output statistic replay."""
import argparse
import array
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

Q = Path(__file__).resolve().parent
sys.path[:0] = [str(Q), str(Q.parent)]
from dory_layer0_serial_contract_v2 import require, read_pinned
from dory_generation_execution_provenance_v1 import review_source_pins
from dory_first5_cache_replay_v1 import FirstFiveCacheReplay
from dory_generation_cache_replay_v1 import OrdinaryFiles
from dory_generation_journal_replay_v1 import SavedMetadata
from review_dory_layer0_projection_saved_v2 import bound_metadata, row
from review_dory_long_matrix_saved_v3 import finite
from explain_dory_long_cache_differences_v1 import differences
from nemotron9b_package_controller_v16 import acquire, release
from run_dory_layer0_serial_v2 import save, free, memory_free
import review_dory_layer0_serial_snapshot_v1 as prior


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def load(ref):
    return read_pinned(ref['path'], ref['sha256'])


def selected(files, descriptor):
    with files.open(descriptor['file'], descriptor['bytes']) as stream:
        blob = stream.read(descriptor['bytes']+1)
    require(len(blob) == descriptor['bytes'] and sha(blob) == descriptor['sha256'] and finite(blob, 'BF16'),
            'Original complete selected tensor bytes')
    return blob


def visible(raw, length, start, end):
    return b''.join(raw[(head*length+start)*512:(head*length+end)*512] for head in range(4))


def visible_stats(a, b, start, end):
    left, right = array.array('H'), array.array('H')
    left.frombytes(a)
    right.frombytes(b)
    require(sys.byteorder == 'little' and len(a) == len(b) == 4*(end-start)*512,
            'Original full visible per-head histories')
    count, first = 0, None
    for i, (x, y) in enumerate(zip(left, right)):
        if x != y:
            count += 1
            position = start+(i%((end-start)*256))//256
            first = position if first is None else min(first, position)
    return dict(start=start, end=end, different_bf16_words=count, first_different_position=first,
                incremental_sha256=sha(a), reference_sha256=sha(b))


def verify(out, native, receipt, guard):
    root = Path(receipt['result']['path']).parent
    pins = {}
    for path in root.glob('*.json'):
        blob = path.read_bytes()
        pins[path.name] = dict(bytes=len(blob), sha256=sha(blob))
    metadata = SavedMetadata(root, pins, guard)
    read = lambda name:metadata.json(name, 32*2**20)
    files = OrdinaryFiles(root, guard)
    save(out/'native-metadata-pins.json', pins)
    require(read('result.json') == native, 'Exact externally pinned producer result')
    spec = load(receipt['spec'])
    prior_exit = load(spec['previous_review_exit'])
    contract = load(prior_exit['post_exit_contract'])
    producer = load(contract['producer_spec'])
    original = Path(producer['output'])/'diagnostic'
    old_read = bound_metadata(original, contract['metadata_pins'], guard)
    old_files = OrdinaryFiles(original, guard)
    schedule = old_read(original/'replay-history.json', 64*2**20)
    plan = read_pinned(producer['first5_plan'], producer['source_sha256'][producer['first5_plan']])
    positions = plan['original_prefix_plan']['reference_positions']
    cache = FirstFiveCacheReplay(original, guard)
    frames = {(f['event'], f['layer']):f for f in native['frames']}
    full = {(f['fixture'], f['layer']):f for f in native['full_query_controls']}
    require(len(frames) == 261 and len(full) == 6 and len(native['frames']) == 261, 'All original cases, no duplicates')
    references, used, raw_bytes = {}, set(), 0
    row_checks = row_exact = full_checks = full_exact = 0

    def outputs(desc):
        nonlocal raw_bytes
        require(desc['file'] not in used, 'Each output archive consumed once')
        used.add(desc['file'])
        values, cursor, whole = {}, 0, hashlib.sha256()
        with files.open(desc['file'], desc['bytes']) as stream:
            for name, tensor in desc['tensors'].items():
                guard()
                require(tensor['offset'] == cursor and tensor['dtype'] == 'BF16'
                        and tensor['shape'][-1] == 4096 and tensor['bytes'] == tensor['shape'][0]*8192,
                        'Exact contiguous BF16 output extent')
                blob = stream.read(tensor['bytes'])
                require(len(blob) == tensor['bytes'] and sha(blob) == tensor['sha256'] and finite(blob, 'BF16'),
                        'Complete finite raw attention output')
                values[name] = blob
                cursor += len(blob)
                whole.update(blob)
            require(stream.read(1) == b'' and cursor == desc['bytes'] and whole.hexdigest() == desc['sha256'],
                    'Complete original output file hash')
        raw_bytes += cursor
        return values

    for action in schedule['actions']:
        guard()
        if not action['native_prefix_call']:
            cache.accept(action)
            continue
        original_record = old_read(original/f"event-{action['event']:06d}-replay.json")
        cache.accept(action, original_record)
        if not action['selected_local_rows']:
            continue
        event, fixture, end = action['event'], action['fixture'], action['end']
        bank = cache.banks[action['slot']]['heads']
        if action['phase'] == 'full_causal':
            for layer in (0, 2, 4):
                f = full[fixture, layer]
                require(read(f'fixture-{fixture}-layer-{layer}-full-query-control.json') == f, 'Original full-query case metadata')
                qdesc, adesc = [original_record['tensors'][f'{layer}-{name}'] for name in ('query', 'attention')]
                qraw, expected = [selected(old_files, desc) for desc in (qdesc, adesc)]
                kv = {kind:b''.join(bank[f'cache-{layer}-{kind}']) for kind in ('key', 'value')}
                qfull = bytearray(2060*8192)
                for i, position in enumerate(positions):
                    qfull[position*8192:(position+1)*8192] = row(qraw, 4096, i)
                require(f['original_selected_query'] == qdesc and f['original_selected_attention'] == adesc
                        and f['inputs']['query']['sha256'] == sha(qfull) and f['positions'] == positions
                        and f['whole_original_query_reproduced'] is False, 'Explicit zero-other-row query intervention')
                for name in ('key', 'value'):
                    require(f['inputs'][name]['sha256'] == sha(kv[name])
                            and f['inputs'][name]['shape'] == [4,2060,256]
                            and f['inputs'][name]['stride'] == [256,1024,1], 'Exact full reference KV values and original layout')
                blobs = outputs(f['output'])
                require(list(blobs) == ['selected'], 'Exactly retained selected full-query output')
                comparisons = [differences(row(expected,4096,i),row(blobs['selected'],4096,i)) for i in range(33)]
                require(comparisons == f['comparisons_to_original_selected_attention'], 'Independent198 reference comparisons')
                full_checks += 33
                full_exact += sum(x['bitwise_equal_words'] == x['elements'] for x in comparisons)
                references[fixture, layer] = dict(query=qraw, expected=expected, **kv)
            continue
        position, index = end-1, positions.index(end-1)
        for layer in (0, 2, 4):
            f = frames[event, layer]
            require(read(f'event-{event:06d}-layer-{layer}-attention-factors.json') == f, 'Original factor metadata')
            for key in ('event','fixture','slot','epoch','request','original_pointwise_failed','original_FP32_metrics'):
                require(f[key] == action[key], 'Same original request/action: '+key)
            require(f['position'] == position and f['original_query_rows'] == end-action['start']
                    and f['original_query_shape_replayed'] == (end-action['start'] == 1), 'Wider original calls remain labeled')
            ref = references[fixture, layer]
            qdesc, adesc = [original_record['tensors'][f'{layer}-{name}'] for name in ('query','attention')]
            qi, expected_i = [selected(old_files, desc) for desc in (qdesc, adesc)]
            require(f['original_query'] == qdesc and f['original_attention'] == adesc, 'Exact original selected descriptors')
            sources = dict(QI=qi, QF=row(ref['query'],4096,index),
                KI=b''.join(bank[f'cache-{layer}-key']), VI=b''.join(bank[f'cache-{layer}-value']),
                KF=visible(ref['key'],2060,0,end), VF=visible(ref['value'],2060,0,end))
            for name, blob in sources.items():
                require(f['inputs'][name]['sha256'] == sha(blob) and f['inputs'][name]['dtype'] == 'BF16',
                        'Every actual Q/K/V provider bound to original saved input')
                shape, stride = ([1,4096],[4096,1]) if name.startswith('Q') else ([4,end,256],[end*256,256,1])
                require(f['inputs'][name]['shape'] == shape and f['inputs'][name]['stride'] == stride,
                        'Fixed common input layout in eight factor arms')
            begin = max(0, position-1023)
            for label, key in [('key','K'),('value','V')]:
                found = visible_stats(visible(sources[key+'I'],end,begin,end),visible(sources[key+'F'],end,begin,end),begin,end)
                require(found == f['visible_history_differences'][label], 'Full visible history word differences independently counted')
            blobs = outputs(f['output'])
            require(list(blobs) == ['III','IIF','IFI','IFF','FII','FIF','FFI','FFF',
                'reference_full_extent_strided','reference_full_extent_contiguous','incremental_row_replay'], 'All original factor arms')
            expected_f = row(ref['expected'],4096,index)
            for field, baseline in [('comparisons_to_original_incremental',expected_i),
                                    ('comparisons_to_original_full_reference',expected_f),('comparisons_to_III',blobs['III'])]:
                require({name:differences(baseline, blob) for name,blob in blobs.items()} == f[field],
                        'Every raw-output statistic independently recomputed')
            if f['original_query_shape_replayed']:
                row_checks += 1
                row_exact += blobs['incremental_row_replay'] == expected_i
        if len(used)%30 == 0:
            print(json.dumps(dict(stage='saved_factor_review', raw_files=len(used))), flush=True)
    reconstructed = cache.finish()
    require(used == {p.name for p in root.glob('*.bin')} and len(used) == 267 and raw_bytes == 25141248,
            'Every original raw output archive and byte accounted for')
    require((row_checks,row_exact,full_checks,full_exact) == (249,249,198,198)
            and native['operator_controls_passed'] is True and native['sdpa_calls'] == 2877,
            'Complete original reproduction controls and scientific scope')
    require(not native['original_FP32_acceptance_passed'] and native['original_pointwise_failures'] == 7,
            'Original failures remain failures')
    return dict(complete=True, raw_output_bytes_verified=raw_bytes, raw_files_verified=267,
        original_cache_reconstruction=reconstructed, marked_frames=87, layer_frames=261,
        original_row_shape_checks=row_checks, original_row_shape_exact=row_exact,
        full_query_selected_rows=full_checks, full_query_selected_rows_exact=full_exact,
        all_saved_input_provider_hashes_reconstructed=True, all_recorded_output_statistics_exact=True,
        original_seven_failures_preserved=True, new_speedup_claim=False, full_model_qualified=False)


def main():
    ap = argparse.ArgumentParser()
    for name in ('spec','spec-sha256'):
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    spec = read_pinned(args.spec, args.spec_sha256)
    require(spec['schema'] == 'dory_attention_factors_saved_review_v1' and spec['reviewer'] == '/root'
            and spec['passed_source_review'] is True, 'Exact root-reviewed saved-data audit')
    receipt = load(spec['native_exit'])
    require(receipt['actual_external_exit_observed'] and (receipt['session_id'],receipt['exit_code'],receipt['tool_chunk'])
            == (99315,0,'c0c997'), 'Actual native tool exit')
    native, admission, terminal = [load(receipt[k]) for k in ('result','admission','terminal')]
    require(admission['owner'] == terminal['owner'] == receipt['owner'] and terminal['complete'], 'Exact completed native owner')
    out = Path(spec['output'])
    require(not out.exists(), 'Fresh saved review namespace')
    p = prior.p
    p.low.user_bus()
    fd, owner, lock = acquire()
    start, last, initial = time.monotonic(), [0.], free('/mnt/nvme')
    canonical, created = p.canonical(), False
    def guard():
        now = time.monotonic()
        require(now-start < 1800, 'Bounded saved review')
        if now-last[0] >= .5:
            p.low.lock_proof(owner, lock)
            p.low.clear_window()
            p.low.active_heavy()
            require(free('/') >= 28*2**30 and free('/mnt/nvme') >= max(8*2**30, initial-64*2**20)
                    and memory_free() >= 3*2**30, 'Original review resource floors')
            last[0] = now
    try:
        guard()
        require(memory_free() >= 4*2**30 and initial >= 8*2**30+64*2**20, 'Fresh saved-review reservation')
        sources = review_source_pins(spec['source_sha256'],[str(Path(__file__).resolve())],
                                    max_total_bytes=spec['source_read_cap_bytes'],guard=guard)
        history = load(admission['predecessor_history'])
        require(len(history['processes']) == 288 and len(history['rows']) == 1096, 'Complete native predecessor snapshot')
        history['processes'].append(receipt['owner'])
        drain = prior.physical_drain(history,[])
        out.mkdir()
        created = True
        os.sched_setaffinity(0,{4})
        os.nice(10)
        resource.setrlimit(resource.RLIMIT_AS,(4*2**30,)*2)
        resource.setrlimit(resource.RLIMIT_CPU,(1700,1705))
        save(out/'admission.json',dict(owner=owner,lock=lock,spec_sha256=args.spec_sha256,
            predecessor_history=save(out/'predecessor-history.json',history),
            predecessor_drain=save(out/'predecessor-drain.json',drain),performance_measurement=False))
        save(out/'source-files.json',sources)
        print(json.dumps(dict(owner=owner,admission=str(out/'admission.json'))),flush=True)
        result = verify(out,native,receipt,guard)
        require(p.canonical() == canonical, 'Original canonical unchanged')
        result.update(owner=owner,elapsed_seconds=time.monotonic()-start,actual_self_exit_claimed=False)
        save(out/'result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        if created:
            save(out/'failure.json',dict(error=repr(error),owner=owner))
        raise
    finally:
        release(fd)


if __name__ == '__main__':
    main()
