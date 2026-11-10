"""Compare authenticated SDK input histories to the previous official frame.

Read-only, serial, stdlib-only diagnostic. No model or attention is executed.
Full SDK stores are reconstructed; distances use only the active valid prefix.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import struct
import sys

R = Path(__file__).resolve().parents[1]
Q = R / 'research'
sys.path.insert(0, str(R))
import minicpm5_layer1_native_cache_runner_v1 as runner
from research import minicpm5_layer1_decode_stream_v2 as stream
from research import minicpm5_layer1_after_saved_audit_v3 as collected
from research import minicpm5_layer1_payload_io_v1 as payload

p = runner.p
require = p.require


def aligned_prefix(case, frame, native_row, cache, native_bytes, kind):
    """Match same fixture's previous step, original physical slot and KV head."""
    require(kind in ('key', 'value') and frame in case['physical_frame_order'], 'Actual decode frame')
    rank = case['rank']; slot = frame['slot']; length = case['positions'][slot]
    require(type(rank) is int and rank in (0, 1) and frame['step'] in (1, 2, 3), 'Original TP2 decode')
    previous = native_row['frame']
    require(previous['fixture'] == frame['fixture'] and previous['slot'] == slot
            and previous['step'] == frame['step'] - 1, 'Same fixture, slot and preceding native frame')
    require(native_row['cache_length'] == length and
            native_row['cache_metadata'][kind]['shape'] == [1, 2, length, 128], 'Complete native prior history')
    require(type(cache) is bytes and len(cache) == 4 * 4224 * 128 * 2
            and type(native_bytes) is bytes and len(native_bytes) == 2 * length * 128 * 2,
            'Both full physical SDK cache and both native heads retained')
    width = length * 128 * 2
    sdk_start = slot * 4224 * 128 * 2
    return cache[sdk_start:sdk_start + width], native_bytes[rank * width:(rank + 1) * width]


def distance(actual, reference):
    require(len(actual) == len(reference) and len(actual) % 2 == 0, 'Equal full BF16 extents')
    def decode(data):
        words = [x[0] for x in struct.iter_unpack('<H', data)]
        require(all(w & 0x7f80 != 0x7f80 for w in words), 'Finite BF16 cache values')
        return [struct.unpack('<f', struct.pack('<I', w << 16))[0] for w in words]
    a = decode(actual); b = decode(reference)
    squared = math.fsum((x-y)*(x-y) for x, y in zip(a, b))
    base = math.fsum(y*y for y in b)
    return dict(elements=len(a), changed_elements=sum(x != y for x, y in zip(a, b)),
                raw_exact=actual == reference, l2_squared=squared, reference_l2_squared=base,
                relative_l2=math.sqrt(squared/base) if base else None,
                max_abs=max((abs(x-y) for x, y in zip(a, b)), default=0.0))


def run():
    p.user_bus(); fd, owner, lock = runner.post.external.acquire_lock()
    try:
        os.sched_setaffinity(0, {6}); os.nice(19)
        resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (240, 245))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 2**20,) * 2)
        signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError('Cache comparison deadline')))
        signal.alarm(300)
        endref = runner.reference(Q/'minicpm5-layer1-native-cache-post-exit-audit-actual-exit-20261110-v1.json')
        end = p.read_ref(endref)
        require(end['actual_tool_exit_observed'] and end['actual_exit_code'] == 0
                and end['exact_owner_gone'] and end['serial_lock_reacquired']
                and end['lock_identity'] == lock, 'Actual successful independent native audit exit')
        audit = p.read_ref(end['result']); model_end = p.read_ref(audit['model_exit'])
        spec = p.read_ref(model_end['spec']); plan = p.read_ref(spec['plan']); review = p.read_ref(model_end['review'])
        require(review['passed'] and review['spec'] == model_end['spec']
                and review['source_sha256'] == spec['source_sha256'] == plan['source_sha256'], 'Frozen native source review')
        for path, digest in spec['source_sha256'].items(): require(p.sha(path) == digest, 'Frozen source ' + path)
        require(model_end['actual_exit_code'] == 0 and model_end['all_three_processes_gone']
                and model_end['cgroup_empty'] and audit['complete'], 'Completed native collection and reread')
        bundle = collected.completed(plan['layer1_own_history_replay'], p.read_ref)
        def guard():
            p.lock_proof(owner, lock); p.clear_window()
            for old in (end['owner'], model_end['owner'], model_end['coordinator'], model_end['native']): p.gone(old)
            p.empty_group(p.GROUP_PREFIX + spec['unit'] + '.service')
        guard(); p.active_heavy(); collected.physical(bundle, guard)
        native = p.read_ref(model_end['outputs']['native_result'])
        require(runner.validate_native_result(native, plan, guard) == audit['result'], 'Complete independent native bytes')
        native_rows = {(r['frame']['fixture'], r['frame']['step']): r for r in native['records']}
        rows = []; proofs = []
        def read_json(path, sha): return p.read_ref(dict(path=path, sha256=sha))
        for rank in (0, 1):
            rank_result = p.read_ref(bundle['collector_exit']['outputs']['rank%d_result' % rank])
            inputs = stream.RankStream(bundle['collector_plan'], rank_result, bundle['audit_result'], read_json, guard)
            for item in inputs:
                case = item['case']
                for frame in case['physical_frame_order']:
                    prior = native_rows[frame['fixture'], frame['step']-1]
                    fixture = plan['fixtures'][frame['fixture']]; prompt_length = len(fixture['input_ids'])
                    for kind, cache_name in zip(('key', 'value'), stream.raw.CACHES):
                        desc = prior['outputs'][kind]
                        raw_native = payload.read_owned_payload(dict(desc, payload_bytes=desc['raw_bytes'],
                            payload_sha256=desc['sha256']), Path(desc['path']), guard)
                        sdk, hf = aligned_prefix(case, frame, prior, item['cache_raw'][cache_name], raw_native, kind)
                        row = dict(rank=rank, arm=case['arm'], call=case['call']['index'], frame=frame,
                            kind=kind, previous_native_frame=prior['frame'], valid_positions=len(sdk)//256,
                            original_call=case['original_call'], native_payload=desc,
                            sdk_full_cache_sha256=hashlib.sha256(item['cache_raw'][cache_name]).hexdigest(),
                            sdk_valid_prefix_sha256=hashlib.sha256(sdk).hexdigest(),
                            native_head_prefix_sha256=hashlib.sha256(hf).hexdigest(),
                            full=distance(sdk, hf), prompt=distance(sdk[:prompt_length*256], hf[:prompt_length*256]))
                        row['appended'] = distance(sdk[prompt_length*256:], hf[prompt_length*256:])
                        rows.append(row)
                guard()
            proofs.append(inputs.finish())
        require(len(rows) == 288 and len({(r['rank'], r['arm'], r['frame']['index'], r['kind']) for r in rows}) == 288,
                'All 2 heads x 3 arms x 24 decode frames x 2 cache kinds')
        summaries = []
        for arm in stream.raw.ARMS:
            for kind in ('key', 'value'):
                selected = [r for r in rows if r['arm'] == arm and r['kind'] == kind]
                stats = {}
                for scope in ('full', 'prompt', 'appended'):
                    ds = [r[scope] for r in selected if r[scope]['elements']]
                    total = math.fsum(d['l2_squared'] for d in ds)
                    norm = math.fsum(d['reference_l2_squared'] for d in ds)
                    stats[scope] = dict(head_frame_pairs=len(ds), exact_pairs=sum(d['raw_exact'] for d in ds),
                        elements=sum(d['elements'] for d in ds), changed_elements=sum(d['changed_elements'] for d in ds),
                        aggregate_relative_l2=math.sqrt(total/norm) if norm else None,
                        mean_relative_l2=math.fsum(d['relative_l2'] for d in ds)/len(ds),
                        max_abs=max(d['max_abs'] for d in ds))
                summaries.append(dict(arm=arm, kind=kind, stats=stats))
        require(not {'torch', 'numpy', 'torch_neuronx', 'neuronxcc'} & set(sys.modules), 'Standard library only')
        guard()
        target = Q/'minicpm5-layer1-native-cache-comparison-20261110-v1.json'
        p.once(target, dict(schema='minicpm5_layer1_native_cache_comparison_v1', complete=True,
            observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), owner=owner, lock_identity=lock,
            source=runner.reference(__file__), native_audit_exit=endref,
            sdk_binding=plan['layer1_own_history_replay'], source_streams=proofs, comparisons=288,
            records=rows, summary=summaries, descriptive_precision='stdlib FP64 from BF16',
            scope='Original 24 decode input histories, each matched to same fixture previous official frame; no attention replay',
            sdk_cache_location='CPU SDK reference adapter; not Neuron device capture',
            attention_causality_proven=False, full_model_qualified=False, new_speedup_claim=False))
        print(json.dumps(runner.reference(target)))
    finally:
        signal.alarm(0); runner.post.external.release_lock(fd)


if __name__ == '__main__': run()
