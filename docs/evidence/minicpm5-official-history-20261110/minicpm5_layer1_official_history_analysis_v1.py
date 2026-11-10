"""Descriptive output distances to official current-frame attention targets."""
import hashlib
import math
from pathlib import Path
from research import minicpm5_layer1_after_native_cache_v1 as official
from research import minicpm5_layer1_official_history_inputs_v1 as inputs
from research import minicpm5_layer1_official_history_v1 as contract
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research.compare_minicpm5_layer1_native_cache_v1 import distance
require = boundary.require


def analyze(proof, plan, read, guard):
    require(proof['complete'] is True and proof['decode_calls'] == 180
            and proof['all_original_and_restored_outputs_exact'] is True, 'Verified actual official-history pair')
    bundle = official.completed(plan['layer1_official_history'], read)
    native = inputs.load(plan, bundle, guard)
    outputs = {}
    for row in proof['rank_proofs'][0]['outputs']:
        guard(); raw = Path(row['path']).read_bytes()
        require(len(raw) == 16384 and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Exact verified complete output')
        outputs[row['arm'], row['call'], row['mode']] = (row, raw)
    records = []
    for arm in boundary.ARMS:
        for ci in boundary.DECODE_CALLS:
            row, baseline = outputs[arm, ci, 'own']
            target = b''.join(native[f['fixture'], f['step']]['targets'][boundary.ATTENTION]
                              for f in row['physical_frame_order'])
            require(hashlib.sha256(target).hexdigest() == row['official_target_sha256'], 'Same official current targets')
            for mode in contract.MODES:
                _, raw = outputs[arm, ci, mode]
                for slot, frame in enumerate(row['physical_frame_order']):
                    start = slot*4096; stop = start+4096
                    metric = distance(raw[start:stop], target[start:stop])
                    original = distance(baseline[start:stop], target[start:stop])
                    change = distance(raw[start:stop], baseline[start:stop])
                    records.append(dict(arm=arm, call=ci, mode=mode, slot=slot, frame=frame,
                        official_distance=metric, own_official_distance=original, change_from_own=change,
                        improved=metric['l2_squared'] < original['l2_squared'],
                        worsened=metric['l2_squared'] > original['l2_squared']))
    require(len(records) == 360, 'All90 replicated outputs x4 rows, counted once')
    summaries = []
    for arm in boundary.ARMS:
        for mode in contract.MODES:
            rows = [r for r in records if r['arm'] == arm and r['mode'] == mode]
            require(len(rows) == 24, 'Every original decode frame once')
            norms = [r['official_distance'] for r in rows]
            num = math.fsum(x['l2_squared'] for x in norms); den = math.fsum(x['reference_l2_squared'] for x in norms)
            summaries.append(dict(arm=arm, mode=mode, frames=24,
                aggregate_relative_l2=math.sqrt(num/den),
                mean_relative_l2=math.fsum(x['relative_l2'] for x in norms)/24,
                max_abs=max(x['max_abs'] for x in norms),
                exact_official_frames=sum(x['raw_exact'] for x in norms),
                changed_from_own_frames=sum(r['change_from_own']['changed_elements'] > 0 for r in rows),
                improved_frames=sum(r['improved'] for r in rows), worsened_frames=sum(r['worsened'] for r in rows)))
    guard()
    return dict(schema='minicpm5_layer1_official_history_descriptive_analysis_v1', complete=True,
        records=records, summary=summaries, descriptive_precision='stdlib FP64 distances of full BF16 rows',
        scope='Fixed original current input, only official valid prior histories substituted; official target is current native frame',
        cache_intervention_applied=True, root_cause_proven=False, full_model_qualified=False, new_speedup_claim=False)
