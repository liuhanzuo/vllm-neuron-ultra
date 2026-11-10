"""Descriptive output distances to official current-frame attention targets."""
import hashlib
import math
import struct
from pathlib import Path
from research import minicpm5_layer1_after_native_cache_v1 as official
from research import minicpm5_layer1_official_history_inputs_v1 as inputs
from research import minicpm5_layer1_current_history_factorial_v1 as contract
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research.compare_minicpm5_layer1_native_cache_v1 import distance
require = boundary.require


def interaction(cells,target):
    """Difference of input effects with official versus original past histories."""
    require(set(cells)==set(contract.MODES), 'All four factorial cells and restore required')
    require(cells['own']==cells['restore'], 'Restored output must equal own output')
    require(type(target)is bytes and len(target)>0 and len(target)%2==0
            and all(type(v)is bytes and len(v)==len(target)for v in cells.values()), 'Equal complete BF16 vectors')
    def values(raw):
        words=[x[0]for x in struct.iter_unpack('<H',raw)]
        require(all(w&0x7f80!=0x7f80 for w in words),'Finite factorial values')
        return [struct.unpack('<f',struct.pack('<I',w<<16))[0]for w in words]
    y00=values(cells['own']);y10=values(cells['official_current'])
    y01=values(cells['official_history']);y11=values(cells['official_both']);t=values(target)
    delta=[d-b-c+a for a,b,c,d in zip(y00,y10,y01,y11)]
    squared=math.fsum(x*x for x in delta);den=math.fsum(x*x for x in t)
    current=math.fsum((b-a)**2 for a,b in zip(y00,y10))
    history=math.fsum((c-a)**2 for a,c in zip(y00,y01))
    combined=math.fsum((d-a)**2 for a,d in zip(y00,y11))
    return dict(elements=len(t),interaction_l2_squared=squared,official_target_l2_squared=den,
        interaction_relative_l2=math.sqrt(squared/den)if den else None,
        interaction_max_abs=max(abs(x)for x in delta),current_effect_l2_squared=current,
        history_effect_l2_squared=history,combined_effect_l2_squared=combined)


def analyze(proof, plan, read, guard):
    require(proof['complete'] is True and proof['decode_calls'] == 180
            and proof['all_original_and_restored_outputs_exact'] is True
            and proof['previous_real_controls_full_bytes_exact'] is True
            and proof['previous_control_calls']==108
            and proof['official_current_intervention_performed'] is True, 'Verified actual factorial pair with full previous real controls')
    bundle = official.completed(plan['layer1_official_history'], read)
    native = inputs.load(plan, bundle, guard)
    outputs = {}
    for row in proof['rank_proofs'][0]['outputs']:
        guard(); raw = Path(row['path']).read_bytes()
        require(len(raw) == 16384 and hashlib.sha256(raw).hexdigest() == row['sha256'], 'Exact verified complete output')
        outputs[row['arm'], row['call'], row['mode']] = (row, raw)
    records = []; interactions = []
    for arm in boundary.ARMS:
        for ci in boundary.DECODE_CALLS:
            row, baseline = outputs[arm, ci, 'own']
            target = b''.join(native[f['fixture'], f['step']]['targets'][boundary.ATTENTION]
                              for f in row['physical_frame_order'])
            require(hashlib.sha256(target).hexdigest() == row['official_target_sha256'], 'Same official current targets')
            for slot, frame in enumerate(row['physical_frame_order']):
                start=slot*4096; stop=start+4096
                cells={mode:outputs[arm,ci,mode][1][start:stop] for mode in contract.MODES}
                interactions.append(dict(arm=arm,call=ci,slot=slot,frame=frame,
                    **interaction(cells,target[start:stop])))
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
    require(len(interactions)==72, 'All three arms and24 frames retained for factorial comparison')
    interaction_summary=[]
    for arm in boundary.ARMS:
        rows=[r for r in interactions if r['arm']==arm]
        numerator=math.fsum(r['interaction_l2_squared'] for r in rows)
        denominator=math.fsum(r['official_target_l2_squared'] for r in rows)
        interaction_summary.append(dict(arm=arm,frames=len(rows),
            aggregate_interaction_relative_l2=math.sqrt(numerator/denominator) if denominator else None,
            nonzero_interaction_frames=sum(r['interaction_l2_squared']>0 for r in rows)))
    guard()
    return dict(schema='minicpm5_layer1_current_history_factorial_descriptive_analysis_v1', complete=True,
        records=records, summary=summaries, interactions=interactions, interaction_summary=interaction_summary,
        interaction_definition="y11 - y10 - y01 + y00; vector difference, not an error contribution fraction",
        current_qkv_change_together=True, descriptive_precision='stdlib FP64 distances of full BF16 rows',
        scope='Original/official current normalized input x original/official valid prior histories; official target is current native frame; current input changes Q/K/V jointly',
        cache_intervention_applied=True, root_cause_proven=False, full_model_qualified=False, new_speedup_claim=False)
