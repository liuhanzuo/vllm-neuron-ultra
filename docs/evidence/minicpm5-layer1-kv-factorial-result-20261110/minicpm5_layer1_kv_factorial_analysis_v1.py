"""Descriptive K/V intervention distances; no official-quality/speed claims."""
import math
from pathlib import Path
from research.minicpm5_layer1_cross_history_analysis_v1 import values,distance
from research import minicpm5_layer1_decode_boundary_v1 as boundary


def analyze(proof,plan,guard):
    boundary.require(proof['complete']is True and proof['matrix_cells']==324 and proof['all_previous_paired_outputs_exact']is True,
        'Verified complete factorial with historical paired controls')
    data={}
    for row in proof['rank_proofs'][0]['outputs']:
        guard();raw=Path(row['path']).read_bytes()
        boundary.require(len(raw)==16384 and boundary.sha(raw)==row['sha256'],'Complete authenticated factorial bytes')
        if row['phase']=='matrix':data[row['call'],row['arm'],row['key_arm'],row['value_arm']]=values(raw)
    a,*candidates=boundary.ARMS;records=[]
    for ci in boundary.DECODE_CALLS:
        for candidate in candidates:
            for query in (a,candidate):
                arrays={name:data[ci,query,k,v]for name,k,v in (('own_reference',a,a),('key_only',candidate,a),('value_only',a,candidate),('paired',candidate,candidate))}
                for slot in range(4):
                    sliced={name:arr[slot*2048:(slot+1)*2048]for name,arr in arrays.items()}
                    row=dict(call=ci,slot=slot,candidate=candidate,query_arm=query,
                        frame=next(f for f in plan['calls'][ci]['frames']if f['slot']==slot))
                    for name in ('key_only','value_only','paired'):row[name]=distance(sliced[name],sliced['own_reference'])
                    residual=[sliced['paired'][j]-sliced['key_only'][j]-sliced['value_only'][j]+sliced['own_reference'][j]for j in range(2048)]
                    row['interaction_l2']=math.sqrt(math.fsum(x*x for x in residual));records.append(row)
    summaries=[]
    for candidate in candidates:
        for query in (a,candidate):
            rows=[r for r in records if r['candidate']==candidate and r['query_arm']==query];stats={}
            for name in ('key_only','value_only','paired'):
                ds=[r[name]for r in rows]
                stats[name]=dict(rows=len(ds),changed_rows=sum(d['changed_elements']>0 for d in ds),
                    mean_relative_l2=math.fsum(d['relative_l2']for d in ds)/len(ds),
                    max_relative_l2=max(d['relative_l2']for d in ds),max_abs=max(d['max_abs']for d in ds))
            summaries.append(dict(candidate=candidate,query_arm=query,stats=stats,
                value_effect_larger_rows=sum(r['value_only']['l2']>r['key_only']['l2']for r in rows),
                interaction_nonzero_rows=sum(r['interaction_l2']>0 for r in rows)))
    return dict(schema='minicpm5_layer1_kv_factorial_descriptive_analysis_v1',complete=True,records=records,summary=summaries,
        unique_matrix_cells=162,unique_restored_controls=18,descriptive_precision='stdlib FP64 distances of complete BF16 vectors',
        scope='Differences relative to same current input with anchor K/V; not official-reference quality or additive contributions',
        original_fp64_scalar_audit_failure_unchanged=True,full_model_qualified=False,new_speedup_claim=False)
