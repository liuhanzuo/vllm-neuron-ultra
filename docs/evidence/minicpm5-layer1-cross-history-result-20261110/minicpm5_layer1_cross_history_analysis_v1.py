"""FP64 descriptive distances on authenticated BF16 outputs; no quality gates."""
import math
import struct
from pathlib import Path
from research import minicpm5_layer1_payload_io_v1 as payload
from research import minicpm5_layer1_decode_boundary_v1 as boundary


def values(raw):
    words=[w for (w,)in struct.iter_unpack('<H',raw)]
    result=[struct.unpack('<f',struct.pack('<I',w<<16))[0]for w in words]
    boundary.require(len(result)==8192 and all(math.isfinite(x)for x in result),'Complete finite output')
    return result


def distance(left,right):
    norm=math.sqrt(math.fsum(x*x for x in right));delta=[x-y for x,y in zip(left,right)]
    l2=math.sqrt(math.fsum(x*x for x in delta))
    return dict(changed_elements=sum(x!=y for x,y in zip(left,right)),l2=l2,
        relative_l2=l2/norm if norm else None,max_abs=max(abs(x)for x in delta),reference_l2=norm)


def analyze(proof,plan,guard):
    boundary.require(proof['complete']is True and proof['matrix_cells']==108,'Verified real complete cross matrix')
    data={}
    for row in proof['rank_proofs'][0]['outputs']:
        guard();path=Path(row['path']);raw=path.read_bytes()
        boundary.require(len(raw)==16384 and boundary.sha(raw)==row['sha256'],'Authenticated full matrix output')
        if row['phase']=='matrix':data[row['call'],row['arm'],row['history_arm']]=values(raw)
    a,*candidates=boundary.ARMS;records=[]
    for ci in boundary.DECODE_CALLS:
        for candidate in candidates:
            arrays={name:data[ci,q,h] for name,q,h in (('baseline',a,a),('history_only',a,candidate),
                ('input_only',candidate,a),('combined',candidate,candidate))}
            for slot in range(4):
                sliced={name:arr[slot*2048:(slot+1)*2048]for name,arr in arrays.items()}
                row=dict(call=ci,slot=slot,candidate=candidate,
                    frame=next(f for f in plan['calls'][ci]['frames']if f['slot']==slot))
                for name in ('history_only','input_only','combined'):row[name]=distance(sliced[name],sliced['baseline'])
                residual=[sliced['combined'][j]-sliced['history_only'][j]-sliced['input_only'][j]+sliced['baseline'][j]for j in range(2048)]
                row['interaction_l2']=math.sqrt(math.fsum(x*x for x in residual));records.append(row)
    summaries=[]
    for candidate in candidates:
        rows=[r for r in records if r['candidate']==candidate]
        stats={}
        for name in ('history_only','input_only','combined'):
            ds=[r[name]for r in rows]
            stats[name]=dict(rows=len(ds),changed_rows=sum(d['changed_elements']>0 for d in ds),
                mean_relative_l2=math.fsum(d['relative_l2']for d in ds)/len(ds),
                max_relative_l2=max(d['relative_l2']for d in ds),max_abs=max(d['max_abs']for d in ds))
        summaries.append(dict(candidate=candidate,stats=stats,
            history_effect_larger_rows=sum(r['history_only']['l2']>r['input_only']['l2']for r in rows)))
    return dict(schema='minicpm5_layer1_cross_history_descriptive_analysis_v1',complete=True,records=records,summary=summaries,
        unique_matrix_cells=54,unique_restored_controls=18,descriptive_precision='stdlib FP64 distances of complete BF16 vectors',
        scope='Layer1 attention intervention only; not native-reference quality, full-model qualification or speed',
        original_fp64_scalar_audit_failure_unchanged=True,full_model_qualified=False,new_speedup_claim=False)
