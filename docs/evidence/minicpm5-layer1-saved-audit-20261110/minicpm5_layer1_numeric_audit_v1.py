"""Recompute every original full-vocabulary HF metric for layer1 collection.

No launcher or admission. The caller must authenticate metadata/source closure,
hold the serial lock and repeatedly prove physical/resource conditions via guard.
"""
import hashlib
import math
from pathlib import Path
from research import minicpm5_layer1_logits_archive_v1 as archive
from research import minicpm5_layer1_runtime_results_v1 as results
from research import minicpm5_layer1_payload_io_v1 as cache_io

require=archive.require
V=archive.V
TOLERANCE=archive.TOLERANCE
ARMS=tuple(results.ARMS)

def load_reference(plan,guard):
    ref=plan['hf_reference'];registered=plan['inputs'][ref['path']]
    require(registered==dict(sha256=ref['sha256'],identity=ref['identity']),'Exact admitted original HF input')
    payload=dict(path=ref['path'],sha256=ref['sha256'],bytes=ref['identity']['size'])
    data=archive.read_exact(payload,ref['path'],32*2**20,guard,expected_identity=ref['identity'])
    values={};hashes={}
    with archive.Archive(data,32*2**20,guard)as source:
        require(type(source.data)is list and len(source.data)==len(plan['fixtures'])==8,
                'All8 original reference fixtures')
        for fixture,row in zip(plan['fixtures'],source.data):
            guard()
            for key in ('id','input_ids','continuation_ids'):
                require(row[key]==fixture[key],'Original HF prompt/continuation '+key)
            logits,_=source.tensor(row['logits'],[4,V])
            for step in range(4):
                key=row['id'],step
                require(key not in values,'Distinct original reference frame')
                frame=logits[step*V:(step+1)*V]
                values[key]=frame;hashes[key]=hashlib.sha256(frame.tobytes()).hexdigest()
    require(len(values)==32,'Every original HF frame')
    return values,hashes

def check_reference_witnesses(ranks,plan,hashes):
    require(len(ranks)==2 and len(plan['frames'])==32,'Exactly two replicated ranks,32 original frames')
    for rank in (0,1):
        rows=ranks[rank]['native32_evidence']
        require(type(rows)is list and len(rows)==32
            and [x['frame']for x in rows]==plan['frames'],'No missing, repeated or reordered official reference witnesses')
        for row in rows:
            frame=row['frame']
            require(row['original_logits_raw_exact']is True
                and row['hf_reference_logits_sha256']==hashes[frame['fixture'],frame['step']],
                'Every rank binds actual official full-vocabulary reference bytes')

def call_outputs(call,folder,guard):
    item=call['payload']
    data=archive.read_exact(item,folder/'output.pt',3*2**20,guard)
    shape=call['output_metadata']['shape']
    require(shape==[len(call['physical_rows']),1,V]and call['output_metadata']['dtype']=='torch.float32',
            'Original FP32 complete vocabulary logits')
    with archive.Archive(data,3*2**20,guard)as source:
        require(type(source.data)is dict and set(source.data)=={'output'},'Only original full output tensor')
        values,digest=source.tensor(source.data['output'],shape)
    require(digest==call['output_metadata']['sha256'],'Every complete output word matches recorded tensor')
    bundle=call['selected_tap_bundle']
    ref=dict(path=bundle['path'],sha256=bundle['sha256'],bytes=bundle['raw_bytes'])
    taps=archive.read_exact(ref,folder/'selected-taps.bf16',65536,guard,expected_identity=bundle['identity'])
    require(bundle['encoding']=='raw_bf16_le'
        and [e['name']for e in bundle['entries']]==results.SELECTED_TAPS,'Four original selected prediction taps')
    offset=0;size=len(call['physical_rows'])*2048*2
    physical=sorted(call['physical_rows'],key=lambda row:row['physical_row'])
    order=[x['frame']['index']for x in physical]
    require([x['physical_row']for x in physical]==list(range(len(physical))),'Complete original physical mapping')
    for entry in bundle['entries']:
        require(entry['offset']==offset and entry['raw_bytes']==size
            and entry['dtype']=='torch.bfloat16'and entry['shape']==[len(physical),1,2048]
            and entry['frame_indices']==order
            and entry['source_metadata']==call['stage_metadata'][entry['name']],
            'Every original selected tap byte span')
        raw=taps[offset:offset+size];offset+=size
        require(hashlib.sha256(raw).hexdigest()==entry['sha256']==call['stage_sha256'][entry['name']],
                'Saved original prediction tap digest')
    require(offset==len(taps),'No unreferenced selected tap bytes')
    return values,digest,len(taps)

def score_pair(calls,arrays,hashes,reference,deviations,guard):
    require(len(calls)==len(arrays)==len(hashes)==2,'Both actual replicated ranks required')
    require(hashes[0]==hashes[1]and arrays[0]==arrays[1],'Both rank logits equal; count once')
    require(calls[0]['HF_metrics']==calls[1]['HF_metrics'],'Both rank metrics agree')
    physical=calls[0]['physical_rows']
    require(physical==calls[1]['physical_rows']
        and sorted(x['physical_row']for x in physical)==list(range(len(physical))),
        'Both complete original physical row mappings')
    scored=calls[0]['HF_metrics']
    require(len(scored)==len(physical)
        and [x['frame']for x in scored]==[x['frame']for x in physical],
        'Every physical prediction frame scored exactly once')
    output=[]
    for row,mapping in zip(scored,physical):
        guard()
        require(all(row[k]==mapping[k]for k in ('frame','physical_row','current_position')),
                'Metric attaches to exact physical row and position')
        index=row['physical_row'];frame=row['frame']
        got=arrays[0][index*V:(index+1)*V];expected=reference[frame['fixture'],frame['step']]
        m=archive.metrics(got,expected)
        require(all(math.isfinite(value)for key,value in m.items()if key in TOLERANCE),'Finite independently recomputed metrics')
        for key,tolerance in TOLERANCE.items():
            recorded=row['metrics'][key]
            require(type(recorded)in (int,float)and math.isfinite(recorded),'Finite recorded metric')
            delta=abs(m[key]-recorded)
            require(delta<=tolerance,'Original independent float64 tolerance: '+key)
            deviations[key]=max(deviations[key],delta)
        for key in ('argmax_match','reference_top1','neuron_top1'):
            require(m[key]==row['metrics'][key],'Exact original top1 and reference decision')
        failed=archive.failed(m)
        require(failed==row['failed_criteria'],'Original numerical pass/fail classification unchanged')
        output.append(dict(frame=frame,metrics=m,failed_criteria=failed))
    return output

def verify_numeric_trajectory(plan,rank_results,read_json,guard):
    require(callable(read_json)and callable(guard),'External serial guard and authenticated JSON reader')
    guard();controls=results.validate_rank_pair(rank_results,plan);guard()
    reference,reference_hashes=load_reference(plan,guard)
    check_reference_witnesses(rank_results,plan,reference_hashes)
    root=archive.owned.safe_path(plan['output_root'])
    deviations={k:0. for k in TOLERANCE};raw_files=[];all_rows=[];summaries={};tap_bytes=0
    for ai,arm in enumerate(ARMS):
        reports=[]
        for rank in (0,1):
            entry=rank_results[rank]['arm_reports'][ai];folder=root/('cpu-rank'+str(rank))/arm
            require(entry['arm']==arm and entry['report']==str(folder/'result.json'),'Exact original collected arm')
            report=read_json(entry['report'],entry['sha256'])
            require([x['call']for x in report['reports']]==list(range(14)),'All14 calls in original order')
            reports.append(report)
        computed=[]
        for ci in range(14):
            calls=[];arrays=[];hashes=[]
            for rank in (0,1):
                guard();folder=root/('cpu-rank'+str(rank))/arm/('call-%02d'%ci)
                entry=reports[rank]['reports'][ci]
                require(entry['report']==str(folder/'result.json'),'Exact original call metadata')
                call=read_json(entry['report'],entry['sha256'])
                require([call['rank'],call['arm'],call['call']]==[rank,arm,plan['calls'][ci]],
                        'Actual rank/arm/original call identity')
                values,digest,n=call_outputs(call,folder,guard)
                arrays.append(values);hashes.append(digest);calls.append(call)
                raw_files.append(call['payload']);tap_bytes+=n
            computed.extend(score_pair(calls,arrays,hashes,reference,deviations,guard))
        require([x['frame']for x in computed]==plan['frames'],'All32 original frames scored, neither missing nor duplicated')
        summary=dict(passed_frames=sum(not x['failed_criteria']for x in computed),
            top1_matches=sum(x['metrics']['argmax_match']for x in computed),
            failed_frames=[x['frame']['index']for x in computed if x['failed_criteria']])
        for rank in (0,1):
            recorded=rank_results[rank]['summaries'][arm]
            require(recorded['failed_frames']==summary['failed_frames']
                and recorded['argmax_match_fraction']==summary['top1_matches']/32,
                'Both recorded arm summaries agree with independently recomputed failures/top1')
            require(recorded['passed']is (summary['passed_frames']==32 and summary['top1_matches']/32>=.9),
                    'Original aggregate gate unchanged')
        summaries[arm]=summary;all_rows.append(dict(arm=arm,rows=computed))
    guard()
    require(len(raw_files)==84 and tap_bytes==3145728,'Every84 full logits and original selected tap bundle')
    return dict(schema='minicpm5_layer1_full_logits_numeric_audit_v1',complete=True,
        metadata_controls=controls,summaries=summaries,rows=all_rows,logits_payloads=raw_files,
        logits_payload_count=84,original_selected_tap_bytes=3145728,official_reference_frames=32,
        candidate_metric_frames=96,replicated_ranks_verified_and_counted_once=True,
        metrics_independently_recomputed=True,independent_float64_tolerances=dict(TOLERANCE),
        maximum_metric_deviations=deviations,exact_classification_and_argmax=True,
        model_reexecuted=False,matrix_multiplication_reexecuted=False,
        requires_separate_runtime_admission=True,full_model_qualified=False,new_speedup_claim=False)

def audit_collection(plan,rank_results,read_json,guard):
    numeric=verify_numeric_trajectory(plan,rank_results,read_json,guard)
    guard()
    cache=cache_io.audit_trajectory(plan,rank_results,read_json,guard)
    guard()
    require(numeric['complete']is cache['complete']is True
        and numeric['candidate_metric_frames']==96 and cache['calls']==84
        and cache['full_cache_records']==168,'Complete original numeric and whole-cache review')
    return dict(schema='minicpm5_layer1_complete_saved_data_audit_v1',complete=True,
        numeric=numeric,layer1_cache=cache,requires_separate_runtime_admission=True,
        model_reexecuted=False,attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)
