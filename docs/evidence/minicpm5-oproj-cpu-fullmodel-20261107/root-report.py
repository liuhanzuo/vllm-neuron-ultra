"""Summarize completed CPU trajectories from bounded JSON only; no tensor imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ARMS = ('inclusive_ownhistory_control', 'strict_cpu_s0', 'strict_cpu_s1', 'strict_fp32_oproj_candidate')

def read(path):
    path = Path(path)
    if path.stat().st_size > 8*2**20: raise ValueError('JSON too large')
    return json.loads(path.read_text())

def ref(path):
    path = Path(path)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main(root, exit_path, prefix):
    ended = read(exit_path)
    if ended.get('actual_exit_code') != 0 or not ended.get('actual_exit_observed'):
        raise ValueError('Root actual successful exit required')
    rows = {}; evidence = []
    for rank in (0,1):
        for arm in ARMS:
            path = root/f'cpu-rank{rank}'/arm/'result.json'; arm_result = read(path)
            if not arm_result['complete'] or arm_result['model_forwards'] != 14 or len(arm_result['rows']) != 32:
                raise ValueError('Incomplete full trajectory')
            if arm_result['seed_import_count'] != 1 or not arm_result['continuous_actual_cache_chain']:
                raise ValueError('Own-history chain required')
            flat = []
            for index, item in enumerate(arm_result['reports']):
                call_path = Path(item['report']); call = read(call_path)
                if ref(call_path)['sha256'] != item['sha256']: raise ValueError('Changed call JSON')
                if call['rank'] != rank or call['arm'] != arm or call['call']['index'] != index:
                    raise ValueError('Wrong call identity')
                flat.extend(call['HF_metrics'])
            if flat != arm_result['rows']: raise ValueError('Summary rows differ from actual calls')
            rows[rank,arm] = flat; evidence.append(ref(path))
    for arm in ARMS:
        if rows[0,arm] != rows[1,arm]: raise ValueError('Replicated full logits metrics disagree')
    if rows[0,ARMS[1]] != rows[0,ARMS[2]]: raise ValueError('Strict baseline repeat differs')
    columns = ['arm','frame','fixture','step','passed','failed_criteria','KL','cosine','max_abs','argmax_match']
    output_rows = []; summaries = {}
    for arm in ARMS:
        data = rows[0,arm]
        summaries[arm] = dict(frames=32,passed_frames=sum(not r['failed_criteria'] for r in data),
            passed_prefill=sum(not r['failed_criteria'] for r in data if r['frame']['step']==0),
            passed_decode=sum(not r['failed_criteria'] for r in data if r['frame']['step']>0),
            top1_matches=sum(r['metrics']['argmax_match'] for r in data),
            max_KL=max(r['metrics']['kl_reference_to_neuron'] for r in data),
            failed=[r for r in data if r['failed_criteria']])
        for r in data:
            m=r['metrics']; f=r['frame']
            output_rows.append(dict(arm=arm,frame=f['index'],fixture=f['fixture'],step=f['step'],
                passed=not r['failed_criteria'],failed_criteria='|'.join(r['failed_criteria']),
                KL=m['kl_reference_to_neuron'],cosine=m['cosine'],max_abs=m['max_abs_error'],argmax_match=m['argmax_match']))
    contrasts=[]
    for base,candidate in zip(rows[0,ARMS[1]],rows[0,ARMS[3]]):
        if base['frame']!=candidate['frame']: raise ValueError('Frame order mismatch')
        a=base['metrics']; b=candidate['metrics']
        contrasts.append(dict(frame=base['frame'],baseline_failed=base['failed_criteria'],candidate_failed=candidate['failed_criteria'],
            baseline_KL=a['kl_reference_to_neuron'],candidate_KL=b['kl_reference_to_neuron'],
            KL_delta=b['kl_reference_to_neuron']-a['kl_reference_to_neuron'],
            baseline_max_abs=a['max_abs_error'],candidate_max_abs=b['max_abs_error']))
    result=dict(schema='minicpm5_original32_oproj_cpu_root_summary_v1',actual_exit=ref(exit_path),
        sources=evidence,replicated_ranks_counted_once=True,strict_baseline_repeat_exact=True,
        summaries=summaries,strict_candidate_contrasts=contrasts,
        newly_passed_frames=[x['frame']['index'] for x in contrasts if x['baseline_failed'] and not x['candidate_failed']],
        newly_failed_frames=[x['frame']['index'] for x in contrasts if not x['baseline_failed'] and x['candidate_failed']],
        limits='CPU TP2 full trajectory only. Device, cold-cache, long-decode, slot/inactive, quality96 and paired-serving qualification remain.',
        full_model_qualified=False,new_speedup_claim=False)
    with Path(str(prefix)+'.json').open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    with Path(str(prefix)+'.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(output_rows)
    print(json.dumps({k:result[k] for k in ['summaries','newly_passed_frames','newly_failed_frames']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--actual-exit',type=Path,required=True);parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.actual_exit,args.output_prefix)
