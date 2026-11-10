"""Explain independently reviewed residual differences without changing gates.

Only saved-review metadata is read. The entry point requires the actual complete
producer/reviewer success chain; no live tensor reads, model run, or new metrics.
All87 original marked calls and all224 visit boundaries remain in the output.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

Q=Path(__file__).resolve().parent
sys.path[:0]=[str(Q),str(Q.parent)]
import nemotron9b_after_dory_residual_review_v1 as completion
from dory_residual_saved_observations_v1 import BOUNDARIES,VISITS

PLAN=dict(path=str(Q/'dory-first5-propagation-plan-20261109-v1.json'),
    sha256='a6ff5ba1f8c237e62d536725f9b19eb374a69d9ab4b516cbfdf52fd891be7ecf')
require=completion.require


def event_number(path):
    match=re.fullmatch(r'event-([0-9]{6})\.json',Path(path).name)
    require(match is not None,'Original event filename')
    return int(match[1])


def original_frames(plan):
    rows=plan['reference_and_marked_calls']
    require(len(rows)==89 and sum(row['phase']=='full_causal'for row in rows)==2,
            'Both original references and all87 marked calls')
    result={event_number(row['event']['path']):row for row in rows if row['phase']!='full_causal'}
    require(len(result)==87 and sum(row['original_pointwise_failed']for row in result.values())==7,
            'Every original marked event and all seven failures')
    return result


def stats_valid(stats):
    require(stats['elements']==2560 and type(stats['bitwise_equal_words'])is int
            and type(stats['numerically_equal_values'])is int
            and 0<=stats['bitwise_equal_words']<=stats['numerically_equal_values']<=2560,
            'Complete finite BF16 residual row counts')
    for name in ('max_abs','rms','reference_rms'):
        require(type(stats[name])in(int,float) and math.isfinite(stats[name]) and stats[name]>=0,
                'Finite nonnegative saved explanatory metric')
    require(type(stats['mean_delta'])in(int,float) and math.isfinite(stats['mean_delta']), 'Finite saved signed error')
    for name in ('relative_rms','cosine'):
        require(stats[name]is None or (type(stats[name])in(int,float) and math.isfinite(stats[name])),
                'Finite or explicitly undefined relative metric')
    require(stats['relative_rms']is None or stats['relative_rms']>=0,'Nonnegative relative RMS')
    require(stats['cosine']is None or -1-1e-12<=stats['cosine']<=1+1e-12,'Finite cosine range')
    require(stats['acceptance_metric']is False and stats['arithmetic']==
        'Exact BF16-to-float values with Python float64 accumulation; explanatory only',
        'Original explanatory metric semantics, not original FP32 acceptance')
    for name in ('reference_row_sha256','incremental_row_sha256'):
        require(isinstance(stats[name],str) and re.fullmatch('[0-9a-f]{64}',stats[name]),'Exact reviewed row hash')
    equal=stats['bitwise_equal_words']==2560
    require((stats['reference_row_sha256']==stats['incremental_row_sha256'])==equal,
            'Saved bit identity and hash identity agree')
    if stats['numerically_equal_values']==2560:
        require(stats['max_abs']==stats['rms']==stats['mean_delta']==0,'Equal numeric values have zero errors')
    return stats


def transition(input_stats,output_stats):
    """Describe observed residual error growth; this is not causal attribution."""
    before,after=input_stats['rms'],output_stats['rms']
    return dict(input_rms=before,output_rms=after,rms_change=after-before,
        rms_ratio=after/before if before else None,
        numeric_difference_first_appears_here=before==0 and after>0,
        numeric_difference_disappears_here=before>0 and after==0,
        root_cause_established=False)


def analyze(saved,plan):
    completion.saved_scope(saved)
    original=original_frames(plan)
    frames=saved['observations']['frames']
    require(len(frames)==87 and [row['event']for row in frames]==sorted(original),
            'Chronological complete original87 frames, including repeated token positions')
    by_boundary={key:dict(boundary=key,frames=0,bit_different_frames=0,numeric_different_frames=0,
        failed_original_frames_with_difference=0,max_abs=0.0,max_rms=0.0,
        max_relative_rms=None,min_cosine=None,largest_rms_event=None)for key in BOUNDARIES}
    first_bit,first_numeric=Counter(),Counter()
    explained=[]
    for frame in frames:
        source=original[frame['event']]
        require(frame['fixture']==source['fixture'] and frame['position']==source['end']-1
                and frame['original_pointwise_failed']==source['original_pointwise_failed']
                and frame['original_FP32_metrics']==source['original_FP32_metrics'],
                'Original fixture/event/position/FP32 metrics and outcome unchanged')
        require(frame['original_metrics_recomputed']is False and frame['comparison_is_acceptance_metric']is False
                and frame['root_cause_established']is frame['full_model_qualified']is False,
                'Do not promote residual diagnostic metrics to model acceptance')
        metrics=frame['comparisons']
        require(set(metrics)==set(BOUNDARIES) and len(metrics)==224,'Every executed visit boundary retained')
        for key in BOUNDARIES:stats_valid(metrics[key])
        first=next((key for key in BOUNDARIES if metrics[key]['bitwise_equal_words']<2560),None)
        numeric=next((key for key in BOUNDARIES if metrics[key]['numerically_equal_values']<2560),None)
        require(first==frame['first_nonidentical_boundary'],'Recomputed first boundary matches saved review')
        first_bit[first or 'all_equal']+=1;first_numeric[numeric or 'all_equal']+=1
        visits=[]
        for index,(layer,loop)in enumerate(VISITS):
            prefix=f'layer-{layer}-loop-{loop}'
            a,b=metrics[prefix+'-input'],metrics[prefix+'-residual_output']
            visits.append(dict(visit=index,physical_layer=layer,recurrent_loop=loop,
                               **transition(a,b)))
        jumps=sorted(visits,key=lambda row:(-row['rms_change'],row['visit']))
        # Preserve all boundaries/visits; top changes are only a convenience view.
        explained.append(dict(event=frame['event'],fixture=source['fixture'],slot=source['slot'],epoch=source['epoch'],
            phase=source['phase'],position=frame['position'],first_bit_difference=first,
            first_numeric_difference=numeric,original_pointwise_failed=frame['original_pointwise_failed'],
            original_FP32_metrics=copy.deepcopy(frame['original_FP32_metrics']),
            boundary_metrics=copy.deepcopy(metrics),visits=visits,largest_rms_increases=jumps[:5]))
        for key in BOUNDARIES:
            stats=metrics[key];row=by_boundary[key];row['frames']+=1
            bit=stats['bitwise_equal_words']<2560;num=stats['numerically_equal_values']<2560
            row['bit_different_frames']+=bit;row['numeric_different_frames']+=num
            row['failed_original_frames_with_difference']+=bool(bit and frame['original_pointwise_failed'])
            row['max_abs']=max(row['max_abs'],stats['max_abs'])
            if row['largest_rms_event']is None or stats['rms']>row['max_rms']:
                row['max_rms']=stats['rms'];row['largest_rms_event']=frame['event']
            if stats['relative_rms']is not None:
                row['max_relative_rms']=max(row['max_relative_rms']or 0,stats['relative_rms'])
            if stats['cosine']is not None:
                row['min_cosine']=stats['cosine']if row['min_cosine']is None else min(row['min_cosine'],stats['cosine'])
    return dict(schema='dory_saved_residual_propagation_summary_v1',complete=True,
        scope=dict(marked_calls=87,original_failed_calls=7,physical_layers=72,executed_visits=112,boundaries=224,
                   selected_calls=89,selected_rows=153),
        first_bit_difference_counts=dict(first_bit),first_numeric_difference_counts=dict(first_numeric),
        boundary_summary=list(by_boundary.values()),frames=explained,
        failed_events=[row['event']for row in explained if row['original_pointwise_failed']],
        original_failure_injection=copy.deepcopy(saved['observations']['failure_review']),
        residual_metrics_recomputed_here=False,original_FP32_metrics_recomputed=False,
        metric_arithmetic='Previously independently reviewed BF16 rows, float64 explanatory differences',
        observed_growth_is_causal_proof=False,original_FP32_acceptance_passed=False,
        root_cause_established=False,production_lifecycle_qualified=False,full_model_qualified=False,new_speedup_claim=False)


def from_actual_review(actual_exit_ref):
    # All real producer/reviewer exits and the original full saved scope are checked first.
    completed=completion.completed(actual_exit_ref)
    reader=completion.Reader()
    receipt=reader(actual_exit_ref);saved=reader(receipt['saved_review']);plan=reader(PLAN)
    summary=analyze(saved,plan)
    summary.update(actual_saved_review_exit=copy.deepcopy(actual_exit_ref),saved_review=receipt['saved_review'],
        selection_plan=PLAN,completed_history_owners=len(completed['history']['processes']),
        metadata_only=True,raw_tensor_bytes_read=0,weight_bytes_read=0)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-review-exit',required=True)
    parser.add_argument('--saved-review-exit-sha256',required=True)
    args=parser.parse_args()
    result=from_actual_review(dict(path=args.saved_review_exit,sha256=args.saved_review_exit_sha256))
    from completed_storage_files_v1 import once
    print(json.dumps(once(Q/'dory-saved-residual-propagation-summary-20261109-v1.json',result)))
