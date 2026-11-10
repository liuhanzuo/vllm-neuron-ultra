"""Bind the new saved-byte diagnostic to the previous exact FP32 replay.

Metadata only. Does not rerun native metrics, admit replay, or turn the failed
float64 scalar tolerance gate into a pass. Both endpoints have independent
saved-byte results; compare their complete authenticated call metadata.
"""
import datetime
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[1]
Q = R / 'research'
sys.path.insert(0, str(R))
from research import minicpm5_layer1_metric_diagnostic_runner_v1 as runner

p = runner.p
ref = runner.collector.reference
require = runner.require
NEW_EXIT = Q / 'minicpm5-layer1-metric-diagnostic-r1-actual-exit-20261110-v1.json'
OLD_EXIT = Q / 'minicpm5-layer0-full-bf16-cpu-root-exit-r1-20261108-v1.json'
OLD_REPLAY = Q / 'minicpm5-full-bf16-metric-replay-actual-exit-20261108-v1.json'
OLD_FLOAT = Q / 'minicpm5-full-bf16-float64-analysis-actual-exit-20261108-v1.json'
FIELDS = ('call', 'bucket', 'input_metadata', 'execution_input_metadata',
          'integer_widening', 'changed_mask_cells', 'output_metadata',
          'stage_names', 'stage_metadata', 'stage_sha256', 'before_metadata',
          'input_after_metadata', 'post_metadata', 'route_records',
          'physical_rows', 'HF_metrics')


def calls(ended):
    out = {}
    for rank in (0, 1):
        result = p.read_ref(ended['outputs']['rank' + str(rank) + '_result'])
        require(result['rank'] == rank and result['complete'], 'Actual complete rank')
        for entry in result['arm_reports']:
            arm = p.read_ref(dict(path=entry['report'], sha256=entry['sha256']))
            for ci, item in enumerate(arm['reports']):
                reference = dict(path=item['report'], sha256=item['sha256'])
                call = p.read_ref(reference)
                key = rank, entry['arm'], ci
                require(key not in out and call['call']['index'] == ci, 'Distinct exact call')
                out[key] = reference, call
    require(len(out) == 84, 'All two-rank, three-arm, fourteen-call metadata')
    return out


def run():
    p.user_bus()
    fd, owner, lock = runner.collector.post.external.acquire_lock()
    try:
        p.clear_window()
        p.active_heavy()
        ne = p.read(NEW_EXIT)
        require(ne['actual_exit_observed'] and ne['actual_exit_code'] == 0,
                'Actual independent diagnostic exit')
        p.gone(ne['owner'])
        terminal = p.read_ref(ne['terminal'])
        require(terminal['owner'] == ne['owner'] and terminal['complete']
                and not terminal['audit_passed'], 'Completed diagnostic retains tolerance failure')
        require(terminal['lock_identity'] == lock, 'Same serial lock')
        ce = p.read_ref(p.read_ref(ne['spec'])['collector_exit'])
        old = p.read(OLD_EXIT)
        require(ce['actual_exit_code'] == old['actual_exit_code'] == 0,
                'Both model collections actually completed')
        for ended in (ce, old):
            for process in [ended['owner'], ended['coordinator'], *ended['ranks']]:
                p.gone(process)
        replay_exit = p.read(OLD_REPLAY)
        float_exit = p.read(OLD_FLOAT)
        for ended in (replay_exit, float_exit):
            require(ended['actual_tool_exit_observed'] and ended['actual_exit']['exit_code'] == 0,
                    'Historical independent replay actually exited successfully')
            p.gone(ended['owner'])
        replay = p.read_ref(replay_exit['result'])
        previous_float = p.read_ref(float_exit['result'])
        require(replay['complete'] and replay['all96_original_metrics_exact']
                and replay['full_original_layout_restored'] and replay['frames'] == 96,
                'Previous exact producer-layout metrics reproduced')
        current_plan = p.read_ref(p.read_ref(ce['spec'])['plan'])
        previous_plan = p.read_ref(p.read_ref(old['spec'])['plan'])
        require(current_plan['hf_reference'] == previous_plan['hf_reference'], 'Same official logits')
        require(current_plan['source_sha256'][replay['actual_original_metric_source']['path']]
                == replay['actual_original_metric_source']['sha256'], 'Same original FP32 metric source')
        numeric = terminal['result']['numeric']
        require(numeric['replicated_ranks_verified_and_counted_once']
                and numeric['exact_classification_and_argmax'], 'Independent current rank and classification proof')
        require(numeric['maximum_metric_deviations'] == previous_float['maximum_metric_deviations'],
                'Exactly the previously observed float64 deviations')
        before, after = calls(old), calls(ce)
        require(set(before) == set(after), 'All original calls preserved')
        checked = []
        payloads = {x['path']: x for x in numeric['logits_payloads']}
        for key in sorted(before):
            br, b = before[key]
            ar, a = after[key]
            require(all(a[field] == b[field] for field in FIELDS), 'Exact saved call and model observations')
            require(payloads[a['payload']['path']] == a['payload'], 'New full logits were independently read')
            checked.append(dict(rank=key[0], arm=key[1], call=key[2], previous=br, current=ar,
                                output_sha256=a['output_metadata']['sha256']))
        original_rows = {(x['arm'], x['frame']): x for x in replay['rows']}
        count = 0
        for (rank, arm, ci), (_, call) in after.items():
            if rank:
                continue
            for row in call['HF_metrics']:
                oldrow = original_rows[arm, row['frame']['index']]
                require(oldrow['reproduced_exact'] and oldrow['original'] == row['metrics'],
                        'Same exact metrics covered by historical original-layout replay')
                count += 1
        require(count == 96, 'All ninety-six unique prediction metrics covered')
        p.lock_proof(owner, lock)
        result = dict(schema='minicpm5_layer1_metric_history_equivalence_v1', complete=True,
                      reviewer='/root', observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      verifier=ref(__file__), diagnostic_exit=ref(NEW_EXIT), previous_collection=ref(OLD_EXIT),
                      previous_metric_replay=ref(OLD_REPLAY), previous_float64_analysis=ref(OLD_FLOAT),
                      compared_fields=list(FIELDS), calls=checked, call_count=84, metric_frames=count,
                      all_saved_observations_equal=True, original_metric_replay_applies_to_same_bytes=True,
                      native_metric_reexecuted=False, strict_float64_tolerance_passed=False,
                      own_history_runtime_admitted=False, attention_causality_proven=False,
                      full_model_qualified=False, new_speedup_claim=False)
        target = Q / 'minicpm5-layer1-metric-history-equivalence-20261110-v1.json'
        p.once(target, result)
        print(json.dumps(ref(target)))
    finally:
        runner.collector.post.external.release_lock(fd)


if __name__ == '__main__':
    run()
