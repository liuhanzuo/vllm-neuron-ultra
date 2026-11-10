"""Authenticate complementary FP32, FP64 and cache evidence without relabeling failure.

Metadata only; the caller authenticates every reference and later verifies all
processes have exited under its serial lock. This permits only the unchanged
diagnostic own-history replay, never model qualification or performance work.
"""
import math
from pathlib import Path
from research import minicpm5_layer1_metric_history_equivalence_v1 as original
from research import minicpm5_layer1_logits_archive_v1 as archive

require = archive.require
ARMS = original.runner.numeric.ARMS


def read_calls(ended, read):
    out = {}
    for rank in (0, 1):
        result = read(ended['outputs']['rank' + str(rank) + '_result'])
        require(result['rank'] == rank and result['complete'] is True, 'Complete historical rank')
        require([e['arm'] for e in result['arm_reports']] == list(ARMS), 'Exact three historical arms')
        for entry in result['arm_reports']:
            report = read(dict(path=entry['report'], sha256=entry['sha256']))
            require(len(report['reports']) == 14, 'All fourteen historical calls')
            for ci, item in enumerate(report['reports']):
                ref = dict(path=item['report'], sha256=item['sha256'])
                call = read(ref)
                require([call['rank'], call['arm'], call['call']['index']] == [rank, entry['arm'], ci],
                        'Exact historical call identity')
                out[rank, entry['arm'], ci] = ref, call
    require(len(out) == 84, 'Complete eighty-four call lineage')
    return out


def validate_equivalence(binding, terminal, ce, cs, cp, read):
    require(callable(read), 'Authenticated reference reader')
    proof = read(binding['metric_equivalence'])
    require(proof['schema'] == 'minicpm5_layer1_metric_history_equivalence_v1'
            and proof['reviewer'] == '/root' and proof['complete'] is True
            and proof['diagnostic_exit'] == binding['diagnostic_exit']
            and proof['call_count'] == 84 and proof['metric_frames'] == 96
            and proof['compared_fields'] == list(original.FIELDS), 'Exact independent history proof')
    for key in ('all_saved_observations_equal', 'original_metric_replay_applies_to_same_bytes'):
        require(proof[key] is True, 'Complete original endpoint equivalence ' + key)
    for key in ('native_metric_reexecuted', 'strict_float64_tolerance_passed',
                'own_history_runtime_admitted', 'attention_causality_proven',
                'full_model_qualified', 'new_speedup_claim'):
        require(proof[key] is False, 'Do not promote the diagnostic ' + key)
    old = read(proof['previous_collection'])
    require(old['actual_exit_observed'] is True and type(old['actual_exit_code']) is int
            and old['actual_exit_code'] == 0 and old['all_four_processes_gone'] is True
            and old['cgroup_empty'] is True, 'Actual prior model collection completed')
    old_spec = read(old['spec'])
    old_plan = read(old_spec['plan'])
    require(cp['hf_reference'] == old_plan['hf_reference'] and cp['frames'] == old_plan['frames'],
            'Exact same official full-vocabulary frames')
    replay_exit = read(proof['previous_metric_replay'])
    float_exit = read(proof['previous_float64_analysis'])
    for ended in (replay_exit, float_exit):
        require(ended['actual_tool_exit_observed'] is True and ended['exact_owner_gone'] is True
                and type(ended['actual_exit']['exit_code']) is int and ended['actual_exit']['exit_code'] == 0,
                'Actual historical independent analysis exit')
    replay = read(replay_exit['result'])
    previous_float = read(float_exit['result'])
    review = read(replay['review'])
    require(replay['owner'] == replay_exit['owner'] and replay['complete'] is replay['passed'] is True
            and replay['all96_original_metrics_exact'] is replay['full_original_layout_restored'] is True
            and replay['frames'] == 96 and replay['original_metric_disagreements'] == [],
            'Exact original-layout FP32 replay result and process')
    require(review['passed'] is True and review['reviewer'] == '/root'
            and review['native_exit'] == proof['previous_collection']
            and review['float64_exit'] == proof['previous_float64_analysis'], 'Historical source review binds both executions')
    source = replay['actual_original_metric_source']
    require(source['path'] == str(original.R / 'validate_real_logits.py')
            and cs['source_sha256'].get(source['path']) == source['sha256']
            and old_spec['source_sha256'].get(source['path']) == source['sha256'], 'Same frozen FP32 metric function')
    result = terminal['result']
    require(result['schema'] == 'minicpm5_layer1_saved_data_diagnostic_v1'
            and result['complete'] is result['diagnostic_only'] is True
            and result['audit_passed'] is False, 'Full diagnostic retains its failure')
    numeric = result['numeric']
    require(numeric['schema'] == 'minicpm5_layer1_full_logits_metric_diagnostic_v1'
            and numeric['strict_numeric_audit_passed'] is False
            and numeric['independent_float64_tolerances'] == archive.TOLERANCE,
            'Original independent tolerance and failure unchanged')
    require(previous_float['complete'] is previous_float['raw_payloads_passed'] is True
            and previous_float['original_metric_tolerance_gate_passed'] is False
            and previous_float['exact_classification_and_argmax'] is True
            and numeric['maximum_metric_deviations'] == previous_float['maximum_metric_deviations'],
            'Original independent float64 diagnostic exactly retained')
    before, after = read_calls(old, read), read_calls(ce, read)
    require(set(before) == set(after) and len(proof['calls']) == 84, 'Both complete trajectories')
    payloads = {x['path']: x for x in numeric['logits_payloads']}
    require(len(payloads) == 84, 'All distinct current output files audited')
    original_rows = {(x['arm'], x['frame']): x for x in replay['rows']}
    require(len(replay['rows']) == len(original_rows) == 96, 'No missing or duplicate FP32 replay frames')
    require([x['arm'] for x in numeric['rows']] == list(ARMS), 'All three numeric arms')
    float_rows = {}
    historical_float_rows = {(a['arm'], r['frame']['index']): r
                             for a in previous_float['rows'] for r in a['rows']}
    require(len(historical_float_rows) == 96, 'Complete historical independent FP64 frames')
    for arm in numeric['rows']:
        require([r['frame'] for r in arm['rows']] == cp['frames'], 'All original FP64 prediction positions')
        for row in arm['rows']:
            float_rows[arm['arm'], row['frame']['index']] = row
    count = 0
    violations = []
    maxima = {k: 0. for k in archive.TOLERANCE}
    for key, saved in zip(sorted(after), proof['calls']):
        br, b = before[key]
        ar, a = after[key]
        require(saved == dict(rank=key[0], arm=key[1], call=key[2], previous=br, current=ar,
                              output_sha256=a['output_metadata']['sha256']), 'Each equality witness binds actual call references')
        require(all(a[field] == b[field] for field in original.FIELDS), 'Complete original observation equality')
        require(a['output_metadata']['dtype'] == 'torch.float32'
                and payloads[a['payload']['path']] == a['payload'], 'Audited original FP32 output')
        if key[0]:
            continue
        for row in a['HF_metrics']:
            frame = row['frame']['index']
            prior = original_rows[key[1], frame]
            require(prior['reproduced_exact'] is True and prior['original'] == row['metrics'],
                    'Exact original metric dictionary covered by real FP32 replay')
            computed = float_rows[key[1], frame]
            require({k: v for k, v in computed.items() if k != 'metric_tolerance_violations'}
                    == historical_float_rows[key[1], frame], 'Same independently calculated FP64 results')
            require(computed['failed_criteria'] == row['failed_criteria'] == archive.failed(computed['metrics']),
                    'Original unchanged scientific failure classification')
            for name in ('argmax_match', 'reference_top1', 'neuron_top1'):
                require(computed['metrics'][name] == row['metrics'][name], 'Exact top1 comparison')
            expected_violations = []
            for name, tolerance in archive.TOLERANCE.items():
                require(math.isfinite(computed['metrics'][name]) and math.isfinite(row['metrics'][name]),
                        'Finite statistics in both precisions')
                delta = abs(computed['metrics'][name] - row['metrics'][name])
                maxima[name] = max(maxima[name], delta)
                if delta > tolerance:
                    expected_violations.append(dict(metric=name, recorded=row['metrics'][name],
                        recomputed=computed['metrics'][name], absolute_delta=delta, tolerance=tolerance))
            require(computed['metric_tolerance_violations'] == expected_violations, 'Every strict scalar failure retained')
            violations.extend(expected_violations)
            count += 1
    require(count == 96 and violations and maxima == numeric['maximum_metric_deviations'], 'Full numeric coverage and truthful failure totals')
    enriched = dict(schema='minicpm5_layer1_reviewed_replay_inputs_v1', complete=True,
                    numeric=numeric, layer1_cache=result['layer1_cache'], metric_equivalence=proof,
                    original_fp32_metrics_exact_by_authenticated_equivalence=True,
                    strict_float64_scalar_tolerance_passed=False, attention_causality_proven=False,
                    full_model_qualified=False, new_speedup_claim=False)
    return enriched, [old['owner'], old['coordinator'], *old['ranks'], replay_exit['owner'], float_exit['owner']]
