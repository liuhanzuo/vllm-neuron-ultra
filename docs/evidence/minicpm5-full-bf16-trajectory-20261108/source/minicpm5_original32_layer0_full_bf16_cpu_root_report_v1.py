"""Finite three-arm summary from completed JSON reports; no tensor operations."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ARMS = ('strict_anchor_s0', 'strict_layer0_full_bf16_prefill_only',
        'strict_layer0_full_bf16_both_phases')


def read(path):
    path = Path(path)
    if path.stat().st_size > 8 * 2**20:
        raise ValueError('Bounded JSON report required')
    return json.loads(path.read_text())


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(root, exit_path, prefix):
    root = root.resolve()
    ended = read(exit_path)
    require(ended.get('actual_exit_observed') is True
            and type(ended.get('actual_exit_code')) is int
            and ended['actual_exit_code'] == 0,
            'Root-observed successful actual exit required')
    outer = ended['outputs']['outer_result']
    require(Path(outer['path']).resolve() == root / 'outer-result.json'
            and ref(outer['path']) == outer, 'Actual exit binds this output tree')
    require(read(outer['path'])['complete'] is True, 'Complete outer result')
    rows, evidence = {}, []
    for rank in (0, 1):
        for arm in ARMS:
            folder = root / f'cpu-rank{rank}' / arm
            path = folder / 'result.json'
            result = read(path)
            require(result['complete'] is True and result['rank'] == rank
                    and result['arm'] == arm and result['model_forwards'] == 14
                    and len(result['reports']) == 14 and len(result['rows']) == 32,
                    'Complete exact arm identity and coverage')
            require(result['seed_import_count'] == 1
                    and result['continuous_actual_cache_chain'] is True,
                    'Complete own-history trajectory')
            flat = []
            for index, item in enumerate(result['reports']):
                call_path = folder / f'call-{index:02d}' / 'result.json'
                require(item['call'] == index
                        and Path(item['report']).resolve() == call_path
                        and ref(call_path)['sha256'] == item['sha256'],
                        'Exact authenticated call report')
                call = read(call_path)
                require(call['rank'] == rank and call['arm'] == arm
                        and call['call']['index'] == index and call['complete'] is True,
                        'Complete call identity')
                require(call['full_model_qualified'] is False
                        and call['new_speedup_claim'] is False,
                        'Diagnostic result only')
                flat.extend(call['HF_metrics'])
            require(flat == result['rows'], 'Arm rows match actual call rows')
            require([row['frame']['index'] for row in flat] == list(range(32)),
                    'Every original frame in order')
            for row in flat:
                metric = row['metrics']
                require(all(math.isfinite(metric[key]) for key in
                            ('cosine', 'kl_reference_to_neuron', 'max_abs_error', 'reference_margin')),
                        'Finite endpoint metrics')
                failed = []
                if metric['cosine'] < .995:
                    failed.append('cosine')
                if metric['kl_reference_to_neuron'] > .02:
                    failed.append('KL')
                if metric['max_abs_error'] > 2:
                    failed.append('max_abs')
                if metric['reference_margin'] > .5 and not metric['argmax_match']:
                    failed.append('high_margin_argmax')
                require(sorted(failed) == sorted(row['failed_criteria']),
                        'Unchanged original per-frame thresholds')
            rows[rank, arm] = flat
            evidence.append(ref(path))
    for arm in ARMS:
        require(rows[0, arm] == rows[1, arm], 'Replicated rank endpoint metrics agree')

    summaries, output_rows, contrasts = {}, [], {}
    for arm in ARMS:
        data = rows[0, arm]
        summaries[arm] = dict(frames=32,
            passed_frames=sum(not row['failed_criteria'] for row in data),
            passed_prefill=sum(not row['failed_criteria'] for row in data if row['frame']['step'] == 0),
            passed_decode=sum(not row['failed_criteria'] for row in data if row['frame']['step'] > 0),
            top1_matches=sum(row['metrics']['argmax_match'] for row in data),
            overall_top1_gate=sum(row['metrics']['argmax_match'] for row in data) / 32 >= .9,
            failed=[row for row in data if row['failed_criteria']])
        for row in data:
            frame, metric = row['frame'], row['metrics']
            output_rows.append(dict(arm=arm, frame=frame['index'], fixture=frame['fixture'],
                step=frame['step'], passed=not row['failed_criteria'],
                failed_criteria='|'.join(row['failed_criteria']),
                KL=metric['kl_reference_to_neuron'], cosine=metric['cosine'],
                max_abs=metric['max_abs_error'], argmax_match=metric['argmax_match']))
    for baseline_arm, candidate_arm in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
        changes = []
        for baseline, candidate in zip(rows[0, baseline_arm], rows[0, candidate_arm]):
            require(baseline['frame'] == candidate['frame'], 'Paired frame identity')
            changes.append(dict(frame=baseline['frame'],
                baseline_failed=baseline['failed_criteria'], candidate_failed=candidate['failed_criteria'],
                baseline_metrics=baseline['metrics'], candidate_metrics=candidate['metrics']))
        contrasts[baseline_arm + '__vs__' + candidate_arm] = dict(
            newly_passed_frames=[x['frame']['index'] for x in changes if x['baseline_failed'] and not x['candidate_failed']],
            newly_failed_frames=[x['frame']['index'] for x in changes if not x['baseline_failed'] and x['candidate_failed']],
            frames=changes)
    result = dict(schema='minicpm5_original32_layer0_full_bf16_cpu_root_summary_v1',
        actual_exit=ref(exit_path), sources=evidence, replicated_ranks_counted_once=True,
        summaries=summaries, contrasts=contrasts,
        limits=['JSON endpoint summary only; not an independent raw tensor recomputation.',
                'Selected prediction taps require separate numerical analysis; hashes do not measure error.',
                'CPU diagnostic only. Device, cold-cache, long decode, slot reuse/inactive-slot, quality96 and paired serving remain required.'],
        full_model_qualified=False, new_speedup_claim=False)
    json_path, csv_path = Path(str(prefix) + '.json'), Path(str(prefix) + '.csv')
    require(not json_path.exists() and not csv_path.exists(), 'Fresh immutable report paths')
    with json_path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with csv_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({arm: {key: value for key, value in summary.items() if key != 'failed'}
                      for arm, summary in summaries.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--actual-exit', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.actual_exit, args.output_prefix)
