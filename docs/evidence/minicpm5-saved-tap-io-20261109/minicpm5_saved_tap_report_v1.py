"""Complete, callback-driven MiniCPM5 saved-tap numeric report.

This module has no CLI or filesystem reader. A future serial worker must
authenticate metadata and supply a guarded, bounded raw reader. Tests pass
generated in-memory data only. Calling this function is not runtime admission.
"""
import copy
import json

import analyze_minicpm5_saved_tap_metadata_v1 as metadata
from minicpm5_saved_tap_analysis_v1 import (
    ARMS, TAPS, WIDTH, decode_bundle, frame_order, perturbation, require,
)

PAIRS = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))


def propagation(early, late):
    """A descriptive late/early ratio, including the undefined-zero case."""
    a, b = early['delta_rms'], late['delta_rms']
    return dict(early_delta_rms=a, late_delta_rms=b,
                late_to_early_rms_ratio=b/a if a else None,
                zero_early_nonzero_late=(a == 0 and b != 0),
                causal_amplification_established=False)


def analyze(read_metadata, read_bundle, guard):
    """Validate all metadata, then consume all 84 bundles, one call at a time.

    The caller authenticates every metadata reference. read_bundle(ref, size)
    must enforce its exact allowed path, size, stable identity and budget.
    This module independently verifies raw size, bundle/entry SHA256, finite
    BF16 values, and byte equality of replicated ranks. guard() is called on
    both sides of each callback and each call's numeric computation.
    """
    calls, metadata_refs = {}, []

    def collect(ref):
        guard()
        value = read_metadata(copy.deepcopy(ref))
        guard()
        require(ref not in metadata_refs, 'Each original metadata record read exactly once')
        metadata_refs.append(copy.deepcopy(ref))
        if 'selected_tap_bundle' in value:
            key = (value['rank'], value['arm'], value['call']['index'])
            require(key not in calls, 'No duplicate original call')
            calls[key] = copy.deepcopy(value)
        return value

    diagnosis = metadata.analyze(collect)
    require(len(metadata_refs) == 95 and len({r['path'] for r in metadata_refs}) == 95,
            'All95 distinct original metadata records')
    expected_keys = {(rank, arm, ci) for rank in (0, 1) for arm in ARMS for ci in range(14)}
    require(set(calls) == expected_keys, 'All84 original rank-arm-call records')
    require(len({c['selected_tap_bundle']['path'] for c in calls.values()}) == 84,
            'All84 distinct original bundle paths')
    for rank in (0, 1):
        for arm in ARMS:
            trajectory = [calls[rank, arm, ci] for ci in range(14)]
            require([f['index'] for c in trajectory for f in c['call']['frames']] == list(range(32)),
                    'Every original frame0..31 exactly once per arm/rank')
            phases = [c['call']['phase'] for c in trajectory]
            require(phases.count('prefill') == 8 and phases.count('decode') == 6,
                    'Eight original prefills and six original decodes')
            for c in trajectory:
                require(len(c['call']['frames']) == (1 if c['call']['phase'] == 'prefill' else 4),
                        'Original phase-specific frame geometry')
                require(c['HF_metrics'] == [dict(x, frame=r['frame'], physical_row=r['physical_row'])
                        for x, r in zip(c['HF_metrics'], c['physical_rows'])],
                        'Original endpoint physical rows match tap mapping')
    for arm in ARMS:
        for ci in range(14):
            left, right = calls[0, arm, ci], calls[1, arm, ci]
            require(left['call'] == right['call'] and left['physical_rows'] == right['physical_rows'],
                    'Replicated ranks preserve identical original call and physical mapping')

    reports = {pair: [] for pair in PAIRS}
    bundle_refs, raw_bytes = [], 0
    for ci in range(14):
        guard()
        decoded = {}
        for arm in ARMS:
            rank0_raw = None
            for rank in (0, 1):
                call = calls[rank, arm, ci]
                bundle = call['selected_tap_bundle']
                ref = dict(path=bundle['path'], sha256=bundle['sha256'])
                guard()
                raw = read_bundle(copy.deepcopy(ref), bundle['raw_bytes'])
                guard()
                values = decode_bundle(raw, call)
                if rank == 0:
                    rank0_raw, decoded[arm] = raw, values
                else:
                    require(raw == rank0_raw, 'Both replicated rank bundles must be byte identical')
                raw_bytes += len(raw)
                bundle_refs.append(dict(rank=rank, arm=arm, call=ci, **ref, raw_bytes=len(raw)))
                del raw, values
            del rank0_raw
        for left_arm, right_arm in PAIRS:
            left, right = calls[0, left_arm, ci], calls[0, right_arm, ci]
            require(left['call'] == right['call'] and left['physical_rows'] == right['physical_rows'],
                    'Same original trajectory across candidate arms')
            endpoints = {arm: {r['frame']['index']: r for r in calls[0, arm, ci]['HF_metrics']}
                         for arm in (left_arm, right_arm)}
            for physical_row, frame in enumerate(frame_order(left)):
                taps = {}
                for tap in TAPS:
                    stats = perturbation(decoded[left_arm][tap][frame], decoded[right_arm][tap][frame])
                    # The third pair compares candidates, not the strict anchor.
                    stats['reference_is_failing_cpu_anchor'] = left_arm == ARMS[0]
                    taps[tap] = stats
                reports[left_arm, right_arm].append(dict(
                    call_index=ci, phase=left['call']['phase'], frame_index=frame,
                    physical_row=physical_row, physical_frame_order=frame_order(left),
                    left_original_endpoint=copy.deepcopy(endpoints[left_arm][frame]),
                    right_original_endpoint=copy.deepcopy(endpoints[right_arm][frame]),
                    taps=taps, propagation=dict(
                        attention=propagation(taps[TAPS[0]], taps[TAPS[2]]),
                        layer_output=propagation(taps[TAPS[1]], taps[TAPS[3]]))))
        del decoded
        guard()
    require(len(bundle_refs) == 84 and raw_bytes == 3145728, 'Full84 bundles and3145728 bytes')
    pair_reports = []
    for pair in PAIRS:
        rows = reports[pair]
        require(len(rows) == 32 and sorted(r['frame_index'] for r in rows) == list(range(32)),
                'All32 frames retained for each comparison')
        pair_reports.append(dict(left_arm=pair[0], right_arm=pair[1], frames=rows))
    result = dict(schema='minicpm5_saved_tap_numeric_report_v1', complete=True,
        original_actual_exit=diagnosis['original_actual_exit'],
        original_spec=diagnosis['original_spec'], original_plan=diagnosis['original_plan'],
        metadata_inputs=metadata_refs, bundle_inputs=bundle_refs,
        original_summaries=diagnosis['original_summaries'],
        original_remaining_gates=diagnosis['original_remaining_gates'],
        comparisons=pair_reports, frames_per_pair=32, taps=list(TAPS),
        hidden_width=WIDTH, pairwise_vector_comparisons=384,
        raw_bytes_validated=raw_bytes, replicated_ranks_byte_equal=True,
        ranks_counted_once=True, original_FP32_metrics_recomputed=False,
        scientific_thresholds_changed=False, absolute_HF_tap_error_measured=False,
        runtime_admission_proven_by_this_module=False,
        full_model_qualified=False, new_speedup_claim=False,
        limitations=[
            'Only four saved tap positions; no numerical trace of the other209 stages.',
            'Both strict anchor and candidates fail original CPU gates; deltas are not absolute HF errors.',
            'Float64 descriptions and late/early ratios do not replace original FP32 endpoint gates.',
            'Candidate-to-candidate comparisons are labeled separately from anchor-relative comparisons.',
            'Every frame is included; rank replicas are verified in bytes and counted once.',
            'Input authenticity, resource/serial ownership and actual exit need a separately reviewed executor.'
        ])
    require(len(json.dumps(result, allow_nan=False).encode()) <= 4*2**20, 'Bounded finite report')
    guard()
    return result
