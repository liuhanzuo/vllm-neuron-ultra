"""Prepare a bounded, metadata-only Dory follow-up design; never launch it.

Only JSON, JSONL and Python source bytes are opened. Tensor/checkpoint payloads,
native frameworks, existing producer sources and canonical state are untouched.
This is a design inventory, not a runtime spec or runtime admission.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import stat


ROOT = Path('/home/ubuntu/experiments/trn1-latest-models-20261031/research')
BASE = Path('/mnt/nvme/dory-official-cpu-20261108-v1')
ORIGINAL = BASE / 'long-matrix-r1/diagnostic'
REPLAY = BASE / 'residual-replay-r2/diagnostic'
OUTPUT = ROOT / 'dory-targeted-mixer-residual-design-20261110-v1.json'
TARGETS = ((32, 0, 32), (32, 1, 104), (56, 0, 56))
FAILURES = [3607, 4105, 4113, 6160, 6164, 6176, 6181]
CACHE_IDS = tuple(sorted((*range(0, 16, 2), *range(16, 56, 2),
                          *range(88, 128, 2), *range(56, 72, 2))))
BF16_ROW_BYTES = 2560 * 2
READS = {}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def metadata(path, expected=None, cap=24 * 2**20):
    path = Path(path)
    require(path.suffix in ('.json', '.jsonl', '.py'), 'Metadata/source suffix only')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap,
                'Bounded regular metadata/source file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            data = stream.read(cap + 1)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after) and len(data) == before.st_size,
                'Unchanged complete metadata/source read')
    finally:
        os.close(fd)
    sha = hashlib.sha256(data).hexdigest()
    require(expected is None or expected == sha, 'Pinned metadata/source SHA256')
    pin = dict(path=str(path), sha256=sha, bytes=len(data))
    READS[str(path)] = pin
    if path.suffix == '.json':
        return json.loads(data), pin
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in data.splitlines()], pin
    return data, pin


def main():
    plan, plan_pin = metadata(ROOT / 'dory-first5-propagation-plan-20261109-v1.json',
        'a6ff5ba1f8c237e62d536725f9b19eb374a69d9ab4b516cbfdf52fd891be7ecf')
    summary, summary_pin = metadata(ROOT / 'dory-saved-residual-propagation-summary-after-retry-20261110-v2.json')
    review, review_pin = metadata(summary['saved_review']['path'], summary['saved_review']['sha256'])
    exit_record, exit_pin = metadata(summary['actual_saved_review_exit']['path'], summary['actual_saved_review_exit']['sha256'])
    completion, completion_pin = metadata(ROOT / 'dory-residual-saved-review-r3-completion-validation-20261110-v2.json')
    previous_spec, previous_spec_pin = metadata(ROOT / 'dory-residual-native-r2-runtime-spec-20261109-v1.json')
    model, model_pin = metadata(REPLAY / 'model.json')
    require(summary['complete'] is True and summary['failed_events'] == FAILURES
            and review['complete'] is True and review['both_original_and_new_successful_payloads_verified'] is True
            and review['original_events'] == 6200 and review['original_successful_calls'] == 6185,
            'Completed original and residual saved-byte review')
    require(exit_record['actual_external_exit_observed'] is True and exit_record['exit_code'] == 0
            and exit_record['exact_owner_gone'] is True and completion['complete'] is True
            and completion['actual_exit']['sha256'] == exit_pin['sha256'],
            'Existing recorded reviewer exit and completion audit')
    require(summary['full_model_qualified'] is False and summary['root_cause_established'] is False
            and review['original_FP32_metrics_recomputed'] is False,
            'Retain actual limits of completed diagnostics')

    source_pins = {}
    sources = [
        BASE / 'vllm/vllm/model_executor/models/dory.py',
        BASE / 'vllm/tests/models/test_dory.py',
        ROOT / 'dory_residual_visit_capture_v1.py',
        ROOT / 'dory_residual_worker_preparation_v2.py',
        ROOT / 'dory_residual_replay_runtime_v2.py',
        ROOT / 'dory_residual_event_stream_v1.py',
        ROOT / 'dory_residual_replay_history_v1.py',
        ROOT / 'dory_residual_saved_archives_v2.py',
        ROOT / 'dory_residual_saved_observations_v1.py',
        ROOT / 'run_dory_attention_factors_v1.py',
    ]
    for source in sources:
        expected = ('483a3adbbef4d2b66e275579e1e505a3d812f3c1c09b10c349e6cbf286df517b'
                    if source.name == 'dory.py' else previous_spec['source_sha256'].get(str(source)))
        _, source_pins[str(source)] = metadata(source, expected)
    _, builder_pin = metadata(Path(__file__))
    cfg = model['config']
    require(tuple(cfg[k] for k in ('hidden_size', 'num_attention_heads', 'num_key_value_heads',
                                    'head_dim', 'sliding_window')) == (2560, 16, 4, 256, 1024)
            and cfg['recurrent_kv_cache_mode'] == 'per_loop'
            and all(cfg['layer_types'][layer] == 'sliding_attention' for layer, _, _ in TARGETS),
            'Exact original target geometry and independent loop histories')

    original_index, original_index_pin = metadata(ORIGINAL / 'events.jsonl',
        'f67286b8313896b8a18eb5c75ad2bc8cbee3a1d39ba3b364fbb571dbaa2ae029')
    replay_index, replay_index_pin = metadata(REPLAY / 'residual-events.jsonl')
    require(len(original_index) == len(replay_index) == 6200
            and [r['event'] for r in original_index] == list(range(6200))
            and [r['event'] for r in replay_index] == list(range(6200)),
            'Complete original and residual event indexes')
    frames = {frame['event']: frame for frame in summary['frames']}
    calls, gaps = [], []
    selected_rows = marked_input_rows = total_selected_call_rows = 0
    selected = plan['reference_and_marked_calls']
    require(len(selected) == 89 and len(frames) == 87, 'All reference and marked calls retained')
    for item in selected:
        original, original_pin = metadata(item['event']['path'], item['event']['sha256'])
        event = original['event']
        require(original_index[event]['sha256'] == original_pin['sha256']
                and original_index[event]['file'] == Path(original_pin['path']).name,
                'Selected original event belongs to pinned original index')
        replay, replay_pin = metadata(REPLAY / replay_index[event]['file'], replay_index[event]['sha256'])
        require(replay['original_event'] == item['event'] and replay['event'] == event
                and replay['archive_comparison']['exact'] is True
                and replay['actual_archive'] == original['official'],
                'Selected residual call retains exact original output archive metadata')
        for field in ('fixture', 'slot', 'epoch', 'start', 'end', 'phase'):
            require(item[field] == original[field] == replay[field], 'Original selected call identity')
        require(original['kind'] == replay['kind'] == 'call'
                and original['request'] == replay['request'], 'Original owner and event type')
        rows = original['end'] - original['start']
        chosen = item['selected_local_rows']
        require(replay['selected_local_rows'] == chosen, 'Original selected rows')
        full = original['phase'] == 'full_causal'
        require(chosen == (plan['original_prefix_plan']['reference_positions'] if full else [rows - 1]),
                'Both full references and every original mark')
        original_archive = original['official']
        residual_archive = replay['selected_archive']
        require(len(original_archive['tensors']) == 114 and len(residual_archive['tensors']) == 224,
                'Complete archive descriptor coverage')
        require(residual_archive['complete'] is True
                and residual_archive['bytes'] == len(chosen) * 112 * 2 * BF16_ROW_BYTES,
                'Original selected-only residual extent')
        visits = []
        for physical, loop, cache in TARGETS:
            boundaries = {}
            for label in ('input', 'residual_output'):
                key = f'layer-{physical}-loop-{loop}-{label}'
                value = residual_archive['tensors'][key]
                require(value['shape'] == [len(chosen), 2560] and value['dtype'] == 'BF16'
                        and value['bytes'] == len(chosen) * BF16_ROW_BYTES
                        and value['finite'] is True, 'Authenticated selected boundary geometry')
                boundaries[label] = dict(label=key, **value)
            caches = {}
            for kind in ('key', 'value'):
                label = f'cache-{cache}-{kind}'
                value = original_archive['tensors'][label]
                require(value['shape'] == [4, original['end'], 256] and value['dtype'] == 'BF16'
                        and value['bytes'] == rows * 2048 and value['prefix_valid'] is True,
                        'Exact original loop-specific history geometry')
                caches[kind] = dict(label=label, **value)
            visits.append(dict(physical_layer=physical, recurrent_loop=loop,
                               virtual_cache_id=cache, saved_boundaries=boundaries,
                               original_cache_metadata=caches))
        failed = item['original_pointwise_failed']
        if not full:
            require(event in frames and failed == frames[event]['original_pointwise_failed']
                    and item['original_FP32_metrics'] == frames[event]['original_FP32_metrics'],
                    'All existing acceptance results retained verbatim')
            marked_input_rows += rows
        entry = dict(event=event, fixture=original['fixture'], slot=original['slot'],
            epoch=original['epoch'], request=original['request'], phase=original['phase'],
            start=original['start'], end=original['end'], original_query_rows=rows,
            selected_local_rows=chosen, selected_positions=item['selected_positions'],
            original_event=original_pin, residual_event=replay_pin,
            original_output_archive=dict(path=str(ORIGINAL / original_archive['file']),
                                         bytes=original_archive['bytes'], sha256=original_archive['sha256']),
            selected_residual_archive=dict(path=str(REPLAY / residual_archive['file']),
                                          bytes=residual_archive['bytes'], sha256=residual_archive['sha256']),
            saved_input_contains_complete_original_matrix=(len(chosen) == rows),
            missing_original_input_rows=rows-len(chosen),
            original_pointwise_failed=failed, original_FP32_metrics=item['original_FP32_metrics'],
            targets=visits)
        if rows > 1:
            require(original['start'] == 0, 'Only zero-history wide calls need full-input capture')
            gaps.append(dict(event=event, fixture=entry['fixture'], slot=entry['slot'], epoch=entry['epoch'],
                             request=entry['request'], phase=entry['phase'], start=0, end=entry['end'],
                             full_input_rows=rows, saved_rows=len(chosen), missing_rows=rows-len(chosen)))
        calls.append(entry)
        selected_rows += len(chosen)
        total_selected_call_rows += rows

    require([g['event'] for g in gaps] == [1, 4, 7, 9, 11, 6179]
            and [g['full_input_rows'] for g in gaps] == [2060, 2060, 23, 127, 511, 31]
            and selected_rows == 153 and marked_input_rows == 775
            and total_selected_call_rows == 4895
            and sum(c['saved_input_contains_complete_original_matrix'] for c in calls) == 83,
            'Exact six-wide and eighty-three-single-row split')
    require([c['event'] for c in calls if c['original_pointwise_failed']] == FAILURES
            and all(c['original_query_rows'] == 1 for c in calls if c['original_pointwise_failed']),
            'Every original failure remains covered')
    wide_rows = sum(g['full_input_rows'] for g in gaps)
    raw_components = dict(
        six_full_model_original_hidden_logits_all_56_histories=sum(
            c['original_output_archive']['bytes'] for c in calls if c['original_query_rows'] > 1),
        six_full_model_target_input_mixer_residual=wide_rows * 3 * 3 * BF16_ROW_BYTES,
        six_full_shape_target_layer_controls=2 * 2060 * 3 * 2 * BF16_ROW_BYTES,
        eight_input_past_key_past_value_arms=marked_input_rows * 3 * 8 * 2 * BF16_ROW_BYTES,
        restored_original_controls=marked_input_rows * 3 * 2 * BF16_ROW_BYTES,
        four_residual_mixer_combinations=marked_input_rows * 3 * 4 * 2560 * (2 + 2 + 4 + 2))
    require(raw_components['six_full_model_original_hidden_logits_all_56_histories'] == 1165094912
            and sum(raw_components.values()) == 1965750272, 'Explicit prospective raw-byte arithmetic')
    failed_frames = [dict(event=f['event'], fixture=f['fixture'], slot=f['slot'], epoch=f['epoch'],
        phase=f['phase'], position=f['position'], first_numeric_difference=f['first_numeric_difference'],
        largest_rms_increases=f['largest_rms_increases'][:3], original_FP32_metrics=f['original_FP32_metrics'])
        for f in summary['frames'] if f['original_pointwise_failed']]
    design = dict(
        schema='dory_targeted_mixer_residual_design_v1', diagnostic_only=True, metadata_only=True,
        author='/root/dory_next_diagnostic_20261110',
        status='Source and metadata design only; native worker/reviewer/admission not prepared',
        prior_evidence=dict(summary=summary_pin, saved_review=review_pin, actual_review_exit=exit_pin,
            completion_validation=completion_pin, previous_runtime_spec=previous_spec_pin,
            selection_plan=plan_pin, model_metadata=model_pin,
            original_event_index=original_index_pin, residual_event_index=replay_index_pin),
        producer_implementation='Official vLLM Dory CPU oneDNN with upstream ReferenceAttention; not Neuron',
        existing_result_limits=dict(original_FP32_failures=7, original_top1_matches=87,
            FP32_metrics_recomputed_by_this_preparation=False, raw_payloads_reverified_here=False,
            runtime_native_libraries_JIT_identity_reverified_here=False,
            residual_growth_is_causal_proof=False),
        original_acceptance=dict(cosine_min=0.995, max_abs_max=2.0, KL_max=0.02,
            relaxations_allowed=False, metric_key_kl_reference_to_neuron_is_legacy_name=True),
        scope=dict(original_events=6200, original_successful_calls=6185, cache_histories=56,
            physical_layers=72, recurrent_visits=112, original_residual_boundaries=224,
            marked_calls=87, full_reference_calls=2, selected_calls=89, selected_rows=153,
            complete_single_row_calls=83, wide_calls_requiring_full_input_capture=6,
            wide_full_input_rows=wide_rows, wide_missing_input_rows=sum(g['missing_rows'] for g in gaps),
            marked_full_shape_rows=marked_input_rows),
        targets=[dict(physical_layer=p, recurrent_loop=l, virtual_cache_id=c,
                      layer_type='sliding_attention', window=1024) for p, l, c in TARGETS],
        source_pins=source_pins, metadata_builder=builder_pin,
        weight_provenance=dict(recorded_checkpoint=previous_spec['checkpoint'],
            recorded_checkpoint_sha256=previous_spec['checkpoint_sha256'],
            weight_payload_read_here=False, loaded_parameter_values_read_here=False,
            required_before_native_use='Fresh full checkpoint/source/environment validation and official packed mapping'),
        full_input_gaps=gaps, selected_call_manifest=calls, seven_failures=failed_frames,
        prospective_native_stages=[
            dict(name='A_exact_full_input_capture', unchanged_full_model_forwards=6,
                events=[g['event'] for g in gaps], selected_target_visits=3,
                saved_full_boundaries=['input', 'mixer_output', 'residual_output'],
                tokens='Exact pinned original fixture IDs [0:end]; original call shape and output-head rows',
                caches='Every call starts empty at original position 0; independent diagnostic owners retain original slot/epoch/request labels',
                strict_controls=['All 114 original hidden/logit/cache tensor hashes and top1 match',
                    'All original selected residual rows at all 224 boundaries match r2',
                    'Original 112 physical/loop visits and 281 oneDNN projection calls per full forward',
                    'Hooks copy values and return None; record original shape/stride/dtype/storage offset',
                    'Read back six new original-output archives and target captures after actual producer exit'],
                production_slot_reuse_or_failure_recovery_qualified=False),
            dict(name='B_input_and_past_history_factors',
                scope='Every one of 87 original marked calls at all 3 visits, including all 4 marked wide calls and 7 failures',
                arms=[''.join(x) for x in itertools.product('IF', repeat=3)],
                axes=['original_call_shape_input', 'past_key_prefix', 'past_value_prefix'],
                I='Incremental original call input / original slot+epoch+request history',
                F='True full-causal captured input [start:end] / full-causal original history [:start]',
                input_factor_changes=['Q', 'gate', 'current K', 'current V', 'residual input'],
                history_factor_changes='Past positions only; new positions still come from the chosen input',
                empty_prefix_factor_warning='The four marked wide calls start at zero: K/V contrasts are identical empty prefixes and cannot establish a history effect',
                own_control='Run III first and after all arms using independent fresh cache objects; both must match original residual bytes and full appended target K/V hashes',
                full_shape_controls='Replay each true 2060-row reference at every target visit with empty cache; require exact full input/mixer/residual and K/V results',
                shape_comparison='FFF uses the original marked-call row count and end extent; compare against true full-reference rows and report any remaining shape/extent difference',
                no_sparse_zero_filled_queries=True, original_source_precision_unchanged=True,
                arm_layer_calls=2088, restored_layer_calls=261, full_shape_layer_calls=6,
                cache_reconstruction='Authenticate all 6200 original events and 6185 complete 114-tensor archives with CompleteArchiveReplay; capture reference target histories before their release',
                cache_isolation='Clone prefixes into diagnostic objects at start; never append intervention results to chronological banks or another arm; preserve IDs 32/104/56 separately',
                source_layout_caveat='Saved row archives do not prove original strides. Record reconstructed/native strides; require exact own controls. A failed control invalidates causal attribution.'),
            dict(name='C_residual_mixer_factors', arms=['II', 'IF', 'FI', 'FF'],
                axes=['residual_input', 'mixer_output'],
                I='Original call input and newly observed original III mixer output',
                F='Original full-causal input and mixer output sliced to [start:end]',
                implementation='Unchanged official slerp_alpha/justnorm, original BF16 alpha and source alpha scale; no dtype substitution',
                exact_arithmetic='ya=justnorm(a).float(); yb=justnorm(b).float(); lr=abs(alpha.float()*alpha_scale); z=ya+lr.float()*(yb-ya); y=justnorm(z).to(a.dtype)',
                BF16_rounding='Both inner justnorm results round to their BF16 input dtype before float(); z and the outer normalization accumulate in FP32',
                saved=['inner ya BF16', 'inner yb BF16', 'pre-normalization z FP32', 'output y BF16'],
                diagnostics=['Effective lr coordinate statistics (do not assume lr is in [0,1])',
                    'Per-row norms and cancellation before final normalization',
                    'RMS/max_abs/cosine differences before and after final normalization',
                    'Input-only, mixer-only and joint substitution effects; interactions reported explicitly',
                    'II equals native III residual; FF versus true full-reference output retains shape-effect label'],
                permitted_conclusion='Local controlled sensitivity only; no final-logit root cause or model acceptance claim'),
        ],
        prospective_resources=dict(admitted=False, all_numbers_are_planned_not_actual=True,
            raw_output_components_bytes=raw_components, raw_output_total_bytes=sum(raw_components.values()),
            proposed_new_nvme_reservation_bytes=3 * 2**30,
            unchanged_root_free_floor_bytes=28 * 2**30,
            unchanged_nvme_free_floor_bytes=30 * 2**30,
            floor_is_in_addition_to_reservation=True,
            full_original_archive_read_bytes=previous_spec['expected_original_raw_bytes'],
            original_checkpoint_recorded_bytes=previous_spec['checkpoint']['size'],
            conservative_saved_cache_payload_upper_bound_bytes=6 * 2060 * 112 * 4 * 256 * 2,
            original_full_model_parameters=3975229440,
            target_layer_calls=2355, target_oneDNN_calls=7065,
            whole_model_capture_oneDNN_calls=1686, total_oneDNN_calls=8751,
            memory_address_space_CPU_wall_limits='Unfinalized: full official model initialization plus native caches and saved-byte reconstruction must be separately budgeted; do not borrow old admission',
            possible_reuse='One fresh original model setup may serve A then B/C serially to avoid a second checkpoint initialization; not an approved implementation'),
        completion_requirements=['Separately reviewed new native producer, bounded archive writer and saved reviewer',
            'Fresh exact predecessor exit, full history drain, owner and capacity proof before serial admission',
            'All six full-input gaps filled with original complete matrices and exact original controls',
            'All 87 marks and seven failures remain, including slot 1 continuing while slot 3 is reused at event 6181',
            'Actual producer exit and independent saved-byte review before interpretation',
            'Cold cache, long decode, slot reuse, full quality and HTTP serving gates remain pending for production qualification'],
        tensor_payload_bytes_read=0, weight_payload_bytes_read=0, native_framework_imported=False,
        native_worker_prepared=False, runtime_admitted=False, automatic_launch=False,
        runtime_executed=False, canonical_state_modified=False, frozen_sources_modified=False,
        original_FP32_acceptance_passed=False, root_cause_established=False,
        production_lifecycle_qualified=False, full_model_qualified=False, new_speedup_claim=False)
    design['metadata_read_inventory'] = list(READS.values())
    design['metadata_files_read'] = len(READS)
    design['metadata_bytes_read'] = sum(r['bytes'] for r in READS.values())
    blob = (json.dumps(design, indent=2, ensure_ascii=False) + '\n').encode()
    with OUTPUT.open('xb') as stream:
        stream.write(blob)
    print(json.dumps(dict(output=str(OUTPUT), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                         selected_calls=len(calls), full_input_gaps=len(gaps), prospective_raw_bytes=sum(raw_components.values()),
                         metadata_files_read=design['metadata_files_read'], tensor_payload_bytes_read=0,
                         native_executed=False)))


if __name__ == '__main__':
    main()
