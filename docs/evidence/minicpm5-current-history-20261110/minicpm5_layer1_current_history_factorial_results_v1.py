"""Independent raw-output and full-history review of current-input/history factorial interventions."""
import hashlib
import struct
from pathlib import Path
from research import minicpm5_layer1_after_saved_audit_v3 as history
from research import minicpm5_layer1_after_native_cache_v1 as official
from research import minicpm5_layer1_after_official_history_v1 as completed
from research import minicpm5_layer1_current_history_factorial_controls_v1 as controls
from research import minicpm5_layer1_decode_stream_v2 as stream
from research import minicpm5_layer1_official_history_inputs_v1 as inputs
from research import minicpm5_layer1_current_history_factorial_v1 as contract
from research import minicpm5_layer1_payload_io_v1 as payload
import minicpm5_layer0_cpu_constructor_v1 as constructor
p = history.p
require = p.require


def validate_rank_result(result, plan, rank, guard):
    guard()
    require(result['schema'] == 'minicpm5_layer1_current_history_factorial_cpu_driver_v1'
            and result['complete'] is True and type(rank) is int and rank in (0, 1) and result['rank'] == rank,
            'Complete actual current-input/history factorial rank')
    for key in ('full_model_qualified', 'new_speedup_claim', 'attention_causality_proven',
                'full_model_forward_reexecuted', 'self_exit_claimed'):
        require(result[key] is False, 'Diagnostic boundary ' + key)
    for key in ('complete_original_model_constructed', 'layer1_attention_reexecuted',
                'all_own_history_outputs_exact', 'official_history_intervention_performed',
                'official_current_intervention_performed','current_qkv_change_together','previous_real_controls_full_bytes_exact'):
        require(result[key] is True, 'Complete native execution ' + key)
    require([result[k] for k in ('replay_calls', 'physical_rows', 'raw_output_bytes')] == [90, 360, 1474560],
            'All18 calls and5 modes per rank')
    for result_key, plan_key in (('completed_layer1_data', 'layer1_own_history_replay'),
        ('completed_own_history', 'layer1_cross_history'), ('completed_cross_history', 'layer1_kv_factorial'),
        ('completed_factorial', 'layer1_native_cache'), ('completed_native_cache', 'layer1_official_history'),
        ('completed_official_history','layer1_current_history_factorial'),('completed_storage','completed_storage_maintenance')):
        require(result[result_key] == plan[plan_key], 'Exact successful prerequisite ' + result_key)
    def read(ref):
        guard(); require(plan['inputs'][ref['path']]['sha256'] == ref['sha256'], 'Frozen actual metadata input')
        value = p.read_ref(ref); guard(); return value
    bundle = history.completed(plan['layer1_own_history_replay'], read)
    native = official.completed(plan['layer1_official_history'], read)
    previous = completed.completed(plan['layer1_current_history_factorial'], read)
    completed.storage_completed(plan['completed_storage_maintenance'], previous, read)
    constructor.check_load_metadata(result['load_proof'], rank)
    original = previous['records']['rank%d_result' % rank]
    require(result['load_proof'] == original['load_proof']
            and result['official_layer1_shards'] == original['official_layer1_shards'],
            'Same original full model and newly verified official projection shards')
    require(len(result['records']) == 90, 'Every five-mode row retained')
    data = inputs.load(plan, native, guard)
    source = stream.RankStream(bundle['collector_plan'], bundle['collector_chain']['rank_results'][rank],
        bundle['audit_result'], lambda path, sha: read(dict(path=path, sha256=sha)), guard)
    outputs = []; index = 0
    for item in source:
        case = item['case']; ci = case['call']['index']; arm = case['arm']
        donors, current, target, refs = contract.for_call(plan, data, case)
        old_controls = controls.for_call(plan, previous, case, guard)
        for mode in contract.MODES:
            row = result['records'][index]; index += 1; actual = row['result']; desc = row['output']
            _, _, _, proof = contract.select(case, item['normalized_raw'], item['cache_raw'], donors, current, mode)
            require(actual['schema'] == 'minicpm5_layer1_current_history_factorial_callable_result_v1'
                    and actual['complete'] is True and [actual[k] for k in ('rank', 'arm', 'call', 'mode')]
                    == [rank, arm, case['call'], mode] and actual['intervention'] == proof,
                    'Exact original current input, native donor and intervention order')
            for key in ('normalized_original_layout', 'full_input_cache_unchanged',
                        'original_official_weights_unchanged', 'attention_reexecuted'):
                require(actual[key] is True, 'Original attention contract ' + key)
            for key in ('model_reexecuted', 'attention_causality_proven', 'full_model_qualified', 'new_speedup_claim'):
                require(actual[key] is False, 'No whole-model promotion ' + key)
            require(actual['official_history_intervention_performed'] is (mode in ('official_history','official_both'))
                    and actual['official_current_intervention_performed'] is (mode in ('official_current','official_both'))
                    and actual['current_qkv_change_together'] is (mode in ('official_current','official_both')),
                    'Explicit current QKV and prior-history intervention flags')
            require(row['independent_audit_payload_sha256'] == item['audited_call']['payload_sha256']
                    and row['official_target_refs'] == refs and row['official_target_sha256'] == hashlib.sha256(target).hexdigest(),
                    'Actual original current input and current-frame official output target')
            require(row['bucket'] == dict(bucket=case['bucket'], bucket_index=constructor.BUCKETS.index(case['bucket']),
                phase='decode', n_positions=case['bucket'], n_active_tokens=1, cache_aliases=84, configuration_unchanged=True),
                'Original complete bucket configuration')
            path = Path(plan['output_root']) / ('cpu-rank%d' % rank) / arm / ('call-%02d' % ci) / mode / 'attention.bf16'
            require(desc['path'] == str(path) and desc['raw_bytes'] == 16384
                    and desc['sha256'] == actual['attention_output_sha256'], 'Owned complete four-row output')
            raw = payload.read_owned_payload(dict(desc, payload_bytes=16384, payload_sha256=desc['sha256']), path, guard)
            require(all(word & 0x7f80 != 0x7f80 for (word,) in struct.iter_unpack('<H', raw)), 'All output BF16 values finite')
            require(row['previous_control'] == controls.verify(old_controls,mode,raw),
                    'Full previous real own/history/restore output bytes independently reproduced')
            exact = raw == item['expected_attention_raw']
            require(actual['all_four_rows_exact'] is exact, 'Output equality independently observed')
            if mode in ('own', 'restore'): require(exact, 'Every original/restored output reproduced exactly')
            outputs.append(dict(rank=rank, arm=arm, call=ci, mode=mode, path=str(path), raw_bytes=16384,
                sha256=hashlib.sha256(raw).hexdigest(), original_exact=exact,
                official_target_sha256=row['official_target_sha256'], physical_frame_order=case['physical_frame_order']))
    require(index == 90 and source.finish() == result['source_stream'], 'Complete original history stream independently reconstructed')
    guard()
    return dict(complete=True, rank=rank, replay_calls=90, physical_rows=360, raw_bytes=1474560, outputs=outputs,
        all_original_and_restored_outputs_exact=True, every_native_history_overlay_independently_reconstructed=True,
        every_output_independently_reread=True, previous_real_controls_full_bytes_exact=True,
        previous_control_calls=54, full_model_qualified=False, new_speedup_claim=False)


def validate_rank_pair(results, plan, guard):
    require(type(results) is list and len(results) == 2, 'Both actual TP ranks')
    proofs = [validate_rank_result(result, plan, rank, guard) for rank, result in enumerate(results)]
    keys = ('arm', 'call', 'mode', 'sha256', 'official_target_sha256', 'physical_frame_order')
    require([tuple(row[k] for k in keys) for row in proofs[0]['outputs']]
            == [tuple(row[k] for k in keys) for row in proofs[1]['outputs']], 'Complete attention output replication across ranks')
    guard()
    return dict(schema='minicpm5_layer1_current_history_factorial_pair_verification_v1', complete=True,
        ranks=2, decode_calls=180, original_controls=36, restored_controls=36, official_interventions=108,
        physical_rows=720, unique_replicated_output_rows=360, raw_bytes=2949120, rank_proofs=proofs,
        every_output_independently_reread=True, all_original_and_restored_outputs_exact=True,
        every_native_history_overlay_independently_reconstructed=True, official_history_intervention_performed=True,
        official_current_intervention_performed=True,current_qkv_change_together=True,
        previous_real_controls_full_bytes_exact=True,previous_control_calls=108,
        attention_causality_proven=False, full_model_qualified=False, new_speedup_claim=False)
