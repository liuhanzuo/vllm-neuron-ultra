"""Compact scientific findings from the complete independently reviewed matrix."""
import json
from pathlib import Path
from dory_layer0_serial_contract_v2 import read_pinned, require

Q = Path(__file__).resolve().parent
EXIT = dict(path=str(Q/'dory-attention-factors-saved-review-actual-exit-20261109-v1.json'),
            sha256='2f8c008db3ba7bb3a10f52364b57cac1df658ae2965442a5cb8252ecbc66401e')


def load(ref):
    return read_pinned(ref['path'],ref['sha256'],cap=16*2**20)


def summarize():
    receipt = load(EXIT)
    require(receipt['actual_external_exit_observed'] and receipt['exit_code'] == 0, 'Actual completed independent review')
    review = load(receipt['result'])
    spec = load(receipt['spec'])
    native_exit = load(spec['native_exit'])
    native = load(native_exit['result'])
    require(review['complete'] and review['raw_output_bytes_verified'] == 25141248
            and review['all_saved_input_provider_hashes_reconstructed']
            and review['all_recorded_output_statistics_exact'] and review['original_seven_failures_preserved'],
            'Every saved original input and output independently checked')
    neq = lambda f,a,b:f['output']['tensors'][a]['sha256'] != f['output']['tensors'][b]['sha256']
    per_layer, same_inputs = [], []
    for layer in (0,2,4):
        frames = [f for f in native['frames'] if f['layer'] == layer]
        require(len(frames) == 87, 'All original marks per layer')
        shape = [f['comparisons_to_original_full_reference']['reference_full_extent_strided'] for f in frames]
        per_layer.append(dict(layer=layer, marked_points=87,
            query_row_count_changed_outputs=sum(x['bitwise_equal_words'] != x['elements'] for x in shape),
            query_row_count_max_abs=max(x['max_abs'] for x in shape),
            masked_future_key_extent_changed_outputs=sum(neq(f,'FFF','reference_full_extent_contiguous') for f in frames),
            key_value_storage_layout_changed_outputs=sum(neq(f,'reference_full_extent_strided','reference_full_extent_contiguous') for f in frames),
            native_row_replay_vs_III_changed_outputs=sum(neq(f,'III','incremental_row_replay') for f in frames),
            visible_key_history_different=sum(f['visible_history_differences']['key']['different_bf16_words']>0 for f in frames),
            visible_value_history_different=sum(f['visible_history_differences']['value']['different_bf16_words']>0 for f in frames),
            changed_outputs_when_replacing_only_Q=sum(neq(f,'FII','III') for f in frames),
            changed_outputs_when_replacing_only_K=sum(neq(f,'IFI','III') for f in frames),
            changed_outputs_when_replacing_only_V=sum(neq(f,'IIF','III') for f in frames),
            changed_outputs_when_replacing_QKV=sum(neq(f,'FFF','III') for f in frames)))
        for f in frames:
            if f['inputs']['QI']['sha256'] == f['inputs']['QF']['sha256'] and all(
                    x['different_bf16_words'] == 0 for x in f['visible_history_differences'].values()):
                same_inputs.append(dict(event=f['event'],layer=layer,position=f['position'],
                    original_query_shape_replayed=f['original_query_shape_replayed'],
                    row_replay_differs_from_full_reference=f['comparisons_to_original_full_reference']['incremental_row_replay']['bitwise_equal_words'] != 4096,
                    original_pointwise_failed=f['original_pointwise_failed']))
    return dict(schema='dory_attention_factors_reviewed_findings_v1',saved_review_exit=EXIT,
        saved_review=receipt['result'],native_exit=spec['native_exit'],native_result=native_exit['result'],
        sdpa_calls=2877,original_marked_points=87,attention_layer_points=261,
        original_row_shape_exact='249/249',full_query_selected_rows_exact='198/198',
        original_wider_prefill_points=4,raw_output_bytes_verified=25141248,
        original_cache_append_bytes_verified=135794688,per_layer=per_layer,
        same_query_and_visible_history_cases=same_inputs,
        original_seven_failures_preserved=True,original_FP32_acceptance_passed=False,
        final_failure_root_cause_established=False,full_model_qualified=False,new_speedup_claim=False,
        interpretation=[
            'The full-query control keeps2060 query rows, fills the33saved rows exactly,and explicitly zeros unobserved rows; only198selected outputs are validated.',
            'The query-row comparison holds the target query,complete2060KVvalues/strides,and its causal/window mask row fixed.',
            'The key-extent comparison keeps the same target query and visibleKVvalues; it removes only masked future keys and necessarily changes key extent/stride geometry.',
            'Input replacement counts show sensitivity in the controlled single-row layout,not accuracy improvement or speedup.',
            'All original7full-model failures and all complete-model,Neuron,generation,quality,cold4096,long,slotreuse,HTTP4 and paired-performance requirements remain.'])


if __name__ == '__main__':
    result = summarize()
    out = Q/'dory-attention-factors-reviewed-findings-20261109-v1.json'
    with out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(out),per_layer=result['per_layer'])))
