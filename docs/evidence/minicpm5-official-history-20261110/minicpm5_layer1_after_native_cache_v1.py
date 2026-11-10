"""Authenticate the actual native32 capture, independent audit and comparison.

Metadata alone is not admission. The new caller must hold the serial lock and
check physical drainage before touching model state or raw payloads.
"""
from pathlib import Path
import minicpm5_layer1_native_cache_runner_v1 as original
p = original.p
require = p.require


def completed(binding, read):
    require(set(binding) == {'schema', 'model_exit', 'audit_exit', 'comparison_exit'}
            and binding['schema'] == 'minicpm5_layer1_completed_native_cache_v1', 'Exact completed native binding')
    ended = read(binding['model_exit']); ae = read(binding['audit_exit']); audit = read(ae['result'])
    ce = read(binding['comparison_exit']); comparison = read(ce['result'])
    spec = read(ended['spec']); plan = read(spec['plan']); review = read(ended['review']); live = read(ended['live'])
    records = {k: read(ref) for k, ref in ended['outputs'].items()}
    require(ended['schema'] == 'minicpm5_layer1_native_cache_actual_exit_v1'
            and ended['actual_exit_observed'] is True and type(ended['actual_exit_code']) is int
            and ended['actual_exit_code'] == 0 and ended['all_three_processes_gone'] is ended['cgroup_empty'] is True,
            'Actual successful native collector exit')
    require(ended['owner'] == live['owner'] and ended['actual_execution_handle'] == live['actual_execution_handle']
            and live['spec'] == ended['spec'] and live['review'] == ended['review'], 'Actual native owner binding')
    require(review['passed'] is True and review['blockers'] == [] and review['spec'] == ended['spec']
            and spec['source_sha256'] == plan['source_sha256'] == review['source_sha256'], 'Executed native source review')
    for receipt, schema in ((ae, 'minicpm5_layer1_native_cache_post_exit_audit_actual_exit_v1'),
                            (ce, 'minicpm5_layer1_native_cache_comparison_actual_exit_v1')):
        require(receipt['schema'] == schema and receipt['actual_tool_exit_observed'] is True
                and type(receipt['actual_exit_code']) is int and receipt['actual_exit_code'] == 0
                and receipt['exact_owner_gone'] is receipt['serial_lock_reacquired'] is True,
                'Actual independent post-exit program completion')
    require(audit['schema'] == 'minicpm5_layer1_native_cache_post_exit_audit_v1'
            and audit['complete'] is audit['every_raw_byte_reread_after_all_model_processes_exited'] is True
            and audit['model_exit'] == binding['model_exit'] and audit['owner'] == ae['owner'], 'Exact independent native audit')
    require(comparison['schema'] == 'minicpm5_layer1_native_cache_comparison_v1' and comparison['complete'] is True
            and comparison['native_audit_exit'] == binding['audit_exit'] and comparison['owner'] == ce['owner']
            and comparison['comparisons'] == 288 and len(comparison['records']) == 288
            and comparison['sdk_binding'] == plan['layer1_own_history_replay'], 'Actual complete official cache comparison')
    lock = ended['lock_identity']
    require(all(x['lock_identity'] == lock for x in (ae, audit, ce, comparison)), 'One unchanged serial lock')
    for result, name in ((audit, 'audit_minicpm5_layer1_native_cache_result_v1.py'),
                         (comparison, 'compare_minicpm5_layer1_native_cache_v1.py')):
        require(result['source']['path'] == str(Path(__file__).parent / name)
                and p.sha(result['source']['path']) == result['source']['sha256'], 'Original completed audit source')
    require(records['outer']['complete'] is True and records['outer']['owner'] == ended['owner'] == records['admission']['outer_owner']
            and records['children']['exit_codes'] == records['coordinator']['native_exits'] == [0]
            and records['service_exit']['successful'] is records['cleanup']['physically_drained'] is True
            and records['cleanup']['original_failure'] is None, 'Actual successful native service and cleanup')
    require(records['admission']['spec'] == ended['spec'] and records['admission']['review'] == ended['review']
            and records['admission']['lock_identity'] == lock, 'Actual native runtime admission')
    native = records['native_result']
    require(native['process'] == ended['native'] and native['admission'] == ended['outputs']['admission']
            and native == records['coordinator']['result'], 'Exact actual native result')
    proof = audit['result']
    require(proof == records['coordinator']['validated'] and proof['complete'] is True
            and [proof[k] for k in ('frames', 'cache_tensors', 'full_cache_bytes', 'full_logits_bytes', 'selected_taps')]
            == [32, 64, 5894144, 16711680, 6816]
            and proof['every_raw_byte_independently_reread'] is proof['original_full_logits_and_all_selected_taps_exact']
            is proof['all_native_cache_prefix_links_exact'] is True, 'Full independently verified native32 history')
    require(native['native_provenance'] == plan['native_provenance'] and len(native['records']) == 32, 'Original official model')
    for value in (ended, ae, audit, proof, ce, comparison):
        require(value['full_model_qualified'] is value['new_speedup_claim'] is False, 'Diagnostic evidence only')
    owners = [ended['owner'], ended['coordinator'], ended['native'], ae['owner'], ce['owner']]
    require(len({(v['pid'], v['start_ticks']) for v in owners}) == 5, 'Five distinct completed processes')
    return dict(binding=binding, ended=ended, spec=spec, plan=plan, records=records, native=native,
                audit=audit, audit_exit=ae, comparison=comparison, comparison_exit=ce, owners=owners, lock_identity=lock)


def physical(bundle, guard):
    guard()
    for owner in bundle['owners']: p.gone(owner)
    ended = bundle['ended']; admission = bundle['records']['admission']
    proof = original.verify_owned_drain(bundle['spec'], ended['outputs']['admission'],
        admission['attempt_id'], ended['coordinator'], True)
    guard()
    return dict(native_service=proof, owners_checked=bundle['owners'], independent_auditor_and_comparator_gone=True)
