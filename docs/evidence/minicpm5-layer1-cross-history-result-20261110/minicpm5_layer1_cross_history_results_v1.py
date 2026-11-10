"""Guarded independent verification of all native own-history output bytes."""
from pathlib import Path
import struct
from research import minicpm5_layer1_cross_history_callable_v1 as native
from research import minicpm5_layer1_decode_boundary_v1 as boundary
from research import minicpm5_layer1_official_shards_v2 as shards
from research import minicpm5_layer1_payload_io_v1 as payload
from research import minicpm5_layer1_after_saved_audit_v3 as history
import minicpm5_layer0_cpu_constructor_v1 as constructor
import minicpm5_original32_layer1_observation_cpu_runner_v2 as collector
require=boundary.require;p=collector.p


def validate_rank_result(result,plan,rank,guard):
    require(callable(guard),'Actual serial result-verification guard');guard()
    require(type(rank)is int and rank in (0,1)and result['rank']==rank
        and result['schema']=='minicpm5_layer1_cross_history_cpu_driver_v1'and result['complete']is True,
        'Complete original own-history rank')
    for key in ('full_model_qualified','new_speedup_claim','attention_causality_proven',
                'full_model_forward_reexecuted','self_exit_claimed'):
        require(result[key]is False,'Preserve diagnostic boundary '+key)
    require(result['layer1_attention_reexecuted']is result['all_own_history_outputs_exact']is True
        and result['complete_original_model_constructed']is True
        and [result[k]for k in ('replay_calls','physical_rows','raw_output_bytes')]==[72,288,1179648],
        'All18 complete four-row attention calls from original full model')
    require(result['completed_layer1_data']==plan['layer1_own_history_replay'],
        'Result bound to actual completed collection and independent audit')
    constructor.check_load_metadata(result['load_proof'],rank)
    load=result['load_proof']
    require(load['phase']=='decode'and load['rank']==rank
        and load['compiled']is load['resharded']is load['sdk_math_modified']is False
        and load['official_model_class']=='neuronx_distributed_inference.models.llama.modeling_llama.NeuronLlamaModel',
        'Original unchanged decode model construction')
    stream=result['source_stream']
    require(stream['schema']=='minicpm5_layer1_rank_decode_stream_v1'and stream['complete']is True
        and stream['rank']==rank and [stream[k]for k in ('all_history_calls','decode_calls','physical_decode_rows',
            'full_cache_records','restored_full_cache_bytes')]==[42,18,72,84,84*4325376]
        and stream['all_read_bytes_match_saved_audit']is True and stream['prefill_replayed']is False,
        'All original history calls and whole input caches were checked')
    def read_json(path,sha):
        guard();require(plan['inputs'][path]['sha256']==sha,'Original admitted result input reference')
        value=p.read_ref(dict(path=path,sha256=sha));guard();return value
    completed=history.completed(plan['layer1_own_history_replay'],lambda ref:read_json(ref['path'],ref['sha256']))
    audited=completed['audit_result']['layer1_cache']['raw_records']
    cases=[case for case in boundary.all_boundaries(plan,read_json)if case['rank']==rank]
    weight=result['official_layer1_shards'];source=plan['official_layer0_weights']['path']
    require(weight['schema']=='minicpm5_layer1_official_shards_v1'and weight['complete']is True
        and weight['rank']==rank and weight['all_four_projection_shards_raw_exact']is True
        and weight['official_file']==dict(path=source,**plan['inputs'][source])
        and weight['header']==plan['safetensors_header']and weight['precision_changed']is False,
        'Actual original official weight file and all4 raw comparisons')
    header=read_json(plan['safetensors_header']['path'],plan['safetensors_header']['sha256'])
    expected=[row for row in boundary.weight_slices(plan,header)if row['rank']==rank]
    require(len(weight['weights'])==4 and set(weight['hashes'])=={shards.local_key(n)for n in shards.PROJECTIONS},
        'All four actual official projections')
    for row,spec in zip(weight['weights'],expected):
        key=shards.local_key(spec['official_key'].split('.')[-2]);shape=list(spec['header_entry']['shape']);shape[spec['shard_axis']]//=2
        require(row['official_key']==spec['official_key']and row['local_key']==key and row['rank']==rank
            and all(row[k]==spec[k]for k in ('shard_axis','start','stop'))and row['shape']==shape
            and row['shard_bytes']==2*shape[0]*shape[1]and row['all_raw_bytes_equal']is True
            and row['shard_sha256']==weight['hashes'][key]
            and payload.raw.hash_string(row['shard_sha256'])and payload.raw.hash_string(row['official_matrix_sha256']),
            'Exact original rank-local official shard proof')
        require(len(row['shard_sha256'])==len(row['official_matrix_sha256'])==64,
            'Complete official matrix and shard digests')
    require(result['paired_history_intervention_performed']is True
        and result['completed_own_history']==plan['layer1_cross_history'], 'Explicit cross-history admission')
    rows=result['records'];require(len(rows)==72,'All54 matrix and18 restored controls')
    cases={(v['arm'],v['call']['index']):v for v in cases}
    order=[(ci,arm,donor,phase) for ci in boundary.DECODE_CALLS for arm in boundary.ARMS
        for donor,phase in [(arm,'matrix')]+[(d,'matrix')for d in boundary.ARMS if d!=arm]+[(arm,'restore')]]
    output=[]
    for row,(ci,arm,donor_arm,phase) in zip(rows,order):
        guard();case=cases[arm,ci];donor=cases[donor_arm,ci];actual=row['result'];descriptor=row['output']
        intervention=native.contract(case,donor);diagonal=arm==donor_arm
        require(actual['schema']=='minicpm5_layer1_cross_history_callable_result_v1'
            and actual['complete']is True and [actual[k]for k in ('rank','arm','call','history_arm')]==[rank,arm,case['call'],donor_arm]
            and actual['intervention']==intervention and row['phase']==phase,'Exact native matrix/control identity and order')
        for name in ('normalized_original_layout','full_input_cache_unchanged','original_official_weights_unchanged','attention_reexecuted'):
            require(actual[name]is True,'Original native contract '+name)
        for name in ('model_reexecuted','attention_causality_proven','full_model_qualified','new_speedup_claim'):
            require(actual[name]is False,'Original diagnostic boundary '+name)
        require(actual['paired_history_intervention_performed']is (not diagonal),'Explicit intervention identity')
        for key,source_case in (('independent_audit_payload_sha256',case),('history_audit_payload_sha256',donor)):
            index=rank*42+payload.raw.ARMS.index(source_case['arm'])*14+ci
            require(row[key]==audited[index]['payload_sha256'],'Both inputs from authenticated complete source payloads')
        require(row['collected_original_call']==case['original_call'] and row['history_original_call']==donor['original_call'],
            'Query and paired K/V donor original sources')
        require(row['bucket']==dict(bucket=case['bucket'],bucket_index=constructor.BUCKETS.index(case['bucket']),
            phase='decode',n_positions=case['bucket'],n_active_tokens=1,cache_aliases=84,configuration_unchanged=True),
            'Original SDK decode bucket')
        path=Path(plan['output_root'])/('cpu-rank'+str(rank))/arm/('call-%02d'%ci)/donor_arm/phase/'attention.bf16'
        require(descriptor['path']==str(path)and descriptor['raw_bytes']==16384
            and descriptor['sha256']==actual['attention_output_sha256'],'Complete owned output path')
        data=payload.read_owned_payload(dict(descriptor,payload_bytes=16384,payload_sha256=descriptor['sha256']),path,guard)
        require(len(data)==16384 and all((word & 0x7f80)!=0x7f80 for (word,)in struct.iter_unpack('<H',data)),
            'Independent complete BF16 finiteness verification')
        exact=boundary.sha(data)==case['expected_attention']['logical_sha256']
        require(actual['all_four_rows_exact']is exact,'Equality is observed, not required for off-diagonal cells')
        if diagonal:require(exact,'Diagonal and restored own-history outputs must be exact')
        output.append(dict(rank=rank,arm=arm,history_arm=donor_arm,call=ci,phase=phase,path=str(path),
            sha256=boundary.sha(data),raw_bytes=len(data),exact_query_original=exact))
    guard()
    return dict(complete=True,rank=rank,replay_calls=72,physical_rows=288,raw_bytes=1179648,outputs=output,
        every_output_independently_reread=True,all_diagonal_and_restored_outputs_exact=True,
        full_model_qualified=False,new_speedup_claim=False)


def validate_rank_pair(results,plan,guard):
    require(callable(guard),'Actual serial complete-rank verification guard');guard()
    require(type(results)is list and len(results)==2,'Both actual TP ranks required')
    proofs=[validate_rank_result(value,plan,rank,guard)for rank,value in enumerate(results)]
    fields=('arm','history_arm','call','phase','sha256')
    require([tuple(v[k]for k in fields)for v in proofs[0]['outputs']]==
        [tuple(v[k]for k in fields)for v in proofs[1]['outputs']], 'Complete attention outputs replicated across ranks')
    for left,right in zip(results[0]['official_layer1_shards']['weights'],results[1]['official_layer1_shards']['weights']):
        require(left['official_key']==right['official_key'] and left['official_matrix_sha256']==right['official_matrix_sha256'],
            'Both rank shards originate from the same official matrix')
    guard();return dict(schema='minicpm5_layer1_cross_history_pair_verification_v1',complete=True,
        ranks=2,decode_calls=144,matrix_cells=108,restored_controls=36,physical_rows=576,unique_replicated_output_rows=288,
        raw_bytes=2359296,rank_proofs=proofs,every_output_independently_reread=True,
        own_history_attention_reproduced=True,paired_history_intervention_performed=True,
        attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)
