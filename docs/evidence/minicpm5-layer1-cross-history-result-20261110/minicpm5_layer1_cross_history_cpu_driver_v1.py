"""Native TP2 own-history driver core; not a launcher or runtime admission.

A separately reviewed worker must first authenticate actual successful collector
and saved-audit exit lineage, create the exact RuntimeContext, import the original
SDK in its environment, and initialize the original gloo TP2 groups. This core
then builds one unchanged full decode model per rank and uses original attention.
"""
import hashlib
import os
from pathlib import Path
from research import minicpm5_layer1_decode_stream_v2 as stream
from research import minicpm5_layer1_official_shards_v2 as shards
from research import minicpm5_layer1_cross_history_callable_v1 as native
import minicpm5_layer0_cpu_constructor_v1 as constructor
require=stream.require


def save_output(context,arm,call,history_arm,phase,raw,guard):
    require(callable(guard),'Actual exclusive output guard')
    guard();context.guard('cpu')
    require(arm in stream.raw.ARMS and type(call)is int and call in stream.boundary.DECODE_CALLS
        and type(raw)is bytes and len(raw)==16384,'Only complete original decode output')
    root=stream.payload_io.safe_path(context.plan['output_root'])
    require(root.is_dir()and root.stat().st_uid==os.getuid()==1000,'Admitted existing output root')
    require(history_arm in stream.raw.ARMS and phase in ('matrix','restore'), 'Exact cross-history output cell')
    rank=context.permit['rank'];require(type(rank)is int and rank in (0,1),'Actual output rank')
    folder=stream.payload_io.safe_path(root/('cpu-rank'+str(rank))/arm/('call-%02d'%call)/history_arm/phase)
    folder.mkdir(parents=True,exist_ok=False);path=folder/'attention.bf16'
    guard()
    with path.open('xb')as f:
        require(f.write(raw)==len(raw),'Complete raw attention output write');f.flush();os.fsync(f.fileno())
    guard()
    descriptor=dict(path=str(path),raw_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
        identity=stream.payload_io.file_identity(path,path.lstat()))
    reread=stream.payload_io.read_owned_payload(dict(descriptor,payload_bytes=len(raw),
        payload_sha256=descriptor['sha256']),path,guard)
    require(reread==raw,'Independently reread every saved attention output byte')
    return descriptor


def run_rank(context,torch,collector_plan,rank_result,audit,read_json,guard):
    """Paired history interventions. Caller owns actual admission and output I/O.

    guard() must bind the complete successful collection+audit lineage, exact
    plan/result/audit arguments, source closure, physical drains, serial lock and
    output/resource limits. Caller must keep context.activate() in scope. All
    four BF16 output rows are saved and independently reread in the owned tree.
    """
    require(callable(guard)and callable(read_json),
        'Externally admitted actual lineage and metadata reader')
    guard();context.guard('cpu')
    require(context._sources_checked,'Full source closure authenticated before native model access')
    for module in (stream,shards,native,constructor):
        path=str(Path(module.__file__).resolve())
        require(context.plan['source_sha256'].get(path)==hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'Exact replay helper in admitted full source closure')
    self_path=str(Path(__file__).resolve())
    require(context.plan['source_sha256'].get(self_path)==hashlib.sha256(Path(self_path).read_bytes()).hexdigest(),
        'Exact native driver source admitted')
    rank=context.permit['rank']
    require(type(rank)is int and rank in (0,1)and rank_result['rank']==rank,
        'Exact actual TP rank')
    require(torch.is_inference_mode_enabled()and torch.get_num_threads()==1
        and torch.distributed.is_initialized()and torch.distributed.get_world_size()==2
        and torch.distributed.get_rank()==rank and torch.distributed.get_backend()=='gloo',
        'Original admitted CPU inference and gloo TP2 environment')
    # Guard authenticates provenance. The stream separately rechecks all saved
    # scopes, complete histories, current payload bytes and decode boundaries.
    inputs=stream.RankStream(collector_plan,rank_result,audit,read_json,guard)
    guard();model,load_proof=constructor.build(rank,'decode')
    constructor.check_load_metadata(load_proof,rank)
    guard();weight_proof=shards.verify_model(context,torch,model,rank)
    require(weight_proof['complete']is True and weight_proof['rank']==rank
        and weight_proof['all_four_projection_shards_raw_exact']is True,'Every official rank-local layer1 shard')
    # Retain complete immutable bytes for all18 decode inputs: about150MiB/rank.
    # K and V are always taken from the same authenticated donor item.
    items={(v['case']['arm'],v['case']['call']['index']):v for v in inputs}
    read_proof=inputs.finish();guard()
    require(len(items)==18,'All18 complete same-rank decode inputs')
    output=[];byte_count=0
    for ci in stream.boundary.DECODE_CALLS:
        for arm in stream.raw.ARMS:
            item=items[arm,ci];case=item['case']
            order=[(arm,'matrix')]+[(donor,'matrix')for donor in stream.raw.ARMS if donor!=arm]+[(arm,'restore')]
            for history_arm,phase in order:
                guard();donor=items[history_arm,ci]
                bucket=constructor.select_bucket(model,case['bucket'])
                result,actual=native.replay(torch,model,case,donor['case'],item['normalized_raw'],
                    donor['cache_raw'],item['expected_attention_raw'],weight_proof['hashes'],guard)
                require(result['complete']is True and result['rank']==rank and result['arm']==arm
                    and result['history_arm']==history_arm and result['call']==case['call'], 'Actual cross cell identity')
                require(type(actual)is bytes and len(actual)==16384
                    and hashlib.sha256(actual).hexdigest()==result['attention_output_sha256'], 'Complete actual four-row bytes')
                if history_arm==arm:
                    require(actual==item['expected_attention_raw']and result['all_four_rows_exact']is True,
                        'Own-history diagonal and restored control exact before/after intervention')
                descriptor=save_output(context,arm,ci,history_arm,phase,actual,guard)
                byte_count+=len(actual)
                output.append(dict(result=result,output=descriptor,bucket=bucket,phase=phase,
                    collected_original_call=case['original_call'],history_original_call=donor['case']['original_call'],
                    independent_audit_payload_sha256=item['audited_call']['payload_sha256'],
                    history_audit_payload_sha256=donor['audited_call']['payload_sha256']))
    guard();require(len(output)==72 and byte_count==1179648,'All54matrix+18restore four-row calls per rank')
    return dict(schema='minicpm5_layer1_cross_history_cpu_driver_v1',complete=True,rank=rank,
        load_proof=load_proof,official_layer1_shards=weight_proof,source_stream=read_proof,
        replay_calls=72,physical_rows=288,raw_output_bytes=byte_count,records=output,
        complete_original_model_constructed=True,full_model_forward_reexecuted=False,
        layer1_attention_reexecuted=True,all_own_history_outputs_exact=True,
        paired_history_intervention_performed=True,attention_causality_proven=False,
        full_model_qualified=False,new_speedup_claim=False,self_exit_claimed=False)
