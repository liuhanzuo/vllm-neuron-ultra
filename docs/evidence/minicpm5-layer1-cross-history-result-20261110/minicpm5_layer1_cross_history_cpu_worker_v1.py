"""TP2 native worker for admitted complete layer1 own-history replay only."""
import importlib
from pathlib import Path
from research import minicpm5_layer1_after_own_history_v1 as own
from research import minicpm5_layer1_after_saved_audit_v3 as history
from research import minicpm5_layer1_cross_history_cpu_driver_v1 as driver
from minicpm5_layer0_runtime_auth_v1 import identity
require=driver.require


def load_binding(context):
    context.guard('cpu');require(context._sources_checked,'All sources authenticated before actual lineage')
    observed={}
    def read(ref):
        context.guard('cpu');value=context.read_json(ref['path'],ref['sha256'])
        observed[ref['path']]=context.plan['inputs'][ref['path']]['identity'];return value
    bundle=history.completed(context.plan['layer1_own_history_replay'],read)
    replay=own.completed(context.plan['layer1_cross_history'],read)
    original=bundle['collector_plan'];allowed={'source_sha256','inputs','output_root','layer1_own_history_replay','layer1_cross_history'}
    require({k:v for k,v in context.plan.items()if k not in allowed}==
        {k:v for k,v in original.items()if k not in allowed},'Complete original scientific execution plan')
    for key in ('inputs','source_sha256'):
        require(all(context.plan[key].get(path)==value for path,value in original[key].items()),
            'Every original collector input and source retained')
    def guard():
        context.guard('cpu')
        require(all(identity(path)==wanted for path,wanted in observed.items()),
            'Every actual completed collection/audit metadata identity unchanged')
        for old in replay['owners']:history.p.gone(old)
        history.p.empty_group(history.p.GROUP_PREFIX+replay['spec']['unit']+'.service')
        for owner in [*bundle['collector_chain']['owners'],bundle['saved_audit_owner']]:history.p.gone(owner)
        history.p.empty_group(history.p.GROUP_PREFIX+bundle['collector_spec']['unit']+'.service')
    guard();history.physical(bundle,guard);own.physical(replay,guard)
    return bundle,guard


def run(context,directory):
    context.guard('cpu');require(context._sources_checked,'Exact sources before native imports')
    rank=context.permit['rank']
    require(Path(directory)==Path(context.plan['output_root'])/('cpu-rank'+str(rank)),
        'Only admitted native rank output tree')
    bundle,guard=load_binding(context)
    with context.activate():
        guard()
        import torch
        importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
        from neuronx_distributed.parallel_layers import parallel_state
        context.record_cpu_import();torch.set_num_threads(1)
        torch.distributed.init_process_group('gloo')
        try:
            parallel_state.initialize_model_parallel(2,skip_collective_init=True)
            try:
                require(torch.distributed.get_rank()==rank and torch.distributed.get_world_size()==2,
                    'Original actual TP2 process identity')
                with torch.inference_mode():
                    result=driver.run_rank(context,torch,bundle['collector_plan'],
                        bundle['collector_chain']['rank_results'][rank],bundle['audit_result'],
                        context.read_json,guard)
                    guard();result['completed_layer1_data']=context.plan['layer1_own_history_replay']
                    result['completed_own_history']=context.plan['layer1_cross_history']
                    return result
            finally:parallel_state.destroy_model_parallel()
        finally:torch.distributed.destroy_process_group()
