"""TP2 worker for admitted complete layer1 current-input/history factorial calls."""
import importlib
from pathlib import Path
from research import minicpm5_layer1_after_cross_history_v1 as crossed
from research import minicpm5_layer1_after_kv_factorial_v1 as factored
from research import minicpm5_layer1_after_native_cache_v1 as official
from research import minicpm5_layer1_after_official_history_v1 as completed
from research import minicpm5_layer1_after_own_history_v1 as own
from research import minicpm5_layer1_after_saved_audit_v3 as history
from research import minicpm5_layer1_current_history_factorial_cpu_driver_v1 as driver
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
    factorial=crossed.completed(context.plan['layer1_kv_factorial'],read)
    actual_factorial=factored.completed(context.plan['layer1_native_cache'],read)
    native=official.completed(context.plan['layer1_official_history'],read)
    previous=completed.completed(context.plan['layer1_current_history_factorial'],read)
    storage=completed.storage_completed(context.plan['completed_storage_maintenance'],previous,read)
    original=bundle['collector_plan'];allowed={'source_sha256','inputs','output_root','layer1_own_history_replay','layer1_cross_history','layer1_kv_factorial','layer1_native_cache','layer1_official_history','layer1_current_history_factorial','completed_storage_maintenance'}
    require({k:v for k,v in context.plan.items()if k not in allowed}==
        {k:v for k,v in original.items()if k not in allowed},'Complete original scientific execution plan')
    for key in ('inputs','source_sha256'):
        require(all(context.plan[key].get(path)==value for path,value in original[key].items()),
            'Every original collector input and source retained')
    def guard():
        context.guard('cpu')
        require(all(identity(path)==wanted for path,wanted in observed.items()),
            'Every actual completed collection/audit metadata identity unchanged')
        for old in storage['owners']:history.p.gone(old)
        for later in (actual_factorial,native,previous):
            for old in later['owners']:history.p.gone(old)
            history.p.empty_group(history.p.GROUP_PREFIX+later['spec']['unit']+'.service')
        for old in factorial['owners']:history.p.gone(old)
        history.p.empty_group(history.p.GROUP_PREFIX+factorial['spec']['unit']+'.service')
        for old in replay['owners']:history.p.gone(old)
        history.p.empty_group(history.p.GROUP_PREFIX+replay['spec']['unit']+'.service')
        for owner in [*bundle['collector_chain']['owners'],bundle['saved_audit_owner']]:history.p.gone(owner)
        history.p.empty_group(history.p.GROUP_PREFIX+bundle['collector_spec']['unit']+'.service')
    guard();history.physical(bundle,guard);own.physical(replay,guard);crossed.physical(factorial,guard)
    factored.physical(actual_factorial,guard);official.physical(native,guard)
    completed.physical(previous,guard);completed.storage_physical(storage,guard)
    return bundle,native,previous,guard


def run(context,directory):
    context.guard('cpu');require(context._sources_checked,'Exact sources before native imports')
    rank=context.permit['rank']
    require(Path(directory)==Path(context.plan['output_root'])/('cpu-rank'+str(rank)),
        'Only admitted native rank output tree')
    bundle,native,previous,guard=load_binding(context)
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
                        context.read_json,guard,native,previous)
                    guard();result['completed_layer1_data']=context.plan['layer1_own_history_replay']
                    result['completed_own_history']=context.plan['layer1_cross_history']
                    result['completed_cross_history']=context.plan['layer1_kv_factorial']
                    result['completed_factorial']=context.plan['layer1_native_cache']
                    result['completed_native_cache']=context.plan['layer1_official_history']
                    result['completed_official_history']=context.plan['layer1_current_history_factorial']
                    result['completed_storage']=context.plan['completed_storage_maintenance']
                    return result
            finally:parallel_state.destroy_model_parallel()
        finally:torch.distributed.destroy_process_group()
