"""Partition-aware TP2 SDK batch/layout v2 worker; exact outer native receipt inherited."""
import importlib
from pathlib import Path
from research import minicpm5_layer1_sdk_batch_layout_completion_v3 as latest
from research import minicpm5_layer1_sdk_batch_layout_cpu_driver_v1 as driver
from minicpm5_layer0_runtime_auth_v1 import identity
from minicpm5_original88_decode_mask_analysis_v1 import p
from research import minicpm5_layer1_sdk_batch_layout_sources_v2 as source_auth
require=driver.require


def authenticate_outer_drain(context,done):
    # Runtime receipts did not exist when the scientific plan was frozen. Their
    # immutable references are authenticated by the externally guarded permit.
    context.guard('cpu');permit=context.permit;root=Path(context.plan['output_root']);observed={}
    def read(ref,expected):
        context.guard('cpu')
        require(type(ref)is dict and set(ref)=={'path','sha256'} and ref['path']==str(expected),
            'Exact externally admitted runtime receipt path')
        before=identity(ref['path'])
        require(before['size']<16*2**20,'Bounded complete outer runtime metadata')
        value=source_auth.load(ref);require(identity(ref['path'])==before,'Stable complete outer metadata read')
        observed[ref['path']]=before;return value
    a=read(permit['runtime_admission'],root/'admission.json')
    require(a['schema']=='minicpm5_layer1_sdk_batch_layout_cpu_admission_v2'
        and a['outer_owner']==permit['outer_owner'] and a['lock_identity']==permit['lock_identity']==done['lock_identity']
        and a['plan']==permit['runtime_plan'] and a['observed_drains']==permit['observed_drains']
        and a['source_contract']==context.plan['source_contract']
        and a['source_authentication']==permit['source_authentication']
        and context._inherited_native_receipt['source_contract']==a['source_contract']
        and a['attempt_id']==permit['attempt_id'] and a['latest_completion']==done['binding']==context.plan['latest_completion']
        and a['runtime_admitted']is a['predecessors_checked']is True
        and a['full_model_qualified']is a['new_speedup_claim']is False,
        'Actual externally acknowledged outer admission and latest lineage')
    proof=read(a['observed_drains'],root/'predecessor-drain.json');physical=proof['latest_physical']
    require(proof['complete']is True and proof['latest_completion']==a['latest_completion']
        and proof['outer_owner']==a['outer_owner'] and proof['lock_identity']==a['lock_identity']
        and physical['complete']is physical['physical_checks_performed']is True
        and physical['lock_identity']==done['lock_identity']
        and physical['all_additional_owners_checked']==done['owners']
        and physical['all_current_and_prior_services_preserved']is True
        and physical['nemotron']['fresh_physical_drain_verified']is True
        and physical['nemotron']['services']==1308,
        'Complete outer prelaunch latest physical drain, every original service and owner')
    p.lock_proof(a['outer_owner'],a['lock_identity'])
    for owner in done['owners']:context.guard('cpu');p.gone(owner)
    return observed


def load_binding(context):
    context.guard('cpu');require(context._sources_checked,'All sources authenticated before actual lineage')
    binding=context.plan['latest_completion'];latest.require_complete_binding(binding)
    observed={}
    def read(ref):
        context.guard('cpu');value=context.read_json(ref['path'],ref['sha256'])
        observed[ref['path']]=context.plan['inputs'][ref['path']]['identity'];return value
    done=latest.completed(binding,read,lambda:context.guard('cpu'))
    bundle,native,previous=(done[k]for k in ('collector','native','current_history'))
    require(previous['binding']==context.plan['layer1_sdk_batch_layout']
        and native['binding']==context.plan['layer1_official_history'],
        'Actual current/history and native prerequisites')
    previous_plan=previous['plan'];extensions={'source_sha256','inputs','output_root','layer1_sdk_batch_layout','latest_completion','source_contract'}
    require({k:v for k,v in context.plan.items()if k not in extensions}==
        {k:v for k,v in previous_plan.items()if k not in extensions},
        'Every completed current/history scientific and supplemental field retained')
    for key in ('inputs','source_sha256'):
        require(all(context.plan[key].get(path)==value for path,value in previous_plan[key].items()),
            'Every completed current/history input and source retained')
    original=bundle['collector_plan']
    allowed={'source_sha256','inputs','output_root','layer1_own_history_replay','layer1_cross_history',
        'layer1_kv_factorial','layer1_native_cache','layer1_official_history',
        'layer1_current_history_factorial','completed_storage_maintenance','layer1_sdk_batch_layout','latest_completion','source_contract'}
    require({k:v for k,v in context.plan.items()if k not in allowed}==
        {k:v for k,v in original.items()if k not in allowed},'Complete original scientific execution plan')
    for key in ('inputs','source_sha256'):
        require(all(context.plan[key].get(path)==value for path,value in original[key].items()),
            'Every original collector input and source retained')
    require(all(context.plan['source_sha256'].get(path)==sha for path,sha in done['source_sha256'].items()),
        'Every completed lineage source retained')
    observed.update(authenticate_outer_drain(context,done))
    require(hasattr(context,'sdk_guard') and callable(context.sdk_guard),'Actual runner fast/full guard required')
    context.sdk_guard.watch_metadata(observed);context.sdk_guard.full('completed_metadata_authenticated')
    def guard():
        context.guard('cpu')
        # sdk_guard checks every historical metadata identity in full at all
        # lifecycle boundaries and at most one-second intervals between calls.
    # The outer still owns the same serial lock. Replaying an idle-only full
    # drain inside our own live service would reject that service as competing.
    guard()
    return bundle,native,previous,guard


def run(context,directory):
    context.guard('cpu');require(context._sources_checked,'Exact sources before native imports')
    rank=context.permit['rank']
    require(Path(directory)==Path(context.plan['output_root'])/('cpu-rank'+str(rank)),
        'Only admitted native rank output tree')
    bundle,native,previous,guard=load_binding(context)
    require(callable(context.verify_loaded_sources),'Actual complete late-import source check required')
    context.verify_loaded_sources();context.sdk_guard.full('before_native_import')
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
                context.verify_loaded_sources();context.sdk_guard.full('before_model_replay')
                with torch.inference_mode():
                    result=driver.run_rank(context,torch,bundle['collector_plan'],
                        bundle['collector_chain']['rank_results'][rank],bundle['audit_result'],
                        context.read_json,guard,native,previous)
                    context.verify_loaded_sources();context.sdk_guard.full('after_model_replay');guard();result['latest_completion']=context.plan['latest_completion']
                    return result
            finally:parallel_state.destroy_model_parallel()
        finally:torch.distributed.destroy_process_group()
