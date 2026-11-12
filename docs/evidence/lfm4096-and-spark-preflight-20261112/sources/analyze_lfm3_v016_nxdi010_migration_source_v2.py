"""AST/source/metadata-only migration inventory; never imports investigated code."""
import ast
import hashlib
import json
import os
from pathlib import Path
import resource

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=ROOT/'research'
OLD=ROOT/'lfm3-package-native-core-v1/work/target/vllm_neuron'
NEW=Q/'vllm-neuron-053/vllm_neuron'
VLLM=Q/'vllm016-source/vllm'
SITE=Path('/home/ubuntu/venvs/vllm_neuron_053_py312/lib/python3.12/site-packages')
OLD_SITE=Path('/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages')
NXDI=SITE/'neuronx_distributed_inference'
SOURCES={};EVIDENCE={}

def raw(path):
    path=Path(path)
    assert path.suffix in ('.py','.json','.toml','.txt','.pth') or path.name=='METADATA'
    assert path.is_file() and path.stat().st_size<=8*2**20
    with path.open('rb') as stream:value=stream.read(8*2**20+1)
    assert len(value)<=8*2**20;value.decode('utf8')
    SOURCES[str(path)]=dict(sha256=hashlib.sha256(value).hexdigest(),bytes=len(value))
    return value

def index(path):
    tree=ast.parse(raw(path));result={}
    def walk(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                name=prefix+node.name;result[name]=node
                if isinstance(node,ast.ClassDef):walk(node.body,name+'.')
    walk(tree.body)
    return tree,result

def excerpt(name,path,start,end):
    lines=raw(path).decode().splitlines();assert 1<=start<=end<=len(lines)
    EVIDENCE[name]=dict(path=str(path),sha256=SOURCES[str(path)]['sha256'],start_line=start,end_line=end,text='\n'.join(lines[start-1:end]))

def definition(name,path,qualified):
    tree,nodes=index(path);node=nodes[qualified];excerpt(name,path,node.lineno,node.end_lineno)


def literal(path,name):
    tree,_=index(path)
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in node.targets))


def build():
    excerpt('old_identity_model_scope',OLD/'models/lfm3/identity.py',16,107)
    definition('old_scope_admission',OLD/'models/lfm3/identity.py','validate_scope')
    definition('old_software_identity',OLD/'models/lfm3/identity.py','software_identity')
    definition('old_variant',OLD/'models/lfm3/identity.py','selected_variant')
    definition('old_source_identity',OLD/'models/lfm3/identity.py','source_identity')
    excerpt('old_registration',OLD/'models/lfm3/registration.py',1,61)
    excerpt('old_optional_plugin',OLD/'models/lfm3/__init__.py',1,13)
    excerpt('new_platform_entrypoint',NEW.parent/'pyproject.toml',1,30)
    excerpt('new_nxdi_model_types',NXDI/'utils/constants.py',49,68)
    definition('new_arch_resolution',NEW/'worker/neuronx_distributed_model_loader.py','_get_neuron_model_cls')
    definition('new_model_loader_entry',NEW/'worker/neuronx_distributed_model_loader.py','get_neuron_model')
    definition('old_hybrid_cache',OLD/'models/lfm3/modeling.py','Lfm3Cache')
    definition('old_nxdi_model',OLD/'models/lfm3/modeling.py','NeuronLfm3Model')
    definition('old_nxdi_app',OLD/'models/lfm3/modeling.py','NeuronLfm3ForCausalLM')
    definition('old_config',OLD/'models/lfm3/modeling.py','Lfm3InferenceConfig')
    definition('old_one_token_prefill',OLD/'models/lfm3/modeling.py','Lfm3CausalLMBase')
    definition('old_attention',OLD/'models/lfm3/text_core.py','Attention')
    definition('old_conv',OLD/'models/lfm3/text_core.py','ShortConv')
    definition('old_fp32_rope',OLD/'models/lfm3/text_core.py','apply_rope')
    definition('old_partition_helper',OLD/'models/lfm3/helpers.py','partitioned_parameter')
    definition('old_artifact_admission',OLD/'models/lfm3/artifacts.py','load_or_compile')
    definition('old_artifact_config',OLD/'models/lfm3/artifacts.py','admit_saved_configuration')
    definition('old_serial_compiler',OLD/'models/lfm3/artifacts.py','serial_compiler')
    definition('new_loader_generic_fallback',NEW/'worker/neuronx_distributed_model_loader.py','NeuronModelBase._load_weights_common')
    definition('new_loader_generic_deletion',NEW/'worker/neuronx_distributed_model_loader.py','NeuronModelBase._get_compiled_model_path')
    definition('new_default_config',NEW/'worker/neuronx_distributed_model_loader.py','_get_default_neuron_config')
    definition('new_prompt_override',NEW/'worker/neuronx_distributed_model_loader.py','_validate_override_neuron_config')
    excerpt('new_legacy_sampler_import',NEW/'worker/neuronx_distributed_model_loader.py',49,81)
    excerpt('new_runner_sampler_setup',NEW/'worker/neuronx_distributed_model_runner.py',31,44)
    excerpt('new_runner_sampler_instance',NEW/'worker/neuronx_distributed_model_runner.py',180,193)
    definition('new_deferred_sampling',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner.sample_tokens')
    definition('new_execute',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner.execute_model')
    definition('new_proxy_cache_specs',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner.get_kv_cache_spec')
    definition('new_proxy_cache_memory',NEW/'worker/neuron_worker.py','NeuronWorker._calculate_contiguous_kv_memory')
    definition('new_request_slot_allocate',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner._process_new_request_for_continuous_batching')
    definition('new_request_slot_cached',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner._process_cached_request_for_continuous_batching')
    definition('new_request_update',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner._update_states')
    definition('new_cache_no_allocation',NEW/'worker/neuronx_distributed_model_runner.py','NeuronxDistributedModelRunner.initialize_kv_cache')
    definition('new_platform_config',NEW/'platform.py','NeuronPlatform.check_and_update_config')
    definition('new_cpu_sampler',VLLM/'v1/sample/sampler.py','Sampler.sample')
    definition('new_topk_native',VLLM/'v1/sample/ops/topk_topp_sampler.py','TopKTopPSampler.forward_native')
    definition('old_sampler_wrap',OLD/'worker/cpu_sampler.py','BoundedTopKTopPSampler')
    definition('old_sampler_enable',OLD/'worker/cpu_sampler.py','maybe_enable_cpu_sampler')
    definition('new_hybrid_interface',VLLM/'model_executor/models/interfaces.py','IsHybrid')
    definition('new_hybrid_config',VLLM/'model_executor/models/config.py','HybridAttentionMambaModelConfig')
    definition('new_mamba_config',VLLM/'model_executor/models/config.py','MambaModelConfig')
    excerpt('new_async_scheduler_default',VLLM/'config/vllm.py',647,692)
    definition('new_cte_buckets',NXDI/'modules/autobucketing.py','generate_buckets_for_cte')
    definition('new_tkg_buckets',NXDI/'modules/autobucketing.py','generate_buckets_for_tkg')
    definition('new_generate_buckets',NXDI/'modules/autobucketing.py','generate_buckets')
    excerpt('new_stage_configs',NXDI/'models/model_base.py',3107,3154)
    excerpt('new_alias_output_contract',NXDI/'models/model_wrapper.py',1588,1637)
    excerpt('new_seq_ids_mask',NXDI/'models/model_wrapper.py',601,653)
    excerpt('new_config_base',NXDI/'models/config.py',92,215)
    excerpt('new_config_mask_warmup_long',NXDI/'models/config.py',608,626)
    definition('new_nxdi_output',NXDI/'models/model_base.py','NeuronBaseForCausalLM._get_constructed_outputs')
    definition('new_builder',NXDI/'models/application_base.py','NeuronApplicationBase.get_builder')
    excerpt('new_parallel_compiler',SITE/'neuronx_distributed/trace/model_builder.py',702,750)
    excerpt('old_nki_import',OLD/'models/lfm3/nki_write.py',1,38)
    excerpt('old_nki_cast',OLD/'models/lfm3/nki_write_torch.py',1,10)
    definition('new_nki_jit',SITE/'nki/__init__.py','jit')
    definition('typed_normalize',ROOT/'slot_cache_nki05_typed_torch_v1.py','apply_typed_writer')
    definition('typed_production_dispatch',ROOT/'lfm3_nki05_typed_dispatch_v1.py','make_writer')
    definition('preserved_full_gates',ROOT/'lfm3_typed_slot_writer_contract_v1.py','required_gates')
    definition('current_device_matrix',ROOT/'lfm3_nki05_device_matrix_v1.py','device_specs')
    definition('current_alias_matrix',ROOT/'lfm3_nki05_device_matrix_v1.py','alias_specs')
    for p in (SITE/'__editable___vllm_0_16_0_empty_finder.py',SITE/'__editable___vllm_neuron_0_5_3_finder.py'):
        excerpt(p.stem,p,1,17)
    raw(VLLM/'_version.py');raw(VLLM/'__init__.py')
    api={}
    targets={
        'neuronx_distributed_inference/models/config.py':['NeuronConfig.__init__','InferenceConfig.__init__','InferenceConfig.get_text_config','InferenceConfig.get_neuron_config_cls'],
        'neuronx_distributed_inference/models/model_base.py':['NeuronBaseModel.__init__','NeuronBaseForCausalLM.forward','NeuronBaseForCausalLM._is_prefill'],
        'neuronx_distributed_inference/models/application_base.py':['NeuronApplicationBase.__init__','NeuronApplicationBase.get_neuron_config_cls','NeuronApplicationBase.compile','NeuronApplicationBase.load','NeuronApplicationBase.get_state_dict'],
        'neuronx_distributed/parallel_layers/layers.py':['ParallelEmbedding.__init__','ColumnParallelLinear.__init__','RowParallelLinear.__init__','BaseParallelLinear.__init__']}
    for relative,names in targets.items():
        _,left=index(OLD_SITE/relative);_,right=index(SITE/relative)
        for name in names:
            a,b=left[name],right[name];api[relative+':'+name]=dict(old_path=str(OLD_SITE/relative),new_path=str(SITE/relative),old_line=a.lineno,new_line=b.lineno,
                old_signature=ast.unparse(a.args),new_signature=ast.unparse(b.args),same_signature=ast.dump(a.args)==ast.dump(b.args),same_function_ast=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
    model=literal(OLD/'models/lfm3/identity.py','EXPECTED_MODEL');layers=model['layer_types'];states=[]
    for layer,kind in enumerate(layers):
        for suffix in (['conv'] if kind=='conv' else ['k','v']):
            states.append(dict(index=len(states),layer=layer,kind=kind,state=suffix,shape_per_rank_context4096=['capacity',1024,7] if kind=='conv' else ['capacity',2,4096,128],dtype='bfloat16'))
    budgets=[]
    for length in (2048,4096):
        per_slot=layers.count('conv')*1024*7*2+2*(len(layers)-layers.count('conv'))*2*length*128*2
        for capacity in (4,8,16):budgets.append(dict(context=length,capacity=capacity,per_rank_single_logical_pool_bytes=per_slot*capacity,tp2_single_logical_pool_bytes=per_slot*capacity*2,actual_hbm_or_compile_peak=None))
    raw(Q/'vllm053-sampler-class-import-root-reproduction-20261112-v1.json')
    raw(Q/'lfm3-nki05-concrete-device-root-source-review-20261112-v1.json')
    return dict(schema='lfm3_v016_nxdi010_source_migration_evidence_v2',prior_inventory_failure=dict(script='analyze_lfm3_v016_nxdi010_migration_source_v1.py',terminal_chunk='bba219',exit_code=1,reason='excerpt end 70 exceeded constants.py length 68; no evidence output written'),source_only=True,runtime_admitted=False,source_sha256=SOURCES,evidence=EVIDENCE,api_comparison=api,
        api_summary=dict(compared=len(api),exact_function_ast=sum(x['same_function_ast'] for x in api.values()),unchanged_signatures=sum(x['same_signature'] for x in api.values())),
        state_ledger=dict(layers=len(layers),conv_layers=layers.count('conv'),full_attention_layers=layers.count('full_attention'),sliding_attention_layers=layers.count('sliding_attention'),states_per_rank=len(states),states_tp2=2*len(states),states=states),logical_state_budgets=budgets,
        source_analysis_not_runtime_proof=True,imports_of_investigated_packages=False,compiler_or_binary_or_weights_read=False,installed_or_frozen_sources_edited=False,services_locks_or_handoffs_created=False,
        existing_full_model_gates_retained=True,context4096_qualified=False,new_speedup_claim=False)

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**20,2*2**20))
    os.sched_setaffinity(0,{7});os.nice(19)
    value=build();path=Q/'lfm3-v016-nxdi010-source-migration-evidence-20261112-v2.json'
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sources=len(SOURCES),evidence_items=len(EVIDENCE),api=value['api_summary'],state_summary={k:v for k,v in value['state_ledger'].items() if k!='states'},logical_state_budgets=value['logical_state_budgets']),indent=2))
