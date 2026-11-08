"""Official native HF32 loader behind independent layer0 runtime admission."""
from pathlib import Path
from minicpm5_layer0_runtime_auth_v1 import require,identity
import minicpm5_baseline_native_v4 as original
from minicpm5_loading_info_v4 import normalize_loading_info

def load(context):
    context.guard('native');require(context._sources_checked,'Complete authenticated sources')
    plan=context.plan
    for row in plan['model_files']:context.verify_input(row['path'])
    import torch
    import transformers.models.llama.modeling_llama as native
    actual=Path(native.__file__).resolve()
    require(actual==original.NATIVE_LLAMA.resolve() and str(actual) in plan['source_sha256'],'Original native implementation path')
    torch.set_num_threads(2)
    model,loading=native.LlamaForCausalLM.from_pretrained(str(original.MODEL_PATH),dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True,trust_remote_code=False,output_loading_info=True)
    loading=normalize_loading_info(loading)
    require(not any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'Original official loading info')
    model.eval();header=context.read_json(plan['safetensors_header']['path'],plan['safetensors_header']['sha256']);header.pop('__metadata__',None)
    state=model.state_dict();require(set(state)==set(header) and len(state)==381,'Original official keys')
    for name,value in state.items():require(list(value.shape)==header[name]['shape'] and value.dtype==torch.bfloat16 and value.device.type=='cpu','Original official tensor shape/dtype')
    require(sum(value.numel() for value in model.parameters())==original.PARAMETERS,'Original parameter count')
    require(not model.config.tie_word_embeddings and model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr(),'Independent embedding/head')
    require(model.config.rope_parameters==dict(rope_type='default',rope_theta=5000000),'Original RoPE')
    require(model.config.eos_token_id==[1,130073] and model.config.pad_token_id==1 and model.config._attn_implementation=='eager','Original model config')
    original._check_config(context.read_json(str(original.MODEL_PATH/'config.json')))
    provenance=dict(model_class='transformers.models.llama.modeling_llama.LlamaForCausalLM',source=str(actual),
        source_sha256=plan['source_sha256'][str(actual)],revision=original.REVISION,dtype='torch.bfloat16',attention='eager',
        parameters=original.PARAMETERS,tensors=381,independent_embedding_and_lm_head=True,loading_info=loading,
        native_threads=2,device='cpu',math_implementation_modified=False)
    require(provenance==plan['native_provenance'],'Original native provenance')
    for row in plan['model_files']:require(identity(row['path'])==row['identity'],'Official file changed during load')
    reference=context.load_pt(plan['hf_reference']['path'],32*2**20)
    require(len(reference)==8 and [x['id'] for x in reference]==list(range(8)),'Original reference32 fixture set')
    for fixture,saved in zip(plan['fixtures'],reference):
        require(fixture['id']==saved['id'] and fixture['input_ids']==saved['input_ids'] and fixture['continuation_ids']==saved['continuation_ids'],'Original native fixture bytes')
        require(len(saved['logits'])==4 and all(list(v.shape)==[130560] and str(v.dtype)=='torch.float32' for v in saved['logits']),'Original32 full vocabulary')
    records=[context.read_json(row['path'],row['sha256']) for row in plan['hf_frame_records']]
    require(len(records)==32 and [row['frame']['index'] for row in records]==list(range(32)),'All retained native frame records')
    del state,header
    context.guard('native');return torch,model,reference,records

def run(context,writer_factory):
    from minicpm5_layer0_capture_native_driver_v3 import run as driver
    with context.activate():
        torch,model,reference,records=load(context)
        import importlib
        from minicpm5_layer0_python_trace_source_v1 import source_text
        module=importlib.import_module(type(model.model.layers[0].self_attn).__module__)
        source=source_text(context,module,'native')
        return driver(torch,model,context.plan,reference,records,original._cache,writer_factory,context.guard,source)
