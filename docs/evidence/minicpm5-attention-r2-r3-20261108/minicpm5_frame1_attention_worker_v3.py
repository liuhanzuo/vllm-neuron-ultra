"""Original frame1 prefix plus attention observations; admitted execution required.

Import/source checks are stdlib only. No runtime spec is created by this file.
"""
import importlib,sys
from pathlib import Path
import minicpm5_frame1_upstream_worker_v1 as upstream
import minicpm5_frame1_attention_observer_v3 as attention_observation
from minicpm5_frame1_upstream_worker_v1 import (
    require,same,phase,ACTIVATIONS,activation_source,activation_dispatch,weights,
    raw,metadata,same_bytes,FRAME,LENGTH,BUCKET,ARM,validate_baseline,
    original_dispatch,check_dispatch,norm_dispatch,norm_weights,Observers,
    PrefixDone,anchor,check_full_anchor,capture_bytes,once)


def validate_science(plan):
    original=upstream.validate_science(plan)
    same(plan['frame1_attention'],attention_observation.science(),'Exact full attention diagnostic science')
    for ref in attention_observation.UPSTREAM:
        same(plan['inputs'][ref['path']]['sha256'],ref['sha256'],'All prior full upstream references admitted')
    require(attention_observation.science()['total_raw_bytes']<attention_observation.OUTPUT_CAP,'Bounded complete capture and controls')
    return original


def execute(context,directory,native,rank):
    context.verify_sources();spec=validate_science(context.plan);directory=Path(directory);directory.mkdir(exist_ok=False)
    require(sys.byteorder=='little','Explicit captured BF16 little-endian host')
    activation_path=ACTIVATIONS['native'if native else'cpu']
    require(activation_path in context.plan['source_sha256'],'Actual activation source admitted before model load')
    activation_source(Path(activation_path).read_text())
    with context.activate():
        if native:
            from minicpm5_layer0_native_adapter_v2 import load
            torch,model,_,records=load(context);core=model.model;old=records[1];proof=context.plan['native_provenance'];transition=None
        else:
            import torch
            importlib.import_module('neuronx_distributed_inference.modules.lora_serving')
            from neuronx_distributed.parallel_layers import parallel_state
            from minicpm5_layer0_cpu_adapter_v2 import CPUAdapter
            transition=context.record_cpu_import();torch.set_num_threads(1)
            torch.distributed.init_process_group('gloo');initialized=False
            try:
                require(torch.distributed.get_rank()==rank and torch.distributed.get_world_size()==2,'Actual TP2')
                parallel_state.initialize_model_parallel(2,skip_collective_init=True);initialized=True
                runtime=CPUAdapter(context,torch);model,proof=runtime.build_original_model(rank,'prefill');core=model
                old=context.read_json(**dict(path=spec['cpu_anchors'][rank]['path'],wanted=spec['cpu_anchors'][rank]['sha256']))
                return capture(context,torch,model,core,old,proof,directory,False,rank,runtime,transition)
            finally:
                phase.base.cleanup_actions(([parallel_state.destroy_model_parallel]if initialized else[])+
                                           [torch.distributed.destroy_process_group],'TP cleanup')
        return capture(context,torch,model,core,old,proof,directory,True,-1,None,transition)


def capture(context,torch,model,core,old,proof,directory,native,rank,runtime,transition):
    observer=None;attention_observer=None;official=None;params=None;dispatch=None
    with torch.inference_mode():
        try:
            official,weight_proof=weights(torch,context.plan,context.guard)
            weight_rows={};mlp=core.layers[0].mlp
            from transformers.activations import SiLUActivation
            activation_path=ACTIVATIONS['native'if native else'cpu']
            same(str(Path(importlib.import_module(SiLUActivation.__module__).__file__).resolve()),
                 activation_path,'Original activation runtime source')
            activation_proof=activation_dispatch(torch,mlp.act_fn,SiLUActivation,Path(activation_path).read_text(),activation_path)
            require(all(getattr(mlp,n+'_proj').bias is None for n in ('gate','up','down')),'Original bias-free MLP')
            if not native:
                require(mlp.mlp_kernel_enabled is False and mlp.sequence_parallel_enabled is False,
                        'Original unfused native SDK MLP')
                down=mlp.down_proj
                require(down.input_is_parallel is True and down.reduce_output is True and
                        down.reduce_dtype==torch.bfloat16 and down.sequence_parallel_enabled is False and
                        down.tensor_parallel_group.size()==2 and down.pad_size==0,'Actual BF16 TP2 down path')
            for name,full in official.items():
                local=getattr(mlp,name+'_proj').weight
                wanted=full if native else (full[:,rank*3072:(rank+1)*3072]if name=='down'else full[rank*3072:(rank+1)*3072])
                same_bytes(raw(torch,local),raw(torch,wanted),'Exact official MLP '+name+' shard')
                weight_rows[name]=metadata(torch,local)
            params={name:getattr(mlp,name+'_proj').weight for name in official};del official
            if native:
                same(old['frame'],FRAME,'Native exact frame1')
                fixture=context.plan['fixtures'][7];require(fixture['id']==7 and len(fixture['input_ids'])==LENGTH,'Original long fixture')
                args=dict(input_ids=torch.tensor([fixture['input_ids']]),attention_mask=torch.ones((1,LENGTH),dtype=torch.long),
                          position_ids=torch.arange(LENGTH)[None],use_cache=True,logits_to_keep=1)
                original_inputs={k:metadata(torch,v)for k,v in args.items()if hasattr(v,'dtype')}
                invoke=lambda:model(**args)
            else:
                graph=context.plan['graph_calls'][1];same(graph['call'],old['call'],'Original call1');same(graph['bucket'],BUCKET,'Original bucket')
                validate_baseline(old,rank,full=False)
                original=runtime.graph_inputs(graph);inputs,widening=runtime.cpu_indices(original)
                same(widening,old['integer_widening'],'Lossless original integer widening')
                original_inputs=[runtime.tensor_metadata(x)for x in inputs]
                same(original_inputs,old['execution_input_metadata'],'All seven original argument values/layouts')
                runtime.verify_original_positions_masks(inputs,graph['call'],context.plan['fixtures'])
                runtime.select_original_bucket(model,BUCKET)
                same(phase.selected_targets(ARM,'prefill'),(),'Strict baseline has no precision intervention')
                dispatch=original_dispatch(core)
                invoke=lambda:model(*inputs)
            attention=core.layers[0].self_attn
            module=importlib.import_module('transformers.models.llama.modeling_llama'if native else 'neuronx_distributed_inference.modules.attention.attention_base')
            head_proof=(attention_observation.verify_native_projection_weights(torch,attention,context)if native else runtime.verify_head_shard_proof(core,rank))
            head_parameters=attention_observation.projection_parameters(attention,native)
            head_metadata={name:metadata(torch,value)for name,value in head_parameters.items()}
            attention_directory=directory/'attention';attention_directory.mkdir(exist_ok=False)
            norms=norm_dispatch(torch,core,native,context);norm_saved=norm_weights(torch,core,context,directory)
            observer=Observers(torch,core,native,directory,context.guard)
            attention_observer=attention_observation.Observer(torch,attention,module,native,rank,context.plan,attention_directory,context.guard)
            observer.prediction.begin(torch.tensor([LENGTH]),True)
            try:
                with attention_observation.profile(attention_observer.calls):invoke()
            except PrefixDone as done:require(done is observer.done,'Only own terminal prefix signal')
            else:raise RuntimeError('Prefix failed to stop before downstream layers')
            require(observer.completed==observer.routes==1,'Exactly one completed prefix')
            taps,tapmeta=observer.prediction.finish();checked=anchor(torch,taps,tapmeta,old,native)
            if native:same({k:metadata(torch,v)for k,v in args.items()if hasattr(v,'dtype')},original_inputs,'Native args unchanged')
            else:same([runtime.tensor_metadata(x)for x in inputs],original_inputs,'Seven CPU args unchanged')
            for name,parameter in params.items():
                require(getattr(mlp,name+'_proj').weight is parameter,'Original MLP parameter object')
                same(metadata(torch,parameter),weight_rows[name],'Original MLP parameter values')
            if not native: check_dispatch(core,dispatch)
            attention_report=attention_observer.summary()
            upstream_anchor=attention_observation.verify_upstream(context,observer.values,native,rank)
            full_anchor=check_full_anchor(context,observer.values,old,rank,native)
            same(norm_dispatch(torch,core,native,context),norms,'Original norm dispatch/weights unchanged')
            captures=list(observer.values.values());same(sum(x['bytes']for x in captures),capture_bytes(native),'Exact raw capture budget')
            result=dict(schema='minicpm5_frame1_attention_capture_v1',complete=True,native=native,rank=rank,frame=FRAME,
                        attention=attention_report,upstream_anchor=upstream_anchor,official_head_weights=head_proof,
                        anchor=checked,captures=captures,weights=weight_proof,actual_weight_shards=weight_rows,constructor=proof,
                        oproj_witnesses=[],precision_interventions=[],cpu_full_anchor=full_anchor,environment_transition=transition,actual_prefix_calls=1,
                        activation_dispatch=activation_proof,norm_dispatch=norms,norm_weights=norm_saved,
                        history_consumed=False,original_full_history_replayed=False,downstream_layers_executed=0,
                        full_model_qualified=False,new_speedup_claim=False)
        finally:
            phase.base.cleanup_actions(([attention_observer.close]if attention_observer else[])+([observer.close]if observer else[]),'capture cleanup')
        # Replay only after every capture hook and profiler has been removed.
        controls=attention_observation.own_input_controls(torch,attention,module,native,rank,attention_report,attention_directory,context.guard)
        attention_observer.check_dispatch()
        actual_parameters=attention_observation.projection_parameters(attention,native)
        for name,parameter in head_parameters.items():
            require(actual_parameters[name]is parameter,'Original projection parameter identity retained')
            same(metadata(torch,parameter),head_metadata[name],'Original projection parameter values/layout retained')
        if not native:check_dispatch(core,dispatch)
        result['own_input_controls']=controls
        result['own_input_controls_passed']=controls['passed']
        once(directory/'capture.json',result)
        return result
