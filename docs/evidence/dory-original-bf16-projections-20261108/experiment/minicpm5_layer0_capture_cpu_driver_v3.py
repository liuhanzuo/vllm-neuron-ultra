"""Original14-call CPU capture loop for one admitted TP2 rank and history arm.

No model construction or guard substitution is provided. Caller supplies exact
reviewed constructor/runtime hooks under a future successor admission. The old
constructor's frozen guard must not be monkeypatched to make this loop runnable.
"""
import gc
import importlib
import minicpm5_layer0_sdk_python_trace_observer_v1 as observe
import minicpm5_layer0_capture_endpoints_v1 as endpoint


def run(torch,plan,arm,rank,expected_calls,graph_calls,runtime,writer_factory,admission_guard):
    require=endpoint.require;admission_guard()
    require(arm in ('C_Hd_control','C_Hc_history_generation') and rank in (0,1),'Original CPU arm/rank')
    require(len(expected_calls)==len(graph_calls)==len(plan['calls'])==14,'All14calls required')
    require(torch.distributed.is_initialized() and torch.distributed.get_world_size()==2 and torch.distributed.get_rank()==rank,'Actual TP2 rank required')
    require(torch.distributed.get_backend()=='gloo' and torch.get_num_threads()==1,'Original CPU runtime')
    reports=[];model=None;hooks=None;phase=None;previous_post=None;previous_expected=None
    try:
        with torch.inference_mode():
            for call,expected,graph in zip(plan['calls'],expected_calls,graph_calls):
                admission_guard();require(expected['call']==graph['call']==call and expected['arm']==arm and expected['rank']==rank,'Original call identity')
                # Original graphinputs and CPUindex widening preserve placeholders.
                inputs=runtime.graph_inputs(graph);require(len(inputs)==7,'Original7inputs')
                require([runtime.tensor_metadata(x) for x in inputs]==expected['input_metadata'],'Original graphinput values/layout')
                inputs,widening=runtime.cpu_indices(inputs)
                require(widening==expected['integer_widening'] and [runtime.tensor_metadata(x) for x in inputs]==expected['execution_input_metadata'],'Original CPUindex adaptation')
                require(inputs[3].tolist()==([call['frames'][0]['slot']] if call['phase']=='prefill' else [0,1,2,3]),'Actual physical seq_ids')
                rowmap=endpoint.physical_rows(call,inputs[3].tolist())
                runtime.verify_original_positions_masks(inputs,call,plan['fixtures'])
                if phase!=call['phase']:
                    if hooks is not None:hooks.close()
                    hooks=None;model=None;gc.collect()
                    model,proof=runtime.build_original_model(rank,call['phase'])
                    require(proof['compiled'] is proof['resharded'] is proof['sdk_math_modified'] is False,'Original constructor required')
                    runtime.verify_head_shard_proof(model,rank)
                    hooks=runtime.prediction_hooks(model);phase=call['phase']
                before_records=runtime.manifest(expected['imported_manifest'],rank)
                after_records=runtime.manifest(expected['after'],rank)
                require(len(before_records)==len(after_records)==84,'Original84state manifests')
                if arm=='C_Hc_history_generation' and call['index']>0:
                    require(previous_post is not None and previous_expected is not None,'Missing generated CPU history')
                    require(expected['history']['origin']==dict(kind='cpu_history_previous_post',arm=arm,call=call['index']-1),'History generation order')
                    endpoint.verify_states(torch,previous_post,before_records,rank,'previous generated post')
                    # Includes logical reset between groups; never zero or replace
                    # the generated history with a fresh device-cache snapshot.
                    for parameter,value in zip(model.kv_mgr.past_key_values,previous_post):parameter.copy_(value)
                else:
                    require(expected['history']['origin']==dict(kind='frozen_v2_device_before',call=call['index']),'Original frozen before history')
                    for parameter,record in zip(model.kv_mgr.past_key_values,before_records):parameter.copy_(runtime.load_state(record))
                endpoint.verify_states(torch,model.kv_mgr.past_key_values,before_records,rank,'imported input')
                previous_post=None;runtime.select_original_bucket(model,graph['bucket'])
                prefill=phase=='prefill';lengths=inputs[1].sum(-1).clamp_min(1) if prefill else inputs[2][:,0]+1
                positions=(lengths-1).tolist() if prefill else [0]*len(call['frames'])
                physical_frames=sorted(call['frames'],key=lambda f:inputs[3].tolist().index(f['slot']))
                writer=writer_factory(arm,rank,call['index'],frame_index=None)
                try:
                    sink=observe.SnapshotSink(torch,writer.tensor)
                    capture=observe.SDKCapture(torch,sink,arm,rank,physical_frames,call['index'],phase,positions)
                    capture.cache_backing_before=dict(manifest=expected['imported_manifest'],layer0=[x for x in before_records if x['layer']==0])
                    capture.cache_backing_after=dict(manifest=expected['after'],layer0=[x for x in after_records if x['layer']==0])
                    module=importlib.import_module(type(model.layers[0].self_attn).__mro__[1].__module__)
                    runtime.verify_attention_module(module)
                    capture.layer0_owner=model.layers[0].self_attn
                    source=runtime.attention_trace_source(module)
                    with observe.installed_sdk(model.layers[0].self_attn,module,torch,source['text'],source['path'],source['sha256'],phase) as trace_counts:
                        hooks.begin(lengths,prefill)
                        with capture.activate():output,*post=model(*inputs)
                        taps,tap_metadata=hooks.finish()
                    checked=endpoint.cpu_endpoint(torch,output,taps,tap_metadata,model.kv_mgr.past_key_values,post,expected,before_records,after_records,rank)
                    receipt=capture.complete();receipt['snapshots']=sink.records;receipt['physical_row_map']=rowmap
                    observe.validate_sdk_capture_summary(receipt,trace_counts['contract'])
                    roles=['qk0']if prefill else['qk0','qk1']
                    require(trace_counts['calls']==1 and trace_counts['other_phase_calls']==0 and trace_counts['observed_scale_events']==len(roles)and trace_counts['roles']==roles,'Exact SDK method/event counts')
                    require(runtime.inputs_original_roundtrip_exact(inputs,expected),'Input tensors modified')
                    reports.append(writer.finish(receipt,checked))
                    # Persist actual produced poststate across all14calls, not the
                    # frozen expected state; all84 were checked against old hashes.
                    if arm=='C_Hc_history_generation':previous_post=post
                    previous_expected=expected;del output,taps,inputs
                finally:writer.close()
        require(len(reports)==14,'Incomplete CPU history capture');admission_guard()
        return dict(complete=True,arm=arm,rank=rank,calls=14,frames=32,all_original_outputs_taps_input_poststate_raw_exact=True,
            generated_cpu_history_preserved=arm=='C_Hc_history_generation',reports=reports,scale_observation_schema=observe.SCHEMA,scale_events=20,full_model_qualified=False)
    finally:
        if hooks is not None:hooks.close()
